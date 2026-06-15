# Backchannel and Listener-Response Classification — Implementation Plan

## Status: Research Complete

Key findings:
- **Whisper actively deletes backchannels** via text normalisation. A pure rule-based approach (match Whisper output against filler-word dictionary) would achieve <40% recall.
- **Two-stage hybrid** is the optimal strategy: high-precision rule-based filter first, lightweight acoustic classifier for missed cases.
- **YAMNet's 1024-dim embeddings are already computed** for every 960 ms frame by the existing `AudioClassifyWorker`. A downstream SVM/MLP on these embeddings can classify backchannels with near-zero additional compute.

## Recommended Approach: Two-Stage Hybrid

### Stage 1 — Rule-Based Filter (Immediate, Zero-Cost)

Integrate into the existing `AudioClassifyWorker`:

```python
BACKCHANNEL_LEXICON = {"mhm", "uh-huh", "mm-hmm", "uh", "um",
                       "yeah", "right", "okay", "sure", "yep"}
BACKCHANNEL_MAX_MS = 800

for segment in whisper_segments:
    text = segment.text.strip().lower().rstrip(".,!?")
    duration = segment.end_ms - segment.start_ms
    if duration <= BACKCHANNEL_MAX_MS and text in BACKCHANNEL_LEXICON:
        segment.is_backchannel = True
        segment.backchannel_confidence = 1.0  # Rule-based = high precision
        segment.backchannel_source = "rule"
```

**Expected recall:** ~40% (misses non-lexical backchannels, ASR-dropped tokens, regional variants)
**Expected precision:** >95% (rule is strict — false positives require ASR hallucination)
**Cost:** Negligible — pure string matching on Whisper output that's already in memory.

---

### Stage 2 — Lightweight Acoustic Classifier (Post-Processing)

For the 60% of backchannels missed by Stage 1, train a lightweight classifier using **transfer learning on YAMNet embeddings**.

**Architecture:**
```
YAMNet (MobileNetV1) → freeze → penultimate layer (1024-dim embedding)
                                               ↓
                                    SVM or 2-layer MLP (1024→128→2)
                                               ↓
                                    backchannel / non-backchannel
```

**Training data:** Switchboard SwDA corpus for initial training; fine-tune on oral-history-like data if available.

**Integration:**
1. For each Whisper segment that Stage 1 did NOT flag as backchannel:
   - Extract the corresponding audio frames
   - Pass through YAMNet (already loaded and running in AudioClassifyWorker)
   - Intercept the 1024-dim penultimate embedding (before final classification layer)
   - Run through lightweight MLP (ONNX, <10 MB)
   - If score > threshold, label as `is_backchannel=True`, `backchannel_source="acoustic"`

**Expected performance:**
- SVM on MFCCs: 75-85.7% accuracy (Switchboard)
- YAMNet-embedding + MLP: expected similar or better (richer features)
- Precision: ~85%, Recall: ~70% (combining Stage 1 + Stage 2)

**RAM:** <1 MB (MLP), piggybacks on YAMNet's existing ~15 MB footprint
**Latency:** Sub-millisecond per segment (embedding already computed)

---

### Stage 3 (Future) — Pragmatic Function Classification

Once detection is reliable, classify *function*:
- **Agreement/Continuer**: rising pitch + short duration + standard token
- **Surprise**: high F0 (>350 Hz), wide range, longer duration
- **Impatience**: rapid repetition, low pitch, high intensity, staccato cadence

This requires the acoustic features from Parselmouth (F0, intensity) fused with lexical features. HuBERT audio backbone alone achieved 0.805 F1 on Switchboard for pragmatic classification; audio+text fusion improved to 0.894 F1.

**Not recommended until Stage 1 & 2 are stable and validated on real-world Lore audio.**

---

## Data Model

```python
@dataclass
class BackchannelMetadata:
    is_backchannel: bool = False
    confidence: float = 0.0
    source: str = ""  # "rule" | "acoustic" | "manual"
    function: Optional[str] = None  # "agreement" | "surprise" | "impatience" | None
```

Attached to `Segment.backchannel: Optional[BackchannelMetadata] = None`.

## Priority

**Stage 1** can be implemented immediately — it's a few lines of code in the existing `AudioClassifyWorker` with zero overhead. Worth doing alongside the next `AudioClassifyWorker` change.

**Stage 2** requires a trained model and ONNX export. Worth scoping after Stage 1 is validated and we have a small corpus of real-world backchannel examples from Lore users.

## References

See `docs/research-papers/Backchannel Detection and Classification Research.md` and `docs/research-papers/A Two-Stage Strategy for Efficient Backchannel Detection_ Integrating High-Precision Rules with Lightweight Models in CPU-Only Workflows.md`.
