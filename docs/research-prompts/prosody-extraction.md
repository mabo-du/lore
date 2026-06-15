# Prosody and Emphasis Extraction from Speech — Implementation Plan

## Status: Research Complete

The standout finding: **SenseVoice-Small via ONNX Runtime** runs at ~140× real-time on CPU (~25 seconds for a 60-minute file), consumes ~1.1 GB RAM, and outputs inline emotion/event tags (`<|HAPPY|>`, `<|LAUGHTER|>`, `<|SAD|>`) directly into the text stream. This single model replaces a cascade of alignment → feature extraction → classification.

For quantitative features (F0, intensity), **Parselmouth** (<100 MB, >50× real-time) is the recommended companion.

## Recommended Approach: Two-Layer Architecture

### Layer 1 — High-Level Categorical Labels (SenseVoice-Small)

**Goal:** Add emotion and audio event tags to segments.

**Steps:**
1. Add SenseVoice-Small ONNX as an optional post-processing worker (runs after transcription completes, not during).
2. The model outputs inline tags: `<|HAPPY|>`, `<|SAD|>`, `<|ANGRY|>`, `<|NEUTRAL|>`, `<|LAUGHTER|>`, `<|APPLAUSE|>`, `<|COUGH|>`.
3. Map these to existing `Segment` fields:
   - `<|LAUGHTER|>` → existing `AudioClassifyWorker` event (duplicate detection is fine — compare confidence sources)
   - `<|HAPPY|>`, `<|SAD|>`, etc. → new `Segment.emotion: Optional[str]` field
4. Store the tags inline in a new `Segment.emotion` attribute for downstream LLMWorker consumption.

**Integration:**
```python
class SenseVoiceWorker(QThread):
    finished = pyqtSignal(Transcript)
    error = pyqtSignal(str)

    def __init__(self, audio_path: Path, transcript: Transcript):
        self.audio_path = audio_path
        self.transcript = transcript
        # ONNX session with CPUExecutionProvider

    def run(self):
        # Run SenseVoice ONNX inference
        # Parse inline tags from output
        # Align tags to segments by timestamp
        # Emit finished(updated_transcript)
```

**RAM management:** SenseVoice runs after Whisper/NLLB are unloaded. At ~1.1 GB, it fits within the 8 GB budget when other heavy models are not resident.

**When to skip:**
- If the audio is very short (<5 minutes), the overhead of loading the ONNX model may exceed the processing time. Use a minimum duration threshold.

---

### Layer 2 — Quantitative Features (Parselmouth + Forced Alignment)

**Goal:** Extract word-level F0, intensity, and speaking rate for archival metadata.

**Steps:**
1. Run forced alignment via WhisperX or similar to get precise word-level timestamps (Lore's current Whisper output has segment-level timestamps but word-level alignment is coarse).
2. Extract F0 contours and RMS intensity per word using Parselmouth.
3. Normalize: compute per-speaker z-scores (median + IQR to handle pitch-halving outliers).
4. Compute binary emphasis flag: `is_emphasized = abs(z_pitch) > 1.5 or abs(z_intensity) > 1.5`.
5. Store as hidden metadata on each `Word` object (or `Segment` for segment-level aggregates).

**Data model:**
```python
@dataclass
class WordProsody:
    word_index: int
    f0_mean: float        # z-score normalized
    f0_range: float       # z-score normalized
    intensity_mean: float  # z-score normalized
    duration_ms: int
    is_emphasized: bool
    speaking_rate_wpm: float
```

**Storage:**
- Internal: `Word.prosody: Optional[WordProsody] = None`
- LLM prompt: expose only `is_emphasized` flag inline (e.g., `*really*` for emphasized words)
- Export: write to TEI `<shift feature="loud" new="strong"/>` elements, or store in OHMS custom metadata

**RAM estimate:** Parselmouth requires <100 MB. The prosody worker should be its own QThread that starts after transcription and aligns results back.

---

## Abstraction Decision: Categorical Labels Only for LLM

| Consumer | What to send | Why |
|----------|-------------|-----|
| LLMWorker (summarization) | Inline emotion tags, emphasis markers | LLMs can't interpret raw float arrays |
| OHMS export (archival) | TEI `<shift>` elements, `<vocal>` events | Standards-compliant long-term preservation |
| Internal query/search | z-score vectors in `WordProsody` | Enables future computational linguistics use |

## Priority

**Layer 1 (SenseVoice)** is P2 — valuable, but requires adding a new model dependency and ONNX workflow. Worth doing after overlap detection and summarization enrichment are stable.

**Layer 2 (Parselmouth)** is P3 — primarily archival value. Only implement if an archive partner explicitly requests TEI-compliant prosody encoding.

## References

See `docs/research-papers/Prosody Extraction for Oral History.md`.
