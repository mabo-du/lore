# Overlapping Speech Detection in Diarization Pipelines — Implementation Plan

## Status: Research Complete

Two deep-research reports produced aligned findings. The core insight: ONNX Runtime strips the PyTorch overhead that makes the current 3.1 pipeline memory-unsafe (1.5 GiB → <200 MB), and the `community-1` model solves the long-form speaker confusion that 3.1 struggles with.

## Recommended Approach: Three-Phase Incremental Rollout

### Phase 1 — Flag Overlap Regions (MVP)

**Goal:** Tell the user *when* overlap occurred, even without resolving who said what.

**Steps:**
1. Integrate `pyannote-onnx` and/or `pyannote-onnx-extended` as an optional backend alongside the existing `DiarizationEngine`.
2. Deploy the `pyannote/overlapped-speech-detection` model (lightweight, ONNX, <10 MB). This model outputs frame-level binary overlap probabilities.
3. When overlap probability exceeds threshold (e.g., 0.6) for >500 ms, create an `OverlapFlag` entry in a new `overlap_regions` list on the Transcript object.
4. Surface overlap flags in the UI (e.g., highlighted regions on the waveform or transcript strip).
5. OHMS export: inject `<overlap>` annotations or use the WebVTT `<v>` speaker tags to indicate simultaneous speech regions (OHMS XML 6.0 supports overlapping WebVTT cues natively).

**Models to evaluate:**
- `onnx-community/pyannote-segmentation-3.0` (5.99 MB ONNX) — full segmentation with native OSD
- `pyannote/overlapped-speech-detection` — standalone, even smaller

**Data model change:**
```python
@dataclass
class OverlapRegion:
    start_ms: int
    end_ms: int
    confidence: float
```
Added to `Transcript.overlap_regions: list[OverlapRegion] = field(default_factory=list)`.

**Success criteria:**
- Overlap regions detected and stored in transcript
- UI highlights overlapping regions
- OHMS export preserves overlap information via WebVTT cues
- RAM impact <100 MB additional

---

### Phase 2 — Migrate to pyannote Community-1 + ONNX

**Goal:** Replace PyTorch pyannote 3.1 with the ONNX-exported community-1 pipeline for all diarisation.

**Steps:**
1. Replace `pyannote/speaker-diarization-3.1` with `pyannote/speaker-diarization-community-1` via ONNX backend.
2. Ensure `exclusive_speaker_diarization` is **disabled** (the community-1 model has this mode; for oral history we want the raw multi-speaker output, not the single-speaker-flattened version).
3. Retain the ONNX segmenter + WeSpeaker ResNet34 embedding model from Phase 1.
4. Benchmark DER against the current 3.1 pipeline on Lore's test corpus.

**Models:**
- `pyannote/speaker-diarization-community-1`
- `onnx-community/wespeaker-voxceleb-resnet34-LM` (26.5 MB ONNX)

**Expected improvements over current 3.1:**
- AMI (IHM): 18.8% → 17.0% DER
- RAM: 1.5 GiB → <200 MB
- CPU speed: ~7× faster than PyTorch (based on diarize library benchmarks)

---

### Phase 3 — Full Multi-Speaker Overlap Attribution

**Goal:** Resolve *who* was speaking during overlapping regions, not just *that* overlap occurred.

**Steps:**
1. Introduce `OverlapSegment` data model with `speaker_set: set[str]` attribute replacing the single `speaker_label`.
2. When Phase 1 flags an overlap region, invoke a conditional second pass using the full community-1 pipeline's multi-label output (bypassing `exclusive_speaker_diarization`).
3. Store resolved speakers in the `speaker_set`.
4. OHMS export: use the `<vtt_transcript>` element with overlapping WebVTT cues (OHMS 6.0 supports this natively).
5. Non-overlapping segments continue using the fast single-speaker path.

**Data model change:**
```python
@dataclass
class OverlapSegment:
    start_ms: int
    end_ms: int
    text: str
    speaker_set: set[str]
    # ... other fields
```

**Conditional execution logic:**
```
if overlap_regions exists:
    run community-1 full pipeline on these regions
    populate OverlapSegment.speaker_set
else:
    use existing fast Resemblyzer/3.1 path
```

**Success criteria:**
- Overlapping segments have >1 speaker assigned
- Non-overlapping segments unaffected
- OHMS export produces valid XML with overlapping WebVTT cues

---

## Key Risks

| Risk | Mitigation |
|------|------------|
| ONNX conversion quality not matching PyTorch accuracy | Validate DER on test corpus before switching default |
| `exclusive_speaker_diarization` accidentally enabled | Explicitly configure the flag; add validation test |
| community-1 model license changes | It's MIT/Apache-2.0 compatible per Hugging Face |

## References

See `docs/research-papers/Overlap Detection in Diarization Pipelines.md` and `docs/research-papers/Lightweight Flagging, Full Attribution Later_ An Incremental Strategy for Overlapping Speech Diarization in CPU-Constrained Lore Applications.md` for full research backing these recommendations.
