# Audio-Visual Active Speaker Detection — Implementation Plan

## Status: Research Complete

**Core finding:** CPU-based ASD is feasible (~2-3 GB RAM, 10-20 FPS) but the value proposition is narrow. It provides significant improvements only in group/panel settings (3+ speakers) where audio diarisation degrades. For Lore's primary use case (1-on-1 oral history interviews), the benefit is marginal and the compute cost is hard to justify.

**Decision: P3 priority. Only implement if:**
1. Lore adds video file input support, AND
2. A user segment explicitly requests group interview diarisation improvements, AND
3. The audio-only ONNX migration (Phase 2 of overlap plan) has been completed first.

## Architecture (if implemented)

### Pipeline

```
Video frames at 30 FPS
    │
    ├─→ Silero VAD (acoustic gate) — only activate below when speech detected
    │
    └─→ Face Detection (Ultra-Light-Fast-Generic-Face-Detector-1MB)
        │
        ├─→ Multi-Object Tracking (BoT-SORT or LITE)
        │
        └─→ Active Speaker Detection (LR-ASD via ONNX)
            │
            └─→ Fusion with audio diarization output
```

### Key design decisions

1. **ASD is a post-hoc correction layer, not a replacement.** Audio diarization runs first. ASD only runs on regions where:
   - Audio diarization confidence is low (entangled embeddings indicate overlap)
   - ≥3 speakers are detected from initial audio analysis
   - Speech overlap ratio >30%

2. **Silero VAD gate prevents wasted inference.** The visual pipeline only activates when speech is detected. A 500 ms cooldown prevents toggling during micro-pauses.

3. **Off-screen speaker detection.** If VAD detects speech but ASD finds no visible face → label segment as "Interviewer (Off-Screen)". This is a clean integration point with immediate user value.

4. **Fallback hierarchy:**
   - ASD confidence >70% AND tracking stable → use ASD-enhanced labels
   - ASD confidence 30-70% → weight ASD and audio results by confidence
   - ASD confidence <30% OR track loss >30% → revert to audio-only

### Component selection

| Component | Model | Disk | RAM | CPU latency/frame |
|-----------|-------|------|-----|-------------------|
| Acoustic gate | Silero VAD (ONNX) | 1.5 MB | <10 MB | 0.09-0.15 ms |
| Face detection | Ultra-Light-Fast-Generic-Face-Detector-1MB | 1 MB | ~50 MB | ~4 ms |
| Tracking | BoT-SORT | — | ~20 MB | ~5 ms |
| Active speaker | LR-ASD (ONNX) | ~5 MB | ~15 MB | 1.5-4.5 ms |
| **Total** | | | **<100 MB active** | **10-14 ms per frame** |

### Hardware requirement

Modern 4-core/8-thread CPU with AVX2 support (Intel 8th-gen i5 / AMD Ryzen 3000 or newer). The pipeline processes at ~70 FPS theoretical on a single thread; real-world at 30 FPS input it runs faster-than-real-time.

### Integration with Lore

```python
class ActiveSpeakerWorker(QThread):
    """Optional post-processing worker for video files."""

    def __init__(self, video_path: Path, audio_diarization: DiarizationResult):
        self.video_path = video_path
        self.audio_result = audio_diarization

    def run(self):
        # 1. Run Silero VAD to find speech regions
        # 2. For each speech region with audio ambiguity:
        #    a. Run face detection + tracking
        #    b. Run LR-ASD on each tracked face
        #    c. Fuse with audio diarization labels
        # 3. Emit corrected diarization result
```

## Decision Framework for Activation

| Condition | Action |
|-----------|--------|
| Input is audio-only (no video file) | Skip entirely. ASD not applicable. |
| Input is video, ≤2 speakers detected | Skip. Audio-only is sufficient. |
| Input is video, ≥3 speakers AND overlap >30% | Activate ASD. Conditional execution — only on high-entropy regions. |
| Visual quality poor (resolution, lighting) | Fall back to audio-only. Log reason. |

## Pre-requisites

This feature depends on:
- **ONNX Runtime** integration (same as overlap plan — the framework is shared)
- **Video file input support** in Lore (currently audio-only)
- **`DiarizationEngine` refactoring** to support fusion outputs

None of these are blockers in isolation, but together they make this a significant engineering investment.

## References

See `docs/research-papers/CPU Active Speaker Detection Research.md` and `docs/research-papers/Enhancing Oral History Diarization_ A Decision Framework for Integrating CPU-Friendly Audio-Visual Active Speaker Detection.md`.
