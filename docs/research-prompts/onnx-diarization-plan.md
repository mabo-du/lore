# ONNX Diarisation Pipeline — Implementation Plan

## Goal

Replace the torch-dependent diarisation backends (Resemblyzer, Pyannote 3.1)
with a pure ONNX Runtime pipeline — eliminating the ~1.5 GiB PyTorch memory
footprint while maintaining or improving diarisation accuracy.

## Strategy

Build the pipeline from directly loaded ONNX models, referencing `diarize`
and `pyannote-onnx-extended` as architecture references but **not** as
dependencies. Neither library's published dependency list guarantees
a torch-free install.

## Pipeline components

| Step | Current | Replacement | Size |
|------|---------|-------------|------|
| Voice Activity Detection | (implicit in diarization) | Silero VAD ONNX via `onnxruntime.InferenceSession` | ~1.7 MB |
| Speaker embedding | Resemblyzer (torch) or Pyannote 3.1 (torch) | WeSpeaker ResNet34-LM ONNX | 26.5 MB |
| Speaker clustering | KMeans / AHC | sklearn spectral clustering (GMM-BIC for count estimation, from `diarize` paper) | 0 MB (already dep) |
| Overlap detection | ✅ Already done (Phase 1) | Same ONNX segmentation model | 6 MB |
| Average Stitching | N/A | Implement locally (from pyannote-onnx-extended paper) | ~50 lines |
| Overlap-free clustering | N/A | Extract overlap regions before clustering, assign after | ~100 lines |

**Total new ONNX model weight:** ~28 MB  
**Peak RAM for diarization:** drops from ~1.5 GiB → ~200 MB

## Implementation phases

### Phase 2a: Silero VAD ONNX integration

Silero VAD already ships ONNX weights (`silero_vad.onnx`).
The official repo and Hugging Face mirrors provide them. Load via:

```python
import onnxruntime
session = onnxruntime.InferenceSession("silero_vad.onnx", providers=["CPUExecutionProvider"])
```

This replaces the implicit VAD step in the current Resemblyzer path.

### Phase 2b: WeSpeaker embedding + sklearn clustering

Load `wespeaker-voxceleb-resnet34-LM.onnx` (26.5 MB, CC-BY-4.0/Apache-2.0)
via ONNX Runtime. Extract embeddings from speech segments. Cluster with
sklearn spectral clustering using GMM-BIC for automatic speaker-count
estimation (from `diarize`'s approach).

This replaces both the Resemblyzer and Pyannote embedding/clustering paths.

### Phase 2c: Average Stitching

Implement the sliding-window smoothing technique from the
pyannote-onnx-extended paper. This prevents hard boundary cutoffs between
overlapping windows when classifying frames.

### Phase 2d: Two-stage overlap-aware clustering

1. Run overlap detection (Phase 1, already done)
2. Remove overlapping frames from the embedding stream
3. Cluster on clean (non-overlapping) embeddings only
4. Assign overlapping frames to the nearest cluster — **best-guess single
   speaker label** only. The assignment is not multi-label; overlapping
   frames receive whichever speaker cluster is nearest, and the existing
   Phase 1/2 UI surfaces (overlap strip, segment badges) continue to flag
   these regions as containing overlap. This prevents the diarization
   output from appearing confidently wrong — the speaker label is a best
   guess, and the overlap flag tells the researcher it's uncertain.

This prevents speaker confusion caused by entangled overlapping embeddings
(the source of ~50% of diarisation errors, per the paper's oracle analysis).

## Validation plan

| Level | Data | Success criteria |
|-------|------|-----------------|
| **Floor** | VoxConverse (downloadable benchmark, YouTube/news audio) | DER matches or beats Lore's own Resemblyzer + Pyannote 3.1 measurements on the same files, not a third-party published baseline |
| **Spot-check** | User's flagged recordings (known speaker count) | Manual review: no speaker swaps on known segments |

VoxConverse as the objective floor, user's own files as the realistic
oral-history spot-check.

## Gating

- Both old paths (Resemblyzer, Pyannote 3.1) remain installed and selectable
- User can switch back via Settings at any time
- Old paths are NOT removed until ONNX path is validated on both VoxConverse
  AND user's own recordings with user sign-off
- After sign-off: `pyproject.toml` deps updated to remove `pyannote-audio`,
  `resemblyzer`, and `torch` (if not needed elsewhere — check first)

## Files changed

| File | Change |
|------|--------|
| `src/lore_core/diarization.py` | Add `_run_onnx()` method; old paths become fallbacks |
| `src/utils/model_manager.py` | Add `WHISPER_REPO_IDS`-style mapping for VAD + WeSpeaker ONNX models |
| `pyproject.toml` | Add `pyannote.core`? (check transitive deps first); remove after validation |
| `tests/test_diarization.py` | Integration tests against VoxConverse sample |
| `docs/credits.md` | Add WeSpeaker CC-BY-4.0 attribution |

## Open questions

1. **`pyannote.core`** — used by `pyannote-onnx-extended` for its data model;
   the `diarize` library doesn't need it. Scoping whether we need it for
   our implementation. If not, skip the dep entirely.
2. **Transitive torch from Silero VAD** — only if you import `silero_vad`
   from PyPI. Loading the ONNX weights directly via `onnxruntime` avoids
   this entirely (confirmed safe).
3. **GMM-BIC implementation** — sklearn doesn't ship this directly; we
   may need to port ~50 lines from the `diarize` source (Apache 2.0).
   Apache 2.0 requires retaining the original copyright and license notice
   with copied code — `docs/credits.md` needs a diarize entry alongside
   the WeSpeaker CC-BY-4.0 one.
