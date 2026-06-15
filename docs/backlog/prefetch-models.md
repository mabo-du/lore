# Feature: Model Pre-Fetch for Offline Field Use

**Filed:** 2026-06-15  
**Priority:** High (blocking v0.2.0)  
**Status:** Open

## Problem

Lore downloads models on first demand. A user clicking "Auto-Generate Abstract" 
triggers a ~1.1 GB download of Qwen 2.5 1.5B GGUF without warning. For field 
researchers with no internet access during a session, this is a workflow hazard:
the summary feature silently fails or hangs.

The same applies to: Whisper model (if not cached), NLLB translation model, 
Segmentation ONNX model.

## Solution Options

**Option A — First-run setup wizard:**
On first launch (or `lore --setup`), enumerate all required models, show a 
progress bar with estimated download sizes, and let the user pre-cache them 
over a known-good connection.

**Option B — CLI pre-fetch flag:**
`lore --prefetch-models` downloads all models to the cache directory without 
launching the GUI. Could also support `lore --prefetch-models --tier Fast` 
to limit to a specific model tier.

**Option C — Hybrid:**
Pre-fetch CLI flag + a "Download required models" button in Settings with 
per-model progress.

## Models to Pre-fetch

| Model | Size | Used By |
|-------|------|---------|
| faster-whisper-small (Fast) | ~500 MB | TranscriptionWorker |
| faster-whisper-medium (Balanced) | ~1.0 GB | TranscriptionWorker |
| faster-whisper-large-v3-turbo (Best Quality) | ~1.5 GB | TranscriptionWorker |
| Qwen 2.5 1.5B GGUF | ~1.1 GB | LLMWorker |
| NLLB-200 distilled 600M CT2 INT8 | ~600 MB | TranslationWorker |
| GLiNER2 ONNX | ~200 MB | NERWorker |
| pyannote-segmentation-3.0 ONNX | ~5.99 MB | OverlapDetector |
| YAMNet ONNX | ~15 MB | AudioClassifyWorker |

Total (Best Quality tier): ~3.4 GB  
Total (Fast tier): ~2.4 GB

## Acceptance Criteria

- [ ] User can pre-cache all models before going offline
- [ ] Visual progress feedback during download
- [ ] If a model download fails mid-way, retry is supported
- [ ] Cache check before model load: if not present, show clear error
      message with pre-fetch instructions, not a silent hang
- [ ] Documented in USER_GUIDE under "Offline Preparation"
