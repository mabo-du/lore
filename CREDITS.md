# Credits

Lore uses the following open-source models and libraries. Attributions are
provided as required by their respective licenses.

## Models

### WeSpeaker ResNet34-LM (ONNX)
- **Source:** `onnx-community/wespeaker-voxceleb-resnet34-LM` (Hugging Face)
- **License:** CC-BY-4.0 / Apache-2.0
- **Used for:** Speaker embedding extraction in the ONNX diarization pipeline
- **Attribution:** When distributing Lore, include a notice that this model
  is derived from the WeSpeaker project (Apache-2.0). See
  https://github.com/wenet-e2e/wespeaker for details.

### Silero VAD (ONNX)
- **Source:** `onnx-community/silero-vad` (Hugging Face)
- **License:** MIT
- **Used for:** Voice Activity Detection in the ONNX diarization pipeline

### pyannote-segmentation-3.0 (ONNX)
- **Source:** `onnx-community/pyannote-segmentation-3.0` (Hugging Face)
- **License:** MIT (community ONNX export)
- **Used for:** Overlapping speech detection

## Reference implementations

### diarize library
- **Source:** https://github.com/ (CPU-only speaker diarization)
- **License:** Apache-2.0
- **Used as:** Architecture reference for GMM-BIC speaker-count estimation
- **Notice:** Portions of the clustering logic (`estimate_num_speakers`) are
  adapted from the diarize library approach, used under Apache-2.0. The
  original copyright and license notice is preserved in the source files.

## Previous backends (removed in v0.1.7)

### Resemblyzer
- **Used for:** Speaker embedding (replaced by WeSpeaker ONNX)
- **License:** MIT

### pyannote.audio 3.1
- **Used for:** Speaker diarization pipeline (replaced by ONNX path)
- **License:** MIT (gated, requires HuggingFace token acceptance)
