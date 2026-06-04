# Deep Research Prompt: Phase 14 — Multilingual Translation via NLLB-200 for Lore

## Context

Lore is an offline, local-first PyQt6 desktop application for oral history transcription. It targets archivists working with sensitive materials (indigenous testimonies, post-conflict narratives, institutional oral histories). The app currently features:

- **Faster-Whisper** (CTranslate2) for ASR
- **Pyannote** for speaker diarization
- **GLiNER2-ONNX** for named entity recognition
- **Qwen2.5-1.5B** (via llama-cpp-python) for abstract generation
- **YAMNet** (ONNX) for audio event detection
- **Whisper logprob-based** anomaly detection for confidence scoring

### Hard Constraints
- **No PyTorch dependencies** — the entire app must run without torch
- **PyInstaller compatible** — must bundle into a `--onedir` package
- **Sub-400MB initial bundle size** (models downloaded on demand via HuggingFace Hub)
- **Completely offline** — zero network access during operation
- **≤8GB RAM target** — must not OOM on consumer hardware
- **CTranslate2 is already bundled** as the Faster-Whisper backend

## Feature Description

We want to add offline multilingual translation to Lore. The user transcribes an interview in Language A, then clicks a button to generate a synchronised translation into Language B. The translated output is mapped to OHMS XML 6.0's bilingual sync track schema.

### Architecture Decisions Already Made
- **Inference engine:** CTranslate2 (already bundled with Faster-Whisper — zero additional engine size)
- **Model:** NLLB-200-600M distilled variant, converted to CTranslate2 format with INT8 quantisation
- **Alignment strategy:** Segment-level timestamp inheritance (translated text inherits the source segment's start/end times)
- **OHMS XML mapping:** Use `<vtt_transcript_alt>` for the translated track, plus `Include Translation` and `Language for Translation` metadata flags
- **Model delivery:** Downloaded on demand via ModelManager (not bundled in PyInstaller package)

### Priority Languages
The following language corridors are highest priority for oral history archiving:
- **UNESCO endangered:** Australian Aboriginal languages (Pitjantjatjara, Yolŋu Matha), te reo Māori, Native American (Navajo, Cherokee), Welsh, Irish Gaelic
- **High-demand community:** Spanish, Mandarin Chinese, Arabic

## Research Questions

### 1. CTranslate2 + NLLB-200 Integration
- What is the exact process for converting NLLB-200-600M from HuggingFace format to CTranslate2 format? Provide the exact `ct2-opus-mt-converter` or `ct2-fairseq-converter` command, or whatever tool is appropriate.
- Does the CTranslate2 conversion process itself require PyTorch? If so, can we perform the conversion offline (as a one-time build step) and distribute only the converted model?
- What is the disk size of the NLLB-200-600M model after CTranslate2 INT8 conversion?
- Does CTranslate2's NLLB support require the `sentencepiece` tokenizer? If so, what is its size and does it have any PyTorch transitive dependencies?
- Can we load Faster-Whisper's CTranslate2 instance AND the NLLB CTranslate2 instance simultaneously in the same process, or will they fight over shared C++ state? How do we manage the memory of having two CTranslate2 models loaded?

### 2. Translation Quality for Priority Languages
- What is NLLB-200-600M's BLEU/chrF score for translations involving low-resource languages in our priority list? Specifically:
  - Pitjantjatjara ↔ English
  - Yolŋu Matha ↔ English
  - te reo Māori ↔ English
  - Navajo ↔ English
  - Cherokee ↔ English
  - Welsh ↔ English
  - Irish Gaelic ↔ English
- For any languages where quality is known to be poor, what are the alternatives? Are there community-trained models that could supplement NLLB?
- Should we display a quality confidence indicator per language (e.g., "High quality", "Experimental") in the UI? What data exists to calibrate this?
- Are there any of the 200 NLLB languages that are so poor quality they should be excluded entirely to avoid misleading archivists?

### 3. Performance Benchmarking
- What is the expected tokens-per-second throughput of NLLB-200-600M (INT8, CTranslate2) on a mid-range consumer CPU (Intel i5-12400, 8GB RAM)?
- For a typical 1-hour oral history interview (~8,000 words / ~10,000 tokens), what is the estimated total translation time?
- Is the sub-5-minute target for a 1-hour transcript realistic, or should we position this as a background task?
- What is the peak RAM usage when NLLB-200-600M (INT8) is loaded alongside Faster-Whisper large-v3-turbo (INT8)? Will this cause OOM on 8GB systems?
- Should we unload the Whisper model before loading NLLB to stay within memory limits? What is the CTranslate2 model load/unload latency?

### 4. OHMS XML 6.0 Bilingual Schema
- Provide the exact OHMS XML 6.0 schema structure for bilingual transcripts. What elements are required beyond `<vtt_transcript_alt>`?
- Is `<vtt_transcript_alt>` a standard element in the publicly available OHMS XSD, or is it only documented in the OHMS User Guide? Provide the XSD reference if possible.
- When the OHMS Viewer renders a bilingual transcript, does it expect the `<vtt_transcript>` and `<vtt_transcript_alt>` to have identical cue timestamps? Or can the translated track have different timing?
- Are there any working examples of bilingual OHMS XML files we can reference? Provide URLs or code samples.
- What are the `Include Translation`, `Language`, and `Language for Translation` field values expected by Aviary/Omeka OHMS Embed module?

### 5. Language-Specific Model Pruning
- The research identified "Memory-efficient NLLB-200" (arXiv:2212.09811) which uses language-specific expert pruning. Is there an implementation available that works with CTranslate2?
- Could we create per-language-pair pruned models (e.g., a 200MB Māori↔English model instead of the full 600M model)?
- What is the accuracy loss from aggressive pruning for low-resource language pairs?
- Is it more practical to ship one full NLLB-200-600M model, or multiple pruned per-pair models? What are the storage trade-offs?

### 6. UI/UX Considerations
- How should we handle translation of segments that contain code-switching (speaker switches between languages mid-sentence)?
- Should we translate the entire transcript at once, or allow per-segment translation?
- How do we handle segments that the anomaly detector has flagged as low-confidence — should we skip translation for those, or translate with a warning?

## Deliverables

Please provide:
1. A step-by-step CTranslate2 conversion guide for NLLB-200-600M (including the exact commands)
2. Translation quality assessment for each priority language pair
3. Performance benchmarks or estimates for INT8 CPU inference
4. Complete OHMS XML 6.0 bilingual schema reference with a working sample file
5. Memory management strategy for running Whisper + NLLB simultaneously on 8GB RAM
6. A recommendation on whether to ship a single model or per-language pruned models
