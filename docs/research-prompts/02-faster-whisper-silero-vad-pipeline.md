# Deep Research Prompt 02 — faster-whisper + silero-VAD Pipeline: Technical Integration for Long-Form Audio

## Purpose

Lore uses `faster-whisper` (CTranslate2 backend) as its transcription engine and `silero-vad` for voice activity detection pre-processing. This prompt requests comprehensive technical research covering the specific integration challenges, performance characteristics, and best practices for building a production-quality transcription pipeline for oral history audio (sessions of 1–4 hours, single or dual speakers, variable recording quality).

---

## Research Questions

### 1. faster-whisper — Current State (2025–2026)

- What is the **current stable version** of `faster-whisper` and what significant changes have been made since v1.0?
- What are the **available model sizes** (tiny, base, small, medium, large-v2, large-v3, large-v3-turbo, distil-*)? Provide a table with:
  - Model name
  - Approximate disk size
  - Recommended VRAM (GPU) / RAM (CPU)
  - Expected real-time factor on CPU (e.g., 0.5× = processes 1 hour in 30 minutes)
  - Word Error Rate (WER) on standard benchmarks
- What is the **recommended model for CPU-only deployment** on mid-range hardware (8–16 GB RAM, no GPU)? Is `large-v3-turbo` or a distilled model now preferred over `medium` for CPU work?
- Does `faster-whisper` require **CUDA, cuDNN**, or any GPU libraries when running CPU-only? What are the exact pip dependencies for a pure-CPU install?
- What is the `compute_type` parameter? Which value (`int8`, `int8_float16`, `float16`, `float32`) gives the best speed/accuracy trade-off on CPU?

### 2. silero-VAD — Integration with faster-whisper

- What is the **current recommended approach** for using silero-VAD as a pre-processing step before faster-whisper?
  - Does silero-VAD run on the raw audio waveform or on a resampled version?
  - What sample rate does silero-VAD require?
  - What is the output of silero-VAD — a list of timestamp intervals, a mask array, or something else?
- How should the VAD output be used to **strip silence** before passing audio to faster-whisper? Is it better to:
  a. Remove silent chunks and concatenate the remaining audio into a new file?
  b. Pass the VAD timestamps to faster-whisper's `vad_filter=True` parameter?
- Does `faster-whisper` have a **built-in VAD option** (`vad_filter` parameter)? If so, does it use silero-VAD internally? Is the built-in VAD sufficient, or does pre-processing with silero-VAD separately give better results?
- What are the **known failure modes** of silero-VAD on oral history audio (e.g., quiet speech, background music, overlapping speakers, telephone-quality recordings)?
- What `torch` or `onnxruntime` version is required for silero-VAD? Are there dependency conflicts with `faster-whisper`?

### 3. Long-Form Audio Processing

- faster-whisper processes audio internally in **30-second chunks** (the Whisper context window). How does it handle recordings longer than 30 seconds? Is there anything a developer needs to do explicitly for 2–4 hour recordings, or does it handle them automatically?
- What is the **hallucination problem** for long recordings? Specifically:
  - What types of hallucinations occur (repeated phrases, "Thank you for watching", filler text)?
  - Does silero-VAD pre-processing reliably prevent these, or do hallucinations still occur after VAD?
  - Are there any faster-whisper parameters (`condition_on_previous_text`, `no_speech_threshold`, `log_prob_threshold`, `compression_ratio_threshold`) that help reduce hallucinations? What are the recommended values?
- For a **4-hour oral history session**, what is the expected wall-clock transcription time on:
  - A modern Intel i7/i9 CPU with 16 GB RAM using the `medium` model?
  - The same hardware using `large-v3`?
  - An Apple M1/M2 (if relevant)?
- What is the **memory footprint** of faster-whisper during inference on a 4-hour file? Does it load the entire audio file into RAM, or stream it?

### 4. Progress Reporting

- How can a developer implement **meaningful progress reporting** during transcription? The scope doc requires showing "time elapsed, estimated remaining, chunk count."
  - Does faster-whisper expose a progress callback or generator that yields results per chunk?
  - What does `transcribe()` return — a generator of segment objects, or a single result after the entire file is processed?
  - How can chunk number and estimated remaining time be derived from the generator output?
- What is the correct pattern for running faster-whisper in a **QThread** (PyQt6) without blocking the event loop? Provide the signal/slot pattern for progress updates.

### 5. ffmpeg Normalisation

- What is the **recommended ffmpeg command** for normalising oral history audio before transcription? Cover:
  - Converting from MP3/M4A/OGG/FLAC/WAV to a format faster-whisper accepts
  - Sample rate (16kHz is required by Whisper — confirm)
  - Mono conversion (is stereo supported, or must it be mixed to mono?)
  - Volume normalisation (loudnorm filter vs simple gain — which is better for speech?)
- How should Lore handle **ffmpeg not being installed**? What is the best way to detect ffmpeg and guide the user to install it?

### 6. Model Download and Caching

- Where does faster-whisper cache downloaded models by default? What environment variable controls the cache path?
- Can the cache path be **overridden programmatically** so Lore can point to a shared directory (shared with HOARD if HOARD ever adopts Whisper)?
- What is the download protocol — does it use the Hugging Face Hub API, direct HTTPS, or something else? Are there proxy/firewall issues to be aware of?
- Can models be **pre-downloaded** and shipped as part of a PyInstaller build, or is the disk size prohibitive? What is the minimum viable model for a bundled install?

### 7. Whisper Word-Level Timestamps

- Does faster-whisper support **word-level timestamps** (not just segment-level)? If so, what parameter enables this?
- For the OHMS XML export, are word-level timestamps useful, or is segment-level (sentence/phrase level) sufficient?
- How should **punctuation** be handled — does faster-whisper add punctuation, or does the developer need to post-process?

---

## Context

**Project:** Lore — oral history transcription tool  
**Platform:** Desktop (Linux, macOS, Windows), CPU-only, no GPU requirement  
**Users:** Non-technical archivists and oral historians — no command-line access  
**Audio:** MP3/WAV/M4A/OGG/FLAC, 1–4 hours, single or dual speakers, variable recording quality  
**Privacy:** Audio must never leave the machine. No cloud APIs during transcription.

---

## Deliverables Requested

1. Model comparison table (size, RAM, CPU speed, WER)
2. Recommended model for CPU-only mid-range hardware in 2026
3. Definitive answer on built-in VAD vs external silero-VAD pre-processing
4. Recommended hallucination-reduction parameter set for long-form oral history audio
5. Progress reporting pattern for faster-whisper in a PyQt6 QThread
6. ffmpeg normalisation command for Lore's audio pipeline
7. Model cache path control — how to point to a shared directory

---

*This research will directly inform `lore/src/transcription/engine.py`, `lore/src/transcription/worker.py` (QThread), and `lore/src/audio/normalise.py`.*
