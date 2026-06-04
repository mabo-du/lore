# Deep Research Prompt 03 — pyannote-audio 3.x Speaker Diarization: Integration, Licensing, and UX

## Purpose

Lore includes optional speaker diarization powered by `pyannote-audio`. Diarization is deliberately opt-in due to the HuggingFace token requirement. This prompt requests comprehensive technical research covering the current state of pyannote-audio 3.x, its licensing friction, integration patterns for a desktop application, and practical accuracy expectations for oral history audio.

---

## Research Questions

### 1. pyannote-audio Current State (2025–2026)

- What is the **current stable version** of pyannote-audio? Has there been a major version change since 3.0 (the pipeline architecture rewrite)?
- What **pre-trained pipelines** are available on HuggingFace Hub? Specifically:
  - `pyannote/speaker-diarization-3.1` — is this still the recommended pipeline?
  - Are there any newer or better pipelines (e.g., 3.2, 3.3)?
  - Are there any **CPU-optimised** or lighter-weight variants available?
- What are the **hardware requirements** for running pyannote-audio on CPU-only hardware? Is GPU required, or is CPU inference practical?
  - What is the expected processing time for a 1-hour recording on CPU?
  - What is the RAM footprint during inference?
- What are the **pip dependencies**? Are there conflicts with `faster-whisper` or `torch`? What version of `torch` does pyannote-audio 3.x require?

### 2. HuggingFace Gating and Token Flow

- What is the **exact consent/gating process** a user must complete to use `pyannote/speaker-diarization-3.1`?
  - Must they create a HuggingFace account?
  - Must they accept a specific licence agreement on HuggingFace.co?
  - Must they generate a HuggingFace access token (read scope or write scope)?
- **Has the gating changed** since pyannote-audio 3.0 was released? Are the models still gated, or have they been made public?
- What is the **token format**? `hf_...` tokens — where are they stored and how are they passed to pyannote at runtime? Is `use_auth_token` the correct parameter, or has it been renamed to `token`?
- Can the token be stored **locally** (e.g., in `~/.cache/huggingface/token`) so the user only needs to enter it once? Or must it be passed on every pipeline instantiation?
- What happens if the user's token is **expired or revoked**? Does pyannote fail gracefully?

### 3. Pipeline Output — Data Model

- What does `pipeline(audio_file)` return? A `pyannote.core.Annotation` object?
  - How do you iterate over speaker turns from an `Annotation`?
  - What is the format of speaker labels — `SPEAKER_00`, `SPEAKER_01`, etc.?
  - What is the format of timestamps — seconds as float?
- How do you get the **number of detected speakers** from the output?
- Is there a way to **hint the number of speakers** (e.g., `num_speakers=2` for a two-person interview) to improve accuracy?
- Can the pipeline output be **aligned with faster-whisper segments**? What is the recommended algorithm for merging diarization timestamps with transcription timestamps?

### 4. Whisper + pyannote Alignment

- What is the **standard approach** for combining faster-whisper transcription output with pyannote diarization output?
  - Is it better to run both independently and then align by timestamp?
  - Are there any libraries (e.g., `whisperx`) that do this alignment automatically?
  - What is the alignment algorithm — interval overlap, nearest centroid, or something else?
- What happens when a speaker turn **starts mid-word** in the Whisper output? How should the segment be split or attributed?
- Is `whisperx` (which bundles diarization) a better option than separate faster-whisper + pyannote integration? What are the trade-offs?

### 5. Accuracy for Oral History Audio

- What is the **Diarization Error Rate (DER)** of pyannote-audio 3.x on:
  - Two-speaker interviews (the primary oral history use case)?
  - Multi-speaker panel recordings?
  - Telephone-quality or cassette-digitised recordings?
- What are the known **failure modes** for oral history audio?
  - Overlapping speech?
  - Long pauses between turns?
  - Background noise (field recordings)?
  - Speaker similarity (similar voice characteristics)?
- Are there any **lightweight alternatives** to pyannote-audio that have lower hardware requirements or no gating requirements? (e.g., `resemblyzer`, `nemo_toolkit`, `simple-diarizer`)

### 6. Privacy and Network Traffic

- Does `pyannote-audio` make **network calls during inference**? Specifically, does the pipeline phone home to HuggingFace during transcription, or only during initial model download?
- Lore's value proposition is total local processing. Can pyannote-audio be used in **fully air-gapped mode** after models are downloaded (i.e., no internet connection required during diarization)?
- If the user runs diarization with no internet connection, will pyannote fail with an error, or degrade gracefully?

### 7. Integration UX Patterns

- What is the **recommended UX flow** for an opt-in feature that requires a HuggingFace token?
  - Where should the token entry prompt appear in a desktop app?
  - Should Lore validate the token immediately (make a test API call) or only validate it when diarization is first attempted?
  - What error messages should be shown if the token is invalid, if the user has not accepted the model licence, or if the model download fails?
- Are there any **desktop application examples** (Python, Electron, or other) that have implemented a HuggingFace token setup flow that could serve as a reference?

---

## Context

**Project:** Lore — oral history transcription tool  
**Feature:** Optional speaker diarization (Phase 4, explicitly opt-in)  
**Users:** Non-technical archivists — the HuggingFace token flow must be explained clearly in plain language  
**Constraint:** Audio never leaves the machine after models are downloaded  
**Primary use case:** Two-person oral history interview (interviewer + interviewee)

---

## Deliverables Requested

1. Current pyannote-audio 3.x status: recommended pipeline, CPU feasibility, hardware requirements
2. Step-by-step HuggingFace gating flow (account → licence → token → usage)
3. Token storage and runtime usage pattern (code snippet)
4. Data model: how to iterate speaker turns from `Annotation` output
5. Recommended Whisper + pyannote alignment algorithm with code sketch
6. Accuracy expectations (DER) for two-speaker oral history interviews
7. Confirmation that pyannote runs fully air-gapped after download
8. Recommended alternatives if pyannote is unsuitable (lightweight, no gating)

---

*This research will directly inform `lore/src/diarization/pipeline.py` and the HuggingFace token setup flow in the settings UI (Phase 4).*
