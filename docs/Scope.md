

1. Lore — Oral History Transcription

Named after: the oral tradition itself — lore as in the accumulated knowledge, stories, and history passed down through speech. Fits the ecosystem's naming register alongside Libby, Fritts, and Trowel.

The Problem

Cultural anthropologists, oral historians, community archivists, and ethnographers record interviews and oral testimonies as audio files. Transcription is the single largest time sink in the oral history workflow: industry estimates run to four hours of typing per one hour of audio. Commercial transcription services (Rev, Otter, Sonix) are expensive and require uploading recordings to third-party servers — an ethical barrier when recordings contain sensitive cultural knowledge, endangered language material, or community-restricted content.

OpenAI's Whisper model (MIT licence, runs entirely locally) achieves near-human accuracy on clean speech, and faster-whisper (also MIT) reduces inference time by 3–4× over the reference implementation on CPU. No adequate GUI wrapper exists for this specific use case — existing Whisper frontends are generic; none produce OHMS XML, support speaker diarization, or are designed for the oral history archivist's workflow.

Who It's For
Cultural anthropologists, oral historians, community archivists, ethnographers, documentary filmmakers, endangered-language linguists, university special collections.

Tech Stack

Layer
Choice
Rationale
Language
Python 3.12+
Consistent with Trowel; best Whisper ecosystem
Transcription
faster-whisper (CTranslate2)
3–4× faster than vanilla; CPU-only; MIT
Speaker diarization
pyannote-audio (optional)
State of the art; requires one-time HF token
Audio I/O
ffmpeg subprocess
Universal format support (MP3/WAV/M4A/OGG/FLAC)
UI
PyQt6
Consistent with Trowel
Transcript editing
Custom QTextEdit + SRT sync
Click timestamp → jump in playback
Audio playback
pygame.mixer or sounddevice
Lightweight playback for review
Export
Plain text, SRT, OHMS XML
Standard formats for repository deposit
Packaging
PyInstaller
Models downloaded on first run, not bundled

Key Technical Challenges
Model distribution. Whisper models range from 75 MB (tiny) to 1.5 GB (medium) to 3 GB (large-v3). They cannot ship inside the installer. Lore should download models to a shared cache directory on first use — and that cache should be shared with HOARD if HOARD ever adopts Whisper for handwriting-to-audio crossover work. Let the user choose the model on first launch with a clear speed/accuracy trade-off explanation. Medium is the right default for most machines.

Long recordings. Oral history sessions can run 2–4 hours. faster-whisper handles long audio via chunked processing internally, but the UI must show meaningful progress (time elapsed, estimated remaining, chunk count) rather than a spinning indicator. Process in a QThread; never block the event loop.

Speaker diarization UX. pyannote-audio requires the user to accept a licence on HuggingFace and provide a token. This is non-trivial friction. Make diarization entirely opt-in with a clear setup guide in the UI. When disabled, the transcript is a flat text with timestamps; when enabled, each segment is labelled [Speaker A], [Speaker B], etc. Do not attempt to name speakers automatically — let the archivist rename them in the editor.

OHMS XML format. The Oral History Metadata Synchronizer format (University of Kentucky Libraries) is the deposit standard for oral history platforms (CONTENTdm, Omeka-S with Oral History plugin). The schema is publicly documented. An OHMS file wraps timestamped transcript segments in a metadata envelope (interviewee name, interviewer, date, collection, repository, rights statement). No Python library exists for this; implement a lightweight serialiser. This is 100–150 lines of straightforward XML generation.

Whisper hallucination on silence. Whisper is known to hallucinate text during long silences (it generates filler phrases like "Thank you for watching"). Apply voice activity detection (VAD) pre-processing using silero-vad (also MIT, lightweight) to strip silent regions before passing audio to Whisper. This dramatically reduces hallucinations and speeds up processing.

Privacy signalling. The entire value proposition includes that audio never leaves the machine. A persistent, prominent indicator in the UI ("🔒 Local only — no network access during transcription") builds trust. Verify in code that faster-whisper makes no outbound connections during inference (it does not, but assert it in tests).
Phase Outline
Phase
Description
0
POC: faster-whisper transcribing a 10-minute audio file on target hardware. Measure wall-clock time. Confirm VAD integration reduces hallucinations.
1
Audio ingest (file picker, ffmpeg normalisation, waveform display)
2
Transcription engine (model selector, QThread worker, progress reporting)
3
Transcript editor (SRT viewer, click-to-seek, inline editing, speaker label editing)
4
Speaker diarization (pyannote-audio, opt-in, HF token setup flow)
5
Export (TXT, SRT, OHMS XML, VTT)
6
Metadata form (interviewee, interviewer, date, collection, rights, language)
7
UI polish, dark theme, keyboard shortcuts
8
PyInstaller packaging, first-run model download flow, CI builds

Ecosystem Integration
HOARD: If HOARD ever processes audio field notes alongside paper context sheets, Lore's model cache and faster-whisper wrapper should be importable as a shared library rather than duplicated. Define a lore-core package separated from the UI.
Cache & Carry: Transcripts can be attached to oral history object records as text assets. Export a JSON sidecar compatible with Cache & Carry's media/event import format.
Trowel: Out of scope for Trowel's report pipeline, but interview excerpts produced by Lore could be attached to the documentary evidence sections of heritage impact reports.
Estimate
3–5 weeks. The core engine (Phases 0–3) is achievable in under two weeks given that faster-whisper does all the hard work. OHMS XML and diarization add time. The first-run model download UX and PyInstaller packaging are where unexpected time tends to go.
