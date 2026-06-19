# Lore User Guide

Welcome to Lore! This guide will walk you through the end-to-end process of transcribing, editing, and archiving oral histories completely offline.

## Table of Contents
1. [Installation](#installation)
2. [Getting Started](#getting-started)
3. [Configuring Custom Vocabulary](#configuring-custom-vocabulary)
4. [Offline Preparation](#offline-preparation)
5. [Transcription & Diarization](#transcription--diarization)
6. [Translation](#translation)
7. [Editing and Review](#editing-and-review)
   - [Overlapping Speech](#overlapping-speech)
8. [Metadata & Archival Export](#metadata--archival-export)
9. [Global Search](#global-search)

---

## Installation

### Pre-built Installers

The easiest way to install Lore is via the pre-built installers available on the [GitHub Releases page](https://github.com/mabo-du/lore/releases/latest):

| Platform | File | Instructions |
|----------|------|--------------|
| **Windows** | `lore-windows-x86_64.zip` | Extract the zip and run `lore.exe`. |
| **macOS** | `lore-macos-arm64.tar.gz` | Extract and run `./lore` from Terminal. |
| **Linux** | Install via PyPI (`pip install lore-ai`) or [build from source](#from-source). |

### PyPI

```bash
pip install lore-ai
lore
```

### From Source

Requires Python 3.12+.

```bash
git clone https://github.com/mabo-du/lore.git
cd lore
python -m venv .venv
source .venv/bin/activate
pip install -e .
lore
```

---

## 2. Getting Started

<img src="https://raw.githubusercontent.com/mabo-du/lore/main/docs/images/lore_main.png" alt="Lore Main Window" width="800">

When you launch Lore, you will be greeted by the **File Picker**.
- Click **Browse** to locate your audio file. Lore supports common formats like `.wav`, `.mp3`, `.m4a`, `.ogg`, and `.flac`.
- The application will automatically normalize the audio in the background to ensure it is in the optimal format (`16kHz`, mono) for AI transcription.
- Once normalization is complete, the main editor interface will open.

## 3. Configuring Custom Vocabulary

<img src="https://raw.githubusercontent.com/mabo-du/lore/main/docs/images/lore_settings.png" alt="Lore Settings" width="400">

Before transcribing, it's highly recommended to provide Lore with contextual terms.

1. Click the **⚙️ Settings** button on the top right of the toolbar.
2. In the **Custom Vocabulary** field, enter a comma-separated list of terms that are specific to your recording.
   - *Example: "Nunn Center, oral history, John Doe, Appalachia"*
3. Select your **Model Quality Tier**: Fast (small model, lower accuracy), Balanced (medium), or Best Quality (large-v3-turbo, highest accuracy). Best Quality is the default and recommended for oral history work.
4. These terms are passed to the AI as an "initial prompt." The AI will use this prompt to heavily bias its decoding towards spelling these localized terms correctly, without slowing down the transcription speed.
5. Set the **Number of Speakers** (1–20) expected in the recording. Leave at 0 for automatic detection (GMM-BIC estimation). The default of 2 works for most oral history interviews. Increase for panel discussions or group recordings.

## 4. Offline Preparation

Lore downloads AI models on first use. If you're heading to the field without reliable internet, pre-cache everything beforehand:

```bash
# Pre-fetch all models for the default tier (from Settings)
lore --prefetch-models

# Pre-fetch for a specific model tier
lore --prefetch-models --tier "Best Quality"

# Check what's cached
lore --prefetch-models --check

# Switch to offline mode (fails fast on missing models)
lore --offline
```

Both `--prefetch-models` and normal Lore operation respect `--offline`: when set, any missing model raises an immediate error instead of hanging while `huggingface_hub` times out trying to reach the internet.

**Model cache location:** `~/.local/share/heritage-tools/`

## 5. Transcription & Diarization

With your audio loaded and vocabulary set, you're ready to extract the text.

1. **Diarize (Speaker Identification):** If your audio features multiple speakers, check the **Diarize** checkbox on the toolbar. The AI will analyze the audio to separate and tag who is speaking (e.g., `SPEAKER_00`, `SPEAKER_01`).
2. **Transcribe:** Click the **Transcribe** button. Lore will begin processing the audio.
   - *Note: Lore runs entirely on your CPU. Transcription speed depends on your processor. For a 1-hour file on a modern CPU, it typically takes 10-20 minutes.*
3. As segments are processed, they will appear in real-time in the main transcript window.

## 6. Translation

Lore includes a massive, offline translation engine capable of translating between 200+ languages.

1. Once your English (or source language) transcription is complete, locate the **Translate to:** dropdown on the toolbar.
2. Select your desired target language.
3. Click **Translate**. The AI will generate a translated version of each segment. The original text will remain available, allowing you to export dual-language (bilingual) archives.
4. To cancel an in-progress translation, click the **✕ Cancel** button that appears next to the Translate button.

## 7. Editing and Review

AI is powerful, but it makes mistakes. Lore provides tools to quickly review and correct errors.

- **Word-Level Confidence:** Lore scores every single word the AI produces. If the AI was less than 60% confident about a word, it will be highlighted with a **pale red underline**. This allows you to instantly scan a massive transcript for potential hallucinations or tricky auditory moments.
- **Volume Control:** Use the volume slider (🔊) next to the Play button to adjust playback volume.
- **Audio Sync:** Click on any text segment in the transcript, and the audio waveform at the top of the screen will instantly seek to that exact timestamp. Press the Play button to listen.
- **Editing:** Double-click any segment to modify the text or correct a speaker label.
- **Load a New File:** Click the **📂 New File** button on the toolbar to return to the file picker and open a different audio file without restarting the app.

### Overlapping Speech

Lore automatically detects when two or more speakers talk simultaneously using a lightweight ONNX model (6 MB, no GPU required). Overlap information is presented in two ways:

- **Overlap strip** — a thin horizontal bar between the waveform and the transcript view shows coloured blocks where overlap was detected. Click any block to jump directly to that moment in the transcript.
- **Segment badges** — transcript segments that overlap with a detected region display a subtle `⟪ overlap ⟫` badge and a warm amber left-border indicator. Overlaps in oral history are common (backchannels, affirmations), so the visual weight is kept low and informational.

Overlap detection runs automatically during transcription. No configuration needed.

### Backchannel Data Logging

Lore can optionally log rule-based backchannel detection decisions to a local JSONL
file at `~/.local/share/heritage-tools/backchannel-log.jsonl`. This data is intended
to train a future acoustic backchannel classifier (Stage 2). The feature is enabled by
default and can be disabled in **Settings → Offline & Pre-fetch → Backchannel data
logging (local only)**.

**Privacy:** The log is purely local — no data ever leaves your machine. It records
only timestamps, durations, and text of transcribed segments that the rule classifier
evaluated, along with whether each was classified as a backchannel. No audio, no
speaker embeddings, and no network requests are involved.

## 8. Metadata & Archival Export

Lore is built for archival standards. On the right side of the screen, you will find the **Metadata Form**.

1. Fill out the required OHMS XML fields (Interview Title, Repository Name, Interviewee, format, rights, etc.). You can optionally set a **Record ID (CMS Ref)** to identify this interview in your content management system. If left blank, a random ID is generated on export.
2. **Auto-Generate Abstract:** Click the purple `✨ Auto-Generate Abstract` button to have Lore's local AI (Qwen 2.5) read the entire transcript and generate a concise, professional summary for your archives.
3. **Export Options:**
   - **Export OHMS XML:** Saves a raw `transcript.xml` file conforming to the OHMS XML 6.0 schema, including your transcript, translation, metadata, and auto-extracted named entities.
   - **📦 Export BagIt Package:** Generates a full RFC 8493 compliant folder structure. It packages your original audio and the OHMS XML together, generating SHA-256 cryptographic checksums to guarantee data integrity for long-term cold storage.

## 9. Global Search

Lore automatically indexes every transcript you export into a global, unified database.

1. Click the **🔍 Global Search** button on the toolbar.
2. **Exact/Keyword Search:** Instantly find exact phrases across all your past projects.
3. **Semantic/Concept Search:** Powered by local vector embeddings. Search for concepts like "space exploration" and find segments where someone is talking about "NASA" or "rockets" even if the word "space" was never explicitly spoken.
4. The search results will show the exact timestamp and snippet.
