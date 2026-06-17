<div align="center">
  <img src="https://img.shields.io/badge/Local_First-100%25-brightgreen.svg?style=for-the-badge" alt="Local First">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg?style=for-the-badge&logo=python" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/PyQt6-UI-blueviolet.svg?style=for-the-badge" alt="PyQt6">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="MIT License">
  <img src="https://img.shields.io/github/actions/workflow/status/mabo-du/lore/ci.yml?branch=main&style=for-the-badge" alt="CI">
  <img src="https://img.shields.io/pypi/v/lore-ai.svg?style=for-the-badge" alt="PyPI">

  <h1>Lore 🎙️</h1>

  <p><strong>Privacy-First, Local-Only Oral History Transcription & Archiving</strong></p>
</div>

---

**Lore** is a desktop application designed for historians, archivists, and researchers. It provides state-of-the-art AI transcription, speaker diarization, named entity recognition, and translation—**100% offline, on your own hardware.**

No data leaves your computer. No cloud subscriptions. Just powerful, open-source AI packaged into a clean, intuitive PyQt6 interface.

<img src="https://raw.githubusercontent.com/mabo-du/lore/main/docs/images/lore_main.png" alt="Lore Main Window" width="800">

## ✨ Features

- 🎧 **Offline Transcription:** Powered by `faster-whisper`, optimized for CPU inference with low memory overhead (< 8GB RAM).
- 🗣️ **Speaker Diarization:** Automatically identifies and labels different speakers using `pyannote.audio` or `Resemblyzer`, with configurable speaker count (1–20).
- 🔀 **Overlapping Speech Detection:** Flags when multiple speakers talk simultaneously via a lightweight ONNX segmentation model (6 MB). Overlap regions are visualised on a dedicated strip and per-segment badges in the transcript view.
- ⚡ **Model Tier Selection:** Choose between Fast (small), Balanced (medium), or Best Quality (large-v3-turbo) Whisper models in Settings.
- 🔍 **Word-Level Confidence:** Low-confidence words are visually highlighted so you can quickly spot potential hallucinations.
- 🌍 **Local Translation:** Translate transcripts to over 200 languages completely offline using Meta's `NLLB-200` model.
- 📖 **Custom Vocabulary:** Provide local jargon, proper nouns, and historical terms to guide Whisper's decoding graph for maximum accuracy.
- 🏷️ **Named Entity Recognition:** Uses `GLiNER` to automatically extract people, organizations, dates, and locations.
- 📦 **Archival Exporting:** Export your work to the **OHMS XML** format or create an **RFC 8493 BagIt** archival package with SHA-256 checksum verification.
- 🔎 **Global Archive Search:** A unified SQLite database (`FTS5` + `sqlite-vec`) lets you instantly search across all your past projects using keyword or semantic/conceptual search.

## 🚀 Installation

### Option 1: Pre-built Installers (Recommended)

Download the installer for your platform from the [latest release](https://github.com/mabo-du/lore/releases/latest):

| Platform | Installer |
|----------|-----------|
| 🪟 **Windows** | `lore-windows-x86_64.zip` — Extract and run `lore.exe` |
| 🍎 **macOS** | `lore-macos-arm64.tar.gz` — Extract and run `lore` |
| 🐧 **Linux** | Install via PyPI (`pip install lore-ai`) or [build from source](#option-3-install-from-source) |

### Option 2: Install from PyPI

```bash
pip install lore-ai
lore
```

### Option 3: Install from Source

Lore requires **Python 3.12+** and is cross-platform (Windows, macOS, Linux).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mabo-du/lore.git
   cd lore
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the application:**
   ```bash
   pip install -e .
   ```

## 🎮 Usage

Start the Lore application:

```bash
lore
```

Or launch from your system's application menu if installed via the pre-built installer.

1. **Select an Audio File:** Click "Browse" to select any standard audio format (WAV, MP3, M4A, OGG, FLAC).
2. **Configure Settings:** Click the ⚙️ Settings icon to set your Custom Vocabulary and speaker diarization preferences.

   <img src="https://raw.githubusercontent.com/mabo-du/lore/main/docs/images/lore_settings.png" alt="Lore Settings" width="400">

3. **Transcribe & Diarize:** Click "Transcribe" on the toolbar. If recording has multiple speakers, check the "Enable Speaker Diarization" box.
4. **Edit & Review:** Play the audio, click on segments to edit them, and review any low-confidence words highlighted in red.
5. **Translate:** Select a target language from the dropdown and click "Translate" for fully offline translation.
6. **Export:** Fill out the Metadata panel and export to **OHMS XML** or an Archival **BagIt Package**.

For detailed instructions, see the [User Guide](USER_GUIDE.md).

## 🏗️ Architecture

Lore is designed with strict sequential memory management to run on older hardware.
- Models are loaded into memory one at a time (e.g., Whisper loads, transcribes, unloads → NLLB loads, translates, unloads).
- Heavy use of CTranslate2 (INT8 quantization) ensures models run blazingly fast without needing a dedicated GPU.
- The UI runs asynchronously using PyQt6's `QThread` and Signals, keeping the interface completely responsive during heavy AI workloads.

## 🤝 Contributing

Lore is an open-source project. We welcome pull requests, bug reports, and feature requests. Please see our [User Guide](USER_GUIDE.md) for more detailed workflows and documentation on the codebase.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
