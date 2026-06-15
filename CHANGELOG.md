# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] — 2026-06-15

### Added
- **Model Quality Tier selector** in Settings dialog — choose Fast (small), Balanced (medium), or Best Quality (large-v3-turbo) Whisper model tier; saved to QSettings and respected by both auto-load and manual re-transcription paths
- **Number of Speakers** spinbox in Settings (range 1-20) — replaces the hardcoded `KMeans(n_clusters=2)` in the Resemblyzer diarisation path; respected across Pyannote and Resemblyzer engines
- **Volume slider** in player controls (0-100 range, default 80) — wired to `AudioPlayer.set_volume()`, previously implemented but never connected to a UI element
- **📂 New File** button on editor toolbar — stops all running workers, clears state, and returns to the file picker so users can open a different file without restarting the app
- **✕ Cancel** button for translations — visible during in-progress translations, calls worker `terminate()` and returns to the editor
- **Record ID (CMS Ref)** optional field in the metadata form — exported to OHMS XML as `record/@id` instead of the hardcoded `"1"`; falls back to a random UUID if left empty
- **Rights field** now editable with expanded licence options — changed from a non-editable `QComboBox` to an editable one with CC BY 4.0, CC BY-SA 4.0, CC BY-NC 4.0, CC BY-NC-SA 4.0, CC0, Copyrighted, and Restricted Access
- **Auto-tagging (RAGWorker)** — the fully implemented domain taxonomy auto-tagging thread is now started automatically after transcription finishes (if a taxonomy database is available)
- **OS keyring support** in `token_vault.py` — HuggingFace tokens can now be stored in the system keyring via the `keyring` library, with Fernet-based MAC-address encryption as fallback
- **Expanded WHISPER_TO_NLLB_MAP** — from 35 to 87 language code mappings, covering European, Asian, Middle Eastern, African, Pacific, and Indigenous languages; logs a warning before defaulting to English for unmapped codes
- **Global search now threaded** — `SearchWorker` QThread prevents UI freeze during vector inference; dialog shows "Searching…" indicator while results are computed

### Fixed
- **BUG-01 (crash):** `MinMaxDownsampler.downsample()` now receives the required `x` (arange) parameter — previously passed only `y` and `n_out`, causing a `TypeError` on every file load
- **BUG-02 (UI freeze):** FFmpeg audio normalisation moved from `QTimer.singleShot` to a dedicated `AudioLoadWorker` QThread — the main thread no longer blocks for potentially many seconds
- **BUG-03 (crash):** `waveform_widget.py` `load_audio()` now wraps `wave.open()` in try/except — corrupt WAV files no longer crash the app
- **BUG-04 (stuck UI):** `btn_transcribe` is re-enabled in both `_on_transcription_finished()` and `_on_transcription_error()` — the button was permanently greyed out after a single transcription
- **BUG-05 (stuck UI):** Transcription errors now show a `QMessageBox.critical()` popup and automatically redirect back to the file picker page so the user can try again
- **BUG-06 (silent wrong behaviour):** The editor toolbar diarization checkbox is synced from QSettings saved preferences when the editor page becomes visible — previously the file-picker state was silently ignored on re-transcription
- **BUG-07 (race condition):** `start_transcription()` now stops any running workers (`.stop()`, `.quit()`, `.wait()`) before creating new ones — prevents stale threads from emitting into overwritten state
- **BUG-08 (wrong output):** `TranslationWorker` now receives the transcript's detected source language mapped through `WHISPER_TO_NLLB_MAP` — NLLB previously always received `eng_Latn` regardless of the actual source language
- **BUG-10 (deprecation):** `Pipeline.from_pretrained(use_auth_token=…)` changed to `token=…` — resolves a `FutureWarning` from pyannote-audio 3.x
- **BUG-11 (thread leak):** `NERWorker` thread is explicitly stopped with `ner_worker.stop() + wait()` in `_on_transcription_finished()` — previously lived for the entire app session
- **BUG-12 (data bloat):** Orphaned vector rows in `segments_vec` are now cleaned up before re-indexing — previously accumulated dead rows due to vec0 not supporting `DELETE WHERE`
- **BUG-13 (UI freeze):** `GlobalSearchIndex` model loading is now lazy — initialised on first search rather than at dialog construction, avoiding a multi-second synchronous load
- **STUB-01 (hardcoded value):** OHMS record ID is now generated from metadata or UUID rather than always `"1"`
- **STUB-02 (wrong output):** OHMS `<keywords>` and `<subjects>` are now populated separately — keywords from person/organisation/location entities, subjects from taxonomy tags
- **STUB-03 (fragile):** YAMNet class indices are now loaded dynamically from `yamnet_class_map.csv` with hardcoded fallback — resilient to CSV format changes
- **STUB-04 (fragile):** BagIt `Payload-Oxum` file count computed dynamically from actual payload files — previously hardcoded to `2`

### Changed
- **Author field** in `pyproject.toml` updated from `"Digital Heritage Lab"` to `"Mark Bouck"`
- **Entry point** changed from `src.main:main` to `main:main` — matches the src-layout package discovery in `[tool.setuptools.packages.find] where = ["src"]`
- **Taxonomy embedding model** consolidated to `BAAI/bge-small-en-v1.5` (same as `global_search.py`) — halves RAM usage by sharing the in-memory model
- **Dependencies:** Added `scikit-learn>=1.3`, `onnxruntime>=1.17`, `ctranslate2>=4.0`, `transformers>=4.40`, `huggingface-hub>=0.22`, `keyring>=25.0` to main deps; `pytest-qt>=4.4`, `pytest-mock>=3.12` to dev deps; removed duplicate `pyannote-audio` / `Resemblyzer` from optional deps
- **Token vault** now prefers OS keyring over Fernet MAC-address encryption — tokens remain decryptable across network adapter changes
- **`exporters/__init__.py`** populated with a package docstring (previously empty — reserved for future refactoring)

## [0.1.2] — 2026-05-28

### Fixed
- Package name corrected from `lore-oral-history` to `lore-ai` for PyPI publishing

## [0.1.1] — 2026-05-28

### Added
- Initial public release
- Offline transcription via faster-whisper
- Speaker diarization via Pyannote and Resemblyzer
- Named Entity Recognition via GLiNER
- Local translation via NLLB-200
- OHMS XML and BagIt archival exports
- Global search with FTS5 and semantic vector search
- Custom vocabulary injection for Whisper
- Word-level confidence scoring
