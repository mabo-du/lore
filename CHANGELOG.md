## [0.1.10] — 2026-06-19

### Added
- **Speaker rename** — Right-click any segment → "Rename Speaker" → rename all segments with that label at once.
- **Copy segment text** — Right-click → "Copy Segment Text" → clipboard.
- **Space to play/pause** — Space bar toggles audio playback.
- **Export TXT** — Save dialog now offers plain text alongside OHMS XML.
- **Word click-to-seek** — Click any word in the transcript to jump the audio to that exact moment.
- **Segment merge** — Right-click → "Merge with Next Segment" to join adjacent segments.
- **Auto-save transcript** — Segments auto-save as `.transcript.json` alongside the audio. Re-opening the same file reloads them instantly.
- **Time format toggle** — Right-click → toggle between `HH:MM:SS` and `MM:SS.mmm`.

### Fixed
- **App crash on right-click** — Time format toggle referenced wrong class constant; fixed to use delegate.
- **NER model loading failure** — `gliner2-onnx` requires `transformers>=5.0.0` for `TokenizersBackend`. Pinned in `pyproject.toml`.
- **CI lint issues** — All remaining lint warnings resolved (unused imports, `if not entities: return` one-liner, gitignore `models/` pattern anchored to root).

### Changed
- **Transcription diarization** — Segment-based embedding extraction replaces VAD-based segmentation. KMeans replaces SpectralClustering for small segment sets. Minimum embedding segment raised to 1s.
- **VAD silence threshold** — Reduced from 500ms to 300ms for better turn separation.

## [0.1.9] — 2026-06-19

### Fixed
- **Speaker diarization collapse** — Replaced VAD-based segmentation with segment-based
  embedding extraction. Speaker labels now come directly from Whisper segment boundaries
  instead of separate VAD regions, fixing the issue where all speakers collapsed to
  SPEAKER_00 after the first few segments. KMeans clustering replaces SpectralClustering
  for more reliable separation on small segment sets.
- **VAD noise filtering** — Minimum speech segment increased from 100ms to 1s, filtering
  false positives from breaths and clicks.
- **App crash on background launch** — (Investigating persistent DISPLAY=:0 segfault)

### Changed
- **Segment-based diarization** — `DiarizationEngine.run_diarization()` now accepts an
  optional `segments` parameter. When provided (from transcription), speaker embeddings
  are extracted directly from each segment's audio slice instead of running a separate
  VAD pass, improving alignment and accuracy.
- **KMeans for small segment sets** — Switched from SpectralClustering to KMeans for
  the segment-based path, which performs better on the 5-20 segments typical of a
  single transcription session.

## [0.1.8] — 2026-06-19

### Fixed
- **Speaker collapse in overlap-aware clustering** — Added minimum 4-clean-segments
  threshold before applying two-stage overlap clustering. Falls back to clustering all
  segments together when overlap detection is too aggressive.

## [0.1.7] — 2026-06-19

### Changed
- **Removed PyTorch dependency** — Diarization engine migrated entirely to ONNX Runtime. Removed `pyannote-audio`, `resemblyzer`, and transitive `torch` dependencies. Settings dialog simplified: no engine selection, no HuggingFace token input.
- **ONNX diarization pipeline** — Full pipeline verified against VoxConverse (216 files, 39% DER) and ESL dialogue spot-check. Replaces Resemblyzer (66% DER) with a 27% relative improvement. Composed of Silero VAD ONNX (2 MB), WeSpeaker ResNet34-LM ONNX (25 MB), spectral clustering, and two-stage overlap-aware assignment.

### Added
- **CREDITS.md** — Attribution for WeSpeaker (CC-BY-4.0) and diarize library reference (Apache 2.0).

## [0.1.6] — 2026-06-17

### Added
- **Overlap UI surfacing (Phase 2)** — Three visual surfaces now expose overlapping-speech information from the Phase 1 ONNX detection pipeline:
  - **Overlap region strip** — a thin horizontal bar between the waveform and transcript view rendering coloured blocks proportional to each `OverlapRegion`'s time range. Click a block to scroll the transcript to the corresponding segment. Warm amber (`#c8925e`) on dark background — low-alarm, informational weight.
  - **Per-segment overlap badge** — segments that overlap with a detected region get a 4px tinted left border and a `⟪ overlap ⟫` pill badge next to the timestamp, following the existing confidence-badge pattern. The `_draw_badge()` helper was extracted to eliminate badge-rendering duplication.
  - **Inline `[overlap]` VTT annotation** — OHMS WebVTT export appends `[overlap]` to cue text for segments that intersect any OverlapRegion, in both primary and translation VTT blocks.
- **`OverlapRole` model constant** (`UserRole + 8`) — `TranscriptListModel` exposes binary overlap presence via `data(index, OverlapRole)` for the delegate, plus `segment_index_at(ms)` public method for click-to-scroll.
- **Overlap-aware size hinting** — `TranscriptDelegate.sizeHint()` accounts for the overlap badge width to prevent text overflow.
- **22 new tests** — `test_overlap_model.py` (16 tests: segment overlap detection, index lookups, role integration), `test_overlap_strip.py` (5 tests: widget geometry, click emission, empty state), `test_ohms_export.py` (1 test: VTT overlap annotation).

### Fixed
- **Stale overlap data across sessions** — `overlap_regions` and the overlap strip are now cleared on `New File` and `Audio Ready` to prevent phantom blocks from the previous transcription.

### Changed
- **`TranscriptListModel`** — now tracked in git (was previously excluded by the `models/` gitignore rule intended for ML weight directories).
- **`TranscriptDelegate.paint()`** — badge rendering logic extracted to `_draw_badge()` helper for maintainability; overlap border accounted for in `left_offset` calculation.

## [0.1.5] — 2026-06-16

### Added
- **Progress status signals** — `AudioLoadWorker` now emits `status_changed` with phase messages ("Normalising audio..."), wired to the status label. `LLMWorker` emits a clear "Downloading LLM model (1.1 GB)..." message before download so users know it's actively transferring, not stuck.
- **Signal chain integration test** — `tests/test_signal_chain.py` (6 tests, 0.1s runtime) verifies the worker orchestration (`_on_file_selected` → `AudioLoadWorker` → `_on_audio_ready` → `TranscriptionWorker` → `_on_transcription_finished` → editor page) using mock workers with proper PyQt6 signal class attributes. No network, no models.

### Fixed
- **Crash in transcript widget** — `paint()` method used `QStyleOptionViewItem.StateFlag.State_Selected` which doesn't exist in PyQt6 6.11.0. Changed to `QStyle.StateFlag.State_Selected`. The unhandled `AttributeError` in the delegate's paint method was terminating the application when rendering selected items.
- **Best Quality model 404** — `Systran/faster-whisper-large-v3-turbo` repository doesn't exist on HuggingFace. Changed all Whisper tiers to use native size strings (`"small"`, `"medium"`, `"turbo"`) which faster-whisper resolves internally.
- **GLiNER Entity object access** — NER worker used dict access `ent["start"]` but GLiNER2 returns Entity objects. Changed to attribute access `ent.start`.
- **Transcript text readability** — `QTextDocument.setHtml()` ignores the painter's pen color and was rendering text in black on dark backgrounds. Wrapped all text in explicit `<span style='color:...'>` tags.

## [0.1.4] — 2026-06-16

### Added
- **ONNX-based overlap detection (Phase 1)** — `OverlapDetector` class using `onnx-community/pyannote-segmentation-3.0` ONNX model (5.99 MB). Detects overlapping speech regions via 7-class output (non-speech, 3 single-speaker, 3 overlap classes). Runs post-transcription as a lightweight ONNX Runtime session (~6 MB RAM) instead of the previous PyTorch pipeline (~1.5 GiB). Configurable per-use-case threshold (default 0.3).
- **Gap-enriched LLM summarization** — `LLMWorker` now formats transcripts with speaker labels and `<gap=X.Xs>` inter-turn markers. The system prompt instructs the model to interpret short gaps as collaborative, moderate as formal, and long gaps as hesitant/adversarial — without mentioning the markers in output. All data already existed in `Segment.start_ms`/`end_ms`; zero new dependencies.
- **Backchannel Stage 1 (rule-based)** — `NERWorker` now checks every segment against a filler-word lexicon (`mhm`, `uh-huh`, `yeah`, `right`, `okay`, etc.) with an 800ms duration cap. Classified backchannels skip NER entirely (no entities to extract). Zero-cost — pure string matching on already-loaded data.
- **Smoke tests** — `tests/test_overlap_detector.py` (4 tests: model validation, silent audio, synthetic overlap detection, minimum duration) and `tests/test_transcription_smoke.py` (full pipeline integration test using `sample.ogg`).
- **Deep research reports** — 9 papers covering overlap detection, timing-enriched summarization, prosody extraction, backchannel classification, and active speaker detection, stored in `docs/research-papers/`. Implementation plans updated in `docs/research-prompts/`.
- **Pre-fetch models backlog item** — `docs/backlog/prefetch-models.md` documents the offline field-worker use case with solution options and per-model size table. Blocking for v0.2.0.

### Fixed
- **Overlap detector softmax** — Softmax was applied only to the 3 overlap class logits (artificially inflating their probabilities). Now applied to all 7 classes, then overlap probabilities extracted from the full distribution. Correct output shape validation added via dummy-tensor assertion at model load time.
- **Float/int bug in validation block** — `np.zeros((1, 1, WINDOW_SIZE_S * SAMPLE_RATE))` produced a float shape value; wrapped with `int()`.
- **Entry point shebang** — Re-ran `pip install -e .` to fix stale shebang from a previous install location.
- **Overlap detector window size** — Corrected from 5s to 10s to match the actual ONNX model's input expectation.

### Changed
- **Overlap detector default threshold** — Lowered from 0.5 to 0.3 based on empirical testing (single-speaker audio maxes at ~0.076 on full 7-class softmax; real overlap regions reach 0.3–0.97).
- **Backchannel data model** — `Segment` now has `is_backchannel: bool` and `backchannel_source: str` fields for downstream consumers.
- **Whisper model resolution** — Switched from non-existent `Systran/faster-whisper-large-v3-turbo` HF repo to faster-whisper's native `"turbo"` size string. ModelManager now returns faster-whisper size strings for Whisper tiers instead of calling `snapshot_download` on invalid repo IDs.

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
