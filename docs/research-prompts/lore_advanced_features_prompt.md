# Lore — Advanced Feature Feasibility Analysis: NER, Audio Event Detection & Local LLM

---

## Role & Project Context

You are a senior Python desktop application architect with deep expertise in local NLP inference, audio signal processing, and privacy-first archival software design. You are providing a detailed technical feasibility analysis for **Lore**, an active production project currently in Phase 4 of an 8-phase development roadmap.

**The Application**: Lore is a fully offline, privacy-first oral history transcription desktop application built with **Python 3.13** and **PyQt6**. Its primary users are non-technical archivists, oral historians, and community memory workers. It uses OpenAI Whisper (via `faster-whisper`) for local speech-to-text and exports to **OHMS XML 6.0** — the institutional deposit standard for CONTENTdm and Omeka-S. No internet connection is ever required or assumed after first-run model download. All audio, transcripts, and metadata remain entirely on-device.

---

## Established Architecture — Hard Constraints (Do Not Re-Litigate)

The following decisions are final and documented. All proposals must work *within* them.

**Core Engine**

- **ASR**: `faster-whisper 1.2.1` \+ `CTranslate2` backend. The recommended default model is `large-v3-turbo` (INT8, \~3.0 GB RAM, \~8–12 min for a 4-hour file, \~9.5% WER baseline).  
- **VAD**: `silero-vad-lite` (ONNX-based, no PyTorch) via faster-whisper's `vad_filter=True`. **PyTorch is banned from the Lore bundle** — it was explicitly rejected to keep the bundle under 400 MB. Any proposed library must have an ONNX, pure-Python, or CTranslate2 inference path. If a library's only backend is PyTorch, it is ruled out.  
- **Established `transcribe()` parameters**: `compute_type="int8"`, `device="cpu"`, `condition_on_previous_text=False` (critical for hallucination suppression — means each 30-second window is independently transcribed, so cross-boundary context is deliberately severed), `word_timestamps=True`, `no_speech_threshold=0.6`, `compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`, `vad_filter=True`.  
- **Audio I/O**: `imageio-ffmpeg` (bundled). All audio preprocessed to **16 kHz mono WAV with loudnorm** via ffmpeg subprocess before being passed to faster-whisper.

**UI & Threading**

- `QMediaPlayer` for playback; `positionChanged()` at 50 ms intervals.  
- Custom `QWidget` \+ `QPainter` waveform widget with Level-of-Detail (LoD) downsampled data; sub-200 MB memory budget for waveform structures.  
- `QPlainTextEdit` transcript editor with **`QTextBlockUserData`** storing `SegmentData` (start\_ms, end\_ms, text, speaker\_label) per block; `bisect`\-based O(log n) segment lookup.  
- **`QThread` \+ typed `pyqtSignal`** for all background work. Queued connections (`Qt.ConnectionType.QueuedConnection`) for all inter-thread communication. This pattern is non-negotiable for all new AI features.

**Distribution**

- PyInstaller with a hand-crafted `.spec` file (explicit `CTranslate2` native binaries, `qt6_plugins` Tree directive, `silero-vad-lite`, `imageio-ffmpeg`).  
- Current bundle baseline: **\~370 MB** (before any new AI features are added).  
- Hard target: **sub-400 MB** — approximately **30 MB of remaining bundled headroom**.  
- Models cached in `platformdirs.user_data_dir('Lore', 'LoreProject') / 'models'` — shared with the sibling application HOARD if it ever adopts Whisper.  
- Code-signed GitHub Actions CI/CD pipeline (macOS notarisation required; Windows SmartScreen).

**Export**

- **OHMS XML 6.0** using `<vtt_transcript>` containing WebVTT-formatted content in a CDATA block (HH:MM:SS.mmm timecode precision). This supersedes the legacy `<sync time="HH:MM:SS">` format used in schema versions 5.x.  
- `<index>` block: `<point>` entries with `<time>` in HH:MM:SS (whole-second, macro-level), `<title>`, `<synopsis>`, `<keywords>` (semicolon-delimited), `<subjects>` (LCSH-style).  
- `<record>` envelope: `<title>`, `<interviewer>`, `<interviewee>`, `<date>`, `<repository>`, `<rights>`, `<collection>`, `<description>`.  
- Speaker labels use WebVTT `<v Speaker Name>` syntax within the vtt\_transcript block.  
- Optional Dublin Core sidecar (derived from the same metadata the user enters).  
- Additional exports: plain text, SRT, WebVTT.

**Current Phase**: Phase 4 — optional speaker diarization via `pyannote-audio 3.x` (opt-in, HuggingFace token flow). `Resemblyzer` is the lightweight fallback under parallel evaluation for users who won't create a HuggingFace account.

---

## Precise RAM Budget

| Component | RAM (peak) | Notes |
| :---- | :---- | :---- |
| OS (macOS/Win/Linux) | \~2.0 GB | Conservative baseline |
| Qt app \+ PyQt6 | \~0.3 GB |  |
| `large-v3-turbo` (INT8, loaded during transcription) | \~3.0 GB | Unloaded via `WhisperModel.__del__` after transcription |
| **Concurrent headroom** (during transcription) | **\~2.7 GB** | Available for NER \+ audio classifier running alongside |
| **LLM headroom** (after Whisper unloaded) | **\~5.7 GB** | Available for abstract generation |

These are the hard envelopes. Proposals that exceed them are not viable.

---

## Global Evaluation Framework

For each of the three features below, structure your analysis across these five dimensions and close each section with a summary comparison table:

| Dimension | Definition |
| :---- | :---- |
| **Technical Feasibility** | Production maturity, Python 3.13 compat, fully offline, **PyTorch-free path** |
| **UX Friction** | Workflow disruption for a non-technical archivist — Low / Medium / High |
| **Bundle & Disk Impact** | MB added to the \~370 MB baseline; mandatory bundle vs. optional post-install download |
| **Runtime Memory & CPU** | RAM \+ CPU cost; concurrent vs. sequential with transcription |
| **Maintenance & Licensing** | Dependency risk, update cadence, license (prefer Apache 2.0 or MIT) |

Conclude **each** section with:

- ✅ **Recommended stack** — specific library \+ model version to ship in mid-2026  
- 📦 **Bundle delta** — exact estimated disk cost (mandatory bundle or optional download, and why)  
- 🏗️ **Integration sketch** — how this fits into the existing `QThread`/signal/slot architecture, where in the Phase 0–8 roadmap it lands, and how results surface in the `QTextBlockUserData` editor or the OHMS export pipeline

---

## Feature 1 — Offline Named Entity Recognition (NER)

**Goal**: Automatically tag mentions of people, places, and organisations in the Whisper-generated transcript, surfacing them as inline annotations in the `QPlainTextEdit` editor (extending the existing `QTextBlockUserData` pattern) and as a filterable sidebar panel. Zero archivist configuration required.

**Critical architectural constraint**: `spaCy en_core_web_trf` uses PyTorch — banned. Standard GLiNER also uses PyTorch — banned unless a validated PyTorch-free ONNX inference path exists. All proposals must specify a concrete PyTorch-free inference route.

### Libraries to Evaluate

- **`spaCy en_core_web_sm` (CPU-only, no transformer)**: What is the bundled disk size (library \+ model blob) on all three platforms? What is per-segment inference latency on the 30-second chunks faster-whisper already yields? What is realistic NER accuracy on conversational oral history text — informal grammar, regional accents, non-standard proper nouns — compared to the `trf` pipeline that is now off the table?  
- **`spaCy en_core_web_md` (CPU-only)**: Is the accuracy gain over `sm` on this domain worth the additional bundle cost? Quantify both.  
- **GLiNER via ONNX export**: Is there a community-maintained ONNX export of GLiNER that eliminates the PyTorch dependency? If so: ONNX model size, inference speed on CPU, and accuracy versus `spaCy sm` on zero-shot oral history entity types? Is the ONNX export stable enough for production use?  
- **`edsnlp` or other ONNX-native NER alternatives**: Are there alternative libraries with production-grade NER that have native ONNX inference paths and no PyTorch dependency at runtime?

### Specific Questions

1. **Domain accuracy**: Faster-whisper runs with `condition_on_previous_text=False`, which means each 30-second window is transcribed independently without cross-window context. This can produce inconsistent proper noun spellings across chunk boundaries (a surname may be spelled differently in chunk 3 vs. chunk 7). How does this affect NER reliability, and what mitigation is possible at the entity aggregation level?  
     
2. **Streaming NER on the generator**: Faster-whisper's `transcribe()` is a generator that yields one `Segment` at a time. Can NER run incrementally on each yielded segment (streaming NER), processing in parallel with the transcription `QThread`? Propose a concrete threading pattern: does NER run in the same `TranscriptionWorker`, in a sibling `QThread` consuming segments from a `queue.Queue`, or post-processing after the full transcript is available?  
     
3. **`QTextBlockUserData` extension**: The existing `SegmentData` stores `start_ms`, `end_ms`, `text`, and `speaker_label` per `QTextBlock`. How should NER entities be stored — as additional fields on `SegmentData`, as a parallel `EntityData` object on the same block, or as `QTextCharFormat` overlays applied after rendering? What is the performance impact of applying entity highlights to 10,000 blocks without degrading scroll responsiveness?  
     
4. **Oral history-specific entity types**: Standard NER models (trained on news corpora) miss entities common in oral history: Indigenous place names, historical organisations, family surnames used informally, eras and periods ("the Depression", "the war"). Is there a zero-shot path — using GLiNER ONNX or another approach — that handles these without requiring the archivist to annotate training examples?  
     
5. **OHMS pipeline integration**: The OHMS XML 6.0 `<index>/<point>/<keywords>` and `<subjects>` fields need to be populated with domain-relevant terms. Can NER-extracted entities be automatically surfaced as candidate keywords in the metadata form, reducing the archivist's manual effort? Propose the data flow from NER output → metadata form field pre-population.

---

## Feature 2 — Audio-Signal Non-Verbal Event Detection

**Goal**: Detect and timestamp non-verbal audio events — `[Laughter]`, `[Crying]`, `[Silence]`, `[Crosstalk]`, `[Inaudible]` — and inject them at the correct positions in the faster-whisper transcript. This is **audio signal classification** only, not text-based sentiment analysis.

**Build on what already exists before proposing a new classifier**:

- `silero-vad-lite` (ONNX) is already running via `vad_filter=True`. It produces a list of `speech_timestamps` (start/end in seconds). Non-speech gaps in this list are already known to the pipeline. Evaluate whether `[Silence]` injection can be implemented by thresholding these existing gaps (e.g., gaps \> 3 s \= `[Silence: Xs]`) **without any additional model**.  
- Faster-whisper already runs with `word_timestamps=True`, so segment-level boundaries are precise.  
- The audio pipeline already produces a **16 kHz mono WAV buffer** (post-ffmpeg/loudnorm). Any classifier must consume **this same buffer**, not re-decode the source file.  
- Faster-whisper's sliding window processes audio in 30-second chunks. A classifier should ideally operate on the same 30-second windows to avoid introducing a separate decode pass.

### Models/Libraries to Evaluate

- **`[Silence]` from existing VAD gaps**: Specify the exact algorithm for converting `speech_timestamps` into `[Silence: Xs]` annotations using the already-available VAD output. What duration threshold is appropriate for oral history (contemplative pauses vs. true silence)?  
- **YAMNet via `tflite-runtime` (not full TensorFlow)**: `tflite-runtime` is a minimal package (\~2–3 MB) that runs `.tflite` models without the full TF stack. What is the YAMNet `.tflite` model size? Can it classify `[Laughter]`, `[Crying]`, and `[Crosstalk]` with acceptable accuracy on the 16 kHz buffer? What is per-30-second-window inference latency on a modern CPU? Is `tflite-runtime` PyInstaller-packagable without pathological bundle inflation?  
- **PANNs via ONNX export**: Are there production-ready ONNX exports of PANNs models (CNN14 or MobileNetV2 variants) for audio tagging? What is the ONNX model size, and how does accuracy on target event classes compare to YAMNet? What `onnxruntime` version compatibility issues exist alongside the `silero-vad-lite` ONNX runtime already in the bundle?  
- **Silero VAD probability signal re-use**: Can the per-frame VAD probability output from silero (already available inside faster-whisper) be analysed for characteristic burst patterns that correlate with `[Laughter]` (short rapid high-confidence bursts) without running a second model? Evaluate the feasibility of this "free" detection path.

### Specific Questions

1. **Parallel pipeline architecture**: Propose a concrete `QThread` topology where audio classification runs as a sibling worker to `TranscriptionWorker`, both consuming the same 16 kHz audio buffer. Specifically: how is the pre-decoded audio buffer shared between threads without copying it (use of `memoryview` or `numpy` array with shared backing)? How are classifier results (timestamped event labels) merged with the transcription segment stream on the main thread before being stored in `QTextBlockUserData`?  
     
2. **Conflict resolution**: When faster-whisper emits text for a 30-second segment that the classifier simultaneously marks as `[Laughter]`, what merge strategy is recommended? Propose a specific data model for this merged output and describe how the archivist would review and accept or reject each auto-inserted annotation in the editor.  
     
3. **Archival audio accuracy**: Oral history collections frequently contain digitised cassette tape audio with tape hiss, frequency roll-off, SNR below 15 dB, and elderly speakers with soft or trailing voices. What is the realistic false positive rate for `[Crying]` vs. `[Laughter]` on such audio? At what point does the false positive rate require a mandatory archivist review step before annotations are committed to the transcript?  
     
4. **OHMS XML 6.0 representation**: The `<vtt_transcript>` block uses WebVTT cue format. How should `[Laughter]`, `[Crying]`, and `[Silence]` annotations appear in the WebVTT payload? Is there a WebVTT extension syntax for non-verbal cues (e.g., using `<c.laughter>` CSS-class notation), or should they appear as inline text tokens within the cue? What does the OHMS Viewer do with non-standard VTT annotations?  
     
5. **Phase placement**: This feature could run as a post-processing step after transcription completes, or as a parallel pipeline during transcription. Given that a 4-hour interview takes \~10 minutes to transcribe, which architecture minimises total wall-clock time for the archivist?

---

## Feature 3 — Local LLM Integration for Abstract & Metadata Generation

**Goal**: From a single button click, generate: (1) a plain-language interview abstract for the OHMS `<record>/<description>` field, (2) a pre-populated OHMS `<index>` block with thematic `<point>` entries, (3) keyword and LCSH-style subject term suggestions, and (4) an optional Dublin Core sidecar — all via a locally-running quantised LLM, with no external API calls.

**OHMS XML 6.0 structured output mapping** (the LLM's JSON output must map directly to these):

| LLM output field | OHMS XML 6.0 target |
| :---- | :---- |
| `abstract` (prose) | `<record>/<description>` |
| `global_keywords[]` | `<record>/<keywords>` |
| `themes[].title` | `<index>/<point>/<title>` |
| `themes[].synopsis` | `<index>/<point>/<synopsis>` |
| `themes[].keywords[]` | `<index>/<point>/<keywords>` |
| `themes[].subjects[]` | `<index>/<point>/<subjects>` (LCSH-style) |
| `themes[].timestamp` | `<index>/<point>/<time>` (HH:MM:SS) |

**Timing constraint**: Abstract generation must complete in **under 5 minutes for a 60-minute interview** and under **10 minutes for a 2-hour interview** on a modern CPU (Intel i7/i9 or equivalent; Apple M-series is a bonus). If a model exceeds these limits, it fails the design bar.

**LLM RAM ceiling**: 5.7 GB available (large-v3-turbo unloaded first; see RAM budget table above). Do not recommend models that exceed 5 GB loaded at the recommended quantisation level.

**Transcript input characteristics** (arising from established faster-whisper configuration):

- `condition_on_previous_text=False` means each 30-second chunk is transcribed independently. Proper noun spellings may be inconsistent across chunks. The LLM chunking strategy must account for this.  
- `word_timestamps=True` is enabled, so each segment has precise start/end times usable as `<index>/<point>/<time>` candidates.  
- A 60-minute interview at \~130 WPM yields approximately 7,800 words (\~10,000 tokens) of transcript text — evaluate whether this exceeds common 3B model context windows.

### Models & Runtimes to Evaluate

- **Llama 3.2 3B (Q4\_K\_M via llama.cpp)**: RAM at Q4\_K\_M, tokens/second on Intel i7, AMD Ryzen 5/7, and Apple M1/M2. Is 3B parameter count sufficient for generating coherent OHMS subject headings and thematic synopsis text for archival content?  
- **Phi-3.5 Mini 3.8B (Q4\_K\_M via llama.cpp)**: Compare to Llama 3.2 3B on summarisation and structured JSON instruction-following. Does Microsoft's training emphasis on reasoning improve OHMS metadata generation quality?  
- **Qwen 2.5 3B (Q4\_K\_M via llama.cpp)**: Oral history collections frequently include interviews in languages other than English (Spanish, te reo Māori, Cantonese). Does Qwen 2.5 3B's multilingual capability offer meaningful value for generating English-language OHMS metadata from non-English transcripts? Quantify the accuracy trade-off vs. Llama 3.2 3B on English-only tasks.  
- **Mistral 7B (Q4\_K\_M)**: Evaluate as the ceiling option — does it fit within 5.7 GB at Q4\_K\_M? If so, does the quality gain over 3B models justify the RAM cost and inference time penalty for this specific task?  
- **MLX (Apple Silicon only)**: For macOS builds on M-series hardware, how much faster is MLX inference vs. llama.cpp CPU-only for the recommended model? Is the speedup significant enough to warrant a platform-specific inference backend in the macOS PyInstaller build, given the added maintenance complexity of shipping two code paths?

### Specific Questions

1. **GBNF grammar for OHMS JSON**: llama.cpp supports GBNF (Grammar-Based Normalised Form) grammars that constrain output to a specific JSON schema, eliminating malformed output without post-processing. Provide:  
     
   - A GBNF grammar definition that constrains the model's output to the OHMS mapping table above  
   - An example system prompt written from the persona of a professional archivist following OHA Best Practices, optimised for extracting themes and generating LCSH-compatible subject headings  
   - A JSON Schema (draft-07) matching the GBNF grammar, usable for programmatic validation before XML serialisation

   

2. **Chunking strategy for `condition_on_previous_text=False` output**: Because each Whisper chunk is transcribed independently, the full transcript is a list of `Segment` objects that may have subtle discontinuities at 30-second boundaries. For LLM summarisation, compare:  
     
   - **Map-reduce**: Summarise each N-minute section independently, then synthesise across section summaries. What section granularity (5 min? 10 min?) minimises token throughput while preserving theme coherence?  
   - **Sliding window with overlap**: Feed overlapping transcript chunks to preserve cross-boundary context. What overlap size compensates for Whisper's `condition_on_previous_text=False` discontinuities without doubling token cost?  
   - For a 60-minute interview (\~10,000 transcript tokens), which strategy fits within the recommended 3B model's context window without truncation?

   

3. **Single model recommendation**: Given the RAM ceiling (5.7 GB), timing constraint (\<5 min for 60-min interview), OHMS metadata quality requirements, and Python 3.13 \+ llama.cpp compatibility — provide one concrete model recommendation with explicit reasoning. Include the exact GGUF filename, quantisation level, expected RAM at that quantisation, and realistic tokens/second on a mid-range Intel i7 and an Apple M1.  
     
4. **Lazy load/unload lifecycle**: The LLM should not hold weights in RAM during normal transcription work. Using `llama_cpp.Llama`, specify:  
     
   - The exact instantiation call (model path from `platformdirs` cache, `n_ctx`, `n_gpu_layers=0` for CPU, `verbose=False`)  
   - How to explicitly free the model from RAM after generation completes (does `del model` \+ `gc.collect()` reliably free the CTypes memory, or is a manual `llama_cpp` free call needed?)  
   - Expected model load time from NVMe SSD vs. spinning HDD for the recommended GGUF file  
   - How the `QThread` worker signals the main thread during streaming token generation (emit a `pyqtSignal(str)` per token, or buffer to sentences?)

   

5. **Optional download UX**: LLM weights (1.5–4 GB) cannot be bundled. Using the existing download manager architecture (HTTP Range requests, `platformdirs` model cache, progress `pyqtSignal`), answer:  
     
   - Should the LLM download be offered in the Phase 0 first-run wizard alongside Whisper model selection, or deferred until the archivist first clicks "Generate Abstract"?  
   - The existing wizard presents Whisper model choices as "Fast / Balanced / Best Quality." How should the LLM option be presented to a non-technical archivist who does not understand what a language model is?

   

6. **Multilingual OHMS metadata**: CONTENTdm and Omeka-S expect English-language LCSH subject headings and keywords for cross-collection discoverability. If the interview is in Spanish or another language, should the LLM prompt instruct it to (a) generate English metadata from the non-English transcript, (b) generate bilingual metadata, or (c) detect language and adapt? Recommend a prompt strategy and identify which 3B models handle this task reliably.

---

## Synthesis & Architectural Recommendation

After evaluating all three features independently, provide the following.

### 1\. Feature Priority Ranking

Given combined constraints — non-technical users, \~370 MB bundle baseline, 2.7 GB concurrent RAM headroom during transcription, 5.7 GB LLM ceiling, CPU-only, no PyTorch — which feature delivers the highest **value-to-complexity ratio** for the oral history archivist's workflow and should be built first? Justify your ranking with specific reference to the OHMS export pipeline, the existing Phase 0–8 roadmap, and the research already completed on pyannote/Resemblyzer in Phase 4\.

### 2\. Cumulative Overhead Estimate

If all three features are shipped together (NER bundled or optional, audio classifier bundled, LLM as optional download), provide realistic estimates for:

- **Total PyInstaller bundle size**  
- **Peak concurrent RAM** during active transcription (large-v3-turbo \+ NER \+ audio classifier all running simultaneously)  
- **Peak RAM** during abstract generation (Whisper unloaded, LLM loaded)

Flag any scenario where cumulative overhead violates the hard constraints.

### 3\. Phase Placement

Propose where each feature slots into the existing Phase 0–8 roadmap. Should any be deferred to Phase 9+? Note that Phase 4 (speaker diarization) produces `speaker_label` data that is an input to NER (entities tagged per speaker) and potentially to the LLM (speaker-attributed transcript improves theme extraction). How does this dependency affect sequencing?

### 4\. Unified Pipeline Architecture

Describe the signal/slot topology for all three AI pipelines co-existing in a single PyQt6 application without blocking the main thread. Specifically address:

- How NER and audio classification run concurrently with `TranscriptionWorker`  
- How their outputs are merged with the segment stream before `QTextBlockUserData` is populated  
- How the LLM worker integrates with the OHMS metadata form (streaming token output vs. batch)  
- How the existing single-model download flow is extended to manage multiple optional model downloads without UX complexity for the non-technical archivist

### 5\. Shared Model Cache Extension

The existing `platformdirs` cache at `user_data_dir('Lore', 'LoreProject') / 'models'` stores Whisper GGUF weights. Propose a directory structure extension for:

- NER model data (if not bundled)  
- Audio classifier ONNX/TFLite weights  
- LLM GGUF weights (potentially multiple — user may download Phi-3.5 Mini and Llama 3.2 3B)  
- Which of these, if any, should be in a shared path accessible to the HOARD sibling application?

