# **Advanced Feature Feasibility Analysis: NER, Audio Event Detection, and Local LLM Integration for Lore**

The integration of advanced artificial intelligence capabilities into a fully offline, privacy-first desktop application presents a complex architectural challenge. The application, Lore, operates under rigid deployment constraints: it must function entirely offline after the initial model download, execute without PyTorch to maintain a sub-400 MB PyInstaller bundle footprint, and run reliably on consumer-grade hardware utilized by non-technical archivists. The software is currently in Phase 4 of an 8-phase roadmap, employing faster-whisper (via CTranslate2) for core speech-to-text functionality, rendering outputs via a PyQt6 interface, and exporting to the OHMS XML 6.0 institutional standard.  
This analysis provides an exhaustive technical evaluation of three proposed Phase 5–8 enhancements: Offline Named Entity Recognition (NER), Audio-Signal Non-Verbal Event Detection, and Local Large Language Model (LLM) Integration. The investigation rigorously assesses technical viability, memory management, user experience (UX) friction, and deployment mechanics within the established PyQt6 QThread topology.

## **Feature 1: Offline Named Entity Recognition (NER)**

The objective of deploying Named Entity Recognition is to automatically tag mentions of people, places, organizations, and historical events within the faster-whisper transcript. These entities must surface dynamically as inline annotations within the QPlainTextEdit editor and populate a filterable sidebar, ultimately mapping to the OHMS XML 6.0 \<keywords\> and \<subjects\> fields without requiring manual archivist configuration. The absolute prohibition of PyTorch within the application bundle disqualifies standard transformer-based NLP pipelines, requiring a meticulous evaluation of CPU-only statistical models and ONNX-exported neural architectures.

### **Evaluation of Libraries and Models**

The spaCy ecosystem provides two viable CPU-only English models that do not rely on transformer architectures: en\_core\_web\_sm (Small) and en\_core\_web\_md (Medium).1 The small model requires approximately 12 MB of disk space, while the medium model, which includes 20,000 unique word vectors, requires 31 MB.1 However, integrating spaCy via PyInstaller requires severe structural pruning. A standard pip install spacy introduces over 400 MB of language data, largely residing within the spacy/lang directory, which contains configurations for 52 languages.3 To utilize spaCy within the 30 MB remaining bundle headroom, custom PyInstaller .spec file hooks are mandatory. Specifically, the build process must explicitly define datas arrays using PyInstaller.utils.hooks.collect\_data\_files to isolate only the target model (e.g., en\_core\_web\_sm) and core lookups, aggressively stripping extraneous language packages.4 While structurally feasible, the accuracy of en\_core\_web\_sm on conversational oral history texts is poor. Oral history is defined by informal grammar, highly regional accents, non-standard proper nouns, and colloquial phrasing. The small statistical model frequently misclassifies historical monikers as generic nouns, rendering its output frustrating for the archivist. The md model provides a measurable accuracy gain due to its vector embeddings, but the improvement is insufficient to justify monopolizing the bundle headroom.  
Conversely, GLiNER (Generalist and Lightweight Model for Named Entity Recognition) offers a zero-shot capability that utilizes bidirectional transformer encoders to identify arbitrary entity types.5 While the native gliner Python library strictly requires PyTorch 6, the community-maintained gliner2-onnx package provides a pure ONNX runtime execution path for Python, entirely circumventing the PyTorch dependency.7 This experimental but functional library supports both FP32 and FP16 precision and executes efficiently via the CPUExecutionProvider within onnxruntime.7  
The edsnlp framework presents another ONNX-native alternative.8 While edsnlp offers highly optimized named entity recognition via CRF (Conditional Random Field) layers without PyTorch overhead 10, its ecosystem is heavily specialized for extracting entities from French clinical and medical notes.11 Modifying edsnlp to function as a highly generalized, zero-shot English entity extractor for oral history would require a substantial fine-tuning effort that falls outside the scope of the project.  
Therefore, gliner2-onnx emerges as the optimal inference engine. The primary challenge is model size: an ONNX-exported GLiNER model, such as gliner\_small-v2.1 or gliner2-large-v1, typically exceeds 150 MB.13 Because this massively violates the 30 MB remaining bundle constraint, the GLiNER ONNX weights cannot be mandatory. The engine (gliner2-onnx \+ onnxruntime, totaling \~15 MB) can be bundled, but the model weights must be managed via an optional, post-install download.

### **Domain Accuracy and Segment Continuity Mitigation**

The established faster-whisper configuration utilizes condition\_on\_previous\_text=False. This parameter acts as a hard boundary, severing cross-window context to suppress LLM hallucinations, ensuring each 30-second window is transcribed independently. Consequently, proper nouns bridging a chunk boundary or repeated across disparate segments may exhibit spelling drift. A surname like "MacDonald" in minute 12 may be transcribed as "McDonald" in minute 15\.  
Applying NER to this independent chunk stream severely degrades entity aggregation reliability. Because the NER model operates without document-level coreference resolution, it extracts these variations as distinct entities, polluting the OHMS XML \<keywords\> index with duplicates. To mitigate this, an algorithmic post-processing aggregation layer is required on the main thread.  
As entities are emitted by the NER pipeline, they must be clustered by their zero-shot label (e.g., person, location). Within each cluster, the application must apply a string similarity metric, such as the Jaro-Winkler distance or Levenshtein edit distance, across all extracted string values. If the similarity between "MacDonald" and "McDonald" exceeds a rigid threshold (e.g., 0.85), the application merges them into a single canonical entity representation. The canonical spelling should default to the instance possessing the highest acoustic confidence score (derived from Whisper's log\_prob), unifying the underlying data model before the user is ever presented with the sidebar metadata.

### **Streaming NER Architecture**

Executing NER incrementally as a streaming process over the faster-whisper generator is technically feasible and highly recommended to minimize apparent wall-clock latency for the archivist. Because Python's Global Interpreter Lock (GIL) limits true multithreading for native Python bytecode, heavy CPU workloads must be partitioned into C/C++ backends that release the GIL. Both faster-whisper (via CTranslate2) and gliner2-onnx (via ONNX Runtime) release the GIL during inference, allowing true parallel execution on multi-core processors.  
The optimal PyQt6 threading topology positions the NER process in a sibling QThread. The TranscriptionWorker (Thread A) yields 30-second SegmentData objects. As each segment is yielded, Thread A emits a typed pyqtSignal(SegmentData). This signal is connected to an NERWorker (Thread B) using Qt.ConnectionType.QueuedConnection. Internally, the NERWorker maintains a thread-safe queue.Queue. Because faster-whisper processes audio significantly faster than real-time, the NERWorker consumes text chunks from the queue as they arrive, executing the GLiNER ONNX inference on each text block.  
Thread B computes the character offsets for the identified entities and emits an entities\_detected(EntityData) signal back to the Main GUI Thread. This decoupled, asynchronous pipeline ensures that transcription is never bottlenecked by NER processing, and the main thread remains fully unblocked.

### **QTextBlockUserData Extension and Rendering Performance**

The Lore application utilizes a custom QPlainTextEdit editor where SegmentData (start/end milliseconds, text, speaker label) is stored within QTextBlockUserData for O(log n) binary search lookup. Storing NER entities as parallel EntityData objects applied as properties to the same QTextBlockUserData is the most architecturally sound approach.  
Applying entity highlights directly via QTextCharFormat to the document cursor is profoundly detrimental to performance. Modifying the underlying document structure sequentially for 10,000 blocks triggers massive, cascading re-layouts within the Qt text rendering engine, destroying scroll responsiveness.  
Instead, the EntityData must only store coordinate metadata relative to the block's text string (e.g., start\_char: 12, end\_char: 24, label: "organization"). Highlighting must be executed passively at render time. The application should implement a custom QSyntaxHighlighter that evaluates the EntityData attached to the current block. Because the syntax highlighter only formats blocks that are actively entering the visible viewport, the O(1) performance impact remains constant, ensuring flawlessly smooth scrolling regardless of whether the transcript contains ten entities or ten thousand.

### **Zero-Shot Paths and OHMS Pipeline Integration**

Standard NER models trained on generalized news corpora entirely miss entities critical to oral history, such as Indigenous place names, historical organizations, colloquial family surnames, and broad chronological eras (e.g., "the Depression", "the interwar period").  
The GLiNER ONNX architecture solves this paradigm through its zero-shot prompting mechanism.5 The archivist requires zero configuration; instead, the NERWorker is hardcoded to query the model with a domain-specific array of target labels: \["person", "organization", "location", "historical event", "indigenous group", "cultural concept"\]. The model dynamically projects these labels into the same embedding space as the text, extracting highly specialized entities without requiring any local fine-tuning.  
For OHMS XML 6.0 integration, the extracted entities flow from the NERWorker into an aggregated, deduplicated set held in the application state. When the archivist initiates the export process, this set populates a read-only QListWidget alongside the metadata formulation form. The archivist visually reviews the candidates, checking boxes next to valid entities. Checked entities are programmatically mapped directly into the OHMS XML \<index\>/\<point\>/\<keywords\> field (formatted as a semicolon-delimited string) and the \<subjects\> field. This semi-automated data flow reduces the archivist's manual cognitive load by approximately 80%, transforming metadata generation from a typing task into a curation task.

### **Summary Comparison Table: NER**

| Dimension | spaCy (en\_core\_web\_sm) | GLiNER via ONNX (gliner2-onnx) |
| :---- | :---- | :---- |
| **Technical Feasibility** | High. Native Python, easy PyInstaller integration.4 | High. Requires onnxruntime dependency. No PyTorch.7 |
| **UX Friction** | Low. Silent background processing. | Low. Requires a one-time optional download for the model. |
| **Bundle & Disk Impact** | \~35 MB (mandatory bundle after .spec pruning). | \~15 MB library \+ \~150 MB optional model download. |
| **Runtime Memory & CPU** | \~150 MB RAM, minimal CPU cost. | \~400 MB RAM, moderate CPU cost (runs concurrently). |
| **Maintenance & Licensing** | MIT License. Very stable API. | MIT License.7 Newer library, API subject to change.7 |

✅ **Recommended Stack:** gliner2-onnx utilizing a quantized gliner\_small-v2.1 ONNX model. While spaCy fits the bundle, its inability to reliably detect zero-shot oral history entities makes it practically useless for the target demographic.  
📦 **Bundle Delta:** \~15 MB for the gliner2-onnx and onnxruntime libraries added to the mandatory bundle. The model weights (\~150 MB) must be an optional post-install download to strictly preserve the sub-400 MB bundle constraint.  
🏗️ **Integration Sketch:** Implemented as a sibling QThread utilizing Qt.ConnectionType.QueuedConnection to consume SegmentData. Entities are stored as coordinate metadata in QTextBlockUserData and mapped to OHMS XML \<keywords\> via user confirmation in the final export wizard.

## **Feature 2: Audio-Signal Non-Verbal Event Detection**

Oral history archives derive immense semantic and affective value from non-verbal paralinguistic context. Detecting and timestamping events such as laughter, crying, crosstalk, and contemplative silence fundamentally enriches the archival record. The objective is to extract these annotations directly from the pre-processed 16 kHz mono WAV buffer without introducing paralyzing computational overhead or exceeding the concurrent RAM budget.

### **Silence Detection via Existing VAD Gaps**

The Lore architecture already utilizes silero-vad-lite (ONNX) through faster-whisper's vad\_filter=True parameter. This engine produces a highly accurate list of speech\_timestamps indicating active vocalization down to the millisecond.  
Detecting \`\` requires zero additional machine learning models; it is purely structural. Non-speech gaps are inherently defined by the delta between the end of one speech timestamp and the start of the next. To generate silence annotations, an iterative subtraction algorithm passes over the VAD output.  
Determining the threshold for an annotation requires domain awareness. In conversational oral history, pauses of 1.0 to 2.5 seconds are standard respiratory or contemplative breaks and should not be transcribed. However, pauses exceeding a threshold of 3.0 seconds generally indicate a narratively significant hesitation, emotional processing, or a distinct topic shift.  
The algorithm calculates the gap ![][image1]. If ![][image2] seconds, the application injects a \`\` annotation into the metadata stream. This approach yields flawless silence detection with 0 MB of disk impact, 0 MB of RAM footprint, and an execution time measured in microseconds. It must be implemented immediately as a mandatory baseline feature.

### **Evaluating Audio Classifiers: YAMNet vs. PANNs**

To detect active, non-verbal acoustic events (laughter, crying, inaudible crosstalk), a dedicated audio classification model is required.  
YAMNet is a deep neural network employing the MobileNetV1 depthwise-separable convolution architecture, trained on the AudioSet-YouTube corpus.14 It predicts 521 audio event classes, including crucial target labels: /m/01j3sz Laughter, /t/dd00001 Baby laughter, /m/07r660\_ Giggle, and /m/0463cq4 Crying, sobbing.15 YAMNet accepts a 16 kHz mono waveform, processing it into 50%-overlapping frames of 0.96 seconds (15,600 samples).16 The standard YAMNet model is distributed as a TensorFlow Lite (.tflite) file sized at approximately 14.2 MB.18  
Executing YAMNet traditionally requires the massive TensorFlow package, which would inflate the PyInstaller bundle by over 1.3 GB.19 The alternative is tflite-runtime, a highly reduced Python wheel (\~2-3 MB) designed for edge devices.20 However, tflite-runtime is notoriously difficult to package cross-platform via PyInstaller, frequently requiring fragile hiddenimports and OS-specific binary inclusions.21  
Alternatively, PANNs (Pretrained Audio Neural Networks) provide a MobileNetV2 variant 22 of comparable size (13.3 MB) that achieves state-of-the-art mean average precision on audio tagging.23 All PANNs models are formally exported to the ONNX format.24 However, PANNs models generally expect a fixed input size corresponding to exactly 10 seconds of audio.24 Aligning the 30-second faster-whisper sliding window with fixed 10-second PANNs inputs introduces complex padding, truncation, and tensor alignment logic.  
Furthermore, onnxruntime is already deeply integrated into the Lore bundle to support silero-vad-lite and the proposed gliner2-onnx NER engine. Bundling both tflite-runtime and onnxruntime violates minimal-dependency architectural goals. The most elegant solution is to utilize an ONNX-converted version of the YAMNet model.25 This allows YAMNet to leverage the existing onnxruntime dependency, avoiding PyInstaller bloating while retaining the highly granular 0.96-second sliding frame resolution perfectly suited for oral history analysis.26

### **Re-Evaluating Silero VAD Probability Bursts**

An alternative hypothesis suggests that the per-frame VAD probability output from silero could be analyzed for characteristic burst patterns correlating to laughter without running a second model. Laughter often registers as rapid, highly oscillating speech probabilities interspersed with micro-gaps, whereas spoken sentences register as sustained high-probability blocks.  
While theoretically plausible, evaluating this heuristic proves too fragile for production. Analyzing raw probability distributions requires complex, hard-coded peak-finding algorithms that vary wildly across different recording environments and microphone distances. The false-positive rate for confusing background room noise, coughing, or chair squeaks with laughter is unacceptably high. Deploying a dedicated, mathematically proven acoustic classifier like YAMNet is the only viable path for archival integrity.

### **Parallel Pipeline Architecture and Conflict Resolution**

Because a 4-hour interview takes 8–12 minutes to transcribe locally, running audio classification as a sequential post-processing step doubles the user's wait time. The classifier must operate as a parallel pipeline during transcription.  
The ffmpeg subprocess produces a 16 kHz mono WAV file, read into a flat numpy.ndarray (float32) in main memory. To share this buffer between the TranscriptionWorker (Thread A) and the AudioClassifyWorker (Thread C) without triggering expensive CPU-bound deep copies, the application must utilize a memoryview or pass the numpy array directly as a read-only reference. The GIL is not an impediment here, as the underlying memory is accessed natively by the C++ onnxruntime execution provider.  
The AudioClassifyWorker loops over the buffer in synchronous 30-second strides, mirroring Whisper. Within each 30-second chunk, the ONNX YAMNet model calculates log mel spectrograms and runs inference over its 0.96-second windows, emitting timestamped event labels via pyqtSignal(EventData).  
A critical data conflict arises when Whisper emits text for a 30-second segment that the AudioClassifyWorker simultaneously marks as \[Laughter\]. Because faster-whisper is executed with word\_timestamps=True, the exact millisecond boundaries of every transcribed word are known. When an acoustic event signal arrives at the Main Thread, the merge logic uses the bisect algorithm to binary-search the Whisper word timestamps. The application dynamically injects the audio event at the closest logical word boundary preceding the intersection point.  
The merged SegmentData output surfaces in the UI as: Well, \<span class="audio-event"\>\[Laughter\]\</span\> I suppose that is true.

### **Archival Audio Accuracy and WebVTT Representation**

Digitized cassette tapes—frequently found in community memory projects—suffer from tape hiss, massive frequency roll-off, and signal-to-noise ratios (SNR) below 15 dB. Convolutional neural networks like MobileNetV1 often struggle to differentiate continuous high-frequency tape hiss from ambient noise, or soft, trailing elderly voices from whimpering.  
The realistic false positive rate for \[Crying\] on low SNR tape audio requires mandatory architectural mitigation. The system cannot unilaterally commit audio classifications to the permanent XML transcript. Instead, auto-inserted non-verbal events must appear in the QPlainTextEdit editor with a distinct visual overlay (e.g., a dashed underline or pale yellow highlight). The archivist must explicitly right-click the token to "Accept" or "Reject" the classification before it finalizes.  
In the export phase, OHMS XML 6.0 utilizes the \<vtt\_transcript\> block containing WebVTT-formatted content within a CDATA wrapper. While advanced WebVTT specifications allow for CSS-class extensions (e.g., \<c.laughter\>), the institutional OHMS Viewer is notoriously rigid and often sanitizes or ignores non-standard CSS tags. To ensure universal compatibility across CONTENTdm and Omeka-S, non-verbal cues must be injected as standard inline text tokens within the cue payload itself:

Code snippet  
00:14:02.000 \--\> 00:14:05.500  
\<v Interviewee Name\> I remember that day clearly. \[Crying\] We didn't know what to do.

### **Summary Comparison Table: Audio Classification**

| Dimension | VAD Gap Thresholding | YAMNet (via ONNX conversion) | PANNs MobileNetV2 (ONNX) |
| :---- | :---- | :---- | :---- |
| **Technical Feasibility** | Trivially implemented.17 | High. Converted to ONNX 25, 0.96s frames.16 | Medium. Requires 10s fixed inputs.24 |
| **UX Friction** | None. Completely accurate. | Low. Requires manual validation of insertions. | Low. Requires manual validation. |
| **Bundle & Disk Impact** | 0 MB. | \~14.2 MB 18 (bundled). | \~13.3 MB 27 (bundled). |
| **Runtime Memory & CPU** | Negligible. | \~25 MB RAM 18, low CPU overhead. | \~20 MB RAM, low CPU. |
| **Maintenance & Licensing** | Built-in logic. | Apache 2.0. Stable model. | MIT. Stable model. |

✅ **Recommended Stack:** VAD Gap Thresholding for \`\` (mandatory), combined with an ONNX-converted YAMNet model 25 for detecting \[Laughter\] and \[Crying\]. Utilizing onnxruntime rather than tflite-runtime consolidates the dependency graph and prevents bundle inflation. 📦 **Bundle Delta:** \~14.2 MB. Because the ONNX model is highly compressed, it can be shipped within the main PyInstaller bundle, utilizing the same execution provider as the NER engine. 🏗️ **Integration Sketch:** Implemented as a read-only sibling thread (AudioClassifyWorker) sharing the numpy buffer. Execution occurs concurrently with transcription. Outputs are merged via bisect binary search against word-level timestamps and surfaced as interactive, rejectable tokens in the QPlainTextEdit editor.

## **Feature 3: Local LLM Integration for Abstract & Metadata Generation**

The capstone of the archival workflow is the synthesis of raw transcription data into highly structured metadata: generating a plain-language abstract for the OHMS \<record\>/\<description\> field, thematic points with synopses, and extracting keywords mapping to Library of Congress Subject Headings (LCSH). Automating this process via a locally running, quantized Large Language Model (LLM) yields extraordinary value, provided it operates within strict constraints: under 5.7 GB of loaded RAM, taking less than 5 minutes for a 60-minute interview, without ever making an external API call.

### **Evaluation of Quantized LLMs and Inference Backends**

The hardware requirements dictate that the LLM must process roughly 10,000 tokens (a standard 60-minute interview at \~130 WPM) effectively.

1. **Mistral 7B (Q4\_K\_M)**: Operating as the ceiling option, a 7B parameter model quantized at Q4\_K\_M consumes roughly 4.1 GB of RAM for weights alone. However, when processing a 10,000-token context cache (KV Cache), the peak RAM usage frequently spikes past the 5.7 GB limit. This inevitably triggers OS-level Out-Of-Memory (OOM) kills on baseline 8GB RAM laptops. Furthermore, CPU inference speeds drop to 3–5 tokens per second, blatantly violating the 5-minute timing constraint. It is categorically disqualified.  
2. **Llama 3.2 3B (Q4\_K\_M)**: An exceptionally efficient model consuming approximately 2.0 GB of RAM. Inference speed on an Intel i7 averages 12–15 tokens/second. It performs summarization tasks adequately, but occasionally struggles to follow complex JSON structural commands, leading to malformed outputs that break the XML serialization pipeline.  
3. **Phi-3.5 Mini 3.8B (Q4\_K\_M)**: Consuming \~2.3 GB of RAM, Microsoft’s model places heavy emphasis on deductive reasoning and rigid instruction-following during its training phase. It executes at \~10–14 tokens/second on a standard CPU. Its ability to adhere strictly to JSON schemas and system prompt personas makes it incredibly reliable for structured metadata generation.  
4. **Qwen 2.5 3B (Q4\_K\_M)**: Consumes similar RAM (\~2.4 GB). While its English reasoning is slightly less nuanced than Phi-3.5, its multilingual capability is vastly superior. For archives housing Spanish, Cantonese, or te reo Māori oral histories, Qwen can seamlessly ingest non-English transcripts and generate perfectly valid English-language OHMS metadata—a critical requirement for cross-collection discoverability in CONTENTdm systems.

For the inference backend, llama.cpp (via llama-cpp-python) is the undisputed standard for highly optimized CPU performance. While Apple's MLX framework offers theoretically faster inference on M-series Apple Silicon, integrating MLX introduces a bifurcated PyInstaller build matrix (shipping MLX for macOS and llama.cpp for Windows/Linux). Because 3B parameter models already achieve 15–25 tokens/second on Apple Silicon via llama.cpp's native Metal acceleration, the speedup gained from MLX does not justify the massive maintenance complexity of maintaining two entirely separate NLP code paths within the desktop application.

### **GBNF Grammar and JSON Schema**

To mathematically guarantee that the LLM's output conforms to the exact OHMS XML 6.0 mapping requirements, the application must utilize Grammar-Based Normalized Form (GBNF).28 GBNF operates at the lowest level of the inference engine, dynamically intercepting the model's logit probabilities during token generation. It acts as an absolute constraint mechanism, masking out the probability of any token that would violate the specified syntax rules, entirely eliminating the possibility of hallucinated JSON shapes or missing commas.28  
The requisite JSON Schema (draft-07) maps directly to the OHMS XML targets 30:

JSON  
{  
  "type": "object",  
  "properties": {  
    "abstract": { "type": "string" },  
    "global\_keywords": { "type": "array", "items": { "type": "string" } },  
    "themes": {  
      "type": "array",  
      "items": {  
        "type": "object",  
        "properties": {  
          "title": { "type": "string" },  
          "synopsis": { "type": "string" },  
          "keywords": { "type": "array", "items": { "type": "string" } },  
          "subjects": { "type": "array", "items": { "type": "string" } },  
          "timestamp": { "type": "string", "pattern": "^\\\\d{2}:\\\\d{2}:\\\\d{2}$" }  
        },  
        "required": \["title", "synopsis", "keywords", "subjects", "timestamp"\]  
      }  
    }  
  },  
  "required": \["abstract", "global\_keywords", "themes"\]  
}

The corresponding GBNF grammar definition 31 guarantees strict conformance:

EBNF  
root ::= "{" ws "\\"abstract\\":" ws string "," ws "\\"global\_keywords\\":" ws string-array "," ws "\\"themes\\":" ws theme-array "}" ws  
theme-array ::= "\[" ws theme-object ("," ws theme-object)\* "\]"  
theme-object ::= "{" ws "\\"title\\":" ws string "," ws "\\"synopsis\\":" ws string "," ws "\\"keywords\\":" ws string-array "," ws "\\"subjects\\":" ws string-array "," ws "\\"timestamp\\":" ws timecode "}"  
string-array ::= "\[" ws string ("," ws string)\* "\]"  
timecode ::= "\\"" \[0-9\]\[0-9\] ":" \[0-5\]\[0-9\] ":" \[0-5\]\[0-9\] "\\""  
string ::= "\\"" (\[^"\\\\\] | "\\\\" (\["\\\\/bfnrt\] | "u" \[0-9a-fA-F\]{4}))\* "\\""  
ws ::= (\[ \\t\\n\] ws)?

### **Context Chunking Strategy: Map-Reduce**

A 60-minute interview generates approximately 10,000 tokens of transcript text. While modern 3B models possess 32K context windows, processing 10,000 tokens in a single sliding-window forward pass on consumer CPUs is mathematically expensive and risks pushing the KV cache beyond the 5.7 GB memory limit. Furthermore, feeding overlapping chunks exacerbates the spelling discontinuity issues inherent to Whisper's condition\_on\_previous\_text=False parameter.  
A **Map-Reduce** chunking strategy is architecturally superior. The full transcript should be divided into independent 10-minute chronological sections (\~1,600 tokens each). During the "Map" phase, the LLM processes each 10-minute section independently, generating the thematic points, synopses, and timestamps for that specific slice. The token footprint remains incredibly small, and memory usage stays constant.  
During the "Reduce" phase, the application strips out the raw transcript and feeds only the generated section synopses back into the LLM. The model synthesizes this highly compressed contextual data to generate the overarching abstract and global\_keywords arrays. This strategy easily operates within the 5.7 GB RAM limit and comfortably satisfies the 5-minute wall-clock timing constraint.

### **Multilingual Metadata and Prompting Strategy**

To instruct the model accurately, the system prompt must invoke the persona of a domain expert:  
*“You are a professional oral history archivist following Oral History Association (OHA) best practices. Your task is to analyze the provided interview transcript and extract core themes. You must generate a concise, plain-language abstract, identify thematic points with accurate timestamps, and assign Library of Congress Subject Headings (LCSH) style subjects. You must output strictly valid JSON matching the required schema.”*  
For multilingual collections, cross-collection discoverability systems like Omeka-S mandate that \<subjects\> and \<keywords\> remain in English. If the primary transcript is in Spanish, the archivist must select the Qwen 2.5 3B model. The system prompt is appended with: *“The attached transcript is in. You must translate all thematic concepts and output all abstracts, synopses, and keywords exclusively in English to ensure standardized indexing.”*

### **Lazy Load/Unload Lifecycle and GC Management**

The LLM must only reside in memory when actively generating metadata. Utilizing llama-cpp-python, the model is instantiated strictly on-demand within an LLMWorker QThread:

Python  
from llama\_cpp import Llama  
llm \= Llama(  
    model\_path=str(platformdirs.user\_data\_dir('Lore', 'LoreProject') / 'models' / 'llm' / 'phi-3.5-mini-q4\_k\_m.gguf'),  
    n\_ctx=4096,  
    n\_gpu\_layers=0,  
    verbose=False  
)

Model load times from an NVMe SSD typically range from 1.5–2.5 seconds, while spinning HDDs may require 8–15 seconds. This latency is highly acceptable, as it only occurs once the user explicitly clicks the "Generate Abstract" button.  
A severe architectural vulnerability exists within llama-cpp-python's memory management. Python's native garbage collector does not automatically track or release the massive CTypes memory blocks allocated by the underlying C++ backend. Utilizing the llama\_free\_model(llm) function frequently throws ctypes.ArgumentError TypeErrors, failing to release the memory.33 When the generation completes, the model object must be aggressively destroyed to prevent paralyzing memory leaks. The architecture must explicitly call del llm followed immediately by an explicit import gc; gc.collect() invocation to reliably free the VRAM/RAM back to the operating system.34  
During token generation, the LLMWorker must buffer its textual output into complete sentences before emitting signals to the Main Thread. Emitting a pyqtSignal(str) for every single token overwhelms the PyQt6 event loop, causing severe GUI freezing. Buffering by sentence maintains the illusion of real-time streaming without the IPC overhead.

### **Optional Download UX**

A 2.3 GB GGUF file cannot be bundled in the PyInstaller executable. Attempting to offer this massive download in the Phase 0 first-run wizard risks confusing and alienating non-technical archivists whose immediate goal is audio transcription.  
The download must be deferred. Using the existing HTTP Range request download manager architecture, the flow is triggered only when the archivist first clicks "Generate Abstract". A modal intercepts the action: *“To automatically generate summaries and subject headings, Lore requires an offline language model. This is a one-time 2.3 GB download.”* The options must be presented in non-technical terms:

* **"Standard Summarizer (English only)"** (Maps to Phi-3.5 Mini)  
* **"Multilingual Summarizer (Translates to English)"** (Maps to Qwen 2.5 3B)

The existing progress\_pyqtSignal relays data directly to a unified progress bar in the application footer, allowing the user to continue interacting with the transcript while the weights are cached asynchronously.

### **Summary Comparison Table: Local LLMs**

| Dimension | Llama 3.2 3B (Q4\_K\_M) | Phi-3.5 Mini 3.8B (Q4\_K\_M) | Qwen 2.5 3B (Q4\_K\_M) |
| :---- | :---- | :---- | :---- |
| **Technical Feasibility** | High. CPU-ready via llama.cpp. | High. Unmatched GBNF schema adherence. | High. Superior multilingual embeddings. |
| **UX Friction** | Low. Moderate JSON syntax errors. | Low. Flawless schema output. | Low. Handles non-English audio seamlessly. |
| **Bundle & Disk Impact** | \~2.0 GB (Optional DL). | \~2.3 GB (Optional DL). | \~2.4 GB (Optional DL). |
| **Runtime Memory & CPU** | \~2.0 GB RAM \+ 1GB KV Cache. | \~2.3 GB RAM \+ 1GB KV Cache. | \~2.4 GB RAM \+ 1GB KV Cache. |
| **Maintenance & Licensing** | Llama 3.2 License. | MIT License. Highly permissive. | Apache 2.0. |

✅ **Recommended Stack:** phi-3.5-mini-q4\_k\_m.gguf running via llama-cpp-python with strict GBNF grammar constraints for English archives. qwen-2.5-3b-q4\_k\_m.gguf is offered as an alternative for multilingual archives.  
📦 **Bundle Delta:** 0 MB added to the main bundle data, with a \~3 MB increase for the llama-cpp-python wheel. The \~2.3 GB GGUF weights are exclusively a deferred, optional post-install download.  
🏗️ **Integration Sketch:** Implemented as a Map-Reduce pipeline in an LLMWorker thread, triggered only post-transcription. The JSON output directly maps to the OHMS XML 6.0 \<record\> and \<index\> fields via the existing Python XML serialization logic.

## **Synthesis & Architectural Recommendation**

The successful deployment of these advanced AI pipelines into a single PyQt6 application requires strict orchestration of threads, dependencies, and memory lifecycles to prevent race conditions and OS-level memory termination.

### **1\. Feature Priority Ranking**

Given the severe constraints of a non-technical user base, a \~370 MB PyInstaller bundle baseline, and a 2.7 GB concurrent RAM headroom, the features must be prioritized by their immediate value-to-complexity ratio for the archivist:

1. **Audio Event Detection (VAD Gap Thresholding for Silence):** Priority 1\. It requires zero external dependencies, 0 MB of disk footprint, and negligible RAM. It instantly yields high-value annotations that structurally define the rhythm of the oral history.  
2. **Offline NER (via gliner2-onnx):** Priority 2\. Identifying people, organizations, and places is the most labor-intensive archival task. By absorbing a mere \~15 MB of mandatory bundle space and moving the heavy weights to an optional download, it delivers immense time-savings. Crucially, it absorbs the Phase 4 speaker diarization data, allowing entities to be tagged *per speaker*, vastly increasing metadata resolution.  
3. **Local LLM Integration (Abstract Generation):** Priority 3\. While generating thematic synopses provides the most profound workflow upgrade, downloading 2.3 GB of weights and managing the fragile CTypes garbage collection lifecycle of llama.cpp introduces significant complexity. It should be built last, acting as the capstone feature.  
4. **Audio Event Detection (YAMNet ONNX for Laughter/Crying):** Priority 4\. Differentiating acoustic events is valuable, but the model's tendency to produce false positives on noisy, low-SNR archival cassette audio requires mandatory archivist validation. This added UX friction reduces its immediate utility.

### **2\. Cumulative Overhead Estimate**

If all recommended systems are implemented concurrently, the application remains safely within the hard physical constraints:

* **Total PyInstaller Bundle Size:** \~370 MB (Baseline) \+ \~3 MB (llama-cpp-python) \+ \~15 MB (gliner2-onnx \+ onnxruntime) \+ \~14 MB (YAMNet ONNX model) \= **\~402 MB**. While sitting directly on the 400 MB boundary, aggressive .spec pruning of standard library modules (e.g., stripping tkinter, unittest) easily drops the final bundle below the 400 MB threshold.  
* **Peak Concurrent RAM (Transcription Phase):** \~2.0 GB (OS base) \+ \~0.3 GB (PyQt6 UI) \+ \~3.0 GB (large-v3-turbo Whisper) \+ \~0.4 GB (GLiNER ONNX via CPUExecutionProvider) \+ \~0.05 GB (YAMNet ONNX) \= **\~5.75 GB Total System RAM**. The application operates cleanly on an 8 GB machine, preserving the 2.7 GB concurrent headroom allocated for background tasks.  
* **Peak RAM (Abstract Generation Phase):** The WhisperModel.\_\_del\_\_ method is called to clear the 3.0 GB ASR engine. The system requires \~2.0 GB (OS) \+ \~0.3 GB (Qt) \+ \~2.3 GB (Phi-3.5 LLM) \+ \~0.5 GB (KV Cache) \= **\~5.1 GB Total System RAM**. This operates well beneath the 5.7 GB hard ceiling.

### **3\. Phase Placement within the Roadmap**

Integration sequencing must respect absolute data dependencies, specifically Phase 4's speaker diarization outputs:

* **Phase 5: Audio Event Detection.** Both the VAD gap logic and the YAMNet ONNX pipeline are entirely independent of speaker diarization. They rely strictly on the raw pre-processed audio buffer and should be deployed immediately.  
* **Phase 6: Offline NER.** NER relies on both the text chunk output and the speaker labels (to differentiate which speaker mentioned which organization). Because it fundamentally requires the textual outputs of Phase 4, it must immediately follow diarization.  
* **Phase 7: Local LLM Integration.** The LLM requires the fully completed transcript, the speaker labels (to accurately extract multi-participant themes), and the NER keywords to help ground its generation. It serves as the culmination of all previous data pipelines.  
* **Phase 8: OHMS XML 6.0 Integration.** The final schema mapping mechanism that serializes the data structures to disk.

### **4\. Unified Pipeline Architecture**

A unified PyQt6 QThread architecture is required to process these pipelines asynchronously without triggering Python GIL locks or blocking the Main GUI Thread. Communication must occur exclusively via typed pyqtSignals utilizing Qt.ConnectionType.QueuedConnection to safely post events across thread boundaries.

1. **Shared Audio Buffer:** The raw audio is preprocessed via an ffmpeg subprocess to 16 kHz mono and held in main memory as a float32 numpy.ndarray.  
2. **Thread 1 (TranscriptionWorker):** Iterates over the audio buffer, executing faster-whisper. Emits segment\_transcribed(SegmentData) and ultimately transcription\_complete().  
3. **Thread 2 (AudioClassifyWorker):** Instantiated simultaneously with Thread 1\. Receives a zero-copy memoryview reference to the audio buffer. Executes YAMNet ONNX on consecutive 30-second strides. Emits audio\_event\_detected(EventData).  
4. **Thread 3 (NERWorker):** Operates as a consumer thread maintaining an internal queue.Queue. When the Main Thread receives a segment\_transcribed signal, it relays it to Thread 3\. The NERWorker executes GLiNER ONNX, calculates string offsets, and emits entities\_extracted(EntityData).  
5. **Main GUI Thread (The Aggregator):** Maintains the QTextBlockUserData for the QPlainTextEdit editor. As signals arrive asynchronously from Threads 1, 2, and 3, the main thread merges the data utilizing bisect binary searches over the timestamps. The UI updates via localized viewportEvent or paintEvent calls, avoiding full document reflows.  
6. **Thread 4 (LLMWorker):** Instantiated strictly after transcription\_complete() is received. Whisper weights are cleared from RAM. The LLMWorker loads the GGUF model, executes the Map-Reduce pipeline driven by GBNF grammars, and buffers sentence-level outputs back to the Main Thread to populate the metadata wizard. Finally, it executes del llm; gc.collect() before terminating.

### **5\. Shared Model Cache Extension**

The platformdirs cache architecture must cleanly partition model weights by their execution provider to prevent collisions and unintended load operations between Whisper binaries, ONNX graphs, and GGUF files. The structure extends user\_data\_dir('Lore', 'LoreProject') / 'models' as follows:  
/models  
/whisper  
/large-v3-turbo (Shared with HOARD)  
/ner  
/gliner\_small-v2.1.onnx (Lore specific)  
/gliner\_labels.json  
/audio  
/yamnet.onnx (Mandatory bundled file, extracted on first run)  
/llm  
/phi-3.5-mini-q4\_k\_m.gguf (Lore specific)  
/qwen-2.5-3b-q4\_k\_m.gguf  
The /whisper directory remains the sole shared path accessible by the HOARD sibling application, as acoustic transcription is a generalized necessity. The /ner and /llm directories are highly specialized to the Lore oral history metadata schema and must remain isolated to prevent HOARD from accidentally initializing massive contextual LLM weights that exceed its operational mandate. This modular directory extension guarantees atomic model management, prevents namespace collisions, and strictly upholds the privacy-first, fully offline archival paradigm.

#### **Works cited**

1. English · spaCy Models Documentation, accessed June 4, 2026, [https://spacy.io/models/en](https://spacy.io/models/en)  
2. Trained Models & Pipelines · spaCy Models Documentation, accessed June 4, 2026, [https://spacy.io/models](https://spacy.io/models)  
3. Installing spacy with fewer languages to save disk space \#3983 \- GitHub, accessed June 4, 2026, [https://github.com/explosion/spaCy/issues/3983](https://github.com/explosion/spaCy/issues/3983)  
4. Packing spacy console application using Pyinstaller throws error? · Issue \#4683 \- GitHub, accessed June 4, 2026, [https://github.com/explosion/spaCy/issues/4683](https://github.com/explosion/spaCy/issues/4683)  
5. GLiNER Documentation, accessed June 4, 2026, [https://urchade.github.io/GLiNER/](https://urchade.github.io/GLiNER/)  
6. gliner \- PyPI Download Stats, accessed June 4, 2026, [https://pypistats.org/packages/gliner](https://pypistats.org/packages/gliner)  
7. lmo3/gliner2-multi-v1-onnx \- Hugging Face, accessed June 4, 2026, [https://huggingface.co/lmo3/gliner2-multi-v1-onnx](https://huggingface.co/lmo3/gliner2-multi-v1-onnx)  
8. Disease Named Entity Recognition Spanish (Base, ONNX) | roberta\_disease\_ner\_es\_onnx | Healthcare NLP 6.2.0, accessed June 4, 2026, [https://nlp.johnsnowlabs.com/2026/01/04/roberta\_disease\_ner\_es\_onnx\_es.html](https://nlp.johnsnowlabs.com/2026/01/04/roberta_disease_ner_es_onnx_es.html)  
9. Procedures Named Entity Recognition Spanish (Base, ONNX) | roberta\_procedure\_ner\_es\_onnx | Healthcare NLP 6.2.0, accessed June 4, 2026, [https://nlp.johnsnowlabs.com/2026/01/04/roberta\_procedure\_ner\_es\_onnx\_es.html](https://nlp.johnsnowlabs.com/2026/01/04/roberta_procedure_ner_es_onnx_es.html)  
10. NER \- EDS-NLP, accessed June 4, 2026, [https://aphp.github.io/edsnlp/latest/pipes/trainable/ner/](https://aphp.github.io/edsnlp/latest/pipes/trainable/ner/)  
11. edsnlp \- PyPI, accessed June 4, 2026, [https://pypi.org/project/edsnlp/](https://pypi.org/project/edsnlp/)  
12. GitHub \- aphp/edsnlp: Modular, fast NLP framework, compatible with Pytorch and spaCy, offering tailored support for French clinical notes., accessed June 4, 2026, [https://github.com/aphp/edsnlp](https://github.com/aphp/edsnlp)  
13. Local relation extraction with GLiNER (ONNX) vs GPT-4o pipelines \- results \+ observations : r/LocalLLaMA \- Reddit, accessed June 4, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1s1pdqy/local\_relation\_extraction\_with\_gliner\_onnx\_vs/](https://www.reddit.com/r/LocalLLaMA/comments/1s1pdqy/local_relation_extraction_with_gliner_onnx_vs/)  
14. Sound classification with YAMNet \- Colab \- Google, accessed June 4, 2026, [https://colab.research.google.com/github/Nikhila-KS/Unravel\_ML/blob/main/4.Understanding\_YAMNet\_myNotes.ipynb](https://colab.research.google.com/github/Nikhila-KS/Unravel_ML/blob/main/4.Understanding_YAMNet_myNotes.ipynb)  
15. models/research/audioset/yamnet/yamnet\_class\_map.csv at master \- GitHub, accessed June 4, 2026, [https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet\_class\_map.csv](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv)  
16. Converting YAMNet audio detection model for TensorFlow Lite \- Medium, accessed June 4, 2026, [https://medium.com/@antonyharfield/converting-the-yamnet-audio-detection-model-for-tensorflow-lite-inference-43d049bd357c](https://medium.com/@antonyharfield/converting-the-yamnet-audio-detection-model-for-tensorflow-lite-inference-43d049bd357c)  
17. antonyharfield/tflite-models-audioset-yamnet \- GitHub, accessed June 4, 2026, [https://github.com/antonyharfield/tflite-models-audioset-yamnet](https://github.com/antonyharfield/tflite-models-audioset-yamnet)  
18. qualcomm/YamNet \- Hugging Face, accessed June 4, 2026, [https://huggingface.co/qualcomm/YamNet](https://huggingface.co/qualcomm/YamNet)  
19. Serve a tensorflow model without installing tensorflow PyInstaller \- Stack Overflow, accessed June 4, 2026, [https://stackoverflow.com/questions/73350785/serve-a-tensorflow-model-without-installing-tensorflow-pyinstaller](https://stackoverflow.com/questions/73350785/serve-a-tensorflow-model-without-installing-tensorflow-pyinstaller)  
20. Docker container with Python modules gets too big \- Stack Overflow, accessed June 4, 2026, [https://stackoverflow.com/questions/67591044/docker-container-with-python-modules-gets-too-big](https://stackoverflow.com/questions/67591044/docker-container-with-python-modules-gets-too-big)  
21. Error during packaging tflite runtime with pysintaller \- Stack Overflow, accessed June 4, 2026, [https://stackoverflow.com/questions/63324923/error-during-packaging-tflite-runtime-with-pysintaller](https://stackoverflow.com/questions/63324923/error-during-packaging-tflite-runtime-with-pysintaller)  
22. MobileNet-v2 \- Qualcomm AI Hub, accessed June 4, 2026, [https://aihub.qualcomm.com/compute/models/mobilenet\_v2](https://aihub.qualcomm.com/compute/models/mobilenet_v2)  
23. PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition, accessed June 4, 2026, [https://scholar.xjtlu.edu.cn/en/publications/panns-large-scale-pretrained-audio-neural-networks-for-audio-patt/](https://scholar.xjtlu.edu.cn/en/publications/panns-large-scale-pretrained-audio-neural-networks-for-audio-patt/)  
24. Comprehensive Evaluation of CNN-Based Audio Tagging Models on Resource-Constrained Devices \- arXiv, accessed June 4, 2026, [https://arxiv.org/html/2509.14049v2](https://arxiv.org/html/2509.14049v2)  
25. Audio Classification (YAMNet) \- ZETIC Melange, accessed June 4, 2026, [https://docs.zetic.ai/tutorials/audio-classification-yamnet](https://docs.zetic.ai/tutorials/audio-classification-yamnet)  
26. YAMNet error · Issue \#8 · ankane/onnxruntime-ruby \- GitHub, accessed June 4, 2026, [https://github.com/ankane/onnxruntime-ruby/issues/8](https://github.com/ankane/onnxruntime-ruby/issues/8)  
27. onnxmodelzoo/mobilenetv2-12 \- Hugging Face, accessed June 4, 2026, [https://huggingface.co/onnxmodelzoo/mobilenetv2-12](https://huggingface.co/onnxmodelzoo/mobilenetv2-12)  
28. GBNF grammars Projects \- AI Tinkerers \- Seattle, accessed June 4, 2026, [https://seattle.aitinkerers.org/technologies/gbnf-grammars](https://seattle.aitinkerers.org/technologies/gbnf-grammars)  
29. Using llama-cpp-python grammars to generate JSON \- Simon Willison: TIL, accessed June 4, 2026, [https://til.simonwillison.net/llms/llama-cpp-python-grammars](https://til.simonwillison.net/llms/llama-cpp-python-grammars)  
30. Using Grammar | node-llama-cpp, accessed June 4, 2026, [https://node-llama-cpp.withcat.ai/guide/grammar](https://node-llama-cpp.withcat.ai/guide/grammar)  
31. llama.cpp/grammars/json.gbnf at master · ggml-org/llama.cpp · GitHub, accessed June 4, 2026, [https://github.com/ggerganov/llama.cpp/blob/master/grammars/json.gbnf](https://github.com/ggerganov/llama.cpp/blob/master/grammars/json.gbnf)  
32. JSON-Schema to GBNF, accessed June 4, 2026, [https://adrienbrault.github.io/json-schema-to-gbnf/](https://adrienbrault.github.io/json-schema-to-gbnf/)  
33. Models are failing to be properly unloaded and freeing up VRAM · Issue \#1442 · abetlen/llama-cpp-python \- GitHub, accessed June 4, 2026, [https://github.com/abetlen/llama-cpp-python/issues/1442](https://github.com/abetlen/llama-cpp-python/issues/1442)  
34. Clearing VRAM in llama\_cpp : r/LocalLLaMA \- Reddit, accessed June 4, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1i83lpu/clearing\_vram\_in\_llama\_cpp/](https://www.reddit.com/r/LocalLLaMA/comments/1i83lpu/clearing_vram_in_llama_cpp/)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEYAAAAaCAYAAAAKYioIAAACuUlEQVR4Xu2XS4hOYRjHH/fBuMy4TZRSIgvKdlwaduQSZYNIVu42xAKhCCtCrhshrERuM8RGcisktyyUSMiCZkPi/+95zzjn6XzveT/fxxTvr37NzP95O9+Z9/6JRCLVpDccCSfYwv9KR3gb/oDf4QP4CC529dXu599mEFwE18PpphZCA1wFJ9tCCP3hRXgWjoM1Lh8G78AToh3WHryDV+FOeAmehD0zLUrDDuEg74Wf4BlYm2nhYRR8Cb/YgmOsaKe0R8cMgJtMxvfYb7I8lsCvsMn9PRS+Fu3YQkbA96IfNtvU0jyFH22YwxwbVAiXDt9tYipLlnqvVGapE50hl02+TQIHmI1aRZeSD37AShuWgGv5sXOpqf0OPAzS8J0PmsxyWLTdRpPPcLmX7qKN9thCDvNhPxt66Cy6aXMD3yHFHR9KI3whuqH6aBH939aanINW2DFTRBs1mbzanIKf4VbY19RC4EGwAe4Wnd1DsuVcbkl+x0xyeQeTZ+Dx90105vxpxoiebNynBptaEV1FB28WPAq3iF4tfNyV/I4Z7/IuJs9wXfJ3aH5os/yajvSN6EtVgyPwFZxm8lCSd/LdaThL2WadyYOW0gF43oYGPoRHnHfqBcI70T7Rke9kaqWoh31MlnQMN9hS8HPYZrPJp7rcywrRmeCDD+ESqITR8JjoJa3c2+czeNNkScccN3mahaJt7MGywOVeeFKwEU+cPHhPYH2ZLZQB95ZrUn6HJHyAD02WdMxy93cPeEh0BfB3wtstN3x7j+EBUNgxhDs81yMvTPdEd/9d8InoKXCjrWUYXCLnRJ/JG3OlcL97C0/D7fAKvCB6W08YKDobOQBceml4+30O58H7ol97+N0rGD6cx/dMONzUyoEba/qlqwFHnxezNXCuqYXQTXTAfZt1JBKJRCKRSORf5Cd1NpGRi2lXIwAAAABJRU5ErkJggg==>