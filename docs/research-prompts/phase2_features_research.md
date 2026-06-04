# Phase 2.0 Feature Exploration: Lore

Please conduct deep research into the following four areas for the next major version of the Lore transcription tool. Our primary architectural constraints remain: all features must execute entirely locally/offline, strictly manage memory to stay under 8GB RAM, and avoid introducing heavy native runtime dependencies (like PyTorch) that would complicate PyInstaller bundling.

For each area, please provide a detailed feasibility analysis, recommended libraries/tools, implementation gotchas, and specific code patterns where applicable.

### 1. Custom Vocabulary / Hotword Prompting (Whisper)
Investigate the feasibility of implementing a "Custom Vocabulary/Hotword" feature in faster-whisper. 
* Does injecting a comma-separated list of terms into the `initial_prompt` effectively bias the CTranslate2 model to correctly spell proper nouns, indigenous terms, or local town names? 
* Are there specific prompt engineering tricks for Whisper to enforce spelling without triggering hallucination loops? 
* Is there an alternative (like biasing the beam search or using `prefix`) that works better in CTranslate2?

### 2. Word-Level Confidence UI & Inline Editing (PyQt6)
We want to highlight individual low-confidence words (e.g., underlining them in red) within a transcript segment, rather than just highlighting the segment as a whole.
* Research how to efficiently render word-level formatting inside a PyQt6 `QListView` using `QTextDocument` and HTML.
* How can we reliably map a click event within the painted rich-text area back to the specific word timestamp in the underlying data model? We need this to allow instantaneous inline editing of a single word without opening a large text editor.

### 3. Automated Archival Packaging (BagIt Integration)
Institutions require deposits in strict standard formats. We want to export the audio, OHMS XML, and metadata directly into a BagIt package.
* Research the feasibility of generating BagIt (RFC 8493) archival packages natively in Python within a PyInstaller bundle. 
* Are there lightweight, pure-Python libraries for creating BagIt structures and generating SHA-256 manifests that do not require external system dependencies? 
* What metadata profiles (e.g., BagIt-Profiles) are standard for oral history deposits that we should support?

### 4. Local Global Search Index (FTS5 + Semantic)
As users process dozens of interviews, they need to search across all of them simultaneously. We want to expand our existing `sqlite-vec` database into a global index.
* How can we build a unified local SQLite database combining FTS5 (Full Text Search) and `sqlite-vec` (Semantic Vector Search) for a desktop Python application? 
* We need to index thousands of transcript segments across multiple projects. Evaluate the performance implications, database schema requirements, and optimal query strategies to do hybrid lexical + semantic search completely offline.
