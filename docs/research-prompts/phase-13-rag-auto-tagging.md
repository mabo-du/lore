# Deep Research Prompt: Phase 13 — RAG-Based Domain Auto-Tagging for Lore

## Context

Lore is an offline, local-first PyQt6 desktop application for oral history transcription. It targets archivists working with sensitive materials (indigenous testimonies, post-conflict narratives, institutional oral histories). The app currently features:

- **Faster-Whisper** (CTranslate2) for ASR
- **Pyannote** for speaker diarization
- **GLiNER2-ONNX** for named entity recognition
- **Qwen2.5-1.5B** (via llama-cpp-python) for abstract generation
- **YAMNet** (ONNX) for audio event detection
- **Whisper logprob-based** anomaly detection for confidence scoring

### Hard Constraints
- **No PyTorch dependencies** — the entire app must run without torch
- **PyInstaller compatible** — must bundle into a `--onedir` package
- **Sub-400MB initial bundle size** (models downloaded on demand)
- **Completely offline** — zero network access during operation
- **≤8GB RAM target** — must not OOM on consumer hardware

## Feature Description

We want to implement automatic domain taxonomy tagging using a local Retrieval-Augmented Generation (RAG) pipeline. The architecture involves:

1. **Embedding engine:** FastEmbed (ONNX-based, no PyTorch) using `all-MiniLM-L6-v2` or similar
2. **Vector store:** sqlite-vec (pure C SQLite extension, ~1MB)
3. **Taxonomy structure:** A "base pack" of ~200 general oral history terms is always active. Users can toggle additional domain-specific packs (indigenous heritage, transitional justice, etc.) via a Settings UI. Custom user-created packs are also supported.

The pipeline: transcript segments are embedded → queried against the sqlite-vec taxonomy index → top-k matching tags are surfaced per segment.

## Research Questions

### 1. FastEmbed Integration Architecture
- What is the exact dependency chain of `fastembed` on PyPI? Does it truly avoid PyTorch at runtime, or does it pull in `transformers`/`torch` transitively?
- What is the total installed size of `fastembed` + its ONNX model (`all-MiniLM-L6-v2`)?
- How does FastEmbed handle tokenization? Does it use compiled ONNX tokenization graphs (via `onnxruntime-extensions`), or does it fall back to `transformers.AutoTokenizer`? This is critical — if it imports `transformers`, it may trigger `torch`.
- Are there alternative lightweight ONNX embedding libraries that are guaranteed PyTorch-free? Evaluate `sentence-transformers` ONNX export + raw `onnxruntime` as a fallback.
- What is the cold-start latency of FastEmbed model loading on CPU? How does this compare to raw ONNX Runtime?

### 2. sqlite-vec Integration
- What is the exact installation method for sqlite-vec in a PyInstaller context? Is it a pip-installable package, or must we bundle the `.so`/`.dll`/`.dylib` manually?
- How do we handle `enable_load_extension()` within Python's `sqlite3` module — is this blocked by default in some Python builds?
- What is the query latency for a `vec0` virtual table containing ~500 taxonomy terms (384-dimensional float32 vectors) on a consumer CPU?
- Can sqlite-vec operate on an in-memory database, or must it use a file-backed DB? What are the memory implications?
- How do we handle cross-platform DLL loading in PyInstaller (especially the Windows `vec0.dll` "specified module could not be found" issue documented in sqlite-vec GitHub issue #45)?

### 3. Taxonomy Pack Architecture
- What is the optimal file format for taxonomy packs? Evaluate JSON, CSV, and SQLite (pre-embedded) as distribution formats.
- If we ship pre-embedded packs (vectors already computed), users won't need the embedding model just to load taxonomies. Is this viable, or does the query path also require the embedding model to vectorize the transcript text?
- How should we handle taxonomy pack versioning and updates?
- What is a reasonable taxonomy term structure? Just `{term, definition}`, or should we include `{term, definition, broader_term, narrower_terms, related_terms}` for hierarchical taxonomies?
- Estimate the sqlite-vec index file size for 500 terms × 384 dimensions × float32.

### 4. Runtime Pipeline Performance
- What is the expected per-segment embedding + query latency on a mid-range CPU (Intel i5, 8GB RAM)?
- Should we batch multiple segments before querying, or process each segment individually as it arrives from the transcription pipeline?
- How do we prevent the RAG pipeline from competing for CPU with the concurrent Faster-Whisper inference? Should we use thread priority, CPU affinity, or simply defer RAG until transcription completes?
- What is the memory footprint of having both the Whisper model AND the MiniLM embedding model loaded simultaneously?

### 5. Accuracy and Relevance
- What similarity threshold (cosine distance) is appropriate for auto-tagging? How do we balance precision (avoiding false tags) vs. recall (missing relevant tags)?
- How well does `all-MiniLM-L6-v2` handle domain-specific terminology that it was never trained on (e.g., indigenous cultural concepts like "songline", "dreaming", "sorry business")?
- Should we use the model's "query" and "passage" prefixes for asymmetric search, or treat both taxonomy terms and transcript segments identically?
- Are there any ONNX-compatible multilingual embedding models that would handle non-English terminology better while staying under 100MB?

## Deliverables

Please provide:
1. A verified dependency audit of `fastembed` confirming PyTorch-free operation
2. A concrete sqlite-vec integration guide for PyInstaller `--onedir` builds, including cross-platform DLL loading
3. Recommended taxonomy pack format with a sample schema
4. Benchmark estimates for the full pipeline (embed + query) on consumer hardware
5. A risk matrix covering bundle size, memory, latency, and accuracy trade-offs
