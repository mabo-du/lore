# **Architecture and Implementation Report: Local-First RAG Pipeline for Domain Auto-Tagging**

## **Introduction and Contextual Overview**

The development of the Lore desktop application represents a highly specialized endeavor in the domain of archival software engineering. Designed explicitly for archivists processing sensitive oral histories—ranging from indigenous testimonies and post-conflict narratives to institutional records—the software must operate under strict data sovereignty requirements. These requirements necessitate a completely offline, local-first architecture where zero network access is permitted during operation. The application currently features a robust suite of audio processing and natural language understanding tools, including Faster-Whisper for automated speech recognition, Pyannote for speaker diarization, GLiNER2-ONNX for named entity recognition, and Qwen2.5-1.5B for abstract generation.  
The next phase of architectural evolution involves implementing automatic domain taxonomy tagging via a local Retrieval-Augmented Generation (RAG) pipeline. This feature will allow transcript segments to be automatically tagged with highly specialized vocabulary from domain-specific taxonomy packs. However, this implementation is bounded by severe hardware and distribution constraints: the application must remain free of heavy deep learning frameworks like PyTorch, it must compile seamlessly into a PyInstaller \--onedir bundle, it must maintain an initial bundle size of under 400 megabytes, and it must operate reliably on consumer hardware with a maximum of eight gigabytes of random access memory (RAM).  
This comprehensive report evaluates the integration of FastEmbed as the embedding engine and sqlite-vec as the vector store to achieve this RAG-based domain auto-tagging. The analysis addresses the dependency chains, cross-platform dynamic library loading mechanisms, taxonomy schema design, runtime resource contention, and semantic accuracy tuning required to deploy this feature successfully within the specified operational envelope.

## **FastEmbed Integration Architecture and Dependency Audit**

The embedding engine serves as the computational core of the RAG pipeline, responsible for translating raw transcript text and taxonomy definitions into high-dimensional geometric representations. The primary candidate for this role is FastEmbed, a library engineered specifically to bypass the massive storage and memory overhead associated with traditional deep learning frameworks.

### **Dependency Chain and PyTorch-Free Verification**

A rigorous audit of the FastEmbed dependency tree confirms its absolute suitability for a PyTorch-free environment. Traditional text embedding libraries, such as the widely adopted sentence-transformers, rely heavily on the transformers library provided by Hugging Face. This dependency, in turn, mandates the installation of torch (PyTorch) to handle tensor operations and neural network graph execution.1 The inclusion of PyTorch introduces gigabytes of compiled C++ and CUDA binaries, immediately violating the sub-400 megabyte initial bundle size constraint of the Lore application.3  
In stark contrast, an analysis of the pyproject.toml configuration for FastEmbed reveals a highly restricted and optimized dependency graph. The core dependencies required for runtime execution are strictly limited to lightweight utility and computational packages.

| FastEmbed Dependency | Primary Function within Pipeline | PyTorch / Transformers Transitive Link |
| :---- | :---- | :---- |
| onnxruntime | Executes the compiled neural network graph via the CPU execution provider. | None. Operates independently as a C++ inference engine.4 |
| tokenizers | Handles text encoding and subword token generation via Rust bindings. | None. Replaces transformers.AutoTokenizer entirely.4 |
| numpy | Manages the output arrays and vector geometry in memory. | None. Standard scientific computing library.5 |
| huggingface-hub | Facilitates the downloading of model weights and configuration files. | None. Handles network I/O and caching only.5 |
| requests & tqdm | Manages HTTP requests and progress visualization during model acquisition. | None. Standard networking and UI utilities.4 |
| loguru | Provides structured logging for diagnostic output. | None. Standard logging utility.5 |

This dependency architecture guarantees that FastEmbed operates entirely without triggering any hidden imports of deep learning training frameworks. By utilizing the Open Neural Network Exchange (ONNX) format, FastEmbed shifts the computational burden from PyTorch to the ONNX Runtime, executing inference directly via the CPU execution provider. This architectural decision ensures that the heavy abstractions necessary for training neural networks are entirely bypassed during production inference.4

### **Tokenization Implementation Mechanisms**

A critical vulnerability in deploying lightweight language models is the tokenization phase. If an embedding library imports transformers.AutoTokenizer to convert raw strings into input tensors, it risks inadvertently importing the broader transformers ecosystem and, consequently, torch.6 FastEmbed circumvents this entirely by leveraging the standalone Hugging Face tokenizers package.4  
The tokenizers library is written natively in Rust and provides multi-architecture native bindings for Python.7 When transcript segments are passed into FastEmbed, this Rust-based engine processes the strings into the required numerical representations: input IDs, attention masks, and token type IDs. Because the tokenization logic is decoupled from the transformers ecosystem and compiled as a native C-extension, it operates with exceptional memory efficiency and speed.6 The resulting arrays are passed directly as contiguous C-arrays into the ONNX Runtime session, completely isolating the process from any PyTorch-related memory bloat.

### **Installed Size and Cold-Start Latency Analysis**

The total installed footprint of the embedding pipeline is dictated by the size of the Python packages and the specific quantization state of the selected model. The FastEmbed library and its associated dependencies require approximately 150 megabytes of disk space, with onnxruntime and numpy constituting the bulk of this volume.  
When evaluating the all-MiniLM-L6-v2 model, the size varies significantly based on numerical precision. Exported to the ONNX format in full 32-bit floating-point precision (fp32), the model requires roughly 90 megabytes.9 However, by utilizing INT8 quantized weights—where the 32-bit floats are compressed into 8-bit integers—the ONNX graph size is reduced dramatically to approximately 22 to 23 megabytes.9 Integrating the INT8 quantized model alongside the FastEmbed dependencies results in a combined application footprint of less than 180 megabytes. This leaves ample room for the PyQt6 GUI framework, the SQLite binaries, and the Whisper components, ensuring the application comfortably satisfies the sub-400 megabyte initial bundle constraint.  
Cold-start latency represents the initialization penalty incurred when the application loads the ONNX session and maps the model weights into the physical memory of the CPU. On standard consumer hardware, benchmarks indicate that loading the quantized all-MiniLM-L6-v2 model via FastEmbed requires between 230 and 280 milliseconds.11 This initialization is typically performed asynchronously during the application's boot sequence. Because this latency penalty is incurred only once per application lifecycle, it poses no disruption to the user experience.

### **Alternative Fallback: Raw ONNX Runtime Integration**

While FastEmbed provides an excellent abstraction layer, architectural resilience demands evaluating a raw fallback approach. If packaging conflicts arise—such as the known issues regarding onnxruntime-gpu overriding CPU binaries in certain virtual environments 13—the development team can directly implement a raw onnxruntime and tokenizers pipeline.14  
This raw implementation involves exporting a sentence-transformers model to ONNX format manually, distributing the tokenizer.json and model.onnx files, and writing custom Python bindings to initialize an onnxruntime.InferenceSession.15 The custom script would utilize the Rust tokenizers library to generate the input tensors and feed them directly into the ONNX session. While this approach strips away the downloading, caching, and batching logic provided by FastEmbed, potentially saving a few megabytes of code, it significantly increases the maintenance burden on the engineering team.16 Given that FastEmbed already utilizes this precise underlying mechanism, retaining FastEmbed is highly recommended unless intractable dependency resolution failures occur during the PyInstaller build process.

## **SQLite-Vec Integration and Cross-Platform Distribution**

The vector store component of the RAG pipeline must operate locally and offline, matching transcript embeddings against the taxonomy pack embeddings to surface relevant tags. The sqlite-vec library, a pure C extension for SQLite, introduces high-performance vector processing capabilities without the need for external service dependencies or complex standalone vector databases.18

### **PyInstaller \--onedir Execution and Dynamic Library Resolution**

Integrating a dynamically loadable SQLite extension into a PyInstaller \--onedir bundle presents significant filesystem resolution challenges. When PyInstaller compiles a Python application, it bundles the interpreter, dependencies, and external binaries into a specific directory layout. At runtime, the application extracts necessary components into a temporary directory (typically referenced via the sys.\_MEIPASS environment variable).19  
By default, the sqlite\_vec.load() Python helper attempts to locate the compiled binary within its standard site-packages structure.20 In a frozen PyInstaller environment, this relative path resolution frequently fails, leading to an OperationalError. To circumvent this, the vec0 extension binaries must be explicitly declared in the PyInstaller .spec file under the binaries or datas arrays.22  
During runtime execution, the application must construct an absolute path to the vec0 binary by combining the sys.\_MEIPASS path with the binary filename.19 The SQLite extension loading sequence within Python must follow a strict procedural order. First, the database connection is initialized. Second, extension loading is explicitly enabled via the enable\_load\_extension(True) method. Third, the dynamically resolved absolute path is passed to the load\_extension() method. Finally, extension loading is disabled to prevent arbitrary code execution vulnerabilities.20

### **The Windows \#45 DLL Resolution Vulnerability**

A critical cross-platform deployment risk involves Windows execution, documented extensively in the sqlite-vec repository as Issue \#45. Users frequently report that attempting to load the vec0.dll binary results in a silent failure or an explicit error stating: "The specified module could not be found".21  
This error does not indicate that the vec0.dll file is missing from the disk; rather, it indicates that the Windows OS loader failed to resolve the transitive dependencies required by the DLL. This issue stems from compiler disparities.21 When the vec0.dll is compiled using MinGW (Minimalist GNU for Windows), it dynamically links against specific GNU runtime libraries, such as libgcc\_s\_seh-1.dll.25 These libraries are not native to standard consumer Windows installations. When the Lore application attempts to load the extension on a generic Windows machine, the OS cannot locate the GNU runtime, aborting the module load.27  
To permanently mitigate this vulnerability, the PyInstaller build pipeline must package the MSVC (Microsoft Visual C++) compiled version of vec0.dll instead of the MinGW version.21 The MSVC build dynamically links against the standard Microsoft Visual C++ Redistributable runtime, which is ubiquitous on modern consumer hardware. If the MSVC build is unavailable and the MinGW build is strictly required, the engineering team must utilize a dependency walker to identify all transitive GNU DLLs and explicitly bundle them within the PyInstaller binaries specification, ensuring they reside in the same directory as the executable.

### **macOS enable\_load\_extension System Restrictions**

Deploying SQLite extensions on macOS introduces a distinct architectural hurdle. The default system-provided Python distributions on macOS, which link against the Apple-provided native SQLite library, completely disable the enable\_load\_extension API for system security reasons. Attempting to call this function results in an AttributeError or an OperationalError.20  
Because the Lore application is distributed to end-users who cannot be expected to recompile their system libraries, relying on the host OS Python is an unviable strategy. The PyInstaller build system must package an independent, statically linked Python distribution alongside a custom-compiled libsqlite3.dylib that has extension loading explicitly enabled during compilation.20 Alternatively, the application can bypass the standard library sqlite3 module entirely by statically compiling the pysqlite3 or apsw (Another Python SQLite Wrapper) packages. These third-party wrappers bundle their own modern, statically linked SQLite engines, ensuring that the application maintains absolute control over the SQLite compilation flags and successfully bypasses Apple's artificial system restrictions.28

### **In-Memory Execution and Vector Query Latency**

The domain auto-tagging feature relies on taxonomy packs consisting of approximately 200 to 500 specialized terms. Given this highly constrained dataset size, the sqlite-vec extension can and should operate entirely within an in-memory database (:memory:) rather than a file-backed database on disk.20  
Operating the vector store in-memory completely eliminates the latency associated with disk I/O operations. The vec0 virtual table utilizes Single Instruction, Multiple Data (SIMD) instruction sets—specifically Advanced Vector Extensions (AVX) on x86 architectures and NEON on ARM architectures—to accelerate vector mathematics at the hardware level.30 Consequently, executing a brute-force cosine similarity scan across 500 vectors of 384 dimensions each requires sub-millisecond computation time.

| Metric | Estimated Performance on Mid-Range CPU (i5, 8GB RAM) |
| :---- | :---- |
| Database Initialization (:memory:) | \< 5 ms |
| Vector Index Loading (500 terms) | \< 10 ms |
| Brute-force Cosine Similarity Query (Top-K) | \< 1 ms |
| Memory Overhead of Vector Index | \< 2 MB |

The query latency for a virtual table containing 500 taxonomy terms on a standard consumer processor is virtually unmeasurable. Because the vector space is so small, there is no need to implement complex Approximate Nearest Neighbor (ANN) indices like Hierarchical Navigable Small World (HNSW) graphs; a brute-force exact nearest neighbor search is both faster to initialize and perfectly accurate.30 The computational bottleneck of the RAG pipeline shifts entirely to the embedding model, completely exonerating the SQLite vector index.

## **Taxonomy Pack Architecture and Semantic Schema**

The distribution format and structural schema of the domain taxonomy packs dictate the initialization speed, semantic accuracy, and operational flexibility of the auto-tagging feature.

### **File Format Evaluation and Pre-Embedding Optimization**

Taxonomy packs can be distributed in various formats, including raw text formats like JSON or CSV, or as pre-computed, pre-embedded SQLite database files. Given the offline nature of the Lore application and the strict computational budgets, distributing pre-embedded SQLite databases is the mathematically optimal strategy.  
If a user toggles an "indigenous heritage" taxonomy pack distributed as a JSON file, the application must extract the text and pass all 500 terms through the FastEmbed ONNX model upon application startup. While the model is efficient, embedding 500 terms introduces a noticeable CPU spike and unnecessary memory allocation phase, delaying the availability of the tagging feature.  
By distributing the taxonomy packs as pre-computed SQLite databases, the application simply executes an ATTACH DATABASE 'domain\_tax.db' AS domain command. The vectors are instantly available in memory for the vec0 virtual table to query against.32  
It is crucial to clarify the runtime model requirement in this scenario. While shipping pre-embedded packs means the user does not need the embedding model to load the taxonomies into the database, the query path absolutely still requires the embedding model to remain active in memory. As the archivist processes audio, the resulting transcript segments must be vectorized in real-time by the FastEmbed model to generate the query vector that searches against the pre-embedded taxonomy database.

### **Hierarchical SKOS Schema Design**

A flat taxonomy structure, consisting merely of a {term, definition} pairing, is insufficient for mapping complex historical, legal, or sensitive archival contexts. Semantic similarity models calculate relevance based on geometric proximity in high-dimensional space. To maximize this relevance, the taxonomy schema should adopt a hierarchical model inspired by the Simple Knowledge Organization System (SKOS) standard.  
A robust SKOS-aligned schema within the SQLite database should encompass the following fields:

* term\_id: The primary key identifier.  
* preferred\_term: The primary nomenclature (e.g., "Sorry Business").  
* definition: A detailed, context-rich explanation of the term.  
* broader\_term: The parent category (e.g., "Funerary Rites").  
* related\_terms: A comma-separated list of semantically adjacent concepts (e.g., "Mourning, Ceremony, Bereavement").  
* embedding: A float or int8 vector storing the representation of the data.29

To generate the optimal embedding for the pre-computed database, the text passed to the embedding model should concatenate the preferred\_term, the definition, and the related\_terms. Embedding this rich combination significantly enhances the semantic density of the resulting vector, providing a substantially larger and more accurate geometric target for the transcript segments to match against during the retrieval phase.

### **Index Size Estimation and Versioning Mechanisms**

The storage footprint of these pre-embedded taxonomy packs is remarkably small, reinforcing their viability as bundled assets. A single 384-dimensional vector utilizing 32-bit floating-point (float32) precision requires exactly 1,536 bytes of storage space (384 dimensions multiplied by 4 bytes per dimension).34  
For a taxonomy pack containing 500 specialized terms, the raw vector data consumes only 768,000 bytes (approximately 768 kilobytes). When adding the overhead of the sqlite-vec indexing structure, the row identifiers, and the raw text metadata stored alongside the vectors, a single taxonomy pack database file will reliably remain under 1.5 megabytes.33 This minuscule footprint permits the inclusion of dozens of distinct domain packs without threatening the 400 megabyte initial bundle constraint.  
Handling taxonomy pack versioning and updates requires embedding a metadata table within the SQLite pack format. A simple pack\_metadata table containing version, last\_updated, and schema\_revision allows the application to check for updates if the user connects to the internet, or simply validates compatibility when importing custom user-created packs from external offline drives.

## **Runtime Pipeline Performance and Hardware Profiling**

Integrating a real-time RAG pipeline into an application that is already executing asynchronous audio transcription imposes severe CPU scheduling constraints. Profiling the execution on the target mid-range hardware (Intel i5 processor, 8GB of RAM) is vital to ensure application stability.

### **Embedding and Query Latency**

The performance breakdown per transcript segment—typically representing a 5 to 15-second audio utterance yielding 20 to 50 words—demonstrates that the pipeline is highly capable of real-time execution.

| Pipeline Phase | Execution Mechanism | Estimated Latency per Segment |
| :---- | :---- | :---- |
| **Tokenization** | Rust-based Hugging Face tokenizers | 2 \- 5 ms 6 |
| **Vector Generation** | ONNX Runtime CPU Provider (INT8 weights) | 15 \- 35 ms 11 |
| **Vector Search** | sqlite-vec in-memory cosine matching | \< 1 ms |
| **Total Segment Latency** | Full RAG execution path | **\~20 \- 40 ms** |

Given that a transcript segment takes several seconds to be spoken and subsequently transcribed by Faster-Whisper, a 40-millisecond tagging delay is imperceptible to the end-user.

### **Streaming Execution versus Batch Processing**

While FastEmbed supports efficient batch processing—passing multiple segments simultaneously to leverage matrix multiplication efficiencies within the CPU cache 4—a streaming approach is highly recommended for this desktop architecture.  
Processing segments individually as they stream asynchronously from the Faster-Whisper pipeline ensures real-time UI updates, providing the archivist with immediate visual feedback as auto-tags appear live alongside the transcribing audio. Batching should be strictly reserved for post-processing scenarios where a user imports an entire pre-existing text transcript for retrospective tagging.

### **Resource Contention and Thread Affinity Management**

The most significant existential threat to the stability of the Lore application is CPU thrashing. Both the Faster-Whisper ASR engine (utilizing CTranslate2) and the FastEmbed engine (utilizing the ONNX Runtime) heavily rely on OpenMP or standard C++ thread pools to accelerate complex matrix operations by parallelizing them across multiple physical CPU cores.36  
If a consumer machine features a 4-core processor, and the application attempts to run a Whisper inference utilizing all four threads concurrently with a RAG pipeline triggering an ONNX inference also demanding four threads, the operating system's CPU scheduler will thrash. This contention leads to severe context-switching overhead, thermal throttling, and extreme latency spikes across the entire application interface.  
To mitigate this resource contention, the engineering team must implement stringent thread management strategies. First, sequential execution is preferred. The RAG embedding sequence should be deferred until the active transcription of the current segment is entirely complete; the two neural networks must not execute their forward passes at the exact same millisecond. Second, thread affinity limits must be explicitly configured on the embedding execution provider. The FastEmbed ONNX session should be explicitly restricted to a single thread: session\_options.intra\_op\_num\_threads \= 1 session\_options.inter\_op\_num\_threads \= 1 Because the RAG pipeline processes very short text segments requiring only 30 milliseconds of compute time, restricting it to a single CPU core is highly efficient.12 This architectural decision frees the remaining cores to focus entirely on the computationally heavier Whisper audio processing.

### **Comprehensive Memory Footprint**

Operating within the 8GB RAM target requires meticulous memory accounting. The simultaneous execution of the models yields the following approximate resident set size (RSS) allocations:

* **Faster-Whisper (INT8 Quantized)**: Consumes approximately 1.5GB to 2.0GB of RAM, depending heavily on the configured beam size and vocabulary caching mechanisms.  
* **PyQt6 GUI and Python Runtime**: Consumes roughly 200MB to 300MB of RAM for the graphical interface, event loop, and standard library objects.  
* **FastEmbed (all-MiniLM-L6-v2 INT8)**: While the quantized ONNX model file is only \~23MB on disk 9, the actual memory required when loaded into the ONNX Runtime session, including execution buffers and tensor allocations, is approximately 70MB to 120MB.11  
* **sqlite-vec In-Memory Database**: Consumes less than 5MB for the indices and vector mathematics.30

The cumulative simultaneous memory footprint of the entire application demands roughly 2.5 gigabytes of RAM. This remains safely and conservatively below the 8GB threshold, leaving over 5 gigabytes of overhead for the host operating system and other concurrent background applications. Out-of-memory (OOM) application crashes are highly improbable under this specific architectural configuration.

## **Semantic Accuracy, Relevance, and Multilingual Adaptability**

Vector search operates strictly on geometric and mathematical similarity rather than traditional keyword or boolean matching. Tuning the retrieval parameters is critical to ensure archivists working with sensitive data are not overwhelmed with hallucinatory or false-positive tags.

### **Cosine Distance Thresholding**

When executing a MATCH query in sqlite-vec, a cosine distance or L2 distance scalar is returned representing the proximity of the vectors.18 Cosine distance typically ranges from 0.0 (indicating identical vectors) to 2.0 (indicating diametrically opposed vectors). A raw K-Nearest Neighbors (KNN) query, written as ORDER BY distance LIMIT k 18, is insufficient for auto-tagging. A raw KNN query will always return the top-k results, even if the transcript segment is completely irrelevant to every term in the taxonomy.  
A hard mathematical threshold must be applied to the query results. Based on typical MiniLM distributions in semantic space, a cosine distance threshold of \<= 0.35 (which correlates to a cosine similarity of \>= 0.65) provides a robust baseline for semantic relevance. This threshold balances precision (avoiding spurious, incorrect tags) against recall (ensuring relevant tags are not ignored). The application should expose a "Tagging Sensitivity" slider in the Settings UI, allowing the archivist to adjust this mathematical threshold dynamically based on the noise level of the transcript.

### **Out-of-Domain Vocabulary and Asymmetric Search Considerations**

General-purpose embedding models like all-MiniLM-L6-v2 are trained on massive, generalized public datasets, including Wikipedia, Reddit, and generic web crawls.38 Consequently, they face severe semantic degradation when encountering highly specific domain terminology, such as indigenous cultural concepts ("songline", "dreaming", "sorry business") or specialized transitional justice legal terminology.38  
This degradation occurs fundamentally at the tokenization layer. Standard models utilize subword tokenization algorithms like WordPiece or Byte-Pair Encoding (BPE).16 When an unknown indigenous term is encountered, the tokenizer shatters the word into nonsensical sub-tokens, effectively destroying its semantic meaning before it even enters the neural network. To mitigate this, the SKOS taxonomy schema (as previously established) is vital. By embedding highly descriptive, standard-English definitions alongside the specialized terms, the model receives a massive injection of context. When the transcript mentions "sorry business", the embedding model might struggle with the specific phrase, but the surrounding contextual text of the speaker's utterance will mathematically align with the rich definition vector stored in the taxonomy pack.  
Furthermore, the pipeline must account for asymmetric versus symmetric search. Models belonging to the FlagEmbedding family (such as BAAI/bge-small-en-v1.5) utilize prefix tuning during training. They explicitly expect the prefix query: to be prepended to the search text, and the prefix passage: to be prepended to the indexed documents.1 If a bge model is utilized in the Lore application, all taxonomy terms must be embedded with the passage: prefix, and the incoming transcript segments must be prepended with the query: prefix.4 Failure to respect these asymmetric prefixes results in catastrophic accuracy degradation. Conversely, standard MiniLM models generally perform symmetric search and do not require any prefixing logic.

### **Multilingual Alternatives under 100 Megabytes**

Oral history archives frequently deal with mixed-language testimonies, extensive code-switching, or entirely non-English content. While all-MiniLM-L6-v2 is exceptionally lightweight and fast, it is strictly an English-only model.9 Relying on it for diverse indigenous testimonies poses a severe risk of semantic failure and misclassification.  
Fortunately, highly optimized, ONNX-compatible multilingual models have recently been developed that perfectly fit the strict sub-400 megabyte application requirement:

1. **ibm-granite/granite-embedding-107m-multilingual (or 97m-r2)**: This modern model supports over 100 languages natively. When exported and quantized to INT8 precision in the ONNX format, the model weights are reduced to approximately 98 megabytes.10 It produces 384-dimensional output vectors, making it a perfect, drop-in architectural replacement for the all-MiniLM-L6-v2 model.10 It provides vastly superior zero-shot performance on non-English text while maintaining exceptionally high throughput on standard CPU architectures.  
2. **intfloat/multilingual-e5-small**: This is another highly capable model supporting over 100 languages. While the fp32 ONNX version sits at approximately 470 megabytes 40, INT8 quantization reduces it to approximately 112 megabytes.42 It also yields 384-dimensional output and provides excellent semantic clustering capabilities for non-English archival data.43

Transitioning the Lore architecture from the default all-MiniLM-L6-v2 to the ibm-granite-107m-multilingual INT8 model is strongly recommended for this project. The additional 75 megabytes of model weight is easily absorbed by the overarching 400 megabyte bundle constraint, and the semantic accuracy improvements for post-conflict and indigenous oral histories will be transformational for the archivists utilizing the software.

## **Risk Matrix and Architectural Trade-offs**

The integration of complex, local-first machine learning components requires a delicate balancing of conflicting constraints. The following matrix details the primary architectural risks inherent to this RAG pipeline and their corresponding mitigation strategies.

| Risk Category | Specific Threat Profile | Impact on Application | Proposed Mitigation Strategy |
| :---- | :---- | :---- | :---- |
| **Bundle Size Constraint** | The embedding model and dependencies push the PyInstaller bundle over the 400MB threshold. | Immediate violation of hard constraints; deployment failure for offline users. | Strictly utilize INT8 quantized ONNX models (reducing weights to 22MB \- 112MB). Download models dynamically on first launch to keep the initial executable footprint minimal.9 |
| **Cross-Platform Compatibility** | The vec0.dll binary fails to load on target Windows machines due to missing MinGW runtime libraries. | The application crashes during sqlite-vec extension initialization, rendering the feature useless.44 | Package the MSVC-compiled variant of sqlite-vec. Inject absolute load paths dynamically via the PyInstaller sys.\_MEIPASS variable at runtime.19 |
| **Resource Contention** | FastEmbed and Faster-Whisper simultaneously spawn OpenMP thread pools, thrashing the OS CPU scheduler. | UI freezing, thermal hardware throttling, and extreme latency spikes across all operations. | Isolate execution scopes. Restrict the ONNX session to a single thread (intra\_op\_num\_threads=1).12 Process RAG operations synchronously only *after* transcript segment generation completes. |
| **Semantic Inaccuracy** | The English-only all-MiniLM-L6-v2 model catastrophically misinterprets sensitive indigenous terms due to subword shattering.9 | The generation of offensive, irrelevant, or hallucinatory tags applied to sensitive archival materials. | Transition to the ibm-granite-107m-multilingual INT8 model (\~98MB).10 Mandate the use of SKOS hierarchical taxonomy structures to define terms contextually in standard English. |
| **Memory Exhaustion (OOM)** | The cumulative RAM allocation of Whisper, PyQt6, FastEmbed, and SQLite exceeds 8GB. | The application is terminated unexpectedly by the host operating system's OOM killer on mid-range hardware. | Execute sqlite-vec exclusively via :memory: databases (requiring \<5MB overhead). Retain heavily quantized INT8 models to cap the total application resident memory well below 2.5GB.11 |

By adhering strictly to these mitigation strategies, the domain auto-tagging RAG pipeline can be integrated seamlessly into the Lore application, delivering powerful semantic analysis to archivists without compromising the software's offline, local-first integrity.

#### **Works cited**

1. fastembed \- PyPI, accessed June 4, 2026, [https://pypi.org/project/fastembed/](https://pypi.org/project/fastembed/)  
2. fastembed \- PyPI, accessed June 4, 2026, [https://pypi.org/project/fastembed/0.1.2/](https://pypi.org/project/fastembed/0.1.2/)  
3. \[D\] Embeddings and docker file \- comparison between two libraries \- Is there something better than ONNX? : r/MachineLearning \- Reddit, accessed June 4, 2026, [https://www.reddit.com/r/MachineLearning/comments/1gn87vi/d\_embeddings\_and\_docker\_file\_comparison\_between/](https://www.reddit.com/r/MachineLearning/comments/1gn87vi/d_embeddings_and_docker_file_comparison_between/)  
4. FastEmbed: Qdrant's Efficient Python Library for Embedding Generation, accessed June 4, 2026, [https://qdrant.tech/articles/fastembed/](https://qdrant.tech/articles/fastembed/)  
5. ITK-bm25s-extended-fastembed \- PyPI Package Dependencies \- S, accessed June 4, 2026, [https://socket.dev/pypi/package/ITK-bm25s-extended-fastembed/dependencies/0.1.2/tar-gz](https://socket.dev/pypi/package/ITK-bm25s-extended-fastembed/dependencies/0.1.2/tar-gz)  
6. fastembed \- NPM, accessed June 4, 2026, [https://www.npmjs.com/package/fastembed](https://www.npmjs.com/package/fastembed)  
7. fastembed \- crates.io: Rust Package Registry, accessed June 4, 2026, [https://crates.io/crates/fastembed](https://crates.io/crates/fastembed)  
8. FastEmbed-js ⚡️ \- Generate vector embeddings in NodeJS \- GitHub, accessed June 4, 2026, [https://github.com/Anush008/fastembed-js](https://github.com/Anush008/fastembed-js)  
9. mobile\_rag\_engine/docs/guides/model\_setup.md at main \- GitHub, accessed June 4, 2026, [https://github.com/dev07060/mobile\_rag\_engine/blob/main/docs/guides/model\_setup.md](https://github.com/dev07060/mobile_rag_engine/blob/main/docs/guides/model_setup.md)  
10. Granite Embedding Multilingual R2: Open Apache 2.0 Multilingual Embeddings with 32K Context — Best Sub-100M Retrieval Quality \- Hugging Face, accessed June 4, 2026, [https://huggingface.co/blog/ibm-granite/granite-embedding-multilingual-r2](https://huggingface.co/blog/ibm-granite/granite-embedding-multilingual-r2)  
11. GSoC 2026 Proposal Draft – Idea 3: AI-Based Categorisation – Sasha \- Joplin Forum, accessed June 4, 2026, [https://discourse.joplinapp.org/t/gsoc-2026-proposal-draft-idea-3-ai-based-categorisation-sasha/49327/3](https://discourse.joplinapp.org/t/gsoc-2026-proposal-draft-idea-3-ai-based-categorisation-sasha/49327/3)  
12. Building Sentence Transformers in Rust: A Practical Guide with Burn, ONNX Runtime, and Candle \- DEV Community, accessed June 4, 2026, [https://dev.to/mayu2008/building-sentence-transformers-in-rust-a-practical-guide-with-burn-onnx-runtime-and-candle-281k](https://dev.to/mayu2008/building-sentence-transformers-in-rust-a-practical-guide-with-burn-onnx-runtime-and-candle-281k)  
13. GPU users: onnxruntime (CPU) overwrites onnxruntime-gpu binaries when both are installed by pip/uv · Issue \#608 · qdrant/fastembed \- GitHub, accessed June 4, 2026, [https://github.com/qdrant/fastembed/issues/608](https://github.com/qdrant/fastembed/issues/608)  
14. Text Embedding in Ruby \- Fanatical Code, accessed June 4, 2026, [https://fanaticalcode.com/blog/text-embedding-in-ruby/](https://fanaticalcode.com/blog/text-embedding-in-ruby/)  
15. No Python, No Problem: Model Inference with ONNX in Java | by Carlos Martínez \- Medium, accessed June 4, 2026, [https://medium.com/@CarlosMartes/no-python-no-problem-model-inference-with-onnx-in-java-2001cf014dd5](https://medium.com/@CarlosMartes/no-python-no-problem-model-inference-with-onnx-in-java-2001cf014dd5)  
16. Transformers.js vs ONNX Runtime Web: Browser ML 2026 \- PkgPulse, accessed June 4, 2026, [https://www.pkgpulse.com/guides/transformersjs-vs-onnx-runtime-web-2026](https://www.pkgpulse.com/guides/transformersjs-vs-onnx-runtime-web-2026)  
17. High level LLM libraries? \- Help & Support \- Crystal Forum, accessed June 4, 2026, [https://forum.crystal-lang.org/t/high-level-llm-libraries/8221](https://forum.crystal-lang.org/t/high-level-llm-libraries/8221)  
18. asg017/sqlite-vec: A vector search SQLite extension that runs anywhere\! \- GitHub, accessed June 4, 2026, [https://github.com/asg017/sqlite-vec](https://github.com/asg017/sqlite-vec)  
19. Problem with SQLite3 and python using Pyinstaller \- Stack Overflow, accessed June 4, 2026, [https://stackoverflow.com/questions/63443661/problem-with-sqlite3-and-python-using-pyinstaller](https://stackoverflow.com/questions/63443661/problem-with-sqlite3-and-python-using-pyinstaller)  
20. sqlite-vec in Python | sqlite-vec \- Alex Garcia, accessed June 4, 2026, [https://alexgarcia.xyz/sqlite-vec/python.html](https://alexgarcia.xyz/sqlite-vec/python.html)  
21. Pre-compiled loadable extension won't load on Python (3.10.8, Win11) · Issue \#13 · asg017/sqlite-vec \- GitHub, accessed June 4, 2026, [https://github.com/asg017/sqlite-vec/issues/13](https://github.com/asg017/sqlite-vec/issues/13)  
22. Problem Creating One File exe with pyinstaller and pysqlcipher \- Stack Overflow, accessed June 4, 2026, [https://stackoverflow.com/questions/61658608/problem-creating-one-file-exe-with-pyinstaller-and-pysqlcipher](https://stackoverflow.com/questions/61658608/problem-creating-one-file-exe-with-pyinstaller-and-pysqlcipher)  
23. Technetium1/stars: My stars. View raw for full list. \- GitHub, accessed June 4, 2026, [https://github.com/Technetium1/stars](https://github.com/Technetium1/stars)  
24. Pre-compiled extension does not load on Windows 11 (\`The ..., accessed June 4, 2026, [https://github.com/asg017/sqlite-vec/issues/45](https://github.com/asg017/sqlite-vec/issues/45)  
25. Author here, happy to answer any questions\! Been working on this for a while, so... | Hacker News, accessed June 4, 2026, [https://news.ycombinator.com/item?id=41140506](https://news.ycombinator.com/item?id=41140506)  
26. Sqlite-vec: Work-in-progress vector search SQLite extension that runs anywhere | Hacker News, accessed June 4, 2026, [https://news.ycombinator.com/item?id=41137658](https://news.ycombinator.com/item?id=41137658)  
27. Unable to load SQLite extension from other folder \- Stack Overflow, accessed June 4, 2026, [https://stackoverflow.com/questions/66210522/unable-to-load-sqlite-extension-from-other-folder](https://stackoverflow.com/questions/66210522/unable-to-load-sqlite-extension-from-other-folder)  
28. TILs on sqlite \- Simon Willison, accessed June 4, 2026, [https://til.simonwillison.net/sqlite](https://til.simonwillison.net/sqlite)  
29. Embedded Intelligence: How SQLite-vec Delivers Fast, Local Vector Search for AI., accessed June 4, 2026, [https://dev.to/aairom/embedded-intelligence-how-sqlite-vec-delivers-fast-local-vector-search-for-ai-3dpb](https://dev.to/aairom/embedded-intelligence-how-sqlite-vec-delivers-fast-local-vector-search-for-ai-3dpb)  
30. How sqlite-vec Works for Storing and Querying Vector Embeddings | by Stephen Collins, accessed June 4, 2026, [https://medium.com/@stephenc211/how-sqlite-vec-works-for-storing-and-querying-vector-embeddings-165adeeeceea](https://medium.com/@stephenc211/how-sqlite-vec-works-for-storing-and-querying-vector-embeddings-165adeeeceea)  
31. Semantic search in Rails using sqlite-vec, Kamal and Docker \- Blog of Marian Posaceanu, accessed June 4, 2026, [https://marianposaceanu.com/articles/semantic-search-in-rails-using-sqlite-vec-kamal-and-docker](https://marianposaceanu.com/articles/semantic-search-in-rails-using-sqlite-vec-kamal-and-docker)  
32. Vector search in 7 different programming languages using SQL | Alex Garcia's Blog, accessed June 4, 2026, [https://alexgarcia.xyz/blog/2024/sql-vector-search-languages/index.html](https://alexgarcia.xyz/blog/2024/sql-vector-search-languages/index.html)  
33. Retrieval Augmented Generation in SQLite | Towards Data Science, accessed June 4, 2026, [https://towardsdatascience.com/retrieval-augmented-generation-in-sqlite/](https://towardsdatascience.com/retrieval-augmented-generation-in-sqlite/)  
34. reference.yaml \- asg017/sqlite-vec \- GitHub, accessed June 4, 2026, [https://github.com/asg017/sqlite-vec/blob/main/reference.yaml](https://github.com/asg017/sqlite-vec/blob/main/reference.yaml)  
35. \[New Plugin\] NoCoddo AI: Run LLMs inside the browser. 100% Free, Private & Offline, accessed June 4, 2026, [https://forum.bubble.io/t/new-plugin-nocoddo-ai-run-llms-inside-the-browser-100-free-private-offline/387065](https://forum.bubble.io/t/new-plugin-nocoddo-ai-run-llms-inside-the-browser-100-free-private-offline/387065)  
36. FAISS | Haystack \- deepset AI, accessed June 4, 2026, [https://haystack.deepset.ai/integrations/faiss](https://haystack.deepset.ai/integrations/faiss)  
37. Embedding \+ Rerank Gateway: Rust vs Python (28% Faster, 67% Less RAM) | NavyaAI, accessed June 4, 2026, [https://www.navyaai.com/blog/embedding-rerank-gateway-high-performance](https://www.navyaai.com/blog/embedding-rerank-gateway-high-performance)  
38. Why, When and How to Fine-Tune a Custom Embedding Model | Weaviate, accessed June 4, 2026, [https://weaviate.io/blog/fine-tune-embedding-model](https://weaviate.io/blog/fine-tune-embedding-model)  
39. paraphrase-multilingual-MiniLM-L12-v2 | PromptLayer Models, accessed June 4, 2026, [https://www.promptlayer.com/models/paraphrase-multilingual-minilm-l12-v2/](https://www.promptlayer.com/models/paraphrase-multilingual-minilm-l12-v2/)  
40. intfloat/multilingual-e5-small at main \- onnx \- Hugging Face, accessed June 4, 2026, [https://huggingface.co/intfloat/multilingual-e5-small/blob/main/onnx/model.onnx](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/onnx/model.onnx)  
41. onnx/model.onnx · liuyixin617/paraphrase-multilingual-MiniLM-L12-v2 at main, accessed June 4, 2026, [https://huggingface.co/liuyixin617/paraphrase-multilingual-MiniLM-L12-v2/blob/main/onnx/model.onnx](https://huggingface.co/liuyixin617/paraphrase-multilingual-MiniLM-L12-v2/blob/main/onnx/model.onnx)  
42. Teradata/multilingual-e5-small \- Hugging Face, accessed June 4, 2026, [https://huggingface.co/Teradata/multilingual-e5-small](https://huggingface.co/Teradata/multilingual-e5-small)  
43. Enhance Your Semantic Similarity Search with Multilingual Support \- Oracle Blogs, accessed June 4, 2026, [https://blogs.oracle.com/machinelearning/enhance-your-semantic-similarity-search-with-multilingual-support](https://blogs.oracle.com/machinelearning/enhance-your-semantic-similarity-search-with-multilingual-support)  
44. SqliteError: The specified module could not be found on Windows · Issue \#526 · WiseLibs/better-sqlite3 \- GitHub, accessed June 4, 2026, [https://github.com/JoshuaWise/better-sqlite3/issues/526](https://github.com/JoshuaWise/better-sqlite3/issues/526)  
45. Thank you for this, it's really super exciting\! The link on \*See, accessed June 4, 2026, [https://news.ycombinator.com/item?id=41144962](https://news.ycombinator.com/item?id=41144962)