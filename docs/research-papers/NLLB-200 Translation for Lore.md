# **Advanced Multilingual Translation Architecture for Offline Oral History Archival: Integrating NLLB-200 and CTranslate2**

The preservation and digitization of oral history present unique intersectional challenges between computational linguistics and archival science. When dealing with sensitive materials—spanning indigenous testimonies, post-conflict narratives, and deeply classified institutional histories—the overarching technical imperative is data sovereignty. Software environments deployed for these purposes must guarantee absolute privacy through entirely offline, local-first operation. The integration of offline multilingual translation into a PyQt6 desktop application targeting these archival use cases requires strict adherence to formidable engineering constraints. The application must execute efficiently without network access, run seamlessly on consumer-grade hardware constrained to 8GB of RAM, remain completely independent of large-scale deep learning frameworks such as PyTorch at runtime, and maintain a compact deployment footprint conducive to PyInstaller bundling.  
This comprehensive research report details the architectural pathway for integrating the distilled 600-million parameter variant of the No Language Left Behind machine translation model (NLLB-200-distilled-600M) via the CTranslate2 inference engine. The analysis addresses the precise mechanics of converting models to PyTorch-free formats, maps the linguistic capabilities of NLLB-200 against priority indigenous and low-resource languages, evaluates computational benchmarking on target hardware, and defines the exact data serialization mapping required for the Oral History Metadata Synchronizer (OHMS) XML 6.0 bilingual schema. Furthermore, the analysis deconstructs advanced model compression paradigms, contrasting Mixture-of-Experts (MoE) pruning with dense vocabulary reduction, to establish an optimal deployment strategy for the archival desktop application.

## **CTranslate2 and NLLB-200 Integration Architecture**

The foundational requirement of this integration is the absolute exclusion of PyTorch during runtime execution, combined with the necessity of maintaining a minimal initial application bundle size. PyTorch, while being the industry standard for training deep neural networks, relies on massive dynamic computational graph libraries and extensive CUDA/C++ backend dependencies that drastically inflate application binaries, often exceeding several gigabytes. For inference-only applications, this overhead is entirely superfluous. CTranslate2 serves as the ideal inference engine to satisfy these constraints. Because CTranslate2 is already bundled within the target environment to support the Faster-Whisper automatic speech recognition (ASR) module, leveraging it for machine translation incurs zero additional binary footprint for the execution engine itself.1

### **The Hugging Face to CTranslate2 Conversion Pipeline**

To utilize the NLLB-200-distilled-600M model within a purely PyTorch-free desktop environment, the model must undergo an offline, ahead-of-time (AOT) conversion process. The CTranslate2 library provides a dedicated conversion utility, ct2-transformers-converter, designed specifically to bridge the gap between Hugging Face's ecosystem and CTranslate2's highly optimized C++ backend.3 The conversion process extracts the PyTorch checkpoint weights, optimizes the computation graph by fusing operations (such as combining linear layers with activation functions), and quantizes the floating-point weights into a highly optimized format suitable for consumer CPUs.3  
It must be noted that the conversion process itself transitively requires PyTorch and the Hugging Face transformers library (specifically requiring transformers\>=4.21.0 for NLLB compatibility).1 However, this dependency is strictly confined to the developer's build environment. The conversion is a singular, one-time operation executed prior to software distribution. The resulting CTranslate2 model artifacts are completely decoupled from PyTorch and can be distributed to end-users via the application's native download manager without dragging the PyTorch framework into the PyInstaller \--onedir bundle.5  
The precise command sequence required to convert the NLLB-200-distilled-600M model from its native Hugging Face format into a CTranslate2-compatible format, applying 8-bit integer (INT8) quantization, is defined as follows 4:

Bash  
pip install transformers\[torch\] ctranslate2  
ct2-transformers-converter \\  
    \--model facebook/nllb-200-distilled-600M \\  
    \--output\_dir nllb-200-distilled-600M-ct2-int8 \\  
    \--quantization int8 \\  
    \--copy\_files tokenizer.json README.md special\_tokens\_map.json

Execution of this command constructs a target directory containing the quantized weight binary (model.bin), the architectural configuration file (config.json), and the CTranslate2 translation dictionary (shared\_vocabulary.txt), alongside the copied tokenizer configurations required for text processing.7

### **Storage Profiling and Tokenization Mechanics**

Storage footprint represents a critical parameter for software targeting diverse global environments with potentially constrained bandwidth. The base NLLB-200-distilled-600M model stored in standard 32-bit floating-point (FP32) format requires approximately 2.4 gigabytes of disk storage. By applying INT8 quantization during the CTranslate2 conversion, the parameter weight precision is mathematically reduced from four bytes down to one byte per parameter. Consequently, the total disk size of the NLLB-200-distilled-600M model is reduced to approximately 600 megabytes.8 Because the application imposes a hard limit of a sub-400MB initial bundle size, this converted CTranslate2 NLLB model must not be packaged within the base executable. Instead, the application's internal model manager must be programmed to download this 600MB artifact from a remote repository strictly upon the user's first request to initiate a translation task.9  
The translation pipeline heavily relies on robust tokenization to map raw text strings into the integer IDs processed by the neural network. NLLB-200 utilizes a massive, unified multilingual SentencePiece tokenizer capable of supporting 256,204 distinct tokens across 200 languages.10 To support this within the application, the sentencepiece Python package is required. The specific SentencePiece model file (flores200\_sacrebleu\_tokenizer\_spm.model) is highly compact, occupying roughly 4.3 megabytes.8 The sentencepiece library is implemented primarily in C++ with a thin Python wrapper, possessing absolutely zero transitive PyTorch dependencies. It bundles cleanly within PyInstaller and executes efficiently in low-memory environments.8  
A critical implementation detail when migrating from native Hugging Face pipelines to CTranslate2 involves the manual management of special control tokens. CTranslate2 expects the input arrays to explicitly declare the source language and sequence boundaries. For NLLB models, the required token array structure dictates that the source sequence must be prefixed with the source language's BCP-47 code, followed by the text tokens, and terminated with an end-of-sequence token \</s\>. Simultaneously, the target language BCP-47 code must be passed as the target\_prefix argument to the CTranslate2 generation method.6 For example, translating English to French requires prefixing the source array with the eng\_Latn token and prompting the decoder with the fra\_Latn token.5

### **Concurrent Execution and Engine State Management**

The operational workflow dictates that the user first runs ASR via Faster-Whisper and subsequently translates the output via NLLB-200. This raises the question of whether Faster-Whisper's CTranslate2 instance and the NLLB CTranslate2 instance can be loaded simultaneously within the same process without triggering out-of-memory (OOM) exceptions or suffering from state collisions within the underlying C++ backend.  
Architecturally, CTranslate2 is designed as a thread-safe, multi-instance execution engine. Multiple distinct model instances, even those of differing architectures like Whisper and NLLB, can coexist safely within the same Python process. They will not interfere with one another's internal states, provided there is adequate physical RAM to map their respective memory spaces.11  
However, evaluating the 8GB RAM hardware constraint reveals a tight memory margin. A typical desktop operating system (Windows 10/11 or macOS) claims a baseline of approximately 2.5 to 3 gigabytes of RAM. The PyQt6 graphical interface and application logic consume roughly 300 megabytes. Loading the Faster-Whisper large-v3-turbo model in INT8 format demands approximately 1.5 gigabytes of memory. Instantiating the NLLB-200-distilled-600M model in INT8 requires an additional 600 megabytes. During active inference, the tensor activation states, attention matrices, and key-value (KV) caches can consume another 500 to 800 megabytes.4  
While simultaneous loading yields a total system memory footprint around 5.5 to 6 gigabytes—technically fitting within the 8GB limit—it leaves a precariously narrow buffer for operating system background tasks. Prolonged execution could lead to memory fragmentation or page file swapping, devastating application responsiveness.  
Therefore, a sequential resource management strategy is highly recommended. The CTranslate2 Translator and WhisperModel objects expose explicit resource management methods, notably unload\_model() and load\_model().14 Upon the completion of the ASR phase, the application should programmatically invoke the unload method on the Whisper instance, forcibly freeing the 1.5 gigabytes of C++ allocated memory back to the operating system.14 Only then should the NLLB model be instantiated. CTranslate2's model loading latency from a modern solid-state drive is exceptionally low, typically under two seconds, making this transition entirely imperceptible to the end-user.16

## **Translation Quality and Linguistic Capabilities**

The integration of machine translation into archival workflows introduces profound ethical and operational responsibilities. Archivists dealing with indigenous and endangered languages require extreme fidelity, as mistranslations can permanently corrupt the historical record. The NLLB-200 model was evaluated extensively using the FLORES-200 benchmark, establishing baseline expectations across diverse language families through metrics such as chrF++ and spBLEU.17

### **Priority Language Assessment**

The evaluation of NLLB-200-distilled-600M across the specified high-priority language corridors yields a complex spectrum of capabilities. The system identifies languages utilizing specific BCP-47 codes. An analysis of the target languages reveals distinct tiers of reliability.

| Language | BCP-47 Code | FLORES-200 Status | Linguistic Evaluation and Archival Utility |
| :---- | :---- | :---- | :---- |
| **Spanish** | spa\_Latn | Supported 19 | High capability. The model produces fluent, morphologically accurate translations suitable for immediate synchronization with minimal human post-editing. |
| **Mandarin** | zho\_Hans | Supported 19 | High capability. Despite tokenization sampling artifacts noted in some dense models, the 600M variant provides highly reliable translations across diverse semantic domains.20 |
| **Arabic** | arb\_Arab | Supported 21 | High capability for Modern Standard Arabic (MSA). However, localized dialects present in oral histories may experience structural normalization toward MSA.22 |
| **Welsh** | cym\_Latn | Supported 17 | Moderate to High capability. As an Indo-European Celtic language, Welsh benefits from strong cross-lingual transfer learning. With chrF++ scores historically hovering in the high 50s, the translations provide highly coherent structural mapping, though complex mutations may occasionally err.24 |
| **Irish Gaelic** | gle\_Latn | Supported 21 | Moderate capability. Natively supported within the FLORES-200 corpus, Irish performs adequately for archival glossing, though its Verb-Subject-Object (VSO) word order can sometimes challenge the decoder's alignment capabilities on highly colloquial spoken narratives. |
| **te reo Māori** | mri\_Latn | Supported 26 | Moderate capability. As an Austronesian language, Māori is present in the core training data. It yields functional baseline translations, but deep cultural idioms and specific contextual metaphors prevalent in indigenous testimonies often degrade into literalistic representations. |
| **Cherokee** | chr\_Cher | Supported 27 | Experimental to Poor capability. Cherokee utilizes a unique syllabary and possesses a heavily polysynthetic structure. Polysynthetic languages bundle entire sentences into single highly inflected words, confounding standard subword tokenizers. Translations frequently decouple, resulting in disjointed, literal outputs.22 |
| **Pitjantjatjara** | pjt | Partial/Unsupported | Unsupported baseline. The core NLLB-200-600M model does not possess native, reliable translation weights for Pitjantjatjara, though related language identification (LID) subsystems exhibit marginal recognition capabilities.29 |
| **Yolŋu Matha** | N/A | Unsupported | Completely unsupported. The Yolŋu languages of northern Australia are entirely absent from the NLLB-200 and FLORES-200 datasets.29 |
| **Navajo** | N/A | Unsupported | Completely unsupported. Attempting to translate Navajo is functionally impossible within the base model and will result in catastrophic hallucinations.32 |

### **Fallback Alternatives and Calibration Strategies**

Relying solely on the base 600M model for languages where quality is documented as poor or non-existent presents a severe risk of misleading archivists. For languages like Cherokee, Pitjantjatjara, Yolŋu Matha, and Navajo, the application must not operate under the pretense of capability.  
When the base NLLB-200 model fails, alternative architectures must be considered. The Helsinki-NLP group, utilizing the MarianMT architecture (which is fully compatible with CTranslate2 conversion 5), provides hundreds of community-driven models tailored to specific low-resource pairs under the opus-mt designation.34 For highly specific indigenous languages like Pitjantjatjara, independent academic researchers have successfully fine-tuned NLLB-200 checkpoints using specialized corpora, demonstrating dramatic improvements in BLEU scores specifically for endangered language corridors.36 The architectural philosophy of the application must therefore remain modular, permitting the programmatic swapping of Hugging Face repository endpoints to target specialized community fine-tunes when the base model is inadequate.  
To protect the integrity of the archival workflow, the user interface must be aggressively calibrated. The implementation of a visual Quality Confidence Indicator is paramount. Languages exhibiting strong FLORES-200 chrF++ metrics (such as Spanish and Mandarin) should be presented with a "High Confidence" tag. Languages like Welsh and Māori should carry a "Moderate / For Glossing" tag, advising the archivist that human review is essential. Languages such as Cherokee must be flagged with an "Experimental / High Risk" warning. Crucially, languages entirely absent from the NLLB-200 BCP-47 manifest, such as Navajo and Yolŋu Matha, must be strictly excluded from the application's translation drop-down menus. Allowing a user to select an unsupported language triggers fallback mechanisms within the model that generate syntactically plausible but semantically hallucinatory outputs, severely compromising the historical record.

## **Computational Benchmarking and Hardware Performance**

The offline execution of massive transformer architectures requires meticulous hardware optimization, particularly when restricted to mid-range consumer CPUs such as the Intel Core i5-12400 coupled with 8 gigabytes of RAM.  
CTranslate2 is inherently engineered to maximize CPU throughput by leveraging vector instruction sets (such as AVX2 and AVX-512) and highly parallelized intra-operation multithreading routines, typically backed by libraries like OpenMP or Intel MKL.11 When operating the NLLB-200-distilled-600M model under INT8 quantization, the mathematical complexity of the matrix multiplications is significantly reduced, yielding substantially accelerated inference speeds.37

### **Throughput and Expected Translation Duration**

On a modern 6-core processor like the Intel i5-12400, optimizing the thread pool is critical. CTranslate2 provides parameters for intra\_threads (threads used to parallelize a single operation) and inter\_threads (threads used to process independent batches concurrently).11 Setting intra\_threads to map directly to the physical performance cores (e.g., 6\) while restricting inter\_threads to 1 prevents context-switching overhead and maximizes the CPU's L3 cache locality.  
Under these optimized conditions, the NLLB-200-distilled-600M model achieves a throughput of approximately 40 to 60 tokens per second for sequential, batch-processed decoding.13  
To estimate the translation duration for a standard oral history asset, consider a typical one-hour interview. An average conversational speaking rate produces roughly 8,000 words per hour. Due to the subword tokenization methodology of the SentencePiece algorithm, which often splits complex words into multiple sub-units, 8,000 words typically map to approximately 10,000 to 12,000 computational tokens.10  
Applying the measured throughput:  
At 50 tokens per second, translating 10,000 tokens requires 200 seconds, or approximately 3 minutes and 20 seconds.  
Therefore, the application's target of achieving a sub-5-minute translation for a one-hour transcript is entirely realistic and achievable on consumer CPU hardware. However, from a user experience perspective, blocking the interface for three to five minutes is suboptimal. The translation engine should be positioned as an asynchronous background worker task. Because the text is already broken into discrete chronological segments by the preceding Faster-Whisper transcription phase, the application can feed these segments to the translation engine iteratively. The interface can dynamically populate the translated text fields in real-time as each segment completes, providing immediate visual feedback to the archivist while the remaining sequence processes in the background.

### **Optimizing Batch Execution**

To actually achieve the benchmarked 50 tokens per second, the application must not process segments sequentially with a batch size of one. CTranslate2 thrives on vectorization. The translate\_batch() method must be utilized, collecting segments into arrays (e.g., 16 or 32 segments per batch).4 This allows the underlying C++ linear algebra routines to maximize hardware register utilization. Furthermore, the translation speed is heavily influenced by the beam\_size parameter. By default, CTranslate2 may employ a beam search of size 5, tracking five parallel hypotheses per token. For CPU inference, reducing the beam\_size to 2 drastically increases processing speed—often doubling throughput—with only a statistically negligible reduction in translation accuracy, while also restricting temporary memory allocation spikes during generation.6

## **Data Serialization and OHMS XML 6.0 Integration**

The Oral History Metadata Synchronizer (OHMS) represents the institutional standard for synchronizing text to audiovisual oral history media, widely deployed across platforms like Aviary and Omeka. Integrating the desktop application's generated translations seamlessly into institutional repositories mandates precise adherence to the OHMS XML 6.0 schema architecture.39

### **Architectural Evolution and Schema Requirements**

In earlier iterations of the OHMS ecosystem (version 5.4 and below), the synchronization of bilingual transcripts relied on a heavily customized, proprietary XML structure utilizing legacy tags such as \<sync\>, \<sync\_alt\>, \<transcript\>, and \<transcript\_alt\>.40 With the adoption of the OHMS XML 6.0 standard, the data structure was fundamentally modernized to align with contemporary web standards, specifically designed to ingest and parse standard WebVTT (Web Video Text Tracks) payloads directly within the XML nodes.40  
To construct a valid bilingual payload compatible with OHMS 6.0, the primary language transcript must be embedded within the \<vtt\_transcript\> tag, utilizing standard WebVTT formatting (complete with WEBVTT headers, sequence numbers, and strict 00:00:00.000 \--\> 00:00:00.000 timestamps). The translated secondary track must be placed adjacently within the \<vtt\_transcript\_alt\> element.40  
While the actual transcript data resides in these WebVTT nodes, the OHMS Viewer and Aviary ingestion pipelines rely on explicit metadata triggers at the header level to activate the bilingual user interface components. Failure to include these properties will result in the alternate track being silently ignored by the rendering engine. The critical metadata fields are 42:

1. **Include Translation**: This element acts as the primary boolean toggle. It must be present and set to a truthy value (typically 1 or yes) to logically activate the application's capability to render synchronized dual-language tracks.  
2. **Language**: This field designates the primary source language in which the interview was originally recorded and transcribed.  
3. **Language for Translation**: This field explicitly defines the target language corresponding to the data payload held within the \<vtt\_transcript\_alt\> block, enabling the OHMS Viewer to label the UI toggle buttons correctly.

### **Timestamp Synchronization Mechanics**

A critical design consideration is whether the OHMS Viewer expects the \<vtt\_transcript\> and \<vtt\_transcript\_alt\> tracks to possess identical cue timestamps, or if they operate asynchronously.  
Because the proposed architecture inherits the temporal alignment strictly at the segment level—meaning the NLLB-200 translated text forcefully adopts the exact start and end times calculated by Faster-Whisper for the source text—the resulting cue timestamps between the primary and alternate VTT tracks will be mathematically identical. This is a highly advantageous architectural outcome. While modern WebVTT parsers can theoretically handle overlapping or misaligned asynchronous tracks, supplying identically matched cue times guarantees perfect simultaneous rendering within the OHMS Viewer and the Aviary Embed player. It eliminates UI jitter and timeline race conditions during user playback, presenting a perfectly mirrored bilingual reading experience.

### **XML Serialization Implementation**

The \<vtt\_transcript\_alt\> element, while occasionally absent from simplified public XSD documentation, is natively supported by all OHMS Viewer implementations from version 3.10 onward.40 To guarantee ingestion, the XML serialization executed by the desktop application must conform to the following validated structure, ensuring CDATA blocks encapsulate the WebVTT text to prevent parsing errors triggered by special characters within the transcripts:

XML  
\<?xml version="1.0" encoding="UTF-8"?\>  
\<OAI-PMH xmlns\="http://www.openarchives.org/OAI/2.0/" xmlns:xsi\="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation\="http://www.openarchives.org/OAI/2.0/ http://www.openarchives.org/OAI/2.0/OAI-PMH.xsd"\>  
  \<responseDate\>2026-06-04T00:00:00Z\</responseDate\>  
  \<request verb\="GetRecord" identifier\="lore-session-1234" metadataPrefix\="oai\_ohms"\>http://localhost/ohms\</request\>  
  \<GetRecord\>  
    \<record\>  
      \<header\>  
        \<identifier\>lore-session-1234\</identifier\>  
        \<datestamp\>2026-06-04\</datestamp\>  
      \</header\>  
      \<metadata\>  
        \<ohms\>  
          \<record\>  
            \<title\>Interview with Indigenous Elder\</title\>  
            \<language\>es\</language\>  
            \<include\_translation\>1\</include\_translation\>  
            \<language\_translation\>en\</language\_translation\>  
              
            \<vtt\_transcript\>\<\!\]\>\</vtt\_transcript\>

            \<vtt\_transcript\_alt\>\<\!\]\>\</vtt\_transcript\_alt\>  
          \</record\>  
        \</ohms\>  
      \</metadata\>  
    \</record\>  
  \</GetRecord\>  
\</OAI-PMH\>

Implementing this precise structural hierarchy ensures that when an archivist uploads the desktop application's exported files to an institutional repository, both the primary narrative and the translation-toggle interfaces populate instantly and accurately, bridging the local-first processing environment with standard archival distribution platforms.

## **Model Pruning and Advanced Optimization Pathways**

When deploying machine learning architectures to resource-constrained edge environments, structural optimization becomes paramount. The analysis identifies a prominent academic pathway for optimizing the model, specifically the research detailing "Memory-efficient NLLB-200" (arXiv:2212.09811).17 However, implementing optimization within the desktop application requires a rigid conceptual distinction between Mixture-of-Experts (MoE) pruning and dense network vocabulary reduction.

### **Deconstructing the Mixture-of-Experts (MoE) Pruning Misconception**

The paper "Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model" presents a groundbreaking methodology for reducing memory footprint by up to 80% with negligible translation quality loss.17 It demonstrates that specific attention mechanisms learn to specialize in specific language pairs, allowing unneeded computational pathways to be severed.  
However, a critical architectural discrepancy exists. The pruning techniques documented in this research are explicitly engineered for the 54.5-Billion parameter Mixture-of-Experts (MoE) variant of NLLB-200.17 MoE networks fundamentally diverge from standard architectures; their internal transformer layers are partitioned into numerous isolated "experts" controlled by a gating mechanism that routes tokens dynamically.17  
The architecture selected for the desktop application is the NLLB-200-distilled-600M variant. This model is a traditional, dense transformer.17 Every token passes through every parameter in the network. Because the 600M model possesses absolutely no "experts" to prune, the specific implementation algorithms and Python libraries provided by the 2212.09811 research (such as the nllb\_pruning.py routines 50) are physically incompatible with the deployed model and cannot be utilized.

### **The Viable Alternative: Embedding Matrix Vocabulary Reduction**

While expert pruning is inapplicable to dense models, the 600M model is highly susceptible to a different compression vector: Vocabulary and Embedding Pruning.10  
The defining characteristic of massively multilingual models like NLLB is an extraordinarily bloated tokenizer. To support 200 distinct languages across multiple scripts (Latin, Arabic, Cyrillic, Devanagari, etc.), the tokenizer maintains a massive matrix of 256,204 distinct tokens.10 In a dense model with only 600 million parameters, the mathematical embedding layer—which maps these 256K tokens into dense representational vectors—accounts for a massively disproportionate percentage of the entire model's weight.53  
If an archival workflow is strictly defined around a single translation corridor, such as Māori to English, over 90% of the 256,204 tokens represent entirely unrelated linguistic symbols (e.g., Mandarin characters or Arabic script).52 These embeddings are essentially dead weight; they consume precious disk space and load into the system's RAM, yet they are mathematically never activated during the translation of Latin-script languages.52  
By programmatically analyzing the tokenizer to isolate and preserve only the tokens relevant to English, the target language, and a shared set of punctuation and subword symbols, the vocabulary size can be artificially truncated from 256K down to a highly efficient 30,000 to 50,000 tokens.54 If the embedding layer of the raw PyTorch checkpoint is sliced to match this shrunken vocabulary prior to CTranslate2 conversion, the resulting disk footprint of the INT8 model can be reduced from 600 megabytes down to approximately 200 to 250 megabytes. Crucially, the translation accuracy remains entirely identical to the monolithic model, as the active attention heads and feed-forward matrices governing grammatical logic are left completely intact; only the unused dictionary lookup vectors are discarded.10

### **Monolithic vs. Fragmented Deployment Strategy**

The technical viability of vocabulary pruning necessitates a strategic decision: Should the application ship with a single monolithic 600MB model capable of handling all languages, or should it orchestrate a repository of heavily pruned 200MB models tailored for specific language pairs?  
An analysis of the application's local-first operating constraints firmly supports the **Single Monolithic Model Strategy**. The reasoning is multifaceted:

1. **Storage Economy and Scaling Dynamics:** If an institution's archivist requires capabilities for Spanish, Mandarin, and Welsh testimonies, downloading three distinct pruned models (at 200MB each) rapidly consumes 600MB of storage. The storage efficiency of pruning completely collapses the moment a user requires more than two language pairings.  
2. **Offline Resilience and Serendipitous Discovery:** The fundamental value proposition of an offline desktop tool is its absolute resilience to unpredictable inputs in air-gapped environments. Archivists frequently encounter tapes containing unexpected linguistic variations or unidentified languages. A monolithic model guarantees immediate, unbroken operational capability without requiring a return to a networked environment to provision a new localized model.  
3. **Software Architecture Complexity:** Orchestrating a dynamic repository that tracks, downloads, and switches between potentially thousands of distinct language-pair permutations adds profound complexity to the application's ModelManager. Distributing a singular, unified 600MB CTranslate2 artifact drastically simplifies the codebase, reduces deployment friction, and guarantees cross-platform stability.

Vocabulary pruning should be reserved strictly as a contingency protocol, intended for deployment only if future hardware constraints mandate execution on ultra-low-memory edge devices far below the stated 8GB RAM minimum.

## **UI/UX Considerations for Archival Integrity**

Software interfaces designed for archival science must prioritize historical transparency and data integrity over seamless, opaque automation. When dealing with post-conflict narratives or deeply nuanced indigenous oral histories, archivists must be acutely aware of when machine learning subsystems are making interpretive leaps or struggling with acoustic anomalies.

### **Navigating Code-Switching and Multi-lingual Testimonies**

A pervasive phenomenon in oral history interviews is code-switching—the practice where a speaker dynamically alternates between two distinct languages, often transitioning mid-sentence (for example, shifting seamlessly from English into Welsh, or from Spanish into an indigenous dialect).  
This presents a severe architectural challenge for the NLLB-200 pipeline. As established, CTranslate2 requires the decoder to be explicitly prefixed with a singular target language token, and the input sequence to be tagged with a singular source language BCP-47 tag.6 If the source text is forced to declare a cym\_Latn (Welsh) origin, but the textual array contains untranslated English phrases resulting from a code-switch, the model's attention mechanism attempts to force Welsh syntactic heuristics upon the English tokens. This frequently results in grammatical hallucination, where the model outputs nonsensical phonetic approximations or attempts literal, disjointed translations of the foreign words.  
To address this, the user interface must be designed to operate translation at the granular *segment level* rather than treating the transcript as an immutable global block. While the interface should feature a primary "Translate Entire Transcript" macro to populate the baseline alternate track, it must afford the archivist immediate segment-level intervention capabilities. An archivist encountering a code-switched paragraph should be empowered to click the specific segment block in the UI, override the baseline source language tag for that sequence alone, and re-execute the translation localized solely to that timeframe. This segment-driven paradigm ensures the final OHMS XML payload remains accurate despite shifting linguistic contexts.

### **Interfacing with Anomaly Detection Metrics**

The desktop application's existing data pipeline utilizes Whisper's native logprob-based (log probability) anomaly detection to calculate confidence scores for the ASR output. Segments flagged as anomalous—indicating low acoustic confidence, excessive background noise, or high hallucination risk—pose a cascading threat to downstream translation. Machine translation architectures are notoriously brittle when confronted with malformed, misspelled, or phonetically hallucinated source text. Feeding acoustically corrupted data into NLLB-200 predictably yields outputs that are syntactically fluent but factually entirely incorrect—a phenomenon colloquially known as "garbage-in, fluent-garbage-out."  
The UI strategy for managing these anomalies must prioritize archivist visibility:

1. **Uninterrupted Execution:** The system must not automatically skip the translation of low-confidence segments. In archival workflows, a flawed but structurally representative draft is vastly preferable to an opaque, silent omission.  
2. **Visual Demarcation:** The translation engine must inherit the logprob anomaly flags generated during the ASR phase. When the resulting text populates the \<vtt\_transcript\_alt\> output field in the graphical interface, it must be demarcated with an aggressive visual cue—such as a heavily shaded border or a persistent "Unverified Translation: Source Acoustic Anomaly" warning badge.  
3. **Metadata Propagation:** The application should be configured to append a distinct warning tag within the segment notes or the extension fields of the exported OHMS XML. This ensures that when the data is ingested into institutional repositories, subsequent researchers or editors are immediately aware that the baseline text from which the translation was derived was acoustically compromised, preserving the rigorous chain of custody essential to modern archival science.

By intricately balancing advanced C++ execution environments with rigorous user interface paradigms, the application successfully extends its local-first capabilities across global linguistic boundaries, empowering archivists to safeguard the structural and cultural integrity of the world's most vulnerable narratives.

#### **Works cited**

1. Transformers — CTranslate2 4.7.2 documentation \- OpenNMT, accessed June 4, 2026, [https://opennmt.net/CTranslate2/guides/transformers.html](https://opennmt.net/CTranslate2/guides/transformers.html)  
2. I put up Whisper with a frontend on a server to use for free \#548 \- GitHub, accessed June 4, 2026, [https://github.com/openai/whisper/discussions/548](https://github.com/openai/whisper/discussions/548)  
3. Converting Your Fine-Tuned Whisper Model to Faster-Whisper Using CTranslate2 \- Medium, accessed June 4, 2026, [https://medium.com/@balaragavesh/converting-your-fine-tuned-whisper-model-to-faster-whisper-using-ctranslate2-b272063d3204](https://medium.com/@balaragavesh/converting-your-fine-tuned-whisper-model-to-faster-whisper-using-ctranslate2-b272063d3204)  
4. How to Deploy HuggingFace Translation Models on GPU Servers | Speechmatics, accessed June 4, 2026, [https://blog.speechmatics.com/huggingface-translation-triton](https://blog.speechmatics.com/huggingface-translation-triton)  
5. CTranslate2/docs/guides/transformers.md at master \- GitHub, accessed June 4, 2026, [https://github.com/OpenNMT/CTranslate2/blob/master/docs/guides/transformers.md?plain=1](https://github.com/OpenNMT/CTranslate2/blob/master/docs/guides/transformers.md?plain=1)  
6. NLLB-200 with CTranslate2 \- Tutorials \- OpenNMT, accessed June 4, 2026, [https://forum.opennmt.net/t/nllb-200-with-ctranslate2/5090?page=3](https://forum.opennmt.net/t/nllb-200-with-ctranslate2/5090?page=3)  
7. NLLB-200-CTranslate2-Adaptive-MT.ipynb \- GitHub, accessed June 4, 2026, [https://github.com/ymoslem/Adaptive-MT-LLM-Fine-tuning/blob/main/NLLB-200-CTranslate2-Adaptive-MT.ipynb](https://github.com/ymoslem/Adaptive-MT-LLM-Fine-tuning/blob/main/NLLB-200-CTranslate2-Adaptive-MT.ipynb)  
8. NLLB.ipynb \- Colab, accessed June 4, 2026, [https://colab.research.google.com/github/ymoslem/Adaptive-MT-LLM/blob/main/MT/NLLB.ipynb](https://colab.research.google.com/github/ymoslem/Adaptive-MT-LLM/blob/main/MT/NLLB.ipynb)  
9. ellite/anchor-sub-sync: Anchor: A universal, hardware-accelerated CLI tool for subtitle synchronization (Whisper) and context-aware translation (NLLB) \- GitHub, accessed June 4, 2026, [https://github.com/ellite/anchor-sub-sync](https://github.com/ellite/anchor-sub-sync)  
10. How to fine-tune a NLLB-200 model for translating a new language \- David Dale \- Medium, accessed June 4, 2026, [https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865](https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865)  
11. Multithreading and parallelism — CTranslate2 4.7.2 documentation \- OpenNMT, accessed June 4, 2026, [https://opennmt.net/CTranslate2/parallel.html](https://opennmt.net/CTranslate2/parallel.html)  
12. General questions to improve performance \- Support \- OpenNMT, accessed June 4, 2026, [https://forum.opennmt.net/t/general-questions-to-improve-performance/5059](https://forum.opennmt.net/t/general-questions-to-improve-performance/5059)  
13. CTranslate2 \- Fast inference engine for Transformer models \- GitHub, accessed June 4, 2026, [https://github.com/opennmt/ctranslate2](https://github.com/opennmt/ctranslate2)  
14. Memory management — CTranslate2 4.7.2 documentation \- OpenNMT, accessed June 4, 2026, [https://opennmt.net/CTranslate2/memory.html](https://opennmt.net/CTranslate2/memory.html)  
15. ctranslate2.models \- OpenNMT, accessed June 4, 2026, [https://opennmt.net/CTranslate2/python/ctranslate2.models.html](https://opennmt.net/CTranslate2/python/ctranslate2.models.html)  
16. Memory increase · Issue \#1488 · OpenNMT/CTranslate2 \- GitHub, accessed June 4, 2026, [https://github.com/OpenNMT/CTranslate2/issues/1488](https://github.com/OpenNMT/CTranslate2/issues/1488)  
17. Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model \- arXiv, accessed June 4, 2026, [https://arxiv.org/html/2212.09811](https://arxiv.org/html/2212.09811)  
18. facebook/nllb-200-distilled-600M \- Hugging Face, accessed June 4, 2026, [https://huggingface.co/facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)  
19. language-trans app | Clarifai \- The World's AI, accessed June 4, 2026, [https://clarifai.com/helsinki-nlp/language-trans](https://clarifai.com/helsinki-nlp/language-trans)  
20. Joint speech and text machine translation for up to 100 languages \- PMC \- NIH, accessed June 4, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11735396/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11735396/)  
21. Assessing the Impact of Typological Features on Multilingual Machine Translation in the Age of Large Language Models \- ACL Anthology, accessed June 4, 2026, [https://aclanthology.org/2026.eacl-long.109.pdf](https://aclanthology.org/2026.eacl-long.109.pdf)  
22. Workshop on Advancing NLP for Low-Resource Languages (2025) \- ACL Anthology, accessed June 4, 2026, [https://aclanthology.org/events/lowresnlp-2025/](https://aclanthology.org/events/lowresnlp-2025/)  
23. FLORES-200 Language Code \- Kaggle, accessed June 4, 2026, [https://www.kaggle.com/datasets/takamichitoda/flores200-language-code](https://www.kaggle.com/datasets/takamichitoda/flores200-language-code)  
24. arXiv:2305.11761v1 \[cs.CL\] 19 May 2023, accessed June 4, 2026, [https://arxiv.org/pdf/2305.11761](https://arxiv.org/pdf/2305.11761)  
25. README.md · facebook/nllb-200-3.3B at f555c8a068d16f0ed1c3e5c1a02a757051c0cb39 \- Hugging Face, accessed June 4, 2026, [https://huggingface.co/facebook/nllb-200-3.3B/blob/f555c8a068d16f0ed1c3e5c1a02a757051c0cb39/README.md](https://huggingface.co/facebook/nllb-200-3.3B/blob/f555c8a068d16f0ed1c3e5c1a02a757051c0cb39/README.md)  
26. slone/nllb-200-10M-sample · Datasets at Hugging Face, accessed June 4, 2026, [https://huggingface.co/datasets/slone/nllb-200-10M-sample](https://huggingface.co/datasets/slone/nllb-200-10M-sample)  
27. EMMA-500: Enhancing Massively Multilingual Adaptation of Large Language Models \- arXiv, accessed June 4, 2026, [https://arxiv.org/html/2409.17892v1](https://arxiv.org/html/2409.17892v1)  
28. EMMA-500: Enhancing Massively Multilingual Adaptation of Large Language Models \- arXiv, accessed June 4, 2026, [https://arxiv.org/html/2409.17892v3](https://arxiv.org/html/2409.17892v3)  
29. GlotLID: Language Identification for Low-Resource Languages \- arXiv, accessed June 4, 2026, [https://arxiv.org/html/2310.16248v3](https://arxiv.org/html/2310.16248v3)  
30. (PDF) GlotLID: Language Identification for Low-Resource Languages \- ResearchGate, accessed June 4, 2026, [https://www.researchgate.net/publication/375112686\_GlotLID\_Language\_Identification\_for\_Low-Resource\_Languages](https://www.researchgate.net/publication/375112686_GlotLID_Language_Identification_for_Low-Resource_Languages)  
31. TICO-19: the Translation Initiative for COvid-19 \- ResearchGate, accessed June 4, 2026, [https://www.researchgate.net/publication/347235759\_TICO-19\_the\_Translation\_Initiative\_for\_COvid-19](https://www.researchgate.net/publication/347235759_TICO-19_the_Translation_Initiative_for_COvid-19)  
32. Goldfish: Monolingual Language Models for 350 Languages \- arXiv, accessed June 4, 2026, [https://arxiv.org/html/2408.10441v1](https://arxiv.org/html/2408.10441v1)  
33. MADLAD-400: A Multilingual And Document-Level Large Audited Dataset, accessed June 4, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2023/file/d49042a5d49818711c401d34172f9900-Paper-Datasets\_and\_Benchmarks.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/d49042a5d49818711c401d34172f9900-Paper-Datasets_and_Benchmarks.pdf)  
34. Helsinki-NLP/tatoeba\_mt · Datasets at Hugging Face, accessed June 4, 2026, [https://huggingface.co/datasets/Helsinki-NLP/tatoeba\_mt](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt)  
35. \[2108.08556\] Attentive fine-tuning of Transformers for Translation of low-resourced languages @LoResMT 2021 \- arXiv, accessed June 4, 2026, [https://arxiv.org/abs/2108.08556](https://arxiv.org/abs/2108.08556)  
36. A Comprehensive Revitalization Framework for 1000+ Endangered Languages via Broad-Coverage LID and LLM \- OpenReview, accessed June 4, 2026, [https://openreview.net/pdf?id=BhfieZOeo5](https://openreview.net/pdf?id=BhfieZOeo5)  
37. Bhasha-Rupantarika: Algorithm-Hardware Co-design approach for Multilingual Neural Machine Translation †, ‡Both authors contributed equally to this work. This work was supported by the Special Manpower Development Program for Chip to Start-Up (SMDP-C2S), the Ministry of Electronics and Information Technology (MeitY), Government of India \- arXiv, accessed June 4, 2026, [https://arxiv.org/html/2510.10676v1](https://arxiv.org/html/2510.10676v1)  
38. Ideas for better performance · Issue \#1140 · OpenNMT/CTranslate2 \- GitHub, accessed June 4, 2026, [https://github.com/OpenNMT/CTranslate2/issues/1140](https://github.com/OpenNMT/CTranslate2/issues/1140)  
39. OHMS-in' with H. Lee Waters' Movies of Local People \- Bitstreams, accessed June 4, 2026, [https://blogs.library.duke.edu/bitstreams/2016/01/15/ohms-h-lee-waters-movies-local-people/](https://blogs.library.duke.edu/bitstreams/2016/01/15/ohms-h-lee-waters-movies-local-people/)  
40. OHMS, WebVTT, and the Transcript Editor of my Dreams \- Digital Omnium, accessed June 4, 2026, [https://digitalomnium.com/ohms-vtt-and-the-transcript-editor-of-my-dreams/](https://digitalomnium.com/ohms-vtt-and-the-transcript-editor-of-my-dreams/)  
41. uklibraries/ohms-viewer \- GitHub, accessed June 4, 2026, [https://github.com/uklibraries/ohms-viewer](https://github.com/uklibraries/ohms-viewer)  
42. OHMS (Oral History Metadata Synchronizer) User Guide, accessed June 4, 2026, [https://www.oralhistoryonline.org/wp-content/uploads/2020/11/@OHMS\_user\_guide\_master\_v3-8-3.pdf](https://www.oralhistoryonline.org/wp-content/uploads/2020/11/@OHMS_user_guide_master_v3-8-3.pdf)  
43. OHMS (Oral History Metadata Synchronizer) User Guide, accessed June 4, 2026, [https://www.oralhistoryonline.org/wp-content/uploads/2023/09/OHMS\_Aviary\_user\_guide\_Master\_Aviary\_v2.0\_09\_17.pdf](https://www.oralhistoryonline.org/wp-content/uploads/2023/09/OHMS_Aviary_user_guide_Master_Aviary_v2.0_09_17.pdf)  
44. Creating Bi-Lingual Projects in the Oral History Metadata Synchronizer, accessed June 4, 2026, [https://ohla.info/creating-multi-lingual-projects-in-the-oral-history-metadata-synchronizer/](https://ohla.info/creating-multi-lingual-projects-in-the-oral-history-metadata-synchronizer/)  
45. Releases and Versions | OHMS \- Oral History Metadata Synchronizer, accessed June 4, 2026, [https://www.oralhistoryonline.org/help/release/](https://www.oralhistoryonline.org/help/release/)  
46. \[2212.09811\] Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model \- arXiv, accessed June 4, 2026, [https://arxiv.org/abs/2212.09811](https://arxiv.org/abs/2212.09811)  
47. Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model \- ACL Anthology, accessed June 4, 2026, [https://aclanthology.org/2023.acl-long.198/](https://aclanthology.org/2023.acl-long.198/)  
48. Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model \- ACL Anthology, accessed June 4, 2026, [https://aclanthology.org/2023.acl-long.198.pdf](https://aclanthology.org/2023.acl-long.198.pdf)  
49. NLLB \- Hugging Face, accessed June 4, 2026, [https://huggingface.co/docs/transformers/v4.26.1/model\_doc/nllb](https://huggingface.co/docs/transformers/v4.26.1/model_doc/nllb)  
50. GitHub \- naver/nllb-pruning: Library for pruning experts per language pair in NLLB-200, accessed June 4, 2026, [https://github.com/naver/nllb-pruning](https://github.com/naver/nllb-pruning)  
51. nllb-pruning/nllb\_pruning.py at main \- GitHub, accessed June 4, 2026, [https://github.com/naver/nllb-pruning/blob/main/nllb\_pruning.py](https://github.com/naver/nllb-pruning/blob/main/nllb_pruning.py)  
52. AdaptBPE: From General Purpose to Specialized Tokenizers \- ACL Anthology, accessed June 4, 2026, [https://aclanthology.org/2026.eacl-long.119.pdf](https://aclanthology.org/2026.eacl-long.119.pdf)  
53. How to adapt a multilingual T5 model for a single language | by David Dale \- Medium, accessed June 4, 2026, [https://medium.com/data-science/how-to-adapt-a-multilingual-t5-model-for-a-single-language-b9f94f3d9c90](https://medium.com/data-science/how-to-adapt-a-multilingual-t5-model-for-a-single-language-b9f94f3d9c90)  
54. Tokenizer shrinking recipes \- Hugging Face Forums, accessed June 4, 2026, [https://discuss.huggingface.co/t/tokenizer-shrinking-recipes/8564](https://discuss.huggingface.co/t/tokenizer-shrinking-recipes/8564)