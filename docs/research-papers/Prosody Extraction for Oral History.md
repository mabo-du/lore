# **Computational Extraction and Archival Representation of Speech Prosody in Text-Based Transcription Pipelines**

## **The Imperative for Paralinguistic Preservation in Spoken Language Understanding**

The conversion of spoken audio into written text has historically functioned as a highly lossy compression algorithm. Standard Automatic Speech Recognition (ASR) systems capture the lexical content—the discrete words and their syntactic arrangement—but systematically discard the paralinguistic and prosodic phenomena that envelop them. Within digital platforms such as Lore, which currently captures word-level confidence metrics from Whisper via average log probabilities and compression ratios, the absence of prosodic metadata represents a critical limitation. Prosody, encompassing variations in fundamental frequency, amplitude, speaking rate, and vocal quality, serves as a primary vector for semantic and pragmatic meaning. A statement such as "I did not steal this car" carries a fundamentally different intention depending on whether contrastive stress is placed on the pronoun "I" (implying someone else stole it), the verb "steal" (implying the car was borrowed), or the demonstrative "this" (implying a different car was stolen).1  
In humanistic disciplines, particularly oral history, the acoustic realization of a sentence is inextricably linked to its historical and emotional record. The transcript should ideally preserve not just what was said, but precisely how it was said, within the structural constraints of a text-based representation.3 As Spoken Language Understanding (SLU) pipelines increasingly rely on Large Language Models (LLMs) to perform downstream tasks such as automated summarization, semantic extraction, and entity resolution, the absence of prosodic markers in raw transcripts produces significant analytical blind spots.5 Text-bound LLMs routinely fail to perceive sarcasm, hesitation, emotional volatility, or localized emphasis, leading to flat, semantically inaccurate, or factually inconsistent interpretations.7  
This comprehensive report provides an exhaustive analysis of the architectural, computational, and archival methodologies required to bridge the gap between acoustic prosody and text-based data models. It evaluates open-source, CPU-feasible prosody extraction models capable of operating within strict hardware constraints, specifically the requirement to run on a standard CPU with less than 8GB of system RAM, while maintaining inference speeds that do not exceed a 2× real-time factor.9 The analysis further establishes a theoretical framework for determining the optimal level of prosodic abstraction, assessing whether raw acoustic contours, normalized feature vectors, or discrete high-level labels survive the text transcription process most effectively. Finally, the report examines the empirical impact of prosody-aware transcripts on downstream NLP tasks and synthesizes best practices for encoding these features within established oral history archival standards, including the Text Encoding Initiative (TEI) and the Oral History Metadata Synchronizer (OHMS).11

## **The Acoustic Correlates of Prosody and Computational Extraction Mechanisms**

Before evaluating specific software libraries and machine learning models, it is necessary to define the physical characteristics of prosody and understand the deterministic algorithms used to extract them. Prosody is multidimensional, manifesting through simultaneous variations in several acoustic correlates across the temporal span of an utterance. The primary acoustic correlates of prosody include fundamental frequency, intensity, duration, and spectral distribution, each of which contributes to the perception of linguistic stress and emotional affect.13  
The fundamental frequency, commonly denoted as F0 and measured in Hertz (Hz), corresponds to the perceptual quality of pitch. Pitch tracking algorithms attempt to identify the rate of vocal fold vibration during voiced speech segments. Variations in F0 over time create pitch contours, which are instrumental in signaling syntactic boundaries, such as rising intonation for interrogatives, and indicating emotional arousal or psychological stress.15 Traditional pitch detection algorithms operate primarily in the time domain to ensure computational efficiency. Techniques such as the Average Magnitude Difference Function (AMDF) and autocorrelation functions are widely utilized to estimate pitch periods, while more modern implementations like the Yin algorithm provide highly accurate F0 tracking with minimal error rates.15  
Intensity, measured in decibels (dB), corresponds to the perceptual quality of loudness or total acoustic energy. Localized spikes in root-mean-square (RMS) intensity frequently correlate with emphasized or prominent syllables within an utterance.19 The calculation of intensity requires analyzing the amplitude of the speech signal over defined sliding windows, often utilizing filter banks to isolate frequencies critical to human speech perception.21  
Duration refers to the temporal length of linguistic units, encompassing individual phonemes, syllables, words, and inter-word pauses. Articulatory slowing, elongated vowels, and the strategic insertion of pauses are primary indicators of semantic weight, cognitive load, or hesitation.19 The reliable measurement of duration requires highly accurate forced alignment algorithms that can map orthographic text to the acoustic signal at the phoneme or word level, establishing precise start and end boundaries for each linguistic unit.24  
Spectral qualities describe the voice quality and phonation characteristics of the speaker. Features such as Cepstral Peak Prominence (CPP), jitter (frequency perturbation), and shimmer (amplitude perturbation) provide critical cues regarding the physiological condition and emotional state of the speaker.18 A breathy, tense, or creaky voice alters the spectral tilt and the harmonic-to-noise ratio. In clinical applications, these parameters serve as biomarkers for pathologies like Parkinson's disease, but in conversational analysis, they distinguish between states such as relaxed intimacy and high-stress confrontation.25

| Acoustic Feature | Perceptual Correlate | Linguistic and Paralinguistic Function | Computational Extraction Method |
| :---- | :---- | :---- | :---- |
| Fundamental Frequency (F0) | Pitch | Intonation, syntactic boundaries, emotional arousal | Autocorrelation, AMDF, Yin Algorithm |
| RMS Amplitude | Intensity / Loudness | Syllabic stress, prominence, aggression | Energy windowing, Filterbanks |
| Temporal Boundaries | Duration / Rate | Hesitation, emphasis, cognitive load | Forced alignment (wav2vec2, pyfoal) |
| Jitter, Shimmer, CPP | Voice Quality | Emotion (tension, breathiness), pathology | Spectral analysis, inverse filtering |

## **Evaluating Abstraction Levels for Text-Based Data Models**

Integrating multidimensional acoustic data into a discrete, text-bound structure, such as Lore's existing Segment and Word data models, requires deliberate downsampling. The chosen abstraction must survive the lossy compression of text without misleading downstream consumers or overwhelming natural language parsers with irrelevant mathematical noise. There are three primary levels of abstraction to consider: raw acoustic contours, normalized feature vectors, and high-level categorical labels.  
The first level of abstraction involves raw acoustic contours, which are time-series vectors representing physical measurements. This approach involves sampling F0, intensity, and spectral features at high frequencies, often every ten milliseconds, resulting in dense numerical arrays for every spoken word.28 While mathematically precise, raw contours are highly problematic for text integration. They massively bloat the data payload, are structurally incompatible with standard JSON or XML text nodes, and are entirely uninterpretable by standard LLMs, which operate on discrete semantic tokens rather than continuous acoustic vectors.30 Furthermore, raw pitch is heavily influenced by a speaker's biological baseline, such as physiological vocal tract length and gender, rendering raw Hertz values useless without cross-speaker calibration.1  
The second level of abstraction involves normalized feature vectors. Instead of retaining raw arrays, this methodology calculates word-level or segment-level summary statistics, such as mean F0, maximum F0, pitch range, and RMS energy. To account for biological variance and recording environment disparities, these values are typically normalized using z-scores relative to the speaker's baseline over the entire audio file.32 Robust scaling techniques often replace the mean and standard deviation with the median and interquartile range to prevent extreme outliers, such as algorithmic pitch-halving errors, from skewing the normalization.32 While arrays of normalized floats are highly robust for statistical modeling and traditional machine learning applications 14, they still impose a high cognitive load on human readers and require complex prompt engineering to be utilized effectively by zero-shot LLMs.  
The third and highest level of abstraction maps acoustic features into discrete linguistic and emotional labels. This approach utilizes acoustic thresholds or neural classifiers to output boolean flags, such as indicating whether a word is emphasized, or categorical strings describing the speaker's affect, such as angry, happy, or possessing a fast tempo.9 For oral history transcripts and LLM-driven downstream tasks, high-level categorical labels represent the optimal abstraction. Humanistic disciplines and natural language processing models both rely fundamentally on semantic interpretation. Providing an LLM with text containing embedded structural markers is computationally actionable and aligns perfectly with established XML transcription standards.11  
However, to preserve the empirical historical record without sacrificing usability, archival systems should ideally adopt a dual-layer approach. The system should retain the normalized summary statistics, specifically the robust z-scores for pitch, intensity, and duration, as hidden metadata attributes attached to the word token object in the JSON or XML schema. Simultaneously, the system should expose only the categorical labels and boolean emphasis flags to the text renderer and the downstream LLM worker. This ensures that the transcript is immediately useful for semantic analysis while maintaining the mathematical provenance required for future computational linguistic research.

## **Architectural Survey of CPU-Feasible Prosody Extraction Models**

The deployment constraints for Lore mandate that the selected extraction tools must operate on a standard CPU architecture, occupy less than 8GB of system RAM, and process audio at a speed not exceeding a 2× real-time factor, meaning a 60-minute file must be processed in under 120 minutes. Furthermore, there is a strong architectural preference for deployable models utilizing ONNX Runtime or CTranslate2 over raw PyTorch implementations, as the former provide significant memory optimizations and inference accelerations on edge hardware.37 The following survey evaluates the leading open-source candidates capable of extracting prosodic features, temporal alignments, and emotional metadata under these strict conditions.

### **Parselmouth and Praat Bindings**

Praat is the academic gold standard for phonetic analysis, utilizing highly optimized deterministic digital signal processing algorithms rather than deep neural networks.39 Parselmouth is a Python library that provides a seamless, Pythonic interface directly to Praat's core C and C++ source code, allowing developers to execute Praat's exact algorithms without relying on its graphical interface or its proprietary, idiosyncratic scripting language.28  
Parselmouth is capable of calculating F0 using advanced autocorrelation algorithms, extracting intensity profiles, and measuring formants, jitter, shimmer, and harmonic-to-noise ratios with extreme precision.33 It operates directly on raw audio waveforms, generating time-series data that can be easily aggregated into word-level metrics using Python array operations.41 Because it relies on deterministic signal processing rather than loading billions of neural network parameters, Parselmouth's memory footprint is negligible. It requires only enough system RAM to hold the audio array in memory, which is typically under 100MB for a full hour of uncompressed audio.28  
Furthermore, the inference speed is exceptionally fast. Algorithms like autocorrelation are computationally inexpensive on modern CPUs, allowing Parselmouth to process audio at speeds that are orders of magnitude faster than real-time.28 However, Parselmouth lacks inherent semantic awareness. It cannot output "emphasis" natively; it only outputs the physical properties of pitch and intensity. To integrate Parselmouth into Lore, developers must write heuristic algorithms that normalize the extracted contours and identify statistically significant deviations, such as flagging a word for emphasis if its median pitch exceeds 1.5 standard deviations above the speaker's global mean.16

### **WhisperX and Phonetic Forced Alignment**

Standard Whisper models, while highly accurate at transcribing lexical content, possess notoriously poor temporal resolution, outputting utterance-level timestamps that can drift by several seconds. Precise temporal boundaries are an absolute prerequisite for any word-level prosodic analysis. Without exact word start and end times, it is mathematically impossible to map a pitch contour generated by Parselmouth to a specific word in the transcript, nor is it possible to calculate word duration and localized speech rate.19  
WhisperX resolves this deficiency by performing forced alignment using a phoneme-based ASR model, most commonly a lightweight wav2vec2.0 variant.10 By aligning the orthographic transcription to the acoustic signal, WhisperX generates highly accurate, word-level bounding boxes.24 The primary challenge with Whisper-based pipelines is memory consumption. However, WhisperX utilizes the faster-whisper backend driven by CTranslate2. By applying INT8 quantization, the memory footprint of the Whisper Large-v2 model is reduced to under 4GB, allowing it to comfortably fit within the 8GB RAM constraint.10  
Regarding inference speed, WhisperX utilizes Voice Activity Detection (VAD) via Silero or Pyannote to chunk the audio prior to inference, allowing the system to entirely bypass the processing of silent segments.24 On a standard CPU, processing times heavily depend on the core count, but optimized INT8 models operating via CTranslate2 generally perform at or slightly below a 1× real-time factor, fully satisfying the speed constraints while providing the foundational temporal architecture required for subsequent prosody extraction.10

### **The Emphases Library**

The emphases library, developed by the Interactive Audio Lab, is a dedicated Python package designed specifically for automatic speech prominence estimation.35 Unlike general-purpose acoustic toolkits, it is explicitly built to output binary word-level emphasis labels. The library accepts an audio file and its corresponding text file, performs forced alignment using the Penn Phonetic Forced Aligner (pyfoal) or accepts pre-computed TextGrids, and utilizes a pre-trained neural model to detect emphasized words.35  
The output is a highly structured list of tuples containing the word string, precise start and end times, a continuous float-valued prominence score, and a boolean flag indicating whether the word is emphasized.35 Because the underlying neural model is narrow in scope and optimized for prominence detection rather than full vocabulary generation, it is lightweight and executes well within standard CPU memory limits.46 The inference speed is highly efficient on CPUs, easily meeting the sub-2x real-time constraint. While it provides high accuracy for prominence detection, its utility is limited strictly to emphasis; it does not capture broader affective states, voice quality anomalies, or acoustic events.47

### **SpeechBrain and Speech Emotion Recognition**

SpeechBrain is a comprehensive, open-source conversational AI toolkit built entirely on PyTorch.48 It encompasses a vast array of speech processing tasks, including speech recognition, speaker verification, speech enhancement, and emotion recognition.49 Within the context of prosody, SpeechBrain offers a vocal\_features module that mirrors much of Praat's functionality, capable of calculating jitter, shimmer, and complex glottal parameters directly within a deep learning pipeline.21  
More importantly for abstraction purposes, the HuggingFace model hub hosts numerous pre-trained SpeechBrain models for Speech Emotion Recognition (SER), often utilizing fine-tuned wav2vec2 architectures.51 A standard SpeechBrain wav2vec2-base model requires approximately 1.2GB to 2GB of system RAM, fitting well within the hardware constraints.52 However, PyTorch CPU inference can be sluggish when processing heavy transformer models.54 Processing a multi-hour file with a raw PyTorch wav2vec2 model on a CPU may approach or exceed the 2× real-time limit depending on the hardware generation. To mitigate this, SpeechBrain supports dynamic batching and provides detailed methodologies for model quantization, allowing developers to convert FP32 precision weights to INT8, significantly accelerating CPU inference times and reducing latency.55

### **SenseVoice-Small via ONNX Runtime**

Developed by FunAudioLLM, SenseVoice-Small represents a paradigm shift from traditional cascaded pipelines. It is an exceptionally powerful, multi-task speech foundation model that simultaneously executes Automatic Speech Recognition (ASR), Spoken Language Identification (LID), Speech Emotion Recognition (SER), and Audio Event Detection (AED) within a single forward pass.34  
Rather than relying on external classifiers to append metadata after transcription, SenseVoice generates "rich transcriptions." It identifies the lexical content while simultaneously embedding inline tags representing the speaker's emotional state—such as \<|HAPPY|\>, \<|SAD|\>, \<|ANGRY|\>, or \<|NEUTRAL|\>—and non-verbal acoustic events—such as \<|Laughter|\>, \<|Applause|\>, or \<|Cough|\>—directly into the output text stream.9 The structural representation of these tags is highly standardized, utilizing a JSON Lines format during processing to map explicit emotion targets and event targets to the audio source.34  
The performance profile of SenseVoice-Small is perfectly aligned with Lore's constraints. The model is highly optimized for edge deployment, providing official export features for ONNX Runtime (funasr-onnx).34 When deployed via ONNX on a CPU, the model consumes approximately 1.1GB of system RAM, leaving ample overhead for concurrent background processes.58 Furthermore, because the model utilizes a non-autoregressive end-to-end framework, its inference speed is extraordinarily fast. Benchmarks indicate that SenseVoice-Small processes 10 seconds of audio in approximately 70 milliseconds.34 Extrapolated to a 60-minute oral history file, the inference time is roughly 25 seconds, meaning the model operates at nearly 140× faster than real-time, completely obliterating the 2× real-time constraint.9 It eliminates the computational overhead of running separate, discrete classifiers for transcription, alignment, and emotion detection.

| Model / Library | Primary Architectural Function | Peak CPU RAM | CPU Inference Speed | Output Abstraction | Deployment Framework |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Parselmouth** | Raw acoustic feature extraction (F0, Intensity) | \< 100MB | \> 50x Real-time | Raw Contours / Vectors | Python Package |
| **WhisperX** | Precise word-level timestamps (Forced Alignment) | \~4GB (INT8) | \~1x Real-time | Temporal Boundaries | CTranslate2 / Python |
| **Emphases** | Word-level prominence detection | \< 1GB | \> 10x Real-time | Boolean / Float Scores | Python Package |
| **SpeechBrain** | Emotion classification / Vocal quality features | \~2GB | \~0.5x \- 1x Real-time | Categorical Labels | PyTorch / ONNX |
| **SenseVoice-Small** | Multi-task ASR, Emotion & Event categorization | \~1.1GB | \~140x Real-time | Inline Categorical Tags | ONNX Runtime |

## **The Empirical Impact of Prosody on Downstream NLP Tasks**

A core research question concerns whether preserving prosodic features mathematically justifies the engineering overhead required to extract them. The empirical evidence strongly indicates that passing prosody-aware transcripts to downstream LLMs significantly improves outcomes in natural language processing, semantic reasoning, and automated summarization.  
Current Spoken Language Understanding pipelines rely almost exclusively on cascaded architectures: an ASR model produces a flat transcript, which is subsequently fed into an LLM for summarization, entity extraction, or sentiment analysis.5 This paradigm suffers from severe, documented limitations. Research analyzing human-written summaries versus machine-generated summaries demonstrates that ASR transcription errors propagate rapidly, and the total loss of paralinguistic data—such as speaker emphasis, prosody, and emotion—measurably degrades the informativeness and coherence of the final output.5  
A flat transcript strips the contextual markers required to resolve semantic ambiguity. In the StressTest benchmark—a comprehensive suite specifically designed to evaluate an LLM's ability to distinguish spoken sentence meanings based on stress patterns—leading language models consistently fail when evaluated on raw text alone.7 The benchmark evaluates models on two tasks: Sentence Stress Detection (SSD) and Sentence Stress Reasoning (SSR). Without prosody, LLMs achieve near-random performance on SSR tasks.6 For example, the sentence "She's really driving him to the sci-fi convention?" implies surprise at the action of driving if the stress is placed on "really," whereas placing the stress on "sci-fi convention" implies surprise at the unexpected destination.2 Standard LLMs operating on flat text cannot distinguish between these two contrastive intents because the acoustic cues necessary to infer the speaker's true intention have been entirely obliterated by the ASR system.6  
Published evidence confirms that integrating prosodic awareness into the text payload fundamentally enhances NLP performance. Studies have demonstrated that end-to-end models like WhisperPro, which simultaneously generate text tokens alongside prosody embeddings representing speaking style and emotion, provide downstream LLMs with a richer, more compatible view of the audio.60 When an LLMWorker is tasked with summarizing an hour-long oral history interview, it typically divides the text into manageable chunks based on token limits. Without explicit prosodic markers, the LLM's self-attention mechanism treats all words with relatively equal semantic weight, occasionally missing the core thesis of a paragraph.30  
However, if the data payload passed to the LLM explicitly includes boolean emphasis markers or affective categorical labels, the LLM's attention mechanism can leverage these engineered features. Experimental models trained to leverage this paralinguistic information produce summaries that are measurably more accurate, contextually faithful, and factually consistent compared to text-only models.31 Furthermore, fine-tuned models like StresSLM demonstrate that when an LLM is trained to ingest explicitly stressed word indicators, its ability to reason about contrastive intention and new information focus drastically improves, narrowing the performance gap between text-based LLMs and human listeners.2  
Therefore, injecting boolean emphasis markers (derived from Parselmouth intensity heuristics) and utterance-level emotion tags (derived from SenseVoice) directly into the prompt context provided to the Lore LLMWorker will measurably elevate the nuance and fidelity of automated archival summarizations.

## **Digital Archival Precedents: Encoding Prosody in OHMS and TEI**

For disciplines such as oral history, the transcript serves as the permanent historical record. If prosodic information is computationally extracted by machine learning models, it must be stored using standardized, interoperable archival formats. While proprietary JSON structures are highly efficient for internal API routing and LLM processing, they fail to meet the long-term preservation standards required by digital humanities institutions. Two primary frameworks dictate the encoding of spoken language in modern archival contexts: the Text Encoding Initiative (TEI) and the Oral History Metadata Synchronizer (OHMS).

### **The Text Encoding Initiative (TEI) Guidelines**

The TEI is a scholarly XML standard utilized globally for encoding digital texts. It encompasses a highly mature, specialized module specifically designed for "Transcriptions of Speech".4 The TEI schema operates on a layered approach, enforcing a strict hierarchical structure that excels at representing complex paralinguistic and prosodic events without disrupting the lexical flow of the transcript.64 The Dartmouth Digital History Initiative (DDHI), for instance, utilizes a layered TEI approach to encode utterances and named entities, establishing a robust framework for digital oral history.64  
Within TEI, the base unit of spoken text is the \<u\> (utterance) element, which contains a stretch of speech and is routinely bound to a specific speaker via a @who attribute, ensuring proper diarization tracking.11 To map machine-extracted prosodic metadata to TEI XML, the standard provides distinct mechanisms for event-based phenomena and contour-based prosodic shifts:  
First, non-verbal events are handled via the \<vocal\>, \<kinesic\>, and \<incident\> elements. SenseVoice's inline audio event tags, such as \<|Laughter|\> or \<|Cough|\>, map seamlessly to TEI's event elements.9 A laugh detected mid-utterance is transcribed as \<vocal desc="laugh"/\>, while extralinguistic background noise (e.g., \<|BGM|\>) maps to the \<incident\> element, preserving the integrity of the spoken transcript while isolating non-lexical phenomena.11  
Second, and most critically for prosody, variations in pitch, intensity, and speaking rate are managed via the \<shift\> element.11 Rather than wrapping an emphasized word in arbitrary bolding or italics, \<shift\> uses specific attributes to define the acoustic change computationally. The @feature attribute designates the prosodic dimension being altered, accepting values such as tempo, loud, pitch, tension, rhythm, or voice.36 The @new attribute specifies the state condition of that feature.36  
For example, if the Parselmouth-derived heuristic algorithm detects a significant spike in RMS intensity indicating word-level emphasis, it is encoded using empty shift tags to denote the start and end of the phenomenon: I \<shift feature="loud" new="strong"/\> really \<shift feature="loud" new="normal"/\> trusted them..36 Similarly, if the SenseVoice model detects a segment-level emotional shift to anger, the \<shift\> element can denote a change in voice quality or tension at the beginning of the utterance.36 This standardization ensures that machine-extracted prosody remains mathematically parseable for future researchers while remaining accessible to qualitative historians rendering the XML into readable HTML interfaces.63

| Machine Extracted Feature | Output Abstraction | Corresponding TEI P5 Element | Attribute Structure |
| :---- | :---- | :---- | :---- |
| SenseVoice \<|Laughter|\> | Audio Event Tag | \<vocal\> | \<vocal desc="laugh"/\> |
| Parselmouth Intensity Spike | Boolean Emphasis | \<shift\> | \<shift feature="loud" new="strong"/\> |
| SenseVoice \<|ANGRY|\> | Segment Emotion Label | \<shift\> | \<shift feature="voice" new="angry"/\> |
| SenseVoice \<|BGM|\> | Background Noise Event | \<incident\> | \<incident desc="bgm"/\> |

### **OHMS 6.0 and Aviary Interoperability**

The Oral History Metadata Synchronizer (OHMS) is a web-based system widely adopted by institutions to synchronize text transcripts with corresponding audio and video files.3 Historically, OHMS relied on a segment-based synchronization mechanism, automatically embedding time codes into the transcript text at specific intervals, such as minute-by-minute, to facilitate user navigation.3  
However, with the evolution to OHMS 6.0 and its deep integration with the Aviary platform, the schema now supports highly granular, full timecode transcripts, parsing modern formats like WebVTT and highly structured OHMS XML.12 The OHMS XML schema is flexible, supporting detailed item-level metadata, indexed thematic segments, and granular footnote annotations directly within the transcript nodes.69  
While OHMS does not natively define complex linguistic prosodic markers in the strict manner of TEI's \<shift\> element, it readily accepts HTML-like structural markup within its transcript nodes, making it highly extensible.71 The optimal archival strategy for Lore is to generate an interoperable synthesis: extracting high-level prosody using TEI nomenclature and embedding those TEI-compliant tags directly within the text payload of the OHMS XML export.12 This architecture allows the OHMS Viewer to maintain perfect audio-text temporal synchronization while enabling advanced researchers to parse the underlying text nodes for embedded paralinguistic and emotional metadata.12

## **Proposed Architectural Integration for Lore**

Integrating a comprehensive prosody extraction pipeline into Lore requires careful orchestration of the selected models. Because acoustic features like pitch and intensity are fundamentally dependent on the temporal boundaries of individual words, prosody extraction must occur strictly *after* the initial transcription and forced alignment phase, but *before* the data is sent to the LLM worker for summarization.

### **Pipeline Topology: The Standalone Prosody Worker**

The architecture should follow a sequential, microservice-inspired pipeline, utilizing a standalone worker for prosody to ensure modularity and prevent memory bottlenecks during transcription.  
First, during the Transcription and Diarization Phase, a high-speed engine, preferably WhisperX utilizing CTranslate2 and INT8 quantization, processes the audio. This phase outputs the lexical words, speaker diarization labels, and exact start/end timestamps.24 Alternatively, SenseVoice-Small via ONNX can be utilized here to generate the base transcript alongside its embedded emotional tags.9  
Second, the Standalone Prosody Worker ingests the original raw audio file alongside the temporal bounding array generated in the first phase. Utilizing the Parselmouth library, the worker calculates the mean F0 and intensity for the entire file to establish the biological baseline. It then slices the audio based on the WhisperX word boundaries and computes the normalized z-score for each discrete word, relying on robust scaling (median and interquartile range) to prevent outliers from distorting the calculations.32  
Third, during the Classification and Abstraction Phase, the worker evaluates the normalized statistics against pre-defined computational heuristics. For example, if a word's intensity z-score is greater than 1.5, or its pitch z-score is greater than 2.0, the worker appends a boolean emphasis flag to the word object.14 Concurrently, utterance-level emotion tags generated by SenseVoice are mapped to the broader segment object.34  
Finally, during the LLM Synthesis Phase, the enriched JSON payload is stringified and formatted. Boolean emphasis flags are transformed into explicit typographical markers (e.g., transforming emphasis: true into an uppercase or asterisk-wrapped word within the prompt), and the entire enriched text is passed to the LLMWorker to generate contextually nuanced, emotion-aware summaries.2

### **Expansion of the Segment Data Model**

To prevent schema bloat and maintain compatibility with existing downstream consumers, the Lore data model should retain the raw, normalized machine statistics within a nested object, while exposing the abstracted, LLM-ready labels at the top level of the Word and Segment hierarchies. The following JSON schema demonstrates how prosody data seamlessly attaches to an existing segment with minimal structural disruption:

JSON  
{  
  "segment\_id": "seg\_0042",  
  "speaker": "SPEAKER\_01",  
  "start": 14.250,  
  "end": 17.800,  
  "text": "I really trusted them.",  
  "emotion": "angry",  
  "events":,  
  "words": \[  
    {  
      "word": "I",  
      "start": 14.250,  
      "end": 14.500,  
      "emphasis": false,  
      "acoustic\_features": {  
        "pitch\_z": \-0.2,  
        "intensity\_z": 0.1  
      }  
    },  
    {  
      "word": "really",  
      "start": 14.500,  
      "end": 15.200,  
      "emphasis": true,  
      "acoustic\_features": {  
        "pitch\_z": 2.4,  
        "intensity\_z": 1.9,  
        "duration\_z": 1.5  
      }  
    },  
    {  
      "word": "trusted",  
      "start": 15.200,  
      "end": 16.000,  
      "emphasis": false,  
      "acoustic\_features": {  
        "pitch\_z": 0.3,  
        "intensity\_z": 0.5  
      }  
    },  
    {  
      "word": "them.",  
      "start": 16.000,  
      "end": 16.500,  
      "emphasis": false,  
      "acoustic\_features": {  
        "pitch\_z": \-0.8,  
        "intensity\_z": \-0.4  
      }  
    }  
  \]  
}

When this JSON payload is subsequently formatted for an archival XML export conforming to OHMS and TEI guidelines, the transformation mechanism seamlessly converts the boolean emphasis flag and the segment emotion label into the appropriate \<shift\> elements, ensuring long-term preservation of the paralinguistic data:

XML  
\<u who\="\#SPEAKER\_01" start\="14.250" end\="17.800"\>  
  \<shift feature\="voice" new\="angry"/\>  
  I   
  \<shift feature\="loud" new\="strong"/\>  
  really  
  \<shift feature\="loud" new\="normal"/\>  
  trusted them.  
  \<shift feature\="voice" new\="normal"/\>  
\</u\>

The systematic extraction of prosodic features from spoken audio represents a critical evolution in digital transcription and archival science. By adhering to a dual-layered abstraction strategy—retaining normalized acoustic statistics for backend analysis while exposing categorical labels and boolean emphasis flags to downstream LLMs—the pipeline ensures maximum computational compatibility. Furthermore, by translating these programmatic variables into TEI-compliant XML structures integrated within an OHMS 6.0 framework, archives can guarantee that the rich, multidimensional nature of human speech is permanently preserved within the digital historical record.

#### **Works cited**

1. Using Deepfake Technologies for Word Emphasis Detection \- ACL Anthology, accessed June 15, 2026, [https://aclanthology.org/2023.paclic-1.72.pdf](https://aclanthology.org/2023.paclic-1.72.pdf)  
2. StressTest: Can YOUR Speech LM Handle the Stress? \- CS.HUJI, accessed June 15, 2026, [https://pages.cs.huji.ac.il/adiyoss-lab/stresstest/](https://pages.cs.huji.ac.il/adiyoss-lab/stresstest/)  
3. OHMS \- Oral History in the Digital Age, accessed June 15, 2026, [https://ohda.matrix.msu.edu/2012/06/ohms-2/](https://ohda.matrix.msu.edu/2012/06/ohms-2/)  
4. A TEI-based Approach to Standardising Spoken Language Transcription, accessed June 15, 2026, [https://journals.openedition.org/jtei/142](https://journals.openedition.org/jtei/142)  
5. Summarizing Speech: A Comprehensive Survey \- ACL Anthology, accessed June 15, 2026, [https://aclanthology.org/2025.emnlp-main.1388.pdf](https://aclanthology.org/2025.emnlp-main.1388.pdf)  
6. StressTest: Can YOUR Speech LM Handle the Stress? \[Quick Review\] \- Liner, accessed June 15, 2026, [https://liner.com/review/stresstest-can-your-speech-lm-handle-stress](https://liner.com/review/stresstest-can-your-speech-lm-handle-stress)  
7. StressTest: Can YOUR Speech LM Handle the Stress? \- arXiv, accessed June 15, 2026, [https://arxiv.org/pdf/2505.22765](https://arxiv.org/pdf/2505.22765)  
8. The official repo of the paper "StressTest: Can YOUR Speech LM Handle the Stress?" \- GitHub, accessed June 15, 2026, [https://github.com/slp-rl/StressTest](https://github.com/slp-rl/StressTest)  
9. Feature: FunASR as STT engine — 17x realtime on CPU, 50+ languages, emotion detection · Issue \#3695 \- GitHub, accessed June 15, 2026, [https://github.com/screenpipe/screenpipe/issues/3695](https://github.com/screenpipe/screenpipe/issues/3695)  
10. Interview transcription using WhisperX model, Part 1\. \- Valor Software, accessed June 15, 2026, [https://valor-software.com/articles/interview-transcription-using-whisperx-model-part-1](https://valor-software.com/articles/interview-transcription-using-whisperx-model-part-1)  
11. 8 Transcriptions of Speech \- The TEI Guidelines \- Text Encoding Initiative, accessed June 15, 2026, [https://tei-c.org/release/doc/tei-p5-doc/de/html/TS.html](https://tei-c.org/release/doc/tei-p5-doc/de/html/TS.html)  
12. OHMS, WebVTT, and the Transcript Editor of my Dreams \- Digital Omnium, accessed June 15, 2026, [https://digitalomnium.com/ohms-vtt-and-the-transcript-editor-of-my-dreams/](https://digitalomnium.com/ohms-vtt-and-the-transcript-editor-of-my-dreams/)  
13. Recognition of Speech With Dynamic Pitch Manipulation in Noise \- PMC \- NIH, accessed June 15, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11000783/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11000783/)  
14. An Acoustic Measure for Word Prominence in Spontaneous Speech \- PMC \- NIH, accessed June 15, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC2864931/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2864931/)  
15. SPEECH PITCH DETECTION 1\. Introduction \- University of Rochester, accessed June 15, 2026, [https://hajim.rochester.edu/ece/sites/zduan/teaching/ece472/projects/2014/Lio\_Chen\_SpeechPitchDetection.pdf](https://hajim.rochester.edu/ece/sites/zduan/teaching/ece472/projects/2014/Lio_Chen_SpeechPitchDetection.pdf)  
16. PITCH-BASED EMPHASIS DETECTION FOR CHARACTERIZATION OF MEETING RECORDINGS Lyndon S. Kennedy and Daniel P.W. Ellis LabROSA, Dept. \- Electrical Engineering, accessed June 15, 2026, [https://www.ee.columbia.edu/\~dpwe/pubs/asru03-emph.pdf](https://www.ee.columbia.edu/~dpwe/pubs/asru03-emph.pdf)  
17. pitch-based emphasis detection for segmenting speech recordings \- MIT Media Lab, accessed June 15, 2026, [https://www.media.mit.edu/speech/papers/1994/arons\_ICSLP94\_emphasis\_detection.pdf](https://www.media.mit.edu/speech/papers/1994/arons_ICSLP94_emphasis_detection.pdf)  
18. Voice Lab Interface — VoiceLab: Automated Reproducible Acoustic Analysis, accessed June 15, 2026, [https://voice-lab.github.io/VoiceLab/](https://voice-lab.github.io/VoiceLab/)  
19. USING WORD-LEVEL FEATURES FOR PROSODIC PROMINENCE DETECTION IN CONVERSATIONAL SPEECH \- International Phonetic Association, accessed June 15, 2026, [https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS2023/full\_papers/298.pdf](https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS2023/full_papers/298.pdf)  
20. Pitch behavior detection for automatic prominence recognition \- ISCA Archive, accessed June 15, 2026, [https://www.isca-archive.org/speechprosody\_2010/abete10\_speechprosody.pdf](https://www.isca-archive.org/speechprosody_2010/abete10_speechprosody.pdf)  
21. speechbrain.processing.vocal\_features module \- Read the Docs, accessed June 15, 2026, [https://speechbrain.readthedocs.io/en/latest/API/speechbrain.processing.vocal\_features.html](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.processing.vocal_features.html)  
22. speechbrain/speechbrain/lobes/features.py at develop \- GitHub, accessed June 15, 2026, [https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/features.py](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/features.py)  
23. Standing out in context: Prominence in the production and perception of public speech | Laboratory Phonology, accessed June 15, 2026, [https://www.journal-labphon.org/article/id/6417/](https://www.journal-labphon.org/article/id/6417/)  
24. WhisperX: Automatic Speech Recognition with Word-level Timestamps (& Diarization) \- GitHub, accessed June 15, 2026, [https://github.com/m-bain/whisperx](https://github.com/m-bain/whisperx)  
25. A Tutorial on Clinical Speech AI Development: From Data Collection to Model Validation, accessed June 15, 2026, [https://arxiv.org/html/2410.21640v1](https://arxiv.org/html/2410.21640v1)  
26. Multilingual evaluation of interpretable biomarkers to represent language and speech patterns in Parkinson's disease \- PMC, accessed June 15, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10017962/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10017962/)  
27. Utilization of Machine Learning-Based Computer Vision and Voice Analysis to Derive Digital Biomarkers of Cognitive Functioning in Trauma Survivors \- Karger Publishers, accessed June 15, 2026, [https://karger.com/dib/article/5/1/16/100189/Utilization-of-Machine-Learning-Based-Computer](https://karger.com/dib/article/5/1/16/100189/Utilization-of-Machine-Learning-Based-Computer)  
28. SwiftF0: Fast and Accurate Monophonic Pitch Detection \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2508.18440v1](https://arxiv.org/html/2508.18440v1)  
29. Structure in conversation: Evidence for the vocabulary, semantics, and syntax of prosody | PNAS, accessed June 15, 2026, [https://www.pnas.org/doi/10.1073/pnas.2403262122](https://www.pnas.org/doi/10.1073/pnas.2403262122)  
30. An End-to-End Speech Summarization Using Large Language Model \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2407.02005v1](https://arxiv.org/html/2407.02005v1)  
31. Advancing Speech Summarization in Multi-modal LLMs with Reinforcement Learning \- arXiv, accessed June 15, 2026, [https://arxiv.org/pdf/2509.19631?](https://arxiv.org/pdf/2509.19631)  
32. An Experimental Analysis on Multicepstral Projection Representation Strategies for Dysphonia Detection \- PMC, accessed June 15, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10256083/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10256083/)  
33. Parselmouth Documentation, accessed June 15, 2026, [https://parselmouth.readthedocs.io/\_/downloads/en/stable/pdf/](https://parselmouth.readthedocs.io/_/downloads/en/stable/pdf/)  
34. FunAudioLLM/SenseVoice: Multilingual speech ... \- GitHub, accessed June 15, 2026, [https://github.com/FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)  
35. interactiveaudiolab/emphases: Crowdsourced and ... \- GitHub, accessed June 15, 2026, [https://github.com/interactiveaudiolab/emphases](https://github.com/interactiveaudiolab/emphases)  
36. TEI element shift (shift) \- Text Encoding Initiative, accessed June 15, 2026, [https://tei-c.org/release/doc/tei-p5-doc/en/html/ref-shift.html](https://tei-c.org/release/doc/tei-p5-doc/en/html/ref-shift.html)  
37. Look Once to Hear: Target Speech Hearing with Noisy Examples \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2405.06289v3](https://arxiv.org/html/2405.06289v3)  
38. WhisperX \- Nic's notes, accessed June 15, 2026, [https://notes.nicolasdeville.com/python/library-whisperx/](https://notes.nicolasdeville.com/python/library-whisperx/)  
39. Speech Categorization with Prosodic Features and Deep Learning \- Chalmers ODR, accessed June 15, 2026, [https://odr.chalmers.se/bitstreams/589f27e0-94a1-419e-9b6f-617c6daf1165/download](https://odr.chalmers.se/bitstreams/589f27e0-94a1-419e-9b6f-617c6daf1165/download)  
40. Analyzing Vocal Features for Pathology — SpeechBrain 0.5.0 documentation, accessed June 15, 2026, [https://speechbrain.readthedocs.io/en/latest/tutorials/preprocessing/voice-analysis.html](https://speechbrain.readthedocs.io/en/latest/tutorials/preprocessing/voice-analysis.html)  
41. Open-source packages for using speech data in ML \- DrivenData, accessed June 15, 2026, [https://drivendata.co/blog/speech-for-ml](https://drivendata.co/blog/speech-for-ml)  
42. arXiv:2404.10440v1 \[cs.CL\] 16 Apr 2024, accessed June 15, 2026, [https://arxiv.org/pdf/2404.10440](https://arxiv.org/pdf/2404.10440)  
43. Collection and Analysis of Repeated Speech Samples: Methodological Framework and Example Protocol \- PMC, accessed June 15, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12326161/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12326161/)  
44. linto-ai/whisper-timestamped: Multilingual Automatic Speech Recognition with word-level timestamps and confidence \- GitHub, accessed June 15, 2026, [https://github.com/linto-ai/whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)  
45. Silero VAD: pre-trained enterprise-grade Voice Activity Detector \- GitHub, accessed June 15, 2026, [https://github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad)  
46. Nathan Pruyne NathanPruyne \- GitHub, accessed June 15, 2026, [https://github.com/NathanPruyne](https://github.com/NathanPruyne)  
47. crowdsource · GitHub Topics, accessed June 15, 2026, [https://github.com/topics/crowdsource?l=python\&o=asc\&s=updated](https://github.com/topics/crowdsource?l=python&o=asc&s=updated)  
48. GitHub \- speechbrain/speechbrain: A PyTorch-based Speech Toolkit, accessed June 15, 2026, [https://github.com/speechbrain/speechbrain](https://github.com/speechbrain/speechbrain)  
49. SpeechBrain: Open-Source Conversational AI for Everyone, accessed June 15, 2026, [https://speechbrain.github.io/](https://speechbrain.github.io/)  
50. SpeechBrain | OVHcloud Worldwide, accessed June 15, 2026, [https://www.ovhcloud.com/en/case-studies/speechbrain/](https://www.ovhcloud.com/en/case-studies/speechbrain/)  
51. prithivMLmods/Speech-Emotion-Classification-ONNX \- Hugging Face, accessed June 15, 2026, [https://huggingface.co/prithivMLmods/Speech-Emotion-Classification-ONNX](https://huggingface.co/prithivMLmods/Speech-Emotion-Classification-ONNX)  
52. \[2510.07052\] Enhancing Speech Emotion Recognition via Fine-Tuning Pre-Trained Models and Hyper-Parameter Optimisation \- arXiv, accessed June 15, 2026, [https://arxiv.org/abs/2510.07052](https://arxiv.org/abs/2510.07052)  
53. Enhancing Speech Emotion Recognition via Fine-Tuning Pre-Trained Models and Hyper-Parameter Optimisation \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2510.07052v1](https://arxiv.org/html/2510.07052v1)  
54. Language Identification: Building an End-to-End AI Solution using PyTorch, accessed June 15, 2026, [https://pytorch.org/blog/language-identification/](https://pytorch.org/blog/language-identification/)  
55. SpeechBrain Advanced \- Read the Docs, accessed June 15, 2026, [https://speechbrain.readthedocs.io/en/latest/tutorials/advanced.html](https://speechbrain.readthedocs.io/en/latest/tutorials/advanced.html)  
56. FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2407.04051v1](https://arxiv.org/html/2407.04051v1)  
57. FunAudioLLM/SenseVoiceSmall \- Hugging Face, accessed June 15, 2026, [https://huggingface.co/FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)  
58. Fast and accurate speech-to-text on RK3588 with SenseVoice Small : r/RockchipNPU, accessed June 15, 2026, [https://www.reddit.com/r/RockchipNPU/comments/1g3cetq/fast\_and\_accurate\_speechtotext\_on\_rk3588\_with/](https://www.reddit.com/r/RockchipNPU/comments/1g3cetq/fast_and_accurate_speechtotext_on_rk3588_with/)  
59. Advancing speech summarization in Multi-modal LLMs with Reinforcement Learning \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2509.19631v1](https://arxiv.org/html/2509.19631v1)  
60. Minimizing Modality Gap from the Input Side: Your Speech LLM Can Be a Prosody-Aware Text LLM \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2605.05927v2](https://arxiv.org/html/2605.05927v2)  
61. Summarize Podcast Transcripts and Long Texts Better with NLP and AI \- Medium, accessed June 15, 2026, [https://medium.com/data-science/summarize-podcast-transcripts-and-long-texts-better-with-nlp-and-ai-e04c89d3b2cb](https://medium.com/data-science/summarize-podcast-transcripts-and-long-texts-better-with-nlp-and-ai-e04c89d3b2cb)  
62. \[2505.22765\] StressTest: Can YOUR Speech LM Handle the Stress? \- arXiv, accessed June 15, 2026, [https://arxiv.org/abs/2505.22765](https://arxiv.org/abs/2505.22765)  
63. Synchronizing Oral History Text and Speech: A Tools Overview, accessed June 15, 2026, [https://academicworks.cuny.edu/cgi/viewcontent.cgi?article=1043\&context=jj\_pubs](https://academicworks.cuny.edu/cgi/viewcontent.cgi?article=1043&context=jj_pubs)  
64. DDHI Encoding Guidelines | Dartmouth Digital History Initiative, accessed June 15, 2026, [http://ddhi.dartmouth.edu/ddhi-encoding-guidelines](http://ddhi.dartmouth.edu/ddhi-encoding-guidelines)  
65. Challenging Colonial Discourse Through TEI Markup in Maria Callcott's “Letters” \- Women Writers Project, accessed June 15, 2026, [https://wwp.northeastern.edu/blog/tei-markup-maria-callcott-letters/](https://wwp.northeastern.edu/blog/tei-markup-maria-callcott-letters/)  
66. Developing Linguistic Corpora: a Guide to Good Practice, accessed June 15, 2026, [https://llds.ling-phil.ox.ac.uk/guides/dlc/chapter5.htm](https://llds.ling-phil.ox.ac.uk/guides/dlc/chapter5.htm)  
67. 8 Transcriptions of Speech \- The TEI Guidelines, accessed June 15, 2026, [https://www.sfu.ca/\~takeda/tei-guidelines/issue\_2382\_eventName/html/TS.html](https://www.sfu.ca/~takeda/tei-guidelines/issue_2382_eventName/html/TS.html)  
68. 8 Transcriptions of Speech \- The TEI Guidelines \- Text Encoding Initiative, accessed June 15, 2026, [https://www.tei-c.org/release/doc/tei-p5-doc/it/html/TS.html](https://www.tei-c.org/release/doc/tei-p5-doc/it/html/TS.html)  
69. Connecting Historical and Digital Frontiers: Enhancing Access to the Latah County Oral History Collection Utilizing OHMS (Oral History Metadata Synchronizer) and Isotope \- The Code4Lib Journal, accessed June 15, 2026, [https://journal.code4lib.org/articles/10643](https://journal.code4lib.org/articles/10643)  
70. ohms OHMS in Aviary \- Aviary User Guide, accessed June 15, 2026, [https://coda.aviaryplatform.com/ohms-in-aviary-115](https://coda.aviaryplatform.com/ohms-in-aviary-115)  
71. OHMS (Oral History Metadata Synchronizer) User Guide, accessed June 15, 2026, [https://www.oralhistoryonline.org/wp-content/uploads/2020/11/@OHMS\_user\_guide\_master\_v3-8-3.pdf](https://www.oralhistoryonline.org/wp-content/uploads/2020/11/@OHMS_user_guide_master_v3-8-3.pdf)  
72. 3- Transcript Formatting in OHMS: Prepping before upload (optional), accessed June 15, 2026, [https://ohla.info/transcript-formatting-in-ohms/](https://ohla.info/transcript-formatting-in-ohms/)  
73. PROMINENCE AND INFORMATION STRUCTURE IN PRONUNCIATION TEACHING MATERIALS John M. Levis, Iowa State University Alif O. Silpachai, accessed June 15, 2026, [https://www.iastatedigitalpress.com/psllt/article/15356/galley/13569/view/](https://www.iastatedigitalpress.com/psllt/article/15356/galley/13569/view/)