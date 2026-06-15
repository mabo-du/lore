# **Timing-Enriched Conversational Summarization: Structural Recovery and Inference Optimization in Resource-Constrained Large Language Models**

## **Introduction to the Interactional Dynamics of Dialogue**

The transcription of human conversation has historically prioritized the preservation of lexical and semantic content at the expense of the structural and temporal choreography that governs actual spoken interaction. In human-to-human dialogue, conversation operates on an intricate, highly coordinated system of turn-taking. This system is defined not merely by what is said, but by the precise temporal spacing between utterances.1 Traditional automatic speech recognition (ASR) pipelines inherently strip this temporal choreography from the final output, rendering dynamic, multi-party conversations as flat, sequential text arrays.  
When these flattened, lexically isolated transcripts are fed into Large Language Models (LLMs) for downstream tasks such as abstractive summarization, the models are fundamentally deprived of critical paralinguistic cues that dictate pragmatic meaning.3 Sociolinguistic research, particularly within the framework of Conversation Analysis, establishes that the duration of silences between turns—known as the inter-turn gap—serves as a primary indicator of speaker intent, cognitive load, and the collaborative or adversarial nature of the exchange.5 A silence of 0.2 seconds may indicate a fluent, collaborative turn transition or enthusiastic agreement, whereas a silence of 2.5 seconds may signal hesitation, reluctance, disagreement, or an adversarial shift in the discussion paradigm.1 Without these timing metrics embedded into the context window, an LLM cannot structurally differentiate between a patient, reflective medical interview and a rapid-fire, high-tension cross-examination.7  
The core hypothesis driving current conversational artificial intelligence research is that re-injecting temporal data into text transcripts restores the structural scaffolding necessary for accurate, context-aware abstractive summarization. If the language model cannot perceive *when* events occurred relative to one another, its capacity to reason about *what* transpired is severely impaired.  
This comprehensive report evaluates the integration of timing-enriched conversational data for LLM summarization, specifically tailored to the architectural constraints of small language models (SLMs). It examines published benchmarks across major conversational datasets, evaluates the optimal syntactic representations for temporal enrichment, isolates the single most impactful feature for compute-constrained environments—such as the Qwen 2.5 1.5B architecture executing via GGUF quantization on central processing unit (CPU) hardware—and establishes a robust, offline evaluation framework capable of assessing summarization quality without relying on human annotators or external API dependencies.

## **The Pragmatics of Time in Spoken Discourse**

To understand why temporal enrichment improves algorithmic summarization, it is necessary to first examine the mechanics of human turn-taking and how flat transcripts fail to capture these dynamics. Human conversation relies on unconscious signals indicating when one speaker has finished and another may begin, known as Transition Relevance Places (TRPs).1

### **Sociolinguistic Foundations of Turn-Taking**

In natural conversation, the transition space between a first pair part (e.g., a question) and a preferred second pair part (e.g., an affirmative answer) is typically contiguous, featuring minimal delay. Conversely, the transition space preceding a dispreferred response (e.g., a refusal or disagreement) is commonly characterized by an overlong gap.5 This differential positioning is both empirical and normative. When an ASR system transcribes this exchange, it removes the silence, effectively neutralizing the dispreferred nature of the response. To an LLM reading the flat transcript, the refusal appears immediate and direct, altering the perceived tone of the speaker from hesitant to aggressively blunt.  
Furthermore, conversational pacing fluctuates based on the macroscopic goal of the dialogue. Brainstorming sessions exhibit frequent speaker switches, high rates of terminal overlap, and minimal inter-turn silences. Disciplinary meetings or formal interrogations feature highly structured turn-taking, significant pauses as speakers weigh their responses, and a distinct lack of overlapping speech.9

### **The Modality Gap in Natural Language Processing**

The discrepancy between the rich acoustic reality of spoken dialogue and the sterile textual representation provided to LLMs is referred to in the literature as the "modality gap".11 While large audio-language models (LALMs) attempt to bridge this gap by processing audio waveforms directly end-to-end 13, these architectures require immense computational overhead, placing them far beyond the capabilities of CPU-bound, local environments.  
The intermediate, highly efficient solution is transcript enrichment. By translating acoustic timing features into explicit textual tokens, researchers can leverage the advanced reasoning and instruction-following capabilities of text-based transformer models without the prohibitive hardware requirements of multimodal processing.15 The self-attention mechanisms inherent in transformer architectures are uniquely suited to this task; they natively learn to correlate the presence of a temporal token (e.g., a pause marker) with the semantic shift in the surrounding text, assigning higher attention weights to utterances bounded by significant temporal deviations.17

## **Empirical Benchmarks on Conversation Datasets**

The integration of acoustic and temporal features into text-based summarization paradigms has been the subject of extensive empirical evaluation. Research consistently demonstrates that models operating exclusively on flat text overlook vital discourse boundaries, leading to summaries that suffer from chronological hallucinations, omitted action items, and misattributed emphasis.18  
To address the first research question—whether timing enrichment measurably improves LLM summarization—it is necessary to review the performance benchmarks across the primary conversational datasets utilized in the NLP community: MeetingBank, QMSum, and MediaSum.

### **MeetingBank Benchmarks**

MeetingBank comprises 1,250 transcripts from city council meetings across various municipalities, characterized by highly formal, heavily structured dialogue with significant bureaucratic pauses.20 The dataset contains long-context dependencies, with an average meeting duration of over two hours and transcripts exceeding 28,000 tokens.22  
In formal settings like city council meetings, silences serve a distinct administrative function, often demarcating the transition between agenda items or signifying the period during which a motion is silently reviewed before a vote. When LLMs attempt to summarize MeetingBank transcripts using text alone, they frequently struggle with topic segmentation, hallucinating continuous debates where distinct, separate agenda items were actually discussed.15  
Recent studies incorporating temporal boundaries into such transcripts demonstrated profound improvements. By converting continuous timestamps into a sequence of discrete temporal tokens, researchers found that LLMs could maintain constant chronological precision regardless of audio length.23 When inter-sentence pause durations were infused into the text, the models exhibited a markedly higher capacity to generate accurate tables of contents and segment-level summaries, as the timing gaps acted as explicit indicators of topic boundaries.15

### **QMSum Benchmarks**

QMSum is a specialized dataset designed for query-based multi-domain meeting summarization.25 It aggregates transcripts from diverse sources, including product design meetings (the AMI corpus), academic research discussions (the ICSI corpus), and parliamentary committee debates.4  
The AMI and ICSI subsets are particularly relevant because they capture natural, spontaneous dialogue dynamics, including interruptions, sentence restarts, and collaborative overlapping.3 Historical research on these specific corpora has consistently shown that non-verbal cues, particularly speaker activity, dominance, and turn-taking delays, serve as critical indicators of "salient" utterances.18  
In a benchmark evaluation analyzing multi-turn dialogues within these domains, baseline text models failed to differentiate between trivial backchanneling (e.g., "yeah," "uh-huh") and substantive agreement.10 However, when timing features—specifically the temporal proximity of the responses—were exposed to the summarization models, the systems achieved state-of-the-art performance on downstream benchmarks. The timing data allowed the models to correctly identify overlapping speech as collaborative rather than independent, standalone semantic statements, thereby preventing the summary from being cluttered with irrelevant interpersonal acknowledgments.8

### **MediaSum and Spoken DialogSum Benchmarks**

MediaSum and Spoken DialogSum encompass media interviews, podcast transcripts, and large-scale spoken dialogues.13 These datasets highlight the importance of conversational pacing and emotional expressivity.  
In unstructured interviews, the pacing of the conversation is tightly coupled with the factual consistency of the dialogue. A study specifically analyzing the impact of pause duration on sequential coherence scores found that temporal hesitation strongly co-occurs with semantic shifts.29 Longer pauses were statistically associated with reduced local semantic similarity, meaning that a long pause almost always precedes a change in topic or a pivot in the narrative.29  
When transcript-free models (which inherently process all audio timing) were benchmarked against text-based models, the audio-aware models substantially outperformed the text-only baselines.16 Most critically, a 2026 study titled *Beyond Transcripts: A Renewed Perspective on Audio Chaptering* isolated various acoustic features to determine their individual contributions. The findings were definitive: when augmenting text-based models, *pause duration drove nearly all the performance gains*, far exceeding the contributions of speaker identity features or pitch variations.16

### **Quantitative Synthesis of Timing Enrichment**

The integration of timing features fundamentally shifts the performance ceiling of summarization algorithms across all major datasets. The table below synthesizes the general findings across the literature regarding the transition from flat text to temporally enriched text.

| Dataset Corpus | Domain | Baseline Limitation (Flat Text) | Proven Benefit of Temporal Enrichment | Primary Metric Improvement |
| :---- | :---- | :---- | :---- | :---- |
| **MeetingBank** | Formal Council Meetings | Failure to detect administrative agenda transitions. | Exact identification of topic boundaries via prolonged administrative silences. | Topic boundary precision, Chronological accuracy |
| **QMSum (AMI/ICSI)** | Unscripted Multi-party | Misinterpretation of backchanneling and overlaps. | Clarification of adversarial vs. collaborative turns; accurate salience weighting. | ROUGE-L relative improvement, Extractive recall |
| **MediaSum / DialogSum** | Interviews & Podcasts | Poor narrative segmentation, high verbosity. | Alignment of semantic shifts with temporal hesitations for superior chaptering. | Coherence scores, Contextual faithfulness |

The empirical consensus is unequivocal: timing enrichment measurably and significantly improves the LLM summarization of conversation. By providing the model with the same pacing cues utilized by human listeners, the resulting summaries exhibit superior factual consistency, reduced hallucination, and a much more nuanced understanding of the interactional dynamics.16

## **Evaluating Temporal Representation Strategies**

Establishing that timing data is valuable is only the first step; to successfully operationalize this enrichment, the data must be formatted in a syntax that the LLM can natively parse and weight through its self-attention mechanisms. The chosen representation strategy dictates both the token overhead and the cognitive load placed on the model. The literature suggests several potential prompt structures, each carrying distinct advantages and computational trade-offs.

### **Option 1: Structured Preludes**

**Example Format:**  
"System Context: This conversation consists of 142 turns from 2 speakers. The average inter-turn gap is 0.8s. There are 12 overlapping interruption events."  
**Theoretical Analysis:** Providing a structured prelude at the beginning of the prompt injects macro-level statistical context into the LLM's system prompt.32 It informs the model of the overall pacing and aggression level of the dialogue.  
**Limitations:** While computationally inexpensive, this approach is temporally disconnected from the specific localized events within the transcript. The LLM understands the *overall* pacing but cannot apply this statistical data to determine if a specific statement at minute 14 was a collaborative interjection or an aggressive interruption. Due to the nature of transformer attention span, global context provided at the beginning of a 10,000-token prompt heavily dilutes as the model processes the deep middle of the transcript.33 The LLM fails to anchor the temporal data to the localized lexical data, rendering this approach largely ineffective for granular structural recovery.

### **Option 2: Per-Interval Metrics**

**Example Format:**  
Appending continuous metrics to every speaker utterance timestamp:  
\[00:14:22, gap=0.4s, rate=140wpm, overlap=False\] Speaker A: I agree.  
**Theoretical Analysis:** This strategy provides maximum fidelity, offering the LLM a continuous stream of paralinguistic data. Models processing this level of detail can theoretically map speech rate fluctuations and turn-taking behavior with high precision.35  
**Limitations:** This strategy suffers from catastrophic token bloat. For a small, 1.5B parameter model operating on a CPU with a constrained memory bandwidth, generating embeddings for per-interval metrics across thousands of utterances will rapidly exhaust the effective context window and degrade inference speeds to an unusable state.34 Furthermore, excessive non-semantic tokens can overwhelm the model's self-attention layers. When the ratio of metadata to actual semantic content becomes too high, the model loses the narrative thread of the conversation, focusing disproportionately on the structural syntax rather than the meaning of the words.33

### **Option 3: Overlap Flags**

**Example Format:**  
Injecting tags specifically at collision points:  
Speaker A: I think we should proceed with the \<overlap\> deployment.  
Speaker B: \<overlap\> No, the backend isn't ready.  
**Theoretical Analysis:** Overlaps are excellent sociolinguistic indicators of high-engagement exchanges, whether enthusiastic agreement or hostile interruption.8 Tagging these boundaries explicitly helps the LLM understand instances where the ASR transcript may seem disjointed or grammatically incorrect due to two people speaking simultaneously.  
**Limitations:** While token-efficient, an overlap flag is a binary metric. It indicates that a simultaneous event occurred but fails to quantify the broader pacing of the conversation. It provides no information regarding the 95% of the conversation that does not feature overlapping speech. It cannot indicate hesitation, reluctance, or thoughtful pauses, thereby missing the majority of the structural nuances required for advanced summarization.5

### **Option 4: Inline Gap Durations**

**Example Format:**  
Placing explicit pause duration markers between speaker segments:  
Speaker A: We need to finalize the schedule.  
\<gap=3.2s\>  
Speaker B: I am not sure we are ready.  
**Theoretical Analysis:** The academic literature strongly supports this specific modality. In a foundational study on multi-level transcript segmentation, researchers formatted transcripts such that each sentence or turn included a pause annotation, effectively exposing the timing gaps to the text model. Because of the precise alignment performed during preprocessing, pause intervals were placed directly at sentence boundaries.15  
The self-attention mechanism in the Transformer architecture natively correlates the length of the numeric gap with the preceding and succeeding tokens. By forcing the model to read the temporal gap immediately prior to processing the semantic payload, the model mathematically weights the significance of the pause against the semantic shift.17 The aforementioned study demonstrated that providing explicit inline pause durations resulted in significant improvements over established topic segmentation baselines, validating this format as the optimal bridge between audio realities and text processing.24

### **Synthesis of Representation Strategies**

| Representation Strategy | Token Overhead | Contextual Fidelity | Drawbacks for CPU/SLM Architecture |
| :---- | :---- | :---- | :---- |
| **Structured Prelude** | Extremely Low (\< 50 tokens) | Macro only; no local anchoring. | Fails to resolve specific conversational ambiguities. |
| **Per-Interval Metrics** | Extremely High (20+ tokens per turn) | Perfect temporal fidelity. | Causes severe CPU bottlenecking and attention dilution. |
| **Overlap Flags** | Very Low (2 tokens per overlap) | Captures simultaneous speech only. | Misses all hesitation, pacing, and deliberation cues. |
| **Inline Gap Durations** | Low (5 tokens per turn transition) | Captures pacing, hesitation, and implicit overlaps. | Requires minor upstream preprocessing of ASR data. |

## **Minimum Viable Enrichment for Resource-Constrained Architectures**

The operational constraints of the target environment fundamentally shape the solution. Deploying a 1.5B parameter GGUF model (specifically the Qwen 2.5 architecture) locally on a CPU imposes strict computational boundaries regarding memory bandwidth, thermal throttling, and time-to-first-token (TTFT) metrics.37  
The Qwen 2.5 1.5B model features a robust 32,768-token theoretical context window.34 However, utilizing the entirety of this window on a standard CPU architecture results in severe degradation. Empirical benchmarks of the Qwen 1.5B model on general-purpose hardware reveal that as the context approaches 16,000 tokens, generation speeds slow down dramatically, and the model becomes highly susceptible to "lost-in-the-middle" phenomena where it forgets instructions placed at the beginning of the prompt.34 Furthermore, utilizing 4-bit (Q4\_K\_M) or 8-bit (Q8\_0) integer quantization to fit the model into CPU RAM introduces minor perplexity penalties, requiring the input prompt to be as clean and deterministic as possible.40  
Therefore, throughput on edge devices or standard CPUs dictates that any transcript enrichment must be highly economical. The "Minimum Viable Enrichment" (MVE) must deliver the maximum contextual signal per token of overhead introduced, avoiding the saturation of the model's working memory.

### **The Optimal Feature: Inter-Turn Gap Durations**

Based on empirical benchmarks 16, the single most impactful timing feature to add to a flat transcript is the **inter-turn gap duration**, represented as a localized, inline token.  
Instead of tracking every micro-pause within a single speaker's continuous utterance (which bloats the token count), the pipeline should calculate and inject only the silence duration that elapses between speaker switches. This minimizes token overhead while capturing the macro-rhythm and interactional dynamics of the dialogue.  
**Token Economy Calculation:** Adding a marker such as \<gap=X.Xs\> at turn boundaries introduces merely 5 to 6 tokens per speaker switch, depending on the tokenizer's specific byte-pair encoding (BPE) for special characters. In a typical one-hour meeting containing 400 turn transitions, this equates to roughly 2,000 to 2,400 additional tokens. This overhead is well within the operational limits of Qwen 2.5 1.5B, leaving ample context window for the transcript itself and the system prompt without triggering severe latency degradation.34  
**Semantic Yield Calculation:**  
Gap lengths explicitly and efficiently differentiate the interaction modality.

* **Collaborative/Urgent:** A sequence of \<gap=0.1s\> markers or negative values (clamped to \<gap=0.0s\> to denote overlaps) indicates a highly collaborative, fast-paced discussion or a heated argument.9  
* **Formal/Structured:** A sequence of \<gap=0.8s\> markers indicates standard, formal turn-taking typical of administrative meetings or controlled interviews.  
* **Hesitant/Adversarial:** A sequence of \<gap=3.5s\> markers indicates careful deliberation, hesitation, cognitive load, or tension.1

By utilizing inter-turn gap durations as the singular MVE, the system achieves the optimal balance: it captures the critical paralinguistic cues proven to drive summarization improvements 30 while strictly adhering to the memory and throughput limitations of a 1.5B quantized CPU deployment.40

### **Implementation Syntax and Prompt Architecture**

To leverage this enrichment, the LLM must be explicitly instructed on how to interpret the markers. Small language models (under 3 billion parameters) possess strong instruction-following capabilities but require highly deterministic, unambiguous prompt structures to avoid hallucinating or ignoring formatting constraints.40 The prompt must establish a clear analytical rubric for the temporal markers.

#### **Optimal System Prompt Template**

You are an expert dialogue analyst and executive summarizer.  
You will be provided with a conversation transcript. To assist your understanding of the conversation's tone and structure, the transcript contains explicit timing markers formatted as \<gap=X.Xs\>. These markers indicate the duration of silence (in seconds) between spoken turns.  
Your task is to generate a comprehensive, professional summary of the conversation. You must use the semantic content AND the \<gap\> markers to accurately interpret the structural dynamics of the exchange.  
When analyzing the conversation, apply the following temporal rules to understand the context:

1. Short gaps (\< 0.5s) or 0.0s gaps indicate fast-paced, collaborative, or highly urgent exchanges. They may also signify interruptions or overlapping speech.  
2. Moderate gaps (0.5s \- 1.5s) indicate standard, formal turn-taking.  
3. Long gaps (\> 1.5s) indicate hesitation, careful deliberation, reluctance, or major shifts in topic.

Ensure your summary captures the factual content, the key decisions made, and the underlying tone of the interaction (e.g., collaborative, adversarial, formal, or hesitant) as evidenced by the conversational pacing.  
CRITICAL CONSTRAINT: Do not mention the gap markers explicitly in your generated summary (e.g., do not write "After a 3-second gap"). Use them solely as internal context to inform your analysis of the conversation's tone and progression.  
Transcript:  
{enriched\_transcript}

#### **Transcript Data Formatting**

The upstream pipeline (Lore's LLMWorker pre-processor) must format the transcript as follows before injecting it into the prompt variable:  
Speaker A: We need to finalize the deployment schedule by tomorrow.  
\<gap=3.2s\> Speaker B: I don't think the backend is ready for that kind of load.  
\<gap=0.2s\> Speaker A: The load testing cleared yesterday.  
\<gap=0.0s\> Speaker B: But the database migration hasn't started.  
In this sequence, the 3.2-second gap clearly highlights Speaker B's hesitation or reluctance. The subsequent 0.2s and 0.0s gaps (effectively an overlap) highlight the escalating urgency and mildly adversarial nature of Speaker A's pushback. The LLM, guided by the system prompt, will synthesize this not just as a disagreement over database readiness, but as a "tense exchange where backend readiness was challenged."

## **Reference-Free, Local Evaluation Methodology**

Establishing that temporal enrichment improves summarization requires a rigorous, objective evaluation framework. The final research objective necessitates a methodology that can evaluate summary quality locally, offline, and entirely without human annotators. Traditional natural language processing evaluation metrics present severe theoretical and practical limitations when applied to generative dialogue summarization, necessitating a shift toward modern "LLM-as-a-judge" architectures.44

### **The Deficiencies of Traditional Metrics**

Automated offline metrics such as ROUGE (Recall-Oriented Understudy for Gisting Evaluation), BLEU, and BERTScore were originally engineered for highly extractive tasks or machine translation. They evaluate output quality by measuring overlap against a human-authored "gold standard" reference text.47

1. **ROUGE and BLEU Limitations:** These metrics strictly measure surface-level token recall and precision, calculating n-gram overlaps.48 They heavily penalize abstractive summaries that successfully capture the correct pragmatic meaning but utilize different vocabulary than the reference text.51 In conversational domains, where original speech is inherently disfluent, repetitive, and fragmented, ROUGE scores frequently fail to correlate with human judgments of summary quality.12  
2. **BERTScore Constraints:** While BERTScore improves upon ROUGE by utilizing contextual embeddings from models like BERT to measure semantic similarity rather than exact token matches, it still fundamentally requires a human-written reference summary.47 For a small-scale, local project lacking an established dataset of manually annotated summaries, generating hundreds of "gold-standard" references for testing is cost-prohibitive and unscalable.  
3. **Hallucination Blindness:** Modern LLMs, particularly when summarizing dialogue, frequently engage in "Contextual Inference"—generating highly plausible statements based on circumstantial conversational cues that were never explicitly stated.31 Traditional metrics cannot detect these nuanced, span-level factual inconsistencies, nor can they measure whether the abstractive tone of the summary matches the actual pacing of the audio.55

### **The CREAM Evaluation Framework**

To overcome the absence of reference summaries and the severe limitations of n-gram metrics, the most optimal architecture for this specific use case is the **CREAM** (Comparison-Based Reference-Free ELO-Ranked Automatic Evaluation for Meeting Summarization) framework.57  
Developed collaboratively by researchers at JPMorgan Chase and Columbia University, CREAM was specifically engineered to evaluate long-context, dialogue-based meeting summarizations without requiring gold-standard references or the original source document during the final comparison phase.60 It evaluates candidate summaries strictly on two axes: *Completeness* and *Conciseness*, utilizing an Elo ranking system derived from automated pairwise comparisons.58  
By deploying the local Qwen 2.5 1.5B model iteratively as an automated judge, the system can reliably benchmark the "Baseline" (flat transcript) summary against the "Candidate" (timing-enriched transcript) summary entirely offline.62

### **Implementing CREAM Locally: A Step-by-Step Methodology**

Executing the CREAM framework using a local SLM requires carefully structuring the prompts to avoid overwhelming the model's context window or triggering its inherent biases.

#### **Step 1: Fact Extraction (Establishing the Baseline)**

Comparing two lengthy summaries directly against a massive 10,000-token source transcript can overwhelm a 1.5B parameter model, causing it to lose focus and issue random judgments.61 Therefore, the CREAM framework relies on distilling the source material first.  
The evaluator LLM is fed the original, enriched source transcript and instructed to extract a definitive list of core facts.  
**Prompt for Fact Extraction:**  
You are an expert information extraction system. Review the following conversation transcript carefully.  
Extract a definitive, chronological list of the core factual events, decisions made, actions assigned, and key opinions expressed.  
Present the output as a concise, numbered list of discrete facts. Focus strictly on objective data. Do not include introductory or concluding remarks.  
Transcript:  
{enriched\_transcript}  
This extraction process yields a dense, highly objective fact\_list that serves as the proxy ground-truth for the subsequent evaluation.

#### **Step 2: Pairwise Comparison for Completeness and Conciseness**

The evaluator LLM is then presented with the extracted fact\_list, Summary A (generated from the flat baseline transcript), and Summary B (generated from the timing-enriched transcript). It is instructed to perform a blinded, pairwise comparison.45  
**Prompt for Pairwise Evaluation:**  
You are an impartial judge evaluating two summaries of a conversation.  
You will be provided with a list of "Core Facts" extracted from the original transcript, followed by Summary A and Summary B.  
Evaluate the summaries based on the following criteria:

1. Completeness: Which summary captures more of the Core Facts accurately? Does one summary capture the tone or structural progression of the conversation better?  
2. Conciseness: Which summary avoids unnecessary verbosity, redundancy, or hallucinations while effectively delivering the facts?

Core Facts:  
{fact\_list}  
Summary A:  
{baseline\_summary}  
Summary B:  
{enriched\_summary}  
Compare both summaries step-by-step. Consider which summary strikes the best balance between complete factual representation and concise delivery.  
You must conclude your response with exactly one of the following declarations on a new line:  
WINNER: Summary A  
WINNER: Summary B  
WINNER: Tie

#### **Step 3: Mitigating Judge Bias in Local SLMs**

Using a 1.5B parameter model as a judge requires programmatic safeguards. SLMs are highly susceptible to "position bias" (systematically preferring Summary A simply because it appears first in the prompt context) and "verbosity bias" (preferring longer summaries regardless of accuracy or conciseness).44  
To explicitly correct for position bias, the local Python evaluation script must implement a randomized swap protocol.67 For every pairwise comparison, the script should run the inference twice:

1. **Iteration 1:** Baseline \= Summary A, Enriched \= Summary B.  
2. **Iteration 2:** Enriched \= Summary A, Baseline \= Summary B.

The script then parses the WINNER: outputs:

* If the judge selects the Baseline model's output in *both* positions, it is recorded as a definitive win for the Baseline.  
* If the judge selects the Enriched model's output in *both* positions, it is recorded as a definitive win for the Enriched configuration.  
* If the judge selects "Summary A" in both iterations (indicating position bias rather than qualitative judgment) or returns conflicting results, the script must record the result as a Tie.45

To correct for verbosity bias, the explicit inclusion of "Conciseness" in the CREAM evaluation prompt acts as a counterweight, penalizing the model for selecting artificially inflated, verbose text.66

#### **Step 4: Elo Rating Calculation**

The results of the bias-mitigated pairwise comparisons are tracked using the Elo rating system, a standard statistical mechanism for calculating relative skill levels.57  
Each configuration (Baseline Flat Text vs. Timing-Enriched Text) begins with a baseline Elo rating of ![][image1].  
When the Baseline summary competes against the Enriched summary, the expected score ![][image2] for the Enriched configuration is calculated using the standard logistic curve:  
![][image3]  
Following the LLM judge's double-blinded decision, the actual score ![][image4] is determined (1 for a win, 0.5 for a tie, 0 for a loss). The Elo rating is then updated using a dynamic K-factor (typically set to 32 to allow rapid convergence over small sample sizes):  
![][image5]  
By automating this pipeline locally across a validation set of 50 to 100 transcripts, the system will mathematically converge on a definitive Elo gap between the baseline configuration and the timing-enriched configuration. This provides rigorous, quantitative proof of the enrichment's efficacy without requiring a single human evaluation, external API call, or gold-standard reference dataset.67

## **Strategic Implications for the Data Pipeline**

Implementing this timing-enriched architecture within the existing Lore LLMWorker pipeline necessitates minor, programmatic adjustments to the upstream data pre-processing logic.

1. **Gap Calculation Logic:** The extraction scripts parsing the primary data model must be updated to calculate the mathematical delta between the end timestamp of utterance ![][image6] and the start timestamp of utterance ![][image7].72  
2. **Negative Gaps (Overlaps):** In instances where utterance ![][image7] begins before utterance ![][image6] concludes (simultaneous speech), the resulting calculation will be mathematically negative. To optimize token comprehension for the LLM, negative values should be clamped to 0.0s or explicitly flagged as \<gap=0.0s\>. This ensures the language model correctly interprets the event as an immediate interruption or overlap, rather than attempting to logically parse a negative passage of time.9  
3. **Thresholding for Token Efficiency:** To prevent token saturation within the CPU's context window, intra-speaker micro-pauses (e.g., a speaker taking a brief breath mid-sentence) should be filtered out entirely. The pipeline should only inject \<gap\> tokens at absolute speaker-switch boundaries, or during continuous single-speaker utterances if a pause exceeds a predefined threshold (e.g., \> 1.5 seconds) indicating a significant cognitive break or topic shift. The focus must remain steadfastly on the inter-turn transitions where the interactional dynamics are most exposed.10

## **Conclusion**

The hypothesis that flattened transcripts fundamentally handicap LLM summarization by stripping out temporal pragmatics is thoroughly supported by current literature and empirical benchmarks. Conversational structures—ranging from collaborative ideation to hostile interrogation—are defined not just by vocabulary, but by turn-taking pacing, hesitation, and inter-turn latency.  
For a local, compute-constrained infrastructure utilizing the Qwen 2.5 1.5B GGUF model on CPU architecture, the optimal path forward is the lexical injection of **inter-turn gap durations** formatted as inline \<gap=X.Xs\> markers. This approach represents the precise Minimum Viable Enrichment. It delivers the highest ratio of structural context to token overhead, avoiding the catastrophic latency and memory bandwidth bottlenecks associated with per-interval metrics or LALM architectures. By anchoring the temporal data directly adjacent to the semantic data, the Transformer's self-attention mechanisms can seamlessly map pacing to intent, recovering the lost paralinguistic context.  
Validation of this enhancement can be executed entirely offline using the CREAM framework. By utilizing the local LLM as an automated judge to extract key facts and perform bias-mitigated, position-swapped pairwise comparisons, the system can dynamically calculate an Elo rating for the timing-enriched prompts against the baseline. This elegant methodology circumvents the fatal limitations of ROUGE scoring and the prohibitive costs of human annotation, providing a mathematically sound, fully automated mechanism to definitively quantify the performance gains derived from temporal enrichment.

#### **Works cited**

1. What is turn-taking in conversational AI? \- Decagon, accessed June 15, 2026, [https://decagon.ai/glossary/what-is-turn-taking-in-conversational-ai](https://decagon.ai/glossary/what-is-turn-taking-in-conversational-ai)  
2. Turn-Taking in Voice AI: The Hidden Problem That Breaks Most Demos | Retell AI, accessed June 15, 2026, [https://www.retellai.com/blog/turn-taking-voice-ai-hidden-problem](https://www.retellai.com/blog/turn-taking-voice-ai-hidden-problem)  
3. Don't Stop the Multi-Party\! On Generating Synthetic Written Multi-Party Conversations with Constraints, accessed June 15, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/40548/44509](https://ojs.aaai.org/index.php/AAAI/article/view/40548/44509)  
4. What's under the hood: Investigating Automatic Metrics on Meeting Summarization \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2404.11124v1](https://arxiv.org/html/2404.11124v1)  
5. SEQUENCE ORGANIZATION IN INTERACTION \- ResearchGate, accessed June 15, 2026, [https://www.researchgate.net/profile/Emanuel\_Schegloff/publication/280745816\_Sequence\_Organization\_in\_Interaction\_A\_Primer\_in\_Conversation\_Analysis/links/587d571608ae4445c06b6dcb/Sequence-Organization-in-Interaction-A-Primer-in-ConversationAnalysis.pdf](https://www.researchgate.net/profile/Emanuel_Schegloff/publication/280745816_Sequence_Organization_in_Interaction_A_Primer_in_Conversation_Analysis/links/587d571608ae4445c06b6dcb/Sequence-Organization-in-Interaction-A-Primer-in-ConversationAnalysis.pdf)  
6. (Studies in Interactional Sociolinguistics) Elinor Ochs, Emanuel A. Schegloff, Sandra A. Thompson-Interaction and Grammar \- Cambridge University Press (1997) | PDF | Linguistics \- Scribd, accessed June 15, 2026, [https://www.scribd.com/document/759066347/Studies-in-Interactional-Sociolinguistics-Elinor-Ochs-Emanuel-a-Schegloff-Sandra-a-Thompson-Interaction-and-Grammar-Cambridge-University-Press](https://www.scribd.com/document/759066347/Studies-in-Interactional-Sociolinguistics-Elinor-Ochs-Emanuel-a-Schegloff-Sandra-a-Thompson-Interaction-and-Grammar-Cambridge-University-Press)  
7. PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2605.20414](https://arxiv.org/html/2605.20414)  
8. Text Overlap: An LLM with Human-like Conversational Behaviors \- ACL Anthology, accessed June 15, 2026, [https://aclanthology.org/2025.sicon-1.10.pdf](https://aclanthology.org/2025.sicon-1.10.pdf)  
9. Multi-party open-ended conversation with a social robot \- Frontiers, accessed June 15, 2026, [https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1766383/full](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1766383/full)  
10. Generative Spoken Dialogue Language Modeling | Transactions of the Association for Computational Linguistics \- MIT Press Direct, accessed June 15, 2026, [https://direct.mit.edu/tacl/article/doi/10.1162/tacl\_a\_00545/115240/Generative-Spoken-Dialogue-Language-Modeling](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00545/115240/Generative-Spoken-Dialogue-Language-Modeling)  
11. An End-to-End Speech Summarization Using Large Language Model \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2407.02005v1](https://arxiv.org/html/2407.02005v1)  
12. Incorporating Speaker and Discourse Features into Speech Summarization \- ACL Anthology, accessed June 15, 2026, [https://aclanthology.org/N06-1047.pdf](https://aclanthology.org/N06-1047.pdf)  
13. Spoken DialogSum: An Emotion-Rich Conversational Dataset for Spoken Dialogue Summarization \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2512.14687v1](https://arxiv.org/html/2512.14687v1)  
14. TimeAudio: Bridging Temporal Gaps in Large Audio-Language Models \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2511.11039v1](https://arxiv.org/html/2511.11039v1)  
15. Towards Multi-Level Transcript Segmentation: LoRA Fine-Tuning for Table-of-Contents Generation \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2601.02128v1](https://arxiv.org/html/2601.02128v1)  
16. Beyond Transcripts: A Renewed Perspective on Audio Chaptering \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2602.08979v2](https://arxiv.org/html/2602.08979v2)  
17. Why Attention Patterns Exist: A Unifying Temporal Perspective Analysis \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2601.21709v1](https://arxiv.org/html/2601.21709v1)  
18. Exploring Methods for Predicting Important Utterances Contributing to Meeting Summarization \- MDPI, accessed June 15, 2026, [https://www.mdpi.com/2414-4088/3/3/50](https://www.mdpi.com/2414-4088/3/3/50)  
19. SPECTRUM: Speaker-Enhanced Pre-Training for Long Dialogue Summarization \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2401.17597v1](https://arxiv.org/html/2401.17597v1)  
20. MeetingBank: A Benchmark Dataset for Meeting Summarization | Request PDF, accessed June 15, 2026, [https://www.researchgate.net/publication/372916005\_MeetingBank\_A\_Benchmark\_Dataset\_for\_Meeting\_Summarization](https://www.researchgate.net/publication/372916005_MeetingBank_A_Benchmark_Dataset_for_Meeting_Summarization)  
21. Topic-Conversation Relevance (TCR) Dataset and Benchmarks \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2411.00038v2](https://arxiv.org/html/2411.00038v2)  
22. MeetingBank: A Benchmark Dataset for Meeting Summarization \- ACL Anthology, accessed June 15, 2026, [https://aclanthology.org/2023.acl-long.906.pdf](https://aclanthology.org/2023.acl-long.906.pdf)  
23. Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models \- AAAI Publications, accessed June 15, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/39827/43788](https://ojs.aaai.org/index.php/AAAI/article/view/39827/43788)  
24. Towards Multi-Level Transcript Segmentation: LoRA Fine-Tuning for Table-of-Contents Generation | Request PDF \- ResearchGate, accessed June 15, 2026, [https://www.researchgate.net/publication/399477975\_Towards\_Multi-Level\_Transcript\_Segmentation\_LoRA\_Fine-Tuning\_for\_Table-of-Contents\_Generation](https://www.researchgate.net/publication/399477975_Towards_Multi-Level_Transcript_Segmentation_LoRA_Fine-Tuning_for_Table-of-Contents_Generation)  
25. QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization \- ACL Anthology, accessed June 15, 2026, [https://aclanthology.org/2021.naacl-main.472.pdf](https://aclanthology.org/2021.naacl-main.472.pdf)  
26. QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization, accessed June 15, 2026, [https://tldr.takara.ai/p/2104.05938](https://tldr.takara.ai/p/2104.05938)  
27. Incorporating Speaker and Discourse Features into Speech Summarization \- Idiap Research Institute — EN, accessed June 15, 2026, [https://www.idiap.ch/webarchives/sites/publications.amiproject.org/Murray06.pdf](https://www.idiap.ch/webarchives/sites/publications.amiproject.org/Murray06.pdf)  
28. Report on the SIGDial 2021 Special Session on Summarization of Dialogues and Multi-Party Meetings (SummDial) \- SIGIR, accessed June 15, 2026, [https://www.sigir.org/wp-content/uploads/2022/02/p12.pdf](https://www.sigir.org/wp-content/uploads/2022/02/p12.pdf)  
29. Reading Between the Lines: Combining Pause Dynamics and Semantic Coherence for Automated Assessment of Thought Disorder \- PMC, accessed June 15, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC13172256/](https://pmc.ncbi.nlm.nih.gov/articles/PMC13172256/)  
30. Beyond Transcripts: A Renewed Perspective on Audio Chaptering \- arXiv, accessed June 15, 2026, [https://arxiv.org/pdf/2602.08979](https://arxiv.org/pdf/2602.08979)  
31. Analyzing LLM Behavior in Dialogue Summarization: Unveiling Circumstantial Hallucination Trends \- ACL Anthology, accessed June 15, 2026, [https://aclanthology.org/2024.acl-long.677/](https://aclanthology.org/2024.acl-long.677/)  
32. Chain-of-Thought Supervision Enables Explainable and Efficient Medical Summarization with Small Language Models \- TechRxiv, accessed June 15, 2026, [https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177083629.95649324](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177083629.95649324)  
33. What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations \- ACL Anthology, accessed June 15, 2026, [https://aclanthology.org/2025.acl-long.310.pdf](https://aclanthology.org/2025.acl-long.310.pdf)  
34. Solve Context Window Limits With Context Packing | Docker, accessed June 15, 2026, [https://www.docker.com/blog/context-packing-context-window/](https://www.docker.com/blog/context-packing-context-window/)  
35. Multimodal Conversation Structure Understanding \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2505.17536v1](https://arxiv.org/html/2505.17536v1)  
36. Hearing Between the Lines: Unlocking the Reasoning Power of LLMs for Speech Evaluation \- ACL Anthology, accessed June 15, 2026, [https://aclanthology.org/2026.findings-eacl.151.pdf](https://aclanthology.org/2026.findings-eacl.151.pdf)  
37. LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2603.23640v2](https://arxiv.org/html/2603.23640v2)  
38. Towards Multi-Level Transcript Segmentation: LoRA Fine-Tuning for Table-of-Contents Generation \- arXiv, accessed June 15, 2026, [https://arxiv.org/pdf/2601.02128](https://arxiv.org/pdf/2601.02128)  
39. Run terminal-qwen-1.5b API | Serverless Inference | 32K Context | Flat-Rate Pricing \- Featherless.ai, accessed June 15, 2026, [https://featherless.ai/models/MarcUss01/terminal-qwen-1.5b](https://featherless.ai/models/MarcUss01/terminal-qwen-1.5b)  
40. Empirical Analysis of Small Language Model Quantization for CPU-Only Cloud Infrastructure \- TechRxiv, accessed June 15, 2026, [https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177220011.12966071](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177220011.12966071)  
41. From Independence to Interaction: Speaker-Aware Simulation of Multi-Speaker Conversational Timing \- arXiv, accessed June 15, 2026, [https://arxiv.org/pdf/2509.15808](https://arxiv.org/pdf/2509.15808)  
42. Qwen 2.5: Multimodal LLM Evolution \- Emergent Mind, accessed June 15, 2026, [https://www.emergentmind.com/topics/qwen-2-5](https://www.emergentmind.com/topics/qwen-2-5)  
43. Evaluating Small Language Models for News Summarization: Implications and Factors Influencing Performance \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2502.00641v1](https://arxiv.org/html/2502.00641v1)  
44. A Survey on LLM-as-a-Judge \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2411.15594v6](https://arxiv.org/html/2411.15594v6)  
45. LLM-as-a-judge: a complete guide to using LLMs for evaluations \- Evidently AI, accessed June 15, 2026, [https://www.evidentlyai.com/llm-guide/llm-as-a-judge](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)  
46. LLM as a Judge \- Primer and Pre-Built Evaluators \- Arize AI, accessed June 15, 2026, [https://arize.com/llm-as-a-judge/](https://arize.com/llm-as-a-judge/)  
47. Evaluating LLMs for Text Summarization: An Introduction \- Software Engineering Institute, accessed June 15, 2026, [https://www.sei.cmu.edu/blog/evaluating-llms-for-text-summarization-introduction/](https://www.sei.cmu.edu/blog/evaluating-llms-for-text-summarization-introduction/)  
48. Understanding Offline Evaluation Metrics in NLP and LLMs: A Deep Dive \- Medium, accessed June 15, 2026, [https://medium.com/@xiaxiami/understanding-offline-evaluation-metrics-in-nlp-and-llms-a-deep-dive-b5233c36669b](https://medium.com/@xiaxiami/understanding-offline-evaluation-metrics-in-nlp-and-llms-a-deep-dive-b5233c36669b)  
49. A Practical Guide for Evaluating LLMs and LLM-Reliant Systems \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2506.13023v2](https://arxiv.org/html/2506.13023v2)  
50. LLM Evaluation Metrics: The Ultimate LLM Evaluation Guide \- Confident AI, accessed June 15, 2026, [https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)  
51. Metrics for evaluating summarizations done by LLMs? : r/LanguageTechnology \- Reddit, accessed June 15, 2026, [https://www.reddit.com/r/LanguageTechnology/comments/188ftf9/metrics\_for\_evaluating\_summarizations\_done\_by\_llms/](https://www.reddit.com/r/LanguageTechnology/comments/188ftf9/metrics_for_evaluating_summarizations_done_by_llms/)  
52. Reasoning or Not? A Comprehensive Evaluation of Reasoning LLMs for Dialogue Summarization \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2507.02145v1](https://arxiv.org/html/2507.02145v1)  
53. How to Evaluate Large Language Models \- Galileo AI, accessed June 15, 2026, [https://galileo.ai/blog/llm-evaluation-step-by-step-guide](https://galileo.ai/blog/llm-evaluation-step-by-step-guide)  
54. Best Practices and Methods for LLM Evaluation | Databricks Blog, accessed June 15, 2026, [https://www.databricks.com/blog/best-practices-and-methods-llm-evaluation](https://www.databricks.com/blog/best-practices-and-methods-llm-evaluation)  
55. \[2406.03487\] Analyzing LLM Behavior in Dialogue Summarization: Unveiling Circumstantial Hallucination Trends \- arXiv, accessed June 15, 2026, [https://arxiv.org/abs/2406.03487](https://arxiv.org/abs/2406.03487)  
56. Tailored Cross-Lingual Conversational Data Summarization and Evaluation, accessed June 15, 2026, [https://ncsu-las.org/2025/11/cross-lingual-conversation-summarization-evaluation-bbn/](https://ncsu-las.org/2025/11/cross-lingual-conversation-summarization-evaluation-bbn/)  
57. \[Literature Review\] CREAM: Comparison-Based Reference-Free ELO-Ranked Automatic Evaluation for Meeting Summarization \- Moonlight, accessed June 15, 2026, [https://www.themoonlight.io/en/review/cream-comparison-based-reference-free-elo-ranked-automatic-evaluation-for-meeting-summarization](https://www.themoonlight.io/en/review/cream-comparison-based-reference-free-elo-ranked-automatic-evaluation-for-meeting-summarization)  
58. CREAM : Comparison-Based Reference-Free ELO-Ranked Automatic Evaluation for Meeting Summarization \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2409.10883v1](https://arxiv.org/html/2409.10883v1)  
59. Papers \- JPMorganChase, accessed June 15, 2026, [https://www.jpmorganchase.com/about/technology/research/machine-learning/papers](https://www.jpmorganchase.com/about/technology/research/machine-learning/papers)  
60. arXiv:2409.10883v1 \[cs.CL\] 17 Sep 2024, accessed June 15, 2026, [https://arxiv.org/pdf/2409.10883](https://arxiv.org/pdf/2409.10883)  
61. (PDF) CREAM: Comparison-Based Reference-Free ELO-Ranked Automatic Evaluation for Meeting Summarization \- ResearchGate, accessed June 15, 2026, [https://www.researchgate.net/publication/384085799\_CREAM\_Comparison-Based\_Reference-Free\_ELO-Ranked\_Automatic\_Evaluation\_for\_Meeting\_Summarization](https://www.researchgate.net/publication/384085799_CREAM_Comparison-Based_Reference-Free_ELO-Ranked_Automatic_Evaluation_for_Meeting_Summarization)  
62. LLM evaluation: a beginner's guide \- Evidently AI, accessed June 15, 2026, [https://www.evidentlyai.com/llm-guide/llm-evaluation](https://www.evidentlyai.com/llm-guide/llm-evaluation)  
63. Evaluate AI models with Vertex AI & LLM Comparator | Google Cloud Blog, accessed June 15, 2026, [https://cloud.google.com/blog/products/ai-machine-learning/evaluate-ai-models-with-vertex-ai--llm-comparator](https://cloud.google.com/blog/products/ai-machine-learning/evaluate-ai-models-with-vertex-ai--llm-comparator)  
64. LLM-as-a-Judge Simply Explained: The Complete Guide to Run LLM Evals at Scale, accessed June 15, 2026, [https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method)  
65. LLMs as Judges: A Comprehensive Survey on LLM-Based Evaluation Methods \- Arize AI, accessed June 15, 2026, [https://arize.com/blog/llm-as-judge-survey-paper/](https://arize.com/blog/llm-as-judge-survey-paper/)  
66. A Reference-Free Conciseness Evaluation Metric for LLM-Generated Answers \- arXiv, accessed June 15, 2026, [https://arxiv.org/pdf/2511.16846](https://arxiv.org/pdf/2511.16846)  
67. LLM Arena-as-a-Judge: LLM-Evals for Comparison-Based Regression Testing \- Confident AI, accessed June 15, 2026, [https://www.confident-ai.com/blog/llm-arena-as-a-judge-llm-evals-for-comparison-based-testing](https://www.confident-ai.com/blog/llm-arena-as-a-judge-llm-evals-for-comparison-based-testing)  
68. A Reference-Free Conciseness Evaluation Metric for LLM-Generated Answers \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2511.16846v1](https://arxiv.org/html/2511.16846v1)  
69. RAGElo is a set of tools that helps you selecting the best RAG-based LLM agents by using an Elo ranker · GitHub, accessed June 15, 2026, [https://github.com/zetaalphavector/RAGElo](https://github.com/zetaalphavector/RAGElo)  
70. Elo as a tool for ranking LLMs. In a previous blog post, Vikas and… | by Rahul | Thomson Reuters Labs | Medium, accessed June 15, 2026, [https://medium.com/tr-labs-ml-engineering-blog/elo-as-a-tool-for-ranking-llms-dab056dc9713](https://medium.com/tr-labs-ml-engineering-blog/elo-as-a-tool-for-ranking-llms-dab056dc9713)  
71. Prediction-Powered Ranking of Large Language Models \- arXiv, accessed June 15, 2026, [https://arxiv.org/html/2402.17826v2](https://arxiv.org/html/2402.17826v2)  
72. WhisperX with Diarization | Guides \- Clore.ai, accessed June 15, 2026, [https://docs.clore.ai/guides/audio-and-voice/whisperx](https://docs.clore.ai/guides/audio-and-voice/whisperx)  
73. Improving the Naturalness of Simulated Conversations for End-to-End Neural Diarization \- ISCA Archive, accessed June 15, 2026, [https://www.isca-archive.org/odyssey\_2022/yamashita22\_odyssey.pdf](https://www.isca-archive.org/odyssey_2022/yamashita22_odyssey.pdf)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFQAAAAaCAYAAAApOXvdAAADHElEQVR4Xu2YW6hNQRjHP/dIuZWUcHJ5RpFbHnhwSZIi4eGE8KLUeXEppORBoZS8E6GEJCWXlEsk8sCDSHlwKbmGkPj++5tvz6xvz+y9PFjnxPzqX7P+85991v72rFkzhyiTyWS6PIOsEdCTtZ31mXWbta7YXQeZa+RzMYayDrDusy6zphW7/w3msu5Y09GLdZL1hDWAtZr1kbUvDJHkkJlHPmczw1kPWZdYE1l7WD9YS8JQK5azfiX0jnWKNaeerhbcwxeS2YJ2jK0kfeE97nReCHL2eyATzsCLrPesgYF3lfWK1T/wSrGA9Y01zF3jEVvJuk7yh7c4v0rGs3qwplJjgUB31muSInQLfMyuMK+5MAOQOWSuzwTXYKPzVxi/JVdIHgPLUvIztrNIFRQ/OHysiyEoIPzR7lpzFv1emkN7l++uMdv5z43flH4ks3Ow7WBuknzgW9tRIamC4qmBf952MJ/IP+KasyADX3Nob/bdNaY4PzY+CRZ9OwC/8mHnf2ctK3ZXSqqgu0n8C7aD+cBa6NqasyADX3No26VtsvNj45PsJz8o1E/WMdYYH01ygxrHN9MRGVaKVEHxloYfK+gb8m9nzVmQga85tG1BdT2OjY/SRhLGh4dgAX9Kzfd/VZEqaAelC4rZh3FAcxadoZqLFVRnKHYbpVhLMuC07SDx//jt9hdIFXQViY99o+Ura5xra86CDHzNob3Dd9eY7vxnxk9ygmTAJuPjMYe/xvidQaqgk0j8u8bHJh4+tlxAcxZ4mKWaw7Xd7Ov75azxo+DIhrc3Bsw0fe3O1wW7FbdI8mWFH7IsqYJiWXpBjVuaNirmNWex94Fru7a3O3+D8aNsIwk/IrmJEF2MF5EU/lyxu1L0S81i9TZ9Q0jWerxYQV+S082DekJADhnd3COHTPh5M0i2j4vd9VjWS9bBeiLBSGqcMbEZcJTk/Iu9KE4MVWPvT3UvDDETSF4ae1mPSQo1opAQkMFuRHOxjL5TcOTG03uc1aeQ+E/APzxwXl9PMvtiIDOffC7FKJITImaqPa5mMplMJpPJZDJdgt8Y+fUnTSsqCQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE0AAAAaCAYAAADygtH/AAACzUlEQVR4Xu2XS6iNURTHF3m/ykUhFPK4DISBKEJeoSgDJlKuPEqIlPe9E+888iqMZISJKAOS6zVASgaUCYURKcoAA/7/1tqdddY9de53OCP7V//O2mt/e39772+vtfcRyWQymUzGsR763U5lHN2hPdBnqLf5OkKjoCnQOuiD+TOOt9Ca4Bvq7C3OzoCxouE3JPi3Ozsu6H/PBmmbsxZAk4Mv47gqbZM+1cE/lCkxSXSBvgX/FWd3E92NiYXQHeiL/d6D3oj2c8M9VysrRHNsOpQq0Rm6BZ2LFQWYLjrmxbGiGsxbbHg7+J85eyN01pVJH+h48B2BLgRfLSyCdkRnBTpBK6OzIJz7oOisBheLDVuCP8EQfQVNCP650JLgYw7cFXz1ZCY0ODoLwrkVoif0U3TR5oW6xFqoNTrBUajBlbkzZsnff/kiNEdHDZyKjmowvLhgz2OF6IWXC/pRNAw8DM1f0EHohWgfU139MKv7IfoO5h/2Mcfq6esrmg9Z9x06YXXMn0+gbVbmh30MrbYy39XDxP7TXfISdN5s9sE8y/4J3zfNbF7gx5u9U7SfdsFG6YSsphZtUgZzDusSD0UXZbSVJ5r9SPSfBRlhfrIM6gfNtzJ3J5N+L6hRtO+RVndGdNESaSfPlvIx8BnmVMIUwY/KBeWiXxPdBF2h3fYM2e/sunNIygecdoU/cUmzs3mYeJZK5a98ALrvytztlf6N7INem828xvHMsPJT6B20F2qCBpifp6TPzQ+cXXdaoZfRKTrARH/RwyLBidCXJnbY1XneQ6ugcdBw0cXg1SByV0rhuFm0XbpX8uQ/ZrZnk5R2Pnc1PwhTBsO5rvBlzCU+gXKwDDWGQGK5aDgQLhYnv1X0LxvxIef5KjqJk6LtP0n5KX1dNJQ5Yd7nyE3otJROUkbCRbMHQpehMaL9cNG6WD3HFK9SmUwmk8lk/gl/AJodko7LLECTAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEoAAAAaCAYAAAAQXsqGAAACyElEQVR4Xu2XSehPURTHj3nIvEAIURQLrEjoV2zMCYUi00Z2KFNJSrKg2MhONoaFDRki/oYMZUqmLGxEkoUNksT5dM7zrtP7lR9+G7/7qW/v3HPve/e+c+897z6RTCaTyfy3tFNdUH1VvVFdU3VXXVf1TNq1NP1VV1WLVL1Vw1XrxIJ1r2zW2uxXfVfNjxVi/lp0tio3xQIyPVYor1Udo7NVuSEWqFuq9qFuZCi3NGvEAoXeqU6otorlrUwdSOZHpQxc5KKY/73bbarP7jtWNvtjlql6RGfgrOqL6mCsaIAXqm+qLrGiHoOjw2FFVQWqr5h/b+Ijhx1w/S2zo6MCXu6jammsaADG2tDXfG10OIOkOlAzxfzxhaaqNgRfs6iJjWFg8DfCHWlwYldHhzNHqgPFUYIDaS8vd/DrLNVCt5vNDtXj6GwQtu686KzHSrGjwRWxlxyqGq+6pDot1ccCgsdXco/qkZdTeMYSsby1T+wZtKl5PYF+ojovtoVoB+RFyne9DGNUL90+JzYu4J421RAvpytjhNhfRcFtvzIW7CL/kV9/myNiK2KL2ABJ0ARogfurSLcdvzwEDUb7dYJYMDjlUw/cw0sDE0J5hpdXiP0eUR7rdUD/D1W7vcxheIrbtElz667EJqAP3B6lOuw29xT3w7bEbgp0SEIv2CS2ao4nPl6Q7VHwPLHhk6pz8AGr9LLbTAZ9jSurf/I0sQnYJLf7iN3DDiFf8gUt+nkr5cSRNqa53TTuR4eySn79AjHY9JTPqiDxFjN6JqkrYAW9Ui1XDVOtF9ti8QAMhxJ7o1/Jkfyj0ndV7jmV2ExCJ9XcxPdPYcbYVinM0jOxjgv4dBfnE2Z8smqzlKd8jh4Rtt4HVVexVcBXlJfuJ7ZiWSEEvJtqsd8DbDXy4k4v83fBdoaJYmc7gn3SfeQ12gxQbXdfJpPJZDIZ+QHFq4jv6rKbGAAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABDCAYAAAAh8FnvAAAL9UlEQVR4Xu3dB6xsRRnA8c+GvXdReYoFa2yxw7sIKooFS2yo72nAAhoxdmxPBXuvsb6HBUtiYomxywUVsdeIsd4gKoqxETVojM4/M+POztu9u3te8Xrv/5dM7tk5u/fufmfPnDnfzDk3QpIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZK0Bty4r5AWdJm+QpIk7T6PT+UCfaW0oK+mcvG+UpKmuXIqS6k8K5XzUzmkPKaspPLv/LTYksp9UvlzKrdL5Q7NOrw+lQum8txUjk7l6qlcqXnOCeXnAak8oCyvVzWmfHZiyjLlyMgxJdaazw1jFMuzyzLfs2pzKr9MZVvk79uedJXI7+HYfkVx61SWI7+nS6ayPZUfpXJy8xxNd5NUPhh5Wx8co/2G8uZS/6781HXh0pHb00f0KyRpNc9O5bNdHZ03GsmbpnJEKhdN5RVl3cNS+WNZPjCVl6TywlQeW+p4Dcs/LY+Xy0/QMG8ExG61mG4kdGJ3BTE7qa9MXpnKxfrKPeQ1qfwhJmdFLpfK72K8M4mPxmif0GzsL+2JYHWhyG3Jif2K/3OvjlEbKUlzOTeVO3V1x0duPO9VHt8/lUuV5X9EzpRdJ/KBbJJ/pnKPGM/G7Ttave7xmWfFdKM4tK9YADEkZu28sdtEPlHYm3gP/N1Jfh6TOxpkozU/YjgpjiBTSRu03nypr5CkaTgQ0khyFttaKfXVG5rlv0fuvO2TyuExyiww9FlT/L9P5X6pHBOj3/OC8nO9I6ZkINuYEq+VVE5r6jaKu/YVC3hqKr9pHj86la83j+fBUBrfxUn4fc/oKyc4L5UL95XFtI7GLfoKrYoY0m5UO1I5rCx/IpXbj1atG6+LPDwqSTO9M0YHnLa8KWxIhpoU039FjulGdPe+YgG/TuVJqVw+lVMjx/IGY8+YD3PM7tLVMRx1za5uEq7oW23+1FtjfFv/JJVN7RM0Eyd9xI7pGdgvlW+OVq9bB6VyXF8pSZP8MMazA8yrYIh0o7hn7Ny5mlbm7SgQ05uXZeb+EdObjVZvOEM7bFykQtwfGfng/bLy+EXtkxZwTip3LssPiXxRwzzYlmzDacg0vyPGvytkBTc1z9Hq2D/6/a3N6q9XfLc2ysiDpF3woMgN43JX/61U7tbVaT41pj3qJsX0fZHX0SH5RuQDPY/7bNC8mIjfX+zQ4gDB79/a1c/C5PkzY/Jna3FhSn/gnVS4unOW7ZGfu6mpq68fiiE2snaLdKAZ0n1yX7mKmi36fFP3zFROT+VnkbfPSuR5nrvyWR6cyhX6ysYnI09fWBTbmvfFxRR7y3si3+qixdXmYK4sFz5VxJL3Ryy/GKPpG9tGT1nYSkyP5UUix5HM+aJm7TNkb/nskrQq5k/QmLQXDlwi8kUF3G9Ki6sxbRFT6qbFlHVXbB6flcr1mseL4GB2776yw98b8vtfHPlWFYsammEjDn0sfzyhbhF02H4Vi3XY7hjTb8fyqr6i4D0y76rFRTjtxHnmNfbPWUSd3zUNc+7+2lfOaZ7tzPxCOp+zykPrC1axkspLm8ft1b8fSOVRzWMQ3zaWxJH5iENxQdBqiOOQ3z9rn+EK6h19pST1yOrQ8LWN4VKpI1PUo/ElA0TmgGUm1vNczkCH4AakZ/SVDbJBXEW1taufB/PF/hdzYGpMW0ulblJM8bXu8ftj8u0jdpeVvmJObPchWYahHTZi1sfy6aXusl39PMhacvsNrnCm0zYvhk4Zju3RIfpTX1nwHvtOwLbYOYsz7b5uuwPzo/r4zWvIdh5q/8jvc1IGGlyI0O8PPL+NJXEc+j2bB3+P97moWfsM0yymXWkvSf9FI0Snq71/1FNK/W3L43a+EDcEHTLEsivqJORF0Ync2lfuYdyuo8a01caUSe5tTMm61JsJnxLjnd93p/KFVJ4WOZvx3mbd21L5VOSz8zr0xhW6dP7abB2dFIYp31Iek7ngAEKncEeM/4sc3ttrY3S/PZBZInvBUCtDeEOGyYYeSIkZMemRcVn0SlFuybHU1TGkdu2ubhpi3au3aekRc+bItcimEb+qzqUDnRG2NR2WrZEvcHhCWUcG5suRT6oYPmc+H7an8u2yDIYNufEsc+1q5omTqs+VOrZ3xXbmb7CduSktOHliW5MRYlsP2c5D1Ys2JnlM5M/WIpZ8T8FNu9tYEsdPR44lw9jtxSJ8LmJJHD9c6jgpJJab65Mi/703xvi8ReK4o9TV+alYbZ8hlrP2Gb4n7LeSNBENzlLkRvI7ZbmiAWJ+0QMjH9zb7FudoL83cdAZ4sAYzYHZG4gpc1FqTBlGq4gp9cSUjlYb0xMjN/a8loNCi44X8wlrg9/efoI77/M7633uuNM/k7RZrrc/IKPHe+FqX+7Sj6XIB0g631wlubXUc5D6blmmE1M78fXAxgFo6D2jFu2wcQBeivxZ6IS0N1vmVhkcYFnHZO22wzkNWQzmrfWOSuX7ka9AnYXMXI85YsSMbcdnPCRyFm3SSQ3rec90MNi/2iG/LTH9P4nQGWAbsh+Q0atXGvN65jxW3AuO2HCz3pq15n3QQUHbceE9E1O2M3PA2NYvj/FtvbcwXaC/j92NIu+/dLwmtTfEkk47sTw/xmNJHIkZsQSx5LsOMnLEkjjW9Ww7Xk/mFgdEfj8glmA9cayd+5oxm7XPYNY+Q9zJ4ErSbnVa5AMXB482WwA6AXQufhC5g0HngoPBvpH/EwIN71L5Sd2Rke+N9bcYoTHltXQu6MQsRX5+bXDpYFSnRh4WW4qceaIRrgcnMhbnluW1rM4XrDhYo875OSLy7UDAwWClLIP4LDePqxOaZYaSiEuLjF3Fwa7Gljgzib3FdiPzAt7ntCGrWRbtsK1FfN+5vchQy5H3D9R/o8UJBXMeQef80LJMJ6LNphL7fkiQbcM+gMNTuVWzrmK+H8jS1e8Zf6ffzjgn8rbuv5Nr0XKzfHT52caR73Iby4q5nZM+G7GseG0fS04MKmJJBhWz9hliudo+wz79l75SknYHGrs6TFCHAOo8EhouDmp1wjtDcB+P/G+YOIP8SqknC4F9Ig/TnFQeXzfygQfHRr4y8PmR72kFhkHoYICOGg3x8yJ3UMiyPC5G84loKOtz17KDY+cMAh0sYgPmttBJRs1u3jfyDXmpf05ZV+0f+WKCwyL/HjIUrf5gTPw5uBD39iBXfahZJlvDNmDYd1HzDjuuZXzfyMYNxfexv00FmaA6vMYwJvEF24hhcjI4+Ez52SILRKaW7f3E2PnfYoFbjYD9iTl7bDtOiPrtjI+Un+yfbOsh23lvINs1ad9u40j8iCWZU5bJPBJLstl9LPn+E0sya8SS393Hsj3JIZbsf8Rn1j5DLFfbZ2grJ30WSdplNFBXbR5fK8b/x2PtXKA+l8aNRmtzsw5HRe6M1caxbegqGtstZZkhEDoYdC5olNu5XiDzx1wxcEZ8SuROzlpFlvK8yA02yxQOlO0ZN8N4HETAAfx7MZqfRge4z56RyaJDXYf4GH6l08dcGrIGXKXaZm5Ojjw/Dhyw3h65E1gvjrha5O37scjbloP6euh8DUUWp84XnNemyNuW/YEMcN3WPG6/n+0cubMjn4CA/ef6zbrqrBhllcC8rHbbMeRZh4uZ/0g2rmZT2c4MAfLcmg0iM8625nvGtl6L23lTKr+IHLsaR2LVnvQQR6YeYL/I629ZHjMU3ceS7zixpFOHYyLHkpNF4kMczyzrQCy5AIVYztpniOW0fYYsOllN2lBJ2u3aSc7gvkkHNY+3NcvLkW8aC4YF2nkm4MBHQ1eHH45v1h0XeSIuDfE1Sh1zThhGpbPG/C0aUjw8cgN6RuTXMX+LRpoM4LQzW2kIThJO7yulAX4b4xdLSJIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSWvZfwC7328LXXkj6gAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAAA9ElEQVR4Xu2TMQtBURTHj1IyGBkMEpPJYjLZLSaLyWDjM8imGCyyGyySzXeQkqRksGGyyyD+17mvd+99z7VK71e/Ovf+X+f27juPKOCn6MKnYk2P36i5o5UOvMGlGYAEnMIzLMGolhrE4BVGiE+d6PGbNcybm36U4VzWotkdJt2Y4vCkrK30YVPWB+KGbTemKhwraysrmJN1i7jZhfi1BSNYl7UVv1N7xA23MAR3evyZIXlPTcMHcUNxnwMttXCEKXOT+IuKZjNYMTJfMnBvbkqK5A6oGB0r4nI3cAGzRuYg7vPrtBfI+3s0tCeYMPGoBAT8Dy8pPzSnCzFQSAAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAaCAYAAAD8K6+QAAABmUlEQVR4Xu2WPyhFURzHf/JvYVAyykIWRSkxSdkMyN9sBhlkt9lkIMRgMFlEGayy+DvKYiAZjEooGSS+p9893r2/9653zrm3LnU+9em99/ude979vnPvuY/I4/EkZBA2y+J/phYuwC/YL3qxzBIfoF2LtqkSfgQ97UFkRHG6YL0sGnIOb+AjWQbTjBEf+AbrRE+FGyfuu5zgMvH8LrQFr4fkGOweLhEf/ADLom2ahpuiZoq6ClyDaXQwdZ8Z0wRvYQP8JJ5gIjwA7MARUTNlHY7KoiVOwdRqqC9XbBNPcJFrUyl8Ir6JXdigjFZsFw4E79spt0loOuBZ6LMtmQTrhKei1kc8ibpEFcew+6f7O/pHMdUUHWxINuKYg/OiVgKv4Basgc+wPDLCjjRXbFg24jiiwqsxCd/hFNwXPVvSDGa0gVURbwqFVkM9u9REL3BG9GxJM1jR3bUCnsBL2EJ8+Ule4Z0sOpA0WDW8Jg62KnoRWin/Jlb/DiQrcFEWHUgSbI/yz1XZEx6UFUmC/Wl6YaMsejwej8fjwDccwW0XQfzgDAAAAABJRU5ErkJggg==>