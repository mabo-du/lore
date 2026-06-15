# Timing-Enriched LLM Summarization — Implementation Plan

## Status: Research Complete

Two deep-research reports converged on the same recommendation: **inline `<gap=X.Xs>` markers at inter-turn boundaries** are the single most impactful timing enrichment per token of overhead. Overlap flags provide binary information; gap durations provide continuous quantitative signal about conversational pace.

The approach is implementable today — no new models, no new dependencies. The data (gap durations between segments) already exists in the `Segment.start_ms`/`end_ms` fields.

## Implementation Steps

### Step 1 — Add gap computation to Transcript model

Add a method to compute inter-turn gap durations from the existing segment timeline:

```python
def compute_gaps(self) -> list[float]:
    """Return gap durations (in seconds) between consecutive segments."""
    gaps = []
    for i in range(1, len(self.segments)):
        gap = (self.segments[i].start_ms - self.segments[i-1].end_ms) / 1000.0
        gaps.append(max(0.0, gap))
    return gaps
```

Zero-cost — purely arithmetic on existing data.

### Step 2 — Create the enriched transcript formatter

```python
def format_enriched(transcript: Transcript) -> str:
    lines = []
    for i, seg in enumerate(transcript.segments):
        if i > 0:
            gap = (seg.start_ms - transcript.segments[i-1].end_ms) / 1000.0
            lines.append(f"<gap={max(0.0, gap):.1f}s>")
        speaker = seg.speaker_label or "SPEAKER"
        lines.append(f"{speaker}: {seg.text}")
    return "\n".join(lines)
```

Output format:
```
SPEAKER_00: I think we should proceed with the deployment.
<gap=3.2s>
SPEAKER_01: I don't think the backend is ready.
<gap=0.2s>
SPEAKER_00: The load testing cleared yesterday.
```

### Step 3 — Update LLMWorker system prompt

Replace the current summarisation prompt with the timing-aware version:

```
You are an expert dialogue analyst and executive summarizer.
You will be provided with a conversation transcript. To assist your
understanding of the conversation's tone and structure, the transcript
contains explicit timing markers formatted as <gap=X.Xs>. These markers
indicate the duration of silence (in seconds) between spoken turns.

Your task is to generate a comprehensive, professional summary. You must
use the semantic content AND the <gap> markers to accurately interpret
the structural dynamics of the exchange.

Apply the following temporal rules:
1. Short gaps (<0.5s) or 0.0s gaps = fast-paced, collaborative, or urgent
2. Moderate gaps (0.5s-1.5s) = formal turn-taking
3. Long gaps (>1.5s) = hesitation, deliberation, reluctance, or topic shift

CRITICAL: Do NOT mention the gap markers explicitly in your generated
summary. Use them solely as internal analytical context.
```

### Step 4 — Run ablation study to validate

**Control:** Qwen 2.5 1.5B + flat transcript
**Experimental:** Qwen 2.5 1.5B + gap-enriched transcript

**Metric:** ROUGE-L + BERTScore comparison on a small corpus
**Method:** Generate reference summaries using Qwen on the training set, then compare both conditions against those references.

Expected improvement based on QMSum benchmarks: +0.42 ROUGE-L, +0.34 BERTScore.

### Token cost analysis

- ~5-6 tokens per speaker switch
- 1-hour interview with ~400 turns = ~2,000-2,400 extra tokens
- Qwen 2.5 1.5B has 32K context window
- Overhead is <10% of context — well within safe margins

## Evaluation Framework (CREAM)

If basic ROUGE/BERTScore metrics don't show clear improvement, implement the **CREAM** framework:
1. Feed enriched transcript → LLM as judge → extract factual claims
2. Pairwise compare summaries from flat vs enriched conditions
3. Score on *completeness* (fact coverage) and *conciseness* (no hallucination)
4. Use Qwen 2.5 as the judge (no API calls needed)

## When NOT to do this

Skip gap enrichment if the LLMWorker's context window is already >90% saturated (unlikely for typical oral history transcripts, but possible for very long interviews). In that case, the gap tokens would push the prompt past the effective window and degrade quality.

## References

See `docs/research-papers/Timing-Enriched LLM Summarization Research.md` and `docs/research-papers/Gap Durations Over Overlap Flags_ A Minimalist Approach to Improving Local LLM Summarization with Timing Data.md`.
