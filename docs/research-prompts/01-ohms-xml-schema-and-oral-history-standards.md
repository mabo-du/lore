# Deep Research Prompt 01 — OHMS XML Schema & Oral History Deposit Standards

## Purpose

Lore is an open-source, privacy-first oral history transcription application built on `faster-whisper`. Its primary differentiator is producing output in formats acceptable to institutional oral history repositories. The most critical of these is OHMS XML.

This prompt requests comprehensive technical and domain research covering the OHMS format, related oral history standards, and the repository ecosystem Lore must integrate with.

---

## Research Questions

### 1. OHMS XML Schema — Technical Specification

The Oral History Metadata Synchronizer (OHMS) is developed by the Louie B. Nunn Center for Oral History at the University of Kentucky Libraries.

- What is the **current version** of the OHMS XML schema (as of 2025–2026)? Is the XSD or DTD publicly available, and where?
- Provide a **complete annotated sample OHMS XML file** showing all required and commonly used optional elements, including:
  - The `<record>` envelope (collection, repository, accession number, rights)
  - The `<transcript>` block with `<sync>` elements (timecode → text mapping)
  - The `<index>` block (keyword/segment index entries)
  - The `<keywords>` and `<subjects>` blocks
  - Speaker/interviewee metadata fields
- Which fields are **mandatory** vs optional for a valid OHMS import? What causes OHMS to reject a file?
- What is the **timecode format** used in `<sync>` elements — seconds as float, HH:MM:SS, or milliseconds? What precision does OHMS expect?
- Does OHMS XML support **multiple speakers** / speaker labelling? If so, what element carries speaker identity?
- What is the OHMS **character encoding** requirement? UTF-8? Any entities that must be escaped?

### 2. OHMS Ecosystem — Platforms and Repositories

- Which repository platforms currently support OHMS XML as a native deposit format? Specifically:
  - **CONTENTdm** — what version added OHMS support, and how is the XML ingested?
  - **Omeka-S** with the Oral History plugin — what are the exact import requirements?
  - **Aviary** (AVP) — does it support OHMS XML, or its own format?
  - **AtoM** (Access to Memory) — OHMS support status?
- Are there any **deprecation risks** in OHMS XML format? Is OHMS being superseded by another standard?
- What is the **OHMS Viewer** and does Lore need to know anything about it to produce compatible output?

### 3. Competing and Complementary Standards

- **WebVTT** — Is WebVTT (.vtt) accepted as an alternative to OHMS XML at major oral history repositories? What metadata does WebVTT lack that OHMS XML provides?
- **SRT** — Same question for SRT subtitle format.
- **JSON-LD oral history profiles** — Are there any emerging JSON-LD or schema.org oral history metadata profiles that Lore should be aware of?
- **Dublin Core sidecar** — Should Lore produce a Dublin Core XML sidecar alongside OHMS XML for repository deposits that require DC metadata separately?

### 4. OHMS Index vs Transcript — Use Cases

- What is the **difference** between the OHMS transcript block (verbatim text with timecodes) and the OHMS index block (keyword/topic segments)? Are both required or is one optional?
- Do institutions typically use **both** index and transcript, or one or the other?
- Can a Lore output file contain **only transcript** (no index) and still be valid for deposit?

### 5. Oral History Association — Ethical and Metadata Standards

- What does the **Oral History Association Best Practices** document (most recent edition) say about:
  - Minimum metadata required for archived oral histories?
  - Rights statements and licensing standards (Creative Commons variants used)?
  - Speaker consent documentation — does any standard require linking consent forms to the transcript record?
- What is **TRAILS** (Transfer of Rights and Acquisition in Library Systems) and is it relevant to Lore's output?

### 6. Practical Implementation Guidance

- Are there any **open-source Python or JavaScript OHMS generators** (even incomplete ones) in public repositories? What do they get right, and what do they get wrong?
- What are the **most common errors** archivists report when submitting OHMS XML files to CONTENTdm or Omeka-S? (Check community forums, listservs, or GitHub issues.)
- The Scope document specifies "100–150 lines of straightforward XML generation." Is this estimate realistic for a compliant OHMS serialiser, or are there edge cases that make it more complex?

---

## Context

**Project:** Lore — oral history transcription tool  
**Stack:** Python 3.12+, faster-whisper, PyQt6, OHMS XML export  
**Key constraint:** Audio never leaves the machine. No cloud dependencies during transcription.  
**Ecosystem:** Lore transcripts may be attached to Cache & Carry (collections management) object records as text assets. The JSON sidecar format must be compatible.

---

## Deliverables Requested

1. A technically precise description of the OHMS XML schema with a complete annotated sample
2. A table of platforms and their OHMS support status
3. A comparison of OHMS XML vs WebVTT vs SRT for oral history repository deposit
4. A summary of OHA best practice metadata requirements
5. A list of known implementation gotchas / common submission errors
6. Any open-source OHMS generators to reference

---

*This research will directly inform the implementation of `lore/src/exporters/ohms.py` and the metadata form in Phase 5–6 of the Lore build.*
