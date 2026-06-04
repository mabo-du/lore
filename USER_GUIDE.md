# Lore User Guide

Welcome to Lore! This guide will walk you through the end-to-end process of transcribing, editing, and archiving oral histories completely offline.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Configuring Custom Vocabulary](#configuring-custom-vocabulary)
3. [Transcription & Diarization](#transcription--diarization)
4. [Translation](#translation)
5. [Editing and Review](#editing-and-review)
6. [Metadata & Archival Export](#metadata--archival-export)
7. [Global Search](#global-search)

---

## 1. Getting Started

When you launch Lore, you will be greeted by the **File Picker**. 
- Click **Browse** to locate your audio file. Lore supports common formats like `.wav`, `.mp3`, `.m4a`, `.ogg`, and `.flac`.
- The application will automatically normalize the audio in the background to ensure it is in the optimal format (`16kHz`, mono) for AI transcription.
- Once normalization is complete, the main editor interface will open.

## 2. Configuring Custom Vocabulary

Before transcribing, it's highly recommended to provide Lore with contextual terms.

1. Click the **⚙️ Settings** button on the top right of the toolbar.
2. In the **Custom Vocabulary** field, enter a comma-separated list of terms that are specific to your recording. 
   - *Example: "Nunn Center, oral history, John Doe, Appalachia"*
3. These terms are passed to the AI as an "initial prompt." The AI will use this prompt to heavily bias its decoding towards spelling these localized terms correctly, without slowing down the transcription speed.

## 3. Transcription & Diarization

With your audio loaded and vocabulary set, you're ready to extract the text.

1. **Diarize (Speaker Identification):** If your audio features multiple speakers, check the **Diarize** checkbox on the toolbar. The AI will analyze the audio to separate and tag who is speaking (e.g., `SPEAKER_00`, `SPEAKER_01`).
2. **Transcribe:** Click the **Transcribe** button. Lore will begin processing the audio. 
   - *Note: Lore runs entirely on your CPU. Transcription speed depends on your processor. For a 1-hour file on a modern CPU, it typically takes 10-20 minutes.*
3. As segments are processed, they will appear in real-time in the main transcript window.

## 4. Translation

Lore includes a massive, offline translation engine capable of translating between 200+ languages.

1. Once your English (or source language) transcription is complete, locate the **Translate to:** dropdown on the toolbar.
2. Select your desired target language.
3. Click **Translate**. The AI will generate a translated version of each segment. The original text will remain available, allowing you to export dual-language (bilingual) archives.

## 5. Editing and Review

AI is powerful, but it makes mistakes. Lore provides tools to quickly review and correct errors.

- **Word-Level Confidence:** Lore scores every single word the AI produces. If the AI was less than 60% confident about a word, it will be highlighted with a **pale red underline**. This allows you to instantly scan a massive transcript for potential hallucinations or tricky auditory moments.
- **Audio Sync:** Click on any text segment in the transcript, and the audio waveform at the top of the screen will instantly seek to that exact timestamp. Press the Play button (or Spacebar) to listen.
- **Editing:** Double-click any segment to modify the text or correct a speaker label.

## 6. Metadata & Archival Export

Lore is built for archival standards. On the right side of the screen, you will find the **Metadata Form**.

1. Fill out the required OHMS XML fields (Interview Title, Repository Name, Interviewee, format, rights, etc.).
2. **Auto-Generate Abstract:** Click the purple `✨ Auto-Generate Abstract` button to have Lore's local AI (Llama 3) read the entire transcript and generate a concise, professional summary for your archives.
3. **Export Options:**
   - **Export OHMS XML:** Saves a raw `transcript.xml` file conforming to the OHMS XML 6.0 schema, including your transcript, translation, metadata, and auto-extracted named entities.
   - **📦 Export BagIt Package:** Generates a full RFC 8493 compliant folder structure. It packages your original audio and the OHMS XML together, generating SHA-256 cryptographic checksums to guarantee data integrity for long-term cold storage.

## 7. Global Search

Lore automatically indexes every transcript you export into a global, unified database.

1. Click the **🔍 Global Search** button on the toolbar.
2. **Exact/Keyword Search:** Instantly find exact phrases across all your past projects.
3. **Semantic/Concept Search:** Powered by local vector embeddings. Search for concepts like "space exploration" and find segments where someone is talking about "NASA" or "rockets" even if the word "space" was never explicitly spoken. 
4. The search results will show the exact timestamp and snippet.
