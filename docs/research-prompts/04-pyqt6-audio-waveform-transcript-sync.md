# Deep Research Prompt 04 — PyQt6 Audio Waveform Display, Playback, and Transcript Sync

## Purpose

Lore requires a custom PyQt6 UI that displays an audio waveform, provides playback controls, and synchronises the transcript editor so that clicking a timestamp in the transcript seeks the audio player, and advancing playback highlights the current segment in the editor. This is a non-trivial UI challenge. This prompt requests comprehensive technical research covering the specific PyQt6 patterns for waveform rendering, audio playback, and bidirectional sync with a text editor.

---

## Research Questions

### 1. Audio Playback in PyQt6

- What is the **recommended audio playback library** for a PyQt6 desktop application in 2025–2026?
  - The scope proposes `pygame.mixer` or `sounddevice`. Which is better for this use case (seeking to arbitrary timestamps, detecting playback position for sync)?
  - Is `QMediaPlayer` (via `PyQt6.QtMultimedia`) now viable for audio playback on Linux/Windows/macOS? What are its limitations compared to pygame or sounddevice?
  - Does any of these allow **precise position queries** (e.g., current playback position in milliseconds)?
  - Does any support **seeking** to an arbitrary position by timestamp?
- What is the **format compatibility** of each option? Can they play MP3, M4A, OGG, FLAC, and WAV natively, or does the audio need to be decoded first?
- How should audio playback run in relation to the **Qt event loop**? Should it run on the main thread, a QThread, or a separate process?

### 2. Waveform Rendering

- What is the **most efficient approach** for rendering an audio waveform in a PyQt6 widget?
  - Option A: Custom `QWidget` using `QPainter` to draw amplitude bars
  - Option B: `pyqtgraph` for interactive waveform display
  - Option C: Pre-render the waveform to an image (e.g., using `librosa` or `scipy`) and display it in a `QLabel`
  - Which approach supports the best performance for a 4-hour audio file?
- How should a **4-hour waveform** be efficiently downsampled for display? What is the recommended approach to avoid loading millions of samples into the painter at once?
- How should the **playhead** (a vertical line tracking current position) be rendered and updated in real time during playback without excessive repaints?
- Is there a **PyQt6-compatible audio waveform widget library** that already implements these features? (Search GitHub for maintained examples.)

### 3. Transcript Editor — Click-to-Seek

- Lore's transcript editor requires that clicking a timestamp in the `QTextEdit` seeks the audio player to that position. What is the **recommended implementation pattern**?
  - Should timestamps be stored as QTextEdit `QTextCharFormat` with custom data, or as HTML anchors?
  - How do you detect a click on a specific word/timestamp within a `QTextEdit`?
  - Is `QTextBrowser` (read-only with link handling) better for the read phase, with a separate editing mode?
- How should **inline segment highlighting** work — as the audio plays, the current segment should be highlighted in the editor? What is the Qt mechanism for dynamically updating character formatting in a `QTextEdit`?
- What is the correct approach for **storing timecode metadata** alongside transcript text in a QTextEdit document (e.g., `QTextBlock.setUserData()`)?

### 4. SRT Format — Parsing and Display

- What is the **SRT file format spec**? Provide the exact format including CRLF vs LF requirements, timecode format, and block structure.
- Is there a Python SRT parsing library (maintained, MIT/Apache licence) that handles edge cases (missing blank lines, UTF-8 BOM, overlapping timestamps)?
- How should Lore **map SRT segments** to its internal transcript model? The internal model needs: segment index, start_ms, end_ms, speaker_label (optional), text.

### 5. Playback Position Sync — Timer Approach

- What is the **recommended approach** for polling playback position and updating the transcript highlight?
  - A `QTimer` firing every 100ms that queries the player position and updates the highlight?
  - A callback/signal from the audio backend?
- What is the performance impact of updating `QTextEdit` character formatting **100 times per second** on long transcripts? Are there more efficient approaches (e.g., only searching the visible viewport)?
- How should the editor **auto-scroll** to keep the currently playing segment visible without jarring jumps?

### 6. Speaker Label Editing UX

- The scope requires letting archivists **rename speaker labels** (e.g., changing `[Speaker A]` to `[Dr Smith]`). What is the best implementation?
  - A find-and-replace approach?
  - A separate speaker management panel that maps labels to names?
  - Double-click on a speaker label to rename?
- How should renamed labels be **persisted** in the project file so they survive save/reload?

### 7. Trowel and HOARD Pattern Reference

- Trowel (a sibling PyQt6 project) has a `harris_editor.py` (15 KB) and `review.py` (11 KB). **What patterns does Trowel use** for its custom editors that Lore should reuse or adapt? (Specifically: signal/slot patterns for editor→data sync, undo/redo implementation, and any custom widget patterns.)

---

## Context

**Project:** Lore — oral history transcription tool  
**Framework:** PyQt6 (matching Trowel's stack)  
**Target hardware:** Desktop (Linux, Windows, macOS), mid-range consumer hardware  
**Audio files:** 1–4 hours, loaded from disk  
**Transcript:** Up to ~10,000 segments for a 4-hour recording at 30-second chunks  
**Design target:** Dark theme, keyboard-navigable, accessible

---

## Deliverables Requested

1. Recommended playback library for PyQt6 with seek support (pygame vs sounddevice vs QMediaPlayer)
2. Efficient waveform rendering approach for 4-hour audio files in PyQt6
3. Click-to-seek implementation pattern using QTextEdit character data
4. SRT parser library recommendation with edge-case handling
5. Playback position polling pattern (QTimer + highlight update) with performance notes
6. Speaker label renaming UX pattern
7. Any existing open-source PyQt/PySide audio editor widgets that could be referenced

---

*This research will directly inform `lore/src/ui/waveform_widget.py`, `lore/src/ui/transcript_editor.py`, and `lore/src/audio/player.py`.*
