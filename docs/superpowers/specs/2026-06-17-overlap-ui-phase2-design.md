# Overlap UI Surfacing — Phase 2 Design

**Date:** 2026-06-17
**Status:** Draft
**Builds on:** Phase 1 overlap detection engine (`OverlapDetector`, `OverlapRegion` data model, Phase 3 integration in `TranscriptionWorker`)

---

## Overview

Surface overlapping-speech information from the Phase 1 detection pipeline in the Lore
UI and OHMS export. Two visual surfaces plus one serialization target:

| Surface | Priority | Description |
|---------|----------|-------------|
| A | Ship first | Per-segment overlap badge + tinted left border in transcript view |
| C | Ship first | Dedicated overlap-region strip between waveform and segment list |
| VTT | Ship last | Overlapping WebVTT cue pairs in OHMS export |

---

## A — Per-Segment Overlap Badge

### What it looks like

Follows the existing confidence-badge pattern in `TranscriptDelegate`: a small
coloured pill next to the timestamp, and a 4px left-border tint on the segment
row. Both use a warm amber (`#c8925e`) colour — visually distinct from the
existing badge colours (gold/medium, orange/low, red/hallucination, grey/non-speech)
and low-alarm: overlaps in oral history are ordinary (backchannels, affirmations),
so the visual weight should be informational, not a warning.

Examples:

```
[01:23:45 - 01:23:50] | Speaker A     ⬩⟪ overlap ⟫     ← pill badge
│███                                                      ← 4px left border tint
│  <segment text here>
```

When a segment has *no* overlap: no badge, no border tint — unaffected segments
remain clean.

### Data path

1. `TranscriptListModel` gains a new `OverlapRole` constant (UserRole + 8,
   see audit below):
   - Checks `Transcript.overlap_regions` against the segment's `[start_ms, end_ms)` interval
   - Returns `True` if the segment overlaps any region, `False` otherwise
   - The helper `_segment_overlaps()` returns `bool` — the delegate only
     needs binary presence, not duration

2. `TranscriptDelegate.paint()`:
   - Reads `OverlapRole` from the model index
   - If truthy, draws the 4px left border in `#c8925e` (same pattern as confidence borders)
   - Draws the pill badge `⟪ overlap ⟫` next to the timestamp using same colour

3. `TranscriptDelegate.sizeHint()`:
   - Adds badge width to the time header area if overlap present (same pattern as confidence badge sizing)

### Colour reference

```python
OVERLAP_COLOR = QColor("#c8925e")       # warm amber — used for border + badge
OVERLAP_BADGE_BG = QColor("#3d3020")     # dark amber background for pill
OVERLAP_BADGE_TEXT = "⟪ overlap ⟫"       # pill label
```

### Role constant audit

Existing roles in `TranscriptListModel`:

```python
StartMsRole         = Qt.ItemDataRole.UserRole + 1
EndMsRole           = Qt.ItemDataRole.UserRole + 2
TextRole            = Qt.ItemDataRole.UserRole + 3
SpeakerRole         = Qt.ItemDataRole.UserRole + 4
ConfidenceLevelRole = Qt.ItemDataRole.UserRole + 5
TranslationRole     = Qt.ItemDataRole.UserRole + 6
WordsRole           = Qt.ItemDataRole.UserRole + 7
# +8 is free ← OverlapRole lives here
```

`OverlapRole` added as a named constant alongside the others:

```python
OverlapRole = Qt.ItemDataRole.UserRole + 8  # free slot, see audit above
```

### Model helper

`TranscriptListModel` overlap check — O(n) on overlap_regions, which
are typically <50 even for long recordings. No indexing needed:

```python
def _segment_overlaps(self, seg_start_ms: int, seg_end_ms: int) -> bool:
    """Returns True if segment overlaps any OverlapRegion, else False."""
    for region in self._transcript.overlap_regions:
        lo = max(seg_start_ms, region.start_ms)
        hi = min(seg_end_ms, region.end_ms)
        if lo < hi:
            return True
    return False
```

---

## C — Overlap Region Strip

### Layout position

Inserted between the waveform widget and the transcript view, in the same
container as the waveform. The layout becomes:

```
┌──────────────────────────────────┐
│           Waveform               │  WaveformWidget
├──────────────────────────────────┤
│    ╔══╗     ╔════╗    ╔══╗       │  OverlapStripWidget  ← NEW
├──────────────────────────────────┤
│   Segment list (transcript)      │  TranscriptWidget (QListView)
│   [01:23:45]  ⟪ overlap ⟫       │
│   <text>                         │
└──────────────────────────────────┘
```

### Widget design

- **Height:** 24px (short, unobtrusive)
- **Background:** `#1e1e1e` (matches waveform/transcript background)
- **Blocks:** Each `OverlapRegion` renders as a filled rectangle in `#c8925e` at 50% opacity
  (`alpha=128`), positioned by its `start_ms`/`end_ms` relative to total duration
- **Empty state:** No regions → nothing drawn (widget is a thin empty bar)
- **Cursor:** Changes to hand pointer when hovering a block
- **Click:** Emits `overlap_clicked(ms: int)` — `MainWindow` connects this to scroll
  the transcript view to the segment that contains that time position

### Click behaviour

1. User clicks a block in the strip
2. Strip emits `overlap_clicked(ms)` where `ms` is the midpoint of that region
3. `MainWindow._on_overlap_clicked(ms)`: finds the first segment where
   `start_ms <= ms < end_ms` and calls `transcript_view.scrollTo(index)`
   with `PositionAtCenter`

### Implementation

`OverlapStripWidget(QWidget)` — new file `src/ui/overlap_strip.py`:

```python
class OverlapStripWidget(QWidget):
    overlap_clicked = pyqtSignal(int)  # ms

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self.overlap_regions: list[OverlapRegion] = []
        self.total_duration_ms: int = 0
        self._hovered_region: Optional[int] = None

    def set_regions(self, regions: list[OverlapRegion], duration_ms: int):
        self.overlap_regions = regions
        self.total_duration_ms = duration_ms
        self.update()

    def paintEvent(self, event):
        # Draw filled rects for each region, positioned proportionally
        # to start_ms/duration_ms. Colour: #c8925e at alpha=128 (normal)
        # or alpha=200 (hovered).

    def mousePressEvent(self, event):
        # Map x-coordinate to ms, find containing region, emit midpoint

    def mouseMoveEvent(self, event):
        # Track _hovered_region for hover highlight
```

### New public model method — `segment_index_at()`

To keep `MainWindow` from reaching into `transcript_model._transcript.segments`,
expose a focused public method on `TranscriptListModel`:

```python
def segment_index_at(self, ms: int) -> Optional[QModelIndex]:
    """Return the QModelIndex of the segment containing `ms`, or None."""
    for i, seg in enumerate(self._transcript.segments):
        if seg.start_ms <= ms < seg.end_ms:
            return self.index(i, 0)
    return None
```

### Wiring in MainWindow

In `_on_transcription_finished()` (line ~418), after the waveform is loaded:

```python
transcript = self.transcript_model.get_transcript()
self.overlap_strip.set_regions(
    transcript.overlap_regions,
    transcript.metadata.duration_ms
)
```

And the click handler:

```python
self.overlap_strip.overlap_clicked.connect(self._on_overlap_strip_clicked)

def _on_overlap_strip_clicked(self, ms: int):
    """Scroll transcript to the segment containing ms."""
    idx = self.transcript_model.segment_index_at(ms)
    if idx is not None:
        self.transcript_view.scrollTo(
            idx, QAbstractItemView.ScrollHint.PositionAtCenter
        )
        self.transcript_view.setCurrentIndex(idx)
```

The `_on_overlap_detected` signal (line 261/388) already populates
`transcript.overlap_regions` — the strip simply re-reads from the same model
at transcription finish time.

---

## VTT Overlap Cues (OHMS Export)

Implemented last, after A + C are merged and tested.

### Approach

OHMS XML v6.0 wraps transcript data in a `<vtt_transcript>` CDATA block
containing standard WebVTT. The VTT spec forbids overlapping cue timelines,
so duplicate-cue approaches are fragile. The cleanest strategy is inline
annotation: append `[overlap]` to the cue text for any segment whose time
range intersects an `OverlapRegion`.

This is unambiguous, OHMS-player-safe, and trivial to implement. The overlap
detector does not distinguish *which* speaker overlaps, so there is nothing
lost by omitting NOTE blocks — and nothing gained by duplicating cues with
the same speaker label.

**Decision:** Strategy 2 — inline `[overlap]` annotation.

### VTT output example

```webvtt
WEBVTT

00:01:23.450 --> 00:01:23.900
<v Speaker A>right, exactly [overlap]

00:01:24.000 --> 00:01:30.000
<v Speaker B>and that's when I realised...
```

Implementation in `ohms_exporter.py` — in the main VTT generation loop, check
each segment against `transcript.overlap_regions` (same `_segment_overlaps()`
helper from the model) and append ` [overlap]` to the text if overlapping.

---

## Files changed

| File | Change |
|------|--------|
| `src/models/transcript_model.py` | Add `OverlapRole` constant, `_segment_overlaps()` helper, `segment_index_at()` public method |
| `src/ui/transcript_widget.py` | `TranscriptDelegate`: render overlap badge + left border |
| `src/ui/overlap_strip.py` | **NEW** — `OverlapStripWidget` |
| `src/ui/main_window.py` | Layout: insert strip between waveform and transcript; wire click handler |
| `src/ui/waveform_widget.py` | No changes (strip sits below it, not inside it) |
| `src/lore_core/ohms_exporter.py` | Inline `[overlap]` annotation on affected cues (last) |

## Tests

- **`tests/test_overlap_strip.py`**: Unit tests for `OverlapStripWidget`:
  - No regions → empty paint
  - Single region at known position → rect coordinates match proportion
  - Click at region midpoint → emits correct ms
  - Multiple regions → each click targets the correct region
- **`tests/test_overlap_model.py`**: Unit tests for `_segment_overlaps()`:
  - Segment fully inside overlap → True
  - Segment partially overlapping → True (partial is still overlap)
  - Segment outside all regions → False
  - No overlap regions at all → False
  - Multiple regions, segment overlaps one → True
  - Multiple regions, segment overlaps none → False
- **`tests/test_ohms_export.py`**: Add overlap cue test case — segment in overlap region emits `[overlap]` in VTT cue text; non-overlapping segment unaffected

## Risk and mitigation

| Risk | Mitigation |
|------|------------|
| Strip at 24px is too small to click accurately | Start at 24px; bump to 32px if feedback says so |
| Overlap badge adds visual noise for long transcripts | Only shown on segments that overlap; most segments are clean |
| OHMS player rejects inline `[overlap]` annotation | Unlikely — it's just text in the cue; remove if a player chokes on brackets inside cue payload |
| `overlap_strip` needs audio player seek sync | Already handled — click emits ms, same as waveform click |
