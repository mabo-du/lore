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

1. `TranscriptListModel` gains a new `OverlapRole` (UserRole + 8):
   - Checks `Transcript.overlap_regions` against the segment's `[start_ms, end_ms)` interval
   - Returns `True` if the segment overlaps any region, `None` otherwise
   - The helper `_segment_overlaps()` returns `Optional[bool]` — the delegate only
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

### Model helper

`TranscriptListModel` overlap check — O(n) on overlap_regions, which
are typically <50 even for long recordings. No indexing needed:

```python
def _segment_overlaps(self, seg_start_ms: int, seg_end_ms: int) -> bool:
    """Returns True if segment overlaps any OverlapRegion."""
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
    for i, seg in enumerate(self.transcript_model._transcript.segments):
        if seg.start_ms <= ms < seg.end_ms:
            idx = self.transcript_model.index(i, 0)
            self.transcript_view.scrollTo(
                idx, QAbstractItemView.ScrollHint.PositionAtCenter
            )
            self.transcript_view.setCurrentIndex(idx)
            break
```

The `_on_overlap_detected` signal (line 261/388) already populates
`transcript.overlap_regions` — the strip simply re-reads from the same model
at transcription finish time.

---

## VTT Overlap Cues (OHMS Export)

Implemented last, after A + C are merged and tested.

### Approach

OHMS XML v6.0 wraps transcript data in a `<vtt_transcript>` CDATA block
containing standard WebVTT. WebVTT supports `NOTE` comments but has no
native "overlapping cue" construct (the spec forbids overlapping cue
timelines). Two viable strategies:

1. **NOTE-annotated duplicate cues** (preferred): Emit a normal cue for the
   primary speaker, then immediately after emit a second cue for the same
   time range with a `NOTE overlap` annotation. The OHMS player renders
   both cues — the overlap is evident from the duplication.

2. **Overlap as inline annotation**: Append `(overlap)` to the cue text
   for segments that coincide with an overlap region. Simpler but loses
   the ability to distinguish which speaker overlaps.

**Decision:** Use strategy 1 (duplicate cues with NOTE) for now. If OHMS
players can't handle duplicate time ranges, fall back to strategy 2.

### VTT output example

```webvtt
WEBVTT

00:01:23.450 --> 00:01:23.900
<v Speaker A>right, exactly

NOTE overlap region: 00:01:23.450-00:01:23.900 (confidence 0.87)

00:01:23.450 --> 00:01:23.900
<v Speaker A>right, exactly

00:01:24.000 --> 00:01:30.000
<v Speaker B>and that's when I realised...
```

Implementation in `ohms_exporter.py` — after the main VTT loop, iterate
`transcript.overlap_regions` and for each region, find overlapping segments
and emit duplicate cues prefixed with `NOTE overlap`.

---

## Files changed

| File | Change |
|------|--------|
| `src/models/transcript_model.py` | Add `OverlapRole`, `_segment_overlaps()` helper, restore `get_transcript()` etc. |
| `src/ui/transcript_widget.py` | `TranscriptDelegate`: render overlap badge + left border |
| `src/ui/overlap_strip.py` | **NEW** — `OverlapStripWidget` |
| `src/ui/main_window.py` | Layout: insert strip between waveform and transcript; wire click handler |
| `src/ui/waveform_widget.py` | No changes (strip sits below it, not inside it) |
| `src/lore_core/ohms_exporter.py` | VTT overlap cue blocks (last) |

## Tests

- **`tests/test_overlap_strip.py`**: Unit tests for `OverlapStripWidget`:
  - No regions → empty paint
  - Single region at known position → rect coordinates match proportion
  - Click at region midpoint → emits correct ms
  - Multiple regions → each click targets the correct region
- **`tests/test_overlap_model.py`**: Unit tests for model overlap helper:
  - Segment fully inside overlap → returns full duration
  - Segment partially overlapping → returns partial duration
  - Segment outside overlap → returns None
  - Multiple overlap regions → sums correctly
- **`tests/test_ohms_export.py`**: Add overlap cue test case (strategy 1 or 2)

## Risk and mitigation

| Risk | Mitigation |
|------|------------|
| Strip at 24px is too small to click accurately | Start at 24px; bump to 32px if feedback says so |
| Overlap badge adds visual noise for long transcripts | Only shown on segments that overlap; most segments are clean |
| OHMS player rejects duplicate VTT time ranges | Fall back to inline `(overlap)` annotation in cue text |
| `overlap_strip` needs audio player seek sync | Already handled — click emits ms, same as waveform click |
