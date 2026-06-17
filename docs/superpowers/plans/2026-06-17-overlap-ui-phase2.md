# Overlap UI Surfacing — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface overlapping-speech information in the Lore UI via per-segment overlap badges (A), an overlap-region strip widget (C), and inline `[overlap]` annotations in OHMS WebVTT export (VTT-last).

**Architecture:** Three consumer layers consume the Phase 1 `OverlapRegion` data model: the transcript delegate renders a tinted border + pill badge for overlapping segments; a new `OverlapStripWidget` sits between the waveform and transcript list showing coloured blocks proportional to overlap region timing; the OHMS exporter appends `[overlap]` to affected segment cues in the VTT block. The model layer exposes an `OverlapRole` for binary overlap queries and a `segment_index_at()` public method for click-to-scroll. All share the same intersection-check helper (`_segment_overlaps()`).

**Tech Stack:** Python 3.12+, PyQt6, ONNX Runtime (existing), lxml (existing), pytest.

---

### Task 1: Model layer — OverlapRole, helper, and public index lookup

**Files:**
- Modify: `src/models/transcript_model.py:9-16` (add OverlapRole constant)
- Modify: `src/models/transcript_model.py:25-46` (add role handler)
- Create: `tests/test_overlap_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_overlap_model.py`:

```python
"""Tests for TranscriptListModel overlap helper."""

import pytest
from models.transcript import Transcript, OverlapRegion, Segment
from models.transcript_model import TranscriptListModel


def _make_model(segments: list[Segment], regions: list[OverlapRegion] = None):
    t = Transcript(segments=segments, overlap_regions=regions or [])
    return TranscriptListModel(t)


class TestSegmentOverlaps:
    def test_segment_fully_inside_overlap(self):
        """Segment entirely within an overlap region → True."""
        model = _make_model(
            segments=[Segment(start_ms=1000, end_ms=2000, text="hello")],
            regions=[OverlapRegion(start_ms=500, end_ms=2500, confidence=0.8)],
        )
        assert model._segment_overlaps(1000, 2000) is True

    def test_segment_partial_overlap(self):
        """Segment partially overlapping a region → True."""
        model = _make_model(
            segments=[Segment(start_ms=1000, end_ms=2000, text="hi")],
            regions=[OverlapRegion(start_ms=1500, end_ms=3000, confidence=0.8)],
        )
        assert model._segment_overlaps(1000, 2000) is True

    def test_segment_outside_all_regions(self):
        """Segment touching no overlap region → False."""
        model = _make_model(
            segments=[Segment(start_ms=3000, end_ms=4000, text="bye")],
            regions=[OverlapRegion(start_ms=500, end_ms=2500, confidence=0.8)],
        )
        assert model._segment_overlaps(3000, 4000) is False

    def test_no_overlap_regions(self):
        """No overlap regions at all → False."""
        model = _make_model(
            segments=[Segment(start_ms=0, end_ms=1000, text="none")],
        )
        assert model._segment_overlaps(0, 1000) is False

    def test_multiple_regions_overlaps_one(self):
        """Multiple regions, segment overlaps exactly one → True."""
        model = _make_model(
            segments=[Segment(start_ms=3000, end_ms=4000, text="mid")],
            regions=[
                OverlapRegion(start_ms=500, end_ms=2500, confidence=0.8),
                OverlapRegion(start_ms=3000, end_ms=4000, confidence=0.9),
            ],
        )
        assert model._segment_overlaps(3000, 4000) is True

    def test_multiple_regions_overlaps_none(self):
        """Multiple regions, segment overlaps none → False."""
        model = _make_model(
            segments=[Segment(start_ms=5000, end_ms=6000, text="late")],
            regions=[
                OverlapRegion(start_ms=500, end_ms=2500, confidence=0.8),
                OverlapRegion(start_ms=3000, end_ms=4000, confidence=0.9),
            ],
        )
        assert model._segment_overlaps(5000, 6000) is False


class TestSegmentIndexAt:
    def test_returns_index_for_exact_hit(self):
        seg = Segment(start_ms=1000, end_ms=2000, text="hit")
        model = _make_model(segments=[seg])
        idx = model.segment_index_at(1500)
        assert idx is not None
        assert idx.row() == 0

    def test_returns_index_for_start_boundary(self):
        seg = Segment(start_ms=1000, end_ms=2000, text="edge")
        model = _make_model(segments=[seg])
        idx = model.segment_index_at(1000)
        assert idx is not None
        assert idx.row() == 0

    def test_returns_none_for_miss(self):
        seg = Segment(start_ms=1000, end_ms=2000, text="miss")
        model = _make_model(segments=[seg])
        idx = model.segment_index_at(999)
        assert idx is None

    def test_returns_none_for_empty_model(self):
        model = _make_model(segments=[])
        idx = model.segment_index_at(500)
        assert idx is None

    def test_returns_correct_index_among_many(self):
        segs = [
            Segment(start_ms=0, end_ms=1000, text="first"),
            Segment(start_ms=1000, end_ms=2000, text="second"),
            Segment(start_ms=2000, end_ms=3000, text="third"),
        ]
        model = _make_model(segments=segs)
        idx = model.segment_index_at(1500)
        assert idx is not None
        assert idx.row() == 1


class TestOverlapRole:
    def test_role_true_for_overlapping_segment(self):
        seg = Segment(start_ms=1000, end_ms=2000, text="x")
        model = _make_model(
            segments=[seg],
            regions=[OverlapRegion(start_ms=500, end_ms=2500, confidence=0.8)],
        )
        idx = model.index(0, 0)
        val = model.data(idx, model.OverlapRole)
        assert val is True

    def test_role_false_for_clean_segment(self):
        seg = Segment(start_ms=3000, end_ms=4000, text="x")
        model = _make_model(
            segments=[seg],
            regions=[OverlapRegion(start_ms=500, end_ms=2500, confidence=0.8)],
        )
        idx = model.index(0, 0)
        val = model.data(idx, model.OverlapRole)
        assert val is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/mark/Projects/lore && python -m pytest tests/test_overlap_model.py -v --no-header -x 2>&1`
Expected: FAIL with import/attribute errors (role and methods not yet defined)

- [ ] **Step 3: Add OverlapRole constant and data handler**

In `src/models/transcript_model.py`, add after line 16 (the `WordsRole = UserRole + 7` line):

```python
    OverlapRole = Qt.ItemDataRole.UserRole + 8  # free slot, see role audit in design spec
```

In `src/models/transcript_model.py`, add after the existing role chain (before `return None` at line 46):

```python
        elif role == self.OverlapRole:
            return self._segment_overlaps(segment.start_ms, segment.end_ms)
```

- [ ] **Step 4: Add `_segment_overlaps()` helper**

Add to `TranscriptListModel` class, anywhere after the existing methods:

```python
    def _segment_overlaps(self, seg_start_ms: int, seg_end_ms: int) -> bool:
        """Returns True if segment time range intersects any OverlapRegion."""
        for region in self._transcript.overlap_regions:
            lo = max(seg_start_ms, region.start_ms)
            hi = min(seg_end_ms, region.end_ms)
            if lo < hi:
                return True
        return False
```

- [ ] **Step 5: Add `segment_index_at()` public method**

Add to `TranscriptListModel` class:

```python
    def segment_index_at(self, ms: int) -> QModelIndex | None:
        """Return the QModelIndex of the segment containing `ms`, or None."""
        for i, seg in enumerate(self._transcript.segments):
            if seg.start_ms <= ms < seg.end_ms:
                return self.index(i, 0)
        return None
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/mark/Projects/lore && python -m pytest tests/test_overlap_model.py -v --no-header 2>&1`
Expected: 9 passed (6 SegmentOverlaps + 5 SegmentIndexAt + 2 OverlapRole)

- [ ] **Step 7: Commit**

```bash
git add tests/test_overlap_model.py src/models/transcript_model.py
git commit -m "feat: add OverlapRole, _segment_overlaps, and segment_index_at to TranscriptListModel"
```

---

### Task 2: OverlapStripWidget — new widget

**Files:**
- Create: `src/ui/overlap_strip.py`
- Create: `tests/test_overlap_strip.py`

- [ ] **Step 1: Write the failing widget tests**

Create `tests/test_overlap_strip.py`:

```python
"""Tests for OverlapStripWidget rendering and click behaviour."""

import pytest
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest

from models.transcript import OverlapRegion
from ui.overlap_strip import OverlapStripWidget


@pytest.fixture(scope="module")
def qapp():
    """Return a QApplication instance for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestOverlapStripWidget:
    def test_empty_no_regions(self, qapp):
        """No overlap regions → nothing crashes, widget is empty."""
        strip = OverlapStripWidget()
        strip.set_regions([], 10000)
        assert strip.overlap_regions == []
        assert strip.total_duration_ms == 10000
        # Paint should not raise

    def test_single_region_position(self, qapp):
        """Single region at known position → rect x matches proportion."""
        strip = OverlapStripWidget()
        strip.resize(1000, 24)
        regions = [OverlapRegion(start_ms=0, end_ms=5000, confidence=0.9)]
        strip.set_regions(regions, 10000)
        # Region covers 0-50% of 1000px → rect width should be ~500
        rect = strip._rect_for_region(regions[0])
        assert rect.x() == 0
        assert rect.width() == pytest.approx(500, abs=5)

    def test_region_in_middle(self, qapp):
        """Region in middle → rect starts and ends at correct x."""
        strip = OverlapStripWidget()
        strip.resize(1000, 24)
        regions = [OverlapRegion(start_ms=2500, end_ms=5000, confidence=0.9)]
        strip.set_regions(regions, 10000)
        rect = strip._rect_for_region(regions[0])
        assert rect.x() == pytest.approx(250, abs=3)
        assert rect.width() == pytest.approx(250, abs=3)

    def test_click_emits_midpoint(self, qapp):
        """Clicking a region block emits overlap_clicked with the region midpoint."""
        strip = OverlapStripWidget()
        strip.resize(1000, 24)
        regions = [OverlapRegion(start_ms=0, end_ms=5000, confidence=0.9)]
        strip.set_regions(regions, 10000)

        emitted = []
        strip.overlap_clicked.connect(emitted.append)

        # Click at x=250 (midpoint of the 0-500px block)
        QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(250, 12))
        assert len(emitted) == 1
        # Midpoint of 0-5000ms = 2500ms
        assert emitted[0] == pytest.approx(2500, abs=100)

    def test_click_outside_region_no_emit(self, qapp):
        """Clicking the empty part of the strip does nothing."""
        strip = OverlapStripWidget()
        strip.resize(1000, 24)
        regions = [OverlapRegion(start_ms=0, end_ms=2000, confidence=0.9)]
        strip.set_regions(regions, 10000)

        emitted = []
        strip.overlap_clicked.connect(emitted.append)

        # Click at x=900 which is past the region's right edge (200px)
        QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(900, 12))
        assert len(emitted) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/mark/Projects/lore && python -m pytest tests/test_overlap_strip.py -v --no-header -x 2>&1`
Expected: FAIL with import error (overlap_strip module not found)

- [ ] **Step 3: Create `src/ui/overlap_strip.py`**

```python
"""Overlap strip widget — a thin horizontal bar showing overlapping speech regions."""

from typing import Optional

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QMouseEvent, QCursor
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QRectF

from models.transcript import OverlapRegion


OVERLAP_COLOR = QColor("#c8925e")
OVERLAP_COLOR_HOVER = QColor("#e8b882")


class OverlapStripWidget(QWidget):
    """
    A thin horizontal bar rendered between the waveform and transcript view.

    Each OverlapRegion is drawn as a filled rectangle proportional to its
    time range. Clicking a block emits overlap_clicked(ms) with the
    midpoint of the region, allowing MainWindow to scroll the transcript
    to the corresponding segment.
    """

    overlap_clicked = pyqtSignal(int)  # ms — midpoint of the clicked region

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self.setMouseTracking(True)
        self.overlap_regions: list[OverlapRegion] = []
        self.total_duration_ms: int = 0
        self._hovered_index: Optional[int] = None

    def set_regions(self, regions: list[OverlapRegion], duration_ms: int) -> None:
        """Set the overlap regions to display and trigger a repaint."""
        self.overlap_regions = regions
        self.total_duration_ms = duration_ms or 1  # avoid div-by-zero
        self._hovered_index = None
        self.update()

    # ── Geometry ──────────────────────────────────────────────────────────

    def _rect_for_region(self, region: OverlapRegion) -> QRect:
        """Return the pixel rectangle for a single OverlapRegion."""
        w = self.width()
        x1 = int(region.start_ms / self.total_duration_ms * w)
        x2 = int(region.end_ms / self.total_duration_ms * w)
        return QRect(x1, 0, max(x2 - x1, 2), self.height())

    def _region_index_at_x(self, x: int) -> Optional[int]:
        """Return the index of the region whose rect contains x, or None."""
        for i, region in enumerate(self.overlap_regions):
            r = self._rect_for_region(region)
            if r.left() <= x <= r.right():
                return i
        return None

    # ── Events ────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))

        for i, region in enumerate(self.overlap_regions):
            rect = self._rect_for_region(region)
            color = OVERLAP_COLOR_HOVER if i == self._hovered_index else OVERLAP_COLOR
            color.setAlpha(160 if i == self._hovered_index else 100)
            painter.fillRect(rect, color)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        idx = self._region_index_at_x(int(event.position().x()))
        if idx != self._hovered_index:
            self._hovered_index = idx
            self.setCursor(Qt.CursorShape.PointingHandCursor if idx is not None else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, event) -> None:
        self._hovered_index = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        idx = self._region_index_at_x(int(event.position().x()))
        if idx is not None:
            region = self.overlap_regions[idx]
            midpoint = (region.start_ms + region.end_ms) // 2
            self.overlap_clicked.emit(midpoint)
```

- [ ] **Step 4: Run widget tests to verify they pass**

Run: `cd /home/mark/Projects/lore && python -m pytest tests/test_overlap_strip.py -v --no-header 2>&1`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_overlap_strip.py src/ui/overlap_strip.py
git commit -m "feat: add OverlapStripWidget for overlapping-speech visualisation"
```

---

### Task 3: TranscriptDelegate — overlap badge and tinted left border

**Files:**
- Modify: `src/ui/transcript_widget.py` (add overlap badge rendering to paint/sizeHint)

- [ ] **Step 1: Read the existing delegate code to verify anchoring**

Read: `src/ui/transcript_widget.py` lines 55-227 (paint + sizeHint)

- [ ] **Step 2: Add overlap colour constants after the CONFIDENCE_STYLES dict (line 53)**

```python
    # Overlap indicator style (same colour family as OverlapStripWidget)
    OVERLAP_COLOR = QColor("#c8925e")
    OVERLAP_BADGE_BG = QColor("#3d3020")
    OVERLAP_BADGE_TEXT = "⟪ overlap ⟫"
```

- [ ] **Step 3: Add overlap flag reading early in `paint()`, after the `style = ...` line (~line 69)**

```python
        is_overlap = bool(index.data(index.model().OverlapRole))
```

- [ ] **Step 4: Add overlap left-border drawing after the confidence border block (after line 94)**

```python
        # Overlap left-border indicator (drawn on top of confidence border)
        if is_overlap:
            painter.fillRect(
                option.rect.left(),
                option.rect.top(),
                4,
                option.rect.height(),
                self.OVERLAP_COLOR,
            )
```

- [ ] **Step 5: Draw the overlap pill badge after the confidence badge (after the `painter.setPen(QColor(border_color))` line ~138)**

```python
        # Overlap badge (next to confidence badge or timestamp)
        if is_overlap:
            overlap_badge_x = badge_x + badge_width + 10 if badge_text else rect.left() + time_rect.width() + 10
            overlap_font = QFont(font)
            overlap_font.setPointSize(8)
            overlap_font.setBold(True)
            painter.setFont(overlap_font)
            overlap_fm = QFontMetrics(overlap_font)
            overlap_badge_width = overlap_fm.horizontalAdvance(self.OVERLAP_BADGE_TEXT) + 12
            overlap_badge_height = overlap_fm.height() + 4
            overlap_badge_rect = QRect(
                overlap_badge_x,
                rect.top() + (time_rect.height() - overlap_badge_height) // 2,
                overlap_badge_width,
                overlap_badge_height,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.OVERLAP_BADGE_BG)
            painter.drawRoundedRect(overlap_badge_rect, 4, 4)
            painter.setPen(self.OVERLAP_COLOR)
            painter.drawText(overlap_badge_rect, Qt.AlignmentFlag.AlignCenter, self.OVERLAP_BADGE_TEXT)
```

Reset the font to the text font after drawing badges (after the overlap badge block):

```python
        font.setBold(False)
        font.setPointSize(12)
        painter.setFont(font)
        painter.setPen(text_color)
```

Note: The existing code at ~line 142 does `font.setBold(False)` etc. already. The overlap badge code runs before that, so the font is still the badge font. Ensure the existing text-drawing font reset runs after the overlap badge block.

- [ ] **Step 6: Update `sizeHint()` to account for the overlap badge width**

In `sizeHint()`, after the confidence badge width calculation (around line 227), add overlap badge width:

```python
        if is_overlap:
            badge_width += overlap_fm.horizontalAdvance("⟪ overlap ⟫") + 12 + 10
```

This requires reading `is_overlap` from the model index in `sizeHint()` as well. Add after the `words` / `translation` reads in sizeHint:

```python
        is_overlap = bool(index.data(index.model().OverlapRole))
```

The `badge_width` variable needs to be hoisted: currently it's computed inside the `if badge_text:` block. Change the structure so `badge_width` is always available. The cleanest approach: compute `badge_width = 0` before the confidence badge block, add to it inside, then add overlap badge width after.

- [ ] **Step 7: Commit**

```bash
git add src/ui/transcript_widget.py
git commit -m "feat: add overlap indicator pill and tinted left border to transcript delegate"
```

---

### Task 4: MainWindow — layout insertion and click wiring

**Files:**
- Modify: `src/ui/main_window.py`

- [ ] **Step 1: Read `src/ui/main_window.py` lines 102-203 (editor page layout) and 418-453 (transcription finished + overlap handler)**

- [ ] **Step 2: Import OverlapStripWidget at top of file**

Add after existing waveform import (line 20):
```python
from .overlap_strip import OverlapStripWidget
```

- [ ] **Step 3: Create OverlapStripWidget instance and insert into layout**

After line 162 (`self.waveform = WaveformWidget()`), add:
```python
        self.overlap_strip = OverlapStripWidget()
```

Insert it into the layout between waveform and transcript by adding it to the `tc_layout` after the waveform's layout row. In the existing layout at line 188 (`tc_layout.addLayout(control_layout)`) — the waveform is in `control_layout`. The strip goes between the control layout and the transcript view.

After line 188 (`tc_layout.addLayout(control_layout)`), add:
```python
        tc_layout.addWidget(self.overlap_strip)
```

- [ ] **Step 4: Wire the click handler**

After waveform seek connection (line 163), add:
```python
        self.overlap_strip.overlap_clicked.connect(self._on_overlap_strip_clicked)
```

- [ ] **Step 5: Add the click handler method to MainWindow**

Add after `_on_overlap_detected` (after line ~452):
```python
    def _on_overlap_strip_clicked(self, ms: int):
        """Scroll the transcript to the segment containing the clicked time."""
        idx = self.transcript_model.segment_index_at(ms)
        if idx is not None:
            self.transcript_view.scrollTo(
                idx, QAbstractItemView.ScrollHint.PositionAtCenter
            )
            self.transcript_view.setCurrentIndex(idx)
```

- [ ] **Step 6: Import QAbstractItemView at top of file**

Add to existing PyQt6.QtWidgets imports (or use from QtCore):
```python
from PyQt6.QtWidgets import QAbstractItemView
```

- [ ] **Step 7: Update `_on_transcription_finished()` to populate the strip**

After line 430 (`self.waveform.load_audio(...)`), add:
```python
        # Populate overlap strip
        transcript = self.transcript_model.get_transcript()
        self.overlap_strip.set_regions(
            transcript.overlap_regions,
            transcript.metadata.duration_ms,
        )
```

- [ ] **Step 8: Commit**

```bash
git add src/ui/main_window.py
git commit -m "feat: wire OverlapStripWidget into MainWindow layout with click-to-scroll"
```

---

### Task 5: OHMS export — inline `[overlap]` VTT annotation (last)

**Files:**
- Modify: `src/lore_core/ohms_exporter.py`
- Modify: `tests/test_ohms_export.py`

- [ ] **Step 1: Read the existing VTT generation loop in `ohms_exporter.py` (lines 124-134)**

- [ ] **Step 2: Add overlap check inside the VTT loop**

In the main VTT loop (line 125), after setting `vtt_content += ...`, add:
```python
            # Check for overlap annotation
            for region in transcript.overlap_regions:
                if region.start_ms < seg.end_ms and region.end_ms > seg.start_ms:
                    seg_text += " [overlap]"
                    break
```

The full loop becomes:
```python
        vtt_content = "WEBVTT\n\n"
        for seg in transcript.segments:
            start = _format_vtt_time(seg.start_ms / 1000.0)
            end = _format_vtt_time(seg.end_ms / 1000.0)
            seg_text = seg.text
            # Append [overlap] if segment intersects any OverlapRegion
            for region in transcript.overlap_regions:
                if region.start_ms < seg.end_ms and region.end_ms > seg.start_ms:
                    seg_text += " [overlap]"
                    break
            vtt_content += f"{start} --> {end}\n"
            if seg.speaker_label:
                vtt_content += f"<v {seg.speaker_label}>{seg_text}\n\n"
            else:
                vtt_content += f"{seg_text}\n\n"
```

Apply the same pattern to the `vtt_alt` (translation) loop if present (lines 139-151), appending `[overlap]` to the translation text.

- [ ] **Step 3: Add overlap test case to `test_ohms_export.py`**

Add a new test to the existing test file:

```python
def test_export_with_overlap(tmp_path):
    """Overlapping segments get [overlap] annotation in VTT cues."""
    from models.transcript import Transcript, Segment, OverlapRegion
    from lxml import etree

    transcript = Transcript(
        segments=[
            Segment(start_ms=0, end_ms=1000, text="no overlap here"),
            Segment(start_ms=2000, end_ms=3000, text="overlap here"),
            Segment(start_ms=4000, end_ms=5000, text="also clean"),
        ],
        overlap_regions=[
            OverlapRegion(start_ms=2000, end_ms=3000, confidence=0.9),
        ],
    )
    output = tmp_path / "test_overlap.xml"
    OhmsExporter.export(transcript, {}, output)

    tree = etree.parse(str(output))
    ns = {"o": OhmsExporter.OHMS_NS}
    vtt = tree.find(".//o:vtt_transcript", ns)
    assert vtt is not None
    vtt_text = vtt.text

    # Clean segments unchanged
    assert "no overlap here" in vtt_text
    assert "no overlap here [overlap]" not in vtt_text
    # Overlap segment has annotation
    assert "overlap here [overlap]" in vtt_text
    # Other clean segment
    assert "also clean" in vtt_text
    assert "also clean [overlap]" not in vtt_text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/mark/Projects/lore && python -m pytest tests/test_ohms_export.py -v --no-header 2>&1`
Expected: Existing tests pass + new overlap test passes

- [ ] **Step 5: Commit**

```bash
git add src/lore_core/ohms_exporter.py tests/test_ohms_export.py
git commit -m "feat: add inline [overlap] annotation to OHMS VTT export"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] A — per-segment overlap badge (Task 3)
- [x] C — overlap region strip (Task 2)
- [x] VTT — inline `[overlap]` annotation (Task 5)
- [x] Model — OverlapRole, _segment_overlaps(), segment_index_at() (Task 1)
- [x] MainWindow — layout, wiring, click handler (Task 4)
- [x] Overlap role slot +8 audit (Task 1 step 3)
- [x] Colour reference `#c8925e` / `#3d3020` used consistently (Tasks 2, 3)
- [x] `segment_index_at()` public method, no private attr access (Task 1 step 5, Task 4 step 5)
- [x] VTT strategy 2 — inline `[overlap]` (Task 5)

**Placeholder scan:** All steps contain complete code, exact paths, and expected test output. No TBD/TODO.

**Type consistency:** `_segment_overlaps()` returns `bool` everywhere. `OverlapRole` returns `True`/`False`. `segment_index_at()` returns `QModelIndex | None`. All consistent.

**Scope check:** Single focused feature (Phase 2 overlap UI). Five small tasks that can be done independently.
