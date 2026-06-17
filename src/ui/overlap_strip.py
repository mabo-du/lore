"""Overlap strip widget — a thin horizontal bar showing overlapping speech regions."""

from typing import Optional

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QMouseEvent, QPaintEvent
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QEvent

from models.transcript import OverlapRegion


OVERLAP_COLOR = QColor("#c8925e")


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

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))

        for i, region in enumerate(self.overlap_regions):
            rect = self._rect_for_region(region)
            color = QColor(OVERLAP_COLOR)
            color.setAlpha(160 if i == self._hovered_index else 100)
            painter.fillRect(rect, color)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        idx = self._region_index_at_x(int(event.position().x()))
        if idx != self._hovered_index:
            self._hovered_index = idx
            self.setCursor(Qt.CursorShape.PointingHandCursor if idx is not None else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, event: QEvent) -> None:
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
