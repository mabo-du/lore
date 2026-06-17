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
