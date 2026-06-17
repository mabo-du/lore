"""Tests for TranscriptListModel overlap helper."""

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

    def test_segment_ends_at_region_start(self):
        """Segment end equals region start (adjacent, not overlapping) → False."""
        model = _make_model(
            segments=[Segment(start_ms=1000, end_ms=2000, text="adj")],
            regions=[OverlapRegion(start_ms=2000, end_ms=3000, confidence=0.8)],
        )
        assert model._segment_overlaps(1000, 2000) is False

    def test_region_ends_at_segment_start(self):
        """Region end equals segment start (adjacent, not overlapping) → False."""
        model = _make_model(
            segments=[Segment(start_ms=2000, end_ms=3000, text="adj")],
            regions=[OverlapRegion(start_ms=1000, end_ms=2000, confidence=0.8)],
        )
        assert model._segment_overlaps(2000, 3000) is False


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

    def test_returns_none_at_end_boundary(self):
        """ms equal to segment end_ms falls outside half-open interval → None."""
        seg = Segment(start_ms=1000, end_ms=2000, text="boundary")
        model = _make_model(segments=[seg])
        idx = model.segment_index_at(2000)
        assert idx is None


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
