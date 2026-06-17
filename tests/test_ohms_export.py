"""Tests for OHMS XML export with VTT overlap annotation."""


from lxml import etree


def test_export_with_overlap(tmp_path):
    """Overlapping segments get [overlap] annotation in VTT cues."""
    from models.transcript import Transcript, Segment, OverlapRegion
    from lore_core.ohms_exporter import OhmsExporter

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
