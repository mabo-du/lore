"""
Smoke test for the ONNX-based OverlapDetector.

Prerequisites (run once):
    python -c "from utils.model_manager import ModelManager; ModelManager.ensure_model('Segmentation')"

This test requires:
    - onnxruntime (pip install onnxruntime)
    - The segmentation model downloaded (see above)
    - A real audio file — synthetic WAV is generated in the fixture

If the model isn't cached, ModelManager.ensure_model() will download it
(~5.99 MB) on first run.
"""

import struct
import numpy as np
import pytest
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_sine_wav(path: Path, duration_s: float, freq_hz: float,
                   amplitude: float = 0.3, sample_rate: int = 16000):
    """Write a mono sine wave to path as 16-bit PCM WAV."""
    import wave
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    samples = (amplitude * np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def silent_wav(tmp_path) -> Path:
    """3 seconds of pure silence (all zeros)."""
    p = tmp_path / "silent.wav"
    import wave
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * (16000 * 3))
    return p


@pytest.fixture
def overlap_wav(tmp_path) -> Path:
    """
    6-second WAV with a known overlap region:

        0.0s – 2.0s:  440 Hz tone (Speaker A)
        2.0s – 4.0s:  440 Hz + 660 Hz (Speaker A + Speaker B — OVERLAP)
        4.0s – 6.0s:  660 Hz tone (Speaker B)
    """
    p = tmp_path / "overlap.wav"
    sr = 16000
    t1 = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    t2 = np.linspace(2.0, 4.0, int(sr * 2.0), endpoint=False)
    t3 = np.linspace(4.0, 6.0, int(sr * 2.0), endpoint=False)

    seg1 = (0.3 * np.sin(2 * np.pi * 440 * t1) * 32767).astype(np.int16)
    # Overlap: both tones at reduced amplitude to avoid clipping
    seg2 = (0.2 * np.sin(2 * np.pi * 440 * t2) + 0.2 * np.sin(2 * np.pi * 660 * t2))
    seg2 = (seg2 * 32767).astype(np.int16)
    seg3 = (0.3 * np.sin(2 * np.pi * 660 * t3) * 32767).astype(np.int16)

    samples = np.concatenate([seg1, seg2, seg3])

    import wave
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return p


# ── Tests ────────────────────────────────────────────────────────────────

class TestOverlapDetectorSmoke:
    """Smoke tests that require the ONNX model to be cached locally."""

    def test_model_loads_and_validates(self):
        """The hard validation block in _load_session must pass."""
        from lore_core.overlap_detector import OverlapDetector
        d = OverlapDetector()
        d._load_session()
        # If we got here without RuntimeError("expected 7 output classes"), pass
        assert d._session is not None

    def test_silent_audio_no_overlap(self, silent_wav):
        """
        Silent audio should produce zero overlap regions.
        (non-speech class dominates all frames)
        """
        from lore_core.overlap_detector import OverlapDetector
        d = OverlapDetector(threshold=0.5)
        regions = d.detect(silent_wav)
        assert len(regions) == 0, (
            f"Silent audio produced {len(regions)} overlap regions: {regions}"
        )

    def test_overlap_segment_detected(self, overlap_wav):
        """
        The overlap segment (2.0s-4.0s) must produce at least one
        OverlapRegion whose time bounds intersect the overlap window.
        """
        from lore_core.overlap_detector import OverlapDetector
        d = OverlapDetector(threshold=0.3)  # Lower threshold for synthetic tones
        regions = d.detect(overlap_wav)

        assert len(regions) > 0, (
            f"Expected at least 1 overlap region in overlapping audio, got 0"
        )

        # At least one region should overlap with the 2.0s-4.0s window
        overlap_start_ms = 2000
        overlap_end_ms = 4000
        found = False
        for r in regions:
            if r.start_ms < overlap_end_ms and r.end_ms > overlap_start_ms:
                found = True
                break
        assert found, (
            f"None of the {len(regions)} regions intersect the 2.0-4.0s overlap window:\n"
            + "\n".join(f"  {r.start_ms}ms-{r.end_ms}ms (conf={r.confidence:.2f})" for r in regions)
        )

    def test_overlap_region_duration_at_least_minimum(self, overlap_wav):
        """
        The detected overlap should exceed MIN_OVERLAP_DURATION_S (0.5s).
        This confirms the region merging threshold isn't filtering out
        real overlap.
        """
        from lore_core.overlap_detector import OverlapDetector
        d = OverlapDetector(threshold=0.3)
        regions = d.detect(overlap_wav)

        # At least one region should be >500ms
        long_enough = [r for r in regions if (r.end_ms - r.start_ms) >= 500]
        assert len(long_enough) > 0, (
            f"No region exceeds 500ms minimum. Durations: "
            + ", ".join(f"{r.end_ms - r.start_ms}ms" for r in regions)
        )
