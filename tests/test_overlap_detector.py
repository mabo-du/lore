"""
Smoke test for the ONNX-based OverlapDetector.

Prerequisites (run once):
    python -c "from utils.model_manager import ModelManager; ModelManager.ensure_model('Segmentation')"

This test requires:
    - onnxruntime (installed with lore-ai)
    - The segmentation model downloaded (see above)
    - A source WAV of speech to create the overlap fixture

The overlap fixture uses real human speech (the pyannote sample WAV
shipped with pyannote-audio) and overlays two segments to create a
realistic 2-speaker overlap that the model was trained to recognise.
"""

import wave
import numpy as np
import pytest
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────

_SPEECH_WAV = Path(__file__).parent.parent / ".venv" / "lib" / "python3.13" / \
    "site-packages" / "pyannote" / "audio" / "sample" / "sample.wav"

_SAMPLE_WAV = Path(__file__).parent.parent / "sample.norm.wav"


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file, return (samples as float32 [-1,1], sample_rate)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def silent_wav(tmp_path) -> Path:
    """3 seconds of pure silence (all zeros)."""
    p = tmp_path / "silent.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * (16000 * 3))
    return p


@pytest.fixture
def overlap_wav(tmp_path) -> Path:
    """
    10-second WAV with two overlapping speech segments from different
    time regions of the same source (acoustically distinct pockets).

        0.0s – 5.0s:  Speech segment A (pyannote sample, beginning)
        2.0s – 7.0s:  Speech segment B (same source, later region — acoustically distinct)
        5.0s – 10.0s: Speech segment A alone

    The overlap window is 2.0s–5.0s. The two segments come from well-separated
    time regions of the source so the model treats them as distinct speakers.

    Falls back to sample.norm.wav if pyannote sample is unavailable.
    """
    p = tmp_path / "overlap.wav"

    source = _SPEECH_WAV if _SPEECH_WAV.exists() else _SAMPLE_WAV
    speech, sr = _read_wav(source)

    ws = int(sr * 10.0)  # 10-second window (model input size)
    seg_a = speech[:ws]  # First 10 seconds

    # Take segment B from a different time region of the same file
    # (the pyannote sample is 30s, so 12-22s is well-separated)
    seg_b_source = speech[min(int(sr * 12.0), len(speech) - ws):]
    seg_b_source = seg_b_source[:ws]
    if len(seg_b_source) < ws:
        seg_b_source = np.pad(seg_b_source, (0, ws - len(seg_b_source)))
    seg_b = seg_b_source

    # Silence seg_b except for a 3-second window that overlaps with seg_a
    silence = np.zeros(ws, dtype=np.float32)
    silence[int(sr * 2.0):int(sr * 5.0)] = 1.0
    seg_b = seg_b * silence * 0.6

    # Mix: keep seg_a at full, add seg_b at reduced gain
    mixed = seg_a + seg_b
    mixed = np.clip(mixed, -1.0, 1.0)
    samples = (mixed * 32767).astype(np.int16)

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
        d = OverlapDetector(threshold=0.3)
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
