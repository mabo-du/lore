"""Tests for Silero VAD ONNX integration."""

from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np

from lore_core.vad import SileroVAD


def test_load_model():
    """Loading a valid model path should not raise."""
    with patch("onnxruntime.InferenceSession") as mock_session:
        vad = SileroVAD()
        vad.load(Path("/mock/model.onnx"))
        mock_session.assert_called_once()


def test_is_speech_without_model():
    """Calling is_speech without loading raises."""
    vad = SileroVAD()
    try:
        vad.is_speech(np.zeros(16000, dtype=np.float32))
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "not loaded" in str(e)


def test_is_speech_returns_bool():
    """is_speech should return a boolean."""
    vad = SileroVAD()
    vad._session = MagicMock()
    vad._session.get_inputs.return_value = [MagicMock(name="input")]
    vad._session.run.return_value = [np.array([[0.8]])]

    result = vad.is_speech(np.zeros(16000, dtype=np.float32))
    assert isinstance(result, bool)
    assert result is True


def test_is_speech_below_threshold():
    """Probability below 0.5 returns False."""
    vad = SileroVAD()
    vad._session = MagicMock()
    vad._session.get_inputs.return_value = [MagicMock(name="input")]
    vad._session.run.return_value = [np.array([[0.3]])]

    result = vad.is_speech(np.zeros(16000, dtype=np.float32))
    assert result is False
