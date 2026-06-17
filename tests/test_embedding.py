"""Tests for SpeakerEmbedding (WeSpeaker ONNX)."""

from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path

from lore_core.embedding import SpeakerEmbedding


def test_load_model():
    with patch("onnxruntime.InferenceSession") as mock_session:
        emb = SpeakerEmbedding()
        emb.load(Path("/mock/model.onnx"))
        mock_session.assert_called_once()


def test_extract_without_model():
    emb = SpeakerEmbedding()
    try:
        emb.extract(np.zeros(16000, dtype=np.float32))
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "not loaded" in str(e)


def test_extract_returns_vector():
    emb = SpeakerEmbedding()
    emb._session = MagicMock()
    emb._session.get_inputs.return_value = [MagicMock(name="input")]
    emb._session.run.return_value = [np.random.randn(256).astype(np.float32)]

    result = emb.extract(np.zeros(16000, dtype=np.float32))
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
