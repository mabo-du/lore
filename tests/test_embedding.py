"""Tests for SpeakerEmbedding — downloads and runs real WeSpeaker ONNX."""

import pytest
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download

from lore_core.embedding import SpeakerEmbedding, SAMPLE_RATE


@pytest.fixture(scope="session")
def embedding_model_path() -> Path:
    """Download the real WeSpeaker ONNX model once per test session."""
    return Path(
        hf_hub_download(
            repo_id="onnx-community/wespeaker-voxceleb-resnet34-LM",
            filename="onnx/model.onnx",
            repo_type="model",
        )
    )


@pytest.fixture
def embedder(embedding_model_path) -> SpeakerEmbedding:
    return SpeakerEmbedding(embedding_model_path)


class TestSpeakerEmbeddingReal:
    def test_loads_model(self, embedding_model_path):
        """Loading the real model should not raise."""
        emb = SpeakerEmbedding(embedding_model_path)
        assert emb._session is not None

    def test_silence_produces_embedding(self, embedder):
        """Even silence should produce a 256-dim embedding vector."""
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second
        result = embedder.extract(silence)
        assert isinstance(result, np.ndarray)
        assert result.shape == (256,)
        assert result.dtype == np.float32

    def test_noise_produces_different_embedding(self, embedder):
        """Different audio should produce different embeddings."""
        noise1 = (np.random.rand(SAMPLE_RATE).astype(np.float32) - 0.5) * 0.5
        noise2 = (np.random.rand(SAMPLE_RATE).astype(np.float32) - 0.5) * 0.5
        emb1 = embedder.extract(noise1)
        emb2 = embedder.extract(noise2)
        # Different inputs should give different embeddings
        diff = np.linalg.norm(emb1 - emb2)
        assert diff > 0.01, f"Expected different embeddings, diff={diff}"

    def test_short_audio_produces_embedding(self, embedder):
        """Very short audio (100ms) should still produce an embedding."""
        short = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
        result = embedder.extract(short)
        assert result.shape == (256,)
