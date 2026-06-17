"""
Speaker embedding extraction via WeSpeaker ResNet34-LM ONNX.
Replaces the torch-dependent Resemblyzer VoiceEncoder.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import onnxruntime


class SpeakerEmbedding:
    """Extract speaker embeddings from audio using WeSpeaker ONNX model."""

    def __init__(self, model_path: Optional[Path] = None):
        self._session = None
        if model_path:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract a speaker embedding vector from an audio segment.

        Args:
            audio: float32 array normalized to [-1, 1], shape (samples,) or (1, samples)
            sr: sample rate (must be 16000)

        Returns:
            embedding: float32 array of shape (embedding_dim,)
        """
        if self._session is None:
            raise RuntimeError("Embedding model not loaded")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: audio})
        # WeSpeaker returns embeddings as the first output
        embedding = outputs[0]
        # Squeeze batch dimension if present
        return embedding.squeeze()
