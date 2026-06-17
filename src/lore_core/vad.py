"""
Silero VAD (Voice Activity Detection) via ONNX Runtime.

The official Silero VAD model ships pre-exported ONNX weights.
This module loads them directly through onnxruntime, avoiding
the PyTorch dependency of the official silero-vad PyPI package.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import onnxruntime


SAMPLE_RATE = 16000
FRAME_SIZE = 512  # 32ms @ 16kHz
HOP_SIZE = 160    # 10ms @ 16kHz


class SileroVAD:
    """Voice Activity Detection using Silero VAD ONNX model."""

    def __init__(self, model_path: Optional[Path] = None):
        self._session = None
        if model_path:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        """Load the Silero VAD ONNX model."""
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    def is_speech(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> bool:
        """Returns True if the audio chunk contains speech."""
        if self._session is None:
            raise RuntimeError("VAD model not loaded")
        # Silero VAD expects float32 input, normalized to [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        input_name = self._session.get_inputs()[0].name
        # Input shape: (batch, samples) or (batch, 1, samples)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        outputs = self._session.run(None, {input_name: audio})
        # Output is probability of speech
        prob = outputs[0]
        return float(prob[0][0]) > 0.5

    def detect_speech_regions(
        self, audio: np.ndarray, sr: int = SAMPLE_RATE
    ) -> list[tuple[int, int]]:
        """
        Process full audio and return list of (start_sample, end_sample)
        for each speech region.

        Implementation follows the standard Silero VAD sliding window approach.
        For now, return a stub — full implementation comes when the
        model is available for testing.
        """
        return [(0, len(audio))]
