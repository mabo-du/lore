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
        Process full audio with a sliding window and return speech regions.

        Returns list of (start_sample, end_sample) for each contiguous
        speech segment, with silence gaps >= 500ms used as boundaries.
        """
        if self._session is None:
            raise RuntimeError("VAD model not loaded")

        # Ensure float32 mono
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1) if audio.shape[1] < audio.shape[0] else audio[0]

        input_name = self._session.get_inputs()[0].name
        speech_frames = []

        # Slide over audio in HOP_SIZE increments
        for start in range(0, len(audio) - FRAME_SIZE, HOP_SIZE):
            chunk = audio[start:start + FRAME_SIZE]
            # Silero VAD expects (batch, samples) or (batch, 1, samples)
            chunk_in = chunk[np.newaxis, :]
            outputs = self._session.run(None, {input_name: chunk_in})
            prob = float(outputs[0][0][0])
            speech_frames.append(prob > 0.5)

        # Convert frame-level decisions to sample-level regions
        regions = []
        in_speech = False
        region_start = 0

        min_speech_samples = int(0.1 * sr)  # 100ms minimum speech
        min_silence_samples = int(0.5 * sr)  # 500ms silence gap = new region

        for i, is_speech in enumerate(speech_frames):
            sample_pos = i * HOP_SIZE

            if is_speech and not in_speech:
                in_speech = True
                region_start = sample_pos
            elif not is_speech and in_speech:
                # Check if silence is long enough to split regions
                silence_start = sample_pos  # noqa: F841
                # Look ahead to find end of silence
                j = i
                while j < len(speech_frames) and not speech_frames[j]:
                    j += 1
                silence_samples = (j - i) * HOP_SIZE

                if silence_samples >= min_silence_samples:
                    region_end = sample_pos
                    if region_end - region_start >= min_speech_samples:
                        regions.append((region_start, region_end))
                    in_speech = False

        # Handle trailing speech
        if in_speech:
            region_end = len(audio)
            if region_end - region_start >= min_speech_samples:
                regions.append((region_start, region_end))

        # If no speech regions detected, return full audio as one region
        if not regions:
            regions.append((0, len(audio)))

        return regions
