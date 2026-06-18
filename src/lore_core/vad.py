"""
Silero VAD (Voice Activity Detection) via ONNX Runtime (streaming variant).

Model: onnx-community/silero-vad (onnx/model.onnx)
License: MIT

The streaming variant maintains internal recurrent state across calls,
allowing frame-by-frame processing without re-encoding the full audio
history. Input shape: (batch, samples). Requires sr (sample rate)
and state tensors.

Verified against the actual model weights:
  Inputs:  input [None, None] float32,  state [2, None, 128] float32,
           sr scalar int64
  Outputs: output [batch, 1] float32,  stateN [2, ...] float32
"""

from pathlib import Path
from typing import Optional
import numpy as np
import onnxruntime


SAMPLE_RATE = 16000
FRAME_SIZE = 512  # 32ms @ 16kHz
HOP_SIZE = 160    # 10ms @ 16kHz


def _new_state(batch: int = 1) -> np.ndarray:
    """Create zero-filled initial hidden state for the streaming VAD."""
    return np.zeros((2, batch, 128), dtype=np.float32)


class SileroVAD:
    """Voice Activity Detection using Silero VAD ONNX model (streaming)."""

    def __init__(self, model_path: Optional[Path] = None):
        self._session = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._state: Optional[np.ndarray] = None
        if model_path:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        """Load the Silero VAD ONNX model."""
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        self.reset_state()

    def reset_state(self) -> None:
        """Reset the internal recurrent state for a new audio stream."""
        self._state = _new_state()

    def is_speech(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> bool:
        """
        Returns True if the audio chunk contains speech.

        Args:
            audio: float32 array normalized to [-1, 1], shape (samples,) or (1, samples)
            sr: sample rate (must be 16000)

        Maintains internal recurrent state across calls.
        """
        if self._session is None:
            raise RuntimeError("VAD model not loaded")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # (1, samples)

        # Build input dict matching model's expected names
        feed = {
            self._input_names[0]: audio,    # input
            self._input_names[1]: self._state,  # state
            self._input_names[2]: np.array(sr, dtype=np.int64),  # sr
        }
        outputs = self._session.run(self._output_names, feed)
        self._state = outputs[1]  # stateN for next call
        return float(outputs[0][0][0]) > 0.5

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

        # Reset state for new stream
        self.reset_state()

        speech_frames = []

        # Slide over audio in HOP_SIZE increments
        for start in range(0, len(audio) - FRAME_SIZE, HOP_SIZE):
            chunk = audio[start:start + FRAME_SIZE]
            prob = self.is_speech(chunk, sr)
            speech_frames.append(prob)

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
