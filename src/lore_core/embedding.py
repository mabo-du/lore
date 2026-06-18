"""
Speaker embedding extraction via WeSpeaker ResNet34-LM ONNX.
Replaces the torch-dependent Resemblyzer VoiceEncoder.

Model: onnx-community/wespeaker-voxceleb-resnet34-LM (onnx/model.onnx)
License: CC-BY-4.0 / Apache-2.0

Verified against actual model weights:
  Input:  input_features ['B', 'T', 80] float32 — 80-band mel spectrogram frames
  Output: last_hidden_state ['B', 256] float32 — 256-dim speaker embedding

Preprocessing: 80 mel filterbanks, 25ms window, 10ms hop, 16kHz sample rate.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import onnxruntime
import librosa


# WeSpeaker preprocessing constants
SAMPLE_RATE = 16000
N_MELS = 80
WINDOW_LENGTH = 400  # 25ms @ 16kHz = 400 samples
HOP_LENGTH = 160     # 10ms @ 16kHz = 160 samples


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

    def _mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute 80-band log-mel spectrogram matching WeSpeaker preprocessing.

        Args:
            audio: float32 array normalized to [-1, 1], shape (samples,)
            sr: sample rate

        Returns:
            mel: float32 array of shape (T, 80) where T = number of frames
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=N_MELS,
            n_fft=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window="hann",
            center=True,
            power=2.0,
        )
        # Convert to log scale (dB)
        log_mel = librosa.amplitude_to_db(mel_spec, ref=np.max, top_db=None)
        # Normalize to [-1, 1] range (typical WeSpeaker preprocessing)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        # Transpose to (T, 80) — model expects [B, T, 80]
        return log_mel.T.astype(np.float32)

    def extract(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """
        Extract a speaker embedding vector from an audio segment.

        Args:
            audio: float32 array normalized to [-1, 1], shape (samples,) or (1, samples)
            sr: sample rate (must be 16000)

        Returns:
            embedding: float32 array of shape (256,)
        """
        if self._session is None:
            raise RuntimeError("Embedding model not loaded")

        # Ensure mono float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1) if audio.shape[1] < audio.shape[0] else audio[0]

        # Compute mel spectrogram features
        features = self._mel_spectrogram(audio, sr)  # (T, 80)

        # Add batch dimension: model expects (B, T, 80)
        features = features[np.newaxis, :, :]  # (1, T, 80)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: features})
        # Output shape: (1, 256)
        return outputs[0][0]  # squeeze batch dim
