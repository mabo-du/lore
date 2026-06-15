"""
ONNX-based overlap detector using pyannote-segmentation-3.0.

The segmentation model (5.99 MB ONNX) analyses audio at the frame level
(16 ms frames, 5 s sliding window, 500 ms step) and outputs multi-class
probabilities. When multiple speaker nodes exceed the confidence threshold
simultaneously, that frame region is classified as overlapping speech.

Using ONNX Runtime instead of PyTorch reduces the memory footprint from
~1.5 GiB to under 200 MB for the full diarisation pipeline. This module
only loads the segmentation component (~6 MB) for overlap flagging.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from models.transcript import OverlapRegion
from utils.model_manager import ModelManager


class OverlapDetector:
    """
    Lightweight ONNX-based detector that flags overlapping speech regions.

    This is the MVP for Phase 1 of the overlap strategy. It detects *when*
    overlap occurs without resolving *who* is speaking — that is Phase 3.
    """

    # The segmentation model uses a 5 s sliding window with 500 ms step
    WINDOW_SIZE_S = 5.0
    WINDOW_STEP_S = 0.5
    FRAME_LENGTH_S = 0.016  # 16 ms per frame
    # Minimum duration for a flagged overlap region (avoid spurious flags)
    MIN_OVERLAP_DURATION_S = 0.5

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Confidence threshold for considering a frame as overlapping.
                       0.5 means >50% probability of multiple active speakers.
        """
        self.threshold = threshold
        self._session = None
        self._input_name = None

    def _ensure_model(self) -> Path:
        """Download the segmentation ONNX model if not already cached."""
        return Path(ModelManager.ensure_model("Segmentation"))

    def _load_session(self):
        """Lazy-load the ONNX Runtime session."""
        if self._session is not None:
            return

        import onnxruntime as ort

        model_dir = self._ensure_model()
        onnx_path = model_dir / "onnx" / "model.onnx"
        if not onnx_path.exists():
            # Try alternative locations (HF hub layout varies)
            onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Could not find segmentation ONNX model at {model_dir}"
            )

        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, audio_path: Path) -> List[OverlapRegion]:
        """
        Run overlap detection on an audio file.

        Args:
            audio_path: Path to a 16 kHz mono WAV file.

        Returns:
            List of OverlapRegion entries sorted by start_ms.
        """
        self._load_session()

        # Read audio
        import wave

        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            if sample_rate != 16000:
                raise ValueError(f"Overlap detector requires 16000 Hz, got {sample_rate} Hz")
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)

        audio_np = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        total_duration_s = len(audio_np) / 16000.0
        window_samples = int(self.WINDOW_SIZE_S * 16000)
        step_samples = int(self.WINDOW_STEP_S * 16000)
        frames_per_window = int(self.WINDOW_SIZE_S / self.FRAME_LENGTH_S)

        # Accumulator for frame-level overlap scores across all windows
        # We track how many windows each frame participates in and the
        # cumulative overlap confidence.
        n_frames_total = int(total_duration_s / self.FRAME_LENGTH_S) + 1
        overlap_accum = np.zeros(n_frames_total, dtype=np.float32)
        window_count = np.zeros(n_frames_total, dtype=np.int32)

        # Slide over the audio
        for start_sample in range(0, len(audio_np) - window_samples + 1, step_samples):
            window = audio_np[start_sample : start_sample + window_samples]

            # Run inference — output shape depends on model architecture
            # pyannote-segmentation-3.0 outputs (batch, frames, speakers)
            # where speakers dimension has Kmax=3 nodes
            outputs = self._session.run(None, {self._input_name: [window]})
            logits = outputs[0]  # Shape: (1, n_frames, n_speakers)

            # Convert logits to probabilities via softmax per frame
            probs = self._softmax(logits[0])  # Shape: (n_frames, n_speakers)

            # A frame is "overlapping" if >1 speaker has probability > threshold
            # We use a simpler heuristic: max - 2nd_max > threshold means one dominant
            # speaker. If the top two are close, it's likely overlap.
            for frame_idx in range(probs.shape[0]):
                sorted_probs = np.sort(probs[frame_idx])[::-1]
                # Overlap score: how close the top two speakers are
                # If they're nearly equal (>0.3 each), that's overlap
                overlap_score = sorted_probs[1]  # 2nd speaker probability

                global_frame = start_sample // step_samples * (window_samples // step_samples) + frame_idx
                if global_frame < n_frames_total:
                    overlap_accum[global_frame] += overlap_score
                    window_count[global_frame] += 1

        # Average overlap scores
        mask = window_count > 0
        overlap_accum[mask] /= window_count[mask]

        # Threshold to binary overlap regions
        is_overlap = overlap_accum > self.threshold

        # Merge contiguous overlap frames into regions
        regions = []
        in_overlap = False
        region_start = 0

        for frame_idx in range(len(is_overlap)):
            if is_overlap[frame_idx] and not in_overlap:
                in_overlap = True
                region_start = frame_idx
            elif not is_overlap[frame_idx] and in_overlap:
                in_overlap = False
                duration_s = (frame_idx - region_start) * self.FRAME_LENGTH_S
                if duration_s >= self.MIN_OVERLAP_DURATION_S:
                    regions.append(OverlapRegion(
                        start_ms=int(region_start * self.FRAME_LENGTH_S * 1000),
                        end_ms=int(frame_idx * self.FRAME_LENGTH_S * 1000),
                        confidence=float(np.mean(overlap_accum[region_start:frame_idx])),
                    ))

        # Handle overlap at end of audio
        if in_overlap:
            duration_s = (len(is_overlap) - region_start) * self.FRAME_LENGTH_S
            if duration_s >= self.MIN_OVERLAP_DURATION_S:
                regions.append(OverlapRegion(
                    start_ms=int(region_start * self.FRAME_LENGTH_S * 1000),
                    end_ms=int(len(is_overlap) * self.FRAME_LENGTH_S * 1000),
                    confidence=float(np.mean(overlap_accum[region_start:])),
                ))

        return regions

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
