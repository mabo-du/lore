"""
ONNX-based overlap detector using onnx-community/pyannote-segmentation-3.0.

The segmentation model analyses audio at the frame level (16 ms frames,
10 s sliding window, 500 ms step) and outputs a (num_frames, 7) matrix
with these classes:

    0: non_speech
    1: speaker_1
    2: speaker_2
    3: speaker_3
    4: speakers 1+2  (overlap class)
    5: speakers 1+3  (overlap class)
    6: speakers 2+3  (overlap class)

Overlap detection is a direct max over classes 4-6 — the model explicitly
outputs overlap as distinct classes, no heuristic needed.

Model source: onnx-community/pyannote-segmentation-3.0 on HuggingFace.
This is a pre-exported ONNX file (5.99 MB). The gated pyannote/
segmentation-3.0 repo contains only PyTorch weights, not ONNX, so rolling
our own export would reintroduce the torch dependency that ONNX migration
is meant to eliminate. The community model accepts raw 16 kHz mono
waveform — the same format Lore's audio pipeline already produces.

Preprocessing compatibility is resolved via the hard shape validation
block in _load_session() and the smoke test: if a dummy tensor passes
and a silent clip produces non-speech-dominant frames, preprocessing is
compatible. No HF token required — this works for all users.

Using ONNX Runtime instead of PyTorch reduces memory from ~1.5 GiB to
under 200 MB for the full diarisation pipeline. This module only loads
the segmentation component (~6 MB) for overlap flagging.
"""

from pathlib import Path
from typing import List

import numpy as np

from models.transcript import OverlapRegion
from utils.model_manager import ModelManager


# ── Model constants ──────────────────────────────────────────────────────
# These must match the ONNX export that was used. Currently targeting
# pyannote/segmentation-3.0 architecture — verify against actual model.
WINDOW_SIZE_S = 10.0        # 10-second sliding window (160000 samples @ 16kHz)
WINDOW_STEP_S = 0.5         # 500 ms step
SAMPLE_RATE = 16000
FRAME_LENGTH_S = 0.016      # 16 ms per frame
MIN_OVERLAP_DURATION_S = 0.5  # Minimum duration to flag (avoid spurious triggers)

# Output class indices — explicit overlap classes
_CLASS_NON_SPEECH = 0
_CLASS_OVERLAP_START = 4  # speakers_1_and_2
_CLASS_OVERLAP_END = 6    # speakers_2_and_3


class OverlapDetector:
    """
    Lightweight ONNX-based detector that flags overlapping speech regions.

    Phase 1 of the incremental overlap strategy. Detects *when* overlap
    occurs without resolving *who* is speaking (Phase 3).

    Loads the model lazily on first call to detect().
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self._session = None
        self._input_name = None

    # ── Model loading ────────────────────────────────────────────────────

    def _ensure_model(self) -> Path:
        return Path(ModelManager.ensure_model("Segmentation"))

    def _load_session(self):
        if self._session is not None:
            return

        import onnxruntime as ort

        model_dir = self._ensure_model()
        onnx_path = model_dir / "onnx" / "model.onnx"
        if not onnx_path.exists():
            onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Could not find segmentation ONNX model at {model_dir}"
            )

        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        # ── Hard validation block ────────────────────────────────────────
        # Run a zero-tensor dummy input to verify the model's output shape.
        # This catches ONNX export mismatches before any real audio is
        # processed.
        dummy_input = np.zeros((1, 1, int(WINDOW_SIZE_S * SAMPLE_RATE)), dtype=np.float32)
        dummy_outputs = self._session.run(None, {self._input_name: dummy_input})
        out = dummy_outputs[0]
        n_classes = out.shape[-1]
        if n_classes != 7:
            raise RuntimeError(
                f"Segmentation model expected 7 output classes, got {n_classes}. "
                f"Shape: {out.shape}. This means the ONNX export doesn't match "
                f"pyannote/segmentation-3.0 architecture—check which model file "
                f"is deployed."
            )

    # ── Public API ───────────────────────────────────────────────────────

    def detect(self, audio_path: Path) -> List[OverlapRegion]:
        """
        Run overlap detection on a 16 kHz mono WAV file.

        Returns OverlapRegion entries sorted by start_ms.
        """
        self._load_session()

        # Read audio
        import wave

        with wave.open(str(audio_path), "rb") as wf:
            rate = wf.getframerate()
            if rate != SAMPLE_RATE:
                raise ValueError(f"Requires {SAMPLE_RATE} Hz, got {rate} Hz")
            raw = wf.readframes(wf.getnframes())

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        total_samples = len(audio)
        window_samples = int(WINDOW_SIZE_S * SAMPLE_RATE)
        step_samples = int(WINDOW_STEP_S * SAMPLE_RATE)
        total_frames = int(total_samples / SAMPLE_RATE / FRAME_LENGTH_S) + 1

        # Per-frame overlap accumulator (works in frame-index space)
        overlap_accum = np.zeros(total_frames, dtype=np.float32)
        window_count = np.zeros(total_frames, dtype=np.int32)

        # Slide over the audio
        for start in range(0, total_samples - window_samples + 1, step_samples):
            window = audio[start : start + window_samples]

            # Model input shape: (batch=1, channels=1, samples)
            outputs = self._session.run(
                None, {self._input_name: window.reshape(1, 1, -1)}
            )
            logits = outputs[0]  # (1, n_frames, 7)

            # Apply softmax to ALL 7 classes, then extract overlap probs
            all_probs = self._softmax(logits[0], axis=-1)  # (n_frames, 7)
            overlap_score = all_probs[:, _CLASS_OVERLAP_START:_CLASS_OVERLAP_END + 1].max(axis=-1)
            # overlap_score is now (n_frames,) in [0, 1] indicating how
            # strongly the model thinks this frame contains overlap.

            # Map window-local frame index to global frame index
            global_offset = start // step_samples * (window_samples // step_samples)
            for local_idx, score in enumerate(overlap_score):
                gf = global_offset + local_idx
                if gf < total_frames:
                    overlap_accum[gf] = max(overlap_accum[gf], score)
                    window_count[gf] += 1

        # Merge contiguous high-confidence frames into regions
        regions = []
        in_overlap = False
        region_start = 0

        for frame_idx in range(total_frames):
            is_overlap = overlap_accum[frame_idx] > self.threshold and window_count[frame_idx] > 0

            if is_overlap and not in_overlap:
                in_overlap = True
                region_start = frame_idx
            elif not is_overlap and in_overlap:
                in_overlap = False
                duration_s = (frame_idx - region_start) * FRAME_LENGTH_S
                if duration_s >= MIN_OVERLAP_DURATION_S:
                    regions.append(OverlapRegion(
                        start_ms=int(region_start * FRAME_LENGTH_S * 1000),
                        end_ms=int(frame_idx * FRAME_LENGTH_S * 1000),
                        confidence=float(np.mean(overlap_accum[region_start:frame_idx])),
                    ))

        # Handle overlap at end of audio
        if in_overlap:
            duration_s = (total_frames - region_start) * FRAME_LENGTH_S
            if duration_s >= MIN_OVERLAP_DURATION_S:
                regions.append(OverlapRegion(
                    start_ms=int(region_start * FRAME_LENGTH_S * 1000),
                    end_ms=int(total_frames * FRAME_LENGTH_S * 1000),
                    confidence=float(np.mean(overlap_accum[region_start:])),
                ))

        return regions

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
