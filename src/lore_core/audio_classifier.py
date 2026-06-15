from pathlib import Path
import wave
import numpy as np
import onnxruntime as ort
from PyQt6.QtCore import QThread, pyqtSignal

from models.transcript import Segment
from utils.model_manager import ModelManager


class AudioClassifyWorker(QThread):
    """
    Runs YAMNet Audio Classification in a background thread to detect Non-Verbal events.
    Emits events like [Laughter] and [Crying].
    """

    event_detected = pyqtSignal(Segment)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, audio_path: Path, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.target_indices = {}  # Populated dynamically in run()
        self.threshold = 0.35  # Confidence threshold

    def run(self):
        try:
            model_path_str = ModelManager.ensure_model("YAMNet")
            model_dir = Path(model_path_str)
            onnx_file = model_dir / "yamnet.onnx"

            # Build the target index map dynamically from yamnet_class_map.csv
            class_map_path = model_dir / "yamnet_class_map.csv"
            target_labels = {"Laughter", "Baby cry, infant cry"}
            self.target_indices = {}
            if class_map_path.exists():
                import csv
                with open(class_map_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        label = row.get("display_name", "")
                        if label in target_labels:
                            category = "Laughter" if "laugh" in label.lower() else "Crying"
                            self.target_indices[int(row["index"])] = category
            else:
                # Fallback hardcoded indices if class map is unavailable
                self.target_indices = {13: "Laughter", 14: "Laughter", 17: "Laughter", 19: "Crying", 20: "Crying"}

            # CPU only
            session = ort.InferenceSession(
                str(onnx_file), providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name

            # Read 16kHz Mono PCM Audio
            with wave.open(str(self.audio_path), "rb") as wf:
                framerate = wf.getframerate()
                if framerate != 16000:
                    raise ValueError(f"YAMNet requires 16000Hz, got {framerate}Hz")
                if wf.getnchannels() != 1:
                    raise ValueError("YAMNet requires Mono audio")

                n_frames = wf.getnframes()
                audio_data = wf.readframes(n_frames)

            # Convert to float32 [-1.0, 1.0]
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # YAMNet can process arbitrary length 1D array
            outputs = session.run(None, {input_name: audio_np})
            scores = outputs[0]  # Shape (N, 521)

            # Each frame in YAMNet is roughly 0.48s hop size
            hop_size_s = 0.48

            # Process frames and merge contiguous events of the same type
            current_event = None
            current_start_s = 0.0

            for frame_idx, frame_scores in enumerate(scores):
                frame_time_s = frame_idx * hop_size_s

                # Find the target class with the highest score above threshold
                best_class = None
                best_score = 0.0
                for idx, label in self.target_indices.items():
                    if (
                        frame_scores[idx] > self.threshold
                        and frame_scores[idx] > best_score
                    ):
                        best_score = frame_scores[idx]
                        best_class = label

                if best_class:
                    if current_event == best_class:
                        # Continue event
                        pass
                    else:
                        # Close previous event if exists
                        if current_event:
                            self._emit_event(
                                current_event, current_start_s, frame_time_s
                            )
                        # Start new event
                        current_event = best_class
                        current_start_s = frame_time_s
                else:
                    if current_event:
                        # Close previous event
                        self._emit_event(current_event, current_start_s, frame_time_s)
                        current_event = None

            # Close hanging event
            if current_event:
                self._emit_event(
                    current_event, current_start_s, (len(scores)) * hop_size_s
                )

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))

    def _emit_event(self, event_label: str, start_s: float, end_s: float):
        # Only emit events that last at least 0.5s to avoid spurious noise
        duration = end_s - start_s
        if duration >= 0.48:
            segment = Segment(
                start_ms=int(start_s * 1000),
                end_ms=int(end_s * 1000),
                text=f"[{event_label}]",
                speaker_label="SYSTEM",
            )
            self.event_detected.emit(segment)
