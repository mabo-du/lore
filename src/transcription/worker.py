from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from utils.model_manager import ModelManager
from lore_core.engine import TranscriptionEngine
from models.transcript import Segment


class TranscriptionWorker(QThread):
    """
    Runs the model download and transcription in a background thread.
    Emits signals for UI updates.
    """

    # Signals
    status_changed = pyqtSignal(str)  # E.g. "Downloading model...", "Transcribing..."
    progress_changed = pyqtSignal(float)  # 0.0 to 1.0 — transcription progress
    segment_completed = pyqtSignal(Segment)  # Emitted as each segment is yielded
    diarization_completed = pyqtSignal(
        list
    )  # Emits updated list of segments with speakers
    overlap_detected = pyqtSignal(list)  # Emits list[OverlapRegion]
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        audio_path: Path,
        quality_tier: str = "Best Quality",
        enable_diarization: bool = False,
        custom_vocab: str = "",
        num_speakers: int = 2,
        enable_overlap_detection: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.audio_path = audio_path
        self.quality_tier = quality_tier
        self.enable_diarization = enable_diarization
        self.custom_vocab = custom_vocab
        self.num_speakers = num_speakers
        self.enable_overlap_detection = enable_overlap_detection

    def run(self):
        try:
            self.status_changed.emit(
                "Ensuring model is downloaded (this may take a while)..."
            )
            # Blocking call, but we are in a QThread
            model_path_str = ModelManager.ensure_model(self.quality_tier)
            model_path = Path(model_path_str)

            self.status_changed.emit("Loading model into memory...")
            engine = TranscriptionEngine(model_path)

            self.status_changed.emit("Transcribing...")

            all_segments = []
            last_end_ms = 0
            audio_duration_s = getattr(engine, 'audio_duration_s', None)

            # transcribe() yields segments generator
            for segment in engine.transcribe(
                self.audio_path, initial_prompt=self.custom_vocab
            ):
                # Phase 9: Audio Event Detection (Silence gaps)
                gap_ms = segment.start_ms - last_end_ms
                if gap_ms >= 3000:  # 3 seconds
                    silence_seg = Segment(
                        start_ms=last_end_ms,
                        end_ms=segment.start_ms,
                        text=f"[Silence: {gap_ms / 1000.0:.1f}s]",
                        speaker_label="SYSTEM",
                    )
                    all_segments.append(silence_seg)
                    self.segment_completed.emit(silence_seg)

                all_segments.append(segment)
                self.segment_completed.emit(segment)
                last_end_ms = segment.end_ms

                # Emit progress 0.0–1.0 based on how far through the audio we are
                if audio_duration_s and audio_duration_s > 0:
                    progress = min(segment.end_ms / 1000.0 / audio_duration_s, 1.0)
                    self.progress_changed.emit(progress)

            if self.enable_diarization:
                self.status_changed.emit("Running Speaker Diarization...")
                from lore_core.diarization import DiarizationEngine

                diarizer = DiarizationEngine(num_speakers=self.num_speakers)
                d_results = diarizer.run_diarization(self.audio_path, segments=all_segments)

                self.status_changed.emit("Aligning speakers...")
                diarizer.align_speakers_to_segments(all_segments, d_results)
                self.diarization_completed.emit(all_segments)

            # Phase 3: Overlap Detection (post-diarization, via lightweight ONNX model)
            if self.enable_overlap_detection:
                self.status_changed.emit("Detecting overlapping speech...")
                try:
                    from lore_core.overlap_detector import OverlapDetector

                    detector = OverlapDetector()
                    overlap_regions = detector.detect(self.audio_path)
                    if overlap_regions:
                        self.overlap_detected.emit(overlap_regions)
                except Exception as e:
                    # Overlap detection is optional — log and continue
                    import logging
                    logging.getLogger(__name__).warning(
                        "Overlap detection failed (non-fatal): %s", e
                    )

            self.status_changed.emit("Transcription complete.")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
