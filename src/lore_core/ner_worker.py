import queue
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, QSettings
from gliner2_onnx import GLiNER2ONNXRuntime
import logging
import json
from datetime import datetime, timezone
from pathlib import Path

from models.transcript import Segment, EntityData
from utils.model_manager import ModelManager

logger = logging.getLogger(__name__)

# Stage 1 backchannel detection: rule-based lexicon + duration filter
# Whisper actively strips many disfluencies, so this catches only the
# subset that survives text normalisation. Stage 2 (acoustic classifier)
# will catch the rest.
#
# NOTE: Accumulating enough real-world examples to train Stage 2 requires
# either long-term personal usage or an opt-in data-sharing framework
# where willing users contribute anonymised, feature-extracted logs
# (never raw audio). Both approaches have open design questions that are
# out of scope for this implementation — this module collects the raw
# data, but the collection-and-sharing pipeline is a future concern.
_BACKCHANNEL_LEXICON = frozenset({
    "mhm", "uh-huh", "mm-hmm", "uh", "um", "yeah", "right", "okay",
    "sure", "yep", "yup", "nah", "nope", "ah", "oh", "huh", "aha",
})
_BACKCHANNEL_MAX_MS = 800

_BACKCHANNEL_LOG_PATH = Path.home() / ".local" / "share" / "heritage-tools" / "backchannel-log.jsonl"


def _is_rule_backchannel(segment: Segment) -> bool:
    """Rule-based backchannel check: short segment + filler-word text."""
    duration = segment.end_ms - segment.start_ms
    if duration > _BACKCHANNEL_MAX_MS:
        return False
    text = segment.text.strip().lower().rstrip(".,!?\"';:-")
    return text in _BACKCHANNEL_LEXICON


class NERWorker(QThread):
    """
    Consumes transcribed segments and runs GLiNER ONNX for Named Entity Recognition.
    Also performs Stage-1 rule-based backchannel detection inline (zero extra cost).

    Executes in a separate thread to prevent blocking the GUI or the TranscriptionWorker.
    """

    # Signal emitted when a segment has been processed and entities found
    entities_detected = pyqtSignal(list)  # List[EntityData]
    # Signal emitted when a segment is classified as a backchannel
    backchannel_detected = pyqtSignal(Segment)
    error = pyqtSignal(str)

    def __init__(self, parent=None, logging_enabled=True):
        super().__init__(parent)
        self._queue = queue.Queue()
        self._is_running = True
        self._model = None
        self._labels = [
            "person",
            "organization",
            "location",
            "historical event",
            "indigenous group",
            "cultural concept",
        ]
        # Logging enabled check: parameter can force-disable, otherwise
        # defer to the persistent QSettings toggle.
        self._logging_enabled = logging_enabled
        if self._logging_enabled:
            settings = QSettings("HeritageTools", "Lore")
            self._logging_enabled = settings.value(
                "backchannel/logging_enabled", True, type=bool
            )

    @staticmethod
    def _log_backchannel_decision(segment: Segment, is_backchannel: bool, source: str) -> None:
        """Append one JSONL entry recording a backchannel rule decision.

        Both hits and misses are logged — misses are as valuable as hits
        for training a future Stage 2 acoustic classifier.

        Log path: ``~/.local/share/heritage-tools/backchannel-log.jsonl``
        """
        if not _BACKCHANNEL_LOG_PATH.parent.exists():
            _BACKCHANNEL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "segment_start_ms": segment.start_ms,
            "segment_end_ms": segment.end_ms,
            "duration_ms": segment.duration_ms,
            "text": segment.text,
            "is_backchannel": is_backchannel,
            "backchannel_source": source if is_backchannel else "",
            "source": "ner_worker",
        }
        try:
            with open(_BACKCHANNEL_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("Failed to write backchannel log entry", exc_info=True)

    @pyqtSlot(Segment)
    def enqueue_segment(self, segment: Segment):
        """Called asynchronously via Qt.ConnectionType.QueuedConnection to add a segment."""
        self._queue.put(segment)

    def stop(self):
        self._is_running = False
        self._queue.put(None)  # Sentinel to unblock get()

    def run(self):
        try:
            # Lazy load the model in this thread, downloading if necessary
            model_path_str = ModelManager.ensure_model("NER")
            if not model_path_str:
                self.error.emit("Failed to download NER model.")
                return

            self._model = GLiNER2ONNXRuntime(model_path_str)

            while self._is_running:
                segment = self._queue.get()
                if segment is None:
                    break  # Stop signal received

                # Stage 1: Rule-based backchannel detection (zero-cost, before NER)
                is_bc = _is_rule_backchannel(segment)
                if is_bc:
                    segment.is_backchannel = True
                    segment.backchannel_source = "rule"
                    self.backchannel_detected.emit(segment)

                # Log every decision (hit and miss) for future Stage 2 training
                if self._logging_enabled:
                    NERWorker._log_backchannel_decision(segment, is_bc, "rule")

                if is_bc:
                    # Backchannels are filler words — skip NER (no entities to extract)
                    continue

                # Perform NER
                raw_entities = self._model.extract_entities(segment.text, self._labels)

                entities = []
                for ent in raw_entities:
                    # GLiNER2 returns Entity objects with attribute access
                    entity_data = EntityData(
                        start_char=ent.start,
                        end_char=ent.end,
                        text=ent.text,
                        label=ent.label,
                        score=ent.score,
                        segment_start_ms=segment.start_ms,
                    )
                    entities.append(entity_data)

                if entities:
                    self.entities_detected.emit(entities)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.error.emit(f"NER Error: {str(e)}")
