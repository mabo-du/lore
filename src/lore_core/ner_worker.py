import queue
from typing import List
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from gliner2_onnx import GLiNER2ONNXRuntime

from models.transcript import Segment, EntityData
from utils.model_manager import ModelManager

class NERWorker(QThread):
    """
    Consumes transcribed segments and runs GLiNER ONNX for Named Entity Recognition.
    Executes in a separate thread to prevent blocking the GUI or the TranscriptionWorker.
    """
    # Signal emitted when a segment has been processed and entities found
    entities_detected = pyqtSignal(list) # List[EntityData]
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = queue.Queue()
        self._is_running = True
        self._model = None
        self._labels = ["person", "organization", "location", "historical event", "indigenous group", "cultural concept"]

    @pyqtSlot(Segment)
    def enqueue_segment(self, segment: Segment):
        """Called asynchronously via Qt.ConnectionType.QueuedConnection to add a segment."""
        self._queue.put(segment)

    def stop(self):
        self._is_running = False
        self._queue.put(None) # Sentinel to unblock get()

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
                    break # Stop signal received
                
                # Perform NER
                raw_entities = self._model.extract_entities(segment.text, self._labels)
                
                entities = []
                for ent in raw_entities:
                    # Entity format: {'start': int, 'end': int, 'text': str, 'label': str, 'score': float}
                    entity_data = EntityData(
                        start_char=ent['start'],
                        end_char=ent['end'],
                        text=ent['text'],
                        label=ent['label'],
                        score=ent['score'],
                        segment_start_ms=segment.start_ms
                    )
                    entities.append(entity_data)
                    
                if entities:
                    self.entities_detected.emit(entities)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"NER Error: {str(e)}")
