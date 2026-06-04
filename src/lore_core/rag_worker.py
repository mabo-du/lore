import time
from PyQt6.QtCore import QThread, pyqtSignal
from models.transcript import Transcript, Segment, TagData
from lore_core.taxonomy_manager import TaxonomyManager
import logging

logger = logging.getLogger(__name__)

class RAGWorker(QThread):
    """
    Runs the domain auto-tagging sequentially on a background thread.
    Takes a Transcript object, iterates over segments, and assigns taxonomy tags.
    Executes AFTER transcription/diarization to prevent CPU scheduler thrashing.
    """
    status_changed = pyqtSignal(str)
    segment_tagged = pyqtSignal(Segment)
    finished = pyqtSignal(Transcript)
    error = pyqtSignal(str)

    def __init__(self, transcript: Transcript, taxonomy_db_path: str = None, threshold: float = 0.5, k: int = 3, parent=None):
        super().__init__(parent)
        self.transcript = transcript
        self.taxonomy_db_path = taxonomy_db_path
        self.threshold = threshold
        self.k = k

    def run(self):
        try:
            self.status_changed.emit("Initializing domain taxonomy manager...")
            # Instantiate TaxonomyManager which runs with intra_op_num_threads=1 internally
            tm = TaxonomyManager(db_path=self.taxonomy_db_path)
            
            self.status_changed.emit("Auto-tagging segments...")
            total = len(self.transcript.segments)
            
            for i, segment in enumerate(self.transcript.segments):
                # Only tag actual speech segments, skip SYSTEM messages (like silence gaps)
                if segment.speaker_label == "SYSTEM":
                    continue
                
                # Skip very short segments or empty text to save computation
                if not segment.text or len(segment.text.strip()) < 5:
                    continue
                
                # Query the taxonomy manager.
                # FastEmbed executes synchronously here on this background thread.
                # The model is loaded lazily on the first query.
                results = tm.query(segment.text, k=self.k, threshold=self.threshold)
                
                if results:
                    for r in results:
                        tag = TagData(
                            preferred_term=r["preferred_term"],
                            definition=r["definition"],
                            distance=r["distance"],
                            segment_start_ms=segment.start_ms
                        )
                        segment.tags.append(tag)
                
                # Yield slightly to keep the UI thread buttery smooth
                time.sleep(0.001)
                
                # Emit update for UI so tags can appear live if they're watching this segment
                self.segment_tagged.emit(segment)
                
                if i % 10 == 0:
                    self.status_changed.emit(f"Tagging segments ({i}/{total})...")
            
            tm.close()
            self.status_changed.emit("Auto-tagging complete.")
            self.finished.emit(self.transcript)
            
        except Exception as e:
            logger.exception("Error in RAGWorker")
            self.error.emit(str(e))
