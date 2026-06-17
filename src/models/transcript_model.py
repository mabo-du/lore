from PyQt6.QtCore import QAbstractListModel, Qt, QModelIndex, pyqtSignal
from models.transcript import Transcript, Segment

class TranscriptListModel(QAbstractListModel):
    """
    QAbstractListModel wrapper around a Transcript object.
    Provides efficient data to the QListView.
    """
    # Custom roles for the item delegate
    StartMsRole = Qt.ItemDataRole.UserRole + 1
    EndMsRole = Qt.ItemDataRole.UserRole + 2
    TextRole = Qt.ItemDataRole.UserRole + 3
    SpeakerRole = Qt.ItemDataRole.UserRole + 4
    ConfidenceLevelRole = Qt.ItemDataRole.UserRole + 5
    TranslationRole = Qt.ItemDataRole.UserRole + 6
    WordsRole = Qt.ItemDataRole.UserRole + 7
    OverlapRole = Qt.ItemDataRole.UserRole + 8  # slot +8 is free after WordsRole (+7)

    def __init__(self, transcript: Transcript = None, parent=None):
        super().__init__(parent)
        self._transcript = transcript if transcript is not None else Transcript()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._transcript.segments)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._transcript.segments)):
            return None

        segment = self._transcript.segments[index.row()]

        if role == self.StartMsRole:
            return segment.start_ms
        elif role == self.EndMsRole:
            return segment.end_ms
        elif role == self.TextRole:
            return segment.text
        elif role == self.SpeakerRole:
            return segment.speaker_label or ""
        elif role == self.ConfidenceLevelRole:
            return segment.confidence_level
        elif role == self.TranslationRole:
            return segment.translation
        elif role == self.WordsRole:
            return segment.words
        elif role == self.OverlapRole:
            return self._segment_overlaps(segment.start_ms, segment.end_ms)

        return None

    def add_segment(self, segment: Segment):
        """Insert a newly transcribed segment to the model in chronological order."""
        import bisect
        # Find correct index to maintain chronological order by start_ms
        starts = [s.start_ms for s in self._transcript.segments]
        idx = bisect.bisect_left(starts, segment.start_ms)
        
        self.beginInsertRows(QModelIndex(), idx, idx)
        self._transcript.segments.insert(idx, segment)
        self.endInsertRows()

    def clear_segments(self):
        self.beginResetModel()
        self._transcript.segments.clear()
        self.endResetModel()

    def update_segments(self, segments: list):
        """Replaces or updates all segments (e.g. after diarization completes)"""
        self.beginResetModel()
        self._transcript.segments = segments
        self.endResetModel()

    def update_segment_text(self, row: int, new_text: str):
        if 0 <= row < len(self._transcript.segments):
            self._transcript.segments[row].text = new_text
            idx = self.index(row, 0)
            self.dataChanged.emit(idx, idx, [self.TextRole])
            
    def update_segment_speaker(self, row: int, new_speaker: str):
        if 0 <= row < len(self._transcript.segments):
            self._transcript.segments[row].speaker_label = new_speaker
            idx = self.index(row, 0)
            self.dataChanged.emit(idx, idx, [self.SpeakerRole])
            
    def refresh_segment(self, segment: Segment):
        try:
            row = self._transcript.segments.index(segment)
            idx = self.index(row, 0)
            self.dataChanged.emit(idx, idx, [self.TranslationRole, self.TextRole])
        except ValueError:
            pass

    def get_transcript(self) -> Transcript:
        return self._transcript

    def add_entities(self, entities: list['EntityData']):
        """Adds entities to the corresponding segment based on start_ms."""
        import bisect
        if not entities: return
        starts = [s.start_ms for s in self._transcript.segments]
        idx = bisect.bisect_left(starts, entities[0].segment_start_ms)
        if idx < len(self._transcript.segments) and self._transcript.segments[idx].start_ms == entities[0].segment_start_ms:
            self._transcript.segments[idx].entities.extend(entities)
            model_idx = self.index(idx, 0)
            # Emit dataChanged to trigger a UI refresh for this segment if needed
            self.dataChanged.emit(model_idx, model_idx, [self.TextRole])
        
    def _segment_overlaps(self, seg_start_ms: int, seg_end_ms: int) -> bool:
        """Returns True if segment time range intersects any OverlapRegion."""
        for region in self._transcript.overlap_regions:
            lo = max(seg_start_ms, region.start_ms)
            hi = min(seg_end_ms, region.end_ms)
            if lo < hi:
                return True
        return False

    def segment_index_at(self, ms: int) -> QModelIndex | None:
        """Return the QModelIndex of the segment containing `ms`, or None."""
        for i, seg in enumerate(self._transcript.segments):
            if seg.start_ms <= ms < seg.end_ms:
                return self.index(i, 0)
        return None

    def clear(self):
        self.beginResetModel()
        self._transcript.segments.clear()
        self.endResetModel()
