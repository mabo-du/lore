from PyQt6.QtWidgets import (
    QListView,
    QMenu,
    QStyledItemDelegate,
    QTextEdit,
    QStyleOptionViewItem,
    QStyle,
)
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QFontMetrics,
    QFont,
    QTextDocument,
)
from PyQt6.QtCore import Qt, QSize, QModelIndex, QRect, pyqtSignal


class TranscriptDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.margins = 10
        self.spacing = 5

    TIME_FORMAT_LONG = "long"    # HH:MM:SS
    TIME_FORMAT_SHORT = "short"  # MM:SS.mmm

    def _format_time(self, ms: int) -> str:
        seconds = ms / 1000.0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        # Read format preference from settings
        fmt = getattr(self, '_time_format', self.TIME_FORMAT_LONG)
        if fmt == self.TIME_FORMAT_SHORT:
            return f"{minutes:02d}:{secs:05.2f}"
        # Default: HH:MM:SS
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"

    # Confidence level → visual styling map
    CONFIDENCE_STYLES = {
        "high": {"border": None, "badge": None, "badge_bg": None, "text_alpha": 255},
        "medium": {
            "border": "#b8860b",
            "badge": "⚠ Review",
            "badge_bg": "#3d3520",
            "text_alpha": 255,
        },
        "low": {
            "border": "#cc7a00",
            "badge": "⚠ Low Confidence",
            "badge_bg": "#3d2e10",
            "text_alpha": 255,
        },
        "hallucination": {
            "border": "#cc3333",
            "badge": "🔴 Hallucination?",
            "badge_bg": "#3d1515",
            "text_alpha": 200,
        },
        "non_speech": {
            "border": "#555555",
            "badge": "🔇 Non-Speech",
            "badge_bg": "#2a2a2a",
            "text_alpha": 120,
        },
        "unknown": {"border": None, "badge": None, "badge_bg": None, "text_alpha": 255},
    }

    # Overlap indicator style (same colour family as OverlapStripWidget)
    OVERLAP_COLOR = QColor("#c8925e")
    OVERLAP_BADGE_BG = QColor("#3d3020")
    OVERLAP_BADGE_TEXT = "⟪ overlap ⟫"

    def _draw_badge(self, painter, font, text, text_color, bg_color,
                     left_anchor_x, top_anchor_y, time_header_height):
        """Draw a rounded pill badge returning the x-anchor for the next badge."""
        badge_font = QFont(font)
        badge_font.setPointSize(8)
        badge_font.setBold(True)
        painter.setFont(badge_font)
        fm = QFontMetrics(badge_font)
        badge_width = fm.horizontalAdvance(text) + 12
        badge_height = fm.height() + 4
        badge_rect = QRect(
            left_anchor_x,
            top_anchor_y + (time_header_height - badge_height) // 2,
            badge_width,
            badge_height,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg_color)
        painter.drawRoundedRect(badge_rect, 4, 4)
        painter.setPen(text_color)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, text)
        return left_anchor_x + badge_width + 10

    def paint(
        self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex
    ):
        painter.save()

        # Get data from model
        start_ms = index.data(index.model().StartMsRole)
        end_ms = index.data(index.model().EndMsRole)
        text = index.data(index.model().TextRole)
        translation = index.data(index.model().TranslationRole)
        speaker = index.data(index.model().SpeakerRole)
        confidence = index.data(index.model().ConfidenceLevelRole) or "unknown"
        words = index.data(index.model().WordsRole)

        style = self.CONFIDENCE_STYLES.get(
            confidence, self.CONFIDENCE_STYLES["unknown"]
        )
        is_overlap = bool(index.data(index.model().OverlapRole))

        time_str = f"[{self._format_time(start_ms)} - {self._format_time(end_ms)}]"
        if speaker:
            time_str += f" | {speaker}"

        # Background
        bg_color = (
            QColor("#2d2d2d")
            if bool(option.state & QStyle.StateFlag.State_Selected)
            else QColor("#1e1e1e")
        )
        painter.fillRect(option.rect, bg_color)

        # Left border indicator for flagged segments
        border_color = style["border"]
        if border_color:
            painter.fillRect(
                option.rect.left(),
                option.rect.top(),
                4,
                option.rect.height(),
                QColor(border_color),
            )

        # Overlap left-border indicator (drawn on top of confidence border)
        if is_overlap:
            painter.fillRect(
                option.rect.left(),
                option.rect.top(),
                4,
                option.rect.height(),
                self.OVERLAP_COLOR,
            )

        # Adjust rect for content (with extra left margin if bordered)
        left_offset = 8 if border_color or is_overlap else 0
        rect = option.rect.adjusted(
            self.margins + left_offset, self.margins, -self.margins, -self.margins
        )

        # Draw time/speaker header
        text_color = QColor("#cccccc")
        text_color.setAlpha(style["text_alpha"])
        time_color = QColor("#888888")
        time_color.setAlpha(style["text_alpha"])

        font = option.font
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(time_color)

        fm = QFontMetrics(font)
        time_rect = fm.boundingRect(rect, Qt.TextFlag.TextSingleLine, time_str)
        painter.drawText(rect.left(), rect.top() + fm.ascent(), time_str)

        # Draw confidence badge (pill) next to timestamp
        next_badge_x = rect.left() + time_rect.width() + 10  # anchor for next badge
        badge_text = style["badge"]
        if badge_text:
            next_badge_x = self._draw_badge(
                painter, font, badge_text,
                QColor(border_color), QColor(style["badge_bg"]),
                next_badge_x, rect.top(), time_rect.height(),
            )

        # Overlap badge (next to confidence badge or timestamp)
        if is_overlap:
            self._draw_badge(
                painter, font, self.OVERLAP_BADGE_TEXT,
                self.OVERLAP_COLOR, self.OVERLAP_BADGE_BG,
                next_badge_x, rect.top(), time_rect.height(),
            )

        # Draw Text
        font.setBold(False)
        font.setPointSize(12)
        painter.setFont(font)
        painter.setPen(text_color)

        text_y = rect.top() + time_rect.height() + self.spacing
        text_rect = QRect(
            rect.left(),
            text_y,
            rect.width(),
            rect.height() - time_rect.height() - self.spacing,
        )

        # Using QTextDocument for rich text/word wrapping.
        # QTextDocument ignores the painter's pen — must set color via HTML.
        doc = QTextDocument()
        doc.setDefaultFont(font)

        # Convert QColor to hex string for inline CSS
        text_hex = text_color.name()

        if words:
            html_words = []
            for w in words:
                if w.get("prob", 1.0) < 0.6:
                    html_words.append(
                        f"<span style='color: #ffaa99; text-decoration: underline;'>{w['word']}</span>"
                    )
                else:
                    html_words.append(
                        f"<span style='color: {text_hex};'>{w['word']}</span>"
                    )
            display_text = "".join(html_words)
        else:
            display_text = f"<span style='color: {text_hex};'>{text}</span>"

        if translation:
            display_text += f"<br><br><span style='color: #a0c0ff;'><b>Translation:</b> {translation}</span>"

        doc.setHtml(display_text)
        doc.setTextWidth(text_rect.width())

        painter.translate(text_rect.topLeft())
        doc.drawContents(painter)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        text = index.data(index.model().TextRole)
        translation = index.data(index.model().TranslationRole)
        words = index.data(index.model().WordsRole)
        is_overlap = bool(index.data(index.model().OverlapRole))

        font = option.font
        font.setBold(True)
        font.setPointSize(10)
        fm_time = QFontMetrics(font)
        time_height = fm_time.height()

        font.setBold(False)
        font.setPointSize(12)

        # Account for overlap badge width if present
        overlap_margin = 0
        if is_overlap:
            overlap_font = QFont(font)
            overlap_font.setPointSize(8)
            overlap_font.setBold(True)
            overlap_fm = QFontMetrics(overlap_font)
            # badge_width = horizontalAdvance(text) + 12 (paint padding) + 10 (gap to content)
            overlap_margin = overlap_fm.horizontalAdvance(self.OVERLAP_BADGE_TEXT) + 22

        doc = QTextDocument()
        doc.setDefaultFont(font)

        display_text = text
        if words:
            html_words = []
            for w in words:
                if w.get("prob", 1.0) < 0.6:
                    html_words.append(
                        f"<span style='color: #ffaa99; text-decoration: underline;'>{w['word']}</span>"
                    )
                else:
                    html_words.append(w["word"])
            display_text = "".join(html_words)

        if translation:
            display_text += f"<br><br><span style='color: #a0c0ff;'><b>Translation:</b> {translation}</span>"
        doc.setHtml(display_text)
        doc.setTextWidth(option.rect.width() - 2 * self.margins - overlap_margin)

        text_height = int(doc.size().height())

        total_height = self.margins * 2 + time_height + self.spacing + text_height
        return QSize(option.rect.width(), total_height)

    def createEditor(self, parent, option, index):
        editor = QTextEdit(parent)
        editor.setAcceptRichText(False)
        editor.setStyleSheet(
            "background-color: #2d2d2d; color: white; border: 1px solid #007acc;"
        )
        return editor

    def setEditorData(self, editor, index):
        text = index.data(index.model().TextRole)
        editor.setPlainText(text)

    def setModelData(self, editor, model, index):
        model.update_segment_text(index.row(), editor.toPlainText())


class TranscriptWidget(QListView):
    word_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(TranscriptDelegate(self))
        self.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(False)
        self.setStyleSheet("""
            QListView {
                background-color: #1e1e1e;
                border: none;
                outline: none;
            }
            QListView::item:selected {
                background-color: #2d2d2d;
            }
        """)

    def mousePressEvent(self, event):
        """Handle left-click on words to seek audio playback."""
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.indexAt(event.pos())
            if index.isValid():
                words = index.data(index.model().WordsRole)
                if words:
                    delegate = self.itemDelegate()
                    rect = self.visualRect(index)

                    font = self.font()
                    font.setBold(True)
                    font.setPointSize(10)
                    fm = QFontMetrics(font)
                    time_height = fm.height()

                    margins = delegate.margins
                    spacing = delegate.spacing

                    # Text area bounds (matching delegate paint())
                    text_area_left = rect.left() + margins
                    text_area_top = rect.top() + margins + time_height + spacing
                    text_area_width = rect.width() - 2 * margins
                    text_area_height = rect.height() - 2 * margins - time_height - spacing

                    text_rect = QRect(
                        text_area_left, text_area_top,
                        text_area_width, text_area_height,
                    )

                    if text_rect.contains(event.pos()):
                        # Build QTextDocument matching delegate's paint() layout
                        font.setBold(False)
                        font.setPointSize(12)
                        doc = QTextDocument()
                        doc.setDefaultFont(font)

                        text_hex = "#cccccc"
                        html_words = []
                        for w in words:
                            if w.get("prob", 1.0) < 0.6:
                                html_words.append(
                                    f"<span style='color: #ffaa99;'>{w['word']}</span>"
                                )
                            else:
                                html_words.append(
                                    f"<span style='color: {text_hex};'>{w['word']}</span>"
                                )
                        display_text = "".join(html_words)
                        doc.setHtml(display_text)
                        doc.setTextWidth(text_area_width)

                        # Map click position to character position in document
                        doc_pos = event.pos() - text_rect.topLeft()
                        hit_pos = doc.documentLayout().hitTest(
                            doc_pos, Qt.HitTestAccuracy.FuzzyHit
                        )

                        if hit_pos >= 0:
                            # Map character position to word index
                            plain_text = ""
                            word_boundaries = []
                            for w in words:
                                word_text = w["word"]
                                start = len(plain_text)
                                plain_text += word_text
                                end = len(plain_text) - 1
                                word_boundaries.append((start, end))

                            for i, (start, end) in enumerate(word_boundaries):
                                if start <= hit_pos <= end:
                                    word_start_ms = words[i].get("start")
                                    if word_start_ms is not None:
                                        self.word_clicked.emit(int(word_start_ms))
                                        return

        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        """Show right-click context menu for segment actions."""
        index = self.indexAt(event.pos())
        if not index.isValid():
            return

        menu = QMenu(self)

        # Rename speaker (if segment has one)
        speaker = index.data(index.model().SpeakerRole)
        if speaker:
            rename_action = menu.addAction(f"Rename Speaker ({speaker})")
            rename_action.triggered.connect(lambda: self._rename_speaker(index))

        # Copy text
        copy_action = menu.addAction("Copy Segment Text")
        copy_action.triggered.connect(lambda: self._copy_segment_text(index))

        # Merge with next (only if not last segment)
        if index.row() < index.model().rowCount() - 1:
            merge_action = menu.addAction("Merge with Next Segment")
            merge_action.triggered.connect(lambda: self._merge_with_next(index))

        menu.addSeparator()

        # Time format toggle (constants live on TranscriptDelegate)
        delegate = self.itemDelegate()
        current_fmt = getattr(delegate, '_time_format', delegate.TIME_FORMAT_LONG)
        fmt_label = "Use MM:SS.mmm Time Format" if current_fmt == delegate.TIME_FORMAT_LONG else "Use HH:MM:SS Time Format"
        fmt_action = menu.addAction(fmt_label)
        fmt_action.triggered.connect(self._toggle_time_format)

        menu.exec(event.globalPos())

    def _rename_speaker(self, index):
        """Rename all segments with the same speaker label."""
        from PyQt6.QtWidgets import QInputDialog

        old_speaker = index.data(index.model().SpeakerRole)
        if not old_speaker:
            return

        new_speaker, ok = QInputDialog.getText(
            self, "Rename Speaker",
            f"New name for {old_speaker}:",
            text=old_speaker
        )
        if not ok or not new_speaker.strip():
            return

        new_speaker = new_speaker.strip()
        model = index.model()
        transcript = model.get_transcript()

        # Update all segments with this speaker label
        for i, seg in enumerate(transcript.segments):
            if seg.speaker_label == old_speaker:
                model.update_segment_speaker(i, new_speaker)

    def _copy_segment_text(self, index):
        """Copy segment text to clipboard."""
        from PyQt6.QtWidgets import QApplication
        text = index.data(index.model().TextRole)
        QApplication.clipboard().setText(text)

    def _toggle_time_format(self):
        """Toggle between HH:MM:SS and MM:SS.mmm time display."""
        delegate = self.itemDelegate()
        current = getattr(delegate, '_time_format', delegate.TIME_FORMAT_LONG)
        delegate._time_format = delegate.TIME_FORMAT_SHORT if current == delegate.TIME_FORMAT_LONG else delegate.TIME_FORMAT_LONG
        # Repaint all items
        self.viewport().update()

    def _merge_with_next(self, index):
        """Merge the selected segment with the segment below it."""
        row = index.row()
        model = index.model()
        transcript = model.get_transcript()

        if row >= len(transcript.segments) - 1:
            return  # No next segment

        current = transcript.segments[row]
        next_seg = transcript.segments[row + 1]

        # Merge text and extend end time
        merged_text = f"{current.text} {next_seg.text}".strip()
        current.end_ms = next_seg.end_ms
        current.text = merged_text

        # Remove the next segment
        transcript.segments.pop(row + 1)

        # Notify the model of changes
        model.beginResetModel()
        model.endResetModel()
