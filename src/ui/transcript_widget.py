from PyQt6.QtWidgets import (
    QListView,
    QStyledItemDelegate,
    QTextEdit,
    QStyleOptionViewItem,
    QStyle,
)
from PyQt6.QtGui import QPainter, QColor, QFontMetrics, QFont, QTextDocument
from PyQt6.QtCore import Qt, QSize, QModelIndex, QRect
import datetime


class TranscriptDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.margins = 10
        self.spacing = 5

    def _format_time(self, ms: int) -> str:
        seconds = ms // 1000
        td = datetime.timedelta(seconds=seconds)
        # return HH:MM:SS format
        return str(td)

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

        # Adjust rect for content (with extra left margin if bordered)
        left_offset = 8 if border_color else 0
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
        badge_text = style["badge"]
        if badge_text:
            badge_font = QFont(font)
            badge_font.setPointSize(8)
            badge_font.setBold(True)
            painter.setFont(badge_font)
            badge_fm = QFontMetrics(badge_font)
            badge_width = badge_fm.horizontalAdvance(badge_text) + 12
            badge_height = badge_fm.height() + 4
            badge_x = rect.left() + time_rect.width() + 10
            badge_y = rect.top() + (time_rect.height() - badge_height) // 2

            # Badge background
            badge_rect = QRect(badge_x, badge_y, badge_width, badge_height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(style["badge_bg"]))
            painter.drawRoundedRect(badge_rect, 4, 4)

            # Badge text
            painter.setPen(QColor(border_color))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

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
        text_alpha = text_color.alpha()

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

        font = option.font
        font.setBold(True)
        font.setPointSize(10)
        fm_time = QFontMetrics(font)
        time_height = fm_time.height()

        font.setBold(False)
        font.setPointSize(12)

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
        doc.setTextWidth(option.rect.width() - 2 * self.margins)

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
