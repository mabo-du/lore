from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings


class FilePickerWidget(QWidget):
    """
    Handles drag-and-drop and file dialog selection for audio files.
    """

    file_selected = pyqtSignal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 200)

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border: 2px dashed #444444;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("Drag and drop an audio file here\nor click to browse")
        self.label.setStyleSheet("color: #cccccc; font-size: 14px; border: none;")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.browse_btn = QPushButton("Browse Files")
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0098ff;
            }
        """)
        self.browse_btn.clicked.connect(self._open_file_dialog)
        self.browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        layout.addWidget(self.label)
        layout.addSpacing(16)

        self.chk_diarize = QCheckBox("Enable Speaker Diarization (Beta)")
        self.chk_diarize.setStyleSheet("color: #cccccc; border: none; font-size: 13px;")

        layout.addWidget(self.chk_diarize, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(16)
        layout.addWidget(self.browse_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Restore last state
        settings = QSettings("HeritageTools", "Lore")
        self.chk_diarize.setChecked(
            settings.value("diarization/enabled", False, type=bool)
        )

    def _open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.ogg *.flac);;All Files (*)",
        )
        if file_path:
            settings = QSettings("HeritageTools", "Lore")
            settings.setValue("diarization/enabled", self.chk_diarize.isChecked())
            self.file_selected.emit(Path(file_path))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = Path(urls[0].toLocalFile())
                if path.suffix.lower() in [".mp3", ".wav", ".m4a", ".ogg", ".flac"]:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            settings = QSettings("HeritageTools", "Lore")
            settings.setValue("diarization/enabled", self.chk_diarize.isChecked())
            self.file_selected.emit(path)
