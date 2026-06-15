from PyQt6.QtWidgets import (
    QWidget,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QTextEdit,
    QVBoxLayout,
    QScrollArea,
    QPushButton,
)
from PyQt6.QtCore import pyqtSignal


class MetadataWidget(QWidget):
    """
    Form for editing OHMS XML metadata.
    """

    export_requested = pyqtSignal(dict)  # Emits the collected metadata dictionary
    bagit_export_requested = pyqtSignal(dict)
    generate_abstract_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px;
                color: white;
            }
            QLabel {
                font-weight: bold;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #0098ff;
            }
        """)

        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        form = QFormLayout(container)
        form.setSpacing(10)

        # Fields mapped to OHMS XML required/common fields
        self.fields = {
            "title": QLineEdit(),
            "repository": QLineEdit(),
            "interviewee": QLineEdit(),
            "interviewer": QLineEdit(),
            "format": QComboBox(),
            "media_url": QLineEdit(),
            "rights": QComboBox(),
            "record_id": QLineEdit(),
            "abstract": QTextEdit(),
        }
        self.fields["abstract"].setMinimumHeight(100)

        # Populate combo boxes
        self.fields["format"].addItems(
            ["audio/mp3", "audio/wav", "audio/m4a", "audio/ogg", "audio/flac"]
        )
        self.fields["rights"].setEditable(True)
        self.fields["rights"].addItems([
            "CC BY 4.0",
            "CC BY-SA 4.0",
            "CC BY-NC 4.0",
            "CC BY-NC-SA 4.0",
            "Public Domain (CC0)",
            "Copyrighted - All Rights Reserved",
            "Restricted Access",
        ])

        form.addRow("Interview Title:", self.fields["title"])
        form.addRow("Repository Name:", self.fields["repository"])
        form.addRow("Interviewee:", self.fields["interviewee"])
        form.addRow("Interviewer:", self.fields["interviewer"])
        form.addRow("Media Format:", self.fields["format"])
        form.addRow("Media URL (Online):", self.fields["media_url"])
        form.addRow("Rights/License:", self.fields["rights"])
        form.addRow("Record ID (CMS Ref):", self.fields["record_id"])
        form.addRow("Abstract:", self.fields["abstract"])

        self.generate_btn = QPushButton("✨ Auto-Generate Abstract")
        self.generate_btn.setStyleSheet("background-color: #6a1b9a;")
        self.generate_btn.clicked.connect(self.generate_abstract_requested.emit)
        form.addRow("", self.generate_btn)

        scroll.setWidget(container)
        layout.addWidget(scroll)

        self.export_btn = QPushButton("Export OHMS XML")
        self.export_btn.clicked.connect(self._on_export)
        layout.addWidget(self.export_btn)

        self.bagit_btn = QPushButton("📦 Export BagIt Package")
        self.bagit_btn.setStyleSheet("background-color: #2e7d32;")
        self.bagit_btn.clicked.connect(self._on_bagit_export)
        layout.addWidget(self.bagit_btn)

    def _get_metadata_dict(self):
        return {
            "title": self.fields["title"].text().strip(),
            "repository": self.fields["repository"].text().strip(),
            "interviewee": self.fields["interviewee"].text().strip(),
            "interviewer": self.fields["interviewer"].text().strip(),
            "format": self.fields["format"].currentText(),
            "media_url": self.fields["media_url"].text().strip(),
            "rights": self.fields["rights"].currentText(),
            "record_id": self.fields["record_id"].text().strip(),
            "abstract": self.fields["abstract"].toPlainText().strip(),
        }

    def _on_export(self):
        self.export_requested.emit(self._get_metadata_dict())

    def _on_bagit_export(self):
        self.bagit_export_requested.emit(self._get_metadata_dict())

    def set_abstract(self, text: str):
        self.fields["abstract"].setPlainText(text)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("✨ Auto-Generate Abstract")

    def set_generating_state(self, message: str = "Generating..."):
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText(message)
