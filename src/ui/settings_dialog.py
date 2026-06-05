from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QLineEdit,
    QFormLayout,
    QGroupBox,
    QMessageBox,
)
from PyQt6.QtCore import QSettings


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)

        self.settings = QSettings("HeritageTools", "Lore")

        layout = QVBoxLayout(self)

        # Transcription Group
        transcription_group = QGroupBox("Transcription Settings")
        transcription_layout = QVBoxLayout(transcription_group)
        self.vocab_input = QLineEdit()
        self.vocab_input.setPlaceholderText("e.g., Antigravity, O'Connor, NumPy")
        t_form_layout = QFormLayout()
        t_form_layout.addRow("Custom Vocabulary:", self.vocab_input)
        transcription_layout.addLayout(t_form_layout)
        t_desc = QLabel(
            "Comma-separated list of terms (proper nouns, local places) to inject "
            "into Whisper to improve spelling accuracy."
        )
        t_desc.setWordWrap(True)
        t_desc.setStyleSheet("color: gray; font-size: 11px;")
        transcription_layout.addWidget(t_desc)
        layout.addWidget(transcription_group)

        # Diarization Group
        diarization_group = QGroupBox("Speaker Diarization Engine")
        diarization_layout = QVBoxLayout(diarization_group)

        self.engine_group = QButtonGroup(self)

        self.radio_resemblyzer = QRadioButton("Resemblyzer (Fast, No Token Required)")
        self.radio_pyannote = QRadioButton(
            "Pyannote 3.1 (High Accuracy, Requires Token)"
        )

        self.engine_group.addButton(self.radio_resemblyzer, 0)
        self.engine_group.addButton(self.radio_pyannote, 1)

        diarization_layout.addWidget(self.radio_resemblyzer)
        diarization_layout.addWidget(self.radio_pyannote)

        # HuggingFace Token
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.token_input.setPlaceholderText("hf_...")

        form_layout = QFormLayout()
        form_layout.addRow("HuggingFace Token:", self.token_input)
        diarization_layout.addLayout(form_layout)

        # Description
        desc = QLabel(
            "Note: Diarization attributes transcribed text to specific speakers (e.g., SPEAKER_00).\n"
            "Pyannote requires a HuggingFace account and acceptance of their model license."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: gray; font-size: 11px;")
        diarization_layout.addWidget(desc)

        layout.addWidget(diarization_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Save")
        btn_cancel = QPushButton("Cancel")
        btn_save.clicked.connect(self.save_settings)
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        self.load_settings()

        self.radio_pyannote.toggled.connect(self._toggle_token_input)
        self._toggle_token_input()

    def _toggle_token_input(self):
        self.token_input.setEnabled(self.radio_pyannote.isChecked())

    def load_settings(self):
        use_pyannote = self.settings.value("diarization/use_pyannote", False, type=bool)
        if use_pyannote:
            self.radio_pyannote.setChecked(True)
        else:
            self.radio_resemblyzer.setChecked(True)

        token = self.settings.value("diarization/hf_token", "")
        self.token_input.setText(token)

        vocab = self.settings.value("transcription/custom_vocab", "")
        self.vocab_input.setText(vocab)

    def save_settings(self):
        if self.radio_pyannote.isChecked() and not self.token_input.text().strip():
            QMessageBox.warning(
                self,
                "Token Required",
                "Please enter a HuggingFace token to use Pyannote.",
            )
            return

        self.settings.setValue(
            "diarization/use_pyannote", self.radio_pyannote.isChecked()
        )
        self.settings.setValue("diarization/hf_token", self.token_input.text().strip())
        self.settings.setValue(
            "transcription/custom_vocab", self.vocab_input.text().strip()
        )
        self.accept()
