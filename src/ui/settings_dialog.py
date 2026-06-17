from PyQt6.QtWidgets import (
    QCheckBox,
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
    QComboBox,
    QSpinBox,
)
from PyQt6.QtCore import QSettings
from utils.model_manager import ModelManager
from utils.token_vault import encrypt_token, decrypt_token


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

        self.model_tier_combo = QComboBox()
        self.model_tier_combo.addItems(["Fast", "Balanced", "Best Quality"])
        t_form_layout.addRow("Model Quality Tier:", self.model_tier_combo)

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

        self.speaker_count_spin = QSpinBox()
        self.speaker_count_spin.setRange(1, 20)
        self.speaker_count_spin.setValue(2)
        self.speaker_count_spin.setToolTip(
            "Number of speakers expected (Resemblyzer only).\n"
            "1 = monologue, 2 = interview, 3+ = panel."
        )
        form_layout.addRow("Number of Speakers:", self.speaker_count_spin)

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

        # Offline / Pre-fetch Group
        offline_group = QGroupBox("Offline & Pre-fetch")
        offline_layout = QVBoxLayout(offline_group)

        self.offline_checkbox = QCheckBox(
            "Offline Mode (fail fast on missing models, no network checks)"
        )

        self.prefetch_btn = QPushButton("Download Models")
        self.prefetch_btn.clicked.connect(self._run_prefetch)
        self.prefetch_status = QLabel("")
        self.prefetch_status.setStyleSheet("color: gray; font-size: 11px;")

        offline_layout.addWidget(self.offline_checkbox)
        offline_layout.addWidget(self.prefetch_btn)
        offline_layout.addWidget(self.prefetch_status)

        # Backchannel data logging
        offline_layout.addSpacing(8)
        self.backchannel_log_checkbox = QCheckBox(
            "Backchannel data logging (local only)"
        )
        self.backchannel_log_checkbox.setChecked(True)
        offline_layout.addWidget(self.backchannel_log_checkbox)

        log_desc = QLabel(
            "Records rule-based backchannel decisions to a local log file for future "
            "classifier training. All data stays on your machine. Disable for sensitive "
            "recordings."
        )
        log_desc.setWordWrap(True)
        log_desc.setStyleSheet("color: gray; font-size: 11px;")
        offline_layout.addWidget(log_desc)

        layout.addWidget(offline_group)

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

    def _run_prefetch(self):
        """Download all models for the currently selected tier."""
        tier = self.model_tier_combo.currentText()
        self.prefetch_btn.setEnabled(False)
        self.prefetch_status.setText(f"Downloading models for {tier} tier...")
        self.prefetch_status.repaint()

        try:
            ModelManager.prefetch(tier)
            self.prefetch_status.setText("All models downloaded.")
        except Exception as e:
            self.prefetch_status.setText(f"Error: {e}")
        finally:
            self.prefetch_btn.setEnabled(True)

    def load_settings(self):
        use_pyannote = self.settings.value("diarization/use_pyannote", False, type=bool)
        if use_pyannote:
            self.radio_pyannote.setChecked(True)
        else:
            self.radio_resemblyzer.setChecked(True)

        token = self.settings.value("diarization/hf_token", "")
        self.token_input.setText(decrypt_token(token))

        vocab = self.settings.value("transcription/custom_vocab", "")
        self.vocab_input.setText(vocab)

        model_tier = self.settings.value("transcription/model_tier", "Best Quality")
        idx = self.model_tier_combo.findText(model_tier)
        if idx >= 0:
            self.model_tier_combo.setCurrentIndex(idx)

        num_speakers = self.settings.value("diarization/num_speakers", 2, type=int)
        self.speaker_count_spin.setValue(num_speakers)

        offline_enabled = self.settings.value("offline/enabled", False, type=bool)
        self.offline_checkbox.setChecked(offline_enabled)

        bc_logging = self.settings.value("backchannel/logging_enabled", True, type=bool)
        self.backchannel_log_checkbox.setChecked(bc_logging)

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
        self.settings.setValue(
            "diarization/hf_token",
            encrypt_token(self.token_input.text().strip()),
        )
        self.settings.setValue(
            "transcription/custom_vocab", self.vocab_input.text().strip()
        )
        self.settings.setValue(
            "transcription/model_tier", self.model_tier_combo.currentText()
        )
        self.settings.setValue(
            "diarization/num_speakers", self.speaker_count_spin.value()
        )
        self.settings.setValue(
            "offline/enabled", self.offline_checkbox.isChecked()
        )
        self.settings.setValue(
            "backchannel/logging_enabled", self.backchannel_log_checkbox.isChecked()
        )
        self.accept()
