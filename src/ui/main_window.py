from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QSplitter,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer
from .file_picker import FilePickerWidget
from .waveform_widget import WaveformWidget
from .transcript_widget import TranscriptWidget
from .metadata_widget import MetadataWidget
from audio.normalise import normalise
from audio.player import AudioPlayer
from transcription.worker import TranscriptionWorker
from lore_core.audio_classifier import AudioClassifyWorker
from lore_core.ner_worker import NERWorker
from lore_core.llm_worker import LLMWorker
from lore_core.translation_worker import TranslationWorker
from lore_core.languages import get_supported_languages
from models.transcript_model import TranscriptListModel
from lore_core.ohms_exporter import OhmsExporter
from ui.settings_dialog import SettingsDialog
from PyQt6.QtCore import QSettings
from utils.token_vault import decrypt_token


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lore: Oral History Transcription")
        self.resize(1000, 700)

        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QLabel { color: #cccccc; }
        """)

        # Core State
        self.original_audio_path = None
        self.working_audio_path = None
        self.audio_player = AudioPlayer(self)
        self.transcript_model = TranscriptListModel()

        # UI Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.stack = QStackedWidget()
        self.main_layout.addWidget(self.stack)

        # Page 1: File Picker
        self.picker_page = FilePickerWidget()
        self.picker_page.file_selected.connect(self._on_file_selected)
        self.stack.addWidget(self.picker_page)

        # Page 2: Loading/Transcribing Status
        self.status_page = QWidget()
        status_layout = QVBoxLayout(self.status_page)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        self.stack.addWidget(self.status_page)

        # Page 3: Main Editor
        self.editor_page = QWidget()
        editor_layout = QVBoxLayout(self.editor_page)

        # Splitter for transcript and metadata
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Transcript area
        transcript_container = QWidget()
        tc_layout = QVBoxLayout(transcript_container)
        tc_layout.setContentsMargins(0, 0, 0, 0)
        self.transcript_view = TranscriptWidget()
        self.transcript_view.setModel(self.transcript_model)

        toolbar_layout = QHBoxLayout()
        self.btn_transcribe = QPushButton("Transcribe")
        self.btn_transcribe.clicked.connect(self.start_transcription)

        self.chk_diarize = QCheckBox("Enable Speaker Diarization (Beta)")

        # Translation UI
        self.lang_combo = QComboBox()
        self.supported_langs = get_supported_languages()
        for code, info in self.supported_langs.items():
            # Format: Name (Quality Tier)
            label = f"{info['name']} ({info['tier']})"
            self.lang_combo.addItem(label, code)

        self.btn_translate = QPushButton("Translate")
        self.btn_translate.clicked.connect(self.start_translation)

        self.btn_search = QPushButton("🔍 Global Search")
        self.btn_search.clicked.connect(self.open_global_search)

        self.btn_settings = QPushButton("⚙️ Settings")
        self.btn_settings.clicked.connect(self.open_settings)

        toolbar_layout.addWidget(self.btn_transcribe)
        toolbar_layout.addWidget(self.chk_diarize)
        toolbar_layout.addSpacing(10)
        toolbar_layout.addWidget(QLabel("Translate to:"))
        toolbar_layout.addWidget(self.lang_combo)
        toolbar_layout.addWidget(self.btn_translate)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.btn_search)
        toolbar_layout.addWidget(self.btn_settings)

        tc_layout.addLayout(toolbar_layout)
        tc_layout.addWidget(self.transcript_view)

        # Audio Player and Waveform
        self.waveform = WaveformWidget()
        self.waveform.seek_requested.connect(self.audio_player.seek)

        control_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setStyleSheet(
            "background-color: #007acc; color: white; border-radius: 4px; padding: 6px;"
        )

        self.time_label = QLabel("00:00 / 00:00")

        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.time_label)
        control_layout.addWidget(self.waveform, stretch=1)

        tc_layout.addLayout(control_layout)

        # Metadata area
        self.metadata_form = MetadataWidget()
        self.metadata_form.export_requested.connect(self._on_export)
        self.metadata_form.bagit_export_requested.connect(self._on_bagit_export)
        self.metadata_form.generate_abstract_requested.connect(
            self._on_generate_abstract
        )

        self.splitter.addWidget(transcript_container)
        self.splitter.addWidget(self.metadata_form)
        self.splitter.setSizes([700, 300])

        editor_layout.addWidget(self.splitter)
        self.stack.addWidget(self.editor_page)

        # Connect audio player signals
        self.audio_player.position_changed.connect(self._on_position_changed)
        self.audio_player.duration_changed.connect(self._on_duration_changed)
        self.audio_player.state_changed.connect(self._on_player_state_changed)

    def open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()

    def open_global_search(self):
        from ui.global_search_dialog import GlobalSearchDialog

        dlg = GlobalSearchDialog(self)
        dlg.exec()

    def start_transcription(self):
        if not self.working_audio_path:
            return

        settings = QSettings("HeritageTools", "Lore")
        use_pyannote = settings.value("diarization/use_pyannote", False, type=bool)
        hf_token = decrypt_token(settings.value("diarization/hf_token", ""))

        enable_diarization = self.chk_diarize.isChecked()

        self.btn_transcribe.setEnabled(False)
        self.status_label.setText("Initializing model...")
        self.transcript_model.clear_segments()

        custom_vocab = settings.value("transcription/custom_vocab", "")

        # We pass diarization settings to the worker
        self.worker = TranscriptionWorker(
            self.working_audio_path,
            enable_diarization=enable_diarization,
            use_pyannote=use_pyannote,
            hf_token=hf_token,
            custom_vocab=custom_vocab,
        )

        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.segment_completed.connect(self.transcript_model.add_segment)
        self.worker.diarization_completed.connect(self.transcript_model.update_segments)
        self.worker.finished.connect(self._on_transcription_finished)
        self.worker.error.connect(self._on_transcription_error)

        # Start Audio Classifier worker in parallel
        self.classifier_worker = AudioClassifyWorker(self.working_audio_path)
        self.classifier_worker.event_detected.connect(self.transcript_model.add_segment)
        self.classifier_worker.error.connect(lambda e: print(f"Classifier error: {e}"))

        # Start NER Worker
        self.ner_worker = NERWorker()
        self.worker.segment_completed.connect(
            self.ner_worker.enqueue_segment, Qt.ConnectionType.QueuedConnection
        )
        self.ner_worker.entities_detected.connect(self.transcript_model.add_entities)
        self.ner_worker.error.connect(lambda e: print(f"NER Error: {e}"))

        self.ner_worker.start()
        self.classifier_worker.start()
        self.worker.start()

    def start_translation(self):
        if not self.transcript_model.get_transcript().segments:
            QMessageBox.warning(
                self, "Warning", "No transcript available to translate."
            )
            return

        target_lang_code = self.lang_combo.currentData()
        if not target_lang_code:
            return

        self.btn_translate.setEnabled(False)
        self.status_label.setText("Preparing translation...")
        self.stack.setCurrentIndex(
            1
        )  # Switch back to status page to show translation progress

        self.translation_worker = TranslationWorker(
            transcript=self.transcript_model.get_transcript(),
            target_lang_code=target_lang_code,
        )
        self.translation_worker.status_changed.connect(self.status_label.setText)
        self.translation_worker.segment_translated.connect(self._on_segment_translated)
        self.translation_worker.finished.connect(self._on_translation_finished)
        self.translation_worker.error.connect(self._on_transcription_error)
        self.translation_worker.start()

    def _on_segment_translated(self, segment):
        # Notify the model that a segment was updated so the UI redraws it
        # The model just needs a dataChanged emission.
        # TranscriptListModel can handle an update request
        self.transcript_model.refresh_segment(segment)

    def _on_translation_finished(self, transcript):
        self.stack.setCurrentIndex(2)  # Switch back to editor
        self.btn_translate.setEnabled(True)
        QMessageBox.information(
            self, "Translation Complete", "The translation has finished successfully."
        )

    def _on_file_selected(self, path: Path):
        self.original_audio_path = path
        self.stack.setCurrentIndex(1)  # Status page
        self.status_label.setText("Normalising audio (16kHz mono)...")

        # Use QTimer to allow UI to update before blocking
        QTimer.singleShot(100, self._process_audio)

    def _process_audio(self):
        try:
            # We save the normalised file to a temp or alongside
            self.working_audio_path = self.original_audio_path.with_suffix(".norm.wav")
            normalise(self.original_audio_path, self.working_audio_path)

            self.status_label.setText("Preparing transcription engine...")

            settings = QSettings("HeritageTools", "Lore")
            enable_diarization = settings.value("diarization/enabled", False, type=bool)
            use_pyannote = settings.value("diarization/use_pyannote", False, type=bool)
            hf_token = decrypt_token(settings.value("diarization/hf_token", ""))
            custom_vocab = settings.value("transcription/custom_vocab", "")

            self.transcript_model.clear_segments()

            # Start transcription worker
            self.worker = TranscriptionWorker(
                self.working_audio_path,
                enable_diarization=enable_diarization,
                use_pyannote=use_pyannote,
                hf_token=hf_token,
                custom_vocab=custom_vocab,
            )
            self.worker.status_changed.connect(self.status_label.setText)
            self.worker.segment_completed.connect(self.transcript_model.add_segment)
            self.worker.diarization_completed.connect(
                self.transcript_model.update_segments
            )
            self.worker.finished.connect(self._on_transcription_finished)
            self.worker.error.connect(self._on_transcription_error)

            # Start Audio Classifier worker in parallel
            self.classifier_worker = AudioClassifyWorker(self.working_audio_path)
            self.classifier_worker.event_detected.connect(
                self.transcript_model.add_segment
            )
            self.classifier_worker.error.connect(
                lambda e: print(f"Classifier error: {e}")
            )

            # Start NER Worker
            self.ner_worker = NERWorker()
            self.worker.segment_completed.connect(
                self.ner_worker.enqueue_segment, Qt.ConnectionType.QueuedConnection
            )
            self.ner_worker.entities_detected.connect(
                self.transcript_model.add_entities
            )
            self.ner_worker.error.connect(lambda e: print(f"NER Error: {e}"))

            self.ner_worker.start()
            self.classifier_worker.start()
            self.worker.start()

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def _on_transcription_finished(self):
        # Load audio into player & waveform
        self.audio_player.load(str(self.working_audio_path))
        self.waveform.load_audio(str(self.working_audio_path))
        self.stack.setCurrentIndex(2)  # Editor page

    def _on_transcription_error(self, err_msg: str):
        self.status_label.setText(f"Transcription Error:\n{err_msg}")

    def _on_player_state_changed(self, state):
        """Update play button when playback state changes (e.g., reaches end)."""
        from PyQt6.QtMultimedia import QMediaPlayer

        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_btn.setText("Pause")
        else:
            self.play_btn.setText("Play")

    def _toggle_playback(self):
        from PyQt6.QtMultimedia import QMediaPlayer

        if self.audio_player._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.audio_player.pause()
        else:
            self.audio_player.play()

    def _on_position_changed(self, ms: int):
        self.waveform.set_position(ms)
        self._update_time_label()

    def _on_duration_changed(self, ms: int):
        self._update_time_label()

    def _update_time_label(self):
        pos_s = self.audio_player.position_ms() // 1000
        dur_s = self.audio_player.duration_ms() // 1000
        self.time_label.setText(
            f"{pos_s // 60:02d}:{pos_s % 60:02d} / {dur_s // 60:02d}:{dur_s % 60:02d}"
        )

    def _on_export(self, metadata: dict):
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save OHMS XML",
            str(self.original_audio_path.with_suffix(".xml")),
            "XML Files (*.xml)",
        )
        if save_path:
            try:
                OhmsExporter.export(
                    self.transcript_model.get_transcript(), metadata, Path(save_path)
                )
                QMessageBox.information(
                    self, "Export Successful", f"Saved to {save_path}"
                )

                # Auto-index
                try:
                    from lore_core.global_search import GlobalSearchIndex

                    idx = GlobalSearchIndex()
                    project_id = metadata.get("title", "Untitled").replace(" ", "_")
                    idx.index_transcript(
                        project_id, self.transcript_model.get_transcript()
                    )
                except Exception as e:
                    print(f"Failed to auto-index: {e}")

            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export:\n{str(e)}"
                )

    def _on_bagit_export(self, metadata: dict):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for BagIt Package",
            str(self.original_audio_path.parent),
        )
        if dir_path:
            try:
                from lore_core.bagit_exporter import BagItPackager
                import uuid

                project_id = (
                    metadata.get("title", "Untitled").replace(" ", "_")
                    + "_"
                    + str(uuid.uuid4())[:8]
                )
                packager = BagItPackager()
                bag_dir = packager.create_bag(
                    audio_path=self.original_audio_path,
                    transcript=self.transcript_model.get_transcript(),
                    output_dir=Path(dir_path),
                    project_id=project_id,
                )
                QMessageBox.information(
                    self, "Export Successful", f"BagIt package created at:\n{bag_dir}"
                )

                # Auto-index
                try:
                    from lore_core.global_search import GlobalSearchIndex

                    idx = GlobalSearchIndex()
                    idx.index_transcript(
                        project_id, self.transcript_model.get_transcript()
                    )
                except Exception as e:
                    print(f"Failed to auto-index: {e}")

            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export BagIt:\n{str(e)}"
                )

    def _on_generate_abstract(self):
        transcript = self.transcript_model.get_transcript()
        if not transcript.segments:
            QMessageBox.warning(
                self, "Warning", "No transcript available to summarize."
            )
            return

        self.metadata_form.set_generating_state("Downloading / Loading LLM...")

        self.llm_worker = LLMWorker(transcript)
        self.llm_worker.status_changed.connect(self.metadata_form.set_generating_state)
        self.llm_worker.finished.connect(self._on_abstract_generated)
        self.llm_worker.error.connect(self._on_abstract_error)
        self.llm_worker.start()

    def _on_abstract_generated(self, abstract_text: str):
        self.metadata_form.set_abstract(abstract_text)

    def _on_abstract_error(self, err_msg: str):
        self.metadata_form.set_abstract(f"Error generating abstract:\n{err_msg}")
        QMessageBox.warning(self, "LLM Error", err_msg)
