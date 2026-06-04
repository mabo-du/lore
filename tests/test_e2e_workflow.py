import pytest
from PyQt6.QtCore import Qt
from ui.main_window import MainWindow
from models.transcript import Segment
from lore_core.bagit_exporter import BagItPackager

@pytest.fixture
def app_window(qtbot):
    """Provides a MainWindow instance to tests."""
    window = MainWindow()
    qtbot.addWidget(window)
    return window

def test_full_workflow(qtbot, app_window, dummy_audio_file, mocker, tmp_path):
    """
    Simulates a full user workflow:
    1. Select file
    2. Transcribe (mocked)
    3. Export BagIt
    """
    # ==========================================
    # 1. Mock the Heavy AI Workers
    # ==========================================
    
    # We mock the TranscriptionWorker's run method to immediately emit dummy data
    def mock_run(self):
        self.status_changed.emit("Mock Transcribing...")
        # Emit a segment
        seg = Segment(start_ms=0, end_ms=2000, text="Mock transcription successful.")
        self.segment_completed.emit(seg)
        # We also need to emit diarization_completed if it was enabled, but let's just emit finished.
        self.status_changed.emit("Transcription complete.")
        self.finished.emit()

    mocker.patch("transcription.worker.TranscriptionWorker.run", mock_run)
    
    # We also mock AudioClassifyWorker and NERWorker so they don't do real work
    mocker.patch("lore_core.audio_classifier.AudioClassifyWorker.run", lambda self: self.finished.emit() if hasattr(self, 'finished') else None)
    mocker.patch("lore_core.ner_worker.NERWorker.run", lambda self: None)
    mocker.patch("lore_core.ner_worker.NERWorker.enqueue_segment", lambda self, s: None)
    
    # Mock normalise so it doesn't try to call ffmpeg on our 0-byte file
    import shutil
    mocker.patch("ui.main_window.normalise", lambda in_path, out_path: shutil.copy2(in_path, out_path))
    
    # ==========================================
    # 2. Simulate User Actions
    # ==========================================
    
    # A. Select File
    app_window._on_file_selected(dummy_audio_file)
    
    # The app uses QTimer.singleShot(100) to transition.
    # Wait until it reaches the Editor Page (index 2)
    # Wait for the mocked transcription to finish
    qtbot.waitUntil(lambda: app_window.stack.currentIndex() == 2, timeout=2000)
    
    # Check that our mock segment was loaded into the model
    model = app_window.transcript_model
    assert model.rowCount() == 1
    assert model.data(model.index(0, 0), model.TextRole) == "Mock transcription successful."
    
    # B. Export BagIt Package
    # We need to mock QFileDialog.getExistingDirectory so it doesn't show a UI
    export_dir = tmp_path / "bag_export"
    export_dir.mkdir()
    
    mocker.patch(
        "PyQt6.QtWidgets.QFileDialog.getExistingDirectory",
        return_value=str(export_dir)
    )
    # We also mock QMessageBox so the test doesn't hang on the "Export Successful" popup
    mocker.patch("PyQt6.QtWidgets.QMessageBox.information")
    mocker.patch("PyQt6.QtWidgets.QMessageBox.critical")
    
    # Also we should probably mock the global_search auto-indexer so it doesn't fail
    mocker.patch("lore_core.global_search.GlobalSearchIndex.index_transcript")
    
    # Trigger export
    app_window.metadata_form._on_bagit_export()
    
    # Verify BagIt was created in the export_dir
    # Since the project_id is random UUID, we just look for any directory starting with lore_bag
    bag_dirs = list(export_dir.glob("lore_bag_*"))
    assert len(bag_dirs) == 1
    
    bag_dir = bag_dirs[0]
    assert (bag_dir / "bagit.txt").exists()
    assert (bag_dir / "data" / "transcript.xml").exists()
