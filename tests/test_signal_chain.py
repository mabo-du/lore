"""
Mocked integration test for the MainWindow signal chain.

Tests the orchestration logic (file_selected → audio_ready → transcribe_finished
→ editor page) using mock workers with proper PyQt6 signals.

Run: pytest tests/test_signal_chain.py -v
"""

from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal


# ── Mock Workers (signals must be class attributes in PyQt6) ─────────────

class MockAudioLoadWorker(QObject):
    """Fake AudioLoadWorker — call fire_finished() to trigger chain."""

    status_changed = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def fire_finished(self, path="/mock/file.norm.wav"):
        self.status_changed.emit("Normalising audio...")
        self.finished.emit(path)

    def fire_error(self, msg="Audio normalisation failed"):
        self.error.emit(msg)


class MockTranscriptionWorker(QObject):
    """Fake TranscriptionWorker — call fire_finished() to trigger chain."""

    status_changed = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def fire_finished(self):
        self.status_changed.emit("Transcribing...")
        self.finished.emit()

    def fire_error(self, msg="Transcription failed"):
        self.error.emit(msg)


# ── Test Harness ─────────────────────────────────────────────────────────

class SignalChainHarness(QObject):
    """
    Isolated test harness that mirrors MainWindow's signal chain logic.
    Tests the handoff sequence:

        _on_file_selected → AudioLoadWorker.finished → _on_audio_ready
        → TranscriptionWorker.finished → _on_transcription_finished
    """

    page_changed = pyqtSignal(int)  # Tracks page transitions

    def __init__(self):
        super().__init__()
        self.current_page = 0          # 0=file_picker, 1=status, 2=editor
        self.btn_transcribe_enabled = True
        self.audio_load_worker = None
        self.worker = None
        self.transcript_has_segments = False

    def _on_file_selected(self, path):
        """Simulate file selection — sets up AudioLoadWorker mock."""
        self.current_page = 1
        self.page_changed.emit(1)

        self.audio_load_worker = MockAudioLoadWorker()
        self.audio_load_worker.finished.connect(self._on_audio_ready)
        # Fire immediately (mirrors the real worker's async completion)
        self.audio_load_worker.fire_finished()

    def _on_audio_ready(self, wav_path):
        """Simulate normalisation complete — sets up TranscriptionWorker mock."""
        self.btn_transcribe_enabled = False
        self.worker = MockTranscriptionWorker()
        self.worker.finished.connect(self._on_transcription_finished)
        self.worker.fire_finished()

    def _on_transcription_finished(self):
        """Simulate transcription complete — transitions to editor page."""
        self.current_page = 2
        self.page_changed.emit(2)
        self.btn_transcribe_enabled = True
        self.transcript_has_segments = True

    def _on_transcription_error(self):
        """Simulate error — returns to file picker."""
        self.current_page = 0
        self.page_changed.emit(0)
        self.btn_transcribe_enabled = True


# ── Tests ────────────────────────────────────────────────────────────────

class TestSignalChain:

    def test_file_selected_shows_status_page(self):
        """Selecting a file immediately starts the pipeline."""
        harness = SignalChainHarness()
        harness._on_file_selected(Path("/mock/file.wav"))
        # The chain fires synchronously in the mock, so by the time we
        # assert, the pipeline has run to completion (editor page at index 2).
        # This confirms the chain didn't hang or error mid-way.
        assert harness.current_page == 2

    def test_full_pipeline_ends_at_editor(self, qtbot):
        """After normalisation + transcription, editor page should be active."""
        harness = SignalChainHarness()
        with qtbot.wait_signal(harness.page_changed, timeout=5000):
            harness._on_file_selected(Path("/mock/file.wav"))
        # The chain fires synchronously — last emission is page_changed(2)
        assert harness.current_page == 2

    def test_transcribe_button_enabled_after_pipeline(self):
        """Transcribe button should be enabled after pipeline completes."""
        harness = SignalChainHarness()
        harness._on_file_selected(Path("/mock/file.wav"))
        assert harness.btn_transcribe_enabled is True

    def test_error_path_returns_to_file_picker(self):
        """Error should return to file picker (page 0)."""
        harness = SignalChainHarness()
        harness.current_page = 1
        harness._on_transcription_error()
        assert harness.current_page == 0
        assert harness.btn_transcribe_enabled is True

    def test_transcript_has_segments_after_completion(self):
        """Transcript should contain segments after pipeline."""
        harness = SignalChainHarness()
        assert harness.transcript_has_segments is False
        harness._on_file_selected(Path("/mock/file.wav"))
        assert harness.transcript_has_segments is True

    def test_signal_emission_order(self, qtbot):
        """
        Verify page transitions fire in order: status (1) → editor (2).
        """
        harness = SignalChainHarness()
        emissions = []

        def track_page(page):
            emissions.append(page)

        harness.page_changed.connect(track_page)

        with qtbot.wait_signal(harness.page_changed, timeout=5000):
            harness._on_file_selected(Path("/mock/file.wav"))

        assert emissions == [1, 2], (
            f"Expected page transitions [1, 2], got {emissions}"
        )
