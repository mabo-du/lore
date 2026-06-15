"""
Integration smoke test for the core transcription pipeline.

This tests that the AudioLoadWorker threading refactor, the
TranscriptionWorker, and the model loading all work end-to-end.

Requires:
    - A working PyQt6 display (or xvfb)
    - Internet to download the Whisper model on first run (~1.5 GB)
    - sample.ogg in the project root (11 seconds, included in repo)

Run: python -m pytest tests/test_transcription_smoke.py -v --timeout=600
"""

from pathlib import Path
import pytest
import time

SAMPLE_FILE = Path(__file__).parent.parent / "sample.ogg"


@pytest.mark.integration
class TestTranscriptionSmoke:
    """Full pipeline test from raw audio to transcribed segments."""

    @pytest.fixture(autouse=True)
    def _setup_app(self, request):
        """Ensure a QApplication exists for the worker's signals."""
        from PyQt6.QtWidgets import QApplication
        import sys
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        yield

    def _run_transcription(self, quality_tier="Fast", timeout_s=300):
        """
        Run transcription and return (success, segment_count, error_msg, elapsed).
        Uses the "Fast" model tier by default to minimise download time.
        """
        from transcription.worker import TranscriptionWorker

        result = {"success": False, "segments": 0, "error": None}

        def on_segment(seg):
            result["segments"] += 1

        def on_finished():
            result["success"] = True
            self.app.quit()

        def on_error(err):
            result["error"] = err
            self.app.quit()

        start = time.time()

        worker = TranscriptionWorker(
            audio_path=SAMPLE_FILE,
            quality_tier=quality_tier,
            enable_diarization=False,
        )
        worker.segment_completed.connect(on_segment)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.start()

        # Run event loop with timeout
        timer = self.app.timer = self.app  # prevent GC
        timer.singleShot(int(timeout_s * 1000), self.app.quit)
        self.app.exec()

        elapsed = time.time() - start
        return (
            result["success"],
            result["segments"],
            result["error"],
            elapsed,
        )

    def test_fast_model_transcribes(self):
        """
        Load the small Whisper model and transcribe sample.ogg.
        This is the most important integration test — it validates the
        entire pipeline from file → normalisation → engine → segments.
        """
        ok, count, err, elapsed = self._run_transcription(
            quality_tier="Fast", timeout_s=600
        )

        assert ok, f"Transcription failed: {err}"
        assert count > 0, (
            f"Transcription completed but produced 0 segments "
            f"(elapsed={elapsed:.1f}s)"
        )
        print(f"\n  Transcribed {count} segments in {elapsed:.1f}s")

    def test_transcription_speed(self, quality_tier="Fast"):
        """Transcription should complete within a reasonable time."""
        ok, count, err, elapsed = self._run_transcription(
            quality_tier=quality_tier, timeout_s=600
        )
        assert ok, f"Transcription failed: {err}"
        # 11s file with "Fast" model should finish in under 5 minutes
        assert elapsed < 300, (
            f"Transcription too slow: {elapsed:.1f}s for 11s file"
        )
