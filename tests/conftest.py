import sys
import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Mock heavy AI libraries so tests can run without massive dependencies
sys.modules['faster_whisper'] = MagicMock()
sys.modules['ctranslate2'] = MagicMock()
sys.modules['pyannote'] = MagicMock()
sys.modules['pyannote.audio'] = MagicMock()
sys.modules['resemblyzer'] = MagicMock()
sys.modules['gliner2'] = MagicMock()
sys.modules['gliner2_onnx'] = MagicMock()
sys.modules['llama_cpp'] = MagicMock()

import numpy as np

class MockTextEmbedding:
    def __init__(self, model_name):
        pass
    def embed(self, texts, **kwargs):
        return [np.zeros(384, dtype=np.float32) for _ in texts]

sys.modules['fastembed'] = MagicMock()
sys.modules['fastembed'].TextEmbedding = MockTextEmbedding

# Add src to sys.path so we can import modules
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@pytest.fixture
def dummy_audio_file(tmp_path):
    """Creates a valid 1-second silent WAV file for testing."""
    import wave
    import struct
    audio_file = tmp_path / "test_audio.wav"
    with wave.open(str(audio_file), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        # Write 1 second of silence
        data = struct.pack('<h', 0) * 16000
        w.writeframes(data)
    return audio_file

@pytest.fixture
def dummy_transcript():
    """Provides a minimal Transcript object for tests."""
    from models.transcript import Transcript, Segment
    t = Transcript()
    t.segments.append(
        Segment(
            start_ms=0,
            end_ms=1000,
            text="This is a test segment.",
            avg_logprob=-0.5,
            compression_ratio=1.0,
            no_speech_prob=0.1
        )
    )
    return t
