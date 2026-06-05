from pathlib import Path
from typing import Generator
from faster_whisper import WhisperModel
from models.transcript import Segment


class TranscriptionEngine:
    """
    Core engine wrapping faster-whisper.
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        """Lazy load the model on the CPU with INT8 precision."""
        if self._model is None:
            # We strictly enforce CPU+INT8 to avoid segfaults and ensure max portability
            self._model = WhisperModel(
                str(self.model_path), device="cpu", compute_type="int8"
            )

    def transcribe(
        self, audio_path: Path, initial_prompt: str = ""
    ) -> Generator[Segment, None, None]:
        """
        Yields Segments as they are transcribed.

        Rules:
            - Must capture avg_logprob, compression_ratio, no_speech_prob
              from each faster-whisper segment for anomaly detection.
            - word_timestamps=True enables per-word probability access.
        """
        self._load_model()

        # condition_on_previous_text=False prevents hallucination loops in silent/noisy parts
        # vad_filter=True avoids transcribing ambient noise
        # word_timestamps=True enables word-level confidence scoring
        segments_gen, _info = self._model.transcribe(
            str(audio_path),
            vad_filter=True,
            condition_on_previous_text=False,
            word_timestamps=True,
            initial_prompt=initial_prompt if initial_prompt else None,
        )

        for s in segments_gen:
            word_dicts = []
            if s.words:
                for w in s.words:
                    word_dicts.append(
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "prob": w.probability,
                        }
                    )

            yield Segment(
                start_ms=int(s.start * 1000),
                end_ms=int(s.end * 1000),
                text=s.text.strip(),
                avg_logprob=s.avg_logprob,
                compression_ratio=s.compression_ratio,
                no_speech_prob=s.no_speech_prob,
                words=word_dicts,
            )
