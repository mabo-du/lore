import ctranslate2
import transformers
from PyQt6.QtCore import QThread, pyqtSignal
from models.transcript import Transcript, Segment
from utils.model_manager import ModelManager
import gc
import logging
import time

logger = logging.getLogger(__name__)


class TranslationWorker(QThread):
    """
    Translates a Transcript segment-by-segment using NLLB-200.
    Ensures that Whisper is unloaded, loads NLLB into memory, performs the translation,
    and then unloads NLLB to strictly stay under the 8GB RAM target.
    """

    status_changed = pyqtSignal(str)
    segment_translated = pyqtSignal(
        Segment
    )  # Emits segment once its .translation is set
    finished = pyqtSignal(Transcript)
    error = pyqtSignal(str)

    def __init__(
        self,
        transcript: Transcript,
        target_lang_code: str,
        source_lang_code: str = "eng_Latn",
        parent=None,
    ):
        super().__init__(parent)
        self.transcript = transcript
        self.target_lang_code = target_lang_code
        self.source_lang_code = source_lang_code

    def run(self):
        try:
            self.status_changed.emit("Ensuring NLLB translation model is downloaded...")
            model_dir = ModelManager.ensure_model("Translation")

            # Unload any residual large memory objects before loading NLLB
            gc.collect()

            self.status_changed.emit("Loading NLLB Tokenizer...")
            # We use transformers AutoTokenizer which does not strictly require PyTorch to operate
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir, src_lang=self.source_lang_code
            )

            self.status_changed.emit("Loading NLLB Model into RAM...")
            # Load with INT8 precision
            translator = ctranslate2.Translator(model_dir, compute_type="int8")

            self.status_changed.emit("Translating...")

            # Record the target language metadata
            self.transcript.metadata.target_language = self.target_lang_code

            total = len(self.transcript.segments)
            for i, segment in enumerate(self.transcript.segments):
                if segment.speaker_label == "SYSTEM":
                    # For system messages like silence gaps
                    segment.translation = segment.text
                    self.segment_translated.emit(segment)
                    continue

                if not segment.text.strip():
                    segment.translation = ""
                    self.segment_translated.emit(segment)
                    continue

                # Tokenize
                source = tokenizer.convert_ids_to_tokens(tokenizer.encode(segment.text))
                target_prefix = [self.target_lang_code]

                # Inference
                results = translator.translate_batch(
                    [source], target_prefix=[target_prefix]
                )

                # Decode
                target = results[0].hypotheses[0][1:]  # Remove the language token
                translated_text = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(target)
                )

                # Set translation
                segment.translation = translated_text

                # Emit signal
                self.segment_translated.emit(segment)

                # Yield to keep GUI event loop healthy
                time.sleep(0.001)

                if i % 5 == 0:
                    self.status_changed.emit(f"Translating segments ({i}/{total})...")

            # Clean up the large model explicitly
            self.status_changed.emit("Unloading NLLB model...")
            del translator
            del tokenizer
            gc.collect()

            self.status_changed.emit("Translation complete.")
            self.finished.emit(self.transcript)

        except Exception as e:
            logger.exception("Translation error")
            self.error.emit(str(e))
