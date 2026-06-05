from PyQt6.QtCore import QThread, pyqtSignal
from models.transcript import Transcript
from utils.model_manager import ModelManager
import os
import glob


class LLMWorker(QThread):
    """
    Worker for generating abstracts and themes using a local LLM via llama.cpp.
    It takes the full Transcript, constructs a prompt, and extracts the results.
    """

    finished = pyqtSignal(str)  # Emits the generated abstract
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, transcript: Transcript, parent=None):
        super().__init__(parent)
        self.transcript = transcript
        self._is_running = True

    def run(self):
        try:
            self.status_changed.emit("Ensuring LLM is downloaded...")
            model_dir = ModelManager.ensure_model("LLM")
            if not model_dir:
                self.error.emit("Could not download or locate LLM model.")
                return

            # Find the actual .gguf file
            gguf_files = glob.glob(os.path.join(model_dir, "*.gguf"))
            if not gguf_files:
                self.error.emit("Could not find .gguf file in the model directory.")
                return

            model_path = gguf_files[0]

            self.status_changed.emit("Loading LLM model into memory...")
            try:
                from llama_cpp import Llama
            except ImportError:
                self.error.emit("llama-cpp-python is not installed.")
                return

            # Initialize Llama (with some sensible defaults for Qwen2.5 1.5B)
            # Use 4096 context window to prevent OOM
            llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)

            self.status_changed.emit("Preparing transcript...")

            # Combine transcript text
            full_text = " ".join([seg.text for seg in self.transcript.segments])

            # If transcript is massively long, we just truncate it for this baseline
            # A true map-reduce is complex for a POC, so we take the first ~2500 words (~3000 tokens)
            words = full_text.split()
            if len(words) > 2500:
                truncated_text = " ".join(words[:2500]) + "... [TRUNCATED]"
            else:
                truncated_text = full_text

            self.status_changed.emit("Generating abstract and themes...")

            system_prompt = "You are a professional archivist. Summarize the following oral history transcript."
            user_prompt = f"Please provide a concise 3-paragraph abstract of this interview, followed by a bulleted list of 5 key themes discussed.\n\nTranscript:\n{truncated_text}"

            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.3,
            )

            result_text = output["choices"][0]["message"]["content"]
            self.finished.emit(result_text)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.error.emit(f"LLM Error: {str(e)}")
