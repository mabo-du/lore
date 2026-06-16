from PyQt6.QtCore import QThread, pyqtSignal
from models.transcript import Transcript
from utils.model_manager import ModelManager
import os
import glob


def _format_gap(seconds: float) -> str:
    """Format a gap duration as an inline marker token."""
    return f"<gap={max(0.0, seconds):.1f}s>"


def _format_enriched_transcript(transcript: Transcript, max_words: int = 2500) -> str:
    """
    Format transcript as speaker-labelled lines with inter-turn gap markers.

    Output format:
        SPEAKER_00: I think we should proceed.
        <gap=3.2s>
        SPEAKER_01: I don't think the backend is ready.
        <gap=0.2s>
        SPEAKER_00: The load testing cleared yesterday.
    """
    lines = []
    word_count = 0
    for i, seg in enumerate(transcript.segments):
        # Truncate if over the word budget (applied at segment boundary)
        seg_words = seg.text.split()
        if word_count + len(seg_words) > max_words:
            remaining = max_words - word_count
            if remaining > 0:
                lines.append(f"{seg.speaker_label or 'SPEAKER'}: {' '.join(seg_words[:remaining])} ...")
            else:
                lines.append("... [transcript truncated]")
            break

        # Inter-turn gap (skip SYSTEM segments for gap calculation)
        if i > 0 and seg.speaker_label != "SYSTEM" and transcript.segments[i-1].speaker_label != "SYSTEM":
            gap = (seg.start_ms - transcript.segments[i-1].end_ms) / 1000.0
            if gap > 0.0:
                lines.append(_format_gap(gap))

        speaker = seg.speaker_label or "SPEAKER"
        lines.append(f"{speaker}: {seg.text}")
        word_count += len(seg_words)

    return "\n".join(lines)


def _build_summarization_prompt(enriched_text: str) -> tuple[str, str]:
    """
    Build system + user prompt for timing-aware summarisation.
    """
    system_prompt = (
        "You are a professional archivist and dialogue analyst.\n\n"
        "The transcript below contains timing markers formatted as <gap=X.Xs> "
        "between speaker turns. These indicate the duration of silence in seconds.\n\n"
        "Use the following temporal rules to interpret the conversation's dynamics:\n"
        "1. Short gaps (<0.5s) indicate fast-paced, collaborative, or urgent exchanges.\n"
        "2. Moderate gaps (0.5s-1.5s) indicate formal, structured turn-taking.\n"
        "3. Long gaps (>1.5s) indicate hesitation, deliberation, reluctance, or topic shifts.\n\n"
        "Apply these cues when assessing the tone and structure of the interview.\n"
        "IMPORTANT: Do NOT mention the gap markers in your summary. Use them only as "
        "internal context to inform your analysis."
    )
    user_prompt = (
        "Please provide a concise 3-paragraph abstract of this oral history interview, "
        "followed by a bulleted list of 5 key themes discussed.\n\n"
        f"Transcript:\n{enriched_text}"
    )
    return system_prompt, user_prompt


class LLMWorker(QThread):
    """
    Worker for generating abstracts and themes using a local LLM via llama.cpp.
    Takes the full Transcript, constructs a timing-enriched prompt, and extracts results.
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
            self.status_changed.emit("Downloading LLM model (1.1 GB)...")

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

            # Initialize Llama with 4096 context window to prevent OOM
            llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)

            self.status_changed.emit("Preparing enriched transcript...")

            # Format with speaker labels and inter-turn gap markers
            enriched_text = _format_enriched_transcript(self.transcript)

            self.status_changed.emit("Generating abstract and themes...")

            system_prompt, user_prompt = _build_summarization_prompt(enriched_text)

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
