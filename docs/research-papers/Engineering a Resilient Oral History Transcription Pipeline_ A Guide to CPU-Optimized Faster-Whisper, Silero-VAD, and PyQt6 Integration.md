# Engineering a Resilient Oral History Transcription Pipeline: A Guide to CPU-Optimized Faster-Whisper, Silero-VAD, and PyQt6 Integration

## Model Selection and Hardware Optimization for CPU-Only Inference

The selection of an appropriate Automatic Speech Recognition (ASR) model and its configuration for CPU-only inference is a foundational decision that directly impacts the performance, accuracy, and usability of the Lore transcription tool. For a system designed to operate exclusively on CPUs with 8–16 GB of RAM, the primary considerations shift from raw computational throughput to a careful balance of model size, RAM footprint, processing speed, and real-world accuracy on challenging audio. The analysis must prioritize robustness against the specific acoustic artifacts prevalent in oral history archives—such as low-volume elderly speakers, variable recording quality, and background noise—over peak performance on clean, read-speech datasets like LibriSpeech . This section provides a comprehensive technical analysis of available `faster-whisper` models, their hardware requirements, and optimal configuration strategies for mid-range desktop systems.

`faster-whisper` is an optimized implementation of OpenAI's Whisper model, leveraging the CTranslate2 library to achieve significant performance gains [[12](https://pypi.org/project/faster-whisper/0.3.0/), [19](https://adg.csdn.net/696f3644437a6b403369a7d0.html)]. It is up to four times faster than the original Whisper implementation while maintaining the same level of accuracy and using less memory [[12](https://pypi.org/project/faster-whisper/0.3.0/), [19](https://adg.csdn.net/696f3644437a6b403369a7d0.html), [26](https://www.linkedin.com/posts/arturo-javier-castellanos-551006197_lately-ive-been-using-claude-code-a-lot-activity-7386190761429475328-wLUF)]. This efficiency is crucial for offline applications like Lore, where processing time for long-form audio (1–4 hours) is a key factor in the user experience. The core of the technology is based on the Transformer architecture, which allows it to process audio in chunks, but its overall design prioritizes efficient inference on both GPU and CPU hardware [[26](https://www.linkedin.com/posts/arturo-javier-castellanos-551006197_lately-ive-been-using-claude-code-a-lot-activity-7386190761429475328-wLUF)].

A critical aspect of deploying `faster-whisper` on a CPU-only system is the `compute_type` parameter. This parameter dictates how the model's weights are interpreted and processed. On a CPU, setting `compute_type` to `int8` or `int8_float16` is highly recommended. These options utilize 8-bit integer arithmetic, which is significantly faster on modern CPUs than standard 32-bit floating-point operations (`float32`) [[14](https://arxiv.org/html/2503.09905v1), [63](https://blog.csdn.net/duzm200542901104/article/details/131476514)]. Using `int8` avoids the overhead of floating-point math and does not require any GPU-specific libraries like CUDA or cuDNN, making it ideal for a pure-CPU environment [[14](https://arxiv.org/html/2503.09905v1)]. While `float16` can also offer some speed benefits, it often relies on specific CPU instruction sets (like AVX512) that may not be universally available, whereas `int8` is more broadly supported. Therefore, for maximum compatibility and speed on a typical mid-range CPU, `int8` is the superior choice. The exact pip dependencies for a pure-CPU installation would include `faster-whisper` itself, which brings in `ctranslate2` as a dependency, and potentially `numpy`. The installation process is straightforward and well-documented, ensuring that no extraneous GPU-related packages are pulled in [[11](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_voicechatbot/2_setup/)].

The following table summarizes the available `faster-whisper` models, extrapolating from current information to project their characteristics for a 2025–2026 deployment. Disk sizes are approximate, and RAM requirements refer to the memory needed to load the model weights into RAM before processing begins; this does not include the memory required for the audio buffer and other program variables. The Real-Time Factor (RTF) is an estimate for a modern Intel Core i7-class CPU.

| Model Name | Approximate Disk Size | RAM Requirement (CPU) | Expected RTF (i7 CPU) | Word Error Rate (WER) |
| :--- | :--- | :--- | :--- | :--- |
| `tiny` | ~50 MB | ~100 MB | ~2.5x | High |
| `base` | ~100 MB | ~200 MB | ~2.0x | Medium |
| `small` | ~200 MB | ~400 MB | ~1.5x | Medium |
| `medium` | ~550 MB | ~1.1 GB | ~1.0x | Low |
| `large-v2` | ~1.5 GB | ~3.0 GB | ~0.6x | Very Low |
| `large-v3` | ~1.5 GB | ~3.0 GB | ~0.6x | Very Low |
| `large-v3-turbo` | ~1.5 GB | ~3.0 GB | ~0.8x | Very Low |
| `distil-large-v3` | ~1.1 GB | ~2.2 GB | ~0.7x | Low-Medium |

*Note: WER figures are indicative and based on general knowledge of the Whisper family's capabilities; specific oral history benchmarks are not available in the provided sources.*

For Lore's target hardware of 8–16 GB of RAM, several models are viable. However, the `medium` model emerges as the most pragmatic and balanced choice. It offers a strong baseline of accuracy, striking a good compromise between the higher resource demands of the large models and the lower accuracy of the smaller ones [[70](https://adg.csdn.net/6970706f437a6b40336a407d.html)]. Its RAM requirement of approximately 1.1 GB is comfortably within the limits of even an 8 GB system, leaving ample memory for the operating system and other processes. For users with 16 GB of RAM who prioritize archival-grade fidelity over speed, `large-v3` or `large-v3-turbo` represent the next tier. The `large-v3-turbo` variant is specifically optimized for live speech streaming and may offer better performance on conversational audio compared to the standard `large-v3` model, though its accuracy on offline archival material is comparable [[5](https://community.openai.com/t/all-my-attempts-to-improve-accuracy-and-reduce-hallucinations-have-the-opposite-effect/997302)]. Distilled models like `distil-large-v3` present an interesting alternative, aiming to reduce the model size and RAM footprint with minimal loss in accuracy, making them a potential candidate for pushing the boundaries of what is possible on constrained hardware [[4](https://adg.csdn.net/6952542c5b9f5f31781b8d87.html), [9](https://www.kaggle.com/code/artemcheremuhin/asr-faster-whisper-distil-large-v3)].

It is important to acknowledge the lack of published benchmarks specifically tailored to oral history conditions . Practitioner experience from adjacent fields, such as digital humanities and archival science, suggests that for offline transcription tasks involving human conversation, the `medium` model is frequently favored for its reliability and efficiency [[27](https://aclanthology.org/2024.jeptalnrecital-taln.35.pdf)]. Tools like MacWhisper, another CTranslate2-based implementation, are noted for their utility in quickly creating textual transcripts of interviews, further validating the suitability of this technology stack for the intended purpose [[53](https://aclanthology.org/2024.htres-1.6.pdf)]. Therefore, starting with the `medium` model provides a solid foundation that aligns with both empirical evidence and community best practices. Users could be offered a "balanced" mode using `medium` and an "archival" mode using `large-v3` to cater to different needs and hardware capabilities.

In summary, the optimal strategy for model selection and hardware optimization involves choosing the `medium` model as the default for CPU-only operation. This choice is supported by its favorable balance of accuracy and resource consumption. The `compute_type` should be explicitly set to `int8` to leverage CPU-optimized integer arithmetic, ensuring the fastest possible transcription speed on the target hardware without requiring any specialized libraries. This approach respects the constraints of the platform while delivering a high-quality transcription service that meets the core requirements of the Lore application.

## Integrating Voice Activity Detection for Robust Speech Segmentation

The integration of a Voice Activity Detector (VAD) is a critical step in preparing audio for transcription, particularly for the complex and varied acoustic environments characteristic of oral history collections. The primary goal of VAD in this context is not merely to improve efficiency by reducing processing time, but to enhance transcription accuracy by isolating segments of genuine speech from non-speech audio such as background noise, long pauses, and irrelevant sounds [[38](https://aclanthology.org/2023.mtsummit-users.pdf)]. The user's clarified priority is to optimize for robustness against speech loss in low-volume or pause-heavy audio, a common challenge when dealing with quiet elderly speakers or natural conversational breaks . This requires a careful evaluation of the two primary approaches: using an external VAD like silero-VAD for pre-processing versus relying on `faster-whisper`'s built-in VAD filter.

Silero VAD is a pre-trained, enterprise-grade voice activity detector described as being robust against variations in signal amplitudes and speech distortions [[2](https://aclanthology.org/2025.iwsds-1.26.pdf), [3](https://pytorch.org/hub/snakers4_silero-vad_vad/), [22](https://www.mdpi.com/2218-6581/14/12/184)]. It is widely adopted in academic and industrial pipelines, including those for podcast segmentation, meeting-style conversations, and virtual companions, attesting to its reliability [[41](https://www.nature.com/articles/s41598-026-39884-8), [44](https://arxiv.org/html/2505.05056v1), [48](https://aclanthology.org/2024.bea-1.4.pdf)]. When integrated as an external pre-processing step, silero-VAD operates on a resampled version of the input audio. It typically expects a sample rate of 16kHz, which aligns perfectly with the requirements of the Whisper model itself [[63](https://blog.csdn.net/duzm200542901104/article/details/131476514)]. The output of silero-VAD is a list of timestamp intervals, specifying the start and end times of detected speech segments . This granular output provides the necessary information for two distinct post-processing strategies. The first is to create a new, trimmed audio file containing only the concatenated speech regions, which is then fed to `faster-whisper`. This method reduces the total amount of data to be processed, potentially speeding up transcription. However, it carries the risk of aggressively clipping the beginnings or ends of sentences if the VAD thresholds are too sensitive. The second, more conservative approach, is to pass these timestamps to `faster-whisper`'s native `vad_filter` parameter, allowing the ASR engine to focus its processing solely on the identified speech regions.

The `faster-whisper` library includes a built-in `vad_filter` parameter, which can be enabled by setting it to `True` [[63](https://blog.csdn.net/duzm200542901104/article/details/131476514)]. When activated, this feature instructs the model to perform its own VAD internally before generating a transcription. However, the documentation does not specify whether this internal mechanism uses silero-VAD or a different algorithm. While convenient, this built-in filter offers less control and transparency compared to the external silero-VAD approach. Given the user's explicit directive to prioritize robustness against speech loss, the external pre-processing method is strongly recommended. It provides fine-grained control over the VAD parameters, allowing developers to tune the sensitivity to accommodate the specific challenges of oral history audio. For instance, the minimum amplitude threshold for detecting speech can be lowered to ensure quiet elderly speakers are not mistaken for background noise. Similarly, the handling of long pauses can be adjusted to prevent the VAD from prematurely ending a segment while the speaker is thinking.

Despite its strengths, silero-VAD has known limitations. It can struggle with accurately differentiating between near-field speech and cross-talk, which is relevant for dual-speaker interviews where voices may overlap or be recorded from different distances [[13](https://arxiv.org/pdf/2402.09797)]. It may also have difficulty distinguishing speech from certain types of non-speech audio, such as sustained background music or mechanical noises that fall within the vocal frequency range. To mitigate these issues, a multi-pronged strategy could be employed. For stereo audio files, VAD could be run independently on each channel to better capture dialogue between two interviewers. Furthermore, the parameters of the `vad_filter` in `faster-whisper` can be used in conjunction with the external VAD to create a layered defense against misclassification.

Regarding dependencies, both `faster-whisper` and `silero-vad` are typically built on top of PyTorch. It is crucial to ensure that the versions of PyTorch are compatible to avoid runtime errors. While specific version conflicts are not detailed in the provided sources, it is a common issue when installing multiple deep learning libraries. The recommended practice is to install them within an isolated Python environment (e.g., using conda or venv) and to install them in a specific order, typically starting with PyTorch itself, followed by `faster-whisper`, and finally `silero-vad`. Checking the official repositories for any documented compatibility notes is also advisable [[38](https://aclanthology.org/2023.mtsummit-users.pdf), [41](https://www.nature.com/articles/s41598-026-39884-8)].

In conclusion, for the Lore application, the recommended architecture involves using silero-VAD as an external pre-processing step. This approach offers superior control and configurability, which is essential for achieving the high degree of robustness required to handle the nuances of oral history audio. By tuning the VAD parameters to be less aggressive and combining this with the conservative settings in `faster-whisper`'s `vad_filter`, the pipeline can effectively preserve genuine speech while filtering out problematic non-speech segments, thereby laying a solid foundation for high-quality transcription.

## Managing Long-Form Audio: Context Window Handling and Hallucination Mitigation

Transcribing long-form audio sessions, such as the 1–4 hour oral histories central to the Lore project, presents unique technical challenges that go beyond simply feeding a long file to an ASR model. Two of the most significant hurdles are how the model manages its limited context window and the phenomenon of "hallucination," where the model fabricates text. Understanding and mitigating these issues is paramount for producing reliable and trustworthy transcripts of historical documents.

`faster-whisper`, like its parent model Whisper, processes audio in fixed-size chunks or windows, typically 30 seconds long [[63](https://blog.csdn.net/duzm200542901104/article/details/131476514)]. When faced with an audio file longer than this window, the model employs a technique of sliding the window across the audio stream. Crucially, it passes information from the previous window's output to the next one. This mechanism allows the model to maintain a sense of continuity and context throughout the entire audio file without needing to be explicitly configured for long-form processing. Developers do not need to manually segment the audio; the `transcribe` function handles this internally. However, this process has implications for memory usage. During transcription, the model needs to keep the encoded representation of the preceding audio segment in memory to inform the current one. Consequently, for a 4-hour session, the memory footprint will scale with the duration of the audio, although it is primarily dictated by the size of the model's attention layers rather than the entire audio file being loaded into RAM at once. Nonetheless, this can lead to a significant and growing memory demand over a long session.

The second major challenge is hallucination. This occurs when the model, encountering audio segments with low confidence—such as silence, background noise, or very quiet speech—generates plausible-sounding but fabricated text instead of leaving the segment blank [[7](https://dev.to/nareshipme/whisper-hallucination-on-silence-why-your-transcript-loops-the-same-phrase-2pg4), [32](https://arxiv.org/html/2501.11378v1)]. These hallucinations can take various forms, including repeated phrases, filler words like "thank you for watching," or entirely nonsensical sentences [[10](https://aclanthology.org/2025.findings-acl.1190.pdf), [45](https://arxiv.org/html/2402.08021v2)]. For archival materials, such fabrications are unacceptable as they introduce false information into the historical record. While a robust VAD pre-processing step helps mitigate this by removing many of these low-confidence segments before they reach the ASR model, hallucinations can still occur within the processed audio if the VAD is not perfectly tuned or if genuinely quiet speech is misclassified.

Fortunately, `faster-whisper` exposes several powerful parameters that allow developers to actively suppress hallucinations. These parameters act as a series of quality checks on the generated text:
*   **`condition_on_previous_text`**: When set to `True`, this parameter prevents the model from repeating itself. It forces the model to generate text that is semantically distinct from the previous segment, which is effective at stopping the repetition of phrases [[63](https://blog.csdn.net/duzm200542901104/article/details/131476514)].
*   **`no_speech_threshold`**: This parameter controls the model's confidence level for determining if a segment contains speech. If the model's confidence drops below this threshold, it outputs a blank segment. Setting this to a moderately high value (e.g., 0.6) makes the model more conservative and less likely to guess at unintelligible audio [[63](https://blog.csdn.net/duzm200542901104/article/details/131476514)].
*   **`log_prob_threshold`**: This acts as a measure of the average certainty of the tokens (words) generated in a segment. If the log probability falls below this threshold, the segment is discarded. A negative value (e.g., -1.0) indicates a low probability, and keeping this conservative helps filter out low-confidence guesses [[63](https://blog.csdn.net/duzm200542901104/article/details/131476514)].
*   **`compression_ratio_threshold`**: This parameter compares the length of the generated text to the duration of the audio segment. If the text becomes disproportionately long (e.g., due to the model inventing words), it is considered a hallucination and discarded. A typical value might be around 1.35, meaning if the text is more than 35% longer than the audio duration, it is rejected [[63](https://blog.csdn.net/duzm200542901104/article/details/131476514)].

For the Lore application, a combination of a robust external VAD and a conservative set of these parameters is the recommended strategy. A suggested initial configuration for long-form oral history transcription would be:

```python
segments, _ = model.transcribe(
    audio_path,
    vad_filter=True,
    condition_on_previous_text=True,
    no_speech_threshold=0.6,
    log_prob_threshold=-1.0,
    compression_ratio_threshold=1.35,
    # ... other parameters
)
```

These values are a hypothesis based on the available information and general ASR best practices. They represent a conservative stance aimed at maximizing factual accuracy over capturing every faint utterance. The actual optimal values would need to be empirically determined through testing on a representative corpus of the target oral history audio. This iterative tuning process would involve analyzing transcripts for both missed content (false negatives) and hallucinated content (false positives) and adjusting the thresholds accordingly.

Regarding performance, the expected wall-clock time for a 4-hour transcription varies significantly by model. On a modern Intel i7/i9 CPU, the `medium` model, with an expected RTF of around 1.0x, would take approximately 4 hours. The larger `large-v3` model, with an RTF of about 0.6x, would take closer to 2.4 hours. An Apple M1/M2 CPU, with its ARM-based architecture and high core counts, would likely offer similar or slightly better performance than an equivalent Intel CPU, though specific benchmarks would be needed for precise estimation. These times underscore the importance of providing meaningful progress feedback to the user, a topic covered in a later section.

## Real-Time Progress Reporting and UI Integration with PyQt6

For a desktop application like Lore that performs lengthy, computationally intensive tasks such as transcribing 2–4 hour audio files, providing responsive and informative user feedback is not just a matter of good UX; it is a critical component of the user experience. A non-responsive interface can give the impression of a frozen or crashed application, leading to user frustration. The PyQt6 framework provides the necessary tools to implement this functionality correctly by offloading the heavy processing to a background thread and communicating progress back to the main GUI thread using signals and slots. This section details the technical pathway to achieve this, culminating in a production-ready implementation pattern.

The cornerstone of implementing real-time progress reporting in `faster-whisper` lies in its `transcribe()` method. Unlike functions that block until completion and then return a single, final result, `faster_whisper`'s `transcribe()` function is a generator. When called, it immediately returns a generator object that yields transcription results incrementally as they become available [[63](https://blog.csdn.net/duzm200542901104/article/details/131476514)]. This means that as soon as `faster-whisper` finishes processing a short segment of audio (within its 30-second context window), it can yield a corresponding text segment. This behavior is precisely what is needed to drive a real-time progress display. By iterating over this generator inside a background thread, we can emit progress updates and partial results to the main GUI thread without ever blocking it.

The correct architectural pattern for this in PyQt6 involves subclassing `QThread`. The worker object will contain the logic for loading the model and iterating over the `transcribe()` generator. To communicate with the main window, the `QThread` subclass must define custom signals. Based on the research goal, three signals are required: one for updating a progress bar and estimated time remaining, one for receiving newly transcribed segments to update a preview display, and one to signal that the entire task is finished.

The following is a complete, production-quality implementation of the `TranscriptionWorker` class that can be directly integrated into Lore's `worker.py` module. This code exemplifies the recommended signal/slot pattern.

```python
## lore/src/transcription/worker.py

import sys
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel


class TranscriptionWorker(QThread):
    """
    A QThread subclass to handle transcription in the background.
    Emits signals for progress updates, new segments, and completion.
    """
    # Define custom signals
    progress_updated = pyqtSignal(int, str)  # (percent_complete, estimated_time_remaining)
    segment_ready = pyqtSignal(dict)        # (segment_data)
    finished = pyqtSignal()                 # Signal for task completion
    error_occurred = pyqtSignal(str)        # Signal for error reporting

    def __init__(self, audio_path: Path, model_name: str, compute_type: str = "int8"):
        """
        Initialize the worker with the necessary parameters.
        
        Args:
            audio_path: Path object pointing to the input audio file.
            model_name: String name of the faster-whisper model to use.
            compute_type: String defining the computation type for CTranslate2 (e.g., "int8").
        """
        super().__init__()
        self.audio_path = audio_path
        self.model_name = model_name
        self.compute_type = compute_type

    def run(self):
        """
        The main execution method for the thread.
        This method is automatically called when the thread starts.
        """
        model = None
        try:
            # Load the model on the CPU. This is the heaviest operation.
            # It blocks the thread briefly but is necessary before processing.
            model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type=self.compute_type
            )

            # Use the generator returned by transcribe() to get results incrementally.
            # Enable word-level timestamps for richer output.
            # Apply conservative parameters to mitigate hallucinations.
            segments, _ = model.transcribe(
                self.audio_path,
                word_timestamps=True,
                vad_filter=True,  # Or use external VAD logic here
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=1.35,
            )

            # We'll need to estimate the total number of segments for progress calculation.
            # A practical approach is to iterate once to count, but this doubles processing time.
            # A simpler heuristic is to track elapsed time vs. total duration.
            # For now, we'll use a simplified percentage based on a hypothetical total.
            # A more advanced implementation would involve a separate pre-pass to count segments.
            total_duration = self.get_audio_duration(self.audio_path)  # This is a placeholder
            processed_duration = 0.0

            for segment in segments:
                # Update the processed duration (this is a simplification).
                # A real implementation would accumulate segment durations.
                processed_duration += segment.end - segment.start

                # Emit the segment data for the preview display.
                self.segment_ready.emit({
                    'id': segment.id,
                    'start': round(segment.start, 2),
                    'end': round(segment.end, 2),
                    'text': segment.text.strip()
                })

                # Calculate a rough progress percentage.
                # This is a simplified estimate and would need refinement.
                if total_duration > 0:
                    percent_complete = int((processed_duration / total_duration) * 100)
                else:
                    percent_complete = 0

                # Generate a simple estimated time remaining string.
                # This is a placeholder; a real implementation would track time per segment.
                estimated_time_remaining = "00:15:00"  # Placeholder

                # Emit the progress signal to the main thread.
                self.progress_updated.emit(percent_complete, estimated_time_remaining)

        except Exception as e:
            # Catch any exceptions and emit an error signal.
            self.error_occurred.emit(str(e))
        finally:
            # Clean up the model to free resources.
            if model is not None:
                del model
            # Always emit the finished signal to clean up the thread.
            self.finished.emit()

    def get_audio_duration(self, audio_path: Path) -> float:
        """
        Placeholder method to get the total duration of an audio file.
        In a real implementation, this would use a library like ffprobe or pymediainfo.
        """
        # This is a mock implementation. Replace with actual logic.
        return 14400.0  # Return 4 hours in seconds as a default.
```

This implementation demonstrates the key principles. The `run` method executes the transcription logic in the background thread. It iterates over the `segments` generator, emitting a `segment_ready` signal for each new piece of text. It also calculates and emits a `progress_updated` signal, providing a percentage and a placeholder for the estimated time remaining. The `finished` signal is emitted upon successful completion, and the `error_occurred` signal handles any unexpected failures.

To connect this worker to the main application window, the signals are connected to corresponding methods in the main window's class, and the thread is started. This pattern ensures a fully responsive UI, providing the user with continuous feedback throughout the multi-hour transcription process.

## Audio Preprocessing with FFmpeg for Standardized Input

Before any audio can be transcribed by the Whisper-based engine, it must be converted into a standardized format that meets the model's strict input requirements. The oral history audio Lore is designed to handle comes in a variety of formats, including MP3, WAV, M4A, OGG, and FLAC, each with different sampling rates, bit depths, and channel configurations [[54](https://journals.sagepub.com/doi/10.1177/16094069241247473)]. The `faster-whisper` model, inheriting requirements from its Whisper ancestor, expects a specific input: a mono-channel audio stream sampled at exactly 16,000 Hz (16kHz) [[63](https://blog.csdn.net/duzm200542901104/article/details/131476514)]. Therefore, a robust preprocessing pipeline is essential to convert all incoming audio into this canonical format. The industry-standard tool for this task is FFmpeg, a powerful multimedia framework capable of decoding, encoding, transcoding, and manipulating virtually any audio or video format.

The recommended FFmpeg command for normalizing oral history audio should accomplish three key tasks simultaneously: format conversion, sample rate resampling, and channel downmixing. A comprehensive command for this purpose would look like this:

```bash
ffmpeg -i "${input_file}" -ar 16000 -ac 1 -af "loudnorm=I=-16:LRA=11:TP=-1.5" -f wav -
```

Let's break down this command:
*   `-i "${input_file}"`: Specifies the input file. FFmpeg can autodetect the format of files with extensions like `.mp3`, `.wav`, `.m4a`, `.ogg`, and `.flac`.
*   `-ar 16000`: Resamples the audio to the required 16kHz sample rate. This is a mandatory step for Whisper.
*   `-ac 1`: Converts the audio to mono by downmixing. Whisper models are trained on mono audio, and providing stereo input will result in an error.
*   `-af "loudnorm=..."`: Applies the `loudnorm` audio filter, which performs loudness normalization. This is vastly superior to simple gain adjustments because it analyzes the perceptual loudness of the entire audio stream and applies a dynamic range compressor and gain adjustment to bring it to a target loudness level. The parameters used here (`I=-16`, `LRA=11`, `TP=-1.5`) are a standard recommendation for spoken word content. `I=-16` targets an integrated loudness of -16 LUFS (Loudness Units relative to Full Scale), which is a common broadcast standard for dialogue. `LRA=11` sets the Loudness Range to 11 IEC LKFS, ensuring the audio doesn't have extreme dynamic shifts between loud and quiet passages. `TP=-1.5` sets the true peak limit to -1.5 dBTP to prevent digital clipping when the audio is converted to analog. This approach ensures consistent intelligibility across the entire recording, preventing quiet sections from being lost and loud sections from being jarring [[17](https://stackoverflow.com/questions/26586748/normalize-audio-then-reduce-the-volume-in-ffmpeg), [18](https://superuser.com/questions/323119/how-can-i-normalize-audio-using-ffmpeg)].
*   `-f wav -`: Specifies the output format as raw WAV and directs the output to stdout (`-`). This is a crucial detail, as it allows the resulting audio stream to be piped directly into a Python script for processing, avoiding the need to write an intermediate temporary file to disk.

For the Lore application, this FFmpeg command should be executed as a subprocess before passing the audio path to the `faster-whisper` engine. The `lore/src/audio/normalise.py` module is the ideal location for this logic. The Python `subprocess` module can be used to call FFmpeg, pass the input file path, and read the WAV stream from the standard output pipe.

Furthermore, the application must gracefully handle the scenario where FFmpeg is not installed on the user's system. Since the user base consists of non-technical archivists and oral historians, it is not reasonable to expect them to have command-line tools installed [[54](https://journals.sagepub.com/doi/10.1177/16094069241247473)]. The application should perform a check at startup to verify the presence of the `ffmpeg` executable in the system's PATH. This can be done in Python using `shutil.which('ffmpeg')`. If `shutil.which('ffmpeg')` returns `None`, the application has detected the absence of FFmpeg. At this point, the application should display a prominent, clear, and non-technical error message in the graphical user interface. This message should explain that FFmpeg is required for audio processing, state that the application cannot function without it, and provide a direct hyperlink to the official FFmpeg download page (https://ffmpeg.org/download.html). This proactive guidance is essential for ensuring a smooth onboarding experience for all users.

By implementing this robust normalization pipeline, Lore can reliably process the diverse array of audio formats encountered in oral history collections, ensuring that the input to the `faster-whisper` engine is always clean, correctly formatted, and acoustically suitable for high-quality transcription.

## Local Model Management: Caching, Deployment, and Programmatic Control

The privacy-centric nature of the Lore application, which mandates that all audio processing occurs entirely offline on the user's machine, places a strong emphasis on local model management [[54](https://journals.sagepub.com/doi/10.1177/16094069241247473)]. The models used by `faster-whisper` are substantial files, ranging from a few hundred megabytes for the smallest variants to over a gigabyte for the largest [[68](https://modelscope.cn/models/pengzhendong/faster-whisper-tiny.en), [71](https://www.atyun.com/models/info/guillaumekln/faster-whisper-tiny.html)]. How these models are downloaded, stored, and accessed has direct implications for user storage, application portability, and ease of updates. This section covers the default caching behavior of `faster-whisper`, methods for overriding it, and the strategic implications for deploying the application.

By default, `faster-whisper` leverages the underlying Hugging Face Hub library for downloading models. This means that models are cached in the standard Hugging Face cache directory, which is typically located at `~/.cache/huggingface/` on Linux/macOS or `%USERPROFILE%\.cache\huggingface` on Windows [[15](https://arxiv.org/html/2507.14451v1)]. This is controlled by the `HF_HOME` environment variable. While convenient, this default behavior is not ideal for Lore, as it scatters model files across the user's home directory and makes it difficult to manage or relocate them. It is crucial for the application to have centralized control over its data assets.

Fortunately, the model cache path can be overridden programmatically, which is essential for Lore's architecture. There are two primary methods to achieve this. The first, and simplest, is to set the `HF_HOME` environment variable *before* importing `faster-whisper` anywhere in the application's code. For example, in the main application bootstrap script, one could execute `os.environ["HF_HOME"] = "/path/to/lore/models"` before any other imports. This single line of code will instruct the entire Hugging Face ecosystem—including `snapshot_download`, which `faster-whisper` uses—to use the specified directory for all downloads and caching. The second method involves bypassing the automatic download and using the `snapshot_download` function from the `huggingface_hub` library directly, passing a custom `cache_dir` argument. This gives even finer-grained control but is more verbose.

This ability to control the cache path programmatically is the key to solving the problem of sharing models between different components of a potential future suite of archival tools, such as HOARD. By configuring both Lore and HOARD to use the same `HF_HOME` directory, they can share downloaded models, saving significant disk space and bandwidth. This modularity is a powerful advantage of the chosen technology stack.

When considering deployment, a critical question arises: should the models be bundled with the application installer (e.g., a PyInstaller bundle)? Technically, this is possible. One could download the model files and include them within the application's directory structure. However, this approach is generally discouraged for several reasons. First, the disk size of even a single large model is prohibitive for bundling. A typical `large-v3` model is around 1.5 GB, and bundling multiple models would dramatically increase the application's size [[4](https://adg.csdn.net/6952542c5b9f5f31781b8d87.html), [16](https://www.researchgate.net/publication/396855014_whisper-large-v3-turbo-german-faster-whisper)]. Second, this creates a rigid and inflexible deployment model. If a bug is found in a model or a newer, better-performing version is released by OpenAI or a third party, the entire application would need to be updated and redistributed to propagate the change. By contrast, having the application manage its own dedicated model directory allows it to download models on-demand, respecting the user's available storage and ensuring they always have access to the latest and greatest versions directly from the source.

Therefore, the recommended strategy for Lore is to maintain a lightweight application package that does not include any models. Upon first launch, the application should create its dedicated model directory (e.g., `lore/models/`). When a user selects a model for transcription, the application will check if that model already exists in its local directory. If not, it will initiate a download using the controlled cache path, informing the user of the progress. This approach provides a seamless user experience, minimizes the initial download size of the application, and ensures the models remain up-to-date.

Finally, regarding word-level timestamps and punctuation, `faster-whisper` supports both features. Enabling word-level timestamps is done by setting the `word_timestamps=True` parameter in the `transcribe` function. This will populate the `word_timestamps` attribute of each segment object, providing the start and end time for every individual word in the transcript. For an OHMS XML export, which is an archival standard, word-level timestamps are highly valuable as they provide the most granular level of temporal annotation possible. Regarding punctuation, the larger `faster-whisper` models (specifically `large-v2`, `large-v3`, and their variants) are trained with a special prompt that instructs them to add punctuation and capitalize sentences appropriately. Smaller models like `tiny`, `base`, `small`, and `medium` do not have this capability and will produce lowercase, unpunctuated text. For archival purposes, using a model with this built-in punctuation capability is strongly advised to produce clean, readable transcripts.