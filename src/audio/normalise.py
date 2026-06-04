import subprocess
from pathlib import Path
import imageio_ffmpeg

class AudioNormaliseError(Exception):
    """Raised when audio normalization fails."""
    pass

def normalise(input_path: Path, output_path: Path) -> None:
    """
    Normalise audio to 16kHz mono PCM with EBU R128 loudness normalisation.
    This is required for optimal faster-whisper transcription of archival audio.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    cmd = [
        ffmpeg_exe,
        "-y",               # Overwrite output files
        "-i", str(input_path),
        "-af", "loudnorm",  # EBU R128 loudness normalization
        "-ar", "16000",     # 16 kHz sample rate
        "-ac", "1",         # Mono (1 channel)
        "-acodec", "pcm_s16le", # 16-bit PCM
        str(output_path)
    ]
    
    try:
        # Run ffmpeg subprocess, capturing output for error reporting
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise AudioNormaliseError(f"FFmpeg failed with exit code {e.returncode}:\n{e.stderr}")
    except Exception as e:
        raise AudioNormaliseError(f"Failed to execute FFmpeg: {str(e)}")
