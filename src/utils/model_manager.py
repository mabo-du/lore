from pathlib import Path
import platformdirs
from huggingface_hub import snapshot_download


class ModelManager:
    """
    Manages the local model cache for faster-whisper.
    We use the huggingface_hub library natively because it supports
    resumable downloads and robust caching.
    """

    # Whisper tiers use faster-whisper's native size strings so model
    # resolution is handled by the library itself (no HF repo ID needed).
    # This is more maintainable than tracking community CTranslate2 repos
    # and will automatically pick up official Systran releases when available.
    WHISPER_SIZES = {
        "Fast": "small",
        "Balanced": "medium",
        "Best Quality": "turbo",
    }

    MODELS = {
        **WHISPER_SIZES,
        "YAMNet": "zeropointnine/yamnet-onnx",
        "NER": "lmo3/gliner2-large-v1-onnx",
        "LLM": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "Translation": "JustFrederik/nllb-200-distilled-600M-ct2-int8",
        "Segmentation": "onnx-community/pyannote-segmentation-3.0",
    }

    @staticmethod
    def get_cache_dir() -> Path:
        """
        Returns the shared cache directory.
        We use a generic 'heritage-tools' directory so models can be shared with HOARD.
        """
        # Using user_data_dir for cross-app sharing
        base = Path(
            platformdirs.user_data_dir(appname="heritage-tools", appauthor=False)
        )
        models_dir = base / "whisper-models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    @staticmethod
    def get_ner_cache_dir() -> Path:
        """
        Returns the cache directory for NER models, specific to Lore.
        """
        base = Path(platformdirs.user_data_dir(appname="heritage-tools", appauthor=False))
        models_dir = base / "models" / "ner"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    @classmethod
    def ensure_model(cls, quality_tier: str = "Best Quality") -> str:
        """
        Ensures the model is downloaded. In a CLI context this just blocks.
        In a GUI context, we would run this inside a QThread.

        Returns a local path string for non-Whisper models, or a Whisper
        size string (e.g. "turbo") for Whisper tiers — faster-whisper's
        WhisperModel() accepts size strings natively and handles HF
        download internally.
        """
        identifier = cls.MODELS.get(quality_tier, cls.MODELS["Best Quality"])

        # Whisper tiers use size strings — faster-whisper handles resolution
        if quality_tier in cls.WHISPER_SIZES:
            return identifier

        # All other models: download via huggingface_hub
        if quality_tier in ["NER", "LLM", "Translation"]:
            cache_dir = cls.get_ner_cache_dir()
        else:
            cache_dir = cls.get_cache_dir()

        if quality_tier == "YAMNet":
            model_path = snapshot_download(
                repo_id=identifier,
                cache_dir=cache_dir,
                local_files_only=False,
                allow_patterns=["yamnet.onnx", "yamnet_class_map.csv"],
            )
        elif quality_tier == "LLM":
            model_path = snapshot_download(
                repo_id=identifier,
                cache_dir=cache_dir,
                local_files_only=False,
                allow_patterns=["*q4_k_m.gguf"],
            )
        else:
            model_path = snapshot_download(
                repo_id=identifier, cache_dir=cache_dir, local_files_only=False
            )
        return model_path

    @classmethod
    def get_model_path(cls, quality_tier: str = "Best Quality") -> Path | None:
        """Returns the local path if it exists, otherwise None.

        For Whisper tiers (size strings like "turbo") this returns None
        since faster-whisper handles resolution internally."""
        identifier = cls.MODELS.get(quality_tier, cls.MODELS["Best Quality"])

        # Whisper tiers don't map to a local path — handled by faster-whisper
        if quality_tier in cls.WHISPER_SIZES:
            return None

        if quality_tier in ["NER", "LLM", "Translation"]:
            cache_dir = cls.get_ner_cache_dir()
        else:
            cache_dir = cls.get_cache_dir()

        try:
            if quality_tier == "YAMNet":
                path = snapshot_download(
                    repo_id=identifier,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    allow_patterns=["yamnet.onnx", "yamnet_class_map.csv"],
                )
            elif quality_tier == "LLM":
                path = snapshot_download(
                    repo_id=identifier,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    allow_patterns=["*q4_k_m.gguf"],
                )
            else:
                path = snapshot_download(
                    repo_id=identifier, cache_dir=cache_dir, local_files_only=True
                )
            return Path(path)
        except Exception:
            return None
