from pathlib import Path
import platformdirs
from huggingface_hub import snapshot_download


class ModelManager:
    """
    Manages the local model cache for faster-whisper.
    We use the huggingface_hub library natively because it supports
    resumable downloads and robust caching.
    """

    MODELS = {
        "Fast": "Systran/faster-whisper-small",
        "Balanced": "Systran/faster-whisper-medium",
        "Best Quality": "Systran/faster-whisper-large-v3-turbo",
        "YAMNet": "zeropointnine/yamnet-onnx",
        "NER": "lmo3/gliner2-large-v1-onnx",
        "LLM": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "Translation": "JustFrederik/nllb-200-distilled-600M-ct2-int8",
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
        base = Path(platformdirs.user_data_dir(appname="LoreProject", appauthor="Lore"))
        models_dir = base / "models" / "ner"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    @classmethod
    def ensure_model(cls, quality_tier: str = "Best Quality") -> str:
        """
        Ensures the model is downloaded. In a CLI context this just blocks.
        In a GUI context, we would run this inside a QThread.
        """
        repo_id = cls.MODELS.get(quality_tier, cls.MODELS["Best Quality"])
        if quality_tier in ["NER", "LLM", "Translation"]:
            cache_dir = cls.get_ner_cache_dir()  # Lore-specific models
        else:
            cache_dir = cls.get_cache_dir()
            cache_dir = cls.get_cache_dir()

        # snapshot_download will only download missing/updated files
        if quality_tier == "YAMNet":
            model_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=False,
                allow_patterns=["yamnet.onnx", "yamnet_class_map.csv"],
            )
        elif quality_tier == "LLM":
            model_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=False,
                allow_patterns=[
                    "*q4_k_m.gguf"
                ],  # Specifically target a small, solid quant
            )
        else:
            model_path = snapshot_download(
                repo_id=repo_id, cache_dir=cache_dir, local_files_only=False
            )
        return model_path

    @classmethod
    def get_model_path(cls, quality_tier: str = "Best Quality") -> Path | None:
        """Returns the local path if it exists, otherwise None"""
        repo_id = cls.MODELS.get(quality_tier, cls.MODELS["Best Quality"])
        if quality_tier in ["NER", "LLM", "Translation"]:
            cache_dir = cls.get_ner_cache_dir()
        else:
            cache_dir = cls.get_cache_dir()

        # Check if we can find it local_files_only
        try:
            if quality_tier == "YAMNet":
                path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    allow_patterns=["yamnet.onnx", "yamnet_class_map.csv"],
                )
            elif quality_tier == "LLM":
                path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    allow_patterns=["*q4_k_m.gguf"],
                )
            else:
                path = snapshot_download(
                    repo_id=repo_id, cache_dir=cache_dir, local_files_only=True
                )
            return Path(path)
        except Exception:
            return None
