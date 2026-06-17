from pathlib import Path
import platformdirs
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm


class ModelManager:
    """
    Manages the local model cache for faster-whisper.
    We use the huggingface_hub library natively because it supports
    resumable downloads and robust caching.
    """

    VAD_FILENAME = "silero_vad.onnx"

    # Whisper tiers use faster-whisper's native size strings so model
    # resolution is handled by the library itself (no HF repo ID needed).
    # This is more maintainable than tracking community CTranslate2 repos
    # and will automatically pick up official Systran releases when available.
    WHISPER_SIZES = {
        "Fast": "small",
        "Balanced": "medium",
        "Best Quality": "turbo",
    }

    # Whisper HF repo IDs — hardcoded to avoid depending on faster-whisper's
    # private _MODELS dict and to bypass its disabled_tqdm wrapper.
    WHISPER_REPO_IDS = {
        "small": "Systran/faster-whisper-small",
        "medium": "Systran/faster-whisper-medium",
        "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    }

    # Maps Settings UI tier names to Whisper size strings
    _TIER_TO_WHISPER = {
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
        "VAD": "snakers4/silero-vad",
        "WeSpeaker": "onnx-community/wespeaker-voxceleb-resnet34-LM",
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

    @staticmethod
    def _get_cache_dir(model_key: str) -> Path:
        """
        Returns the appropriate cache directory for a model key.
        NER/LLM/Translation use their own cache dir; others share the default.
        """
        if model_key in ("NER", "LLM", "Translation"):
            return ModelManager.get_ner_cache_dir()
        return ModelManager.get_cache_dir()

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
        elif quality_tier == "VAD":
            model_path = snapshot_download(
                repo_id=identifier,
                cache_dir=cache_dir,
                local_files_only=False,
                allow_patterns=["silero_vad.onnx"],
            )
        elif quality_tier == "WeSpeaker":
            model_path = snapshot_download(
                repo_id=identifier,
                cache_dir=cache_dir,
                local_files_only=False,
                allow_patterns=["model.onnx"],
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
            elif quality_tier == "VAD":
                path = snapshot_download(
                    repo_id=identifier,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    allow_patterns=["silero_vad.onnx"],
                )
            elif quality_tier == "WeSpeaker":
                path = snapshot_download(
                    repo_id=identifier,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    allow_patterns=["model.onnx"],
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

    @classmethod
    def prefetch(cls, quality_tier: str, progress_callback=None):
        """Downloads all models for the given quality tier.

        Downloads the Whisper model for this tier plus all shared hub models
        (YAMNet, NER, LLM, Translation, Segmentation). Prints progress headers
        and handles errors per model without aborting the batch.
        """
        whisper_size = cls._TIER_TO_WHISPER.get(quality_tier)

        if whisper_size:
            repo_id = cls.WHISPER_REPO_IDS[whisper_size]
            print(f"Downloading Whisper ({whisper_size})...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cls.get_cache_dir(),
                    tqdm_class=tqdm,
                    allow_patterns=[
                        "config.json",
                        "preprocessor_config.json",
                        "model.bin",
                        "tokenizer.json",
                        "vocabulary.*",
                    ],
                )
            except Exception as e:
                print(f"  Error downloading {repo_id}: {e}")

        # Hub models shared across all tiers
        hub_keys = ["YAMNet", "NER", "LLM", "Translation", "Segmentation", "VAD", "WeSpeaker"]
        for key in hub_keys:
            repo_id = cls.MODELS[key]
            cache_dir = cls._get_cache_dir(key)
            print(f"Downloading {key}...")
            try:
                if key == "YAMNet":
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        tqdm_class=tqdm,
                        allow_patterns=["yamnet.onnx", "yamnet_class_map.csv"],
                    )
                elif key == "VAD":
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        tqdm_class=tqdm,
                        allow_patterns=["silero_vad.onnx"],
                    )
                elif key == "WeSpeaker":
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        tqdm_class=tqdm,
                        allow_patterns=["model.onnx"],
                    )
                elif key == "LLM":
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        tqdm_class=tqdm,
                        allow_patterns=["*q4_k_m.gguf"],
                    )
                else:
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        tqdm_class=tqdm,
                    )
            except Exception as e:
                print(f"  Error downloading {repo_id}: {e}")

    @classmethod
    def prefetch_status(cls, quality_tier: str) -> dict[str, bool]:
        """Returns {model_name: is_cached} for all models in the given tier.

        Checks Whisper via snapshot_download with local_files_only=True and
        hub models via the same pattern as get_model_path(). Never raises.
        """
        result: dict[str, bool] = {}
        whisper_size = cls._TIER_TO_WHISPER.get(quality_tier)

        if whisper_size:
            repo_id = cls.WHISPER_REPO_IDS[whisper_size]
            try:
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cls.get_cache_dir(),
                    local_files_only=True,
                    allow_patterns=["model.bin"],
                )
                result[f"whisper_{whisper_size}"] = True
            except Exception:
                result[f"whisper_{whisper_size}"] = False

        hub_keys = ["YAMNet", "NER", "LLM", "Translation", "Segmentation", "VAD", "WeSpeaker"]
        for key in hub_keys:
            repo_id = cls.MODELS[key]
            cache_dir = cls._get_cache_dir(key)
            try:
                if key == "YAMNet":
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        allow_patterns=["yamnet.onnx", "yamnet_class_map.csv"],
                    )
                elif key == "VAD":
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        allow_patterns=["silero_vad.onnx"],
                    )
                elif key == "WeSpeaker":
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        allow_patterns=["model.onnx"],
                    )
                elif key == "LLM":
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        allow_patterns=["*q4_k_m.gguf"],
                    )
                else:
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        local_files_only=True,
                    )
                result[key] = True
            except Exception:
                result[key] = False

        return result
