import json
from pathlib import Path
from platformdirs import user_data_dir

# Fallback basic list if the user hasn't run the pre-validation script yet
DEFAULT_LANGUAGES = {
    "eng_Latn": {"name": "English", "tier": "High Quality"},
    "spa_Latn": {"name": "Spanish", "tier": "High Quality"},
    "zho_Hans": {"name": "Chinese (Simplified)", "tier": "High Quality"},
    "zho_Hant": {"name": "Chinese (Traditional)", "tier": "High Quality"},
    "arb_Arab": {"name": "Arabic", "tier": "High Quality"},
    "fra_Latn": {"name": "French", "tier": "High Quality"},
    "mri_Latn": {"name": "Maori", "tier": "Usable"},
    "cym_Latn": {"name": "Welsh", "tier": "Usable"},
    "nav_Latn": {"name": "Navajo", "tier": "Experimental"},
    # We will add others dynamically or from pre-validation
}


def get_supported_languages():
    """
    Returns a dict of NLLB FLORES-200 codes mapping to Name and Quality Tier.
    Loads from a pre-validated JSON file if it exists, otherwise uses defaults.
    """
    config_path = Path(user_data_dir("lore", "lore_app")) / "language_tiers.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_LANGUAGES
