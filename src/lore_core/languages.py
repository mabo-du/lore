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


# Map from Whisper ISO 639-1 language codes to NLLB FLORES-200 codes
# Covers the most common transcription languages
WHISPER_TO_NLLB_MAP = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ur": "urd_Arab",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "uk": "ukr_Cyrl",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "no": "nob_Latn",
    "fi": "fin_Latn",
    "cs": "ces_Latn",
    "hu": "hun_Latn",
    "ro": "ron_Latn",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "tl": "tgl_Latn",
    "mi": "mri_Latn",
    "cy": "cym_Latn",
    "nv": "nav_Latn",
    "fa": "pes_Arab",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "mr": "mar_Deva",
    "te": "tel_Telu",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ne": "npi_Deva",
    "si": "sin_Sinh",
    "km": "khm_Khmr",
    "my": "mya_Mymr",
    "mn": "khk_Cyrl",
    "ka": "kat_Geor",
    "hy": "hye_Armn",
    "az": "azj_Latn",
    "uz": "uzn_Latn",
    "kk": "kaz_Cyrl",
    "af": "afr_Latn",
    "zu": "zul_Latn",
    "am": "amh_Ethi",
    "sr": "srp_Cyrl",
    "hr": "hrv_Latn",
    "bg": "bul_Cyrl",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "et": "est_Latn",
    "is": "isl_Latn",
    "mt": "mlt_Latn",
    "sq": "als_Latn",
    "mk": "mkd_Cyrl",
    "bs": "bos_Latn",
    "ga": "gle_Latn",
    "eu": "eus_Latn",
    "gl": "glg_Latn",
    "ca": "cat_Latn",
    "fy": "fry_Latn",
    "lb": "ltz_Latn",
    "oc": "oci_Latn",
    "yi": "ydd_Hebr",
    "be": "bel_Cyrl",
    "jv": "jav_Latn",
    "su": "sun_Latn",
    "ceb": "ceb_Latn",
    "ilo": "ilo_Latn",
    "haw": "haw_Latn",
    "smo": "smo_Latn",
    "tet": "tet_Latn",
    "la": "lat_Latn",
}


def get_supported_languages():
    """
    Returns a dict of NLLB FLORES-200 codes mapping to Name and Quality Tier.
    Loads from a pre-validated JSON file if it exists, otherwise uses defaults.
    """
    config_path = Path(user_data_dir(appname="heritage-tools", appauthor=False)) / "language_tiers.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_LANGUAGES
