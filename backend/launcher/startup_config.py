from __future__ import annotations

import locale
import os
import pickle
from pathlib import Path

from backend.resource_paths import get_resource_path


def load_startup_config() -> dict[str, object]:
    config_path = Path(get_resource_path(os.path.join("docs_and_configs", "config")))
    try:
        with config_path.open("rb") as config_file:
            config = pickle.load(config_file)
    except Exception:
        return {}

    return config if isinstance(config, dict) else {}


def startup_uses_dark_mode() -> bool:
    return bool(load_startup_config().get("dark_mode", False))


def system_startup_language() -> str:
    lang_code, _ = locale.getlocale()
    if not lang_code:
        try:
            lang_code = locale.getdefaultlocale()[0]
        except Exception:
            lang_code = None

    normalized = str(lang_code or "").strip().lower()
    if normalized.startswith("zh"):
        return "zh"
    return "en"


def startup_language() -> str:
    return system_startup_language()

