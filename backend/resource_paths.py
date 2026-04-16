from __future__ import annotations

import os
import sys
from pathlib import Path


def get_resource_base_path() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent.parent


def get_resource_path(relative_path: str) -> str:
    base_path = get_resource_base_path()
    path = (base_path / relative_path).resolve()
    if path.exists():
        return str(path)

    fallback = Path(relative_path)
    if fallback.exists():
        return str(fallback.resolve())

    return str(path)
