"""Small, account-owned presentation and human-play preferences."""

from __future__ import annotations

import json
import re
from pathlib import Path

from backend.auth.db import auth_db

from .service import iso


THEMES_PATH = Path(__file__).resolve().parents[2] / "docs_and_configs" / "themes.json"
THEME_NAMES = frozenset(json.loads(THEMES_PATH.read_text(encoding="utf-8")))
BOOLEAN_KEYS = frozenset({
    "dark_mode", "use_custom_theme", "do_animation",
    "alwaysConfirmRestart", "showSpeed", "showFourPercent",
    "share_play_analysis",
})
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
ALLOWED_KEYS = BOOLEAN_KEYS | {
    "language", "theme", "custom_colors", "font_size_factor", "ui_scale", "saved_theme_id",
}


def validate_changes(changes: object) -> dict:
    if not isinstance(changes, dict) or len(changes) > len(ALLOWED_KEYS):
        raise ValueError("invalid_preferences")
    if set(changes) - ALLOWED_KEYS:
        raise ValueError("invalid_preferences")
    if len(json.dumps(changes, separators=(",", ":"))) > 4096:
        raise ValueError("invalid_preferences")
    result = {}
    for key, value in changes.items():
        if key in BOOLEAN_KEYS:
            if type(value) is not bool:
                raise ValueError("invalid_preferences")
        elif key == "language":
            if value not in ("en", "zh"):
                raise ValueError("invalid_preferences")
        elif key == "theme":
            if not isinstance(value, str) or value not in THEME_NAMES:
                raise ValueError("invalid_preferences")
        elif key == "custom_colors":
            if not isinstance(value, list) or len(value) != 36 or any(
                not isinstance(color, str) or not COLOR_RE.fullmatch(color) for color in value
            ):
                raise ValueError("invalid_preferences")
        elif key == "font_size_factor":
            if type(value) is not int or value < 50 or value > 150 or value % 5:
                raise ValueError("invalid_preferences")
        elif key == "ui_scale":
            if type(value) is not int or value < 90 or value > 125 or value % 5:
                raise ValueError("invalid_preferences")
        elif key == "saved_theme_id":
            if type(value) is not int or value < 0:
                raise ValueError("invalid_preferences")
        result[key] = value
    return result


def get_preferences(user_id: int) -> dict:
    with auth_db() as db:
        row = db.execute(
            "SELECT preferences_json,revision FROM user_preferences WHERE user_id=?",
            (int(user_id),),
        ).fetchone()
    return {"preferences": json.loads(row[0]) if row else {}, "revision": int(row[1]) if row else 0}


def patch_preferences(user_id: int, changes: object, *, only_if_missing: bool = False) -> dict:
    validated = validate_changes(changes)
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        saved_theme_id = validated.get("saved_theme_id")
        if saved_theme_id and db.execute(
            "SELECT 1 FROM user_saved_themes WHERE id=? AND user_id=?", (saved_theme_id, int(user_id))
        ).fetchone() is None:
            raise ValueError("invalid_preferences")
        row = db.execute(
            "SELECT preferences_json,revision FROM user_preferences WHERE user_id=?",
            (int(user_id),),
        ).fetchone()
        current = json.loads(row[0]) if row else {}
        applied = {key: value for key, value in validated.items()
                   if not only_if_missing or key not in current}
        if applied:
            current.update(applied)
            revision = int(row[1]) + 1 if row else 1
            db.execute(
                "INSERT INTO user_preferences(user_id,preferences_json,revision,updated_at) "
                "VALUES(?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET "
                "preferences_json=excluded.preferences_json,revision=excluded.revision,"
                "updated_at=excluded.updated_at",
                (int(user_id), json.dumps(current, separators=(",", ":")), revision, iso()),
            )
        else:
            revision = int(row[1]) if row else 0
    return {"preferences": current, "revision": revision}
