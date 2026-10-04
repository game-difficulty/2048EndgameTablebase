"""Account-owned 2048Verse-compatible theme files."""

from __future__ import annotations

import json
import re
import unicodedata

from backend.auth.db import auth_db

from .service import iso


MAX_THEMES = 32
MAX_PAYLOAD_BYTES = 64 * 1024
TILE_KEYS = tuple(str(2 ** exponent) for exponent in range(1, 17))
OPTIONAL_KEYS = frozenset({"Super", "super"})
STYLE_KEYS = (
    "--tile-color",
    "--tile-background",
    "--tile-shadow-color",
    "--tile-outline-color",
)
COLOR_RE = re.compile(
    r"^#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2}|\s*/\s*(?:0(?:\.\d+)?|1(?:\.0+)?|\d{1,3}%))?$"
)


class ThemeError(ValueError):
    def __init__(self, code: str, status: int = 400):
        self.code = code
        self.status = status
        super().__init__(code)


def init_schema(db) -> None:
    db.execute("""CREATE TABLE IF NOT EXISTS user_saved_themes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        name_key TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(user_id,name_key),
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS user_saved_themes_user_updated ON user_saved_themes(user_id,updated_at DESC)")


def _name(value: object) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ThemeError("invalid_theme_name")
    name = unicodedata.normalize("NFKC", value).strip()
    if not 1 <= len(name) <= 64 or any(ord(char) < 32 or char in "/\\" for char in name):
        raise ThemeError("invalid_theme_name")
    return name, name.casefold()


def _color(value: object) -> str:
    if not isinstance(value, str) or len(value) > 40 or not COLOR_RE.fullmatch(value.strip()):
        raise ThemeError("invalid_theme_color")
    return value.strip()


def normalize_payload(value: object) -> dict:
    if not isinstance(value, dict) or not value or set(value) - {"light", "dark"}:
        raise ThemeError("invalid_theme_file")
    normalized: dict[str, dict] = {}
    for mode, entries in value.items():
        if not isinstance(entries, dict) or set(entries) - set(TILE_KEYS) - OPTIONAL_KEYS:
            raise ThemeError("invalid_theme_file")
        # An empty Verse palette means this mode inherits the supplied palette.
        if not entries:
            continue
        mode_result = {}
        for tile, style in entries.items():
            if not isinstance(style, dict) or set(style) != set(STYLE_KEYS):
                raise ThemeError("invalid_theme_file")
            canonical_tile = "Super" if tile == "super" else tile
            if tile == "super" and "Super" in entries:
                continue
            mode_result[canonical_tile] = {key: _color(style[key]) for key in STYLE_KEYS}
        if not all(tile in mode_result for tile in TILE_KEYS):
            raise ThemeError("theme_tiles_missing")
        normalized[mode] = mode_result
    if not normalized:
        raise ThemeError("theme_tiles_missing")
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ThemeError("theme_too_large", 413)
    return normalized


def _summary(row) -> dict:
    return {key: row[key] for key in ("id", "name", "created_at", "updated_at")}


def list_themes(user_id: int) -> dict:
    with auth_db() as db:
        init_schema(db)
        rows = db.execute(
            "SELECT id,name,created_at,updated_at FROM user_saved_themes WHERE user_id=? ORDER BY name_key,id",
            (int(user_id),),
        ).fetchall()
    return {"themes": [_summary(row) for row in rows], "limit": MAX_THEMES}


def get_theme(user_id: int, theme_id: int) -> dict:
    with auth_db() as db:
        init_schema(db)
        row = db.execute(
            "SELECT id,name,payload_json,created_at,updated_at FROM user_saved_themes WHERE id=? AND user_id=?",
            (int(theme_id), int(user_id)),
        ).fetchone()
    if row is None:
        raise ThemeError("theme_not_found", 404)
    return {**_summary(row), "theme": json.loads(row["payload_json"])}


def create_theme(user_id: int, raw_name: object, raw_theme: object) -> dict:
    name, name_key = _name(raw_name)
    theme = normalize_payload(raw_theme)
    payload = json.dumps(theme, ensure_ascii=False, separators=(",", ":"))
    now = iso()
    with auth_db() as db:
        init_schema(db)
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT count(*) FROM user_saved_themes WHERE user_id=?", (int(user_id),)).fetchone()[0] >= MAX_THEMES:
            raise ThemeError("theme_limit_reached", 409)
        try:
            cursor = db.execute(
                "INSERT INTO user_saved_themes(user_id,name,name_key,payload_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (int(user_id), name, name_key, payload, now, now),
            )
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                raise ThemeError("theme_name_exists", 409) from exc
            raise
        theme_id = cursor.lastrowid
    return get_theme(user_id, theme_id)


def update_theme(user_id: int, theme_id: int, raw_name: object, raw_theme: object) -> dict:
    name, name_key = _name(raw_name)
    theme = normalize_payload(raw_theme)
    payload = json.dumps(theme, ensure_ascii=False, separators=(",", ":"))
    with auth_db() as db:
        init_schema(db)
        try:
            cursor = db.execute(
                "UPDATE user_saved_themes SET name=?,name_key=?,payload_json=?,updated_at=? WHERE id=? AND user_id=?",
                (name, name_key, payload, iso(), int(theme_id), int(user_id)),
            )
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                raise ThemeError("theme_name_exists", 409) from exc
            raise
        if not cursor.rowcount:
            raise ThemeError("theme_not_found", 404)
    return get_theme(user_id, theme_id)


def delete_theme(user_id: int, theme_id: int) -> None:
    with auth_db() as db:
        init_schema(db)
        cursor = db.execute("DELETE FROM user_saved_themes WHERE id=? AND user_id=?", (int(theme_id), int(user_id)))
        if not cursor.rowcount:
            raise ThemeError("theme_not_found", 404)
        row = db.execute("SELECT preferences_json,revision FROM user_preferences WHERE user_id=?", (int(user_id),)).fetchone()
        if row:
            preferences = json.loads(row[0])
            if preferences.get("saved_theme_id") == int(theme_id):
                preferences["saved_theme_id"] = 0
                db.execute("UPDATE user_preferences SET preferences_json=?,revision=?,updated_at=? WHERE user_id=?",
                           (json.dumps(preferences, separators=(",", ":")), int(row[1]) + 1, iso(), int(user_id)))
