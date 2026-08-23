from __future__ import annotations

import re
import unicodedata


DISPLAY_NAME_MIN_LENGTH = 2
DISPLAY_NAME_MAX_LENGTH = 24
RESERVED_DISPLAY_NAME_KEYS = {
    "2048tables",
    "admin",
    "administrator",
    "moderator",
    "official",
    "root",
    "staff",
    "support",
    "system",
}
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_display_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return _WHITESPACE_RE.sub(" ", normalized.strip())


def canonical_display_name_key(value: str) -> str:
    return normalize_display_name(value).casefold()


def validate_display_name(value: str) -> tuple[str, str]:
    normalized = normalize_display_name(value)
    if len(normalized) < DISPLAY_NAME_MIN_LENGTH:
        raise ValueError(
            f"Username must contain at least {DISPLAY_NAME_MIN_LENGTH} characters."
        )
    if len(normalized) > DISPLAY_NAME_MAX_LENGTH:
        raise ValueError(
            f"Username must contain at most {DISPLAY_NAME_MAX_LENGTH} characters."
        )

    has_letter_or_number = False
    for character in normalized:
        category = unicodedata.category(character)
        if category.startswith(("L", "N")):
            has_letter_or_number = True
            continue
        if character in {" ", "_", "-"}:
            continue
        raise ValueError(
            "Username may only contain letters, numbers, spaces, underscores, and hyphens."
        )
    if not has_letter_or_number:
        raise ValueError("Username must contain at least one letter or number.")

    key = normalized.casefold()
    if key in RESERVED_DISPLAY_NAME_KEYS:
        raise ValueError("This username is reserved.")
    return normalized, key

