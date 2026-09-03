from __future__ import annotations

import json
from typing import Any


CHAT_ROLES = ("host", "player", "spectator")
DEFAULT_CHAT_ROLES = CHAT_ROLES


def normalize_chat_roles(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_CHAT_ROLES)
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError("invalid_chat_roles")
    selected = {str(role or "").strip().lower() for role in value}
    if any(role not in CHAT_ROLES for role in selected):
        raise ValueError("invalid_chat_roles")
    return [role for role in CHAT_ROLES if role in selected]


def decode_chat_roles(value: Any) -> list[str]:
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
        return normalize_chat_roles(decoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        return list(DEFAULT_CHAT_ROLES)


def effective_chat_role(
    *, room: Any, member: Any, user_id: int | None = None, actor_key: str | None = None
) -> str:
    identity_key = str(actor_key or "") or (f"u:{int(user_id)}" if user_id is not None else "")
    room_host = str(room["host_actor_key"] or f"u:{room['host_user_id']}")
    if identity_key and room_host == identity_key:
        return "host"
    return str(member["role"] or "").strip().lower()
