from __future__ import annotations

from typing import Any


def room_host_actor_key(room: Any) -> str | None:
    value = str(room.get("host_actor_key") or "").strip()
    if value:
        return value
    user_id = room.get("host_user_id")
    return f"u:{int(user_id)}" if user_id is not None else None


def is_room_host(room: Any, actor_key: str) -> bool:
    return bool(actor_key) and room_host_actor_key(room) == str(actor_key)


def is_permanent_room(room: Any) -> bool:
    return str(room.get("lifecycle_kind") or "normal") == "permanent"


def is_platform_sponsored(room: Any) -> bool:
    return str(room.get("billing_policy") or "user") == "platform"


def room_billing_user_id(room: Any) -> int | None:
    if is_platform_sponsored(room):
        return None
    value = room.get("host_user_id")
    return int(value) if value is not None else None
