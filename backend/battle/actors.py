from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BattleActor:
    kind: str
    actor_key: str
    user_id: int | None
    guest_id: str | None
    display_name: str

    @property
    def is_user(self) -> bool:
        return self.kind == "user"

    @property
    def is_guest(self) -> bool:
        return self.kind == "guest"


def user_actor(user_id: int, display_name: str = "") -> BattleActor:
    normalized = int(user_id)
    return BattleActor(
        kind="user",
        actor_key=f"u:{normalized}",
        user_id=normalized,
        guest_id=None,
        display_name=str(display_name or ""),
    )


def coerce_actor(
    actor: Any | None = None,
    *,
    user_id: int | None = None,
    display_name: str = "",
) -> BattleActor:
    """Normalize the shared ActorRef without making Battle own auth state."""
    if actor is None:
        if user_id is None:
            raise ValueError("actor_required")
        return user_actor(user_id, display_name)
    if isinstance(actor, BattleActor):
        return actor
    if isinstance(actor, str):
        if actor.startswith("u:") and actor[2:].isdigit():
            return user_actor(int(actor[2:]), display_name)
        if actor.startswith("g:") and len(actor) > 2:
            return BattleActor(
                kind="guest",
                actor_key=actor,
                user_id=None,
                guest_id=actor[2:],
                display_name=str(display_name or ""),
            )
    kind = str(getattr(actor, "kind", "") or "").lower()
    actor_key = str(getattr(actor, "actor_key", "") or "")
    actor_user_id = getattr(actor, "user_id", None)
    guest_id = getattr(actor, "guest_id", None)
    name = str(getattr(actor, "display_name", "") or display_name or "")
    if kind == "user" and actor_user_id is not None:
        return user_actor(int(actor_user_id), name)
    if kind == "guest" and guest_id and actor_key == f"g:{guest_id}":
        return BattleActor(
            kind="guest",
            actor_key=actor_key,
            user_id=None,
            guest_id=str(guest_id),
            display_name=name,
        )
    raise ValueError("invalid_actor")


def actor_from_session(session: Any) -> BattleActor:
    shared_actor = getattr(session, "actor", None) or getattr(session, "actor_ref", None)
    if shared_actor is not None:
        return coerce_actor(shared_actor)
    if getattr(session, "user_id", None) is not None:
        return user_actor(
            int(session.user_id),
            str(getattr(session, "user_display_name", "") or ""),
        )
    guest_id = getattr(session, "guest_id", None)
    actor_key = getattr(session, "actor_key", None)
    if guest_id and actor_key == f"g:{guest_id}":
        return BattleActor(
            kind="guest",
            actor_key=str(actor_key),
            user_id=None,
            guest_id=str(guest_id),
            display_name=str(getattr(session, "guest_display_name", "") or ""),
        )
    raise ValueError("actor_required")
