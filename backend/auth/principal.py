from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ActorKind = Literal["user", "guest"]


@dataclass(frozen=True)
class ActorRef:
    kind: ActorKind
    actor_key: str
    display_name: str
    user_id: int | None = None
    guest_id: str | None = None
    session_id: int | None = None
    role: str = ""

    @property
    def is_user(self) -> bool:
        return self.kind == "user" and self.user_id is not None

    @property
    def is_guest(self) -> bool:
        return self.kind == "guest" and bool(self.guest_id)

    def public_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "actor_key": self.actor_key,
            "display_name": self.display_name,
            "user_id": self.user_id,
            "guest_id": self.guest_id,
        }

    @classmethod
    def from_user(cls, user: dict[str, Any]) -> "ActorRef":
        user_id = int(user["id"])
        return cls(
            kind="user",
            actor_key=f"u:{user_id}",
            user_id=user_id,
            session_id=(int(user["session_id"]) if user.get("session_id") else None),
            display_name=str(user.get("display_name") or user.get("email") or f"User {user_id}"),
            role=str(user.get("role") or "user"),
        )

    @classmethod
    def from_guest(cls, guest: dict[str, Any]) -> "ActorRef":
        guest_id = str(guest["guest_id"])
        return cls(
            kind="guest",
            actor_key=f"g:{guest_id}",
            guest_id=guest_id,
            display_name=str(guest.get("display_name") or "Guest"),
            role="guest",
        )
