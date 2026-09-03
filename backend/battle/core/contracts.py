from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from typing import Any


class BattleModeError(RuntimeError):
    """Raised when a room references an unavailable or invalid Battle mode."""


class BattleMode(ABC):
    """Boundary between reusable room lifecycle code and one ruleset.

    Implementations own settings validation, round artifact generation, player
    actions and mode-specific result serialization. The room core owns members,
    permissions, room status transitions and realtime fan-out.
    """

    key: str
    version: int
    artifact_kind: str

    @abstractmethod
    def validate_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize untrusted create-room settings."""

    @abstractmethod
    def repository_fields(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Return legacy relational fields needed by the current schema."""

    @abstractmethod
    def public_settings(self, room: dict[str, Any]) -> dict[str, Any]:
        """Return the mode settings safe to expose in room snapshots."""

    @property
    @abstractmethod
    def token_operation_key(self) -> str:
        """Quota operation reserved when a new round artifact is prepared."""

    async def create_room(
        self,
        *,
        user_id: int,
        session_id: int | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def start_room(
        self,
        room_code: str,
        *,
        user_id: int,
        session_id: int | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def ensure_permanent_room(self, definition) -> dict[str, Any]:
        """Create or recover one configured platform-sponsored room."""
        raise NotImplementedError

    async def normalize_lobby_settings_patch(
        self, room: dict[str, Any], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate settings that may change before a round starts."""
        raise NotImplementedError

    def artifact_payload(
        self,
        room_code: str,
        round_id: str,
        *,
        actor_key: str,
    ) -> tuple[bytes, dict[str, Any]]:
        raise NotImplementedError

    def handle_action(
        self,
        room_code: str,
        *,
        actor_key: str,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def handle_action_async(
        self,
        room_code: str,
        *,
        actor_key: str,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve a potentially blocking mode action without blocking the event loop."""
        return await asyncio.to_thread(
            self.handle_action,
            room_code,
            actor_key=actor_key,
            action=action,
            payload=payload,
        )

    def sanitize_snapshot(
        self,
        payload: dict[str, Any],
        *,
        viewer_actor_key: str,
        viewer_user_id: int | None,
    ) -> dict[str, Any]:
        """Attach viewer-safe mode state to a generic room snapshot."""
        del viewer_actor_key, viewer_user_id
        return payload

    def settle_unstarted_round(self, room_id: str, *, reason: str) -> None:
        raise NotImplementedError

    def forfeit_round(
        self,
        room_code: str,
        *,
        actor_key: str,
        round_id: str,
    ) -> dict[str, Any]:
        """End one participant's current round without removing room membership."""
        raise NotImplementedError

    async def startup(self) -> None:
        """Start optional background work owned by this ruleset."""

    async def shutdown(self) -> None:
        """Stop optional background work owned by this ruleset."""
