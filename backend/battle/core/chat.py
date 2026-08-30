from __future__ import annotations

import asyncio
import math
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.auth.db import auth_db

from .. import repository
from .chat_policy import decode_chat_roles, effective_chat_role
from .errors import BattleServiceError


CHAT_HISTORY_LIMIT = 50
CHAT_MESSAGE_CODEPOINT_LIMIT = 20
CHAT_MESSAGE_BYTE_LIMIT = 160
CHAT_RATE_LIMIT = 5
CHAT_RATE_WINDOW = timedelta(minutes=1)
CHAT_ACTIVE_ROOM_STATUSES = frozenset({"preparing", "waiting", "running"})


class BattleChatRateLimit(BattleServiceError):
    def __init__(self, retry_after_seconds: int) -> None:
        retry_after = max(1, int(retry_after_seconds))
        super().__init__(
            "CHAT_RATE_LIMITED",
            "Too many chat messages. Please try again shortly.",
            429,
            extra={
                "limit": CHAT_RATE_LIMIT,
                "window_seconds": int(CHAT_RATE_WINDOW.total_seconds()),
                "retry_after_seconds": retry_after,
            },
        )
        self.retry_after_seconds = retry_after


@dataclass(frozen=True)
class ChatInsertResult:
    message: dict[str, Any]
    created: bool


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_chat_content(value: Any) -> str:
    content = unicodedata.normalize("NFC", str(value or "")).strip()
    if not content:
        raise BattleServiceError("CHAT_EMPTY", "Chat message cannot be empty.")
    if any(character in "\r\n\t" or unicodedata.category(character) == "Cc" for character in content):
        raise BattleServiceError(
            "CHAT_INVALID_CONTENT",
            "Chat message contains unsupported characters.",
        )
    if len(content) > CHAT_MESSAGE_CODEPOINT_LIMIT:
        raise BattleServiceError(
            "CHAT_TOO_LONG",
            f"Chat messages are limited to {CHAT_MESSAGE_CODEPOINT_LIMIT} characters.",
        )
    if len(content.encode("utf-8")) > CHAT_MESSAGE_BYTE_LIMIT:
        raise BattleServiceError(
            "CHAT_TOO_LARGE",
            "Chat message is too large.",
        )
    return content


def _message_payload(db, message_id: int) -> dict[str, Any]:
    row = db.execute(
        """
        SELECT message.message_id, message.room_id, message.user_id,
               message.content, message.created_at,
               user.display_name, profile.avatar_key
        FROM battle_chat_messages AS message
        JOIN users AS user ON user.id = message.user_id
        LEFT JOIN user_profiles AS profile ON profile.user_id = message.user_id
        WHERE message.message_id = ?
        """,
        (int(message_id),),
    ).fetchone()
    if row is None:
        raise BattleServiceError("CHAT_MESSAGE_NOT_FOUND", "Chat message was not found.", 404)
    payload = dict(row)
    avatar_key = payload.pop("avatar_key", None)
    payload["avatar_url"] = f"/media/avatars/{avatar_key}" if avatar_key else None
    return payload


def _assert_active_member(db, room_id: str, user_id: int):
    member = db.execute(
        """
        SELECT role FROM battle_members
        WHERE room_id = ? AND user_id = ? AND status = 'active'
        """,
        (str(room_id), int(user_id)),
    ).fetchone()
    if member is None:
        raise BattleServiceError(
            "CHAT_NOT_IN_ROOM",
            "You are not an active member of this room.",
            403,
        )
    return member


def recent_messages(
    room_ref: str,
    *,
    user_id: int,
    limit: int = CHAT_HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), CHAT_HISTORY_LIMIT))
    with auth_db() as db:
        room = repository._find_room(db, str(room_ref))
        _assert_active_member(db, str(room["room_id"]), int(user_id))
        rows = db.execute(
            """
            SELECT message.message_id, message.room_id, message.user_id,
                   message.content, message.created_at,
                   user.display_name, profile.avatar_key
            FROM battle_chat_messages AS message
            JOIN users AS user ON user.id = message.user_id
            LEFT JOIN user_profiles AS profile ON profile.user_id = message.user_id
            WHERE message.room_id = ?
            ORDER BY message.message_id DESC
            LIMIT ?
            """,
            (str(room["room_id"]), limit),
        ).fetchall()
    messages: list[dict[str, Any]] = []
    for row in reversed(rows):
        payload = dict(row)
        avatar_key = payload.pop("avatar_key", None)
        payload["avatar_url"] = f"/media/avatars/{avatar_key}" if avatar_key else None
        messages.append(payload)
    return messages


def post_message(
    room_ref: str,
    *,
    user_id: int,
    request_id: str,
    content: Any,
    now: datetime | None = None,
) -> ChatInsertResult:
    normalized_request_id = str(request_id or "").strip()[:160]
    if not normalized_request_id:
        raise BattleServiceError("CHAT_REQUEST_ID_REQUIRED", "Chat request id is required.")
    normalized_content = normalize_chat_content(content)
    current = (now or _utcnow()).astimezone(timezone.utc)
    cutoff = current - CHAT_RATE_WINDOW

    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room = repository._find_room(db, str(room_ref))
        room_id = str(room["room_id"])
        if str(room["status"]) not in CHAT_ACTIVE_ROOM_STATUSES:
            raise BattleServiceError("CHAT_NOT_ALLOWED", "Chat is unavailable for this room.", 409)
        member = _assert_active_member(db, room_id, int(user_id))
        role = effective_chat_role(room=room, member=member, user_id=int(user_id))
        if role not in decode_chat_roles(room["chat_roles_json"]):
            raise BattleServiceError(
                "CHAT_ROLE_NOT_ALLOWED",
                "Your room role is not allowed to send chat messages.",
                403,
            )

        existing = db.execute(
            """
            SELECT message_id FROM battle_chat_messages
            WHERE room_id = ? AND user_id = ? AND request_id = ?
            """,
            (room_id, int(user_id), normalized_request_id),
        ).fetchone()
        if existing is not None:
            return ChatInsertResult(
                message=_message_payload(db, int(existing["message_id"])),
                created=False,
            )

        recent = db.execute(
            """
            SELECT created_at FROM battle_chat_messages
            WHERE room_id = ? AND user_id = ? AND created_at >= ?
            ORDER BY created_at ASC
            """,
            (room_id, int(user_id), _iso(cutoff)),
        ).fetchall()
        if len(recent) >= CHAT_RATE_LIMIT:
            retry_at = _parse_iso(str(recent[0]["created_at"])) + CHAT_RATE_WINDOW
            retry_after = math.ceil((retry_at - current).total_seconds())
            raise BattleChatRateLimit(retry_after)

        cursor = db.execute(
            """
            INSERT INTO battle_chat_messages
            (room_id, user_id, request_id, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                room_id,
                int(user_id),
                normalized_request_id,
                normalized_content,
                _iso(current),
            ),
        )
        message_id = int(cursor.lastrowid)
        db.execute(
            """
            DELETE FROM battle_chat_messages
            WHERE room_id = ? AND message_id NOT IN (
              SELECT message_id FROM battle_chat_messages
              WHERE room_id = ?
              ORDER BY message_id DESC
              LIMIT ?
            )
            """,
            (room_id, room_id, CHAT_HISTORY_LIMIT),
        )
        return ChatInsertResult(
            message=_message_payload(db, message_id),
            created=True,
        )


def cleanup_closed_room_messages() -> int:
    with auth_db() as db:
        cursor = db.execute(
            """
            DELETE FROM battle_chat_messages
            WHERE room_id IN (
              SELECT room_id FROM battle_rooms WHERE status IN ('closed', 'expired')
            )
            """
        )
        return max(0, int(cursor.rowcount or 0))


_cleanup_task: asyncio.Task | None = None


async def _cleanup_loop() -> None:
    while True:
        await asyncio.to_thread(cleanup_closed_room_messages)
        await asyncio.sleep(300)


async def startup() -> None:
    global _cleanup_task
    await asyncio.to_thread(cleanup_closed_room_messages)
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_loop())


async def shutdown() -> None:
    global _cleanup_task
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        await asyncio.gather(_cleanup_task, return_exceptions=True)
        _cleanup_task = None
