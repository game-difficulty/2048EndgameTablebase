from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from starlette.websockets import WebSocketState

from backend.actions import Action, Message
from backend.auth.db import auth_db, init_auth_db
from backend.battle import realtime, repository
from backend.battle.core import chat
from backend.battle.core.errors import BattleServiceError


class FakeWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.messages.append(payload)


class BattleChatTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._old_auth_db = os.environ.get("CLOUD_AUTH_DB")
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["CLOUD_AUTH_DB"] = os.path.join(self._tmpdir.name, "battle-chat.sqlite3")
        init_auth_db()
        repository.init_battle_db()
        self.host_id = self._create_user("chat-host@example.com", "Host")
        self.player_id = self._create_user("chat-player@example.com", "Player")
        self.spectator_id = self._create_user("chat-spectator@example.com", "Spectator")
        self.room = repository.create_room(
            host_user_id=self.host_id,
            pattern="442t",
            target=512,
            full_pattern="442t_512",
            room_code="CHT234",
            max_players=2,
            status="waiting",
        )
        repository.join_room("CHT234", user_id=self.player_id, preferred_role="player")
        repository.join_room("CHT234", user_id=self.spectator_id, preferred_role="spectator")

    async def asyncTearDown(self) -> None:
        if self._old_auth_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self._old_auth_db
        self._tmpdir.cleanup()

    def _create_user(self, email: str, name: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with auth_db() as db:
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_identity, password_hash, display_name,
                 registered_with_invite, role, status, created_at, updated_at)
                VALUES (?, ?, 'hash', ?, 0, 'user', 'active', ?, ?)
                """,
                (email, email, name, now, now),
            )
            return int(cursor.lastrowid)

    async def test_all_active_room_roles_can_chat_and_read_history(self) -> None:
        base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
        for index, user_id in enumerate((self.host_id, self.player_id, self.spectator_id)):
            result = chat.post_message(
                "CHT234",
                user_id=user_id,
                request_id=f"role-{index}",
                content=f"message {index}",
                now=base + timedelta(seconds=index),
            )
            self.assertTrue(result.created)
        history = chat.recent_messages("CHT234", user_id=self.spectator_id)
        self.assertEqual([item["display_name"] for item in history], ["Host", "Player", "Spectator"])
        self.assertEqual([item["content"] for item in history], ["message 0", "message 1", "message 2"])

    async def test_room_chat_roles_block_only_sending_for_unselected_roles(self) -> None:
        with auth_db() as db:
            db.execute(
                "UPDATE battle_rooms SET chat_roles_json = ? WHERE room_id = ?",
                ('["host","spectator"]', self.room["room_id"]),
            )
        chat.post_message(
            "CHT234",
            user_id=self.host_id,
            request_id="host-allowed",
            content="host",
        )
        chat.post_message(
            "CHT234",
            user_id=self.spectator_id,
            request_id="spectator-allowed",
            content="spectator",
        )
        with self.assertRaises(BattleServiceError) as raised:
            chat.post_message(
                "CHT234",
                user_id=self.player_id,
                request_id="player-blocked",
                content="player",
            )
        self.assertEqual(raised.exception.detail["code"], "CHAT_ROLE_NOT_ALLOWED")
        self.assertEqual(len(chat.recent_messages("CHT234", user_id=self.player_id)), 2)

    async def test_request_id_is_idempotent_and_does_not_consume_rate_twice(self) -> None:
        now = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
        first = chat.post_message(
            "CHT234",
            user_id=self.host_id,
            request_id="same-request",
            content="first",
            now=now,
        )
        duplicate = chat.post_message(
            "CHT234",
            user_id=self.host_id,
            request_id="same-request",
            content="changed",
            now=now + timedelta(seconds=1),
        )
        self.assertTrue(first.created)
        self.assertFalse(duplicate.created)
        self.assertEqual(duplicate.message["message_id"], first.message["message_id"])
        self.assertEqual(duplicate.message["content"], "first")

    async def test_rolling_rate_limit_rejects_sixth_message(self) -> None:
        base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
        for index in range(5):
            chat.post_message(
                "CHT234",
                user_id=self.player_id,
                request_id=f"rate-{index}",
                content=f"m{index}",
                now=base + timedelta(seconds=index),
            )
        with self.assertRaises(chat.BattleChatRateLimit) as raised:
            chat.post_message(
                "CHT234",
                user_id=self.player_id,
                request_id="rate-5",
                content="blocked",
                now=base + timedelta(seconds=5),
            )
        self.assertEqual(raised.exception.retry_after_seconds, 55)
        accepted = chat.post_message(
            "CHT234",
            user_id=self.player_id,
            request_id="rate-after-window",
            content="accepted",
            now=base + timedelta(seconds=61),
        )
        self.assertTrue(accepted.created)

    async def test_unicode_length_controls_and_invalid_characters(self) -> None:
        accepted = chat.post_message(
            "CHT234",
            user_id=self.host_id,
            request_id="twenty",
            content="测" * 20,
        )
        self.assertEqual(accepted.message["content"], "测" * 20)
        with self.assertRaisesRegex(BattleServiceError, "20 characters"):
            chat.post_message(
                "CHT234",
                user_id=self.host_id,
                request_id="twenty-one",
                content="测" * 21,
            )
        with self.assertRaises(BattleServiceError):
            chat.normalize_chat_content("line\nbreak")

    async def test_room_keeps_only_latest_fifty_messages(self) -> None:
        base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
        for index in range(51):
            chat.post_message(
                "CHT234",
                user_id=self.host_id,
                request_id=f"history-{index}",
                content=f"message-{index}",
                now=base + timedelta(seconds=index * 61),
            )
        history = chat.recent_messages("CHT234", user_id=self.host_id)
        self.assertEqual(len(history), 50)
        self.assertEqual(history[0]["content"], "message-1")
        self.assertEqual(history[-1]["content"], "message-50")
        with auth_db() as db:
            count = db.execute(
                "SELECT COUNT(*) AS count FROM battle_chat_messages WHERE room_id = ?",
                (self.room["room_id"],),
            ).fetchone()["count"]
        self.assertEqual(count, 50)

    async def test_left_member_and_closed_room_cannot_chat(self) -> None:
        repository.kick_member(
            "CHT234",
            host_user_id=self.host_id,
            target_user_id=self.player_id,
        )
        with self.assertRaisesRegex(BattleServiceError, "active member"):
            chat.post_message(
                "CHT234",
                user_id=self.player_id,
                request_id="kicked",
                content="blocked",
            )
        repository.close_room("CHT234", host_user_id=self.host_id)
        with self.assertRaisesRegex(BattleServiceError, "unavailable"):
            chat.post_message(
                "CHT234",
                user_id=self.host_id,
                request_id="closed",
                content="blocked",
            )

    async def test_closed_room_cleanup_removes_persisted_history(self) -> None:
        chat.post_message(
            "CHT234",
            user_id=self.host_id,
            request_id="cleanup",
            content="temporary",
        )
        repository.close_room("CHT234", host_user_id=self.host_id)
        self.assertEqual(chat.cleanup_closed_room_messages(), 1)
        with auth_db() as db:
            count = db.execute(
                "SELECT COUNT(*) AS count FROM battle_chat_messages"
            ).fetchone()["count"]
        self.assertEqual(count, 0)

    async def test_accepted_message_broadcasts_delta_without_room_snapshot(self) -> None:
        sender = FakeWebSocket()
        observer = FakeWebSocket()
        realtime._socket_room[sender] = self.room["room_id"]
        realtime._socket_room[observer] = self.room["room_id"]
        realtime._socket_user[sender] = self.host_id
        realtime._socket_user[observer] = self.player_id
        realtime._room_sockets[self.room["room_id"]].update({sender, observer})
        message = {
            "message_id": 8,
            "room_id": self.room["room_id"],
            "user_id": self.host_id,
            "display_name": "Host",
            "content": "hello",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "avatar_url": None,
        }
        try:
            with patch.object(
                realtime.chat,
                "post_message",
                return_value=chat.ChatInsertResult(message=message, created=True),
            ):
                handled = await realtime.handle_battle_action(
                    Action.BATTLE_CHAT_SEND,
                    {"request_id": "accepted", "content": "hello"},
                    SimpleNamespace(user_id=self.host_id),
                    sender,
                )
            self.assertTrue(handled)
            self.assertEqual(sender.messages[-1]["action"], Message.BATTLE_CHAT_MESSAGE)
            self.assertEqual(observer.messages[-1]["action"], Message.BATTLE_CHAT_MESSAGE)
            self.assertNotIn(
                Message.BATTLE_ROOM_STATE,
                [item["action"] for item in sender.messages + observer.messages],
            )
        finally:
            for websocket in (sender, observer):
                realtime._socket_room.pop(websocket, None)
                realtime._socket_user.pop(websocket, None)
                realtime._room_sockets[self.room["room_id"]].discard(websocket)
            if not realtime._room_sockets[self.room["room_id"]]:
                realtime._room_sockets.pop(self.room["room_id"], None)

    async def test_rate_limit_message_is_private_to_sender(self) -> None:
        sender = FakeWebSocket()
        observer = FakeWebSocket()
        realtime._socket_room[sender] = self.room["room_id"]
        realtime._socket_room[observer] = self.room["room_id"]
        realtime._socket_user[sender] = self.host_id
        realtime._socket_user[observer] = self.player_id
        try:
            with patch.object(
                realtime.chat,
                "post_message",
                side_effect=chat.BattleChatRateLimit(37),
            ):
                handled = await realtime.handle_battle_action(
                    Action.BATTLE_CHAT_SEND,
                    {"request_id": "limited", "content": "hello"},
                    SimpleNamespace(user_id=self.host_id),
                    sender,
                )
            self.assertTrue(handled)
            self.assertEqual(sender.messages[-1]["action"], Message.BATTLE_CHAT_RATE_LIMITED)
            self.assertEqual(sender.messages[-1]["data"]["retry_after_seconds"], 37)
            self.assertEqual(observer.messages, [])
        finally:
            for websocket in (sender, observer):
                realtime._socket_room.pop(websocket, None)
                realtime._socket_user.pop(websocket, None)


if __name__ == "__main__":
    unittest.main()
