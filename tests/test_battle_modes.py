from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from starlette.websockets import WebSocketState

from backend.auth.db import auth_db, init_auth_db
from backend.actions import Action, Message
from backend.battle import realtime, repository, service
from backend.battle.core import lifecycle
from backend.battle.core.contracts import BattleMode
from backend.battle.core.registry import get_battle_mode, register_battle_mode


class FakeBattleMode(BattleMode):
    key = "test-mode"
    version = 3
    artifact_kind = "test-artifact"
    token_operation_key = "test_operation"

    def __init__(self) -> None:
        self.cancelled: list[tuple[str, str]] = []
        self.actions: list[dict[str, Any]] = []
        self.started = 0
        self.stopped = 0

    def validate_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"rounds": max(1, int(payload.get("rounds") or 1))}

    def repository_fields(self, settings: dict[str, Any]) -> dict[str, Any]:
        return {"pattern": "test", "target": 1, "full_pattern": "test_1"}

    def public_settings(self, room: dict[str, Any]) -> dict[str, Any]:
        return dict(room.get("settings") or {})

    async def create_room(self, **kwargs):
        return {"mode": self.key, "payload": kwargs["payload"]}

    async def start_room(self, room_code: str, **kwargs):
        return {"room_code": room_code, "started": True}

    def artifact_payload(self, room_code: str, round_id: str, *, actor_key: str):
        return b"test", {"room_code": room_code, "round_id": round_id, "actor_key": actor_key}

    def handle_action(
        self,
        room_code: str,
        *,
        actor_key: str,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        accepted = {
            "room_code": room_code,
            "actor_key": actor_key,
            "action": action,
            "payload": payload,
        }
        self.actions.append(accepted)
        return accepted

    def settle_unstarted_round(self, room_id: str, *, reason: str) -> None:
        self.cancelled.append((room_id, reason))

    async def startup(self) -> None:
        self.started += 1

    async def shutdown(self) -> None:
        self.stopped += 1


class FakeWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.messages.append(payload)


class BattleModeArchitectureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._old_auth_db = os.environ.get("CLOUD_AUTH_DB")
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["CLOUD_AUTH_DB"] = os.path.join(self._tmpdir.name, "battle-modes.sqlite3")
        init_auth_db()
        repository.init_battle_db()
        self.mode = FakeBattleMode()
        register_battle_mode(self.mode, replace=True)
        now = datetime.now(timezone.utc).isoformat()
        with auth_db() as db:
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_identity, password_hash, display_name,
                 registered_with_invite, role, status, created_at, updated_at)
                VALUES ('mode-host@example.com', 'mode-host@example.com', 'hash',
                        'mode-host', 0, 'user', 'active', ?, ?)
                """,
                (now, now),
            )
            self.user_id = int(cursor.lastrowid)

    async def asyncTearDown(self) -> None:
        if self._old_auth_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self._old_auth_db
        self._tmpdir.cleanup()

    async def test_service_dispatches_create_to_registered_mode(self) -> None:
        result = await service.create_room(
            user_id=self.user_id,
            session_id=None,
            payload={"mode_key": self.mode.key, "rounds": 4},
        )
        self.assertEqual(result["mode"], self.mode.key)
        self.assertEqual(result["payload"]["rounds"], 4)
        self.assertIs(get_battle_mode(self.mode.key), self.mode)

    async def test_service_adds_generic_artifact_metadata(self) -> None:
        repository.create_room(
            host_user_id=self.user_id,
            pattern="test",
            target=1,
            full_pattern="test_1",
            room_code="ART234",
            mode_key=self.mode.key,
            mode_version=self.mode.version,
            settings={"rounds": 1},
        )
        blob, metadata = service.route_payload(
            "ART234", "round-1", user_id=self.user_id
        )
        self.assertEqual(blob, b"test")
        self.assertEqual(metadata["artifact_kind"], self.mode.artifact_kind)
        self.assertEqual(len(metadata["artifact_hash"]), 64)

    async def test_generic_realtime_envelope_dispatches_mode_action(self) -> None:
        websocket = FakeWebSocket()
        realtime._socket_room[websocket] = "room-1"
        realtime._socket_user[websocket] = self.user_id
        try:
            with (
                patch.object(
                    realtime,
                    "handle_mode_action_async",
                    new=AsyncMock(return_value={"sequence": 8, "accepted": True}),
                ) as handler,
                patch.object(realtime, "broadcast_room", new=AsyncMock()) as broadcast,
            ):
                handled = await realtime.handle_battle_action(
                    Action.BATTLE_ACTION,
                    {
                        "room_code": "ART234",
                        "mode_action": "move",
                        "request_id": "req-8",
                        "payload": {"direction": "left", "sequence": 8},
                    },
                    SimpleNamespace(user_id=self.user_id),
                    websocket,
                )
            self.assertTrue(handled)
            handler.assert_awaited_once_with(
                "ART234",
                actor=realtime.user_actor(self.user_id),
                action="move",
                payload={"direction": "left", "sequence": 8, "request_id": "req-8"},
            )
            broadcast.assert_awaited_once_with("room-1")
            self.assertEqual(websocket.messages[-1]["action"], Message.BATTLE_ACTION_ACCEPTED)
            self.assertEqual(websocket.messages[-1]["data"]["request_id"], "req-8")
        finally:
            realtime._socket_room.pop(websocket, None)
            realtime._socket_user.pop(websocket, None)

    async def test_temporary_snapshot_failure_does_not_close_active_room(self) -> None:
        room = repository.create_room(
            host_user_id=self.user_id,
            pattern="test",
            target=1,
            full_pattern="test_1",
            room_code="TMP234",
            mode_key=self.mode.key,
            mode_version=self.mode.version,
            settings={"rounds": 1},
        )
        websocket = FakeWebSocket()
        realtime._socket_room[websocket] = room["room_id"]
        realtime._socket_user[websocket] = self.user_id
        realtime._room_sockets[room["room_id"]].add(websocket)
        try:
            with patch.object(
                realtime,
                "room_snapshot",
                side_effect=service.BattleServiceError(
                    "BATTLE_MODE_UNAVAILABLE", "Temporary mode failure.", 409
                ),
            ):
                await realtime.broadcast_room(room["room_id"])
            self.assertEqual(websocket.messages, [])
            self.assertEqual(realtime._socket_room.get(websocket), room["room_id"])
        finally:
            await realtime._detach(websocket)

    async def test_authoritative_room_close_identifies_the_closed_room(self) -> None:
        room = repository.create_room(
            host_user_id=self.user_id,
            pattern="test",
            target=1,
            full_pattern="test_1",
            room_code="CLS234",
            mode_key=self.mode.key,
            mode_version=self.mode.version,
            settings={"rounds": 1},
        )
        websocket = FakeWebSocket()
        realtime._socket_room[websocket] = room["room_id"]
        realtime._socket_user[websocket] = self.user_id
        realtime._room_sockets[room["room_id"]].add(websocket)
        repository.close_room(room["room_id"], host_user_id=self.user_id)
        try:
            await realtime.broadcast_room(room["room_id"])
            message = websocket.messages[-1]
            self.assertEqual(message["action"], Message.BATTLE_ROOM_STATE)
            self.assertTrue(message["data"]["closed"])
            self.assertEqual(message["data"]["room_id"], room["room_id"])
            self.assertEqual(message["data"]["code"], "ROOM_CLOSED")
        finally:
            await realtime._detach(websocket)

    async def test_repository_round_trips_mode_settings_and_core_closes_room(self) -> None:
        room = repository.create_room(
            host_user_id=self.user_id,
            pattern="test",
            target=1,
            full_pattern="test_1",
            room_code="TST234",
            status="preparing",
            mode_key=self.mode.key,
            mode_version=self.mode.version,
            settings={"rounds": 7},
        )
        self.assertEqual(room["mode_key"], self.mode.key)
        self.assertEqual(room["mode_version"], self.mode.version)
        self.assertEqual(room["settings"], {"rounds": 7})
        snapshot = lifecycle.room_snapshot(room["room_id"], user_id=self.user_id)
        self.assertEqual(snapshot["mode_settings"], {"rounds": 7})

        lifecycle.leave_room(room["room_id"], user_id=self.user_id)
        self.assertEqual(
            self.mode.cancelled,
            [(room["room_id"], "battle_host_closed_room")],
        )
        self.assertEqual(repository.get_room(room["room_id"])["status"], "closed")

    async def test_core_lifecycle_does_not_import_goodness_rules(self) -> None:
        source = inspect.getsource(lifecycle)
        self.assertNotIn("route_codec", source)
        self.assertNotIn("route_generator", source)
        self.assertNotIn("goodness_of_fit", source)


if __name__ == "__main__":
    unittest.main()
