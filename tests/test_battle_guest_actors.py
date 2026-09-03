from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone

from backend.auth.db import auth_db, init_auth_db
from backend.auth.dependencies import require_actor, require_user
from backend.auth.principal import ActorRef
from backend.battle import repository
from backend.battle.core import chat
from backend.battle.core.errors import BattleServiceError
from backend.battle.routes import router


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BattleGuestActorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = {
            name: os.environ.get(name)
            for name in (
                "CLOUD_AUTH_DB",
                "GUEST_IP_HASH_SECRET",
                "BATTLE_GUEST_JOIN_LIMIT_PER_HOUR",
                "BATTLE_GUEST_JOIN_IP_LIMIT_PER_HOUR",
            )
        }
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["CLOUD_AUTH_DB"] = os.path.join(self._tmpdir.name, "battle.sqlite3")
        os.environ["GUEST_IP_HASH_SECRET"] = "battle-guest-test-secret"
        init_auth_db()
        repository.init_battle_db()
        self.host_id = self._create_user("host@example.com")

    def tearDown(self) -> None:
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._tmpdir.cleanup()

    def _create_user(self, email: str) -> int:
        now = _now()
        with auth_db() as db:
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_identity, password_hash, display_name,
                 registered_with_invite, role, status, created_at, updated_at)
                VALUES (?, ?, 'hash', ?, 0, 'user', 'active', ?, ?)
                """,
                (email, email, email.split("@", 1)[0], now, now),
            )
            return int(cursor.lastrowid)

    def _guest(self, suffix: str) -> ActorRef:
        guest_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"battle-{suffix}"))
        return ActorRef(
            kind="guest",
            actor_key=f"g:{guest_id}",
            guest_id=guest_id,
            display_name=f"Guest-{suffix}",
        )

    def _room(self, index: int, *, allow_guest_chat: bool = False) -> dict:
        host_id = self.host_id if index == 0 else self._create_user(f"host{index}@example.com")
        return repository.create_room(
            host_user_id=host_id,
            pattern="L3",
            target=128,
            status="waiting",
            max_players=8,
            allow_guest_chat=allow_guest_chat,
        )

    def _leave_guest(self, room_id: str, actor: ActorRef) -> None:
        with auth_db() as db:
            db.execute(
                """
                UPDATE battle_members
                SET status = 'left', ready = 0, left_at = ?, updated_at = ?
                WHERE room_id = ? AND actor_key = ?
                """,
                (_now(), _now(), room_id, actor.actor_key),
            )

    def test_guest_join_is_idempotent_and_ip_is_hmac_only(self) -> None:
        room = self._room(0)
        actor = self._guest("A")
        first = repository.join_room(
            room["room_id"], actor=actor, ip_address="203.0.113.8"
        )
        second = repository.join_room(
            room["room_id"], actor=actor, ip_address="203.0.113.8"
        )
        self.assertEqual(first["member_id"], second["member_id"])

        self._leave_guest(room["room_id"], actor)
        repository.join_room(room["room_id"], actor=actor, ip_address="203.0.113.8")
        with auth_db() as db:
            events = db.execute("SELECT * FROM battle_guest_join_events").fetchall()
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(battle_guest_join_events)")
            }
        self.assertEqual(len(events), 1)
        self.assertNotEqual(events[0]["ip_hash"], "203.0.113.8")
        self.assertEqual(len(str(events[0]["ip_hash"])), 64)
        self.assertNotIn("ip_address", columns)

    def test_default_guest_join_limits_are_ten_and_thirty_per_hour(self) -> None:
        self.assertEqual(repository.DEFAULT_GUEST_JOIN_LIMIT, 10)
        self.assertEqual(repository.DEFAULT_GUEST_JOIN_IP_LIMIT, 30)
        self.assertEqual(repository.GUEST_JOIN_WINDOW.total_seconds(), 3600)

    def test_guest_hourly_limit_counts_distinct_rooms(self) -> None:
        os.environ["BATTLE_GUEST_JOIN_LIMIT_PER_HOUR"] = "2"
        os.environ["BATTLE_GUEST_JOIN_IP_LIMIT_PER_HOUR"] = "30"
        actor = self._guest("LIMIT")
        for index in range(2):
            room = self._room(index)
            repository.join_room(room["room_id"], actor=actor, ip_address=f"203.0.113.{index}")
            self._leave_guest(room["room_id"], actor)
        room = self._room(2)
        with self.assertRaises(repository.BattleRateLimitError) as raised:
            repository.join_room(room["room_id"], actor=actor, ip_address="203.0.113.9")
        self.assertEqual(raised.exception.code, "GUEST_JOIN_RATE_LIMITED")
        self.assertGreater(raised.exception.retry_after_seconds, 0)

    def test_guest_ip_hourly_limit_counts_distinct_guests(self) -> None:
        os.environ["BATTLE_GUEST_JOIN_LIMIT_PER_HOUR"] = "10"
        os.environ["BATTLE_GUEST_JOIN_IP_LIMIT_PER_HOUR"] = "2"
        for index in range(2):
            room = self._room(index)
            repository.join_room(
                room["room_id"], actor=self._guest(f"IP-{index}"), ip_address="198.51.100.4"
            )
        room = self._room(2)
        with self.assertRaises(repository.BattleRateLimitError) as raised:
            repository.join_room(
                room["room_id"], actor=self._guest("IP-2"), ip_address="198.51.100.4"
            )
        self.assertEqual(raised.exception.code, "GUEST_NETWORK_JOIN_RATE_LIMITED")

    def test_guest_identity_payload_and_chat_policy(self) -> None:
        room = self._room(0)
        actor = self._guest("CHAT")
        repository.join_room(room["room_id"], actor=actor, ip_address="192.0.2.2")
        snapshot = repository.get_room(room["room_id"])
        guest = next(member for member in snapshot["members"] if member["actor_key"] == actor.actor_key)
        self.assertTrue(guest["is_guest"])
        self.assertEqual(guest["actor_kind"], "guest")
        self.assertEqual(guest["display_name"], actor.display_name)
        self.assertIsNone(guest["avatar_url"])

        with self.assertRaises(BattleServiceError) as raised:
            chat.post_message(
                room["room_id"], actor=actor, request_id="blocked", content="hello"
            )
        self.assertEqual(raised.exception.code, "CHAT_GUEST_NOT_ALLOWED")
        with auth_db() as db:
            db.execute(
                "UPDATE battle_rooms SET allow_guest_chat = 1 WHERE room_id = ?",
                (room["room_id"],),
            )
        sent = chat.post_message(
            room["room_id"], actor=actor, request_id="allowed", content="hello"
        )
        self.assertTrue(sent.created)
        self.assertTrue(sent.message["is_guest"])
        self.assertEqual(sent.message["display_name"], actor.display_name)

    def test_user_join_does_not_consume_guest_limit(self) -> None:
        room = self._room(0)
        player_id = self._create_user("player@example.com")
        repository.join_room(room["room_id"], user_id=player_id, ip_address="203.0.113.1")
        with auth_db() as db:
            count = int(
                db.execute("SELECT COUNT(*) FROM battle_guest_join_events").fetchone()[0]
            )
        self.assertEqual(count, 0)

    def test_route_dependency_matrix(self) -> None:
        routes = {
            (route.path, method): route
            for route in router.routes
            for method in getattr(route, "methods", set())
        }

        def dependencies(path: str, method: str):
            return {item.call for item in routes[(path, method)].dependant.dependencies}

        self.assertEqual(dependencies("/api/battle/rooms", "GET"), set())
        self.assertIn(require_user, dependencies("/api/battle/rooms", "POST"))
        for path, method in (
            ("/api/battle/me", "GET"),
            ("/api/battle/rooms/{room_code}", "GET"),
            ("/api/battle/rooms/{room_code}/join", "POST"),
            ("/api/battle/rooms/{room_code}/leave", "POST"),
            ("/api/battle/rooms/{room_code}/ready", "POST"),
            ("/api/battle/rooms/{room_code}/role", "POST"),
            ("/api/battle/rooms/{room_code}/kick", "POST"),
            ("/api/battle/rooms/{room_code}/start", "POST"),
            ("/api/battle/rooms/{room_code}/settings", "PATCH"),
            ("/api/battle/rooms/{room_code}/host/renew", "POST"),
            ("/api/battle/rooms/{room_code}/rounds/{round_id}/forfeit", "POST"),
            ("/api/battle/rooms/{room_code}/rounds/{round_id}/artifact", "GET"),
            ("/api/battle/rooms/{room_code}/rounds/{round_id}/route", "GET"),
            ("/api/battle/rooms/{room_code}/rounds/{round_id}/replay", "GET"),
        ):
            self.assertIn(require_actor, dependencies(path, method), (path, method))
if __name__ == "__main__":
    unittest.main()
