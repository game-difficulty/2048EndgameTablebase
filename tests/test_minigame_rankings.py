from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
import unittest

from backend.auth.db import auth_db, init_auth_db
from backend.minigame_rankings.service import game_leaderboard, submit_score, trophy_leaderboard
from backend.minigame_rankings.catalog import MINIGAME_BY_ID


class MinigameRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.old_db = os.environ.get("CLOUD_AUTH_DB")
        os.environ["CLOUD_AUTH_DB"] = str(Path(self.tempdir.name) / "auth.sqlite3")
        init_auth_db()

    def tearDown(self) -> None:
        if self.old_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self.old_db
        self.tempdir.cleanup()

    def _add_user(self, email: str, name: str, supporter: bool = False) -> int:
        with auth_db() as db:
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_identity, password_hash, display_name, display_name_key,
                 role, status, created_at, updated_at)
                VALUES (?, ?, 'hash', ?, ?, 'user', 'active',
                        '2026-08-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00')
                """,
                (email, email, name, name.lower()),
            )
            user_id = int(cursor.lastrowid)
            db.execute(
                """
                INSERT INTO user_entitlements (user_id, tier, created_at, updated_at)
                VALUES (?, ?, '2026-08-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00')
                """,
                (user_id, "supporter" if supporter else "free"),
            )
            db.execute(
                """
                INSERT INTO user_profiles (user_id, avatar_key, created_at, updated_at)
                VALUES (?, ?, '2026-08-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00')
                """,
                (user_id, f"{user_id}/avatar.webp" if supporter else None),
            )
            return user_id

    @staticmethod
    def _submit(user_id: int, game_id: str, score: int, trophy: int, difficulty: int = 1):
        return submit_score(
            user_id=user_id,
            game_id=game_id,
            difficulty=difficulty,
            score=score,
            trophy_tier=trophy,
            highest_tile_exp=10,
            final_board=[0, 1, 2, 3] * 4,
            board_rows=4,
            board_cols=4,
        )

    def test_keeps_one_row_and_updates_score_and_trophy_independently(self) -> None:
        user_id = self._add_user("alice@example.com", "Alice")
        first = self._submit(user_id, "column-chaos", 1_000, 1)
        self.assertTrue(first["score_updated"])
        self.assertTrue(first["trophy_updated"])

        lower_score_higher_trophy = self._submit(user_id, "column-chaos", 800, 3)
        self.assertFalse(lower_score_higher_trophy["score_updated"])
        self.assertTrue(lower_score_higher_trophy["trophy_updated"])
        self.assertEqual(lower_score_higher_trophy["personal_best"], 1_000)
        self.assertEqual(lower_score_higher_trophy["trophy_tier"], 3)

        ignored = self._submit(user_id, "column-chaos", 999, 2)
        self.assertFalse(ignored["score_updated"])
        self.assertFalse(ignored["trophy_updated"])
        with auth_db() as db:
            count = db.execute("SELECT COUNT(*) AS count FROM minigame_high_scores").fetchone()
        self.assertEqual(int(count["count"]), 1)

    def test_game_board_orders_score_then_first_arrival(self) -> None:
        alice = self._add_user("alice@example.com", "Alice", supporter=True)
        bob = self._add_user("bob@example.com", "Bob")
        self._submit(alice, "tricky-tiles", 2_000, 2)
        self._submit(bob, "tricky-tiles", 2_000, 1)
        with auth_db() as db:
            db.execute(
                "UPDATE minigame_high_scores SET score_achieved_at = ? WHERE user_id = ?",
                ("2026-08-02T00:00:00+00:00", alice),
            )
            db.execute(
                "UPDATE minigame_high_scores SET score_achieved_at = ? WHERE user_id = ?",
                ("2026-08-01T00:00:00+00:00", bob),
            )
        payload = game_leaderboard("tricky-tiles", difficulty=1, limit=10)
        self.assertEqual([entry["display_name"] for entry in payload["entries"]], ["Bob", "Alice"])
        self.assertTrue(payload["entries"][1]["is_supporter"])
        self.assertEqual(payload["entries"][1]["avatar_url"], f"/media/avatars/{alice}/avatar.webp")

    def test_overall_board_compares_grand_then_gold_silver_bronze(self) -> None:
        alice = self._add_user("alice@example.com", "Alice")
        bob = self._add_user("bob@example.com", "Bob")
        carol = self._add_user("carol@example.com", "Carol")
        self._submit(alice, "column-chaos", 100, 4)
        self._submit(bob, "column-chaos", 200, 3)
        self._submit(bob, "tricky-tiles", 200, 3)
        self._submit(carol, "column-chaos", 300, 2)
        payload = trophy_leaderboard(difficulty=1, limit=10)
        self.assertEqual([entry["display_name"] for entry in payload["entries"]], ["Alice", "Bob", "Carol"])
        self.assertEqual(payload["entries"][0]["trophies"], {
            "grand": 1,
            "gold": 1,
            "silver": 1,
            "bronze": 1,
        })
        self.assertEqual(payload["entries"][1]["trophies"]["gold"], 2)

    def test_difficulty_boards_are_isolated(self) -> None:
        user_id = self._add_user("alice@example.com", "Alice")
        self._submit(user_id, "ice-age", 500, 1, difficulty=0)
        self._submit(user_id, "ice-age", 900, 2, difficulty=1)
        self.assertEqual(game_leaderboard("ice-age", difficulty=0, limit=10)["entries"][0]["score"], 500)
        self.assertEqual(game_leaderboard("ice-age", difficulty=1, limit=10)["entries"][0]["score"], 900)

    def test_rejects_unknown_game_and_invalid_board(self) -> None:
        user_id = self._add_user("alice@example.com", "Alice")
        with self.assertRaises(ValueError):
            self._submit(user_id, "not-a-game", 10, 0)
        with self.assertRaises(ValueError):
            submit_score(
                user_id=user_id,
                game_id="column-chaos",
                difficulty=1,
                score=10,
                trophy_tier=0,
                highest_tile_exp=2,
                final_board=[0, 1],
                board_rows=4,
                board_cols=4,
            )

    def test_backend_catalog_matches_frontend_registry(self) -> None:
        registry_path = Path(__file__).resolve().parents[1] / "frontend" / "src" / "features" / "minigames" / "engine" / "registry.js"
        source = registry_path.read_text(encoding="utf-8")
        frontend_ids = set(re.findall(r"\bid:\s*'([^']+)'", source))
        self.assertEqual(frontend_ids, set(MINIGAME_BY_ID))


if __name__ == "__main__":
    unittest.main()
