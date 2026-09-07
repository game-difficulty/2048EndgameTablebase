from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
import base64
import binascii
import uuid
import json
from unittest.mock import patch

from fastapi import Response, Request, HTTPException

from backend.auth.db import auth_db, init_auth_db
from backend.minigame_rankings import routes as minigame_routes
from backend.minigame_rankings.service import (
    MGO_RECORD_PREFIX,
    RunTokenError,
    RunTokenExpired,
    _normalize_summary,
    _derive_seed_hex,
    _apply_verified_result,
    personal_records,
    abandon_ranked_run,
    claim_ranked_run,
    claim_pending,
    claim_pending_checkpoint,
    create_ranked_run,
    finish_rejected,
    finish_checkpoint_verified,
    finish_verified,
    get_ranked_run,
    game_leaderboard,
    heartbeat_ranked_run,
    qualify_ranked_run,
    submit_ranked_run,
    submit_ranked_checkpoint,
    submit_score,
    trophy_leaderboard,
)
from backend.minigame_rankings.catalog import MINIGAME_BY_ID
from backend.minigame_rankings import verifier as minigame_verifier


class MinigameRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.old_db = os.environ.get("CLOUD_AUTH_DB")
        self.old_secret = os.environ.get("MINIGAME_RANKING_SECRET")
        os.environ["CLOUD_AUTH_DB"] = str(Path(self.tempdir.name) / "auth.sqlite3")
        os.environ["MINIGAME_RANKING_SECRET"] = "minigame-ranking-test-secret"
        init_auth_db()

    def tearDown(self) -> None:
        if self.old_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self.old_db
        if self.old_secret is None:
            os.environ.pop("MINIGAME_RANKING_SECRET", None)
        else:
            os.environ["MINIGAME_RANKING_SECRET"] = self.old_secret
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
        result = submit_score(
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
        with auth_db() as db:
            db.execute(
                """
                UPDATE minigame_high_scores
                SET verification_level = 'verified', best_tile_exp = MAX(best_tile_exp, highest_tile_exp)
                WHERE user_id = ? AND game_id = ? AND difficulty = ?
                """,
                (int(user_id), str(game_id), int(difficulty)),
            )
        return result

    def test_personal_records_independent_metrics_and_score_replay_metadata(self) -> None:
        user = self._add_user("records@example.com", "Records")
        run = {"user_id": user, "game_id": "column-chaos", "difficulty": 1, "run_id": "one"}
        def apply(score, trophy, tile):
            result = self._summary(score=score, trophy=trophy)
            result["highest_tile_exp"] = tile
            with auth_db() as db:
                _apply_verified_result(db, run=run, verified=result, record_hash="hash",
                                       record_blob="record", verified_at="2026-09-07T00:00:00+00:00")
        apply(1000, 1, 10)
        apply(500, 3, 14)
        apply(2000, 2, 11)
        record = personal_records(user)["records"][0]
        self.assertEqual((record["best_score"], record["trophy_tier"], record["best_tile_exp"]), (2000, 3, 14))
        with auth_db() as db:
            saved = db.execute("SELECT highest_tile_exp FROM minigame_high_scores").fetchone()
        self.assertEqual(saved["highest_tile_exp"], 11)
        self.assertEqual(game_leaderboard("column-chaos", difficulty=1, limit=100)["entries"][0]["highest_tile"], 2**14)
        # Improving only the tile does not require a score or trophy improvement.
        apply(400, 0, 15)
        self.assertEqual(personal_records(user)["records"][0]["best_tile_exp"], 15)

    def test_personal_records_require_login_and_are_account_and_difficulty_scoped(self) -> None:
        alice = self._add_user("records-a@example.com", "A")
        bob = self._add_user("records-b@example.com", "B")
        self._submit(alice, "ice-age", 100, 1, difficulty=0)
        self._submit(alice, "ice-age", 200, 2, difficulty=1)
        self._submit(bob, "ice-age", 300, 3)
        request = Request({"type": "http", "headers": []})
        with self.assertRaises(HTTPException) as error:
            minigame_routes.get_personal_records(request, Response())
        self.assertEqual(error.exception.status_code, 401)
        response = Response()
        with patch.object(minigame_routes, "require_user", return_value={"id": alice}):
            result = minigame_routes.get_personal_records(request, response)
        self.assertEqual(result["user_id"], alice)
        self.assertEqual([r["best_score"] for r in result["records"]], [100, 200])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("record_blob", result["records"][0])

    def test_legacy_data_cannot_enter_personal_records_or_either_leaderboard(self) -> None:
        user = self._add_user("legacy-hidden@example.com", "Legacy")
        self._submit(user, "ice-age", 100000, 4)
        with auth_db() as db:
            db.execute("UPDATE minigame_high_scores SET verification_level = 'legacy'")
        self.assertEqual(personal_records(user)["records"], [])
        self.assertEqual(game_leaderboard("ice-age", difficulty=1, limit=100)["entries"], [])
        self.assertEqual(trophy_leaderboard(difficulty=1, limit=100)["entries"], [])

    def test_tile_only_improvement_is_a_submission_candidate(self) -> None:
        user = self._add_user("tile-only@example.com", "Tile")
        self._submit(user, "column-chaos", 2000, 4)
        summary = self._summary(score=500, trophy=1)
        summary["highest_tile_exp"] = 11
        run = self._run(user)
        qualified = self._qualify(run, user, **summary)
        self.assertTrue(qualified["candidate"])
        self.assertIn("tile_improvement", qualified["reasons"])

    def test_tile_migration_uses_retained_verified_results_once(self) -> None:
        user = self._add_user("tile-migration@example.com", "Migration")
        self._submit(user, "column-chaos", 2000, 3)
        run = self._run(user)
        with auth_db() as db:
            db.execute("UPDATE minigame_ranked_runs SET status='verified', verified_summary_json=? WHERE run_id=?",
                       (json.dumps({"highest_tile_exp": 12}), run["run_id"]))
            db.execute("""
                INSERT INTO minigame_ranked_checkpoints
                (run_id, revision, status, claimed_summary_json, record_hash, action_count,
                 submitted_at, verified_summary_json)
                VALUES (?, 1, 'verified', '{}', 'migration-hash', 1, '2026-09-07', ?)
            """, (run["run_id"], json.dumps({"highest_tile_exp": 14})))
            db.execute("ALTER TABLE minigame_high_scores DROP COLUMN best_tile_exp")
        init_auth_db()
        self.assertEqual(personal_records(user)["records"][0]["best_tile_exp"], 14)
        with auth_db() as db:
            db.execute("UPDATE minigame_high_scores SET best_tile_exp = 11")
        init_auth_db()
        self.assertEqual(personal_records(user)["records"][0]["best_tile_exp"], 11)

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

    def test_accepts_registered_special_board_cells(self) -> None:
        summary = self._summary()
        summary["final_board"] = [-3, -2, -1, 0] * 4
        normalized = _normalize_summary(**summary)
        self.assertEqual(normalized["final_board"], summary["final_board"])

        user_id = self._add_user("special-board@example.com", "Special Board")
        saved = submit_score(
            user_id=user_id,
            game_id="isolated-island",
            difficulty=1,
            score=100,
            trophy_tier=0,
            highest_tile_exp=4,
            final_board=summary["final_board"],
            board_rows=4,
            board_cols=4,
        )
        self.assertTrue(saved["score_updated"])

    def test_accepts_shape_shifter_board_dimensions(self) -> None:
        summary = self._summary()
        summary.update({
            "final_board": [-1] * 55,
            "board_rows": 5,
            "board_cols": 11,
        })
        normalized = _normalize_summary(**summary)
        self.assertEqual(normalized["board_rows"], 5)
        self.assertEqual(normalized["board_cols"], 11)

        payload = minigame_routes.SubmitRankedCheckpointRequest.model_validate({
            "run_token": "r" * 16,
            "lease_token": "l" * 16,
            "revision": 1,
            "record_encoding": "x",
            **summary,
        })
        self.assertEqual(len(payload.final_board), 55)

    def test_rejects_unknown_negative_board_cell(self) -> None:
        summary = self._summary()
        summary["final_board"] = [-4, 0, 1, 2] * 4
        with self.assertRaisesRegex(ValueError, "invalid_board_data"):
            _normalize_summary(**summary)

    def test_backend_catalog_matches_frontend_registry(self) -> None:
        registry_path = Path(__file__).resolve().parents[1] / "frontend" / "src" / "features" / "minigames" / "engine" / "registry.js"
        source = registry_path.read_text(encoding="utf-8")
        frontend_ids = set(re.findall(r"\bid:\s*'([^']+)'", source))
        self.assertEqual(frontend_ids, set(MINIGAME_BY_ID))

    @staticmethod
    def _summary(score: int = 1_000, trophy: int = 1) -> dict:
        return {
            "score": score,
            "trophy_tier": trophy,
            "highest_tile_exp": 10,
            "final_board": [0, 1, 2, 3] * 4,
            "board_rows": 4,
            "board_cols": 4,
            "action_count": 120,
            "elapsed_ms": 60_000,
        }

    @staticmethod
    def _record(run: dict, summary: dict | None = None, *, end_reason: int = 0) -> str:
        claimed = summary or MinigameRankingTests._summary()

        def uleb(value: int) -> bytes:
            result = bytearray()
            remaining = int(value)
            while True:
                byte = remaining & 0x7F
                remaining >>= 7
                result.append(byte | (0x80 if remaining else 0))
                if not remaining:
                    return bytes(result)

        content = bytearray(b"MGO1\x01")
        content.extend(uleb(int(run["rules_version"])))
        content.extend((3, int(run["difficulty"]), 0))
        content.extend(uuid.UUID(str(run["run_id"])).bytes)
        content.extend(bytes.fromhex(str(run["seed_hex"])))
        action_count = int(claimed["action_count"])
        elapsed_ms = int(claimed["elapsed_ms"])
        for index in range(action_count):
            content.append(index % 4)
            content.extend(uleb(elapsed_ms if index == 0 else 0))
        content.extend((0x7F, 0, int(end_reason) & 0xFF))
        content.extend((binascii.crc32(content) & 0xFFFFFFFF).to_bytes(4, "little"))
        return MGO_RECORD_PREFIX + base64.b64encode(content).decode("ascii")

    def _run(
        self,
        user_id: int,
        request_id: str = "run-request",
        *,
        now: datetime | None = None,
    ) -> dict:
        return create_ranked_run(
            user_id=user_id,
            request_id=request_id,
            game_id="column-chaos",
            difficulty=1,
            ip_address="203.0.113.10",
            lease_token=f"lease-{request_id}-0123456789abcdef",
            now=now,
        )

    def _qualify(self, run: dict, user_id: int, **summary) -> dict:
        return qualify_ranked_run(
            run_id=run["run_id"],
            user_id=user_id,
            run_token=run["run_token"],
            lease_token=run["lease_token"],
            **(summary or self._summary()),
        )

    def test_run_creation_is_idempotent_and_seed_is_hmac_derived(self) -> None:
        user_id = self._add_user("seed@example.com", "Seed")
        fixed_salt = "0123456789abcdef0123456789abcdef"
        with patch("backend.minigame_rankings.service.secrets.token_hex", return_value=fixed_salt):
            first = self._run(user_id, "same-request")
        second = self._run(user_id, "same-request")
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["run_token"], second["run_token"])
        self.assertEqual(first["seed_hex"], second["seed_hex"])
        self.assertEqual(
            first["seed_hex"],
            _derive_seed_hex(
                run_id=first["run_id"],
                user_id=user_id,
                game_id="column-chaos",
                difficulty=1,
                rules_version=1,
                salt_hex=fixed_salt,
                started_at=first["started_at"],
                ip_address="203.0.113.10",
            ),
        )
        self.assertEqual(len(first["seed_hex"]), 32)
        self.assertNotEqual(
            first["seed_hex"],
            _derive_seed_hex(
                run_id=first["run_id"],
                user_id=user_id,
                game_id="column-chaos",
                difficulty=1,
                rules_version=1,
                salt_hex=fixed_salt,
                started_at=first["started_at"],
                ip_address="198.51.100.20",
            ),
        )
        with auth_db() as db:
            row = db.execute(
                "SELECT seed_salt_hex, start_ip FROM minigame_ranked_runs WHERE run_id = ?",
                (first["run_id"],),
            ).fetchone()
        self.assertEqual(row["seed_salt_hex"], fixed_salt)
        self.assertEqual(row["start_ip"], "203.0.113.10")

    def test_death_checkpoints_keep_parent_run_active_and_update_one_pb_row(self) -> None:
        user_id = self._add_user("checkpoint@example.com", "Checkpoint")
        run = self._run(user_id, "checkpoint-run")
        first_summary = self._summary(score=1_000, trophy=1)
        first_record = self._record(run, first_summary)
        first = submit_ranked_checkpoint(
            run_id=run["run_id"],
            user_id=user_id,
            run_token=run["run_token"],
            lease_token=run["lease_token"],
            revision=1,
            record_encoding=first_record,
            ip_address="203.0.113.10",
            **first_summary,
        )
        duplicate = submit_ranked_checkpoint(
            run_id=run["run_id"],
            user_id=user_id,
            run_token=run["run_token"],
            lease_token=run["lease_token"],
            revision=1,
            record_encoding=first_record,
            ip_address="203.0.113.10",
            **first_summary,
        )
        self.assertEqual(first["checkpoint_id"], duplicate["checkpoint_id"])

        claimed = claim_pending_checkpoint()
        self.assertEqual(claimed["checkpoint_id"], first["checkpoint_id"])
        finish_checkpoint_verified(claimed["checkpoint_id"], **first_summary)
        self.assertEqual(get_ranked_run(run_id=run["run_id"], user_id=user_id)["status"], "active")

        second_summary = self._summary(score=1_500, trophy=2)
        second_summary["action_count"] = 121
        second_record = self._record(run, second_summary)
        second = submit_ranked_checkpoint(
            run_id=run["run_id"],
            user_id=user_id,
            run_token=run["run_token"],
            lease_token=run["lease_token"],
            revision=2,
            record_encoding=second_record,
            ip_address="203.0.113.10",
            **second_summary,
        )
        claimed = claim_pending_checkpoint()
        self.assertEqual(claimed["checkpoint_id"], second["checkpoint_id"])
        finish_checkpoint_verified(claimed["checkpoint_id"], **second_summary)

        with auth_db() as db:
            rows = db.execute(
                "SELECT best_score, trophy_tier, score_run_id, trophy_run_id FROM minigame_high_scores"
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["best_score"]), 1_500)
        self.assertEqual(int(rows[0]["trophy_tier"]), 2)
        self.assertEqual(rows[0]["score_run_id"], run["run_id"])
        self.assertEqual(rows[0]["trophy_run_id"], run["run_id"])

    def test_checkpoint_verifier_queue_does_not_close_the_parent_run(self) -> None:
        user_id = self._add_user("checkpoint-worker@example.com", "Checkpoint Worker")
        run = self._run(user_id, "checkpoint-worker-run")
        summary = self._summary(score=2_000, trophy=3)
        checkpoint = submit_ranked_checkpoint(
            run_id=run["run_id"],
            user_id=user_id,
            run_token=run["run_token"],
            lease_token=run["lease_token"],
            revision=1,
            record_encoding=self._record(run, summary),
            ip_address="203.0.113.10",
            **summary,
        )
        with patch.object(minigame_verifier, "verify_ranked_run", return_value=summary):
            self.assertTrue(minigame_verifier.process_one_pending_run())
        with auth_db() as db:
            saved_checkpoint = db.execute(
                "SELECT status, pending_record FROM minigame_ranked_checkpoints WHERE id = ?",
                (checkpoint["checkpoint_id"],),
            ).fetchone()
            saved_run = db.execute(
                "SELECT status, lease_token_hash FROM minigame_ranked_runs WHERE run_id = ?",
                (run["run_id"],),
            ).fetchone()
        self.assertEqual(saved_checkpoint["status"], "verified")
        self.assertIsNone(saved_checkpoint["pending_record"])
        self.assertEqual(saved_run["status"], "active")
        self.assertTrue(saved_run["lease_token_hash"])

    def test_active_run_is_exclusive_and_explicit_replacement_requires_lease(self) -> None:
        user_id = self._add_user("exclusive@example.com", "Exclusive")
        first = self._run(user_id, "exclusive-first")
        with self.assertRaisesRegex(RuntimeError, "active_run_exists"):
            self._run(user_id, "exclusive-second")
        with self.assertRaisesRegex(PermissionError, "lease_mismatch"):
            create_ranked_run(
                user_id=user_id,
                request_id="exclusive-second",
                game_id="column-chaos",
                difficulty=1,
                ip_address="203.0.113.10",
                lease_token="lease-exclusive-second-0123456789",
                replace_run_id=first["run_id"],
                replace_lease_token="wrong-lease-token-0123456789",
            )
        replacement = create_ranked_run(
            user_id=user_id,
            request_id="exclusive-second",
            game_id="column-chaos",
            difficulty=1,
            ip_address="203.0.113.10",
            lease_token="lease-exclusive-second-0123456789",
            replace_run_id=first["run_id"],
            replace_lease_token=first["lease_token"],
        )
        self.assertNotEqual(replacement["run_id"], first["run_id"])

    def test_submit_route_forwards_the_lease_token(self) -> None:
        payload = minigame_routes.SubmitRankedRunRequest(
            submission_token="submission-token-0123456789",
            lease_token="lease-token-0123456789",
            record_encoding="MINIGAME_v1MGO_B64_AAAA",
        )
        with (
            patch.object(minigame_routes, "require_user", return_value={"id": 7}),
            patch.object(minigame_routes, "client_ip", return_value="203.0.113.20"),
            patch.object(minigame_routes, "_check_submit_rate"),
            patch.object(
                minigame_routes,
                "submit_ranked_run",
                return_value={"status": "pending"},
            ) as submit,
        ):
            result = minigame_routes.submit_run(
                "run-1",
                payload,
                object(),
                Response(),
            )
        self.assertEqual(result["status"], "pending")
        submit.assert_called_once_with(
            run_id="run-1",
            user_id=7,
            submission_token="submission-token-0123456789",
            lease_token="lease-token-0123456789",
            record_encoding="MINIGAME_v1MGO_B64_AAAA",
            ip_address="203.0.113.20",
        )

    def test_lease_heartbeat_and_expired_claim_rotate_generation(self) -> None:
        user_id = self._add_user("lease@example.com", "Lease")
        started = datetime.now(timezone.utc)
        run = self._run(user_id, "lease-run", now=started)
        heartbeat = heartbeat_ranked_run(
            run_id=run["run_id"], user_id=user_id,
            lease_token=run["lease_token"], now=started + timedelta(seconds=15),
        )
        self.assertEqual(heartbeat["lease_generation"], 1)
        with self.assertRaisesRegex(RuntimeError, "lease_active"):
            claim_ranked_run(
                run_id=run["run_id"], user_id=user_id,
                lease_token="replacement-lease-token-0123456789",
                now=started + timedelta(seconds=30),
            )
        claimed = claim_ranked_run(
            run_id=run["run_id"], user_id=user_id,
            lease_token="replacement-lease-token-0123456789",
            now=started + timedelta(seconds=76),
        )
        self.assertEqual(claimed["lease_generation"], 2)
        with self.assertRaisesRegex(PermissionError, "lease_mismatch"):
            heartbeat_ranked_run(
                run_id=run["run_id"], user_id=user_id,
                lease_token=run["lease_token"], now=started + timedelta(seconds=77),
            )

    def test_qualification_is_frozen_and_same_summary_retry_is_idempotent(self) -> None:
        user_id = self._add_user("freeze@example.com", "Freeze")
        run = self._run(user_id, "freeze-run")
        first = self._qualify(run, user_id, **self._summary(score=2_000, trophy=2))
        repeated = self._qualify(run, user_id, **self._summary(score=2_000, trophy=2))
        self.assertEqual(repeated["submission_token"], first["submission_token"])
        with self.assertRaisesRegex(ValueError, "qualification_conflict"):
            self._qualify(run, user_id, **self._summary(score=2_001, trophy=2))

    def test_run_ownership_and_run_token_tamper_and_expiry(self) -> None:
        owner = self._add_user("owner@example.com", "Owner")
        stranger = self._add_user("stranger@example.com", "Stranger")
        run = self._run(owner)
        with self.assertRaises(PermissionError):
            self._qualify(run, stranger)

        tampered = dict(run)
        tampered["run_token"] = run["run_token"][:-1] + ("A" if run["run_token"][-1] != "A" else "B")
        with self.assertRaises(RunTokenError):
            self._qualify(tampered, owner)
        abandon_ranked_run(
            run_id=run["run_id"], user_id=owner, lease_token=run["lease_token"]
        )

        old_start = datetime.now(timezone.utc) - timedelta(days=2)
        expired = self._run(owner, "expired-run", now=old_start)
        with self.assertRaises(RunTokenExpired):
            self._qualify(expired, owner)

    def test_candidate_filter_requires_pb_trophy_or_strict_top_100(self) -> None:
        user_id = self._add_user("candidate@example.com", "Candidate")
        self._submit(user_id, "column-chaos", 2_000, 2)
        for index in range(99):
            other = self._add_user(f"top-{index}@example.com", f"Top {index}")
            self._submit(other, "column-chaos", 1_000 + index, 1)

        not_candidate_run = self._run(user_id, "not-candidate")
        not_candidate = self._qualify(
            not_candidate_run,
            user_id,
            **self._summary(score=1_000, trophy=2),
        )
        self.assertFalse(not_candidate["candidate"])
        self.assertEqual(not_candidate["status"], "not_candidate")

        pb_run = self._run(user_id, "pb-candidate")
        pb = self._qualify(pb_run, user_id, **self._summary(score=2_001, trophy=2))
        self.assertTrue(pb["candidate"])
        self.assertIn("personal_best", pb["reasons"])
        abandon_ranked_run(
            run_id=pb_run["run_id"], user_id=user_id,
            lease_token=pb_run["lease_token"],
        )

        trophy_run = self._run(user_id, "trophy-candidate")
        trophy = self._qualify(trophy_run, user_id, **self._summary(score=999, trophy=3))
        self.assertTrue(trophy["candidate"])
        self.assertIn("trophy_improvement", trophy["reasons"])

        top_user = self._add_user("top-candidate@example.com", "Top Candidate")
        top_run = self._run(top_user, "top-candidate")
        top = self._qualify(top_run, top_user, **self._summary(score=1_001, trophy=2))
        self.assertTrue(top["candidate"])
        self.assertIn("personal_best", top["reasons"])
        self.assertIn("top_100", top["reasons"])

    def test_submission_token_is_one_time_and_pending_claim_is_atomic(self) -> None:
        user_id = self._add_user("submit@example.com", "Submit")
        run = self._run(user_id)
        qualified = self._qualify(run, user_id)
        record = self._record(run)
        submitted = submit_ranked_run(
            run_id=run["run_id"],
            user_id=user_id,
            submission_token=qualified["submission_token"],
            lease_token=run["lease_token"],
            record_encoding=record,
            ip_address="203.0.113.11",
        )
        self.assertEqual(submitted["status"], "pending")

        repeated = submit_ranked_run(
            run_id=run["run_id"],
            user_id=user_id,
            submission_token=qualified["submission_token"],
            lease_token=run["lease_token"],
            record_encoding=record,
            ip_address="203.0.113.11",
        )
        self.assertEqual(repeated["status"], "pending")
        with self.assertRaisesRegex(ValueError, "submission_already_used"):
            submit_ranked_run(
                run_id=run["run_id"],
                user_id=user_id,
                submission_token=qualified["submission_token"],
                lease_token=run["lease_token"],
                record_encoding=self._record(run, end_reason=1),
                ip_address="203.0.113.11",
            )

        claimed = claim_pending()
        self.assertEqual(claimed["run_id"], run["run_id"])
        self.assertEqual(claimed["game_id"], "column-chaos")
        self.assertEqual(claimed["difficulty"], 1)
        self.assertEqual(claimed["claimed_summary"]["score"], 1_000)
        self.assertIsNone(claim_pending())
        rejected = finish_rejected(run["run_id"], "fixture_rejected")
        self.assertEqual(rejected["status"], "rejected")
        with auth_db() as db:
            row = db.execute(
                "SELECT pending_record, submission_token_consumed_at FROM minigame_ranked_runs WHERE run_id = ?",
                (run["run_id"],),
            ).fetchone()
        self.assertIsNone(row["pending_record"])
        self.assertIsNotNone(row["submission_token_consumed_at"])

    def test_verifier_outage_requeues_without_rejecting_the_run(self) -> None:
        user_id = self._add_user("outage@example.com", "Outage")
        run = self._run(user_id, "outage-run")
        qualified = self._qualify(run, user_id)
        submit_ranked_run(
            run_id=run["run_id"],
            user_id=user_id,
            submission_token=qualified["submission_token"],
            lease_token=run["lease_token"],
            record_encoding=self._record(run),
            ip_address="203.0.113.11",
        )
        minigame_verifier._retry_after_monotonic = 0.0
        try:
            with patch.object(
                minigame_verifier,
                "verify_ranked_run",
                side_effect=minigame_verifier.MinigameVerifierUnavailable("offline"),
            ):
                self.assertFalse(minigame_verifier.process_one_pending_run())
        finally:
            minigame_verifier._retry_after_monotonic = 0.0
        with auth_db() as db:
            saved = db.execute(
                "SELECT status, error_code FROM minigame_ranked_runs WHERE run_id = ?",
                (run["run_id"],),
            ).fetchone()
        self.assertEqual(saved["status"], "pending")
        self.assertIsNone(saved["error_code"])

    def test_finish_verified_updates_verified_pb_and_discards_pending_payload(self) -> None:
        user_id = self._add_user("verified@example.com", "Verified")
        run = self._run(user_id)
        summary = self._summary(score=3_000, trophy=3)
        qualified = self._qualify(run, user_id, **summary)
        submit_ranked_run(
            run_id=run["run_id"],
            user_id=user_id,
            submission_token=qualified["submission_token"],
            lease_token=run["lease_token"],
            record_encoding=self._record(run, summary),
            ip_address="203.0.113.12",
        )
        self.assertIsNotNone(claim_pending())
        result = finish_verified(run["run_id"], **summary)
        self.assertEqual(result["status"], "verified")
        self.assertTrue(result["score_updated"])
        self.assertTrue(result["trophy_updated"])
        with auth_db() as db:
            run_row = db.execute(
                "SELECT pending_record, verified_summary_json FROM minigame_ranked_runs WHERE run_id = ?",
                (run["run_id"],),
            ).fetchone()
            score_row = db.execute(
                """
                SELECT score_run_id, trophy_run_id, verification_level,
                       score_verified_at, trophy_verified_at, record_blob
                FROM minigame_high_scores
                WHERE user_id = ? AND game_id = 'column-chaos' AND difficulty = 1
                """,
                (user_id,),
            ).fetchone()
        self.assertIsNone(run_row["pending_record"])
        self.assertIsNotNone(run_row["verified_summary_json"])
        self.assertEqual(score_row["score_run_id"], run["run_id"])
        self.assertEqual(score_row["trophy_run_id"], run["run_id"])
        self.assertEqual(score_row["verification_level"], "verified")
        self.assertIsNotNone(score_row["score_verified_at"])
        self.assertIsNotNone(score_row["trophy_verified_at"])
        self.assertTrue(str(score_row["record_blob"]).startswith(MGO_RECORD_PREFIX))

    def test_first_verified_result_replaces_a_higher_legacy_score(self) -> None:
        user_id = self._add_user("legacy@example.com", "Legacy")
        self._submit(user_id, "column-chaos", 99_999, 4)
        with auth_db() as db:
            db.execute(
                """
                UPDATE minigame_high_scores
                SET verification_level = 'legacy'
                WHERE user_id = ? AND game_id = 'column-chaos' AND difficulty = 1
                """,
                (user_id,),
            )

        run = self._run(user_id, "replace-legacy")
        summary = self._summary(score=1_500, trophy=1)
        qualified = self._qualify(run, user_id, **summary)
        self.assertTrue(qualified["candidate"])
        submit_ranked_run(
            run_id=run["run_id"],
            user_id=user_id,
            submission_token=qualified["submission_token"],
            lease_token=run["lease_token"],
            record_encoding=self._record(run, summary),
            ip_address="203.0.113.12",
        )
        self.assertIsNotNone(claim_pending())
        finish_verified(run["run_id"], **summary)
        with auth_db() as db:
            score = db.execute(
                """
                SELECT best_score, trophy_tier, verification_level
                FROM minigame_high_scores
                WHERE user_id = ? AND game_id = 'column-chaos' AND difficulty = 1
                """,
                (user_id,),
            ).fetchone()
        self.assertEqual(int(score["best_score"]), 1_500)
        self.assertEqual(int(score["trophy_tier"]), 1)
        self.assertEqual(score["verification_level"], "verified")

    def test_submission_token_tamper_and_expiry_are_rejected(self) -> None:
        user_id = self._add_user("token@example.com", "Token")
        run = self._run(user_id)
        qualified = self._qualify(run, user_id)
        token = qualified["submission_token"]
        tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
        with self.assertRaises(RunTokenError):
            submit_ranked_run(
                run_id=run["run_id"], user_id=user_id,
                submission_token=tampered, lease_token=run["lease_token"],
                record_encoding=self._record(run),
                ip_address="203.0.113.11",
            )
        future = datetime.now(timezone.utc) + timedelta(minutes=11)
        with self.assertRaises(RunTokenExpired):
            submit_ranked_run(
                run_id=run["run_id"], user_id=user_id,
                submission_token=token, lease_token=run["lease_token"],
                record_encoding=self._record(run),
                ip_address="203.0.113.11", now=future,
            )

    def test_pending_limits_and_record_size_are_enforced_before_consumption(self) -> None:
        user_id = self._add_user("limited@example.com", "Limited")
        first = self._run(user_id, "pending-first")
        first_qualified = self._qualify(first, user_id)
        submit_ranked_run(
            run_id=first["run_id"], user_id=user_id,
            submission_token=first_qualified["submission_token"],
            lease_token=first["lease_token"],
            record_encoding=self._record(first), ip_address="203.0.113.13",
        )

        second = self._run(user_id, "pending-second")
        second_qualified = self._qualify(second, user_id)
        with self.assertRaisesRegex(RuntimeError, "user_pending_limit"):
            submit_ranked_run(
                run_id=second["run_id"], user_id=user_id,
                submission_token=second_qualified["submission_token"],
                lease_token=second["lease_token"],
                record_encoding=self._record(second), ip_address="203.0.113.13",
            )
        with auth_db() as db:
            second_row = db.execute(
                "SELECT status, submission_token_consumed_at FROM minigame_ranked_runs WHERE run_id = ?",
                (second["run_id"],),
            ).fetchone()
        self.assertEqual(second_row["status"], "qualified")
        self.assertIsNone(second_row["submission_token_consumed_at"])

        oversized_user = self._add_user("oversized@example.com", "Oversized")
        oversized = self._run(oversized_user, "oversized")
        oversized_qualified = self._qualify(oversized, oversized_user)
        oversized_record = MGO_RECORD_PREFIX + base64.b64encode(b"x" * (256 * 1024 + 1)).decode("ascii")
        with self.assertRaisesRegex(ValueError, "record_too_large"):
            submit_ranked_run(
                run_id=oversized["run_id"], user_id=oversized_user,
                submission_token=oversized_qualified["submission_token"],
                lease_token=oversized["lease_token"],
                record_encoding=oversized_record, ip_address="203.0.113.14",
            )

    def test_existing_score_table_is_safely_extended_by_migration(self) -> None:
        user_id = self._add_user("migration@example.com", "Migration")
        with auth_db() as db:
            db.execute("DROP TABLE minigame_high_scores")
            db.execute("DROP TABLE minigame_ranked_runs")
            db.execute(
                """
                CREATE TABLE minigame_high_scores (
                  user_id INTEGER NOT NULL,
                  game_id TEXT NOT NULL,
                  difficulty INTEGER NOT NULL,
                  best_score INTEGER NOT NULL DEFAULT 0,
                  trophy_tier INTEGER NOT NULL DEFAULT 0,
                  highest_tile_exp INTEGER NOT NULL DEFAULT 0,
                  final_board_json TEXT NOT NULL,
                  board_rows INTEGER NOT NULL DEFAULT 4,
                  board_cols INTEGER NOT NULL DEFAULT 4,
                  score_achieved_at TEXT NOT NULL,
                  trophy_achieved_at TEXT,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(user_id, game_id, difficulty)
                )
                """
            )
            db.execute(
                """
                CREATE TABLE minigame_ranked_runs (
                  run_id TEXT PRIMARY KEY,
                  user_id INTEGER NOT NULL,
                  request_id TEXT NOT NULL,
                  game_id TEXT NOT NULL,
                  difficulty INTEGER NOT NULL,
                  rules_version INTEGER NOT NULL,
                  seed_salt_hex TEXT NOT NULL,
                  seed_hex TEXT NOT NULL,
                  status TEXT NOT NULL,
                  started_at TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  start_ip TEXT,
                  qualified_at TEXT,
                  qualification_expires_at TEXT,
                  submission_token_hash TEXT,
                  submission_token_consumed_at TEXT,
                  claimed_summary_json TEXT,
                  pending_record TEXT,
                  record_hash TEXT,
                  action_count INTEGER,
                  submitted_at TEXT,
                  submit_ip TEXT,
                  validation_started_at TEXT,
                  completed_at TEXT,
                  verified_summary_json TEXT,
                  error_code TEXT,
                  UNIQUE(user_id, request_id)
                )
                """
            )
            db.execute(
                """
                INSERT INTO minigame_ranked_runs
                (run_id, user_id, request_id, game_id, difficulty, rules_version,
                 seed_salt_hex, seed_hex, status, started_at, expires_at)
                VALUES ('legacy-active', ?, 'legacy-request', 'column-chaos', 1, 1,
                        ?, ?, 'active', ?, ?)
                """,
                (
                    user_id,
                    "00" * 16,
                    "11" * 16,
                    "2026-08-01T00:00:00+00:00",
                    "2026-09-01T00:00:00+00:00",
                ),
            )
        init_auth_db()
        with auth_db() as db:
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(minigame_high_scores)")
            }
            run_table = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'minigame_ranked_runs'"
            ).fetchone()
            run_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(minigame_ranked_runs)")
            }
            migrated_run = db.execute(
                "SELECT status, error_code FROM minigame_ranked_runs WHERE run_id = 'legacy-active'"
            ).fetchone()
        self.assertTrue(
            {
                "score_run_id", "trophy_run_id", "verification_level",
                "score_verified_at", "trophy_verified_at", "record_hash", "record_blob",
            }.issubset(columns)
        )
        self.assertIsNotNone(run_table)
        self.assertTrue(
            {
                "lease_token_hash", "lease_expires_at", "lease_last_seen_at", "lease_generation",
            }.issubset(run_columns)
        )
        self.assertEqual(migrated_run["status"], "expired")
        self.assertEqual(migrated_run["error_code"], "lease_required")

    def test_global_pending_queue_is_capped_at_32(self) -> None:
        user_id = self._add_user("queue-target@example.com", "Queue Target")
        filler_id = self._add_user("queue-filler@example.com", "Queue Filler")
        run = self._run(user_id, "queue-target")
        qualified = self._qualify(run, user_id)
        now = datetime.now(timezone.utc)
        with auth_db() as db:
            for index in range(32):
                db.execute(
                    """
                    INSERT INTO minigame_ranked_runs
                    (run_id, user_id, request_id, game_id, difficulty, rules_version,
                     seed_salt_hex, seed_hex, status, started_at, expires_at,
                     submitted_at, pending_record, record_hash)
                    VALUES (?, ?, ?, 'column-chaos', 1, 1, ?, ?, 'pending', ?, ?, ?, ?, ?)
                    """,
                    (
                        f"filler-run-{index}", filler_id, f"filler-request-{index}",
                        "00" * 16, "11" * 16, now.isoformat(),
                        (now + timedelta(days=1)).isoformat(), now.isoformat(),
                        self._record(run), f"hash-{index}",
                    ),
                )
        with self.assertRaisesRegex(RuntimeError, "queue_full"):
            submit_ranked_run(
                run_id=run["run_id"], user_id=user_id,
                submission_token=qualified["submission_token"],
                lease_token=run["lease_token"],
                record_encoding=self._record(run), ip_address="203.0.113.15",
            )

    def test_get_run_does_not_expose_salt_or_pending_record(self) -> None:
        user_id = self._add_user("get@example.com", "Get")
        run = self._run(user_id)
        public = get_ranked_run(run_id=run["run_id"], user_id=user_id)
        self.assertNotIn("seed_salt_hex", public)
        self.assertNotIn("pending_record", public)
        self.assertNotIn("run_token", public)



if __name__ == "__main__":
    unittest.main()
