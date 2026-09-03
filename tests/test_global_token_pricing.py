from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.auth.db import auth_db, init_auth_db
from backend.quota.config import (
    MULTIPLIER_UNIT,
    apply_pricing_multipliers,
    clear_token_pricing_cache,
    resolve_pricing_snapshot,
)
from backend.quota.service import (
    consume_operation_tokens,
    consume_operation_tokens_once,
    finalize_reservation,
    get_token_balance,
    load_token_reservation,
    reserve_operation_tokens,
)


class GlobalTokenPricingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_db = os.environ.get("CLOUD_AUTH_DB")
        self.previous_pricing = os.environ.get("CLOUD_TOKEN_PRICING_CONFIG")
        self.previous_override = os.environ.get("CLOUD_TOKEN_GLOBAL_MULTIPLIER")
        self.db_path = Path(self.tempdir.name) / "auth.sqlite3"
        self.pricing_path = Path(self.tempdir.name) / "pricing.json"
        os.environ["CLOUD_AUTH_DB"] = str(self.db_path)
        os.environ["CLOUD_TOKEN_PRICING_CONFIG"] = str(self.pricing_path)
        os.environ.pop("CLOUD_TOKEN_GLOBAL_MULTIPLIER", None)
        self._write_pricing(1, "standard")
        init_auth_db()
        with auth_db() as db:
            now = "2026-09-03T00:00:00+00:00"
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_identity, password_hash, display_name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("pricing@example.com", "pricing@example.com", "test", "Pricing", now, now),
            )
            self.user_id = int(cursor.lastrowid)
            db.execute(
                """
                INSERT INTO token_accounts
                (user_id, bonus_balance_units, paid_balance_units, created_at, updated_at)
                VALUES (?, 100000000, 0, ?, ?)
                """,
                (self.user_id, now, now),
            )

    def tearDown(self) -> None:
        if self.previous_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self.previous_db
        if self.previous_pricing is None:
            os.environ.pop("CLOUD_TOKEN_PRICING_CONFIG", None)
        else:
            os.environ["CLOUD_TOKEN_PRICING_CONFIG"] = self.previous_pricing
        if self.previous_override is None:
            os.environ.pop("CLOUD_TOKEN_GLOBAL_MULTIPLIER", None)
        else:
            os.environ["CLOUD_TOKEN_GLOBAL_MULTIPLIER"] = self.previous_override
        clear_token_pricing_cache()
        self.tempdir.cleanup()

    def _write_pricing(self, multiplier: float, policy_key: str) -> None:
        self.pricing_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "policy_key": policy_key,
                    "global_multiplier": multiplier,
                }
            ),
            encoding="utf-8",
        )
        clear_token_pricing_cache()

    def test_default_multiplier_preserves_existing_costs(self) -> None:
        pricing = resolve_pricing_snapshot()
        self.assertEqual(pricing.global_multiplier_units, MULTIPLIER_UNIT)
        self.assertEqual(pricing.policy_key, "standard")
        self.assertEqual(apply_pricing_multipliers(1_000, 8_000, 1_000), 8_000)

    def test_global_discount_applies_after_table_or_override_multiplier(self) -> None:
        self._write_pricing(0.5, "half_price")
        consume_operation_tokens(
            user_id=self.user_id,
            session_id=None,
            operation_key="trainer_lookup_hit",
            full_pattern="free10_512",
        )
        consume_operation_tokens(
            user_id=self.user_id,
            session_id=None,
            operation_key="replay_load",
            multiplier_override_units=MULTIPLIER_UNIT,
        )
        self.assertEqual(get_token_balance(self.user_id)["total"], 99_994.5)
        with auth_db() as db:
            rows = db.execute(
                """
                SELECT operation_key, table_multiplier_units, global_multiplier_units,
                       pricing_policy_key, final_cost_units
                FROM token_ledger
                WHERE event_type = 'consume'
                ORDER BY id
                """
            ).fetchall()
        self.assertEqual(
            [tuple(row) for row in rows],
            [
                ("trainer_lookup_hit", 8_000, 500, "half_price", 4_000),
                ("replay_load", 1_000, 500, "half_price", 1_500),
            ],
        )

    def test_reservation_uses_original_pricing_snapshot_after_reload(self) -> None:
        self._write_pricing(0.5, "half_price")
        reservation = reserve_operation_tokens(
            user_id=self.user_id,
            session_id=None,
            operation_key="trainer_lookup_hit",
            full_pattern="free10_512",
        )
        self.assertIsNotNone(reservation)
        restored = load_token_reservation(reservation.ledger_id)
        self.assertEqual(restored.global_multiplier_units, 500)
        self.assertEqual(restored.pricing_policy_key, "half_price")
        self.assertEqual(restored.reserved_units, 4_000)

        self._write_pricing(2, "peak_price")
        finalize_reservation(
            restored,
            actual_operation_key="trainer_lookup_miss",
        )
        self.assertEqual(get_token_balance(self.user_id)["total"], 99_999.2)
        with auth_db() as db:
            finalized = db.execute(
                """
                SELECT global_multiplier_units, pricing_policy_key, final_cost_units
                FROM token_ledger
                WHERE event_type = 'finalize'
                """
            ).fetchone()
        self.assertEqual(tuple(finalized), (500, "half_price", 800))

    def test_idempotent_consumption_charges_discounted_cost_once(self) -> None:
        self._write_pricing(0.5, "half_price")
        first = consume_operation_tokens_once(
            request_id="discounted-replay-request",
            user_id=self.user_id,
            session_id=None,
            operation_key="replay_load",
            multiplier_override_units=MULTIPLIER_UNIT,
        )
        second = consume_operation_tokens_once(
            request_id="discounted-replay-request",
            user_id=self.user_id,
            session_id=None,
            operation_key="replay_load",
            multiplier_override_units=MULTIPLIER_UNIT,
        )
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(get_token_balance(self.user_id)["total"], 99_998.5)

    def test_environment_override_has_priority(self) -> None:
        self._write_pricing(0.5, "half_price")
        os.environ["CLOUD_TOKEN_GLOBAL_MULTIPLIER"] = "1.25"
        clear_token_pricing_cache()
        pricing = resolve_pricing_snapshot()
        self.assertEqual(pricing.global_multiplier_units, 1_250)
        self.assertEqual(pricing.policy_key, "environment_override")

    def test_file_changes_reload_and_invalid_changes_keep_last_good_value(self) -> None:
        self._write_pricing(1, "standard")
        with patch("backend.quota.config.PRICING_REFRESH_SECONDS", 0):
            self.assertEqual(resolve_pricing_snapshot().global_multiplier_units, 1_000)
            self.pricing_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "policy_key": "promotion",
                        "global_multiplier": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(resolve_pricing_snapshot().global_multiplier_units, 500)
            self.pricing_path.write_text("{invalid", encoding="utf-8")
            self.assertEqual(resolve_pricing_snapshot().global_multiplier_units, 500)

    def test_new_ledger_columns_are_migrated(self) -> None:
        with auth_db() as db:
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(token_ledger)").fetchall()
            }
        self.assertIn("global_multiplier_units", columns)
        self.assertIn("pricing_policy_key", columns)

if __name__ == "__main__":
    unittest.main()
