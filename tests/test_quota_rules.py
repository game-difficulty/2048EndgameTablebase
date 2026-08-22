from __future__ import annotations

import unittest

from backend.quota.config import (
    table_multiplier_config,
    table_threshold_config,
    token_cost_config,
)
from backend.quota.routes import public_quota_rules


class PublicQuotaRulesTests(unittest.TestCase):
    def setUp(self) -> None:
        table_multiplier_config.cache_clear()
        table_threshold_config.cache_clear()
        token_cost_config.cache_clear()

    def tearDown(self) -> None:
        table_multiplier_config.cache_clear()
        table_threshold_config.cache_clear()
        token_cost_config.cache_clear()

    def test_public_rules_match_weekly_grants_and_cost_configuration(self) -> None:
        rules = public_quota_rules()

        self.assertEqual(
            rules["weekly_grants"],
            {
                "public": 512,
                "invited": 4096,
                "supporter": 32768,
                "interval_days": 7,
            },
        )
        groups = {
            pattern: group["multiplier"]
            for group in rules["table_groups"]
            for pattern in group["patterns"]
        }
        self.assertEqual(groups["L3"], 1)
        self.assertEqual(groups["3x3free8"], 1)
        self.assertEqual(groups["free10"], 8)
        self.assertEqual(groups["free11"], 50)
        self.assertEqual(groups["4421"], 5)
        self.assertEqual(groups["444"], 5)
        self.assertEqual(groups["LL"], 8)
        self.assertNotIn("4442f", groups)
        self.assertEqual(rules["operation_costs"]["analysis_per_replay"], 100)
        self.assertEqual(rules["operation_costs"]["replay_load"], 3)
        thresholds = {
            row["full_pattern"]: row
            for row in rules["tablebase_thresholds"]
        }
        self.assertEqual(len(thresholds), 20)
        self.assertEqual(
            thresholds["444_1024"],
            {
                "full_pattern": "444_1024",
                "threshold": 0.7,
                "mode": "absolute",
            },
        )
        self.assertEqual(thresholds["L3_1024"]["mode"], "relative")
        self.assertEqual(thresholds["3x3free8_512"]["threshold"], 0.0)
        self.assertIsNone(thresholds["442t_512"]["threshold"])
        self.assertTrue(all("path" not in row for row in thresholds.values()))


if __name__ == "__main__":
    unittest.main()
