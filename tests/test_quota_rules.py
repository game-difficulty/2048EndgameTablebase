from __future__ import annotations

import unittest

from backend.quota.config import table_multiplier_config, token_cost_config
from backend.quota.routes import public_quota_rules


class PublicQuotaRulesTests(unittest.TestCase):
    def setUp(self) -> None:
        table_multiplier_config.cache_clear()
        token_cost_config.cache_clear()

    def tearDown(self) -> None:
        table_multiplier_config.cache_clear()
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
        self.assertEqual(groups["free10"], 5)
        self.assertEqual(groups["free11"], 50)
        self.assertEqual(groups["4442f"], 50)
        self.assertEqual(rules["operation_costs"]["analysis_per_replay"], 100)
        self.assertEqual(rules["operation_costs"]["replay_load"], 3)


if __name__ == "__main__":
    unittest.main()
