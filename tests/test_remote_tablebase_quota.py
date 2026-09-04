from __future__ import annotations

import unittest

from backend.quota.config import (
    apply_multiplier,
    operation_cost_units,
    table_multiplier_config,
    table_multiplier_units,
)


class RemoteTablebaseQuotaTests(unittest.TestCase):
    def setUp(self) -> None:
        table_multiplier_config.cache_clear()

    def tearDown(self) -> None:
        table_multiplier_config.cache_clear()

    def test_remote_tables_use_configured_multipliers(self) -> None:
        self.assertEqual(table_multiplier_units("free11_512"), 50_000)
        self.assertEqual(table_multiplier_units("free11_1024"), 50_000)
        self.assertEqual(table_multiplier_units("free10_128"), 8_000)
        self.assertEqual(table_multiplier_units("free10_256"), 8_000)
        self.assertEqual(table_multiplier_units("free10_512"), 8_000)
        self.assertEqual(table_multiplier_units("4421_1024"), 5_000)
        self.assertEqual(table_multiplier_units("4421_2048"), 5_000)
        self.assertEqual(table_multiplier_units("2432t_2048"), 5_000)
        self.assertEqual(table_multiplier_units("444_1024"), 5_000)
        self.assertEqual(table_multiplier_units("444_2048"), 5_000)
        self.assertEqual(table_multiplier_units("LL_1024"), 8_000)

    def test_remote_lookup_and_analysis_costs(self) -> None:
        multiplier = table_multiplier_units("free11_512")
        hit = apply_multiplier(operation_cost_units("trainer_lookup_hit"), multiplier)
        miss = apply_multiplier(operation_cost_units("trainer_lookup_miss"), multiplier)
        analysis = apply_multiplier(
            operation_cost_units("analysis_per_replay"), multiplier
        )

        self.assertEqual(hit, 50_000)
        self.assertEqual(miss, 10_000)
        self.assertEqual(analysis, 5_000_000)


if __name__ == "__main__":
    unittest.main()
