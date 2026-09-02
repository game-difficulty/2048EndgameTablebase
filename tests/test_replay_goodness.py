from __future__ import annotations

import unittest

import numpy as np

from backend.analysis_core import Analyzer
from engine_core.replay_utils import (
    REPLAY_DTYPE,
    analyze_replay,
    replay_change_is_forced,
)


class ReplayGoodnessTests(unittest.TestCase):
    @staticmethod
    def _record(change: int, rates: tuple[int, int, int, int]):
        record = np.zeros(1, dtype=REPLAY_DTYPE)
        record[0]["f1"] = np.uint8(change)
        for index, field in enumerate(("f2", "f3", "f4", "f5")):
            record[0][field] = np.uint32(rates[index])
        return record

    def test_zero_selected_rate_is_not_discarded(self):
        record = self._record(1 << 5, (2_000_000_000, 0, 0, 0))
        analysis = analyze_replay(record)

        self.assertEqual(analysis["losses"].tolist(), [0.0])
        self.assertEqual(analysis["summary"]["final_gof"], 0.0)

    def test_one_hundred_percent_best_is_not_implicitly_forced(self):
        record = self._record(
            1 << 5,
            (4_000_000_000, 2_000_000_000, 0, 0),
        )
        analysis = analyze_replay(record)

        self.assertEqual(analysis["forced"].tolist(), [False])
        self.assertEqual(analysis["losses"].tolist(), [0.5])

    def test_explicit_forced_bit_excludes_the_step(self):
        change = Analyzer.encode(1, 3, 1, forced=True)
        self.assertTrue(replay_change_is_forced(change))

        record = self._record(
            int(change),
            (4_000_000_000, 2_000_000_000, 0, 0),
        )
        analysis = analyze_replay(record)

        self.assertEqual(analysis["forced"].tolist(), [True])
        self.assertEqual(analysis["losses"].tolist(), [1.0])
        self.assertEqual(analysis["summary"]["total_moves"], 0)


if __name__ == "__main__":
    unittest.main()
