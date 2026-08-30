from __future__ import annotations

import math
import unittest

from backend.battle.modes.free_goodness.rules import (
    SpawnRiskState,
    decide_move,
    deterministic_spawn_choice,
    evaluate_spawn,
    risk_multiplier,
)


class FreeGoodnessRuleTests(unittest.TestCase):
    def test_death_risk_uses_multiplicative_change(self) -> None:
        self.assertAlmostEqual(risk_multiplier(0.99, 0.98), 2.0)
        self.assertAlmostEqual(risk_multiplier(0.10, 0.0), 1.0 / 0.9)
        self.assertTrue(math.isinf(risk_multiplier(1.0, 0.99)))

    def test_zero_selected_success_is_always_corrected(self) -> None:
        decision = decide_move(
            selected_direction="right",
            results={"left": 0.5, "right": 0.0, "up": 0.4, "down": 0.3},
            legal_directions={"left", "right", "up", "down"},
        )
        self.assertIsNotNone(decision)
        self.assertTrue(decision.corrected)
        self.assertEqual(decision.executed_direction, "left")
        self.assertEqual(decision.goodness, 0.0)

    def test_spawn_to_zero_can_be_accepted_from_low_success(self) -> None:
        accepted, multiplier, state = evaluate_spawn(
            executed_success=0.10,
            next_success=0.0,
            risk_state=SpawnRiskState(),
        )
        self.assertTrue(accepted)
        self.assertAlmostEqual(multiplier, 1.0 / 0.9)
        self.assertLessEqual(state.drawdown, 1.20)

    def test_cumulative_spawn_drawdown_is_multiplicative(self) -> None:
        state = SpawnRiskState()
        accepted, _multiplier, state = evaluate_spawn(
            executed_success=0.50,
            next_success=0.45,
            risk_state=state,
        )
        self.assertTrue(accepted)
        accepted, _multiplier, state = evaluate_spawn(
            executed_success=0.50,
            next_success=0.44,
            risk_state=state,
        )
        self.assertFalse(accepted)
        self.assertGreater(state.drawdown, 1.20)

    def test_deterministic_spawn_is_stable(self) -> None:
        seed = "00112233445566778899aabbccddeeff" * 2
        first = deterministic_spawn_choice(
            seed, 17, 3, [1, 4, 7, 12], spawn_rate=0.1
        )
        second = deterministic_spawn_choice(
            seed, 17, 3, [1, 4, 7, 12], spawn_rate=0.1
        )
        self.assertEqual(first, second)
        self.assertIn(first[0], {1, 4, 7, 12})
        self.assertIn(first[1], {2, 4})


if __name__ == "__main__":
    unittest.main()
