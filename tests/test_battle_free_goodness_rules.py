from __future__ import annotations

import math
import unittest

from backend.battle.modes.free_goodness.rules import (
    SpawnRiskState,
    accumulate_goodness,
    decide_move,
    deterministic_spawn_choice,
    evaluate_spawn,
    risk_multiplier,
)


class FreeGoodnessRuleTests(unittest.TestCase):
    def test_goodness_uses_tester_cumulative_product(self) -> None:
        goodness = accumulate_goodness(1.0, 0.8)
        self.assertAlmostEqual(goodness, 0.8)
        goodness = accumulate_goodness(goodness, 1.0)
        self.assertAlmostEqual(goodness, 0.8)
        goodness = accumulate_goodness(goodness, 0.5)
        self.assertAlmostEqual(goodness, 0.4)

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

    def test_move_correction_requires_relative_and_absolute_risk_increase(self) -> None:
        tiny_absolute_change = decide_move(
            selected_direction="right",
            results={"left": 0.999, "right": 0.9988},
            legal_directions={"left", "right"},
        )
        self.assertIsNotNone(tiny_absolute_change)
        self.assertGreater(tiny_absolute_change.risk_multiplier, 1.15)
        self.assertFalse(tiny_absolute_change.corrected)
        self.assertEqual(tiny_absolute_change.executed_direction, "right")

        material_change = decide_move(
            selected_direction="right",
            results={"left": 0.99, "right": 0.988},
            legal_directions={"left", "right"},
        )
        self.assertIsNotNone(material_change)
        self.assertTrue(material_change.corrected)
        self.assertEqual(material_change.executed_direction, "left")
        self.assertEqual(material_change.correction_reason, "risk_limit")

    def test_spawn_to_zero_can_be_accepted_from_low_success(self) -> None:
        accepted, multiplier, state = evaluate_spawn(
            executed_success=0.10,
            next_success=0.0,
            risk_state=SpawnRiskState(),
        )
        self.assertTrue(accepted)
        self.assertAlmostEqual(multiplier, 1.0 / 0.9)
        self.assertLessEqual(state.drawdown, 1.20)

    def test_spawn_that_eliminates_death_risk_is_accepted(self) -> None:
        previous = SpawnRiskState(
            log_index=-8.637161503024732,
            log_floor=-8.637161503024732,
        )
        accepted, multiplier, state = evaluate_spawn(
            executed_success=0.999993678,
            next_success=1.0,
            risk_state=previous,
        )
        self.assertTrue(accepted)
        self.assertEqual(multiplier, 0.0)
        self.assertEqual(state.log_index, 0.0)
        self.assertEqual(state.log_floor, 0.0)
        self.assertEqual(state.drawdown, 1.0)

        accepted, multiplier, state = evaluate_spawn(
            executed_success=1.0,
            next_success=0.99999,
            risk_state=state,
        )
        self.assertFalse(accepted)
        self.assertTrue(math.isinf(multiplier))
        self.assertGreater(state.drawdown, 1.20)

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
