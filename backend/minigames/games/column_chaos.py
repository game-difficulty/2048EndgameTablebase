from __future__ import annotations

import random
from typing import Any

from ..engine.base import BaseMinigameEngine


class ColumnChaosEngine(BaseMinigameEngine):
    def __init__(self, definition, difficulty: int) -> None:
        self.count_down = 40 - 10 * int(difficulty)
        super().__init__(definition, difficulty)

    def load_legacy_extra(self, extra_state: list[Any]) -> None:
        self.count_down = int(extra_state[0]) if extra_state else 40 - 10 * int(self.difficulty)

    def export_legacy_extra(self) -> list[Any]:
        return [int(self.count_down)]

    def setup_new_game(self) -> None:
        super().setup_new_game()
        self.count_down = 40 - 10 * int(self.difficulty)
        self.save_to_config()

    def after_gen_num(self) -> None:
        self.count_down -= 1
        if self.count_down > 0:
            return
        self.count_down = 40 - 10 * int(self.difficulty)
        col1, col2 = random.sample(range(self.cols), 2)
        swap_effects: list[dict[str, int]] = []
        for row in range(self.rows):
            left_value = int(self.board[row, col1])
            right_value = int(self.board[row, col2])
            if left_value > 0:
                swap_effects.append(
                    {
                        "type": "column_swap_move",
                        "fromIndex": int(row * self.cols + col1),
                        "toIndex": int(row * self.cols + col2),
                        "value": left_value,
                        "durationMs": 1250,
                    }
                )
            if right_value > 0:
                swap_effects.append(
                    {
                        "type": "column_swap_move",
                        "fromIndex": int(row * self.cols + col2),
                        "toIndex": int(row * self.cols + col1),
                        "value": right_value,
                        "durationMs": 1250,
                    }
                )
        self.board[:, [col1, col2]] = self.board[:, [col2, col1]]
        if swap_effects:
            self.set_follow_up_animation(
                {
                    "kind": "effects",
                    "delayMs": 270,
                    "durationMs": 1250,
                    "effects": swap_effects,
                }
            )

    def build_hud(self) -> dict[str, Any]:
        hud = super().build_hud()
        hud["customPanels"] = [
            {
                "type": "remainingSteps",
                "title": "Next Chaos",
                "value": int(self.count_down),
                "suffix": "steps",
            }
        ]
        return hud

    def get_info_text(self) -> str:
        return "Unpredictable shifts in columns are coming!"
