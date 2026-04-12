from __future__ import annotations

from typing import Any

from ..engine.base import BaseMinigameEngine


class FerrisWheelEngine(BaseMinigameEngine):
    OUTER_RING = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 3), (2, 3),
        (3, 3), (3, 2), (3, 1), (3, 0),
        (2, 0), (1, 0),
    ]

    def __init__(self, definition, difficulty: int) -> None:
        self.count_down = 40 - int(difficulty) * 10
        super().__init__(definition, difficulty)

    def load_legacy_extra(self, extra_state: list[Any]) -> None:
        self.count_down = int(extra_state[0]) if extra_state else 40 - int(self.difficulty) * 10

    def export_legacy_extra(self) -> list[Any]:
        return [int(self.count_down)]

    def setup_new_game(self) -> None:
        super().setup_new_game()
        self.count_down = 40 - int(self.difficulty) * 10
        self.save_to_config()

    def after_gen_num(self) -> None:
        self.count_down -= 1
        if self.count_down > 0:
            return
        self.count_down = 40 - int(self.difficulty) * 10
        values = [int(self.board[row, col]) for row, col in self.OUTER_RING]
        rotated_values = values[-1:] + values[:-1]
        swap_effects: list[dict[str, int]] = []
        for (from_row, from_col), (to_row, to_col), value in zip(
            self.OUTER_RING,
            self.OUTER_RING[1:] + self.OUTER_RING[:1],
            values,
        ):
            if value > 0:
                swap_effects.append(
                    {
                        "type": "ring_rotate_move",
                        "fromIndex": int(from_row * self.cols + from_col),
                        "toIndex": int(to_row * self.cols + to_col),
                        "value": value,
                        "durationMs": 1000,
                    }
                )
        for (row, col), value in zip(self.OUTER_RING, rotated_values):
            self.board[row, col] = value
        if swap_effects:
            self.set_follow_up_animation(
                {
                    "kind": "effects",
                    "delayMs": 270,
                    "durationMs": 1000,
                    "effects": swap_effects,
                }
            )

    def build_hud(self) -> dict[str, Any]:
        hud = super().build_hud()
        hud["customPanels"] = [
            {
                "type": "remainingSteps",
                "title": "Next Rotation",
                "value": int(self.count_down),
                "suffix": "steps",
            }
        ]
        return hud

    def get_info_text(self) -> str:
        return "The Earth revolves around the Sun."
