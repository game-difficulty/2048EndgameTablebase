from __future__ import annotations

from typing import Any

import numpy as np

from ..engine.base import BaseMinigameEngine


PATTERNS: dict[str, np.ndarray] = {
    "Design Master1": np.array(
        [[0, 0, 0, 0], [0, 2, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]], dtype=float
    ),
    "Design Master2": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    ),
    "Design Master3": np.array(
        [[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float
    ),
    "Design Master4": np.array(
        [
            [0, 0, 0, 0],
            [0, 0.0078125, 1, 0.5],
            [0, 0.015625, 2, 0.25],
            [0, 0.03125, 0.0625, 0.125],
        ],
        dtype=float,
    ),
}


class DesignMasterEngine(BaseMinigameEngine):
    def __init__(self, definition, difficulty: int) -> None:
        self.pattern = np.array(PATTERNS[definition.legacy_name], dtype=float)
        super().__init__(definition, difficulty)
        self.current_max_num = max(int(self.max_num), 7)

    def setup_new_game(self) -> None:
        super().setup_new_game()
        self.current_max_num = 7

    def _build_target_board(self) -> np.ndarray:
        scale = max(self.current_max_num, self.max_num) + 1
        return self.pattern * float(2**scale)

    @staticmethod
    def _format_target_value(value: float) -> str:
        if value == 0:
            return ""
        target = int(round(value))
        return str(target) if target < 1000 else f"{target // 1000}k"

    def _formatted_target_lines(self) -> list[str]:
        target = self._build_target_board()
        rows: list[str] = []
        for row in target:
            parts = []
            for item in row:
                if not item:
                    parts.append("  _")
                    continue
                target_val = int(round(float(item)))
                text = str(target_val) if target_val < 1000 else f"{target_val // 1000}k"
                parts.append(text.rjust(3, " "))
            rows.append(" ".join(parts))
        return rows

    def build_view_state(self) -> dict[str, Any]:
        view = super().build_view_state()
        labels: list[str] = []
        for row in self._build_target_board():
            for item in row:
                labels.append(self._format_target_value(float(item)))
        view["smallLabels"] = labels
        return view

    def build_hud(self) -> dict[str, Any]:
        hud = super().build_hud()
        hud["customPanels"] = [
            {"type": "patternText", "title": "Target Pattern", "lines": self._formatted_target_lines()}
        ]
        return hud

    def get_info_text(self) -> str:
        return "Fit a particular pattern.\n" + "\n".join(self._formatted_target_lines())

    def check_pattern(self) -> int | bool:
        mask = self.pattern != 0
        masked_original = np.where(mask, np.power(2.0, np.maximum(self.board, 0)), 0.0)
        for num in range(8, 15):
            if np.allclose(masked_original, self.pattern * float(2**num)):
                return num
        return False

    def check_game_passed(self) -> None:
        self._refresh_highest_tile_exp()
        baseline_level = max(int(self.max_num), 7)
        pattern_level = self.check_pattern()
        if not pattern_level or pattern_level <= baseline_level:
            self.current_max_num = baseline_level
            return
        self.current_max_num = int(pattern_level)
        if self.current_max_num > self.max_num:
            self.max_num = self.current_max_num
            self.is_passed = {10: 3, 9: 2, 8: 1}.get(self.max_num, 4)
            level = {10: "gold", 9: "silver", 8: "bronze"}.get(self.max_num, "gold")
            self.queue_message(
                "trophy",
                {"level": level, "message": f"You achieved {2 ** self.max_num}! You get a {level} trophy!"},
            )
        else:
            level = {10: "gold", 9: "silver", 8: "bronze"}.get(self.current_max_num, "gold")
            self.queue_message(
                "trophy",
                {
                    "level": level,
                    "message": f"You achieved {2 ** self.current_max_num}! Take it further!",
                },
            )
