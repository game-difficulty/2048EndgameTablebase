from __future__ import annotations

import numpy as np

from ..animation import build_minigame_move_animation_metadata
from ..engine.base import BaseMinigameEngine


class GravityTwistEngine(BaseMinigameEngine):
    def __init__(self, definition, difficulty: int) -> None:
        self.variant = 2 if definition.legacy_name.endswith("2") else 1
        super().__init__(definition, difficulty)

    def _compress_down(self) -> np.ndarray:
        board = np.array(self.board, copy=True)
        for col in range(self.cols):
            current_col = board[:, col]
            non_zero = current_col[current_col != 0]
            zero_elements = np.zeros(self.rows - len(non_zero), dtype=np.int32)
            board[:, col] = np.concatenate((zero_elements, non_zero))
        return board

    def _build_drop_only_follow_up(self, board_before: np.ndarray, board_after: np.ndarray) -> dict[str, object]:
        slide_distances = [0] * (self.rows * self.cols)
        pop_positions = [0] * (self.rows * self.cols)
        for col in range(self.cols):
            source_rows = [row for row in range(self.rows) if int(board_before[row, col]) > 0]
            target_rows = [row for row in range(self.rows) if int(board_after[row, col]) > 0]
            for source_row, target_row in zip(reversed(source_rows), reversed(target_rows)):
                slide_distances[source_row * self.cols + col] = max(0, target_row - source_row)
        return {
            "kind": "move",
            "direction": "down",
            "slide_distances": slide_distances,
            "pop_positions": pop_positions,
            "delayMs": 360,
            "durationMs": 180,
            "lockInput": False,
        }

    def after_gen_num(self) -> None:
        board_before_gravity = np.array(self.board, copy=True)
        if self.variant == 1:
            self.board = self._compress_down()
        else:
            board_new, new_score = self.mover.move_board(np.array(self.board, copy=True), 4)
            if np.any(board_new != self.board):
                self.board = np.array(board_new, dtype=np.int32)
                self.score += int(new_score)
                self.max_score = max(self.max_score, self.score)
        if np.any(self.board != board_before_gravity):
            if self.variant == 1:
                self.set_follow_up_animation(self._build_drop_only_follow_up(board_before_gravity, self.board))
            else:
                follow_up = build_minigame_move_animation_metadata(board_before_gravity, "down")
                self.set_follow_up_animation(
                    {
                        "kind": "move",
                        "direction": "down",
                        "slide_distances": follow_up.get("slide_distances", []),
                        "pop_positions": follow_up.get("pop_positions", []),
                        "delayMs": 360,
                        "durationMs": 180,
                        "lockInput": False,
                    }
                )

    def get_info_text(self) -> str:
        return "The tiles are affected by the unusual gravity."
