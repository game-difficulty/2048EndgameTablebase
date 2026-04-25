from __future__ import annotations

from typing import Any

import numpy as np

from engine_core.Calculator import find_merge_positions

from ..engine.base import BaseMinigameEngine


class MysteryMergeEngine(BaseMinigameEngine):
    def __init__(self, definition, difficulty: int) -> None:
        self.variant = 2 if definition.legacy_name.endswith("2") else 1
        self.peek_count = 0
        self.peek_active = False
        self.reveal_all = False
        self.masked_ = np.zeros((4, 4), dtype=bool)
        super().__init__(definition, difficulty)

    def load_legacy_extra(self, extra_state: list[Any]) -> None:
        self.peek_count = int(extra_state[0]) if extra_state else 0
        if len(extra_state) > 1:
            self.masked_ = np.array(extra_state[1], dtype=bool)
        else:
            self.masked_ = np.zeros((self.rows, self.cols), dtype=bool)
        self.reveal_all = False
        self.peek_active = False

    def export_legacy_extra(self) -> list[Any]:
        return [int(self.peek_count), np.array(self.masked_, dtype=bool)]

    def setup_new_game(self) -> None:
        self.masked_ = np.zeros((self.rows, self.cols), dtype=bool)
        self.peek_count = 0
        self.peek_active = False
        self.reveal_all = False
        super().setup_new_game()
        if self.variant == 2 and self.newtile_pos >= 0:
            self.after_gen_num()
        self.save_to_config()

    def _can_peek(self) -> bool:
        if self.variant not in (1, 2):
            return False
        if int(self.difficulty) == 0:
            return True
        return self.peek_count <= self.score // 10000

    def _remaining_peek_count(self) -> str:
        if int(self.difficulty) == 0:
            return "∞"
        return str(max(0, self.score // 10000 - self.peek_count + 1))

    def handle_custom_action(self, key: str, phase: str = "trigger") -> bool:
        normalized_key = str(key or "").strip().lower()
        normalized_phase = str(phase or "trigger").strip().lower()
        if normalized_key != "peek":
            return False
        if normalized_phase == "start":
            if self.has_possible_move() and self._can_peek():
                self.peek_active = True
                self.peek_count += 1
                self.save_to_config()
                return True
            return False
        if normalized_phase in {"end", "cancel"}:
            if self.peek_active:
                self.peek_active = False
                return True
            return False
        if self.has_possible_move() and self._can_peek():
            self.peek_active = not self.peek_active
            if self.peek_active:
                self.peek_count += 1
            self.save_to_config()
            return True
        return False

    def before_move(self, direct: int) -> None:
        if self.variant == 1:
            direction = ("left", "right", "up", "down")[direct - 1]
            self.masked_ = np.array(find_merge_positions(self.board, direction), dtype=bool)
        else:
            if direct == 1:
                for row in range(self.rows):
                    self.masked_[row, :] = self._update_mask_line(self.board[row, :], self.masked_[row, :])
            elif direct == 2:
                for row in range(self.rows):
                    self.masked_[row, :] = self._update_mask_line(
                        self.board[row, :], self.masked_[row, :], reverse=True
                    )
            elif direct == 3:
                for col in range(self.cols):
                    self.masked_[:, col] = self._update_mask_line(self.board[:, col], self.masked_[:, col])
            elif direct == 4:
                for col in range(self.cols):
                    self.masked_[:, col] = self._update_mask_line(
                        self.board[:, col], self.masked_[:, col], reverse=True
                    )
        self.peek_active = False
        self.reveal_all = False

    def after_gen_num(self) -> None:
        if self.variant == 2 and self.newtile_pos >= 0:
            row, col = divmod(self.newtile_pos, self.cols)
            self.masked_[row, col] = True

    @staticmethod
    def _update_mask_line(line, mask, reverse: bool = False) -> np.ndarray:
        current_line = np.array(line, copy=True)
        current_mask = np.array(mask, copy=True)
        if reverse:
            current_line = current_line[::-1]
            current_mask = current_mask[::-1]

        non_zero = [int(value) for value in current_line if value != 0]
        non_zero_mask = [bool(flag) for value, flag in zip(current_line, current_mask) if value != 0]
        merged_mask: list[int] = []
        skip = False

        for idx in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if idx + 1 < len(non_zero) and non_zero[idx] == non_zero[idx + 1]:
                merged_mask.append(0)
                skip = True
            else:
                merged_mask.append(1 if non_zero_mask[idx] else 0)

        merged_mask += [0] * (len(current_mask) - len(merged_mask))
        result = np.array(merged_mask, dtype=bool)
        return result[::-1] if reverse else result

    def check_game_over(self) -> None:
        if self.has_possible_move():
            return
        self.reveal_all = True
        self.peek_active = False
        self.is_over = True

    def build_view_state(self) -> dict[str, Any]:
        view = super().build_view_state()
        hidden_mask = np.zeros((self.rows, self.cols), dtype=bool)
        if not self.peek_active and not self.reveal_all:
            for row in range(self.rows):
                for col in range(self.cols):
                    index = row * self.cols + col
                    value = int(self.board[row, col])
                    if value <= 0:
                        continue
                    if self.variant == 1:
                        hidden_mask[row, col] = index != self.newtile_pos and not bool(self.masked_[row, col])
                    else:
                        hidden_mask[row, col] = bool(self.masked_[row, col]) or index == self.newtile_pos
        view["hiddenMask"] = hidden_mask.flatten().tolist()
        return view

    def build_hud(self) -> dict[str, Any]:
        hud = super().build_hud()
        hud["customPanels"] = [
            {
                "type": "actionButton",
                "title": "Peek",
                "key": "peek",
                "label": "Peek",
                "hold": True,
                "pressed": bool(self.peek_active),
                "enabled": bool(self._can_peek() and self.has_possible_move()),
                "meta": self._remaining_peek_count(),
            }
        ]
        return hud

    def get_info_text(self) -> str:
        if self.variant == 1:
            return "Show only empty spaces and newly generated tiles."
        return "Not sure what newly generated tiles are unless a merge occurs."
