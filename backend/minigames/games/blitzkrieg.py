from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..engine.base import BaseMinigameEngine


class BlitzkriegEngine(BaseMinigameEngine):
    def __init__(self, definition, difficulty: int) -> None:
        self.remaining_ms = 180000
        self.timer_running = False
        self.timer_anchor_ms: int | None = None
        self.count_1k = 0
        super().__init__(definition, difficulty)
        self.count_1k = int(np.sum(self.board == 10))

    def load_legacy_extra(self, extra_state: list[Any]) -> None:
        if extra_state:
            self.remaining_ms = max(0, int(float(extra_state[0]) * 60 * 1000))

    def export_legacy_extra(self) -> list[Any]:
        self._sync_timer()
        return [self.remaining_ms / (60 * 1000)]

    def setup_new_game(self) -> None:
        super().setup_new_game()
        self.remaining_ms = 180000
        self.timer_running = False
        self.timer_anchor_ms = None
        self.count_1k = int(np.sum(self.board == 10))
        self.is_over = False
        self.save_to_config()

    def _sync_timer(self) -> None:
        if not self.timer_running or self.timer_anchor_ms is None or self.is_over:
            return
        now_ms = int(time.time() * 1000)
        elapsed = max(0, now_ms - self.timer_anchor_ms)
        self.remaining_ms = max(0, self.remaining_ms - elapsed)
        self.timer_anchor_ms = now_ms
        if self.remaining_ms == 0:
            self.is_over = True
            self.timer_running = False
            self.check_game_passed()

    def do_move(self, direction: str) -> None:
        self._sync_timer()
        if self.is_over:
            return
        if not self.timer_running:
            self.timer_running = True
            self.timer_anchor_ms = int(time.time() * 1000)
        previous_count = int(np.sum(self.board == 10))
        super().do_move(direction)
        self._sync_timer()
        if not self._last_valid_move:
            return
        current_count = int(np.sum(self.board == 10))
        if current_count > previous_count:
            bonus_minutes = 0.75 if int(self.difficulty) == 1 else 1.0
            gained_ms = int((current_count - previous_count) * bonus_minutes * 60 * 1000)
            self.remaining_ms += gained_ms
            if self.timer_running:
                self.timer_anchor_ms = int(time.time() * 1000)
        self.count_1k = current_count
        if self.is_over:
            self.timer_running = False
        self.save_to_config()

    def check_game_passed(self) -> None:
        self._refresh_highest_tile_exp()
        self.current_max_num = max(self.current_max_num, self._positive_max(self.board))
        self.max_num = max(self.max_num, self.current_max_num)
        if not self.is_over:
            return
        if self.max_num > 9:
            self.is_passed = {12: 3, 11: 2, 10: 1}.get(self.max_num, 4)
        if self.current_max_num <= 9:
            return
        level = {12: "gold", 11: "silver", 10: "bronze"}.get(self.current_max_num, "gold")
        if self.score == self.max_score:
            message = f"You achieved {self.score} score! You get a {level} trophy!"
        else:
            message = f"You achieved {self.score} score! Nice game!"
        self.queue_message("trophy", {"level": level, "message": message})

    def check_game_over(self) -> None:
        self._sync_timer()
        if not self.is_over and self.remaining_ms > 0 and self.has_possible_move():
            return
        self.is_over = True
        self.timer_running = False
        self.check_game_passed()

    def build_hud(self) -> dict[str, Any]:
        self._sync_timer()
        hud = super().build_hud()
        hud["customPanels"] = [
            {
                "type": "countdown",
                "title": "Countdown",
                "remainingMs": int(self.remaining_ms),
                "running": bool(self.timer_running and not self.is_over),
                "syncedAt": int(time.time() * 1000),
            }
        ]
        return hud

    def get_info_text(self) -> str:
        return "Act quickly to earn bonus time and rack up the highest score!"
