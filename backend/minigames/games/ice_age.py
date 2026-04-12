from __future__ import annotations

from typing import Any

import numpy as np

from ..engine.base import BaseMinigameEngine


class IceAgeEngine(BaseMinigameEngine):
    def __init__(self, definition, difficulty: int) -> None:
        self.frozen_step = 80 + int(difficulty) * 20
        self.count_down = np.zeros((4, 4), dtype=np.int32)
        self.movement_track = np.zeros((4, 4), dtype=bool)
        super().__init__(definition, difficulty)

    def load_legacy_extra(self, extra_state: list[Any]) -> None:
        self.count_down = np.array(extra_state[0], dtype=np.int32) if extra_state else np.zeros((self.rows, self.cols), dtype=np.int32)

    def export_legacy_extra(self) -> list[Any]:
        return [np.array(self.count_down, dtype=np.int32)]

    def setup_new_game(self) -> None:
        self.count_down = np.zeros((self.rows, self.cols), dtype=np.int32)
        super().setup_new_game()
        self.save_to_config()

    def before_move(self, direct: int) -> None:
        self.movement_track = self._move_and_track(direct)

    def before_gen_num(self, _direct: int) -> None:
        reveal_effects: list[dict[str, Any]] = []
        for row in range(self.rows):
            for col in range(self.cols):
                value = int(self.board[row, col])
                if value == -1:
                    continue
                if not bool(self.movement_track[row, col]):
                    self.count_down[row, col] = 0
                    continue
                previous_count_down = int(self.count_down[row, col])
                self.count_down[row, col] += 1
                if self.count_down[row, col] >= self.frozen_step:
                    self.count_down[row, col] = value
                    self.board[row, col] = -1
                    reveal_effects.append(self._build_stage_reveal(row, col, "icetrap.png"))
                    continue
                reveal_sprite = self._sprite_for_threshold(previous_count_down, int(self.count_down[row, col]))
                if reveal_sprite:
                    reveal_effects.append(self._build_stage_reveal(row, col, reveal_sprite))
        if reveal_effects:
            self.queue_move_effects(reveal_effects)

    def _build_stage_reveal(self, row: int, col: int, sprite: str) -> dict[str, Any]:
        return {
            "type": "ice_stage_reveal",
            "index": int(row * self.cols + col),
            "sprite": sprite,
            "durationMs": 280,
            "animDurationMs": 220,
        }

    def _sprite_for_threshold(self, previous_count: int, current_count: int) -> str | None:
        thresholds = [
            (20, "crystal1.png"),
            (36 + self.difficulty * 4, "crystal3.png"),
            (50 + self.difficulty * 10, "crystal2.png"),
            (64 + self.difficulty * 16, "ice_overlay.png"),
            (self.frozen_step - 5, "icetrap0.png"),
            (self.frozen_step, "icetrap.png"),
        ]
        for threshold, sprite in thresholds:
            if previous_count < threshold <= current_count:
                return sprite
        return None

    @staticmethod
    def _track_movement(line: np.ndarray, reverse: bool = False) -> np.ndarray:
        current_line = np.array(line, copy=True)
        if reverse:
            current_line = current_line[::-1]

        result = np.zeros_like(current_line, dtype=bool)
        segments = []
        current_segment = []
        for value in current_line:
            if int(value) == -1:
                if current_segment:
                    segments.append(current_segment)
                segments.append([-1])
                current_segment = []
            else:
                current_segment.append(int(value))
        if current_segment:
            segments.append(current_segment)

        result_idx = 0
        for segment in segments:
            if segment == [-1]:
                result[result_idx] = False
                result_idx += 1
                continue
            moved = False
            for idx in range(len(segment)):
                if moved:
                    result[result_idx + idx] = False
                    continue
                if segment[idx] == 0:
                    moved = True
                step = 1
                while idx + step < len(segment):
                    if segment[idx] == segment[idx + step]:
                        moved = True
                        break
                    if segment[idx + step] != 0:
                        break
                    step += 1
                result[result_idx + idx] = not moved
            result_idx += len(segment)

        return result[::-1] if reverse else result

    def _move_and_track(self, direction: int) -> np.ndarray:
        movement_occurred = np.zeros_like(self.board, dtype=bool)
        if direction == 1:
            for row in range(self.rows):
                movement_occurred[row, :] = self._track_movement(self.board[row, :])
        elif direction == 2:
            for row in range(self.rows):
                movement_occurred[row, :] = self._track_movement(self.board[row, :], reverse=True)
        elif direction == 3:
            for col in range(self.cols):
                movement_occurred[:, col] = self._track_movement(self.board[:, col])
        elif direction == 4:
            for col in range(self.cols):
                movement_occurred[:, col] = self._track_movement(self.board[:, col], reverse=True)
        return movement_occurred

    def build_view_state(self) -> dict[str, Any]:
        view = super().build_view_state()
        tile_text_override: dict[str, str] = {}
        tile_style_variant: dict[str, Any] = {}
        cover_sprites: dict[str, list[str]] = {}

        for row in range(self.rows):
            for col in range(self.cols):
                index = row * self.cols + col
                value = int(self.board[row, col])
                count_down = int(self.count_down[row, col])
                if count_down == 0 and value != -1:
                    continue

                sprites: list[str] = []
                if count_down >= 20 or value == -1:
                    sprites.append("crystal1.png")
                if count_down >= 36 + self.difficulty * 4 or value == -1:
                    sprites.append("crystal3.png")
                if count_down >= 50 + self.difficulty * 10 or value == -1:
                    sprites.append("crystal2.png")
                if count_down >= 64 + self.difficulty * 16 or value == -1:
                    sprites.append("ice_overlay.png")
                if count_down >= self.frozen_step - 5 or value == -1:
                    sprites.append("icetrap0.png")
                if count_down >= self.frozen_step or value == -1:
                    sprites.append("icetrap.png")
                if sprites:
                    cover_sprites[str(index)] = sprites

                if value == -1 and count_down > 0:
                    tile_text_override[str(index)] = str(2 ** count_down)
                    tile_style_variant[str(index)] = {"kind": "frozen", "exponent": count_down}

        view["tileTextOverride"] = tile_text_override
        view["tileStyleVariant"] = tile_style_variant
        view["coverSprites"] = cover_sprites
        return view

    def get_info_text(self) -> str:
        return "Tiles freeze in place if they stand still for too long."
