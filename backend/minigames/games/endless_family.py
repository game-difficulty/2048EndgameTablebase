from __future__ import annotations

import random
from typing import Any

import numpy as np

from egtb_core.Calculator import count_zeros
from Config import SingletonConfig

from ..engine.base import BaseMinigameEngine


class EndlessFamilyEngine(BaseMinigameEngine):
    def __init__(self, definition, difficulty: int) -> None:
        self.variant = self._detect_variant(definition.legacy_name)
        self.bomb_pos: tuple[int, int] | None = None
        self.current_level = 0
        self.bomb_type = 0
        self.target_pos: tuple[int, int] | None = None
        self.count_down = np.zeros((4, 4), dtype=np.int32)
        self.has_just_exploded: int | bool = False
        self.pending_resolution_kind: str | None = None
        self.pending_resolution_positions: list[tuple[int, int]] = []
        self.levels = self._levels_for_variant()
        self.bomb_gen_rate = 0.05 if self.variant == "hybrid" else 0.03
        super().__init__(definition, difficulty)

    @staticmethod
    def _detect_variant(legacy_name: str) -> str:
        mapping = {
            "Endless Explosions": "explosions",
            "Endless Giftbox": "giftbox",
            "Endless Factorization": "factorization",
            "Endless Hybrid": "hybrid",
            "Endless AirRaid": "airraid",
        }
        return mapping.get(legacy_name, "explosions")

    def _levels_for_variant(self) -> list[tuple[int, int, str | None]]:
        if self.variant == "hybrid":
            return [(150000, 4, None), (100000, 3, "gold"), (50000, 2, "silver"), (20000, 1, "bronze")]
        return [(300000, 4, None), (200000, 3, "gold"), (100000, 2, "silver"), (40000, 1, "bronze")]

    def load_legacy_extra(self, extra_state: list[Any]) -> None:
        if self.variant == "airraid":
            self.count_down = np.array(extra_state[0], dtype=np.int32) if extra_state else np.zeros((self.rows, self.cols), dtype=np.int32)
            self.target_pos = tuple(extra_state[1]) if len(extra_state) > 1 and extra_state[1] is not None else None
            self.current_level = int(extra_state[2]) if len(extra_state) > 2 else 0
            return
        self.bomb_pos = tuple(extra_state[0]) if extra_state and extra_state[0] is not None else None
        self.current_level = int(extra_state[1]) if len(extra_state) > 1 else 0
        if self.variant == "hybrid":
            self.bomb_type = int(extra_state[2]) if len(extra_state) > 2 else 0

    def export_legacy_extra(self) -> list[Any]:
        if self.variant == "airraid":
            return [np.array(self.count_down, dtype=np.int32), self.target_pos, int(self.current_level)]
        data: list[Any] = [self.bomb_pos, int(self.current_level)]
        if self.variant == "hybrid":
            data.append(int(self.bomb_type))
        return data

    def setup_new_game(self) -> None:
        self.has_just_exploded = False
        self.pending_resolution_kind = None
        self.pending_resolution_positions = []
        self.current_level = 0
        self.target_pos = None
        self.count_down = np.zeros((self.rows, self.cols), dtype=np.int32)
        if self.variant == "hybrid":
            self.bomb_type = random.randint(0, 2)
        super().setup_new_game()
        if self.variant == "airraid":
            self.target_pos = None
            self.count_down = np.zeros((self.rows, self.cols), dtype=np.int32)
        else:
            zero_positions = np.argwhere(self.board == 0)
            self.bomb_pos = tuple(random.choice(zero_positions.tolist())) if zero_positions.size else None
        self.save_to_config()

    def _queue_score_trophy(self) -> None:
        level_name = None
        for score_threshold, level_number, trophy_name in self.levels:
            if self.score >= score_threshold and self.current_level < level_number:
                self.is_passed = max(self.is_passed, level_number)
                self.current_level = level_number
                level_name = trophy_name
                break
        if not level_name:
            return
        score_text = f"{self.max_score // 1000}k"
        message = (
            f"You achieved {score_text} score! You get a {level_name} trophy!"
            if self.max_score == self.score
            else f"You achieved {score_text} score! Let's go!"
        )
        self.queue_message("trophy", {"level": level_name, "message": message})

    def check_game_passed(self) -> None:
        self._refresh_highest_tile_exp()
        self.current_max_num = max(self.current_max_num, self._positive_max(self.board))
        self.max_num = max(self.max_num, self.current_max_num)
        self._queue_score_trophy()

    def has_possible_move(self) -> bool:
        if self.variant != "airraid" and self.bomb_pos is not None:
            return True
        if np.sum(self.board == 0) > 0:
            return True
        for direction in (1, 2, 3, 4):
            board_new, _, is_valid = self.move_and_check_validity(direction)
            if is_valid and np.any(board_new != self.board):
                return True
        return False

    def _empty_positions(self) -> list[tuple[int, int]]:
        return [tuple(pos) for pos in np.argwhere(self.board == 0).tolist()]

    def _place_random_bomb(self) -> bool:
        empty_positions = self._empty_positions()
        if not empty_positions:
            return False
        self.bomb_pos = random.choice(empty_positions)
        if self.variant == "hybrid":
            self.bomb_type = random.randint(0, 2)
        return True

    def _bomb_slide_distance(self, direct: int) -> tuple[int, int] | None:
        if self.bomb_pos is None:
            return None
        row, col = self.bomb_pos
        if direct == 1:
            return row, 0
        if direct == 2:
            return row, self.cols - 1
        if direct == 3:
            return 0, col
        if direct == 4:
            return self.rows - 1, col
        return None

    def _first_hit_in_direction(self, direct: int) -> tuple[int, int] | None:
        if self.bomb_pos is None:
            return None
        row, col = self.bomb_pos
        if direct == 1:
            search = [(row, idx) for idx in range(col - 1, -1, -1)] + [(row, idx) for idx in range(col + 1, self.cols)]
        elif direct == 2:
            search = [(row, idx) for idx in range(col + 1, self.cols)] + [(row, idx) for idx in range(col - 1, -1, -1)]
        elif direct == 3:
            search = [(idx, col) for idx in range(row - 1, -1, -1)] + [(idx, col) for idx in range(row + 1, self.rows)]
        else:
            search = [(idx, col) for idx in range(row + 1, self.rows)] + [(idx, col) for idx in range(row - 1, -1, -1)]
        for candidate_row, candidate_col in search:
            if int(self.board[candidate_row, candidate_col]) != 0:
                return candidate_row, candidate_col
        return None

    def _explode_target(self, row: int, col: int) -> None:
        value = int(self.board[row, col])
        if self.variant == "explosions":
            self.board[row, col] = 0
            self.queue_move_effects(
                [
                    {
                        "type": "explosion",
                        "index": int(row * self.cols + col),
                        "delayMs": 100,
                        "durationMs": 500,
                        "animDurationMs": 500,
                    }
                ]
            )
            self.has_just_exploded = False
            return
        if self.variant == "giftbox":
            self.has_just_exploded = value
            self.board[row, col] = -2
            self.pending_resolution_kind = "giftbox_burst"
            self.pending_resolution_positions = [(row, col)]
            return
        if self.variant == "factorization":
            self.has_just_exploded = value
            self.board[row, col] = -2
            self.pending_resolution_kind = "factorization_burst"
            self.pending_resolution_positions = [(row, col)]
            if value > 1 and self.bomb_pos is not None:
                self.board[self.bomb_pos] = -2
                self.pending_resolution_positions.append(tuple(self.bomb_pos))
            elif self.bomb_pos is not None:
                self.board[self.bomb_pos] = 0
            return
        if self.variant == "hybrid":
            if self.bomb_type == 0:
                self.board[row, col] = 0
                self.queue_move_effects(
                    [
                        {
                            "type": "explosion",
                            "index": int(row * self.cols + col),
                            "delayMs": 100,
                            "durationMs": 500,
                            "animDurationMs": 500,
                        }
                    ]
                )
                self.has_just_exploded = False
                return
            if self.bomb_type == 1:
                self.has_just_exploded = value
                self.board[row, col] = -2
                self.pending_resolution_kind = "factorization_burst"
                self.pending_resolution_positions = [(row, col)]
                if value > 1 and self.bomb_pos is not None:
                    self.board[self.bomb_pos] = -2
                    self.pending_resolution_positions.append(tuple(self.bomb_pos))
                elif self.bomb_pos is not None:
                    self.board[self.bomb_pos] = 0
                return
            self.has_just_exploded = value
            self.board[row, col] = -2
            self.pending_resolution_kind = "giftbox_burst"
            self.pending_resolution_positions = [(row, col)]

    def _current_object_visual(self) -> tuple[str, str]:
        if self.variant == "explosions":
            return "bomb.png", ""
        if self.variant == "giftbox":
            return "giftbox.png", ""
        if self.variant == "factorization":
            return "tilebg.png", ""
        if self.variant == "hybrid":
            if self.bomb_type == 0:
                return "bomb.png", ""
            if self.bomb_type == 1:
                return "tilebg.png", ""
            return "giftbox.png", ""
        return "bomb.png", ""

    def _object_slide_target(self, direct: int) -> tuple[int, int]:
        if self.bomb_pos is None:
            return -1, -1
        row, col = self.bomb_pos
        if direct == 1:
            return row, col - int(count_zeros(self.board[row, :col][::-1]))
        if direct == 2:
            return row, col + int(count_zeros(self.board[row, col + 1 :]))
        if direct == 3:
            return row - int(count_zeros(self.board[:row, col][::-1])), col
        if direct == 4:
            return row + int(count_zeros(self.board[row + 1 :, col])), col
        return row, col

    def _current_object_visual(self) -> tuple[str, str]:
        if self.variant == "explosions":
            return "bomb.png", ""
        if self.variant == "giftbox":
            return "giftbox.png", ""
        if self.variant == "factorization":
            return "tilebg.png", ""
        if self.variant == "hybrid":
            if self.bomb_type == 0:
                return "bomb.png", ""
            if self.bomb_type == 1:
                return "tilebg.png", ""
            return "giftbox.png", ""
        return "bomb.png", ""

    def _queue_object_slide(
        self,
        from_pos: tuple[int, int],
        to_pos: tuple[int, int],
        *,
        hide_target: bool = False,
        fade_out_at_end: bool = False,
        duration_ms: int = 100,
    ) -> None:
        if from_pos == to_pos and not hide_target:
            return
        sprite, label_text = self._current_object_visual()
        effect = {
            "type": "object_slide",
            "fromIndex": int(from_pos[0] * self.cols + from_pos[1]),
            "toIndex": int(to_pos[0] * self.cols + to_pos[1]),
            "durationMs": duration_ms,
            "animDurationMs": duration_ms,
            "sprite": sprite,
            "labelText": label_text,
            "fadeOutAtEnd": bool(fade_out_at_end),
        }
        if hide_target:
            effect["hideIndices"] = [int(to_pos[0] * self.cols + to_pos[1])]
        self.queue_move_effects([effect])

    def _resolve_post_explosion(self) -> None:
        if not self.has_just_exploded:
            return
        positions = [tuple(pos) for pos in np.argwhere(self.board == -2).tolist()]
        original_value = int(self.has_just_exploded)
        if self.variant in {"giftbox"} or (self.variant == "hybrid" and self.bomb_type == 2):
            if not positions:
                self.has_just_exploded = False
                self.pending_resolution_kind = None
                self.pending_resolution_positions = []
                return
            row, col = positions[0]
            exponents = np.arange(2, 11)
            weights = 1 / (exponents ** 1.5)
            weights /= weights.sum()
            new_value = original_value
            while new_value == original_value:
                new_value = int(np.random.choice(exponents, p=weights))
            self.board[row, col] = new_value
            self.queue_move_effects(
                [
                    {
                        "type": "giftbox_burst",
                        "index": int(row * self.cols + col),
                        "delayMs": 100,
                        "durationMs": 620,
                        "animDurationMs": 500,
                        "hideIndices": [int(row * self.cols + col)],
                    }
                ]
            )
            self.has_just_exploded = False
            self.pending_resolution_kind = None
            self.pending_resolution_positions = []
            return
        if self.variant in {"factorization"} or (self.variant == "hybrid" and self.bomb_type == 1):
            if original_value <= 1:
                if positions:
                    row, col = positions[0]
                    self.board[row, col] = 0
                    self.queue_move_effects(
                        [
                            {
                                "type": "factorization_burst",
                                "index": int(row * self.cols + col),
                                "delayMs": 100,
                                "durationMs": 620,
                                "animDurationMs": 500,
                                "hideIndices": [int(row * self.cols + col)],
                            }
                        ]
                    )
                self.has_just_exploded = False
                self.pending_resolution_kind = None
                self.pending_resolution_positions = []
                return
            if len(positions) >= 2:
                (row0, col0), (row1, col1) = positions[:2]
                factor1 = random.randint(1, max(original_value - 1, 1))
                factor0 = original_value - factor1
                self.board[row0, col0] = factor1
                self.board[row1, col1] = factor0
                self.queue_move_effects(
                    [
                        {
                            "type": "factorization_burst",
                            "index": int(row0 * self.cols + col0),
                            "delayMs": 100,
                            "durationMs": 620,
                            "animDurationMs": 500,
                            "hideIndices": [int(row0 * self.cols + col0)],
                        },
                        {
                            "type": "factorization_burst",
                            "index": int(row1 * self.cols + col1),
                            "delayMs": 100,
                            "durationMs": 620,
                            "animDurationMs": 500,
                            "hideIndices": [int(row1 * self.cols + col1)],
                        },
                    ]
                )
            self.has_just_exploded = False
            self.pending_resolution_kind = None
            self.pending_resolution_positions = []

    def before_gen_num(self, _direct: int) -> None:
        if self.variant == "airraid":
            mask = self.count_down > 0
            self.count_down[mask] -= 1
            just_zero = (self.count_down == 0) & mask
            self.board[just_zero] = 0
            if self.target_pos is not None:
                if int(self.board[self.target_pos]) != 0:
                    row, col = self.target_pos
                    self.board[row, col] = -1
                    self.count_down[row, col] = 60 + int(self.difficulty) * 40
                    self.queue_move_effects(
                        [
                            {
                                "type": "airraid_fire_drop",
                                "index": int(row * self.cols + col),
                                "durationMs": 250,
                                "animDurationMs": 250,
                            },
                            {
                                "type": "airraid_explosion",
                                "index": int(row * self.cols + col),
                                "delayMs": 250,
                                "durationMs": 450,
                                "animDurationMs": 200,
                            },
                        ]
                    )
                self.target_pos = None
            return
        self._resolve_post_explosion()

    def gen_new_num(self) -> None:
        if self.variant == "airraid":
            board, empty_count, pos, val = self.mover.gen_new_num(
                np.array(self.board, copy=True),
                SingletonConfig().config.get("4_spawn_rate", 0.1),
            )
            self.board = np.array(board, dtype=np.int32)
            self.newtile_pos, self.newtile = int(pos), int(val)
            probability = max(0.08 - float(np.sum(self.count_down > 0)) / 40, 0.01)
            if empty_count > 1 and self.target_pos is None and random.random() < probability:
                empty_positions = self._empty_positions()
                if empty_positions:
                    self.target_pos = random.choice(empty_positions)
            return

        if self.bomb_pos is None and random.random() < self.bomb_gen_rate:
            if self._place_random_bomb():
                self.newtile_pos = int(self.bomb_pos[0] * self.cols + self.bomb_pos[1])
                self.newtile = 0
                return

        board_for_spawn = np.array(self.board, copy=True)
        if self.bomb_pos is not None:
            board_for_spawn[self.bomb_pos] = 1
        board, _, pos, val = self.mover.gen_new_num(
            board_for_spawn,
            SingletonConfig().config.get("4_spawn_rate", 0.1),
        )
        if self.bomb_pos is not None:
            board[self.bomb_pos] = 0
        if self.variant == "hybrid":
            spawn_chance = 0.03 - float(np.sum(board == -3)) * 0.02 + int(self.difficulty) * 0.015
            zero_positions = np.argwhere(board == 0)
            if zero_positions.size and random.random() < spawn_chance:
                row, col = random.choice(zero_positions.tolist())
                board[row, col] = -3
                self.newtile_pos, self.newtile = int(row * self.cols + col), -3
                self.board = np.array(board, dtype=np.int32)
                return
        self.board = np.array(board, dtype=np.int32)
        self.newtile_pos, self.newtile = int(pos), int(val)

    def move_and_check_validity(self, direct: int) -> tuple[np.ndarray, int, bool]:
        if self.variant == "airraid":
            return super().move_and_check_validity(direct)

        is_valid_move = False
        if self.bomb_pos is not None:
            original_bomb_pos = tuple(self.bomb_pos)
            slide_target = self._object_slide_target(direct)
            hit = self._first_hit_in_direction(direct)
            if hit is not None:
                self._queue_object_slide(
                    original_bomb_pos,
                    slide_target,
                    hide_target=False,
                    fade_out_at_end=True,
                    duration_ms=100,
                )
                self._explode_target(*hit)
                self.bomb_pos = None
                is_valid_move = True
            else:
                target = self._bomb_slide_distance(direct)
                if target is not None and target != self.bomb_pos:
                    self._queue_object_slide(original_bomb_pos, target, hide_target=True)
                    self.bomb_pos = target
                    is_valid_move = True
        board_new, new_score = self.mover.move_board(np.array(self.board, copy=True), direct)
        if np.any(board_new != self.board):
            is_valid_move = True
        return np.array(board_new, dtype=np.int32), int(new_score), is_valid_move

    def build_view_state(self) -> dict[str, Any]:
        view = super().build_view_state()
        cover_sprites: dict[str, list[str]] = {}
        tile_text_override: dict[str, str] = {}
        tile_style_variant: dict[str, Any] = {}
        blocked_mask = np.array(view["blockedMask"], dtype=bool)

        if self.variant == "airraid":
            for row in range(self.rows):
                for col in range(self.cols):
                    index = row * self.cols + col
                    if self.target_pos == (row, col):
                        cover_sprites[str(index)] = ["target.png"]
                    if int(self.board[row, col]) == -1:
                        blocked_mask[index] = True
                        cover_sprites[str(index)] = [
                            "crater1.png" if int(self.count_down[row, col]) > (60 + int(self.difficulty) * 40) / 2 else "crater2.png"
                        ]
            view["blockedMask"] = blocked_mask.tolist()
            view["coverSprites"] = cover_sprites
            return view

        if self.bomb_pos is not None:
            bomb_index = int(self.bomb_pos[0] * self.cols + self.bomb_pos[1])
            if self.variant == "explosions":
                cover_sprites[str(bomb_index)] = ["bomb.png"]
            elif self.variant == "giftbox":
                cover_sprites[str(bomb_index)] = ["giftbox.png"]
            elif self.variant == "factorization":
                cover_sprites[str(bomb_index)] = ["tilebg.png"]
            elif self.variant == "hybrid":
                if self.bomb_type == 0:
                    cover_sprites[str(bomb_index)] = ["bomb.png"]
                elif self.bomb_type == 1:
                    cover_sprites[str(bomb_index)] = ["tilebg.png"]
                else:
                    cover_sprites[str(bomb_index)] = ["giftbox.png"]

        if self.variant == "hybrid":
            for index, value in enumerate(self.board.flatten().tolist()):
                if int(value) == -3:
                    cover_sprites[str(index)] = ["portal.png"]
                    tile_style_variant[str(index)] = {"kind": "portal"}

        view["coverSprites"] = cover_sprites
        view["tileTextOverride"] = tile_text_override
        view["tileStyleVariant"] = tile_style_variant
        return view

    def get_info_text(self) -> str:
        messages = {
            "explosions": "A small chance of generating a bomb that destroys the first tile it encounters!",
            "giftbox": "A small chance of generating a gift box that magically changes the first tile it encounters!",
            "factorization": "A small chance of generating a power-up that halves the first tile it encounters!",
            "hybrid": "All-Stars.",
            "airraid": "Airstrikes incoming! Avoid marked targets!",
        }
        return messages[self.variant]
