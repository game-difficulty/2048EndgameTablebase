from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from Config import SingletonConfig

from ..animation import build_minigame_move_animation_metadata
from ..registry import MinigameDefinition
from .mover import create_mover


DirectionMap = {
    "left": 1,
    "right": 2,
    "up": 3,
    "down": 4,
}


@dataclass
class AnimationState:
    appear_index: int | None = None
    appear_value: int | None = None
    direction: str | None = None
    valid_move: bool = False
    slide_distances: list[int] | None = None
    pop_positions: list[int] | None = None
    effects: list[dict[str, Any]] | None = None
    page_effects: list[dict[str, Any]] | None = None
    follow_up: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        if not self.valid_move and not self.effects and not self.page_effects:
            payload: dict[str, Any] = {}
            if self.follow_up:
                payload["followUp"] = dict(self.follow_up)
            return payload
        payload: dict[str, Any] = {
            "direction": self.direction,
            "slide_distances": list(self.slide_distances or []),
            "pop_positions": list(self.pop_positions or []),
        }
        if self.appear_index is not None and self.appear_value is not None:
            payload["appearTile"] = {
                "index": int(self.appear_index),
                "value": int(self.appear_value),
            }
        if self.effects:
            payload["effects"] = list(self.effects)
        if self.page_effects:
            payload["pageEffects"] = list(self.page_effects)
        if self.follow_up:
            payload["followUp"] = dict(self.follow_up)
        return payload


class BaseMinigameEngine:
    def get_initial_shape(self) -> tuple[int, int]:
        return 4, 4

    def __init__(self, definition: MinigameDefinition, difficulty: int) -> None:
        self.definition = definition
        self.game_id = definition.id
        self.title = definition.title
        self.legacy_name = definition.legacy_name
        self.difficulty = int(difficulty)

        self.rows, self.cols = self.get_initial_shape()
        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.mover = create_mover((self.rows, self.cols))

        self.score = 0
        self.max_score = 0
        self.max_num = 0
        self.current_max_num = 0
        self.highest_tile_exp = 0
        self.is_passed = 0
        self.newtile_pos = -1
        self.newtile = 0
        self.is_over = False

        self._pending_messages: dict[str, Any] = {}
        self._animation = AnimationState()
        self._queued_follow_up_animation: dict[str, Any] | None = None
        self._queued_move_effects: list[dict[str, Any]] = []
        self._queued_page_effects: list[dict[str, Any]] = []
        self._last_valid_move = False
        self._last_direction: str | None = None
        self._last_move_at_ms = 0

        self._load_or_initialize()

    @staticmethod
    def _positive_max(board: np.ndarray) -> int:
        positive_values = board[board > 0]
        return int(positive_values.max()) if positive_values.size else 0

    def _load_or_initialize(self) -> None:
        config = SingletonConfig().config
        state_by_difficulty = config.setdefault("minigame_state", [dict(), dict()])
        saved_entry = state_by_difficulty[self.difficulty].get(self.legacy_name)
        if saved_entry:
            try:
                main_state = saved_entry[0]
                self.board = np.array(main_state[0], dtype=np.int32)
                self.score = int(main_state[1])
                self.max_score = int(main_state[2])
                self.max_num = int(main_state[3])
                self.is_passed = int(main_state[4])
                self.newtile_pos = int(main_state[5])
                self.highest_tile_exp = int(main_state[6]) if len(main_state) > 6 else int(main_state[3])
                extra_state = saved_entry[1] if len(saved_entry) > 1 else []
                self.load_legacy_extra(extra_state)
                if self.board.shape != (self.rows, self.cols):
                    self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
            except Exception:
                self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
                self.score = 0
                self.max_score = 0
                self.max_num = 0
                self.highest_tile_exp = 0
                self.is_passed = 0
                self.newtile_pos = -1

        self.current_max_num = max(self.max_num, self._positive_max(self.board))
        self._refresh_highest_tile_exp()
        if np.all(self.board <= 0):
            self.setup_new_game()

    def load_legacy_extra(self, extra_state: list[Any]) -> None:
        _ = extra_state

    def export_legacy_extra(self) -> list[Any]:
        return []

    def _refresh_highest_tile_exp(self) -> None:
        self.highest_tile_exp = max(int(self.highest_tile_exp), self._positive_max(self.board))

    def setup_new_game(self) -> None:
        spawn_rate = SingletonConfig().config.get("4_spawn_rate", 0.1)
        board = np.zeros((self.rows, self.cols), dtype=np.int32)
        board, _, _, _ = self.mover.gen_new_num(board, spawn_rate)
        board, _, pos, val = self.mover.gen_new_num(board, spawn_rate)
        self.board = np.array(board, dtype=np.int32)
        self.score = 0
        self.newtile_pos = int(pos)
        self.newtile = int(val)
        self.current_max_num = self._positive_max(self.board)
        self.highest_tile_exp = self.current_max_num
        self.is_over = False
        self._pending_messages.clear()
        self._animation = AnimationState(
            appear_index=self.newtile_pos,
            appear_value=self.newtile,
            valid_move=True,
        )
        self.save_to_config()

    def setup_new_round(self) -> None:
        self.setup_new_game()

    def save_to_config(self) -> None:
        config = SingletonConfig().config
        config.setdefault("minigame_state", [dict(), dict()])
        config["minigame_state"][self.difficulty][self.legacy_name] = (
            [
                np.array(self.board, dtype=np.int32),
                int(self.score),
                int(self.max_score),
                int(self.max_num),
                int(self.is_passed),
                int(self.newtile_pos),
                int(self.highest_tile_exp),
            ],
            self.export_legacy_extra(),
        )
        SingletonConfig().save_config(config)

    def queue_message(self, key: str, value: Any) -> None:
        self._pending_messages[key] = value

    def pop_messages(self) -> dict[str, Any]:
        payload = dict(self._pending_messages)
        self._pending_messages.clear()
        return payload

    def gen_new_num(self) -> None:
        board, _, pos, val = self.mover.gen_new_num(
            np.array(self.board, copy=True),
            SingletonConfig().config.get("4_spawn_rate", 0.1),
        )
        self.board = np.array(board, dtype=np.int32)
        self.newtile_pos = int(pos)
        self.newtile = int(val)
        self._refresh_highest_tile_exp()

    def before_move(self, direct: int) -> None:
        _ = direct

    def before_gen_num(self, direct: int) -> None:
        _ = direct

    def after_gen_num(self) -> None:
        return

    def move_and_check_validity(self, direct: int) -> tuple[np.ndarray, int, bool]:
        new_board, new_score = self.mover.move_board(np.array(self.board, copy=True), direct)
        is_valid = bool(np.any(new_board != self.board))
        return np.array(new_board, dtype=np.int32), int(new_score), is_valid

    def has_possible_move(self) -> bool:
        if np.any(self.board == 0):
            return True
        for direction in (1, 2, 3, 4):
            board_new, _, is_valid = self.move_and_check_validity(direction)
            if is_valid and np.any(board_new != self.board):
                return True
        return False

    def check_game_passed(self) -> None:
        current_peak = self._positive_max(self.board)
        self._refresh_highest_tile_exp()
        previous_peak = int(self.current_max_num)
        previous_best = int(self.max_num)
        self.current_max_num = max(self.current_max_num, current_peak)
        self.max_num = max(self.max_num, self.current_max_num)
        if self.current_max_num <= 9:
            return
        if self.current_max_num > previous_peak:
            if self.current_max_num > previous_best:
                self.is_passed = {12: 3, 11: 2, 10: 1}.get(self.max_num, 4)
                level = {12: "gold", 11: "silver", 10: "bronze"}.get(
                    self.current_max_num, "gold"
                )
                self.queue_message(
                    "trophy",
                    {
                        "level": level,
                        "message": f"You achieved {2 ** self.max_num}! You get a {level} trophy!",
                    },
                )
            else:
                level = {12: "gold", 11: "silver", 10: "bronze"}.get(
                    self.current_max_num, "gold"
                )
                self.queue_message(
                    "trophy",
                    {
                        "level": level,
                        "message": f"You achieved {2 ** self.current_max_num}! Take it further!",
                    },
                )

    def check_game_over(self) -> None:
        was_over = bool(self.is_over)
        self.is_over = not self.has_possible_move()
        if self.is_over and not was_over:
            self.queue_message("gameOver", "Game Over")

    def request_info(self) -> None:
        self.queue_message("infoDialog", self.get_info_text())

    def get_info_text(self) -> str:
        return "More minigames."

    def handle_custom_action(self, key: str, phase: str = "trigger") -> bool:
        _ = key, phase
        return False

    def clear_animation(self) -> None:
        self._animation = AnimationState()
        self._queued_follow_up_animation = None
        self._queued_move_effects = []
        self._queued_page_effects = []

    def set_special_effects(self, effects: list[dict[str, Any]] | None) -> None:
        self._animation = AnimationState(effects=list(effects or []))
        self._queued_follow_up_animation = None
        self._queued_move_effects = []
        self._queued_page_effects = []

    def queue_move_effects(self, effects: list[dict[str, Any]] | None) -> None:
        if not effects:
            return
        self._queued_move_effects.extend(dict(effect) for effect in effects)

    def queue_page_effects(self, effects: list[dict[str, Any]] | None) -> None:
        if not effects:
            return
        self._queued_page_effects.extend(dict(effect) for effect in effects)

    def set_follow_up_animation(self, animation: dict[str, Any] | None) -> None:
        self._queued_follow_up_animation = dict(animation) if animation else None

    def do_move(self, direction: str) -> None:
        direction_key = str(direction or "").strip().lower()
        direct = DirectionMap.get(direction_key)
        self._last_direction = direction_key if direct else None
        self._last_valid_move = False
        self.clear_animation()
        if direct is None:
            return

        board_before = np.array(self.board, copy=True)
        self.before_move(direct)
        board_new, new_score, is_valid_move = self.move_and_check_validity(direct)
        if not is_valid_move:
            self.check_game_over()
            return

        self.board = np.array(board_new, dtype=np.int32)
        self.score += int(new_score)
        self.max_score = max(self.max_score, self.score)
        self.before_gen_num(direct)
        self.gen_new_num()
        self.after_gen_num()
        self._refresh_highest_tile_exp()
        self.check_game_passed()
        self.check_game_over()

        self._last_valid_move = True
        self._last_move_at_ms = int(time.time() * 1000)
        animation_metadata = build_minigame_move_animation_metadata(
            board_before,
            direction_key,
            spawn_index=self.newtile_pos if self.newtile_pos >= 0 else None,
            spawn_value=self.newtile if self.newtile > 0 else None,
        )
        self._animation = AnimationState(
            appear_index=animation_metadata.get("appear_tile", {}).get("index"),
            appear_value=animation_metadata.get("appear_tile", {}).get("value"),
            direction=animation_metadata.get("direction"),
            valid_move=True,
            slide_distances=animation_metadata.get("slide_distances"),
            pop_positions=animation_metadata.get("pop_positions"),
            effects=list(self._queued_move_effects),
            page_effects=list(self._queued_page_effects),
            follow_up=self._queued_follow_up_animation,
        )
        self._queued_follow_up_animation = None
        self._queued_move_effects = []
        self._queued_page_effects = []
        self.save_to_config()

    def build_view_state(self) -> dict[str, Any]:
        cell_count = self.rows * self.cols
        return {
            "hiddenMask": [False] * cell_count,
            "blockedMask": [False] * cell_count,
            "smallLabels": [""] * cell_count,
            "tileOverlays": {},
            "tileTextOverride": {},
            "tileStyleVariant": {},
            "coverSprites": {},
        }

    def build_hud(self) -> dict[str, Any]:
        return {
            "score": int(self.score),
            "best": int(self.max_score),
            "infoText": self.get_info_text(),
            "customPanels": [],
        }

    def build_powerups(self) -> dict[str, Any]:
        config = SingletonConfig().config
        init_count = 1 if self.difficulty == 1 else 5
        counts = config.setdefault("power_ups_state", [dict(), dict()])[self.difficulty].get(
            self.legacy_name,
            [init_count, init_count, init_count],
        )
        return {
            "enabled": bool(self.definition.supports_powerups),
            "counts": {
                "bomb": int(counts[0]),
                "glove": int(counts[1]),
                "twist": int(counts[2]),
            },
            "activeMode": None,
        }

    def build_interaction(self) -> dict[str, Any]:
        return {
            "active": False,
            "mode": None,
            "targetType": None,
            "phase": None,
        }

    def build_messages(self) -> dict[str, Any]:
        return self.pop_messages()

    def build_animation(self) -> dict[str, Any]:
        return self._animation.to_dict()

    def serialize_state(self) -> dict[str, Any]:
        messages = self.build_messages()
        status = "game_over" if self.is_over else "running"
        if "trophy" in messages:
            status = "trophy"
        return {
            "gameId": self.game_id,
            "title": self.title,
            "difficulty": int(self.difficulty),
            "board": self.board.flatten().astype(int).tolist(),
            "shape": {"rows": int(self.rows), "cols": int(self.cols)},
            "score": int(self.score),
            "best": int(self.max_score),
            "status": status,
            "animation": self.build_animation(),
            "view": self.build_view_state(),
            "hud": self.build_hud(),
            "powerups": self.build_powerups(),
            "interaction": self.build_interaction(),
            "messages": messages,
        }
