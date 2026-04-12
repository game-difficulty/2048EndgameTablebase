from __future__ import annotations

import random
from typing import Any

import numpy as np

from Config import SingletonConfig

from .session import MinigameSessionState


POWERUP_KEYS = ("bomb", "glove", "twist")


def _default_count_for_difficulty(difficulty: int) -> int:
    return 1 if int(difficulty) == 1 else 5


def _new_game_default_counts(state: MinigameSessionState) -> dict[str, int]:
    if state.engine is not None and getattr(state.engine, "legacy_name", "") == "Blitzkrieg":
        return {"bomb": 10, "glove": 10, "twist": 10}
    default_count = _default_count_for_difficulty(state.difficulty)
    return {key: default_count for key in POWERUP_KEYS}


def load_powerup_counts(state: MinigameSessionState) -> dict[str, int]:
    if state.engine is None:
        return {"bomb": 0, "glove": 0, "twist": 0}
    config = SingletonConfig().config
    difficulty_state = config.setdefault("power_ups_state", [dict(), dict()])[state.difficulty]
    saved_counts = difficulty_state.get(getattr(state.engine, "legacy_name", ""), None)
    if not saved_counts:
        return _new_game_default_counts(state)
    values = list(saved_counts)
    while len(values) < len(POWERUP_KEYS):
        values.append(_default_count_for_difficulty(state.difficulty))
    return {key: max(0, int(values[index])) for index, key in enumerate(POWERUP_KEYS)}


def save_powerup_counts(state: MinigameSessionState) -> None:
    if state.engine is None:
        return
    config = SingletonConfig().config
    difficulty_state = config.setdefault("power_ups_state", [dict(), dict()])[state.difficulty]
    difficulty_state[getattr(state.engine, "legacy_name", "")] = tuple(
        int(state.powerup_counts.get(key, 0)) for key in POWERUP_KEYS
    )
    SingletonConfig().save_config(config)


def reset_powerup_counts(state: MinigameSessionState) -> None:
    state.powerup_counts = _new_game_default_counts(state)
    save_powerup_counts(state)


def _valid_bomb_targets(engine) -> list[int]:
    return [index for index, value in enumerate(engine.board.flatten().tolist()) if int(value) > 0]


def _valid_glove_sources(engine) -> list[int]:
    return [index for index, value in enumerate(engine.board.flatten().tolist()) if int(value) > 0]


def _valid_glove_targets(engine) -> list[int]:
    return [index for index, value in enumerate(engine.board.flatten().tolist()) if int(value) == 0]


def _valid_twist_targets(engine) -> list[int]:
    valid_targets: list[int] = []
    for row in range(engine.rows - 1):
        for col in range(engine.cols - 1):
            sub_board = np.array(engine.board[row : row + 2, col : col + 2], copy=False)
            if np.all(sub_board == 0):
                continue
            if np.any(sub_board == -1):
                continue
            valid_targets.append(row * engine.cols + col)
    return valid_targets


def build_powerups_payload(state: MinigameSessionState) -> dict[str, Any]:
    enabled = state.engine is not None
    return {
        "enabled": enabled,
        "counts": {key: int(state.powerup_counts.get(key, 0)) for key in POWERUP_KEYS},
        "activeMode": state.active_mode,
    }


def build_interaction_payload(state: MinigameSessionState) -> dict[str, Any]:
    engine = state.engine
    if engine is None or not state.active_mode:
        return {
            "active": False,
            "mode": None,
            "targetType": None,
            "phase": None,
            "validTargets": [],
            "selectedIndices": [],
            "hintKey": "",
        }

    if state.active_mode == "bomb":
        return {
            "active": True,
            "mode": "bomb",
            "targetType": "singleTile",
            "phase": "target",
            "validTargets": _valid_bomb_targets(engine),
            "selectedIndices": [],
            "hintKey": "minigames.powerups.hints.bomb",
        }

    if state.active_mode == "glove" and state.interaction_phase == 1:
        return {
            "active": True,
            "mode": "glove",
            "targetType": "singleTile",
            "phase": "source",
            "validTargets": _valid_glove_sources(engine),
            "selectedIndices": [],
            "hintKey": "minigames.powerups.hints.gloveSource",
        }

    if state.active_mode == "glove" and state.interaction_phase == 2:
        source_index = int((state.selection_cache or {}).get("source_index", -1))
        return {
            "active": True,
            "mode": "glove",
            "targetType": "singleTile",
            "phase": "target",
            "validTargets": _valid_glove_targets(engine),
            "selectedIndices": [source_index] if source_index >= 0 else [],
            "hintKey": "minigames.powerups.hints.gloveTarget",
        }

    if state.active_mode == "twist":
        return {
            "active": True,
            "mode": "twist",
            "targetType": "subgridTopLeft",
            "phase": "target",
            "validTargets": _valid_twist_targets(engine),
            "selectedIndices": [],
            "hintKey": "minigames.powerups.hints.twist",
        }

    return {
        "active": False,
        "mode": None,
        "targetType": None,
        "phase": None,
        "validTargets": [],
        "selectedIndices": [],
        "hintKey": "",
    }


def cancel_powerup_interaction(
    state: MinigameSessionState, *, clear_animation: bool = True
) -> None:
    if clear_animation and state.engine is not None:
        state.engine.clear_animation()
    state.active_mode = None
    state.interaction_phase = 0
    state.selection_cache = None


def activate_powerup(state: MinigameSessionState, mode: str) -> bool:
    engine = state.engine
    normalized_mode = str(mode or "").strip().lower()
    if engine is None or normalized_mode not in POWERUP_KEYS:
        return False
    engine.clear_animation()
    if int(state.powerup_counts.get(normalized_mode, 0)) <= 0:
        return False
    if state.active_mode == normalized_mode:
        cancel_powerup_interaction(state)
        return True
    state.active_mode = normalized_mode
    state.interaction_phase = 1
    state.selection_cache = None
    return True


def _consume_powerup(state: MinigameSessionState, mode: str) -> None:
    state.powerup_counts[mode] = max(0, int(state.powerup_counts.get(mode, 0)) - 1)
    save_powerup_counts(state)


def _post_powerup_update(state: MinigameSessionState) -> None:
    engine = state.engine
    if engine is None:
        return
    engine.check_game_passed()
    engine.check_game_over()
    engine.save_to_config()


def _apply_bomb(state: MinigameSessionState, index: int) -> bool:
    engine = state.engine
    if engine is None:
        return False
    row, col = divmod(index, engine.cols)
    if row < 0 or row >= engine.rows or col < 0 or col >= engine.cols:
        engine.clear_animation()
        return False
    if int(engine.board[row, col]) <= 0:
        engine.clear_animation()
        return False
    engine.board[row, col] = 0
    engine.set_special_effects([{"type": "explosion", "index": int(index)}])
    _consume_powerup(state, "bomb")
    _post_powerup_update(state)
    return True


def _apply_glove_step(state: MinigameSessionState, index: int) -> bool:
    engine = state.engine
    if engine is None:
        return False
    row, col = divmod(index, engine.cols)
    if row < 0 or row >= engine.rows or col < 0 or col >= engine.cols:
        cancel_powerup_interaction(state)
        return False

    if state.interaction_phase == 1:
        value = int(engine.board[row, col])
        if value <= 0:
            engine.clear_animation()
            return False
        state.interaction_phase = 2
        state.selection_cache = {"source_index": int(index), "value": value}
        engine.set_special_effects(
            [{"type": "grab", "index": int(index), "value": int(value)}]
        )
        return False

    source_index = int((state.selection_cache or {}).get("source_index", -1))
    if source_index < 0:
        cancel_powerup_interaction(state)
        return False

    source_row, source_col = divmod(source_index, engine.cols)
    source_value = int(engine.board[source_row, source_col])
    success = int(engine.board[row, col]) == 0 and source_value > 0
    if success:
        engine.board[row, col] = source_value
        engine.board[source_row, source_col] = 0
        engine.set_special_effects(
            [
                {
                    "type": "glove_move",
                    "fromIndex": int(source_index),
                    "toIndex": int(index),
                    "value": int(source_value),
                }
            ]
        )
        _consume_powerup(state, "glove")
        _post_powerup_update(state)
    else:
        engine.clear_animation()
    cancel_powerup_interaction(state, clear_animation=not success)
    return success


def _rotate_2x2_clockwise(board: np.ndarray, row: int, col: int) -> None:
    sub_board = np.array(board[row : row + 2, col : col + 2], copy=True)
    board[row : row + 2, col : col + 2] = np.array(
        [[sub_board[1, 0], sub_board[0, 0]], [sub_board[1, 1], sub_board[0, 1]]],
        dtype=board.dtype,
    )


def _apply_twist(state: MinigameSessionState, index: int) -> bool:
    engine = state.engine
    if engine is None:
        return False
    row, col = divmod(index, engine.cols)
    if row < 0 or col < 0 or row >= engine.rows - 1 or col >= engine.cols - 1:
        engine.clear_animation()
        return False
    sub_board = np.array(engine.board[row : row + 2, col : col + 2], copy=False)
    if np.all(sub_board == 0) or np.any(sub_board == -1):
        engine.clear_animation()
        return False
    top_left = row * engine.cols + col
    top_right = top_left + 1
    bottom_left = (row + 1) * engine.cols + col
    bottom_right = bottom_left + 1
    twist_tiles = []
    twist_mapping = (
        (top_left, top_right, int(engine.board[row, col])),
        (top_right, bottom_right, int(engine.board[row, col + 1])),
        (bottom_left, top_left, int(engine.board[row + 1, col])),
        (bottom_right, bottom_left, int(engine.board[row + 1, col + 1])),
    )
    for from_index, to_index, value in twist_mapping:
        if value > 0:
            twist_tiles.append(
                {
                    "fromIndex": int(from_index),
                    "toIndex": int(to_index),
                    "value": int(value),
                }
            )
    _rotate_2x2_clockwise(engine.board, row, col)
    engine.set_special_effects(
        [
            {
                "type": "twist",
                "index": int(index),
                "tiles": twist_tiles,
            }
        ]
    )
    _consume_powerup(state, "twist")
    _post_powerup_update(state)
    return True


def apply_target_action(state: MinigameSessionState, index: int) -> bool:
    if state.engine is None or not state.active_mode:
        return False
    success = False
    if state.active_mode == "bomb":
        success = _apply_bomb(state, index)
        cancel_powerup_interaction(state, clear_animation=not success)
        return success
    if state.active_mode == "glove":
        return _apply_glove_step(state, index)
    if state.active_mode == "twist":
        success = _apply_twist(state, index)
        cancel_powerup_interaction(state, clear_animation=not success)
        return success
    cancel_powerup_interaction(state)
    return False


def maybe_award_random_powerup(state: MinigameSessionState, score_delta: int) -> str | None:
    if score_delta < 300 or score_delta >= 1024:
        return None
    probability = 0.05 if int(state.difficulty) == 1 else 0.25
    if random.random() >= probability:
        return None
    awarded = random.choice(list(POWERUP_KEYS))
    state.powerup_counts[awarded] = int(state.powerup_counts.get(awarded, 0)) + 1
    save_powerup_counts(state)
    return awarded
