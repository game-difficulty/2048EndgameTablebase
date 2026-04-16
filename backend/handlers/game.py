from __future__ import annotations

from typing import Any

import numpy as np
from Config import SingletonConfig, logger
from fastapi import WebSocket
from egtb_core.VBoardMover import decode_board
from egtb_core.BoardMover import (
    encode_board as r_encode_board,
    s_gen_new_num as r_gen_new_num,
    s_move_board as r_move_board,
)

from ..actions import Action, Message
from ..animation import build_move_animation_metadata
from ..session import GameSession
from ..session import np_u64, u64
from ..state import ConnectionManager
from ..state import save_game_state

GAMER_TOP_TILE = 32768


def _log_ai_step_debug(best_move: str | None, table_name: str | None):
    message = f"AI step move={best_move}, table={table_name or 'Unknown'}"
    print(message)
    logger.warning(message)


def _gamer_board_with_special_tiles(
    board_encoded: np.uint64, special_tiles: dict[int, int]
):
    board = decode_board(np.uint64(u64(board_encoded))).copy()
    flat = board.reshape(-1)
    for index, value in special_tiles.items():
        if 0 <= int(index) < flat.size and int(flat[int(index)]) == GAMER_TOP_TILE:
            flat[int(index)] = int(value)
    return board


def _gamer_collapse_board(actual_board):
    collapsed = np.where(actual_board >= GAMER_TOP_TILE, GAMER_TOP_TILE, actual_board)
    return np.uint64(u64(r_encode_board(collapsed.astype(np.int32))))


def _gamer_extract_special_tiles(actual_board):
    special_tiles = {}
    for index, value in enumerate(actual_board.reshape(-1).tolist()):
        if int(value) > GAMER_TOP_TILE:
            special_tiles[index] = int(value)
    return special_tiles


def _gamer_get_ai_state(session: GameSession):
    actual_board = _gamer_board_with_special_tiles(
        session.board_encoded, getattr(session, "gamer_special_tiles", {})
    )
    special_tiles = getattr(session, "gamer_special_tiles", {})
    if not special_tiles:
        return actual_board, np.uint64(u64(session.board_encoded)), True

    return actual_board, np.uint64(u64(session.board_encoded)), False


def _gamer_line_positions(direction_str: str):
    if direction_str == "left":
        return [[(row, col) for col in range(4)] for row in range(4)]
    if direction_str == "right":
        return [[(row, col) for col in range(3, -1, -1)] for row in range(4)]
    if direction_str == "up":
        return [[(row, col) for row in range(4)] for col in range(4)]
    if direction_str == "down":
        return [[(row, col) for row in range(3, -1, -1)] for col in range(4)]
    return []


def _gamer_simulate_line(values):
    result = [0, 0, 0, 0]
    distances = [0, 0, 0, 0]
    pops = [0, 0, 0, 0]
    score_delta = 0

    non_zero = [
        (idx, int(value)) for idx, value in enumerate(values) if int(value) != 0
    ]
    read = 0
    write = 0
    while read < len(non_zero):
        source_index, value = non_zero[read]
        if read + 1 < len(non_zero) and non_zero[read + 1][1] == value:
            next_source_index, _ = non_zero[read + 1]
            merged_value = value * 2
            result[write] = merged_value
            score_delta += merged_value
            distances[source_index] = source_index - write
            distances[next_source_index] = next_source_index - write
            pops[write] = 1
            read += 2
        else:
            result[write] = value
            distances[source_index] = source_index - write
            read += 1
        write += 1

    return result, distances, pops, score_delta


def _gamer_simulate_move(actual_board, direction_str: str):
    next_board = np.zeros((4, 4), dtype=np.int32)
    slide_distances = [0] * 16
    pop_positions = [0] * 16
    score_delta = 0

    for line_positions in _gamer_line_positions(direction_str):
        line_values = [int(actual_board[row, col]) for row, col in line_positions]
        next_values, line_distances, line_pops, line_score = _gamer_simulate_line(
            line_values
        )
        score_delta += line_score

        for offset, (row, col) in enumerate(line_positions):
            board_index = row * 4 + col
            slide_distances[board_index] = int(line_distances[offset])
            if line_pops[offset]:
                pop_positions[board_index] = 1
            next_board[row, col] = int(next_values[offset])

    return next_board, slide_distances, pop_positions, score_delta


async def handle_game_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
    manager: ConnectionManager,
) -> bool:
    spawn_rate4 = float(SingletonConfig().config.get("4_spawn_rate", 0.1))

    if action == Action.INIT_GAME:
        session.board_encoded = np_u64(0)
        session.score = 0
        session.gamer_special_tiles = {}
        session.ensure_evil_gen().reset_board(u64(session.board_encoded))

        initial_board, _, _, _ = r_gen_new_num(session.board_encoded, spawn_rate4)
        initial_board, _, _, _ = r_gen_new_num(initial_board, spawn_rate4)
        session.board_encoded = np_u64(initial_board)

        session.history = [(session.board_encoded, session.score)]
        session.move_history = [None]
        session.gamer_special_history = [dict(session.gamer_special_tiles)]
        session.played_length = 0
        await manager.send_state(websocket)
        return True

    if action == Action.USER_MOVE:
        direction_str = str(payload.get("dir") or "")
        direction_map = {"left": 1, "right": 2, "up": 3, "down": 4}
        if direction_str not in direction_map:
            return True

        direction = direction_map[direction_str]
        old_board_encoded = np_u64(session.board_encoded)
        old_actual_board = _gamer_board_with_special_tiles(
            old_board_encoded, getattr(session, "gamer_special_tiles", {})
        )
        moved_actual_board, slide_distances, pop_positions, move_score = (
            _gamer_simulate_move(old_actual_board, direction_str)
        )
        if np.array_equal(moved_actual_board, old_actual_board):
            return True

        collapsed_after_move = _gamer_collapse_board(moved_actual_board)
        session.score += int(move_score)
        if session.score > session.best_score:
            session.best_score = session.score

        if np.random.rand() > session.difficulty:
            new_board_with_num, _, num_pos_1d, val_exp = r_gen_new_num(
                np_u64(collapsed_after_move), spawn_rate4
            )
        else:
            evil_gen = session.ensure_evil_gen()
            evil_gen.reset_board(u64(collapsed_after_move))
            new_board_with_num, num_pos_1d, val_exp = evil_gen.gen_new_num(depth=5)

        session.board_encoded = np_u64(new_board_with_num)
        spawned_actual_board = moved_actual_board.copy()
        if num_pos_1d is not None and num_pos_1d >= 0 and val_exp > 0:
            spawned_actual_board[num_pos_1d // 4, num_pos_1d % 4] = 2**val_exp
        session.gamer_special_tiles = _gamer_extract_special_tiles(spawned_actual_board)
        session.history.append((session.board_encoded, session.score))
        session.move_history.append(direction_str)
        session.gamer_special_history.append(dict(session.gamer_special_tiles))
        session.played_length = len(session.history) - 1

        metadata = {
            "direction": direction_str,
            "slide_distances": slide_distances,
            "pop_positions": pop_positions,
            "appear_tile": {
                "index": int(num_pos_1d),
                "value": int(2**val_exp if val_exp > 0 else 0),
            },
        }
        await manager.send_state(websocket, metadata)
        return True

    if action == Action.UNDO:
        if len(session.history) > 1:
            from ..trainer_helpers import _clear_record_replay

            if session.client_id.startswith("trainer_"):
                _clear_record_replay(session)
            session.history.pop()
            session.move_history.pop() if len(session.move_history) > 1 else None
            if session.client_id.startswith("gamer_") and getattr(
                session, "gamer_special_history", None
            ):
                session.gamer_special_history.pop()
            last_state = session.history[-1]
            session.board_encoded = last_state[0]
            session.score = last_state[1]
            if session.client_id.startswith("gamer_"):
                session.gamer_special_tiles = (
                    dict(session.gamer_special_history[-1])
                    if getattr(session, "gamer_special_history", None)
                    else {}
                )
            session.played_length = len(session.history) - 1
            await manager.send_state(websocket)
        return True

    if action == Action.SET_BOARD:
        hex_str = str(payload.get("hex_str") or "")
        try:
            new_encoded = np_u64(int(hex_str, 16))
            from ..trainer_helpers import _clear_record_replay

            if session.client_id.startswith("trainer_"):
                _clear_record_replay(session)
            session.board_encoded = new_encoded
            session.score = 0
            session.gamer_special_tiles = {}
            session.history = [(session.board_encoded, session.score)]
            session.move_history = [None]
            session.gamer_special_history = [dict(session.gamer_special_tiles)]
            session.played_length = 0
            await manager.send_state(websocket)
        except ValueError:
            pass
        return True

    if action == Action.SAVE_GAME_STATE:
        save_game_state(payload)
        return True

    if action == Action.AI_STEP:
        if session.client_id.startswith("gamer_"):
            board_2d, ai_board_encoded, allow_resolve_32768 = _gamer_get_ai_state(
                session
            )
        else:
            board_2d = decode_board(np.uint64(u64(session.board_encoded)))
            ai_board_encoded = np_u64(session.board_encoded)
            allow_resolve_32768 = False

        ai_dispatcher = session.ensure_ai_dispatcher()
        ai_dispatcher.reset(board_2d, u64(ai_board_encoded))
        best_move = ai_dispatcher.dispatcher()
        valid_moves = {"left", "right", "up", "down"}

        if best_move == "AI":
            if allow_resolve_32768:
                from ai_and_sort import ai_core

                ai_board_encoded = ai_core.resolve_32768_doubles(u64(ai_board_encoded))
            player, logic = session.ensure_ai_fallback(
                ai_board_encoded,
                SingletonConfig().config["4_spawn_rate"],
                getattr(ai_dispatcher, "time_limit_ratio", 1.0),
            )
            best_move_code = logic.calculate_step(
                player, board_2d, ai_dispatcher.counts
            )
            logger.warning(
                f"{player.do_check} {player.masked_count} {player.max_d} {player.prune}"
            )
            move_map = {1: "left", 2: "right", 3: "up", 4: "down"}
            best_move = move_map.get(best_move_code, None)
        else:
            best_move = best_move.lower() if best_move else None

        _log_ai_step_debug(
            best_move if best_move in valid_moves else None,
            getattr(session.ai_dispatcher, "current_table", None),
        )

        await websocket.send_json(
            {
                "action": Message.DO_AI_MOVE_CMD,
                "data": {"dir": best_move if best_move in valid_moves else None},
            }
        )
        return True

    return False
