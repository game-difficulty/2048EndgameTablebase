from __future__ import annotations

from typing import Any

import numpy as np
from Config import SingletonConfig
from fastapi import WebSocket
from egtb_core.VBoardMover import decode_board
from egtb_core.BoardMover import s_gen_new_num as r_gen_new_num, s_move_board as r_move_board

from ..actions import Action, Message
from ..animation import build_move_animation_metadata
from ..session import GameSession
from ..session import u64
from ..state import ConnectionManager
from ..state import save_game_state


async def handle_game_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
    manager: ConnectionManager,
) -> bool:
    spawn_rate4 = float(SingletonConfig().config.get("4_spawn_rate", 0.1))

    if action == Action.INIT_GAME:
        session.board_encoded = np.uint64(0)
        session.score = 0
        session.evil_gen.reset_board(u64(session.board_encoded))

        initial_board, _, _, _ = r_gen_new_num(session.board_encoded, spawn_rate4)
        initial_board, _, _, _ = r_gen_new_num(initial_board, spawn_rate4)
        session.board_encoded = np.uint64(u64(initial_board))

        session.history = [(session.board_encoded, session.score)]
        session.move_history = [None]
        session.played_length = 0
        await manager.send_state(websocket)
        return True

    if action == Action.USER_MOVE:
        direction_str = str(payload.get("dir") or "")
        direction_map = {"left": 1, "right": 2, "up": 3, "down": 4}
        if direction_str not in direction_map:
            return True

        direction = direction_map[direction_str]
        old_board_encoded = session.board_encoded

        new_board, move_score = r_move_board(old_board_encoded, direction)
        new_board = np.uint64(u64(new_board))
        if new_board == old_board_encoded:
            return True

        session.score += int(move_score)
        if session.score > session.best_score:
            session.best_score = session.score

        if np.random.rand() > session.difficulty:
            new_board_with_num, _, num_pos_1d, val_exp = r_gen_new_num(
                new_board, spawn_rate4
            )
        else:
            session.evil_gen.reset_board(u64(new_board))
            new_board_with_num, num_pos_1d, val_exp = session.evil_gen.gen_new_num(
                depth=5
            )

        session.board_encoded = np.uint64(u64(new_board_with_num))
        session.history.append((session.board_encoded, session.score))
        session.move_history.append(direction_str)
        session.played_length = len(session.history) - 1

        metadata = build_move_animation_metadata(
            direction_str,
            board_encoded=old_board_encoded,
            spawn_index=num_pos_1d,
            spawn_value=2**val_exp if val_exp > 0 else 0,
            error_prefix="Animation metadata error",
            trace=True,
        )
        await manager.send_state(websocket, metadata)
        return True

    if action == Action.UNDO:
        if len(session.history) > 1:
            from ..trainer_helpers import _clear_record_replay

            if session.client_id.startswith("trainer_"):
                _clear_record_replay(session)
            session.history.pop()
            session.move_history.pop() if len(session.move_history) > 1 else None
            last_state = session.history[-1]
            session.board_encoded = last_state[0]
            session.score = last_state[1]
            session.played_length = len(session.history) - 1
            await manager.send_state(websocket)
        return True

    if action == Action.SET_BOARD:
        hex_str = str(payload.get("hex_str") or "")
        try:
            new_encoded = np.uint64(int(hex_str, 16))
            from ..trainer_helpers import _clear_record_replay

            if session.client_id.startswith("trainer_"):
                _clear_record_replay(session)
            session.board_encoded = new_encoded
            session.score = 0
            session.history = [(session.board_encoded, session.score)]
            session.move_history = [None]
            session.played_length = 0
            await manager.send_state(websocket)
        except ValueError:
            pass
        return True

    if action == Action.SAVE_GAME_STATE:
        save_game_state(payload)
        return True

    if action == Action.AI_STEP:
        board_2d = decode_board(np.uint64(u64(session.board_encoded)))
        session.ai_dispatcher.reset(board_2d, u64(session.board_encoded))
        best_move = session.ai_dispatcher.dispatcher()
        valid_moves = {"left", "right", "up", "down"}

        if best_move == "AI":
            from ai_and_sort import ai_core
            from egtb_core.AIPlayer import CoreAILogic

            player = ai_core.AIPlayer(u64(session.board_encoded))
            player.update_spawn_rate(SingletonConfig().config["4_spawn_rate"])
            logic = CoreAILogic()
            logic.time_limit_ratio = getattr(
                session.ai_dispatcher, "time_limit_ratio", 1.0
            )
            best_move_code = logic.calculate_step(
                player, board_2d, session.ai_dispatcher.counts
            )
            move_map = {1: "left", 2: "right", 3: "up", 4: "down"}
            best_move = move_map.get(best_move_code, None)
        else:
            best_move = best_move.lower() if best_move else None

        await websocket.send_json(
            {
                "action": Message.DO_AI_MOVE_CMD,
                "data": {"dir": best_move if best_move in valid_moves else None},
            }
        )
        return True

    return False
