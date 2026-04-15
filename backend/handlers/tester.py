from __future__ import annotations

from typing import Any

import numpy as np
from Config import SingletonConfig, category_info
from fastapi import WebSocket
from egtb_core.VBoardMover import s_gen_new_num as v_gen_new_num, s_move_board as v_move_board
from egtb_core.BoardMover import s_gen_new_num as r_gen_new_num, s_move_board as r_move_board

from ..actions import Action, Message
from ..animation import build_move_animation_metadata
from ..notebook import mistakes_book_store
from ..session import GameSession
from ..session import np_u64, u64
from ..tester import (
    TESTER_REPLAY_SENTINEL,
    _cache_tester_replay,
    _tester_append_log,
    _tester_append_summary,
    _tester_board_lines,
    _tester_compute_results,
    _tester_evaluation_of_performance,
    _tester_feedback_lines,
    _tester_format_rate_for_log,
    _tester_prepare_selection,
    _tester_random_rotate,
    _tester_record_step,
    _tester_reset_history,
    _tester_reset_last_step,
    _tester_reset_metrics,
    _tester_reset_record,
    _tester_restore_success_rate,
    _tester_start_practice,
    send_tester_state,
)


async def handle_tester_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
) -> bool:
    spawn_rate4 = float(SingletonConfig().config.get("4_spawn_rate", 0.1))

    if action == Action.TESTER_GET_INIT:
        session.tester_text_visible = bool(
            SingletonConfig().config.get("dis_text", True)
        )
        await websocket.send_json(
            {
                "action": Message.TESTER_BOOTSTRAP,
                "data": {
                    "categories": category_info,
                    "target_tiles": [2**i for i in range(6, 15)],
                    "settings": {
                        "colors": SingletonConfig().config.get("colors", []),
                        "dis_32k": SingletonConfig().config.get("dis_32k", False),
                        "dis_text": SingletonConfig().config.get("dis_text", True),
                        "language": SingletonConfig().config.get("language", "en"),
                    },
                },
            }
        )
        await send_tester_state(websocket, session)
        return True

    if action == Action.TESTER_SELECT_PATTERN:
        pattern = str(payload.get("pattern") or "")
        target = str(payload.get("target") or "")
        found, path_list = _tester_prepare_selection(session, pattern, target)
        _tester_reset_history(session, session.board_encoded, 0)
        _tester_reset_record(session)
        _tester_reset_metrics(session)
        session.tester_logs = []
        session.tester_results = {}
        session.tester_result_dtype = "?"
        session.tester_best_move = None
        session.tester_ready = False
        _tester_reset_last_step(session)

        if found and path_list:
            try:
                random_board = session.ensure_book_reader().get_random_state(
                    path_list, session.tester_full_pattern
                )
                random_board = _tester_random_rotate(random_board, pattern)
                _tester_start_practice(session, random_board, "We'll start from:")
            except Exception as e:
                session.tester_status = f"Failed to initialize board: {e}"
                session.tester_logs = [
                    f"Selected pattern: {session.tester_full_pattern}",
                    session.tester_status,
                ]
        else:
            session.tester_logs = [
                f"Selected pattern: {session.tester_full_pattern or '?'}",
                session.tester_status,
            ]

        await send_tester_state(websocket, session)
        return True

    if action == Action.TESTER_RESET_RANDOM:
        pattern = session.tester_pattern[0]
        target = session.tester_pattern[1]
        found, path_list = _tester_prepare_selection(session, pattern, target)
        if found and path_list:
            random_board = session.ensure_book_reader().get_random_state(
                path_list, session.tester_full_pattern
            )
            random_board = _tester_random_rotate(random_board, pattern)
            _tester_start_practice(session, random_board, "We'll start from:")
        else:
            _tester_reset_history(session, session.board_encoded, 0)
            _tester_reset_record(session)
            _tester_reset_metrics(session)
            session.tester_results = {}
            session.tester_result_dtype = "?"
            session.tester_best_move = None
            session.tester_ready = False
            _tester_reset_last_step(session)
            session.tester_logs = [
                f"Selected pattern: {session.tester_full_pattern or '?'}",
                session.tester_status,
            ]
        await send_tester_state(websocket, session)
        return True

    if action == Action.TESTER_SET_BOARD:
        hex_str = str(payload.get("hex_str") or "").strip()
        pattern = session.tester_pattern[0]
        target = session.tester_pattern[1]
        found, _ = _tester_prepare_selection(session, pattern, target)
        try:
            board_encoded = np_u64(int(hex_str, 16))
        except ValueError:
            session.tester_status = "Invalid hex board."
            await send_tester_state(websocket, session)
            return True

        if found:
            _tester_start_practice(session, board_encoded, "Manual board:")
            session.tester_status = (
                f"Manual board loaded for {session.tester_full_pattern}"
            )
        else:
            _tester_reset_history(session, board_encoded, 0)
            _tester_reset_record(session)
            _tester_reset_metrics(session)
            session.tester_ready = False
            session.tester_results = {}
            session.tester_result_dtype = "?"
            session.tester_best_move = None
            _tester_reset_last_step(session)
            session.tester_logs = [
                f"Selected pattern: {session.tester_full_pattern or '?'}",
                session.tester_status,
                *_tester_board_lines(session.board_encoded),
            ]
        await send_tester_state(websocket, session)
        return True

    if action == Action.TESTER_SET_TEXT_VISIBLE:
        visible = bool(payload.get("visible", True))
        session.tester_text_visible = visible
        config = SingletonConfig().config
        config["dis_text"] = visible
        SingletonConfig().save_config(config)
        await send_tester_state(websocket, session)
        return True

    if action == Action.TESTER_MOVE:
        direction_str = str(payload.get("dir") or "")
        direction_map = {"left": 1, "right": 2, "up": 3, "down": 4}
        if (
            not session.tester_ready
            or direction_str not in direction_map
            or not session.tester_results
        ):
            return True

        selected_rate = _tester_restore_success_rate(
            session.tester_results.get(direction_str),
            session.tester_result_dtype,
        )
        best_move = session.tester_best_move
        best_rate = _tester_restore_success_rate(
            session.tester_results.get(best_move),
            session.tester_result_dtype,
        )
        if selected_rate is None or best_move is None or best_rate is None:
            return True

        old_board_encoded = np_u64(session.board_encoded)

        move_fn = v_move_board if session.use_variant else r_move_board
        gen_fn = v_gen_new_num if session.use_variant else r_gen_new_num
        new_board, move_score = move_fn(
            old_board_encoded, direction_map[direction_str]
        )
        new_board = np.uint64(u64(new_board))
        if new_board == old_board_encoded:
            return True

        result_lines = []
        for key, value in session.tester_results.items():
            label = key[:1].upper()
            display = (
                "--"
                if value is None
                else _tester_format_rate_for_log(value, session.tester_result_dtype)
            )
            result_lines.append(f"{label}: {display}")
        result_lines.append("")

        previous_board_lines = _tester_board_lines(old_board_encoded)
        structured_result_lines = []
        for key in ("left", "right", "down", "up"):
            value = session.tester_results.get(key)
            label = key[:1].upper()
            display = (
                "--"
                if value is None
                else _tester_format_rate_for_log(value, session.tester_result_dtype)
            )
            structured_result_lines.append(f"{label}: {display}")

        evaluation = "Perfect!"
        loss = 1.0
        if abs(best_rate - selected_rate) <= 3e-10:
            session.tester_combo += 1
            session.tester_max_combo = max(
                session.tester_max_combo, session.tester_combo
            )
            session.tester_performance_stats["Perfect!"] += 1
            result_lines.append(f"Perfect! Combo: {session.tester_combo}x")
            result_lines.append(
                f"You pressed {direction_str.capitalize()}. And the best move is {best_move.capitalize()}."
            )
        else:
            session.tester_combo = 0
            loss = selected_rate / best_rate if best_rate > 0 else 1.0
            session.tester_goodness_of_fit *= loss
            evaluation = _tester_evaluation_of_performance(loss)
            session.tester_performance_stats[evaluation] += 1
            mistakes_book_store.add_mistake(
                session.tester_full_pattern,
                old_board_encoded,
                loss,
                best_move,
            )
            result_lines.append(evaluation)
            result_lines.append(
                f"one-step loss: {1 - loss:.4f}, goodness of fit: {session.tester_goodness_of_fit:.4f}"
            )
            result_lines.append(
                f"You pressed {direction_str.capitalize()}. But the best move is {best_move.capitalize()}."
            )

        feedback_lines = _tester_feedback_lines(
            evaluation,
            direction_str,
            best_move,
            session.tester_combo,
            loss,
            session.tester_goodness_of_fit,
        )
        session.tester_last_step = {
            "board_lines": previous_board_lines,
            "result_lines": structured_result_lines,
            "results": dict(session.tester_results),
            "dtype": session.tester_result_dtype,
            "message_lines": feedback_lines,
            "evaluation": evaluation,
            "direction": direction_str,
            "best_move": best_move,
            "loss": float(1 - loss),
            "goodness_of_fit": float(session.tester_goodness_of_fit),
        }

        session.score += int(move_score)
        session.best_score = max(session.best_score, session.score)
        new_board, _, num_pos_1d, val_exp = gen_fn(new_board, spawn_rate4)
        session.board_encoded = np_u64(new_board)
        session.history.append((session.board_encoded, session.score))
        session.move_history.append(direction_str)
        session.played_length = len(session.history) - 1
        _tester_record_step(session, direction_str, num_pos_1d, val_exp)
        _cache_tester_replay(session)

        _tester_append_log(session, result_lines)
        _tester_append_log(session, "--------------------------------------------------")
        _tester_compute_results(session)
        _tester_append_log(session, _tester_board_lines(session.board_encoded))
        _tester_append_log(session, "")

        next_best_rate = _tester_restore_success_rate(
            session.tester_results.get(session.tester_best_move),
            session.tester_result_dtype,
        )
        if session.tester_best_move is None:
            _tester_append_log(session, "Game Over: no possible moves left.")
            _tester_append_summary(session)
        elif next_best_rate is not None and next_best_rate >= 1 - 3e-10:
            _tester_append_log(
                session,
                "Congratulations! You're about to reach the target tile.",
            )
            _tester_append_summary(session)
        else:
            _tester_append_log(
                session,
                f"Total Goodness of Fit: {session.tester_goodness_of_fit:.4f}",
                f"Maximum Combo: {session.tester_max_combo}",
            )

        metadata = build_move_animation_metadata(
            direction_str,
            board_encoded=old_board_encoded,
            spawn_index=num_pos_1d,
            spawn_value=2**val_exp if val_exp > 0 else 0,
        )
        await send_tester_state(websocket, session, metadata)
        return True

    if action == Action.TESTER_SAVE_LOG:
        path = str(payload.get("path") or "").strip()
        if path:
            if not path.lower().endswith(".txt"):
                path += ".txt"
            with open(path, "w", encoding="utf-8") as file:
                file.write("\n".join(session.tester_logs))
            session.tester_status = f"Saved log to {path}"
            await send_tester_state(websocket, session)
        return True

    if action == Action.TESTER_SAVE_REPLAY:
        path = str(payload.get("path") or "").strip()
        if path and session.tester_step_count > 0:
            if not path.lower().endswith(".rpl"):
                path += ".rpl"
            replay = session.tester_record[: session.tester_step_count + 1].copy()
            replay[session.tester_step_count] = TESTER_REPLAY_SENTINEL
            replay.tofile(path)
            session.tester_status = f"Saved replay to {path}"
            await send_tester_state(websocket, session)
        return True

    return False
