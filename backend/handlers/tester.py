from __future__ import annotations

import base64
import time
from typing import Any

import numpy as np
from Config import SingletonConfig, category_info
from fastapi import WebSocket
from engine_core.VBoardMover import decode_board, encode_board, s_move_board as v_move_board
from engine_core.BoardMover import s_move_board as r_move_board
from engine_core.replay_utils import replay_sentinel, replay_step_goodness_ratio

from ..actions import Action, Message
from ..remote_workers.errors import RemoteTablebaseError
from ..remote_workers.registry import remote_worker_registry
from ..session import GameSession
from ..session import np_u64, safe_hex, u64
from ..tester import (
    PERFORMANCE_PERFECT_LABEL,
    _cache_tester_replay,
    _tester_append_log,
    _tester_board_lines,
    _tester_evaluation_of_performance,
    _tester_feedback_lines,
    _tester_format_rate_for_log,
    _tester_mark_lookup_pending,
    _tester_prepare_selection,
    _tester_random_rotate,
    _tester_record_step,
    _tester_reset_history,
    _tester_reset_last_step,
    _tester_reset_metrics,
    _tester_reset_record,
    _tester_restore_success_rate,
    _tester_start_practice,
    send_tester_move_accepted,
    send_tester_state,
    tester_replay_filename,
)


async def _tester_get_random_state(session: GameSession, path_list) -> int:
    if session.tester_tablebase_provider_kind == "remote":
        response = await remote_worker_registry.random_state(
            full_pattern=session.tester_full_pattern,
            pattern=str(session.tester_pattern[0]),
            target=str(session.tester_pattern[1]),
        )
        board = response.get("board") or response.get("board_hex")
        if not isinstance(board, (str, int)):
            raise RuntimeError("Remote tablebase returned an invalid random state.")
        return int(board, 16) if isinstance(board, str) else int(board)
    return int(
        session.ensure_book_reader().get_random_state(
            path_list, session.tester_full_pattern
        )
    )


def _client_revision(payload: dict[str, Any]) -> int | None:
    try:
        revision = int(payload.get("client_revision"))
    except (TypeError, ValueError):
        return None
    return revision if revision >= 0 else None


async def _send_tester_board_seed(
    websocket: WebSocket,
    session: GameSession,
    *,
    request_id: str,
    client_revision: int | None,
    board_encoded: int | None,
) -> None:
    ready = bool(session.tester_table_found and board_encoded is not None)
    encoded = np_u64(board_encoded or 0)
    logs = [f"Selected pattern: {session.tester_full_pattern or '?'}"]
    logs.append("We'll start from:" if ready else session.tester_status)
    await websocket.send_json(
        {
            "action": Message.TESTER_BOARD_SEED,
            "data": {
                "load_request_id": str(request_id or "")[:160],
                "client_revision": client_revision,
                "board": decode_board(encoded).flatten().tolist(),
                "hex_str": safe_hex(encoded),
                "pattern": session.tester_pattern[0],
                "target": session.tester_pattern[1],
                "full_pattern": session.tester_full_pattern,
                "ready": ready,
                "table_found": bool(session.tester_table_found),
                "status": session.tester_status,
                "use_variant": bool(session.use_variant),
                "logs": logs,
            },
        }
    )


async def _send_tester_tablebase_ready(
    websocket: WebSocket,
    session: GameSession,
    *,
    request_id: str,
) -> None:
    await websocket.send_json(
        {
            "action": Message.TESTER_TABLEBASE_READY,
            "data": {
                "request_id": str(request_id or "")[:160],
                "pattern": session.tester_pattern[0],
                "target": session.tester_pattern[1],
                "full_pattern": session.tester_full_pattern,
                "ready": bool(session.tester_table_found),
                "table_found": bool(session.tester_table_found),
                "status": session.tester_status,
                "use_variant": bool(session.use_variant),
            },
        }
    )


async def _start_requested_tablebase_query(
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
) -> None:
    query_id = str(payload.get("query_id") or "").strip()[:160]
    if not query_id or session.user_id is None:
        return
    from .tablebase_query import handle_tablebase_query_action

    await handle_tablebase_query_action(
        Action.TABLEBASE_QUERY,
        {
            "page": "tester",
            "query_id": query_id,
            "full_pattern": session.tester_full_pattern,
            "board_hex": f"{int(session.board_encoded):016x}",
            "prefetch_rng": payload.get("prefetch_rng"),
        },
        session,
        websocket,
    )


async def handle_tester_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
) -> bool:
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
        if not bool(payload.get("client_local_board")):
            await send_tester_state(websocket, session)
        return True

    if action == Action.TESTER_SELECT_PATTERN:
        pattern = str(payload.get("pattern") or "")
        target = str(payload.get("target") or "")
        client_local_board = bool(payload.get("client_local_board"))
        found, path_list = _tester_prepare_selection(
            session,
            pattern,
            target,
            reset_board=not client_local_board,
        )
        if client_local_board:
            if bool(payload.get("preserve_client_board")):
                await _send_tester_tablebase_ready(
                    websocket,
                    session,
                    request_id=str(payload.get("request_id") or ""),
                )
                return True
            random_board = None
            if found:
                try:
                    random_board = await _tester_get_random_state(session, path_list)
                    random_board = _tester_random_rotate(random_board, pattern)
                except Exception as exc:
                    if isinstance(exc, RemoteTablebaseError):
                        session.tester_table_found = False
                        session.tester_status = "The selected tablebase is temporarily unavailable."
                    else:
                        session.tester_status = "Failed to initialize the selected tablebase."
            await _send_tester_board_seed(
                websocket,
                session,
                request_id=str(payload.get("request_id") or ""),
                client_revision=_client_revision(payload),
                board_encoded=random_board,
            )
            return True

        _tester_reset_history(session, session.board_encoded, 0)
        _tester_reset_record(session)
        _tester_reset_metrics(session)
        session.tester_logs = []
        session.tester_results = {}
        session.tester_result_dtype = "?"
        session.tester_best_move = None
        session.tester_ready = False
        session.tester_lookup_pending = False
        _tester_reset_last_step(session)

        if found:
            try:
                random_board = await _tester_get_random_state(session, path_list)
                random_board = _tester_random_rotate(random_board, pattern)
                _tester_start_practice(
                    session,
                    random_board,
                    "We'll start from:",
                    compute_results=False,
                )
            except Exception as exc:
                if isinstance(exc, RemoteTablebaseError):
                    session.tester_table_found = False
                    session.tester_status = "The selected tablebase is temporarily unavailable."
                else:
                    session.tester_status = "Failed to initialize the selected tablebase."
                session.tester_logs = [
                    f"Selected pattern: {session.tester_full_pattern}",
                    session.tester_status,
                ]
        else:
            session.tester_logs = [
                f"Selected pattern: {session.tester_full_pattern or '?'}",
                session.tester_status,
            ]

        await send_tester_state(
            websocket,
            session,
            load_request_id=str(payload.get("request_id") or "")[:160],
        )
        return True

    if action == Action.TESTER_RESET_RANDOM:
        pattern = session.tester_pattern[0]
        target = session.tester_pattern[1]
        client_local_board = bool(payload.get("client_local_board"))
        found, path_list = _tester_prepare_selection(
            session,
            pattern,
            target,
            reset_board=not client_local_board,
        )
        if client_local_board:
            random_board = None
            if found:
                try:
                    random_board = await _tester_get_random_state(session, path_list)
                    random_board = _tester_random_rotate(random_board, pattern)
                except Exception as exc:
                    if isinstance(exc, RemoteTablebaseError):
                        session.tester_table_found = False
                        session.tester_status = "The selected tablebase is temporarily unavailable."
                    else:
                        session.tester_status = "Failed to initialize the selected tablebase."
            await _send_tester_board_seed(
                websocket,
                session,
                request_id=str(payload.get("request_id") or ""),
                client_revision=_client_revision(payload),
                board_encoded=random_board,
            )
            return True

        if found:
            try:
                random_board = await _tester_get_random_state(session, path_list)
                random_board = _tester_random_rotate(random_board, pattern)
                _tester_start_practice(
                    session,
                    random_board,
                    "We'll start from:",
                    compute_results=False,
                )
            except Exception as exc:
                if isinstance(exc, RemoteTablebaseError):
                    session.tester_table_found = False
                    session.tester_status = "The selected tablebase is temporarily unavailable."
                else:
                    session.tester_status = "Failed to initialize the selected tablebase."
                session.tester_logs = [
                    f"Selected pattern: {session.tester_full_pattern}",
                    session.tester_status,
                ]
        else:
            _tester_reset_history(session, session.board_encoded, 0)
            _tester_reset_record(session)
            _tester_reset_metrics(session)
            session.tester_results = {}
            session.tester_result_dtype = "?"
            session.tester_best_move = None
            session.tester_ready = False
            session.tester_lookup_pending = False
            _tester_reset_last_step(session)
            session.tester_logs = [
                f"Selected pattern: {session.tester_full_pattern or '?'}",
                session.tester_status,
            ]
        await send_tester_state(
            websocket,
            session,
            load_request_id=str(payload.get("request_id") or "")[:160],
        )
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
            _tester_start_practice(
                session,
                board_encoded,
                "Manual board:",
                compute_results=False,
            )
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
            session.tester_lookup_pending = False
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
        ):
            return True

        old_board_encoded = np_u64(session.board_encoded)
        logs_since = len(session.tester_logs)
        if (
            np_u64(getattr(session, "tester_results_board", 0)) != old_board_encoded
            or not session.tester_results
        ):
            from .tablebase_query import wait_for_tester_query_result

            await wait_for_tester_query_result(session)
        if (
            np_u64(getattr(session, "tester_results_board", 0)) != old_board_encoded
            or not session.tester_results
        ):
            await send_tester_state(websocket, session)
            return True
        post_lookup_context = getattr(session, "tester_post_lookup_context", None)
        if (
            isinstance(post_lookup_context, dict)
            and np_u64(post_lookup_context.get("board_encoded", 0)) == old_board_encoded
        ):
            _tester_append_post_lookup_logs(session)
            session.tester_post_lookup_context = None

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

        move_fn = v_move_board if session.use_variant else r_move_board
        moved_board, move_score = move_fn(
            old_board_encoded, direction_map[direction_str]
        )
        moved_board = np.uint64(u64(moved_board))
        if moved_board == old_board_encoded:
            await send_tester_state(websocket, session)
            return True

        try:
            from_board_encoded = np_u64(int(str(payload.get("from_board_hex") or ""), 16))
            client_board_encoded = np_u64(int(str(payload.get("board_hex") or ""), 16))
            spawn_index = int(payload.get("spawn_index"))
            spawn_value = int(payload.get("spawn_value"))
        except (TypeError, ValueError):
            await send_tester_state(websocket, session)
            return True

        if (
            from_board_encoded != old_board_encoded
            or spawn_index < 0
            or spawn_index >= 16
            or spawn_value not in (2, 4)
        ):
            await send_tester_state(websocket, session)
            return True

        moved_array = decode_board(moved_board).copy()
        spawn_row, spawn_col = divmod(spawn_index, 4)
        if int(moved_array[spawn_row, spawn_col]) != 0:
            await send_tester_state(websocket, session)
            return True
        moved_array[spawn_row, spawn_col] = spawn_value
        validated_board = np_u64(encode_board(moved_array))
        if validated_board != client_board_encoded:
            await send_tester_state(websocket, session)
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

        evaluation = PERFORMANCE_PERFECT_LABEL
        loss = replay_step_goodness_ratio(selected_rate, best_rate)
        if loss == 1.0:
            session.tester_combo += 1
            session.tester_max_combo = max(
                session.tester_max_combo, session.tester_combo
            )
            session.tester_performance_stats[PERFORMANCE_PERFECT_LABEL] += 1
            result_lines.append(
                f"{PERFORMANCE_PERFECT_LABEL} Combo: {session.tester_combo}x"
            )
            result_lines.append(
                f"You pressed {direction_str.capitalize()}. And the best move is {best_move.capitalize()}."
            )
        else:
            session.tester_combo = 0
            session.tester_goodness_of_fit *= loss
            evaluation = _tester_evaluation_of_performance(loss)
            session.tester_performance_stats[evaluation] += 1
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
        session.board_encoded = validated_board
        session.history.append((session.board_encoded, session.score))
        session.move_history.append(direction_str)
        session.played_length = len(session.history) - 1
        _tester_record_step(
            session,
            direction_str,
            spawn_index,
            2 if spawn_value == 4 else 1,
        )
        _cache_tester_replay(session)

        _tester_append_log(session, result_lines)
        _tester_append_log(session, "--------------------------------------------------")
        _tester_append_log(session, _tester_board_lines(session.board_encoded))
        _tester_append_log(session, "")
        session.tester_post_lookup_context = {
            "board_encoded": u64(session.board_encoded),
            "logs_since": len(session.tester_logs),
        }
        _tester_mark_lookup_pending(session)

        await send_tester_move_accepted(
            websocket,
            session,
            session.board_encoded,
            logs_since=logs_since,
        )
        await _start_requested_tablebase_query(payload, session, websocket)
        return True

    if action == Action.TESTER_EXPORT_LOG:
        filename = f"tester_log_{int(time.time())}.txt"
        await websocket.send_json(
            {
                "action": Action.TESTER_EXPORT_LOG,
                "data": {
                    "filename": filename,
                    "mime": "text/plain;charset=utf-8",
                    "text": "\n".join(session.tester_logs),
                },
            }
        )
        return True

    if action == Action.TESTER_EXPORT_REPLAY:
        if session.tester_step_count > 0:
            replay = session.tester_record[: session.tester_step_count + 1].copy()
            replay[session.tester_step_count] = replay_sentinel(session.board_encoded)
            await websocket.send_json(
                {
                    "action": Action.TESTER_EXPORT_REPLAY,
                    "data": {
                        "filename": tester_replay_filename(
                            session.tester_full_pattern,
                            session.tester_goodness_of_fit,
                        ),
                        "mime": "application/octet-stream",
                        "base64": base64.b64encode(replay.tobytes()).decode("ascii"),
                    },
                }
            )
        return True

    return False
