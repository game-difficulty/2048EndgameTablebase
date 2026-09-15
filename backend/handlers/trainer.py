from __future__ import annotations

import asyncio
import os
from typing import Any

import numpy as np
from Config import SingletonConfig, category_info
from engine_core.Calculator import ReverseLR, ReverseUD, RotateL, RotateR
from fastapi import WebSocket
from engine_core.VBoardMover import (
    decode_board,
    encode_board,
    s_gen_new_num as v_gen_new_num,
    s_move_board as v_move_board,
)
from engine_core.BoardMover import s_gen_new_num as r_gen_new_num, s_move_board as r_move_board

from ..actions import Action, EventType, Message
from ..animation import build_move_animation_metadata
from ..session import GameSession
from ..session import np_u64, safe_hex, u64
from ..state import ConnectionManager
from ..remote_workers.errors import RemoteTablebaseError
from ..remote_workers.registry import remote_worker_registry
from ..tablebase_catalog import (
    build_filepath_map_entry,
    is_guest_tablebase_available,
    resolve_configured_tablebase,
)
from ..trainer_helpers import (
    _clear_record_replay,
    compute_spawns_async,
    _decode_record_rates,
    _record_state,
    send_trainer_results,
)
from ..webview_api import Api
from ..trainer_default_lookup import default_lookup_for


def _clear_trainer_results(session: GameSession) -> None:
    session.trainer_results = {}
    session.trainer_results_board = np_u64(0)


async def _get_random_trainer_board(session: GameSession, path_list) -> int:
    if session.tablebase_provider_kind == "remote":
        response = await remote_worker_registry.random_state(
            full_pattern=session.current_pattern,
            pattern=str(session.pattern_settings[0]),
            target=str(session.pattern_settings[1]),
        )
        random_board = response.get("board") or response.get("board_hex")
        if not isinstance(random_board, (str, int)):
            raise RuntimeError("Remote tablebase returned an invalid random state.")
        random_board = int(random_board, 16) if isinstance(random_board, str) else random_board
    else:
        random_board = session.ensure_book_reader().get_random_state(
            path_list, session.current_pattern
        )
    return int(random_board)


def _client_revision(payload: dict[str, Any]) -> int | None:
    try:
        revision = int(payload.get("client_revision"))
    except (TypeError, ValueError):
        return None
    return revision if revision >= 0 else None


async def _set_random_trainer_board(session: GameSession, path_list) -> None:
    random_board = await _get_random_trainer_board(session, path_list)
    session.board_encoded = np_u64(random_board)
    session.score = 0
    session.history = [(session.board_encoded, session.score)]
    session.move_history = [None]
    session.moved = 0
    session.played_length = 0
    _clear_trainer_results(session)


async def _send_trainer_tablebase_ready(
    websocket: WebSocket,
    session: GameSession,
    *,
    request_id: str,
    client_revision: int | None = None,
    board_encoded: int | None = None,
    default_switch: bool | None = None,
) -> None:
    data = {
        "request_id": str(request_id or "")[:160],
        "tablebase_status": session.tablebase_status,
        "tablebase_full_pattern": session.current_pattern,
        "tablebase_dtype": session.success_rate_dtype,
        "use_variant": bool(session.use_variant),
    }
    if client_revision is not None:
        data["client_revision"] = int(client_revision)
    if board_encoded is not None:
        data["board_hex"] = safe_hex(np_u64(board_encoded))
        if default_switch is not None:
            data["default_query_ticket"] = default_lookup_for(session).issue(
                session.current_pattern, board_encoded, switched=default_switch,
            )
    await websocket.send_json(
        {"action": Message.TRAINER_TABLEBASE_READY, "data": data}
    )


async def _start_requested_tablebase_query(
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
) -> None:
    query_id = str(payload.get("query_id") or "").strip()[:160]
    if (
        not query_id
        or not session.current_pattern
        or (session.user_id is None and not getattr(session, "guest_id", None))
        or session.moved == 1
    ):
        return
    from .tablebase_query import handle_tablebase_query_action

    await handle_tablebase_query_action(
        Action.TABLEBASE_QUERY,
        {
            "page": "trainer",
            "query_id": query_id,
            "full_pattern": session.current_pattern,
            "board_hex": f"{int(session.board_encoded):016x}",
            "prefetch_rng": payload.get("prefetch_rng"),
        },
        session,
        websocket,
    )


async def _send_trainer_move_accepted(
    websocket: WebSocket,
    session: GameSession,
    *,
    move_seq: int,
    query_pending: bool,
) -> None:
    await websocket.send_json(
        {
            "action": Message.TRAINER_MOVE_ACCEPTED,
            "data": {
                "move_seq": int(move_seq),
                "board_hex": safe_hex(session.board_encoded),
                "score": {
                    "current": int(session.score),
                    "best": int(session.best_score),
                },
                "record_step": int(session.played_length),
                "record_max": len(session.history),
                "recording_length": int(session.record_length),
                "query_pending": bool(query_pending),
            },
        }
    )


async def handle_trainer_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
    manager: ConnectionManager,
) -> bool:
    spawn_rate4 = float(SingletonConfig().config.get("4_spawn_rate", 0.1))

    if action == Action.TRAINER_SET_EMPTY_PATTERN:
        client_local_board = bool(payload.get("client_local_board"))
        if client_local_board:
            session.current_pattern = ""
            session.pattern_settings = ["", ""]
            session.use_variant = False
            session.tablebase_provider_kind = ""
            session.tablebase_status = "not_selected"
            session.success_rate_dtype = "?"
            await _send_trainer_tablebase_ready(
                websocket,
                session,
                request_id=str(payload.get("request_id") or ""),
                client_revision=_client_revision(payload),
            )
            return True
        current_board = np_u64(session.board_encoded)
        current_score = int(session.score)
        query_handle = getattr(session, "trainer_query_handle", None)
        if query_handle is not None:
            query_handle.cancel()
        session.trainer_query_handle = None
        session.trainer_query_task = None
        _clear_record_replay(session)
        session.board_encoded = current_board
        session.score = current_score
        session.history = [(session.board_encoded, session.score)]
        session.move_history = [None]
        session.played_length = 0
        session.moved = 0
        session.current_pattern = ""
        session.pattern_settings = ["", ""]
        session.use_variant = False
        session.tablebase_provider_kind = ""
        session.tablebase_status = "not_selected"
        session.success_rate_dtype = "?"
        if session.spawn_mode in (1, 2):
            session.spawn_mode = 0
        _clear_trainer_results(session)
        await manager.send_state(websocket)
        return True

    if action == Action.TRAINER_SET_FILEPATH:
        pattern = str(payload.get("pattern", "L3")).strip()
        target = payload.get("target", "32768")
        full_pattern = pattern if "_" in pattern else f"{pattern}_{target}"
        if (
            session.user_id is None
            and getattr(session, "guest_id", None)
            and not is_guest_tablebase_available(full_pattern)
        ):
            await websocket.send_json(
                {
                    "action": Message.TRAINER_TABLEBASE_READY,
                    "data": {
                        "request_id": str(payload.get("request_id") or "")[:160],
                        "tablebase_status": "guest_locked",
                        "tablebase_full_pattern": full_pattern,
                        "code": "GUEST_TABLE_LOGIN_REQUIRED",
                    },
                }
            )
            return True
        base_pattern = full_pattern.split("_")[0]
        client_local_board = bool(payload.get("client_local_board"))
        if not client_local_board:
            current_board = np_u64(session.board_encoded)
            current_score = int(session.score)
            _clear_record_replay(session)
            session.board_encoded = current_board
            session.score = current_score
            session.history = [(session.board_encoded, session.score)]
            session.move_history = [None]
            session.played_length = 0
            session.moved = 0
            _clear_trainer_results(session)
        descriptor = resolve_configured_tablebase(full_pattern, spawn_rate4)
        provider_kind = (
            str(descriptor.get("_provider")) if descriptor else "unavailable"
        )
        path_list = build_filepath_map_entry(full_pattern, spawn_rate4)
        if provider_kind == "local":
            session.ensure_book_reader().dispatch(path_list, base_pattern, target)
        session.current_pattern = full_pattern
        session.pattern_settings = [base_pattern, target]
        session.use_variant = base_pattern in category_info.get("variant", [])
        session.tablebase_provider_kind = provider_kind
        session.success_rate_dtype = str(
            descriptor.get("dtype") if descriptor else "?"
        )
        session.tablebase_status = (
            "loaded"
            if descriptor
            and (
                provider_kind == "local"
                or bool(descriptor.get("_available", False))
            )
            else "temporarily_unavailable"
            if provider_kind == "remote"
            else "not_found"
        )

        client_board_seed = None
        if payload.get("load_default") and session.tablebase_status == "loaded":
            try:
                if client_local_board:
                    client_board_seed = await _get_random_trainer_board(session, path_list)
                else:
                    await _set_random_trainer_board(session, path_list)
            except RemoteTablebaseError:
                session.tablebase_status = "temporarily_unavailable"
            except Exception:
                session.tablebase_status = "not_found"

        if client_local_board:
            await _send_trainer_tablebase_ready(
                websocket,
                session,
                request_id=str(payload.get("request_id") or ""),
                client_revision=_client_revision(payload),
                board_encoded=client_board_seed,
                default_switch=True if payload.get("load_default") else None,
            )
        else:
            await manager.send_state(websocket)
        return True

    if action == Action.TRAINER_GET_RESULTS:
        await send_trainer_results(
            session, websocket, request_id=payload.get("request_id")
        )
        return True

    if action == Action.SET_SPAWN_MODE:
        try:
            next_mode = int(payload.get("mode", 0))
        except (TypeError, ValueError):
            next_mode = 0
        if next_mode not in (0, 1, 2, 3):
            next_mode = 0
        if session.user_id is None and getattr(session, "guest_id", None) and next_mode in (1, 2):
            next_mode = 0
        session.spawn_mode = next_mode
        session.moved = 0
        await manager.send_state(websocket)
        return True

    if action == Action.TRAINER_SPAWN_QUERY:
        request_id = str(payload.get("request_id") or "")[:160]
        try:
            revision = int(payload.get("revision"))
            moved_board = np_u64(int(str(payload.get("board_hex") or ""), 16))
            mode = int(payload.get("mode"))
        except (TypeError, ValueError):
            revision = -1
            moved_board = np_u64(0)
            mode = -1
        data = {
            "request_id": request_id,
            "revision": revision,
            "board_hex": safe_hex(moved_board),
            "found": False,
        }
        if (
            request_id
            and revision >= 0
            and mode in (1, 2)
            and session.current_pattern
            and str(payload.get("full_pattern") or "") == session.current_pattern
        ):
            try:
                spawns = await compute_spawns_async(session, moved_board)
                if spawns:
                    pos, val_exp = (
                        max(spawns, key=spawns.get)
                        if mode == 1
                        else min(spawns, key=spawns.get)
                    )
                    data.update(
                        {
                            "found": True,
                            "index": int(pos),
                            "value": int(2**val_exp),
                        }
                    )
            except RemoteTablebaseError as exc:
                data.update(exc.payload)
            except Exception:
                data["code"] = "TRAINER_SPAWN_QUERY_FAILED"
        await websocket.send_json(
            {"action": Message.TRAINER_SPAWN_RESULT, "data": data}
        )
        return True

    if action == Action.TRAINER_STEP:
        if (
            session.trainer_results
            and np_u64(getattr(session, "trainer_results_board", 0))
            == np_u64(session.board_encoded)
        ):
            move = list(session.trainer_results.keys())[0]
            val = session.trainer_results.get(move)
            if isinstance(val, (int, float)) and val:
                await websocket.send_json(
                    {"action": Message.DO_AI_MOVE_CMD, "data": {"dir": move}}
                )
            else:
                await websocket.send_json(
                    {"action": Message.TRAINER_STEP_FAILED, "data": {}}
                )
        else:
            await websocket.send_json({"action": Message.TRAINER_STEP_FAILED, "data": {}})
        return True

    if action == Action.TRAINER_MOVE:
        direction_str = str(payload.get("dir") or "")
        direction_map = {"left": 1, "right": 2, "up": 3, "down": 4}
        if direction_str not in direction_map:
            return True
        client_optimistic = bool(payload.get("client_optimistic"))
        move_seq = None
        if client_optimistic and payload.get("move_seq") is not None:
            try:
                move_seq = int(payload.get("move_seq"))
            except (TypeError, ValueError):
                await manager.send_state(websocket)
                return True
            expected_move_seq = int(getattr(session, "trainer_move_seq", 0)) + 1
            if move_seq != expected_move_seq:
                await manager.send_state(websocket)
                return True

            from .tablebase_query import wait_for_trainer_query_result

            await wait_for_trainer_query_result(session)
        _clear_record_replay(session)

        if session.spawn_mode == 3 and session.moved == 1:
            return True

        direction = direction_map[direction_str]
        old_board_encoded = np_u64(session.board_encoded)

        move_fn = v_move_board if session.use_variant else r_move_board
        new_board, move_score = move_fn(old_board_encoded, direction)
        new_board = np_u64(new_board)

        if new_board == old_board_encoded:
            return True

        client_board_encoded = None
        if client_optimistic:
            try:
                from_board_encoded = np_u64(
                    int(str(payload.get("from_board_hex") or ""), 16)
                )
                client_board_encoded = np_u64(
                    int(str(payload.get("board_hex") or ""), 16)
                )
            except (TypeError, ValueError):
                await manager.send_state(websocket)
                return True
            if from_board_encoded != old_board_encoded:
                await manager.send_state(websocket)
                return True

        next_score = int(session.score) + int(move_score)

        num_pos_1d, val_exp = -1, 0
        if session.spawn_mode == 0:
            session.moved = 0
            if client_optimistic:
                try:
                    num_pos_1d = int(payload.get("spawn_index"))
                    spawn_value = int(payload.get("spawn_value"))
                except (TypeError, ValueError):
                    await manager.send_state(websocket)
                    return True
                if not 0 <= num_pos_1d < 16 or spawn_value not in (2, 4):
                    await manager.send_state(websocket)
                    return True
                moved_array = decode_board(new_board).copy()
                spawn_row, spawn_col = divmod(num_pos_1d, 4)
                if int(moved_array[spawn_row, spawn_col]) != 0:
                    await manager.send_state(websocket)
                    return True
                moved_array[spawn_row, spawn_col] = spawn_value
                validated_board = np_u64(encode_board(moved_array))
                if validated_board != client_board_encoded:
                    await manager.send_state(websocket)
                    return True
                new_board = validated_board
                val_exp = 1 if spawn_value == 2 else 2
            else:
                gen_fn = v_gen_new_num if session.use_variant else r_gen_new_num
                new_board, _, num_pos_1d, val_exp = gen_fn(new_board, spawn_rate4)
                new_board = np_u64(new_board)
        elif session.spawn_mode == 3:
            if client_optimistic and client_board_encoded != new_board:
                await manager.send_state(websocket)
                return True
            session.moved = 1
        elif session.spawn_mode in (1, 2):
            if client_optimistic and client_board_encoded != new_board:
                await manager.send_state(websocket)
                return True
            session.moved = 0
            try:
                spawns = await compute_spawns_async(session, new_board)
            except RemoteTablebaseError as exc:
                session.tablebase_status = "temporarily_unavailable"
                await manager.send_state(websocket)
                await websocket.send_json(
                    {
                        "action": Message.TABLEBASE_QUERY_RESULT,
                        "data": {
                            "page": "trainer",
                            "query_id": str(payload.get("query_id") or "")[:160],
                            "full_pattern": session.current_pattern,
                            "board_hex": format(int(old_board_encoded), "016x"),
                            "results": {},
                            "dtype": "?",
                            **exc.payload,
                        },
                    }
                )
                return True
            if spawns:
                key = (
                    max(spawns, key=spawns.get)
                    if session.spawn_mode == 1
                    else min(spawns, key=spawns.get)
                )
                pos, val = key
                p = 15 - pos
                new_board = np_u64(new_board | (np.uint64(val) << np.uint64(4 * p)))
                num_pos_1d = pos
                val_exp = val
            else:
                gen_fn = v_gen_new_num if session.use_variant else r_gen_new_num
                new_board, _, num_pos_1d, val_exp = gen_fn(new_board, spawn_rate4)
                new_board = np_u64(new_board)

        session.score = next_score
        if session.score > session.best_score:
            session.best_score = session.score
        session.board_encoded = np_u64(new_board)
        _clear_trainer_results(session)
        session.history.append((session.board_encoded, session.score))
        session.move_history.append(direction_str)
        session.played_length = len(session.history) - 1
        if move_seq is not None:
            session.trainer_move_seq = move_seq

        if client_optimistic:
            metadata = (
                {
                    "appear_tile": {
                        "index": num_pos_1d,
                        "value": 2**val_exp,
                    }
                }
                if session.spawn_mode in (1, 2) and val_exp > 0
                else {}
            )
        else:
            metadata = build_move_animation_metadata(
                direction_str,
                board_encoded=old_board_encoded,
                use_variant=session.use_variant,
                spawn_index=num_pos_1d,
                spawn_value=2**val_exp if val_exp > 0 else 0,
            )
        if session.recording_state:
            _record_state(session, direction_str, num_pos_1d, val_exp)

        if move_seq is not None and session.spawn_mode == 0:
            await _send_trainer_move_accepted(
                websocket,
                session,
                move_seq=move_seq,
                query_pending=bool(str(payload.get("query_id") or "").strip()),
            )
        else:
            await manager.send_state(websocket, metadata)
        await _start_requested_tablebase_query(payload, session, websocket)
        return True

    if action == Action.TRAINER_DEFAULT:
        if bool(payload.get("client_local_board")):
            board_seed = None
            path_list = build_filepath_map_entry(session.current_pattern, spawn_rate4)
            if session.tablebase_provider_kind == "remote" or path_list:
                try:
                    board_seed = await _get_random_trainer_board(session, path_list)
                    session.tablebase_status = "loaded"
                except RemoteTablebaseError:
                    session.tablebase_status = "temporarily_unavailable"
                except Exception:
                    session.tablebase_status = "not_found"
            await _send_trainer_tablebase_ready(
                websocket,
                session,
                request_id=str(payload.get("request_id") or ""),
                client_revision=_client_revision(payload),
                board_encoded=board_seed,
                default_switch=False,
            )
            return True
        _clear_record_replay(session)
        path_list = build_filepath_map_entry(session.current_pattern, spawn_rate4)
        if session.tablebase_provider_kind == "remote" or path_list:
            try:
                await _set_random_trainer_board(session, path_list)
                session.tablebase_status = "loaded"
                await manager.send_state(websocket)
            except RemoteTablebaseError as exc:
                session.tablebase_status = "temporarily_unavailable"
                await manager.send_state(websocket)
                await websocket.send_json(
                    {
                        "action": Message.TABLEBASE_QUERY_RESULT,
                        "data": {
                            "page": "trainer",
                            "full_pattern": session.current_pattern,
                            "results": {},
                            "dtype": "?",
                            **exc.payload,
                        },
                    }
                )
            except Exception:
                session.tablebase_status = "not_found"
                await manager.send_state(websocket)
        return True

    if action == Action.TRAINER_MANUAL_SPAWN:
        if session.spawn_mode == 3 and session.moved == 1:
            _clear_record_replay(session)
            try:
                row = int(payload.get("row", 0))
                col = int(payload.get("col", 0))
                val = int(payload.get("val", 2))
            except (TypeError, ValueError):
                await manager.send_state(websocket)
                return True
            if not 0 <= row < 4 or not 0 <= col < 4 or val not in (2, 4):
                await manager.send_state(websocket)
                return True

            board_2d = decode_board(np.uint64(u64(session.board_encoded)))
            if board_2d[row, col] == 0:
                board_2d[row, col] = val
                next_board_encoded = np_u64(encode_board(board_2d))
                client_optimistic = bool(payload.get("client_optimistic"))
                if client_optimistic:
                    try:
                        from_board_encoded = np_u64(
                            int(str(payload.get("from_board_hex") or ""), 16)
                        )
                        client_board_encoded = np_u64(
                            int(str(payload.get("board_hex") or ""), 16)
                        )
                    except (TypeError, ValueError):
                        await manager.send_state(websocket)
                        return True
                    if (
                        from_board_encoded != session.board_encoded
                        or client_board_encoded != next_board_encoded
                    ):
                        await manager.send_state(websocket)
                        return True
                session.board_encoded = next_board_encoded
                _clear_trainer_results(session)
                session.moved = 0
                session.history.append((session.board_encoded, session.score))
                session.move_history.append("spawn")
                session.played_length = len(session.history) - 1

                num_pos_1d = row * 4 + col
                metadata = (
                    {}
                    if client_optimistic
                    else {"appear_tile": {"index": num_pos_1d, "value": val}}
                )
                await manager.send_state(websocket, metadata)
                await _start_requested_tablebase_query(payload, session, websocket)
            else:
                await manager.send_state(websocket)
        return True

    if action == Action.SET_BOARD:
        hex_str = str(payload.get("hex_str") or "").strip()
        try:
            board_encoded = np_u64(int(hex_str, 16))
        except ValueError:
            await manager.send_state(websocket)
            return True

        _clear_record_replay(session)
        session.board_encoded = board_encoded
        session.score = 0
        session.moved = 0
        _clear_trainer_results(session)
        session.history = [(session.board_encoded, session.score)]
        session.move_history = [None]
        session.played_length = 0
        if bool(payload.get("client_optimistic")):
            await websocket.send_json(
                {
                    "action": Message.TRAINER_BOARD_SYNCED,
                    "data": {
                        "board_hex": f"{int(session.board_encoded):016x}",
                        "edit_source": str(payload.get("edit_source") or "")[:32],
                    },
                }
            )
        else:
            await manager.send_state(websocket)
        await _start_requested_tablebase_query(payload, session, websocket)
        return True

    if action == Action.UNDO:
        _clear_record_replay(session)
        history = getattr(session, "history", [])
        move_history = getattr(session, "move_history", [])
        if len(history) <= 1:
            await manager.send_state(websocket)
            return True

        current_board = np_u64(session.board_encoded)

        def pop_last_state():
            if len(session.history) > 1:
                session.history.pop()
            if len(session.move_history) > 1:
                session.move_history.pop()

        pop_last_state()
        while len(session.history) > 1 and np_u64(session.history[-1][0]) == current_board:
            pop_last_state()

        session.board_encoded, session.score = session.history[-1]
        session.board_encoded = np_u64(session.board_encoded)
        session.played_length = len(session.history) - 1
        _clear_trainer_results(session)
        last_move = session.move_history[-1] if session.move_history else None
        session.moved = (
            1
            if session.spawn_mode == 3 and last_move not in (None, "spawn")
            else 0
        )
        client_optimistic = bool(payload.get("client_optimistic"))
        expected_board = None
        if client_optimistic:
            try:
                expected_board = np_u64(
                    int(str(payload.get("expected_board_hex") or ""), 16)
                )
            except (TypeError, ValueError):
                expected_board = None
        if client_optimistic and expected_board == session.board_encoded:
            await websocket.send_json(
                {
                    "action": Message.TRAINER_BOARD_SYNCED,
                    "data": {
                        "board_hex": f"{int(session.board_encoded):016x}",
                        "edit_source": "undo",
                    },
                }
            )
        else:
            await manager.send_state(websocket)
        return True

    if action == Action.SET_CELL:
        row = payload.get("row", 0)
        col = payload.get("col", 0)
        val = payload.get("val", 0)
        _clear_record_replay(session)

        board_2d = decode_board(np.uint64(u64(session.board_encoded)))
        board_2d[row, col] = int(val)

        session.board_encoded = np_u64(encode_board(board_2d))
        _clear_trainer_results(session)
        session.history = [(session.board_encoded, session.score)]
        session.move_history = [None]
        session.played_length = 0
        await manager.send_state(websocket)
        return True

    if action == Action.ROTATE:
        rotate_func = {
            "UD": ReverseUD,
            "LR": ReverseLR,
            "RL": ReverseLR,
            "R90": RotateR,
            "L90": RotateL,
        }.get(payload.get("type"))
        if rotate_func:
            if session.client_id.startswith("trainer_"):
                _clear_record_replay(session)
            session.board_encoded = np_u64(
                rotate_func(np_u64(session.board_encoded))
            )
            session.score = 0
            session.moved = 0
            _clear_trainer_results(session)
            session.history = [(session.board_encoded, session.score)]
            session.move_history = [None]
            session.played_length = 0
            await manager.send_state(websocket)
        return True

    if action == Action.TRIGGER_RECORD_OPEN:
        path = await asyncio.to_thread(Api().select_open_record)
        if not path:
            return True
        payload = dict(payload)
        payload["path"] = path
        action = Action.RECORD_OPEN

    if action == Action.TRIGGER_RECORD_SAVE:
        path = await asyncio.to_thread(Api().select_save_record)
        if not path:
            return True
        payload = dict(payload)
        payload["path"] = path
        action = Action.RECORD_SAVE

    if action == Action.RECORD_OPEN:
        path = payload.get("path")
        if path and os.path.exists(path):
            _clear_record_replay(session)
            _clear_trainer_results(session)
            session.moved = 0
            filesize = os.path.getsize(path)
            if filesize % 17 == 0:
                dt = np.dtype([("changes", np.uint8), ("rates", np.uint32, (4,))])
                arr = np.fromfile(path, dtype=dt)
                if len(arr) > 0:
                    v_pieces = arr[0]["rates"]
                    init_board = (
                        np.uint64(v_pieces[0])
                        | (np.uint64(v_pieces[1]) << 16)
                        | (np.uint64(v_pieces[2]) << 32)
                        | (np.uint64(v_pieces[3]) << 48)
                    )

                    current_board = np_u64(init_board)
                    current_score = 0
                    session.history = [(current_board, current_score)]
                    session.move_history = [None]
                    session.record_animation_history = [{}]

                    move_inv = {0: 3, 1: 4, 2: 1, 3: 2}
                    move_fn = v_move_board if session.use_variant else r_move_board

                    for i in range(1, len(arr)):
                        c = arr[i]["changes"]
                        move_bits = c & 0b11
                        pos = (c >> 2) & 0b1111
                        val_exp = ((c >> 6) & 0b1) + 1
                        move_label = {
                            0: "up",
                            1: "down",
                            2: "left",
                            3: "right",
                        }.get(move_bits, "up")
                        nb, move_score = move_fn(current_board, move_inv.get(move_bits, 3))
                        current_score += int(move_score)

                        p = 15 - pos
                        nb = np_u64(nb | (np.uint64(val_exp) << np.uint64(4 * p)))
                        current_board = nb
                        session.history.append((current_board, current_score))
                        session.move_history.append(move_label)
                        session.record_animation_history.append(
                            build_move_animation_metadata(
                                move_label,
                                board_encoded=session.history[-2][0],
                                use_variant=session.use_variant,
                                spawn_index=pos,
                                spawn_value=2**val_exp,
                            )
                        )

                    session.record_result_history = [
                        _decode_record_rates(arr[i + 1]["rates"])
                        if i + 1 < len(arr)
                        else None
                        for i in range(len(session.history))
                    ]
                    session.record_result_dtype = "recorded"
                    session.record_playback_loaded = True
                    session.played_length = 0
                    session.board_encoded, session.score = session.history[0]
                    await manager.send_state(websocket)
                    await websocket.send_json(
                        {
                            "type": EventType.RECORD_OPEN,
                            "success": True,
                            "path": path,
                            "total": len(session.history),
                        }
                    )
            else:
                dt = np.dtype([("board", np.uint64), ("score", np.uint32)])
                arr = np.fromfile(path, dtype=dt)
                if len(arr) > 0:
                    session.history = [(row["board"], row["score"]) for row in arr]
                    session.move_history = [None] * len(session.history)
                    session.record_animation_history = [{}] * len(session.history)
                    session.record_playback_loaded = True
                    session.played_length = 0
                    session.board_encoded = np_u64(session.history[0][0])
                    session.score = session.history[0][1]
                    await manager.send_state(websocket)
        return True

    if action == Action.START_RECORDING:
        _clear_record_replay(session)
        session.recording_state = True
        session.records = np.zeros(
            10000, dtype=[("changes", np.uint8), ("rates", np.uint32, (4,))]
        )
        v = np.uint64(u64(session.board_encoded))
        session.records[0] = (
            0,
            [
                np.uint32(v & 0xFFFF),
                np.uint32((v >> 16) & 0xFFFF),
                np.uint32((v >> 32) & 0xFFFF),
                np.uint32((v >> 48) & 0xFFFF),
            ],
        )
        session.record_length = 1
        await websocket.send_json(
            {
                "action": Message.RECORDING_STARTED,
                "data": {"recording_length": session.record_length},
            }
        )
        return True

    if action in (Action.STOP_RECORDING, Action.RECORD_SAVE):
        path = payload.get("path")
        if path and session.recording_state and session.record_length > 2:
            session.records[: session.record_length].tofile(path)
        session.recording_state = False
        session.record_length = 0
        session.records = np.empty(
            0, dtype=[("changes", np.uint8), ("rates", np.uint32, (4,))]
        )
        await websocket.send_json(
            {
                "action": Message.RECORDING_STOPPED,
                "data": {"recording_length": session.record_length},
            }
        )
        return True

    if action == Action.PREPARE_STOP_RECORDING:
        if session.record_length < 2:
            session.recording_state = False
            session.record_length = 0
            session.records = np.empty(
                0, dtype=[("changes", np.uint8), ("rates", np.uint32, (4,))]
            )
            await websocket.send_json(
                {
                    "action": Message.RECORDING_STOPPED,
                    "data": {"recording_length": session.record_length},
                }
            )
        else:
            await websocket.send_json({"action": Message.RECORD_SAVE_REQUIRED})
        return True

    if action == Action.RECORD_STEP:
        step = payload.get("step") or payload.get("dir")
        idx = payload.get("index")

        if not hasattr(session, "played_length"):
            session.played_length = 0
        old_idx = session.played_length

        new_idx = int(idx) if idx is not None else session.played_length + (
            int(step) if step is not None else 1
        )

        if 0 <= new_idx < len(session.history):
            session.played_length = new_idx
            session.board_encoded, session.score = session.history[new_idx]
            metadata = {}
            record_animation_history = getattr(session, "record_animation_history", [])
            if new_idx == old_idx + 1 and new_idx < len(record_animation_history):
                metadata = record_animation_history[new_idx] or {}
            await manager.send_state(websocket, metadata)
        return True

    if action == Action.TRIGGER_SELECT_FOLDER:
        path = await asyncio.to_thread(Api().select_folder)
        await websocket.send_json(
            {"action": Message.FOLDER_SELECTED, "data": {"path": path}}
        )
        return True

    return False
