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
from ..session import np_u64, u64
from ..state import ConnectionManager
from ..trainer_helpers import (
    _clear_record_replay,
    _compute_spawns,
    _decode_record_rates,
    _record_state,
    send_trainer_results,
)
from ..webview_api import Api


def _set_random_trainer_board(session: GameSession, path_list) -> None:
    random_board = session.ensure_book_reader().get_random_state(
        path_list, session.current_pattern
    )
    session.board_encoded = np_u64(random_board)
    session.score = 0
    session.history = [(session.board_encoded, session.score)]
    session.move_history = [None]
    session.moved = 0
    session.played_length = 0
    session.trainer_results = {}


def _configure_trainer_tablebase(
    session: GameSession,
    pattern: str,
    target: str,
    filepath: str | None,
    spawn_rate4: float,
):
    config = SingletonConfig().config
    filepath_map = config["filepath_map"]
    pattern_key = SingletonConfig.get_pattern_key(pattern, spawn_rate4)
    original_path_list = list(filepath_map.get(pattern_key, []))
    SingletonConfig.clean_pattern_paths(pattern, spawn_rate4, persist=False)
    current_path_list = list(filepath_map.get(pattern_key, []))
    normalized_path_list = current_path_list
    updated_config = current_path_list != original_path_list

    if filepath:
        success_rate_dtype = SingletonConfig.read_success_rate_dtype(filepath, pattern)
        table_4sr = SingletonConfig.read_4sr(filepath, pattern)
        table_4sr = table_4sr if table_4sr is not None else spawn_rate4
        pattern_key = SingletonConfig.get_pattern_key(pattern, table_4sr)
        SingletonConfig.clean_pattern_paths(pattern, table_4sr, persist=False)
        current_path_list = list(filepath_map.get(pattern_key, []))
        normalized_path_list = [
            (path, dtype)
            for path, dtype in current_path_list
            if path != filepath
        ]
        normalized_path_list.append((filepath, success_rate_dtype))
        filepath_map[pattern_key] = normalized_path_list
        SingletonConfig.clean_pattern_paths(pattern, table_4sr, persist=False)
        normalized_path_list = list(filepath_map.get(pattern_key, []))
        updated_config = True

    filepath_map[pattern_key] = normalized_path_list
    if updated_config:
        SingletonConfig().save_config(config)

    path_list = list(filepath_map.get(pattern_key, []))
    pattern_name = pattern.rsplit("_", 1)[0]
    session.ensure_book_reader().dispatch(path_list, pattern_name, target)
    session.current_pattern = pattern
    session.pattern_settings = [pattern_name, target]
    session.use_variant = pattern_name in category_info.get("variant", [])
    return path_list


async def handle_trainer_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
    manager: ConnectionManager,
) -> bool:
    spawn_rate4 = float(SingletonConfig().config.get("4_spawn_rate", 0.1))

    if action == Action.TRAINER_SET_FILEPATH:
        filepath = payload.get("filepath")
        pattern = payload.get("pattern", "L3")
        target = payload.get("target", "32768")
        current_board = np_u64(session.board_encoded)
        current_score = int(session.score)
        _clear_record_replay(session)
        session.board_encoded = current_board
        session.score = current_score
        session.history = [(session.board_encoded, session.score)]
        session.move_history = [None]
        session.played_length = 0
        session.moved = 0
        session.trainer_results = {}
        path_list = _configure_trainer_tablebase(
            session, str(pattern), str(target), filepath, spawn_rate4
        )

        if payload.get("load_default") and path_list:
            try:
                _set_random_trainer_board(session, path_list)
            except Exception as e:
                print("TRAINER_SET_FILEPATH default err:", e)

        await manager.send_state(websocket)
        await send_trainer_results(session, websocket)
        return True

    if action == Action.TRAINER_LOAD_POSITION:
        request_id = str(payload.get("request_id") or "")[:160]
        hex_str = str(payload.get("hex_str") or "").strip()
        try:
            board_encoded = np_u64(int(hex_str, 16))
        except (TypeError, ValueError, OverflowError):
            await websocket.send_json(
                {
                    "action": Message.TRAINER_POSITION_LOADED,
                    "data": {"request_id": request_id, "success": False},
                }
            )
            return True

        full_pattern = str(payload.get("full_pattern") or "").strip()
        if full_pattern:
            pattern_parts = full_pattern.rsplit("_", 1)
            if len(pattern_parts) != 2 or not all(pattern_parts):
                await websocket.send_json(
                    {
                        "action": Message.TRAINER_POSITION_LOADED,
                        "data": {"request_id": request_id, "success": False},
                    }
                )
                return True
            _configure_trainer_tablebase(
                session,
                full_pattern,
                pattern_parts[1],
                None,
                spawn_rate4,
            )

        _clear_record_replay(session)
        session.board_encoded = board_encoded
        session.score = 0
        session.history = [(session.board_encoded, session.score)]
        session.move_history = [None]
        session.played_length = 0
        session.moved = 0
        session.trainer_results = {}
        await manager.send_state(websocket)
        await send_trainer_results(session, websocket)
        await websocket.send_json(
            {
                "action": Message.TRAINER_POSITION_LOADED,
                "data": {
                    "request_id": request_id,
                    "success": True,
                    "full_pattern": session.current_pattern,
                },
            }
        )
        return True

    if action == Action.TRAINER_GET_RESULTS:
        await send_trainer_results(
            session, websocket, request_id=payload.get("request_id")
        )
        return True

    if action == Action.SET_SPAWN_MODE:
        session.spawn_mode = int(payload.get("mode", 0))
        session.moved = 0
        return True

    if action == Action.TRAINER_STEP:
        if session.trainer_results:
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

        session.score += int(move_score)
        if session.score > session.best_score:
            session.best_score = session.score

        num_pos_1d, val_exp = -1, 0
        if session.spawn_mode == 0:
            session.moved = 0
            gen_fn = v_gen_new_num if session.use_variant else r_gen_new_num
            new_board, _, num_pos_1d, val_exp = gen_fn(new_board, spawn_rate4)
            new_board = np_u64(new_board)
        elif session.spawn_mode == 3:
            session.moved = 1
        elif session.spawn_mode in (1, 2):
            session.moved = 0
            spawns = _compute_spawns(session, new_board)
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

        session.board_encoded = np_u64(new_board)
        session.history.append((session.board_encoded, session.score))
        session.move_history.append(direction_str)
        session.played_length = len(session.history) - 1

        metadata = build_move_animation_metadata(
            direction_str,
            board_encoded=old_board_encoded,
            use_variant=session.use_variant,
            spawn_index=num_pos_1d,
            spawn_value=2**val_exp if val_exp > 0 else 0,
        )
        if session.recording_state:
            _record_state(session, direction_str, num_pos_1d, val_exp)

        await manager.send_state(websocket, metadata)
        return True

    if action == Action.TRAINER_DEFAULT:
        _clear_record_replay(session)
        pattern_key = SingletonConfig.get_pattern_key(session.current_pattern, spawn_rate4)
        path_list = (
            SingletonConfig()
            .config["filepath_map"]
            .get(pattern_key, [])
        )
        if path_list:
            try:
                _set_random_trainer_board(session, path_list)
                await manager.send_state(websocket)
                await send_trainer_results(session, websocket)
            except Exception as e:
                print("TRAINER_DEFAULT err:", e)
        return True

    if action == Action.TRAINER_MANUAL_SPAWN:
        if session.spawn_mode == 3 and session.moved == 1:
            _clear_record_replay(session)
            row = payload.get("row", 0)
            col = payload.get("col", 0)
            val = payload.get("val", 2)

            board_2d = decode_board(np.uint64(u64(session.board_encoded)))
            if board_2d[row, col] == 0:
                board_2d[row, col] = val
                session.board_encoded = np_u64(encode_board(board_2d))
                session.moved = 0
                session.history.append((session.board_encoded, session.score))
                session.move_history.append("spawn")
                session.played_length = len(session.history) - 1

                num_pos_1d = row * 4 + col
                metadata = {"appear_tile": {"index": num_pos_1d, "value": val}}
                await manager.send_state(websocket, metadata)
        return True

    if action == Action.SET_CELL:
        row = payload.get("row", 0)
        col = payload.get("col", 0)
        val = payload.get("val", 0)
        _clear_record_replay(session)

        board_2d = decode_board(np.uint64(u64(session.board_encoded)))
        board_2d[row, col] = int(val)

        session.board_encoded = np_u64(encode_board(board_2d))
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
            session.trainer_results = {}
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
            session.trainer_results = {}
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
