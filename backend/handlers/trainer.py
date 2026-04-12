from __future__ import annotations

import asyncio
import os
import threading
from typing import Any

import numpy as np
from Config import SingletonConfig, category_info
from egtb_core.Calculator import ReverseLR, ReverseUD, RotateL, RotateR
from fastapi import WebSocket
from egtb_core.VBoardMover import (
    decode_board,
    encode_board,
    s_gen_new_num as v_gen_new_num,
    s_move_board as v_move_board,
)
from egtb_core.BoardMover import s_gen_new_num as r_gen_new_num, s_move_board as r_move_board

from ..actions import Action, EventType, Message
from ..animation import build_move_animation_metadata
from ..session import GameSession
from ..session import u64
from ..state import ConnectionManager
from ..trainer_helpers import (
    _clear_record_replay,
    _compute_spawns,
    _decode_record_rates,
    _record_state,
    send_trainer_results,
)
from ..webview_api import Api


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
        _clear_record_replay(session)

        current_path_list = (
            SingletonConfig().config["filepath_map"].get((pattern, spawn_rate4), [])
        )

        if filepath:
            for path, success_rate_dtype in current_path_list.copy():
                if not os.path.exists(path) or path == filepath:
                    current_path_list.remove((path, success_rate_dtype))

            success_rate_dtype = SingletonConfig.read_success_rate_dtype(filepath, pattern)
            table_4sr = SingletonConfig.read_4sr(filepath, pattern)
            table_4sr = table_4sr if table_4sr is not None else spawn_rate4

            SingletonConfig().config["filepath_map"][(pattern, table_4sr)] = (
                current_path_list + [(filepath, success_rate_dtype)]
            )

        path_list = (
            SingletonConfig().config["filepath_map"].get((pattern, spawn_rate4), [])
        )
        session.book_reader.dispatch(path_list, pattern.split("_")[0], target)
        session.current_pattern = pattern
        session.pattern_settings = [pattern.split("_")[0], target]
        session.use_variant = pattern.split("_")[0] in category_info.get("variant", [])

        if path_list:
            try:
                random_board = session.book_reader.get_random_state(
                    path_list, session.current_pattern
                )
                session.board_encoded = np.uint64(random_board)
                session.score = 0
                session.history = [(session.board_encoded, session.score)]
                session.move_history = [None]
                session.moved = 0
                session.played_length = 0
                session.trainer_results = {}
            except Exception as e:
                print("TRAINER_SET_FILEPATH default load err:", e)

        await manager.send_state(websocket)
        if path_list:
            await send_trainer_results(session, websocket)
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
        old_board_encoded = session.board_encoded

        move_fn = v_move_board if session.use_variant else r_move_board
        new_board, move_score = move_fn(old_board_encoded, direction)

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
        elif session.spawn_mode == 3:
            session.moved = 1
            session.history.append((new_board, session.score))
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
                new_board = new_board | (np.uint64(val) << np.uint64(4 * p))
                num_pos_1d = pos
                val_exp = val
            else:
                gen_fn = v_gen_new_num if session.use_variant else r_gen_new_num
                new_board, _, num_pos_1d, val_exp = gen_fn(new_board, spawn_rate4)

        session.board_encoded = new_board
        if session.spawn_mode != 3:
            session.history.append((session.board_encoded, session.score))
            session.move_history.append(direction_str)
        else:
            session.history.append((new_board, session.score))
            session.move_history.append(direction_str)
        session.played_length = len(session.history) - 1

        metadata = build_move_animation_metadata(
            direction_str,
            board_encoded=old_board_encoded,
            spawn_index=num_pos_1d,
            spawn_value=2**val_exp if val_exp > 0 else 0,
        )
        if session.recording_state:
            _record_state(session, direction_str, num_pos_1d, val_exp)

        await manager.send_state(websocket, metadata)
        return True

    if action == Action.TRAINER_DEFAULT:
        _clear_record_replay(session)
        path_list = (
            SingletonConfig()
            .config["filepath_map"]
            .get((session.current_pattern, spawn_rate4), [])
        )
        if path_list:
            try:
                random_board = session.book_reader.get_random_state(
                    path_list, session.current_pattern
                )
                session.board_encoded = np.uint64(random_board)
                session.score = 0
                session.history = [(session.board_encoded, session.score)]
                session.move_history = [None]
                session.moved = 0
                session.played_length = 0
                session.trainer_results = {}
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
                session.board_encoded = encode_board(board_2d)
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

        session.board_encoded = encode_board(board_2d)
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
            session.board_encoded = np.uint64(rotate_func(np.uint64(u64(session.board_encoded))))
            session.score = 0
            session.moved = 0
            session.trainer_results = {}
            session.history = [(session.board_encoded, session.score)]
            session.move_history = [None]
            session.played_length = 0
            await manager.send_state(websocket)
        return True

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

                    current_board = init_board
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
                        nb |= np.uint64(val_exp) << np.uint64(4 * p)
                        current_board = nb
                        session.history.append((current_board, current_score))
                        session.move_history.append(move_label)
                        session.record_animation_history.append(
                            build_move_animation_metadata(
                                move_label,
                                board_encoded=session.history[-2][0],
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
                    session.played_length = 0
                    session.board_encoded = session.history[0][0]
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
            if not getattr(session, "record_result_history", []):
                await send_trainer_results(session, websocket)
        return True

    if action == Action.TRIGGER_SELECT_FOLDER:
        def _trigger() -> None:
            api = Api()
            path = api.select_folder()
            asyncio.run(
                websocket.send_json(
                    {"action": Message.FOLDER_SELECTED, "data": {"path": path}}
                )
            )

        threading.Thread(target=_trigger).start()
        return True

    if action == Action.TRIGGER_RECORD_OPEN:
        def _trigger() -> None:
            api = Api()
            path = api.select_open_record()
            if path:
                asyncio.run(
                    websocket.send_json(
                        {
                            "action": Message.DO_API_CALLBACK,
                            "data": {"type": Action.RECORD_OPEN, "path": path},
                        }
                    )
                )

        threading.Thread(target=_trigger).start()
        return True

    if action == Action.TRIGGER_RECORD_SAVE:
        def _trigger() -> None:
            api = Api()
            path = api.select_save_record()
            if path:
                asyncio.run(
                    websocket.send_json(
                        {
                            "action": Message.DO_API_CALLBACK,
                            "data": {"type": Action.RECORD_SAVE, "path": path},
                        }
                    )
                )

        threading.Thread(target=_trigger).start()
        return True

    return False
