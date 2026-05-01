from __future__ import annotations

import numpy as np

from engine_core.BoardMover import s_gen_new_num as r_gen_new_num
from engine_core.BookReader import BookReader
from Config import DTYPE_CONFIG, SingletonConfig, pattern_32k_tiles_map
from engine_core.VBoardMover import decode_board

from .serialization import sanitize_config
from .session import safe_hex, u64


def replace_largest_tiles(board_encoded, n, target: str):
    if n == 0:
        return np.uint64(board_encoded)

    be = np.uint64(board_encoded)
    tiles = []
    for i in range(16):
        tile_value = int((be >> np.uint64(4 * i)) & np.uint64(0xF))
        tiles.append(tile_value)

    sorted_tiles = sorted(tiles, reverse=True)
    threshold = sorted_tiles[n - 1]
    if 2**threshold < int(target):
        return be

    count = 0
    for i in range(len(tiles)):
        if count < n and tiles[i] >= threshold:
            tiles[i] = 0xF
            count += 1

    result = np.uint64(0)
    for i in range(15, -1, -1):
        result = np.uint64(result << np.uint64(4))
        result = np.uint64(result | np.uint64(tiles[i]))

    return result


def _compute_spawns(session, new_board):
    results = {}
    _32ks = pattern_32k_tiles_map.get(session.pattern_settings[0], [0])[0]
    target = session.pattern_settings[1]
    bd_encoded = np.uint64(
        u64(replace_largest_tiles(np.uint64(u64(new_board)), _32ks, target))
    )

    for val in (1, 2):
        for tile_pos in range(16):
            p = 15 - tile_pos
            if ((bd_encoded >> np.uint64(4 * p)) & np.uint64(0xF)) == np.uint64(0):
                test_board = bd_encoded | (np.uint64(val) << np.uint64(4 * p))
                result, _ = session.ensure_book_reader().move_on_dic(
                    decode_board(test_board),
                    session.pattern_settings[0],
                    session.pattern_settings[1],
                    session.current_pattern,
                )
                if isinstance(result, dict):
                    first_key = list(result.keys())[0]
                    r0 = result[first_key]
                    if r0 is not None and not isinstance(r0, str):
                        results[(tile_pos, val)] = r0
    return results


async def send_trainer_results(session, websocket, request_id=None):
    if session.book_reader is None:
        return
    _32ks = pattern_32k_tiles_map.get(session.pattern_settings[0], [0])[0]
    target = session.pattern_settings[1]
    try:
        board_encoded_replaced = np.uint64(
            u64(
                replace_largest_tiles(
                    np.uint64(u64(session.board_encoded)), _32ks, target
                )
            )
        )
        board = decode_board(board_encoded_replaced)
        result, dtype = session.ensure_book_reader().move_on_dic(
            board,
            session.pattern_settings[0],
            session.pattern_settings[1],
            session.current_pattern,
        )

        if not isinstance(result, dict):
            result = {}
        else:

            def safe_float(v):
                if v is None or isinstance(v, str):
                    return None
                try:
                    return float(v)
                except Exception:
                    return None

            result = {k: safe_float(v) for k, v in result.items()}

        session.trainer_results = result
        await websocket.send_json(
            {
                "action": "TRAINER_RESULTS",
                "data": {
                    "results": result,
                    "dtype": str(dtype) if result else "?",
                    "board_hex": safe_hex(session.board_encoded),
                    "request_id": request_id,
                },
            }
        )
    except Exception as e:
        print("send_trainer_results Err:", e)


def _clear_record_replay(session):
    session.record_result_history = []
    session.record_result_dtype = None
    session.record_animation_history = []


def _decode_record_rates(raw_rates):
    result = {}
    for i, direction in enumerate(("up", "down", "left", "right")):
        result[direction] = float(raw_rates[i]) / 4e9
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


def _get_current_record_results(session):
    record_result_history = getattr(session, "record_result_history", [])
    if not record_result_history:
        return None, None

    idx = getattr(session, "played_length", 0)
    if 0 <= idx < len(record_result_history):
        return record_result_history[idx], (
            getattr(session, "record_result_dtype", None) or "recorded"
        )

    return None, getattr(session, "record_result_dtype", None) or "recorded"


def _record_state(session, move_str, gen_pos=0, gen_val=1):
    if not session.recording_state or session.record_length >= 10000:
        return

    move_map = {"up": 0, "down": 1, "left": 2, "right": 3}
    move_bits = move_map.get(move_str, 0) & 0b11
    changes = np.uint8(
        move_bits | ((gen_pos & 0b1111) << 2) | (((gen_val - 1) & 0b1) << 6)
    )

    zero_val = DTYPE_CONFIG.get(session.success_rate_dtype, DTYPE_CONFIG["uint32"])[3]
    rates = []
    for d in ("up", "down", "left", "right"):
        r = session.trainer_results.get(d, 0)
        r = r if isinstance(r, (float, int, np.integer, np.floating)) else 0
        rates.append(np.uint32((r - zero_val) * 4e9))

    session.records[session.record_length] = (changes, rates)
    session.record_length += 1
