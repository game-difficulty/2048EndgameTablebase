from __future__ import annotations

import numpy as np

import engine_core.BoardMover as bm
import engine_core.VBoardMover as vbm
from engine_core.Calculator import find_merge_positions, slide_distance

REPLAY_DTYPE = np.dtype("uint64,uint8,uint32,uint32,uint32,uint32")
REPLAY_SENTINEL = (
    np.uint64(0),
    np.uint8(88),
    np.uint32(666666666),
    np.uint32(233333333),
    np.uint32(314159265),
    np.uint32(987654321),
)
REPLAY_DIRS = ("left", "right", "up", "down")
ENGINE_DIR_MAP = {"left": 1, "right": 2, "up": 3, "down": 4}


def _normalize_board_for_animation(board_2d, use_variant=False):
    if not use_variant:
        return board_2d
    normalized = np.array(board_2d, copy=True)
    normalized[normalized == 32768] = -1
    return normalized


def empty_replay():
    return np.empty(0, dtype=REPLAY_DTYPE)


def replay_move_bits_to_dir(move_bits):
    if 0 <= int(move_bits) < len(REPLAY_DIRS):
        return REPLAY_DIRS[int(move_bits)]
    return None


def decode_replay_change(encoded):
    encoded_int = int(encoded)
    move_bits = (encoded_int >> 5) & 0b11
    spawn_pos = (encoded_int >> 1) & 0b1111
    spawn_exp = (encoded_int & 0b1) + 1
    return replay_move_bits_to_dir(move_bits), int(spawn_pos), int(spawn_exp)


def validate_replay_array(record):
    if len(record) < 1:
        return False
    last = record[-1].copy()
    last["f0"] = 0
    return tuple(last.tolist()) == REPLAY_SENTINEL


def strip_replay_sentinel(record):
    if len(record) and validate_replay_array(record):
        return record[:-1].copy()
    return empty_replay()


def load_replay_file(path):
    record = np.fromfile(path, dtype=REPLAY_DTYPE)
    return strip_replay_sentinel(record)


def current_results(record, step):
    if step < 0 or step >= len(record):
        return None
    values = [float(rate) / 4e9 for rate in list(record[step][["f2", "f3", "f4", "f5"]])]
    keys = ("left", "right", "up", "down")
    return dict(sorted(zip(keys, values), key=lambda item: item[1], reverse=True))


def evaluation_of_performance(loss):
    if loss > 1 - 3e-10:
        return "Perfect!"
    if loss >= 0.999:
        return "Excellent!"
    if loss >= 0.99:
        return "Nice try!"
    if loss >= 0.975:
        return "Not bad!"
    if loss >= 0.9:
        return "Mistake!"
    if loss >= 0.75:
        return "Blunder!"
    return "Terrible!"


def analyze_replay(record, marker_threshold=1.0):
    if len(record) == 0:
        return {
            "moves": np.empty(0, dtype=np.uint8),
            "losses": np.empty(0, dtype=float),
            "goodness_of_fit": np.empty(0, dtype=float),
            "combo": np.empty(0, dtype=np.uint16),
            "points_rank": np.empty(0, dtype=int),
            "summary": {
                "total_moves": 0,
                "final_gof": 0.0,
                "max_combo": 0,
                "counts": {},
            },
        }

    moves = ((record["f1"] >> 5) & 0b11).astype(np.uint8)
    arr_rates = np.vstack((record["f2"], record["f3"], record["f4"], record["f5"])).T.astype(float) / 4e9
    optimal = np.max(arr_rates, axis=1)
    optimal[optimal <= 0] = 1
    player = arr_rates[np.arange(len(moves)), moves]

    losses = player / optimal
    losses[losses == 0] = 1
    goodness_of_fit = np.cumprod(losses)

    combo = np.empty(len(losses), dtype=np.uint16)
    count = 0
    for index, loss in enumerate(losses):
        if loss > 1 - 3e-10:
            count += 1
        else:
            count = 0
        combo[index] = count

    threshold = min(float(np.quantile(losses, 0.1)), float(marker_threshold)) if len(losses) else float(marker_threshold)
    points_rank = np.where((losses < threshold) & (losses < 1))[0]

    counts = {}
    for loss in losses:
        label = evaluation_of_performance(float(loss))
        counts[label] = counts.get(label, 0) + 1

    return {
        "moves": moves,
        "losses": losses,
        "goodness_of_fit": goodness_of_fit,
        "combo": combo,
        "points_rank": points_rank,
        "summary": {
            "total_moves": int(len(losses)),
            "final_gof": float(goodness_of_fit[-1]) if len(goodness_of_fit) else 0.0,
            "max_combo": int(np.max(combo)) if len(combo) else 0,
            "counts": counts,
        },
    }


def build_step_transition(record, step, use_variant=False):
    if step < 0 or step >= len(record):
        return None

    board_encoded = np.uint64(int(record[step]["f0"]))
    move_name, spawn_pos, spawn_exp = decode_replay_change(record[step]["f1"])
    if move_name is None:
        return None

    move_fn = vbm.s_move_board if use_variant else bm.s_move_board
    board_2d = vbm.decode_board(board_encoded)
    animation_board = _normalize_board_for_animation(board_2d, use_variant)
    moved_board, move_score = move_fn(board_encoded, ENGINE_DIR_MAP[move_name])
    next_board = np.uint64(int(moved_board) | (int(spawn_exp) << (int(spawn_pos) * 4)))

    return {
        "board_encoded": board_encoded,
        "next_board_encoded": next_board,
        "direction": move_name,
        "slide_distances": slide_distance(animation_board, move_name).flatten().tolist(),
        "pop_positions": find_merge_positions(animation_board, move_name).flatten().tolist(),
        "appear_tile": {
            "index": int(spawn_pos),
            "value": int(2 ** spawn_exp),
        },
        "score_delta": int(move_score),
    }


def board_for_replay_step(record, step, use_variant=False):
    if len(record) == 0:
        return np.uint64(0)
    if step <= 0:
        return np.uint64(int(record[0]["f0"]))
    if step < len(record):
        return np.uint64(int(record[step]["f0"]))
    transition = build_step_transition(record, len(record) - 1, use_variant)
    return np.uint64(int(transition["next_board_encoded"])) if transition else np.uint64(int(record[-1]["f0"]))
