from __future__ import annotations

import numpy as np

import engine_core.BoardMover as bm
import engine_core.VBoardMover as vbm
from engine_core.Calculator import find_merge_positions, slide_distance
from engine_core.performance_evaluation import (
    PERFORMANCE_PERFECT_LABEL,
    evaluation_of_performance as shared_evaluation_of_performance,
    is_perfect_result,
)

REPLAY_DTYPE = np.dtype("uint64,uint8,uint32,uint32,uint32,uint32")
REPLAY_FORCED_FLAG = np.uint8(0x80)
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


def _non_merging_values_for_animation(use_variant=False):
    return (32768, 16384) if use_variant else (32768,)


def empty_replay():
    return np.empty(0, dtype=REPLAY_DTYPE)


def replay_sentinel(terminal_board=0):
    """Build the existing sentinel with an optional terminal board in f0."""
    return (np.uint64(int(terminal_board)), *REPLAY_SENTINEL[1:])


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


def replay_change_is_forced(encoded):
    return bool(int(encoded) & int(REPLAY_FORCED_FLAG))


def replay_step_goodness_ratio(selected_rate, best_rate):
    selected = float(selected_rate)
    best = float(best_rate)
    if is_perfect_result(selected, best):
        return 1.0
    return selected / best if best > 0 else 1.0


def replay_spawn_pos_to_board_pos(spawn_pos):
    """Convert the row-major replay cell index to the packed-board nibble index."""
    return 15 - int(spawn_pos)


def validate_replay_array(record):
    if len(record) < 1:
        return False
    last = record[-1].copy()
    last["f0"] = 0
    return tuple(last.tolist()) == REPLAY_SENTINEL


def strip_replay_sentinel(record):
    replay, _ = split_replay_sentinel(record)
    return replay


def split_replay_sentinel(record):
    """Remove the sentinel and recover its optional terminal board snapshot."""
    if not len(record) or not validate_replay_array(record):
        return empty_replay(), None
    terminal_board = np.uint64(int(record[-1]["f0"]))
    return record[:-1].copy(), terminal_board if terminal_board else None


def load_replay_file(path):
    record = np.fromfile(path, dtype=REPLAY_DTYPE)
    return strip_replay_sentinel(record)


def load_replay_file_with_terminal_board(path):
    record = np.fromfile(path, dtype=REPLAY_DTYPE)
    return split_replay_sentinel(record)


def current_results(record, step):
    if step < 0 or step >= len(record):
        return None
    values = [float(rate) / 4e9 for rate in list(record[step][["f2", "f3", "f4", "f5"]])]
    keys = ("left", "right", "up", "down")
    return dict(sorted(zip(keys, values), key=lambda item: item[1], reverse=True))


def evaluation_of_performance(loss, selected_rate=None, best_rate=None):
    selected = loss if selected_rate is None else selected_rate
    best = 1.0 if best_rate is None else best_rate
    if is_perfect_result(selected, best):
        return PERFORMANCE_PERFECT_LABEL
    return shared_evaluation_of_performance(loss)


def analyze_replay(record, marker_threshold=1.0):
    if len(record) == 0:
        return {
            "moves": np.empty(0, dtype=np.uint8),
            "losses": np.empty(0, dtype=float),
            "goodness_of_fit": np.empty(0, dtype=float),
            "combo": np.empty(0, dtype=np.uint16),
            "forced": np.empty(0, dtype=bool),
            "evaluations": [],
            "points_rank": np.empty(0, dtype=int),
            "summary": {
                "total_moves": 0,
                "final_gof": 0.0,
                "max_combo": 0,
                "counts": {},
            },
        }

    moves = ((record["f1"] >> 5) & 0b11).astype(np.uint8)
    arr_rates_raw = np.vstack(
        (record["f2"], record["f3"], record["f4"], record["f5"])
    ).T
    forced = (record["f1"] & REPLAY_FORCED_FLAG) != 0
    arr_rates = arr_rates_raw.astype(float) / 4e9
    optimal = np.max(arr_rates, axis=1)
    player = arr_rates[np.arange(len(moves)), moves]

    losses = np.ones(len(moves), dtype=float)
    for index in range(len(moves)):
        if not forced[index]:
            losses[index] = replay_step_goodness_ratio(
                player[index], optimal[index]
            )
    goodness_of_fit = np.cumprod(losses)

    combo = np.empty(len(losses), dtype=np.uint16)
    count = 0
    for index, loss in enumerate(losses):
        if forced[index]:
            pass
        elif is_perfect_result(player[index], optimal[index]):
            count += 1
        else:
            count = 0
        combo[index] = count

    threshold = min(float(np.quantile(losses, 0.1)), float(marker_threshold)) if len(losses) else float(marker_threshold)
    points_rank = np.where((losses < threshold) & (losses < 1))[0]

    counts = {}
    evaluations = []
    for index, loss in enumerate(losses):
        if forced[index]:
            evaluations.append(None)
            continue
        label = evaluation_of_performance(
            float(loss), float(player[index]), float(optimal[index])
        )
        evaluations.append(label)
        counts[label] = counts.get(label, 0) + 1

    return {
        "moves": moves,
        "losses": losses,
        "goodness_of_fit": goodness_of_fit,
        "combo": combo,
        "forced": forced,
        "evaluations": evaluations,
        "points_rank": points_rank,
        "summary": {
            "total_moves": int(np.count_nonzero(~forced)),
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
    board_for_move = board_encoded
    score_adjustment = 0
    if not use_variant:
        # Packed boards cap exponents at 15, so emulate a 32k+32k merge as
        # 16k+16k and restore the missing half of the score.
        positions = np.argwhere(board_2d == 32768)
        if len(positions) == 2:
            first, second = positions
            first_row, first_column = (int(first[0]), int(first[1]))
            second_row, second_column = (int(second[0]), int(second[1]))
            horizontal_pair = (
                first_row == second_row
                and move_name in ("left", "right")
                and not np.any(
                    board_2d[
                        first_row,
                        min(first_column, second_column) + 1 : max(
                            first_column, second_column
                        ),
                    ]
                )
            )
            vertical_pair = (
                first_column == second_column
                and move_name in ("up", "down")
                and not np.any(
                    board_2d[
                        min(first_row, second_row) + 1 : max(first_row, second_row),
                        first_column,
                    ]
                )
            )
            if horizontal_pair or vertical_pair:
                board_2d = np.array(board_2d, copy=True)
                board_2d[tuple(first)] = 16384
                board_2d[tuple(second)] = 16384
                board_for_move = np.uint64(vbm.encode_board(board_2d))
                score_adjustment = 32768

    animation_board = _normalize_board_for_animation(board_2d, use_variant)
    non_merging_values = _non_merging_values_for_animation(use_variant)
    moved_board, move_score = move_fn(board_for_move, ENGINE_DIR_MAP[move_name])
    if int(moved_board) == int(board_for_move):
        return None
    board_spawn_pos = replay_spawn_pos_to_board_pos(spawn_pos)
    if (int(moved_board) >> (board_spawn_pos * 4)) & 0xF:
        return None
    next_board = np.uint64(
        int(moved_board) | (int(spawn_exp) << (board_spawn_pos * 4))
    )

    return {
        "board_encoded": board_encoded,
        "next_board_encoded": next_board,
        "direction": move_name,
        "slide_distances": slide_distance(animation_board, move_name, non_merging_values).flatten().tolist(),
        "pop_positions": find_merge_positions(animation_board, move_name, non_merging_values).flatten().tolist(),
        "appear_tile": {
            "index": int(spawn_pos),
            "value": int(2 ** spawn_exp),
        },
        "score_delta": int(move_score) + score_adjustment,
    }


def replay_transition_matches_next_snapshot(
    record, step, use_variant=False, terminal_board=None
):
    """Return whether a recorded move reaches the following board snapshot.

    A terminal snapshot stored in the sentinel validates the final transition.
    Legacy files without one fall back to validating the move itself.
    """
    if step < 0 or step >= len(record):
        return False

    transition = build_step_transition(record, step, use_variant)
    if transition is None:
        return False
    if step + 1 >= len(record):
        return terminal_board is None or int(transition["next_board_encoded"]) == int(
            terminal_board
        )

    expected = int(transition["next_board_encoded"])
    actual = int(record[step + 1]["f0"])
    return expected == actual


def board_for_replay_step(record, step, use_variant=False, terminal_board=None):
    if len(record) == 0:
        return np.uint64(0)
    if step <= 0:
        return np.uint64(int(record[0]["f0"]))
    if step < len(record):
        return np.uint64(int(record[step]["f0"]))
    if terminal_board is not None:
        return np.uint64(int(terminal_board))
    transition = build_step_transition(record, len(record) - 1, use_variant)
    return np.uint64(int(transition["next_board_encoded"])) if transition else np.uint64(int(record[-1]["f0"]))
