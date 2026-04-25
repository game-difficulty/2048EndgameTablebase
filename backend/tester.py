import random

import numpy as np
from Config import (
    DTYPE_CONFIG,
    SingletonConfig,
    category_info,
    formation_info,
)
from engine_core.BookReader import BookReader
from engine_core.VBoardMover import decode_board, encode_board
from engine_core.replay_utils import empty_replay, strip_replay_sentinel

from .serialization import sanitize_config
from .session import np_u64, safe_hex, u64


TESTER_PERFORMANCE_ORDER = (
    "Perfect!",
    "Excellent!",
    "Nice try!",
    "Not bad!",
    "Mistake!",
    "Blunder!",
    "Terrible!",
)
TESTER_MOVE_LABELS = {
    "en": {"left": "Left", "right": "Right", "up": "Up", "down": "Down"},
    "zh": {"left": "左", "right": "右", "up": "上", "down": "下"},
}
TESTER_EVALUATION_LABELS = {
    "en": {
        "Perfect!": "Perfect!",
        "Excellent!": "Excellent!",
        "Nice try!": "Nice try!",
        "Not bad!": "Not bad!",
        "Mistake!": "Mistake!",
        "Blunder!": "Blunder!",
        "Terrible!": "Terrible!",
    },
    "zh": {
        "Perfect!": "Perfect!",
        "Excellent!": "Excellent!",
        "Nice try!": "Nice try!",
        "Not bad!": "Not bad!",
        "Mistake!": "Mistake!",
        "Blunder!": "Blunder!",
        "Terrible!": "Terrible!",
    },
}
TESTER_REPLAY_SENTINEL = (
    np.uint64(0),
    np.uint8(88),
    np.uint32(666666666),
    np.uint32(233333333),
    np.uint32(314159265),
    np.uint32(987654321),
)
LATEST_TESTER_REPLAY = {
    "record": empty_replay(),
    "pattern": "",
    "source": "",
    "use_variant": False,
}


def _tester_reset_metrics(session):
    session.tester_combo = 0
    session.tester_goodness_of_fit = 1.0
    session.tester_max_combo = 0
    session.tester_performance_stats = {
        label: 0 for label in TESTER_PERFORMANCE_ORDER
    }


def _tester_reset_last_step(session):
    session.tester_last_step = {
        "board_lines": [],
        "result_lines": [],
        "results": {},
        "dtype": "?",
        "message_lines": [],
        "evaluation": "",
        "direction": None,
        "best_move": None,
        "loss": None,
        "goodness_of_fit": None,
    }


def _tester_reset_record(session):
    session.tester_record = np.zeros(
        4000, dtype="uint64,uint8,uint32,uint32,uint32,uint32"
    )
    session.tester_step_count = 0
    session.tester_record[0] = (
        np.uint64(u64(session.board_encoded)),
        np.uint8(0),
        np.uint32(0),
        np.uint32(0),
        np.uint32(0),
        np.uint32(0),
    )


def _cache_tester_replay(session):
    if session.tester_step_count <= 0:
        return
    replay = session.tester_record[: session.tester_step_count + 1].copy()
    replay[session.tester_step_count] = TESTER_REPLAY_SENTINEL
    LATEST_TESTER_REPLAY["record"] = strip_replay_sentinel(replay)
    LATEST_TESTER_REPLAY["pattern"] = session.tester_full_pattern
    LATEST_TESTER_REPLAY["source"] = "Tester session"
    LATEST_TESTER_REPLAY["use_variant"] = bool(session.use_variant)


def _tester_reset_history(session, board_encoded, score=0):
    session.board_encoded = np_u64(board_encoded)
    session.score = int(score)
    session.history = [(session.board_encoded, session.score)]
    session.move_history = [None]
    session.played_length = 0


def _tester_append_log(session, *lines):
    for line in lines:
        if isinstance(line, str):
            session.tester_logs.append(line)
        elif isinstance(line, (list, tuple)):
            session.tester_logs.extend(str(item) for item in line)
        else:
            session.tester_logs.append(str(line))


def _tester_board_lines(board_encoded):
    board = decode_board(np.uint64(u64(board_encoded)))
    lines = []
    for row in board:
        rendered = []
        for item in row:
            value = int(item)
            if value == 0:
                rendered.append("_".rjust(4))
            elif value == 32768:
                rendered.append("x".rjust(4))
            elif value >= 1024:
                rendered.append(f"{value // 1024}k".rjust(4))
            else:
                rendered.append(str(value).rjust(4))
        lines.append(" ".join(rendered))
    return lines


def _tester_language():
    lang = str(SingletonConfig().config.get("language", "en") or "en").lower()
    return "zh" if lang.startswith("zh") else "en"


def _tester_move_label(direction, lang=None):
    current_lang = lang or _tester_language()
    return TESTER_MOVE_LABELS.get(current_lang, TESTER_MOVE_LABELS["en"]).get(
        direction, str(direction or "?")
    )


def _tester_evaluation_label(label, lang=None):
    current_lang = lang or _tester_language()
    return TESTER_EVALUATION_LABELS.get(
        current_lang, TESTER_EVALUATION_LABELS["en"]
    ).get(label, str(label or ""))


def _tester_feedback_lines(
    evaluation, direction_str, best_move, combo, loss, goodness_of_fit
):
    lang = _tester_language()
    pressed_label = _tester_move_label(direction_str, lang)
    best_label = _tester_move_label(best_move, lang)
    display_evaluation = _tester_evaluation_label(evaluation, lang)

    if evaluation == "Perfect!":
        if lang == "zh":
            return [
                f"{display_evaluation} Combo: {combo}x",
                f"你走的是 {pressed_label}，最优解正是 {best_label}",
            ]
        return [
            f"{display_evaluation} Combo: {combo}x",
            f"You pressed {pressed_label}. And the best move is {best_label}.",
        ]

    if lang == "zh":
        return [
            display_evaluation,
            f"单步损失: {1 - loss:.4f}，GOF: {goodness_of_fit:.4f}",
            f"你走的是 {pressed_label}，但最优解是 {best_label}",
        ]
    return [
        display_evaluation,
        f"one-step loss: {1 - loss:.4f}, goodness of fit: {goodness_of_fit:.4f}",
        f"You pressed {pressed_label}. But the best move is {best_label}.",
    ]


def _tester_restore_success_rate(value, dtype):
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    _, _, _, zero_val = DTYPE_CONFIG.get(dtype, DTYPE_CONFIG["uint32"])
    if zero_val < 0:
        return float(abs(zero_val) + float(value))
    return float(value)


def _tester_format_rate_for_log(value, dtype):
    _, _, _, zero_val = DTYPE_CONFIG.get(dtype, DTYPE_CONFIG["uint32"])
    if zero_val >= 0 or not isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    if value >= 0 or value < -1e-7:
        return str(abs(zero_val) + value)
    return str(abs(zero_val)).strip(".0") + str(value)


def _tester_sanitize_results(results):
    sanitized = {}
    for key, value in results.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            sanitized[key] = float(value)
        elif value in ("", None):
            sanitized[key] = None
        else:
            sanitized[key] = str(value)
    return sanitized


def _tester_best_move(results):
    for key, value in results.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            return key
    return None


def _tester_random_rotate(board_encoded, pattern):
    operations = BookReader.gen_all_mirror(pattern)
    operation_func = random.choice(operations)[-1] if operations else (lambda x: x)
    rotated = operation_func(decode_board(np.uint64(u64(board_encoded))))
    return np.uint64(u64(encode_board(rotated)))


def _tester_prepare_selection(session, pattern, target):
    session.tester_pattern = [str(pattern or "?"), str(target or "?")]
    session.tester_full_pattern = (
        f"{session.tester_pattern[0]}_{session.tester_pattern[1]}"
        if "?" not in session.tester_pattern
        else ""
    )
    session.current_pattern = session.tester_full_pattern
    session.pattern_settings = session.tester_pattern.copy()
    session.use_variant = pattern in category_info.get("variant", [])

    if session.use_variant and pattern in formation_info:
        session.board_encoded = np_u64(formation_info[pattern][4][0])
    else:
        session.board_encoded = np_u64(0)
    session.score = 0

    if not session.tester_full_pattern:
        session.tester_table_found = False
        return False, []

    spawn_rate4 = SingletonConfig().config["4_spawn_rate"]
    session.tester_table_found = SingletonConfig().check_pattern_file(
        session.tester_full_pattern
    )
    path_list = (
        SingletonConfig()
        .config["filepath_map"]
        .get((session.tester_full_pattern, spawn_rate4), [])
    )
    if session.tester_table_found and path_list:
        session.ensure_book_reader().dispatch(path_list, pattern, target)
        session.tester_status = f"Loaded {session.tester_full_pattern}"
        return True, path_list

    session.tester_status = "Table file path not found."
    return False, path_list


def _tester_compute_results(session):
    if not session.tester_full_pattern or not session.tester_table_found:
        session.tester_results = {}
        session.tester_result_dtype = "?"
        session.tester_best_move = None
        return

    result, dtype = session.ensure_book_reader().move_on_dic(
        decode_board(np.uint64(u64(session.board_encoded))),
        session.tester_pattern[0],
        session.tester_pattern[1],
        session.tester_full_pattern,
    )

    if not isinstance(result, dict):
        session.tester_results = {}
        session.tester_result_dtype = str(dtype or "?")
        session.tester_best_move = None
        return

    session.tester_best_move = _tester_best_move(result)
    session.tester_results = _tester_sanitize_results(result)
    session.tester_result_dtype = str(dtype or "?")
    if session.tester_result_dtype and session.tester_result_dtype != "?":
        session.success_rate_dtype = session.tester_result_dtype


def _tester_start_practice(session, board_encoded, opening_text):
    _tester_reset_metrics(session)
    _tester_reset_last_step(session)
    _tester_reset_history(session, board_encoded, 0)
    _tester_reset_record(session)
    _tester_compute_results(session)
    session.tester_ready = True
    session.tester_logs = [
        f"Selected pattern: {session.tester_full_pattern}",
        opening_text,
        *_tester_board_lines(session.board_encoded),
    ]


def _tester_evaluation_of_performance(loss):
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


def _tester_record_step(session, direction, spawn_pos, spawn_val_exp):
    if session.tester_step_count >= len(session.tester_record) - 1:
        return

    move_bits = {"left": 0, "right": 1, "up": 2, "down": 3}.get(direction, 0)
    changes = np.uint8(
        (((int(move_bits) & 0b11) << 5))
        | (((int(spawn_pos) & 0b1111) << 1))
        | (((int(spawn_val_exp) - 1) & 0b1))
    )

    rates = []
    for d in ("left", "right", "up", "down"):
        rate = _tester_restore_success_rate(
            session.tester_results.get(d), session.tester_result_dtype
        )
        rates.append(np.uint32(max(0.0, rate or 0.0) * 4e9))

    session.tester_record[session.tester_step_count] = (
        np.uint64(u64(session.tester_record[session.tester_step_count][0])),
        changes,
        *rates,
    )
    session.tester_step_count += 1
    session.tester_record[session.tester_step_count] = (
        np.uint64(u64(session.board_encoded)),
        np.uint8(0),
        np.uint32(0),
        np.uint32(0),
        np.uint32(0),
        np.uint32(0),
    )


def _tester_append_summary(session):
    for label in TESTER_PERFORMANCE_ORDER:
        _tester_append_log(
            session, f"{label}: {session.tester_performance_stats.get(label, 0)}"
        )
    _tester_append_log(
        session,
        f"Total Goodness of Fit: {session.tester_goodness_of_fit:.4f}",
        f"Maximum Combo: {session.tester_max_combo}",
    )


async def send_tester_state(websocket, session, metadata=None):
    board_encoded = np_u64(session.board_encoded)
    board_array = decode_board(board_encoded)
    config = SingletonConfig().config

    await websocket.send_json(
        {
            "action": "TESTER_STATE",
            "data": {
                "board": board_array.flatten().tolist(),
                "animation": sanitize_config(metadata or {}),
                "hex_str": safe_hex(session.board_encoded),
                "pattern": session.tester_pattern[0],
                "target": session.tester_pattern[1],
                "full_pattern": session.tester_full_pattern,
                "results": sanitize_config(session.tester_results),
                "dtype": session.tester_result_dtype,
                "best_move": session.tester_best_move,
                "logs": session.tester_logs,
                "last_step": sanitize_config(session.tester_last_step),
                "text_visible": session.tester_text_visible,
                "ready": session.tester_ready,
                "table_found": session.tester_table_found,
                "status": session.tester_status,
                "use_variant": session.use_variant,
                "metrics": {
                    "combo": session.tester_combo,
                    "max_combo": session.tester_max_combo,
                    "goodness_of_fit": session.tester_goodness_of_fit,
                    "performance_stats": session.tester_performance_stats,
                    "score": int(session.score),
                    "best_score": int(session.best_score),
                },
                "record": {"length": session.tester_step_count},
                "settings": {
                    "colors": config.get("colors", []),
                    "dis_32k": config.get("dis_32k", False),
                    "dis_text": config.get("dis_text", True),
                    "language": config.get("language", "en"),
                },
            },
        }
    )
