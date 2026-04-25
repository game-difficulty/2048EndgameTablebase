import os

import numpy as np
from Config import SingletonConfig
from engine_core.VBoardMover import decode_board
from engine_core.replay_utils import (
    analyze_replay,
    board_for_replay_step,
    build_step_transition,
    current_results as replay_current_results,
    decode_replay_change,
    empty_replay,
    evaluation_of_performance as replay_evaluation_of_performance,
)

from .serialization import sanitize_config
from .session import np_u64, safe_hex, u64
from .tester import TESTER_PERFORMANCE_ORDER


def _replay_reset(session, status=""):
    session.replay_record = empty_replay()
    session.replay_pattern = ""
    session.replay_source = ""
    session.replay_status = status
    session.replay_loaded = False
    session.replay_use_variant = False
    session.replay_current_step = 0
    session.replay_board_encoded = np_u64(0)
    session.replay_results = {}
    session.replay_current_move = None
    session.replay_best_move = None
    session.replay_loss = None
    session.replay_gof = None
    session.replay_combo = 0
    session.replay_points_rank = []
    session.replay_losses = []
    session.replay_summary = {
        "total_moves": 0,
        "final_gof": 0.0,
        "max_combo": 0,
        "counts": {},
    }


def _replay_pattern_from_path(path):
    splits = os.path.basename(path).split("_")
    return "_".join(splits[:2]) if len(splits) >= 2 else ""


def _replay_sync_step(session, step, animate=False, previous_step=None):
    total_moves = len(session.replay_record)
    step = max(0, min(int(step), total_moves))
    session.replay_current_step = step
    session.replay_board_encoded = np_u64(
        board_for_replay_step(
        session.replay_record, step, session.replay_use_variant
    )
    )
    session.replay_results = {}
    session.replay_current_move = None
    session.replay_best_move = None
    session.replay_loss = None
    session.replay_gof = None
    session.replay_combo = 0
    metadata = {}

    if step < total_moves:
        current = replay_current_results(session.replay_record, step) or {}
        session.replay_results = sanitize_config(current)
        session.replay_best_move = next(iter(current.keys()), None)
        move_name, _, _ = decode_replay_change(session.replay_record[step]["f1"])
        session.replay_current_move = move_name
        if step < len(session.replay_losses):
            session.replay_loss = float(session.replay_losses[step])
        analysis = analyze_replay(session.replay_record)
        gof_values = analysis["goodness_of_fit"]
        combo_values = analysis["combo"]
        if step < len(gof_values):
            session.replay_gof = float(gof_values[step])
        if step < len(combo_values):
            session.replay_combo = int(combo_values[step])

    if (
        animate
        and previous_step is not None
        and previous_step + 1 == step
        and previous_step < total_moves
    ):
        transition = build_step_transition(
            session.replay_record, previous_step, session.replay_use_variant
        )
        if transition:
            metadata = {
                "direction": transition["direction"],
                "slide_distances": transition["slide_distances"],
                "pop_positions": transition["pop_positions"],
                "appear_tile": transition["appear_tile"],
            }

    return metadata


def _replay_load_record(session, record, pattern="", source="", use_variant=False):
    session.replay_record = record.copy()
    session.replay_pattern = pattern
    session.replay_source = source
    session.replay_use_variant = bool(use_variant)
    session.replay_loaded = len(record) > 0
    session.replay_status = (
        f"Loaded {source}" if source else (f"Loaded {pattern}" if pattern else "")
    )

    if not session.replay_loaded:
        _replay_reset(session, "Recording file corrupted")
        return

    marker_threshold = SingletonConfig().config.get("record_player_slider_threshold", 1)
    analysis = analyze_replay(session.replay_record, marker_threshold)
    session.replay_losses = [float(item) for item in analysis["losses"].tolist()]
    session.replay_points_rank = [int(item) for item in analysis["points_rank"].tolist()]

    summary_counts = {label: 0 for label in TESTER_PERFORMANCE_ORDER}
    summary_counts.update(analysis["summary"]["counts"])
    session.replay_summary = {
        "total_moves": int(analysis["summary"]["total_moves"]),
        "final_gof": float(analysis["summary"]["final_gof"]),
        "max_combo": int(analysis["summary"]["max_combo"]),
        "counts": summary_counts,
    }
    _replay_sync_step(session, 0)


async def send_replay_state(websocket, session, metadata=None):
    board_encoded = np_u64(session.replay_board_encoded)
    board_array = decode_board(board_encoded)
    config = SingletonConfig().config
    summary = sanitize_config(session.replay_summary)
    await websocket.send_json(
        {
            "action": "REPLAY_STATE",
            "data": {
                "board": board_array.flatten().tolist(),
                "animation": sanitize_config(metadata or {}),
                "hex_str": safe_hex(board_encoded),
                "loaded": session.replay_loaded,
                "status": session.replay_status,
                "pattern": session.replay_pattern,
                "source": session.replay_source,
                "current_step": session.replay_current_step,
                "total_steps": int(len(session.replay_record)),
                "results": sanitize_config(session.replay_results),
                "current_move": session.replay_current_move,
                "best_move": session.replay_best_move,
                "loss": session.replay_loss,
                "goodness_of_fit": session.replay_gof,
                "combo": session.replay_combo,
                "evaluation": (
                    replay_evaluation_of_performance(session.replay_loss)
                    if session.replay_loss is not None
                    else None
                ),
                "points_rank": session.replay_points_rank,
                "losses": session.replay_losses,
                "summary": summary,
                "settings": {
                    "colors": config.get("colors", []),
                    "dis_32k": config.get("dis_32k", False),
                    "slider_threshold": config.get(
                        "record_player_slider_threshold", 1
                    ),
                },
            },
        }
    )
