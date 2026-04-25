from __future__ import annotations

import math
import os
import pickle
from collections import defaultdict

import numpy as np
from fastapi import WebSocket

import egtb_core.Calculator as Calculator
from Config import SingletonConfig
from egtb_core.VBoardMover import decode_board

from .actions import Message
from .serialization import sanitize_config
from .session import np_u64, safe_hex, u64


class MistakesBookStore:
    def __init__(self, filename: str | None = None) -> None:
        if filename is None:
            filename = os.path.join(
                os.path.dirname(__file__), "..", "docs_and_configs", "mistakes_book.pkl"
            )
        self.filename = filename
        self.mistakes: defaultdict[str, dict[np.uint64, tuple[int, float, str]]] = defaultdict(dict)
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.filename):
            self.mistakes = defaultdict(dict)
            return
        with open(self.filename, "rb") as handle:
            data = pickle.load(handle)
        self.mistakes = defaultdict(dict, data)

    def save(self) -> None:
        with open(self.filename, "wb") as handle:
            pickle.dump(dict(self.mistakes), handle)

    def get_all_patterns(self) -> list[str]:
        return list(self.mistakes.keys())

    def get_mistakes_for_pattern(self, full_pattern: str) -> dict[np.uint64, tuple[int, float, str]]:
        return self.mistakes.get(full_pattern, {})

    def remove_mistake(self, full_pattern: str, board_encoded: np.uint64) -> None:
        if full_pattern not in self.mistakes:
            return
        pattern_dict = self.mistakes[full_pattern]
        if board_encoded in pattern_dict:
            del pattern_dict[board_encoded]
            if not pattern_dict:
                del self.mistakes[full_pattern]

    def add_mistake(
        self, full_pattern: str, board_encoded: np.uint64, loss: float, best_move: str
    ) -> None:
        if not full_pattern or not best_move:
            return
        threshold = _get_notebook_threshold()
        if loss and loss > threshold:
            return

        pattern_dict = self.mistakes[full_pattern]
        normalized_board = np.uint64(u64(board_encoded))
        all_symm = [
            normalized_board,
            np.uint64(u64(Calculator.ReverseLR(normalized_board))),
            np.uint64(u64(Calculator.ReverseUD(normalized_board))),
            np.uint64(u64(Calculator.ReverseUL(normalized_board))),
            np.uint64(u64(Calculator.ReverseUR(normalized_board))),
            np.uint64(u64(Calculator.Rotate180(normalized_board))),
            np.uint64(u64(Calculator.RotateL(normalized_board))),
            np.uint64(u64(Calculator.RotateR(normalized_board))),
        ]

        for board_key in all_symm:
            if board_key in pattern_dict:
                count, total_loss, move = pattern_dict[board_key]
                pattern_dict[board_key] = (count + 1, total_loss + (1 - loss), move)
                self.save()
                return

        pattern_dict[normalized_board] = (1, 1 - loss, str(best_move))
        self.save()


mistakes_book_store = MistakesBookStore()


def _normalize_notebook_threshold(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.999
    if not math.isfinite(parsed):
        return 0.999
    return min(1.0, max(0.0, parsed))


def _get_notebook_threshold() -> float:
    return _normalize_notebook_threshold(
        SingletonConfig().config.get("notebook_threshold", 0.999)
    )


def _set_notebook_threshold(value: object) -> float:
    normalized = _normalize_notebook_threshold(value)
    config_manager = SingletonConfig()
    config_manager.config["notebook_threshold"] = normalized
    config_manager.save_config(config_manager.config)
    return normalized


def _notebook_pattern_list() -> list[str]:
    return mistakes_book_store.get_all_patterns()


def _notebook_clear_state(session) -> None:
    session.notebook_pattern = ""
    session.notebook_board_encoded = np_u64(0)
    session.notebook_best_move = None
    session.notebook_current_count = 0
    session.notebook_combo = 0
    session.notebook_correct = 0
    session.notebook_incorrect = 0
    session.notebook_weight_mode = 0
    session.notebook_unseen_boards = []
    session.notebook_answered = False
    session.notebook_last_direction = None
    session.notebook_answer_correct = None
    session.notebook_status = ""


def _notebook_calculate_weights(
    boards: list[np.uint64],
    mistakes: dict[np.uint64, tuple[int, float, str]],
    mode: int,
) -> list[float]:
    weights: list[float] = []
    for board in boards:
        count, total_loss, _ = mistakes[board]
        if mode == 0:
            weights.append(float(count**2))
        elif mode == 1:
            weights.append(float(total_loss**2))
        else:
            weights.append(float(count * total_loss))
    if sum(weights) == 0:
        return [1.0] * len(boards)
    return weights


def _notebook_reset_unseen_boards(session) -> None:
    if not session.notebook_pattern:
        session.notebook_unseen_boards = []
        return

    mistakes = mistakes_book_store.get_mistakes_for_pattern(session.notebook_pattern)
    if not mistakes:
        session.notebook_unseen_boards = []
        return

    boards = list(mistakes.keys())
    weights = _notebook_calculate_weights(
        boards, mistakes, int(session.notebook_weight_mode)
    )
    total_weight = sum(weights)
    probabilities = [weight / total_weight for weight in weights]
    shuffled_indices = np.random.choice(
        len(boards),
        size=min(len(boards), 1000),
        replace=False,
        p=probabilities,
    )
    session.notebook_unseen_boards = [np_u64(boards[index]) for index in shuffled_indices]
    session.notebook_correct = 0
    session.notebook_incorrect = 0


def _notebook_load_problem(session, board_encoded: np.uint64) -> None:
    mistakes = mistakes_book_store.get_mistakes_for_pattern(session.notebook_pattern)
    current_count, _, current_best_move = mistakes[board_encoded]
    session.notebook_board_encoded = np_u64(board_encoded)
    session.notebook_current_count = int(current_count)
    session.notebook_best_move = str(current_best_move).lower()
    session.notebook_answered = False
    session.notebook_last_direction = None
    session.notebook_answer_correct = None
    session.notebook_status = (
        f"Correct: {session.notebook_correct}  "
        f"Incorrect: {session.notebook_incorrect}  "
        f"Remaining: {len(session.notebook_unseen_boards)}"
    )


def _notebook_select_pattern(session, pattern: str) -> None:
    session.notebook_pattern = str(pattern or "")
    session.notebook_combo = 0
    session.notebook_correct = 0
    session.notebook_incorrect = 0
    _notebook_reset_unseen_boards(session)
    _notebook_next_problem(session)


def _notebook_next_problem(session) -> None:
    if not session.notebook_pattern:
        session.notebook_board_encoded = np_u64(0)
        session.notebook_best_move = None
        session.notebook_current_count = 0
        session.notebook_status = ""
        return

    mistakes = mistakes_book_store.get_mistakes_for_pattern(session.notebook_pattern)
    if not mistakes:
        if session.notebook_pattern not in mistakes_book_store.get_all_patterns():
            available = _notebook_pattern_list()
            session.notebook_pattern = available[0] if available else ""
        session.notebook_board_encoded = np_u64(0)
        session.notebook_best_move = None
        session.notebook_current_count = 0
        session.notebook_unseen_boards = []
        session.notebook_status = "No mistakes recorded for this pattern"
        return

    if not session.notebook_unseen_boards:
        _notebook_reset_unseen_boards(session)

    if not session.notebook_unseen_boards:
        session.notebook_board_encoded = np_u64(0)
        session.notebook_best_move = None
        session.notebook_current_count = 0
        session.notebook_status = "No mistakes recorded for this pattern"
        return

    next_board = np_u64(session.notebook_unseen_boards.pop(0))
    _notebook_load_problem(session, next_board)


def _notebook_update_mistake_count(session, delta: int) -> None:
    if not session.notebook_pattern or not session.notebook_board_encoded:
        return
    mistakes = mistakes_book_store.get_mistakes_for_pattern(session.notebook_pattern)
    board_key = np_u64(session.notebook_board_encoded)
    if board_key not in mistakes:
        return
    count, total_loss, best_move = mistakes[board_key]
    new_count = max(1, int(count) + int(delta))
    mistakes_book_store.mistakes[session.notebook_pattern][board_key] = (
        new_count,
        total_loss,
        best_move,
    )
    mistakes_book_store.save()
    session.notebook_current_count = new_count


def _notebook_answer(session, direction: str) -> None:
    if (
        not session.notebook_board_encoded
        or not session.notebook_best_move
        or session.notebook_answered
    ):
        return

    selected_direction = str(direction or "").lower()
    if selected_direction not in {"up", "down", "left", "right"}:
        return

    session.notebook_answered = True
    session.notebook_last_direction = selected_direction
    is_correct = selected_direction == session.notebook_best_move
    session.notebook_answer_correct = is_correct
    if is_correct:
        session.notebook_combo += 1
        session.notebook_correct += 1
        session.notebook_status = f"Combo: {session.notebook_combo}x"
        return

    session.notebook_combo = 0
    session.notebook_incorrect += 1
    _notebook_update_mistake_count(session, 1)
    session.notebook_status = (
        f"Correct: {session.notebook_correct}  "
        f"Incorrect: {session.notebook_incorrect}  "
        f"Remaining: {len(session.notebook_unseen_boards)}"
    )


def _notebook_delete_current(session) -> None:
    if not session.notebook_pattern or not session.notebook_board_encoded:
        return

    board_key = np_u64(session.notebook_board_encoded)
    mistakes_book_store.remove_mistake(session.notebook_pattern, board_key)
    mistakes_book_store.save()
    session.notebook_unseen_boards = [
        np_u64(board)
        for board in session.notebook_unseen_boards
        if np_u64(board) != board_key
    ]

    available_patterns = _notebook_pattern_list()
    if session.notebook_pattern not in available_patterns:
        session.notebook_pattern = available_patterns[0] if available_patterns else ""
        session.notebook_combo = 0
        session.notebook_correct = 0
        session.notebook_incorrect = 0
        _notebook_reset_unseen_boards(session)

    _notebook_next_problem(session)


async def send_notebook_state(websocket: WebSocket, session) -> None:
    board_encoded = np_u64(session.notebook_board_encoded)
    board_array = decode_board(board_encoded)
    await websocket.send_json(
        {
            "action": Message.NOTEBOOK_STATE,
            "data": {
                "pattern": session.notebook_pattern,
                "patterns": _notebook_pattern_list(),
                "board": board_array.flatten().tolist(),
                "animation": sanitize_config({}),
                "hex_str": safe_hex(session.notebook_board_encoded),
                "weight_mode": int(session.notebook_weight_mode),
                "notebook_threshold": _get_notebook_threshold(),
                "feedback": {
                    "combo": int(session.notebook_combo),
                    "remaining": int(len(session.notebook_unseen_boards)),
                    "correct": int(session.notebook_correct),
                    "incorrect": int(session.notebook_incorrect),
                },
                "status": session.notebook_status,
                "best_move": session.notebook_best_move,
                "answered": bool(session.notebook_answered),
                "last_direction": session.notebook_last_direction,
                "answer_correct": session.notebook_answer_correct,
                "current_count": int(session.notebook_current_count),
            },
        }
    )
