from __future__ import annotations

import hashlib
import hmac
from dataclasses import replace

from .contracts import ProjectDescriptor, ProjectState, PublicProjectView


DIRECTIONS = {"up", "down", "left", "right"}


def _merge_line(values: list[int]) -> tuple[list[int], int]:
    compact = [value for value in values if value]
    merged: list[int] = []
    gained = 0
    index = 0
    while index < len(compact):
        if index + 1 < len(compact) and compact[index] == compact[index + 1]:
            value = compact[index] * 2
            merged.append(value)
            gained += value
            index += 2
        else:
            merged.append(compact[index])
            index += 1
    return merged + [0] * (4 - len(merged)), gained


def _move_board(
    board: tuple[tuple[int, ...], ...], direction: str
) -> tuple[tuple[tuple[int, ...], ...], int]:
    rows = [list(row) for row in board]
    result = [[0] * 4 for _ in range(4)]
    gained = 0
    for index in range(4):
        if direction in {"left", "right"}:
            source = rows[index] if direction == "left" else list(reversed(rows[index]))
        else:
            column = [rows[row][index] for row in range(4)]
            source = column if direction == "up" else list(reversed(column))
        merged, line_gained = _merge_line(source)
        gained += line_gained
        if direction == "right":
            merged.reverse()
        if direction == "down":
            merged.reverse()
        if direction in {"left", "right"}:
            result[index] = merged
        else:
            for row in range(4):
                result[row][index] = merged[row]
    return tuple(tuple(row) for row in result), gained


def _spawn(
    board: tuple[tuple[int, ...], ...], seed: str, counter: int
) -> tuple[tuple[tuple[int, ...], ...], int]:
    empty = [
        (row, column)
        for row in range(4)
        for column in range(4)
        if board[row][column] == 0
    ]
    if not empty:
        return board, counter
    digest = hmac.new(
        bytes.fromhex(seed), f"spawn:{counter}".encode("ascii"), hashlib.sha256
    ).digest()
    row, column = empty[int.from_bytes(digest[:8], "big") % len(empty)]
    value = 4 if digest[8] % 10 == 0 else 2
    mutable = [list(items) for items in board]
    mutable[row][column] = value
    return tuple(tuple(items) for items in mutable), counter + 1


def _has_moves(board: tuple[tuple[int, ...], ...]) -> bool:
    if any(value == 0 for row in board for value in row):
        return True
    for row in range(4):
        for column in range(4):
            value = board[row][column]
            if row + 1 < 4 and board[row + 1][column] == value:
                return True
            if column + 1 < 4 and board[row][column + 1] == value:
                return True
    return False


class Standard2048Adapter:
    """Server-authoritative test project behind the project adapter boundary."""

    project_id = "standard-2048-test"
    rules_version = "standard-v1"

    def __init__(self, *, target_tile: int = 2048):
        self.target_tile = max(4, int(target_tile))

    @property
    def descriptor(self) -> ProjectDescriptor:
        return ProjectDescriptor(
            project_ref=self.project_id,
            rules_version=self.rules_version,
            display_name="Standard 2048 (test adapter)",
            view_kind="2048-board",
            view_protocol="2048-board-v1",
            test_only=True,
        )

    def initial_state(self, *, seed: str) -> ProjectState:
        board = tuple(tuple(0 for _ in range(4)) for _ in range(4))
        board, counter = _spawn(board, seed, 0)
        board, counter = _spawn(board, seed, counter)
        return ProjectState(
            board=board,
            score=0,
            elapsed_ms=0,
            finished=False,
            seed=seed,
            rng_counter=counter,
        )

    def apply_move(self, state: ProjectState, move: str) -> ProjectState:
        direction = str(move).lower()
        if direction not in DIRECTIONS:
            raise ValueError("invalid direction")
        board, gained = _move_board(state.board, direction)
        if board == state.board:
            raise ValueError("move does not change board")
        board, counter = _spawn(board, state.seed, state.rng_counter)
        score = state.score + gained
        reached_target = max(value for row in board for value in row) >= self.target_tile
        no_moves = not _has_moves(board)
        outcome = "target_reached" if reached_target else "no_moves" if no_moves else None
        return replace(
            state,
            board=board,
            score=score,
            finished=reached_target or no_moves,
            outcome=outcome,
            move_count=state.move_count + 1,
            rng_counter=counter,
        )

    def apply_action(self, state: ProjectState, action: dict) -> ProjectState:
        if str(action.get("type") or "") != "move":
            raise ValueError("unsupported project action")
        return self.apply_move(state, str(action.get("direction") or ""))

    def public_payload(self, state: ProjectState) -> dict:
        return {
            "board": [list(row) for row in state.board],
            "score": state.score,
            "move_count": state.move_count,
            "finished": state.finished,
            "outcome": state.outcome,
            "target_tile": self.target_tile,
        }

    def public_view(
        self, state: ProjectState, *, generation: int = 1
    ) -> PublicProjectView:
        return PublicProjectView(
            view_kind=self.descriptor.view_kind,
            view_protocol=self.descriptor.view_protocol,
            generation=max(1, int(generation)),
            sequence=state.move_count,
            payload=self.public_payload(state),
        )

    def verify_result(self, record: bytes, claimed: ProjectState) -> bool:
        # M5 executes moves server-side; external record verification is reserved for
        # future adapters that do not use this authoritative action path.
        return bool(record) and claimed.finished
