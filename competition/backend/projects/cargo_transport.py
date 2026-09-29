"""Server-authoritative rules for project 01, 真华容道."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .contracts import ProjectDescriptor, ProjectState, PublicProjectView
from .tournament_variants import WALL, _digest, _move


LIMIT_MS = 600_000
FIRST_CARGO_MOVE = 10
SHAPES = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
)
VECTORS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}


def _cells(cargo: dict[str, int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (cargo["row"] + dr, cargo["col"] + dc)
        for dr, dc in SHAPES[cargo["shape"]]
    )


def _flat(board: tuple[tuple[int, ...], ...]) -> list[int]:
    return [value for row in board for value in row]


def _nested(board: list[int]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(board[row * 4:(row + 1) * 4]) for row in range(4))


def _cargo_board_indices(cargo: dict[str, int] | None) -> set[int]:
    return {row * 4 + col for row, col in _cells(cargo) if 0 <= row < 4} if cargo else set()


def _can_shift(cargo: dict[str, int], direction: str, board: list[int]) -> bool:
    if cargo["row"] < 0 and direction != "down":
        return False
    # A piece partly through the outlet may still align sideways; only moving
    # back upward into the board is forbidden.
    if cargo["row"] >= 3 and direction == "up":
        return False
    dr, dc = VECTORS[direction]
    next_cargo = {**cargo, "row": cargo["row"] + dr, "col": cargo["col"] + dc}
    if next_cargo["row"] < 0 and direction != "down":
        return False
    for row, col in _cells(next_cargo):
        if not 0 <= col < 4:
            return False
        if row >= 4 and col not in (1, 2):
            return False
        if 0 <= row < 4 and board[row * 4 + col] != 0:
            return False
    return True


def _move_cargo_board(
    board: tuple[tuple[int, ...], ...], cargo: dict[str, int] | None, direction: str
) -> tuple[tuple[tuple[int, ...], ...], dict[str, int] | None, list[dict[str, Any]], bool]:
    occupied = _cargo_board_indices(cargo)
    masked = _nested([WALL if index in occupied else value for index, value in enumerate(_flat(board))])
    moved, _gained, movements = _move(masked, direction, mirror=False, unmergeable=None)
    next_flat = [0 if value == WALL else value for value in _flat(moved)]
    dr, dc = VECTORS[direction]
    next_cargo = cargo
    while next_cargo is not None and _can_shift(next_cargo, direction, next_flat):
        next_cargo = {**next_cargo, "row": next_cargo["row"] + dr,
                      "col": next_cargo["col"] + dc}
        if next_cargo["row"] >= 4:
            break
    cargo_moved = next_cargo is not cargo
    return _nested(next_flat), next_cargo, movements, cargo_moved


def _has_cargo_move(
    board: tuple[tuple[int, ...], ...], cargo: dict[str, int] | None
) -> bool:
    for direction in VECTORS:
        moved, _shifted, _movements, cargo_moved = _move_cargo_board(
            board, cargo, direction
        )
        if moved != board or cargo_moved:
            return True
    return False


class CargoTransportAdapter:
    project_id = "tournament-cargo-transport-4x4"
    rules_version = "tournament-v1"
    time_limit_ms = LIMIT_MS

    @property
    def descriptor(self) -> ProjectDescriptor:
        return ProjectDescriptor(
            project_ref=self.project_id,
            rules_version=self.rules_version,
            display_name="真华容道（4×4）",
            view_kind="cargo-transport",
            view_protocol="cargo-transport-v1",
        )

    def _spawn_number(
        self, board: tuple[tuple[int, ...], ...], cargo: dict[str, int] | None,
        seed: str, counter: int,
    ) -> tuple[tuple[tuple[int, ...], ...], int, dict[str, int] | None]:
        flat = _flat(board)
        occupied = _cargo_board_indices(cargo) if cargo else set()
        empty = [index for index, value in enumerate(flat) if value == 0 and index not in occupied]
        if not empty:
            return board, counter, None
        digest = _digest(seed, f"cargo-spawn:{counter}")
        index = empty[int.from_bytes(digest[:8], "big") % len(empty)]
        value = 4 if digest[8] / 255 < 0.1 else 2
        flat[index] = value
        return _nested(flat), counter + 1, {"index": index, "value": value}

    def _next_cargo(self, seed: str, counter: int) -> tuple[dict[str, int], int]:
        digest = _digest(seed, f"cargo-shape:{counter}")
        return {
            "id": counter,
            "shape": int.from_bytes(digest[:8], "big") % len(SHAPES),
            "row": -2,
            "col": 1,
        }, counter + 1

    def initial_state(self, *, seed: str) -> ProjectState:
        board = _nested([0] * 16)
        board, counter, first = self._spawn_number(board, None, seed, 0)
        board, counter, second = self._spawn_number(board, None, seed, counter)
        return ProjectState(
            board=board, score=0, elapsed_ms=0, finished=False, seed=seed,
            rng_counter=counter,
            extra={
                "cargo": None, "revision": 0,
                "last_transition": {"kind": "initial", "spawns": [first, second]},
            },
        )

    def apply_move(self, state: ProjectState, move: str) -> ProjectState:
        direction = str(move).lower()
        if direction not in VECTORS:
            raise ValueError("invalid direction")
        if state.finished:
            raise ValueError("project is already complete")
        if state.elapsed_ms >= LIMIT_MS:
            return replace(
                state, finished=True, outcome="time_limit", elapsed_ms=LIMIT_MS,
                extra={**state.extra,
                       "revision": int(state.extra.get("revision", state.move_count)) + 1,
                       "last_transition": {"kind": "time_limit"}},
            )
        cargo_before = dict(state.extra["cargo"]) if state.extra.get("cargo") else None
        board, shifted, movements, cargo_moved = _move_cargo_board(state.board, cargo_before, direction)
        if board == state.board and not cargo_moved:
            raise ValueError("move does not change board")
        delivered = cargo_moved and shifted is not None and shifted["row"] >= 4
        separate_shapes = "shape_counter" in state.extra
        shape_counter = int(state.extra["shape_counter"]) if separate_shapes else state.rng_counter
        next_cargo, shape_counter = (
            self._next_cargo(state.seed, shape_counter)
            if delivered else (shifted, shape_counter)
        )
        counter = state.rng_counter if separate_shapes else shape_counter
        board, counter, spawn = self._spawn_number(board, next_cargo, state.seed, counter)
        if next_cargo is None and state.move_count + 1 >= FIRST_CARGO_MOVE:
            next_cargo, shape_counter = self._next_cargo(
                state.seed, shape_counter if separate_shapes else counter
            )
            if not separate_shapes:
                counter = shape_counter
        dead = not _has_cargo_move(board, next_cargo)
        extra = {
            **state.extra,
            "cargo": next_cargo,
            **({"shape_counter": shape_counter} if separate_shapes else {}),
            "revision": int(state.extra.get("revision", state.move_count)) + 1,
            "last_transition": {
                "kind": "move", "direction": direction,
                "before": _flat(state.board),
                "movements": movements, "spawn": spawn,
                "cargoBefore": cargo_before, "cargoMoved": cargo_moved,
                "cargoExit": shifted if delivered else None,
            },
        }
        return replace(
            state, board=board, score=state.score + int(delivered),
            move_count=state.move_count + 1, rng_counter=counter,
            finished=dead, outcome="no_moves" if dead else None, extra=extra,
        )

    def apply_action(self, state: ProjectState, action: dict[str, Any]) -> ProjectState:
        if str(action.get("type") or "").lower() != "move":
            raise ValueError("unsupported project action")
        return self.apply_move(state, str(action.get("direction") or ""))

    def result_value(self, state: ProjectState) -> int:
        return state.score

    def resolve_winner(self, yellow: ProjectState, white: ProjectState) -> tuple[str, str]:
        winner = (
            "yellow" if yellow.score > white.score else
            "white" if white.score > yellow.score else "draw"
        )
        return winner, "delivered_cargo"

    def public_payload(self, state: ProjectState) -> dict[str, Any]:
        return {
            "board": [list(row) for row in state.board],
            "rows": 4, "cols": 4,
            "score": state.score, "deliveries": state.score,
            "move_count": state.move_count,
            "cargo": state.extra.get("cargo"),
            "elapsed_ms": min(LIMIT_MS, state.elapsed_ms),
            "time_limit_ms": LIMIT_MS,
            "remaining_ms": max(0, LIMIT_MS - state.elapsed_ms),
            "finished": state.finished,
            "outcome": state.outcome,
            "last_transition": state.extra.get("last_transition"),
        }

    def public_view(self, state: ProjectState, *, generation: int = 1) -> PublicProjectView:
        return PublicProjectView(
            view_kind=self.descriptor.view_kind,
            view_protocol=self.descriptor.view_protocol,
            generation=max(1, int(generation)),
            sequence=int(state.extra.get("revision", state.move_count)),
            payload=self.public_payload(state),
        )

    def verify_result(self, record: bytes, claimed: ProjectState) -> bool:
        return bool(record) and claimed.finished


class CargoTransportAdapterV2(CargoTransportAdapter):
    rules_version = "tournament-v2"

    def _spawn_number(
        self, board: tuple[tuple[int, ...], ...], cargo: dict[str, int] | None,
        seed: str, counter: int,
    ) -> tuple[tuple[tuple[int, ...], ...], int, dict[str, int] | None]:
        spawned, next_counter, event = super()._spawn_number(board, cargo, seed, counter)
        return spawned, max(counter + 1, next_counter), event

    def initial_state(self, *, seed: str) -> ProjectState:
        state = super().initial_state(seed=seed)
        return replace(state, extra={**state.extra, "shape_counter": 0})
