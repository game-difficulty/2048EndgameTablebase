"""Server-authoritative adapters for the two later practice projects."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .contracts import ProjectDescriptor, ProjectState, PublicProjectView
from .tournament_variants import (
    DIRECTIONS, WALL, Tournament2048AdapterV2, VariantRules,
    _board_sum, _digest, _move,
)


def _nested(values: list[int], cols: int = 4) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(values[row * cols:(row + 1) * cols]) for row in range(len(values) // cols))


def _flat(board: tuple[tuple[int, ...], ...]) -> list[int]:
    return [value for row in board for value in row]


CORNER_CUTOFF_SEALS = ({2, 5, 8}, {1, 6, 11}, {4, 9, 14}, {7, 10, 13})


class HundredStepSealAdapter(Tournament2048AdapterV2):
    """A separate seal stream; numeric tickets never depend on seal draws."""

    def __init__(self) -> None:
        super().__init__(VariantRules(
            "practice-hundred-step-seal-4x4", "百步封锁（4×4）", 4, 4,
        ))

    @staticmethod
    def _draw_seals(seed: str, previous: list[int], counter: int) -> tuple[list[int], int]:
        candidates = [index for index in range(16) if index not in previous]
        selected: list[int] = []
        for _attempt in range(33):
            available = candidates[:]
            selected = []
            for _ in range(3):
                digest = _digest(seed, f"seal:{counter}")
                counter += 1
                selected.append(available.pop(int.from_bytes(digest[:8], "big") % len(available)))
            if not any(pattern <= set(selected) for pattern in CORNER_CUTOFF_SEALS):
                break
        else:
            replacement = next((
                index for index in candidates
                if index not in selected[:2]
                and not any(pattern <= {*selected[:2], index} for pattern in CORNER_CUTOFF_SEALS)
            ), None)
            if replacement is not None:
                selected[-1] = replacement
        return sorted(selected), counter

    @staticmethod
    def _spawn_unsealed(
        board: tuple[tuple[int, ...], ...], seed: str, counter: int, seals: list[int]
    ) -> tuple[tuple[tuple[int, ...], ...], int, dict[str, int] | None]:
        values = _flat(board)
        empty = [index for index, value in enumerate(values) if value == 0 and index not in seals]
        digest = _digest(seed, f"spawn:{counter}")
        if not empty:
            return board, counter + 1, None
        index = empty[int.from_bytes(digest[:8], "big") % len(empty)]
        value = 4 if digest[8] / 255 < 0.1 else 2
        values[index] = value
        return _nested(values), counter + 1, {"index": index, "value": value}

    @staticmethod
    def _move_with_seals(
        board: tuple[tuple[int, ...], ...], direction: str, seals: list[int]
    ) -> tuple[tuple[tuple[int, ...], ...], int, list[dict[str, Any]]]:
        values = _flat(board)
        masked = _nested([WALL if index in seals else value for index, value in enumerate(values)])
        moved, score, movements = _move(masked, direction, mirror=False, unmergeable=None)
        restored = [values[index] if index in seals else value for index, value in enumerate(_flat(moved))]
        return _nested(restored), score, movements

    def _has_moves_with_seals(self, board: tuple[tuple[int, ...], ...], seals: list[int]) -> bool:
        return any(self._move_with_seals(board, direction, seals)[0] != board for direction in DIRECTIONS)

    def initial_state(self, *, seed: str) -> ProjectState:
        seals, seal_counter = self._draw_seals(seed, [], 0)
        board = _nested([0] * 16)
        board, counter, first = self._spawn_unsealed(board, seed, 0, seals)
        board, counter, second = self._spawn_unsealed(board, seed, counter, seals)
        return ProjectState(
            board=board, score=0, elapsed_ms=0, finished=False, seed=seed,
            rng_counter=counter, extra={
                "sealed_cells": seals, "seal_counter": seal_counter, "seal_round": 1,
                "revision": 0, "last_transition": {
                    "kind": "initial", "spawns": [first, second],
                    "seals": {"released": [], "sealed": seals},
                },
            },
        )

    def initial_state_for_side(self, *, seed: str, side: str) -> ProjectState:
        return self.initial_state(seed=seed)

    def apply_move(self, state: ProjectState, move: str) -> ProjectState:
        direction = str(move).lower()
        if direction not in DIRECTIONS:
            raise ValueError("invalid direction")
        old_seals = list(state.extra["sealed_cells"])
        board, gained, movements = self._move_with_seals(state.board, direction, old_seals)
        if board == state.board:
            raise ValueError("move does not change board")
        move_count = state.move_count + 1
        seals = old_seals
        seal_counter = int(state.extra["seal_counter"])
        seal_round = int(state.extra["seal_round"])
        rotation = None
        if move_count % 100 == 0:
            seals, seal_counter = self._draw_seals(state.seed, old_seals, seal_counter)
            seal_round += 1
            rotation = {"released": old_seals, "sealed": seals}
        board, counter, spawn = self._spawn_unsealed(board, state.seed, state.rng_counter, seals)
        dead = not self._has_moves_with_seals(board, seals)
        return replace(state, board=board, score=state.score + gained,
                       move_count=move_count, rng_counter=counter,
                       finished=dead, outcome="no_moves" if dead else None,
                       extra={**state.extra, "sealed_cells": seals,
                              "seal_counter": seal_counter, "seal_round": seal_round,
                              "revision": int(state.extra["revision"]) + 1,
                              "last_transition": {
                                  "kind": "move", "direction": direction,
                                  "before": _flat(state.board), "movements": movements,
                                  "spawn": spawn, "seals": rotation,
                              }})

    def public_payload(self, state: ProjectState) -> dict[str, Any]:
        return {
            **super().public_payload(state),
            "sealed_cells": list(state.extra["sealed_cells"]),
            "seal_round": int(state.extra["seal_round"]),
            "next_seal_in": 100 - state.move_count % 100,
            "no_moves": not self._has_moves_with_seals(state.board, list(state.extra["sealed_cells"])),
        }


POLY_VECTORS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}


def _poly_shift(cells: list[int], direction: str, rows: int = 4, cols: int = 4) -> list[int] | None:
    dr, dc = POLY_VECTORS[direction]
    shifted = []
    for cell in cells:
        row, col = divmod(cell, cols)
        row += dr
        col += dc
        if not 0 <= row < rows or not 0 <= col < cols:
            return None
        shifted.append(row * cols + col)
    return sorted(set(shifted))


def _poly_move(tiles: list[dict[str, Any]], direction: str, rows: int = 4, cols: int = 4) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], list[dict[str, Any]], bool]:
    if direction not in POLY_VECTORS:
        raise ValueError("invalid direction")
    def lead(tile: dict[str, Any]) -> int:
        positions = [cell % cols if direction in {"left", "right"} else cell // cols for cell in tile["cells"]]
        return min(positions) if direction in {"left", "up"} else -max(positions)
    ordered = sorted(tiles, key=lambda tile: (lead(tile), str(tile["id"])))
    settled: dict[str, dict[str, Any]] = {}
    occupied: dict[int, dict[str, Any]] = {}
    movements: list[dict[str, Any]] = []
    merges: list[dict[str, Any]] = []
    score = 0
    changed = False
    for tile in ordered:
        cells = list(tile["cells"])
        merged = False
        while True:
            candidate = _poly_shift(cells, direction, rows, cols)
            if candidate is None:
                break
            blockers = list({occupied[cell]["id"]: occupied[cell] for cell in candidate if cell in occupied}.values())
            if not blockers:
                cells = candidate
                continue
            if len(blockers) != 1:
                break
            target = blockers[0]
            if target.get("merged") or target["value"] != tile["value"] or tile["value"] >= 256:
                break
            intersection = [cell for cell in candidate if cell in target["cells"]]
            if not intersection:
                break
            merge_position = candidate
            if tile["value"] == 128 and len(intersection) == 1:
                aligned = _poly_shift(candidate, direction, rows, cols)
                if aligned is not None and all(cell in target["cells"] for cell in aligned):
                    merge_position = aligned
            if tile["value"] == 128:
                result_cells = sorted(set(merge_position + target["cells"]))
            elif tile["value"] == 64:
                result_cells = sorted(set(cells + target["cells"]))
            else:
                result_cells = list(target["cells"])
            result = {"id": f"merge-{target['id']}-{tile['id']}", "value": tile["value"] * 2,
                      "cells": result_cells, "merged": True}
            for cell in target["cells"]:
                occupied.pop(cell, None)
            settled.pop(target["id"])
            settled[result["id"]] = result
            for cell in result_cells:
                occupied[cell] = result
            movements.append({"id": tile["id"], "from": list(tile["cells"]),
                              "to": merge_position, "mergeInto": result["id"]})
            for movement in movements:
                if movement["id"] == target["id"]:
                    movement["mergeInto"] = result["id"]
                    break
            merges.append({"tile": {key: value for key, value in result.items() if key != "merged"},
                           "sources": [target["id"], tile["id"]]})
            score += result["value"]
            changed = merged = True
            break
        if merged:
            continue
        if cells != tile["cells"]:
            changed = True
        result = {"id": tile["id"], "value": tile["value"], "cells": cells}
        settled[result["id"]] = result
        for cell in cells:
            occupied[cell] = result
        movements.append({"id": tile["id"], "from": list(tile["cells"]), "to": cells})
    output = [{key: value for key, value in tile.items() if key != "merged"} for tile in settled.values()]
    return output, score, movements, merges, changed


class GrowingTilesAdapter:
    project_id = "practice-growing-tiles-4x4"
    rules_version = "tournament-v2"
    rows = 4
    cols = 4

    @property
    def descriptor(self) -> ProjectDescriptor:
        return ProjectDescriptor(self.project_id, self.rules_version, f"越来越大（{self.rows}×{self.cols}）",
                                 "polyomino-board", "polyomino-board-v1")

    def _board(self, tiles: list[dict[str, Any]]) -> tuple[tuple[int, ...], ...]:
        board = [0] * (self.rows * self.cols)
        for tile in tiles:
            for cell in tile["cells"]:
                board[cell] = tile["value"]
        return _nested(board, self.cols)

    def _spawn(self, tiles: list[dict[str, Any]], seed: str, counter: int, next_id: int) -> tuple[dict[str, Any] | None, int, int]:
        occupied = {cell for tile in tiles for cell in tile["cells"]}
        empty = [index for index in range(self.rows * self.cols) if index not in occupied]
        digest = _digest(seed, f"spawn:{counter}")
        if not empty:
            return None, counter + 1, next_id
        cell = empty[int.from_bytes(digest[:8], "big") % len(empty)]
        value = 4 if digest[8] / 255 < 0.1 else 2
        tile = {"id": f"tile-{next_id}", "value": value, "cells": [cell]}
        tiles.append(tile)
        return tile, counter + 1, next_id + 1

    def initial_state(self, *, seed: str) -> ProjectState:
        tiles: list[dict[str, Any]] = []
        first, counter, next_id = self._spawn(tiles, seed, 0, 0)
        second, counter, next_id = self._spawn(tiles, seed, counter, next_id)
        return ProjectState(self._board(tiles), 0, 0, False, seed=seed, rng_counter=counter,
                            extra={"tiles": tiles, "next_tile_id": next_id, "revision": 0,
                                   "last_transition": {"kind": "initial", "spawns": [first, second]}})

    def apply_move(self, state: ProjectState, move: str) -> ProjectState:
        before = list(state.extra["tiles"])
        tiles, gained, movements, merges, changed = _poly_move(before, str(move).lower(), self.rows, self.cols)
        if not changed:
            raise ValueError("move does not change board")
        spawned, counter, next_id = self._spawn(tiles, state.seed, state.rng_counter,
                                                 int(state.extra["next_tile_id"]))
        dead = not any(_poly_move(tiles, direction, self.rows, self.cols)[-1] for direction in DIRECTIONS)
        return replace(state, board=self._board(tiles), score=state.score + gained,
                       move_count=state.move_count + 1, rng_counter=counter,
                       finished=dead, outcome="no_moves" if dead else None,
                       extra={**state.extra, "tiles": tiles, "next_tile_id": next_id,
                              "revision": int(state.extra["revision"]) + 1,
                              "last_transition": {"kind": "move", "before": before,
                                                  "movements": movements, "merges": merges,
                                                  "spawn": spawned}})

    def apply_action(self, state: ProjectState, action: dict[str, Any]) -> ProjectState:
        if str(action.get("type") or "").lower() != "move":
            raise ValueError("unsupported project action")
        return self.apply_move(state, str(action.get("direction") or ""))

    def result_value(self, state: ProjectState) -> int:
        return state.score

    def resolve_winner(self, yellow: ProjectState, white: ProjectState) -> tuple[str, str]:
        winner = "yellow" if yellow.score > white.score else "white" if white.score > yellow.score else "draw"
        return winner, "score"

    def public_payload(self, state: ProjectState) -> dict[str, Any]:
        return {"board": [list(row) for row in state.board], "rows": self.rows, "cols": self.cols,
                "tiles": state.extra["tiles"], "score": state.score,
                "board_sum": sum(tile["value"] for tile in state.extra["tiles"]),
                "move_count": state.move_count, "elapsed_ms": state.elapsed_ms,
                "finished": state.finished, "outcome": state.outcome,
                "last_transition": state.extra["last_transition"]}

    def public_view(self, state: ProjectState, *, generation: int = 1) -> PublicProjectView:
        return PublicProjectView(self.descriptor.view_kind, self.descriptor.view_protocol,
                                 max(1, int(generation)), int(state.extra["revision"]),
                                 self.public_payload(state))

    def verify_result(self, record: bytes, claimed: ProjectState) -> bool:
        return bool(record) and claimed.finished


class GrowingTilesAdapterV3(GrowingTilesAdapter):
    rules_version = "tournament-v3"
    cols = 5


PRACTICE_ADAPTER_FACTORIES = (HundredStepSealAdapter, GrowingTilesAdapter, GrowingTilesAdapterV3)
