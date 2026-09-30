"""Client-executed games: metadata and opaque state projection, never a replay."""
from __future__ import annotations

from typing import Any
from .projects.contracts import ProjectState, PublicProjectView

PROTOCOL = "client-runtime-v1"


def initial_state(adapter, seed: str) -> ProjectState:
    rules = getattr(adapter, "rules", None)
    rows = int(getattr(adapter, "rows", getattr(rules, "rows", 4)))
    cols = int(getattr(adapter, "cols", getattr(rules, "cols", 4)))
    return ProjectState(
        board=tuple(tuple(0 for _ in range(cols)) for _ in range(rows)),
        score=0, elapsed_ms=0, finished=False, seed=seed,
        extra={"client_sequence": 0,
               "checkpoint": None, "client_payload": {},
               "result_value": 0},
    )


def public_payload(state: ProjectState) -> dict[str, Any]:
    return {**state.extra.get("client_payload", {}),
            "board": [list(row) for row in state.board],
            "rows": len(state.board), "cols": len(state.board[0]),
            "score": state.score, "move_count": state.move_count,
            "elapsed_ms": state.elapsed_ms, "finished": state.finished,
            "outcome": state.outcome,
            "awaiting_client": state.extra.get("checkpoint") is None}


def public_view(adapter, state: ProjectState, generation: int) -> dict[str, Any]:
    return PublicProjectView(
        adapter.descriptor.view_kind, adapter.descriptor.view_protocol,
        generation, int(state.extra.get("client_sequence", 0)), public_payload(state),
    ).as_dict()


def resolve_result(yellow: ProjectState, white: ProjectState, *, race: bool) -> tuple[int, int, str, str]:
    y, w = int(yellow.extra.get("result_value", yellow.score)), int(white.extra.get("result_value", white.score))
    if yellow.outcome == 'surrendered':
        return y, w, 'white', 'yellow_surrendered'
    if white.outcome == 'surrendered':
        return y, w, 'yellow', 'white_surrendered'
    if race:
        yt, wt = yellow.outcome == "target_reached", white.outcome == "target_reached"
        if yt != wt:
            return y, w, "yellow" if yt else "white", "race_target"
        if yt and wt:
            winner = "yellow" if yellow.elapsed_ms < white.elapsed_ms else "white" if white.elapsed_ms < yellow.elapsed_ms else "draw"
            return y, w, winner, "race_elapsed"
    return y, w, "yellow" if y > w else "white" if w > y else "draw", "score"
