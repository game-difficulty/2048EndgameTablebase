"""Client-executed games: metadata and opaque state projection, never a replay."""
from __future__ import annotations

from typing import Any
from backend.stream_snapshots import compact_public_view
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


def public_view(adapter, state: ProjectState, generation: int, after_sequence: int = 0, *, recovery: bool = False) -> dict[str, Any]:
    view = PublicProjectView(
        adapter.descriptor.view_kind, adapter.descriptor.view_protocol,
        generation, int(state.extra.get("client_sequence", 0)), public_payload(state),
    ).as_dict()
    frames = state.extra.get('frames', [])
    view['frame_start'] = frames[0]['sequence'] if frames else view['sequence']
    view['frames'] = [frame for frame in frames if frame['sequence'] > after_sequence]
    if recovery:
        return compact_public_view(view)
    return view


def resolve_result(yellow: ProjectState, white: ProjectState, *, race: bool) -> tuple[int, int, str, str]:
    from .projects.result_policy import ResultPolicy
    return ResultPolicy(race=race).compare(yellow, white)
