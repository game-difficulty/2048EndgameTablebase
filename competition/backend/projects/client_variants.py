"""Descriptors for client-executed tournament projects 13–20.

The match service accepts versioned client checkpoints for these projects;
the server adapter supplies frozen project metadata and public-view routing.
Practice leaderboards remain separate from match adjudication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import ProjectDescriptor, ProjectState, PublicProjectView


@dataclass(frozen=True)
class ClientVariantRules:
    rows: int = 4
    cols: int = 4
    race: bool = False
    result_metric: str = "score"


class ClientVariantAdapter:
    rules_version = "tournament-v4"
    rules = ClientVariantRules()

    def __init__(self, project_id: str, display_name: str, *, special_tiles: bool = False,
                 rows: int = 4, cols: int = 4, rules_version: str = "tournament-v4"):
        self.project_id = project_id
        self.display_name = display_name
        self.special_tiles = special_tiles
        self.rules = ClientVariantRules(rows=rows, cols=cols)
        self.rules_version = rules_version

    @property
    def descriptor(self) -> ProjectDescriptor:
        kind = "polyomino-board" if self.special_tiles else "2048-board"
        protocol = "polyomino-board-v1" if self.special_tiles else "2048-board-v2"
        return ProjectDescriptor(self.project_id, self.rules_version, self.display_name, kind, protocol)

    def initial_state(self, *, seed: str) -> ProjectState:
        return ProjectState(tuple(tuple(0 for _ in range(self.rules.cols)) for _ in range(self.rules.rows)),
                            0, 0, False, seed=seed)

    def apply_move(self, state: ProjectState, move: str) -> ProjectState:
        raise NotImplementedError("This project is executed by the versioned match client.")

    def apply_action(self, state: ProjectState, action: dict[str, Any]) -> ProjectState:
        raise NotImplementedError("This project is executed by the versioned match client.")

    def result_value(self, state: ProjectState) -> int:
        return state.score

    def resolve_winner(self, yellow: ProjectState, white: ProjectState) -> tuple[str, str]:
        side = "yellow" if yellow.score > white.score else "white" if white.score > yellow.score else "draw"
        return side, "score"

    def public_payload(self, state: ProjectState) -> dict[str, Any]:
        return {
            "board": [list(row) for row in state.board], "rows": self.rules.rows, "cols": self.rules.cols,
            "score": state.score, "move_count": state.move_count,
            "elapsed_ms": state.elapsed_ms, "finished": state.finished,
            "outcome": state.outcome,
        }

    def public_view(self, state: ProjectState, *, generation: int = 1) -> PublicProjectView:
        return PublicProjectView(self.descriptor.view_kind, self.descriptor.view_protocol,
                                 generation, state.move_count, self.public_payload(state))

    def verify_result(self, record: bytes, claimed: ProjectState) -> bool:
        return bool(record) and claimed.finished


CLIENT_VARIANT_DEFINITIONS = (
    ("practice-pair-bond-4x4", "出双入对（4×4）", True),
    ("practice-chemical-reaction-4x4", "化学反应（4×4）", True),
    ("practice-timed-bomb-4x4", "定时炸弹（4×4）", True),
    ("practice-full-load-4x4", "满载（4×4）", False),
    ("practice-heavy-tiles-4x4", "越来越重（4×4）", False),
    ("practice-fission-4x4", "裂变（4×4）", False),
    ("practice-aftershock-4x4", "余震（4×4）", False, 4, 4, "tournament-v5"),
    ("practice-look-back-3x4", "回头看看（3×4）", False, 3, 4, "tournament-v5"),
)

CLIENT_VARIANT_ADAPTER_FACTORIES = tuple(
    (lambda selected=item: ClientVariantAdapter(
        selected[0], selected[1], special_tiles=selected[2],
        rows=selected[3] if len(selected) > 3 else 4,
        cols=selected[4] if len(selected) > 4 else 4,
        rules_version=selected[5] if len(selected) > 5 else "tournament-v4"))
    for item in CLIENT_VARIANT_DEFINITIONS
)
