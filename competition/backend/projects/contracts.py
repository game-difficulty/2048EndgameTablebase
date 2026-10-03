from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
from .result_policy import ResultPolicy


@dataclass(frozen=True)
class ProjectState:
    board: tuple[tuple[int, ...], ...]
    score: int
    elapsed_ms: int
    finished: bool
    outcome: str | None = None
    seed: str = ""
    move_count: int = 0
    rng_counter: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectDescriptor:
    """Stable metadata frozen into a competition project-pool snapshot."""

    project_ref: str
    rules_version: str
    display_name: str
    view_kind: str
    view_protocol: str
    test_only: bool = False
    result_policy: ResultPolicy = field(default_factory=ResultPolicy)

    def snapshot(self) -> dict[str, Any]:
        return {
            "project_ref": self.project_ref,
            "rules_version": self.rules_version,
            "display_name": self.display_name,
            "view_kind": self.view_kind,
            "view_protocol": self.view_protocol,
            "test_only": self.test_only,
        }


@dataclass(frozen=True)
class PublicProjectView:
    view_kind: str
    view_protocol: str
    generation: int
    sequence: int
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "view_kind": self.view_kind,
            "view_protocol": self.view_protocol,
            "generation": self.generation,
            "sequence": self.sequence,
            "payload": self.payload,
        }


class ProjectAdapter(Protocol):
    """Boundary for standard 2048 and future board-rule variants."""

    project_id: str
    rules_version: str

    @property
    def descriptor(self) -> ProjectDescriptor: ...

    def initial_state(self, *, seed: str) -> ProjectState: ...

    def apply_move(self, state: ProjectState, move: str) -> ProjectState: ...

    def apply_action(self, state: ProjectState, action: dict[str, Any]) -> ProjectState: ...

    def public_payload(self, state: ProjectState) -> dict[str, Any]: ...

    def public_view(
        self, state: ProjectState, *, generation: int = 1
    ) -> PublicProjectView: ...

    def verify_result(self, record: bytes, claimed: ProjectState) -> bool: ...
