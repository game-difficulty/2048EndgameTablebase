from __future__ import annotations

from pydantic import BaseModel, Field


class PracticeResultRequest(BaseModel):
    score: int
    board_sum: int
    elapsed_ms: int
    outcome: str


class ProjectInput(BaseModel):
    key: str | None = None
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=240)
    project_ref: str = Field(min_length=1, max_length=80)
    adapter_rules_version: str = Field(min_length=1, max_length=80)
    rules_version: str = Field(min_length=1, max_length=80)


class CreateCompetitionRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    room_code: str | None = None
    projects: list[ProjectInput] | None = None


class AssignStaffRequest(BaseModel):
    user_id: int = Field(gt=0)
    role: str


class ClaimSeatRequest(BaseModel):
    side: str
    position: int
    command_id: str


class CommandRequest(BaseModel):
    command_id: str


class ReadinessRequest(BaseModel):
    ready: bool
    command_id: str


class PickBanRequest(BaseModel):
    pick_project_key: str
    ban_project_key: str
    phase_token: str
    command_id: str


class BlindPickRequest(BaseModel):
    project_key: str
    phase_token: str
    command_id: str


class LineupRequest(BaseModel):
    assignments: dict[str, int]
    phase_token: str
    command_id: str


class GameReadinessRequest(BaseModel):
    readiness_role: str
    ready: bool
    phase_token: str
    command_id: str


class GamePhaseRequest(BaseModel):
    phase_token: str
    command_id: str


class GameMoveRequest(BaseModel):
    direction: str
    phase_token: str
    command_id: str


class GameActionRequest(BaseModel):
    action: str
    phase_token: str
    command_id: str


class ResultConfirmationRequest(BaseModel):
    result_revision: int = Field(gt=0)
    phase_token: str
    command_id: str


class IssueReportRequest(BaseModel):
    category: str
    details: str = Field(min_length=3, max_length=500)
    command_id: str


class IssueResolutionRequest(BaseModel):
    status: str
    resolution_note: str = Field(min_length=3, max_length=500)
    command_id: str


class SuspendMatchRequest(BaseModel):
    reason_code: str
    reason_text: str = Field(min_length=3, max_length=500)
    phase_token: str
    command_id: str


class SuspensionReadinessRequest(BaseModel):
    ready: bool
    phase_token: str
    command_id: str


class ResultOverrideRequest(BaseModel):
    yellow_score: int = Field(ge=0)
    white_score: int = Field(ge=0)
    winner_side: str
    reason: str = Field(min_length=3, max_length=500)
    result_revision: int = Field(gt=0)
    phase_token: str
    command_id: str


class ForceAdvanceRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=500)
    phase_token: str
    command_id: str


class ForceFinishRequest(BaseModel):
    winner_side: str
    reason: str = Field(min_length=3, max_length=500)
    phase_token: str
    command_id: str
