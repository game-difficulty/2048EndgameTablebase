from __future__ import annotations

from pydantic import BaseModel, Field, StrictInt, model_validator
from typing import Literal


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
    event_slug: str | None = Field(default=None, max_length=64)
    starts_at: str | None = Field(default=None, max_length=64)
    yellow_team_id: str | None = Field(default=None, max_length=150)
    white_team_id: str | None = Field(default=None, max_length=150)
    rules: dict | None = None


class CreateDuelRequest(BaseModel):
    model_config = {'extra': 'forbid'}
    name: str = Field(min_length=2, max_length=100)
    projects: list[str] = Field(min_length=1, max_length=15)
    command_id: str = Field(min_length=8, max_length=160)
    clock_seconds: int = Field(default=1800, ge=30, le=86400, strict=True)
    predictions_enabled: bool = Field(default=False, strict=True)


class CreateTimeAttackRequest(BaseModel):
    model_config = {'extra': 'forbid'}
    name: str = Field(min_length=2, max_length=100)
    variant: Literal['4x4', '3x4', '2x4', '3x3']
    target_kind: Literal['tile', 'board_sum']
    target_value: int = Field(strict=True)
    clock_seconds: int = Field(default=600, ge=30, le=86400, strict=True)
    command_id: str = Field(min_length=8, max_length=160)
    predictions_enabled: bool = Field(default=False, strict=True)


class TimeAttackCommand(BaseModel):
    model_config = {'extra': 'forbid'}
    attempt_id: str = Field(min_length=1, max_length=64)
    command_id: str = Field(min_length=8, max_length=160)
    action: Literal['submit', 'restart']
    base_sequence: int = Field(default=0, ge=0, strict=True)
    events: list[list[StrictInt]] | None = Field(default=None, max_length=64)


class CreateEventRequest(BaseModel):
    team_size: int = Field(default=3, ge=1, le=16)
    slug: str = Field(min_length=2, max_length=64)
    name: str = Field(min_length=2, max_length=100)
    description: str = Field(default='', max_length=1000)
    rules: str = Field(default='', max_length=10000)


class LinkEventRoomRequest(BaseModel):
    room_code: str = Field(min_length=1, max_length=12)


class AssignEventOrganizerRequest(BaseModel):
    user_id: int = Field(gt=0)
    dry_run: bool = True


class UpdateEventRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    description: str = Field(default='', max_length=1000)
    rules: str = Field(default='', max_length=10000)
    status: str = Field(max_length=20)


class StatisticsRosterEntry(BaseModel):
    user_id: int | None = Field(default=None, gt=0)
    username: str | None = Field(default=None, min_length=1, max_length=100)
    team_name: str = Field(default='', max_length=40)
    is_external: bool = False
    captain: bool = False
    position: int | None = Field(default=None, ge=1, le=16)

    @model_validator(mode='after')
    def identity(self):
        if (self.user_id is None) == (self.username is None):
            raise ValueError('Provide exactly one of user_id or username')
        if self.username is not None and not self.username.strip():
            raise ValueError('Username cannot be blank')
        return self


class EnrollmentRosterRequest(BaseModel):
    entries: list[StatisticsRosterEntry] = Field(min_length=1, max_length=500)
    revision: int = Field(ge=0)
    dry_run: bool = True


class EnrollmentActionRequest(BaseModel):
    action: Literal['signup','withdraw','create_team','invite','accept_invite','decline_invite','cancel_invite',
                    'leave_team','disband_team','submit_team','unsubmit_team','settings','lock_registration','lock_roster','unlock']
    revision: int = Field(ge=0)
    team_id: str | None = Field(default=None, max_length=150)
    user_id: int | None = Field(default=None, gt=0)
    name: str | None = Field(default=None, max_length=40)
    mode: Literal['solo','self_team','organizer_team'] | None = None
    capacity: int | None = Field(default=None, ge=0, le=10000)
    registration_open: bool | None = None
    reason: str | None = Field(default=None, max_length=500)


class StatisticsRosterRequest(BaseModel):
    entries: list[StatisticsRosterEntry] = Field(min_length=20, max_length=20)
    revision: int = Field(ge=0)
    dry_run: bool = True


class AssignStaffRequest(BaseModel):
    user_id: int = Field(gt=0)
    role: str


class ClaimSeatRequest(BaseModel):
    side: str
    position: int
    command_id: str


class CommandRequest(BaseModel):
    command_id: str


class ManageMemberRequest(CommandRequest):
    user_id: int = Field(gt=0)
    remove: bool = True


class ReadinessRequest(BaseModel):
    ready: bool
    command_id: str


class PickBanRequest(BaseModel):
    pick_project_key: str
    ban_project_key: str
    phase_token: str
    command_id: str


class DraftStepRequest(BaseModel):
    picks: list[str] = Field(default_factory=list, max_length=15)
    bans: list[str] = Field(default_factory=list, max_length=15)
    phase_token: str = Field(max_length=100)
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
    stream_protocol: str = ''
    readiness_role: str
    ready: bool
    phase_token: str
    command_id: str


class GamePhaseRequest(BaseModel):
    phase_token: str
    command_id: str


class ClientGameStateRequest(BaseModel):
    instance_id: str = Field(min_length=1, max_length=100)
    sequence: int = Field(ge=1, le=2**53 - 1)
    phase_token: str = Field(max_length=100)
    payload: dict
    checkpoint: dict
    frames: list[dict] = Field(default_factory=list, max_length=128)
    result_value: int = Field(ge=0, le=10**15)
    elapsed_ms: int = Field(ge=0, le=10**9)
    finished: bool
    outcome: str | None = Field(default=None, max_length=40)


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
