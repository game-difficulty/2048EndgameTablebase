from __future__ import annotations

import asyncio
import hmac
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from .auth import auth_user_exists, principal_from_request
from .db import SCHEMA_VERSION
from .domain import Principal
from .errors import CompetitionError
from .schemas import (
    AssignStaffRequest,
    BlindPickRequest,
    ClaimSeatRequest,
    ClientGameStateRequest,
    CommandRequest,
    ManageMemberRequest,
    CreateCompetitionRequest,
    CreateEventRequest,
    LinkEventRoomRequest,
    UpdateEventRequest,
    AssignEventOrganizerRequest,
    StatisticsRosterRequest,
    EnrollmentRosterRequest,
    EnrollmentActionRequest,
    GamePhaseRequest,
    GameReadinessRequest,
    ForceAdvanceRequest,
    ForceFinishRequest,
    IssueReportRequest,
    IssueResolutionRequest,
    LineupRequest,
    PickBanRequest,
    PracticeResultRequest,
    ReadinessRequest,
    ResultConfirmationRequest,
    ResultOverrideRequest,
    SuspendMatchRequest,
    SuspensionReadinessRequest,
)


router = APIRouter(prefix="/api")


@router.post('/competitions/{room_code}/members/manage')
async def manage_member(request: Request, room_code: str, payload: ManageMemberRequest):
    principal = current_principal(request)
    room = await asyncio.to_thread(service_from_request(request).manage_member, room_code, principal,
        user_id=payload.user_id, remove=payload.remove, command_id=payload.command_id)
    await _broadcast(request, room['room_code'])
    return {'competition': room}


def service_from_request(request: Request):
    return request.app.state.competition_service


def hub_from_request(request: Request):
    return request.app.state.competition_hub


def current_principal(request: Request) -> Principal:
    return principal_from_request(request, request.app.state.competition_settings)


PrincipalDependency = Annotated[Principal, Depends(current_principal)]


def optional_principal(request: Request) -> Principal | None:
    try:
        return current_principal(request)
    except HTTPException as exc:
        if exc.status_code != 401:
            raise
        return None


@router.get("/preferences/language")
async def preferred_language(request: Request, response: Response) -> dict:
    """Read the shared account preference; this site never writes it back."""
    response.headers["Cache-Control"] = "private, no-store"
    principal = optional_principal(request)
    if principal is None:
        return {"language": None}
    from backend.profile.preferences import get_preferences
    result = await asyncio.to_thread(get_preferences, principal.user_id)
    language = result.get("preferences", {}).get("language")
    return {"language": language if language in ("zh", "en") else None}


@router.get("/practice/appearance")
async def practice_appearance(principal: PrincipalDependency, response: Response) -> dict:
    """Read only this player's main-site tile appearance for practice boards."""
    from backend.profile.preferences import get_preferences
    from backend.profile.themes import ThemeError, get_theme

    def load() -> dict:
        preferences = get_preferences(principal.user_id)["preferences"]
        appearance = {
            key: preferences[key]
            for key in ("theme", "use_custom_theme", "custom_colors", "font_size_factor", "saved_theme_id")
            if key in preferences
        }
        saved_id = preferences.get("saved_theme_id")
        if type(saved_id) is int and saved_id > 0:
            try:
                appearance["saved_theme"] = get_theme(principal.user_id, saved_id)["theme"]
            except ThemeError as exc:
                if exc.status != 404:
                    raise
        return appearance

    response.headers["Cache-Control"] = "private, no-store"
    return await asyncio.to_thread(load)


@router.get("/practice/{project_id}/leaderboard")
async def practice_leaderboard(request: Request, project_id: str) -> dict:
    viewer = optional_principal(request)
    return await asyncio.to_thread(request.app.state.practice_leaderboard.list, project_id, viewer)


@router.post("/practice/{project_id}/results")
async def submit_practice_result(
    request: Request, project_id: str, result: PracticeResultRequest,
    principal: PrincipalDependency,
) -> dict:
    return await asyncio.to_thread(
        request.app.state.practice_leaderboard.submit, project_id, principal,
        score=result.score, board_sum=result.board_sum,
        elapsed_ms=result.elapsed_ms, outcome=result.outcome,
    )


async def _broadcast(request: Request, room_code: str) -> None:
    service = service_from_request(request)
    hub = hub_from_request(request)
    await hub.broadcast(room_code, lambda viewer: service.snapshot(room_code, viewer))


@router.get("/health")
async def health(request: Request) -> dict:
    return {
        "ok": True,
        "service": "competition",
        "schema_version": SCHEMA_VERSION,
    }


def require_live_internal(request: Request) -> None:
    expected = request.app.state.competition_settings.live_internal_token
    supplied = request.headers.get("x-competition-live-token", "")
    if not expected or not hmac.compare_digest(expected, supplied):
        raise CompetitionError(
            "LIVE_INTERNAL_AUTH_REQUIRED", "Live internal authentication required.", 401
        )


@router.get("/internal/live/rooms")
async def internal_live_rooms(request: Request) -> dict:
    require_live_internal(request)
    rooms = await asyncio.to_thread(service_from_request(request).list_live_rooms)
    return {"rooms": rooms}


@router.get("/internal/live/rooms/{public_key}")
async def internal_live_projection(request: Request, public_key: str, after_yellow: int = 0, after_white: int = 0) -> dict:
    require_live_internal(request)
    projection = await asyncio.to_thread(
        service_from_request(request).live_projection, public_key,
        after_yellow=max(0, after_yellow), after_white=max(0, after_white),
    )
    return {"projection": projection}


@router.get("/internal/live/settlement/{public_key}")
async def internal_prediction_facts(request: Request, public_key: str) -> dict:
    require_live_internal(request)
    facts = await asyncio.to_thread(service_from_request(request).live_prediction_facts, public_key)
    return {"projection": facts}


@router.get("/session")
async def session(request: Request, principal: PrincipalDependency) -> dict:
    service = service_from_request(request)
    return {
        "authenticated": True,
        "user": {
            "id": principal.user_id,
            "display_name": principal.display_name,
            "site_role": principal.site_role,
        },
        "can_create_competition": service._can_create_competition(principal),
    }


@router.get("/competitions")
async def list_competitions(request: Request, principal: PrincipalDependency) -> dict:
    service = service_from_request(request)
    rooms = await asyncio.to_thread(service.list_competitions, principal)
    return {"competitions": rooms}


@router.get('/events')
async def list_events(request: Request):
    principal = optional_principal(request)
    service = service_from_request(request)
    return {'events': await asyncio.to_thread(service.events.list, principal),
            'can_create': bool(principal and service._is_platform_organizer(principal))}


@router.get('/events/{slug}')
async def event_detail(request: Request, slug: str):
    return {'event': await asyncio.to_thread(service_from_request(request).events.detail, slug, optional_principal(request))}


@router.post('/events', status_code=201)
async def create_event(request: Request, payload: CreateEventRequest, principal: PrincipalDependency):
    return {'event': await asyncio.to_thread(service_from_request(request).events.create, principal, **payload.model_dump())}


@router.post('/events/{slug}/rooms')
async def link_event_room(request: Request, slug: str, payload: LinkEventRoomRequest, principal: PrincipalDependency):
    result = await asyncio.to_thread(service_from_request(request).events.link, slug, payload.room_code, principal)
    await _broadcast(request, payload.room_code.strip().upper())
    return {'event': result}


@router.post('/events/{slug}/settings')
async def update_event(request: Request, slug: str, payload: UpdateEventRequest, principal: PrincipalDependency):
    return {'event': await asyncio.to_thread(service_from_request(request).events.update, slug, principal, **payload.model_dump())}


@router.post('/events/{slug}/organizer')
async def assign_event_organizer(request: Request, slug: str, payload: AssignEventOrganizerRequest, principal: PrincipalDependency):
    return await asyncio.to_thread(service_from_request(request).events.assign_organizer, slug, principal, **payload.model_dump())


@router.get('/events/{slug}/statistics')
async def event_statistics(request: Request, slug: str, response: Response):
    response.headers['Cache-Control'] = 'no-store'
    return await asyncio.to_thread(service_from_request(request).events.statistics.standings, slug)


@router.get('/events/{slug}/enrollment')
async def event_enrollment(request: Request, slug: str, response: Response):
    response.headers['Cache-Control'] = 'private, no-store'
    return await asyncio.to_thread(service_from_request(request).events.enrollment.snapshot, slug, optional_principal(request))


@router.post('/events/{slug}/enrollment/actions')
async def enrollment_action(request: Request, slug: str, payload: EnrollmentActionRequest, principal: PrincipalDependency):
    return await asyncio.to_thread(service_from_request(request).events.enrollment.action, slug, principal, **payload.model_dump(exclude_none=True))


@router.post('/events/{slug}/enrollment/import')
async def enrollment_import(request: Request, slug: str, payload: EnrollmentRosterRequest, principal: PrincipalDependency):
    return await asyncio.to_thread(service_from_request(request).events.enrollment.import_roster, slug, principal, **payload.model_dump())


@router.post('/events/{slug}/roster')
async def import_event_roster(request: Request, slug: str, payload: StatisticsRosterRequest, principal: PrincipalDependency):
    return await asyncio.to_thread(service_from_request(request).events.statistics.import_roster, slug, principal, **payload.model_dump())


@router.post("/competitions", status_code=201)
async def create_competition(
    request: Request,
    payload: CreateCompetitionRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.create_competition,
        principal,
        name=payload.name,
        event_slug=payload.event_slug,
        starts_at=payload.starts_at,
        yellow_team_id=payload.yellow_team_id,
        white_team_id=payload.white_team_id,
        room_code=payload.room_code,
        projects=[
            {
                "key": item.key,
                "name": item.name,
                "description": item.description,
                "project_ref": item.project_ref,
                "adapter_rules_version": item.adapter_rules_version,
                "rules_version": item.rules_version,
            }
            for item in payload.projects
        ] if payload.projects is not None else None,
    )
    return {"competition": room}


@router.get("/competitions/{room_code}")
async def competition_snapshot(
    request: Request, room_code: str, principal: PrincipalDependency
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(service.snapshot, room_code, principal)
    return {"competition": room}


@router.post('/competitions/{room_code}/check-in')
async def competition_check_in(request: Request, room_code: str, principal: PrincipalDependency):
    service = service_from_request(request)
    room = await asyncio.to_thread(service.schedule.check_in, room_code, principal)
    await _broadcast(request, room_code)
    return {'competition': room}


@router.post("/competitions/{room_code}/close")
async def close_competition(
    request: Request, room_code: str, payload: CommandRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.close_competition, room_code, principal,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post('/competitions/{room_code}/rematch')
async def rematch_room(request: Request, room_code: str, payload: CommandRequest, principal: PrincipalDependency):
    room = await asyncio.to_thread(service_from_request(request).rematch_before_lineup,room_code,principal,command_id=payload.command_id)
    await _broadcast(request, room_code)
    return {'competition':room}


@router.post("/competitions/{room_code}/staff")
async def assign_staff(
    request: Request,
    room_code: str,
    payload: AssignStaffRequest,
    principal: PrincipalDependency,
) -> dict:
    if not await asyncio.to_thread(auth_user_exists, payload.user_id):
        raise CompetitionError("USER_NOT_FOUND", "Active user not found.", 404)
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.assign_staff,
        room_code,
        principal,
        user_id=payload.user_id,
        role=payload.role,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/seat")
async def claim_seat(
    request: Request,
    room_code: str,
    payload: ClaimSeatRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.claim_seat,
        room_code,
        principal,
        side=payload.side,
        position=payload.position,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/seat/leave")
async def leave_seat(
    request: Request,
    room_code: str,
    payload: CommandRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.leave_seat,
        room_code,
        principal,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/ready")
async def set_ready(
    request: Request,
    room_code: str,
    payload: ReadinessRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.set_ready,
        room_code,
        principal,
        ready=payload.ready,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/draft/pick-ban")
async def submit_pick_ban(
    request: Request,
    room_code: str,
    payload: PickBanRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.submit_pick_ban,
        room_code,
        principal,
        pick_project_key=payload.pick_project_key,
        ban_project_key=payload.ban_project_key,
        phase_token=payload.phase_token,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/draft/blind")
async def submit_blind_pick(
    request: Request,
    room_code: str,
    payload: BlindPickRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.submit_blind_pick,
        room_code,
        principal,
        project_key=payload.project_key,
        phase_token=payload.phase_token,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/lineup")
async def submit_lineup(
    request: Request,
    room_code: str,
    payload: LineupRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.submit_lineup,
        room_code,
        principal,
        assignments=payload.assignments,
        phase_token=payload.phase_token,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/games/current/readiness")
async def set_game_readiness(
    request: Request,
    room_code: str,
    payload: GameReadinessRequest,
    principal: PrincipalDependency,
) -> dict:
    if payload.ready and payload.stream_protocol != 'project-stream-v2':
        raise HTTPException(426, detail={'code': 'CLIENT_UPDATE_REQUIRED', 'message': '请刷新页面后重新准备，当前客户端不支持连续观战。'})
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.set_game_readiness,
        room_code,
        principal,
        readiness_role=payload.readiness_role,
        ready=payload.ready,
        phase_token=payload.phase_token,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/games/current/start")
async def start_current_game(
    request: Request,
    room_code: str,
    payload: GamePhaseRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.start_current_game,
        room_code,
        principal,
        phase_token=payload.phase_token,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/games/current/state")
async def sync_client_game(
    request: Request, room_code: str, payload: ClientGameStateRequest,
    principal: PrincipalDependency,
) -> dict:
    raise HTTPException(426, detail={'code': 'CLIENT_UPDATE_REQUIRED', 'message': '请刷新页面，比赛已升级为连续步骤传输。'})


@router.post("/competitions/{room_code}/games/current/result/confirm")
async def confirm_current_result(
    request: Request,
    room_code: str,
    payload: ResultConfirmationRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.confirm_current_result,
        room_code,
        principal,
        result_revision=payload.result_revision,
        phase_token=payload.phase_token,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/issues")
async def report_issue(
    request: Request,
    room_code: str,
    payload: IssueReportRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.report_issue,
        room_code,
        principal,
        category=payload.category,
        details=payload.details,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/issues/{issue_id}/resolve")
async def resolve_issue(
    request: Request,
    room_code: str,
    issue_id: str,
    payload: IssueResolutionRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.resolve_issue,
        room_code,
        principal,
        issue_id=issue_id,
        status=payload.status,
        resolution_note=payload.resolution_note,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/suspension/start")
async def suspend_match(
    request: Request,
    room_code: str,
    payload: SuspendMatchRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.suspend_match,
        room_code,
        principal,
        reason_code=payload.reason_code,
        reason_text=payload.reason_text,
        phase_token=payload.phase_token,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/suspension/readiness")
async def set_suspension_readiness(
    request: Request,
    room_code: str,
    payload: SuspensionReadinessRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.set_suspension_readiness,
        room_code,
        principal,
        ready=payload.ready,
        phase_token=payload.phase_token,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/suspension/resume")
async def resume_match(
    request: Request,
    room_code: str,
    payload: GamePhaseRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.resume_match,
        room_code,
        principal,
        phase_token=payload.phase_token,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/games/current/result/override")
async def override_current_result(
    request: Request,
    room_code: str,
    payload: ResultOverrideRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.override_current_result,
        room_code,
        principal,
        yellow_score=payload.yellow_score,
        white_score=payload.white_score,
        winner_side=payload.winner_side,
        reason=payload.reason,
        result_revision=payload.result_revision,
        phase_token=payload.phase_token,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/games/current/result/force-advance")
async def force_advance_current_result(
    request: Request,
    room_code: str,
    payload: ForceAdvanceRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.force_advance_current_result,
        room_code,
        principal,
        reason=payload.reason,
        phase_token=payload.phase_token,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}


@router.post("/competitions/{room_code}/force-finish")
async def force_finish_match(
    request: Request,
    room_code: str,
    payload: ForceFinishRequest,
    principal: PrincipalDependency,
) -> dict:
    service = service_from_request(request)
    room = await asyncio.to_thread(
        service.force_finish_match,
        room_code,
        principal,
        winner_side=payload.winner_side,
        reason=payload.reason,
        phase_token=payload.phase_token,
        command_id=payload.command_id,
    )
    await _broadcast(request, room["room_code"])
    return {"competition": room}
