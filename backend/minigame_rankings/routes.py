from __future__ import annotations

from collections import defaultdict, deque
import threading
import time

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from backend.auth.dependencies import client_ip, require_user

from .service import (
    LEADERBOARD_LIMIT,
    RunTokenError,
    RunTokenExpired,
    abandon_ranked_run,
    claim_ranked_run,
    create_ranked_run,
    game_leaderboard,
    get_ranked_run,
    heartbeat_ranked_run,
    minigame_catalog,
    qualify_ranked_run,
    submit_ranked_run,
    trophy_leaderboard,
)


router = APIRouter(prefix="/api/minigame-rankings", tags=["minigame-rankings"])
_RATE_WINDOW_SECONDS = 60
_USER_RATE_LIMIT = 30
_IP_RATE_LIMIT = 60
_rate_events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
_rate_lock = threading.Lock()


class ScoreSubmission(BaseModel):
    game_id: str = Field(min_length=1, max_length=64)
    difficulty: int = Field(ge=0, le=1)
    score: int = Field(ge=0, le=2_147_483_647)
    trophy_tier: int = Field(ge=0, le=4)
    highest_tile_exp: int = Field(ge=0, le=63)
    final_board: list[int] = Field(min_length=1, max_length=64)
    board_rows: int = Field(ge=1, le=8)
    board_cols: int = Field(ge=1, le=8)


class CreateRankedRunRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    game_id: str = Field(min_length=1, max_length=64)
    difficulty: int = Field(ge=0, le=1)
    lease_token: str = Field(min_length=16, max_length=256)
    replace_run_id: str | None = Field(default=None, max_length=64)
    replace_lease_token: str | None = Field(default=None, max_length=256)


class LeaseRequest(BaseModel):
    lease_token: str = Field(min_length=16, max_length=256)


class QualifyRankedRunRequest(BaseModel):
    run_token: str = Field(min_length=16, max_length=2048)
    lease_token: str = Field(min_length=16, max_length=256)
    score: int = Field(ge=0, le=2**63 - 1)
    trophy_tier: int = Field(ge=0, le=4)
    highest_tile_exp: int = Field(ge=0, le=63)
    final_board: list[int] = Field(min_length=1, max_length=64)
    board_rows: int = Field(ge=1, le=8)
    board_cols: int = Field(ge=1, le=8)
    action_count: int = Field(ge=0, le=50_000)
    elapsed_ms: int = Field(ge=0, le=7 * 24 * 60 * 60 * 1000)


class SubmitRankedRunRequest(BaseModel):
    submission_token: str = Field(min_length=16, max_length=2048)
    lease_token: str = Field(min_length=16, max_length=256)
    record_encoding: str = Field(min_length=1, max_length=400_000)


def _check_submit_rate(user_id: int, ip_address: str) -> None:
    now = time.monotonic()
    buckets = (
        (("user", str(int(user_id))), _USER_RATE_LIMIT),
        (("ip", str(ip_address or "unknown")), _IP_RATE_LIMIT),
    )
    with _rate_lock:
        cutoff = now - _RATE_WINDOW_SECONDS
        for key, limit in buckets:
            events = _rate_events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                raise HTTPException(status_code=429, detail="Too many score submissions.")
        for key, _limit in buckets:
            _rate_events[key].append(now)


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


@router.get("/catalog")
def get_catalog(response: Response):
    response.headers["Cache-Control"] = "public, max-age=3600"
    return {"games": minigame_catalog()}


@router.get("/overall")
def get_overall(
    response: Response,
    difficulty: int = Query(1, ge=0, le=1),
    limit: int = Query(LEADERBOARD_LIMIT, ge=1, le=LEADERBOARD_LIMIT),
):
    _no_store(response)
    return trophy_leaderboard(difficulty=difficulty, limit=limit)


@router.get("/games/{game_id}")
def get_game(
    game_id: str,
    response: Response,
    difficulty: int = Query(1, ge=0, le=1),
    limit: int = Query(LEADERBOARD_LIMIT, ge=1, le=LEADERBOARD_LIMIT),
):
    try:
        payload = game_leaderboard(game_id, difficulty=difficulty, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Minigame not found.") from exc
    _no_store(response)
    return payload


@router.post("/scores")
def post_score(payload: ScoreSubmission, request: Request, response: Response):
    require_user(request)
    raise HTTPException(status_code=410, detail="legacy_score_submission_disabled")


@router.post("/runs")
def create_run(
    payload: CreateRankedRunRequest,
    request: Request,
    response: Response,
):
    user = require_user(request)
    _check_submit_rate(int(user["id"]), client_ip(request))
    try:
        result = create_ranked_run(
            user_id=int(user["id"]),
            request_id=payload.request_id,
            game_id=payload.game_id,
            difficulty=payload.difficulty,
            ip_address=client_ip(request),
            lease_token=payload.lease_token,
            replace_run_id=payload.replace_run_id,
            replace_lease_token=payload.replace_lease_token,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _no_store(response)
    return result


@router.post("/runs/{run_id}/heartbeat")
def heartbeat_run(
    run_id: str,
    payload: LeaseRequest,
    request: Request,
    response: Response,
):
    user = require_user(request)
    try:
        result = heartbeat_ranked_run(
            run_id=run_id,
            user_id=int(user["id"]),
            lease_token=payload.lease_token,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Ranked run not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RunTokenExpired as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _no_store(response)
    return result


@router.post("/runs/{run_id}/claim")
def claim_run(
    run_id: str,
    payload: LeaseRequest,
    request: Request,
    response: Response,
):
    user = require_user(request)
    try:
        result = claim_ranked_run(
            run_id=run_id,
            user_id=int(user["id"]),
            lease_token=payload.lease_token,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Ranked run not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RunTokenExpired as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _no_store(response)
    return result


@router.post("/runs/{run_id}/abandon")
def abandon_run(
    run_id: str,
    payload: LeaseRequest,
    request: Request,
    response: Response,
):
    user = require_user(request)
    try:
        result = abandon_ranked_run(
            run_id=run_id,
            user_id=int(user["id"]),
            lease_token=payload.lease_token,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Ranked run not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _no_store(response)
    return result


@router.post("/runs/{run_id}/qualify")
def qualify_run(
    run_id: str,
    payload: QualifyRankedRunRequest,
    request: Request,
    response: Response,
):
    user = require_user(request)
    _check_submit_rate(int(user["id"]), client_ip(request))
    try:
        result = qualify_ranked_run(
            run_id=run_id,
            user_id=int(user["id"]),
            **payload.model_dump(),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Ranked run not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Ranked run belongs to another user.") from exc
    except RunTokenExpired as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except (RunTokenError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _no_store(response)
    return result


@router.post("/runs/{run_id}/submit", status_code=202)
def submit_run(
    run_id: str,
    payload: SubmitRankedRunRequest,
    request: Request,
    response: Response,
):
    user = require_user(request)
    _check_submit_rate(int(user["id"]), client_ip(request))
    try:
        result = submit_ranked_run(
            run_id=run_id,
            user_id=int(user["id"]),
            submission_token=payload.submission_token,
            lease_token=payload.lease_token,
            record_encoding=payload.record_encoding,
            ip_address=client_ip(request),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Ranked run not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Ranked run belongs to another user.") from exc
    except RunTokenExpired as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except RunTokenError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _no_store(response)
    return result


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request, response: Response):
    user = require_user(request)
    try:
        result = get_ranked_run(run_id=run_id, user_id=int(user["id"]))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Ranked run not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Ranked run belongs to another user.") from exc
    _no_store(response)
    return result
