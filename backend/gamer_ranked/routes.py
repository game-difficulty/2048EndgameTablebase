from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.auth.dependencies import client_ip, require_user

from .service import (
    abandon_ranked_run,
    create_ranked_run,
    get_ranked_run,
    heartbeat_ranked_run,
    public_replay,
    submit_ranked_run,
)


router = APIRouter(prefix="/api/gamer", tags=["gamer-ranked"])


class CreateRunRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    lease_token: str = Field(min_length=16, max_length=256)
    # Keep old open browser tabs compatible; those clients always used 10%.
    spawn_rate4: float = 0.1
    replace_run_id: str | None = Field(default=None, max_length=64)
    replace_lease_token: str | None = Field(default=None, max_length=256)


class SubmitRunRequest(BaseModel):
    score: int = Field(ge=0)
    final_board_codes: list[int] = Field(min_length=16, max_length=16)
    record_encoding: str = Field(min_length=1)
    lease_token: str = Field(min_length=16, max_length=256)


class LeaseRequest(BaseModel):
    lease_token: str = Field(min_length=16, max_length=256)


@router.post("/runs")
def create_run(payload: CreateRunRequest, request: Request):
    user = require_user(request)
    try:
        return create_ranked_run(
            user_id=int(user["id"]),
            request_id=payload.request_id,
            ip_address=client_ip(request),
            spawn_rate4=payload.spawn_rate4,
            lease_token=payload.lease_token,
            replace_run_id=payload.replace_run_id,
            replace_lease_token=payload.replace_lease_token,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        if str(exc) == "run_creation_rate_limit":
            raise HTTPException(
                status_code=429, detail=str(exc), headers={"Retry-After": "60"},
            ) from exc
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/runs/{run_id}/heartbeat")
def heartbeat_run(run_id: str, payload: LeaseRequest, request: Request):
    user = require_user(request)
    try:
        return heartbeat_ranked_run(
            run_id=run_id,
            user_id=int(user["id"]),
            lease_token=payload.lease_token,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Ranked run not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/runs/{run_id}/abandon")
def abandon_run(run_id: str, payload: LeaseRequest, request: Request):
    user = require_user(request)
    try:
        return abandon_ranked_run(
            run_id=run_id,
            user_id=int(user["id"]),
            lease_token=payload.lease_token,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Ranked run not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/runs/{run_id}/submit", status_code=202)
def submit_run(run_id: str, payload: SubmitRunRequest, request: Request):
    user = require_user(request)
    try:
        return submit_ranked_run(
            run_id=run_id,
            user_id=int(user["id"]),
            score=payload.score,
            final_board_codes=payload.final_board_codes,
            record_encoding=payload.record_encoding,
            ip_address=client_ip(request),
            lease_token=payload.lease_token,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Ranked run not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Ranked run belongs to another user.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        code = str(exc)
        status = 503 if code in {"queue_full", "user_pending_limit"} else 429
        raise HTTPException(status_code=status, detail=code) from exc


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request):
    user = require_user(request)
    try:
        return get_ranked_run(run_id=run_id, user_id=int(user["id"]))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Ranked run not found.") from exc


@router.get("/replays/{replay_id}")
def get_public_replay(replay_id: str, response: Response):
    try:
        replay = public_replay(replay_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Ranked replay not found.") from exc
    response.headers["Cache-Control"] = "public, max-age=60, stale-while-revalidate=300"
    return {
        "record_encoding": replay["record_blob"],
        "display_name": replay["display_name"],
        "board_key": replay["board_key"],
        "score": int(replay["score"]),
        "max_tile": int(replay["max_tile"]),
        "move_count": int(replay["move_count"]),
        "used_ai": bool(replay["used_ai"]),
    }
