from __future__ import annotations

from collections import defaultdict, deque
import threading
import time

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from backend.auth.dependencies import client_ip, require_user

from .service import (
    LEADERBOARD_LIMIT,
    game_leaderboard,
    minigame_catalog,
    submit_score,
    trophy_leaderboard,
)


router = APIRouter(prefix="/api/minigame-rankings", tags=["minigame-rankings"])
_RATE_WINDOW_SECONDS = 60
_RATE_LIMIT = 30
_rate_events: dict[tuple[int, str], deque[float]] = defaultdict(deque)
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


def _check_submit_rate(user_id: int, ip_address: str) -> None:
    now = time.monotonic()
    key = (int(user_id), str(ip_address or ""))
    with _rate_lock:
        events = _rate_events[key]
        cutoff = now - _RATE_WINDOW_SECONDS
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= _RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Too many score submissions.")
        events.append(now)


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
    user = require_user(request)
    _check_submit_rate(int(user["id"]), client_ip(request))
    try:
        result = submit_score(user_id=int(user["id"]), **payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _no_store(response)
    return result
