from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response

from backend.leaderboards.service import (
    LEADERBOARD_LIMIT,
    leaderboard_catalog,
    leaderboard_payload,
)


router = APIRouter(prefix="/api/leaderboards", tags=["leaderboards"])


def _set_public_cache(response: Response) -> None:
    response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=3600"


@router.get("")
def get_leaderboard_catalog(response: Response):
    _set_public_cache(response)
    return {"boards": leaderboard_catalog()}


@router.get("/{board_key}")
def get_leaderboard(
    board_key: str,
    response: Response,
    limit: int = Query(LEADERBOARD_LIMIT, ge=1, le=LEADERBOARD_LIMIT),
):
    try:
        payload = leaderboard_payload(board_key, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Leaderboard not found.") from exc
    _set_public_cache(response)
    return payload
