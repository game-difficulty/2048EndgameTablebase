from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response

from backend.leaderboards.service import (
    LEADERBOARD_LIMIT,
    leaderboard_catalog,
    leaderboard_payload,
)


router = APIRouter(prefix="/api/leaderboards", tags=["leaderboards"])


def _set_public_cache(
    response: Response,
    *,
    max_age: int = 300,
    stale_while_revalidate: int | None = 3600,
) -> None:
    cache_control = f"public, max-age={max_age}"
    if stale_while_revalidate is not None:
        cache_control += f", stale-while-revalidate={stale_while_revalidate}"
    else:
        cache_control += ", must-revalidate"
    response.headers["Cache-Control"] = cache_control


def _set_live_cache(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


@router.get("")
def get_leaderboard_catalog(response: Response):
    _set_public_cache(response, stale_while_revalidate=None)
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
    is_event_driven = board_key == "supporters" or board_key.startswith("gamer_")
    if is_event_driven:
        _set_live_cache(response)
    else:
        _set_public_cache(response)
    return payload
