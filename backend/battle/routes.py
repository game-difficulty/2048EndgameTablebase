from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response

from backend.auth.dependencies import require_user
from backend.quota.errors import InsufficientTokens
from backend.quota.service import get_token_balance

from . import repository
from .service import (
    BattleServiceError,
    broadcast_room,
    create_room,
    current_room,
    forfeit_round,
    join_room,
    kick,
    leave_room,
    list_rooms,
    player_replay_payload,
    room_snapshot,
    route_payload,
    set_ready,
    set_role,
    start_room,
)


router = APIRouter(prefix="/api/battle", tags=["battle"])


def _raise_service_error(exc: Exception) -> None:
    if isinstance(exc, InsufficientTokens):
        raise HTTPException(status_code=402, detail=exc.payload) from exc
    if isinstance(exc, BattleServiceError):
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    if isinstance(exc, repository.BattleNotFoundError):
        raise HTTPException(status_code=404, detail={"code": "ROOM_NOT_FOUND", "message": "Room not found."}) from exc
    if isinstance(exc, repository.BattlePermissionError):
        raise HTTPException(status_code=403, detail={"code": str(exc).upper(), "message": str(exc).replace("_", " ")}) from exc
    if isinstance(exc, (repository.BattleConflictError, ValueError)):
        raise HTTPException(status_code=409, detail={"code": str(exc).upper(), "message": str(exc).replace("_", " ")}) from exc
    raise exc


@router.get("/me")
async def get_current_battle(user: dict = Depends(require_user)) -> dict[str, Any]:
    return {"room": current_room(user_id=int(user["id"]))}


@router.get("/rooms")
async def get_public_rooms(user: dict = Depends(require_user)) -> dict[str, Any]:
    return {"rooms": list_rooms(user_id=int(user["id"]))}


@router.post("/rooms")
async def create_battle_room(
    payload: dict = Body(...),
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        return await create_room(
            user_id=int(user["id"]),
            session_id=(int(user["session_id"]) if user.get("session_id") else None),
            payload=payload,
        )
    except Exception as exc:
        _raise_service_error(exc)
        raise


@router.get("/rooms/{room_code}")
async def get_battle_room(
    room_code: str,
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        return {"room": room_snapshot(room_code, user_id=int(user["id"]))}
    except Exception as exc:
        _raise_service_error(exc)
        raise


@router.post("/rooms/{room_code}/join")
async def join_battle_room(
    room_code: str,
    payload: dict = Body(default={}),
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        room = join_room(
            room_code,
            user_id=int(user["id"]),
            role=(str(payload.get("role")) if payload.get("role") else None),
        )
        await broadcast_room(str(room["room_id"]))
        return {"room": room}
    except Exception as exc:
        _raise_service_error(exc)
        raise


@router.post("/rooms/{room_code}/leave")
async def leave_battle_room(
    room_code: str,
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        room = repository.get_room(room_code)
        leave_room(room_code, user_id=int(user["id"]))
        await broadcast_room(str(room["room_id"]))
        return {
            "left": True,
            "token_balance": get_token_balance(int(user["id"])),
        }
    except Exception as exc:
        _raise_service_error(exc)
        raise


@router.post("/rooms/{room_code}/ready")
async def ready_battle_room(
    room_code: str,
    payload: dict = Body(...),
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        room = set_ready(room_code, user_id=int(user["id"]), ready=bool(payload.get("ready")))
        await broadcast_room(str(room["room_id"]))
        return {"room": room}
    except Exception as exc:
        _raise_service_error(exc)
        raise


@router.post("/rooms/{room_code}/role")
async def role_battle_room(
    room_code: str,
    payload: dict = Body(...),
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        room = set_role(room_code, user_id=int(user["id"]), role=str(payload.get("role") or ""))
        await broadcast_room(str(room["room_id"]))
        return {"room": room}
    except Exception as exc:
        _raise_service_error(exc)
        raise


@router.post("/rooms/{room_code}/kick")
async def kick_battle_member(
    room_code: str,
    payload: dict = Body(...),
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        room = kick(
            room_code,
            host_user_id=int(user["id"]),
            target_user_id=int(payload.get("user_id")),
        )
        await broadcast_room(str(room["room_id"]))
        return {"room": room}
    except Exception as exc:
        _raise_service_error(exc)
        raise


@router.post("/rooms/{room_code}/start")
async def start_battle_room(
    room_code: str,
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        room = await start_room(
            room_code,
            user_id=int(user["id"]),
            session_id=(int(user["session_id"]) if user.get("session_id") else None),
        )
        return {"room": room, "token_balance": get_token_balance(int(user["id"]))}
    except Exception as exc:
        _raise_service_error(exc)
        raise


@router.post("/rooms/{room_code}/rounds/{round_id}/forfeit")
async def forfeit_battle_round(
    room_code: str,
    round_id: str,
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        room = forfeit_round(
            room_code,
            user_id=int(user["id"]),
            round_id=round_id,
        )
        await broadcast_room(str(room["room_id"]))
        return {"room": room}
    except Exception as exc:
        _raise_service_error(exc)
        raise


@router.get("/rooms/{room_code}/rounds/{round_id}/replay")
async def download_own_battle_replay(
    room_code: str,
    round_id: str,
    user: dict = Depends(require_user),
) -> Response:
    try:
        blob, metadata = player_replay_payload(
            room_code,
            round_id,
            user_id=int(user["id"]),
        )
    except Exception as exc:
        _raise_service_error(exc)
        raise
    return Response(
        blob,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": f'attachment; filename="{metadata["filename"]}"',
            "X-Replay-Pattern": str(metadata["full_pattern"]),
            "X-Replay-Variant": "1" if metadata["use_variant"] else "0",
            "X-Replay-Source": "Battle",
            "X-Replay-Moves": str(metadata["move_count"]),
        },
    )


def _artifact_response(blob: bytes, metadata: dict[str, Any]) -> Response:
    artifact_hash = str(
        metadata.get("artifact_hash") or metadata.get("route_hash") or ""
    )
    headers = {
        "Cache-Control": "private, no-store",
        "X-Battle-Artifact-Kind": str(metadata.get("artifact_kind") or "binary"),
    }
    if artifact_hash:
        headers["ETag"] = f'"{artifact_hash}"'
    if metadata.get("step_count") is not None:
        headers["X-Battle-Route-Steps"] = str(metadata["step_count"])
    if "certainty_step" in metadata:
        headers["X-Battle-Certainty-Step"] = (
            "" if metadata["certainty_step"] is None else str(metadata["certainty_step"])
        )
    if metadata.get("termination_reason") is not None:
        headers["X-Battle-Termination"] = str(metadata["termination_reason"])
    return Response(blob, media_type="application/octet-stream", headers=headers)


async def _download_battle_artifact(
    room_code: str,
    round_id: str,
    user: dict,
) -> Response:
    try:
        blob, metadata = route_payload(
            room_code, round_id, user_id=int(user["id"])
        )
    except Exception as exc:
        _raise_service_error(exc)
        raise
    return _artifact_response(blob, metadata)


@router.get("/rooms/{room_code}/rounds/{round_id}/artifact")
async def download_battle_artifact(
    room_code: str,
    round_id: str,
    user: dict = Depends(require_user),
) -> Response:
    return await _download_battle_artifact(room_code, round_id, user)


@router.get("/rooms/{room_code}/rounds/{round_id}/route")
async def download_battle_route(
    room_code: str,
    round_id: str,
    user: dict = Depends(require_user),
) -> Response:
    """Compatibility alias for the original goodness route endpoint."""
    return await _download_battle_artifact(room_code, round_id, user)
