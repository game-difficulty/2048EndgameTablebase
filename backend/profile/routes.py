from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
import sqlite3
import threading
import time

from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from backend.auth.db import auth_db
from backend.auth.dependencies import client_ip, require_user
from backend.auth.service import public_user

from .service import (
    AvatarUploadDisabledError,
    DisplayNameTakenError,
    ProfileCooldownError,
    assert_avatar_change_allowed,
    delete_avatar,
    update_avatar,
    update_display_name,
)
from .storage import (
    AvatarValidationError,
    get_avatar_input_limit,
    process_avatar_bytes,
    resolve_avatar_key,
)


router = APIRouter(tags=["profile"])
_RATE_WINDOW_SECONDS = 60 * 60
_RATE_LIMITS = {"avatar": 10, "display_name": 20}
_rate_events: dict[tuple[str, int, str], deque[float]] = defaultdict(deque)
_rate_lock = threading.Lock()


def _check_rate_limit(kind: str, user_id: int, ip_address: str) -> None:
    now = time.monotonic()
    key = (kind, int(user_id), str(ip_address or ""))
    limit = _RATE_LIMITS[kind]
    with _rate_lock:
        events = _rate_events[key]
        cutoff = now - _RATE_WINDOW_SECONDS
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= limit:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "PROFILE_RATE_LIMIT",
                    "message": "Too many profile update attempts. Please try again later.",
                },
            )
        events.append(now)


def _profile_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ProfileCooldownError):
        available = datetime.fromisoformat(exc.available_at).astimezone(timezone.utc)
        remaining = max(
            1,
            int((available - datetime.now(timezone.utc)).total_seconds()),
        )
        return HTTPException(
            status_code=429,
            detail={
                "code": "PROFILE_CHANGE_COOLDOWN",
                "message": "This profile field cannot be changed yet.",
                "field": exc.field,
                "available_at": exc.available_at,
                "remaining_seconds": remaining,
            },
        )
    if isinstance(exc, DisplayNameTakenError):
        return HTTPException(
            status_code=409,
            detail={"code": "DISPLAY_NAME_TAKEN", "message": str(exc)},
        )
    if isinstance(exc, AvatarUploadDisabledError):
        return HTTPException(
            status_code=403,
            detail={"code": "AVATAR_UPLOAD_DISABLED", "message": str(exc)},
        )
    if isinstance(exc, AvatarValidationError):
        return HTTPException(
            status_code=400,
            detail={"code": "INVALID_AVATAR", "message": str(exc)},
        )
    return HTTPException(
        status_code=400,
        detail={"code": "INVALID_PROFILE_UPDATE", "message": str(exc)},
    )


def _updated_public_user(user_id: int) -> dict:
    with auth_db() as db:
        row = db.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="User not found.")
        return public_user(row, db=db)


async def _read_avatar_upload(upload: UploadFile) -> bytes:
    limit = get_avatar_input_limit()
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = await upload.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise AvatarValidationError("Avatar image exceeds the upload size limit.")
            chunks.append(chunk)
    finally:
        await upload.close()
    return b"".join(chunks)


@router.patch("/api/profile/display-name")
async def change_display_name(request: Request, payload: dict = Body(...)):
    user = require_user(request)
    user_id = int(user["id"])
    ip_address = client_ip(request)
    _check_rate_limit("display_name", user_id, ip_address)
    try:
        update_display_name(
            user_id,
            str(payload.get("display_name") or ""),
            ip_address=ip_address,
            user_agent=request.headers.get("user-agent", ""),
        )
        return {"user": _updated_public_user(user_id)}
    except (ValueError, sqlite3.IntegrityError) as exc:
        raise _profile_error(exc) from exc


@router.put("/api/profile/avatar")
async def change_avatar(
    request: Request,
    avatar: UploadFile = File(...),
):
    user = require_user(request)
    user_id = int(user["id"])
    ip_address = client_ip(request)
    _check_rate_limit("avatar", user_id, ip_address)
    try:
        assert_avatar_change_allowed(user_id)
        data = await _read_avatar_upload(avatar)
        processed = await asyncio.to_thread(process_avatar_bytes, data)
        update_avatar(
            user_id,
            processed,
            ip_address=ip_address,
            user_agent=request.headers.get("user-agent", ""),
        )
        return {"user": _updated_public_user(user_id)}
    except ValueError as exc:
        raise _profile_error(exc) from exc


@router.delete("/api/profile/avatar")
async def remove_avatar(request: Request):
    user = require_user(request)
    user_id = int(user["id"])
    ip_address = client_ip(request)
    _check_rate_limit("avatar", user_id, ip_address)
    try:
        delete_avatar(
            user_id,
            ip_address=ip_address,
            user_agent=request.headers.get("user-agent", ""),
        )
        return {"user": _updated_public_user(user_id)}
    except ValueError as exc:
        raise _profile_error(exc) from exc


@router.get("/media/avatars/{user_id}/{filename}")
async def get_avatar(user_id: int, filename: str):
    try:
        path = resolve_avatar_key(
            f"{int(user_id)}/{filename}",
            expected_user_id=int(user_id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Avatar not found.") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Avatar not found.")
    return FileResponse(
        path,
        media_type="image/webp",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )
