from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request, WebSocket

from .config import CompetitionSettings
from .domain import Principal


def _dev_principal(raw: str | None) -> Principal | None:
    value = str(raw or "").strip()
    if not value:
        return None
    parts = value.split(":", 2)
    if not parts[0].isdigit():
        return None
    return Principal(
        user_id=int(parts[0]),
        display_name=(parts[1].strip() if len(parts) > 1 else f"Dev {parts[0]}") or f"Dev {parts[0]}",
        site_role=(parts[2].strip() if len(parts) > 2 else "user") or "user",
    )


def _principal_from_user(user: dict[str, Any] | None) -> Principal | None:
    if not user:
        return None
    return Principal(
        user_id=int(user["id"]),
        display_name=str(user.get("display_name") or f"User {user['id']}"),
        site_role=str(user.get("role") or "user"),
        session_id=int(user["session_id"]) if user.get("session_id") is not None else None,
    )


def _token_from_authorization(value: str | None) -> str:
    scheme, _, token = str(value or "").strip().partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def _authenticate_token(token: str | None) -> Principal | None:
    if not token:
        return None
    try:
        from backend.auth.service import authenticate_session_token
    except ImportError:
        return None
    return _principal_from_user(authenticate_session_token(token))


def principal_from_request(request: Request, settings: CompetitionSettings) -> Principal:
    tokens = (
        request.cookies.get("tb_shared_session"),
        request.cookies.get("tb_session"),
        _token_from_authorization(request.headers.get("authorization")),
    )
    for token in tokens:
        principal = _authenticate_token(token)
        if principal is not None:
            return principal
    if settings.allow_dev_auth:
        principal = _dev_principal(request.headers.get("x-competition-dev-user"))
        if principal is not None:
            return principal
    raise HTTPException(status_code=401, detail={"code": "AUTH_REQUIRED", "message": "Authentication required."})


def principal_from_websocket(websocket: WebSocket, settings: CompetitionSettings) -> Principal | None:
    for token in (
        websocket.cookies.get("tb_shared_session"),
        websocket.cookies.get("tb_session"),
        websocket.query_params.get("token"),
    ):
        principal = _authenticate_token(token)
        if principal is not None:
            return principal
    if settings.allow_dev_auth:
        return _dev_principal(websocket.query_params.get("dev_user"))
    return None


def principal_from_socket_auth(payload: dict[str, Any], settings: CompetitionSettings) -> Principal | None:
    principal = _authenticate_token(str(payload.get("token") or ""))
    if principal is not None:
        return principal
    if settings.allow_dev_auth:
        return _dev_principal(str(payload.get("dev_user") or ""))
    return None


def auth_user_exists(user_id: int) -> bool:
    try:
        from backend.auth.db import auth_db
    except ImportError:
        return False
    with auth_db() as db:
        row = db.execute(
            "SELECT 1 FROM users WHERE id = ? AND status = 'active'",
            (int(user_id),),
        ).fetchone()
    return row is not None

