from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, Request, WebSocket

from .service import SESSION_COOKIE_NAME, authenticate_session_token


def cookie_secure() -> bool:
    return os.getenv("AUTH_COOKIE_SECURE", "1") != "0"


def current_user_from_request(request: Request) -> dict[str, Any] | None:
    return authenticate_session_token(request.cookies.get(SESSION_COOKIE_NAME))


def require_user(request: Request) -> dict[str, Any]:
    user = current_user_from_request(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def current_user_from_websocket(websocket: WebSocket) -> dict[str, Any] | None:
    return authenticate_session_token(websocket.cookies.get(SESSION_COOKIE_NAME))


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else ""
