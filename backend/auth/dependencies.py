from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, Request, WebSocket

from .service import SESSION_COOKIE_NAME, authenticate_session_token


def cookie_secure() -> bool:
    return os.getenv("AUTH_COOKIE_SECURE", "1") != "0"


def current_user_from_request(request: Request) -> dict[str, Any] | None:
    for token in auth_tokens_from_request(request):
        user = authenticate_session_token(token)
        if user is not None:
            return user
    return None


def bearer_token_from_authorization(value: str | None) -> str:
    if not value:
        return ""
    scheme, _, token = value.strip().partition(" ")
    if scheme.lower() != "bearer" or not token:
        return ""
    return token.strip()


def auth_tokens_from_request(request: Request) -> list[str]:
    tokens: list[str] = []
    cookie_token = str(request.cookies.get(SESSION_COOKIE_NAME) or "").strip()
    bearer_token = bearer_token_from_authorization(request.headers.get("authorization"))
    for token in (cookie_token, bearer_token):
        if token and token not in tokens:
            tokens.append(token)
    return tokens


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
