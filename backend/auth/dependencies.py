from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, Request, WebSocket

from .guest_service import (
    GUEST_COOKIE_NAME,
    GUEST_TOKEN_HEADER,
    authenticate_guest_token,
)
from .principal import ActorRef
from .service import SESSION_COOKIE_NAME, SHARED_SESSION_COOKIE_NAME, authenticate_session_token


def cookie_secure() -> bool:
    return os.getenv("AUTH_COOKIE_SECURE", "1") != "0"


def shared_cookie_domain(request: Request | WebSocket) -> str | None:
    domain = os.getenv('AUTH_SHARED_COOKIE_DOMAIN', '2048tables.online').strip().lower().lstrip('.')
    host = (request.url.hostname or '').lower()
    return domain if domain and host in {domain, f'www.{domain}', f'live.{domain}'} else None


def current_user_from_request(request: Request) -> dict[str, Any] | None:
    for token in auth_tokens_from_request(request):
        user = authenticate_session_token(token)
        if user is not None:
            request.state.auth_session_token = token
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
    shared_token = request.cookies.get(SHARED_SESSION_COOKIE_NAME) if shared_cookie_domain(request) else None
    for token in (shared_token, cookie_token, bearer_token):
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def require_user(request: Request) -> dict[str, Any]:
    user = current_user_from_request(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def guest_tokens_from_request(request: Request) -> list[str]:
    tokens: list[str] = []
    cookie_token = str(request.cookies.get(GUEST_COOKIE_NAME) or "").strip()
    header_token = str(request.headers.get(GUEST_TOKEN_HEADER) or "").strip()
    for token in (cookie_token, header_token):
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def current_guest_from_request(request: Request) -> dict[str, Any] | None:
    ip_address = client_ip(request)
    for token in guest_tokens_from_request(request):
        guest = authenticate_guest_token(token, ip_address=ip_address)
        if guest is not None:
            return guest
    return None


def current_actor_from_request(request: Request) -> ActorRef | None:
    user = current_user_from_request(request)
    if user is not None:
        return ActorRef.from_user(user)
    guest = current_guest_from_request(request)
    if guest is not None:
        return ActorRef.from_guest(guest)
    return None


def require_actor(request: Request) -> ActorRef:
    actor = current_actor_from_request(request)
    if actor is None:
        raise HTTPException(status_code=401, detail="A guest or user session is required.")
    return actor


def current_user_from_websocket(websocket: WebSocket) -> dict[str, Any] | None:
    if shared_cookie_domain(websocket):
        user = authenticate_session_token(websocket.cookies.get(SHARED_SESSION_COOKIE_NAME))
        if user is not None:
            return user
    return authenticate_session_token(websocket.cookies.get(SESSION_COOKIE_NAME))


def current_guest_from_websocket(websocket: WebSocket) -> dict[str, Any] | None:
    return authenticate_guest_token(
        websocket.cookies.get(GUEST_COOKIE_NAME),
        ip_address=websocket_client_ip(websocket),
    )


def current_actor_from_websocket(websocket: WebSocket) -> ActorRef | None:
    user = current_user_from_websocket(websocket)
    if user is not None:
        return ActorRef.from_user(user)
    guest = current_guest_from_websocket(websocket)
    if guest is not None:
        return ActorRef.from_guest(guest)
    return None


def client_ip(request: Request) -> str:
    cloudflare = request.headers.get("cf-connecting-ip", "").strip()
    if cloudflare:
        return cloudflare
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else ""


def websocket_client_ip(websocket: WebSocket) -> str:
    cloudflare = str(websocket.headers.get("cf-connecting-ip") or "").strip()
    if cloudflare:
        return cloudflare
    forwarded = str(websocket.headers.get("x-forwarded-for") or "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return websocket.client.host if websocket.client else ""
