from __future__ import annotations

from datetime import datetime, timezone
from email.utils import format_datetime

from fastapi import APIRouter, HTTPException, Request, Response

from .dependencies import client_ip, cookie_secure, guest_tokens_from_request
from .guest_service import (
    GUEST_COOKIE_NAME,
    GuestLimitError,
    authenticate_guest_token,
    issue_guest_session,
    revoke_guest_session,
)
from .principal import ActorRef


router = APIRouter(prefix="/api/guest", tags=["guest"])


def _set_guest_cookie(response: Response, token: str, expires_at: str) -> None:
    expires_dt = datetime.fromisoformat(expires_at).astimezone(timezone.utc)
    max_age = max(1, int((expires_dt - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(
        GUEST_COOKIE_NAME,
        token,
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
        max_age=max_age,
        expires=format_datetime(expires_dt, usegmt=True),
        path="/",
    )


def _response(result: dict) -> dict:
    guest = result["guest"]
    return {
        "authenticated": False,
        "guest": guest,
        "actor": ActorRef.from_guest(guest).public_dict(),
        "guest_session_token": result["token"],
        "expires_at": result["expires_at"],
    }


@router.get("/me")
async def guest_me(request: Request):
    for token in guest_tokens_from_request(request):
        guest = authenticate_guest_token(token, ip_address=client_ip(request))
        if guest is not None:
            return {
                "guest": guest,
                "actor": ActorRef.from_guest(guest).public_dict(),
            }
    return {"guest": None, "actor": None}


@router.post("/session")
async def create_guest_session(request: Request, response: Response):
    for token in guest_tokens_from_request(request):
        guest = authenticate_guest_token(token, ip_address=client_ip(request))
        if guest is not None:
            # Restore the primary HttpOnly credential when a mobile browser only
            # retained the JavaScript-readable fallback token.
            _set_guest_cookie(response, token, guest["expires_at"])
            return {
                "authenticated": False,
                "guest": guest,
                "actor": ActorRef.from_guest(guest).public_dict(),
                "guest_session_token": token,
                "expires_at": guest["expires_at"],
            }
    try:
        result = issue_guest_session(ip_address=client_ip(request))
    except GuestLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    _set_guest_cookie(response, result["token"], result["expires_at"])
    return _response(result)


@router.post("/logout")
async def guest_logout(request: Request, response: Response):
    for token in guest_tokens_from_request(request):
        revoke_guest_session(token)
    response.delete_cookie(GUEST_COOKIE_NAME, path="/")
    return {"guest": None, "actor": None}
