from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Request, Response

from .dependencies import client_ip, cookie_secure, current_user_from_request
from .service import (
    SESSION_COOKIE_NAME,
    login_user,
    register_user,
    revoke_session,
    send_register_email_code,
)


router = APIRouter(prefix="/api/auth", tags=["auth"])


def _set_session_cookie(response: Response, token: str, expires_at: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
        expires=expires_at,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")


@router.get("/me")
async def me(request: Request):
    user = current_user_from_request(request)
    return {"authenticated": user is not None, "user": user}


@router.post("/send-email-code")
async def send_email_code(request: Request, payload: dict = Body(...)):
    try:
        result = send_register_email_code(
            email=str(payload.get("email") or ""),
            invite_code=str(payload.get("invite_code") or ""),
            ip_address=client_ip(request),
        )
        return result
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/register")
async def register(request: Request, response: Response, payload: dict = Body(...)):
    try:
        result = register_user(
            email=str(payload.get("email") or ""),
            password=str(payload.get("password") or ""),
            invite_code=str(payload.get("invite_code") or ""),
            verification_code=str(payload.get("verification_code") or ""),
            display_name=str(payload.get("display_name") or ""),
            user_agent=request.headers.get("user-agent", ""),
            ip_address=client_ip(request),
        )
        _set_session_cookie(response, result["token"], result["expires_at"])
        return {"authenticated": True, "user": result["user"]}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/login")
async def login(request: Request, response: Response, payload: dict = Body(...)):
    try:
        result = login_user(
            email=str(payload.get("email") or ""),
            password=str(payload.get("password") or ""),
            user_agent=request.headers.get("user-agent", ""),
            ip_address=client_ip(request),
        )
        _set_session_cookie(response, result["token"], result["expires_at"])
        return {"authenticated": True, "user": result["user"]}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/logout")
async def logout(request: Request, response: Response):
    revoke_session(request.cookies.get(SESSION_COOKIE_NAME))
    _clear_session_cookie(response)
    return {"authenticated": False, "user": None}
