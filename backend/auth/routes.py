from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Request, Response

from .dependencies import client_ip, cookie_secure, current_user_from_request, require_user
from .service import (
    BROWSER_COOKIE_NAME,
    EMAIL_CODE_BROWSER_COOLDOWN_SECONDS,
    EmailCodeCooldownError,
    SESSION_COOKIE_NAME,
    change_password,
    deactivate_account,
    login_user,
    register_user,
    request_account_deactivation_code,
    request_password_reset_code,
    reset_password,
    revoke_session,
    send_register_email_code,
)
from .security import new_token


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


def _browser_id(request: Request, response: Response) -> str:
    existing = str(request.cookies.get(BROWSER_COOKIE_NAME) or "").strip()
    if existing:
        return existing
    token = new_token(18)
    response.set_cookie(
        BROWSER_COOKIE_NAME,
        token,
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
        max_age=365 * 24 * 60 * 60,
        path="/",
    )
    return token


def _cooldown_response(exc: EmailCodeCooldownError) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail={
            "code": "EMAIL_CODE_COOLDOWN",
            "message": "Please wait before requesting another verification code.",
            "retry_after_seconds": exc.retry_after_seconds,
            "cooldown_seconds": EMAIL_CODE_BROWSER_COOLDOWN_SECONDS,
        },
    )


@router.get("/me")
async def me(request: Request):
    user = current_user_from_request(request)
    return {"authenticated": user is not None, "user": user}


@router.post("/send-email-code")
async def send_email_code(request: Request, response: Response, payload: dict = Body(...)):
    try:
        result = send_register_email_code(
            email=str(payload.get("email") or ""),
            invite_code=str(payload.get("invite_code") or ""),
            ip_address=client_ip(request),
            browser_id=_browser_id(request, response),
        )
        return result
    except EmailCodeCooldownError as exc:
        raise _cooldown_response(exc) from exc
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


@router.post("/request-password-reset")
async def request_password_reset(request: Request, response: Response, payload: dict = Body(...)):
    try:
        return request_password_reset_code(
            email=str(payload.get("email") or ""),
            ip_address=client_ip(request),
            browser_id=_browser_id(request, response),
        )
    except EmailCodeCooldownError as exc:
        raise _cooldown_response(exc) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reset-password")
async def reset_password_route(request: Request, response: Response, payload: dict = Body(...)):
    try:
        result = reset_password(
            email=str(payload.get("email") or ""),
            verification_code=str(payload.get("verification_code") or ""),
            new_password=str(payload.get("new_password") or ""),
            user_agent=request.headers.get("user-agent", ""),
            ip_address=client_ip(request),
        )
        _set_session_cookie(response, result["token"], result["expires_at"])
        return {"authenticated": True, "user": result["user"]}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/change-password")
async def change_password_route(request: Request, payload: dict = Body(...)):
    user = require_user(request)
    try:
        result = change_password(
            user_id=int(user["id"]),
            session_id=int(user["session_id"]) if user.get("session_id") else None,
            current_password=str(payload.get("current_password") or ""),
            new_password=str(payload.get("new_password") or ""),
        )
        return {"authenticated": True, "user": result["user"]}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/request-deactivation-code")
async def request_deactivation_code(request: Request, response: Response):
    user = require_user(request)
    try:
        return request_account_deactivation_code(
            user_id=int(user["id"]),
            ip_address=client_ip(request),
            browser_id=_browser_id(request, response),
        )
    except EmailCodeCooldownError as exc:
        raise _cooldown_response(exc) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/deactivate")
async def deactivate(request: Request, response: Response, payload: dict = Body(...)):
    user = require_user(request)
    try:
        deactivate_account(
            user_id=int(user["id"]),
            password=str(payload.get("password") or ""),
            confirm=str(payload.get("confirm") or ""),
            verification_code=str(payload.get("verification_code") or ""),
        )
        _clear_session_cookie(response)
        return {"authenticated": False, "user": None}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/logout")
async def logout(request: Request, response: Response):
    revoke_session(request.cookies.get(SESSION_COOKIE_NAME))
    _clear_session_cookie(response)
    return {"authenticated": False, "user": None}
