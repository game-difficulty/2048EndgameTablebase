from __future__ import annotations

import atexit
import asyncio
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager
import json
import logging
import os
import time

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from backend.actions import Action, Message
from backend.admin.routes import router as admin_router
from backend.auth.db import init_auth_db
from backend.auth.dependencies import client_ip, current_user_from_websocket, require_user
from backend.auth.routes import router as auth_router
from backend.auth.service import authenticate_session_token, record_usage
from backend.cloud_analysis_jobs import (
    analysis_job_payload,
    cleanup_expired_jobs,
    create_analysis_job,
    get_analysis_job,
)
from backend.cloud_files import (
    allowed_extensions_for_kind,
    build_file_download_response,
    cleanup_expired_uploads,
    get_download_root,
    get_max_upload_bytes_for_kind,
    get_download_record,
    get_upload_record,
    register_upload,
    save_upload_file,
)
from backend.cloud_safety import (
    cloud_disabled_message,
    is_cloud_action_blocked,
    is_cloud_mode,
)
from backend.quota.errors import InsufficientTokens
from backend.quota.routes import router as quota_router
from backend.quota.service import (
    cancel_reservation,
    get_token_balance,
    reserve_operation_tokens,
)
from backend.replay_routes import router as replay_router
from backend.remote_workers import remote_worker_registry
CLOUD_MODE = is_cloud_mode()

from backend.handlers.analysis import handle_analysis_action
from backend.handlers.replay import handle_replay_action
from backend.handlers.settings import handle_settings_action
from backend.handlers.tester import handle_tester_action
from backend.handlers.trainer import handle_trainer_action
from backend.handlers.tablebase_query import (
    drain_tablebase_query_tasks,
    handle_tablebase_query_action,
)
from backend.gamer_ranked.routes import router as gamer_ranked_router
from backend.gamer_ranked.service import (
    cleanup_stale_ranked_runs,
    prepare_validation_queue as prepare_gamer_validation_queue,
    process_one_pending_run,
)
from backend.leaderboards.routes import router as leaderboard_router
from backend.minigame_rankings.routes import router as minigame_rankings_router
from backend.minigame_rankings.verifier import (
    cleanup_stale_ranked_runs as cleanup_stale_minigame_ranked_runs,
    close_verifier as close_minigame_verifier,
    prepare_validation_queue as prepare_minigame_validation_queue,
    process_one_pending_run as process_one_pending_minigame_run,
)
from backend.leaderboards.service import refresh_due_leaderboards
from backend.profile.routes import router as profile_router
from backend.preload import start_preload_thread
from backend.resource_paths import get_resource_path
from backend.state import ConnectionManager, save_game_state
from backend.tablebase_catalog import (
    get_available_tablebases,
    get_catalog_version,
    resolve_configured_tablebase,
)
from backend.tablebase_query_service import tablebase_query_scheduler
from Config import SingletonConfig
from error_bridge import publish_frontend_exception

if CLOUD_MODE:
    handle_game_action = None
    handle_notebook_action = None
else:
    from backend.handlers.game import handle_game_action
    from backend.handlers.notebook import handle_notebook_action


manager = ConnectionManager()
logger = logging.getLogger("2048tables.websocket")
mathjax_path = get_resource_path("mathjax")
pic_path = get_resource_path("pic")
minigame_assets_path = pic_path
frontend_dist_path = get_resource_path(os.path.join("frontend", "dist"))
frontend_assets_path = os.path.join(frontend_dist_path, "assets")
frontend_wasm_path = os.path.join(frontend_dist_path, "wasm")
frontend_guides_path = os.path.join(frontend_dist_path, "guides")

WS_MAX_ACTIVE_CONNECTIONS = int(os.getenv("WS_MAX_ACTIVE_CONNECTIONS", "1024"))
WS_MAX_ACTIVE_CONNECTIONS_PER_IP = int(
    os.getenv("WS_MAX_ACTIVE_CONNECTIONS_PER_IP", "24")
)
WS_ACCEPT_RATE_WINDOW_SECONDS = int(
    os.getenv("WS_ACCEPT_RATE_WINDOW_SECONDS", "60")
)
WS_MAX_ACCEPTS_PER_WINDOW_PER_IP = int(
    os.getenv("WS_MAX_ACCEPTS_PER_WINDOW_PER_IP", "60")
)
WS_LOG_INTERVAL_SECONDS = int(os.getenv("WS_LOG_INTERVAL_SECONDS", "60"))
WS_LOG_BURST = int(os.getenv("WS_LOG_BURST", "5"))

_ws_active_by_ip: Counter[str] = Counter()
_ws_accepts_by_ip: defaultdict[str, deque[float]] = defaultdict(deque)
_ws_log_windows: dict[str, dict[str, float | int]] = {}
_WS_CLOSED_ERROR_MARKERS = (
    "WebSocket is not connected",
    "Cannot call \"send\" once a close message has been sent",
    "Cannot call \"receive\" once a disconnect message has been received",
    "after sending 'websocket.close'",
    "Unexpected ASGI message",
    "ClientDisconnected",
    "ConnectionClosed",
)


def _broadcast_tablebase_catalog_update(_epoch: int) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(
        manager.broadcast(
            json.dumps(
                {"action": "TABLEBASE_CATALOG_UPDATED", "data": {}},
                separators=(",", ":"),
            )
        )
    )


class CacheControlledStaticFiles(StaticFiles):
    def __init__(self, *args, cache_control: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_control = cache_control

    async def get_response(self, path, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        if self.cache_control and response.status_code == 200:
            response.headers["Cache-Control"] = self.cache_control
        return response


async def _leaderboard_refresh_loop() -> None:
    while True:
        try:
            await asyncio.to_thread(refresh_due_leaderboards)
        except Exception as exc:
            _rate_limited_log(
                "leaderboard-refresh",
                f"Leaderboard refresh failed: {type(exc).__name__}: {exc}",
            )
        await asyncio.sleep(3600)


async def _gamer_validation_loop() -> None:
    next_cleanup = 0.0
    while True:
        if time.monotonic() >= next_cleanup:
            await asyncio.to_thread(cleanup_stale_ranked_runs)
            next_cleanup = time.monotonic() + 3600
        processed = await asyncio.to_thread(process_one_pending_run)
        await asyncio.sleep(0.05 if processed else 1.0)


async def _minigame_validation_loop() -> None:
    next_cleanup = 0.0
    while True:
        if time.monotonic() >= next_cleanup:
            await asyncio.to_thread(cleanup_stale_minigame_ranked_runs)
            next_cleanup = time.monotonic() + 3600
        processed = await asyncio.to_thread(process_one_pending_minigame_run)
        await asyncio.sleep(0.05 if processed else 1.0)


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    SingletonConfig()
    init_auth_db()
    prepare_gamer_validation_queue()
    prepare_minigame_validation_queue()
    cleanup_expired_uploads()
    cleanup_expired_jobs()
    start_preload_thread()
    remote_worker_registry.add_availability_listener(
        _broadcast_tablebase_catalog_update
    )
    await remote_worker_registry.start()
    leaderboard_refresh_task = asyncio.create_task(_leaderboard_refresh_loop())
    gamer_validation_task = asyncio.create_task(_gamer_validation_loop())
    minigame_validation_task = asyncio.create_task(_minigame_validation_loop())
    try:
        yield
    finally:
        leaderboard_refresh_task.cancel()
        gamer_validation_task.cancel()
        minigame_validation_task.cancel()
        try:
            await leaderboard_refresh_task
        except asyncio.CancelledError:
            pass
        try:
            await gamer_validation_task
        except asyncio.CancelledError:
            pass
        try:
            await minigame_validation_task
        except asyncio.CancelledError:
            pass
        await asyncio.to_thread(close_minigame_verifier)
        remote_worker_registry.remove_availability_listener(
            _broadcast_tablebase_catalog_update
        )
        await remote_worker_registry.close()
        await tablebase_query_scheduler.close()
        await drain_tablebase_query_tasks()


app = FastAPI(lifespan=app_lifespan)
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(replay_router)
app.include_router(quota_router)
app.include_router(leaderboard_router)
app.include_router(minigame_rankings_router)
app.include_router(profile_router)
app.include_router(gamer_ranked_router)


@app.middleware("http")
async def canonicalize_www_domain(request: Request, call_next):
    host = str(request.headers.get("host") or "").lower().split(":", 1)[0]
    if host == "www.2048tables.online":
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"
        return RedirectResponse(
            f"https://2048tables.online{path}",
            status_code=308,
        )
    return await call_next(request)


def _rate_limited_log(
    key: str,
    message: str,
    *,
    exc: BaseException | None = None,
) -> None:
    now = time.monotonic()
    window = _ws_log_windows.get(key)
    if window is None or now - float(window["start"]) >= WS_LOG_INTERVAL_SECONDS:
        if window and int(window.get("suppressed", 0)) > 0:
            logger.warning(
                "Suppressed %s similar websocket log messages for %s.",
                int(window["suppressed"]),
                key,
            )
        _ws_log_windows[key] = {"start": now, "count": 1, "suppressed": 0}
        if exc is None:
            logger.warning(message)
        else:
            logger.warning(message, exc_info=(type(exc), exc, exc.__traceback__))
        return

    if int(window["count"]) < WS_LOG_BURST:
        window["count"] = int(window["count"]) + 1
        if exc is None:
            logger.warning(message)
        else:
            logger.warning(message, exc_info=(type(exc), exc, exc.__traceback__))
        return

    window["suppressed"] = int(window.get("suppressed", 0)) + 1


def _is_websocket_closed_error(exc: BaseException) -> bool:
    if isinstance(exc, WebSocketDisconnect):
        return True
    message = str(exc)
    return any(marker in message for marker in _WS_CLOSED_ERROR_MARKERS)


def _websocket_can_send(websocket: WebSocket) -> bool:
    try:
        return (
            websocket.application_state == WebSocketState.CONNECTED
            and websocket.client_state == WebSocketState.CONNECTED
        )
    except Exception:
        return False


async def _safe_send_json(
    websocket: WebSocket,
    payload: dict,
    *,
    log_key: str,
) -> bool:
    if not _websocket_can_send(websocket):
        return False
    try:
        await websocket.send_json(payload)
        return True
    except Exception as exc:
        if not _is_websocket_closed_error(exc):
            _rate_limited_log(
                log_key,
                f"WebSocket send failed: {type(exc).__name__}: {exc}",
                exc=exc,
            )
        return False


async def _send_ws_error(websocket: WebSocket, message: str) -> bool:
    return await _safe_send_json(
        websocket,
        {"action": Message.ERROR, "data": {"message": message}},
        log_key="send_error",
    )


async def _send_auth_required(websocket: WebSocket) -> bool:
    return await _safe_send_json(
        websocket,
        {
            "action": Message.AUTH_REQUIRED,
            "data": {
                "code": "AUTH_REQUIRED",
                "message": "Authentication required.",
            },
        },
        log_key="send_auth_required",
    )


async def _send_token_required(websocket: WebSocket, exc: InsufficientTokens) -> bool:
    return await _safe_send_json(
        websocket,
        {
            "action": Message.TOKEN_REQUIRED,
            "data": exc.payload,
        },
        log_key="send_token_required",
    )


async def _close_websocket_quietly(websocket: WebSocket, code: int = 1008) -> None:
    try:
        await websocket.close(code=code)
    except Exception as exc:
        if not _is_websocket_closed_error(exc):
            _rate_limited_log(
                "ws_close_failed",
                f"WebSocket close failed: {type(exc).__name__}: {exc}",
                exc=exc,
            )


def _websocket_client_ip(websocket: WebSocket) -> str:
    forwarded = websocket.headers.get("cf-connecting-ip", "").strip()
    if forwarded:
        return forwarded
    forwarded = websocket.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return websocket.client.host if websocket.client else "unknown"


def _reserve_websocket_slot(ip_address: str) -> tuple[bool, str]:
    now = time.monotonic()
    attempts = _ws_accepts_by_ip[ip_address]
    while attempts and now - attempts[0] > WS_ACCEPT_RATE_WINDOW_SECONDS:
        attempts.popleft()

    if sum(_ws_active_by_ip.values()) >= WS_MAX_ACTIVE_CONNECTIONS:
        return False, "global connection limit reached"
    if _ws_active_by_ip[ip_address] >= WS_MAX_ACTIVE_CONNECTIONS_PER_IP:
        return False, "per-client connection limit reached"
    if len(attempts) >= WS_MAX_ACCEPTS_PER_WINDOW_PER_IP:
        return False, "per-client connection rate limit reached"

    attempts.append(now)
    _ws_active_by_ip[ip_address] += 1
    return True, ""


def _release_websocket_slot(ip_address: str) -> None:
    if not ip_address:
        return
    if _ws_active_by_ip[ip_address] <= 1:
        _ws_active_by_ip.pop(ip_address, None)
    else:
        _ws_active_by_ip[ip_address] -= 1


def _bind_auth_user(session, auth_user: dict) -> None:
    session.user_id = int(auth_user["id"])
    session.auth_session_id = (
        int(auth_user["session_id"]) if auth_user.get("session_id") else None
    )
    session.user_email = str(auth_user["email"])
    session.user_role = str(auth_user["role"])
    session.user_entitlement_tier = str(
        (auth_user.get("entitlements") or {}).get("tier") or "free"
    )


def _bind_or_restore_auth_user(
    websocket: WebSocket,
    session,
    auth_user: dict,
):
    restored_session = manager.restore_detached_session(websocket, auth_user) or session
    _bind_auth_user(restored_session, auth_user)
    return restored_session


async def _handle_auth_session(websocket: WebSocket, session, payload: dict):
    token = str((payload or {}).get("token") or "").strip()
    auth_user = authenticate_session_token(token)
    if auth_user is None or (
        session.user_id is not None and int(auth_user["id"]) != int(session.user_id)
    ):
        if session.user_id is not None:
            await _safe_send_json(
                websocket,
                {
                    "action": Action.AUTH_SESSION,
                    "data": {"authenticated": True},
                },
                log_key="send_auth_session",
            )
            return session
        await _send_auth_required(websocket)
        return session
    session = _bind_or_restore_auth_user(websocket, session, auth_user)
    await _safe_send_json(
        websocket,
        {
            "action": Action.AUTH_SESSION,
            "data": {
                "authenticated": True,
                "user": auth_user,
                "token_balance": auth_user.get("token_balance"),
            },
        },
        log_key="send_auth_session",
    )
    return session


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):  # type: ignore
    ws_ip = _websocket_client_ip(websocket)
    slot_reserved = False
    session = None

    allowed, reject_reason = _reserve_websocket_slot(ws_ip)
    if not allowed:
        _rate_limited_log(
            f"ws_reject_{reject_reason}",
            f"Rejected websocket from {ws_ip}: {reject_reason}.",
        )
        await _close_websocket_quietly(websocket)
        return
    slot_reserved = True

    try:
        auth_user = current_user_from_websocket(websocket)
        await manager.connect(websocket, client_id)
        session = manager.active_connections[websocket]
        if auth_user is not None:
            session = _bind_or_restore_auth_user(websocket, session, auth_user)

        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                action = message.get("action") or message.get("type")
                payload = (
                    message.get("data")
                    if isinstance(message.get("data"), dict)
                    else message
                )

                if is_cloud_action_blocked(action):
                    await _send_ws_error(
                        websocket,
                        cloud_disabled_message(action),
                    )
                    continue

                if action == Action.AUTH_SESSION:
                    session = await _handle_auth_session(websocket, session, payload)
                    continue

                if _action_requires_auth(action) and session.user_id is None:
                    await _send_auth_required(websocket)
                    continue

                usage = _usage_for_ws_action(action)
                if usage is not None and session.user_id is not None:
                    quota_key, event_type = usage
                    record_usage(
                        user_id=session.user_id,
                        session_id=session.auth_session_id,
                        event_type=event_type,
                        quota_key=quota_key,
                        cost=0,
                        metadata={"action": action},
                    )

                if action == Action.GET_STATE:
                    await manager.send_state(websocket)

                elif await handle_tablebase_query_action(
                    action, payload, session, websocket
                ):
                    continue

                elif await handle_tester_action(action, payload, session, websocket):
                    continue

                elif await handle_replay_action(action, payload, session, websocket):
                    continue

                elif (
                    not CLOUD_MODE
                    and handle_notebook_action is not None
                    and await handle_notebook_action(
                        action, payload, session, websocket
                    )
                ):
                    continue

                elif await handle_analysis_action(action, payload, session, websocket):
                    continue

                elif await handle_settings_action(
                    action, payload, session, websocket, manager
                ):
                    continue

                elif (
                    not CLOUD_MODE
                    and handle_game_action is not None
                    and await handle_game_action(
                        action, payload, session, websocket, manager
                    )
                ):
                    continue

                elif await handle_trainer_action(
                    action, payload, session, websocket, manager
                ):
                    continue

            except WebSocketDisconnect:
                break
            except InsufficientTokens as exc:
                sent = await _send_token_required(websocket, exc)
                if not sent:
                    break
            except Exception as exc:
                if _is_websocket_closed_error(exc):
                    break
                _rate_limited_log(
                    "ws_action_error",
                    f"WebSocket action error: {type(exc).__name__}: {exc}",
                    exc=exc,
                )
                publish_frontend_exception("WebSocket Action Error", exc)
                sent = await _send_ws_error(websocket, str(exc))
                if not sent:
                    break
    except Exception as exc:
        if not _is_websocket_closed_error(exc):
            _rate_limited_log(
                "ws_connection_error",
                f"WebSocket connection error: {type(exc).__name__}: {exc}",
                exc=exc,
            )
            publish_frontend_exception("WebSocket Connection Error", exc)
    finally:
        if session is not None:
            save_game_state(session)
        manager.disconnect(websocket)
        if slot_reserved:
            _release_websocket_slot(ws_ip)


@app.websocket("/worker-ws/tablebase")
async def tablebase_worker_endpoint(websocket: WebSocket):  # type: ignore
    await remote_worker_registry.handle_connection(websocket)


def _usage_for_ws_action(action: str | None) -> tuple[str, str] | None:
    return None


def _action_requires_auth(action: str | None) -> bool:
    return action in {
        Action.TRAINER_SET_FILEPATH,
        Action.TRAINER_SET_EMPTY_PATTERN,
        Action.TRAINER_GET_RESULTS,
        Action.TRAINER_DEFAULT,
        Action.TRAINER_MOVE,
        Action.TRAINER_MANUAL_SPAWN,
        Action.TRAINER_STEP,
        Action.SET_BOARD,
        Action.SET_CELL,
        Action.UNDO,
        Action.TESTER_SELECT_PATTERN,
        Action.TESTER_RESET_RANDOM,
        Action.TESTER_MOVE,
        Action.TESTER_SET_BOARD,
        Action.TESTER_EXPORT_LOG,
        Action.TESTER_EXPORT_REPLAY,
        Action.REPLAY_LOAD_UPLOAD,
        Action.REPLAY_LOAD_LATEST,
        Action.ANALYSIS_SUBSCRIBE,
        Action.TABLEBASE_QUERY,
    }


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def serve_index():
    path = os.path.join(frontend_dist_path, "index.html")
    if os.path.exists(path):
        return FileResponse(path)

    return {
        "ERROR": "FRONTEND_NOT_FOUND",
        "ATTEMPTED_ABS_PATH": os.path.abspath(path),
        "CWD": os.getcwd(),
        "SYS_MEIPASS": getattr(os.sys, "_MEIPASS", "NOT_BUNDLE"),
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    icon_path = get_resource_path("favicon.ico")
    if os.path.exists(icon_path):
        return FileResponse(icon_path)

    alt_icon_path = get_resource_path(os.path.join("pic", "2048_2.ico"))
    if os.path.exists(alt_icon_path):
        return FileResponse(alt_icon_path)

    return None


@app.get("/api/tablebases")
async def tablebases():
    return {
        "catalog_version": get_catalog_version(),
        "tables": get_available_tablebases(),
    }


@app.post("/api/uploads")
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    kind: str = Form("generic"),
    user: dict = Depends(require_user),
):
    normalized_kind = str(kind or "generic").strip().lower()
    allowed_extensions = allowed_extensions_for_kind(normalized_kind)
    uploads = []
    try:
        for upload in files:
            saved = await save_upload_file(
                upload,
                allowed_extensions=allowed_extensions or None,
                max_bytes=get_max_upload_bytes_for_kind(normalized_kind),
            )
            record = register_upload(
                saved,
                kind=normalized_kind,
                user_id=int(user["id"]),
                session_id=int(user["session_id"]),
            )
            quota_key = (
                "upload_analysis"
                if normalized_kind == "analysis"
                else "upload_replay"
                if normalized_kind == "replay"
                else "upload"
            )
            record_usage(
                user_id=int(user["id"]),
                session_id=int(user["session_id"]),
                event_type=f"upload:{normalized_kind}",
                quota_key=quota_key,
                cost=0,
                metadata={"filename": record.filename, "size": record.size},
                ip_address=client_ip(request),
            )
            uploads.append(
                {
                    "upload_id": record.upload_id,
                    "filename": record.filename,
                    "size": record.size,
                    "content_type": record.content_type,
                    "kind": record.kind,
                }
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"uploads": uploads}


@app.get("/api/downloads/{download_id}")
async def download_file(download_id: str, user: dict = Depends(require_user)):
    try:
        record = get_download_record(download_id, user_id=int(user["id"]))
        return build_file_download_response(
            record.path,
            filename=record.filename,
            media_type=record.media_type,
            root=get_download_root(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Download not found.") from exc


@app.post("/api/analysis/jobs")
async def create_analysis_job_route(
    request: Request,
    files: list[UploadFile] = File(...),
    pattern: str = Form(...),
    target: str = Form(...),
    user: dict = Depends(require_user),
):
    uploads = []
    reservations = []
    job_started = False
    try:
        normalized_pattern = str(pattern or "").strip()
        normalized_target = str(target or "").strip()
        full_pattern = f"{normalized_pattern}_{normalized_target}"
        descriptor = resolve_configured_tablebase(full_pattern)
        if descriptor is None:
            raise ValueError("The selected tablebase is not available.")
        if descriptor.get("_provider") == "remote" and not descriptor.get(
            "_available", False
        ):
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "REMOTE_TABLEBASE_OFFLINE",
                    "message": "The selected tablebase is temporarily unavailable.",
                },
            )
        for upload in files:
            saved = await save_upload_file(
                upload,
                allowed_extensions=allowed_extensions_for_kind("analysis"),
                max_bytes=get_max_upload_bytes_for_kind("analysis"),
            )
            uploads.append(
                register_upload(
                    saved,
                    kind="analysis",
                    user_id=int(user["id"]),
                    session_id=int(user["session_id"]),
                )
            )
        for record in uploads:
            reservations.append(
                reserve_operation_tokens(
                    user_id=int(user["id"]),
                    session_id=int(user["session_id"]),
                    operation_key="analysis_per_replay",
                    full_pattern=full_pattern,
                )
            )
        job = create_analysis_job(
            uploads=uploads,
            pattern=normalized_pattern,
            target=normalized_target,
            user_id=int(user["id"]),
            session_id=int(user["session_id"]),
            quota_reservations=reservations,
        )
        job_started = True
        record_usage(
            user_id=int(user["id"]),
            session_id=int(user["session_id"]),
            event_type="analysis_job",
            quota_key="analysis_job",
            cost=0,
            metadata={"pattern": pattern, "target": target, "total": len(uploads)},
            ip_address=client_ip(request),
        )
    except InsufficientTokens as exc:
        for reservation in reservations:
            cancel_reservation(reservation, reason="analysis_job_not_created")
        raise HTTPException(status_code=402, detail=exc.payload) from exc
    except ValueError as exc:
        for reservation in reservations:
            cancel_reservation(reservation, reason="analysis_job_not_created")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        if not job_started:
            for reservation in reservations:
                cancel_reservation(reservation, reason="analysis_job_not_created")
        raise
    return {
        "job_id": job.job_id,
        "total": job.total,
        "token_balance": get_token_balance(int(user["id"])),
    }


@app.get("/api/analysis/jobs/{job_id}")
async def get_analysis_job_route(job_id: str, user: dict = Depends(require_user)):
    try:
        return analysis_job_payload(get_analysis_job(job_id, user_id=int(user["id"])))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis job not found.") from exc


@app.get("/api/analysis/jobs/{job_id}/download")
async def download_analysis_job(job_id: str, user: dict = Depends(require_user)):
    try:
        job = get_analysis_job(job_id, user_id=int(user["id"]))
        if job.zip_path is None or job.status != "finished":
            raise HTTPException(status_code=409, detail="Analysis job is not finished.")
        return build_file_download_response(
            job.zip_path,
            filename=job.zip_path.name,
            media_type="application/zip",
            root=get_download_root(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis job not found.") from exc


if os.path.exists(mathjax_path):
    app.mount(
        "/mathjax",
        CacheControlledStaticFiles(
            directory=mathjax_path,
            cache_control="public, max-age=31536000, immutable",
        ),
        name="mathjax",
    )

if os.path.exists(minigame_assets_path):
    app.mount(
        "/minigames-assets",
        CacheControlledStaticFiles(
            directory=minigame_assets_path,
            cache_control="public, max-age=604800",
        ),
        name="minigames-assets",
    )

if os.path.exists(frontend_assets_path):
    app.mount(
        "/assets",
        CacheControlledStaticFiles(
            directory=frontend_assets_path,
            cache_control="public, max-age=31536000, immutable",
        ),
        name="frontend-assets",
    )

if os.path.exists(frontend_wasm_path):
    app.mount(
        "/wasm",
        CacheControlledStaticFiles(
            directory=frontend_wasm_path,
            cache_control="public, max-age=31536000, immutable",
        ),
        name="frontend-wasm",
    )

if os.path.exists(frontend_guides_path):
    app.mount(
        "/guides",
        CacheControlledStaticFiles(
            directory=frontend_guides_path,
            cache_control="public, max-age=86400",
        ),
        name="frontend-guides",
    )

if os.path.exists(frontend_dist_path):
    app.mount(
        "/",
        CacheControlledStaticFiles(
            directory=frontend_dist_path,
            html=True,
            cache_control="no-cache",
        ),
        name="frontend",
    )


def persist_runtime_config() -> None:
    try:
        SingletonConfig().save_config(SingletonConfig().config)
    except Exception as exc:
        print(f"Failed to persist config on exit: {exc}")


def run_backend_server(port: int, host: str = "127.0.0.1") -> None:
    atexit.register(persist_runtime_config)
    uvicorn.run(app, host=host, port=port, log_level="error")
