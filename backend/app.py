from __future__ import annotations

import atexit
from contextlib import asynccontextmanager
import json
import os

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.actions import Action, Message
from backend.admin.routes import router as admin_router
from backend.auth.db import init_auth_db
from backend.auth.dependencies import client_ip, current_user_from_websocket, require_user
from backend.auth.routes import router as auth_router
from backend.auth.service import record_usage
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
from backend.quota.service import consume_operation_tokens, get_token_balance
CLOUD_MODE = is_cloud_mode()

from backend.handlers.analysis import handle_analysis_action
from backend.handlers.replay import handle_replay_action
from backend.handlers.settings import handle_settings_action
from backend.handlers.tester import handle_tester_action
from backend.handlers.trainer import handle_trainer_action
from backend.preload import start_preload_thread
from backend.resource_paths import get_resource_path
from backend.state import ConnectionManager, save_game_state
from backend.tablebase_catalog import get_available_tablebases
from Config import SingletonConfig
from error_bridge import publish_frontend_exception

if CLOUD_MODE:
    handle_game_action = None
    handle_notebook_action = None
else:
    from backend.handlers.game import handle_game_action
    from backend.handlers.notebook import handle_notebook_action


manager = ConnectionManager()
mathjax_path = get_resource_path("mathjax")
pic_path = get_resource_path("pic")
minigame_assets_path = pic_path
frontend_dist_path = get_resource_path(os.path.join("frontend", "dist"))
frontend_assets_path = os.path.join(frontend_dist_path, "assets")
frontend_wasm_path = os.path.join(frontend_dist_path, "wasm")


class CacheControlledStaticFiles(StaticFiles):
    def __init__(self, *args, cache_control: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_control = cache_control

    async def get_response(self, path, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        if self.cache_control and response.status_code == 200:
            response.headers["Cache-Control"] = self.cache_control
        return response


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    SingletonConfig()
    init_auth_db()
    cleanup_expired_uploads()
    cleanup_expired_jobs()
    start_preload_thread()
    yield


app = FastAPI(lifespan=app_lifespan)
app.include_router(auth_router)
app.include_router(admin_router)


async def _send_ws_error(websocket: WebSocket, message: str):
    try:
        await websocket.send_json(
            {"action": Message.ERROR, "data": {"message": message}}
        )
    except Exception as send_error:
        print(f"Send error message failed: {send_error}")


async def _send_auth_required(websocket: WebSocket):
    try:
        await websocket.send_json(
            {
                "action": Message.AUTH_REQUIRED,
                "data": {
                    "code": "AUTH_REQUIRED",
                    "message": "Authentication required.",
                },
            }
        )
    except Exception as send_error:
        print(f"Send auth required message failed: {send_error}")


async def _send_token_required(websocket: WebSocket, exc: InsufficientTokens):
    try:
        await websocket.send_json(
            {
                "action": Message.TOKEN_REQUIRED,
                "data": exc.payload,
            }
        )
    except Exception as send_error:
        print(f"Send token required message failed: {send_error}")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):  # type: ignore
    auth_user = current_user_from_websocket(websocket)

    await manager.connect(websocket, client_id)
    session = manager.active_connections[websocket]
    if auth_user is not None:
        session.user_id = int(auth_user["id"])
        session.auth_session_id = int(auth_user["session_id"])
        session.user_email = str(auth_user["email"])
        session.user_role = str(auth_user["role"])

    try:
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
                await _send_token_required(websocket, exc)
            except Exception as exc:
                print(f"WebSocket action error: {exc}")
                publish_frontend_exception("WebSocket Action Error", exc)
                await _send_ws_error(websocket, str(exc))
    except Exception as exc:
        print(f"Connection error: {exc}")
        publish_frontend_exception("WebSocket Connection Error", exc)
    finally:
        save_game_state(session)
        manager.disconnect(websocket)


def _usage_for_ws_action(action: str | None) -> tuple[str, str] | None:
    return None


def _action_requires_auth(action: str | None) -> bool:
    return action in {
        Action.TRAINER_SET_FILEPATH,
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
    return {"tables": get_available_tablebases()}


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
    try:
        for upload in files:
            saved = await save_upload_file(
                upload,
                allowed_extensions=allowed_extensions_for_kind("analysis"),
            )
            uploads.append(
                register_upload(
                    saved,
                    kind="analysis",
                    user_id=int(user["id"]),
                    session_id=int(user["session_id"]),
                )
            )
        consume_operation_tokens(
            user_id=int(user["id"]),
            session_id=int(user["session_id"]),
            operation_key="analysis_per_replay",
            full_pattern=f"{str(pattern or '').strip()}_{str(target or '').strip()}",
            quantity=len(uploads),
            metadata={"pattern": pattern, "target": target, "total": len(uploads)},
        )
        job = create_analysis_job(
            uploads=uploads,
            pattern=str(pattern or "").strip(),
            target=str(target or "").strip(),
            user_id=int(user["id"]),
            session_id=int(user["session_id"]),
        )
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
        raise HTTPException(status_code=402, detail=exc.payload) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
