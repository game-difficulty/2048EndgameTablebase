"""Play-site API process, isolated from the main and live application processes.

Nginx serves the built frontend and forwards only /api/ to this process. The
account database is shared with the main site; game archives use their own DB.
Unlike local_app, this application has no preview account or loopback bypass.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import logging

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile

from backend.admin.routes import router as admin_router
from backend.auth.db import init_auth_db
from backend.auth.activity_middleware import DailyActivityMiddleware
from backend.auth.dependencies import require_user
from backend.auth.routes import router as auth_router
from backend.cloud_analysis_jobs import (
    AnalysisQueueFull,
    analysis_download_filename,
    analysis_job_payload,
    check_analysis_capacity,
    create_analysis_job,
    get_analysis_job,
    start_analysis_worker,
    cleanup_expired_jobs,
)
from backend.analysis_history import cleanup_artifacts, router as analysis_history_router
from backend.cloud_files import (
    allowed_extensions_for_kind,
    build_file_download_response,
    delete_upload,
    get_download_root,
    get_max_upload_bytes_for_kind,
    register_upload,
    save_upload_file,
)
from backend.http_compression import DisplayCompression
from backend.profile.routes import router as profile_router
from backend.quota.errors import InsufficientTokens
from backend.quota.routes import router as quota_router
from backend.quota.service import cancel_reservation, get_token_balance, reserve_operation_tokens_many
from backend.tablebase_catalog import resolve_configured_tablebase
from backend.auth.dependencies import client_ip
from backend.auth.service import record_usage
from backend.token_rewards import settle_human_rolling_weeks
from Config import SingletonConfig

from .routes import router as human_router
from .store import init_db

logger = logging.getLogger(__name__)


async def reward_loop():
    while True:
        try:
            await asyncio.to_thread(settle_human_rolling_weeks)
        except Exception:
            logger.exception("Play weekly reward settlement failed")
        try:
            from .service import maintain_full_leaderboards
            await asyncio.to_thread(maintain_full_leaderboards)
        except Exception:
            logger.exception("Play leaderboard maintenance failed")
        await asyncio.sleep(60)


async def analysis_storage_loop():
    while True:
        try:
            await asyncio.to_thread(cleanup_expired_jobs)
            await asyncio.to_thread(cleanup_artifacts)
        except Exception:
            logger.exception("Analysis storage maintenance failed")
        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    SingletonConfig()
    init_auth_db()
    init_db()
    from .verse_history import start_worker as start_verse_worker

    start_verse_worker()
    start_analysis_worker()
    reward_task = asyncio.create_task(reward_loop())
    storage_task = asyncio.create_task(analysis_storage_loop())
    try:
        yield
    finally:
        reward_task.cancel()
        storage_task.cancel()
        try:
            await reward_task
        except asyncio.CancelledError:
            pass
        try:
            await storage_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="2048 Play API", lifespan=lifespan)
app.add_middleware(DailyActivityMiddleware, site='play')
app.add_middleware(DisplayCompression)
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(profile_router)
app.include_router(quota_router)
app.include_router(human_router)
app.include_router(analysis_history_router)


@app.post("/api/analysis/jobs")
async def analysis_upload(
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
        if descriptor.get("_provider") == "remote" and not descriptor.get("_available", False):
            raise HTTPException(503, detail={
                "code": "REMOTE_TABLEBASE_OFFLINE",
                "message": "The selected tablebase is temporarily unavailable.",
            })
        check_analysis_capacity(int(user["id"]), len(files))
        for upload in files:
            saved = await save_upload_file(
                upload,
                allowed_extensions=allowed_extensions_for_kind("analysis"),
                max_bytes=get_max_upload_bytes_for_kind("analysis"),
            )
            uploads.append(register_upload(
                saved, kind="analysis", user_id=int(user["id"]),
                session_id=int(user["session_id"]),
            ))
        reservations = reserve_operation_tokens_many(
            user_id=int(user["id"]), session_id=int(user["session_id"]),
            operation_key="analysis_per_replay",
            full_patterns=[full_pattern] * len(uploads),
        )
        job = create_analysis_job(
            uploads=uploads, pattern=normalized_pattern, target=normalized_target,
            user_id=int(user["id"]), session_id=int(user["session_id"]),
            quota_reservations=reservations,
        )
        job_started = True
        record_usage(
            user_id=int(user["id"]), session_id=int(user["session_id"]),
            event_type="analysis_job", quota_key="analysis_job", cost=0,
            metadata={"pattern": pattern, "target": target, "total": len(uploads)},
            ip_address=client_ip(request),
        )
    except InsufficientTokens as exc:
        for reservation in reservations:
            cancel_reservation(reservation, reason="analysis_job_not_created")
        raise HTTPException(402, detail=exc.payload) from exc
    except AnalysisQueueFull as exc:
        for reservation in reservations:
            cancel_reservation(reservation, reason="analysis_job_not_created")
        raise HTTPException(429, detail={"code": str(exc)}, headers={"Retry-After": "10"}) from exc
    except ValueError as exc:
        for reservation in reservations:
            cancel_reservation(reservation, reason="analysis_job_not_created")
        raise HTTPException(400, detail=str(exc)) from exc
    except Exception:
        if not job_started:
            for reservation in reservations:
                cancel_reservation(reservation, reason="analysis_job_not_created")
        raise
    finally:
        if not job_started:
            for upload_record in uploads:
                delete_upload(upload_record.upload_id, int(user["id"]))
    return {
        "job_id": job.job_id,
        "total": job.total,
        "token_balance": get_token_balance(int(user["id"])),
    }


@app.get("/api/analysis/jobs/{job_id}")
def analysis_status(job_id: str, request: Request):
    try:
        return analysis_job_payload(get_analysis_job(job_id, user_id=require_user(request)["id"]))
    except FileNotFoundError as exc:
        raise HTTPException(404, "Analysis job not found.") from exc


@app.get("/api/analysis/jobs/{job_id}/download")
def analysis_download(job_id: str, request: Request):
    user = require_user(request)
    try:
        job = get_analysis_job(job_id, user_id=user["id"])
    except FileNotFoundError as exc:
        raise HTTPException(404, "Analysis job not found.") from exc
    if job.status != "finished" or job.zip_path is None:
        raise HTTPException(409, "Analysis job is not finished.")
    return build_file_download_response(
        job.zip_path,
        filename=analysis_download_filename(job, user),
        media_type="application/zip",
        root=get_download_root(),
        preserve_unicode_filename=True,
    )


@app.get("/api/human/health")
def health():
    return {"ok": True}
