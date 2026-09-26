"""Play-site API process, isolated from the main and live application processes.

Nginx serves the built frontend and forwards only /api/ to this process. The
account database is shared with the main site; game archives use their own DB.
Unlike local_app, this application has no preview account or loopback bypass.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import logging

from fastapi import FastAPI, HTTPException, Request

from backend.admin.routes import router as admin_router
from backend.auth.db import init_auth_db
from backend.auth.dependencies import require_user
from backend.auth.routes import router as auth_router
from backend.cloud_analysis_jobs import (
    analysis_download_filename,
    analysis_job_payload,
    get_analysis_job,
    start_analysis_worker,
    cleanup_expired_jobs,
)
from backend.analysis_history import cleanup_artifacts, router as analysis_history_router
from backend.cloud_files import build_file_download_response, get_download_root
from backend.http_compression import DisplayCompression
from backend.profile.routes import router as profile_router
from backend.quota.routes import router as quota_router
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
app.add_middleware(DisplayCompression)
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(profile_router)
app.include_router(quota_router)
app.include_router(human_router)
app.include_router(analysis_history_router)


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
