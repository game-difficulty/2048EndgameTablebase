from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from .analysis import normalize_target_value
from .analysis_core import Analyzer
from .auth.db import auth_db
from .auth.service import iso, utcnow
from .cloud_files import (
    UploadRecord,
    get_download_root,
    register_download_path,
    sanitize_download_filename,
)


@dataclass
class AnalysisJob:
    job_id: str
    user_id: int
    session_id: int | None
    pattern: str
    target: str
    target_value: int
    full_pattern: str
    input_paths: list[Path]
    input_names: dict[str, str]
    output_dir: Path
    total: int
    status: str = "queued"
    completed: int = 0
    done: int = 0
    failed: int = 0
    current_file: str = ""
    entries: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    zip_path: Path | None = None


JOBS: dict[str, AnalysisJob] = {}
JOB_LOCK = threading.Lock()
DEFAULT_ANALYSIS_RESULT_TTL_SECONDS = 5 * 60


def get_analysis_result_ttl_seconds(
    default: int = DEFAULT_ANALYSIS_RESULT_TTL_SECONDS,
) -> int:
    try:
        return max(1, int(os.getenv("CLOUD_ANALYSIS_RESULT_TTL_SECONDS", str(default))))
    except ValueError:
        return default


def get_analysis_root() -> Path:
    root = get_download_root().parent / "analysis_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _public_entry(path: Path, status: str, message: str = "") -> dict[str, Any]:
    payload = {
        "filename": sanitize_download_filename(path.name),
        "status": status,
    }
    if message:
        payload["message"] = message
    return payload


def _run_one_file(job: AnalysisJob, path: Path) -> dict[str, Any]:
    analyzer = Analyzer(
        file_path=str(path),
        pattern=job.pattern,
        target=job.target_value,
        full_pattern=job.full_pattern,
        target_path=str(job.output_dir),
        source_filename=job.input_names.get(str(path), path.name),
    )
    analyzer.generate_reports()
    return _public_entry(path, "done")


def _build_job_zip(job: AnalysisJob) -> Path:
    zip_name = f"analysis_{job.full_pattern}_{job.job_id[:8]}.zip"
    zip_path = get_download_root() / zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in job.output_dir.iterdir():
            if item.is_file():
                archive.write(item, arcname=sanitize_download_filename(item.name))
    register_download_path(
        zip_path,
        filename=zip_name,
        media_type="application/zip",
        user_id=job.user_id,
        session_id=job.session_id,
    )
    return zip_path


def _persist_job(job: AnalysisJob) -> None:
    with auth_db() as db:
        db.execute(
            """
            INSERT OR REPLACE INTO analysis_jobs
            (job_id, user_id, session_id, pattern, target, status, total, done, failed,
             output_dir, zip_path, created_at, updated_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.job_id,
                job.user_id,
                job.session_id,
                job.pattern,
                job.target,
                job.status,
                job.total,
                job.done,
                job.failed,
                str(job.output_dir),
                str(job.zip_path) if job.zip_path else None,
                iso_from_timestamp(job.created_at),
                iso_from_timestamp(job.updated_at),
                iso(utcnow() + timedelta(seconds=get_analysis_result_ttl_seconds())),
            ),
        )


def iso_from_timestamp(value: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _run_job(job_id: str) -> None:
    with JOB_LOCK:
        job = JOBS[job_id]
        job.status = "running"
        job.updated_at = time.time()

    try:
        for path in list(job.input_paths):
            try:
                entry = _run_one_file(job, path)
                done_increment = 1
                failed_increment = 0
            except Exception as exc:
                entry = _public_entry(path, "failed", str(exc))
                done_increment = 0
                failed_increment = 1
            with JOB_LOCK:
                job.completed += 1
                job.done += done_increment
                job.failed += failed_increment
                job.current_file = sanitize_download_filename(path.name)
                job.entries.append(entry)
                job.updated_at = time.time()
                _persist_job(job)

        with JOB_LOCK:
            job.zip_path = _build_job_zip(job)
            job.status = "finished"
            job.updated_at = time.time()
            _persist_job(job)
    except Exception as exc:
        with JOB_LOCK:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = time.time()
            _persist_job(job)


def create_analysis_job(
    *,
    uploads: list[UploadRecord],
    pattern: str,
    target: str,
    user_id: int,
    session_id: int | None = None,
) -> AnalysisJob:
    cleanup_expired_jobs(max_age_seconds=get_analysis_result_ttl_seconds())
    target_tile, target_value, numeric_target = normalize_target_value(target)
    full_pattern = f"{pattern}_{numeric_target}"
    job_id = uuid.uuid4().hex
    output_dir = get_analysis_root() / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    job = AnalysisJob(
        job_id=job_id,
        user_id=user_id,
        session_id=session_id,
        pattern=pattern,
        target=target_tile,
        target_value=target_value,
        full_pattern=full_pattern,
        input_paths=[record.path for record in uploads],
        input_names={str(record.path): record.filename for record in uploads},
        output_dir=output_dir,
        total=len(uploads),
    )
    with JOB_LOCK:
        JOBS[job_id] = job
    _persist_job(job)
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return job


def get_analysis_job(job_id: str, user_id: int | None = None) -> AnalysisJob:
    with JOB_LOCK:
        job = JOBS.get(str(job_id or ""))
        if job is None:
            raise FileNotFoundError("Analysis job not found.")
        if user_id is not None and int(job.user_id) != int(user_id):
            raise FileNotFoundError("Analysis job not found.")
        return job


def analysis_job_payload(job: AnalysisJob) -> dict[str, Any]:
    download_url = (
        f"/api/analysis/jobs/{job.job_id}/download"
        if job.status == "finished" and job.zip_path is not None
        else ""
    )
    return {
        "job_id": job.job_id,
        "pattern": job.pattern,
        "target": job.target,
        "status": job.status,
        "completed": job.completed,
        "total": job.total,
        "done": job.done,
        "failed": job.failed,
        "current_file": job.current_file,
        "entries": job.entries[-8:],
        "message": job.error,
        "download_url": download_url,
    }


def cleanup_expired_jobs(
    now: float | None = None,
    max_age_seconds: int | None = None,
) -> int:
    cutoff = (time.time() if now is None else now) - (
        max_age_seconds
        if max_age_seconds is not None
        else get_analysis_result_ttl_seconds()
    )
    removed = 0
    with JOB_LOCK:
        for job_id, job in list(JOBS.items()):
            if job.status in {"queued", "running"}:
                continue
            cleanup_timestamp = job.updated_at or job.created_at
            if cleanup_timestamp >= cutoff:
                continue
            _remove_job_files(job)
            JOBS.pop(job_id, None)
            removed += 1
    removed += _remove_expired_orphan_files(cutoff)
    return removed


def _remove_job_files(job: AnalysisJob) -> None:
    for path in list(job.input_paths):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    if job.zip_path is not None:
        try:
            job.zip_path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        shutil.rmtree(job.output_dir, ignore_errors=True)
    except OSError:
        pass


def _is_older_than(path: Path, cutoff: float) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    return min(stat.st_mtime, stat.st_ctime) < cutoff


def _remove_expired_orphan_files(cutoff: float) -> int:
    removed = 0
    analysis_root = get_analysis_root()
    if analysis_root.exists():
        active_dirs = set()
        with JOB_LOCK:
            active_dirs = {job.output_dir.resolve() for job in JOBS.values()}
        for item in analysis_root.iterdir():
            try:
                resolved = item.resolve()
                if resolved in active_dirs or not _is_older_than(item, cutoff):
                    continue
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                    removed += 1
                elif item.is_file():
                    item.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue

    download_root = get_download_root()
    if download_root.exists():
        active_zips = set()
        with JOB_LOCK:
            active_zips = {
                job.zip_path.resolve()
                for job in JOBS.values()
                if job.zip_path is not None
            }
        for item in download_root.glob("analysis_*.zip"):
            try:
                resolved = item.resolve()
                if resolved in active_zips or not _is_older_than(item, cutoff):
                    continue
                item.unlink(missing_ok=True)
                removed += 1
            except OSError:
                continue

    return removed
