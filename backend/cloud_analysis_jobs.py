from __future__ import annotations

import os
import json
import logging
import re
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .analysis import normalize_target_value
from .analysis_core import Analyzer
from .auth.db import auth_db
from .auth.service import iso, utcnow
from .cloud_files import (
    UploadRecord,
    delete_upload,
    get_download_root,
    register_download_path,
    sanitize_download_filename,
)
from .quota.service import cancel_reservation, finalize_reservation, get_token_balance, load_token_reservation
from .remote_workers.errors import RemoteTablebaseError


@dataclass
class AnalysisWorkItem:
    path: Path
    filename: str
    pattern: str
    target: str
    reservation: Any = None
    source_run_id: str | None = None
    upload_id: str | None = None
    history_item_id: int | None = None
    subject_user_id: int | None = None
    listing_snapshot: bool = True
    source_ended_at: float | None = None


class AnalysisQueueFull(ValueError):
    pass


class AnalysisLeaseLost(RuntimeError):
    pass


class AnalysisInputUnavailable(RuntimeError):
    pass


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
    quota_reservations: list[Any] = field(default_factory=list, repr=False)
    work_items: list[AnalysisWorkItem] = field(default_factory=list, repr=False)


JOBS: dict[str, AnalysisJob] = {}
JOB_LOCK = threading.Lock()
DEFAULT_ANALYSIS_RESULT_TTL_SECONDS = 60 * 60
logger = logging.getLogger(__name__)
MAX_WAITING_ANALYSIS_ITEMS = 60
_WORKER_ID = uuid.uuid4().hex
_WORKER_STARTED = False
_WORKER_START_LOCK = threading.Lock()


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


def _ensure_queue(db) -> None:
    db.execute("""CREATE TABLE IF NOT EXISTS analysis_queue (
        job_id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, total INTEGER NOT NULL,
        pending INTEGER NOT NULL, status TEXT NOT NULL, lease_owner TEXT,
        lease_until REAL NOT NULL DEFAULT 0, created_at REAL NOT NULL
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_analysis_queue_active ON analysis_queue(status,created_at)")


def _manifest_path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", str(job_id or "")):
        raise ValueError("invalid_analysis_job_id")
    return get_analysis_root() / job_id / "manifest.json"


def _write_manifest(job: AnalysisJob) -> None:
    data = {
        "job_id": job.job_id, "user_id": job.user_id, "session_id": job.session_id,
        "pattern": job.pattern, "target": job.target, "target_value": job.target_value,
        "full_pattern": job.full_pattern, "total": job.total,
        "completed": job.completed, "done": job.done, "failed": job.failed,
        "current_file": job.current_file, "entries": job.entries, "error": job.error,
        "created_at": job.created_at, "updated_at": job.updated_at,
        "zip_path": str(job.zip_path) if job.zip_path else None,
        "items": [{"path": str(item.path), "filename": item.filename,
                   "pattern": item.pattern, "target": item.target,
                   "source_run_id": item.source_run_id,
                   "upload_id": item.upload_id,
                   "history_item_id": item.history_item_id,
                   "subject_user_id": item.subject_user_id,
                   "listing_snapshot": item.listing_snapshot,
                   "source_ended_at": item.source_ended_at,
                   "reservation_id": getattr(item.reservation, "ledger_id", None)}
                  for item in job.work_items],
    }
    path = _manifest_path(job.job_id)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _load_job(job_id: str, *, restore_reservations: bool = True) -> AnalysisJob:
    data = json.loads(_manifest_path(job_id).read_text(encoding="utf-8"))
    items = [AnalysisWorkItem(
        Path(item["path"]), item["filename"], item["pattern"], item["target"],
        load_token_reservation(item.get("reservation_id")) if restore_reservations else None,
        item.get("source_run_id"), item.get("upload_id"), item.get("history_item_id"),
        item.get("subject_user_id"), bool(item.get("listing_snapshot", True)),
        item.get("source_ended_at"))
        for item in data["items"]]
    missing_sources = {item.source_run_id for item in items
                       if item.source_run_id and item.subject_user_id is None}
    if missing_sources:
        from .human_play.store import database
        placeholders = ",".join("?" for _ in missing_sources)
        with database() as db:
            source_rows = db.execute(f"""SELECT id,user_id,ended FROM human_runs
                WHERE id IN ({placeholders})""", sorted(missing_sources)).fetchall()
        sources = {row["id"]: row for row in source_rows}
        for item in items:
            row = sources.get(item.source_run_id)
            if row and item.subject_user_id is None:
                item.subject_user_id = int(row["user_id"])
                item.source_ended_at = row["ended"]
    job = AnalysisJob(
        job_id=data["job_id"], user_id=data["user_id"], session_id=data["session_id"],
        pattern=data["pattern"], target=data["target"], target_value=data["target_value"],
        full_pattern=data["full_pattern"], input_paths=list(dict.fromkeys(item.path for item in items)),
        input_names={str(item.path): item.filename for item in items},
        output_dir=get_analysis_root() / job_id, total=data["total"],
        completed=data["completed"], done=data["done"], failed=data["failed"],
        current_file=data["current_file"], entries=data["entries"], error=data["error"],
        created_at=data["created_at"], updated_at=data["updated_at"],
        zip_path=Path(data["zip_path"]) if data["zip_path"] else None,
        work_items=items, quota_reservations=[item.reservation for item in items],
    )
    return job


def _admit_job(job: AnalysisJob) -> None:
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        _ensure_queue(db)
        now = time.time()
        # Expired leases can be reclaimed after a process crash.
        db.execute("UPDATE analysis_queue SET status='queued',lease_owner=NULL WHERE status='running' AND lease_until<?", (now,))
        waiting = db.execute("""SELECT COALESCE(SUM(CASE WHEN status='queued' THEN pending
            ELSE max(0,pending-1) END),0) FROM analysis_queue WHERE status IN ('queued','running')""").fetchone()[0]
        active = db.execute("SELECT status FROM analysis_queue WHERE user_id=? AND status IN ('queued','running')", (job.user_id,)).fetchall()
        if sum(row["status"] == "queued" for row in active) >= 1 or len(active) >= 2:
            raise AnalysisQueueFull("user_analysis_queue_full")
        if waiting + job.total > MAX_WAITING_ANALYSIS_ITEMS:
            raise AnalysisQueueFull("analysis_queue_full")
        db.execute("INSERT INTO analysis_queue(job_id,user_id,total,pending,status,created_at) VALUES(?,?,?,?,?,?)",
                   (job.job_id, job.user_id, job.total, job.total, "queued", job.created_at))


def check_analysis_capacity(user_id: int, item_count: int) -> None:
    """Cheap early rejection; _admit_job still performs the atomic final check."""
    if not 1 <= item_count <= MAX_WAITING_ANALYSIS_ITEMS:
        raise AnalysisQueueFull("invalid_analysis_batch")
    with auth_db() as db:
        _ensure_queue(db)
        waiting = db.execute("""SELECT COALESCE(SUM(CASE WHEN status='queued' THEN pending
            ELSE max(0,pending-1) END),0) FROM analysis_queue WHERE status IN ('queued','running')""").fetchone()[0]
        active = [row["status"] for row in db.execute(
            "SELECT status FROM analysis_queue WHERE user_id=? AND status IN ('queued','running')", (user_id,))]
    if active.count("queued") >= 1 or len(active) >= 2:
        raise AnalysisQueueFull("user_analysis_queue_full")
    if waiting + item_count > MAX_WAITING_ANALYSIS_ITEMS:
        raise AnalysisQueueFull("analysis_queue_full")


def _claim_job() -> str | None:
    now = time.time()
    with auth_db() as db:
        _ensure_queue(db)
        has_work = db.execute("""SELECT 1 FROM analysis_queue
            WHERE status='queued' OR (status='running' AND lease_until<?) LIMIT 1""",
            (now,)).fetchone()
    if not has_work:
        return None
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        _ensure_queue(db)
        now = time.time()
        db.execute("UPDATE analysis_queue SET status='queued',lease_owner=NULL WHERE status='running' AND lease_until<?", (now,))
        if db.execute("SELECT 1 FROM analysis_queue WHERE status='running' LIMIT 1").fetchone():
            return None
        row = db.execute("SELECT job_id FROM analysis_queue WHERE status='queued' ORDER BY created_at,job_id LIMIT 1").fetchone()
        if not row:
            return None
        db.execute("UPDATE analysis_queue SET status='running',lease_owner=?,lease_until=? WHERE job_id=?",
                   (_WORKER_ID, now + 45, row["job_id"]))
        return row["job_id"]


def _renew_lease(job_id: str, stop: threading.Event) -> None:
    while not stop.wait(10):
        try:
            with auth_db() as db:
                db.execute("UPDATE analysis_queue SET lease_until=? WHERE job_id=? AND lease_owner=? AND status='running'",
                           (time.time() + 45, job_id, _WORKER_ID))
        except Exception:
            # One busy checkpoint must not permanently stop lease renewal.
            continue


def _require_lease(job_id: str) -> None:
    with auth_db() as db:
        row = db.execute("SELECT lease_owner,lease_until,status FROM analysis_queue WHERE job_id=?", (job_id,)).fetchone()
    if not row or row["lease_owner"] != _WORKER_ID or row["status"] != "running" or row["lease_until"] <= time.time():
        raise AnalysisLeaseLost(job_id)


def _worker_loop() -> None:
    while True:
        try:
            job_id = _claim_job()
            if job_id:
                with JOB_LOCK:
                    job = JOBS.get(job_id)
                    if job is None:
                        job = _load_job(job_id)
                        JOBS[job_id] = job
                stop = threading.Event()
                heartbeat = threading.Thread(target=_renew_lease, args=(job_id, stop), daemon=True)
                heartbeat.start()
                try:
                    _run_job(job_id)
                finally:
                    stop.set()
                    heartbeat.join(timeout=1)
                    with auth_db() as db:
                        db.execute("UPDATE analysis_queue SET status=?,pending=?,lease_owner=NULL,lease_until=0,created_at=? "
                                   "WHERE job_id=? AND lease_owner=?",
                                   (job.status, job.total - job.completed, time.time(), job_id, _WORKER_ID))
                continue
        except Exception:
            logger.exception("Analysis worker failed to poll or execute a job")
            time.sleep(5)
            continue
        time.sleep(1)


def start_analysis_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_START_LOCK:
        if not _WORKER_STARTED:
            threading.Thread(target=_worker_loop, name="analysis-worker", daemon=True).start()
            _WORKER_STARTED = True


def _public_entry(path: Path, status: str, message: str = "") -> dict[str, Any]:
    payload = {
        "filename": sanitize_download_filename(path.name),
        "status": status,
    }
    if message:
        payload["message"] = message
    return payload


def _run_one_file(job: AnalysisJob, item: AnalysisWorkItem, index: int) -> tuple[dict[str, Any], dict | None]:
    source = None
    timing_lossy = False
    if item.source_run_id:
        from .human_play.store import database
        with database() as db:
            from .human_play.service import RANKABLE_SQL
            still_visible = db.execute(f"""SELECT source,state,variant FROM human_runs WHERE id=? AND user_id=?
                AND status='sealed' AND archive IS NOT NULL AND has_replay=1 AND visible=1
                AND (?=? OR ({RANKABLE_SQL}))""",
                (item.source_run_id, item.subject_user_id, job.user_id, item.subject_user_id)).fetchone()
        if not still_visible:
            raise AnalysisInputUnavailable("The archived game is no longer available for analysis.")
        source = still_visible["source"]
        timing_lossy = source != "native" and json.loads(still_visible["state"]).get("replay_timing_version") != 2
    # Separate output folders avoid collisions when a single replay is analyzed
    # under more than one formation or target.
    target_name = sanitize_download_filename(f"{index + 1:02d}_{item.pattern}_{item.target}")
    target_dir = job.output_dir / "items" / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_tile, target_value, numeric_target = normalize_target_value(item.target)
    analyzer = Analyzer(
        file_path=str(item.path),
        pattern=item.pattern,
        target=target_value,
        full_pattern=f"{item.pattern}_{numeric_target}",
        target_path=str(target_dir),
        source_filename=item.filename,
    )
    analyzer.generate_reports()
    from .analysis_history import publish_segments
    published = publish_segments(item_id=item.history_item_id, analyzer=analyzer,
                                 pattern=item.pattern, target=target_tile)
    summary = None
    if item.source_run_id and hasattr(analyzer, "segment_summaries"):
        from .human_play.analysis_summary import (
            POSTER_GOALS, build_summary, intervals_from_analysis_input, poster_goal_tile,
        )
        summary = build_summary(published,
                                intervals_from_analysis_input(item.path),
                                source=source, timing_lossy=timing_lossy)
        summary["goal_tile"] = poster_goal_tile(item.pattern, target_tile, still_visible["variant"])
        summary["aggregate"]["poster_eligible"] = (
            summary["aggregate"]["stage_count"] > 0 and summary["goal_tile"] in POSTER_GOALS)
    artifacts = [{"artifact_id": segment.get("artifact_id"), "segment_index": stage_index,
                  "source_start_index": int(segment["start_index"]),
                  "source_end_index": int(segment["end_index"])}
                 for stage_index, segment in enumerate(published) if segment.get("artifact_id")]
    return ({**_public_entry(item.path, "done"), "pattern": item.pattern,
             "target": target_tile, "artifacts": artifacts,
             "stage_count": len(published)}, summary)


def _build_job_zip(job: AnalysisJob) -> Path:
    zip_name = f"analysis_{job.full_pattern}_{job.job_id[:8]}.zip"
    zip_path = get_download_root() / zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in job.output_dir.rglob("*"):
            if item.is_file():
                if item.name == "manifest.json" or "inputs" in item.relative_to(job.output_dir).parts:
                    continue
                archive.write(item, arcname=item.relative_to(job.output_dir).as_posix())
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
    from .analysis_history import set_item_status, set_job_status
    set_job_status(job.job_id, "running", done=job.done, failed=job.failed)

    try:
        processed_reservations = job.completed
        for index in range(job.completed, len(job.work_items)):
            item = job.work_items[index]
            path = item.path
            reservation = item.reservation
            try:
                _require_lease(job.job_id)
                entry, summary = _run_one_file(job, item, index)
                _require_lease(job.job_id)
                if summary is not None:
                    from .human_play.analysis_summary import save_summary
                    entry["summary_id"] = save_summary(
                        run_id=item.source_run_id, user_id=item.subject_user_id,
                        pattern=item.pattern, target=entry["target"],
                        job_id=job.job_id, summary=summary,
                        listed=bool(item.listing_snapshot))
                    from .analysis_history import promote_library_artifacts
                    entry["library_admitted"] = promote_library_artifacts(
                        item.history_item_id, entry["summary_id"],
                        source_run_id=item.source_run_id,
                        subject_user_id=item.subject_user_id,
                        run_ended_at=item.source_ended_at,
                        listed=bool(item.listing_snapshot),
                    )
                    entry["poster_eligible"] = summary["aggregate"]["poster_eligible"]
                    set_item_status(item.history_item_id, "done", stage_count=entry.get("stage_count", 0),
                                    summary_id=entry["summary_id"])
                else:
                    set_item_status(item.history_item_id, "done", stage_count=entry.get("stage_count", 0))
                    from .analysis_history import enforce_limits
                    enforce_limits(job.user_id)
                finalize_reservation(
                    reservation,
                    actual_operation_key="analysis_per_replay",
                    metadata={"job_id": job.job_id, "filename": item.filename},
                )
                done_increment = 1
                failed_increment = 0
            except AnalysisLeaseLost:
                raise
            except AnalysisInputUnavailable as exc:
                from .analysis_history import discard_item_artifacts
                discard_item_artifacts(item.history_item_id)
                cancel_reservation(reservation, reason="analysis_input_unavailable",
                                   metadata={"job_id": job.job_id, "filename": item.filename})
                entry = {**_public_entry(path, "failed", str(exc)), "pattern": item.pattern, "target": item.target}
                done_increment = 0
                failed_increment = 1
                set_item_status(item.history_item_id, "failed", error_code="analysis_input_unavailable")
            except RemoteTablebaseError as exc:
                from .analysis_history import discard_item_artifacts
                discard_item_artifacts(item.history_item_id)
                cancel_reservation(
                    reservation,
                    reason=exc.code.lower(),
                    metadata={"job_id": job.job_id, "filename": item.filename},
                )
                entry = {**_public_entry(
                    path, "failed", "The selected tablebase is temporarily unavailable.",
                ), "pattern": item.pattern, "target": item.target}
                done_increment = 0
                failed_increment = 1
                set_item_status(item.history_item_id, "failed", error_code=exc.code.lower())
            except Exception as exc:
                from .analysis_history import discard_item_artifacts
                discard_item_artifacts(item.history_item_id)
                finalize_reservation(
                    reservation,
                    actual_operation_key="analysis_per_replay",
                    metadata={
                        "job_id": job.job_id,
                        "filename": item.filename,
                        "error": type(exc).__name__,
                    },
                )
                entry = {**_public_entry(path, "failed", str(exc)), "pattern": item.pattern, "target": item.target}
                done_increment = 0
                failed_increment = 1
                set_item_status(item.history_item_id, "failed", error_code=type(exc).__name__)
            _require_lease(job.job_id)
            processed_reservations = index + 1
            with JOB_LOCK:
                job.completed += 1
                job.done += done_increment
                job.failed += failed_increment
                job.current_file = sanitize_download_filename(item.filename)
                job.entries.append(entry)
                job.updated_at = time.time()
                _write_manifest(job)
                _persist_job(job)
                with auth_db() as db:
                    db.execute("UPDATE analysis_queue SET pending=? WHERE job_id=? AND lease_owner=?",
                               (job.total - job.completed, job.job_id, _WORKER_ID))
            if job.completed < job.total:
                with JOB_LOCK:
                    job.status = "queued"
                    job.updated_at = time.time()
                    _write_manifest(job)
                    _persist_job(job)
                set_job_status(job.job_id, "queued", done=job.done, failed=job.failed)
                return

        with JOB_LOCK:
            _require_lease(job.job_id)
            # ZIP remains a one-hour task result. The durable public library
            # itself only registers compact stage replays.
            job.zip_path = _build_job_zip(job)
            _require_lease(job.job_id)
            job.status = "finished"
            job.updated_at = time.time()
            _write_manifest(job)
            _persist_job(job)
            history_status = "failed" if job.failed and not job.done else ("partial" if job.failed else "finished")
            set_job_status(job.job_id, history_status, done=job.done, failed=job.failed, completed=True)
    except AnalysisLeaseLost:
        return
    except Exception as exc:
        for reservation in job.quota_reservations[processed_reservations:]:
            cancel_reservation(
                reservation,
                reason="analysis_job_failed",
                metadata={"job_id": job.job_id, "error": type(exc).__name__},
            )
        with JOB_LOCK:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = time.time()
            _write_manifest(job)
            _persist_job(job)
        set_job_status(job.job_id, "failed", done=job.done, failed=job.failed, completed=True)
    finally:
        if job.status in {"finished", "failed"}:
            for item in job.work_items:
                if item.upload_id:
                    delete_upload(item.upload_id, job.user_id)


def create_analysis_job(
    *,
    uploads: list[UploadRecord] | None = None,
    pattern: str = "",
    target: str = "",
    user_id: int,
    session_id: int | None = None,
    quota_reservations: list[Any] | None = None,
    work_items: list[AnalysisWorkItem] | None = None,
) -> AnalysisJob:
    cleanup_expired_jobs(max_age_seconds=get_analysis_result_ttl_seconds())
    if work_items is None:
        uploads = uploads or []
        reservations = list(quota_reservations or [])
        work_items = [AnalysisWorkItem(record.path, record.filename, pattern, target,
                                       reservations[index] if index < len(reservations) else None,
                                       None, record.upload_id)
                      for index, record in enumerate(uploads)]
    if not work_items or len(work_items) > MAX_WAITING_ANALYSIS_ITEMS:
        raise AnalysisQueueFull("invalid_analysis_batch")
    target_tile, target_value, numeric_target = normalize_target_value(work_items[0].target)
    pattern = work_items[0].pattern
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
        input_paths=list(dict.fromkeys(item.path for item in work_items)),
        input_names={str(item.path): item.filename for item in work_items},
        output_dir=output_dir,
        total=len(work_items),
        quota_reservations=[item.reservation for item in work_items],
        work_items=work_items,
    )
    try:
        input_dir = output_dir / "inputs"
        input_dir.mkdir(exist_ok=True)
        copied: dict[Path, Path] = {}
        for index, item in enumerate(job.work_items):
            if item.path not in copied:
                source = item.path
                target_path = input_dir / f"{index + 1:02d}_{sanitize_download_filename(item.filename)}"
                shutil.copyfile(source, target_path)
                copied[source] = target_path
            item.path = copied[item.path]
        job.input_paths = list(copied.values())
        job.input_names = {str(item.path): item.filename for item in job.work_items}
        from .analysis_history import register_job
        register_job(job)
        _persist_job(job)
        _write_manifest(job)
        _admit_job(job)
    except Exception:
        with auth_db() as db:
            db.execute("DELETE FROM analysis_jobs WHERE job_id=?", (job_id,))
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    with JOB_LOCK:
        JOBS[job_id] = job
    start_analysis_worker()
    return job


def get_analysis_job(job_id: str, user_id: int | None = None) -> AnalysisJob:
    try:
        job = _load_job(job_id, restore_reservations=False)
    except (OSError, ValueError, KeyError) as exc:
        raise FileNotFoundError("Analysis job not found.") from exc
    if user_id is not None and int(job.user_id) != int(user_id):
        raise FileNotFoundError("Analysis job not found.")
    with auth_db() as db:
        _ensure_queue(db)
        row = db.execute("SELECT status FROM analysis_queue WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        raise FileNotFoundError("Analysis job not found.")
    job.status = row["status"]
    return job


def analysis_job_payload(job: AnalysisJob) -> dict[str, Any]:
    download_url = (
        f"/api/analysis/jobs/{job.job_id}/download"
        if job.status == "finished" and job.zip_path is not None
        else ""
    )
    payload = {
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
        "items": [{"pattern": item.pattern, "target": item.target,
                   "status": job.entries[index]["status"] if index < len(job.entries)
                   else ("running" if job.status == "running" and index == job.completed else "queued"),
                   "message": job.entries[index].get("message", "") if index < len(job.entries) else "",
                   "summary_id": job.entries[index].get("summary_id") if index < len(job.entries) else None,
                   "artifacts": job.entries[index].get("artifacts", []) if index < len(job.entries) else [],
                   "library_admitted": bool(job.entries[index].get("library_admitted")) if index < len(job.entries) else False,
                   "poster_eligible": bool(job.entries[index].get("poster_eligible")) if index < len(job.entries) else False}
                  for index, item in enumerate(job.work_items)],
        "message": job.error,
        "download_url": download_url,
    }
    if job.status in {"finished", "failed"}:
        payload["token_balance"] = get_token_balance(job.user_id)
    return payload


def analysis_download_filename(job: AnalysisJob, user: dict[str, Any]) -> str:
    """Return a stable player-facing name for archived play-site analyses."""
    source_ids = {item.source_run_id for item in job.work_items if item.source_run_id}
    if len(source_ids) != 1:
        return job.zip_path.name if job.zip_path else "analysis.zip"
    try:
        from .human_play.store import database
        with database() as db:
            row = db.execute(
                "SELECT variant,ended,state FROM human_runs WHERE id=? AND user_id=?",
                (source_ids.pop(), job.user_id),
            ).fetchone()
        if not row:
            raise LookupError("analysis_source_missing")
        state = json.loads(row["state"])
        ended = datetime.fromtimestamp(float(row["ended"]), timezone(timedelta(hours=8)))
        played_at = ended.strftime("%Y-%m-%d_%H-%M-%S")
        username = str(user.get("display_name") or user.get("email") or f"user-{job.user_id}").strip()
        score = max(0, int(state.get("score") or 0))
        variant = str(row["variant"] or "2048")
        return f"{username}-{played_at}-{variant}-{score}分.zip"
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, LookupError):
        # Do not expose the internal task id if archived metadata becomes unavailable.
        username = str(user.get("display_name") or f"user-{job.user_id}").strip()
        return f"{username}-2048-对局分析.zip"


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
    with auth_db() as db:
        _ensure_queue(db)
        terminal_ids = [row["job_id"] for row in db.execute(
            "SELECT job_id FROM analysis_queue WHERE status IN ('finished','failed')")]
    for job_id in terminal_ids:
        try:
            job = _load_job(job_id, restore_reservations=False)
        except (OSError, ValueError, KeyError):
            continue
        if job.updated_at >= cutoff:
            continue
        _remove_job_files(job)
        with auth_db() as db:
            db.execute("DELETE FROM analysis_queue WHERE job_id=? AND status IN ('finished','failed')", (job_id,))
            db.execute("DELETE FROM analysis_jobs WHERE job_id=?", (job_id,))
        with JOB_LOCK:
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
    with auth_db() as db:
        _ensure_queue(db)
        protected_ids = {row["job_id"] for row in db.execute("SELECT job_id FROM analysis_queue")}
        protected_ids.update(row["job_id"] for row in db.execute(
            "SELECT job_id FROM analysis_jobs WHERE status IN ('queued','running')"))
        protected_zip_paths = {Path(row["zip_path"]).resolve() for row in db.execute(
            "SELECT zip_path FROM analysis_jobs WHERE zip_path IS NOT NULL AND job_id IN (SELECT job_id FROM analysis_queue)")}
    if analysis_root.exists():
        for item in analysis_root.iterdir():
            try:
                resolved = item.resolve()
                if item.name in protected_ids or not _is_older_than(item, cutoff):
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
        for item in download_root.glob("analysis_*.zip"):
            try:
                resolved = item.resolve()
                if resolved in protected_zip_paths or not _is_older_than(item, cutoff):
                    continue
                item.unlink(missing_ok=True)
                removed += 1
            except OSError:
                continue

    return removed
