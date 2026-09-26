"""Durable metadata and bounded replay artifacts for analysis jobs."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import shutil
import time
import uuid
from collections import defaultdict, deque
from threading import BoundedSemaphore, Lock
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from .auth.db import auth_db
from .auth.dependencies import current_user_from_request, require_user
from .auth.entitlements import SUPPORTER_TIER, ensure_user_entitlements
from .cloud_files import get_upload_root
from engine_core.replay_utils import REPLAY_DTYPE, validate_replay_array


router = APIRouter(prefix="/api/analysis", tags=["analysis-history"])
DEFAULT_USER_ARTIFACT_LIMIT = 50
SUPPORTER_ARTIFACT_LIMIT = 100
GLOBAL_ARTIFACT_BYTES = 4 * 1024**3
MIN_FREE_BYTES = 4 * 1024**3
OPEN_TOKEN_TTL_SECONDS = 60
MAX_CONCURRENT_REPLAY_DOWNLOADS = 4
REPLAY_DOWNLOADS_PER_MINUTE = 20
_download_slots = BoundedSemaphore(MAX_CONCURRENT_REPLAY_DOWNLOADS)
_download_windows: dict[int, deque[float]] = defaultdict(deque)
_download_lock = Lock()


def artifact_root() -> Path:
    shared_default = (Path(os.environ["CLOUD_AUTH_DB"]).expanduser().parent / "analysis-replays"
                      if os.getenv("CLOUD_AUTH_DB") else get_upload_root().parent / "analysis_replays")
    root = Path(os.getenv("CLOUD_ANALYSIS_REPLAY_ROOT") or shared_default)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def init_schema(db) -> None:
    db.executescript("""
    CREATE TABLE IF NOT EXISTS analysis_history_jobs (
      job_id TEXT PRIMARY KEY,
      user_id INTEGER NOT NULL,
      origin TEXT NOT NULL,
      status TEXT NOT NULL,
      total INTEGER NOT NULL DEFAULT 0,
      done INTEGER NOT NULL DEFAULT 0,
      failed INTEGER NOT NULL DEFAULT 0,
      source_run_id TEXT,
      metadata_json TEXT NOT NULL DEFAULT '{}',
      created_at REAL NOT NULL,
      completed_at REAL,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS analysis_history_jobs_user
      ON analysis_history_jobs(user_id,created_at DESC,job_id DESC);
    CREATE TABLE IF NOT EXISTS analysis_history_items (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      job_id TEXT NOT NULL,
      work_index INTEGER NOT NULL,
      source_run_id TEXT,
      source_filename TEXT NOT NULL,
      pattern TEXT NOT NULL,
      target TEXT NOT NULL,
      variant TEXT,
      status TEXT NOT NULL DEFAULT 'queued',
      error_code TEXT,
      stage_count INTEGER NOT NULL DEFAULT 0,
      summary_id INTEGER,
      UNIQUE(job_id,work_index),
      FOREIGN KEY(job_id) REFERENCES analysis_history_jobs(job_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS analysis_history_items_job
      ON analysis_history_items(job_id,work_index);
    CREATE TABLE IF NOT EXISTS analysis_replay_artifacts (
      artifact_id TEXT PRIMARY KEY,
      history_item_id INTEGER NOT NULL,
      segment_index INTEGER NOT NULL,
      relative_path TEXT NOT NULL,
      byte_size INTEGER NOT NULL,
      source_start_index INTEGER NOT NULL,
      source_end_index INTEGER NOT NULL,
      replay_move_count INTEGER NOT NULL,
      evaluated_moves INTEGER NOT NULL,
      goodness_of_fit REAL,
      max_combo INTEGER NOT NULL DEFAULT 0,
      performance_counts_json TEXT NOT NULL DEFAULT '{}',
      pattern TEXT NOT NULL,
      target TEXT NOT NULL,
      variant TEXT,
      use_variant INTEGER NOT NULL DEFAULT 0,
      format_version INTEGER NOT NULL DEFAULT 1,
      created_at REAL NOT NULL,
      deleted_at REAL,
      delete_reason TEXT,
      UNIQUE(history_item_id,segment_index),
      FOREIGN KEY(history_item_id) REFERENCES analysis_history_items(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS analysis_replay_artifacts_active
      ON analysis_replay_artifacts(deleted_at,created_at,artifact_id);
    """)


def register_job(job) -> None:
    origin = "human_archive" if any(item.source_run_id for item in job.work_items) else "main_upload"
    source_ids = {item.source_run_id for item in job.work_items if item.source_run_id}
    now = float(job.created_at)
    with auth_db() as db:
        init_schema(db)
        db.execute("""INSERT OR IGNORE INTO analysis_history_jobs
            (job_id,user_id,origin,status,total,done,failed,source_run_id,metadata_json,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (job.job_id, job.user_id, origin, "queued", job.total, 0, 0,
             next(iter(source_ids)) if len(source_ids) == 1 else None,
             json.dumps({"version": 1}, separators=(",", ":")), now))
        for index, item in enumerate(job.work_items):
            db.execute("""INSERT OR IGNORE INTO analysis_history_items
                (job_id,work_index,source_run_id,source_filename,pattern,target,status)
                VALUES(?,?,?,?,?,?,?)""",
                (job.job_id, index, item.source_run_id, item.filename,
                 item.pattern, str(item.target), "queued"))
            row = db.execute("SELECT id FROM analysis_history_items WHERE job_id=? AND work_index=?",
                             (job.job_id, index)).fetchone()
            item.history_item_id = int(row["id"])


def set_job_status(job_id: str, status: str, *, done: int, failed: int,
                   completed: bool = False) -> None:
    with auth_db() as db:
        init_schema(db)
        db.execute("""UPDATE analysis_history_jobs SET status=?,done=?,failed=?,
            completed_at=CASE WHEN ? THEN COALESCE(completed_at,?) ELSE completed_at END
            WHERE job_id=?""", (status, int(done), int(failed), int(completed), time.time(), job_id))


def set_item_status(item_id: int | None, status: str, *, error_code: str | None = None,
                    stage_count: int | None = None, summary_id: int | None = None,
                    variant: str | None = None) -> None:
    if not item_id:
        return
    fields = ["status=?", "error_code=?"]
    values: list[Any] = [status, (str(error_code)[:160] if error_code else None)]
    for name, value in (("stage_count", stage_count), ("summary_id", summary_id),
                        ("variant", variant)):
        if value is not None:
            fields.append(f"{name}=?")
            values.append(value)
    values.append(int(item_id))
    with auth_db() as db:
        init_schema(db)
        db.execute(f"UPDATE analysis_history_items SET {','.join(fields)} WHERE id=?", values)


def _validate_replay(path: Path) -> int:
    size = path.stat().st_size
    if size <= REPLAY_DTYPE.itemsize or size > 16 * 1024 * 1024 or size % REPLAY_DTYPE.itemsize:
        raise ValueError("analysis_replay_invalid")
    record = np.fromfile(path, dtype=REPLAY_DTYPE)
    if not validate_replay_array(record):
        raise ValueError("analysis_replay_invalid")
    return len(record) - 1


def publish_segments(*, item_id: int, analyzer, pattern: str, target: str) -> list[dict]:
    if not item_id:
        raise ValueError("analysis_history_item_missing")
    root = artifact_root()
    month = datetime.now(timezone.utc).strftime("%Y/%m")
    destination_dir = root / month
    destination_dir.mkdir(parents=True, exist_ok=True)
    published: list[dict] = []
    for index, segment in enumerate(analyzer.segment_summaries):
        replay_path = Path(str(segment.get("replay_path") or ""))
        if not replay_path.is_file():
            continue
        replay_moves = _validate_replay(replay_path)
        segment["replay_move_count"] = replay_moves
        artifact_id = uuid.uuid4().hex
        relative = Path(month) / f"{artifact_id}.rpl"
        destination = root / relative
        temporary = destination.with_suffix(".tmp")
        try:
            try:
                os.link(replay_path, temporary)
            except OSError:
                shutil.copyfile(replay_path, temporary)
            temporary.replace(destination)
            now = time.time()
            with auth_db() as db:
                init_schema(db)
                prior = db.execute("""SELECT artifact_id,relative_path FROM analysis_replay_artifacts
                    WHERE history_item_id=? AND segment_index=?""", (item_id, index)).fetchone()
                if prior:
                    prior_path = root / prior["relative_path"]
                    if prior_path.is_file():
                        destination.unlink(missing_ok=True)
                    else:
                        prior_path.parent.mkdir(parents=True, exist_ok=True)
                        destination.replace(prior_path)
                    artifact_id = prior["artifact_id"]
                    db.execute("""UPDATE analysis_replay_artifacts SET byte_size=?,deleted_at=NULL,
                        delete_reason=NULL,created_at=? WHERE artifact_id=?""",
                               (replay_path.stat().st_size, now, artifact_id))
                else:
                    db.execute("""INSERT INTO analysis_replay_artifacts
                        (artifact_id,history_item_id,segment_index,relative_path,byte_size,
                         source_start_index,source_end_index,replay_move_count,evaluated_moves,
                         goodness_of_fit,max_combo,performance_counts_json,pattern,target,variant,
                         use_variant,format_version,created_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (artifact_id, item_id, index, relative.as_posix(), replay_path.stat().st_size,
                         int(segment["start_index"]), int(segment["end_index"]), replay_moves,
                         int(segment["evaluated_moves"]), segment.get("goodness_of_fit"),
                         int(segment.get("max_combo") or 0),
                         json.dumps(segment.get("performance_counts") or {}, separators=(",", ":")),
                         pattern, str(target), analyzer.variant,
                         int(pattern in __import__("Config").category_info.get("variant", [])), 1, now))
            segment["artifact_id"] = artifact_id
            published.append(segment)
        finally:
            temporary.unlink(missing_ok=True)
    set_item_status(item_id, "done", stage_count=len(published), variant=getattr(analyzer, "variant", None))
    enforce_limits(analyzer_user_id(item_id))
    return published


def analyzer_user_id(item_id: int) -> int:
    with auth_db() as db:
        init_schema(db)
        row = db.execute("""SELECT j.user_id FROM analysis_history_items i
            JOIN analysis_history_jobs j ON j.job_id=i.job_id WHERE i.id=?""", (item_id,)).fetchone()
    if not row:
        raise ValueError("analysis_history_item_missing")
    return int(row["user_id"])


def _delete_artifact(db, row, reason: str) -> None:
    try:
        (artifact_root() / row["relative_path"]).unlink(missing_ok=True)
    except OSError:
        pass
    db.execute("""UPDATE analysis_replay_artifacts SET deleted_at=?,delete_reason=?
        WHERE artifact_id=? AND deleted_at IS NULL""", (time.time(), reason, row["artifact_id"]))


def enforce_limits(user_id: int | None = None) -> int:
    removed = 0
    root = artifact_root()
    with auth_db() as db:
        init_schema(db)
        if user_id is not None:
            entitlements = ensure_user_entitlements(db, int(user_id))
            limit = SUPPORTER_ARTIFACT_LIMIT if entitlements["tier"] == SUPPORTER_TIER else DEFAULT_USER_ARTIFACT_LIMIT
            rows = db.execute("""SELECT a.artifact_id,a.relative_path FROM analysis_replay_artifacts a
                JOIN analysis_history_items i ON i.id=a.history_item_id
                JOIN analysis_history_jobs j ON j.job_id=i.job_id
                WHERE j.user_id=? AND a.deleted_at IS NULL
                ORDER BY a.created_at DESC,a.artifact_id DESC""", (int(user_id),)).fetchall()
            for row in rows[limit:]:
                _delete_artifact(db, row, "user_limit")
                removed += 1
        total = int(db.execute("""SELECT COALESCE(SUM(byte_size),0) FROM analysis_replay_artifacts
            WHERE deleted_at IS NULL""").fetchone()[0])
        try:
            free = shutil.disk_usage(root).free
        except OSError:
            free = MIN_FREE_BYTES
        if total > GLOBAL_ARTIFACT_BYTES or free < MIN_FREE_BYTES:
            rows = db.execute("""SELECT artifact_id,relative_path,byte_size FROM analysis_replay_artifacts
                WHERE deleted_at IS NULL ORDER BY created_at,artifact_id""").fetchall()
            for row in rows:
                if total <= GLOBAL_ARTIFACT_BYTES and free >= MIN_FREE_BYTES:
                    break
                _delete_artifact(db, row, "global_capacity")
                total -= int(row["byte_size"])
                free += int(row["byte_size"])
                removed += 1
    return removed


def cleanup_artifacts() -> int:
    """Reconcile missing files and enforce both capacity safety rails."""
    removed = 0
    root = artifact_root()
    with auth_db() as db:
        init_schema(db)
        rows = db.execute("""SELECT artifact_id,relative_path FROM analysis_replay_artifacts
            WHERE deleted_at IS NULL""").fetchall()
        for row in rows:
            if not (root / row["relative_path"]).is_file():
                db.execute("""UPDATE analysis_replay_artifacts SET deleted_at=?,delete_reason='missing'
                    WHERE artifact_id=?""", (time.time(), row["artifact_id"]))
                removed += 1
        user_ids = [int(row[0]) for row in db.execute("""SELECT DISTINCT j.user_id
            FROM analysis_replay_artifacts a
            JOIN analysis_history_items i ON i.id=a.history_item_id
            JOIN analysis_history_jobs j ON j.job_id=i.job_id
            WHERE a.deleted_at IS NULL""").fetchall()]
    for current_user_id in user_ids:
        removed += enforce_limits(current_user_id)
    removed += enforce_limits(None)
    return removed


def _artifact_payload(row) -> dict:
    active = row["deleted_at"] is None and (artifact_root() / row["relative_path"]).is_file()
    return {
        "artifact_id": row["artifact_id"], "segment_index": row["segment_index"],
        "source_start_index": row["source_start_index"], "source_end_index": row["source_end_index"],
        "replay_move_count": row["replay_move_count"], "evaluated_moves": row["evaluated_moves"],
        "goodness_of_fit": row["goodness_of_fit"], "max_combo": row["max_combo"],
        "pattern": row["pattern"], "target": row["target"], "variant": row["variant"],
        "available": active, "deleted_reason": row["delete_reason"] if not active else None,
    }


def list_history(user_id: int, *, limit: int = 20, cursor: str = "",
                 origin: str = "", status: str = "", variant: str = "",
                 pattern: str = "") -> dict:
    params: list[Any] = [int(user_id)]
    where = ["j.user_id=?"]
    for column, value in (("j.origin", origin), ("j.status", status),
                          ("i.variant", variant), ("i.pattern", pattern)):
        if value:
            where.append(f"EXISTS(SELECT 1 FROM analysis_history_items i WHERE i.job_id=j.job_id AND {column}=?)" if column.startswith("i.") else f"{column}=?")
            params.append(value)
    if cursor:
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            created, job_id = base64.urlsafe_b64decode(padded).decode().split("|", 1)
            where.append("(j.created_at<? OR (j.created_at=? AND j.job_id<?))")
            params.extend((float(created), float(created), job_id))
        except (ValueError, UnicodeDecodeError):
            raise ValueError("invalid_analysis_history_cursor")
    size = min(50, max(1, int(limit)))
    with auth_db() as db:
        init_schema(db)
        jobs = db.execute(f"""SELECT j.* FROM analysis_history_jobs j WHERE {' AND '.join(where)}
            ORDER BY j.created_at DESC,j.job_id DESC LIMIT ?""", (*params, size + 1)).fetchall()
        result = [_job_payload(db, row, include_items=False) for row in jobs[:size]]
    next_cursor = ""
    if len(jobs) > size:
        last = jobs[size - 1]
        next_cursor = base64.urlsafe_b64encode(f"{last['created_at']}|{last['job_id']}".encode()).decode().rstrip("=")
    return {"items": result, "next_cursor": next_cursor}


def _job_payload(db, job, *, include_items: bool) -> dict:
    payload = {key: job[key] for key in ("job_id", "origin", "status", "total", "done", "failed",
                                          "source_run_id", "created_at", "completed_at")}
    if include_items:
        items = db.execute("SELECT * FROM analysis_history_items WHERE job_id=? ORDER BY work_index",
                           (job["job_id"],)).fetchall()
        payload["items"] = []
        for item in items:
            artifacts = db.execute("""SELECT * FROM analysis_replay_artifacts
                WHERE history_item_id=? ORDER BY segment_index""", (item["id"],)).fetchall()
            payload["items"].append({
                **{key: item[key] for key in ("id", "work_index", "source_run_id", "source_filename",
                                               "pattern", "target", "variant", "status", "error_code",
                                               "stage_count", "summary_id")},
                "artifacts": [_artifact_payload(row) for row in artifacts],
            })
    return payload


def get_history(job_id: str, user_id: int) -> dict:
    with auth_db() as db:
        init_schema(db)
        row = db.execute("SELECT * FROM analysis_history_jobs WHERE job_id=? AND user_id=?",
                         (job_id, int(user_id))).fetchone()
        if not row:
            raise FileNotFoundError("analysis_history_not_found")
        return _job_payload(db, row, include_items=True)


def _secret() -> bytes:
    value = str(os.getenv("AUTH_SECRET") or os.getenv("REMOTE_TABLEBASE_WORKER_SECRET") or "")
    if not value:
        value = str(Path(__file__).resolve().parents[1])
    return hashlib.sha256(("analysis-replay|" + value).encode()).digest()


def make_open_token(artifact_id: str, user_id: int) -> str:
    payload = json.dumps({"a": artifact_id, "u": int(user_id), "e": int(time.time()) + OPEN_TOKEN_TTL_SECONDS},
                         separators=(",", ":")).encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    signature = base64.urlsafe_b64encode(hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    return f"{encoded}.{signature}"


def verify_open_token(token: str, artifact_id: str) -> int | None:
    try:
        encoded, signature = token.split(".", 1)
        expected = base64.urlsafe_b64encode(hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest()).decode().rstrip("=")
        if not hmac.compare_digest(signature, expected):
            return None
        data = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if data.get("a") != artifact_id or int(data.get("e", 0)) < int(time.time()):
            return None
        return int(data["u"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def resolve_artifact(artifact_id: str, user_id: int):
    with auth_db() as db:
        init_schema(db)
        row = db.execute("""SELECT a.*,j.user_id,i.source_filename FROM analysis_replay_artifacts a
            JOIN analysis_history_items i ON i.id=a.history_item_id
            JOIN analysis_history_jobs j ON j.job_id=i.job_id
            WHERE a.artifact_id=? AND j.user_id=?""", (artifact_id, int(user_id))).fetchone()
    if not row:
        raise FileNotFoundError("analysis_replay_not_found")
    if row["deleted_at"] is not None:
        raise GoneError(row["delete_reason"] or "expired")
    path = (artifact_root() / row["relative_path"]).resolve()
    if artifact_root() not in path.parents or not path.is_file():
        raise GoneError("missing")
    return row, path


class GoneError(FileNotFoundError):
    pass


def _claim_download(user_id: int) -> None:
    now = time.monotonic()
    with _download_lock:
        window = _download_windows[int(user_id)]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= REPLAY_DOWNLOADS_PER_MINUTE:
            raise HTTPException(429, "Too many replay downloads.")
        window.append(now)
    if not _download_slots.acquire(blocking=False):
        raise HTTPException(503, "Replay download capacity is busy.")


def _release_download() -> None:
    _download_slots.release()


@router.get("/history")
def history_route(request: Request, limit: int = Query(20, ge=1, le=50), cursor: str = "",
                  origin: str = "", status: str = "", variant: str = "", pattern: str = ""):
    user = require_user(request)
    try:
        return list_history(user["id"], limit=limit, cursor=cursor, origin=origin,
                            status=status, variant=variant, pattern=pattern)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/history/{job_id}")
def history_detail_route(job_id: str, request: Request):
    try:
        return get_history(job_id, require_user(request)["id"])
    except FileNotFoundError as exc:
        raise HTTPException(404, "Analysis history not found.") from exc


@router.get("/replays/{artifact_id}")
def replay_route(artifact_id: str, request: Request, token: str = ""):
    user = current_user_from_request(request)
    user_id = int(user["id"]) if user else verify_open_token(token, artifact_id)
    if user_id is None:
        raise HTTPException(401, "Authentication required.")
    _claim_download(user_id)
    try:
        row, path = resolve_artifact(artifact_id, user_id)
    except GoneError as exc:
        _release_download()
        raise HTTPException(410, {"code": "ANALYSIS_REPLAY_EXPIRED", "reason": str(exc)}) from exc
    except FileNotFoundError as exc:
        _release_download()
        raise HTTPException(404, "Analysis replay not found.") from exc
    response = FileResponse(path, media_type="application/octet-stream",
                            filename=f"{row['pattern']}_{row['target']}_stage-{int(row['segment_index']) + 1}.rpl")
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Replay-Pattern"] = f"{row['pattern']}_{row['target']}"
    response.headers["X-Replay-Variant"] = "1" if row["use_variant"] else "0"
    response.headers["X-Replay-Source"] = "Analysis history"
    from starlette.background import BackgroundTask
    response.background = BackgroundTask(_release_download)
    return response


@router.post("/replays/{artifact_id}/open-link")
def replay_open_link_route(artifact_id: str, request: Request):
    user = require_user(request)
    try:
        resolve_artifact(artifact_id, user["id"])
    except GoneError as exc:
        raise HTTPException(410, {"code": "ANALYSIS_REPLAY_EXPIRED"}) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "Analysis replay not found.") from exc
    token = make_open_token(artifact_id, user["id"])
    return {"url": f"https://2048tables.online/?tab=replay#analysisReplay={artifact_id}&token={token}"}
