from __future__ import annotations

import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

from fastapi import UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from .auth.db import auth_db
from .auth.service import iso, utcnow


DEFAULT_UPLOAD_ROOT = Path(tempfile.gettempdir()) / "2048tables-cloud" / "uploads"
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_UPLOAD_TTL_SECONDS = 6 * 60 * 60
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SavedUpload:
    path: Path
    original_filename: str
    safe_filename: str
    size: int
    content_type: str


@dataclass(frozen=True)
class UploadRecord:
    upload_id: str
    path: Path
    filename: str
    size: int
    content_type: str
    kind: str
    created_at: float
    user_id: int | None = None
    session_id: int | None = None


@dataclass(frozen=True)
class DownloadRecord:
    download_id: str
    path: Path
    filename: str
    media_type: str
    created_at: float
    user_id: int | None = None
    session_id: int | None = None


UPLOAD_REGISTRY: dict[str, UploadRecord] = {}
DOWNLOAD_REGISTRY: dict[str, DownloadRecord] = {}


ALLOWED_UPLOAD_EXTENSIONS = {
    "replay": {".rpl"},
    "analysis": {".txt", ".vrs", ".rpl", ""},
}


def get_upload_root() -> Path:
    return Path(os.getenv("CLOUD_UPLOAD_ROOT") or DEFAULT_UPLOAD_ROOT)


def get_download_root() -> Path:
    return get_upload_root().parent / "downloads"


def get_max_upload_bytes(default: int = DEFAULT_MAX_UPLOAD_BYTES) -> int:
    try:
        return max(1, int(os.getenv("CLOUD_MAX_UPLOAD_BYTES", str(default))))
    except ValueError:
        return default


def get_upload_ttl_seconds(default: int = DEFAULT_UPLOAD_TTL_SECONDS) -> int:
    try:
        return max(1, int(os.getenv("CLOUD_UPLOAD_TTL_SECONDS", str(default))))
    except ValueError:
        return default


def sanitize_download_filename(filename: str, fallback: str = "download.bin") -> str:
    raw_name = Path(str(filename or "")).name.strip()
    safe_name = SAFE_FILENAME_RE.sub("_", raw_name).strip("._")
    return safe_name or fallback


def normalize_extensions(extensions: Iterable[str] | None) -> set[str]:
    if not extensions:
        return set()
    normalized = set()
    for extension in extensions:
        value = str(extension or "").strip().lower()
        if value == "":
            normalized.add("")
            continue
        normalized.add(value if value.startswith(".") else f".{value}")
    return normalized


def validate_extension(filename: str, allowed_extensions: Iterable[str] | None) -> None:
    allowed = normalize_extensions(allowed_extensions)
    if not allowed:
        return
    suffix = Path(filename or "").suffix.lower()
    if suffix not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported file extension '{suffix}'. Allowed: {allowed_text}")


def allowed_extensions_for_kind(kind: str) -> set[str]:
    return set(ALLOWED_UPLOAD_EXTENSIONS.get(str(kind or "").strip().lower(), set()))


def resolve_within_root(path: Path, root: Path | None = None) -> Path:
    base_root = (root or get_upload_root()).resolve()
    resolved = path.resolve()
    if resolved != base_root and base_root not in resolved.parents:
        raise ValueError("Resolved path is outside the configured cloud file root.")
    return resolved


async def save_upload_file(
    upload: UploadFile,
    *,
    allowed_extensions: Iterable[str] | None = None,
    max_bytes: int | None = None,
    root: Path | None = None,
) -> SavedUpload:
    if not upload.filename:
        raise ValueError("Uploaded file must have a filename.")

    validate_extension(upload.filename, allowed_extensions)
    upload_root = (root or get_upload_root()).resolve()
    upload_root.mkdir(parents=True, exist_ok=True)

    safe_filename = sanitize_download_filename(upload.filename, "upload.bin")
    target_name = f"{int(time.time())}_{uuid.uuid4().hex}_{safe_filename}"
    target_path = resolve_within_root(upload_root / target_name, upload_root)
    byte_limit = max_bytes if max_bytes is not None else get_max_upload_bytes()

    total = 0
    try:
        with target_path.open("wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > byte_limit:
                    raise ValueError(f"Uploaded file exceeds {byte_limit} bytes.")
                output.write(chunk)
    except Exception:
        try:
            target_path.unlink(missing_ok=True)
        finally:
            raise
    finally:
        await upload.close()

    return SavedUpload(
        path=target_path,
        original_filename=upload.filename,
        safe_filename=safe_filename,
        size=total,
        content_type=upload.content_type or "application/octet-stream",
    )


def register_upload(
    saved: SavedUpload,
    *,
    kind: str = "generic",
    user_id: int | None = None,
    session_id: int | None = None,
) -> UploadRecord:
    upload_id = uuid.uuid4().hex
    record = UploadRecord(
        upload_id=upload_id,
        path=saved.path,
        filename=saved.safe_filename,
        size=saved.size,
        content_type=saved.content_type,
        kind=str(kind or "generic"),
        created_at=time.time(),
        user_id=user_id,
        session_id=session_id,
    )
    UPLOAD_REGISTRY[upload_id] = record
    if user_id is not None:
        try:
            with auth_db() as db:
                db.execute(
                    """
                    INSERT OR REPLACE INTO uploads
                    (upload_id, user_id, session_id, kind, filename, path, size,
                     content_type, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        upload_id,
                        int(user_id),
                        session_id,
                        record.kind,
                        record.filename,
                        str(record.path),
                        record.size,
                        record.content_type,
                        iso(),
                        iso(utcnow() + timedelta(seconds=get_upload_ttl_seconds())),
                    ),
                )
        except Exception:
            pass
    return record


def get_upload_record(upload_id: str, user_id: int | None = None) -> UploadRecord:
    record = UPLOAD_REGISTRY.get(str(upload_id or ""))
    if record is None:
        raise FileNotFoundError("Upload not found.")
    if user_id is not None and record.user_id is not None and int(record.user_id) != int(user_id):
        raise FileNotFoundError("Upload not found.")
    resolved = resolve_within_root(record.path, get_upload_root())
    if not resolved.exists() or not resolved.is_file():
        UPLOAD_REGISTRY.pop(record.upload_id, None)
        raise FileNotFoundError("Upload not found.")
    return record


def register_download_path(
    path: Path,
    *,
    filename: str | None = None,
    media_type: str = "application/octet-stream",
    root: Path | None = None,
    user_id: int | None = None,
    session_id: int | None = None,
) -> DownloadRecord:
    resolved = resolve_within_root(Path(path), root or get_download_root())
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError("Download file not found.")
    download_id = uuid.uuid4().hex
    record = DownloadRecord(
        download_id=download_id,
        path=resolved,
        filename=sanitize_download_filename(filename or resolved.name),
        media_type=media_type,
        created_at=time.time(),
        user_id=user_id,
        session_id=session_id,
    )
    DOWNLOAD_REGISTRY[download_id] = record
    return record


def get_download_record(download_id: str, user_id: int | None = None) -> DownloadRecord:
    record = DOWNLOAD_REGISTRY.get(str(download_id or ""))
    if record is None:
        raise FileNotFoundError("Download not found.")
    if user_id is not None and record.user_id is not None and int(record.user_id) != int(user_id):
        raise FileNotFoundError("Download not found.")
    resolved = resolve_within_root(record.path, get_download_root())
    if not resolved.exists() or not resolved.is_file():
        DOWNLOAD_REGISTRY.pop(record.download_id, None)
        raise FileNotFoundError("Download not found.")
    return record


def build_file_download_response(
    path: Path,
    *,
    filename: str | None = None,
    media_type: str = "application/octet-stream",
    root: Path | None = None,
) -> FileResponse:
    resolved = resolve_within_root(Path(path), root)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return FileResponse(
        resolved,
        media_type=media_type,
        filename=sanitize_download_filename(filename or resolved.name),
    )


def build_bytes_download_response(
    content: bytes,
    *,
    filename: str,
    media_type: str = "application/octet-stream",
) -> StreamingResponse:
    headers = {
        "Content-Disposition": f'attachment; filename="{sanitize_download_filename(filename)}"'
    }
    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers=headers,
    )


def cleanup_expired_uploads(
    *,
    root: Path | None = None,
    max_age_seconds: int | None = None,
    now: float | None = None,
) -> int:
    upload_root = (root or get_upload_root()).resolve()
    if not upload_root.exists():
        return 0
    cutoff = (time.time() if now is None else now) - (
        max_age_seconds if max_age_seconds is not None else get_upload_ttl_seconds()
    )
    removed = 0
    for item in upload_root.iterdir():
        try:
            if item.is_file() and item.stat().st_mtime < cutoff:
                item.unlink()
                removed += 1
        except OSError:
            continue
    cutoff = (time.time() if now is None else now) - (
        max_age_seconds if max_age_seconds is not None else get_upload_ttl_seconds()
    )
    for upload_id, record in list(UPLOAD_REGISTRY.items()):
        if record.created_at < cutoff:
            UPLOAD_REGISTRY.pop(upload_id, None)
    for download_id, record in list(DOWNLOAD_REGISTRY.items()):
        if record.created_at < cutoff:
            DOWNLOAD_REGISTRY.pop(download_id, None)
    return removed
