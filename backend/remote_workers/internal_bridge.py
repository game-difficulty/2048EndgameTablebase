"""Loopback bridge from the Play analysis process to the main tablebase worker.

The worker's websocket belongs to the main API process.  Play may use its
availability and lookup operations, but never advertises a table it cannot
actually read.  The bridge is enabled only by an explicit loopback URL.
"""
from __future__ import annotations

import hmac
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request as UrlRequest, urlopen

from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import BaseModel

from .config import configured_remote_tables, worker_secret
from .errors import RemoteTablebaseError, RemoteTablebaseOffline, RemoteTablebaseProtocolError, RemoteTablebaseTimeout
from .registry import remote_worker_registry


router = APIRouter(prefix="/internal/tablebases", include_in_schema=False)


def proxy_enabled() -> bool:
    return bool(os.getenv("REMOTE_TABLEBASE_PROXY_BASE_URL", "").strip())


def _proxy_base() -> str:
    raw = os.getenv("REMOTE_TABLEBASE_PROXY_BASE_URL", "").strip().rstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"} or not parsed.port or parsed.path:
        raise RemoteTablebaseProtocolError("Tablebase proxy must be a loopback HTTP URL.")
    return raw


def _authenticate(secret: str | None) -> None:
    expected = worker_secret()
    if not expected or not secret or not hmac.compare_digest(secret, expected):
        raise HTTPException(status_code=403, detail="Forbidden")


class LookupRequest(BaseModel):
    full_pattern: str
    pattern: str
    target: str
    boards: list[str]
    use_variant: bool
    board_is_lookup: bool = True


@router.get("/online")
def online(response: Response, x_internal_tablebase_secret: str | None = Header(default=None)):
    _authenticate(x_internal_tablebase_secret)
    response.headers["Cache-Control"] = "no-store"
    configured = configured_remote_tables()
    return {"tables": sorted(remote_worker_registry.online_tables() & configured.keys())}


@router.post("/lookup-batch")
async def lookup_batch(request: LookupRequest, x_internal_tablebase_secret: str | None = Header(default=None)):
    _authenticate(x_internal_tablebase_secret)
    configured = configured_remote_tables().get(request.full_pattern)
    if (configured is None or configured["pattern"] != request.pattern
            or configured["target"] != request.target):
        raise HTTPException(status_code=400, detail="Invalid tablebase selection")
    try:
        return await remote_worker_registry.lookup_batch(
            full_pattern=request.full_pattern, pattern=request.pattern,
            target=request.target, boards=request.boards,
            use_variant=request.use_variant, board_is_lookup=request.board_is_lookup,
        )
    except RemoteTablebaseTimeout as exc:
        raise HTTPException(status_code=504, detail=exc.payload) from exc
    except RemoteTablebaseError as exc:
        raise HTTPException(status_code=503, detail=exc.payload) from exc


def _request(path: str, payload: dict | None = None, *, timeout: float = 5.0) -> dict:
    secret = worker_secret()
    if not secret:
        raise RemoteTablebaseOffline()
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = UrlRequest(
        _proxy_base() + path, data=body,
        headers={"X-Internal-Tablebase-Secret": secret, "Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise RemoteTablebaseProtocolError("Tablebase proxy response is too large.")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise RemoteTablebaseProtocolError("Invalid tablebase proxy response.")
        return result
    except HTTPError as exc:
        if exc.code == 504:
            raise RemoteTablebaseTimeout() from exc
        raise RemoteTablebaseOffline() from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RemoteTablebaseOffline() from exc
    except (ValueError, UnicodeError) as exc:
        raise RemoteTablebaseProtocolError("Invalid tablebase proxy response.") from exc


def proxy_online_tables() -> set[str]:
    result = _request("/internal/tablebases/online", timeout=3.0)
    tables = result.get("tables")
    if not isinstance(tables, list) or not all(isinstance(item, str) for item in tables):
        raise RemoteTablebaseProtocolError("Invalid tablebase proxy catalog.")
    return set(tables)


def proxy_lookup_batch(*, full_pattern: str, pattern: str, target: str,
                       boards: list[str], use_variant: bool, board_is_lookup: bool) -> dict:
    return _request("/internal/tablebases/lookup-batch", {
        "full_pattern": full_pattern, "pattern": pattern, "target": target,
        "boards": boards, "use_variant": use_variant,
        "board_is_lookup": board_is_lookup,
    }, timeout=max(65.0, min(605.0, len(boards) * 2.0 + 5.0)))
