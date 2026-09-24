from __future__ import annotations

from urllib.parse import urlsplit
from fastapi import APIRouter, Depends, HTTPException, Request, Response, Query
from fastapi.responses import StreamingResponse
import asyncio
import anyio
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.auth.dependencies import require_identity as require_user, current_identity_from_request as current_user_from_request, client_ip
from . import engine, service, traffic, codec


def same_origin(request: Request):
    if request.method in {"POST", "PUT", "DELETE"}:
        origin = request.headers.get("origin")
        if request.headers.get("sec-fetch-site") == "cross-site" or (origin and urlsplit(origin).netloc != request.headers.get("host")):
            raise HTTPException(403, "origin_rejected")


router = APIRouter(prefix="/api/human", tags=["human-play"], dependencies=[Depends(same_origin)])


async def release_slot(token):
    # Client disconnects must not retain a capacity slot until crash expiry.
    with anyio.CancelScope(shield=True):
        await run_in_threadpool(traffic.release, token)


class ReplayResponse(StreamingResponse):
    def __init__(self, data, token, **kwargs):
        self.token = token
        async def body():
            for offset in range(0, len(data), 16384):
                yield data[offset:offset + 16384]
        super().__init__(body(), **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await release_slot(self.token)


def call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except service.RunError as exc:
        raise HTTPException(exc.status, {"code": exc.code, **exc.details}) from exc


def wire_receipt(request, result):
    return codec.compact_receipt(result) if request.headers.get('x-human-protocol') == '2' else result


class CreateRun(BaseModel):
    browser: str = Field(min_length=32, max_length=128, pattern=r"^[a-zA-Z0-9-]+$")
    variant: str = Field(max_length=8)
    request_id: str = Field(min_length=16, max_length=128)
    writer: str = Field(min_length=16, max_length=128)
    replace_id: str | None = Field(default=None, max_length=64)


class Writer(BaseModel):
    browser: str = Field(min_length=32, max_length=128)
    writer: str = Field(min_length=16, max_length=128)
    epoch: int = Field(ge=1)
    permit: str = Field(default="", max_length=128)


@router.get("/config")
def get_config(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return service.config()


@router.post("/runs")
def create(payload: CreateRun, request: Request):
    user = require_user(request)
    return call(service.create, user["id"], **payload.model_dump())


@router.get("/runs/{run_id}/status")
def status(run_id: str, request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    user = require_user(request)
    return wire_receipt(request, call(service.status, user["id"], request.headers.get("x-human-browser", ""), run_id))


@router.post("/runs/{run_id}/writer")
def writer(run_id: str, payload: Writer, request: Request):
    user = require_user(request)
    return wire_receipt(request, call(service.claim_low, user["id"], payload.browser, run_id, payload.writer, payload.epoch))


@router.post("/runs/{run_id}/online-check")
def online_check(run_id: str, payload: Writer, request: Request):
    user = require_user(request)
    return wire_receipt(request, call(service.online_check, user["id"], payload.browser, run_id, payload.writer, payload.epoch, payload.permit))


@router.post("/runs/{run_id}/{action}")
async def upload(run_id: str, action: str, request: Request):
    user = await run_in_threadpool(require_user, request)
    if action not in {"monitor", "append", "reentry", "seal"}:
        raise HTTPException(404)
    if request.headers.get("content-type", "").split(";")[0] != "application/octet-stream":
        raise HTTPException(415, "binary_body_required")
    encoding = request.headers.get('content-encoding', 'identity').lower()
    layout = request.headers.get('x-human-layout', 'interleaved')
    if encoding not in {'identity', 'gzip'} or layout not in {'interleaved', 'planes5'}:
        raise HTTPException(415, 'unsupported_record_encoding')
    try:
        epoch = int(request.headers["x-human-epoch"])
        start = int(request.headers["x-human-start"])
        local_seq = int(request.headers["x-human-count"])
        if min(epoch, start, local_seq) < 0:
            raise ValueError()
    except (KeyError, ValueError):
        raise HTTPException(400, "invalid_sequence_headers")
    if not 16 <= len(request.headers.get("x-human-writer", "")) <= 128:
        raise HTTPException(400, "invalid_writer")
    try:
        size = int(request.headers.get('content-length', '-1'))
    except ValueError:
        raise HTTPException(400, 'invalid_content_length')
    if size > engine.MAX_BYTES:
        raise HTTPException(413, 'record_too_large')
    lane = 'small' if action != 'seal' and encoding == 'identity' and 0 <= size <= 8192 else 'large'
    token = await run_in_threadpool(traffic.acquire, str(user['id']), lane)
    try:
        data = bytearray()
        try:
            async with asyncio.timeout(20):
                async for part in request.stream():
                    if len(data) + len(part) > (8192 if lane == 'small' else engine.MAX_BYTES):
                        raise HTTPException(413, 'record_too_large')
                    data.extend(part)
        except TimeoutError:
            raise HTTPException(408, 'upload_timeout')
        try:
            canonical = await run_in_threadpool(codec.decode_upload, bytes(data), encoding, layout)
        except ValueError as exc:
            raise HTTPException(413 if str(exc) == 'record_too_large' else 400, str(exc)) from exc
        result = await run_in_threadpool(call, service.submit, user["id"], request.headers.get("x-human-browser", ""), run_id,
            action=action, writer=request.headers.get("x-human-writer", ""), epoch=epoch,
            start=start, prefix_hash=request.headers.get("x-human-prefix", ""), local_seq=local_seq,
            data=canonical, reason=request.headers.get("x-human-reason", ""), permit=request.headers.get('x-human-permit', ''))
        return wire_receipt(request, result)
    finally:
        await release_slot(token)


@router.get("/leaderboards")
def leaderboard(response: Response, variant: str = "4x4", period: str = "all", limit: int = Query(10, ge=1, le=100)):
    response.headers['Cache-Control'] = 'no-store'
    return call(service.leaderboard, variant, period, limit)


@router.get('/me/bests')
def personal_bests(request: Request, response: Response):
    response.headers['Cache-Control'] = 'private, no-store'
    return call(service.personal_bests, require_user(request)['id'])


@router.get("/players/{user_id}")
def history(user_id: int, request: Request, response: Response, before: float | None = None):
    response.headers["Cache-Control"] = "no-store"
    viewer = current_user_from_request(request)
    return call(service.history, user_id, viewer["id"] if viewer else None, before)


@router.get("/replays/{run_id}")
async def replay(run_id: str, request: Request):
    viewer = await run_in_threadpool(current_user_from_request, request)
    identity = str(viewer['id']) if viewer else 'ip:' + client_ip(request)
    token = await run_in_threadpool(traffic.acquire, identity, 'download')
    try:
        data = await run_in_threadpool(call, service.replay, run_id, viewer["id"] if viewer else None)
    except BaseException:
        await release_slot(token)
        raise
    return ReplayResponse(data, token, media_type="application/octet-stream", headers={
        "Content-Encoding": "gzip", "Cache-Control": "private, no-store",
        "Content-Disposition": f'attachment; filename="{run_id}.hpr"'})
