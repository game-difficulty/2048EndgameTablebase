from __future__ import annotations

from urllib.parse import urlsplit
from fastapi import APIRouter, Depends, HTTPException, Request, Response, Query
from fastapi.responses import StreamingResponse
import anyio
from pydantic import BaseModel, Field
from typing import Dict
from starlette.concurrency import run_in_threadpool

from backend.auth.dependencies import require_identity as require_user, current_identity_from_request as current_user_from_request, client_ip
from . import engine, service, traffic, codec
from backend.quota.service import InsufficientTokens


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


class PlayerSettings(BaseModel):
    display_thresholds: Dict[str, int]
    timer_splits: Dict[str, list[str]] | None = None


class HumanLiveControl(BaseModel):
    browser: str = Field(min_length=16, max_length=128)
    writer: str = Field(min_length=16, max_length=128)
    epoch: int = Field(ge=1)


class VerseClaim(BaseModel):
    username: str = Field(min_length=1, max_length=64)


class AnalysisItem(BaseModel):
    pattern: str = Field(min_length=1, max_length=80)
    target: str = Field(min_length=1, max_length=12)


class AnalysisSelection(BaseModel):
    items: list[AnalysisItem] = Field(min_length=1, max_length=6)


class AnalysisCreate(AnalysisSelection):
    expected_cost_units: int = Field(ge=0)
    expected_catalog_version: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=16, max_length=128)


@router.get('/runs/{run_id}/analysis/options')
def analysis_options(run_id: str, request: Request, response: Response):
    from . import analysis as replay_analysis
    response.headers['Cache-Control'] = 'private, no-store'
    return call(replay_analysis.options, run_id, require_user(request)['id'])


@router.get('/runs/{run_id}/analysis/summaries')
def analysis_summaries(run_id: str, request: Request, response: Response):
    from . import analysis as replay_analysis
    viewer = current_user_from_request(request)
    response.headers['Cache-Control'] = 'private, no-store' if viewer else 'public, max-age=30'
    return call(replay_analysis.summaries, run_id, viewer['id'] if viewer else None)


@router.get('/analysis/summaries/{summary_id}')
def analysis_summary(summary_id: int, request: Request, response: Response):
    from . import analysis as replay_analysis
    viewer = current_user_from_request(request)
    response.headers['Cache-Control'] = 'private, no-store' if viewer else 'public, max-age=30'
    return call(replay_analysis.summary, summary_id, viewer['id'] if viewer else None)


@router.post('/runs/{run_id}/analysis/quote')
def analysis_quote(run_id: str, payload: AnalysisSelection, request: Request):
    from . import analysis as replay_analysis
    return call(replay_analysis.quote, run_id, require_user(request)['id'],
                [item.model_dump() for item in payload.items])


@router.post('/runs/{run_id}/analysis/jobs', status_code=202)
def analysis_create(run_id: str, payload: AnalysisCreate, request: Request):
    from . import analysis as replay_analysis
    from backend.cloud_analysis_jobs import AnalysisQueueFull, analysis_job_payload
    user = require_user(request)
    try:
        job = call(replay_analysis.create, run_id, user['id'], user.get('session_id'),
                   [item.model_dump() for item in payload.items], payload.expected_cost_units,
                   payload.expected_catalog_version, payload.request_id)
    except AnalysisQueueFull as exc:
        raise HTTPException(429, {'code': str(exc)}, headers={'Retry-After': '10'}) from exc
    except InsufficientTokens as exc:
        raise HTTPException(402, exc.payload) from exc
    except ValueError as exc:
        raise HTTPException(400, {'code': str(exc)}) from exc
    return analysis_job_payload(job)


@router.get("/config")
def get_config(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return service.config()


@router.get("/live/current")
def live_current(request: Request, response: Response):
    from . import live
    response.headers["Cache-Control"] = "private, no-store"
    return live.current(require_user(request)["id"])


@router.post("/runs/{run_id}/live")
def live_start(run_id: str, payload: HumanLiveControl, request: Request):
    from . import live
    return live.start(require_user(request), run_id, **payload.model_dump())


@router.post("/live/{room_id}/lease")
def live_lease(room_id: str, payload: HumanLiveControl, request: Request):
    from . import live
    return live.renew(require_user(request), room_id, **payload.model_dump())


@router.delete("/live/{room_id}")
def live_stop(room_id: str, request: Request):
    from . import live
    return live.stop(require_user(request)["id"], room_id)


@router.get("/internal/live/runs/{run_id}/milestones/{milestone}")
def live_verified_milestone(run_id: str, milestone: int, request: Request):
    from . import live
    return live.verified_milestone(run_id, milestone,
        request.headers.get("x-human-live-internal", ""))


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
    if action not in {"monitor", "append", "reentry", "seal", "live"}:
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
            # anyio.fail_after supports the Python 3.10 runtime used by the
            # play site; asyncio.timeout was only added in Python 3.11.
            with anyio.fail_after(20):
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
def leaderboard(request: Request, response: Response, variant: str = "4x4", period: str = "all", limit: int = Query(10, ge=1, le=100)):
    response.headers['Cache-Control'] = 'no-store'
    viewer = current_user_from_request(request)
    return call(service.leaderboard, variant, period, limit, viewer['id'] if viewer else None)


@router.get("/leaderboards/catalog")
def full_leaderboard_catalog(response: Response):
    response.headers['Cache-Control'] = 'public, max-age=30'
    return call(service.full_leaderboard_catalog)


@router.get("/leaderboards/full")
def full_leaderboard(response: Response, kind: str = Query("score", alias="type"),
                     variant: str = "4x4", period: str = "all",
                     page: int = Query(1, ge=1, le=100000),
                     page_size: int = Query(50, ge=50, le=50),
                     pattern: str = Query("", max_length=80),
                     target: str = Query("", max_length=12)):
    response.headers['Cache-Control'] = 'public, max-age=30'
    return call(service.full_leaderboard, kind, variant, period, page, pattern, target)


@router.get('/me/bests')
def personal_bests(request: Request, response: Response):
    response.headers['Cache-Control'] = 'private, no-store'
    return call(service.personal_bests, require_user(request)['id'])


@router.get('/me/settings')
def player_settings(request: Request, response: Response):
    response.headers['Cache-Control'] = 'private, no-store'
    return call(service.player_settings, require_user(request)['id'])


@router.get('/me/verse-claim')
def verse_claim_status(request: Request, response: Response):
    from . import verse_history
    response.headers['Cache-Control'] = 'private, no-store'
    return {"claim": verse_history.own_claim(require_user(request)['id'])}


@router.post('/me/verse-claim')
def verse_claim_request(payload: VerseClaim, request: Request):
    from . import verse_history
    return {"claim": call(verse_history.request_claim, require_user(request)['id'], payload.username)}


@router.get('/me/archive-applications')
def archive_applications(request: Request, response: Response):
    from . import manual_archive
    response.headers['Cache-Control'] = 'private, no-store'
    return {"applications": call(manual_archive.mine, require_user(request)['id'])}


@router.post('/me/archive-applications', status_code=201)
async def archive_application_submit(request: Request, variant: str,
                                     ended_at: float, score: int,
                                     filename: str = Query('', max_length=180)):
    from . import manual_archive
    user = await run_in_threadpool(require_user, request)
    if request.headers.get('content-type', '').split(';')[0] != 'application/octet-stream':
        raise HTTPException(415, 'binary_body_required')
    token = await run_in_threadpool(traffic.acquire, str(user['id']), 'large')
    try:
        body = bytearray()
        try:
            with anyio.fail_after(20):
                async for part in request.stream():
                    if len(body) + len(part) > manual_archive.MAX_INPUT:
                        raise HTTPException(413, 'replay_too_large')
                    body.extend(part)
        except TimeoutError as exc:
            raise HTTPException(408, 'upload_timeout') from exc
        application = await run_in_threadpool(call, manual_archive.submit, user['id'],
            variant, ended_at, score, filename, bytes(body))
        return {"application": application}
    finally:
        await release_slot(token)


@router.put('/me/settings')
def save_player_settings(payload: PlayerSettings, request: Request):
    return call(service.save_player_settings, require_user(request)['id'], payload.display_thresholds, payload.timer_splits)


@router.get('/users/{username}/history')
def player_history(username: str, request: Request, response: Response,
                   variant: str = 'all', sort: str = 'newest', page: int = Query(1, ge=1),
                   page_size: int = Query(20, ge=10, le=50)):
    if page_size not in {10, 20, 50}:
        raise HTTPException(400, 'invalid_page_size')
    response.headers['Cache-Control'] = 'private, no-store'
    viewer = current_user_from_request(request)
    user_id = call(service.player_id_for_name, username)
    return call(service.history, user_id, viewer['id'] if viewer else None,
                variant=variant, sort=sort, limit=page_size, page=page)


@router.delete('/runs/{run_id}/history')
def delete_player_history_run(run_id: str, request: Request, response: Response):
    response.headers['Cache-Control'] = 'private, no-store'
    user = require_user(request)
    return call(service.delete_history_run, run_id, user['id'])


@router.get('/users/{username}/best10')
def player_best_ten(username: str, request: Request, response: Response, variant: str = '4x4'):
    response.headers['Cache-Control'] = 'private, no-store'
    viewer = current_user_from_request(request)
    user_id = call(service.player_id_for_name, username)
    return call(service.best_ten, user_id, viewer['id'] if viewer else None, variant)


@router.get('/users/{username}/statistics')
def player_statistics(username: str, response: Response, variant: str = '4x4'):
    response.headers['Cache-Control'] = 'private, max-age=30'
    user_id = call(service.player_id_for_name, username)
    return call(service.player_statistics, user_id, variant)


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
        data, source = await run_in_threadpool(call, service.replay_record, run_id, viewer["id"] if viewer else None)
    except BaseException:
        await release_slot(token)
        raise
    return ReplayResponse(data, token, media_type="application/octet-stream", headers={
        "Content-Encoding": "gzip", "Cache-Control": "private, no-store",
        "Content-Disposition": f'attachment; filename="{run_id}.{"vrs" if source in {"verse", "manual"} else "hpr"}"'})


@router.post("/verse-replays/{run_id}")
async def attach_verse_replay(run_id: str, request: Request):
    from . import verse_replay
    user = await run_in_threadpool(require_user, request)
    token = await run_in_threadpool(traffic.acquire, str(user["id"]), "large")
    try:
        body = bytearray()
        with anyio.fail_after(20):
            async for part in request.stream():
                if len(body) + len(part) > verse_replay.MAX_INPUT:
                    raise HTTPException(413, "replay_too_large")
                body.extend(part)
        return await run_in_threadpool(call, verse_replay.attach, run_id, user["id"], bytes(body))
    except TimeoutError as exc:
        raise HTTPException(408, "upload_timeout") from exc
    finally:
        await release_slot(token)
