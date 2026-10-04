"""Authenticated producer stream and read-only internal live subscription."""
import asyncio
from contextlib import suppress
import hmac
import json
import logging
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from .auth import principal_from_socket_auth, principal_from_websocket
from .errors import CompetitionError
from .hub import SocketWriter
from .schemas import ClientGameStateRequest
from .stream_protocol import PROTOCOL
from backend.projection_delta import ProjectionEncoder
from backend.stream_snapshots import compact_public_view

log = logging.getLogger(__name__)

LIVE_RECHECK_SECONDS = 15


async def relay_live_projections(socket, service, public_key, queue, *, interval=LIVE_RECHECK_SECONDS, deltas=False):
    """Notifications are hints; periodically verify the durable source watermark."""
    loop = asyncio.get_running_loop()
    cursors = {}
    game = None
    revision = None
    encoder = ProjectionEncoder() if deltas else None

    async def publish():
        nonlocal cursors, game, revision
        projection = await asyncio.wait_for(asyncio.to_thread(service.live_projection, public_key,
            after_yellow=cursors.get('yellow', 0), after_white=cursors.get('white', 0)), 5)
        current = (projection.get('generation'), projection.get('current_game'))
        if game is not None and current != game:
            projection = await asyncio.wait_for(asyncio.to_thread(service.live_projection, public_key), 5)
        if game is None or current != game:
            # Recovery establishes a new baseline; ordinary/final batches retain all new frames.
            projection = {**projection, 'project_public_views': {
                s: compact_public_view(v) for s, v in (projection.get('project_public_views') or {}).items()}}
        message = encoder.encode(projection) if encoder else {'type': 'projection', 'projection': projection}
        await asyncio.wait_for(socket.send_json(message), 5)
        game = (projection.get('generation'), projection.get('current_game'))
        revision = (projection['generation'], projection['content_sequence'])
        cursors = {s: v['sequence'] for s, v in projection.get('project_public_views', {}).items() if v}

    await publish()
    check_at = loop.time() + interval
    while True:
        notified = False
        try:
            await asyncio.wait_for(queue.get(), max(0, check_at - loop.time()))
            notified = True
        except asyncio.TimeoutError:
            pass
        if loop.time() >= check_at:
            source_revision = await asyncio.wait_for(asyncio.to_thread(service.live_revision, public_key), 5)
            check_at = loop.time() + interval
            if notified or source_revision != revision:
                await publish()
            else:
                await asyncio.wait_for(socket.send_json({'type': 'heartbeat'}), 5)
        elif notified:
            await publish()

def install_stream_routes(app, service, hub, settings):
    publishers = {}
    @app.websocket('/ws/projects/{room_code}')
    async def producer(socket: WebSocket, room_code: str):
        await socket.accept()
        writer = None
        instance = None
        side = None
        accepted = 0
        code = room_code.strip().upper()
        try:
            hello = await asyncio.wait_for(socket.receive_json(), 8)
            data = hello.get('data') or {}
            if hello.get('type') != 'authenticate' or data.get('protocol') != PROTOCOL:
                await socket.close(code=4406, reason='refresh_required'); return
            principal = principal_from_websocket(socket, settings) or principal_from_socket_auth(data, settings, socket)
            if not principal:
                await socket.close(code=4401); return
            room = await asyncio.to_thread(service.snapshot, code, principal)
            runtime = ((room.get('match') or {}).get('my_session') or {}).get('runtime') or {}
            instance = data.get('instance_id')
            if not instance or instance != runtime.get('instance_id'):
                await socket.close(code=4403, reason='active_player_required'); return
            side = runtime.get('side')
            accepted = runtime['sequence']
            async def detached():
                pass
            writer = SocketWriter(socket, detached)
            old = publishers.get(instance)
            publishers[instance] = writer
            if old:
                await old.close(4409, 'publisher_replaced')
            log.info('producer connected room=%s side=%s accepted=%s replaced=%s', code, side, accepted, bool(old))
            await writer.put({'type': 'stream.ready', 'protocol': PROTOCOL,
                              'instance_id': instance, 'accepted_sequence': runtime['sequence']})
            while not writer.closed:
                message = await asyncio.wait_for(socket.receive_json(), 45)
                if message.get('type') == 'ping':
                    await writer.put({'type': 'pong'}); continue
                if message.get('type') != 'project.batch':
                    await writer.close(4400, 'invalid_message'); break
                raw = message.get('data') or {}
                if raw.get('instance_id') != instance or not raw.get('frames'):
                    await writer.close(4400, 'invalid_stream_batch'); break
                if len(json.dumps(raw).encode()) > 1024 * 1024:
                    await writer.close(1009, 'batch_too_large'); break
                payload = ClientGameStateRequest.model_validate(raw)
                result = await asyncio.to_thread(service.sync_client_game, code, principal, **payload.model_dump())
                accepted = result['accepted_sequence']
                await writer.put({'type': 'stream.ack', 'instance_id': instance,
                                  'accepted_sequence': result['accepted_sequence'],
                                  'stopped': bool(result.get('stopped')),
                                  'resync_room': bool(result.get('competition'))})
                if result.get('update'):
                    await hub.broadcast_message(code, {'type': 'project.snapshot', 'data': result['update']})
                elif result.get('competition'):
                    await hub.broadcast(code, lambda viewer: service.snapshot(code, viewer))
        except CompetitionError as error:
            log.warning('producer error room=%s code=%s', code, error.code)
            close_code = 1012 if error.code in {'CHECKPOINT_BASE_MISMATCH', 'STALE_PHASE', 'MATCH_SUSPENDED'} else 4400
            if writer:
                await writer.finish({'type': 'stream.error', 'error': {**error.detail, 'status': error.status_code}}, close_code, error.code)
            else:
                await socket.close(code=close_code, reason=error.code)
        except WebSocketDisconnect as error:
            log.info('producer peer closed room=%s side=%s close_code=%s reason=%r',
                     code, side, error.code, error.reason)
        except asyncio.TimeoutError:
            log.warning('producer receive timeout room=%s side=%s accepted=%s', code, side, accepted)
        except (ValidationError, ValueError, TypeError):
            if writer:
                await writer.close(4400, 'invalid_stream_payload')
            else:
                await socket.close(code=4400, reason='invalid_stream_payload')
        except Exception:
            log.exception('producer stream failed room=%s', room_code)
        finally:
            if instance and publishers.get(instance) is writer:
                publishers.pop(instance, None)
            if writer:
                log.info('producer disconnected room=%s side=%s accepted=%s', code, side, accepted)
                await writer.close()
            else:
                with suppress(Exception):
                    await asyncio.wait_for(socket.close(), 2)

    @app.websocket('/ws/internal/live/{public_key}')
    async def live(socket: WebSocket, public_key: str):
        expected = settings.live_internal_token
        supplied = socket.headers.get('x-competition-live-token', '')
        if not expected or not hmac.compare_digest(expected, supplied):
            await socket.close(code=4403); return
        with service.database.transaction() as db:
            row = db.execute('SELECT room_code FROM competitions WHERE public_key=?', (public_key,)).fetchone()
        if not row:
            await socket.close(code=4404); return
        code = row['room_code']
        queue = hub.subscribe(code)
        await socket.accept()
        try:
            await relay_live_projections(socket, service, public_key, queue,
                                         deltas=socket.query_params.get('transport') == 'delta-v1')
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
            pass
        except CompetitionError:
            await socket.close(code=4404, reason='live_room_ended')
        finally:
            hub.unsubscribe(code, queue)
            with suppress(Exception):
                await asyncio.wait_for(socket.close(), 2)
