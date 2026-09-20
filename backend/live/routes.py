import asyncio
import contextlib
import hmac
import json
import logging
import os
import time
import unicodedata
from collections import deque
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from backend.auth.dependencies import require_actor, require_user, current_user_from_request, current_user_from_websocket, client_ip, websocket_client_ip
from backend.quota.errors import InsufficientTokens
from . import gifts, audience, lucky_bags, red_envelopes
from backend.auth.dependencies import current_guest_from_websocket
from backend.auth.principal import ActorRef
from .protocol import LiveRun, STEP
from .store import LiveStore, week_bounds

router = APIRouter(prefix='/api/live', tags=['live'])


class LiveHub:
    def __init__(self):
        self.store = None
        self.run = None
        self.producer = None
        self.last_seen = 0
        self.viewers = {}
        self.chat = deque(maxlen=100)
        self.rates = {}
        self.likes = 0
        self.like_total = 0
        self.task = None
        self.gift_task = None
        self.gift_lock = asyncio.Lock()
        self.viewer_users = {}
        self.gift_history = deque(maxlen=100)
        self.audience = audience.Presence()
        self.lucky_task = None
        self.lucky_lock = asyncio.Lock()
        self.lucky_bags = []
        self.lucky_seen = set()
        self.viewer_times = {}
        self.control = dict(enabled=True, revision=0)
        self.control_lock = asyncio.Lock()
        self.producer_ready = False
        self.control_supported = False
        self.control_ack = None
        self.summary_week = None
        self.red_lock = asyncio.Lock()
        self.red_state = dict(active=None, queued=0)
        self.red_task = None
        self.stats_task = None

    async def start(self):
        await asyncio.to_thread(gifts.init_schema)
        await asyncio.to_thread(lucky_bags.init_schema)
        await asyncio.to_thread(red_envelopes.init_schema)
        self.red_state = await asyncio.to_thread(red_envelopes.tick)
        self.lucky_bags = await asyncio.to_thread(lucky_bags.listing)
        self.gift_history.extend(await asyncio.to_thread(gifts.recent_events))
        for event in self.gift_history:
            self.append_gift_chat(event)
        entrances = {}
        for item in self.chat:
            if item.get('type') == 'entrance':
                entrances[item.get('name')] = item
        self.chat = deque(
            [item for item in self.chat if item.get('type') != 'entrance'] + list(entrances.values()),
            maxlen=100,
        )
        history = [*self.chat, *await asyncio.to_thread(red_envelopes.events)]
        self.chat = deque(sorted(history, key=lambda item: item['at'])[-100:], maxlen=100)
        self.store = LiveStore()
        self.control = await asyncio.to_thread(self.store.control)
        summary = await asyncio.to_thread(self.store.summary)
        self.like_total = summary['likes']
        self.summary_week = summary['week']['start']
        saved = await asyncio.to_thread(self.store.load)
        if saved:
            self.run = await asyncio.to_thread(LiveRun.restore, saved)
        self.task = asyncio.create_task(self.maintenance())
        self.gift_task = asyncio.create_task(self.gift_maintenance())
        self.lucky_task = asyncio.create_task(self.lucky_maintenance())
        self.red_task = asyncio.create_task(self.red_maintenance())
        self.stats_task = asyncio.create_task(self.backfill_stats())

    async def backfill_stats(self):
        changed = False
        while await asyncio.to_thread(self.store.backfill_stats, limit=1):
            changed = True
            await asyncio.sleep(0)
        if changed:
            summary = await asyncio.to_thread(self.store.summary)
            await self.broadcast(dict(type='summary', **summary))

    async def stop(self):
        if self.stats_task:
            self.stats_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.stats_task
        if self.red_task:
            self.red_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.red_task
        if self.lucky_task:
            self.lucky_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.lucky_task
        if self.gift_task:
            self.gift_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.gift_task
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        if self.run:
            await asyncio.to_thread(self.store.save, self.run)
        await asyncio.to_thread(self.store.add_likes, self.likes)

    def append_gift_chat(self, event):
        key = 'gift:' + event['combo_id']
        message = dict(event, **event['actor'])
        message['id'] = key
        for index, previous in enumerate(self.chat):
            if previous['id'] == key:
                if event['combo_count'] > previous['combo_count']:
                    message['at'] = previous['at']
                    self.chat[index] = message
                return
        self.chat.append(message)

    def append_entrance_chat(self, event):
        """Keep only the newest entrance notice for each displayed account."""
        actor_name = event.get('name')
        self.chat = deque(
            (item for item in self.chat
             if not (item.get('type') == 'entrance' and item.get('name') == actor_name)),
            maxlen=100,
        )
        self.chat.append(event)

    async def drain_gifts(self):
        async with self.gift_lock:
            pending = await asyncio.to_thread(gifts.pending_events)
            known = {event['id'] for event in self.gift_history}
            for item in pending:
                if item['id'] not in known:
                    self.gift_history.append(item['event'])
                self.append_gift_chat(item['event'])
                if item['fresh']:
                    self.broadcast(item['event'])
            if pending:
                await asyncio.to_thread(gifts.delivered, [item['id'] for item in pending])

    async def gift_maintenance(self):
        last_cleanup = last_error = 0
        while True:
            try:
                await self.drain_gifts()
                if time.monotonic() - last_cleanup > 3600:
                    await asyncio.to_thread(gifts.cleanup)
                    await asyncio.to_thread(lucky_bags.cleanup)
                    last_cleanup = time.monotonic()
            except Exception as error:
                if time.monotonic() - last_error > 60:
                    logging.getLogger(__name__).warning('Gift delivery delayed: %s', type(error).__name__)
                    last_error = time.monotonic()
            await asyncio.sleep(1)

    def snapshot(self):
        return dict(type='snapshot', online=self.control_status()['online'], paused=not self.control['enabled'],
                    run=self.run.snapshot() if self.run else None, viewers=len(self.audience.identities()),
                    lucky_bags=self.lucky_bags, red_envelopes=self.red_state, server_time=time.time())

    async def refresh_red(self):
        async with self.red_lock:
            state = await asyncio.to_thread(red_envelopes.tick)
            if state != self.red_state:
                self.red_state = state
                self.broadcast(dict(type='red_envelopes', **state, server_time=time.time()))
            pending = await asyncio.to_thread(red_envelopes.events, True)
            known = {item['id'] for item in self.chat}
            for event in pending:
                if event['id'] not in known:
                    self.chat.append(event)
                self.broadcast(event)
            if pending:
                await asyncio.to_thread(red_envelopes.delivered, [item['envelope_id'] for item in pending])

    async def red_maintenance(self):
        last_error = 0
        while True:
            try:
                await self.refresh_red()
            except Exception as error:
                if time.monotonic() - last_error > 60:
                    logging.getLogger(__name__).warning('Red envelope settlement delayed: %s', type(error).__name__)
                    last_error = time.monotonic()
            await asyncio.sleep(1)

    def control_status(self):
        connected = bool(self.producer) and self.producer_ready and time.monotonic() - self.last_seen < 20
        applied = connected and self.control_ack == self.control['revision']
        return dict(self.control, connected=connected, supported=self.control_supported,
                    applied=applied, online=connected and self.control['enabled'] and (applied or not self.control_supported),
                    run_id=self.run.id if self.run else None)

    async def set_enabled(self, enabled):
        async with self.control_lock:
            self.control = await asyncio.to_thread(self.store.control, enabled)
            if self.producer_ready and self.control_supported and self.producer:
                try:
                    await asyncio.wait_for(self.producer.send_json(dict(type='control', **self.control)), 5)
                except Exception:
                    # The persisted intent will be delivered on reconnect.
                    with contextlib.suppress(Exception):
                        await self.producer.close(code=1013)
            self.broadcast(self.snapshot())
            return self.control_status()

    def lucky_present(self, deadline):
        now = time.time()
        return {uid for ws, uid in self.viewer_users.items()
                if ws in self.viewer_times and self.viewer_times[ws][0] <= deadline
                and self.viewer_times[ws][1] >= now - 35}

    async def refresh_lucky(self):
        self.lucky_bags = await asyncio.to_thread(lucky_bags.listing)
        self.broadcast(dict(type='lucky_bags', bags=self.lucky_bags, server_time=time.time()))

    async def tick_lucky(self):
        async with self.lucky_lock:
            # Only two milestone inserts per run; no database work on ordinary steps.
            reached = {(self.run.id, value) for value in lucky_bags.RULES
                       if str(value) in self.run.nodes} if self.run else set()
            new = reached - self.lucky_seen
            for run_id, value in sorted(new):
                await asyncio.to_thread(lucky_bags.create, run_id, value)
            self.lucky_seen = reached
            if new:
                await self.refresh_lucky()
            changed = False
            now = time.time()
            for bag in self.lucky_bags:
                if bag['drawn_at'] is None and bag['draw_at'] <= now:
                    present = self.lucky_present(bag['draw_at'])
                    changed |= await asyncio.to_thread(lucky_bags.draw, bag['id'], present)
                elif bag['expires_at'] <= now:
                    changed = True
            if changed:
                await self.refresh_lucky()

    async def lucky_maintenance(self):
        last_error = 0
        while True:
            try:
                await self.tick_lucky()
            except Exception as error:
                if time.monotonic() - last_error > 60:
                    logging.getLogger(__name__).warning('Lucky bag draw delayed: %s', type(error).__name__)
                    last_error = time.monotonic()
            await asyncio.sleep(1)

    def broadcast(self, message):
        for queue in tuple(self.viewers.values()):
            if queue.full():
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(self.snapshot())
            else:
                queue.put_nowait(message)

    def limit(self, key, count, seconds=60, cost=1, commit=True):
        now = time.monotonic()
        if len(self.rates) > 10000:
            self.rates = {k: v for k, v in self.rates.items() if v and now-v[-1] < 60}
            if len(self.rates) > 10000:
                raise HTTPException(429, 'busy')
        recent = self.rates.setdefault(key, deque())
        while recent and now-recent[0] >= seconds:
            recent.popleft()
        if len(recent) + cost > count:
            raise HTTPException(429, 'rate_limit')
        if commit:
            recent.extend([now] * cost)

    async def maintenance(self):
        while True:
            await asyncio.sleep(5)
            if week_bounds()[0] != self.summary_week:
                summary = await asyncio.to_thread(self.store.summary)
                self.summary_week = summary['week']['start']
                self.broadcast({**summary, 'type':'summary', 'likes':self.like_total})
            watch = self.audience.tick()
            if self.snapshot()['online']:
                await asyncio.to_thread(audience.online_tick, watch)
            if self.producer and time.monotonic()-self.last_seen >= 20:
                expired = self.producer
                with contextlib.suppress(RuntimeError):
                    await expired.close(code=1013)
                if self.producer is expired:
                    self.producer = None
            self.broadcast(dict(type='presence', online=self.control_status()['online'],
                                paused=not self.control['enabled'], viewers=len(self.audience.identities())))
            if self.run:
                await asyncio.to_thread(self.store.save, self.run)
            if self.likes:
                count, self.likes = self.likes, 0
                await asyncio.to_thread(self.store.add_likes, count)
                self.broadcast(dict(type='likes', count=self.like_total))


hub = LiveHub()


def same_origin(headers):
    origin = headers.get('origin')
    if not origin:
        return
    if urlsplit(origin).netloc != headers.get('host'):
        raise HTTPException(403, 'origin_not_allowed')


@router.get('/state')
async def state(stats_range: str = Query('all')):
    return {**hub.snapshot(), **await asyncio.to_thread(hub.store.summary, stats_range), 'likes': hub.like_total, 'chat': list(hub.chat),
            'music_url': os.environ.get('LIVE_MUSIC_URL', ''), 'gifts': list(hub.gift_history)}


@router.get('/audience')
async def room_audience(request: Request, response: Response):
    hub.limit(('audience-ip', client_ip(request)), 120)
    response.headers['Cache-Control'] = 'no-store'
    return await asyncio.to_thread(audience.ranking, hub.audience.identities())


@router.get('/history')
async def run_history(request: Request, response: Response, page: int = Query(1, ge=1)):
    hub.limit(('history-ip', client_ip(request)), 120)
    response.headers['Cache-Control'] = 'no-store'
    return await asyncio.to_thread(hub.store.history, page)


@router.get('/lucky-bags')
async def lucky_list(request: Request, response: Response):
    hub.limit(('lucky-read', client_ip(request)), 120)
    user = current_user_from_request(request)
    response.headers['Cache-Control'] = 'no-store'
    return dict(bags=await asyncio.to_thread(lucky_bags.listing, user['id'] if user else None), server_time=time.time())


@router.post('/lucky-bags/{bag_id}/join')
async def lucky_join(bag_id: str, request: Request):
    same_origin(request.headers)
    user = require_user(request)
    hub.limit(('lucky-join', user['id']), 12)
    if user['id'] not in hub.viewer_users.values():
        raise HTTPException(409, 'lucky_bag_not_present')
    async with hub.lucky_lock:
        await asyncio.to_thread(lucky_bags.join, bag_id, user['id'])
        await hub.refresh_lucky()
    return dict(bags=await asyncio.to_thread(lucky_bags.listing, user['id']), server_time=time.time())


async def gift_body(request):
    same_origin(request.headers)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 2048:
            raise HTTPException(413, 'invalid_gift')
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError()
        return body
    except ValueError:
        raise HTTPException(400, 'invalid_gift')


@router.post('/red-envelopes')
async def red_send(request: Request):
    user = require_user(request)
    body = await gift_body(request)
    hub.limit(('red-send', user['id']), 20)
    if user['id'] not in hub.lucky_present(time.time()):
        raise HTTPException(409, 'red_not_present')
    actor = await asyncio.to_thread(gifts.public_actor, user)
    result = await asyncio.to_thread(red_envelopes.create, user['id'], actor, body)
    # The money has committed; a broadcast failure must not report failed payment.
    with contextlib.suppress(Exception):
        await hub.refresh_red()
    return result


@router.get('/red-envelopes/{envelope_id}')
async def red_detail(envelope_id: str, request: Request, response: Response):
    hub.limit(('red-read', client_ip(request)), 120)
    response.headers['Cache-Control'] = 'no-store'
    user = current_user_from_request(request)
    return await asyncio.to_thread(red_envelopes.detail, envelope_id, user['id'] if user else None)


@router.post('/red-envelopes/{envelope_id}/claim')
async def red_claim(envelope_id: str, request: Request):
    same_origin(request.headers)
    user = require_user(request)
    hub.limit(('red-claim', user['id']), 30)
    if user['id'] not in hub.lucky_present(time.time()):
        raise HTTPException(409, 'red_not_present')
    result = await asyncio.to_thread(red_envelopes.claim, envelope_id, user['id'])
    with contextlib.suppress(Exception):
        await hub.refresh_red()
    return result


@router.get('/gifts/catalog')
async def gift_catalog():
    catalog, _ = await asyncio.to_thread(gifts.catalogue)
    return catalog


@router.get('/gifts/me')
async def gift_me(request: Request):
    return await asyncio.to_thread(gifts.mine, require_user(request)['id'])


@router.post('/gifts/preferences')
async def gift_preferences(request: Request):
    user = require_user(request)
    body = await gift_body(request)
    return await asyncio.to_thread(gifts.set_preferences, user['id'], body.get('daily_limit_units'), body.get('entrance_enabled'))


@router.get('/gifts/orders/{request_id}')
async def gift_order(request_id: str, request: Request):
    return await asyncio.to_thread(gifts.order, require_user(request)['id'], request_id)


@router.post('/gifts/send')
async def gift_send(request: Request):
    user = require_user(request)
    body = await gift_body(request)
    hub.limit(('gift-second', user['id']), 5, seconds=1)
    hub.limit(('gift-minute', user['id']), 30)
    hub.limit(('gift-ip', client_ip(request)), 90)
    try:
        result = await asyncio.to_thread(gifts.send, user, body, hub.snapshot()['online'])
    except InsufficientTokens as error:
        raise HTTPException(402, error.payload)
    except ValueError:
        raise HTTPException(409, 'gift_request_conflict')
    # Committed orders stay successful even if broadcasting is temporarily unavailable.
    with contextlib.suppress(Exception):
        await hub.drain_gifts()
    return result


@router.get('/replays/{run_id}')
async def replay(run_id: str):
    content = await asyncio.to_thread(hub.store.replay, run_id)
    if not content:
        raise HTTPException(404, 'replay_expired')
    return Response(content, media_type='text/plain', headers={'Cache-Control': 'public, max-age=3600'})


@router.post('/chat')
async def chat(request: Request):
    same_origin(request.headers)
    actor = require_actor(request)
    body = await request.body()
    if len(body) > 1024:
        raise HTTPException(413, 'message_too_long')
    try:
        content = json.loads(body).get('text', '')
    except (ValueError, AttributeError):
        raise HTTPException(400, 'invalid_message')
    if not isinstance(content, str):
        raise HTTPException(400, 'invalid_message')
    content = content.strip()
    if not 1 <= len(content) <= 32 or any(unicodedata.category(c).startswith('C') for c in content):
        raise HTTPException(400, 'invalid_message')
    hub.limit(('chat-ip', client_ip(request)), 20)
    hub.limit(('chat', actor.actor_key), 5)
    user = current_user_from_request(request) if actor.is_user else None
    identity = await asyncio.to_thread(gifts.public_actor, user) if user else dict(
        name=actor.display_name, avatar_url=None, supporter=False, supporter_level=0)
    message = dict(type='chat', id=str(time.time_ns()), text=content, at=time.time(),
                   **identity, guest=actor.is_guest)
    hub.chat.append(message)
    hub.broadcast(message)
    if hub.snapshot()['online']:
        await asyncio.to_thread(audience.record, actor.actor_key, 'messages')
    return {'ok': True}


@router.post('/like')
async def like(request: Request):
    same_origin(request.headers)
    actor = require_actor(request)
    body = await request.body()
    if len(body) > 256:
        raise HTTPException(413, 'invalid_count')
    try:
        amount = json.loads(body or b'{}').get('count', 1)
    except (ValueError, AttributeError):
        raise HTTPException(400, 'invalid_count')
    if type(amount) is not int or not 1 <= amount <= 20:
        raise HTTPException(400, 'invalid_count')
    limits = [(('like-ip', client_ip(request)), 60), (('like', actor.actor_key), 20)]
    # Check both budgets before consuming either; this block never yields.
    for key, limit in limits:
        hub.limit(key, limit, cost=amount, commit=False)
    for key, limit in limits:
        hub.limit(key, limit, cost=amount)
    hub.likes += amount
    hub.like_total += amount
    if hub.snapshot()['online']:
        await asyncio.to_thread(audience.record, actor.actor_key, 'likes', amount)
    return {'ok': True, 'count': hub.like_total, 'accepted': amount}


@router.websocket('/watch')
async def watch(ws: WebSocket):
    try:
        same_origin(ws.headers)
        hub.limit(('connect', websocket_client_ip(ws)), 30)
    except HTTPException:
        await ws.close(code=1008)
        return
    if len(hub.viewers) >= int(os.environ.get('LIVE_MAX_VIEWERS', '200')):
        await ws.close(code=1013)
        return
    await ws.accept()
    queue = asyncio.Queue(maxsize=32)
    hub.viewers[ws] = queue
    queue.put_nowait(hub.snapshot())
    user = None
    with contextlib.suppress(Exception):
        user = await asyncio.to_thread(current_user_from_websocket, ws)
    if user:
        already_present = user['id'] in hub.viewer_users.values()
        hub.viewer_users[ws] = user['id']
        hub.viewer_times[ws] = (time.time(), time.time())
        if not already_present:
            with contextlib.suppress(Exception):
                event = await asyncio.to_thread(gifts.entrance, user)
                if event:
                    hub.append_entrance_chat(dict(type='entrance', id=event['id'], at=event['at'], **event['actor']))
                    hub.broadcast(event)
    actor = ActorRef.from_user(user) if user else None
    if not actor:
        with contextlib.suppress(Exception):
            guest = await asyncio.to_thread(current_guest_from_websocket, ws)
            if guest:
                actor = ActorRef.from_guest(guest)
    identity = dict(name=actor.display_name if actor else 'Guest', avatar_url=None,
                    supporter=False, supporter_level=0, guest=not bool(user))
    if user:
        with contextlib.suppress(Exception):
            identity = await asyncio.to_thread(gifts.public_actor, user)
    hub.audience.join(ws, actor.actor_key if actor else f'anonymous:{id(ws)}', identity)

    async def send():
        while True:
            item = await queue.get()
            if isinstance(item, bytes):
                await asyncio.wait_for(ws.send_bytes(item), 5)
            else:
                await asyncio.wait_for(ws.send_json(item), 5)

    task = asyncio.create_task(send())
    try:
        while not task.done():
            incoming = await asyncio.wait_for(ws.receive_text(), 90)
            if incoming != 'ping':
                break
            if ws in hub.viewer_times:
                hub.viewer_times[ws] = (hub.viewer_times[ws][0], time.time())
    except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
        pass
    finally:
        hub.viewers.pop(ws, None)
        hub.viewer_users.pop(ws, None)
        hub.viewer_times.pop(ws, None)
        hub.audience.leave(ws)
        task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task
        with contextlib.suppress(Exception):
            await ws.close()


@router.websocket('/publish')
async def publish(ws: WebSocket):
    secret = os.environ.get('LIVE_PUBLISH_TOKEN', '')
    supplied = ws.headers.get('authorization', '')
    if not secret or not hmac.compare_digest(supplied, 'Bearer ' + secret) or ws.headers.get('origin'):
        await ws.close(code=1008)
        return
    if hub.producer:
        await ws.close(code=1008)
        return
    hub.producer = ws
    hub.producer_ready = False
    hub.control_supported = False
    hub.control_ack = None
    hub.last_seen = time.monotonic()
    ready = False
    try:
        await ws.accept()
        while hub.producer is ws:
            packet = await asyncio.wait_for(ws.receive(), 20)
            if packet['type'] == 'websocket.disconnect':
                break
            hub.last_seen = time.monotonic()
            raw = packet.get('bytes')
            if raw is not None:
                if not ready or len(raw) != STEP.size or not hub.run:
                    raise ValueError('invalid_packet')
                if not hub.control['enabled'] and hub.control_ack == hub.control['revision']:
                    raise ValueError('publisher_paused')
                hub.run.apply(raw)
                hub.broadcast(raw)
                continue
            text = packet.get('text') or ''
            if len(text) > 710_000:
                raise ValueError('message_too_large')
            data = json.loads(text)
            action = data.get('type')
            if action == 'hello':
                await asyncio.to_thread(audience.online_tick, {})
                hub.audience.tick()
                proposed = data.get('run')
                if proposed:
                    restored = await asyncio.to_thread(LiveRun.restore, proposed)
                    if hub.run and hub.run.id != restored.id:
                        restored = hub.run
                    elif hub.run and hub.run.seed != restored.seed:
                        raise ValueError('seed_conflict')
                    elif hub.run and len(hub.run.records) > len(restored.records):
                        restored = hub.run
                    elif hub.run and not restored.records.startswith(hub.run.records):
                        raise ValueError('record_conflict')
                    hub.run = restored
                if hub.run:
                    if hub.run.ended:
                        await asyncio.to_thread(hub.store.finish, hub.run)
                    await asyncio.to_thread(hub.store.save, hub.run)
                async with hub.control_lock:
                    hub.control_supported = data.get('control_version') == 1
                    if not hub.control['enabled'] and not hub.control_supported:
                        raise ValueError('control_upgrade_required')
                    ready = hub.producer_ready = True
                    await ws.send_json({'type': 'resume', 'run': hub.run.checkpoint() if hub.run else None,
                                        'control': hub.control})
                hub.broadcast(hub.snapshot())
            elif action == 'control_ack' and ready and hub.control_supported:
                if data.get('revision') == hub.control['revision']:
                    hub.control_ack = data['revision']
                    hub.broadcast(hub.snapshot())
            elif action == 'start' and ready:
                if not hub.control['enabled'] and hub.control_ack == hub.control['revision']:
                    raise ValueError('publisher_paused')
                if hub.run and not hub.run.ended:
                    raise ValueError('run_not_finished')
                hub.run = LiveRun(data['seed'], data['run_id'])
                await asyncio.to_thread(hub.store.save, hub.run)
                hub.broadcast(hub.snapshot())
            elif action == 'source' and ready and hub.run:
                hub.run.source = str(data['source'])[:80]
                hub.broadcast({'type': 'source', 'source': hub.run.source})
            elif action == 'end' and ready and hub.run:
                hub.run.end(time.time() + min(20, max(10, int(data.get('delay', 15)))))
                await asyncio.to_thread(hub.store.finish, hub.run)
                await asyncio.to_thread(hub.store.save, hub.run)
                hub.broadcast(hub.snapshot())
                hub.broadcast({'type': 'summary', **await asyncio.to_thread(hub.store.summary), 'likes': hub.like_total})
            elif action != 'ping':
                raise ValueError('invalid_message')
    except (WebSocketDisconnect, asyncio.TimeoutError, ValueError, KeyError, TypeError):
        with contextlib.suppress(Exception):
            await ws.close(code=1008)
    finally:
        if hub.producer is ws:
            hub.producer = None
            hub.producer_ready = False
            hub.control_ack = None
            hub.broadcast(hub.snapshot())
            if hub.run:
                await asyncio.to_thread(hub.store.save, hub.run)
