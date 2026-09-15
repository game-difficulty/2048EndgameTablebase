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

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from backend.auth.dependencies import require_actor, require_user, current_user_from_request, current_user_from_websocket, client_ip, websocket_client_ip
from backend.quota.errors import InsufficientTokens
from . import gifts, audience, lucky_bags
from backend.auth.dependencies import current_guest_from_websocket
from backend.auth.principal import ActorRef
from .protocol import LiveRun, STEP
from .store import LiveStore

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

    async def start(self):
        await asyncio.to_thread(gifts.init_schema)
        await asyncio.to_thread(lucky_bags.init_schema)
        self.lucky_bags = await asyncio.to_thread(lucky_bags.listing)
        self.gift_history.extend(await asyncio.to_thread(gifts.recent_events))
        for event in self.gift_history:
            self.append_gift_chat(event)
        self.store = LiveStore()
        self.like_total = (await asyncio.to_thread(self.store.summary))['likes']
        saved = await asyncio.to_thread(self.store.load)
        if saved:
            self.run = await asyncio.to_thread(LiveRun.restore, saved)
        self.task = asyncio.create_task(self.maintenance())
        self.gift_task = asyncio.create_task(self.gift_maintenance())
        self.lucky_task = asyncio.create_task(self.lucky_maintenance())

    async def stop(self):
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
        return dict(type='snapshot', online=bool(self.producer) and time.monotonic()-self.last_seen < 20,
                    run=self.run.snapshot() if self.run else None, viewers=len(self.audience.identities()),
                    lucky_bags=self.lucky_bags, server_time=time.time())

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
            watch = self.audience.tick()
            if self.snapshot()['online']:
                await asyncio.to_thread(audience.online_tick, watch)
            if self.producer and time.monotonic()-self.last_seen >= 20:
                expired = self.producer
                with contextlib.suppress(RuntimeError):
                    await expired.close(code=1013)
                if self.producer is expired:
                    self.producer = None
            self.broadcast(dict(type='presence', online=bool(self.producer), viewers=len(self.audience.identities())))
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
async def state():
    return {**hub.snapshot(), **await asyncio.to_thread(hub.store.summary), 'likes': hub.like_total, 'chat': list(hub.chat),
            'music_url': os.environ.get('LIVE_MUSIC_URL', ''), 'gifts': list(hub.gift_history)}


@router.get('/audience')
async def room_audience(request: Request, response: Response):
    hub.limit(('audience-ip', client_ip(request)), 120)
    response.headers['Cache-Control'] = 'no-store'
    return await asyncio.to_thread(audience.ranking, hub.audience.identities())


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
                    hub.chat.append(dict(type='entrance', id=event['id'], at=event['at'], **event['actor']))
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
            incoming = await asyncio.wait_for(ws.receive_text(), 35)
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
                ready = True
                await ws.send_json({'type': 'resume', 'run': hub.run.checkpoint() if hub.run else None})
                hub.broadcast(hub.snapshot())
            elif action == 'start' and ready:
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
            hub.broadcast(hub.snapshot())
            if hub.run:
                await asyncio.to_thread(hub.store.save, hub.run)
