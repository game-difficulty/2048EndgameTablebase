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
from .store import week_bounds
from .rooms import DEFAULT_ROOM, DEFAULT_ROOM_ID, ROOMS, RoomDefinition
from .content import create_content
from . import human_rooms
from backend.room_activities import predictions
from backend.room_activities.runtime import RoomActivities

router = APIRouter(prefix='/api/live', tags=['live'])


class ViewerQueue(asyncio.Queue):
    """Bound both message count and pending payload bytes per spectator."""
    def __init__(self):
        super().__init__(maxsize=32)
        self.bytes = 0
        self.closing = False

    @staticmethod
    def size(item):
        return len(item) if isinstance(item, bytes) else len(json.dumps(item, ensure_ascii=False).encode('utf-8'))

    def put_nowait(self, item):
        size = self.size(item)
        if self.bytes + size > 256 * 1024:
            raise asyncio.QueueFull
        super().put_nowait((item, size))
        self.bytes += size

    def get_nowait(self):
        item, size = super().get_nowait()
        self.bytes -= size
        return item


class LiveHub:
    def __init__(self, room=DEFAULT_ROOM):
        self.room = room
        self.content = create_content(room)
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
        self.room_ended_notified = False
        self.red_lock = asyncio.Lock()
        self.red_state = dict(active=None, queued=0)
        self.red_task = None
        self.stats_task = None
        self.activities = RoomActivities(self)
        self.activity_lock = self.activities.lock
        self.activity_task = None

    @property
    def store(self):
        return self.content.store

    @store.setter
    def store(self, value):
        self.content.store = value

    @property
    def run(self):
        return self.content.run

    @run.setter
    def run(self, value):
        self.content.run = value

    async def start(self):
        await asyncio.to_thread(gifts.init_schema)
        await asyncio.to_thread(lucky_bags.init_schema)
        await asyncio.to_thread(red_envelopes.init_schema)
        await asyncio.to_thread(predictions.init_schema)
        self.red_state = await asyncio.to_thread(red_envelopes.tick, room_id=self.room.id)
        self.lucky_bags = await asyncio.to_thread(lucky_bags.listing, room_id=self.room.id)
        self.gift_history.extend(await asyncio.to_thread(gifts.recent_events, self.room.target))
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
        history = [*self.chat, *await asyncio.to_thread(red_envelopes.events, room_id=self.room.id),
                   *await asyncio.to_thread(predictions.announcement_events, self.room.id)]
        self.chat = deque(sorted(history, key=lambda item: item['at'])[-100:], maxlen=100)
        await self.content.start()
        if hasattr(self.content, 'batch'):
            self.content.hub = self
            await self.content.advance_batch(force=True)
        await self.activities.refresh()
        self.control = await asyncio.to_thread(self.store.control)
        await asyncio.to_thread(self.store.refresh_stats_snapshots)
        summary = await asyncio.to_thread(self.store.summary)
        self.like_total = summary['likes']
        self.summary_week = summary['week']['start']
        self.task = asyncio.create_task(self.maintenance())
        self.gift_task = asyncio.create_task(self.gift_maintenance())
        self.lucky_task = asyncio.create_task(self.lucky_maintenance())
        self.red_task = asyncio.create_task(self.red_maintenance())
        self.stats_task = asyncio.create_task(self.backfill_stats())
        self.activity_task = asyncio.create_task(self.activity_maintenance())

    async def activity_maintenance(self):
        while True:
            try:
                async with self.activity_lock:
                    if hasattr(self.content, 'advance_batch'):
                        await self.content.advance_batch()
                    await self.activities.drain()
                    await self.activities.refresh()
            except Exception:
                logging.getLogger(__name__).exception('Room activity reconciliation delayed')
            await asyncio.sleep(1)

    async def backfill_stats(self):
        changed = False
        while await asyncio.to_thread(self.store.backfill_stats, limit=1):
            changed = True
            await asyncio.sleep(0)
        if changed:
            await asyncio.to_thread(self.store.refresh_stats_snapshots)
            summary = await asyncio.to_thread(self.store.summary)
            self.broadcast(dict(type='summary', **summary))

    async def publisher_joined(self):
        await asyncio.to_thread(audience.online_tick, {}, room_id=self.room.id)
        self.audience.tick()

    async def stop(self):
        if self.activity_task:
            self.activity_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.activity_task
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
        await self.content.save()
        if self.store:
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
            pending = await asyncio.to_thread(gifts.pending_events, self.room.target)
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
        return dict(type='snapshot', room_id=self.room.id, protocol=self.room.protocol, online=self.control_status()['online'], paused=not self.control['enabled'],
                    **self.content.snapshot(), viewers=len(self.audience.identities()),
                    lucky_bags=self.lucky_bags, red_envelopes=self.red_state,
                    predictions=self.activities.state, server_time=time.time())

    async def refresh_red(self):
        async with self.red_lock:
            state = await asyncio.to_thread(red_envelopes.tick, room_id=self.room.id)
            if state != self.red_state:
                self.red_state = state
                self.broadcast(dict(type='red_envelopes', **state, server_time=time.time()))
            pending = await asyncio.to_thread(red_envelopes.events, True, room_id=self.room.id)
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
                    run_id=self.content.run_id,
                    lanes=self.content.control_lanes() if hasattr(self.content, 'control_lanes') else [])

    async def set_enabled(self, enabled):
        async with self.control_lock, self.activity_lock:
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
        present = {uid for ws, uid in self.viewer_users.items()
                if ws in self.viewer_times and self.viewer_times[ws][0] <= deadline
                and self.viewer_times[ws][1] >= now - 35}
        if self.room.content_kind == 'human-play':
            present.discard(int(self.room.metadata.get('owner_user_id', -1)))
        return present

    async def refresh_lucky(self):
        self.lucky_bags = await asyncio.to_thread(lucky_bags.listing, room_id=self.room.id)
        self.broadcast(dict(type='lucky_bags', bags=self.lucky_bags, server_time=time.time()))

    async def tick_lucky(self):
        async with self.lucky_lock:
            # Only two milestone inserts per run; no database work on ordinary steps.
            reached = self.content.reached_milestones() if self.room.milestone_rewards else set()
            new = reached - self.lucky_seen
            for run_id, value in sorted(new):
                creator = lucky_bags.create_human if self.room.content_kind == 'human-play' else lucky_bags.create
                await asyncio.to_thread(creator, run_id, value, room_id=self.room.id)
            self.lucky_seen = reached
            if new:
                await self.refresh_lucky()
            changed = False
            now = time.time()
            for bag in self.lucky_bags:
                if bag['drawn_at'] is None and bag['draw_at'] <= now:
                    present = self.lucky_present(bag['draw_at'])
                    changed |= await asyncio.to_thread(lucky_bags.draw, bag['id'], present, room_id=self.room.id)
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
            if getattr(queue, 'closing', False):
                continue
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                while not queue.empty():
                    queue.get_nowait()
                # Reconnect reloads the room snapshot AND social histories; silently
                # replacing the queue with a board snapshot would lose paid events.
                queue.closing = True
                queue.put_nowait(None)

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
        last_stats_refresh = 0
        while True:
            await asyncio.sleep(5)
            if time.monotonic() - last_stats_refresh >= 60:
                await asyncio.to_thread(self.store.refresh_stats_snapshots, ('24h',))
                last_stats_refresh = time.monotonic()
            if week_bounds()[0] != self.summary_week:
                await asyncio.to_thread(self.store.refresh_stats_snapshots, ('24h', 'recent100', 'all'))
                summary = await asyncio.to_thread(self.store.summary)
                self.summary_week = summary['week']['start']
                self.broadcast({**summary, 'type':'summary', 'likes':self.like_total})
            watch = self.audience.tick()
            if self.snapshot()['online']:
                await asyncio.to_thread(audience.online_tick, watch, room_id=self.room.id)
            if self.producer and time.monotonic()-self.last_seen >= (20 if self.producer_ready else 180):
                expired = self.producer
                with contextlib.suppress(RuntimeError):
                    await expired.close(code=1013)
                if self.producer is expired:
                    self.producer = None
            self.broadcast(dict(type='presence', online=self.control_status()['online'],
                                paused=not self.control['enabled'], viewers=len(self.audience.identities())))
            await self.content.save()
            if self.likes:
                count, self.likes = self.likes, 0
                await asyncio.to_thread(self.store.add_likes, count)
                self.broadcast(dict(type='likes', count=self.like_total))


hub = LiveHub()

room_hubs = {}
dynamic_hubs = {}
dynamic_task = None


def _human_definition(data):
    name = data.get('display_name') or 'Player'
    return RoomDefinition(
        id=data['room_id'], title={'zh': f'{name} 的直播', 'en': f'{name}\'s stream'},
        description={'zh': f"{data['variant']} 玩家实时对局",
                     'en': f"Live {data['variant']} player game"},
        content_kind='human-play', protocol='human-play-v1',
        milestone_rewards=True, dynamic=True,
        metadata={'variant': data['variant'], 'owner_user_id': data['owner_user_id'],
                  'run_id': data['run_id'], 'generation': int(data['generation']),
                  'streamer': {'display_name': name, 'avatar_url': data.get('avatar_url')},
                  'started_at': data['started_at']},
    )


def _dynamic_hub(room_id):
    data = human_rooms.room(room_id)
    if not data:
        raise HTTPException(404, 'room_not_found')
    runtime = dynamic_hubs.get(room_id)
    if (runtime and runtime.room.metadata.get('variant') == data['variant']
            and runtime.room.metadata.get('run_id') == data['run_id']
            and int(runtime.room.metadata.get('generation', 0)) == int(data['generation'])):
        # Name/avatar changes do not create a new run generation. Refresh the
        # mutable presentation payload while preserving viewers and activities.
        current = _human_definition(data)
        runtime.room.title.clear(); runtime.room.title.update(current.title)
        runtime.room.description.clear(); runtime.room.description.update(current.description)
        runtime.room.metadata.clear(); runtime.room.metadata.update(current.metadata)
        return runtime
    if runtime and (runtime.producer or runtime.viewers):
        # A run switch changes the generation; the old publisher must reconnect.
        if runtime.producer:
            asyncio.create_task(runtime.producer.close(code=1012))
    runtime = LiveHub(_human_definition(data))
    # Dynamic rooms share the auth database and retain paid/social state across
    # a live-service restart without allocating a per-room replay database.
    runtime.lucky_bags = lucky_bags.listing(room_id=room_id)
    runtime.gift_history.extend(gifts.recent_events(runtime.room.target))
    for event in runtime.gift_history:
        runtime.append_gift_chat(event)
    dynamic_hubs[room_id] = runtime
    return runtime


def resolve_hub(connection=None):
    room_id = connection.path_params.get('room_id', DEFAULT_ROOM_ID) if connection is not None else DEFAULT_ROOM_ID
    if room_id == DEFAULT_ROOM_ID:
        return hub
    if room_id in ROOMS and room_id in room_hubs:
        return room_hubs[room_id]
    return _dynamic_hub(room_id)


async def _reconcile_dynamic_room(room_id, runtime, definition, now):
    if not definition:
        if not runtime.room_ended_notified:
            runtime.room_ended_notified = True
            runtime.broadcast(dict(type='room_ended', room_id=room_id,
                                   reason='room_ended'))
        if runtime.producer:
            with contextlib.suppress(Exception):
                await runtime.producer.close(code=1008)
        if not runtime.viewers and dynamic_hubs.get(room_id) is runtime:
            dynamic_hubs.pop(room_id, None)
        return
    metadata = definition.metadata if isinstance(definition, RoomDefinition) else definition
    if int(runtime.room.metadata.get('generation', 0)) != int(metadata.get('generation', 0)):
        replacement = _dynamic_hub(room_id)
        if replacement is not runtime:
            # Existing websocket handlers retain the old hub: reconnect to the new one.
            for viewer in tuple(runtime.viewers):
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(viewer.close(code=1012), 5)
        return
    if runtime.producer and now - runtime.last_seen >= 20:
        with contextlib.suppress(Exception):
            await runtime.producer.close(code=1013)
    if int(now) % 5 == 0:
        runtime.broadcast(dict(type='presence', online=runtime.control_status()['online'],
            paused=False, viewers=len(runtime.audience.identities())))


async def dynamic_maintenance():
    cursor = 0
    last_expiry = 0.0
    while True:
        await asyncio.sleep(1)
        items = list(dynamic_hubs.items())
        now = time.monotonic()
        if now - last_expiry >= 15:
            await asyncio.to_thread(human_rooms.expire_stale,
                stale_after=float(os.environ.get('HUMAN_LIVE_RECOVERY_SECONDS', '90')))
            last_expiry = now
        active = {item['room_id']: item for item in await asyncio.to_thread(human_rooms.active_rooms)}
        for room_id, runtime in items:
            await _reconcile_dynamic_room(room_id, runtime, active.get(room_id), now)
        items = list(dynamic_hubs.items())
        if items:
            # Recovery/activity work is deliberately staggered across dynamic rooms.
            _, runtime = items[cursor % len(items)]
            cursor += 1
            try:
                if hasattr(runtime.content, 'verify_pending_milestones'):
                    await runtime.content.verify_pending_milestones()
                await runtime.drain_gifts()
                await runtime.refresh_red()
                await runtime.tick_lucky()
            except Exception:
                logging.getLogger(__name__).exception('Dynamic room reconciliation delayed')


async def start_rooms():
    global dynamic_task
    human_rooms.init_schema()
    await hub.start()
    try:
        for room_id, definition in ROOMS.items():
            if room_id != DEFAULT_ROOM_ID:
                runtime = LiveHub(definition)
                room_hubs[room_id] = runtime
                await runtime.start()
        dynamic_task = asyncio.create_task(dynamic_maintenance())
    except BaseException:
        await stop_rooms()
        raise


async def stop_rooms():
    global dynamic_task
    if dynamic_task:
        dynamic_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await dynamic_task
        dynamic_task = None
    for runtime in [*dynamic_hubs.values()]:
        if runtime.producer:
            with contextlib.suppress(Exception):
                await runtime.producer.close(code=1012)
    dynamic_hubs.clear()
    for runtime in [*room_hubs.values(), hub]:
        await runtime.stop()
    room_hubs.clear()


@router.get('/rooms')
async def list_rooms():
    dynamic = [_human_definition(item).public() for item in human_rooms.active_rooms()]
    return {'rooms': [room.public() for room in ROOMS.values()] + dynamic}


@router.get('/lobby')
async def lobby(response: Response):
    response.headers['Cache-Control'] = 'public, max-age=5, stale-while-revalidate=5'
    entries = []
    for runtime in [hub, *room_hubs.values(), *dynamic_hubs.values()]:
        snapshot = runtime.snapshot()
        if not snapshot.get('online'):
            continue
        run = snapshot.get('run')
        if not run and snapshot.get('lanes'):
            running = [item.get('run') for item in snapshot['lanes'] if item.get('run')]
            run = max(running, key=lambda item: item.get('score', 0), default=None)
        entries.append({
            'id': runtime.room.id, 'path': '/rooms/' + runtime.room.id,
            'title': runtime.room.title, 'content_kind': runtime.room.content_kind,
            'streamer': runtime.room.metadata.get('streamer'),
            'variant': (run or {}).get('variant', runtime.room.metadata.get('variant', '4x4')),
            'board': (run or {}).get('board'), 'score': (run or {}).get('score', 0),
            'appearance': (run or {}).get('appearance'),
            'max_tile': max((run or {}).get('board') or [0]),
            'seq': (run or {}).get('seq', 0), 'viewers': len(runtime.audience.identities()),
            'started_at': (run or {}).get('started_at', runtime.room.metadata.get('started_at')),
        })
    entries.sort(key=lambda item: (-item['viewers'], -item['score'], item['id']))
    return {'rooms': entries, 'generated_at': time.time()}


@router.get('/rooms/{room_id}')
async def room_detail(room_id: str):
    if room_id in ROOMS:
        return ROOMS[room_id].public()
    data = human_rooms.room(room_id)
    if not data:
        raise HTTPException(404, 'room_not_found')
    return _human_definition(data).public()


def same_origin(headers):
    origin = headers.get('origin')
    if not origin:
        return
    if urlsplit(origin).netloc != headers.get('host'):
        raise HTTPException(403, 'origin_not_allowed')


@router.get('/rooms/{room_id}/state')
@router.get('/state')
async def state(request: Request, stats_range: str = Query('all')):
    hub = resolve_hub(request)
    from backend.chat_moderation import visible_messages
    history = await asyncio.to_thread(visible_messages, list(hub.chat))
    return {**hub.snapshot(), **await asyncio.to_thread(hub.store.summary, stats_range), 'likes': hub.like_total, 'chat': history,
            'music_url': os.environ.get('LIVE_MUSIC_URL', ''), 'gifts': list(hub.gift_history)}


@router.get('/status')
async def live_status(response: Response):
    response.headers['Cache-Control'] = 'no-store'
    return {'online': bool(hub.control_status()['online'])}


@router.get('/rooms/{room_id}/audience')
@router.get('/audience')
async def room_audience(request: Request, response: Response):
    hub = resolve_hub(request)
    hub.limit(('audience-ip', client_ip(request)), 120)
    response.headers['Cache-Control'] = 'no-store'
    return await asyncio.to_thread(audience.ranking, hub.audience.identities(), room_id=hub.room.id)


@router.get('/rooms/{room_id}/history')
@router.get('/history')
async def run_history(request: Request, response: Response, page: int = Query(1, ge=1)):
    hub = resolve_hub(request)
    hub.limit(('history-ip', client_ip(request)), 120)
    response.headers['Cache-Control'] = 'no-store'
    return await asyncio.to_thread(hub.store.history, page)


@router.get('/rooms/{room_id}/lucky-bags')
@router.get('/lucky-bags')
async def lucky_list(request: Request, response: Response):
    hub = resolve_hub(request)
    hub.limit(('lucky-read', client_ip(request)), 120)
    user = current_user_from_request(request)
    response.headers['Cache-Control'] = 'no-store'
    return dict(bags=await asyncio.to_thread(lucky_bags.listing, user['id'] if user else None, room_id=hub.room.id), server_time=time.time())


@router.get('/rooms/{room_id}/predictions')
async def prediction_state(request: Request, response: Response, market_id: str = Query(None, max_length=36)):
    hub = resolve_hub(request)
    hub.limit(('prediction-read', client_ip(request)), 120)
    response.headers['Cache-Control'] = 'no-store'
    user = current_user_from_request(request)
    state = await asyncio.to_thread(predictions.listing, hub.room.id, user['id'] if user else None, market_id)
    return dict(state, available=hub.control_status()['online'])


@router.post('/rooms/{room_id}/predictions')
async def prediction_place(request: Request):
    hub = resolve_hub(request)
    user = require_user(request)
    body = await gift_body(request)
    hub.limit(('prediction-write', user['id']), 30)
    async with hub.activity_lock:
        if hasattr(hub.content, 'advance_batch'):
            await hub.content.advance_batch()
        result = await asyncio.to_thread(predictions.place, hub.room.id, user['id'], body,
                                         lambda: hub.control_status()['online'])
    return result


@router.post('/rooms/{room_id}/lucky-bags/{bag_id}/join')
@router.post('/lucky-bags/{bag_id}/join')
async def lucky_join(bag_id: str, request: Request):
    hub = resolve_hub(request)
    same_origin(request.headers)
    user = require_user(request)
    hub.limit(('lucky-join', user['id']), 12)
    if user['id'] not in hub.viewer_users.values():
        raise HTTPException(409, 'lucky_bag_not_present')
    async with hub.lucky_lock:
        await asyncio.to_thread(lucky_bags.join, bag_id, user['id'], room_id=hub.room.id)
        await hub.refresh_lucky()
    return dict(bags=await asyncio.to_thread(lucky_bags.listing, user['id'], room_id=hub.room.id), server_time=time.time())


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


@router.post('/rooms/{room_id}/red-envelopes')
@router.post('/red-envelopes')
async def red_send(request: Request):
    hub = resolve_hub(request)
    user = require_user(request)
    body = await gift_body(request)
    hub.limit(('red-send', user['id']), 20)
    if user['id'] not in hub.lucky_present(time.time()):
        raise HTTPException(409, 'red_not_present')
    actor = await asyncio.to_thread(gifts.public_actor, user)
    result = await asyncio.to_thread(red_envelopes.create, user['id'], actor, body, room_id=hub.room.id)
    # The money has committed; a broadcast failure must not report failed payment.
    with contextlib.suppress(Exception):
        await hub.refresh_red()
    return result


@router.get('/rooms/{room_id}/red-envelopes/{envelope_id}')
@router.get('/red-envelopes/{envelope_id}')
async def red_detail(envelope_id: str, request: Request, response: Response):
    hub = resolve_hub(request)
    hub.limit(('red-read', client_ip(request)), 120)
    response.headers['Cache-Control'] = 'no-store'
    user = current_user_from_request(request)
    return await asyncio.to_thread(red_envelopes.detail, envelope_id, user['id'] if user else None, room_id=hub.room.id)


@router.post('/rooms/{room_id}/red-envelopes/{envelope_id}/claim')
@router.post('/red-envelopes/{envelope_id}/claim')
async def red_claim(envelope_id: str, request: Request):
    hub = resolve_hub(request)
    same_origin(request.headers)
    user = require_user(request)
    hub.limit(('red-claim', user['id']), 30)
    if user['id'] not in hub.lucky_present(time.time()):
        raise HTTPException(409, 'red_not_present')
    result = await asyncio.to_thread(red_envelopes.claim, envelope_id, user['id'], room_id=hub.room.id)
    with contextlib.suppress(Exception):
        await hub.refresh_red()
    return result


@router.get('/rooms/{room_id}/gifts/catalog')
@router.get('/gifts/catalog')
async def gift_catalog(request: Request):
    hub = resolve_hub(request)
    catalog, _ = await asyncio.to_thread(gifts.catalogue)
    return catalog


@router.get('/rooms/{room_id}/gifts/me')
@router.get('/gifts/me')
async def gift_me(request: Request):
    hub = resolve_hub(request)
    return await asyncio.to_thread(gifts.mine, require_user(request)['id'])


@router.post('/rooms/{room_id}/gifts/preferences')
@router.post('/gifts/preferences')
async def gift_preferences(request: Request):
    hub = resolve_hub(request)
    user = require_user(request)
    body = await gift_body(request)
    return await asyncio.to_thread(gifts.set_preferences, user['id'], body.get('daily_limit_units'), body.get('entrance_enabled'))


@router.post('/rooms/{room_id}/gifts/preferences/order')
@router.post('/gifts/preferences/order')
async def gift_order_preferences(request: Request):
    hub = resolve_hub(request)
    if not hub.room.public()['capabilities'].get('gifts', False):
        raise HTTPException(404, 'room_capability_disabled')
    user = require_user(request)
    body = await gift_body(request)
    return await asyncio.to_thread(gifts.set_gift_order, user['id'], body.get('gift_ids'))


@router.get('/rooms/{room_id}/gifts/orders/{request_id}')
@router.get('/gifts/orders/{request_id}')
async def gift_order(request_id: str, request: Request):
    hub = resolve_hub(request)
    return await asyncio.to_thread(gifts.order, require_user(request)['id'], request_id, hub.room.target)


@router.post('/rooms/{room_id}/gifts/send')
@router.post('/gifts/send')
async def gift_send(request: Request):
    hub = resolve_hub(request)
    user = require_user(request)
    body = await gift_body(request)
    hub.limit(('gift-second', user['id']), 5, seconds=1)
    hub.limit(('gift-minute', user['id']), 30)
    hub.limit(('gift-ip', client_ip(request)), 90)
    try:
        result = await asyncio.to_thread(gifts.send, user, body, hub.snapshot()['online'], target=hub.room.target)
    except InsufficientTokens as error:
        raise HTTPException(402, error.payload)
    except ValueError:
        raise HTTPException(409, 'gift_request_conflict')
    # Committed orders stay successful even if broadcasting is temporarily unavailable.
    with contextlib.suppress(Exception):
        await hub.drain_gifts()
    return result


@router.get('/rooms/{room_id}/replays/{run_id}')
@router.get('/replays/{run_id}')
async def replay(run_id: str, request: Request):
    hub = resolve_hub(request)
    content = await asyncio.to_thread(hub.store.replay, run_id)
    if not content:
        raise HTTPException(404, 'replay_expired')
    return Response(content, media_type='text/plain', headers={'Cache-Control': 'public, max-age=3600'})


@router.post('/rooms/{room_id}/chat')
@router.post('/chat')
async def chat(request: Request):
    hub = resolve_hub(request)
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
    from .chat_length import chat_length, LIMIT
    if not 1 <= chat_length(content) <= LIMIT or any(
            unicodedata.category(c).startswith('C') and c != '\u200d' for c in content):
        raise HTTPException(400, 'invalid_message')
    hub.limit(('chat-ip', client_ip(request)), 20)
    hub.limit(('chat', actor.actor_key), 5)
    from backend.chat_moderation import blocked
    if blocked(content):
        raise HTTPException(400, 'message_blocked')
    user = current_user_from_request(request) if actor.is_user else None
    identity = await asyncio.to_thread(gifts.public_actor, user) if user else dict(
        name=actor.display_name, avatar_url=None, supporter=False, supporter_level=0)
    message = dict(type='chat', id=str(time.time_ns()), text=content, at=time.time(),
                   **identity, guest=actor.is_guest, user_id=actor.user_id if actor.is_user else None)
    hub.chat.append(message)
    hub.broadcast(message)
    if hub.snapshot()['online']:
        await asyncio.to_thread(audience.record, actor.actor_key, 'messages', room_id=hub.room.id)
    return {'ok': True}


@router.post('/rooms/{room_id}/like')
@router.post('/like')
async def like(request: Request):
    hub = resolve_hub(request)
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
        await asyncio.to_thread(audience.record, actor.actor_key, 'likes', amount, room_id=hub.room.id)
    return {'ok': True, 'count': hub.like_total, 'accepted': amount}


@router.websocket('/rooms/{room_id}/watch')
@router.websocket('/watch')
async def watch(ws: WebSocket):
    hub = resolve_hub(ws)
    try:
        same_origin(ws.headers)
        hub.limit(('connect', websocket_client_ip(ws)), 30)
    except HTTPException:
        await ws.close(code=1008)
        return
    total_viewers = len(routes_viewers())
    if (len(hub.viewers) >= int(os.environ.get('LIVE_MAX_VIEWERS', '200'))
            or total_viewers >= int(os.environ.get('LIVE_MAX_TOTAL_VIEWERS', '120'))):
        await ws.close(code=1013)
        return
    await ws.accept()
    queue = ViewerQueue()
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
            if item is None:
                await asyncio.wait_for(ws.close(code=1013, reason='slow_consumer'), 5)
                return
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


@router.websocket('/rooms/{room_id}/publish')
@router.websocket('/publish')
async def publish(ws: WebSocket):
    hub = resolve_hub(ws)
    secret = os.environ.get(hub.room.publish_token_env, '')
    supplied = ws.headers.get('authorization', '')
    if not secret or not hmac.compare_digest(supplied, 'Bearer ' + secret) or ws.headers.get('origin'):
        await ws.close(code=1008)
        return
    await hub.content.publish(ws, hub)


def routes_viewers():
    return [viewer for runtime in [hub, *room_hubs.values(), *dynamic_hubs.values()]
            for viewer in runtime.viewers]


@router.websocket('/rooms/{room_id}/human-publish')
async def human_publish(ws: WebSocket):
    hub = resolve_hub(ws)
    if hub.room.content_kind != 'human-play':
        await ws.close(code=1008)
        return
    origin = ws.headers.get('origin', '')
    allowed = {value.rstrip('/') for value in os.environ.get(
        'HUMAN_LIVE_ALLOWED_ORIGINS',
        'https://play.2048tables.online,http://127.0.0.1:8765,http://localhost:8765').split(',') if value}
    if origin not in allowed:
        await ws.close(code=1008)
        return
    user = None
    with contextlib.suppress(Exception):
        user = await asyncio.to_thread(current_user_from_websocket, ws)
    if not user or int(user['id']) != int(hub.room.metadata.get('owner_user_id', -1)):
        await ws.close(code=1008)
        return
    await hub.content.publish(ws, hub, user)
