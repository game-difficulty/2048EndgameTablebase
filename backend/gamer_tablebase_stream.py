"""Bounded Gamer route subscriptions. Transport acknowledgements never gate movement."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from fastapi import HTTPException
from pydantic import ValidationError

from backend.auth.db import auth_db
from backend.auth.service import parse_iso, utcnow
from backend.gamer_tablebase import RouteRequest, check_query_budget
from backend.gamer_tablebase_route import GamerRouteCursor
from backend.gamer_stream_window import StreamWindow, MAX_WINDOW, REMOTE_LOOKAHEAD, STREAM_IDLE_SECONDS, MAX_STREAM_STEPS, wait_stream_event
from backend.quota.config import apply_pricing_multipliers, operation_cost_units, resolve_pricing_snapshot, table_multiplier_units, token_to_units
from backend.quota.errors import InsufficientTokens
from backend.quota.service import consume_operation_tokens_once, get_token_balance, has_numeric_result
from backend.tablebase_catalog import ai_table_metadata, get_catalog_version, resolve_tablebase
from backend.tablebase_query_service import TablebaseLookupSpec, tablebase_query_scheduler
from backend.remote_workers.registry import remote_worker_registry
from engine_core.BookReader import BookReaderDispatcher

OPEN, CREDIT, CANCEL, EVENT = ('GAMER_STREAM_OPEN', 'GAMER_STREAM_CREDIT', 'GAMER_STREAM_CANCEL', 'GAMER_STREAM_EVENT')


def validate_session(user_id, session_id):
    with auth_db() as db:
        row = db.execute('''SELECT s.expires_at, s.revoked_at, u.status FROM sessions s
            JOIN users u ON u.id=s.user_id WHERE s.id=? AND s.user_id=?''', (session_id, user_id)).fetchone()
    if (row is None or row['revoked_at'] or row['status'] != 'active'
            or (parse_iso(row['expires_at']) or utcnow()) <= utcnow()):
        raise HTTPException(401, 'AUTH_REQUIRED')


@dataclass
class RouteSubscription:
    request: RouteRequest
    user_id: int
    session_id: int
    supporter: bool
    descriptor: dict
    socket: object
    sender: object
    window: StreamWindow
    fingerprint: str
    frames: OrderedDict = field(default_factory=OrderedDict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    attached: asyncio.Event = field(default_factory=asyncio.Event)
    task: object = None
    remote: object = None
    reader: object = None
    end: dict | None = None
    auth_checked: float = 0
    balance: dict | None = None

    async def send(self, data):
        async with self.lock:
            if self.socket is not None:
                if not await self.sender(self.socket, {'action': EVENT,
                    'data': {'route_id': self.request.request_id, **data}}):
                    self.socket = None
                    self.attached.clear()

    async def check(self):
        if time.monotonic() - self.auth_checked > 5:
            await asyncio.to_thread(validate_session, self.user_id, self.session_id)
            if self.request.catalog_version != get_catalog_version():
                raise HTTPException(409, 'CATALOG_CHANGED')
            self.auth_checked = time.monotonic()

    def query_cost(self):
        return apply_pricing_multipliers(operation_cost_units('trainer_lookup_hit'),
            table_multiplier_units(self.request.full_pattern), resolve_pricing_snapshot().global_multiplier_units)

    def affordable_limit(self, max_window=None):
        max_window = max_window or self.remote.max_window
        cost = self.query_cost()
        balance = token_to_units(self.balance['total'])
        count = max_window if cost <= 0 else min(max_window, balance // cost)
        return min(self.window.allow_through + REMOTE_LOOKAHEAD,
                   self.window.produced + max(0, count), MAX_STREAM_STEPS - 1)

    async def replenish_remote(self):
        if self.remote is None or not self.attached.is_set():
            return
        allowed = self.affordable_limit()
        # Cloud consumption, not browser consumption, drains the unpaid buffer.
        if allowed > self.remote.allowed and (allowed - self.remote.allowed >= 4
                or self.remote.allowed - self.window.produced <= 4):
            await self.remote.credit(self.window.produced, allowed)

    async def lookup(self, cursor, seq):
        spec = TablebaseLookupSpec(cursor.encoded, self.descriptor['pattern'], str(self.descriptor['target']),
            self.request.full_pattern, False, self.reader, self.descriptor['_provider'], self.request.catalog_version)
        handle = await tablebase_query_scheduler.submit(spec,
            stream_key=f'gamer-stream:{self.user_id}:{self.request.request_id}',
            supporter=self.supporter, lane='foreground' if seq == 0 else 'prefetch', supersede=False)
        try:
            result = await handle.wait()
            return cursor.node(result.results, result.dtype)
        finally:
            handle.cancel()

    def bill(self, seq, item):
        with auth_db() as db:
            consume_operation_tokens_once(request_id=f'gamer-ai:{self.request.request_id}:{seq}',
                user_id=self.user_id, session_id=self.session_id, db=db,
                operation_key='trainer_lookup_hit' if has_numeric_result(item['results']) else 'trainer_lookup_miss',
                full_pattern=self.request.full_pattern, idempotency_scope=f'gamer-ai:{self.fingerprint}:{seq}',
                metadata={'source': 'gamer_ai', 'board_hex': item['lookup_board'], 'route_index': seq})
            return get_token_balance(self.user_id, db=db)

    async def produce(self):
        try:
            await self.check()
            request = self.request
            options = {key: getattr(request, key) for key in
                ('board_codes', 'rng_state', 'difficulty', 'spawn_rate4', 'random_only')}
            options['steps'] = 1
            cursor = GamerRouteCursor(options, ai_table_metadata(self.descriptor)['large_tiles'])
            await asyncio.to_thread(check_query_budget, self.user_id, request, self.fingerprint, 0)
            self.balance = await asyncio.to_thread(get_token_balance, self.user_id)
            if self.descriptor['_provider'] == 'local':
                self.reader = BookReaderDispatcher()
                await asyncio.to_thread(self.reader.dispatch,
                    [(self.descriptor['_absolute_path'], self.descriptor.get('dtype', 'uint32'))],
                    self.descriptor['pattern'], str(self.descriptor['target']))
            if request.advance_first:
                anchor = await self.lookup(cursor, -1)
                if not cursor.advance(anchor['results'], anchor['dtype']):
                    self.end = {'type': 'end', 'reason': 'route_end'}
                    return
                options.update(board_codes=[0 if v == 0 else v.bit_length()-1 for v in cursor.values],
                               rng_state=cursor.rng.state.copy(), random_only=False)
            if self.descriptor['_provider'] == 'remote' and remote_worker_registry.supports_gamer_stream(request.full_pattern):
                allowed = self.affordable_limit(remote_worker_registry.gamer_stream_window(request.full_pattern))
                self.remote = await remote_worker_registry.open_gamer_stream(full_pattern=request.full_pattern,
                    pattern=self.descriptor['pattern'], target=self.descriptor['target'], options=options,
                    allow_through=max(0, allowed))
            for seq in range(MAX_STREAM_STEPS):
                await self.window.wait(seq)
                await wait_stream_event(self.attached, STREAM_IDLE_SECONDS)
                await self.check()
                # Advisory credit uses the last committed balance. The transaction below
                # remains authoritative if another tab spends tokens concurrently.
                if token_to_units(self.balance['total']) < self.query_cost():
                    await asyncio.to_thread(check_query_budget, self.user_id, request, self.fingerprint, seq)
                if self.remote is not None:
                    item = await self.remote.receive()
                    if item is None:
                        break
                    expected = cursor.node(item.get('results', {}), str(item.get('dtype', '')))
                    if item != expected:
                        raise ValueError('Invalid remote route state')
                    tablebase_query_scheduler.cache_stream_result(catalog_version=request.catalog_version,
                        full_pattern=request.full_pattern, board_encoded=cursor.encoded,
                        results=item['results'], dtype=item['dtype'])
                else:
                    item = await self.lookup(cursor, seq)
                # A reconnect replays these paid frames, not a second lookup/charge.
                billing = asyncio.create_task(asyncio.to_thread(self.bill, seq, item))
                try:
                    balance = await asyncio.shield(billing)
                except asyncio.CancelledError:
                    await billing
                    raise
                self.balance = balance
                frame = {'type': 'result', 'seq': seq, **item, 'full_pattern': request.full_pattern,
                         'catalog_version': request.catalog_version, 'token_balance': balance}
                self.frames[seq] = frame
                self.window.produced = seq
                await self.send(frame)
                if not cursor.advance(item['results'], item['dtype']):
                    break
                # Credits are forwarded in advance, never one ACK per node.
                await self.replenish_remote()
            self.end = {'type': 'end', 'reason': 'route_end'}
        except asyncio.CancelledError:
            raise
        except InsufficientTokens as exc:
            self.end = {'type': 'error', 'status': 402, 'detail': exc.payload}
        except HTTPException as exc:
            self.end = {'type': 'error', 'status': exc.status_code, 'detail': exc.detail}
        except Exception:
            self.end = {'type': 'error', 'status': 503, 'detail': 'TABLEBASE_UNAVAILABLE'}
        finally:
            if self.remote is not None:
                await self.remote.close()
            if self.end is not None:
                await self.send(self.end)


class GamerStreamService:
    def __init__(self):
        self.routes = {}
        self.cleanup_tasks = set()

    async def stop(self, key):
        state = self.routes.pop(key, None)
        if state is not None:
            state.task.cancel()
            await asyncio.gather(state.task, return_exceptions=True)

    def finished(self, key, state):
        async def expire():
            await asyncio.sleep(STREAM_IDLE_SECONDS)
            if self.routes.get(key) is state:
                self.routes.pop(key, None)
        task = asyncio.create_task(expire())
        self.cleanup_tasks.add(task)
        task.add_done_callback(self.cleanup_tasks.discard)

    async def close(self):
        for key in list(self.routes):
            await self.stop(key)
        for task in self.cleanup_tasks:
            task.cancel()
        await asyncio.gather(*self.cleanup_tasks, return_exceptions=True)

    def disconnect(self, socket):
        for state in self.routes.values():
            if state.socket is socket:
                state.socket = None
                state.attached.clear()

    async def handle(self, action, data, session, socket, sender):
        if action not in {OPEN, CREDIT, CANCEL}:
            return False
        route_id = str(data.get('route_id') or '')
        key = (session.user_id, route_id)
        state = self.routes.get(key)
        try:
            if not session.user_id or not session.auth_session_id:
                raise HTTPException(401, 'AUTH_REQUIRED')
            if action == CANCEL:
                if state is not None and state.socket is socket:
                    await self.stop(key)
                return True
            if action == OPEN:
                request = RouteRequest(**data.get('request', {}))
                if request.request_id != route_id or request.steps != 1:
                    raise ValueError('Invalid route identifier')
                fingerprint = hashlib.sha256(json.dumps(request.model_dump(), sort_keys=True).encode()).hexdigest()
                received = data.get('received', -1)
                if type(received) is not int or received < -1:
                    raise ValueError('Invalid receipt')
                if state is None:
                    if received != -1 or data.get('resume'):
                        raise HTTPException(409, 'STREAM_GONE')
                    # A socket has one active route. Other tabs keep their own subscription.
                    for old_key, old in list(self.routes.items()):
                        if old.socket is socket:
                            await self.stop(old_key)
                    active = [s for s in self.routes.values() if not s.task.done()]
                    limit = 32 if session.user_entitlement_tier == 'supporter' else 24
                    if (len(active) >= limit or sum(s.user_id == session.user_id for s in active) >= 2
                            or len(self.routes) >= 256):
                        raise HTTPException(429, 'TABLEBASE_BUSY')
                    descriptor = resolve_tablebase(request.full_pattern)
                    if (request.catalog_version != get_catalog_version() or descriptor is None
                            or not ai_table_metadata(descriptor)['compatible']
                            or abs(float(descriptor.get('spawn_rate', .1)) - request.spawn_rate4) >= .01):
                        raise HTTPException(409, 'CATALOG_CHANGED')
                    # Validate exact board/RNG ranges before a task can touch a reader.
                    GamerRouteCursor({name: getattr(request, name) for name in
                        ('board_codes', 'rng_state', 'difficulty', 'spawn_rate4', 'random_only', 'steps')},
                        ai_table_metadata(descriptor)['large_tiles'])
                    state = RouteSubscription(request, session.user_id, session.auth_session_id,
                        session.user_entitlement_tier == 'supporter', descriptor, socket, sender,
                        StreamWindow(data.get('allow_through', 7)), fingerprint)
                    self.routes[key] = state
                    state.attached.set()
                    state.task = asyncio.create_task(state.produce())
                    state.task.add_done_callback(lambda _: self.finished(key, state))
                else:
                    if state.fingerprint != fingerprint or state.session_id != session.auth_session_id:
                        raise HTTPException(409, 'REQUEST_ID_CONFLICT')
                    if received > state.window.produced or (state.frames and received < next(iter(state.frames)) - 1):
                        raise HTTPException(409, 'STREAM_GONE')
                    await state.check()
                    balance = await asyncio.to_thread(get_token_balance, state.user_id)
                    state.balance = balance
                    async with state.lock:
                        state.socket, state.sender = socket, sender
                        state.attached.set()
                        for seq, frame in state.frames.items():
                            if seq > received:
                                await sender(socket, {'action': EVENT, 'data': {'route_id': route_id, **frame, 'token_balance': balance}})
                        if state.end:
                            await sender(socket, {'action': EVENT, 'data': {'route_id': route_id, **state.end}})
            elif state is not None and state.socket is socket:
                consumed, allowed = data.get('consumed'), data.get('allow_through')
                state.window.update(consumed, allowed)
                for seq in list(state.frames):
                    if seq < state.window.consumed:
                        del state.frames[seq]
                try:
                    await state.replenish_remote()
                except Exception:
                    await self.stop(key)
                    raise HTTPException(503, 'TABLEBASE_UNAVAILABLE') from None
                await state.send({'type': 'window', 'allow_through': state.window.allow_through})
        except (ValueError, TypeError, ValidationError):
            await sender(socket, {'action': EVENT, 'data': {'route_id': route_id,
                'type': 'error', 'status': 400, 'detail': 'INVALID_STREAM_REQUEST'}})
        except HTTPException as exc:
            await sender(socket, {'action': EVENT, 'data': {'route_id': route_id,
                'type': 'error', 'status': exc.status_code, 'detail': exc.detail}})
        return True


gamer_stream_service = GamerStreamService()
