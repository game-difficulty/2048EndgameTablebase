"""Three independent games, one publisher, one ordered spectator stream."""
import asyncio
import contextlib
import json
import logging
import time
import uuid
from copy import deepcopy
from fastapi import WebSocketDisconnect
from backend.gamer_ranked.rules import legal_moves
from .protocol import LiveRun, STEP
from .multi_protocol import PROTOCOL, PUBLISH_PROTOCOL, encode_step, encode_source, pack_batch, unpack_uplink
from .multi_store import MultiLiveStore


class MultiAiContent:
    def __init__(self, room):
        self.room, self.store = room, None
        self.runs = [None] * 3
        self.generations = [0] * 3
        self.statuses = ['recovering'] * 3
        self.epoch = str(uuid.uuid4())
        self.seq = 0
        self.sources = ['AI']
        self.pending = []
        self.pending_first = 0
        self.timer = None
        self.hub = None
        self.batch = None
        self.save_lock = asyncio.Lock()
        self.activity_events = []
        self.activity_dirty = False

    @staticmethod
    def participants():
        return [dict(id=name.lower(), name=name, lane=i) for i, name in enumerate(('Lume','Clari','Vero'))]

    @property
    def run_id(self):
        return None  # No room-wide main screen; selection belongs to each viewer.

    async def start(self):
        self.store = await asyncio.to_thread(MultiLiveStore, self.room.store_path())
        for slot in await asyncio.to_thread(self.store.load_slots):
            lane = slot['lane']
            if lane not in range(3):
                raise ValueError('invalid_saved_lane')
            self.generations[lane] = slot['generation']
            if slot['run']:
                self.runs[lane] = await asyncio.to_thread(LiveRun.restore, slot['run'])
                run = self.runs[lane]
                self.statuses[lane] = 'ended' if run.ended else 'running'
                if run.source not in self.sources:
                    self.sources.append(run.source)
                if run.ended:
                    await asyncio.to_thread(self.store.finish_lane, run, lane)
        self.batch = await asyncio.to_thread(self.store.load_batch)
        if not self.batch and any(self.runs):
            # Existing independent games finish together, without opening a market.
            for lane, run in enumerate(self.runs):
                if run is None:
                    self.runs[lane] = LiveRun()
                    self.generations[lane] += 1
                    self.statuses[lane] = 'running'
            self.batch = dict(id=str(uuid.uuid4()), phase='running', transition=True,
                              started_at=time.time(), deadline=0, participants=self.participants(), milestones={})
            await self.save()

    async def save(self):
        if self.store:
            async with self.save_lock:
                slots = [dict(lane=i, generation=self.generations[i], run=r.checkpoint() if r else None)
                         for i, r in enumerate(self.runs)]
                events = self.activity_events[:]
                await asyncio.to_thread(self.store.save_slots, slots, deepcopy(self.batch), events)
                del self.activity_events[:len(events)]

    async def advance_batch(self, force=False):
        """Called under the room activity lock before admitting bets or publishing results."""
        if not self.batch:
            return
        batch = self.batch
        changed = force or self.activity_dirty
        # Every player's target outcome is independent of the first-to-milestone giveaway.
        # Persist these facts with checkpoints before allowing the activity ledger to use them.
        if not batch.get('transition') and batch['phase'] != 'void':
            targets = batch.setdefault('target_results', {})
            for player, run in zip(batch['participants'], self.runs):
                previous = targets.get(player['id'])
                if not run or previous in ('missed', 'reached_combo'):
                    continue
                combo = '65536+32768' in run.nodes or (65536 in run.board and 32768 in run.board)
                reached = combo or previous == 'reached' or '65536' in run.nodes or max(run.board) >= 65536
                if reached or run.ended or (0 not in run.board and not legal_moves(run.board)):
                    outcome = 'reached_combo' if combo else 'reached' if reached else 'missed'
                    if outcome != previous:
                        targets[player['id']] = outcome
                        changed = self.activity_dirty = True
        if batch['phase'] == 'running':
            ended = [r for r in self.runs if r and r.ended]
            # The terminal board itself reveals the result, even before the worker's end message.
            playing = [r for r in self.runs if r and not r.ended and (0 in r.board or legal_moves(r.board))]
            stopped = [r for r in self.runs if r and r not in playing]
            certain = len(stopped) == 3 or (len(playing) == 1 and len(stopped) == 2 and playing[0].score > max(r.score for r in stopped))
            if (certain or batch['deadline'] <= time.time()) and not batch.get('closed'):
                batch['closed'] = True
                changed = True
                self.activity_dirty = True
                await self.save()
            if len(ended) == 3:
                highest = max(r.score for r in ended)
                batch.update(phase='settling', closed=True,
                    winners=[p['id'] for p,r in zip(batch['participants'],self.runs) if r.score == highest],
                    scores={p['id']:r.score for p,r in zip(batch['participants'],self.runs)})
                await self.save()
        activities = getattr(self.hub, 'activities', None)
        if activities and (changed or batch['phase'] in ('settling','void')):
            await self.save()
            await activities.reconcile(batch)
            self.activity_dirty = False
        if batch['phase'] == 'settling':
            batch.update(phase='cooldown', next_at=time.time()+15)
            await self.save()
            if self.hub:
                self.event('batch', batch=deepcopy(batch))

    async def record_milestones(self, lane):
        if not self.batch or self.batch.get('transition') or not self.room.milestone_rewards:
            return
        run = self.runs[lane]
        for value in (32768,65536):
            if str(value) in run.nodes and str(value) not in self.batch['milestones']:
                fact = dict(trigger_key=f"{self.batch['id']}:{value}", batch_id=self.batch['id'], milestone=value,
                            participant_id=self.batch['participants'][lane]['id'], run_id=run.id,
                            content_seq=self.seq+1, created=time.time())
                self.batch['milestones'][str(value)] = fact
                self.activity_events.append(fact)
                await self.save()

    async def begin_batch(self, data):
        if not self.hub.control['enabled']:
            raise ValueError('publisher_paused')
        if self.batch and (self.batch['phase'] not in ('cooldown','void') or time.time()<self.batch.get('next_at',0)):
            raise ValueError('batch_not_finished')
        batch_id = str(uuid.UUID(data['batch_id']))
        if self.batch and batch_id == self.batch['id']:
            raise ValueError('duplicate_batch')
        proposed = data.get('lanes', [])
        if len(proposed) != 3:
            raise ValueError('invalid_batch')
        started = time.time()
        runs = []
        for lane, slot in enumerate(proposed):
            if slot.get('lane') != lane or slot.get('generation') != self.generations[lane]+1 or slot['generation']>0xffffffff:
                raise ValueError('invalid_generation')
            runs.append(LiveRun(slot['seed'],slot['run_id'],started))
        if len({r.id for r in runs}) != 3 or any(r.id in {s.id for s in self.runs if s} for r in runs):
            raise ValueError('duplicate_run')
        self.flush()
        self.runs, self.generations, self.statuses = runs, [s['generation'] for s in proposed], ['running']*3
        self.batch = dict(id=batch_id, phase='running', started_at=started, deadline=started+600,
                          participants=self.participants(), milestones={}, closed=False)
        self.activity_dirty = True
        await self.save()  # Durable fact first; auth DB reconciles idempotently after a crash.
        await self.advance_batch(force=True)
        self.hub.broadcast(self.hub.snapshot())

    async def void_batch(self):
        if not self.batch or self.batch['phase'] in ('cooldown','void'):
            return
        self.batch.update(phase='void', closed=True, next_at=time.time()+15)
        await self.save()
        await self.advance_batch()
        self.hub.broadcast(self.hub.snapshot())

    def control_lanes(self):
        return [dict(lane=i, name=self.participants()[i]['name'], run_id=r.id if r else None, seq=r.seq if r else 0,
                     status=self.statuses[i], score=r.score if r else 0)
                for i, r in enumerate(self.runs)]

    def slot_snapshot(self, lane):
        run = self.runs[lane]
        state = run.snapshot() if run else None
        if state:
            state.pop('nodes', None)
            state['source_id'] = self.sources.index(run.source)
        return dict(lane=lane, participant_id=self.participants()[lane]['id'], name=self.participants()[lane]['name'],
                    generation=self.generations[lane], status=self.statuses[lane], run=state)

    def snapshot(self):
        return dict(stream_epoch=self.epoch, content_seq=self.seq, batch=deepcopy(self.batch),
                    sources={str(i): name for i, name in enumerate(self.sources)},
                    lanes=[self.slot_snapshot(i) for i in range(3)])

    def reached_milestones(self):
        if self.batch:
            return set()  # Batch facts are delivered by the durable room activity outbox.
        return {(r.id, value) for r in self.runs if r for value in (32768, 65536) if str(value) in r.nodes}

    def flush(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None
        if self.pending:
            records, self.pending = self.pending, []
            self.hub.broadcast(pack_batch(self.pending_first, records))

    def ensure_capacity(self):
        if self.seq >= 0xffffffff - 128:
            self.flush()
            self.epoch, self.seq = str(uuid.uuid4()), 0
            self.hub.broadcast(self.hub.snapshot())

    def next_seq(self):
        self.seq += 1
        return self.seq

    def append(self, record):
        seq = self.next_seq()
        if not self.pending:
            self.pending_first = seq
            self.timer = asyncio.get_running_loop().call_later(.05, self.flush)
        self.pending.append(record)
        if len(self.pending) >= 64:
            self.flush()

    def event(self, action, **data):
        self.flush()
        self.hub.broadcast(dict(type=action, stream_epoch=self.epoch, content_seq=self.next_seq(), **data))

    def source(self, lane, name):
        name = str(name)[:80]
        if name not in self.sources:
            if len(self.sources) >= 65536:
                raise ValueError('too_many_sources')
            self.sources.append(name)
            self.event('dictionary', sources={str(len(self.sources) - 1): name})
        run = self.runs[lane]
        if run.source != name:
            run.source = name
            self.append(encode_source(lane, self.sources.index(name)))

    def require_lane(self, data):
        lane = data.get('lane')
        if type(lane) is not int or lane not in range(3):
            raise ValueError('invalid_lane')
        if data.get('generation') != self.generations[lane]:
            raise ValueError('stale_generation')
        return lane

    async def sync_lane(self, data):
        lane = data['lane']
        proposed = data.get('run')
        generation = data.get('generation', 0)
        current = self.runs[lane]
        if proposed:
            restored = await asyncio.to_thread(LiveRun.restore, proposed)
            if not current:
                if type(generation) is not int or not 1 <= generation <= 0xffffffff:
                    raise ValueError('invalid_generation')
                self.generations[lane] = generation
                current = restored
            elif generation == self.generations[lane] and current.id == restored.id:
                if current.seed != restored.seed:
                    raise ValueError('seed_conflict')
                a, b = bytes(current.records), bytes(restored.records)
                if not a.startswith(b) and not b.startswith(a):
                    raise ValueError('record_conflict')
                if not self.batch and (len(b) > len(a) or (len(b) == len(a) and restored.ended and not current.ended)):
                    current = restored
        self.runs[lane] = current
        if current:
            if current.source not in self.sources:
                self.sources.append(current.source)
            self.statuses[lane] = 'ended' if current.ended else 'running'
            if current.ended:
                await asyncio.to_thread(self.store.finish_lane, current, lane)
        return dict(type='resume_lane', lane=lane, generation=self.generations[lane],
                    run=current.checkpoint() if current else None)

    async def publish(self, ws, hub):
        if hub.producer:
            await ws.close(code=1008)
            return
        self.hub = hub
        hub.producer, hub.producer_ready = ws, False
        hub.control_supported, hub.control_ack = False, None
        hub.last_seen = time.monotonic()
        phase, synced = 'hello', 0
        sync_deadline = time.monotonic() + 180
        try:
            await ws.accept()
            while hub.producer is ws:
                remaining = max(.01, sync_deadline - time.monotonic()) if phase != 'ready' else 20
                packet = await asyncio.wait_for(ws.receive(), remaining)
                if packet['type'] == 'websocket.disconnect':
                    break
                hub.last_seen = time.monotonic()
                self.ensure_capacity()
                raw = packet.get('bytes')
                if raw is not None:
                    if phase != 'ready':
                        raise ValueError('not_ready')
                    if not hub.control['enabled'] and hub.control_ack == hub.control['revision']:
                        raise ValueError('publisher_paused')
                    async with getattr(hub, 'activity_lock', asyncio.Lock()):
                        if self.batch and self.batch['phase'] != 'running':
                            raise ValueError('batch_closed')
                        for lane, generation, step in unpack_uplink(raw):
                            if self.generations[lane] != generation or not self.runs[lane]:
                                raise ValueError('stale_generation')
                            self.runs[lane].apply(step)
                            await self.record_milestones(lane)
                            await self.advance_batch()
                            self.append(encode_step(lane, step))
                    continue
                text = packet.get('text') or ''
                if len(text) > 710_000:
                    raise ValueError('message_too_large')
                data = json.loads(text)
                action = data.get('type')
                if action == 'ping':
                    continue
                if phase == 'hello':
                    if action != 'hello' or data.get('protocol') != PUBLISH_PROTOCOL or data.get('control_version') != 1:
                        raise ValueError('protocol_mismatch')
                    await hub.publisher_joined()
                    hub.control_supported = True
                    await ws.send_json(dict(type='sync', lanes=3, control=hub.control))
                    phase = 'sync'
                elif phase == 'sync':
                    if action != 'sync_lane' or type(data.get('lane')) is not int or data['lane'] != synced:
                        raise ValueError('invalid_sync_order')
                    await ws.send_json(await self.sync_lane(data))
                    synced += 1
                    if synced == 3:
                        if not self.batch and all(self.runs):
                            self.batch = dict(id=str(uuid.uuid4()),phase='running',transition=True,
                                started_at=time.time(),deadline=0,participants=self.participants(),milestones={})
                        await self.save()
                        async with getattr(hub, 'activity_lock', asyncio.Lock()):
                            await self.advance_batch(force=True)
                        phase = 'ready'
                        hub.producer_ready = True
                        await ws.send_json(dict(type='ready', control=hub.control, batch=self.batch))
                        hub.broadcast(hub.snapshot())
                elif action == 'control_ack':
                    if data.get('revision') == hub.control['revision']:
                        hub.control_ack = data['revision']
                        hub.broadcast(hub.snapshot())
                elif action == 'batch_start':
                    async with hub.activity_lock:
                        await self.begin_batch(data)
                        await ws.send_json(dict(type='batch_ready', batch=self.batch))
                elif action == 'batch_status':
                    async with hub.activity_lock:
                        await self.advance_batch()
                        await ws.send_json(dict(type='batch_state', batch=self.batch))
                elif action == 'start':
                    if self.batch:
                        raise ValueError('batch_start_required')
                    lane = data.get('lane')
                    if type(lane) is not int or lane not in range(3):
                        raise ValueError('invalid_lane')
                    if not hub.control['enabled']:
                        raise ValueError('publisher_paused')
                    if self.runs[lane] and not self.runs[lane].ended:
                        raise ValueError('run_not_finished')
                    if data.get('generation') != self.generations[lane] + 1 or data['generation'] > 0xffffffff:
                        raise ValueError('invalid_generation')
                    self.flush()
                    self.runs[lane] = LiveRun(data['seed'], data['run_id'])
                    self.generations[lane] = data['generation']
                    self.statuses[lane] = 'running'
                    self.event('lane_start', slot=self.slot_snapshot(lane))
                    await self.save()
                elif action in ('source', 'end', 'lane_status'):
                    lane = self.require_lane(data)
                    run = self.runs[lane]
                    if not run:
                        raise ValueError('run_missing')
                    if action == 'source':
                        self.source(lane, data['source'])
                    elif action == 'lane_status':
                        status = data.get('status')
                        if status not in ('running', 'recovering') or run.ended:
                            raise ValueError('invalid_status')
                        self.statuses[lane] = status
                        self.event('lane_status', lane=lane, generation=self.generations[lane], status=status)
                    else:
                        if not hub.control['enabled'] and hub.control_ack == hub.control['revision']:
                            raise ValueError('publisher_paused')
                        async with getattr(hub, 'activity_lock', asyncio.Lock()):
                            if self.batch and self.batch['phase'] != 'running':
                                if run.ended:
                                    continue  # Reconnect can race a final end acknowledgement.
                                raise ValueError('batch_closed')
                            run.end()
                            self.statuses[lane] = 'ended'
                            await self.save()
                            await self.advance_batch()
                            self.event('lane_end', slot=self.slot_snapshot(lane))
                        await asyncio.to_thread(self.store.finish_lane, run, lane)
                        await self.save()
                        hub.broadcast({**await asyncio.to_thread(self.store.summary), 'type': 'summary', 'likes': hub.like_total})
                else:
                    raise ValueError('invalid_message')
        except (WebSocketDisconnect, asyncio.TimeoutError, ValueError, KeyError, TypeError, OverflowError) as error:
            logging.getLogger(__name__).warning('Multi publisher closed: %s (%s)', type(error).__name__, str(error)[:160])
            with contextlib.suppress(Exception):
                await ws.close(code=1008)
        finally:
            if hub.producer is ws:
                self.flush()
                hub.producer, hub.producer_ready, hub.control_ack = None, False, None
                hub.broadcast(hub.snapshot())
                await self.save()
