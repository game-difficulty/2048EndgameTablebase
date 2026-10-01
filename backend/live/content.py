"""Content adapters own game protocol and storage, never wallets or room identity."""
import asyncio
import contextlib
import json
import time
from fastapi import WebSocketDisconnect
from .protocol import LiveRun, STEP
from .store import LiveStore


class ClassicContent:
    def __init__(self, room):
        self.room = room
        self.run = None
        self.store = None

    async def start(self):
        self.store = await asyncio.to_thread(LiveStore, self.room.store_path())
        saved = await asyncio.to_thread(self.store.load)
        if saved:
            self.run = await asyncio.to_thread(LiveRun.restore, saved)

    async def save(self):
        if self.run:
            await asyncio.to_thread(self.store.save, self.run)

    @property
    def run_id(self):
        return self.run.id if self.run else None

    def snapshot(self):
        return {'run': self.run.snapshot() if self.run else None}

    def reached_milestones(self):
        return {(self.run.id, value) for value in (32768, 65536)
                if str(value) in self.run.nodes} if self.run else set()

    async def publish(self, ws, hub):
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
                    await hub.publisher_joined()
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


from .multi_content import MultiAiContent
from .human_content import HumanPlayContent
from .competition_content import CompetitionMatchContent

CONTENT_FACTORIES = {('classic-ai', 'classic-step-v1'): ClassicContent,
                     ('classic-multi-ai', 'classic-multi-v1'): MultiAiContent,
                     ('human-play', 'human-play-v1'): HumanPlayContent,
                     ('competition-match', 'competition-match-v1'): CompetitionMatchContent}


def create_content(room):
    try:
        return CONTENT_FACTORIES[(room.content_kind, room.protocol)](room)
    except KeyError:
        raise ValueError('unsupported_room_content') from None
