"""Four-variant player stream adapter. The Play game remains authoritative."""
from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import struct
import time
import urllib.error
import urllib.request

from fastapi import HTTPException, WebSocketDisconnect

from backend.human_play import codec, engine

from . import human_rooms


VIEW_STEP = struct.Struct("<IIB")
LIVE_TILE_VALUES = {str(2 ** power) for power in range(1, 28)}


def sanitize_best_score(value):
    if isinstance(value, bool):
        raise ValueError('invalid_live_best')
    return max(0, min(int(value or 0), 1_000_000_000))


def _live_color(value):
    value = value.lower() if isinstance(value, str) else ''
    if len(value) == 7 and value.startswith('#') and all(char in '0123456789abcdef' for char in value[1:]):
        return value
    raise ValueError('invalid_live_appearance')


def sanitize_appearance(value):
    if value is None:
        return None
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('invalid_live_appearance')
    empty = value.get('empty')
    tiles = value.get('tiles')
    if not isinstance(empty, dict) or not isinstance(tiles, dict) or len(tiles) > len(LIVE_TILE_VALUES):
        raise ValueError('invalid_live_appearance')
    normalized = {'version': 1, 'empty': {
        'background': _live_color(empty.get('background')),
        'color': _live_color(empty.get('color')),
    }, 'tiles': {}}
    for key, colors in tiles.items():
        if str(key) not in LIVE_TILE_VALUES or not isinstance(colors, dict):
            raise ValueError('invalid_live_appearance')
        normalized['tiles'][str(key)] = {
            'background': _live_color(colors.get('background')),
            'color': _live_color(colors.get('color')),
        }
    return normalized


class HumanLiveStore:
    """Small in-memory compatibility store; permanent replays remain on Play."""
    path = None

    def __init__(self):
        self.likes = 0

    def control(self, enabled=None):
        return {"enabled": True, "revision": 0}

    def summary(self, _stats_range="all"):
        empty = {"games": 0, "score_sum": 0, "tile32": 0, "tile64": 0,
                 "median_score": None, "stage32_rate": None}
        return {"history": [], "history_total": 0, "best": 0, "likes": self.likes,
                "week": {**empty, "start": "", "end": ""}, "all_time": empty,
                "today": empty, "stats_range": "all"}

    def history(self, page=1):
        return {"history": [], "total": 0, "page": 1, "pages": 1}

    def replay(self, _run_id):
        return None

    def save(self, _run):
        return None

    def finish(self, _run):
        return None

    def refresh_stats_snapshots(self, _ranges=("24h", "recent100", "all")):
        return 0

    def backfill_stats(self, **_kwargs):
        return 0

    def add_likes(self, count):
        self.likes += int(count or 0)


class HumanStreamRun:
    def __init__(self, *, run_id: str, variant: str, seed: str, started_at: float,
                 source: str, raw: bytes, appearance=None, best_score=0):
        if variant not in engine.VARIANTS or len(raw) % engine.EVENT.size:
            raise ValueError("invalid_human_stream")
        self.id = run_id
        self.variant = variant
        self.seed = seed
        self.started = float(started_at)
        self.source = source
        self.appearance = appearance
        self.best_score = sanitize_best_score(best_score)
        self.state = engine.advance(engine.initial(run_id, variant, seed), variant, raw)
        self.ended = None

    @property
    def seq(self):
        return int(self.state["seq"])

    def apply(self, raw: bytes) -> bytes:
        if len(raw) != engine.EVENT.size:
            raise ValueError("invalid_human_stream_step")
        code, delta = engine.EVENT.unpack(raw)
        self.state = engine.advance(self.state, self.variant, raw)
        self.best_score = max(self.best_score, sanitize_best_score(self.state.get("score")))
        return VIEW_STEP.pack(self.seq, delta, code)

    def snapshot(self):
        nodes = {key: int(value.get("elapsed", 0) if isinstance(value, dict) else value)
                 for key, value in self.state.get("nodes", {}).items()}
        rows, cols = engine.VARIANTS[self.variant]
        return {"run_id": self.id, "variant": self.variant, "rows": rows, "cols": cols,
                "board": self.state["board"], "score": self.state["score"],
                "seq": self.seq, "elapsed_ms": self.state["elapsed"], "nodes": nodes,
                "source": self.source, "started_at": self.started,
                "ended_at": self.ended, "restart_at": None,
                "appearance": self.appearance,
                "best_score": max(self.best_score, sanitize_best_score(self.state.get("score")))}


class HumanPlayContent:
    def __init__(self, room):
        self.room = room
        self.run = None
        self.store = HumanLiveStore()
        self.verified_milestones = set()
        self.pending_milestones = set()

    async def start(self):
        return None

    async def save(self):
        return None

    @property
    def run_id(self):
        return self.run.id if self.run else None

    def snapshot(self):
        return {"run": self.run.snapshot() if self.run else None}

    def reached_milestones(self):
        return {(self.run.id, value) for value in self.verified_milestones} if self.run else set()

    def _locally_reached(self):
        if not self.run:
            return set()
        return {value for value in (32768, 65536)
                if value in self.run.state['board'] or str(value) in self.run.state.get('nodes', {})}

    @staticmethod
    def _verify(run_id, milestone):
        base = os.environ.get('HUMAN_PLAY_INTERNAL_URL', 'http://127.0.0.1:8766').rstrip('/')
        request = urllib.request.Request(
            f'{base}/api/human/internal/live/runs/{run_id}/milestones/{milestone}',
            headers={'X-Human-Live-Internal': os.environ.get('HUMAN_LIVE_SIGNING_KEY', '')})
        with urllib.request.urlopen(request, timeout=4) as response:
            return response.status == 200 and bool(json.load(response).get('verified'))

    async def verify_pending_milestones(self):
        if not self.run:
            return
        self.pending_milestones |= self._locally_reached() - self.verified_milestones
        run = self.run
        for milestone in tuple(sorted(self.pending_milestones)):
            try:
                verified = await asyncio.to_thread(self._verify, run.id, milestone)
            except (OSError, TimeoutError, urllib.error.URLError, ValueError):
                continue
            if self.run is not run:
                return
            if verified:
                self.pending_milestones.discard(milestone)
                self.verified_milestones.add(milestone)

    async def _install_run(self, ws, hub, hello, user):
        lease = human_rooms.verify_lease(str(hello.get("lease") or ""),
            room_id=self.room.id, owner_user_id=user["id"])
        proposed_seq = hello.get("seq")
        if type(proposed_seq) is not int or not 0 <= proposed_seq <= engine.MAX_MOVES:
            raise ValueError("invalid_live_prefix")
        retained = self.run
        resume = (hello.get('resume') is True and retained is not None
                  and retained.id == lease['run'] and retained.variant == lease['variant']
                  and retained.seed == lease['seed'] and retained.seq <= proposed_seq)
        start = retained.seq if resume else 0
        if hello.get('resume') is True:
            await ws.send_json({'type': 'prefix_request', 'run_id': lease['run'],
                                'start': start, 'seq': proposed_seq})
        prefix_packet = await asyncio.wait_for(ws.receive_bytes(), 20)
        if len(prefix_packet) < 5 or prefix_packet[:4] != b"HLP1":
            raise ValueError("invalid_live_prefix")
        flags = prefix_packet[4]
        if flags & ~3:
            raise ValueError("invalid_live_prefix")
        encoded = prefix_packet[5:]
        encoding = "gzip" if flags & 1 else "identity"
        layout = "planes5" if flags & 2 else "interleaved"
        raw = await asyncio.to_thread(codec.decode_upload, encoded, encoding, layout)
        if proposed_seq - start != len(raw) // engine.EVENT.size:
            raise ValueError("invalid_live_prefix")
        if resume:
            run = copy.copy(retained)
            run.state = await asyncio.to_thread(engine.advance, copy.deepcopy(retained.state), run.variant, raw)
            run.appearance = sanitize_appearance(hello.get('appearance'))
            run.best_score = max(retained.best_score, sanitize_best_score(hello.get('best_score')),
                                 sanitize_best_score(run.state.get('score')))
        else:
            run = await asyncio.to_thread(HumanStreamRun,
                run_id=lease["run"], variant=lease["variant"], seed=lease["seed"],
                started_at=float(hello.get("started_at") or time.time()),
                source=self.room.metadata.get("streamer", {}).get("display_name", "Player"),
                raw=raw, appearance=sanitize_appearance(hello.get('appearance')),
                best_score=hello.get('best_score', 0),
            )
        human_rooms.verify_lease(str(hello.get("lease") or ""),
            room_id=self.room.id, owner_user_id=user["id"])
        if not await asyncio.to_thread(human_rooms.publisher_seen, self.room.id, lease["generation"]):
            raise ValueError("live_room_ended")
        self.run = run
        if not resume:
            self.verified_milestones.clear()
            self.pending_milestones.clear()
        hub.producer_ready = True
        hub.last_seen = time.monotonic()
        await ws.send_json({"type": "ready", "room_id": self.room.id,
                            "run_id": self.run.id, "seq": self.run.seq,
                            "generation": lease["generation"], "switch_supported": True})
        hub.broadcast(hub.snapshot())
        return lease

    async def publish(self, ws, hub, user=None):
        if hub.producer:
            await ws.close(code=1008)
            return
        hub.producer = ws
        hub.producer_ready = False
        hub.control_supported = False
        hub.control_ack = None
        hub.last_seen = time.monotonic()
        lease = None
        try:
            await ws.accept()
            hello_packet = await asyncio.wait_for(ws.receive_text(), 5)
            if len(hello_packet) > 16_384:
                raise ValueError("invalid_live_hello")
            hello = json.loads(hello_packet)
            if not isinstance(hello, dict) or hello.get("type") != "hello" or not user:
                raise ValueError("invalid_live_hello")
            lease = await self._install_run(ws, hub, hello, user)
            await hub.publisher_joined()
            last_seen_write = time.monotonic()
            rate_window_started = time.monotonic()
            rate_window_steps = 0
            while hub.producer is ws:
                if time.time() > float(lease["exp"]):
                    raise ValueError("live_lease_expired")
                packet = await asyncio.wait_for(ws.receive(), 20)
                if packet["type"] == "websocket.disconnect":
                    break
                hub.last_seen = time.monotonic()
                # Consume the signed switch before touching the previous generation's lease.
                if packet.get("text"):
                    if len(packet["text"]) > 16_384:
                        raise ValueError("invalid_live_message")
                    control = json.loads(packet["text"])
                    if not isinstance(control, dict):
                        raise ValueError("invalid_live_message")
                    if control.get("type") == "switch":
                        lease = await self._install_run(ws, hub, control, user)
                        last_seen_write = rate_window_started = time.monotonic()
                        rate_window_steps = 0
                        continue
                if time.monotonic() - last_seen_write >= 20:
                    if not await asyncio.to_thread(human_rooms.publisher_seen,
                            self.room.id, lease["generation"]):
                        current = await asyncio.to_thread(human_rooms.room, self.room.id)
                        if not current or current["owner_user_id"] != user["id"] or current["generation"] <= lease["generation"]:
                            raise ValueError("live_room_ended")
                    last_seen_write = time.monotonic()
                raw_step = packet.get("bytes")
                if raw_step is not None:
                    now = time.monotonic()
                    if now - rate_window_started >= 1:
                        rate_window_started, rate_window_steps = now, 0
                    rate_window_steps += 1
                    if rate_window_steps > 32:
                        raise ValueError("live_publish_rate_exceeded")
                    view_step = self.run.apply(raw_step)
                    self.pending_milestones |= self._locally_reached() - self.verified_milestones
                    hub.broadcast(view_step)
                    continue
                data = json.loads(packet.get("text") or "{}")
                if data.get("type") == "renew":
                    renewed = human_rooms.verify_lease(str(data.get("lease") or ""),
                        room_id=self.room.id, owner_user_id=user["id"])
                    if renewed["run"] != self.run.id or renewed["generation"] != lease["generation"]:
                        raise ValueError("live_switch_required")
                    lease = renewed
                elif data.get("type") == "end":
                    self.run.ended = time.time()
                    hub.broadcast(hub.snapshot())
                elif data.get("type") == "appearance":
                    self.run.appearance = sanitize_appearance(data.get('appearance'))
                    hub.broadcast({'type': 'appearance', 'appearance': self.run.appearance})
                elif data.get("type") == "best":
                    self.run.best_score = max(self.run.best_score,
                                              sanitize_best_score(self.run.state.get('score')),
                                              sanitize_best_score(data.get('best_score')))
                    hub.broadcast({'type': 'best', 'best_score': self.run.best_score})
                elif data.get("type") != "ping":
                    raise ValueError("invalid_live_message")
        except (WebSocketDisconnect, asyncio.TimeoutError, ValueError, KeyError,
                TypeError, HTTPException):
            with contextlib.suppress(Exception):
                await ws.close(code=1008)
        finally:
            if hub.producer is ws:
                hub.producer = None
                hub.producer_ready = False
                hub.broadcast(hub.snapshot())
