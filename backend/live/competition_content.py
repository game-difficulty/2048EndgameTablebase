"""Read-only competition content. Competition Service remains authoritative."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from copy import deepcopy
from datetime import datetime

from .dynamic_rooms import competition_provider
from .human_content import HumanLiveStore
from backend.stream_snapshots import compact_public_view
from backend.projection_delta import ProjectionDecoder


def _source_time(projection):
    try:
        value = datetime.fromisoformat(projection.get('server_time', ''))
        return value.timestamp() if value.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


class CompetitionMatchContent:
    push_stream = True
    def __init__(self, room):
        self.room = room
        self.store = HumanLiveStore()
        self.run = None
        # The subscription supplies the only bootstrap. Never synchronously fetch
        # a duplicate 128-frame HTTP projection while resolving a room.
        self.projection = None
        self._clock_anchor = None
        self._clock_sample = None
        self._clock_at = time.monotonic()
        self._observe_clock(self.projection)
        self.online = self.projection is not None
        self.incremental_projection = self.projection
        self.stream_task = None
        self.on_update = lambda: None

    def _is_valid(self, projection):
        if not isinstance(projection, dict):
            return False
        try:
            generation_matches = (
                int(projection.get('generation', 0))
                == int(self.room.metadata.get('generation', 0))
            )
        except (TypeError, ValueError):
            return False
        sequence = projection.get('content_sequence')
        return (
            projection.get('match_public_key') == self.room.metadata.get('public_key')
            and generation_matches
            and isinstance(sequence, int)
            and not isinstance(sequence, bool)
        )

    async def start(self):
        self.stream_task = asyncio.create_task(self._subscribe())

    async def stop(self):
        if self.stream_task:
            self.stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.stream_task
            self.stream_task = None

    async def _subscribe(self):
        from websockets.legacy.client import connect
        origin = os.environ.get('COMPETITION_LIVE_API_ORIGIN', '').rstrip('/')
        token = os.environ.get('COMPETITION_LIVE_INTERNAL_TOKEN', '')
        if not origin or not token:
            return
        url = origin.replace('https://', 'wss://').replace('http://', 'ws://')
        url += '/ws/internal/live/' + self.room.metadata['public_key'] + '?transport=delta-v1'
        delay = .5
        while True:
            try:
                async with connect(url, extra_headers={'X-Competition-Live-Token': token},
                                   max_size=2*1024*1024, open_timeout=5, ping_interval=15, ping_timeout=15) as socket:
                    delay = .5
                    decoder = ProjectionDecoder()
                    initial = True
                    while True:
                        message = json.loads(await asyncio.wait_for(socket.recv(), 35))
                        fresh = decoder.decode(message)
                        if fresh is None:
                            continue
                        if not self._is_valid(fresh):
                            raise ValueError('invalid_projection')
                        if initial:
                            # Also bound legacy-server bootstraps during rolling upgrades.
                            fresh = {**fresh, 'project_public_views': {s: compact_public_view(v)
                                for s, v in (fresh.get('project_public_views') or {}).items()}}
                            initial = False
                        if self.accept_projection(fresh):
                            self.on_update()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if self.online:
                    self.online = False
                    self.on_update()
                logging.getLogger(__name__).warning('competition relay reconnect room=%s error=%s', self.room.id, type(error).__name__)
                await asyncio.sleep(delay)
                delay = min(5, delay * 2)

    async def save(self):
        return None

    @property
    def run_id(self):
        return self.room.metadata.get('public_key')

    def snapshot(self, *, incremental=False):
        # Keep a small recovery window for short disconnects, not the entire
        # retained replay. Ordinary deltas bypass this bootstrap compaction.
        projection = self.incremental_projection if incremental else self.projection
        if not incremental and projection is not None and 'project_public_views' in projection:
            projection = {**projection, 'project_public_views': {
                side: compact_public_view(view)
                for side, view in projection['project_public_views'].items()
            }}
        result = {'match': projection}
        if self._clock_anchor is not None:
            # Cached projections retain their original time/remaining-ms pair. Give
            # new viewers a current source-domain clock, never the live host's wall clock.
            result['competition_server_time'] = self._clock_now()
        return result

    def watermark(self):
        return {key: (self.projection or {}).get(key) for key in
                ('match_public_key', 'generation', 'content_sequence')}

    def _clock_now(self):
        if self._clock_anchor is None:
            return None
        return self._clock_anchor + max(0, time.monotonic() - self._clock_at)

    def _observe_clock(self, projection):
        sample = _source_time(projection or {})
        if sample is None or (self._clock_sample is not None and sample <= self._clock_sample):
            return
        current = self._clock_now()
        self._clock_anchor = sample if current is None else max(current, sample)
        self._clock_sample = sample
        self._clock_at = time.monotonic()

    def reached_milestones(self):
        return set()

    async def refresh(self):
        fresh = await asyncio.to_thread(
            competition_provider.projection,
            self.room.metadata.get('public_key', ''),
            {side: int(view.get('sequence', 0)) for side, view in
             ((self.projection or {}).get('project_public_views') or {}).items()},
        )
        return self.accept_projection(fresh)

    def accept_projection(self, fresh):
        previous = self.projection
        was_online = self.online
        if not self._is_valid(fresh):
            # Keep the last verified full projection on a transient source outage.
            # Live presence still tells viewers that the feed is reconnecting.
            self.online = False
            return was_online
        self.online = True
        if previous is None:
            self._observe_clock(fresh)
            self.projection = fresh
            self.incremental_projection = fresh
            return True
        previous_sequence = int(previous.get('content_sequence', 0))
        fresh_sequence = int(fresh.get('content_sequence', 0))
        previous_time, fresh_time = _source_time(previous), _source_time(fresh)
        if fresh_sequence < previous_sequence or (
            fresh_sequence == previous_sequence and previous_time is not None
            and (fresh_time is None or fresh_time < previous_time)
        ):
            # Polls can complete out of order; a late response must not rewind a match.
            return was_online != self.online
        self._observe_clock(fresh)
        self.incremental_projection = fresh
        retained = deepcopy(fresh)
        same_game = previous.get('current_game') == fresh.get('current_game')
        for side, view in (retained.get('project_public_views') or {}).items():
            if view is None:
                continue
            old = ((previous.get('project_public_views') or {}).get(side) or {}) if same_game else {}
            floor = int(view.get('frame_start', 0))
            frames = {frame['sequence']: frame for frame in old.get('frames', [])
                      if frame['sequence'] >= floor}
            frames.update({frame['sequence']: frame for frame in view.get('frames', [])})
            view['frames'] = [frames[key] for key in sorted(frames)[-128:]]
        self.projection = retained
        return fresh_sequence != previous_sequence or was_online != self.online

    async def publish(self, ws, _hub):
        await ws.close(code=1008)
