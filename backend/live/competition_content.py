"""Read-only competition content. Competition Service remains authoritative."""
from __future__ import annotations

import asyncio
from copy import deepcopy

from .dynamic_rooms import competition_provider
from .human_content import HumanLiveStore


class CompetitionMatchContent:
    def __init__(self, room):
        self.room = room
        self.store = HumanLiveStore()
        self.run = None
        self.projection = competition_provider.projection(
            room.metadata.get('public_key', '')
        )
        if not self._is_valid(self.projection):
            self.projection = None
        self.online = self.projection is not None
        self.incremental_projection = self.projection

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
        return None

    async def save(self):
        return None

    @property
    def run_id(self):
        return self.room.metadata.get('public_key')

    def snapshot(self):
        return {'match': self.projection}

    def reached_milestones(self):
        return set()

    async def refresh(self):
        fresh = await asyncio.to_thread(
            competition_provider.projection,
            self.room.metadata.get('public_key', ''),
            {side: int(view.get('sequence', 0)) for side, view in
             ((self.projection or {}).get('project_public_views') or {}).items()},
        )
        previous = self.projection
        was_online = self.online
        if not self._is_valid(fresh):
            # Keep the last verified full projection on a transient source outage.
            # Live presence still tells viewers that the feed is reconnecting.
            self.online = False
            return was_online
        self.online = True
        if previous is None:
            self.projection = fresh
            self.incremental_projection = fresh
            return True
        previous_sequence = int(previous.get('content_sequence', 0))
        fresh_sequence = int(fresh.get('content_sequence', 0))
        if fresh_sequence < previous_sequence:
            # Polls can complete out of order; a late response must not rewind a match.
            return was_online != self.online
        self.incremental_projection = fresh
        retained = deepcopy(fresh)
        same_game = previous.get('current_game') == fresh.get('current_game')
        for side, view in (retained.get('project_public_views') or {}).items():
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
