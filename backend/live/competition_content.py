"""Read-only competition content. Competition Service remains authoritative."""
from __future__ import annotations

import asyncio

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
            return True
        previous_sequence = int(previous.get('content_sequence', 0))
        fresh_sequence = int(fresh.get('content_sequence', 0))
        if fresh_sequence < previous_sequence:
            # Polls can complete out of order; a late response must not rewind a match.
            return was_online != self.online
        self.projection = fresh
        return fresh_sequence != previous_sequence or was_online != self.online

    async def publish(self, ws, _hub):
        await ws.close(code=1008)
