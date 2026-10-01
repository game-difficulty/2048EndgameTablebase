"""Registered dynamic room sources. Providers never share their source database."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol

from . import human_rooms
from .rooms import RoomDefinition


def _human_definition(data):
    name = data.get('display_name') or 'Player'
    return RoomDefinition(
        id=data['room_id'], title={'zh': f'{name} 的直播', 'en': f"{name}'s stream"},
        description={'zh': f"{data['variant']} 玩家实时对局",
                     'en': f"Live {data['variant']} player game"},
        content_kind='human-play', protocol='human-play-v1',
        milestone_rewards=True, dynamic=True,
        metadata={'variant': data['variant'], 'owner_user_id': data['owner_user_id'],
                  'run_id': data['run_id'], 'generation': int(data['generation']),
                  'streamer': {'display_name': name, 'avatar_url': data.get('avatar_url')},
                  'category_label': {'zh': '玩家直播', 'en': 'PLAYER'},
                  'badge': {'zh': '直播中', 'en': 'LIVE'},
                  'subtitle': {'zh': f"{data['variant']} 实时对局", 'en': f"Live {data['variant']} game"},
                  'started_at': data['started_at'], 'provider': 'human-play'},
    )


class DynamicRoomProvider(Protocol):
    name: str

    def list_active_rooms(self) -> list[RoomDefinition]: ...
    def resolve_room(self, room_id: str) -> RoomDefinition | None: ...


class HumanPlayRoomProvider:
    name = 'human-play'

    def list_active_rooms(self):
        return [_human_definition(item) for item in human_rooms.active_rooms()]

    def resolve_room(self, room_id):
        if room_id.startswith('competition-'):
            return None
        data = human_rooms.room(room_id)
        return _human_definition(data) if data else None


class CompetitionMatchRoomProvider:
    name = 'competition-match'

    def __init__(self):
        self._directory_cache = []

    @staticmethod
    def _request(path):
        origin = os.environ.get('COMPETITION_LIVE_API_ORIGIN', '').rstrip('/')
        token = os.environ.get('COMPETITION_LIVE_INTERNAL_TOKEN', '')
        if not origin or not token:
            return None
        request = urllib.request.Request(
            origin + path,
            headers={'X-Competition-Live-Token': token, 'Accept': 'application/json'},
        )
        try:
            with urllib.request.urlopen(request, timeout=2.0) as response:
                return json.loads(response.read(2_000_000))
        except (OSError, ValueError, urllib.error.HTTPError):
            return None

    @staticmethod
    def _definition(data):
        return RoomDefinition(
            id=data['room_id'], title=data['title'],
            description=data.get('subtitle') or {},
            content_kind='competition-match', protocol='competition-match-v1',
            dynamic=True,
            capabilities={'gifts': True, 'red_envelopes': True,
                          'lucky_bags': False, 'predictions': True,
                          'statistics': False},
            metadata={
                'public_key': data['public_key'],
                'generation': int(data['generation']),
                'content_sequence': int(data.get('content_sequence', 0)),
                'category_label': data.get('category_label'),
                'badge': data.get('badge'),
                'subtitle': data.get('subtitle'),
                'preview': data.get('preview'),
                'started_at': data.get('started_at'),
                'expires_at': data.get('expires_at'),
                'provider': 'competition-match',
            },
        )

    def list_active_rooms(self):
        payload = self._request('/api/internal/live/rooms')
        if payload is None:
            self._directory_cache = [room for room in self._directory_cache if self._unexpired(room)]
            return list(self._directory_cache)
        definitions = []
        for item in payload.get('rooms') or []:
            try:
                definition = self._definition(item)
                if self._unexpired(definition):
                    definitions.append(definition)
            except (KeyError, TypeError, ValueError):
                continue
        self._directory_cache = definitions
        return list(definitions)

    @staticmethod
    def _unexpired(room):
        expires = room.metadata.get('expires_at')
        return expires is None or time.time() < float(expires)

    def resolve_room(self, room_id):
        if not room_id.startswith('competition-'):
            return None
        for definition in self._directory_cache:
            if definition.id == room_id and self._unexpired(definition):
                return definition
        for definition in self.list_active_rooms():
            if definition.id == room_id:
                return definition
        # The directory is authoritative for retention/visibility; do not create
        # a room merely because a projection endpoint still responds.
        return None

    def projection(self, public_key, after=None):
        cursors = after or {}
        query = '&'.join(f'after_{side}={int(cursors.get(side, 0))}' for side in ('yellow', 'white'))
        payload = self._request(f'/api/internal/live/rooms/{public_key}?{query}') or {}
        return payload.get('projection')

    def settlement(self, public_key):
        payload = self._request(f'/api/internal/live/settlement/{public_key}') or {}
        return payload.get('projection')


class DynamicRoomRegistry:
    def __init__(self):
        self.providers: list[DynamicRoomProvider] = []

    def register(self, provider):
        if any(item.name == provider.name for item in self.providers):
            raise ValueError('duplicate_dynamic_room_provider')
        self.providers.append(provider)

    def list_active_rooms(self):
        rooms = {}
        for provider in self.providers:
            for definition in provider.list_active_rooms():
                if definition.id in rooms:
                    raise ValueError('duplicate_dynamic_room')
                rooms[definition.id] = definition
        return list(rooms.values())

    def resolve_room(self, room_id):
        for provider in self.providers:
            definition = provider.resolve_room(room_id)
            if definition is not None:
                return definition
        return None


competition_provider = CompetitionMatchRoomProvider()
dynamic_room_registry = DynamicRoomRegistry()
dynamic_room_registry.register(HumanPlayRoomProvider())
dynamic_room_registry.register(competition_provider)
