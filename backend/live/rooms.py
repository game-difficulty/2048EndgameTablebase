"""Explicit room registration. Unknown URLs never allocate a runtime or storage."""
from dataclasses import dataclass, field
from pathlib import Path
import os
import re

DEFAULT_ROOM_ID = 'ai-classic'


@dataclass(frozen=True)
class RoomDefinition:
    id: str
    title: dict
    description: dict = field(default_factory=dict)
    content_kind: str = 'classic-ai'
    protocol: str = 'classic-step-v1'
    publish_token_env: str = ''
    milestone_rewards: bool = False
    dynamic: bool = False
    capabilities: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}', self.id):
            raise ValueError('invalid_room_id')

    @property
    def target(self):
        return 'live:' + self.id

    def store_path(self):
        base = Path(os.environ.get('LIVE_DB_PATH', 'data/live.sqlite3'))
        return base if self.id == DEFAULT_ROOM_ID else base.parent / 'rooms' / (self.id + '.sqlite3')

    def public(self):
        defaults = dict(chat=True, likes=True, music=True, pip=True,
                        gifts=True, red_envelopes=True,
                        lucky_bags=self.milestone_rewards,
                        predictions=self.content_kind=='classic-multi-ai',
                        statistics=self.content_kind not in {'human-play', 'competition-match'})
        defaults.update(self.capabilities)
        return dict(id=self.id, title=self.title, description=self.description,
                    content_kind=self.content_kind, protocol=self.protocol,
                    path='/rooms/' + self.id, api_base='/api/live/rooms/' + self.id,
                    participants=[dict(id=n.lower(),name=n,lane=i) for i,n in enumerate(('Lume','Clari','Vero'))] if self.content_kind=='classic-multi-ai' else [],
                    capabilities=defaults,
                    **self.metadata)


DEFAULT_ROOM = RoomDefinition(
    id=DEFAULT_ROOM_ID, title={'zh': '2048 AI 直播', 'en': '2048 AI Live'},
    content_kind='classic-multi-ai' if os.environ.get('LIVE_AI_LANES') == '3' else 'classic-ai',
    protocol='classic-multi-v1' if os.environ.get('LIVE_AI_LANES') == '3' else 'classic-step-v1',
    description={'zh': 'AI 实时思考，搜索与残局定式共同决策。', 'en': 'Live decisions powered by search and endgame tables.'},
    publish_token_env='LIVE_PUBLISH_TOKEN', milestone_rewards=True,
)
ROOMS = {DEFAULT_ROOM.id: DEFAULT_ROOM}


def register_room(room):
    if room.id in ROOMS:
        raise ValueError('duplicate_room')
    # A room must explicitly choose its publisher credential. Never inherit the default.
    if not room.publish_token_env or any(r.publish_token_env == room.publish_token_env for r in ROOMS.values()):
        raise ValueError('room_requires_distinct_publisher_credential')
    ROOMS[room.id] = room
