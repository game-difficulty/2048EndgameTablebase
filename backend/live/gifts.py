"""Live host adapter for the reusable gift transaction service."""
from backend.gifts.service import *  # Compatibility API for existing callers.
from backend.gifts import service
from backend.auth.db import auth_db
from . import audience


def init_schema():
    service.init_schema()
    with auth_db() as db:
        audience.init_schema(db)


def send(user, data, online, target='live:ai-classic'):
    room_id = target.removeprefix('live:')
    return service.send(user, data, online, target=target,
                        on_paid=lambda db, uid, cost: audience.add(db, f'u:{uid}', 'gift_units', cost, room_id=room_id))
