import asyncio
from contextlib import suppress
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from competition.backend.errors import CompetitionError
from competition.backend.stream_routes import relay_live_projections
from competition.tests.test_client_runtime import setup_game


class Socket:
    def __init__(self):
        self.messages = asyncio.Queue()

    async def send_json(self, message):
        await self.messages.put(message)

    async def read(self):
        return await asyncio.wait_for(self.messages.get(), 2)


async def stop(task):
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def test_durable_change_without_notification_is_reconciled(tmp_path):
    service, _ = setup_game(tmp_path)
    key = service.list_live_rooms()[0]['public_key']

    async def run():
        socket = Socket()
        task = asyncio.create_task(relay_live_projections(socket, service, key, asyncio.Queue(), interval=.02))
        try:
            initial = (await socket.read())['projection']
            with service.database.transaction(immediate=True) as db:
                room = service._room_row(db, 'MATCH5')
                # Simulate a committed transition without a RoomHub notification.
                service._touch(db, room['id'], status='LINEUP')
            while True:
                message = await socket.read()
                if message['type'] == 'projection':
                    break
            assert message['projection']['phase'] == 'LINEUP'
            assert message['projection']['content_sequence'] > initial['content_sequence']
        finally:
            await stop(task)
    asyncio.run(run())


def test_idle_checks_do_not_rebuild_projection_and_source_failure_is_not_heartbeat():
    async def run():
        projection = dict(generation=1, content_sequence=4, current_game='A')
        service = SimpleNamespace(live_projection=Mock(return_value=projection), live_revision=Mock(return_value=(1, 4)))
        socket = Socket()
        task = asyncio.create_task(relay_live_projections(socket, service, 'key', asyncio.Queue(), interval=.01))
        try:
            assert (await socket.read())['type'] == 'projection'
            for _ in range(3):
                assert (await socket.read())['type'] == 'heartbeat'
            assert service.live_projection.call_count == 1
            service.live_revision.side_effect = RuntimeError('source unavailable')
            with pytest.raises(RuntimeError, match='source unavailable'):
                await asyncio.wait_for(task, 2)
            assert socket.messages.empty()
        finally:
            if not task.done():
                await stop(task)
    asyncio.run(run())


@pytest.mark.parametrize('change', [{'generation': 2}, {'current_game': 'B'}])
def test_generation_or_game_change_resets_frame_cursors(change):
    async def run():
        state = dict(generation=1, content_sequence=4, current_game='A', project_public_views={'yellow': {'sequence': 12}})
        calls = []
        def project(_key, **kwargs):
            calls.append(kwargs)
            return deepcopy(state)
        service = SimpleNamespace(live_projection=project, live_revision=lambda _key: (state['generation'], state['content_sequence']))
        socket = Socket()
        task = asyncio.create_task(relay_live_projections(socket, service, 'key', asyncio.Queue(), interval=.01))
        try:
            await socket.read()
            state.update(change, content_sequence=5)
            assert (await socket.read())['type'] == 'projection'
            assert calls[-2]['after_yellow'] == 12
            assert calls[-1] == {}
        finally:
            await stop(task)
    asyncio.run(run())


def test_frequent_notifications_cannot_postpone_source_check():
    async def run():
        state = dict(generation=1, content_sequence=4, current_game='A')
        service = SimpleNamespace(live_projection=lambda _key, **kw: state, live_revision=Mock(return_value=(1, 4)))
        socket, queue = Socket(), asyncio.Queue()
        task = asyncio.create_task(relay_live_projections(socket, service, 'key', queue, interval=.1))
        try:
            await socket.read()
            for _ in range(12):
                queue.put_nowait(True)
                await socket.read()
                await asyncio.sleep(.025)
            assert service.live_revision.call_count > 0
        finally:
            await stop(task)
    asyncio.run(run())


def test_revision_checks_retention_without_building_projection(tmp_path, monkeypatch):
    service, _ = setup_game(tmp_path)
    key = service.list_live_rooms()[0]['public_key']
    projection = service.live_projection(key)
    monkeypatch.setattr(service, '_public_match_projection', Mock(side_effect=AssertionError('expensive projection')))
    assert service.live_revision(key) == (projection['generation'], projection['content_sequence'])
    with service.database.transaction(immediate=True) as db:
        db.execute("UPDATE competitions SET live_ended_at='2000-01-01T00:00:00+00:00' WHERE public_key=?", (key,))
    with pytest.raises(CompetitionError) as error:
        service.live_revision(key)
    assert error.value.code == 'LIVE_ROOM_NOT_FOUND'
