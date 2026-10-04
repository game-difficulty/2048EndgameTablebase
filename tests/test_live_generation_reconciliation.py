import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from backend.live import routes
from backend.live.rooms import RoomDefinition


def runtime(generation):
    viewer = Mock(close=AsyncMock())
    return SimpleNamespace(room=RoomDefinition(id='test-generation', title={},
        metadata={'generation': generation}), producer=None, viewers={viewer: object()},
        content=SimpleNamespace(refresh=AsyncMock(return_value=False)),
        room_ended_notified=False, broadcast=Mock(), control_status=lambda:{'online':False},
        audience=SimpleNamespace(identities=lambda:[]))


def test_generation_change_delegates_replacement_to_runtime_factory():
    for definition in ({'generation': 2}, RoomDefinition(id='test-generation', title={},
                                                        metadata={'generation': 2})):
        old, new = runtime(1), runtime(2)
        with patch.dict(routes.dynamic_hubs, {'test-generation': old}, clear=True), \
                patch.object(routes, '_dynamic_hub', return_value=new) as resolve:
            asyncio.run(routes._reconcile_dynamic_room('test-generation', old, definition, 1))
        resolve.assert_called_once_with('test-generation')
        # The real factory retires both channels (covered by the route integration test).
        old.content.refresh.assert_not_awaited()


def test_same_generation_preserves_connections():
    old = runtime(1)
    with patch.object(routes, '_dynamic_hub') as resolve:
        asyncio.run(routes._reconcile_dynamic_room('test-generation', old, old.room, 1))
    resolve.assert_not_called()
    next(iter(old.viewers)).close.assert_not_awaited()


def test_stale_cleanup_cannot_remove_new_generation():
    old, new = runtime(1), runtime(2)
    old.viewers.clear()
    with patch.dict(routes.dynamic_hubs, {'test-generation': new}, clear=True):
        asyncio.run(routes._reconcile_dynamic_room('test-generation', old, None, 1))
        assert routes.dynamic_hubs['test-generation'] is new
