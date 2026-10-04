import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.live import routes
from tests.test_live_match_delta import make_runtime


def test_competition_maintenance_never_builds_boards_or_empty_statistics():
    runtime = make_runtime()
    runtime.summary_week = ''
    with patch('backend.live.routes.asyncio.sleep', new=AsyncMock(side_effect=[None, None, None, asyncio.CancelledError()])), \
         patch.object(runtime, 'snapshot', side_effect=AssertionError('unneeded board serialization')), \
         patch.object(runtime.store, 'summary', side_effect=AssertionError('empty statistics')), \
         patch.object(runtime.store, 'refresh_stats_snapshots', side_effect=AssertionError('empty statistics')), \
         patch.object(routes.audience, 'online_tick'), patch.object(runtime, 'broadcast') as broadcast:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(runtime.maintenance())
    assert [c.args[0]['type'] for c in broadcast.call_args_list] == ['presence'] * 3


def test_dynamic_reconciliation_has_no_second_presence_publisher():
    runtime = make_runtime()
    runtime.activities.refresh = AsyncMock()
    with patch.object(runtime, 'broadcast') as broadcast:
        asyncio.run(routes._reconcile_dynamic_room(runtime.room.id, runtime, runtime.room, 100))
    runtime.activities.refresh.assert_awaited_once()
    broadcast.assert_not_called()
