import asyncio
from unittest.mock import patch
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response

from backend.live.competition_content import CompetitionMatchContent
from backend.live.dynamic_rooms import CompetitionMatchRoomProvider
from backend.live.routes import require_room_capability
from backend.live import routes


def directory_item():
    return {
        'room_id': 'competition-m1234567890abcdef1234',
        'public_key': 'm1234567890abcdef1234',
        'generation': 1,
        'content_sequence': 7,
        'title': {'zh': '秋季赛', 'en': 'Autumn Cup'},
        'subtitle': {'zh': '黄方 1 : 0 白方', 'en': 'Yellow 1 : 0 White'},
        'category_label': {'zh': '赛事直播', 'en': 'TOURNAMENT'},
        'badge': {'zh': '直播中', 'en': 'LIVE'},
        'preview': {'kind': 'competition-score', 'yellow_score': 1, 'white_score': 0},
        'started_at': '2026-09-27T00:00:00+00:00',
    }


def projection(sequence=7):
    return {
        'match_public_key': 'm1234567890abcdef1234',
        'generation': 1,
        'content_sequence': sequence,
        'phase': 'GAME_A_PLAYING',
        'score': {'yellow': 1, 'white': 0, 'draws': 0},
    }


def test_competition_provider_maps_generic_room_metadata() -> None:
    provider = CompetitionMatchRoomProvider()
    with patch.object(provider, '_request', return_value={'rooms': [directory_item()]}):
        rooms = provider.list_active_rooms()
    assert len(rooms) == 1
    public = rooms[0].public()
    assert public['content_kind'] == 'competition-match'
    assert public['protocol'] == 'competition-match-v1'
    assert public['category_label']['zh'] == '赛事直播'
    assert public['capabilities'] == {
        'chat': True, 'likes': True, 'music': True, 'pip': True,
        'gifts': True, 'red_envelopes': True, 'lucky_bags': False,
        'predictions': True, 'statistics': False,
    }


def test_competition_provider_keeps_directory_during_transient_failure() -> None:
    provider = CompetitionMatchRoomProvider()
    with patch.object(provider, '_request', return_value={'rooms': [directory_item()]}):
        assert len(provider.list_active_rooms()) == 1
    with patch.object(provider, '_request', return_value=None):
        cached = provider.list_active_rooms()
    assert [room.id for room in cached] == ['competition-m1234567890abcdef1234']
    with patch.object(provider, '_request', return_value={'rooms': []}):
        assert provider.list_active_rooms() == []


def test_ended_room_expires_even_when_source_is_unavailable():
    provider = CompetitionMatchRoomProvider()
    item = {**directory_item(), 'expires_at': 1800}
    with patch('backend.live.dynamic_rooms.time.time', return_value=1799), \
         patch.object(provider, '_request', return_value={'rooms': [item]}):
        assert len(provider.list_active_rooms()) == 1
    with patch('backend.live.dynamic_rooms.time.time', return_value=1800), \
         patch.object(provider, '_request', return_value=None):
        assert provider.resolve_room(item['room_id']) is None
        assert provider.list_active_rooms() == []


def test_lobby_does_not_advertise_expired_runtime_with_connected_viewers():
    room = CompetitionMatchRoomProvider()._definition(directory_item())
    retained = SimpleNamespace(room=room, viewers={'connected-viewer'},
                               snapshot=lambda: {'online': True, 'match': projection()})
    offline = SimpleNamespace(room=SimpleNamespace(dynamic=False), snapshot=lambda: {'online': False})
    with patch.object(routes, 'hub', offline), patch.object(routes, 'room_hubs', {}), \
         patch.object(routes, 'dynamic_hubs', {room.id: retained}), \
         patch.object(routes.dynamic_room_registry, 'list_active_rooms', return_value=[]):
        assert asyncio.run(routes.lobby(Response()))['rooms'] == []
        with patch.object(routes.dynamic_room_registry, 'resolve_room', return_value=None):
            with pytest.raises(HTTPException) as captured:
                routes.resolve_hub(SimpleNamespace(path_params={'room_id': room.id}))
            assert captured.value.status_code == 404


def test_transit_keeps_recovery_history_but_broadcasts_only_new_frames():
    room = CompetitionMatchRoomProvider()._definition(directory_item())
    def state(sequence, frames):
        return {**projection(sequence), 'current_game': 'A', 'project_public_views': {
            'yellow': {'sequence': sequence, 'frames': [{'sequence': i, 'payload': {}} for i in frames]}}}
    with patch('backend.live.competition_content.competition_provider.projection', return_value=state(7, [6, 7])):
        content = CompetitionMatchContent(room)
    with patch('backend.live.competition_content.competition_provider.projection', return_value=state(10, [8, 9, 10])) as request:
        assert asyncio.run(content.refresh())
        assert request.call_args.args[1] == {'yellow': 7}
    assert [f['sequence'] for f in content.snapshot()['match']['project_public_views']['yellow']['frames']] == [6, 7, 8, 9, 10]
    assert [f['sequence'] for f in content.incremental_projection['project_public_views']['yellow']['frames']] == [8, 9, 10]


def test_competition_content_rejects_wrong_generation() -> None:
    provider = CompetitionMatchRoomProvider()
    room = provider._definition(directory_item())
    wrong = {**projection(), 'generation': 2}
    with patch('backend.live.competition_content.competition_provider.projection', return_value=wrong):
        content = CompetitionMatchContent(room)
    assert content.online is False
    assert content.snapshot() == {'match': None}


def test_competition_content_retains_last_frame_and_never_rewinds() -> None:
    room = CompetitionMatchRoomProvider()._definition(directory_item())
    with patch(
        'backend.live.competition_content.competition_provider.projection',
        return_value=projection(7),
    ):
        content = CompetitionMatchContent(room)

    with patch(
        'backend.live.competition_content.competition_provider.projection',
        return_value=None,
    ):
        assert asyncio.run(content.refresh()) is True
    assert content.online is False
    assert content.snapshot()['match']['content_sequence'] == 7

    with patch(
        'backend.live.competition_content.competition_provider.projection',
        return_value=projection(7),
    ):
        assert asyncio.run(content.refresh()) is True
    assert content.online is True

    with patch(
        'backend.live.competition_content.competition_provider.projection',
        return_value=projection(6),
    ):
        assert asyncio.run(content.refresh()) is False
    assert content.snapshot()['match']['content_sequence'] == 7

    with patch(
        'backend.live.competition_content.competition_provider.projection',
        return_value=projection(8),
    ):
        assert asyncio.run(content.refresh()) is True
    assert content.snapshot()['match']['content_sequence'] == 8


def test_same_revision_older_time_cannot_poison_cached_live_projection():
    room = CompetitionMatchRoomProvider()._definition(directory_item())
    initial = {**projection(), 'server_time': '2026-10-03T00:00:10+00:00'}
    with patch('backend.live.competition_content.competition_provider.projection', return_value=initial):
        content = CompetitionMatchContent(room)
    for timestamp in ('2026-10-03T00:00:05+00:00', None, 'invalid'):
        assert not content.accept_projection({**initial, 'server_time': timestamp, 'phase': 'LINEUP'})
        assert content.projection == initial
        assert content.incremental_projection == initial
    # Same revision with a newer source sample may refresh the cached time.
    fresh = {**initial, 'server_time': '2026-10-03T00:00:15Z'}
    assert not content.accept_projection(fresh)
    assert content.projection == fresh
    # A new revision is authoritative, including corrections and extra time.
    correction = {**initial, 'content_sequence': 8, 'team_clocks': {'yellow': {'remaining_ms': 90000}}}
    assert content.accept_projection(correction)
    assert content.projection == correction


def test_identical_revision_and_timestamp_can_restore_missing_frames():
    room = CompetitionMatchRoomProvider()._definition(directory_item())
    def state(frames):
        return {**projection(), 'server_time': '2026-10-03T00:00:10+00:00', 'current_game': 'A',
                'project_public_views': {'yellow': {'sequence': 7, 'frames': [{'sequence': n} for n in frames]}}}
    with patch('backend.live.competition_content.competition_provider.projection', return_value=state([7])):
        content = CompetitionMatchContent(room)
    content.accept_projection(state([5, 6, 7]))
    assert [frame['sequence'] for frame in content.projection['project_public_views']['yellow']['frames']] == [5, 6, 7]


def test_relayed_source_clock_advances_across_cached_snapshots_and_late_samples():
    room = CompetitionMatchRoomProvider()._definition(directory_item())
    initial = {**projection(), 'server_time': '1970-01-01T00:03:20+00:00',
               'team_clocks': {'yellow': {'state': 'running', 'remaining_ms': 60000}}}
    with patch('backend.live.competition_content.time.monotonic', return_value=0) as mono, \
         patch('backend.live.competition_content.competition_provider.projection', return_value=initial):
        content = CompetitionMatchContent(room)
        assert content.snapshot()['competition_server_time'] == 200
        mono.return_value = 12
        # No source changes: a new viewer still sees 48s rather than restarting at 60s.
        assert content.snapshot()['competition_server_time'] == 212
        assert content.snapshot()['match']['team_clocks']['yellow']['remaining_ms'] == 60000
        content.accept_projection({**initial, 'server_time': '1970-01-01T00:03:25+00:00'})
        assert content.snapshot()['competition_server_time'] == 212
        mono.return_value = 13
        content.accept_projection(initial)
        assert content.snapshot()['competition_server_time'] == 213
        content.accept_projection({**initial, 'content_sequence': 8, 'server_time': '1970-01-01T00:03:35+00:00'})
        assert content.snapshot()['competition_server_time'] == 215


def test_competition_gifts_red_envelopes_predictions_enabled_but_not_lucky_bags() -> None:
    room = CompetitionMatchRoomProvider()._definition(directory_item())
    runtime = SimpleNamespace(room=room)
    require_room_capability(runtime, 'likes')
    require_room_capability(runtime, 'gifts')
    require_room_capability(runtime, 'red_envelopes')
    require_room_capability(runtime, 'predictions')
    with pytest.raises(HTTPException) as captured:
        require_room_capability(runtime, 'lucky_bags')
    assert captured.value.status_code == 404
