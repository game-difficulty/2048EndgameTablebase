import asyncio
from unittest.mock import patch
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.live.competition_content import CompetitionMatchContent
from backend.live.dynamic_rooms import CompetitionMatchRoomProvider
from backend.live.routes import require_room_capability


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
