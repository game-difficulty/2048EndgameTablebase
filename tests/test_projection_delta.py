import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from backend.projection_delta import ProjectionEncoder, ProjectionDecoder
from backend.live.competition_content import CompetitionMatchContent
from competition.backend.stream_routes import relay_live_projections
from competition.tests.test_live_reconciliation import Socket, stop
from tests.test_competition_live import directory_item
from backend.live.dynamic_rooms import CompetitionMatchRoomProvider


def projection(sequence=128):
    payload = {'board': [[2, 4]] * 4, 'geometry': 'x' * 2000, 'score': sequence}
    return {'match_public_key': 'demo', 'generation': 1, 'current_game': 'A', 'phase': 'GAME_A_PLAYING',
            'content_sequence': sequence, 'projects': [{'description': 'rules ' * 500} for _ in range(20)],
            'project_public_views': {'yellow': {'sequence': sequence, 'payload': payload,
                'frames': [{'sequence': n, 'payload': payload} for n in range(sequence - 127, sequence + 1)]}, 'white': None}}


def test_relay_delta_roundtrip_removals_nulls_and_payload_deduplication():
    encoder, decoder = ProjectionEncoder(), ProjectionDecoder()
    old = projection()
    assert decoder.decode(encoder.encode(old)) == old
    fresh = projection(129)
    fresh['project_public_views']['yellow']['frames'] = fresh['project_public_views']['yellow']['frames'][-1:]
    encoded = encoder.encode(fresh)
    assert encoded['type'] == 'projection_delta'
    assert 'projects' not in encoded['fields']['set']
    assert 'payload' not in encoded['views']['set']['yellow']
    assert len(json.dumps(encoded)) < len(json.dumps(fresh)) / 10
    assert decoder.decode(encoded) == fresh
    final = {**fresh, 'project_public_views': {'yellow': None}}
    del final['projects']
    assert decoder.decode(encoder.encode(final)) == final
    assert old == projection()  # No mutation of historical baselines.


@pytest.mark.parametrize('change', [{'stream_epoch': 'wrong'}, {'base_sequence': 0}, {'stream_sequence': 9}])
def test_relay_rejects_transport_gaps(change):
    encoder, decoder = ProjectionEncoder(), ProjectionDecoder()
    decoder.decode(encoder.encode(projection()))
    message = encoder.encode(projection(129))
    with pytest.raises(ValueError, match='projection_delta_gap'):
        decoder.decode({**message, **change})
    assert decoder.previous['content_sequence'] == 128


def test_legacy_and_reconnected_new_epoch_bootstrap_supported():
    decoder = ProjectionDecoder()
    assert decoder.decode({'type': 'projection', 'projection': projection()}) == projection()
    assert decoder.decode({'type': 'heartbeat'}) is None
    encoder = ProjectionEncoder()
    assert decoder.decode(encoder.encode(projection(129))) == projection(129)
    assert decoder.decode(ProjectionEncoder().encode(projection(130))) == projection(130)


@pytest.mark.parametrize('deltas', [False, True])
def test_relay_bounds_bootstrap_but_never_truncates_ordinary_or_final_batches(deltas):
    async def run():
        state = projection()
        service = SimpleNamespace(live_projection=lambda *a, **kw: deepcopy(state), live_revision=lambda _: (1, state['content_sequence']))
        socket, queue, decoder = Socket(), asyncio.Queue(), ProjectionDecoder()
        task = asyncio.create_task(relay_live_projections(socket, service, 'demo', queue, interval=60, deltas=deltas))
        try:
            initial = decoder.decode(await socket.read())
            assert len(initial['project_public_views']['yellow']['frames']) == 8
            assert initial['project_public_views']['yellow']['frame_start'] == 121
            state.update(projection(178), phase='GAME_A_RESULT')
            state['project_public_views']['yellow']['frames'] = state['project_public_views']['yellow']['frames'][-50:]
            queue.put_nowait(True)
            final = decoder.decode(await socket.read())
            assert len(final['project_public_views']['yellow']['frames']) == 50
            state.update(projection(128), current_game='B', phase='GAME_B_PLAYING')
            queue.put_nowait(True)
            next_game = decoder.decode(await socket.read())
            assert len(next_game['project_public_views']['yellow']['frames']) == 8
        finally:
            await stop(task)
    asyncio.run(run())


def test_room_creation_does_not_fetch_a_duplicate_http_bootstrap():
    room = CompetitionMatchRoomProvider()._definition(directory_item())
    with patch('backend.live.competition_content.competition_provider.projection', Mock(side_effect=AssertionError('duplicate bootstrap'))):
        content = CompetitionMatchContent(room)
    assert content.projection is None
    assert not content.online


@pytest.mark.parametrize('delta_protocol', [False, True])
def test_actual_subscriber_accepts_legacy_and_delta_and_bounds_initial_history(delta_protocol):
    room = CompetitionMatchRoomProvider()._definition(directory_item())
    first, second = projection(), projection(129)
    for state in (first, second):
        state['match_public_key'] = room.metadata['public_key']
    second['project_public_views']['yellow']['frames'] = second['project_public_views']['yellow']['frames'][-1:]
    encoder = ProjectionEncoder()
    messages = [encoder.encode(p) if delta_protocol else {'type': 'projection', 'projection': p} for p in (first, second)]
    class Connection:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def recv(self):
            if not messages:
                raise asyncio.CancelledError()
            return json.dumps(messages.pop(0))
    content = CompetitionMatchContent(room)
    delivered = []
    content.on_update = lambda: delivered.append(deepcopy(content.incremental_projection))
    with patch.dict('os.environ', {'COMPETITION_LIVE_API_ORIGIN': 'http://example.test', 'COMPETITION_LIVE_INTERNAL_TOKEN': 'test-only'}), \
         patch('websockets.legacy.client.connect', return_value=Connection()) as connect:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(content._subscribe())
    assert connect.call_args.args[0].endswith('?transport=delta-v1')
    assert len(delivered) == 2
    assert len(delivered[0]['project_public_views']['yellow']['frames']) == 8
    assert len(delivered[1]['project_public_views']['yellow']['frames']) == 1
    assert content.projection['content_sequence'] == 129
    assert content.projection['project_public_views']['yellow']['payload']['score'] == 129
    assert len(content.projection['project_public_views']['yellow']['frames']) == 9
