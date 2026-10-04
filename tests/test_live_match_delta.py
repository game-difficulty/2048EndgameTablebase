import asyncio
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.live.match_delta import MatchDeltaEncoder
from backend.live import routes
from backend.live.rooms import RoomDefinition


def snapshot(sequence=1):
    payload = {'board': [[2, 4, 0, 0]] * 4, 'score': sequence * 4, 'cargo': [{'cells': [0, 1, 4]}]}
    return {'type': 'snapshot', 'room_id': 'competition-demo', 'protocol': 'competition-match-v1',
            'server_time': sequence, 'online': True, 'match': {
                'match_public_key': 'demo', 'generation': 1, 'current_game': 'A', 'phase': 'GAME_A_PLAYING',
                'content_sequence': sequence, 'server_time': '2026-10-04T12:00:00Z',
                'projects': [{'name': 'Project '+str(n), 'rules': 'Long rules. '*100} for n in range(20)],
                'teams': {'yellow': {'name': 'Yellow'}}, 'score': {'yellow': 0, 'white': 0},
                'project_public_views': {'yellow': {'instance_id': 'yellow-a', 'sequence': sequence,
                    'frame_start': sequence, 'payload': payload,
                    'frames': [{'sequence': sequence, 'payload': payload}]}, 'white': None}}}


def test_delta_omits_static_metadata_and_duplicate_payload_without_mutation():
    encoder = MatchDeltaEncoder()
    first = snapshot(); first_copy = deepcopy(first)
    full = encoder.encode(first)
    second = snapshot(2); second_copy = deepcopy(second)
    delta = encoder.encode(second)
    assert full['type'] == 'snapshot'
    assert delta['type'] == 'match_delta' and delta['base_sequence'] == 1
    assert 'projects' not in delta['match']['set'] and 'teams' not in delta['match']['set']
    assert 'white' not in delta['views']['set']
    view = delta['views']['set']['yellow']
    assert view['payload_from_last_frame'] and 'payload' not in view
    assert view['frames'][0]['payload'] == second['match']['project_public_views']['yellow']['payload']
    assert len(json.dumps(delta)) < len(json.dumps(second)) / 10
    assert first == first_copy and second == second_copy


@pytest.mark.parametrize('field,value', [('phase','LINEUP'), ('current_game','B'), ('generation',2)])
def test_phase_game_and_generation_changes_force_a_complete_snapshot(field, value):
    encoder = MatchDeltaEncoder(); encoder.encode(snapshot())
    next_value = snapshot(2); next_value['match'][field] = value
    full = encoder.encode(next_value)
    assert full['type'] == 'snapshot' and full['match']['projects']


def test_result_transition_never_truncates_an_ordinary_final_frame_batch():
    encoder=MatchDeltaEncoder();encoder.encode(snapshot())
    result=snapshot(51);result['match']['phase']='GAME_A_RESULT'
    result['match']['project_public_views']['yellow']['frames']=[{'sequence':n,'payload':{'score':n}} for n in range(2,52)]
    message=encoder.encode(result)
    assert [frame['sequence'] for frame in message['match']['project_public_views']['yellow']['frames']]==list(range(2,52))


def test_python_encoder_roundtrips_through_actual_browser_decoder():
    if not shutil.which('node'):
        pytest.skip('Node is needed for the cross-language wire contract')
    encoder = MatchDeltaEncoder(); states = [snapshot(n) for n in range(1, 5)]
    states[1]['match']['project_public_views']['white'] = {'sequence': 1, 'payload': {'board': [[64]]}, 'frames': []}
    states[2]['match']['teams'] = None
    states[2]['match']['project_public_views'].pop('white')
    states[3]['match'].pop('teams'); states[3]['match']['score']['yellow'] = 1
    states += [{**snapshot(5), 'match': {**snapshot(5)['match'], 'current_game': 'B', 'phase': 'GAME_B_READY'}}]
    messages = [encoder.encode(state) for state in states]
    script = """import {createMatchDeltaDecoder} from './frontend/src/live/matchDelta.js';
      let text='';for await (const chunk of process.stdin)text+=chunk;
      const messages=JSON.parse(text);const codec=createMatchDeltaDecoder({id:'competition-demo',protocol:'competition-match-v1'});
      console.log(JSON.stringify(messages.map(message=>codec.decode(message))));"""
    result = subprocess.run(['node', '--input-type=module', '-e', script], cwd=Path(__file__).resolve().parents[1],
                            input=json.dumps(messages), text=True, capture_output=True, check=True)
    decoded = json.loads(result.stdout)
    for actual, expected in zip(decoded, states):
        actual.pop('stream_epoch'); actual.pop('stream_sequence')
        assert actual == expected


def make_runtime():
    room = RoomDefinition(id='competition-demo', title={'en': 'Test'}, content_kind='competition-match',
                          protocol='competition-match-v1', dynamic=True, metadata={'public_key':'demo','generation':1})
    with patch('backend.live.competition_content.competition_provider.projection', return_value=snapshot()['match']):
        runtime = routes.LiveHub(room)
        runtime.content.accept_projection(snapshot()['match'])
        return runtime


def test_social_channel_has_no_board_backlog_and_legacy_protocol_is_unchanged():
    runtime = make_runtime()
    legacy, social, board = (routes.ViewerQueue(byte_limit=2*1024*1024) for _ in range(3))
    social.channel = 'social'; board.channel = 'board'
    runtime.viewers = {'legacy':legacy, 'social':social}; runtime.board_viewers = {'board':board}
    for n in range(1, 35):
        runtime.broadcast(snapshot(n))
    # Overflow closes only the stalled board connection. Social events never queue behind boards.
    assert board.closing and social.empty()
    runtime.broadcast({'type':'gift','id':'paid-1'})
    runtime.broadcast({'type':'chat','id':'chat-1'})
    assert [social.get_nowait()['id'],social.get_nowait()['id']] == ['paid-1','chat-1']
    old = []
    while not legacy.empty():old.append(legacy.get_nowait())
    assert all(m['type'] != 'match_delta' for m in old)
    assert next(m for m in old if m['type']=='snapshot')['match']['projects']
    assert board.get_nowait() is None


def test_real_watch_routes_isolate_channels_and_count_presence_only_once():
    runtime = make_runtime(); app = FastAPI(); app.include_router(routes.router)
    with patch.object(routes, 'resolve_hub', return_value=runtime), \
         patch.object(routes, 'current_user_from_websocket', return_value=None), \
         patch.object(routes, 'current_guest_from_websocket', return_value=None), \
         patch.object(routes, 'dynamic_hubs', {runtime.room.id:runtime}), TestClient(app) as client:
        base='/api/live/rooms/competition-demo/watch'
        with client.websocket_connect(base+'?channel=social') as social, client.websocket_connect(base+'?channel=board') as board:
            assert social.receive_json()['type']=='social_snapshot'
            full=board.receive_json(); assert full['type']=='snapshot' and full['stream_epoch']
            assert len(runtime.viewers)==1 and len(runtime.board_viewers)==1
            assert len(runtime.audience.identities())==1
            client.portal.call(runtime.broadcast,snapshot(2))
            assert board.receive_json()['type']=='match_delta'
            client.portal.call(runtime.broadcast,{'type':'gift','id':'paid-test'})
            assert social.receive_json()['id']=='paid-test'
            social.send_text('ping'); assert social.receive_json()['type']=='pong'
            board.send_text('ping'); assert board.receive_json()['type']=='match_watermark'
        assert not runtime.viewers and not runtime.board_viewers and not runtime.audience.identities()


def test_social_history_endpoint_never_builds_or_serializes_a_board_snapshot():
    runtime=make_runtime(); runtime.chat.append({'type':'chat','id':'one','text':'hello'})
    request=SimpleNamespace(path_params={'room_id':runtime.room.id}); response=routes.Response()
    with patch.object(routes,'resolve_hub',return_value=runtime), \
         patch.object(runtime,'snapshot',side_effect=AssertionError('board snapshot used')), \
         patch('backend.chat_moderation.visible_messages',side_effect=lambda values:values):
        data=asyncio.run(routes.social_state(request,response))
    assert data['type']=='social_snapshot' and 'match' not in data
    assert data['chat'][0]['id']=='one' and response.headers['Cache-Control']=='no-store'


def test_retiring_generation_closes_both_sockets_and_stops_old_relay():
    async def run():
        social,board=AsyncMock(),AsyncMock()
        runtime=SimpleNamespace(viewers={social:None},board_viewers={board:None},producer=None,stop=AsyncMock())
        await routes.retire_runtime(runtime)
        social.close.assert_awaited_once_with(code=1012)
        board.close.assert_awaited_once_with(code=1012)
        runtime.stop.assert_awaited_once()
    asyncio.run(run())


def test_real_runtime_replacement_retires_board_and_social_even_on_http_resolution():
    async def run():
        old,new=make_runtime(),make_runtime();new.room.metadata['generation']=2
        board,social=AsyncMock(),AsyncMock()
        old.viewers[social]=routes.ViewerQueue();old.board_viewers[board]=routes.ViewerQueue()
        old.stop=AsyncMock();new.start=AsyncMock()
        with patch.dict(routes.dynamic_hubs,{old.room.id:old},clear=True), \
             patch.object(routes,'LiveHub',return_value=new), \
             patch.object(routes.lucky_bags,'listing',return_value=[]), \
             patch.object(routes.gifts,'recent_events',return_value=[]):
            assert routes._dynamic_hub_from_definition(new.room) is new
            await asyncio.wait_for(old.retirement_task,1)
            await new.start_task
        board.close.assert_awaited_once_with(code=1012);social.close.assert_awaited_once_with(code=1012)
        old.stop.assert_awaited_once()
    asyncio.run(run())
