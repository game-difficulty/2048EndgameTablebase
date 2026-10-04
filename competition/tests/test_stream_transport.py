import asyncio
import contextlib
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from competition.backend.hub import SocketWriter
from competition.backend.stream_protocol import expand_checkpoint, PROTOCOL
from competition.backend.errors import CompetitionError
from competition.tests.test_client_runtime import setup_game, packet

def stream_app(tmp_path, monkeypatch):
    monkeypatch.setenv('CLOUD_AUTH_DB', str(tmp_path / 'auth.sqlite3'))
    service, players = setup_game(tmp_path)
    from competition.backend import app as module
    settings = replace(module.load_settings(), database_path=service.database.path,
                       allow_dev_auth=True, live_internal_token='test-stream-secret')
    monkeypatch.setattr(module, 'load_settings', lambda: settings)
    return module.create_app(), service, players

def hello(socket, runtime, user):
    socket.send_json({'type':'authenticate','data':{'protocol':PROTOCOL,
        'instance_id':runtime['instance_id'],'dev_user':str(user.user_id)}})
    return socket.receive_json()

def test_private_delta_restores_history_and_rejects_wrong_base():
    previous={'version':1,'state':{'undo':[[1],[2]],'randomState':1},'metric_history':[[0,1]]}
    delta={'version':1,'state':{'randomState':2},'delta_base':4,
           'lists':{'undo':{'keep':1,'append':[[3]]},'lookBackHistory':{'keep':0,'append':[]},
                    'metric_history':{'keep':1,'append':[[1,2]]}}}
    next=expand_checkpoint(delta,previous,4)
    assert next['state']['undo']==[[1],[3]]
    assert previous['state']['undo']==[[1],[2]]
    with pytest.raises(CompetitionError): expand_checkpoint(delta,previous,3)

def test_two_producers_room_viewer_and_live_subscription(tmp_path, monkeypatch):
    app, service, players=stream_app(tmp_path,monkeypatch)
    key=service.list_live_rooms()[0]['public_key']
    with TestClient(app) as client:
        with client.websocket_connect('/ws/rooms/MATCH5?dev_user='+str(players[1].user_id)) as viewer, \
             client.websocket_connect('/ws/internal/live/'+key,headers={'X-Competition-Live-Token':'test-stream-secret'}) as live, \
             client.websocket_connect('/ws/projects/MATCH5') as yellow, \
             client.websocket_connect('/ws/projects/MATCH5') as white:
            assert viewer.receive_json()['type']=='room.snapshot'
            assert live.receive_json()['type']=='projection'
            for socket,user in [(yellow,players[0]),(white,players[3])]:
                runtime=service.snapshot('MATCH5',user)['match']['my_session']['runtime']
                assert hello(socket,runtime,user)['accepted_sequence']==0
            for seq in range(1,13):
                for socket,user in [(yellow,players[0]),(white,players[3])]:
                    data=packet(service,user,sequence=seq)
                    data['frames']=[{'sequence':seq,'payload':data['payload']}]
                    socket.send_json({'type':'project.batch','data':data})
            # Pipeline all batches before reading ANY acknowledgements.
            for socket in [yellow,white]:
                assert [socket.receive_json()['accepted_sequence'] for _ in range(12)]==list(range(1,13))
            seen={'yellow':[],'white':[]}
            while any(len(v)<12 for v in seen.values()):
                msg=viewer.receive_json()
                if msg['type']=='project.snapshot':
                    update=msg['data'];seen[update['side']].extend(f['sequence'] for f in update['public_view']['frames'])
            assert seen=={'yellow':list(range(1,13)),'white':list(range(1,13))}
            latest={}
            while not latest or any(latest[s]['sequence']<12 for s in ['yellow','white']):
                msg=live.receive_json()
                if msg['type']=='projection': latest=msg['projection']['project_public_views']
            assert all(v['sequence']==12 for v in latest.values())
            assert 'checkpoint' not in str(latest)
        # Durable cursor survives a connection replacement; duplicate replay is harmless.
        with client.websocket_connect('/ws/projects/MATCH5') as socket:
            runtime=service.snapshot('MATCH5',players[0])['match']['my_session']['runtime']
            assert hello(socket,runtime,players[0])['accepted_sequence']==12
            data=packet(service,players[0],sequence=12)
            data['frames']=[{'sequence':12,'payload':data['payload']}]
            socket.send_json({'type':'project.batch','data':data})
            assert socket.receive_json()['accepted_sequence']==12
        response=client.post('/api/competitions/MATCH5/games/current/state',json=data,
                             headers={'X-Competition-Dev-User':str(players[0].user_id)})
        assert response.status_code==426

def test_writer_closes_instead_of_silently_detaching_on_backpressure():
    async def run():
        socket=AsyncMock(); detached=AsyncMock()
        writer=SocketWriter(socket,detached,byte_limit=10)
        await writer.put({'too':'large'})
        await writer.close_task
        socket.close.assert_awaited_once_with(code=1013,reason='stream_backpressure')
        detached.assert_awaited_once()
        with contextlib.suppress(asyncio.CancelledError):
            await writer.task
    asyncio.run(run())


def test_slow_spectator_close_does_not_block_new_producer_batches():
    async def run():
        socket, detached = AsyncMock(), AsyncMock()
        gate = asyncio.Event()
        async def slow_close(**_):
            await gate.wait()
        socket.close.side_effect = slow_close
        writer = SocketWriter(socket, detached, byte_limit=10)
        await asyncio.wait_for(writer.put({'too':'large'}), .5)
        assert writer.close_task is not None
        # Subsequent enqueue calls also finish without waiting for that peer.
        await asyncio.wait_for(writer.put({'another':'update'}), .5)
        gate.set()
        await writer.close_task
        detached.assert_awaited_once()
    asyncio.run(run())


def test_public_room_broadcast_serializes_once_for_all_viewers():
    from unittest.mock import patch
    from competition.backend.hub import RoomHub
    import json
    async def run():
        hub = RoomHub()
        sockets = [AsyncMock() for _ in range(20)]
        for socket in sockets:
            await hub.connect('room', socket, None)
        with patch('competition.backend.hub.json.dumps', wraps=json.dumps) as encode:
            await hub.broadcast_message('room', {'type':'project.snapshot', 'data':{'instance_id':'A'}})
            assert encode.call_count == 1
        for socket in sockets:
            await hub.disconnect('room', socket)
    asyncio.run(run())


def test_slow_viewer_keeps_latest_board_and_ordered_non_board_events():
    async def run():
        import json
        socket, detached = AsyncMock(), AsyncMock()
        writer = SocketWriter(socket, detached)
        # Keep the sender idle while the producer outruns this viewer.
        writer.task.cancel()
        for seq in range(1, 120):
            await writer.put({'type': 'project.snapshot', 'data': {'instance_id': 'A:yellow',
                'public_view': {'sequence': seq, 'payload': {'score': seq}, 'frames': []}}})
            if seq == 3:
                await writer.put({'type': 'important', 'value': 'preserve'})
        queued = []
        while not writer.queue.empty():
            text, _ = writer.queue.get_nowait()
            queued.append(json.loads(text))
        assert [m['value'] for m in queued if m['type'] == 'important'] == ['preserve']
        views = [m['data']['public_view'] for m in queued if m['type'] == 'project.snapshot']
        assert len(views) < 119
        assert views[-1]['sequence'] == 119
        assert not writer.closed
        await writer.close()
    asyncio.run(run())


def test_delta_ack_persists_full_recovery_and_replacement_has_one_publisher(tmp_path, monkeypatch):
    app, service, players = stream_app(tmp_path, monkeypatch)
    user = players[0]
    with TestClient(app) as client:
        with client.websocket_connect('/ws/projects/MATCH5') as first:
            runtime = service.snapshot('MATCH5', user)['match']['my_session']['runtime']
            hello(first, runtime, user)
            for seq in [1, 2]:
                data = packet(service, user, sequence=seq)
                data['frames'] = [{'sequence': seq, 'payload': data['payload']}]
                if seq == 1:
                    data['checkpoint']['state']['undo'] = [[1], [2]]
                    data['checkpoint']['metric_history'] = [[0, 0]]
                else:
                    data['checkpoint'].update(delta_base=1, lists={
                        'undo': {'keep': 1, 'append': [[3]]},
                        'lookBackHistory': {'keep': 0, 'append': []},
                        'metric_history': {'keep': 1, 'append': [[500, 100]]},
                    })
                first.send_json({'type': 'project.batch', 'data': data})
                assert first.receive_json()['accepted_sequence'] == seq
            restored = service.snapshot('MATCH5', user)['match']['my_session']['runtime']
            assert restored['checkpoint']['state']['undo'] == [[1], [3]]
            assert restored['checkpoint']['metric_history'] == [[0, 0], [500, 100]]
            assert 'delta_base' not in restored['checkpoint']
            with client.websocket_connect('/ws/projects/MATCH5') as second:
                assert hello(second, restored, user)['accepted_sequence'] == 2
                with pytest.raises(WebSocketDisconnect) as closed:
                    first.receive_json()
                assert closed.value.code == 4409


def test_legacy_producer_is_rejected_before_accepting_any_game_state(tmp_path, monkeypatch):
    app, _, _ = stream_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        with client.websocket_connect('/ws/projects/MATCH5') as socket:
            socket.send_json({'type': 'authenticate', 'data': {'protocol': 'old'}})
            with pytest.raises(WebSocketDisconnect) as closed:
                socket.receive_json()
            assert closed.value.code == 4406
