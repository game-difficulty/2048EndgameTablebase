import asyncio
import json
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace
from backend.auth.db import init_auth_db
from backend.gamer_ranked.rules import legal_moves
from backend.live.multi_content import MultiAiContent
from backend.live.multi_protocol import PUBLISH_PROTOCOL, UPLINK
from backend.live.protocol import LiveRun, STEP
from backend.room_activities import predictions, lucky_bags
from backend.room_activities.runtime import RoomActivities
from tests.test_live_multi import Socket, handshake
from tools.live_multi_runner import MultiRunner


class Room:
    id='test-batches'
    milestone_rewards=True
    def __init__(self,path):self.path=path
    def store_path(self):return self.path


class Hub:
    producer=None
    like_total=0
    def __init__(self,content):
        self.content=content;self.room=content.room;self.store=content.store
        self.control=dict(enabled=True,revision=0)
        self.events=[];self.activities=RoomActivities(self);self.activity_lock=self.activities.lock
    def broadcast(self,data):self.events.append(data)
    def snapshot(self):return dict(type='snapshot',**self.content.snapshot())
    async def publisher_joined(self):pass
    async def refresh_lucky(self):pass


def finish_packets(run,lane,generation):
    packets=[]
    while choices:=legal_moves(run.board):
        step=run.make_step(choices[0],80)
        run.apply(step);packets.append(UPLINK.pack(lane,generation,*STEP.unpack(step)))
    packets.append(dict(type='end',lane=lane,generation=generation))
    run.end()
    return packets


class BatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        env=patch.dict(os.environ,CLOUD_AUTH_DB=str(Path(directory.name)/'auth.db'))
        env.start();self.addCleanup(env.stop)
        init_auth_db();predictions.init_schema();lucky_bags.init_schema()
        self.content=MultiAiContent(Room(Path(directory.name)/'live.db'))
        await self.content.start()
        self.hub=Hub(self.content);self.content.hub=self.hub

    def proposal(self):
        runs=[LiveRun() for _ in range(3)]
        return runs,dict(type='batch_start',batch_id=str(uuid.uuid4()),lanes=[
            dict(lane=i,run_id=r.id,seed=r.seed,generation=self.content.generations[i]+1) for i,r in enumerate(runs)])

    async def test_new_market_has_ten_minute_window_and_closes_at_boundary(self):
        await self.content.begin_batch(self.proposal()[1])
        batch=self.content.batch
        self.assertEqual(batch['deadline']-batch['started_at'],600)
        self.assertEqual(predictions.listing(self.hub.room.id,now=batch['started_at']+599)['market']['status'],'open')
        self.assertEqual(predictions.listing(self.hub.room.id,now=batch['deadline'])['market']['status'],'closed')

    async def test_shared_start_early_finish_waits_and_all_finish_settles(self):
        runs,proposal=self.proposal()
        messages=handshake()+[proposal]+finish_packets(runs[0],0,1)
        socket=Socket(messages);await self.content.publish(socket,self.hub)
        self.assertIsNone(socket.closed)
        self.assertEqual(len({r.started for r in self.content.runs}),1)
        self.assertEqual(self.content.batch['phase'],'running')
        self.assertTrue(self.content.runs[0].ended)
        self.assertEqual(self.content.runs[0].board,runs[0].board)
        self.assertFalse(self.content.runs[1].ended)
        with self.assertRaisesRegex(ValueError,'batch_not_finished'):await self.content.begin_batch(self.proposal()[1])
        messages=handshake(self.content.runs)+finish_packets(runs[1],1,1)+finish_packets(runs[2],2,1)
        socket=Socket(messages);await self.content.publish(socket,self.hub)
        self.assertIsNone(socket.closed)
        self.assertEqual(self.content.batch['phase'],'cooldown')
        result=predictions.listing(self.hub.room.id)['market']
        self.assertEqual(result['status'],'settled')
        high=max(r.score for r in runs)
        self.assertEqual(set(result['winners']),{p['id'] for p,r in zip(self.content.participants(),runs) if r.score==high})
        with self.assertRaisesRegex(ValueError,'batch_not_finished'):await self.content.begin_batch(self.proposal()[1])
        self.content.batch['next_at']=0
        await self.content.begin_batch(self.proposal()[1])
        self.assertTrue(all(r.seq==0 and not r.ended for r in self.content.runs))

    async def test_result_known_from_terminal_board_closes_before_end_message(self):
        await self.content.begin_batch(self.proposal()[1])
        board=[2,4,2,4,4,2,4,2,2,4,2,4,4,2,4,2]
        for r in self.content.runs:r.board=board[:]
        await self.content.advance_batch()
        self.assertEqual(predictions.listing(self.hub.room.id)['market']['status'],'closed')
        self.assertEqual(self.content.batch['phase'],'running')

    async def test_outbox_first_crossing_is_atomic_and_maximum_two_per_batch(self):
        await self.content.begin_batch(self.proposal()[1])
        for lane,value in [(2,32768),(0,32768),(1,65536),(2,65536)]:
            self.content.runs[lane].nodes[str(value)]=100
            await self.content.record_milestones(lane)
        events=self.content.store.pending_activities()
        self.assertEqual(len(events),2)
        batch=self.content.store.load_batch()
        self.assertEqual(batch['milestones']['32768']['participant_id'],'vero')
        self.assertEqual(batch['milestones']['65536']['participant_id'],'clari')
        with patch.object(self.content.store,'activity_delivered',side_effect=RuntimeError('crash after auth commit')):
            with self.assertRaises(RuntimeError):await self.hub.activities.drain()
        await self.hub.activities.drain();await self.hub.activities.drain()
        self.assertEqual(len(lucky_bags.listing(room_id=self.hub.room.id)),2)
        self.assertEqual(self.content.store.pending_activities(),[])

    async def test_target_tracks_each_player_and_terminal_failure_independently_of_giveaways(self):
        await self.content.begin_batch(self.proposal()[1])
        self.content.runs[1].nodes['65536']=100
        await self.content.record_milestones(1)
        await self.content.advance_batch()
        self.content.runs[0].nodes['65536']=200
        await self.content.record_milestones(0)
        self.content.runs[2].board=[2,4,2,4,4,2,4,2,2,4,2,4,4,2,4,2]
        await self.content.advance_batch()
        expected={'lume':'reached','clari':'reached','vero':'missed'}
        self.assertEqual(self.content.store.load_batch()['target_results'],expected)
        self.assertEqual(predictions.listing(self.hub.room.id)['market']['target_bet']['outcomes'],expected)
        self.assertEqual(len(self.content.store.pending_activities()),1)

    async def test_target_facts_replay_after_activity_db_failure(self):
        await self.content.begin_batch(self.proposal()[1])
        self.content.runs[0].nodes['65536']=100
        with patch.object(predictions,'record_targets',side_effect=RuntimeError('auth unavailable')):
            with self.assertRaises(RuntimeError):await self.content.advance_batch()
        self.assertEqual(self.content.store.load_batch()['target_results'],{'lume':'reached'})
        restored=MultiAiContent(self.content.room);await restored.start();restored.hub=Hub(restored)
        await restored.advance_batch(force=True)
        self.assertEqual(predictions.listing(self.hub.room.id)['market']['target_bet']['outcomes'],{'lume':'reached'})

    async def test_combo_requires_coexisting_tiles_and_upgrades_persisted_target(self):
        await self.content.begin_batch(self.proposal()[1])
        run=self.content.runs[0]
        run.nodes.update({'32768':50,'65536':100})
        run.board=[65536,2]+[0]*14
        await self.content.advance_batch()
        self.assertEqual(self.content.batch['target_results']['lume'],'reached')
        run.board=[65536,32768]+[0]*14
        with patch.object(predictions,'record_targets',side_effect=RuntimeError('auth unavailable')):
            with self.assertRaises(RuntimeError):await self.content.advance_batch()
        self.assertEqual(self.content.store.load_batch()['target_results']['lume'],'reached_combo')
        restored=MultiAiContent(self.content.room);await restored.start();restored.hub=Hub(restored)
        await restored.advance_batch(force=True)
        self.assertEqual(predictions.listing(self.hub.room.id)['market']['target_bet']['outcomes']['lume'],'reached_combo')
        await restored.advance_batch(force=True)
        self.assertEqual(restored.batch['target_results']['lume'],'reached_combo')

    async def test_recovery_reconciles_saved_batch_when_auth_creation_failed(self):
        runs,proposal=self.proposal()
        with patch.object(predictions,'ensure_market',side_effect=RuntimeError('auth unavailable')):
            with self.assertRaises(RuntimeError):await self.content.begin_batch(proposal)
        restored=MultiAiContent(self.content.room);await restored.start()
        restored.hub=Hub(restored)
        await restored.advance_batch(force=True)
        self.assertEqual(predictions.listing(self.hub.room.id)['market']['id'],proposal['batch_id'])
        self.assertEqual([r.id for r in restored.runs],[r.id for r in runs])

    async def test_transition_preserves_runs_and_does_not_open_market(self):
        runs=[LiveRun() for _ in range(3)]
        self.content.runs=runs;self.content.generations=[1]*3;await self.content.save()
        restored=MultiAiContent(self.content.room);await restored.start();restored.hub=Hub(restored)
        await restored.advance_batch(force=True)
        self.assertTrue(restored.batch['transition'])
        self.assertEqual([r.id for r in restored.runs],[r.id for r in runs])
        self.assertIsNone(predictions.listing(self.hub.room.id)['market'])

    async def test_active_batch_reconnect_uses_cloud_checkpoint_without_changing_seed(self):
        await self.content.begin_batch(self.proposal()[1])
        cloud=self.content.runs[0];ahead=LiveRun.restore(cloud.checkpoint())
        ahead.apply(ahead.make_step(legal_moves(ahead.board)[0],80))
        result=await self.content.sync_lane(dict(lane=0,generation=1,run=ahead.checkpoint()))
        self.assertEqual(result['run']['records'],'')
        self.assertEqual(result['run']['seed'],cloud.seed)

    async def test_void_is_idempotent_and_survives_restore(self):
        await self.content.begin_batch(self.proposal()[1])
        await self.content.void_batch();await self.content.void_batch()
        self.assertEqual(predictions.listing(self.hub.room.id)['market']['status'],'void')
        restored=MultiAiContent(self.content.room);await restored.start()
        self.assertEqual(restored.batch['phase'],'void')

    async def test_room_giveaway_does_not_require_an_ai_or_milestone(self):
        spec=dict(pool=10000,minimum=100,maximum=1000)
        first=lucky_bags.create_activity('viewer-event:123',spec,room_id='human-room')
        repeated=lucky_bags.create_activity('viewer-event:123',spec,room_id='human-room')
        self.assertEqual(first['id'],repeated['id']);self.assertEqual(first['milestone'],0)
        self.assertEqual(len(lucky_bags.listing(room_id='human-room')),1)
        self.assertEqual(lucky_bags.listing(room_id=self.hub.room.id),[])

    async def test_worker_ignores_late_status_while_waiting_for_batch_ack(self):
        runner=object.__new__(MultiRunner)
        runner.batch=dict(id='expected',phase='preparing');runner.runs=[LiveRun() for _ in range(3)]
        runner.batch_ack=asyncio.Event();runner.batch_active=asyncio.Event();runner.save=lambda:None
        await runner.receive_batch(dict(type='batch_state',batch=dict(id='old',phase='cooldown')))
        self.assertEqual(runner.batch['id'],'expected');self.assertFalse(runner.batch_ack.is_set())
        await runner.receive_batch(dict(type='batch_ready',batch=dict(id='expected',phase='running',started_at=123)))
        self.assertEqual([r.started for r in runner.runs],[123]*3)
        self.assertTrue(runner.batch_active.is_set());self.assertTrue(runner.batch_ack.is_set())

    async def test_coordinator_and_server_complete_two_real_replay_batches(self):
        upstream,downstream=asyncio.Queue(),asyncio.Queue()
        class ServerSocket:
            async def accept(self):pass
            async def close(self,code):raise AssertionError(f'unexpected publisher close {code}')
            async def receive(self):return await upstream.get()
            async def send_json(self,data):await downstream.put(json.dumps(data))
        class ClientSocket:
            async def send(self,value):await upstream.put(dict(type='websocket.receive',**({'bytes':value} if isinstance(value,bytes) else {'text':value})))
            async def recv(self):return await downstream.get()
        runner=MultiRunner(SimpleNamespace(checkpoint_dir=self.content.room.path.parent/'worker',legacy_checkpoint=self.content.room.path.parent/'none',tables=None,interval=0,search_interval=0))
        ready=set()
        class Engine:
            process=None
            def __init__(self,lane):self.lane=lane
            async def start(self):self.process=True;ready.add(self.lane)
            async def choose(self,run,generation,revision):
                self_test.assertEqual(ready,{0,1,2})
                await asyncio.sleep(0)
                return legal_moves(run.board)[0],'AI'
            def close(self):self.process=None
        self_test=self
        runner.children=[Engine(i) for i in range(3)]
        server=asyncio.create_task(self.content.publish(ServerSocket(),self.hub))
        with patch('tools.live_multi_runner.wait_for_step',new=AsyncMock()):
            worker=asyncio.create_task(runner.connected(ClientSocket()))
            async def until(predicate):
                async with asyncio.timeout(10):
                    while not predicate():
                        if server.done():server.result();self.fail('server stopped')
                        if worker.done():worker.result();self.fail('worker stopped')
                        await asyncio.sleep(.01)
            try:
                await until(lambda:self.content.batch and self.content.batch['phase']=='cooldown')
                first=self.content.batch['id']
                self.assertTrue(all(r.ended for r in self.content.runs))
                self.assertEqual(predictions.listing(self.hub.room.id)['market']['status'],'settled')
                await until(lambda:runner.batch and runner.batch['phase']=='cooldown')
                self.content.batch['next_at']=0;runner.batch['next_at']=0
                await until(lambda:self.content.batch['id']!=first and self.content.batch['phase']=='cooldown')
                self.assertEqual(self.content.generations,[2,2,2])
                self.assertEqual(len({r.started for r in self.content.runs}),1)
                self.assertTrue(all(r.ended for r in self.content.runs))
            finally:
                worker.cancel();server.cancel();await asyncio.gather(worker,server,return_exceptions=True)


if __name__=='__main__':unittest.main()
