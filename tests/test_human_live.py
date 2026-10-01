import os
import tempfile
import unittest
import asyncio
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException

from backend.auth.db import auth_db, init_auth_db
from backend.gifts import service as gift_service
from backend.live import gifts, human_rooms
from backend.live import routes as live_routes
from backend.live.human_content import HumanPlayContent, sanitize_appearance
from backend.live.rooms import RoomDefinition
from backend.human_play import engine
from backend.gamer_ranked.prng import Xoshiro128StarStar
from backend.room_activities import lucky_bags


class HumanLiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            'CLOUD_AUTH_DB': str(Path(self.directory.name) / 'auth.db'),
            'CLOUD_TOKEN_GLOBAL_MULTIPLIER': '1',
            'HUMAN_LIVE_SIGNING_KEY': 'test-key-' + 'x' * 40,
        })
        self.env.start(); init_auth_db(); gifts.init_schema(); human_rooms.init_schema(); lucky_bags.init_schema()
        with auth_db() as db:
            for user_id, name in ((1, 'Viewer'), (2, 'Streamer')):
                db.execute('''INSERT INTO users(id,email,email_identity,password_hash,display_name,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?)''', (user_id, f'{user_id}@test.invalid', f'{user_id}@test.invalid', 'x', name, 'now', 'now'))
                db.execute('''INSERT INTO token_accounts(user_id,bonus_balance_units,paid_balance_units,created_at,updated_at)
                    VALUES(?,0,1000000,'now','now')''', (user_id,))

    def tearDown(self):
        self.env.stop(); self.directory.cleanup()

    def request(self, gift='heart'):
        catalog, _ = gifts.catalogue(); item = next(row for row in catalog['gifts'] if row['id'] == gift)
        return {'request_id': str(uuid4()), 'gift_id': gift, 'quantity': 1,
                'expected_cost_units': item['totals'][0], 'quote_version': catalog['version']}

    def test_room_is_reused_but_old_lease_is_revoked_on_run_switch(self):
        first = human_rooms.start(owner_user_id=2, run_id='run-one', variant='4x4', display_name='Streamer')
        lease, _ = human_rooms.issue_lease(first, seed='1' * 32, writer='writer-one-000000', epoch=1, browser='browser-one')
        self.assertEqual(human_rooms.verify_lease(lease, room_id=first['room_id'], owner_user_id=2)['run'], 'run-one')
        second = human_rooms.start(owner_user_id=2, run_id='run-two', variant='4x4', display_name='Streamer')
        self.assertEqual(second['room_id'], first['room_id'])
        self.assertGreater(second['generation'], first['generation'])
        with self.assertRaises(HTTPException):
            human_rooms.verify_lease(lease, room_id=first['room_id'], owner_user_id=2)

    def test_stale_room_expires_after_reconnect_window_but_recent_room_survives(self):
        stale = human_rooms.start(owner_user_id=1, run_id='run-old', variant='4x4',
                                  display_name='Viewer', now=100)
        recent = human_rooms.start(owner_user_id=2, run_id='run-new', variant='3x4',
                                   display_name='Streamer', now=180)
        human_rooms.publisher_seen(recent['room_id'], recent['generation'], now=195)
        expired = human_rooms.expire_stale(stale_after=90, now=200)
        self.assertEqual(expired, [stale['room_id']])
        self.assertIsNone(human_rooms.room(stale['room_id']))
        self.assertEqual(human_rooms.room(recent['room_id'])['run_id'], 'run-new')

    def test_switch_in_long_running_room_gets_a_new_recovery_window(self):
        first = human_rooms.start(owner_user_id=2, run_id='old', variant='4x4', display_name='Streamer', now=100)
        second = human_rooms.start(owner_user_id=2, run_id='new', variant='2x4', display_name='Streamer', now=1000)
        self.assertEqual(second['started_at'], first['started_at'])
        self.assertEqual(human_rooms.expire_stale(stale_after=90, now=1001), [])
        self.assertEqual(human_rooms.expire_stale(stale_after=90, now=1091), [first['room_id']])

    def test_human_generation_change_preserves_hub_and_viewers(self):
        first = human_rooms.start(owner_user_id=2, run_id='old', variant='4x4', display_name='Streamer')
        runtime = live_routes._dynamic_hub(first['room_id'])
        viewer = object()
        runtime.viewers[viewer] = live_routes.ViewerQueue()
        human_rooms.start(owner_user_id=2, run_id='new', variant='2x4', display_name='Streamer')
        self.assertIs(live_routes._dynamic_hub(first['room_id']), runtime)
        self.assertIn(viewer, runtime.viewers)
        self.assertEqual(runtime.room.metadata['variant'], '2x4')
        live_routes.dynamic_hubs.pop(first['room_id'], None)

    def test_ended_room_notifies_existing_viewers_once(self):
        room = human_rooms.start(owner_user_id=2, run_id='run-ended', variant='4x4',
                                 display_name='Streamer')
        runtime = live_routes._dynamic_hub(room['room_id'])
        queue = live_routes.ViewerQueue()
        runtime.viewers[object()] = queue
        self.assertTrue(human_rooms.stop(2, room['room_id']))

        asyncio.run(live_routes._reconcile_dynamic_room(
            room['room_id'], runtime, None, now=1))
        asyncio.run(live_routes._reconcile_dynamic_room(
            room['room_id'], runtime, None, now=2))
        self.assertEqual(queue.get_nowait()['type'], 'room_ended')
        self.assertTrue(queue.empty())
        live_routes.dynamic_hubs.pop(room['room_id'], None)

    def test_only_paid_token_spend_credits_half_to_streamer_atomically(self):
        room = human_rooms.start(owner_user_id=2, run_id='run-one', variant='4x4', display_name='Streamer')
        before = 1000000
        result = gifts.send({'id': 1, 'display_name': 'Viewer'}, self.request(), True,
                            target='live:' + room['room_id'])
        self.assertEqual(result['creator_share_units'], result['cost_units'] // 2)
        with auth_db() as db:
            streamer = db.execute('SELECT paid_balance_units FROM token_accounts WHERE user_id=2').fetchone()[0]
            share = db.execute('SELECT * FROM live_creator_shares').fetchone()
            ledger = db.execute("SELECT * FROM token_ledger WHERE event_type='live_creator_share'").fetchone()
        self.assertEqual(streamer, before + result['creator_share_units'])
        self.assertEqual(share['share_units'], result['creator_share_units'])
        self.assertEqual(ledger['paid_delta_units'], result['creator_share_units'])

    def test_bonus_token_spend_does_not_credit_streamer_and_self_gift_is_rejected(self):
        room = human_rooms.start(owner_user_id=2, run_id='run-one', variant='4x4', display_name='Streamer')
        with auth_db() as db:
            db.execute('UPDATE token_accounts SET bonus_balance_units=1000000,paid_balance_units=0 WHERE user_id=1')
        result = gifts.send({'id': 1, 'display_name': 'Viewer'}, self.request(), True,
                            target='live:' + room['room_id'])
        self.assertEqual(result['creator_share_units'], 0)
        with self.assertRaises(HTTPException) as error:
            gifts.send({'id': 2, 'display_name': 'Streamer'}, self.request(), True,
                       target='live:' + room['room_id'])
        self.assertEqual(error.exception.detail, 'self_gift_not_allowed')

    def test_human_lucky_bag_is_idempotent_across_room_ids(self):
        first = lucky_bags.create_human('same-run', 32768, room_id='h-one')
        second = lucky_bags.create_human('same-run', 32768, room_id='h-two')
        self.assertEqual(first['id'], second['id'])
        with auth_db() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM live_lucky_bags WHERE run_id='human:same-run'").fetchone()[0], 1)


class _PublishSocket:
    def __init__(self, hello, prefix, step):
        self.hello = hello; self.prefix = prefix; self.messages = [
            {'type': 'websocket.receive', 'bytes': step},
            {'type': 'websocket.disconnect'},
        ]; self.sent=[]; self.closed=None
    async def accept(self): pass
    async def receive_text(self): return self.hello
    async def receive_bytes(self): return self.prefix
    async def receive(self): return self.messages.pop(0)
    async def send_json(self, value): self.sent.append(value)
    async def close(self, code=1000): self.closed=code


class _PublishHub:
    producer=None; producer_ready=False; control_supported=False; control_ack=None; last_seen=0
    def __init__(self): self.events=[]
    async def publisher_joined(self): pass
    def broadcast(self, value): self.events.append(value)
    def snapshot(self): return {'type':'snapshot', **self.content.snapshot()}


class HumanPublishProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.env=patch.dict(os.environ, {'CLOUD_AUTH_DB':str(Path(self.directory.name)/'auth.db'),
            'HUMAN_LIVE_SIGNING_KEY':'test-key-'+'y'*40})
        self.env.start();init_auth_db();human_rooms.init_schema()
        with auth_db() as db:
            db.execute('''INSERT INTO users(id,email,email_identity,password_hash,display_name,created_at,updated_at)
                VALUES(1,'1@test','1@test','x','Player','now','now')''')

    async def asyncTearDown(self): self.env.stop();self.directory.cleanup()

    async def test_prefix_and_binary_tail_rebuild_rectangular_game(self):
        data=human_rooms.start(owner_user_id=1,run_id='run',variant='3x3',display_name='Player')
        lease,_=human_rooms.issue_lease(data,seed='00000001000000020000000300000004',
            writer='writer-0000000001',epoch=1,browser='browser')
        state=engine.initial('run','3x3','00000001000000020000000300000004')
        for direction in range(4):
            moved,_=engine.move(state['board'],3,3,direction)
            if moved!=state['board']:
                rng=Xoshiro128StarStar(list(state['rng']));index,value=engine.spawn(moved,rng)
                step=engine.EVENT.pack(direction|(index<<2)|(64 if value==4 else 0),123);break
        room=RoomDefinition(id=data['room_id'],title={'en':'x'},content_kind='human-play',protocol='human-play-v1',dynamic=True,
            metadata={'owner_user_id':1,'variant':'3x3','streamer':{'display_name':'Player'}})
        content=HumanPlayContent(room);hub=_PublishHub();hub.content=content
        socket=_PublishSocket(__import__('json').dumps({'type':'hello','lease':lease,'seq':0,'started_at':1}),b'HLP1\0',step)
        await content.publish(socket,hub,{'id':1})
        self.assertEqual(content.run.seq,1)
        self.assertEqual((content.run.snapshot()['rows'],content.run.snapshot()['cols']),(3,3))
        self.assertTrue(any(isinstance(value,bytes) and len(value)==9 for value in hub.events))

    async def test_signed_switch_reuses_socket_and_resets_milestones(self):
        import json
        seed = '00000001000000020000000300000004'
        first = human_rooms.start(owner_user_id=1, run_id='old', variant='4x4', display_name='Player')
        old_lease, _ = human_rooms.issue_lease(first, seed=seed, writer='writer-0000000001', epoch=1, browser='browser')
        room = RoomDefinition(id=first['room_id'], title={'en':'x'}, content_kind='human-play',
            protocol='human-play-v1', dynamic=True, metadata={})
        content = HumanPlayContent(room)
        hub = _PublishHub(); hub.content = content
        class SwitchSocket(_PublishSocket):
            async def receive(self):
                if len(self.sent) == 1:
                    content.verified_milestones.add(32768)
                    second = human_rooms.start(owner_user_id=1, run_id='new', variant='2x4', display_name='Player')
                    lease, _ = human_rooms.issue_lease(second, seed=seed, writer='writer-0000000001', epoch=1, browser='browser')
                    return {'type':'websocket.receive', 'text':json.dumps({'type':'switch','lease':lease,'seq':0})}
                self_same.assertIs(hub.producer, self)
                self_same.assertFalse(content.verified_milestones)
                return {'type':'websocket.disconnect'}
        self_same = self
        socket = SwitchSocket(json.dumps({'type':'hello','lease':old_lease,'seq':0}), b'HLP1\0', b'')
        await content.publish(socket, hub, {'id':1})
        self.assertIsNone(socket.closed)
        self.assertEqual([ack['run_id'] for ack in socket.sent], ['old', 'new'])
        self.assertEqual(content.run.variant, '2x4')

    async def test_switch_rejects_another_owner_and_preserves_previous_run(self):
        import json
        data = human_rooms.start(owner_user_id=1, run_id='old', variant='4x4', display_name='Player')
        lease, _ = human_rooms.issue_lease(data, seed='00000001000000020000000300000004',
            writer='writer-0000000001', epoch=1, browser='browser')
        room = RoomDefinition(id=data['room_id'], title={'en':'x'}, content_kind='human-play', metadata={})
        content = HumanPlayContent(room); hub = _PublishHub(); hub.content = content
        socket = _PublishSocket('{}', b'HLP1\0', b'')
        await content._install_run(socket, hub, {'lease':lease,'seq':0}, {'id':1})
        previous = content.run
        with self.assertRaises(HTTPException):
            await content._install_run(socket, hub, {'lease':lease,'seq':0}, {'id':2})
        self.assertIs(content.run, previous)

    def test_streamer_tile_appearance_is_sanitized_and_persisted_in_snapshot(self):
        appearance = {'version': 1, 'empty': {'background': '#716E69', 'color': '#B5B1AC'},
                      'tiles': {'2': {'background': '#EEE4DA', 'color': '#776E65'}}}
        normalized = sanitize_appearance(appearance)
        self.assertEqual(normalized['tiles']['2']['background'], '#eee4da')
        run = __import__('backend.live.human_content', fromlist=['HumanStreamRun']).HumanStreamRun(
            run_id='theme-run', variant='4x4', seed='00000001000000020000000300000004',
            started_at=1, source='Player', raw=b'', appearance=normalized, best_score=123456)
        self.assertEqual(run.snapshot()['appearance'], normalized)
        self.assertEqual(run.snapshot()['best_score'], 123456)
        with self.assertRaises(ValueError):
            sanitize_appearance({'version': 1, 'empty': {'background': 'url(x)', 'color': '#ffffff'}, 'tiles': {}})

    def test_stream_best_tracks_current_score_and_cannot_be_downgraded(self):
        module = __import__('backend.live.human_content', fromlist=['HumanStreamRun'])
        run = module.HumanStreamRun(
            run_id='best-run', variant='4x4', seed='00000001000000020000000300000004',
            started_at=1, source='Player', raw=b'', best_score=0)
        run.state['score'] = 1200
        self.assertEqual(run.snapshot()['best_score'], 1200)
        run.best_score = max(run.best_score, module.sanitize_best_score(run.state.get('score')),
                             module.sanitize_best_score(400))
        self.assertEqual(run.best_score, 1200)


if __name__ == '__main__':
    unittest.main()
