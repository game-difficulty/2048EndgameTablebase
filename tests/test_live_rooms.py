import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from backend.auth.db import auth_db, init_auth_db
from backend.auth.principal import ActorRef
from backend.gamer_ranked.rules import legal_moves
from backend.live import routes, gifts, audience, red_envelopes, lucky_bags
from backend.live.content import CONTENT_FACTORIES
from backend.live.rooms import RoomDefinition, DEFAULT_ROOM, register_room
from backend.live.protocol import LiveRun
from backend.gifts import service


class LiveRoomTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = patch.dict(os.environ, {
            'CLOUD_AUTH_DB': str(Path(self.directory.name) / 'auth.db'),
            'LIVE_DB_PATH': str(Path(self.directory.name) / 'live.db'),
            'LIVE_PUBLISH_TOKEN': 'default-secret', 'ROOM_SECOND_SECRET': 'second-secret',
            'CLOUD_TOKEN_GLOBAL_MULTIPLIER': '1',
        })
        self.env.start(); self.addCleanup(self.env.stop)
        init_auth_db()
        self.second = RoomDefinition('second-ai', {'zh': '第二房间', 'en': 'Second room'}, publish_token_env='ROOM_SECOND_SECRET')
        self.registry = patch.dict(routes.ROOMS, {DEFAULT_ROOM.id: DEFAULT_ROOM, self.second.id: self.second}, clear=True)
        self.registry.start(); self.addCleanup(self.registry.stop)
        self.runtime = patch.object(routes, 'hub', routes.LiveHub())
        self.runtime.start(); self.addCleanup(self.runtime.stop)

        @asynccontextmanager
        async def lifespan(app):
            await routes.start_rooms()
            try: yield
            finally: await routes.stop_rooms()

        app = FastAPI(lifespan=lifespan)
        app.include_router(routes.router)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(lambda: self.client.__exit__(None, None, None))
        async def pause_outbox_consumers():
            for runtime in [routes.hub, *routes.room_hubs.values()]:
                runtime.gift_task.cancel()
                try: await runtime.gift_task
                except asyncio.CancelledError: pass
        self.client.portal.call(pause_outbox_consumers)
        with auth_db() as db:
            for uid in [1, 2]:
                db.execute('INSERT INTO users(id,email,email_identity,password_hash,display_name,created_at,updated_at) VALUES(?,?,?,?,?,?,?)',
                           (uid, f'{uid}@test.invalid', f'{uid}@test.invalid', 'test', 'Tester', 'now', 'now'))
                db.execute("INSERT INTO token_accounts(user_id,bonus_balance_units,paid_balance_units,created_at,updated_at) VALUES(?,0,100000000,'now','now')", (uid,))
        self.user = {'id': 1, 'display_name': 'Tester'}

    def purchase(self):
        catalog, _ = service.catalogue()
        gift = next(item for item in catalog['gifts'] if item['id'] == 'heart')
        return dict(request_id=str(uuid4()), gift_id='heart', quantity=1,
                    expected_cost_units=gift['totals'][0], quote_version=catalog['version'])

    def test_discovery_legacy_alias_unknown_room_and_credentials(self):
        rooms = self.client.get('/api/live/rooms').json()['rooms']
        self.assertEqual(len(rooms), 2)
        self.assertNotIn('publish_token_env', rooms[0])
        for path in ['/api/live/state', '/api/live/rooms/ai-classic/state']:
            self.assertEqual(self.client.get(path).json()['room_id'], 'ai-classic')
        for path in ['', '/state', '/gifts/catalog', '/like']:
            response = self.client.post('/api/live/rooms/missing'+path, json={}) if path == '/like' else self.client.get('/api/live/rooms/missing'+path)
            self.assertEqual(response.status_code, 404)
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/api/live/rooms/second-ai/publish', headers={'authorization': 'Bearer default-secret'}): pass
        self.assertNotEqual(routes.hub.store.path, routes.room_hubs['second-ai'].store.path)
        with self.assertRaises(ValueError):
            register_room(RoomDefinition('untrusted', {}, publish_token_env='LIVE_PUBLISH_TOKEN'))

    def test_broadcast_steps_presence_and_chat_are_room_local(self):
        first = '/api/live/rooms/ai-classic'; second = '/api/live/rooms/second-ai'
        with self.client.websocket_connect(first+'/watch') as watcher, self.client.websocket_connect(second+'/watch') as other:
            self.assertEqual(watcher.receive_json()['room_id'], 'ai-classic')
            other.receive_json()
            with self.client.websocket_connect(first+'/publish', headers={'authorization': 'Bearer default-secret'}) as publisher:
                publisher.send_json({'type': 'hello'}); publisher.receive_json(); watcher.receive_json()
                run = LiveRun()
                publisher.send_json({'type': 'start', 'seed': run.seed, 'run_id': run.id})
                self.assertEqual(watcher.receive_json()['run']['seq'], 0)
                packet = run.make_step(legal_moves(run.board)[0], 80)
                publisher.send_bytes(packet)
                self.assertEqual(watcher.receive_bytes(), packet)
                self.assertEqual(len(packet), 9)
                actor = ActorRef.from_guest({'guest_id': 'second', 'display_name': 'Guest'})
                with patch.object(routes, 'require_actor', return_value=actor):
                    self.client.post(second+'/chat', json={'text': 'only second'})
                # If any default-room snapshot/step leaked this would not be the chat event.
                self.assertEqual(other.receive_json()['text'], 'only second')
                state = self.client.get(second+'/state').json()
                self.assertIsNone(state['run']); self.assertFalse(state['online'])
                self.assertEqual(self.client.get(first+'/state').json()['chat'], [])

    def test_likes_and_contributions_are_isolated(self):
        actor = ActorRef.from_guest({'guest_id': 'actor', 'display_name': 'Guest'})
        with patch.object(routes, 'require_actor', return_value=actor):
            self.client.post('/api/live/rooms/second-ai/like', json={'count': 3})
        self.assertEqual(self.client.get('/api/live/state').json()['likes'], 0)
        self.assertEqual(self.client.get('/api/live/rooms/second-ai/state').json()['likes'], 3)
        audience.record('u:1', 'likes', 5, room_id='second-ai')
        identities = {'u:1': {'name': 'Tester'}}
        self.assertEqual(audience.ranking(identities)['viewers'][0]['contribution_units'], 0)
        self.assertEqual(audience.ranking(identities, room_id='second-ai')['viewers'][0]['contribution_units'], 5000)

    def test_gift_target_idempotency_combo_outbox_and_global_budget(self):
        request = self.purchase()
        a = gifts.send(self.user, request, True)
        self.assertEqual(a, gifts.send(self.user, request, False))
        with self.assertRaises(HTTPException) as conflict:
            gifts.send(self.user, request, True, target='live:second-ai')
        self.assertEqual(conflict.exception.status_code, 409)
        b = gifts.send(self.user, self.purchase(), True, target='live:second-ai')
        self.assertNotEqual(a['combo_id'], b['combo_id'])
        self.assertEqual(a['combo_count'], b['combo_count'])
        self.assertEqual(len(service.pending_events('live:ai-classic')), 1)
        self.assertEqual(len(service.pending_events('live:second-ai')), 1)
        with self.assertRaises(HTTPException): service.order(1, a['request_id'], 'live:second-ai')
        gifts.set_preferences(1, request['expected_cost_units']*2, True)
        with self.assertRaises(HTTPException) as budget:
            gifts.send(self.user, self.purchase(), True, target='live:second-ai')
        self.assertEqual(budget.exception.detail, 'gift_daily_budget')

    def test_gift_service_can_serve_battle_without_live_hub(self):
        request = self.purchase()
        receipt = service.send(self.user, request, True, target='battle:example:player-2')
        self.assertEqual(receipt['target'], 'battle:example:player-2')
        self.assertEqual(len(service.pending_events('battle:example:player-2')), 1)
        self.assertEqual(service.pending_events(), [])

    def test_legacy_gift_schema_migrates_without_changing_retry_identity(self):
        request = self.purchase()
        original = gifts.send(self.user, request, True)
        with auth_db() as db:
            db.execute('DROP INDEX gift_target_pending')
            db.execute('DROP INDEX gift_target_combo')
            db.execute('ALTER TABLE live_gift_orders DROP COLUMN target')
        service.init_schema()
        self.assertEqual(gifts.send(self.user, request, False), original)
        self.assertEqual(len(service.pending_events()), 1)
        self.assertEqual(service.pending_events('live:second-ai'), [])

    def test_reward_fifo_and_claims_are_room_local(self):
        body = dict(request_id=str(uuid4()), amount=1000, count=5, mode='equal')
        a = red_envelopes.create(1, {'name': 'A'}, body, now=100)
        b = red_envelopes.create(1, {'name': 'A'}, {**body, 'request_id': str(uuid4())}, now=120, room_id='second-ai')
        self.assertEqual(red_envelopes.tick(now=121)['active']['id'], a['id'])
        self.assertEqual(red_envelopes.tick(now=121, room_id='second-ai')['active']['id'], b['id'])
        with self.assertRaises(HTTPException): red_envelopes.claim(a['id'], 2, now=121, room_id='second-ai')
        self.assertEqual(len(red_envelopes.events(room_id='second-ai')), 1)
        bag = lucky_bags.create('same-run', 32768, now=100)
        other = lucky_bags.create('same-run', 32768, now=100, room_id='second-ai')
        self.assertNotEqual(bag['id'], other['id'])
        self.assertEqual(len(lucky_bags.listing(now=101)), 1)
        self.assertEqual(len(lucky_bags.listing(now=101, room_id='second-ai')), 1)
        with self.assertRaises(HTTPException): lucky_bags.join(bag['id'], 2, now=101, room_id='second-ai')

    def test_slow_viewer_reconnect_snapshot_covers_dropped_steps(self):
        hub = routes.room_hubs['second-ai']
        run = LiveRun(); hub.run = run
        queue = asyncio.Queue(maxsize=2); key = object(); hub.viewers[key] = queue
        try:
            queue.put_nowait(b'old'); queue.put_nowait(b'old')
            packet = run.make_step(legal_moves(run.board)[0], 80); run.apply(packet)
            hub.broadcast(packet)
            self.assertIsNone(queue.get_nowait())
            snapshot = hub.snapshot()
            self.assertEqual(snapshot['room_id'], 'second-ai')
            self.assertEqual(snapshot['run']['seq'], 1)
            next_packet = run.make_step(legal_moves(run.board)[0], 80); run.apply(next_packet)
            hub.viewers[key] = queue = asyncio.Queue(maxsize=2)
            hub.broadcast(next_packet)
            self.assertEqual(queue.get_nowait(), next_packet)
        finally: del hub.viewers[key]

    def test_room_shell_accepts_content_without_a_classic_run(self):
        class AlternativeContent:
            def __init__(self, room): self.store = None
            run_id = 'multi-game'
            def snapshot(self): return {'lanes': [{'id': 1, 'seq': 8}, {'id': 2, 'seq': 13}]}
        definition = RoomDefinition('alternative', {'en': 'Alternative'}, content_kind='alternative', protocol='test-v1')
        with patch.dict(CONTENT_FACTORIES, {('alternative', 'test-v1'): AlternativeContent}):
            hub = routes.LiveHub(definition)
            snapshot = hub.snapshot()
            self.assertNotIn('run', snapshot)
            self.assertEqual([lane['seq'] for lane in snapshot['lanes']], [8, 13])
            self.assertEqual(snapshot['room_id'], 'alternative')
            self.assertEqual(hub.control_status()['run_id'], 'multi-game')
