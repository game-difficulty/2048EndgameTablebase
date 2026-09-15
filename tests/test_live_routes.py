from contextlib import asynccontextmanager
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.auth.principal import ActorRef
from backend.auth.db import init_auth_db
from backend.live import routes
from backend.live.protocol import LiveRun
from backend.gamer_ranked.rules import legal_moves


class LiveRouteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'CLOUD_AUTH_DB': str(Path(self.directory.name)/'auth.db'), 'LIVE_DB_PATH': str(Path(self.directory.name)/'live.db'), 'LIVE_PUBLISH_TOKEN': 'local-test-secret'})
        self.env.start()
        init_auth_db()
        self.hub_patch = patch.object(routes, 'hub', routes.LiveHub())
        self.hub_patch.start()

        @asynccontextmanager
        async def lifespan(app):
            await routes.hub.start()
            yield
            await routes.hub.stop()

        app = FastAPI(lifespan=lifespan)
        app.include_router(routes.router)
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.hub_patch.stop()
        self.env.stop()
        self.directory.cleanup()

    def test_single_producer_two_watchers_and_disconnect(self):
        with self.client.websocket_connect('/api/live/watch') as first, self.client.websocket_connect('/api/live/watch') as second:
            self.assertFalse(first.receive_json()['online'])
            second.receive_json()
            with self.client.websocket_connect('/api/live/publish', headers={'authorization': 'Bearer local-test-secret'}) as producer:
                producer.send_json({'type':'hello'})
                self.assertIsNone(producer.receive_json()['run'])
                first.receive_json()
                second.receive_json()
                run = LiveRun()
                producer.send_json({'type':'start','seed':run.seed,'run_id':run.id})
                self.assertEqual(first.receive_json()['run']['board'], run.board)
                second.receive_json()
                packet = run.make_step(legal_moves(run.board)[0], 50)
                producer.send_bytes(packet)
                self.assertEqual(first.receive_bytes(), packet)
                self.assertEqual(second.receive_bytes(), packet)
                run.apply(packet)
            self.assertFalse(first.receive_json()['online'])
            second.receive_json()
            with self.client.websocket_connect('/api/live/publish', headers={'authorization':'Bearer local-test-secret'}) as producer:
                producer.send_json({'type':'hello','run':run.checkpoint()})
                self.assertEqual(producer.receive_json()['run']['records'], run.checkpoint()['records'])

    def test_chat_guest_limits_plain_text_and_origin(self):
        actor = ActorRef.from_guest({'guest_id':'test','display_name':'Guest-test'})
        with patch.object(routes, 'require_actor', return_value=actor):
            self.assertEqual(self.client.post('/api/live/chat',json={'text':'x'*33}).status_code,400)
            self.assertEqual(self.client.post('/api/live/chat',json={'text':'hi'},headers={'Origin':'https://evil.invalid'}).status_code,403)
            for _ in range(5):
                self.assertEqual(self.client.post('/api/live/chat',json={'text':'<b>hello</b>'}).status_code,200)
            self.assertEqual(self.client.post('/api/live/chat',json={'text':'six'}).status_code,429)
            state = self.client.get('/api/live/state').json()
            self.assertEqual(state['chat'][0]['text'],'<b>hello</b>')
            self.assertTrue(state['chat'][0]['guest'])

    def test_missing_replay_and_unauthenticated_chat(self):
        self.assertEqual(self.client.get('/api/live/replays/missing').status_code,404)
        with patch.object(routes, 'require_actor', side_effect=routes.HTTPException(401)):
            self.assertEqual(self.client.post('/api/live/chat',json={'text':'hi'}).status_code,401)

    def test_like_returns_total_before_periodic_broadcast(self):
        actor = ActorRef.from_guest({'guest_id': 'likes', 'display_name': 'Guest-likes'})
        with patch.object(routes, 'require_actor', return_value=actor):
            response = self.client.post('/api/live/like', json={})
            self.assertEqual(response.json(), {'ok': True, 'count': 1, 'accepted': 1})
            self.assertEqual(self.client.get('/api/live/state').json()['likes'], 1)
            response = self.client.post('/api/live/like', json={'count': 19})
            self.assertEqual(response.json(), {'ok': True, 'count': 20, 'accepted': 19})
            self.assertEqual(self.client.post('/api/live/like', json={}).status_code, 429)
            self.assertEqual(self.client.get('/api/live/state').json()['likes'], 20)

    def test_like_batch_validates_count_and_ip_budget(self):
        for value in [True, 0, -1, 21, 1.5, '2']:
            actor = ActorRef.from_guest({'guest_id': 'invalid', 'display_name': 'Guest'})
            with patch.object(routes, 'require_actor', return_value=actor):
                self.assertEqual(self.client.post('/api/live/like', json={'count': value}).status_code, 400)
        for i in range(4):
            actor = ActorRef.from_guest({'guest_id': str(i), 'display_name': 'Guest'})
            with patch.object(routes, 'require_actor', return_value=actor):
                response = self.client.post('/api/live/like', json={'count': 20})
                self.assertEqual(response.status_code, 200 if i < 3 else 429)
        self.assertEqual(self.client.get('/api/live/state').json()['likes'], 60)
        self.assertFalse(routes.hub.rates.get(('like', actor.actor_key)))

    def test_recovered_death_saves_history_once_and_restarts(self):
        run = LiveRun('00000001000000020000000300000004')
        while legal_moves(run.board):
            run.apply(run.make_step(legal_moves(run.board)[0], 50))
        run.end()
        headers = {'authorization': 'Bearer local-test-secret'}
        for _ in range(2):
            with self.client.websocket_connect('/api/live/publish', headers=headers) as producer:
                producer.send_json({'type': 'hello', 'run': run.checkpoint()})
                self.assertIsNotNone(producer.receive_json()['run']['ended'])
        state = self.client.get('/api/live/state').json()
        self.assertEqual(state['today']['games'], 1)
        self.assertEqual(state['history'][0]['score'], run.score)
        self.assertFalse(state['online'])
        self.assertEqual(self.client.get('/api/live/replays/' + run.id).text, run.replay())
        with self.client.websocket_connect('/api/live/publish', headers=headers) as producer:
            producer.send_json({'type': 'hello'})
            producer.receive_json()
            new = LiveRun()
            producer.send_json({'type': 'start', 'run_id': new.id, 'seed': new.seed})
            # A subsequent hello provides an ordered acknowledgement of start.
            producer.send_json({'type': 'hello'})
            self.assertEqual(producer.receive_json()['run']['run_id'], new.id)
