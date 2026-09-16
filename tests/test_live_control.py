import asyncio
import contextlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch, AsyncMock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from backend.admin import routes as admin
from backend.live import routes
from backend.live.protocol import LiveRun
from backend.live.store import LiveStore
from backend.gamer_ranked.rules import legal_moves
from tools.live_runner import run_forever


class LiveControlTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.hub = routes.LiveHub()
        self.hub.store = LiveStore(Path(self.directory.name) / 'live.db')
        self.app = FastAPI()
        self.app.include_router(admin.router)
        self.app.include_router(routes.router)
        self.client = TestClient(self.app)
        self.patches = [patch.object(routes, 'hub', self.hub),
                        patch.dict(os.environ, {'LIVE_PUBLISH_TOKEN': 'x' * 32}),
                        patch.object(routes.audience, 'online_tick')]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def test_admin_auth_validation_and_origin(self):
        for code in [401, 403]:
            with patch.object(admin, '_require_admin', side_effect=HTTPException(code)):
                self.assertEqual(self.client.get('/api/admin/live').status_code, code)
                self.assertEqual(self.client.post('/api/admin/live', json={'enabled': False}).status_code, code)
        with patch.object(admin, '_require_admin'):
            self.assertEqual(self.client.post('/api/admin/live', json={'enabled': 'false'}).status_code, 422)
            self.assertEqual(self.client.post('/api/admin/live', json={'enabled': False},
                                             headers={'origin': 'https://other.invalid'}).status_code, 403)

    def test_offline_intent_persists_and_is_idempotent(self):
        with patch.object(admin, '_require_admin'):
            first = self.client.post('/api/admin/live', json={'enabled': False}).json()
            second = self.client.post('/api/admin/live', json={'enabled': False}).json()
        self.assertEqual(first, second)
        self.assertFalse(first['connected'])
        self.assertFalse(first['applied'])
        self.assertEqual(LiveStore(self.hub.store.path).control(), dict(enabled=False, revision=1))

    def test_publisher_pause_ack_resume_preserves_run(self):
        run = LiveRun()
        with patch.object(admin, '_require_admin'), self.client.websocket_connect(
                '/api/live/publish', headers={'authorization': 'Bearer ' + 'x' * 32}) as ws:
            ws.send_json(dict(type='hello', control_version=1, run=run.checkpoint()))
            reply = ws.receive_json()
            self.assertEqual(reply['control'], dict(enabled=True, revision=0))
            ws.send_json(dict(type='control_ack', revision=0))
            result = self.client.post('/api/admin/live', json={'enabled': False}).json()
            self.assertFalse(result['applied'])
            command = ws.receive_json()
            # A step sent before the pause ACK is valid, not a rollback/disconnect.
            packet = run.make_step(legal_moves(run.board)[0], 80)
            ws.send_bytes(packet)
            ws.send_json(dict(type='control_ack', revision=command['revision']))
            for _ in range(100):
                status = self.client.get('/api/admin/live').json()
                if status['applied']:
                    break
                time.sleep(.005)
            self.assertTrue(status['applied'])
            self.assertFalse(status['online'])
            self.assertTrue(self.hub.snapshot()['paused'])
            self.assertEqual(self.hub.run.seq, 1)
            self.client.post('/api/admin/live', json={'enabled': True})
            command = ws.receive_json()
            ws.send_json(dict(type='control_ack', revision=command['revision']))
            self.assertEqual(self.hub.run.id, run.id)

    def test_paused_handshake_and_old_runner_protection(self):
        self.hub.control = self.hub.store.control(False)
        with self.client.websocket_connect('/api/live/publish', headers={'authorization': 'Bearer ' + 'x'*32}) as ws:
            ws.send_json(dict(type='hello', control_version=1))
            self.assertFalse(ws.receive_json()['control']['enabled'])
        self.hub.producer_ready = True
        self.hub.control_supported = False
        with patch.object(admin, '_require_admin'):
            self.assertEqual(self.client.post('/api/admin/live', json={'enabled': True}).status_code, 409)


class RunnerPauseTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_server_receives_steps_without_control_ack(self):
        run = LiveRun()
        sent, queue = [], asyncio.Queue()
        progressed = asyncio.Event()

        class Socket:
            async def send(self, message):
                sent.append(message if isinstance(message, bytes) else json.loads(message))
                if isinstance(message, bytes):
                    progressed.set()
            async def recv(self):
                return json.dumps(await queue.get())
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        ai = SimpleNamespace(choose=lambda board: (legal_moves(board)[0], 'AI'))
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(checkpoint=str(Path(directory) / 'run.json'), tables=None,
                                   engine_root=directory, threads=1, time_ratio=1,
                                   url='ws://test', interval=.08, search_interval=.05)
            await queue.put(dict(type='resume', run=run.checkpoint()))
            with patch.dict(os.environ, {'LIVE_PUBLISH_TOKEN': 'x'*32}), \
                    patch('tools.live_runner.NativeAI', return_value=ai), \
                    patch('websockets.asyncio.client.connect', return_value=Socket()):
                task = asyncio.create_task(run_forever(args))
                try:
                    await asyncio.wait_for(progressed.wait(), 5)
                    self.assertFalse(any(isinstance(item, dict) and item.get('type') == 'control_ack'
                                         for item in sent))
                finally:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def test_pause_during_search_fences_result_and_resumes_same_game(self):
        queue = asyncio.Queue()
        sent = []
        started, release = threading.Event(), threading.Event()
        run = LiveRun()

        class Socket:
            async def send(self, message):
                sent.append(message if isinstance(message, bytes) else json.loads(message))
            async def recv(self):
                return json.dumps(await queue.get())
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        class AI:
            calls = 0
            def choose(self, board):
                self.calls += 1
                if self.calls == 1:
                    started.set()
                    release.wait(5)
                return legal_moves(board)[0], 'AI'

        async def until(predicate):
            async with asyncio.timeout(5):
                while not predicate():
                    await asyncio.sleep(.005)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'run.json'
            args = SimpleNamespace(checkpoint=str(checkpoint), tables=None, engine_root=directory,
                                   threads=1, time_ratio=1, url='ws://test', interval=.08, search_interval=.05)
            await queue.put(dict(type='resume', run=run.checkpoint(), control=dict(enabled=True, revision=0)))
            with patch.dict(os.environ, {'LIVE_PUBLISH_TOKEN': 'x'*32}), \
                    patch('tools.live_runner.NativeAI', return_value=AI()), \
                    patch('websockets.asyncio.client.connect', return_value=Socket()):
                task = asyncio.create_task(run_forever(args))
                try:
                    await until(started.is_set)
                    await queue.put(dict(type='control', enabled=False, revision=1))
                    await until(lambda: dict(type='control_ack', revision=1) in sent)
                    release.set()
                    await asyncio.sleep(.15)
                    self.assertFalse(any(isinstance(item, bytes) for item in sent))
                    self.assertEqual(json.loads(checkpoint.read_text())['run_id'], run.id)
                    await queue.put(dict(type='control', enabled=True, revision=2))
                    await until(lambda: any(isinstance(item, bytes) for item in sent))
                    first = next(item for item in sent if isinstance(item, bytes))
                    run.apply(first)
                    self.assertEqual(run.seq, 1)
                    self.assertLess(run.elapsed, 150)
                    self.assertFalse(any(isinstance(item, dict) and item.get('type') == 'start' for item in sent))
                finally:
                    release.set()
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
