import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

from backend import gamer_tablebase_stream as stream
from backend.auth.db import init_auth_db, auth_db
from backend.auth.service import create_session
from backend.quota.service import get_token_balance
from backend.gamer_stream_window import StreamWindow


class Cursor:
    def __init__(self, options, count):
        self.index = 0
        self.values = [2] + [0] * 15
        self.rng = SimpleNamespace(state=[1, 2, 3, 4])

    @property
    def encoded(self):
        return self.index + 1

    def node(self, results, dtype):
        return dict(board_codes=[self.index] + [0]*15, rng_state=[self.index+1,2,3,4],
            random_only=False, lookup_board=f'{self.encoded:016x}', results=results, dtype=dtype)

    def advance(self, results, dtype):
        self.index += 1
        return True


class GamerTableStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_app_sender_delivers_frames_and_handles_disconnected_socket(self):
        from backend.app import _send_gamer_stream
        from starlette.websockets import WebSocketState
        socket = SimpleNamespace(client_state=WebSocketState.CONNECTED,
            application_state=WebSocketState.CONNECTED, send_json=AsyncMock())
        frame = {'action':stream.EVENT,'data':{'type':'result','seq':0}}
        self.assertTrue(await _send_gamer_stream(socket,frame))
        socket.send_json.assert_awaited_once_with(frame)
        socket.client_state=WebSocketState.DISCONNECTED
        self.assertFalse(await _send_gamer_stream(socket,frame))

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, CLOUD_AUTH_DB=str(Path(self.temp.name)/'auth.sqlite3'))
        self.env.start()
        init_auth_db()
        with auth_db() as db:
            self.user_id = db.execute('''INSERT INTO users
                (email,email_identity,password_hash,display_name,display_name_key,role,status,created_at,updated_at)
                VALUES ('stream@test.example','stream@test.example','hash','stream','stream','user','active','now','now')''').lastrowid
            _, self.session_id, _ = create_session(db, self.user_id)
        get_token_balance(self.user_id)
        with auth_db() as db:
            db.execute('UPDATE token_accounts SET paid_balance_units=512000 WHERE user_id=?', (self.user_id,))
        self.session = SimpleNamespace(user_id=self.user_id, auth_session_id=self.session_id, user_entitlement_tier='free')
        self.service = stream.GamerStreamService()
        self.socket = object()
        self.messages = []
        self.calls = []
        async def lookup(state, cursor, seq):
            self.calls.append(seq)
            return cursor.node({'left': .9}, 'float64')
        self.patches = [patch.object(stream, 'GamerRouteCursor', Cursor),
            patch.object(stream.RouteSubscription, 'lookup', lookup),
            patch.object(stream, 'get_catalog_version', return_value='v1'),
            patch.object(stream, 'resolve_tablebase', return_value=dict(pattern='L3', target='256',
                _provider='remote', dtype='float64', spawn_rate=.1)),
            patch.object(stream.remote_worker_registry, 'supports_gamer_stream', return_value=False)]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        await self.service.close()
        for item in self.patches:
            item.stop()
        self.env.stop()
        self.temp.cleanup()

    async def send(self, socket, message):
        self.messages.append((socket, message['data']))
        return True

    def opening(self, **changes):
        request = dict(request_id='12345678-1234-1234-1234-123456789012', catalog_version='v1',
            full_pattern='L3_256', board_codes=[1]+[0]*15, rng_state=[1,2,3,4],
            spawn_rate4=.1, difficulty=0, steps=1)
        return dict(route_id=request['request_id'], request=request, **changes)

    async def open(self, **changes):
        data = self.opening(**changes)
        await self.service.handle(stream.OPEN, data, self.session, self.socket, self.send)
        return self.service.routes.get((self.user_id, data['route_id']))

    async def until(self, predicate):
        async with asyncio.timeout(5):
            while not predicate():
                await asyncio.sleep(.005)

    async def test_window_stops_at_eight_then_continues_same_task_past_four_step_boundary(self):
        state = await self.open()
        await self.until(lambda: state.window.produced == 7)
        await asyncio.sleep(.05)
        self.assertEqual(len(self.calls), 8)
        task = state.task
        for consumed in (4, 12, 20):
            await self.service.handle(stream.CREDIT, dict(route_id=state.request.request_id,
                consumed=consumed, allow_through=consumed+16), self.session, self.socket, self.send)
            await self.until(lambda: state.window.produced == consumed+16)
        self.assertIs(state.task, task)
        self.assertEqual(self.calls, list(range(37)))
        self.assertLessEqual(len(state.frames), 33)

    async def test_reconnect_replays_paid_nodes_without_another_lookup_or_charge(self):
        state = await self.open()
        await self.until(lambda: state.window.produced == 7)
        balance = get_token_balance(self.user_id)
        self.service.disconnect(self.socket)
        other = object()
        await self.service.handle(stream.OPEN, self.opening(received=2, resume=True), self.session, other, self.send)
        replayed = [data for socket, data in self.messages if socket is other and data['type']=='result']
        self.assertEqual([data['seq'] for data in replayed], list(range(3,8)))
        self.assertEqual(len(self.calls),8)
        self.assertEqual(get_token_balance(self.user_id),balance)
        self.assertTrue(all(data['token_balance'] == balance for data in replayed))
        self.assertIs(state.socket,other)

    async def test_pause_cancel_and_old_socket_cannot_grant_new_work(self):
        state = await self.open()
        await self.until(lambda: state.window.produced == 7)
        old = self.socket
        self.service.disconnect(old)
        await self.service.handle(stream.CREDIT, dict(route_id=state.request.request_id,
            consumed=7,allow_through=39),self.session,old,self.send)
        self.assertEqual(state.window.allow_through,7)
        await self.service.stop((self.user_id,state.request.request_id))
        self.assertTrue(state.task.cancelled())
        self.assertEqual(len(self.calls),8)

    async def test_oversized_window_and_changed_retry_are_rejected(self):
        self.assertIsNone(await self.open(allow_through=100))
        self.assertEqual(self.messages[-1][1]['status'],400)
        state=await self.open()
        await self.until(lambda:state.window.produced==7)
        data=self.opening(); data['request']['board_codes']=[2]+[0]*15
        await self.service.handle(stream.OPEN,data,self.session,self.socket,self.send)
        self.assertEqual(self.messages[-1][1]['detail'],'REQUEST_ID_CONFLICT')
        await self.service.handle(stream.CREDIT,dict(route_id=state.request.request_id,consumed=100,allow_through=120),
            self.session,self.socket,self.send)
        self.assertEqual(self.messages[-1][1]['status'],400)

    async def test_revoked_session_and_insufficient_balance_stop_before_lookup(self):
        with auth_db() as db:
            db.execute("UPDATE sessions SET revoked_at='now' WHERE id=?",(self.session_id,))
        state=await self.open()
        await self.until(lambda:state.task.done())
        self.assertEqual(state.end['status'],401); self.assertEqual(self.calls,[])
        await self.service.stop((self.user_id,state.request.request_id))
        with auth_db() as db:
            db.execute('UPDATE sessions SET revoked_at=NULL WHERE id=?',(self.session_id,))
            db.execute('UPDATE token_accounts SET paid_balance_units=0,bonus_balance_units=0 WHERE user_id=?',(self.user_id,))
        state=await self.open()
        await self.until(lambda:state.task.done())
        self.assertEqual(state.end['status'],402); self.assertEqual(self.calls,[])

    async def test_resume_missing_state_does_not_start_new_paid_work(self):
        self.assertIsNone(await self.open(received=2,resume=True))
        self.assertEqual(self.messages[-1][1]['detail'],'STREAM_GONE')
        self.assertEqual(self.calls,[])


class StreamWindowTests(unittest.IsolatedAsyncioTestCase):
    async def test_cumulative_credit_and_hard_limit(self):
        window=StreamWindow(7)
        waiting=asyncio.create_task(window.wait(8))
        await asyncio.sleep(0); self.assertFalse(waiting.done())
        window.produced=7
        window.update(4,20); await waiting
        window.update(2,10)
        self.assertEqual((window.consumed,window.allow_through),(4,20))
        for consumed,allowed in [(8,20),(7,40),(True,20),(-1,32)]:
            with self.assertRaises(ValueError): window.update(consumed,allowed)
