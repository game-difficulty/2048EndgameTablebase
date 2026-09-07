import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from fastapi import FastAPI, HTTPException
from backend import gamer_tablebase as api
from backend.auth.db import auth_db, init_auth_db
from backend.quota.service import get_token_balance
from backend.tablebase_query_service import TablebaseQueryScheduler
from backend.remote_workers.registry import remote_worker_registry
from backend.quota.config import clear_token_pricing_cache
from backend.gamer_tablebase_route import generate_route


class Reader:
    calls = 0
    def dispatch(self, *args):
        pass
    def move_on_dic(self, *args):
        Reader.calls += 1
        return {'left': .9, 'right': .8, 'up': .7, 'down': .6}, 'float64'


class GamerTablebaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'CLOUD_AUTH_DB': str(Path(self.temp.name)/'auth.sqlite3')})
        self.env.start()
        init_auth_db()
        with auth_db() as db:
            self.user_id = db.execute("""INSERT INTO users
                (email,email_identity,password_hash,display_name,display_name_key,role,status,created_at,updated_at)
                VALUES ('ai@test.example','ai@test.example','hash','ai','ai','user','active','now','now')""").lastrowid
        self.user = dict(id=self.user_id, session_id=None)
        get_token_balance(self.user_id)
        with auth_db() as db:
            db.execute('UPDATE token_accounts SET paid_balance_units=512000 WHERE user_id=?',(self.user_id,))
        self.scheduler = TablebaseQueryScheduler()
        self.patches = [patch.object(api,'tablebase_query_scheduler',self.scheduler),
            patch.object(api,'get_catalog_version',return_value='v1'),
            patch.object(api,'resolve_tablebase',return_value=dict(pattern='L3',target='256',
                _provider='local',_absolute_path='unused',dtype='float64',spawn_rate=.1)),
            patch.object(api,'BookReaderDispatcher',Reader)]
        for p in self.patches: p.start()
        Reader.calls = 0

    async def asyncTearDown(self):
        await self.scheduler.close()
        for p in self.patches: p.stop()
        self.env.stop(); self.temp.cleanup()

    def request(self, **changes):
        fields = dict(request_id='12345678-1234-1234-1234-123456789012',catalog_version='v1',
            full_pattern='L3_256',board_codes=[1,0,2,0,3,0,4,0,5,0,6,0,7,0,8,0],
            rng_state=[1,2,3,4],spawn_rate4=.1,difficulty=0)
        fields.update(changes)
        return api.RouteRequest(**fields)

    async def read(self, request):
        response = await api.route(request,self.user)
        return [json.loads(line) async for line in response.body_iterator]

    async def test_remote_route_uses_one_rpc_and_populates_shared_node_cache(self):
        descriptor = dict(pattern='L3', target='256', _provider='remote', spawn_rate=.1)
        def produce(**kwargs):
            moves = iter(['left', 'right', 'left', 'right'])
            return generate_route(kwargs['options'], 6, lambda board: ({next(moves): .9}, 'float64'))
        rpc = AsyncMock(side_effect=produce)
        with patch.object(api, 'resolve_tablebase', return_value=descriptor), \
             patch.object(remote_worker_registry, 'supports_gamer_route', return_value=True), \
             patch.object(remote_worker_registry, 'generate_gamer_route', rpc), \
             patch.object(remote_worker_registry, 'lookup', AsyncMock(side_effect=AssertionError('unexpected per-step RPC'))):
            request = self.request(board_codes=[0,1,1,1,1,2,2,2,1,9,10,11,3,12,13,14])
            nodes = await self.read(request)
            self.assertEqual(len(nodes), 4)
            self.assertEqual(len({n['lookup_board'] for n in nodes}), 4)
            rpc.assert_awaited_once()
            self.assertEqual(rpc.call_args.kwargs['options']['steps'], 4)
            for node in nodes:
                self.assertEqual(node['type'], 'result')
                self.assertIsNotNone(self.scheduler.get_cached_result(catalog_version='v1',
                    full_pattern='L3_256', board_encoded=int(node['lookup_board'], 16)))
            self.assertEqual(get_token_balance(self.user_id)['paid'], 508)
            await self.read(request)
            rpc.assert_awaited_once()
            self.assertEqual(get_token_balance(self.user_id)['paid'], 508)

    async def test_remote_route_budget_limits_speculation(self):
        with auth_db() as db:
            db.execute('UPDATE token_accounts SET paid_balance_units=1000,bonus_balance_units=0 WHERE user_id=?', (self.user_id,))
        rpc = AsyncMock(side_effect=lambda **kw: generate_route(kw['options'], 6,
            lambda board: ({'left': .9}, 'float64')))
        with patch.object(api, 'resolve_tablebase', return_value=dict(pattern='L3',target='256',_provider='remote',spawn_rate=.1)), \
             patch.object(remote_worker_registry, 'supports_gamer_route', return_value=True), \
             patch.object(remote_worker_registry, 'generate_gamer_route', rpc):
            nodes = await self.read(self.request())
            self.assertEqual(rpc.call_args.kwargs['options']['steps'], 1)
            self.assertEqual(nodes[-1]['status'], 402)

    async def test_four_step_route_is_idempotently_charged_at_lookup_rate(self):
        before = get_token_balance(self.user_id)['total']
        first = await self.read(self.request())
        self.assertEqual(len(first),4)
        self.assertTrue(all(row['type']=='result' for row in first),first)
        after = get_token_balance(self.user_id)['total']
        self.assertEqual(before-after,4)
        calls = Reader.calls
        second = await self.read(self.request())
        self.assertEqual([x['board_codes'] for x in first],[x['board_codes'] for x in second])
        self.assertEqual(get_token_balance(self.user_id)['total'],after)
        self.assertEqual(Reader.calls,calls)
        self.assertFalse(api._active_users)

    async def test_evil_branch_stops_before_the_next_board_lookup(self):
        data = await self.read(self.request(difficulty=100))
        self.assertEqual(len(data),1)
        self.assertEqual(Reader.calls,1)

    async def test_random_after_undo_applies_only_to_first_spawn(self):
        data = await self.read(self.request(difficulty=100,random_only=True))
        self.assertEqual(len(data),2)
        self.assertEqual([x['random_only'] for x in data],[True,False])

    async def test_extend_route_skips_charging_the_already_purchased_root(self):
        data = await self.read(self.request(steps=1))
        before = get_token_balance(self.user_id)['total']
        more = await self.read(self.request(request_id='22345678-1234-1234-1234-123456789012',advance_first=True))
        self.assertEqual(len(more),4)
        self.assertNotEqual(data[0]['board_codes'],more[0]['board_codes'])
        self.assertEqual(before-get_token_balance(self.user_id)['total'],4)

    async def test_variant_and_stale_catalog_are_rejected_before_lookup(self):
        with patch.object(api,'resolve_tablebase',return_value=dict(pattern='3x3',target='1024',spawn_rate=.1)):
            with self.assertRaises(HTTPException) as error:
                await api.route(self.request(full_pattern='3x3_1024'),self.user)
            self.assertEqual(error.exception.status_code,409)
        with self.assertRaises(HTTPException):
            await api.route(self.request(catalog_version='old'),self.user)
        self.assertEqual(Reader.calls,0)

    async def test_request_id_cannot_be_reused_for_another_board(self):
        await self.read(self.request(steps=1))
        before = get_token_balance(self.user_id)['total']
        with self.assertRaises(HTTPException) as error:
            await self.read(self.request(steps=1,board_codes=[2,0]*8))
        self.assertEqual(error.exception.status_code,409)
        self.assertEqual(get_token_balance(self.user_id)['total'],before)

    async def test_paid_retry_is_delivered_even_after_balance_is_exhausted(self):
        request = self.request(steps=1)
        first = await self.read(request)
        with auth_db() as db:
            db.execute('UPDATE token_accounts SET paid_balance_units=0 WHERE user_id=?',(self.user_id,))
        self.assertEqual(first[0]['results'], (await self.read(request))[0]['results'])

    async def test_remote_table_uses_worker_without_constructing_a_local_reader(self):
        descriptor=dict(pattern='free10',target='512',_provider='remote',dtype='float64',spawn_rate=.1)
        lookup=AsyncMock(return_value={'results':{'left':.9,'right':.8},'dtype':'float64'})
        with patch.object(api,'resolve_tablebase',return_value=descriptor), \
                patch.object(remote_worker_registry,'lookup',lookup):
            before=get_token_balance(self.user_id)['total']
            data=await self.read(self.request(full_pattern='free10_512',steps=1))
        self.assertEqual(data[0]['type'],'result')
        self.assertEqual(Reader.calls,0)
        lookup.assert_awaited_once()
        self.assertEqual(lookup.call_args.kwargs['full_pattern'],'free10_512')
        self.assertEqual(before-get_token_balance(self.user_id)['total'],8)

    async def test_global_discount_applies_to_admission_and_final_charge(self):
        with auth_db() as db:
            db.execute('UPDATE token_accounts SET paid_balance_units=500 WHERE user_id=?',(self.user_id,))
        with patch.dict(os.environ,{'CLOUD_TOKEN_GLOBAL_MULTIPLIER':'0.5'}):
            clear_token_pricing_cache()
            data=await self.read(self.request(steps=1))
        clear_token_pricing_cache()
        self.assertEqual(data[0]['type'],'result')
        self.assertEqual(get_token_balance(self.user_id)['total'],0)

    async def test_unfunded_request_does_not_start_native_work(self):
        with auth_db() as db:
            db.execute('UPDATE token_accounts SET paid_balance_units=0 WHERE user_id=?',(self.user_id,))
        with self.assertRaises(HTTPException) as error:
            await self.read(self.request())
        self.assertEqual(error.exception.status_code,402)
        self.assertEqual(Reader.calls,0)

    async def test_unauthenticated_request_is_rejected(self):
        app=FastAPI(); app.include_router(api.router)
        messages=[]
        async def receive():
            return {'type':'http.request','body':self.request().model_dump_json().encode(),'more_body':False}
        async def send(message): messages.append(message)
        await app({'type':'http','http_version':'1.1','method':'POST','scheme':'http',
            'path':'/api/gamer/tablebase/route','raw_path':b'/api/gamer/tablebase/route',
            'query_string':b'','headers':[(b'content-type',b'application/json')],
            'server':('test',80),'client':('127.0.0.1',1)},receive,send)
        self.assertEqual(messages[0]['status'],401)


if __name__=='__main__': unittest.main()
