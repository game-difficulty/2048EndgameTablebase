import asyncio
import os
import tempfile
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from backend.auth.db import auth_db, init_auth_db
from backend.live import red_envelopes as red, routes


class RedEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'CLOUD_AUTH_DB': str(Path(self.directory.name) / 'auth.db')})
        self.env.start()
        init_auth_db()
        red.init_schema()
        with auth_db() as db:
            for uid in range(1, 26):
                db.execute('''INSERT INTO users(id,email,email_identity,password_hash,display_name,created_at,updated_at)
                    VALUES(?,?,?,?,?,'now','now')''', (uid, f'{uid}@test.invalid', f'{uid}@test.invalid', 'test', str(uid)))
                db.execute("INSERT INTO token_accounts(user_id,bonus_balance_units,paid_balance_units,created_at,updated_at) VALUES(?,123456,100000000,'now','now')", (uid,))

    def tearDown(self):
        self.env.stop()
        self.directory.cleanup()

    def body(self, **kw):
        return dict(request_id=str(uuid.uuid4()), amount=1000, count=5, mode='random', **kw)

    def send(self, uid=1, now=100, body=None):
        return red.create(uid, dict(name=str(uid), supporter=False), body or self.body(), now=now)

    def paid(self, uid):
        with auth_db() as db:
            return db.execute('SELECT paid_balance_units FROM token_accounts WHERE user_id=?', (uid,)).fetchone()[0]

    def test_paid_only_and_strict_integer_bounds(self):
        for key, value in [('amount',999),('amount',50001),('amount',1000.0),('count',4),('count',21),('count',True),('mode','bad'),('request_id','bad')]:
            body=self.body();body[key]=value
            with self.subTest(key=key,value=value), self.assertRaises(HTTPException):
                self.send(body=body)
        with auth_db() as db:
            db.execute('UPDATE token_accounts SET paid_balance_units=999999,bonus_balance_units=99999999 WHERE user_id=1')
        with self.assertRaises(HTTPException) as error:
            self.send()
        self.assertEqual(error.exception.detail, 'red_paid_balance')
        self.assertEqual(self.paid(1),999999)

    def test_idempotent_send_precedes_cooldown_and_payload_conflict(self):
        body=self.body()
        with ThreadPoolExecutor(max_workers=6) as pool:
            results=list(pool.map(lambda _:self.send(body=body),range(6)))
        self.assertEqual(len({item['id'] for item in results}),1)
        self.assertEqual(self.paid(1),99000000)
        self.assertEqual(self.send(now=500,body=body)['id'],results[0]['id'])
        for now in [100,114.999]:
            with self.assertRaises(HTTPException) as error:self.send(now=now)
            self.assertEqual(error.exception.detail,'red_cooldown')
        self.send(now=115)
        with self.assertRaises(HTTPException) as error:self.send(now=200,body={**body,'count':6})
        self.assertEqual(error.exception.detail,'red_request_conflict')

    def test_same_user_concurrent_claim_and_own_rejection(self):
        bag=self.send()
        with self.assertRaises(HTTPException):red.claim(bag['id'],1,now=101)
        with ThreadPoolExecutor(max_workers=8) as pool:
            awards=list(pool.map(lambda _:red.claim(bag['id'],2,now=101)['award'],range(12)))
        self.assertEqual(len(set(awards)),1)
        self.assertEqual(self.paid(2),100000000+awards[0]*1000)
        self.assertEqual(red.detail(bag['id'],2,now=101)['claimed'],1)
        self.assertEqual(red.claim(bag['id'],2,now=500)['award'],awards[0])

    def test_concurrent_claims_never_overdraw_and_hold_fifteen_seconds(self):
        bag=self.send()
        second=self.send(uid=2,now=101)
        self.assertEqual(second['status'],'queued')
        self.assertEqual(red.claim(second['id'],3,now=102)['award'],0)
        with ThreadPoolExecutor(max_workers=12) as pool:
            results=list(pool.map(lambda uid:red.claim(bag['id'],uid,now=102),range(2,26)))
        self.assertEqual(sum(item['award']>0 for item in results),5)
        self.assertEqual(sum(item['award'] for item in results),1000)
        self.assertEqual(red.tick(now=114.999)['active']['id'],bag['id'])
        active=red.tick(now=115)['active']
        self.assertEqual((active['id'],active['started_at'],active['expires_at']),(second['id'],115,175))
        self.assertEqual(red.detail(bag['id'],now=115)['refunded'],0)

    def test_expiry_refunds_once_survives_restart_and_waiting_gets_full_minute(self):
        first=self.send(body={**self.body(),'mode':'equal'})
        second=self.send(uid=2,now=101)
        red.claim(first['id'],3,now=159.999)
        self.assertEqual(red.claim(first['id'],4,now=160)['award'],0)
        red.init_schema()
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(lambda _:red.tick(now=170),range(5)))
        self.assertEqual(self.paid(1),99800000)
        self.assertEqual(red.detail(first['id'],now=170)['refunded'],800)
        self.assertEqual(red.tick(now=170)['active']['expires_at'],230)
        self.assertEqual(red.tick(now=230)['active'],None)
        self.assertEqual(self.paid(2),100000000)
        self.assertEqual(second['status'],'queued')

    def test_claim_expiry_race_conserves_funds(self):
        bag=self.send()
        with ThreadPoolExecutor(max_workers=10) as pool:
            tasks=[pool.submit(red.claim,bag['id'],uid,159.9) for uid in range(2,10)]
            tasks.append(pool.submit(red.tick,160))
            for future in tasks:future.result()
        red.tick(now=161)
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT SUM(paid_balance_units) FROM token_accounts').fetchone()[0],25*100000000)

    def test_equal_remainder_burned_and_random_allocation_has_no_order_bias(self):
        bag=self.send(body={**self.body(),'amount':1003,'mode':'equal'})
        self.assertEqual(bag['remainder'],3)
        self.assertEqual(red.claim(bag['id'],2,now=101)['award'],200)
        red.tick(now=160)
        self.assertEqual(red.detail(bag['id'],now=160)['refunded'],800)
        self.assertEqual(self.paid(1),99797000)
        # All positive compositions have equal probability; an independent uniform
        # permutation preserves the sum and makes every position exchangeable.
        for n in range(5,21):
            for total in [1000,1003,50000]:
                shares=red.split_amount(total,n,'random')
                self.assertEqual((len(shares),sum(shares)),(n,total))
                self.assertTrue(all(type(x) is int and x>0 for x in shares))
        with patch.object(red.secrets,'SystemRandom') as random:
            random.return_value.sample.return_value=[10,20,30,40]
            random.return_value.shuffle.side_effect=lambda values:values.reverse()
            self.assertEqual(red.split_amount(1000,5,'random'),[960,10,10,10,10])
            random.return_value.shuffle.assert_called_once()

    def test_transactions_roll_back_credits_and_debit_on_ledger_failure(self):
        bag=self.send()
        with auth_db() as db:db.execute("CREATE TRIGGER fail_ledger BEFORE INSERT ON token_ledger BEGIN SELECT RAISE(ABORT,'test'); END")
        for action in [lambda:self.send(uid=2),lambda:red.claim(bag['id'],2,now=101),lambda:red.tick(now=160)]:
            with self.assertRaises(Exception):action()
        self.assertEqual(self.paid(2),100000000)
        self.assertEqual(red.detail(bag['id'],now=159)['claimed'],0)
        with auth_db() as db:db.execute('DROP TRIGGER fail_ledger')
        red.tick(now=160)
        self.assertEqual(self.paid(1),100000000)

    def test_public_payload_hides_shares_and_other_users_awards(self):
        bag=self.send()
        red.claim(bag['id'],2,now=101)
        for payload in [bag,red.tick(now=102)['active'],red.detail(bag['id'],3,now=102)]:
            self.assertNotIn('shares_json',payload)
            self.assertNotIn('request_id',payload)
        self.assertEqual(red.detail(bag['id'],3,now=102)['award'],0)
        self.assertGreater(red.detail(bag['id'],2,now=102)['award'],0)
        self.assertEqual(len(red.events(True)),1)
        red.delivered([bag['id']])
        self.assertEqual(red.events(True),[])
        self.assertEqual(len(red.events()),1)

    def test_awards_do_not_grant_sponsorship_or_bonus_after_restart(self):
        bag=self.send()
        red.claim(bag['id'],2,now=101)
        red.tick(now=160)
        init_auth_db()
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT tier FROM user_entitlements WHERE user_id=2').fetchone()[0],'free')
            self.assertEqual(db.execute('SELECT bonus_balance_units FROM token_accounts WHERE user_id=2').fetchone()[0],123456)

    def test_api_requires_login_same_origin_and_fresh_room_presence(self):
        app=FastAPI();app.include_router(routes.router)
        hub=routes.LiveHub()
        with TestClient(app) as client, patch.object(routes,'hub',hub):
            self.assertEqual(client.post('/api/live/red-envelopes',json=self.body()).status_code,401)
            with patch.object(routes,'require_user',return_value={'id':1}),patch.object(routes.gifts,'public_actor',return_value={'name':'one'}):
                self.assertEqual(client.post('/api/live/red-envelopes',json=self.body(),headers={'origin':'https://evil.invalid'}).status_code,403)
                self.assertEqual(client.post('/api/live/red-envelopes',json=self.body()).json()['detail'],'red_not_present')
                with patch.object(hub,'lucky_present',return_value={1}):
                    response=client.post('/api/live/red-envelopes',json=self.body())
                    self.assertEqual(response.status_code,200)
                    bag=response.json()
                    self.assertEqual(len(hub.chat),1)
                    self.assertEqual(hub.red_state['active']['id'],bag['id'])
                    self.assertEqual(client.post('/api/live/red-envelopes/'+bag['id']+'/claim').json()['detail'],'red_own')
                with patch.object(routes,'current_user_from_request',return_value=None):
                    response=client.get('/api/live/red-envelopes/'+bag['id'])
                    self.assertEqual(response.headers['cache-control'],'no-store')

    def test_failed_broadcast_keeps_committed_send_and_outbox_recovers(self):
        app=FastAPI();app.include_router(routes.router)
        hub=routes.LiveHub();body=self.body()
        with TestClient(app) as client, patch.object(routes,'hub',hub), patch.object(routes,'require_user',return_value={'id':1}), patch.object(routes.gifts,'public_actor',return_value={'name':'one'}), patch.object(hub,'lucky_present',return_value={1}):
            with patch.object(hub,'refresh_red',side_effect=RuntimeError('disconnected')):
                response=client.post('/api/live/red-envelopes',json=body)
            self.assertEqual(response.status_code,200)
            self.assertEqual(len(red.events(True)),1)
            repeated=client.post('/api/live/red-envelopes',json=body)
            self.assertEqual(response.json()['id'],repeated.json()['id'])
            self.assertEqual(self.paid(1),99000000)
            self.assertEqual(len(hub.chat),1)
            self.assertEqual(red.events(True),[])
            asyncio.run(hub.refresh_red())
            self.assertEqual(len(hub.chat),1)


if __name__ == '__main__':
    unittest.main()
