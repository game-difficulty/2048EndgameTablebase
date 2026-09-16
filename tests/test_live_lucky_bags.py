import asyncio
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from backend.auth.db import auth_db, init_auth_db
from backend.live import lucky_bags as bags, routes
from backend.live.protocol import LiveRun


class LuckyBagTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'CLOUD_AUTH_DB': str(Path(self.directory.name) / 'auth.db')})
        self.env.start()
        init_auth_db()
        bags.init_schema()
        with auth_db() as db:
            for uid in range(1, 16):
                db.execute('''INSERT INTO users(id,email,email_identity,password_hash,display_name,created_at,updated_at)
                    VALUES(?,?,?,?,?,'now','now')''', (uid, f'{uid}@test.invalid', f'{uid}@test.invalid', 'test', str(uid)))
        self.bag = bags.create('run-one', 32768, now=100)

    def tearDown(self):
        self.env.stop()
        self.directory.cleanup()

    def enter(self, count=15):
        for uid in range(1, count + 1):
            bags.join(self.bag['id'], uid, now=110)

    def test_milestones_are_independent_and_idempotent(self):
        self.assertEqual(bags.create('run-one', 32768, now=150), self.bag)
        large = bags.create('run-one', 65536, now=150)
        self.assertEqual((large['pool'], large['minimum'], large['maximum'], large['draw_at']), (20000, 1000, 3000, 330))
        self.assertEqual(len(bags.listing(now=200)), 2)
        self.assertNotEqual(bags.create('run-two', 32768, now=200)['id'], self.bag['id'])

    def test_award_bounds_and_totals_for_every_winner_count(self):
        for pool, low, high in bags.RULES.values():
            for count in range(11):
                for _ in range(50):
                    result = bags.amounts(count, pool, low, high)
                    self.assertEqual(len(result), count)
                    self.assertLessEqual(sum(result), pool)
                    self.assertTrue(all(low <= value <= high for value in result))

    def test_ten_unique_online_winners_credited_without_claim(self):
        self.enter()
        with auth_db() as db:
            db.execute("INSERT INTO token_accounts(user_id,bonus_balance_units,paid_balance_units,created_at,updated_at) VALUES(1,1234,789,'now','now')")
        self.assertTrue(bags.draw(self.bag['id'], set(range(1, 16)), now=280))
        result = bags.listing(1, now=281)[0]
        self.assertEqual((result['participants'], result['winners']), (15, 10))
        self.assertTrue(3000 <= result['distributed'] <= 5000)
        with auth_db() as db:
            ledger = list(db.execute("SELECT * FROM token_ledger WHERE event_type='live_lucky_award'"))
            self.assertEqual(len(ledger), 10)
            for row in ledger:
                self.assertEqual(row['balance_after_units']-row['balance_before_units'], row['paid_delta_units'])
                self.assertTrue(300000 <= row['paid_delta_units'] <= 1000000)
            account = db.execute('SELECT * FROM token_accounts WHERE user_id=1').fetchone()
            self.assertEqual(account['bonus_balance_units'], 1234)
            self.assertEqual(account['paid_balance_units'], 789+result['award']*1000)
            self.assertEqual(db.execute('SELECT count(*) FROM user_entitlements').fetchone()[0], 0)

    def test_absent_users_excluded_and_small_pools_not_fully_distributed(self):
        self.enter(3)
        with patch.object(bags.secrets, 'randbelow', return_value=123):
            self.assertTrue(bags.draw(self.bag['id'], {1}, now=280))
        self.assertEqual(bags.listing(1, now=281)[0]['award'], 423)
        result = bags.listing(2, now=281)[0]
        self.assertEqual((result['joined'], result['present'], result['award'], result['distributed']), (True, False, 0, 423))

    def test_single_winner_can_receive_either_endpoint(self):
        for pool, low, high in bags.RULES.values():
            with patch.object(bags.secrets, 'randbelow', return_value=0):
                self.assertEqual(bags.amounts(1, pool, low, high), [low])
            with patch.object(bags.secrets, 'randbelow', side_effect=lambda n: n - 1):
                self.assertEqual(bags.amounts(1, pool, low, high), [high])

    def test_amounts_are_shuffled_before_assignment(self):
        with patch.object(bags.secrets, 'randbelow', side_effect=[0, 100, 200]), \
             patch.object(bags.secrets.SystemRandom, 'shuffle', side_effect=lambda values: values.reverse()) as shuffle:
            self.assertEqual(bags.amounts(3, 5000, 300, 1000), [500, 400, 300])
            shuffle.assert_called_once()

    def test_no_participants_or_no_present_users(self):
        self.assertTrue(bags.draw(self.bag['id'], {1}, now=280))
        self.assertEqual(bags.listing(now=281)[0]['distributed'], 0)
        other = bags.create('run-empty', 32768, now=100)
        bags.join(other['id'], 1, now=110)
        self.assertTrue(bags.draw(other['id'], set(), now=280))
        self.assertEqual(bags.listing(1, now=281)[1]['award'], 0)

    def test_draw_and_join_retries_are_idempotent(self):
        self.enter(1)
        self.assertFalse(bags.draw(self.bag['id'], {1}, now=279))
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: bags.draw(self.bag['id'], {1}, now=280), range(4)))
        self.assertEqual(sum(results), 1)
        bags.join(self.bag['id'], 1, now=300)
        bags.init_schema()
        self.assertFalse(bags.draw(self.bag['id'], {1}, now=300))
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM token_ledger').fetchone()[0], 1)
        with self.assertRaises(HTTPException) as error:
            bags.join(self.bag['id'], 2, now=280)
        self.assertEqual(error.exception.status_code, 409)

    def test_transaction_rolls_back_all_awards_on_failure(self):
        self.enter(2)
        with auth_db() as db:
            db.execute("CREATE TRIGGER fail_credit BEFORE INSERT ON token_ledger BEGIN SELECT RAISE(ABORT,'test'); END")
        with self.assertRaises(Exception):
            bags.draw(self.bag['id'], {1, 2}, now=280)
        result = bags.listing(1, now=281)[0]
        self.assertIsNone(result['drawn_at'])
        self.assertEqual(result['award'], 0)
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM token_accounts').fetchone()[0], 0)
            db.execute('DROP TRIGGER fail_credit')
        self.assertTrue(bags.draw(self.bag['id'], {1, 2}, now=281))

    def test_results_retained_three_minutes_and_cleanup_keeps_trigger_key(self):
        self.enter(1)
        bags.draw(self.bag['id'], {1}, now=290)
        self.assertEqual(len(bags.listing(now=469)), 1)
        self.assertEqual(bags.listing(now=470), [])
        bags.cleanup(now=470+31*86400)
        self.assertEqual(bags.create('run-one', 32768, now=470+32*86400)['id'], self.bag['id'])
        self.assertEqual(bags.listing(now=470+32*86400), [])

    def test_inactive_account_cannot_win(self):
        self.enter(1)
        with auth_db() as db:
            db.execute("UPDATE users SET status='disabled' WHERE id=1")
        bags.draw(self.bag['id'], {1}, now=280)
        self.assertEqual(bags.listing(1, now=281)[0]['award'], 0)

    def test_awards_do_not_grant_sponsorship_on_service_restart(self):
        self.enter(1)
        bags.draw(self.bag['id'], {1}, now=280)
        init_auth_db()
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT tier FROM user_entitlements WHERE user_id=1').fetchone()[0], 'free')
            db.execute("UPDATE user_entitlements SET tier='supporter' WHERE user_id=1")
        init_auth_db()
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT tier FROM user_entitlements WHERE user_id=1').fetchone()[0], 'supporter')

    def test_presence_uses_unique_users_live_heartbeats_and_deadline(self):
        hub = routes.LiveHub()
        hub.viewer_users = {'a': 1, 'b': 1, 'stale': 2, 'late': 3}
        hub.viewer_times = {'a': (100, 278), 'b': (200, 279), 'stale': (100, 240), 'late': (280.5, 280.5)}
        with patch.object(routes.time, 'time', return_value=281):
            self.assertEqual(hub.lucky_present(280), {1})

    def test_tick_milestone_resume_no_per_step_database_and_restart_due(self):
        hub = routes.LiveHub()
        hub.run = SimpleNamespace(id='run-one', nodes={'32768': 5, '65536': 8})
        async def check():
            with patch.object(bags.time, 'time', return_value=200):
                await hub.tick_lucky()
                self.assertEqual(len(hub.lucky_bags), 2)
                with patch.object(bags, 'create', side_effect=AssertionError('duplicate work')), patch.object(bags, 'listing', side_effect=AssertionError('ordinary-step IO')):
                    await hub.tick_lucky()
            restored = routes.LiveHub()
            restored.lucky_bags = bags.listing(now=500)
            with patch.object(bags.time, 'time', return_value=500):
                await restored.tick_lucky()
                self.assertTrue(all(bag['drawn_at'] is not None for bag in restored.lucky_bags))
        asyncio.run(check())

    def test_actual_merge_activates_each_milestone_only_once(self):
        hub = routes.LiveHub()
        hub.run = LiveRun()
        async def check():
            for value in [32768, 65536]:
                hub.run.board = [value // 2, value // 2, 0, 0] + [0] * 12
                hub.run.apply(hub.run.make_step('left', 50))
                self.assertIn(str(value), hub.run.nodes)
                await hub.tick_lucky()
                matched = [bag for bag in hub.lucky_bags if bag['run_id'] == hub.run.id]
                self.assertEqual(len(matched), 1 if value == 32768 else 2)
            hub.run.apply(hub.run.make_step('down', 50))
            with patch.object(bags, 'create', side_effect=AssertionError('repeated trigger')):
                await hub.tick_lucky()
        asyncio.run(check())

    def test_http_auth_origin_presence_and_private_result(self):
        app = FastAPI()
        app.include_router(routes.router)
        hub = routes.LiveHub()
        with patch.object(routes, 'hub', hub), TestClient(app) as client, patch.object(bags.time, 'time', return_value=110):
            path = '/api/live/lucky-bags/' + self.bag['id'] + '/join'
            self.assertEqual(client.post(path).status_code, 401)
            with patch.object(routes, 'require_user', return_value={'id': 1}):
                self.assertEqual(client.post(path).status_code, 409)
                hub.viewer_users['socket'] = 1
                self.assertEqual(client.post(path, headers={'Origin':'https://evil.invalid'}).status_code, 403)
                result = client.post(path)
                self.assertEqual(result.status_code, 200)
                self.assertTrue(result.json()['bags'][0]['joined'])
            with patch.object(routes, 'current_user_from_request', return_value=None):
                response = client.get('/api/live/lucky-bags')
                self.assertEqual(response.headers['cache-control'], 'no-store')
                self.assertNotIn('joined', response.json()['bags'][0])
            with patch.object(routes, 'current_user_from_request', return_value={'id': 2}):
                self.assertFalse(client.get('/api/live/lucky-bags').json()['bags'][0]['joined'])


if __name__ == '__main__':
    unittest.main()
