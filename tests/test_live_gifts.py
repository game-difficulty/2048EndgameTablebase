import os
import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import HTTPException, FastAPI
from fastapi.testclient import TestClient
from backend.auth.db import auth_db, init_auth_db
from backend.live import gifts, audience
from backend.live import routes
from backend.live.supporters import supporter_level, public_actor
from backend.quota.errors import InsufficientTokens
from backend.quota.service import get_token_balance


class LiveGiftTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'CLOUD_AUTH_DB': str(Path(self.directory.name) / 'auth.db'), 'CLOUD_TOKEN_GLOBAL_MULTIPLIER': '1'})
        self.env.start()
        init_auth_db()
        gifts.init_schema()
        with auth_db() as db:
            for user_id in [1, 2]:
                db.execute('''INSERT INTO users(id,email,email_identity,password_hash,display_name,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?)''', (user_id, f'{user_id}@test.invalid', f'{user_id}@test.invalid', 'test', 'Tester', 'now', 'now'))
                db.execute('''INSERT INTO token_accounts(user_id,bonus_balance_units,paid_balance_units,created_at,updated_at)
                    VALUES(?,?,?,'now','now')''', (user_id, 100000, 100000000))
        self.user = {'id': 1, 'display_name': 'Tester', 'role': 'admin'}

    def tearDown(self):
        self.env.stop()
        self.directory.cleanup()

    def request(self, gift='heart', quantity=1):
        catalog, _ = gifts.catalogue()
        item = next(item for item in catalog['gifts'] if item['id'] == gift)
        return dict(request_id=str(uuid4()), gift_id=gift, quantity=quantity,
                    expected_cost_units=item['totals'][quantity - 1], quote_version=catalog['version'])

    def payment(self, amount, event_type='admin_topup', delta=100000):
        with auth_db() as db:
            db.execute('''INSERT INTO token_ledger(user_id,event_type,operation_key,paid_delta_units,metadata_json,created_at)
                VALUES(1,?,'add_paid',?,?,'now')''', (event_type, delta, json.dumps({'payment_amount_cny': amount})))

    def test_supporter_levels_use_confirmed_cumulative_payments(self):
        regular = {**self.user, 'role': 'user'}
        self.assertEqual(supporter_level(regular), 0)
        self.payment(9.89)
        self.assertEqual(supporter_level(regular), 0)
        self.payment(.01)
        self.assertEqual(supporter_level(regular), 1)
        for _ in range(8):
            self.payment(9.9)
        self.assertEqual(supporter_level(regular), 1)
        self.payment(9.9)
        self.assertEqual(supporter_level(regular), 2)
        identity = public_actor(regular)
        self.assertEqual(set(identity), {'name', 'avatar_url', 'supporter', 'supporter_level'})
        self.assertTrue(identity['supporter'])

    def test_supporter_levels_do_not_infer_from_tokens_or_adjustments(self):
        regular = {**self.user, 'role': 'user'}
        for value in [None, True, 'NaN', 'Infinity', -99, 'invalid']:
            self.payment(value)
        self.payment(999, 'admin_set_paid_balance')
        self.payment(999, 'consume')
        self.payment(999, delta=-10)
        self.assertEqual(supporter_level(regular), 0)
        self.assertEqual(supporter_level(self.user), 1)
        self.assertEqual(supporter_level({**regular, 'entitlements': {'tier': 'supporter'}}), 1)
        self.payment(99)
        self.assertEqual(supporter_level(self.user), 2)

    def test_gift_and_entrance_include_level_but_not_payment_amount(self):
        self.payment(99)
        entrance = gifts.entrance(self.user)
        self.assertEqual(entrance['actor']['supporter_level'], 2)
        gifts.send(self.user, self.request(), True)
        event = gifts.pending_events()[0]['event']
        self.assertEqual(event['actor']['supporter_level'], 2)
        self.assertNotIn('payment_amount', json.dumps(event))

    def test_chat_level_is_server_derived_and_limit_is_unchanged(self):
        from backend.auth.principal import ActorRef
        self.payment(99)
        app = FastAPI()
        app.include_router(routes.router)
        local_hub = routes.LiveHub()
        with TestClient(app) as client, patch.object(routes, 'hub', local_hub), \
                patch.object(routes, 'require_actor', return_value=ActorRef.from_user(self.user)), \
                patch.object(routes, 'current_user_from_request', return_value=self.user):
            for _ in range(5):
                self.assertEqual(client.post('/api/live/chat', json={'text': 'Hi', 'supporter_level': 0}).status_code, 200)
            self.assertEqual(local_hub.chat[-1]['supporter_level'], 2)
            self.assertEqual(client.post('/api/live/chat', json={'text': 'Six'}).status_code, 429)

    def test_entrance_is_in_chat_and_shared_across_sockets_only_once(self):
        app = FastAPI()
        app.include_router(routes.router)
        local_hub = routes.LiveHub()
        with TestClient(app) as client, patch.object(routes, 'hub', local_hub), \
                patch.object(routes, 'current_user_from_websocket', return_value=self.user):
            with client.websocket_connect('/api/live/watch') as first:
                first.receive_json()
                event = first.receive_json()
                self.assertEqual(event['type'], 'entrance')
                self.assertEqual(event['actor']['supporter_level'], 1)
                with patch.object(gifts, 'entrance', wraps=gifts.entrance) as announce:
                    with client.websocket_connect('/api/live/watch') as second:
                        second.receive_json()
                        self.assertEqual(len(local_hub.chat), 1)
                        self.assertEqual(local_hub.chat[0]['type'], 'entrance')
                    announce.assert_not_called()

    def test_prices_and_bonus_before_paid(self):
        expected = dict(heart=8, flowers=8, two=2, four=4, dealer=32, bug=8, klbm=32,
                        crown=32768, final=1024, legend=65536, knowledge=16, button=16,
                        whale=16, moai=10, meaning=16, rip=16, tea=16, chicken=16,
                        serious=16)
        expected.update({'666': 66, '2048': 2048})
        self.assertEqual({item['id']: item['totals'][0] // 1000 for item in gifts.catalogue()[0]['gifts']}, expected)
        gifts.send(self.user, self.request('dealer', 4), True)
        self.assertEqual(get_token_balance(1)['bonus'], 0)
        self.assertEqual(get_token_balance(1)['paid'], 99972)

    def test_retired_gifts_cannot_be_bought_but_old_receipts_survive(self):
        request = self.request('heart')
        before = get_token_balance(1)
        for gift_id in gifts.RETIRED_GIFT_IDS:
            with self.assertRaises(HTTPException) as error:
                gifts.send(self.user, {**request, 'gift_id': gift_id}, True)
            self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(get_token_balance(1), before)
        receipt = gifts.send(self.user, request, True)
        with patch.object(gifts, 'GIFT_IDS', gifts.GIFT_IDS - {'heart'}), patch.object(gifts, 'RETIRED_GIFT_IDS', {'heart'}):
            self.assertEqual(gifts.send(self.user, request, False), receipt)
            self.assertEqual(gifts.order(1, request['request_id']), receipt)

    def test_cutout_posters_have_real_transparency(self):
        from PIL import Image
        root = Path(__file__).resolve().parents[1] / 'frontend/public/live-gifts'
        for name in ['knowledge-cutout', 'moai-cutout']:
            with Image.open(root / (name + '.webp')) as image:
                self.assertEqual(image.mode, 'RGBA')
                self.assertEqual(image.size, (160, 160))
                self.assertEqual(image.getchannel('A').getextrema(), (0, 255))
                for point in [(0, 0), (159, 0), (0, 159), (159, 159)]:
                    self.assertEqual(image.getpixel(point)[3], 0)

    def test_catalog_change_preserves_paid_receipts(self):
        request = self.request('crown')
        result = gifts.send(self.user, request, True)
        with patch.object(gifts, 'catalogue', side_effect=AssertionError('must not reprice')):
            self.assertEqual(gifts.send(self.user, request, False), result)

    def test_gifts_share_chat_and_combos_update_one_row(self):
        import asyncio
        hub = routes.LiveHub()
        gifts.send(self.user, self.request('button'), True)
        gifts.send(self.user, self.request('button', 5), True)
        asyncio.run(hub.drain_gifts())
        self.assertEqual(len(hub.chat), 1)
        self.assertEqual(hub.chat[0]['combo_count'], 6)
        self.assertEqual(hub.chat[0]['type'], 'gift')
        self.assertEqual(hub.chat[0]['supporter_level'], 1)
        hub.chat.append(dict(id='text', text='Hello', at=time.time()))
        for event in reversed(gifts.recent_events()):
            hub.append_gift_chat(event)
        self.assertEqual(len(hub.chat), 2)
        self.assertEqual(hub.chat[0]['combo_count'], 6)
        self.assertEqual(hub.chat[1]['id'], 'text')

    def test_concurrent_idempotency_and_fingerprint(self):
        request = self.request()
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda _: gifts.send(self.user, request, True), range(4)))
        self.assertEqual(len({item['request_id'] for item in results}), 1)
        self.assertEqual(get_token_balance(1)['total'], 100092)
        self.assertEqual(gifts.send(self.user, request, False)['status'], 'sent')
        for user, body in [(self.user, {**request, 'quantity': 2}), ({'id': 2}, request)]:
            with self.assertRaises(HTTPException) as error:
                gifts.send(user, body, True)
            self.assertEqual(error.exception.status_code, 409)
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM token_ledger').fetchone()[0], 1)

    def test_price_change_daily_budget_and_offline_do_not_charge(self):
        before = get_token_balance(1)
        request = self.request()
        with patch.dict(os.environ, {'CLOUD_TOKEN_GLOBAL_MULTIPLIER': '.5'}):
            with self.assertRaises(HTTPException) as error:
                gifts.send(self.user, request, True)
            self.assertEqual(error.exception.detail, 'gift_price_changed')
            fresh = self.request()
            self.assertEqual(fresh['expected_cost_units'], 4000)
        with self.assertRaises(HTTPException):
            gifts.send(self.user, request, False)
        gifts.set_preferences(1, 7000, True)
        with self.assertRaises(HTTPException) as error:
            gifts.send(self.user, request, True)
        self.assertEqual(error.exception.detail, 'gift_daily_budget')
        self.assertEqual(get_token_balance(1), before)

    def test_failed_order_rolls_back_debit_ledger_and_request(self):
        before = get_token_balance(1)
        with patch.object(gifts, 'receipt', side_effect=RuntimeError('after_insert')):
            with self.assertRaises(RuntimeError):
                gifts.send(self.user, self.request(), True)
        self.assertEqual(get_token_balance(1), before)
        with auth_db() as db:
            for table in ['live_gift_orders', 'token_ledger', 'token_operation_requests', 'live_gift_daily']:
                self.assertEqual(db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0], 0)

    def test_insufficient_tokens_is_atomic(self):
        with auth_db() as db:
            db.execute('UPDATE token_accounts SET bonus_balance_units=0, paid_balance_units=0 WHERE user_id=1')
        with self.assertRaises(InsufficientTokens):
            gifts.send(self.user, self.request(), True)
        self.assertEqual(gifts.mine(1)['orders'], [])

    def test_combo_outbox_and_entrance_cooldown(self):
        first = gifts.send(self.user, self.request(), True)
        second = gifts.send(self.user, self.request(quantity=5), True)
        self.assertEqual(first['combo_id'], second['combo_id'])
        self.assertEqual(second['combo_count'], 6)
        events = gifts.pending_events()
        self.assertEqual(len(events), 2)
        self.assertTrue(all(item['fresh'] for item in events))
        gifts.delivered([item['id'] for item in events])
        self.assertEqual(gifts.pending_events(), [])
        self.assertIsNotNone(gifts.entrance(self.user))
        self.assertIsNone(gifts.entrance(self.user))
        gifts.set_preferences(1, None, False)
        with patch.object(gifts.time, 'time', return_value=time.time() + 1900):
            self.assertIsNone(gifts.entrance(self.user))

    def test_global_multiplier_deducts_exact_quote(self):
        with patch.dict(os.environ, {'CLOUD_TOKEN_GLOBAL_MULTIPLIER': '.333'}):
            request = self.request('two', 5)
            result = gifts.send(self.user, request, True)
        self.assertEqual(result['cost_units'], 3330)
        self.assertEqual(get_token_balance(1)['bonus'], 96.67)

    def test_stale_outbox_does_not_replay_effect_and_cleanup_keeps_receipt(self):
        request = self.request()
        gifts.send(self.user, request, True)
        with auth_db() as db:
            db.execute('UPDATE live_gift_orders SET created_at=?', (time.time() - 40 * 86400,))
        self.assertFalse(gifts.pending_events()[0]['fresh'])
        gifts.delivered([request['request_id']])
        gifts.cleanup()
        self.assertEqual(gifts.recent_events(), [])
        self.assertEqual(gifts.send(self.user, request, False)['status'], 'sent')

    def test_invalid_quantity_and_foreign_receipt(self):
        for value in [True, 0, 1001, 1.5, '1']:
            with self.assertRaises(HTTPException):
                gifts.send(self.user, {**self.request(), 'quantity': value}, True)
        request = self.request()
        gifts.send(self.user, request, True)
        with self.assertRaises(HTTPException) as error:
            gifts.order(2, request['request_id'])
            self.assertEqual(error.exception.status_code, 404)

    def test_bulk_gift_charges_once_and_contributes_actual_tokens(self):
        for quantity in [1, 10, 100, 1000]:
            request = self.request('two')
            request.update(quantity=quantity, expected_cost_units=2000*quantity)
            result = gifts.send(self.user, request, True)
            self.assertEqual(gifts.send(self.user, request, True), result)
        with auth_db() as db:
            row = db.execute('SELECT gift_units FROM live_audience_scores WHERE actor_key=?', ('u:1',)).fetchone()
        self.assertEqual(row[0], 2222000)

    def test_audience_caps_gold_priority_and_private_fields(self):
        audience.online_tick({'u:1': 5}, now=1000)
        with auth_db() as db:
            audience.add(db, 'u:1', 'watch_seconds', 1200)
            audience.add(db, 'u:1', 'likes', 100)
            audience.add(db, 'u:1', 'messages', 100)
            audience.add(db, 'u:1', 'gift_units', 12345)
        identities = {'u:1': dict(name='Normal', supporter_level=0), 'u:2': dict(name='Gold', supporter_level=2)}
        result = audience.ranking(identities)
        self.assertEqual([row['name'] for row in result['viewers']], ['Gold','Normal'])
        self.assertEqual(result['viewers'][1]['contribution_units'], 34345)
        self.assertNotIn('actor_key', json.dumps(result))
        audience.online_tick({}, now=1100)
        self.assertEqual(audience.ranking(identities)['session_id'],result['session_id'])
        audience.online_tick({}, now=3001)
        renewed = audience.ranking(identities)
        self.assertNotEqual(renewed['session_id'],result['session_id'])
        self.assertTrue(all(row['contribution_units']==0 for row in renewed['viewers']))

    def test_presence_deduplicates_tabs_and_excludes_departed_viewers(self):
        presence = audience.Presence()
        presence.join('tab1','u:1',dict(name='One'),now=0)
        presence.join('tab2','u:1',dict(name='One'),now=2)
        self.assertEqual(presence.tick(now=5),{'u:1':5})
        presence.leave('tab1')
        self.assertEqual(presence.tick(now=10),{'u:1':5})
        presence.leave('tab2')
        self.assertEqual(presence.tick(now=20),{})
        self.assertEqual(presence.identities(),{})

    def test_http_charge_recovery_broadcast_failure_and_errors(self):
        app = FastAPI()
        app.include_router(routes.router)
        local_hub = routes.LiveHub()
        local_hub.producer = object()
        local_hub.last_seen = time.monotonic()
        with TestClient(app) as client, patch.object(routes, 'hub', local_hub), patch.object(routes, 'require_user', return_value=self.user):
            request = self.request()
            with patch.object(local_hub, 'drain_gifts', new=AsyncMock(side_effect=RuntimeError('broadcast_down'))):
                response = client.post('/api/live/gifts/send', json=request)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(client.get('/api/live/gifts/orders/' + request['request_id']).json()['status'], 'sent')
            self.assertEqual(client.post('/api/live/gifts/send', json=request).status_code, 200)
            self.assertEqual(get_token_balance(1)['total'], 100092)
            local_hub.rates.clear()
            with auth_db() as db:
                db.execute('UPDATE token_accounts SET bonus_balance_units=0,paid_balance_units=0 WHERE user_id=1')
            response = client.post('/api/live/gifts/send', json=self.request())
            self.assertEqual(response.status_code, 402)
            self.assertEqual(response.json()['detail']['code'], 'INSUFFICIENT_TOKENS')
            self.assertEqual(client.post('/api/live/gifts/send', json={}, headers={'Origin':'https://other.invalid'}).status_code, 403)
            self.assertEqual(client.post('/api/live/gifts/send', content='x'*3000).status_code, 413)
        with TestClient(app) as client, patch.object(routes, 'require_user', side_effect=HTTPException(401)):
            self.assertEqual(client.post('/api/live/gifts/send', json=self.request()).status_code, 401)
            self.assertEqual(client.get('/api/live/gifts/me').status_code, 401)
