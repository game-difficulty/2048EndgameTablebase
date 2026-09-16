import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from backend.gamer_tablebase_common import RouteRequest, check_query_budget
from backend.gamer_tablebase_route import GamerRouteCursor
from backend.auth.db import auth_db, init_auth_db
from backend.quota.config import clear_token_pricing_cache
from backend.quota.errors import InsufficientTokens
from backend.quota.service import consume_operation_tokens_once, get_token_balance


class GamerTablebaseCommonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {'CLOUD_AUTH_DB': str(Path(self.temp.name) / 'auth.sqlite3')})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(clear_token_pricing_cache)
        init_auth_db()
        with auth_db() as db:
            self.user_id = db.execute("""INSERT INTO users
                (email,email_identity,password_hash,display_name,display_name_key,role,status,created_at,updated_at)
                VALUES ('ai@test.example','ai@test.example','hash','ai','ai','user','active','now','now')""").lastrowid
        get_token_balance(self.user_id)
        self.set_balance(512000)
        self.request = RouteRequest(request_id='12345678-1234-1234-1234-123456789012',
            catalog_version='v1', full_pattern='L3_256', board_codes=[1,0,2,0]*4,
            rng_state=[1,2,3,4], spawn_rate4=.1, difficulty=0)

    def set_balance(self, units):
        with auth_db() as db:
            db.execute('UPDATE token_accounts SET paid_balance_units=?,bonus_balance_units=0 WHERE user_id=?',
                       (units, self.user_id))

    def test_budget_check_does_not_charge(self):
        before = get_token_balance(self.user_id)
        check_query_budget(self.user_id, self.request, 'fingerprint', 0)
        self.assertEqual(get_token_balance(self.user_id), before)

    def test_insufficient_balance_and_table_multiplier(self):
        self.set_balance(1000)
        request = self.request.model_copy(update={'full_pattern': 'free11_512'})
        with self.assertRaises(InsufficientTokens):
            check_query_budget(self.user_id, request, 'fingerprint', 0)
        self.set_balance(0)
        with self.assertRaises(InsufficientTokens):
            check_query_budget(self.user_id, self.request, 'fingerprint', 0)

    def test_paid_retry_remains_available_with_empty_balance_but_conflict_is_rejected(self):
        consume_operation_tokens_once(request_id=f'gamer-ai:{self.request.request_id}:0',
            user_id=self.user_id, session_id=None, operation_key='trainer_lookup_hit',
            full_pattern=self.request.full_pattern, idempotency_scope='gamer-ai:fingerprint:0')
        self.set_balance(0)
        check_query_budget(self.user_id, self.request, 'fingerprint', 0)
        for user, fingerprint in ((self.user_id, 'changed'), (self.user_id + 1, 'fingerprint')):
            with self.assertRaises(HTTPException) as error:
                check_query_budget(user, self.request, fingerprint, 0)
            self.assertEqual(error.exception.status_code, 409)

    def test_global_discount_applies_to_admission(self):
        self.set_balance(500)
        with patch.dict(os.environ, {'CLOUD_TOKEN_GLOBAL_MULTIPLIER': '0.5'}):
            clear_token_pricing_cache()
            check_query_budget(self.user_id, self.request, 'fingerprint', 0)

    def test_stream_request_only_accepts_single_cursor(self):
        self.assertEqual(self.request.steps, 1)
        with self.assertRaises(ValidationError):
            RouteRequest(**{**self.request.model_dump(), 'steps': 4})

    def test_retired_http_route_is_not_registered(self):
        from backend.app import app

        self.assertFalse(any(getattr(route, 'path', '') == '/api/gamer/tablebase/route'
                             for route in app.routes))

    def test_cursor_remains_deterministic_and_stops_before_evil(self):
        options = {k: getattr(self.request, k) for k in
                   ('board_codes', 'rng_state', 'spawn_rate4', 'difficulty', 'random_only', 'steps')}
        a, b = GamerRouteCursor(options, 6), GamerRouteCursor(options, 6)
        for _ in range(3):
            self.assertEqual(a.node({'left': .9}, 'float64'), b.node({'left': .9}, 'float64'))
            self.assertEqual(a.advance({'left': .9}, 'float64'), b.advance({'left': .9}, 'float64'))
        evil = GamerRouteCursor({**options, 'difficulty': 100, 'random_only': True}, 6)
        self.assertTrue(evil.advance({'left': .9}, 'float64'))
        self.assertFalse(evil.advance({'right': .9}, 'float64'))


if __name__ == '__main__':
    unittest.main()
