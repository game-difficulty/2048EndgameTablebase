from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import service
from backend.auth.db import auth_db, init_auth_db
from backend.auth.routes import router
from backend.auth.security import hash_password, hash_token, verify_password


class EmailVerificationAttemptsTests(unittest.TestCase):
    purposes = ('register', 'password_reset', 'account_deactivate')
    password = 'Original-password-2048'
    correct = '123456'

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            'CLOUD_AUTH_DB': str(Path(self.directory.name) / 'auth.sqlite3'),
            'AUTH_REGISTRATION_EMAIL_DOMAINS': 'gmail.com',
        })
        self.env.start()
        init_auth_db()
        with auth_db() as db:
            db.execute('''INSERT INTO users
                (id,email,email_identity,password_hash,display_name,display_name_key,created_at,updated_at)
                VALUES (1,?,?,?,?,?,?,?)''', ('owner@gmail.com', 'owner@gmail.com', hash_password(self.password),
                'Owner', 'owner', service.iso(), service.iso()))
            self.token = service.create_session(db, 1)[0]

    def tearDown(self):
        self.env.stop()
        self.directory.cleanup()

    def issue(self, purpose, *, expired=False):
        email = 'new@gmail.com' if purpose == 'register' else 'owner@gmail.com'
        with auth_db() as db:
            return db.execute('''INSERT INTO email_verification_codes
                (email,purpose,code_hash,created_at,expires_at) VALUES (?,?,?,?,?)''',
                (email, purpose, hash_token(self.correct), service.iso(),
                 service.iso(service.utcnow() + timedelta(minutes=-1 if expired else 10)))).lastrowid

    def consume(self, purpose, code):
        if purpose == 'register':
            return service.register_user(email='new@gmail.com', password=self.password, invite_code='',
                                         verification_code=code, display_name='New Player')
        if purpose == 'password_reset':
            return service.reset_password(email='owner@gmail.com', verification_code=code,
                                          new_password='Replacement-password-2048')
        return service.deactivate_account(user_id=1, password=self.password, confirm='DELETE', verification_code=code)

    def row(self, code_id):
        with auth_db() as db:
            return dict(db.execute('SELECT * FROM email_verification_codes WHERE id=?', (code_id,)).fetchone())

    def assert_account_unchanged(self):
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM users').fetchone()[0], 1)
            user = db.execute('SELECT * FROM users WHERE id=1').fetchone()
            self.assertEqual(user['status'], 'active')
            self.assertTrue(verify_password(self.password, user['password_hash']))
        self.assertIsNotNone(service.authenticate_session_token(self.token))

    def test_failures_persist_and_lock_all_three_flows(self):
        for purpose in self.purposes:
            with self.subTest(purpose=purpose):
                code_id = self.issue(purpose)
                for count in range(1, 6):
                    with self.assertRaisesRegex(ValueError, 'Invalid verification code'):
                        self.consume(purpose, '000000')
                    self.assertEqual(self.row(code_id)['attempts'], count)
                for code in ('000000', self.correct):
                    with self.assertRaisesRegex(ValueError, 'Too many verification attempts'):
                        self.consume(purpose, code)
                self.assertEqual(self.row(code_id)['attempts'], 5)
                self.assertIsNone(self.row(code_id)['consumed_at'])
                self.assert_account_unchanged()

    def test_correct_code_succeeds_on_last_attempt_and_cannot_be_reused(self):
        for purpose in self.purposes:
            with self.subTest(purpose=purpose):
                code_id = self.issue(purpose)
                for _ in range(4):
                    with self.assertRaises(ValueError):
                        self.consume(purpose, '000000')
                self.consume(purpose, self.correct)
                self.assertEqual(self.row(code_id)['attempts'], 5)
                self.assertIsNotNone(self.row(code_id)['consumed_at'])
                with self.assertRaises(ValueError):
                    self.consume(purpose, self.correct)
                # Keep the account suitable for the next independent flow.
                if purpose == 'password_reset':
                    with auth_db() as db:
                        db.execute('UPDATE users SET password_hash=? WHERE id=1', (hash_password(self.password),))

    def test_business_failure_rolls_back_code_consumption_and_account_changes(self):
        for purpose in self.purposes:
            with self.subTest(purpose=purpose):
                code_id = self.issue(purpose)
                with self.assertRaises(ValueError):
                    self.consume(purpose, '000000')
                before = self.row(code_id)
                target = '_revoke_user_sessions' if purpose == 'account_deactivate' else 'create_session'
                with patch.object(service, target, side_effect=RuntimeError('downstream failure')):
                    with self.assertRaisesRegex(RuntimeError, 'downstream failure'):
                        self.consume(purpose, self.correct)
                self.assertEqual(self.row(code_id), before)
                self.assert_account_unchanged()

    def test_expired_missing_and_other_purpose_codes_do_not_consume_attempts(self):
        reset_id = self.issue('password_reset')
        with self.assertRaisesRegex(ValueError, 'not found'):
            self.consume('account_deactivate', self.correct)
        self.assertEqual(self.row(reset_id)['attempts'], 0)
        for purpose in self.purposes:
            code_id = self.issue(purpose, expired=True)
            with self.assertRaisesRegex(ValueError, 'expired'):
                self.consume(purpose, self.correct)
            self.assertEqual(self.row(code_id)['attempts'], 0)

    def test_new_code_recovers_after_attempt_limit(self):
        old_id = self.issue('password_reset')
        for _ in range(5):
            with self.assertRaises(ValueError):
                self.consume('password_reset', '000000')
        new_id = self.issue('password_reset')
        self.consume('password_reset', self.correct)
        self.assertEqual(self.row(old_id)['attempts'], 5)
        self.assertIsNotNone(self.row(new_id)['consumed_at'])

    def test_concurrent_wrong_codes_cannot_exceed_attempt_limit(self):
        code_id = self.issue('password_reset')
        barrier = threading.Barrier(8)
        def attempt(_):
            barrier.wait(timeout=10)
            try:
                self.consume('password_reset', '000000')
            except ValueError as exc:
                return str(exc)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(8)))
        self.assertEqual(results.count('Invalid verification code.'), 5)
        self.assertEqual(results.count('Too many verification attempts.'), 3)
        self.assertEqual(self.row(code_id)['attempts'], 5)

    def test_concurrent_correct_code_is_consumed_once(self):
        code_id = self.issue('password_reset')
        barrier = threading.Barrier(2)
        def attempt(_):
            barrier.wait(timeout=10)
            try:
                self.consume('password_reset', self.correct)
                return 'success'
            except ValueError as exc:
                return str(exc)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, range(2)))
        self.assertEqual(results.count('success'), 1)
        self.assertEqual(self.row(code_id)['attempts'], 1)

    def test_http_error_response_keeps_failed_attempt(self):
        code_id = self.issue('password_reset')
        app = FastAPI(); app.include_router(router)
        with TestClient(app) as client:
            response = client.post('/api/auth/reset-password', json={
                'email': 'owner@gmail.com', 'verification_code': '000000', 'new_password': self.password,
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['detail'], 'Invalid verification code.')
        self.assertEqual(self.row(code_id)['attempts'], 1)

    def test_wrong_code_persists_only_attempt_not_other_transaction_writes(self):
        code_id = self.issue('password_reset')
        with self.assertRaisesRegex(ValueError, 'Invalid verification code'):
            with service._email_code_transaction() as db:
                db.execute("UPDATE users SET status='disabled' WHERE id=1")
                service._consume_email_code(db, email='owner@gmail.com', code='000000', purpose='password_reset')
        self.assertEqual(self.row(code_id)['attempts'], 1)
        self.assert_account_unchanged()


if __name__ == '__main__':
    unittest.main()
