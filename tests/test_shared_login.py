import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from backend.auth import routes
from backend.auth.db import auth_db, init_auth_db
from backend.auth.dependencies import current_user_from_websocket
from backend.auth.service import create_session, authenticate_session_token


class SharedLoginTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'CLOUD_AUTH_DB':str(Path(self.directory.name)/'auth.db'),
            'AUTH_COOKIE_SECURE':'1','AUTH_SHARED_COOKIE_DOMAIN':'2048tables.online'})
        self.env.start()
        init_auth_db()
        self.sessions = []
        with auth_db() as db:
            for uid in [1,2]:
                db.execute('''INSERT INTO users(id,email,email_identity,password_hash,display_name,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?)''', (uid,f'{uid}@test.invalid',f'{uid}@test.invalid','test',f'User {uid}','now','now'))
                self.sessions.append(create_session(db,uid))
        token, _, expiry = self.sessions[0]
        self.result = dict(token=token, expires_at=expiry, user={'id':1})
        app = FastAPI()
        app.include_router(routes.router)

        @app.websocket('/test-auth')
        async def ws(websocket: WebSocket):
            await websocket.accept()
            user = current_user_from_websocket(websocket)
            await websocket.send_json({'id':user['id'] if user else None})
            await websocket.close()

        self.client = TestClient(app, base_url='https://2048tables.online')
        self.balance = patch.object(routes,'grant_weekly_tokens_if_due',return_value={})
        self.balance.start()

    def tearDown(self):
        self.client.close()
        self.balance.stop()
        self.env.stop()
        self.directory.cleanup()

    def login(self, url='https://2048tables.online/api/auth/login'):
        with patch.object(routes,'login_user',return_value=self.result):
            return self.client.post(url,json={'email':'test','password':'test'})

    def test_first_main_login_is_available_on_first_live_visit_and_websocket(self):
        response=self.login()
        cookie=next(c for c in response.headers.get_list('set-cookie') if c.startswith('tb_shared_session='))
        for attribute in ['Domain=2048tables.online','HttpOnly','Secure','SameSite=lax','Path=/']:
            self.assertIn(attribute,cookie)
        self.assertTrue(self.client.get('https://live.2048tables.online/api/auth/me').json()['authenticated'])
        with self.client.websocket_connect('wss://live.2048tables.online/test-auth') as ws:
            self.assertEqual(ws.receive_json(),{'id':1})

    def test_shared_identity_wins_over_old_live_cookie_and_bearer(self):
        a,b=self.sessions[0][0],self.sessions[1][0]
        headers={'cookie':f'tb_session={b}; tb_shared_session={a}','authorization':f'Bearer {b}'}
        response=self.client.get('https://live.2048tables.online/api/auth/me',headers=headers)
        self.assertEqual(response.json()['user']['id'],1)
        self.assertIn('tb_session=""',response.headers['set-cookie'])
        self.assertEqual(response.json()['device_session_token'],a)
        with self.client.websocket_connect('wss://live.2048tables.online/test-auth',headers={'cookie':headers['cookie']}) as ws:
            self.assertEqual(ws.receive_json(),{'id':1})

    def test_legacy_main_cookie_upgrades_without_new_session_or_longer_expiry(self):
        token,session_id,expiry=self.sessions[0]
        response=self.client.get('/api/auth/me',headers={'cookie':f'tb_session={token}'})
        self.assertEqual(response.status_code,200)
        self.assertIn('tb_shared_session=',response.headers['set-cookie'])
        self.assertIn('tb_session=""',response.headers['set-cookie'])
        self.assertEqual(response.headers['cache-control'],'no-store')
        self.assertTrue(self.client.get('https://live.2048tables.online/api/auth/me').json()['authenticated'])
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM sessions').fetchone()[0],2)
            self.assertEqual(db.execute('SELECT expires_at FROM sessions WHERE id=?',(session_id,)).fetchone()[0],expiry)

    def test_bearer_only_legacy_login_also_upgrades(self):
        response=self.client.get('/api/auth/me',headers={'authorization':f'Bearer {self.sessions[0][0]}'})
        self.assertIn('tb_shared_session=',response.headers['set-cookie'])

    def test_live_logout_revokes_shared_and_legacy_sessions(self):
        self.login()
        old=self.sessions[1][0]
        response=self.client.post('https://live.2048tables.online/api/auth/logout',headers={'authorization':f'Bearer {old}'})
        self.assertIn('tb_shared_session=""',response.headers['set-cookie'])
        self.assertIsNone(authenticate_session_token(self.sessions[0][0]))
        self.assertIsNone(authenticate_session_token(old))
        self.assertFalse(self.client.get('/api/auth/me').json()['authenticated'])

    def test_localhost_and_unrelated_hosts_keep_host_only_cookies(self):
        for host in ['localhost','127.0.0.1','other.invalid','2048tables.online.attacker.invalid']:
            response=self.login(f'https://{host}/api/auth/login')
            self.assertIn('tb_session=',response.headers['set-cookie'])
            self.assertNotIn('Domain=',response.headers['set-cookie'])
            self.assertNotIn('tb_shared_session',response.headers['set-cookie'])

    def test_new_registration_and_password_reset_also_share_session(self):
        for endpoint, function in [('register','register_user'),('reset-password','reset_password')]:
            with patch.object(routes,function,return_value=self.result):
                response=self.client.post('/api/auth/'+endpoint,json={})
                self.assertEqual(response.status_code,200)
                self.assertIn('tb_shared_session=',response.headers['set-cookie'])

    def test_live_login_shared_back_to_main_and_expired_sessions_do_not_migrate(self):
        self.login('https://live.2048tables.online/api/auth/login')
        self.assertTrue(self.client.get('/api/auth/me').json()['authenticated'])
        self.client.cookies.clear()
        with auth_db() as db:
            db.execute("UPDATE sessions SET expires_at='2000-01-01T00:00:00+00:00'")
        response=self.client.get('/api/auth/me',headers={'cookie':f'tb_session={self.sessions[0][0]}'})
        self.assertFalse(response.json()['authenticated'])
        self.assertNotIn('set-cookie',response.headers)
