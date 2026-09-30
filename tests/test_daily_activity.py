import os
import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI, Depends, Request, WebSocket
from fastapi.testclient import TestClient

from backend.auth.daily_activity import activity_day, record_daily_visit, site_from_host
from backend.auth.activity_middleware import DailyActivityMiddleware
from backend.auth.dependencies import require_user, require_identity, current_user_from_websocket
from competition.backend.auth import principal_from_request
from competition.backend.config import load_settings
from backend.auth.db import auth_db, init_auth_db
from backend.auth.routes import router as auth_router
from backend.auth.service import create_session
from backend.admin.routes import _daily_token_activity
from backend.human_play.store import database, init_db
from scripts.refresh_daily_activity import refresh


class DailyActivityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            'CLOUD_AUTH_DB': self.temp.name + '/auth.db',
            'HUMAN_PLAY_DB': self.temp.name + '/human.db',
        })
        self.env.start()
        init_auth_db()
        init_db()
        with auth_db() as db:
            for user_id in range(1, 5):
                db.execute('''INSERT INTO users
                    (id, email, password_hash, created_at, updated_at)
                    VALUES (?, ?, '!', '2026-09-26T00:00:00+00:00',
                            '2026-09-26T00:00:00+00:00')''',
                           (user_id, f'user{user_id}@test.invalid'))

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_site_visits_deduplicate_account_across_sites(self):
        now = datetime(2026, 9, 26, 16, 1, tzinfo=timezone.utc)
        self.assertEqual(activity_day(now), '2026-09-27')
        self.assertEqual(site_from_host('play.2048tables.online'), 'play')
        self.assertEqual(site_from_host('live.2048tables.online'), 'live')
        self.assertEqual(site_from_host('tournament.2048tables.online:443'), 'tournament')
        record_daily_visit(1, 'main', now=now)
        record_daily_visit(1, 'play', now=now)
        record_daily_visit(1, 'live', now=now)
        record_daily_visit(1, 'live', now=now)
        with auth_db() as db:
            rows = db.execute('SELECT day, site FROM daily_user_activity ORDER BY site').fetchall()
            self.assertEqual([(row['day'], row['site']) for row in rows],
                             [('2026-09-27', 'live'), ('2026-09-27', 'main'),
                              ('2026-09-27', 'play')])
            count = db.execute('SELECT COUNT(DISTINCT user_id) FROM daily_user_activity').fetchone()[0]
            self.assertEqual(count, 1)

    def test_auth_me_records_host_site(self):
        with auth_db() as db:
            token = create_session(db, 1)[0]
        app = FastAPI()
        app.add_middleware(DailyActivityMiddleware)
        app.include_router(auth_router)
        with TestClient(app) as client:
            for host in ('2048tables.online', 'play.2048tables.online',
                         'live.2048tables.online'):
                response = client.get('/api/auth/me', headers={
                    'Host': host, 'Authorization': f'Bearer {token}',
                })
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()['authenticated'])
        with auth_db() as db:
            sites = {row['site'] for row in db.execute(
                'SELECT site FROM daily_user_activity WHERE user_id=1')}
        self.assertEqual(sites, {'main', 'play', 'live'})

    def test_feature_requests_record_all_sites_without_auth_me(self):
        with auth_db() as db:
            token = create_session(db, 1)[0]
        app = FastAPI()
        app.add_middleware(DailyActivityMiddleware)

        @app.get('/feature')
        def feature(user=Depends(require_user)):
            return {'id': user['id']}

        @app.get('/play-feature')
        def play_feature(user=Depends(require_identity)):
            return {'id': user['id']}

        @app.get('/competition-feature')
        def competition_feature(request: Request):
            return {'id': principal_from_request(request, load_settings()).user_id}

        with TestClient(app) as client:
            for host, path in [('2048tables.online', '/feature'),
                               ('live.2048tables.online', '/feature'),
                               ('play.2048tables.online', '/play-feature'),
                               ('tournament.2048tables.online', '/competition-feature')]:
                for _ in range(2):
                    response = client.get(path, headers={'Host': host, 'Authorization': f'Bearer {token}'})
                    self.assertEqual(response.status_code, 200)
            self.assertEqual(client.get('/feature').status_code, 401)
        with auth_db() as db:
            rows = db.execute('SELECT site FROM daily_user_activity').fetchall()
        self.assertEqual({row['site'] for row in rows}, {'main', 'play', 'live', 'tournament'})
        self.assertEqual(len(rows), 4)

    def test_cached_visit_does_not_write_again_and_rolls_over(self):
        with patch('backend.auth.daily_activity.auth_db', wraps=auth_db) as connection:
            record_daily_visit(1, 'main', now=datetime(2026, 9, 30, 15, 59, tzinfo=timezone.utc))
            record_daily_visit(1, 'main', now=datetime(2026, 9, 30, 15, 59, tzinfo=timezone.utc))
            record_daily_visit(1, 'main', now=datetime(2026, 9, 30, 16, 1, tzinfo=timezone.utc))
            self.assertEqual(connection.call_count, 2)

    def test_failed_write_is_not_cached(self):
        with patch('backend.auth.daily_activity.auth_db', side_effect=RuntimeError('busy')):
            with self.assertRaises(RuntimeError):
                record_daily_visit(1, 'tournament')
        record_daily_visit(1, 'tournament')
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM daily_user_activity').fetchone()[0], 1)

    def test_websocket_cross_day_records_incoming_and_outgoing_activity(self):
        with auth_db() as db:
            token = create_session(db, 1)[0]
        app = FastAPI()
        app.add_middleware(DailyActivityMiddleware, site='live')

        @app.websocket('/watch')
        async def watch(ws: WebSocket):
            user = current_user_from_websocket(ws)
            await ws.accept()
            await ws.send_json({'authenticated': bool(user)})
            await ws.receive_text()
            await ws.send_text('pong')

        day = ['2026-09-30']
        with patch('backend.auth.activity_middleware.activity_day', side_effect=lambda: day[0]), \
                patch('backend.auth.daily_activity.activity_day', side_effect=lambda now=None: day[0]), \
                TestClient(app) as client:
            client.cookies.set('tb_session', token)
            with client.websocket_connect('/watch') as ws:
                self.assertTrue(ws.receive_json()['authenticated'])
                day[0] = '2026-10-01'
                ws.send_text('ping')
                self.assertEqual(ws.receive_text(), 'pong')
        with auth_db() as db:
            rows = db.execute('SELECT day, site FROM daily_user_activity ORDER BY day').fetchall()
        self.assertEqual([(row['day'], row['site']) for row in rows],
                         [('2026-09-30', 'live'), ('2026-10-01', 'live')])

    def test_activity_failure_does_not_break_feature_response(self):
        with auth_db() as db:
            token = create_session(db, 1)[0]
        app = FastAPI()
        app.add_middleware(DailyActivityMiddleware)

        @app.get('/feature')
        def feature(user=Depends(require_identity)):
            return {'id': user['id']}

        with patch('backend.auth.activity_middleware.record_daily_visit', side_effect=RuntimeError('busy')), \
                self.assertLogs('backend.auth.activity_middleware', level='ERROR'), TestClient(app) as client:
            response = client.get('/feature', headers={'Authorization': f'Bearer {token}'})
        self.assertEqual(response.status_code, 200)

    def test_competition_app_session_records_real_account(self):
        from competition.backend.app import create_app
        with auth_db() as db:
            token = create_session(db, 1)[0]
        with patch.dict(os.environ, {'COMPETITION_DB': self.temp.name + '/competition.db'}):
            app = create_app()
            with TestClient(app) as client:
                response = client.get('/api/session', headers={'Authorization': f'Bearer {token}'})
        self.assertEqual(response.status_code, 200)
        with auth_db() as db:
            rows = db.execute('SELECT site FROM daily_user_activity').fetchall()
        self.assertEqual([row['site'] for row in rows], ['tournament'])

    def test_competition_socket_token_auth_marks_account_but_dev_auth_does_not(self):
        from dataclasses import replace
        from competition.backend.auth import principal_from_socket_auth
        with auth_db() as db:
            token = create_session(db, 1)[0]
        settings = replace(load_settings(), allow_dev_auth=True)
        scope = {'type': 'websocket', 'path': '/ws/rooms/test', 'headers': []}
        ws = WebSocket(scope, AsyncMock(), AsyncMock())
        principal = principal_from_socket_auth({'dev_user': '1:Test:user'}, settings, ws)
        self.assertEqual(principal.user_id, 1)
        self.assertFalse(hasattr(ws.state, 'daily_activity_user_id'))
        principal = principal_from_socket_auth({'token': token}, settings, ws)
        self.assertEqual(principal.user_id, 1)
        self.assertEqual(ws.state.daily_activity_user_id, 1)

    def test_backfill_includes_play_and_ignores_automatic_grants(self):
        with auth_db() as db:
            db.execute('''INSERT INTO sessions
                (user_id,session_token_hash,created_at,expires_at)
                VALUES (1,'s1','2026-09-26T15:59:00+00:00',
                        '2026-10-01T00:00:00+00:00')''')
            db.execute('''INSERT INTO usage_events(user_id,event_type,created_at)
                VALUES (2,'replay_load','2026-09-26T16:01:00+00:00')''')
            db.execute('''INSERT INTO token_ledger(user_id,event_type,created_at)
                VALUES (2,'reserve','2026-09-26T16:02:00+00:00')''')
            db.execute('''INSERT INTO token_ledger(user_id,event_type,created_at)
                VALUES (4,'weekly_grant','2026-09-26T16:02:00+00:00')''')
        with database() as db:
            db.execute('''INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,created,writer,state)
                VALUES ('run-3',3,'browser-3','4x4','request-3','seed',20,
                        1790438340,'writer','{}')''')
            db.execute('''INSERT INTO human_chunks
                (run_id,start,count,digest,previous_hash,data,received)
                VALUES ('run-3',0,1,X'01',X'00',X'02',1790438460)''')
        first = refresh(first_day=date(2026, 9, 26), last_day=date(2026, 9, 27))
        second = refresh(first_day=date(2026, 9, 26), last_day=date(2026, 9, 27))
        self.assertEqual(first['days'], second['days'])
        with auth_db() as db:
            by_day = {
                row['day']: {item['user_id'] for item in db.execute(
                    'SELECT user_id FROM daily_user_activity WHERE day=?', (row['day'],))}
                for row in first['days']
            }
        self.assertEqual(by_day['2026-09-26'], {1, 3})
        self.assertEqual(by_day['2026-09-27'], {2, 3})
        self.assertNotIn(4, by_day['2026-09-27'])

    def test_admin_chart_shares_beijing_day_and_deduplicates_sites(self):
        record_daily_visit(1, 'main')
        record_daily_visit(1, 'live')
        with auth_db() as db:
            db.execute('''INSERT INTO token_ledger
                (user_id,event_type,final_cost_units,created_at)
                VALUES (1,'consume',1200,?)''',
                (datetime.now(timezone.utc).isoformat(),))
            row = _daily_token_activity(db, 14)[-1]
        self.assertEqual(row['date'], activity_day())
        self.assertEqual(row['active_accounts'], 1)
        self.assertEqual(row['spending_users'], 1)
        self.assertEqual(row['tokens_spent'], 1.2)

    def test_explicit_missing_play_database_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            refresh(first_day=date(2026, 9, 26), last_day=date(2026, 9, 27),
                    play_db=self.temp.name + '/missing.db')

    def test_competition_backfill_uses_actual_actors_and_beijing_dates(self):
        path = self.temp.name + '/competition-history.db'
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE competition_commands(user_id INTEGER, created_at TEXT)')
            db.execute('CREATE TABLE practice_bests(user_id INTEGER, achieved_at TEXT)')
            db.executemany('INSERT INTO competition_commands VALUES (?, ?)',
                           [(1, '2026-09-27T00:01:00+08:00'), (99, '2026-09-26T16:01:00Z')])
            db.execute("INSERT INTO practice_bests VALUES (2, '2026-09-26T16:02:00Z')")
        db.close()
        for _ in range(2):
            refresh(first_day=date(2026, 9, 27), last_day=date(2026, 9, 27), competition_db=path)
        with auth_db() as db:
            rows = db.execute('SELECT day, user_id, site FROM daily_user_activity ORDER BY user_id').fetchall()
        self.assertEqual([tuple(row) for row in rows],
                         [('2026-09-27', 1, 'tournament'), ('2026-09-27', 2, 'tournament')])


if __name__ == '__main__':
    unittest.main()
