import os
import tempfile
import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth.daily_activity import activity_day, record_daily_visit, site_from_host
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


if __name__ == '__main__':
    unittest.main()
