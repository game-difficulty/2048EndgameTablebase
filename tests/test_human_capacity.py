import gzip
import json
import sqlite3
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from backend.auth.db import auth_db
from backend.human_play import service, traffic
from backend.human_play.store import database
from backend.http_compression import DisplayCompression, CacheControlledStaticFiles
from tests import test_human_play as fixtures
BROWSER, WRITER = fixtures.BROWSER, fixtures.WRITER


class CapacityTests(unittest.TestCase):
    setUp = fixtures.HumanPlayTests.setUp
    tearDown = fixtures.HumanPlayTests.tearDown
    new = fixtures.HumanPlayTests.new
    records = fixtures.HumanPlayTests.records
    send = fixtures.HumanPlayTests.send

    def test_signed_heartbeat_is_readonly_and_checks_current_epoch(self):
        run = self.new(threshold=0)
        raw, _, _ = self.records(run, until=lambda state: state['score'] > 0)
        checkpoint = self.send(run, raw, action='monitor')
        expiry = checkpoint['permit_until']
        with patch('backend.human_play.service.time.time', return_value=expiry - 2):
            renewed = service.online_check(1, BROWSER, run['run_id'], WRITER, 1, checkpoint['permit'])
        with database() as db:
            self.assertEqual(db.execute('SELECT permit_until FROM human_runs').fetchone()[0], expiry)
        with patch('backend.human_play.service.time.time', return_value=expiry + 1):
            result = service.online_check(1, BROWSER, run['run_id'], WRITER, 1, renewed['permit'])
            self.assertGreater(result['permit_until'], renewed['permit_until'])
            with self.assertRaisesRegex(service.RunError, 'reentry_required'):
                service.online_check(1, BROWSER, run['run_id'], WRITER, 1, renewed['permit'][:-1] + 'x')
            with database() as db:
                db.execute('UPDATE human_runs SET epoch=epoch+1')
            with self.assertRaisesRegex(service.RunError, 'reentry_required'):
                service.online_check(1, BROWSER, run['run_id'], WRITER, 2, renewed['permit'])

    def test_validation_releases_write_lock_and_rechecks_snapshot(self):
        run = self.new(); raw, _, _ = self.records(run, count=3)
        original = service.engine.advance
        changed = False
        def concurrent_change(*args, **kwargs):
            nonlocal changed
            if not changed:
                changed = True
                with database() as db:
                    db.execute('BEGIN IMMEDIATE')
                    db.execute('UPDATE human_runs SET epoch=epoch+1 WHERE id=?', (run['run_id'],))
            return original(*args, **kwargs)
        with patch.object(service.engine, 'advance', side_effect=concurrent_change):
            with self.assertRaisesRegex(service.RunError, 'progress_changed'):
                self.send(run, raw, reason='restarted')
        with database() as db:
            row = db.execute('SELECT archive,eligibility,state FROM human_runs').fetchone()
            self.assertIsNone(row['archive']); self.assertEqual(row['eligibility'], 'eligible')
            self.assertEqual(json.loads(row['state'])['seq'], 0)

    def test_display_reads_never_select_archive(self):
        run = self.new('2x4'); raw, _, _ = self.records(run); self.send(run, raw)
        original = service.database
        @contextmanager
        def restricted():
            with original() as db:
                db.set_authorizer(lambda op, table, col, *rest: sqlite3.SQLITE_DENY
                                  if op == sqlite3.SQLITE_READ and table == 'human_runs' and col == 'archive'
                                  else sqlite3.SQLITE_OK)
                yield db
        with patch.object(service, 'database', restricted):
            self.assertEqual(len(service.history(1, 1)['entries']), 1)
            self.assertIn('2x4', service.personal_bests(1)['bests'])
            item = service.leaderboard('2x4')['entries'][0]
            self.assertNotIn('nodes', item)
            self.assertEqual(set(item), {'id', 'user_id', 'score', 'max_tile', 'rank',
                                         'display_name', 'source', 'has_replay'})
            service.status(1, BROWSER, run['run_id'])

    def test_top_ten_limit_and_independent_bests(self):
        run = self.new('2x4'); raw, _, _ = self.records(run); self.send(run, raw)
        with auth_db() as db:
            for uid in range(3, 17):
                db.execute("INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at) VALUES(?,?,?,?,'now','now')",
                           (uid, f'{uid}@test.invalid', '!', f'Player {uid}'))
        with database() as db:
            for uid in range(3, 17):
                db.execute('''INSERT INTO human_runs (id,user_id,browser,variant,request_id,seed,threshold,status,
                    reason,created,ended,writer,state) SELECT ?,?,browser,variant,request_id,seed,threshold,status,
                    reason,created,ended,writer,state FROM human_runs WHERE id=?''', (str(uid), uid, run['run_id']))
        self.assertEqual(len(self.client.get('/api/human/leaderboards?variant=2x4').json()['entries']), 10)
        self.assertEqual(len(self.client.get('/api/human/leaderboards?variant=2x4&limit=100').json()['entries']), 15)
        self.assertEqual(set(self.client.get('/api/human/me/bests').json()), {'bests'})

    def test_heavy_admission_limits_and_small_lane_independence(self):
        first = traffic.acquire('1', 'large')
        with self.assertRaises(HTTPException) as error: traffic.acquire('1', 'large')
        self.assertEqual(error.exception.status_code, 429)
        second = traffic.acquire('2', 'large')
        with self.assertRaises(HTTPException): traffic.acquire('3', 'large')
        small = traffic.acquire('1', 'small')
        for token in (first, second, small): traffic.release(token)
        for _ in range(traffic.POLICIES['download'][2]):
            traffic.release(traffic.acquire('1', 'download'))
        with self.assertRaises(HTTPException): traffic.acquire('1', 'download')
        # Saturation is rejected before processing an invalid binary body.
        token = traffic.acquire('1', 'large')
        response = self.client.post('/api/human/runs/ignored/seal', content=b'bad', headers={
            'Content-Type': 'application/octet-stream', 'X-Human-Writer': WRITER,
            'X-Human-Epoch': '1', 'X-Human-Start': '0', 'X-Human-Count': '0'})
        self.assertEqual(response.status_code, 429)
        traffic.release(token)

    def test_heartbeat_auth_never_loads_profile_or_balance(self):
        with patch('backend.auth.service.public_user', side_effect=AssertionError('full profile queried')):
            response = self.client.get('/api/human/me/bests')
            self.assertEqual(response.status_code, 200)
        with auth_db() as db: db.execute('UPDATE sessions SET revoked_at=?', ('revoked',))
        self.assertEqual(self.client.get('/api/human/me/bests').status_code, 401)

    def test_replay_download_releases_slot_and_rate_limits(self):
        run = self.new('2x4'); raw, _, _ = self.records(run); self.send(run, raw)
        for _ in range(traffic.POLICIES['download'][2]):
            response = self.client.get('/api/human/replays/' + run['run_id'])
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.content.startswith(b'HPR2'))
            with traffic.connection() as db:
                self.assertEqual(db.execute('SELECT count(*) FROM slots').fetchone()[0], 0)
        response = self.client.get('/api/human/replays/' + run['run_id'])
        self.assertEqual(response.status_code, 429)
        self.assertIn('retry-after', response.headers)


class CompressionTests(unittest.TestCase):
    def test_json_compression_and_static_negotiation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'asset.js'
            data = b'const repeated = 123;\n' * 100
            path.write_bytes(data); path.with_suffix('.js.gz').write_bytes(gzip.compress(data))
            app = FastAPI(); app.add_middleware(DisplayCompression)
            @app.get('/api/human/leaderboards')
            def board(): return {'entries': ['test' * 100] * 10}
            app.mount('/assets', CacheControlledStaticFiles(directory=directory, cache_control='public, max-age=31536000, immutable'))
            with TestClient(app) as client:
                response = client.get('/api/human/leaderboards', headers={'Accept-Encoding': 'gzip'})
                self.assertEqual(response.headers['content-encoding'], 'gzip')
                compressed = client.get('/assets/asset.js', headers={'Accept-Encoding': 'gzip'})
                self.assertEqual(compressed.content, data)
                self.assertEqual(compressed.headers['content-encoding'], 'gzip')
                self.assertIn('javascript', compressed.headers['content-type'])
                self.assertIn('immutable', compressed.headers['cache-control'])
                plain = client.get('/assets/asset.js', headers={'Accept-Encoding': 'gzip;q=0'})
                self.assertNotIn('content-encoding', plain.headers)
                self.assertEqual(plain.content, data)
                cached = client.get('/assets/asset.js', headers={'Accept-Encoding': 'gzip', 'If-None-Match': compressed.headers['etag']})
                self.assertEqual(cached.status_code, 304)
                self.assertIn('Accept-Encoding', cached.headers['vary'])
