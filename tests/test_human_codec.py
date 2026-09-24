import gzip
import hashlib
import json
import os
import sqlite3
import struct
import base64
import subprocess
from pathlib import Path
import unittest

from backend.human_play import codec, engine, service, admin, permits
from backend.human_play.store import database, init_db
from tests import test_human_play as fixtures

BROWSER, WRITER = fixtures.BROWSER, fixtures.WRITER


class CodecTests(unittest.TestCase):
    def test_planes_roundtrip_uint32_extremes_and_gzip_bounds(self):
        raw = b''.join(engine.EVENT.pack(i % 128, v) for i, v in enumerate([0, 127, 128, 65535, 65536, 2**31, 2**32 - 1]))
        self.assertEqual(codec.from_planes(codec.to_planes(raw)), raw)
        self.assertEqual(codec.gunzip_limited(gzip.compress(raw)), raw)
        for packed in (gzip.compress(raw)[:-1], gzip.compress(raw) + b'garbage', gzip.compress(raw) * 2):
            with self.assertRaisesRegex(ValueError, 'invalid_gzip'): codec.gunzip_limited(packed)
        with self.assertRaisesRegex(ValueError, 'record_too_large'):
            codec.gunzip_limited(gzip.compress(b'x' * 1000001))
        with self.assertRaises(ValueError): codec.from_planes(b'bad')

    def test_replay_versions_recover_identical_events_and_metadata(self):
        run = dict(id='codec-fixture', variant='4x4', seed=fixtures.SEED, reason='restarted', created=123.125)
        raw = b''.join(engine.EVENT.pack(i % 128, i * 713) for i in range(2000))
        for version in (1, 2):
            binary = engine.replay_bytes(run, raw, version)
            header, restored = engine.parse_replay(codec.gunzip_limited(gzip.compress(binary), codec.MAX_REPLAY_BYTES))
            self.assertEqual(restored, raw)
            self.assertEqual(header['started_at'], 123.125)
            self.assertEqual(header['seed'], fixtures.SEED)
            self.assertEqual(header['version'], version)
        self.assertLess(len(gzip.compress(engine.replay_bytes(run, raw, 2))), len(gzip.compress(engine.replay_bytes(run, raw, 1))))


class WireIntegrationTests(unittest.TestCase):
    setUp = fixtures.HumanPlayTests.setUp
    tearDown = fixtures.HumanPlayTests.tearDown
    new = fixtures.HumanPlayTests.new
    records = fixtures.HumanPlayTests.records
    send = fixtures.HumanPlayTests.send

    def headers(self, run, count, **extra):
        return {'Content-Type': 'application/octet-stream', 'X-Human-Browser': BROWSER,
                'X-Human-Writer': WRITER, 'X-Human-Epoch': '1', 'X-Human-Start': '0',
                'X-Human-Count': str(count), 'X-Human-Prefix': run['prefix_hash'], **extra}

    def test_compressed_and_plain_retries_are_identical_batches(self):
        run = self.new(threshold=8)
        raw, state, _ = self.records(run, until=lambda s: s['score'] > 8)
        response = self.client.post(f"/api/human/runs/{run['run_id']}/monitor",
            content=gzip.compress(codec.to_planes(raw)), headers=self.headers(run, state['seq'], **{
                'Content-Encoding': 'gzip', 'X-Human-Layout': 'planes5', 'X-Human-Protocol': '2'}))
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload[:4], [2, state['seq'], 1, 4])
        self.assertLess(len(response.content), 140)
        retry = self.client.post(f"/api/human/runs/{run['run_id']}/monitor", content=raw,
                                 headers=self.headers(run, state['seq']))
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(retry.json()['prefix_hash'], state['hash'])
        with database() as db:
            rows = db.execute('SELECT * FROM human_chunks').fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['data'], raw)
            self.assertEqual(rows[0]['digest'], hashlib.sha256(raw).digest())
            self.assertEqual(len(rows[0]['previous_hash']), 32)
            self.assertEqual(rows[0]['count'], state['seq'])
        self.assertEqual(service.status(1, BROWSER, run['run_id'])['status'], 'active')

    def test_gzip_bomb_and_damaged_payload_never_change_evidence(self):
        run = self.new()
        for packed, status in [(gzip.compress(b'x' * 1000001), 413), (gzip.compress(b'12345')[:-4], 400)]:
            result = self.client.post(f"/api/human/runs/{run['run_id']}/seal", content=packed,
                headers=self.headers(run, 1, **{'Content-Encoding': 'gzip', 'X-Human-Reason': 'restarted'}))
            self.assertEqual(result.status_code, status, result.text)
        self.assertEqual(service.status(1, BROWSER, run['run_id'])['seq'], 0)
        self.assertEqual(service.status(1, BROWSER, run['run_id'])['eligibility'], 'eligible')

    def test_legacy_chunk_migration_preserves_evidence_and_retry(self):
        run = self.new(threshold=8); raw, state, _ = self.records(run, until=lambda s: s['score'] > 8)
        self.send(run, raw, action='monitor')
        with database() as db:
            original = dict(db.execute('SELECT * FROM human_chunks').fetchone())
            db.execute('DROP TABLE human_chunks')
            db.execute('''CREATE TABLE human_chunks(run_id TEXT REFERENCES human_runs(id), start INTEGER,
                count INTEGER, digest TEXT, previous_hash TEXT, data BLOB, received REAL, PRIMARY KEY(run_id,start))''')
            db.execute('INSERT INTO human_chunks VALUES(?,?,?,?,?,?,?)', (original['run_id'], original['start'],
                original['count'], original['digest'].hex(), original['previous_hash'].hex(), raw, original['received']))
        init_db()
        with database() as db:
            self.assertEqual(dict(db.execute('SELECT * FROM human_chunks').fetchone()), original)
            self.assertIn('WITHOUT ROWID', db.execute("SELECT sql FROM sqlite_master WHERE name='human_chunks'").fetchone()[0])
            admin.verified_events(db, admin.get_run(db, run['run_id']))
        self.assertEqual(self.send(run, raw, action='monitor')['seq'], state['seq'])
        self.assertEqual(service.status(1, BROWSER, run['run_id'])['status'], 'active')

    def test_invalid_migration_rolls_back_instead_of_discarding_evidence(self):
        run = self.new()
        with database() as db:
            db.execute('DROP TABLE human_chunks')
            db.execute('''CREATE TABLE human_chunks(run_id TEXT, start INTEGER, count INTEGER,
                digest TEXT, previous_hash TEXT, data BLOB, received REAL, PRIMARY KEY(run_id,start))''')
            db.execute('INSERT INTO human_chunks VALUES(?,?,?,?,?,?,?)', (run['run_id'], 0, 1, 'broken', 'a'*64, b'12345', 1))
        with self.assertRaises(sqlite3.OperationalError): init_db()
        with database() as db:
            self.assertEqual(db.execute('SELECT digest,data FROM human_chunks').fetchone()[:], ('broken', b'12345'))
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='human_chunks_compact'").fetchone())

    def test_legacy_archive_review_preserves_exact_blob(self):
        run = self.new(); raw, state, _ = self.records(run, count=5)
        self.send(run, raw, reason='restarted')
        with database() as db:
            row = admin.get_run(db, run['run_id'])
            old_blob = gzip.compress(engine.replay_bytes(row, raw, version=1), mtime=0)
            db.execute('UPDATE human_runs SET archive=? WHERE id=?', (old_blob, run['run_id']))
        admin.review(run['run_id'], approved=True, operator='test', note='legacy audit',
                     expected_seq=state['seq'], expected_hash=state['hash'])
        self.assertEqual(service.replay(run['run_id'], 2), old_blob)

    def test_old_and_short_signature_encodings_verify_the_same_hmac(self):
        run = self.new()
        with database() as db: row = admin.get_run(db, run['run_id'])
        until = __import__('time').time() + 12
        token = permits.issue(row, until)
        expiry = token.split('.')[0]
        old = expiry + '.' + permits._signature(row, expiry).hex()
        self.assertTrue(permits.valid(row, token)); self.assertTrue(permits.valid(row, old))
        self.assertEqual(len(old) - len(token), 21)

    def test_browser_reads_backend_archives_for_all_variants_and_versions(self):
        fixtures_to_read = []; expected = []
        for variant in engine.VARIANTS:
            run = self.new(variant); raw, state, _ = self.records(run, count=80)
            with database() as db: row = admin.get_run(db, run['run_id'])
            for version in (1, 2):
                fixtures_to_read.append(base64.b64encode(engine.replay_bytes(row, raw, version)).decode())
                expected.append({key: state[key] for key in ('board', 'seq', 'score', 'elapsed')})
        script = """import {parseReplay,buildReplay} from './frontend/src/human/engine.js';
        const results=JSON.parse(process.argv[1]).map(value=>{
          const replay=buildReplay(parseReplay(Uint8Array.from(Buffer.from(value,'base64')).buffer));
          const {board,seq,score,elapsed}=replay.seek(replay.total); return {board,seq,score,elapsed};
        });console.log(JSON.stringify(results));"""
        result = subprocess.check_output(['node', '--input-type=module', '-e', script, json.dumps(fixtures_to_read)],
                                         cwd=Path(__file__).resolve().parents[1], text=True)
        self.assertEqual(json.loads(result), expected)
