import gzip
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from backend import assistance_evidence as evidence
from backend.lookup_mask import lookup_key, play_lookup_key, descriptor
from backend.human_play import engine, service
from backend.human_play.assistance_review import Matcher, decide
from backend.human_play.store import init_db, database
from backend.human_play.wall_timeline import validate, move_instants
from backend.gamer_ranked.prng import Xoshiro128StarStar


def values(code):
    return [2 ** int(c, 16) if c != '0' else 0 for c in code]


class AssistanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'CLOUD_AUTH_DB': self.temp.name + '/auth.db',
            'HUMAN_PLAY_DB': self.temp.name + '/human.db', 'ASSISTANCE_EVIDENCE_DB': self.temp.name + '/evidence.db'})
        self.env.start()
        init_db()
        self.stamp = round(time.time() * 1000)

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_reader_mask_example_and_idempotence(self):
        key = lookup_key(int('10010332ba91fec3', 16), 'free10_512')
        self.assertEqual(key, '10010332fff1fff3')
        self.assertEqual(lookup_key(int(key, 16), 'free10_512'), key)
        self.assertEqual(play_lookup_key(values('10010332ba91fec3'), '4x4', 'free10_512'), key)

    def test_variant_walls_and_large_tiles(self):
        from Config import pattern_catalog
        for pattern in ('2x4', '3x3', '3x4', '3x4free9', '3x3free8'):
            variant = descriptor(pattern + '_128')[4]
            count = engine.VARIANTS[variant][0] * engine.VARIANTS[variant][1]
            board = [2] * count
            board[-1] = 512
            key = play_lookup_key(board, variant, pattern + '_128')
            walls = f'{int(pattern_catalog[pattern]["seed_boards"][0]):016x}'
            self.assertEqual([i for i, c in enumerate(key) if c == 'f'], [i for i, c in enumerate(walls) if c == 'f'])
            self.assertEqual(key.count('e'), 1)
        self.assertIsNone(play_lookup_key([2] * 16, '4x4', '3x3_128'))

    def test_scope_dedupe_and_deterministic_ties(self):
        args = (1, 'query-id-1234567890', 'setboard', int('10010332ba91fec3', 16), 'free10_512', self.stamp)
        for _ in range(2):
            evidence.record_table(*args, {'right': .9, 'up': .9, 'left': .9})
        for source in ('prefetch', 'practice-jump', 'auto'):
            evidence.record_table(1, source, source, 1, 'free10_512', self.stamp, {'left': 1})
        evidence.record_table(1, 'missing', 'palette', 1, 'free10_512', self.stamp, {'left': None})
        rows = evidence.candidates(1, self.stamp - 10, self.stamp + 10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['direction'], 'left')
        self.assertEqual(rows[0]['board_key'], '10010332fff1fff3')
        self.assertEqual(evidence.candidates(2, 0, self.stamp + 100), [])

    def test_upload_authentication_and_origin(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.assistance_routes import router
        app = FastAPI()
        app.include_router(router)
        payload = {'event_id': 'ai-api-test-123456', 'board_codes': [0] * 16,
                   'queried_at_ms': self.stamp, 'direction': 'left'}
        with TestClient(app) as client:
            self.assertEqual(client.post('/api/assistance/ai-setboard', json=payload).status_code, 401)
            with patch('backend.assistance_routes.require_identity', return_value={'id': 1}):
                self.assertEqual(client.post('/api/assistance/ai-setboard', json=payload,
                    headers={'Origin': 'https://example.invalid'}).status_code, 403)
                self.assertEqual(client.post('/api/assistance/ai-setboard', json=payload).status_code, 204)
                self.assertEqual(client.post('/api/assistance/ai-setboard', json={**payload, 'direction': 'bad'}).status_code, 422)

    def test_review_boundary_and_clock_reversal(self):
        board = values('10010332ba91fec3')
        evidence.record_table(1, 'boundary-query-1234', 'palette', int('10010332fff1fff3', 16),
                              'free10_512', self.stamp, {'up': .8})
        for high, relation in ((self.stamp - 1000, 'boundary'), (self.stamp - 3000, None), (self.stamp - 20000, None)):
            timeline = {'version': 1, 'started_at_ms': self.stamp - 10000, 'anchors': [[1, high]]}
            matcher = Matcher({'id': 'r', 'user_id': 1, 'variant': '4x4'}, engine.EVENT.pack(0, 0), timeline)
            matcher.observe({'seq': 0, 'board': board})
            matcher.observe({'seq': 1, 'board': board})
            self.assertEqual([hit['time_relation'] for hit in matcher.hits], [relation] if relation else [])

    def test_masked_match_absolute_interval_and_boundary(self):
        board = values('10010332ba91fec3')
        evidence.record_table(1, 'query-in-interval', 'setboard', int('10010332fff1fff3', 16),
                              'free10_512', self.stamp, {'up': .8})
        timeline = {'version': 1, 'started_at_ms': self.stamp - 10000,
                    'anchors': [[1, self.stamp + 1000]], 'truncated_at_seq': None}
        matcher = Matcher({'id': 'r', 'user_id': 1, 'variant': '4x4'}, engine.EVENT.pack(0, 0), timeline)
        matcher.observe({'seq': 0, 'board': board})
        matcher.observe({'seq': 1, 'board': board})
        self.assertEqual(len(matcher.hits), 1)
        self.assertEqual(matcher.hits[0]['time_relation'], 'during')
        self.assertEqual(matcher.hits[0]['play_board'], board)
        legacy = Matcher({'id': 'r', 'user_id': 1, 'variant': '4x4'}, engine.EVENT.pack(0, 0), None)
        legacy.observe({'seq': 0, 'board': board})
        legacy.observe({'seq': 1, 'board': board})
        self.assertEqual(legacy.hits, [])

    def test_ai_preserves_high_tiles_and_exact_matching(self):
        codes = [0] * 14 + [15, 16]
        evidence.record_ai(1, {'event_id': 'ai-test-123456789', 'board_codes': codes,
                              'queried_at_ms': self.stamp, 'direction': 'left'})
        timeline = {'version': 1, 'started_at_ms': self.stamp - 3000, 'anchors': [[1, self.stamp + 3000]]}
        def match(board):
            matcher = Matcher({'id': 'r', 'user_id': 1, 'variant': '4x4'}, engine.EVENT.pack(3, 0), timeline)
            matcher.observe({'seq': 0, 'board': board})
            matcher.observe({'seq': 1, 'board': board})
            return matcher.hits
        self.assertEqual(len(match([0] * 14 + [32768, 65536])), 1)
        self.assertEqual(match([0] * 14 + [32768, 32768]), [])

    def test_timeline_resume_overflow_and_old_prefix(self):
        timeline = validate({'version': 1, 'anchors': [[2, self.stamp], [4, self.stamp - 500]],
                             'started_at_ms': None}, 4)
        self.assertEqual(move_instants(timeline, [0, 100, 900000, 0]),
                         [None, self.stamp, self.stamp + 900000, self.stamp - 500])
        run = {'id': 'test', 'variant': '4x4', 'seed': '00000001000000020000000300000004',
               'reason': 'abandoned', 'created': time.time(), 'wall_timeline': timeline}
        raw = b''.join(engine.EVENT.pack(0, delta) for delta in [0, 100, 900000, 0])
        for version in (1, 2):
            header, events = engine.parse_replay(engine.replay_bytes(run, raw, version))
            self.assertEqual(header['wall_timeline'], timeline)
            self.assertEqual(events, raw)

    def test_timeline_does_not_mask_reentry_rollback(self):
        run = service.create(1, 'b' * 32, '4x4', 'request-rollback-1234', 'writer-1234567890')
        state = engine.initial(run['run_id'], '4x4', run['seed'])
        state['seq'] = 1
        timeline = {'version': 1, 'anchors': [[1, self.stamp]], 'started_at_ms': self.stamp - 1000}
        with database() as db:
            db.execute('UPDATE human_runs SET state=?,wall_timeline=? WHERE id=?',
                       (json.dumps(state), json.dumps(timeline), run['run_id']))
        with self.assertRaisesRegex(service.RunError, 'rollback_detected'):
            service.submit(1, 'b' * 32, run['run_id'], action='reentry',
                writer='writer-1234567890', epoch=1, start=1, prefix_hash=state['hash'],
                local_seq=0, data=b'', wall_timeline=json.dumps({**timeline, 'anchors': []}))

    def test_seal_preserves_archive_and_creates_one_review(self):
        run = service.create(1, 'b' * 32, '4x4', 'request-1234567890', 'writer-1234567890')
        state = engine.initial(run['run_id'], '4x4', run['seed'])
        direction = next(d for d in range(4) if engine.move(state['board'], 4, 4, d)[0] != state['board'])
        moved, _ = engine.move(state['board'], 4, 4, direction)
        rng = Xoshiro128StarStar(state['rng'])
        index, tile = engine.spawn(moved, rng)
        raw = engine.EVENT.pack(direction | index << 2 | (64 if tile == 4 else 0), 0)
        evidence.record_ai(1, {'event_id': 'seal-ai-1234567890',
            'board_codes': [tile.bit_length() - 1 if tile else 0 for tile in state['board']],
            'queried_at_ms': self.stamp, 'direction': 'left'})
        timeline = {'version': 1, 'anchors': [[1, self.stamp + 1000]], 'started_at_ms': self.stamp - 1000}
        kwargs = dict(action='seal', writer='writer-1234567890', epoch=1, start=0,
            prefix_hash=state['hash'], local_seq=1, data=raw, reason='abandoned', wall_timeline=json.dumps(timeline))
        service.submit(1, 'b' * 32, run['run_id'], **kwargs)
        service.submit(1, 'b' * 32, run['run_id'], **kwargs)
        with database() as db:
            saved = dict(db.execute('SELECT * FROM human_runs WHERE id=?', (run['run_id'],)).fetchone())
            reviews = db.execute('SELECT * FROM human_assistance_reviews').fetchall()
        self.assertEqual(saved['status'], 'sealed')
        self.assertEqual(saved['eligibility'], 'eligible')
        self.assertEqual(len(reviews), 1)
        header, events = engine.parse_replay(gzip.decompress(saved['archive']))
        self.assertEqual(events, raw)
        self.assertEqual(header['wall_timeline']['anchors'], timeline['anchors'])
        from backend.human_play.approval_transactions import list_transactions
        self.assertEqual(list_transactions(kind='assistance')['transactions'][0]['actions'], ['confirm', 'dismiss'])
        decide(reviews[0]['id'], 2, False, 'test review')
        self.assertEqual(list_transactions(kind='assistance')['transactions'][0]['raw_status'], 'dismissed')
