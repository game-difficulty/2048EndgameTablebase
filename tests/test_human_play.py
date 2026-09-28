import gzip
import json
import os
from pathlib import Path
import struct
import subprocess
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.auth.db import init_auth_db, auth_db
from backend.auth.service import create_session, iso
from backend.gamer_ranked.prng import Xoshiro128StarStar
from backend.human_play import admin, engine, rating, service, statistics
from backend.human_play.store import init_db, database
from backend.human_play.routes import router

SEED = '00000001000000020000000300000004'
BROWSER = '11111111-1111-1111-1111-111111111111'
OTHER_BROWSER = '22222222-2222-2222-2222-222222222222'
WRITER = 'test-writer-0000000000001'


def next_event(state, variant, threshold=None):
    for direction in range(4):
        moved, _ = engine.move(state['board'], *engine.VARIANTS[variant], direction)
        if moved != state['board']:
            rng = Xoshiro128StarStar(list(state['rng']))
            index, value = engine.spawn(moved, rng)
            event = engine.EVENT.pack(direction | (index << 2) | (64 if value == 4 else 0), 10 if state['seq'] else 0)
            return event, engine.advance(state, variant, event, threshold)
    return None, state


class HumanPlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'CLOUD_AUTH_DB': self.temp.name + '/auth.db',
            'HUMAN_PLAY_DB': self.temp.name + '/human.db',
            'HUMAN_PLAY_THRESHOLDS': json.dumps({key: 10000000 for key in engine.VARIANTS})})
        self.env.start(); init_auth_db(); init_db()
        with auth_db() as db:
            for uid in (1, 2):
                db.execute('INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at) VALUES(?,?,?,?,?,?)',
                           (uid, f'{uid}@test.invalid', '!disabled', f'Player {uid}', iso(), iso()))
            self.token = create_session(db, 1)[0]
        app = FastAPI(); app.include_router(router)
        self.client = TestClient(app); self.client.headers['Authorization'] = 'Bearer ' + self.token
        self.number = 0

    def tearDown(self):
        self.client.close(); self.env.stop(); self.temp.cleanup()

    def new(self, variant='4x4', browser=BROWSER, threshold=None):
        self.number += 1
        with patch.dict(os.environ, {'HUMAN_PLAY_THRESHOLDS': json.dumps({variant: threshold if threshold is not None else 10000000})}):
            return service.create(1, browser, variant, f'test-request-{self.number:016d}', WRITER)

    def records(self, run, until=None, count=None):
        state = engine.initial(run['run_id'], run['variant'], run['seed'])
        raw = b''; states = [state]
        while count is None or state['seq'] < count:
            event, next_state = next_event(state, run['variant'], run['threshold'])
            if not event: break
            raw += event; state = next_state; states.append(state)
            if until and until(state): break
        return raw, state, states

    def send(self, run, raw, action='seal', reason='game_over', start=0, states=None, local_seq=None, prefix=None, writer=WRITER):
        initial = engine.initial(run['run_id'], run['variant'], run['seed'])
        return service.submit(1, BROWSER, run['run_id'], action=action, writer=writer, epoch=run.get('epoch',1),
            start=start, prefix_hash=prefix or (states[start]['hash'] if states else initial['hash']),
            local_seq=local_seq if local_seq is not None else start + len(raw)//5, data=raw, reason=reason)

    def test_rectangular_movement_and_single_merge(self):
        board = [2,2,2,2, 0,0,0,0, 4,0,4,0]
        moved, score = engine.move(board, 3, 4, 3)
        self.assertEqual(moved, [4,4,0,0, 0,0,0,0, 8,0,0,0]); self.assertEqual(score,16)
        self.assertEqual(engine.move([32768,32768,0,0,0,0,0,0],2,4,3)[0][0],65536)

    def test_default_high_score_monitoring_thresholds(self):
        self.assertEqual(engine.THRESHOLDS, {
            '4x4': 800000,
            '3x4': 70000,
            '2x4': 5000,
            '3x3': 10000,
        })

    def test_single_game_rating_formulas(self):
        self.assertAlmostEqual(rating.single_rating('4x4', [65536]), 3000)
        self.assertAlmostEqual(rating.single_rating('3x4', [8192]), 3032.5)
        self.assertAlmostEqual(rating.single_rating('3x3', [4096]), 4132)
        self.assertAlmostEqual(rating.single_rating('2x4', [2048, 256, 32, 16, 8, 4]), 5884.03234099934)

    def test_all_variant_top_ratings_apply_formula_after_averaging_board_sums(self):
        board_sums = [1024, 4096]
        mean_sum = sum(board_sums) / len(board_sums)
        for variant in engine.VARIANTS:
            self.assertAlmostEqual(
                rating.top_rating(variant, board_sums),
                rating.rating_from_board_sum(variant, mean_sum),
            )
        self.assertNotAlmostEqual(
            rating.top_rating('4x4', board_sums),
            sum(rating.rating_from_board_sum('4x4', value) for value in board_sums) / 2,
        )

    def test_rating_summary_rates_average_board_sum_and_updates_ranks(self):
        for _ in range(2):
            run = self.new('2x4')
            raw, _, _ = self.records(run)
            self.send(run, raw)
        best = service.best_ten(1, 1, '2x4')
        self.assertEqual(best['rating_games'], 2)
        self.assertAlmostEqual(best['rating'], rating.top_rating(
            '2x4', (sum(row['board']) for row in best['entries'])))
        self.assertAlmostEqual(rating.top_rating('4x4', [65284, 61388, 49128, 49128,
            45040, 44558, 40024, 37016, 36124, 32776]), 2713.5789452247955)
        self.assertEqual((best['pb_rank'], best['ra_rank']), (1, 1))

        other = service.create(2, OTHER_BROWSER, '2x4', 'rating-other-00000001', WRITER)
        raw, _, _ = self.records(other)
        service.submit(2, OTHER_BROWSER, other['run_id'], action='seal', writer=WRITER,
            epoch=other['epoch'], start=0,
            prefix_hash=engine.initial(other['run_id'], '2x4', other['seed'])['hash'],
            local_seq=len(raw) // 5, data=raw, reason='game_over')
        first = service.best_ten(1, 1, '2x4')
        second = service.best_ten(2, 2, '2x4')
        self.assertEqual(first['pb_rank'], 1 + int(second['pb_score'] > first['pb_score']))
        self.assertEqual(first['ra_rank'], 1 + int(second['rating'] > first['rating']))
        self.assertEqual(second['pb_rank'], 1 + int(first['pb_score'] > second['pb_score']))
        self.assertEqual(second['ra_rank'], 1 + int(first['rating'] > second['rating']))

    def test_player_statistics_are_persisted_without_replay_reads(self):
        run = self.new('2x4')
        raw, state, _ = self.records(run)
        self.send(run, raw)
        result = service.player_statistics(1, '2x4')
        summary = result['summaries']['2x4']
        self.assertEqual((summary['game_count'], summary['pb_score']), (1, state['score']))
        self.assertIsNone(summary['b10_score'])
        self.assertEqual(len(result['series']), 1)
        response = self.client.get('/api/human/users/Player%201/statistics?variant=2x4')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['summaries']['2x4']['game_count'], 1)
        self.assertEqual(response.headers['cache-control'], 'private, max-age=30')
        with database() as db:
            db.set_authorizer(lambda op, _arg1, column, *_:
                sqlite3.SQLITE_DENY if op == sqlite3.SQLITE_READ and column == 'archive' else sqlite3.SQLITE_OK)
            statistics.payload(db, 1, '2x4')

    def test_statistic_feature_masks_and_32k_stage_tracker(self):
        self.assertEqual(statistics.feature_mask('3x3', [2,4,8,16,32,64,128,256,512]) & 1, 1)
        self.assertEqual(statistics.feature_mask('2x4', [2,4,8,16,32,64,128,512]) & 4, 4)
        tracker = statistics.Rate32kTracker('4x4')
        tracker.observe({'board':[16384,16384,8192,8192,4096,4096] + [0] * 10})
        tracker.observe({'board':[32768,16384,8192] + [0] * 13})
        self.assertEqual(tracker.result(), (2, 3, 1))

    def test_extended_terminal_achievement_masks_and_order(self):
        self.assertEqual(
            [key for key, _label in statistics.FEATURES['2x4']],
            ['full', 'tile_512', 'second_full', '512_256', 'third_full'],
        )

        second_2x4 = statistics.feature_mask(
            '2x4', [2, 4, 8, 16, 32, 64, 128, 512])
        third_2x4 = statistics.feature_mask(
            '2x4', [2, 4, 8, 16, 32, 64, 256, 512])
        self.assertEqual(second_2x4, (1 << 1) | (1 << 2))
        self.assertEqual(third_2x4, (1 << 1) | (1 << 3) | (1 << 4))

        third_3x3 = statistics.feature_mask(
            '3x3', [2, 4, 8, 16, 32, 64, 128, 512, 1024])
        self.assertEqual(third_3x3, (1 << 1) | (1 << 3) | (1 << 4))

        deep_3x4 = statistics.feature_mask(
            '3x4', [4096, 2048, 1024, 512, 256, 128] + [0] * 6)
        self.assertEqual(deep_3x4, (1 << 6) - 1)

    def test_32k_rate_uses_terminal_achievement_levels(self):
        def totals(level_counts):
            passed = total = 0
            for reached in range(6):
                exact = level_counts[reached] - (level_counts[reached + 1]
                                                  if reached < 5 else 0)
                mask = (1 << 1) | sum(1 << index for index in range(2, 2 + reached))
                row_passed, row_total = statistics.rate32k_fact(mask)
                passed += exact * row_passed
                total += exact * row_total
            return passed, total

        self.assertEqual(totals([49, 22, 11, 7, 3, 0]), (43, 92))
        examples = [
            ([13, 8, 7, 5, 5, 4], .7632),
            ([31, 24, 19, 14, 8, 7], .7500),
            ([53, 36, 26, 21, 16, 10], .7171),
            ([22, 6, 4, 3, 1, 1], .4167),
            ([56, 37, 24, 17, 11, 9], .6759),
        ]
        for counts, expected in examples:
            passed, total = totals(counts)
            self.assertAlmostEqual(passed / total, expected, places=4)

    def test_32k_rate_progress_is_persisted_by_eligible_game_index(self):
        run = self.new('4x4')
        board = [32768, 16384, 8192, 4096, 2048] + [0] * 11
        state = engine.initial(run['run_id'], '4x4', run['seed'])
        state.update({'board': board, 'score': 700000})
        with database() as db:
            db.execute("""UPDATE human_runs SET status='sealed',ended=?,reason='game_over',state=?
                WHERE id=?""", (1234.0, json.dumps(state), run['run_id']))
            statistics.upsert_fact(db, {'id': run['run_id'], 'user_id': 1,
                'variant': '4x4', 'ended': 1234.0}, board, 700000, (4, 5, 1))
            statistics.rebuild_player(db, 1, '4x4')
            result = statistics.payload(db, 1, '4x4')
        self.assertEqual(result['rate_32k_series_total'], 0)
        with database() as db:
            db.execute("""UPDATE human_player_rate32k_series SET game_index=10
                WHERE user_id=1""")
            result = statistics.payload(db, 1, '4x4')
        self.assertEqual(result['rate_32k_series_total'], 1)
        self.assertEqual(result['rate_32k_series'][0]['game_index'], 10)
        self.assertEqual((result['rate_32k_series'][0]['passed'],
                          result['rate_32k_series'][0]['total']), (4, 5))
        self.assertAlmostEqual(result['rate_32k_series'][0]['value'], .8)

    def test_rating_cache_rebuilds_from_sealed_state_without_replay_blob(self):
        run = self.new('2x4')
        raw, _, _ = self.records(run)
        self.send(run, raw)
        expected = service.best_ten(1, 1, '2x4')['rating']
        with database() as db:
            db.execute("UPDATE human_runs SET single_rating=NULL WHERE id=?", (run['run_id'],))
            db.execute('DROP TABLE human_player_ratings')
        init_db()
        restored = service.best_ten(1, 1, '2x4')
        self.assertAlmostEqual(restored['rating'], expected)
        self.assertAlmostEqual(restored['entries'][0]['single_rating'], expected)
        self.assertEqual(restored['ra_rank'], 1)

    def test_four_slots_and_other_browser_are_independent(self):
        ids = {self.new(v)['run_id'] for v in engine.VARIANTS}
        ids.add(self.new('4x4', OTHER_BROWSER)['run_id']); self.assertEqual(len(ids),5)
        with self.assertRaisesRegex(service.RunError,'slot_exists'): self.new('4x4')

    def test_idempotent_creation(self):
        a = service.create(1,BROWSER,'3x4','idempotent-request-0001',WRITER)
        b = service.create(1,BROWSER,'3x4','idempotent-request-0001',WRITER)
        self.assertEqual(a['run_id'],b['run_id']); self.assertEqual(a['seed'],b['seed'])

    def test_cross_browser_access_and_active_replay_forbidden(self):
        run = self.new()
        with self.assertRaisesRegex(service.RunError,'run_not_found'): service.status(1,OTHER_BROWSER,run['run_id'])
        with self.assertRaisesRegex(service.RunError,'replay_not_found'): service.replay(run['run_id'],1)
        self.assertNotIn('seed', service.status(1,BROWSER,run['run_id']))
        self.assertNotIn('board', service.status(1,BROWSER,run['run_id']))

    def test_low_score_not_uploaded_until_final_and_binary_archive(self):
        run = self.new('2x4'); raw, state, _ = self.records(run)
        with database() as db: self.assertEqual(db.execute('SELECT count(*) FROM human_chunks').fetchone()[0],0)
        result = self.send(run,raw); self.assertEqual(result['status'],'sealed')
        archive = service.replay(run['run_id'],2)
        binary = gzip.decompress(archive); self.assertEqual(binary[:4],b'HPR2')
        self.assertEqual(engine.parse_replay(binary)[1],raw)
        self.assertEqual(service.leaderboard('2x4')['entries'][0]['score'],state['score'])
        self.assertEqual(service.leaderboard('2x4', viewer_id=1)['me']['score'], state['score'])
        self.assertIsNone(service.leaderboard('2x4', viewer_id=2)['me'])
        self.assertEqual(service.leaderboard('2x4', 'week')['entries'][0]['score'], state['score'])
        self.assertEqual(service.leaderboard('2x4', 'week', viewer_id=1)['me']['rank'], 1)
        self.assertEqual(self.client.get('/api/human/leaderboards?variant=2x4').json()['me']['user_id'], 1)
        with database() as db:
            candidate = db.execute("""SELECT achieved_at,eligible_at FROM rolling_candidates
                WHERE run_id=?""", (run['run_id'],)).fetchone()
            self.assertGreaterEqual(candidate['eligible_at'], candidate['achieved_at'])

    def test_restarted_history_is_visible_and_retained_at_default_zero_threshold(self):
        run = self.new(); raw,_,_ = self.records(run,count=3)
        self.send(run,raw,reason='restarted')
        self.assertEqual(len(service.history(1,1)['entries']),1)
        self.assertEqual(len(service.history(1,2)['entries']),1)
        self.assertTrue(service.replay(run['run_id'],2))
        self.assertEqual(service.leaderboard('4x4')['entries'],[])

    def test_display_threshold_is_fixed_when_run_starts(self):
        service.save_player_settings(1, {key: 100 for key in engine.VARIANTS})
        run = self.new('2x4')
        service.save_player_settings(1, {key: 0 for key in engine.VARIANTS})
        self.send(run, b'', reason='restarted')
        with database() as db:
            row = db.execute('SELECT display_threshold,visible,has_replay FROM human_runs WHERE id=?',
                             (run['run_id'],)).fetchone()
            self.assertEqual(tuple(row), (100, 0, 1))
        self.assertEqual(service.history(1, 1)['entries'], [])
        self.assertEqual(service.history(1, 2)['entries'], [])
        self.assertEqual(service.best_ten(1, 1, '2x4')['entries'], [])
        self.assertTrue(service.replay(run['run_id'], 1))
        with self.assertRaisesRegex(service.RunError, 'replay_not_found'):
            service.replay(run['run_id'], 2)

    def test_named_profile_filters_and_best_ten_have_replay_flag(self):
        run = self.new('3x3')
        self.send(run, b'', reason='restarted')
        run2 = self.new('2x4')
        raw, state, _ = self.records(run2, count=3)
        self.send(run2, raw, reason='restarted')
        uid = service.player_id_for_name('Player 1')
        self.assertEqual(uid, 1)
        self.assertTrue(service.history(uid, 1)['is_owner'])
        self.assertFalse(service.history(uid, 2)['is_owner'])
        self.assertEqual([row['variant'] for row in service.history(uid, 2, variant='2x4')['entries']], ['2x4'])
        self.assertEqual(service.history(uid, 2, variant='2x4')['entries'][0]['board'], state['board'])
        self.assertEqual([row['variant'] for row in service.history(uid, 2, sort='score_desc')['entries']], ['2x4', '3x3'])
        self.assertEqual(service.history(uid, 2, limit=1)['next_offset'], 1)
        first_page = service.history(uid, 2, limit=1, page=1)
        second_page = service.history(uid, 2, limit=1, page=2)
        self.assertEqual((first_page['total'], first_page['page'], first_page['page_count']), (2, 1, 2))
        self.assertEqual((second_page['page'], second_page['page_count']), (2, 2))
        self.assertNotEqual(first_page['entries'][0]['id'], second_page['entries'][0]['id'])
        self.assertEqual(service.best_ten(uid, 2, '2x4')['entries'], [])  # Unapproved restart is not ranked.
        self.approve(run2, state)
        best = service.best_ten(uid, 2, '2x4')['entries'][0]
        self.assertEqual((best['score'], best['has_replay'], len(best['board'])), (state['score'], 1, 8))
        initial_fours = engine.initial(run2['run_id'], '2x4', run2['seed'])['fourCount']
        expected_fours = initial_fours + sum(bool(raw[i] & 64) for i in range(0, len(raw), 5))
        self.assertAlmostEqual(best['four_spawn_rate'], expected_fours / (2 + state['seq']))
        self.assertAlmostEqual(best['single_rating'], rating.single_rating('2x4', state['board']))
        self.assertEqual(service.best_ten(uid, 2, '2x4')['ra_rank'], 1)

        admin.review(run2['run_id'], approved=False, operator='site-owner', note='Approval revoked')
        self.assertIsNone(service.best_ten(uid, 2, '2x4')['rating'])

    def test_named_profile_prefers_exact_legacy_case_conflict(self):
        with auth_db() as db:
            db.execute("UPDATE users SET display_name='xlb',display_name_key='xlb' WHERE id=1")
            db.execute("UPDATE users SET display_name='XLB',display_name_key='legacy-conflict:2:xlb' WHERE id=2")
            db.execute("""INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at)
                VALUES(3,'3@test.invalid','!disabled','Player 3',?,?)""", (iso(), iso()))
        self.assertEqual(service.player_id_for_name('xlb'), 1)
        self.assertEqual(service.player_id_for_name('XLB'), 2)
        self.assertEqual(service.player_id_for_name('XLB', preferred_user_id=1), 1)
        self.assertEqual(service.player_id_for_name('xlb', preferred_user_id=2), 2)
        self.assertEqual(service.player_id_for_name('Player 3', preferred_user_id=2), 3)
        response = self.client.get('/api/human/users/XLB/history')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['player']['id'], 1)
        self.assertTrue(response.json()['is_owner'])

    def test_owner_soft_delete_removes_game_from_all_player_results_but_keeps_archive(self):
        run = self.new('2x4')
        raw, state, _ = self.records(run, count=3)
        self.send(run, raw, reason='restarted')
        self.approve(run, state)
        self.assertEqual(service.history(1, 1)['entries'][0]['id'], run['run_id'])
        self.assertEqual(service.best_ten(1, 1, '2x4')['entries'][0]['id'], run['run_id'])
        self.assertEqual(service.leaderboard('2x4')['entries'][0]['id'], run['run_id'])

        with self.assertRaisesRegex(service.RunError, 'run_not_found'):
            service.delete_history_run(run['run_id'], 2)
        # Deletion must not load the potentially large replay BLOB.
        from contextlib import contextmanager
        @contextmanager
        def narrow_database():
            with database() as db:
                def authorize(action, table, column, *_):
                    if action == sqlite3.SQLITE_READ and table == 'human_runs' and column == 'archive':
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK
                db.set_authorizer(authorize)
                yield db
        with patch.object(service, 'database', narrow_database):
            response = self.client.delete(f"/api/human/runs/{run['run_id']}/history")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            'run_id': run['run_id'], 'deleted': True, 'already_deleted': False})

        with database() as db:
            saved = db.execute("""SELECT visible,deleted_by_user,deleted_by_user_at,archive
                FROM human_runs WHERE id=?""", (run['run_id'],)).fetchone()
            self.assertEqual((saved['visible'], saved['deleted_by_user']), (0, 1))
            self.assertIsNotNone(saved['deleted_by_user_at'])
            self.assertIsNotNone(saved['archive'])
            self.assertIsNone(db.execute("""SELECT 1 FROM human_player_ratings
                WHERE user_id=1 AND variant='2x4'""").fetchone())
            self.assertIsNone(db.execute("""SELECT 1 FROM human_player_statistics
                WHERE user_id=1 AND variant='2x4'""").fetchone())
        self.assertEqual(service.history(1, 1)['entries'], [])
        self.assertEqual(service.best_ten(1, 1, '2x4')['entries'], [])
        self.assertEqual(service.leaderboard('2x4')['entries'], [])
        self.assertEqual(service.leaderboard('2x4', 'week')['entries'], [])
        self.assertTrue(service.replay(run['run_id'], 1))
        with self.assertRaisesRegex(service.RunError, 'replay_not_found'):
            service.replay(run['run_id'], 2)
        self.assertTrue(service.delete_history_run(run['run_id'], 1)['already_deleted'])

    def test_deleting_pb_promotes_remaining_game_and_repeated_delete_is_safe(self):
        runs = []
        for count in (3, 9):
            run = self.new('2x4')
            raw, state, _ = self.records(run, count=count)
            self.send(run, raw, reason='restarted')
            self.approve(run, state)
            runs.append(run)
        best_id = service.best_ten(1, 1, '2x4')['entries'][0]['id']
        remaining_id = next(run['run_id'] for run in runs if run['run_id'] != best_id)
        response = self.client.delete(f'/api/human/runs/{best_id}/history')
        self.assertEqual(response.status_code, 200)
        repeated = self.client.delete(f'/api/human/runs/{best_id}/history')
        self.assertTrue(repeated.json()['already_deleted'])
        self.assertEqual(service.best_ten(1, 1, '2x4')['entries'][0]['id'], remaining_id)
        self.assertEqual(service.leaderboard('2x4')['entries'][0]['id'], remaining_id)
        self.assertEqual(service.leaderboard('2x4', 'week')['entries'][0]['id'], remaining_id)
        self.assertEqual(service.history(1, 1)['total'], 1)
        with database() as db:
            self.assertEqual(db.execute("SELECT game_count FROM human_player_statistics WHERE user_id=1 AND variant='2x4'").fetchone()[0], 1)
        self.client.headers.pop('Authorization')
        self.assertEqual(self.client.delete(f'/api/human/runs/{remaining_id}/history').status_code, 401)

    def test_old_run_without_spawn_counters_still_seals(self):
        run = self.new('2x4')
        with database() as db:
            row = db.execute('SELECT state FROM human_runs WHERE id=?', (run['run_id'],)).fetchone()
            state = json.loads(row['state'])
            state.pop('fourCount'); state.pop('spawnCount')
            db.execute('UPDATE human_runs SET state=? WHERE id=?', (json.dumps(state), run['run_id']))
        raw, final, _ = self.records(run, count=3)
        self.send(run, raw, reason='restarted')
        self.approve(run, final)
        best = service.best_ten(1, 1, '2x4')['entries'][0]
        self.assertIsNotNone(best['four_spawn_rate'])

    def test_four_spawn_rate_is_derived_from_board_and_score(self):
        self.assertEqual(service.four_spawn_rate([2, 2], 0), 0)
        self.assertEqual(service.four_spawn_rate([4, 2], 0), .5)
        self.assertEqual(service.four_spawn_rate([4], 4), 0)
        self.assertIsNone(service.four_spawn_rate([3, 2], 0))
        self.assertIsNone(service.four_spawn_rate([2, 2], 100))

    def approve(self, run, state):
        return admin.review(run['run_id'], approved=True, operator='site-owner', note='Lost local save; retained prefix reviewed',
                            expected_seq=state['seq'], expected_hash=state['hash'])

    def test_manual_checkpoint_ranking_and_revocation(self):
        run, raw, state, states, _ = self.monitored()
        self.assertFalse(engine.game_over(state['board'], *engine.VARIANTS[run['variant']]))
        self.assertEqual(service.leaderboard('4x4')['entries'], [])
        with database() as db:
            db.execute('UPDATE human_chunks SET received=1 WHERE run_id=?', (run['run_id'],))
        result = self.approve(run, state)
        self.assertEqual((result['status'], result['reason'], result['ended_at']), ('sealed', 'interrupted', 1))
        self.assertEqual(service.leaderboard('4x4')['entries'][0]['score'], state['score'])
        self.assertEqual(service.leaderboard('4x4', 'week')['entries'], [])  # Approval cannot move an old score into this week.
        history = service.history(1, 2)
        self.assertEqual(history['bests']['4x4'], state['score'])
        self.assertEqual(history['stats']['completed'], 0)
        binary = gzip.decompress(service.replay(run['run_id'], 2))
        size = struct.unpack_from('<I', binary, 4)[0]
        self.assertEqual(engine.parse_replay(binary)[1], raw)
        self.assertEqual(json.loads(binary[8:8 + size])['reason'], 'interrupted')
        status = service.status(1, BROWSER, run['run_id'])
        self.assertNotIn('board', status)
        self.assertEqual(status['seq'], state['seq'])
        event, _ = next_event(state, run['variant'], run['threshold'])
        with self.assertRaisesRegex(service.RunError, 'run_sealed'):
            self.send(run, event, action='append', start=state['seq'], states=states)
        revoked = admin.review(run['run_id'], approved=False, operator='site-owner', note='Approval withdrawn')
        self.assertEqual(len(revoked['ranking_reviews']), 2)
        self.assertEqual(service.leaderboard('4x4')['entries'], [])
        self.assertEqual(service.history(1, 2)['bests'], {})
        self.assertTrue(service.replay(run['run_id'], 2))
        self.assertEqual(gzip.decompress(service.replay(run['run_id'], 1)), binary)

    def test_manual_review_requires_unchanged_verified_evidence(self):
        run, _, state, _, _ = self.monitored()
        stale = dict(state, seq=state['seq'] - 1)
        with self.assertRaisesRegex(service.RunError, 'review_progress_changed'):
            self.approve(run, stale)
        with database() as db:
            db.execute("UPDATE human_chunks SET digest='broken' WHERE run_id=?", (run['run_id'],))
        with self.assertRaisesRegex(service.RunError, 'retained_replay_invalid'):
            self.approve(run, state)
        self.assertEqual(service.status(1, BROWSER, run['run_id'])['status'], 'active')
        self.assertFalse(admin.inspect(run['run_id'])['manually_approved'])

    def test_manual_review_of_existing_archive_preserves_bytes(self):
        run = self.new(); raw, state, _ = self.records(run, count=5)
        self.send(run, raw, reason='restarted')
        before = service.replay(run['run_id'], 1)
        self.approve(run, state)
        self.assertEqual(service.replay(run['run_id'], 2), before)
        self.assertEqual(service.history(1, 2)['entries'][0]['reason'], 'restarted')
        with database() as db:
            damaged = bytearray(before)
            damaged[-8] ^= 1  # Corrupt gzip CRC32; no separate SHA-256 is needed.
            db.execute("UPDATE human_runs SET archive=? WHERE id=?", (bytes(damaged), run['run_id']))
        admin.review(run['run_id'], approved=False, operator='site-owner', note='Rechecking')
        with self.assertRaisesRegex(service.RunError, 'retained_replay_invalid'):
            self.approve(run, state)

    def test_legacy_archive_digest_migration_preserves_replay(self):
        run = self.new(); raw, _, _ = self.records(run, count=5)
        self.send(run, raw, reason='restarted')
        before = service.replay(run['run_id'], 1)
        with database() as db:
            db.execute('ALTER TABLE human_runs ADD COLUMN archive_hash TEXT')
            db.execute("UPDATE human_runs SET archive_hash='legacy' WHERE id=?", (run['run_id'],))
        init_db(); init_db()
        with database() as db:
            self.assertNotIn('archive_hash', {r['name'] for r in db.execute('PRAGMA table_info(human_runs)')})
        self.assertEqual(service.replay(run['run_id'], 1), before)
        response = self.client.get(f"/api/human/replays/{run['run_id']}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, gzip.decompress(before))
        self.assertNotIn('X-Replay-SHA256', response.headers)

    def test_manual_review_cannot_bypass_disqualification_or_empty_record(self):
        run = self.new()
        state = engine.initial(run['run_id'], run['variant'], run['seed'])
        with self.assertRaisesRegex(service.RunError, 'retained_replay_invalid'):
            self.approve(run, state)
        with database() as db:
            db.execute("UPDATE human_runs SET eligibility='disqualified' WHERE id=?", (run['run_id'],))
        with self.assertRaisesRegex(service.RunError, 'run_disqualified'):
            self.approve(run, state)
        response = self.client.post(f"/api/human/runs/{run['run_id']}/approve", json={})
        self.assertEqual(response.status_code, 404)

    def monitored(self):
        run = self.new(threshold=8)
        raw,state,states = self.records(run, until=lambda s:s['score'] > 8)
        result = self.send(run,raw,action='monitor')
        return run,raw,state,states,result

    def test_threshold_is_strict_and_first_over_required(self):
        run = self.new(threshold=8)
        raw,state,states = self.records(run, until=lambda s:s['score'] > 8)
        prior = states[-2]
        with self.assertRaisesRegex(service.RunError,'below_threshold'):
            self.send(run,raw[:-5],action='monitor')
        self.assertLessEqual(prior['score'],8)
        self.assertTrue(self.send(run,raw,action='monitor')['monitored'])

    def test_successful_game_over_seal_compacts_periodic_chunks(self):
        run = self.new(threshold=0)
        raw, final, states = self.records(run)
        crossing = final['first_over']
        self.assertIsNotNone(crossing)
        self.send(run, raw[:crossing * engine.EVENT.size], action='monitor')
        with database() as db:
            self.assertEqual(db.execute(
                'SELECT count(*) FROM human_chunks WHERE run_id=?',
                (run['run_id'],)).fetchone()[0], 1)
        sealed = self.send(
            run, raw[crossing * engine.EVENT.size:], action='seal',
            reason='game_over', start=crossing, states=states,
            local_seq=final['seq'])
        self.assertEqual(sealed['status'], 'sealed')
        with database() as db:
            self.assertEqual(db.execute(
                'SELECT count(*) FROM human_chunks WHERE run_id=?',
                (run['run_id'],)).fetchone()[0], 0)
        with self.assertRaisesRegex(service.RunError, 'run_sealed'):
            self.send(run, b'', action='append', start=final['seq'],
                      states=states, local_seq=final['seq'])

    def test_offline_active_run_chunks_survive_age_and_database_restart(self):
        run, _raw, state, _states, _result = self.monitored()
        with database() as db:
            db.execute('UPDATE human_runs SET created=0,permit_until=0 WHERE id=?',
                       (run['run_id'],))
            db.execute('UPDATE human_chunks SET received=0 WHERE run_id=?',
                       (run['run_id'],))
        init_db()
        self.assertEqual(service.status(1, BROWSER, run['run_id'])['status'], 'active')
        with database() as db:
            self.assertEqual(db.execute(
                'SELECT count(*) FROM human_chunks WHERE run_id=?',
                (run['run_id'],)).fetchone()[0], 1)

    def test_cannot_skip_high_score_monitoring_and_submit_later(self):
        run = self.new('2x4',threshold=0); raw,_,_ = self.records(run)
        with self.assertRaisesRegex(service.RunError,'monitoring_required'): self.send(run,raw)

    def test_reentry_local_behind_rejected_without_replacing_server(self):
        run,raw,state,states,result = self.monitored()
        with self.assertRaisesRegex(service.RunError,'rollback_detected'):
            self.send(run,b'',action='reentry',start=state['seq'],states=states,local_seq=state['seq']-1)
        after=service.status(1,BROWSER,run['run_id'])
        self.assertEqual(after['seq'],state['seq']); self.assertEqual(after['eligibility'],'disqualified')

    def test_reentry_same_prefix_and_ahead_both_pass(self):
        run,raw,state,states,_ = self.monitored()
        self.send(run,b'',action='reentry',start=state['seq'],states=states)
        event,next_state=next_event(state,run['variant'],run['threshold'])
        result=self.send(run,event,action='reentry',start=state['seq'],states=states)
        self.assertEqual(result['seq'],next_state['seq']); self.assertNotIn('board',result)

    def test_same_count_wrong_prefix_is_disqualified(self):
        run,_,state,states,_=self.monitored()
        with self.assertRaisesRegex(service.RunError,'prefix_conflict'):
            self.send(run,b'',action='reentry',start=state['seq'],states=states,prefix='0'*64)

    def test_ack_loss_exact_upload_retry_is_idempotent(self):
        run,raw,state,states,_=self.monitored()
        result=self.send(run,raw,action='monitor')
        self.assertEqual(result['seq'],state['seq'])
        with database() as db:self.assertEqual(db.execute('SELECT count(*) FROM human_chunks').fetchone()[0],1)

    def test_expired_permit_blocks_append_until_reentry(self):
        run,_,state,states,_=self.monitored()
        with database() as db:db.execute('UPDATE human_runs SET permit_until=0 WHERE id=?',(run['run_id'],))
        event,_=next_event(state,run['variant'],run['threshold'])
        with self.assertRaisesRegex(service.RunError,'reentry_required'):
            self.send(run,event,action='append',start=state['seq'],states=states)
        self.assertTrue(self.send(run,event,action='reentry',start=state['seq'],states=states)['monitored'])

    def test_old_writer_rejected_after_same_browser_claim(self):
        run=self.new(); claimed=service.claim_low(1,BROWSER,run['run_id'],'new-writer-0000000000001',1)
        self.assertEqual(claimed['epoch'],2)
        with self.assertRaisesRegex(service.RunError,'writer_changed'):
            self.send(run,b'',reason='restarted')

    def test_replacing_one_variant_keeps_other_variant(self):
        a=self.new('4x4'); b=self.new('3x4')
        service.create(1,BROWSER,'4x4','new-replacement-00001',WRITER,a['run_id'])
        self.assertEqual(service.status(1,BROWSER,a['run_id'])['status'],'pending_archive')
        self.assertEqual(service.status(1,BROWSER,b['run_id'])['status'],'active')

    def test_retired_run_can_archive_missing_first_checkpoint_but_cannot_resume(self):
        run = self.new(threshold=8)
        raw, state, states = self.records(run, until=lambda s: s['score'] > 8)
        event, final = next_event(state, run['variant'], run['threshold'])
        service.create(1, BROWSER, '4x4', 'replacement-checkpoint-001', WRITER, run['run_id'])
        result = self.send(run, raw, action='monitor')
        self.assertEqual(result['status'], 'pending_archive')
        with self.assertRaisesRegex(service.RunError, 'run_inactive'):
            self.send(run, event, action='append', start=state['seq'], states=states)
        with self.assertRaisesRegex(service.RunError, 'run_inactive'):
            self.send(run, event, action='reentry', start=state['seq'], states=states)
        sealed = self.send(run, event, reason='restarted', start=state['seq'], states=states)
        self.assertEqual(sealed['status'], 'sealed')
        self.assertEqual(sealed['seq'], final['seq'])

    def test_api_auth_origin_and_binary_limits(self):
        self.assertEqual(self.client.post('/api/human/runs',headers={'Origin':'https://untrusted.invalid'},json={}).status_code,403)
        self.assertEqual(TestClient(self.client.app).get('/api/human/runs/no/status').status_code,401)
        run=self.new()
        response=self.client.post(f"/api/human/runs/{run['run_id']}/seal",content=b'bad')
        self.assertEqual(response.status_code,415)
        self.assertEqual(self.client.get(f"/api/human/runs/{run['run_id']}/status",headers={'X-Human-Browser':OTHER_BROWSER}).status_code,404)

    def test_frontend_backend_determinism_all_variants(self):
        root=Path(__file__).resolve().parents[1]
        script="""import {VARIANTS,initialState,initialHash,nextMove,eventHash} from './src/human/engine.js';
        const results=[]; for(const variant of Object.keys(VARIANTS)) {
          let s={...initialState('fixture',variant,'00000001000000020000000300000004'),variant};
          let hash=await initialHash('fixture',variant,'00000001000000020000000300000004'); const events=[];
          for(let i=0;i<80;i++){let n;for(let d=0;d<4;d++){n=nextMove(s,d,i?10:0);if(n)break;}if(!n)break;
            hash=await eventHash(hash,n.event);events.push(n.event);s=n.state;}
          results.push({variant,state:s,events,hash});}console.log(JSON.stringify(results));"""
        output=subprocess.check_output(['node','--input-type=module','-e',script],cwd=root/'frontend',text=True)
        for fixture in json.loads(output):
            raw=b''.join(engine.EVENT.pack(*event) for event in fixture['events'])
            actual=engine.advance(engine.initial('fixture',fixture['variant'],SEED),fixture['variant'],raw)
            for key in ('board','rng','seq','score','elapsed'):self.assertEqual(actual[key],fixture['state'][key],(fixture['variant'],key))
            self.assertEqual(actual['hash'],fixture['hash'])


if __name__ == '__main__': unittest.main()
