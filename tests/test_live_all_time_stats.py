import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.gamer_ranked.rules import legal_moves
from backend.live.protocol import LiveRun
from backend.live.statistics import VERSION, StageCounter, TARGETS, replay_stages
from backend.live.store import LiveStore


class StageCounterTests(unittest.TestCase):
    def merge(self, counter, target, held=()):
        counter.observe([*held, target//2, target//2, 0], [*held, target, 0, 0])

    def test_three_successes_then_death(self):
        c = StageCounter()
        for target in TARGETS[:3]:
            self.merge(c, target)
        self.assertEqual(c.result(True), (3, 1))
        self.assertEqual(c.result(False), (3, 0))

    def test_65k_phase_death_is_not_failure_and_next_cycle_restarts(self):
        c = StageCounter()
        for target in TARGETS:
            self.merge(c, target)
        self.assertEqual(c.result(True), (5, 0))
        self.merge(c, 65536)
        self.assertEqual(c.result(True), (5, 1))
        self.merge(c, 32768, held=(65536,))
        self.assertEqual(c.result(True), (6, 1))

    def test_early_large_merge_cancels_old_stage_and_larger_cycles_work(self):
        c = StageCounter()
        self.merge(c, 32768)
        self.merge(c, 65536)
        self.assertEqual(c.result(False), (1, 0))
        self.assertEqual(c.stage, 0)
        self.merge(c, 131072)
        self.merge(c, 32768)
        self.assertEqual(c.result(True), (2, 1))

    def test_simultaneous_merges_and_unchanged_boards_do_not_double_count(self):
        c = StageCounter()
        c.observe([16384,16384,8192,8192], [32768,16384,0,0])
        self.assertEqual(c.result(False), (2,0))
        c.observe([32768,16384,0,0], [0,32768,16384,0])
        self.assertEqual(c.result(False), (2,0))
        # One 32k is consumed while another is created in the same move.
        c.observe([32768,32768,16384,16384], [65536,32768,0,0])
        self.assertEqual(c.result(False), (3,0))


class AllTimeStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = LiveStore(Path(self.temp.name)/'live.db')

    def tearDown(self):
        self.temp.cleanup()

    def dead_run(self):
        run = LiveRun(seed='00000001000000020000000300000004')
        rng = random.Random(123)
        for _ in range(10000):
            choices = legal_moves(run.board)
            if not choices:
                run.end()
                return run
            run.apply(run.make_step(rng.choice(choices), 15))
        self.fail('Fixture did not die')

    def test_real_replay_and_idempotent_finish_backfill(self):
        run = self.dead_run()
        self.assertEqual(replay_stages(run.replay()), (0,1))
        self.store.finish(run)
        self.store.finish(run)
        self.assertEqual(self.store.summary()['all_time']['stage32'], dict(passed=0,failed=1,analyzed_runs=1))
        with self.store.connect() as db:
            db.execute('DELETE FROM live_run_stages')
        restored = LiveStore(self.store.path)
        restored.backfill_stats()
        with patch('backend.live.store.replay_stages', side_effect=AssertionError('Replayed twice')):
            restored.backfill_stats()
            self.assertEqual(restored.summary()['all_time']['stage32_rate'],0)

    def test_existing_runs_only_and_weighted_rate(self):
        with self.store.connect() as db:
            for key,score,maximum,passed,failed in [('a',100,32768,3,1),('b',300,65536,6,0)]:
                db.execute('INSERT INTO live_runs VALUES (?,?,?,?,?,?,?)', (key,score,maximum,0,1,'2000-01-01','invalid'))
                db.execute('INSERT INTO live_run_stages VALUES (?,?,?,?)', (key,VERSION,passed,failed))
            db.execute("INSERT INTO live_days VALUES ('1999-01-01',99,999999,99,99)")
            db.execute("INSERT INTO live_scores VALUES ('deleted','1999-01-01',999999)")
        stats = self.store.summary()['all_time']
        self.assertEqual((stats['games'],stats['score_sum'],stats['median_score']), (2,400,200))
        self.assertEqual((stats['tile32'],stats['tile64']), (2,1))
        self.assertEqual(stats['stage32_rate'], .9)
        with self.store.connect() as db:
            db.execute("DELETE FROM live_runs WHERE id='a'")
        self.store._invalidate_summary_cache()
        self.assertEqual(self.store.summary()['all_time']['stage32_rate'],1)

    def test_summary_cache_reuses_unchanged_range(self):
        with patch.object(self.store, '_summary_uncached', wraps=self.store._summary_uncached) as uncached:
            first = self.store.summary('all')
            second = self.store.summary('all')
            self.assertIs(first, second)
            self.assertEqual(uncached.call_count, 1)
            self.store._invalidate_summary_cache()
            self.store.summary('all')
            self.assertEqual(uncached.call_count, 2)

    def test_invalid_replay_is_unknown_and_non_dead_end_has_no_failure(self):
        self.assertEqual(replay_stages(LiveRun().replay()), (0,0))
        with self.store.connect() as db:
            db.execute("INSERT INTO live_runs VALUES ('bad',1,2,0,1,'2000-01-01','broken')")
        self.store.backfill_stats()
        stats=self.store.summary()['all_time']
        self.assertEqual(stats['games'],1)
        self.assertEqual(stats['stage32']['analyzed_runs'],0)
        self.assertIsNone(stats['stage32_rate'])
