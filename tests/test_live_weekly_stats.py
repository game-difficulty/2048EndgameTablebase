import asyncio
from datetime import datetime
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, AsyncMock

from backend.live.store import LiveStore, week_bounds
from backend.live.routes import LiveHub


class WeeklyStatsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = LiveStore(Path(self.directory.name) / 'live.db')
        self.week = patch('backend.live.store.week_bounds', return_value=('2026-09-14','2026-09-21'))
        self.week.start()

    def tearDown(self):
        self.week.stop()
        self.directory.cleanup()

    def run_result(self, key, score, day='2026-09-16', maximum=2048):
        return SimpleNamespace(id=str(key), score=score, elapsed=5000, board=[maximum],
            ended=datetime.fromisoformat(day+'T12:00:00+08:00').timestamp(), replay=lambda:'small-replay')

    def test_beijing_monday_boundary_not_utc_or_rolling_seven_days(self):
        self.assertEqual(week_bounds(datetime.fromisoformat('2026-09-13T15:59:59+00:00')), ('2026-09-07','2026-09-14'))
        self.assertEqual(week_bounds(datetime.fromisoformat('2026-09-13T16:00:00+00:00')), ('2026-09-14','2026-09-21'))

    def test_empty_week_and_odd_even_medians(self):
        self.assertIsNone(self.store.summary()['week']['median_score'])
        for key,score in enumerate([100,800,200]):
            self.store.finish(self.run_result(key,score))
        self.assertEqual(self.store.summary()['week']['median_score'],200)
        self.store.finish(self.run_result('fourth',400))
        weekly=self.store.summary()['week']
        self.assertEqual(weekly['median_score'],300)
        self.assertEqual((weekly['games'],weekly['score_sum']),(4,1500))

    def test_week_totals_exclude_previous_and_next_week(self):
        for key,day,maximum in [('a','2026-09-13',65536),('b','2026-09-14',32768),
                                ('c','2026-09-20',65536),('d','2026-09-21',65536)]:
            self.store.finish(self.run_result(key,100,day,maximum))
        weekly=self.store.summary()['week']
        self.assertEqual((weekly['games'],weekly['tile32'],weekly['tile64']),(2,2,1))
        self.assertEqual(weekly['median_score'],100)

    def test_replay_retention_does_not_drop_scores_or_double_count_retries(self):
        first=self.run_result(0,0)
        for key in range(205):
            run=self.run_result(key,key*4)
            run.ended += key
            self.store.finish(run)
        self.assertIsNone(self.store.replay('0'))
        weekly=self.store.summary()['week']
        self.assertEqual(weekly['median_score'],408)
        self.assertEqual(weekly['games'],205)
        self.store.finish(first)
        self.assertEqual(self.store.summary()['week'],weekly)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM live_runs').fetchone()[0],200)
            self.assertEqual(db.execute('SELECT count(*) FROM live_scores').fetchone()[0],205)

    def test_legacy_backfill_and_missing_scores_are_not_an_invented_median(self):
        self.store.finish(self.run_result('legacy',120))
        with self.store.connect() as db:
            db.execute('DELETE FROM live_scores')
        restored=LiveStore(self.store.path)
        self.assertEqual(restored.summary()['week']['median_score'],120)
        with restored.connect() as db:
            db.execute('UPDATE live_days SET games=2,score_sum=360')
        weekly=restored.summary()['week']
        self.assertEqual(weekly['games'],2)
        self.assertIsNone(weekly['median_score'])
        self.assertIsNone(LiveStore(self.store.path).summary()['week']['median_score'])

    def test_idle_stream_broadcasts_week_rollover_without_a_finished_game(self):
        hub=LiveHub()
        hub.store=self.store
        hub.summary_week='2026-09-07'
        messages=[]
        with patch('backend.live.routes.week_bounds',return_value=('2026-09-14','2026-09-21')), \
                patch('backend.live.routes.asyncio.sleep',new=AsyncMock(side_effect=[None,asyncio.CancelledError()])), \
                patch.object(hub,'broadcast',side_effect=messages.append):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(hub.maintenance())
        summaries=[message for message in messages if message.get('type')=='summary']
        self.assertEqual(len(summaries),1)
        self.assertEqual(summaries[0]['week']['start'],'2026-09-14')
        self.assertEqual(summaries[0]['week']['games'],0)
