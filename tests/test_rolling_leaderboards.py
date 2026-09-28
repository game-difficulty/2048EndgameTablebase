import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from backend import rolling_leaderboards as rolling
from backend.auth.db import auth_db, init_auth_db
from backend.human_play.store import init_db, database
from backend.leaderboards.rolling_gamer import payload as gamer_rolling_payload
from backend.leaderboards.service import _period_for
from backend.token_rewards import HUMAN_WEEKLY_REWARDS, settle_human_rolling_weeks, settle_rolling_weeks


class RollingCacheTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.row_factory = sqlite3.Row
        rolling.init_schema(self.db)
        self.now = 1_000_000.0

    def tearDown(self):
        self.db.close()

    def add(self, user, score, age, run=None):
        rolling.add(self.db, board_key='4x4', run_id=run or f'{user}:{score}',
                    user_id=user, score=score, achieved_at=self.now-age,
                    eligible_at=self.now-age, now=self.now)

    def test_expiry_promotes_same_players_next_run_and_eleventh_player(self):
        self.add(1, 1000, rolling.WINDOW_SECONDS-1, 'old-best')
        self.add(1, 100, 10, 'newer-second')
        for user in range(2, 102):
            self.add(user, 1000-user, 20)
        before = rolling.entries(self.db, '4x4', 100, self.now)
        self.assertEqual(before[0]['run_id'], 'old-best')
        self.assertEqual(len(before), 100)
        after = rolling.entries(self.db, '4x4', 100, self.now+2)
        self.assertEqual(after[0]['user_id'], 2)
        self.assertEqual(len(after), 100)
        self.assertEqual(after[-1]['user_id'], 101)
        self.assertEqual(self.db.execute("SELECT run_id FROM rolling_player_best WHERE user_id=1").fetchone()[0],
                         'newer-second')

    def test_eligibility_and_half_open_settlement_window(self):
        self.add(1, 10, rolling.WINDOW_SECONDS)
        self.add(2, 20, 1)
        rolling.add(self.db, board_key='4x4', run_id='late-approval', user_id=3,
                    score=30, achieved_at=self.now-5, eligible_at=self.now+1,
                    now=self.now+2)
        self.assertEqual([r['user_id'] for r in rolling.as_of(self.db, '4x4', self.now, 10)],
                         [2, 1])
        self.assertEqual([r['user_id'] for r in rolling.as_of(self.db, '4x4', self.now+2, 10)],
                         [3, 2])


class RollingRewardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            'CLOUD_AUTH_DB': str(Path(self.temp.name) / 'auth.db'),
            'HUMAN_PLAY_DB': str(Path(self.temp.name) / 'human.db')})
        self.env.start()
        init_auth_db()
        init_db()
        self.boundary = datetime(2026, 10, 5, tzinfo=timezone.utc)
        with auth_db() as db:
            db.execute("UPDATE token_reward_state SET value=? WHERE key='rolling_first_boundary'",
                       (self.boundary.isoformat(),))
            for user in range(1, 13):
                db.execute("""INSERT INTO users(id,email,email_identity,password_hash,
                    display_name,created_at,updated_at) VALUES(?,?,?,'hash',?,'now','now')""",
                    (user, f'{user}@example.test', f'{user}@example.test', str(user)))

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_gamer_seven_day_boards_cross_calendar_week(self):
        now = self.boundary.timestamp()
        with auth_db() as db:
            for board in ('gamer_high_score_weekly', 'gamer_adversarial_weekly'):
                rolling.add(db, board_key=board, run_id=board + ':recent',
                            user_id=1, score=1000, achieved_at=now-6*86400,
                            eligible_at=now-6*86400, now=now-1)
                rolling.add(db, board_key=board, run_id=board + ':expired',
                            user_id=2, score=2000, achieved_at=now-8*86400,
                            eligible_at=now-8*86400, now=now-1)
        for board in ('gamer_high_score_weekly', 'gamer_adversarial_weekly'):
            result = gamer_rolling_payload(board, now=now)
            self.assertEqual(result['period']['start'],
                             (self.boundary-timedelta(days=7)).isoformat())
            self.assertEqual(result['period']['end'], self.boundary.isoformat())
            self.assertEqual([entry['score'] for entry in result['entries']], [1000])
            with self.assertRaises(ValueError):
                _period_for(board, self.boundary)

    def test_play_only_settlement_survives_later_unified_settlement(self):
        stamp = self.boundary.timestamp() - 10
        with database() as db:
            rolling.add(db, board_key='4x4', run_id='human-only', user_id=1,
                        score=1234, achieved_at=stamp, eligible_at=stamp,
                        now=self.boundary.timestamp() - 1)
        settle_human_rolling_weeks(self.boundary - timedelta(seconds=1))
        settle_human_rolling_weeks(self.boundary)
        settle_human_rolling_weeks(self.boundary)
        settle_rolling_weeks(self.boundary)
        with auth_db() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM token_human_rolling_settlements").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM token_rolling_settlements").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM token_rolling_snapshot WHERE board_key='human:4x4'").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM token_reward_receipts WHERE reward_key LIKE 'rolling:%:human:4x4:%'").fetchone()[0], 1)

    def test_main_only_settlement_does_not_read_play_scores(self):
        stamp = self.boundary.timestamp() - 10
        with database() as db:
            rolling.add(db, board_key='4x4', run_id='play-run', user_id=1,
                        score=2000, achieved_at=stamp, eligible_at=stamp,
                        now=self.boundary.timestamp() - 1)
        with auth_db() as db:
            rolling.add(db, board_key='gamer_high_score_weekly', run_id='main-run',
                        user_id=2, score=1000, achieved_at=stamp, eligible_at=stamp,
                        now=self.boundary.timestamp() - 1)
        settle_rolling_weeks(self.boundary, include_human=False)
        with auth_db() as db:
            boards = [row[0] for row in db.execute(
                'SELECT board_key FROM token_rolling_snapshot ORDER BY board_key')]
            self.assertEqual(boards, ['gamer_high_score_weekly'])

    def test_human_top_ten_and_main_top_five_are_frozen_and_idempotent(self):
        t = self.boundary.timestamp()
        with database() as db:
            for user in range(1, 12):
                rolling.add(db, board_key='4x4', run_id=f'human-{user}',
                            user_id=user, score=2000-user, achieved_at=t-10,
                            eligible_at=t-10, now=t-1)
                rolling.add(db, board_key='3x4', run_id=f'human-3x4-{user}',
                            user_id=user, score=1000-user, achieved_at=t-10,
                            eligible_at=t-10, now=t-1)
        with auth_db() as db:
            for user in range(1, 7):
                rolling.add(db, board_key='gamer_high_score_weekly',
                            run_id=f'gamer-{user}', user_id=user,
                            score=1000-user, achieved_at=t-10,
                            eligible_at=t-10, now=t-1)
        settle_rolling_weeks(self.boundary-timedelta(seconds=1))
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM token_rolling_settlements').fetchone()[0], 0)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(settle_rolling_weeks, [self.boundary]*3))
        settle_rolling_weeks(self.boundary+timedelta(seconds=1))
        with auth_db() as db:
            rows = db.execute("""SELECT user_id,tokens FROM token_rolling_snapshot
                WHERE board_key='human:4x4' ORDER BY position""").fetchall()
            self.assertEqual([row['tokens'] for row in rows], list(HUMAN_WEEKLY_REWARDS['4x4']))
            self.assertEqual([row['user_id'] for row in rows], list(range(1, 11)))
            other = [row[0] for row in db.execute("""SELECT tokens FROM token_rolling_snapshot
                WHERE board_key='human:3x4' ORDER BY position""")]
            self.assertEqual(other, list(HUMAN_WEEKLY_REWARDS['3x4']))
            self.assertEqual(HUMAN_WEEKLY_REWARDS['3x3'], HUMAN_WEEKLY_REWARDS['3x4'])
            self.assertEqual(HUMAN_WEEKLY_REWARDS['2x4'], HUMAN_WEEKLY_REWARDS['3x4'])
            self.assertEqual(db.execute("SELECT COUNT(*) FROM token_rolling_snapshot").fetchone()[0], 25)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM token_reward_receipts").fetchone()[0], 25)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM token_rolling_settlements').fetchone()[0], 1)

    def test_pending_main_verification_delays_frozen_snapshot(self):
        t = self.boundary.timestamp()
        with auth_db() as db:
            db.execute("""INSERT INTO gamer_ranked_runs
                (run_id,user_id,request_id,seed_hex,rules_version,status,
                 started_at,expires_at,submitted_at)
                VALUES('pending-1',1,'pending-1','seed',1,'pending',?,?,?)""",
                (self.boundary.isoformat(), (self.boundary+timedelta(days=1)).isoformat(),
                 (self.boundary-timedelta(seconds=1)).isoformat()))
        settle_rolling_weeks(self.boundary)
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM token_rolling_settlements').fetchone()[0], 0)
            db.execute("UPDATE gamer_ranked_runs SET status='verified' WHERE run_id='pending-1'")
            rolling.add(db, board_key='gamer_high_score_weekly', run_id='pending-1',
                        user_id=1, score=100, achieved_at=t-1, eligible_at=t-1,
                        now=t+1)
        settle_rolling_weeks(self.boundary+timedelta(seconds=2))
        with auth_db() as db:
            row = db.execute("""SELECT run_id,tokens FROM token_rolling_snapshot
                WHERE board_key='gamer_high_score_weekly'""").fetchone()
            self.assertEqual(tuple(row), ('pending-1', 10_000))


if __name__ == '__main__':
    unittest.main()
