import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from unittest.mock import patch

from backend.auth.db import auth_db, init_auth_db
from backend import token_rewards as rewards
from backend.gamer_ranked.service import prune_ranked_replays
from backend.minigame_rankings.service import _apply_verified_result


class TokenRewardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'CLOUD_AUTH_DB': str(Path(self.temp.name) / 'auth.db')})
        self.env.start()
        init_auth_db()
        self.week = datetime.fromisoformat('2026-09-07T00:00:00+08:00')
        with auth_db() as db:
            db.execute("UPDATE token_reward_state SET value=? WHERE key='weekly_start'", (self.week.isoformat(),))
            for uid in range(1, 8):
                db.execute("""INSERT INTO users(id,email,email_identity,password_hash,display_name,created_at,updated_at)
                    VALUES(?,?,?,'hash',?,'now','now')""", (uid, f'{uid}@test.invalid', f'{uid}@test.invalid', str(uid)))

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def balance(self, uid=1):
        with auth_db() as db:
            row = db.execute('SELECT paid_balance_units FROM token_accounts WHERE user_id=?', (uid,)).fetchone()
            return row[0] // 1000 if row else 0

    def trophy(self, tier, difficulty=0):
        with auth_db() as db:
            db.execute('BEGIN IMMEDIATE')
            _apply_verified_result(db, run={'user_id': 1, 'game_id': 'test', 'difficulty': difficulty, 'run_id': 'run'},
                verified={'final_board': [0]*16, 'score': 100, 'trophy_tier': tier,
                          'highest_tile_exp': 10, 'board_rows': 4, 'board_cols': 4},
                record_hash='hash', record_blob='record', verified_at=self.week.isoformat())

    def test_verified_upgrade_retries_skipped_tiers_and_difficulty(self):
        self.trophy(2)
        self.assertEqual(self.balance(), 5000)
        self.trophy(2)
        self.trophy(1)
        self.assertEqual(self.balance(), 5000)
        self.trophy(4)
        self.assertEqual(self.balance(), 20000)
        self.trophy(4, difficulty=1)
        self.assertEqual(self.balance(), 40000)
        rewards.backfill_trophies()
        self.assertEqual(self.balance(), 40000)

    def test_backfill_verified_only_once_and_future_upgrade(self):
        with auth_db() as db:
            for uid, level in [(1, 'verified'), (2, 'legacy')]:
                db.execute("""INSERT INTO minigame_high_scores
                    (user_id,game_id,difficulty,trophy_tier,final_board_json,score_achieved_at,updated_at,verification_level)
                    VALUES(?,'test',0,2,'[]','now','now',?)""", (uid, level))
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: rewards.backfill_trophies(), range(3)))
        self.assertEqual(self.balance(), 5000)
        self.assertEqual(self.balance(2), 0)
        self.trophy(4)
        self.assertEqual(self.balance(), 20000)
        init_auth_db()
        rewards.backfill_trophies()
        self.assertEqual(self.balance(), 20000)
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT tier FROM user_entitlements WHERE user_id=1').fetchone()[0], 'free')

    def add_weekly(self, uid, board='gamer_high_score', score=None, week=None):
        week = week or self.week
        run_id = f'{uid}:{board}:{week.isoformat()}'
        with auth_db() as db:
            db.execute("""INSERT INTO gamer_ranked_runs
                (run_id,user_id,request_id,seed_hex,rules_version,status,started_at,expires_at,submitted_at)
                VALUES(?,?,?,'seed',1,'verified',?,?,?)""",
                (run_id, uid, run_id, week.isoformat(), (week+timedelta(days=30)).isoformat(), week.isoformat()))
            db.execute("""INSERT INTO gamer_weekly_high_scores
                (user_id,board_key,week_start,score,max_tile,move_count,final_board,record_blob,replay_id,run_id,achieved_at,updated_at)
                VALUES(?,?,?,?,2,1,'[]','record',?,?,?,?)""",
                (uid, board, week.isoformat(), score if score is not None else 1000-uid,
                 run_id, run_id, week.isoformat(), week.isoformat()))
        return run_id

    def test_weekly_both_boards_ties_top_five_and_idempotency(self):
        for uid in range(1, 7):
            for board in rewards.WEEKLY_REWARDS:
                self.add_weekly(uid, board, score=1000)
        rewards.settle_weeks(self.week + timedelta(days=6, hours=23))
        self.assertEqual(self.balance(), 0)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: rewards.settle_weeks(self.week+timedelta(days=7)), range(3)))
        self.assertEqual([self.balance(uid) for uid in range(1, 7)], [15000,12000,8000,5000,3000,0])
        init_auth_db()
        rewards.settle_weeks(self.week + timedelta(days=7))
        self.assertEqual(self.balance(), 15000)
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM token_reward_receipts').fetchone()[0], 10)
            self.assertEqual(db.execute('SELECT tier FROM user_entitlements WHERE user_id=1').fetchone()[0], 'free')

    def test_pending_validation_delays_settlement_and_cleanup(self):
        run = self.add_weekly(1)
        with auth_db() as db:
            db.execute("UPDATE gamer_ranked_runs SET status='validating' WHERE run_id=?", (run,))
        now = self.week + timedelta(days=21)
        rewards.settle_weeks(now)
        self.assertEqual(self.balance(), 0)
        with patch('backend.gamer_ranked.service._utc_now', return_value=now):
            prune_ranked_replays()
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM gamer_weekly_high_scores').fetchone()[0], 1)
            db.execute("UPDATE gamer_ranked_runs SET status='verified' WHERE run_id=?", (run,))
        rewards.settle_weeks(now)
        self.assertEqual(self.balance(), 10000)
        with patch('backend.gamer_ranked.service._utc_now', return_value=now):
            prune_ranked_replays()
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM gamer_weekly_high_scores').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM token_weekly_settlements').fetchone()[0], 3)

    def test_prelaunch_weeks_and_disabled_accounts_do_not_win(self):
        self.add_weekly(1, week=self.week-timedelta(days=7))
        self.add_weekly(2)
        self.add_weekly(3)
        with auth_db() as db:
            db.execute("UPDATE users SET status='disabled' WHERE id=2")
        rewards.settle_weeks(self.week+timedelta(days=7))
        self.assertEqual(self.balance(), 0)
        self.assertEqual(self.balance(2), 0)
        self.assertEqual(self.balance(3), 10000)

    def test_failed_credit_rolls_back_receipt_balance_and_result(self):
        with auth_db() as db:
            db.execute("CREATE TRIGGER fail_credit BEFORE INSERT ON token_ledger BEGIN SELECT RAISE(ABORT,'test'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.trophy(4)
        self.assertEqual(self.balance(), 0)
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM token_reward_receipts').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM minigame_high_scores').fetchone()[0], 0)
            db.execute('DROP TRIGGER fail_credit')
        self.trophy(4)
        self.assertEqual(self.balance(), 20000)

    def test_weekly_credit_failure_is_retryable_and_atomic(self):
        self.add_weekly(1)
        self.add_weekly(2)
        with auth_db() as db:
            db.execute("""CREATE TRIGGER fail_credit BEFORE INSERT ON token_ledger
                WHEN NEW.user_id=2 BEGIN SELECT RAISE(ABORT,'test'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            rewards.settle_weeks(self.week + timedelta(days=7))
        self.assertEqual(self.balance(), 0)
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM token_weekly_settlements').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM token_reward_receipts').fetchone()[0], 0)
            db.execute('DROP TRIGGER fail_credit')
        rewards.settle_weeks(self.week + timedelta(days=7))
        self.assertEqual(self.balance(), 10000)
        self.assertEqual(self.balance(2), 8000)


if __name__ == '__main__':
    unittest.main()
