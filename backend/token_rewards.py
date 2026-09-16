"""Verified achievement rewards, credited atomically with durable receipts."""
from datetime import datetime, timedelta, timezone
import json

from backend.auth.db import auth_db

TROPHY_REWARDS = {1: 2_000, 2: 3_000, 3: 5_000, 4: 10_000}
WEEKLY_REWARDS = {
    "gamer_high_score": (10_000, 8_000, 5_000, 3_000, 2_000),
    "gamer_adversarial": (5_000, 4_000, 3_000, 2_000, 1_000),
}


def init_schema(db):
    from backend.gamer_ranked.service import _week_start_iso

    db.execute("""CREATE TABLE IF NOT EXISTS token_reward_receipts (
        reward_key TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        tokens INTEGER NOT NULL, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS token_reward_state (
        key TEXT PRIMARY KEY, value TEXT NOT NULL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS token_weekly_settlements (
        week_start TEXT PRIMARY KEY, settled_at TEXT NOT NULL
    )""")
    db.execute("INSERT OR IGNORE INTO token_reward_state VALUES ('weekly_start', ?)",
               (_week_start_iso(),))


def _credit(db, *, key, user_id, tokens, event, metadata, stamp):
    metadata_json = json.dumps(metadata, separators=(",", ":"))
    inserted = db.execute("""INSERT OR IGNORE INTO token_reward_receipts
        (reward_key,user_id,tokens,metadata_json,created_at) VALUES(?,?,?,?,?)""",
        (key, user_id, tokens, metadata_json, stamp)).rowcount
    if not inserted:
        return 0
    db.execute("""INSERT OR IGNORE INTO token_accounts
        (user_id,created_at,updated_at) VALUES(?,?,?)""", (user_id, stamp, stamp))
    before = db.execute("""SELECT bonus_balance_units+paid_balance_units
        FROM token_accounts WHERE user_id=?""", (user_id,)).fetchone()[0]
    units = tokens * 1000
    db.execute("""UPDATE token_accounts SET paid_balance_units=paid_balance_units+?,
        updated_at=? WHERE user_id=?""", (units, stamp, user_id))
    db.execute("""INSERT INTO token_ledger (user_id,event_type,operation_key,
        paid_delta_units,balance_before_units,balance_after_units,metadata_json,created_at)
        VALUES(?,?,?,?,?,?,?,?)""",
        (user_id, event, key, units, before, before + units, metadata_json, stamp))
    return tokens


def award_trophies(db, *, user_id, game_id, difficulty, tier, stamp):
    """Caller owns the verification transaction; skipped tiers are included."""
    total = 0
    for level in range(1, min(4, int(tier)) + 1):
        total += _credit(
            db, key=f"trophy:{user_id}:{game_id}:{difficulty}:{level}",
            user_id=user_id, tokens=TROPHY_REWARDS[level], event="minigame_trophy_award",
            metadata={"game_id": game_id, "difficulty": difficulty, "tier": level}, stamp=stamp,
        )
    return total


def backfill_trophies():
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM token_reward_state WHERE key='trophies_backfilled'").fetchone():
            return
        stamp = datetime.now(timezone.utc).isoformat()
        rows = db.execute("""SELECT user_id,game_id,difficulty,trophy_tier
            FROM minigame_high_scores WHERE verification_level='verified' AND trophy_tier>0""").fetchall()
        for row in rows:
            award_trophies(db, user_id=row['user_id'], game_id=row['game_id'],
                          difficulty=row['difficulty'], tier=row['trophy_tier'], stamp=stamp)
        db.execute("INSERT INTO token_reward_state VALUES ('trophies_backfilled',?)", (stamp,))


def settle_weeks(now=None):
    from backend.gamer_ranked.service import _week_start_iso

    now = now or datetime.now(timezone.utc)
    current = datetime.fromisoformat(_week_start_iso(now))
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        start = db.execute("SELECT value FROM token_reward_state WHERE key='weekly_start'").fetchone()[0]
        latest = db.execute("SELECT MAX(week_start) FROM token_weekly_settlements").fetchone()[0]
        week = datetime.fromisoformat(latest) + timedelta(days=7) if latest else datetime.fromisoformat(start)
        while week < current:
            end = week + timedelta(days=7)
            # A submitted run belongs to its submission week, even if validation is late.
            pending = db.execute("""SELECT 1 FROM gamer_ranked_runs
                WHERE status IN ('pending','validating')
                  AND julianday(submitted_at)>=julianday(?) AND julianday(submitted_at)<julianday(?)
                LIMIT 1""", (week.isoformat(), end.isoformat())).fetchone()
            if pending:
                break
            for board, awards in WEEKLY_REWARDS.items():
                rows = db.execute("""SELECT s.user_id,s.score,s.run_id FROM gamer_weekly_high_scores s
                    JOIN users u ON u.id=s.user_id
                    WHERE s.week_start=? AND s.board_key=? AND u.status='active'
                      AND TRIM(COALESCE(u.display_name,''))<>''
                    ORDER BY s.score DESC,s.achieved_at ASC,s.user_id ASC LIMIT 5""",
                    (week.isoformat(), board)).fetchall()
                for rank, row in enumerate(rows, 1):
                    _credit(db, key=f"weekly:{week.isoformat()}:{board}:{rank}",
                            user_id=row['user_id'], tokens=awards[rank - 1], event="weekly_rank_award",
                            metadata={"week_start": week.isoformat(), "board_key": board,
                                      "rank": rank, "score": row['score'], "run_id": row['run_id']},
                            stamp=now.isoformat())
            db.execute("INSERT INTO token_weekly_settlements VALUES (?,?)", (week.isoformat(), now.isoformat()))
            week = end


def run_maintenance():
    backfill_trophies()
    settle_weeks()
