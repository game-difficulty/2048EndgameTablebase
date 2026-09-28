"""Verified achievement rewards, credited atomically with durable receipts."""
from datetime import datetime, timedelta, timezone
import json

from backend.auth.db import auth_db

TROPHY_REWARDS = {1: 2_000, 2: 3_000, 3: 5_000, 4: 10_000}
WEEKLY_REWARDS = {
    "gamer_high_score": (10_000, 8_000, 5_000, 3_000, 2_000),
    "gamer_adversarial": (5_000, 4_000, 3_000, 2_000, 1_000),
}
HUMAN_WEEKLY_REWARDS = {
    '4x4': (36_000, 24_000, 16_000, 12_000, 10_000, 8_000,
            6_000, 5_000, 4_000, 3_000),
    '3x4': (16_000, 12_000, 10_000, 8_000, 6_000, 5_000,
            4_000, 3_000, 2_000, 1_000),
    '3x3': (16_000, 12_000, 10_000, 8_000, 6_000, 5_000,
            4_000, 3_000, 2_000, 1_000),
    '2x4': (16_000, 12_000, 10_000, 8_000, 6_000, 5_000,
            4_000, 3_000, 2_000, 1_000),
}


def _next_monday_08(now):
    local = now.astimezone(timezone(timedelta(hours=8)))
    days = (7-local.weekday()) % 7
    candidate = (local + timedelta(days=days)).replace(hour=8, minute=0,
        second=0, microsecond=0)
    if candidate <= local:
        candidate += timedelta(days=7)
    return candidate.astimezone(timezone.utc)


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
    db.execute("""CREATE TABLE IF NOT EXISTS token_rolling_settlements (
        boundary TEXT PRIMARY KEY, rule_version INTEGER NOT NULL,
        settled_at TEXT NOT NULL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS token_human_rolling_settlements (
        boundary TEXT PRIMARY KEY, settled_at TEXT NOT NULL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS token_rolling_snapshot (
        boundary TEXT NOT NULL, board_key TEXT NOT NULL, position INTEGER NOT NULL,
        user_id INTEGER NOT NULL, run_id TEXT NOT NULL, score INTEGER NOT NULL,
        achieved_at REAL NOT NULL, tokens INTEGER NOT NULL,
        PRIMARY KEY(boundary,board_key,position)
    )""")
    # A new installation observes a complete 168-hour window before its first
    # rolling payout. Already-paid calendar-week receipts remain untouched.
    first = _next_monday_08(datetime.now(timezone.utc) + timedelta(days=7))
    db.execute("INSERT OR IGNORE INTO token_reward_state VALUES ('rolling_first_boundary',?)",
               (first.isoformat(),))


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
        start = db.execute("SELECT value FROM token_reward_state WHERE key='weekly_start'").fetchone()[0]
        latest = db.execute("SELECT MAX(week_start) FROM token_weekly_settlements").fetchone()[0]
        week = datetime.fromisoformat(latest) + timedelta(days=7) if latest else datetime.fromisoformat(start)
        first = datetime.fromisoformat(db.execute("""SELECT value FROM token_reward_state
            WHERE key='rolling_first_boundary'""").fetchone()[0])
        if week >= current or week + timedelta(days=7) > first - timedelta(days=7):
            return
        db.execute("BEGIN IMMEDIATE")
        # Another worker may have settled while this one waited for the lock.
        latest = db.execute("SELECT MAX(week_start) FROM token_weekly_settlements").fetchone()[0]
        week = datetime.fromisoformat(latest) + timedelta(days=7) if latest else datetime.fromisoformat(start)
        while week < current:
            end = week + timedelta(days=7)
            if end > first - timedelta(days=7):
                break
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


def settle_rolling_weeks(now=None):
    """Freeze six rolling boards at Monday 08:00 Beijing and credit once.

    The live 7-day rankings keep sliding every moment; this fixed boundary
    applies only to which 168-hour snapshot receives the weekly rewards.
    """
    from backend import rolling_leaderboards as rolling
    from backend.human_play.store import database as human_database
    from backend.human_play.rolling import as_of as human_as_of
    now = now or datetime.now(timezone.utc)
    with auth_db() as db:
        first = datetime.fromisoformat(db.execute("""SELECT value FROM token_reward_state
            WHERE key='rolling_first_boundary'""").fetchone()[0])
        latest = db.execute("SELECT MAX(boundary) FROM token_rolling_settlements").fetchone()[0]
    boundary = datetime.fromisoformat(latest) + timedelta(days=7) if latest else first
    while boundary <= now:
        lower = boundary - timedelta(seconds=rolling.WINDOW_SECONDS)
        with auth_db() as db:
            if db.execute("""SELECT 1 FROM gamer_ranked_runs
                WHERE status IN ('pending','validating')
                  AND julianday(submitted_at)>=julianday(?)
                  AND julianday(submitted_at)<julianday(?) LIMIT 1""",
                (lower.isoformat(), boundary.isoformat())).fetchone():
                break
        # Human scores are in a different SQLite file. Serialize its one-time
        # backfill and snapshot without holding the account database write lock.
        with human_database() as human_db:
            human_db.execute('BEGIN IMMEDIATE')
            human_rows = {variant: [dict(row) for row in human_as_of(
                human_db, variant, boundary, 100)]
                for variant in HUMAN_WEEKLY_REWARDS}
        with auth_db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM token_rolling_settlements WHERE boundary=?",
                          (boundary.isoformat(),)).fetchone():
                boundary += timedelta(days=7)
                continue
            pending = db.execute("""SELECT 1 FROM gamer_ranked_runs
                WHERE status IN ('pending','validating')
                  AND julianday(submitted_at)>=julianday(?)
                  AND julianday(submitted_at)<julianday(?) LIMIT 1""",
                (lower.isoformat(), boundary.isoformat())).fetchone()
            if pending:
                break
            boards = {}
            for board in WEEKLY_REWARDS:
                boards[board + '_weekly'] = [dict(row) for row in rolling.as_of(
                    db, board + '_weekly', boundary, 100)]
            for variant, rows in human_rows.items():
                boards['human:' + variant] = rows
            for board, rows in boards.items():
                awards = (HUMAN_WEEKLY_REWARDS[board[6:]] if board.startswith('human:')
                          else WEEKLY_REWARDS[board[:-7]])
                position = 0
                for row in rows:
                    user = db.execute("""SELECT 1 FROM users WHERE id=? AND status='active'
                        AND TRIM(COALESCE(display_name,''))<>''""",
                        (row['user_id'],)).fetchone()
                    if not user:
                        continue
                    position += 1
                    if position > len(awards):
                        break
                    db.execute("""INSERT OR IGNORE INTO token_rolling_snapshot
                        VALUES(?,?,?,?,?,?,?,?)""", (boundary.isoformat(), board,
                        position, row['user_id'], row['run_id'], row['score'],
                        row['achieved_at'], awards[position-1]))
                    _credit(db, key=f"rolling:{boundary.isoformat()}:{board}:{position}",
                            user_id=row['user_id'], tokens=awards[position-1],
                            event='weekly_rank_award', metadata={
                                'rule_version': 2, 'boundary': boundary.isoformat(),
                                'board_key': board, 'rank': position,
                                'score': row['score'], 'run_id': row['run_id']},
                            stamp=now.isoformat())
            db.execute("INSERT INTO token_rolling_settlements VALUES(?,2,?)",
                       (boundary.isoformat(), now.isoformat()))
        boundary += timedelta(days=7)


def settle_human_rolling_weeks(now=None):
    """Play-only settlement while the main site's release remains independent.

    It uses the unified reward keys and snapshots, so a later unified release
    can settle the same boundary without crediting a player twice.
    """
    from backend.human_play.store import database as human_database
    from backend.human_play.rolling import as_of as human_as_of

    now = now or datetime.now(timezone.utc)
    with auth_db() as db:
        first = datetime.fromisoformat(db.execute("""SELECT value FROM token_reward_state
            WHERE key='rolling_first_boundary'""").fetchone()[0])
        latest = db.execute("SELECT MAX(boundary) FROM token_human_rolling_settlements").fetchone()[0]
    boundary = datetime.fromisoformat(latest) + timedelta(days=7) if latest else first
    while boundary <= now:
        with human_database() as human_db:
            human_db.execute('BEGIN IMMEDIATE')
            rows_by_variant = {variant: [dict(row) for row in human_as_of(
                human_db, variant, boundary, 100)] for variant in HUMAN_WEEKLY_REWARDS}
        with auth_db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM token_human_rolling_settlements WHERE boundary=?",
                          (boundary.isoformat(),)).fetchone():
                boundary += timedelta(days=7)
                continue
            for variant, rows in rows_by_variant.items():
                board = 'human:' + variant
                awards = HUMAN_WEEKLY_REWARDS[variant]
                position = 0
                for row in rows:
                    active = db.execute("""SELECT 1 FROM users WHERE id=? AND status='active'
                        AND TRIM(COALESCE(display_name,''))<>''""", (row['user_id'],)).fetchone()
                    if not active:
                        continue
                    position += 1
                    if position > len(awards):
                        break
                    db.execute("""INSERT OR IGNORE INTO token_rolling_snapshot
                        VALUES(?,?,?,?,?,?,?,?)""", (boundary.isoformat(), board,
                        position, row['user_id'], row['run_id'], row['score'],
                        row['achieved_at'], awards[position-1]))
                    _credit(db, key=f"rolling:{boundary.isoformat()}:{board}:{position}",
                            user_id=row['user_id'], tokens=awards[position-1],
                            event='weekly_rank_award', metadata={
                                'rule_version': 2, 'boundary': boundary.isoformat(),
                                'board_key': board, 'rank': position,
                                'score': row['score'], 'run_id': row['run_id']},
                            stamp=now.isoformat())
            db.execute("INSERT INTO token_human_rolling_settlements VALUES(?,?)",
                       (boundary.isoformat(), now.isoformat()))
        boundary += timedelta(days=7)


def maintain_rolling_boards():
    """Keep expiries current without waiting for a visitor, then prune paid data."""
    import time
    from backend import rolling_leaderboards as rolling
    from backend.leaderboards.rolling_gamer import BOARDS, ensure_backfill
    from backend.human_play.rolling import VARIANTS, ensure_backfill as human_backfill
    from backend.human_play.store import database as human_database
    now = time.time()
    with auth_db() as db:
        settled = db.execute("SELECT MAX(boundary) FROM token_rolling_settlements").fetchone()[0]
        cutoff = (datetime.fromisoformat(settled).timestamp() - rolling.WINDOW_SECONDS
                  if settled else None)
        prune_due = cutoff is not None and any(db.execute("""SELECT 1 FROM rolling_candidates c
            WHERE c.board_key=? AND c.achieved_at<=?
              AND NOT EXISTS (SELECT 1 FROM gamer_high_scores h WHERE h.run_id=c.run_id)
              AND NOT EXISTS (SELECT 1 FROM gamer_weekly_high_scores w WHERE w.run_id=c.run_id)
            LIMIT 1""", (board, cutoff)).fetchone()
            for board in BOARDS)
        work_due = (not db.execute("SELECT 1 FROM rolling_meta WHERE key='gamer_rolling_v1'").fetchone()
                    or any(rolling.due(db, board, now) for board in BOARDS) or prune_due)
        if work_due:
            db.execute('BEGIN IMMEDIATE')
            ensure_backfill(db, now)
            for board in BOARDS:
                rolling.maintain(db, board, now)
        if settled:
            if prune_due:
                db.execute("""DELETE FROM gamer_rolling_replays WHERE run_id IN
                    (SELECT c.run_id FROM rolling_candidates c WHERE c.achieved_at<=?)
                    AND run_id NOT IN (SELECT run_id FROM gamer_high_scores)
                    AND run_id NOT IN (SELECT run_id FROM gamer_weekly_high_scores)""", (cutoff,))
                db.execute("""DELETE FROM rolling_candidates WHERE achieved_at<=?
                    AND run_id NOT IN (SELECT run_id FROM gamer_high_scores)
                    AND run_id NOT IN (SELECT run_id FROM gamer_weekly_high_scores)""", (cutoff,))
    with human_database() as db:
        human_prune_due = cutoff is not None and any(db.execute("""SELECT 1 FROM rolling_candidates
            WHERE board_key=? AND achieved_at<=? LIMIT 1""", (variant, cutoff)).fetchone()
            for variant in VARIANTS)
        human_work_due = (not db.execute("SELECT 1 FROM rolling_meta WHERE key='human_rolling_v1'").fetchone()
                          or any(rolling.due(db, variant, now) for variant in VARIANTS)
                          or human_prune_due)
        if human_work_due:
            db.execute('BEGIN IMMEDIATE')
            human_backfill(db, now)
            for variant in VARIANTS:
                rolling.maintain(db, variant, now)
            if human_prune_due:
                db.execute("DELETE FROM rolling_candidates WHERE achieved_at<=?", (cutoff,))


def run_maintenance():
    backfill_trophies()
    settle_weeks()
    settle_rolling_weeks()
    maintain_rolling_boards()
