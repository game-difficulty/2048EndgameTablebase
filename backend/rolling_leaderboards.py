"""Narrow, event-maintained rolling scoreboards shared by the two game sites.

Callers own the SQLite transaction. Replay bodies deliberately live elsewhere.
"""
from __future__ import annotations

from datetime import datetime, timezone
import time

WINDOW_SECONDS = 7 * 24 * 60 * 60
BOARD_LIMIT = 100


def timestamp(value):
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    return datetime.fromisoformat(str(value)).timestamp()


def init_schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS rolling_candidates (
        run_id TEXT PRIMARY KEY, board_key TEXT NOT NULL, user_id INTEGER NOT NULL,
        score INTEGER NOT NULL, max_tile INTEGER NOT NULL DEFAULT 0,
        move_count INTEGER, used_ai INTEGER NOT NULL DEFAULT 0,
        source TEXT NOT NULL DEFAULT 'native', has_replay INTEGER NOT NULL DEFAULT 0,
        replay_id TEXT, achieved_at REAL NOT NULL, eligible_at REAL NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
    )""")
    db.execute("""CREATE INDEX IF NOT EXISTS rolling_candidates_user_score
        ON rolling_candidates(board_key,user_id,score DESC,achieved_at,run_id)""")
    db.execute("""CREATE INDEX IF NOT EXISTS rolling_candidates_time
        ON rolling_candidates(board_key,achieved_at)""")
    db.execute("""CREATE TABLE IF NOT EXISTS rolling_candidate_periods (
        id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, start_at REAL NOT NULL,
        end_at REAL, FOREIGN KEY(run_id) REFERENCES rolling_candidates(run_id) ON DELETE CASCADE
    )""")
    db.execute("""CREATE INDEX IF NOT EXISTS rolling_periods_run
        ON rolling_candidate_periods(run_id,end_at)""")
    db.execute("""CREATE TABLE IF NOT EXISTS rolling_player_best (
        board_key TEXT NOT NULL, user_id INTEGER NOT NULL, run_id TEXT NOT NULL,
        score INTEGER NOT NULL, achieved_at REAL NOT NULL,
        PRIMARY KEY(board_key,user_id)
    )""")
    db.execute("""CREATE INDEX IF NOT EXISTS rolling_best_score
        ON rolling_player_best(board_key,score DESC,achieved_at,user_id,run_id)""")
    db.execute("""CREATE INDEX IF NOT EXISTS rolling_best_expiry
        ON rolling_player_best(board_key,achieved_at)""")
    db.execute("""CREATE TABLE IF NOT EXISTS rolling_board_entries (
        board_key TEXT NOT NULL, position INTEGER NOT NULL, run_id TEXT NOT NULL,
        PRIMARY KEY(board_key,position)
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS rolling_board_state (
        board_key TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0,
        updated_at REAL NOT NULL, next_expiry_at REAL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS rolling_meta (
        key TEXT PRIMARY KEY, value TEXT NOT NULL
    )""")


def _best_run(db, board_key, user_id, now):
    return db.execute("""SELECT run_id,score,achieved_at FROM rolling_candidates
        WHERE board_key=? AND user_id=? AND active=1 AND eligible_at<=?
          AND achieved_at>=? AND achieved_at<?
        ORDER BY score DESC,achieved_at,run_id LIMIT 1""",
        (board_key, user_id, now, now - WINDOW_SECONDS, now)).fetchone()


def _refresh_player(db, board_key, user_id, now):
    old = db.execute("SELECT run_id FROM rolling_player_best WHERE board_key=? AND user_id=?",
                     (board_key, user_id)).fetchone()
    best = _best_run(db, board_key, user_id, now)
    if old and best and old['run_id'] == best['run_id']:
        return False
    if best:
        db.execute("""INSERT INTO rolling_player_best(board_key,user_id,run_id,score,achieved_at)
            VALUES(?,?,?,?,?) ON CONFLICT(board_key,user_id) DO UPDATE SET
            run_id=excluded.run_id,score=excluded.score,achieved_at=excluded.achieved_at""",
            (board_key, user_id, best['run_id'], best['score'], best['achieved_at']))
    elif old:
        db.execute("DELETE FROM rolling_player_best WHERE board_key=? AND user_id=?",
                   (board_key, user_id))
    return bool(old or best)


def _refresh_entries(db, board_key, now):
    selected = db.execute("""SELECT run_id FROM rolling_player_best
        WHERE board_key=? AND achieved_at>=? AND achieved_at<?
        ORDER BY score DESC,achieved_at,user_id,run_id LIMIT ?""",
        (board_key, now - WINDOW_SECONDS, now, BOARD_LIMIT)).fetchall()
    wanted = [row['run_id'] for row in selected]
    existing = [row['run_id'] for row in db.execute(
        "SELECT run_id FROM rolling_board_entries WHERE board_key=? ORDER BY position",
        (board_key,)).fetchall()]
    changed = wanted != existing
    if changed:
        db.execute("DELETE FROM rolling_board_entries WHERE board_key=?", (board_key,))
        db.executemany("INSERT INTO rolling_board_entries VALUES(?,?,?)",
                       [(board_key, index, run_id) for index, run_id in enumerate(wanted, 1)])
    earliest = db.execute("SELECT MIN(achieved_at) FROM rolling_player_best WHERE board_key=?",
                          (board_key,)).fetchone()[0]
    db.execute("""INSERT INTO rolling_board_state(board_key,version,updated_at,next_expiry_at)
        VALUES(?,1,?,?) ON CONFLICT(board_key) DO UPDATE SET
        version=rolling_board_state.version+?,updated_at=excluded.updated_at,
        next_expiry_at=excluded.next_expiry_at""",
        (board_key, now, earliest + WINDOW_SECONDS if earliest is not None else None,
         int(changed)))
    return changed


def maintain(db, board_key, now=None):
    """Repair due expiries; ordinary reads only check a single state row."""
    now = time.time() if now is None else timestamp(now)
    state = db.execute("SELECT next_expiry_at FROM rolling_board_state WHERE board_key=?",
                       (board_key,)).fetchone()
    if state is None:
        users = db.execute("""SELECT DISTINCT user_id FROM rolling_candidates
            WHERE board_key=? AND active=1 AND achieved_at>=? AND achieved_at<?""",
            (board_key, now - WINDOW_SECONDS, now)).fetchall()
        for row in users:
            _refresh_player(db, board_key, row['user_id'], now)
        _refresh_entries(db, board_key, now)
        return True
    if state['next_expiry_at'] is None or state['next_expiry_at'] >= now:
        return False
    expired = db.execute("""SELECT user_id FROM rolling_player_best
        WHERE board_key=? AND achieved_at<?""",
        (board_key, now - WINDOW_SECONDS)).fetchall()
    for row in expired:
        _refresh_player(db, board_key, row['user_id'], now)
    _refresh_entries(db, board_key, now)
    return True


def due(db, board_key, now=None):
    now = time.time() if now is None else timestamp(now)
    row = db.execute("SELECT next_expiry_at FROM rolling_board_state WHERE board_key=?",
                     (board_key,)).fetchone()
    return row is None or (row['next_expiry_at'] is not None
                           and row['next_expiry_at'] < now)


def rebuild(db, board_key, now=None):
    """Rare migration/audit repair, never called by a normal page read."""
    now = time.time() if now is None else timestamp(now)
    db.execute("DELETE FROM rolling_player_best WHERE board_key=?", (board_key,))
    users = db.execute("""SELECT DISTINCT user_id FROM rolling_candidates
        WHERE board_key=? AND active=1 AND achieved_at>=? AND achieved_at<?""",
        (board_key, now - WINDOW_SECONDS, now)).fetchall()
    for row in users:
        _refresh_player(db, board_key, row['user_id'], now)
    _refresh_entries(db, board_key, now)


def add(db, *, board_key, run_id, user_id, score, achieved_at, eligible_at,
        max_tile=0, move_count=None, used_ai=False, source='native',
        has_replay=False, replay_id=None, now=None, refresh=True):
    now = time.time() if now is None else timestamp(now)
    achieved_at, eligible_at = timestamp(achieved_at), timestamp(eligible_at)
    # An archive can be stamped with the same clock reading as this event.
    # The right-open window becomes visible one microsecond later.
    now = max(now, achieved_at + 0.000001, eligible_at)
    old = db.execute("SELECT board_key,user_id,score,achieved_at,active FROM rolling_candidates WHERE run_id=?",
                     (run_id,)).fetchone()
    if old and (old['board_key'], old['user_id'], old['score'], old['achieved_at']) != (
            board_key, user_id, score, achieved_at):
        raise ValueError('rolling_candidate_conflict')
    if old:
        db.execute("""UPDATE rolling_candidates SET has_replay=MAX(has_replay,?),
            replay_id=COALESCE(?,replay_id),active=1,
            eligible_at=CASE WHEN active=0 THEN ? ELSE eligible_at END WHERE run_id=?""",
            (int(has_replay), replay_id, eligible_at, run_id))
        if not old['active']:
            db.execute("INSERT INTO rolling_candidate_periods(run_id,start_at) VALUES(?,?)",
                       (run_id, eligible_at))
    else:
        db.execute("""INSERT INTO rolling_candidates
            (run_id,board_key,user_id,score,max_tile,move_count,used_ai,source,
             has_replay,replay_id,achieved_at,eligible_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, board_key, user_id, score, max_tile, move_count, int(used_ai),
             source, int(has_replay), replay_id, achieved_at, eligible_at))
        db.execute("INSERT INTO rolling_candidate_periods(run_id,start_at) VALUES(?,?)",
                   (run_id, eligible_at))
    if refresh:
        maintain(db, board_key, now)
        if _refresh_player(db, board_key, user_id, now):
            _refresh_entries(db, board_key, now)


def revoke(db, run_id, now=None, refresh=True):
    now = time.time() if now is None else timestamp(now)
    row = db.execute("SELECT board_key,user_id,active FROM rolling_candidates WHERE run_id=?",
                     (run_id,)).fetchone()
    if not row or not row['active']:
        return
    db.execute("UPDATE rolling_candidates SET active=0 WHERE run_id=?", (run_id,))
    db.execute("""UPDATE rolling_candidate_periods SET end_at=?
        WHERE run_id=? AND end_at IS NULL""", (now, run_id))
    if refresh:
        maintain(db, row['board_key'], now)
        if _refresh_player(db, row['board_key'], row['user_id'], now):
            _refresh_entries(db, row['board_key'], now)


def refresh_user(db, board_key, user_id, now=None):
    now = time.time() if now is None else timestamp(now)
    maintain(db, board_key, now)
    if _refresh_player(db, board_key, user_id, now):
        _refresh_entries(db, board_key, now)


def suspend_user(db, board_key, user_id, now=None):
    """Temporarily remove a player without invalidating verified candidates."""
    now = time.time() if now is None else timestamp(now)
    maintain(db, board_key, now)
    removed = db.execute(
        "DELETE FROM rolling_player_best WHERE board_key=? AND user_id=?",
        (board_key, user_id),
    ).rowcount
    if removed:
        _refresh_entries(db, board_key, now)


def entries(db, board_key, limit=10, now=None):
    now = time.time() if now is None else timestamp(now)
    maintain(db, board_key, now)
    return db.execute("""SELECT e.position,c.* FROM rolling_board_entries e
        JOIN rolling_candidates c ON c.run_id=e.run_id
        WHERE e.board_key=? AND c.active=1 AND c.achieved_at>=? AND c.achieved_at<?
        ORDER BY e.position LIMIT ?""",
        (board_key, now - WINDOW_SECONDS, now, max(1, min(BOARD_LIMIT, int(limit))))).fetchall()


def as_of(db, board_key, end, limit):
    """Infrequent settlement query over narrow metadata, never replay bodies."""
    end = timestamp(end)
    return db.execute("""SELECT * FROM (
        SELECT c.*,row_number() OVER (PARTITION BY c.user_id
            ORDER BY c.score DESC,c.achieved_at,c.run_id) AS personal_rank
        FROM rolling_candidates c
        WHERE c.board_key=? AND c.active=1 AND c.achieved_at>=? AND c.achieved_at<?
          AND EXISTS (SELECT 1 FROM rolling_candidate_periods p WHERE p.run_id=c.run_id
            AND p.start_at<? AND (p.end_at IS NULL OR p.end_at>?))
        ) WHERE personal_rank=1
        ORDER BY score DESC,achieved_at,user_id,run_id LIMIT ?""",
        (board_key, end - WINDOW_SECONDS, end, end, end, limit)).fetchall()
