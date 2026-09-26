"""Rolling 168-hour ranked-game display, separate from legacy calendar-week awards."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import time

from backend import rolling_leaderboards as rolling
from backend.auth.db import auth_db

BOARDS = {
    'gamer_high_score_weekly': 'gamer_high_score',
    'gamer_adversarial_weekly': 'gamer_adversarial',
}


def ensure_backfill(db, now=None):
    if db.execute("SELECT 1 FROM rolling_meta WHERE key='gamer_rolling_v1'").fetchone():
        return
    now = time.time() if now is None else rolling.timestamp(now)
    cutoff = datetime.fromtimestamp(now - rolling.WINDOW_SECONDS, timezone.utc).isoformat()
    # Older non-PB records were not retained, so only available verified PBs can be
    # reconstructed. Every newly verified run is recorded from this migration on.
    for table in ('gamer_high_scores', 'gamer_weekly_high_scores'):
        rows = db.execute(f"""SELECT s.user_id,s.board_key,s.run_id,s.score,s.max_tile,
            s.move_count,s.used_ai,s.replay_id,s.achieved_at FROM {table} s
            JOIN users u ON u.id=s.user_id WHERE u.status='active'
            AND TRIM(COALESCE(u.display_name,''))<>''
            AND julianday(s.achieved_at)>=julianday(?)
            AND julianday(s.achieved_at)<julianday(?)""",
            (cutoff, datetime.fromtimestamp(now, timezone.utc).isoformat()))
        for row in rows:
            # Legacy score tables may contain inconsistent duplicated run IDs
            # from old data; keep the first narrow candidate rather than
            # aborting the migration or inventing a new verified run ID.
            prior = db.execute("SELECT 1 FROM rolling_candidates WHERE run_id=?",
                               (row['run_id'],)).fetchone()
            if prior:
                continue
            rolling.add(db, board_key=row['board_key'] + '_weekly',
                        run_id=row['run_id'], user_id=row['user_id'], score=row['score'],
                        max_tile=row['max_tile'], move_count=row['move_count'],
                        used_ai=row['used_ai'], has_replay=True,
                        replay_id=row['replay_id'], achieved_at=row['achieved_at'],
                        eligible_at=row['achieved_at'], now=now, refresh=False)
    for board in BOARDS:
        rolling.rebuild(db, board, now)
    db.execute("INSERT INTO rolling_meta(key,value) VALUES('gamer_rolling_v1',?)",
               (datetime.fromtimestamp(now, timezone.utc).isoformat(),))


def payload(board_key, limit=10, now=None):
    if board_key not in BOARDS:
        raise KeyError(board_key)
    now = time.time() if now is None else rolling.timestamp(now)
    with auth_db() as db:
        if (not db.execute("SELECT 1 FROM rolling_meta WHERE key='gamer_rolling_v1'").fetchone()
                or rolling.due(db, board_key, now)):
            db.execute('BEGIN IMMEDIATE')
        ensure_backfill(db, now)
        rows = rolling.entries(db, board_key, limit, now)
        state = db.execute("SELECT version,updated_at FROM rolling_board_state WHERE board_key=?",
                           (board_key,)).fetchone()
        ids = list({row['user_id'] for row in rows})
        identities = {}
        if ids:
            placeholders = ','.join('?' for _ in ids)
            identities = {row['id']: row for row in db.execute(f"""SELECT u.id,u.display_name,
                u.status,p.avatar_key,e.tier FROM users u
                LEFT JOIN user_profiles p ON p.user_id=u.id
                LEFT JOIN user_entitlements e ON e.user_id=u.id
                WHERE u.id IN ({placeholders})""", ids)}
        entries = []
        for row in rows:
            user = identities.get(row['user_id'])
            if not user or user['status'] != 'active' or not str(user['display_name'] or '').strip():
                continue
            entries.append({
                'entry_key': hashlib.sha256(
                    f"{board_key}:{state['version']}:{row['run_id']}".encode()).hexdigest()[:16],
                'rank': row['position'], 'display_name': user['display_name'].strip(),
                'is_supporter': user['tier'] == 'supporter',
                'avatar_url': f"/media/avatars/{user['avatar_key']}" if user['avatar_key'] else None,
                'score': row['score'], 'max_tile': row['max_tile'],
                'move_count': row['move_count'] or 0, 'used_ai': bool(row['used_ai']),
                'replay_id': row['replay_id'],
            })
        count = db.execute("SELECT COUNT(*) FROM rolling_board_entries WHERE board_key=?",
                           (board_key,)).fetchone()[0]
    iso = lambda value: datetime.fromtimestamp(value, timezone.utc).isoformat()
    return {'key': board_key, 'cadence': 'live', 'score_visible': True,
            'unit': 'points', 'period': {'start': iso(now - rolling.WINDOW_SECONDS),
                                        'end': iso(now)},
            'generated_at': iso(state['updated_at']), 'entry_count': count,
            'entries': entries}
