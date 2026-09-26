"""Narrow rolling-score cache for archived Play games."""
import json
import time

from backend import rolling_leaderboards as core

VARIANTS = ('4x4', '3x4', '3x3', '2x4')


def _add_row(db, row, eligible_at, *, refresh=False, now=None):
    state = json.loads(row['state'])
    core.add(db, board_key=row['variant'], run_id=row['id'],
             user_id=row['user_id'], score=state['score'],
             max_tile=max(state['board']), move_count=state.get('seq'),
             source=row['source'], has_replay=bool(row['has_replay']),
             replay_id=row['id'] if row['has_replay'] else None,
             achieved_at=row['ended'], eligible_at=eligible_at,
             refresh=refresh, now=now)


def ensure_backfill(db, now=None):
    if db.execute("SELECT 1 FROM rolling_meta WHERE key='human_rolling_v1'").fetchone():
        return
    from .service import RANKABLE_SQL, identity_map
    now = time.time() if now is None else core.timestamp(now)
    rows = db.execute(f"""SELECT id,user_id,variant,state,ended,source,has_replay,
        (SELECT created FROM human_rank_approvals a WHERE a.run_id=human_runs.id) AS approved_at,
        (SELECT updated FROM human_external_claims c WHERE c.provider='verse'
            AND human_runs.browser='verse:' || c.id) AS claim_at
        FROM human_runs WHERE status='sealed' AND {RANKABLE_SQL}
          AND source!='manual' AND ended>=? AND ended<?""", (now-core.WINDOW_SECONDS, now)).fetchall()
    active_users = identity_map({row['user_id'] for row in rows})
    for row in rows:
        if row['user_id'] not in active_users:
            continue
        eligible = row['claim_at'] if row['source'] == 'verse' else (
            row['approved_at'] or row['ended'])
        _add_row(db, row, eligible, now=now)
    for variant in VARIANTS:
        core.rebuild(db, variant, now)
    db.execute("INSERT INTO rolling_meta VALUES('human_rolling_v1',?)", (str(now),))


def add_run(db, run_id, eligible_at=None, now=None, refresh=True):
    now = time.time() if now is None else core.timestamp(now)
    ensure_backfill(db, now)
    row = db.execute("""SELECT id,user_id,variant,state,ended,source,has_replay
        FROM human_runs WHERE id=? AND status='sealed' AND visible=1
          AND eligibility='eligible' AND source!='manual'""", (run_id,)).fetchone()
    if row:
        _add_row(db, row, eligible_at or row['ended'], refresh=refresh, now=now)


def revoke_run(db, run_id, now=None, refresh=True):
    core.revoke(db, run_id, now, refresh=refresh)


def entries(db, variant, limit=10, now=None):
    ensure_backfill(db, now)
    return core.entries(db, variant, limit, now)


def player_entry(db, variant, user_id, now=None):
    """Return one player's maintained 168-hour best and score rank."""
    ensure_backfill(db, now)
    core.maintain(db, variant, now)
    row = db.execute("""SELECT c.* FROM rolling_player_best p
        JOIN rolling_candidates c ON c.run_id=p.run_id
        WHERE p.board_key=? AND p.user_id=?""", (variant, user_id)).fetchone()
    if not row:
        return None
    rank = db.execute("""SELECT 1+COUNT(*) FROM rolling_player_best
        WHERE board_key=? AND score>?""", (variant, row['score'])).fetchone()[0]
    return row, rank


def as_of(db, variant, end, limit=100):
    ensure_backfill(db)
    return core.as_of(db, variant, end, limit)
