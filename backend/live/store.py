import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import median


def week_bounds(now=None):
    today = (now or datetime.now(timezone(timedelta(hours=8)))).astimezone(timezone(timedelta(hours=8))).date()
    start = today - timedelta(days=today.weekday())
    return start.isoformat(), (start + timedelta(days=7)).isoformat()


class LiveStore:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get('LIVE_DB_PATH', 'data/live.sqlite3'))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS live_state (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS live_runs (
                    id TEXT PRIMARY KEY, score INTEGER, max_tile INTEGER, elapsed INTEGER,
                    ended REAL, day TEXT, replay TEXT);
                CREATE TABLE IF NOT EXISTS live_days (
                    day TEXT PRIMARY KEY, games INTEGER, score_sum INTEGER, tile32 INTEGER, tile64 INTEGER);
                CREATE TABLE IF NOT EXISTS live_totals (id INTEGER PRIMARY KEY, best INTEGER, likes INTEGER);
                INSERT OR IGNORE INTO live_totals VALUES (1, 0, 0);
                CREATE TABLE IF NOT EXISTS live_control (
                    id INTEGER PRIMARY KEY, enabled INTEGER NOT NULL, revision INTEGER NOT NULL);
                INSERT OR IGNORE INTO live_control VALUES (1, 1, 0);
                CREATE TABLE IF NOT EXISTS live_scores (id TEXT PRIMARY KEY, day TEXT NOT NULL, score INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS live_scores_day ON live_scores(day, score);
                INSERT OR IGNORE INTO live_scores SELECT id, day, score FROM live_runs;
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def load(self):
        with self.connect() as db:
            row = db.execute('SELECT value FROM live_state WHERE id=1').fetchone()
        return json.loads(row[0]) if row else None

    def control(self, enabled=None):
        with self.connect() as db:
            if enabled is not None:
                db.execute('UPDATE live_control SET enabled=?, revision=revision+1 WHERE id=1 AND enabled<>?',
                           (int(enabled), int(enabled)))
            row = db.execute('SELECT enabled, revision FROM live_control WHERE id=1').fetchone()
        return dict(enabled=bool(row[0]), revision=row[1])

    def save(self, run):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO live_state VALUES (1, ?)', (json.dumps(run.checkpoint()),))

    def finish(self, run):
        day = datetime.fromtimestamp(run.ended, timezone(timedelta(hours=8))).date().isoformat()
        maximum = max(run.board)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            inserted = db.execute('INSERT OR IGNORE INTO live_scores VALUES (?,?,?)',
                                  (run.id, day, run.score)).rowcount
            db.execute('INSERT OR IGNORE INTO live_runs VALUES (?,?,?,?,?,?,?)',
                (run.id, run.score, maximum, run.elapsed, run.ended, day, run.replay()))
            if inserted:
                db.execute('''INSERT INTO live_days VALUES (?,1,?,?,?) ON CONFLICT(day) DO UPDATE SET
                    games=games+1, score_sum=score_sum+excluded.score_sum,
                    tile32=tile32+excluded.tile32, tile64=tile64+excluded.tile64''',
                    (day, run.score, int(maximum >= 32768), int(maximum >= 65536)))
                db.execute('UPDATE live_totals SET best=max(best,?) WHERE id=1', (run.score,))
            # At most 200 compact replays; counters remain in the small aggregate tables.
            db.execute('DELETE FROM live_runs WHERE id NOT IN (SELECT id FROM live_runs ORDER BY ended DESC LIMIT 200)')
            while db.execute('SELECT coalesce(sum(length(replay)),0) FROM live_runs').fetchone()[0] > 100 * 1024 * 1024:
                db.execute('DELETE FROM live_runs WHERE id=(SELECT id FROM live_runs ORDER BY ended LIMIT 1)')

    def summary(self):
        day = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        start, end = week_bounds()
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            db.execute('BEGIN')
            history = [dict(row) for row in db.execute('SELECT id,score,max_tile,elapsed,ended FROM live_runs ORDER BY ended DESC LIMIT 20')]
            daily = db.execute('SELECT * FROM live_days WHERE day=?', (day,)).fetchone()
            best, likes = db.execute('SELECT best,likes FROM live_totals WHERE id=1').fetchone()
            weekly = dict(db.execute('''SELECT coalesce(sum(games),0) AS games,
                coalesce(sum(score_sum),0) AS score_sum, coalesce(sum(tile32),0) AS tile32,
                coalesce(sum(tile64),0) AS tile64 FROM live_days WHERE day>=? AND day<?''', (start,end)).fetchone())
            scores = [row[0] for row in db.execute('SELECT score FROM live_scores WHERE day>=? AND day<?', (start,end))]
        weekly.update(start=start, end=end, median_score=median(scores) if scores and len(scores)==weekly['games'] else None)
        return dict(history=history, best=best, likes=likes,
                    week=weekly,
                    today=dict(daily) if daily else dict(day=day, games=0, score_sum=0, tile32=0, tile64=0))

    def replay(self, run_id):
        with self.connect() as db:
            row = db.execute('SELECT replay FROM live_runs WHERE id=?', (run_id,)).fetchone()
        return row[0] if row else None

    def add_likes(self, count):
        if count:
            with self.connect() as db:
                db.execute('UPDATE live_totals SET likes=likes+? WHERE id=1', (count,))
