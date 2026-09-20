import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import median
from .statistics import VERSION, replay_stages


def week_bounds(now=None):
    today = (now or datetime.now(timezone(timedelta(hours=8)))).astimezone(timezone(timedelta(hours=8))).date()
    start = today - timedelta(days=today.weekday())
    return start.isoformat(), (start + timedelta(days=7)).isoformat()


class LiveStore:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get('LIVE_DB_PATH', 'data/live.sqlite3'))
        self._summary_cache = {}
        self._summary_cache_revision = 0
        self._summary_cache_lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS live_state (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS live_runs (
                    id TEXT PRIMARY KEY, score INTEGER, max_tile INTEGER, elapsed INTEGER,
                    ended REAL, day TEXT, replay TEXT);
                CREATE TABLE IF NOT EXISTS live_days (
                    day TEXT PRIMARY KEY, games INTEGER, score_sum INTEGER, tile32 INTEGER, tile64 INTEGER);
                CREATE INDEX IF NOT EXISTS live_runs_ended ON live_runs(ended DESC, id DESC);
                CREATE TABLE IF NOT EXISTS live_totals (id INTEGER PRIMARY KEY, best INTEGER, likes INTEGER);
                INSERT OR IGNORE INTO live_totals VALUES (1, 0, 0);
                CREATE TABLE IF NOT EXISTS live_control (
                    id INTEGER PRIMARY KEY, enabled INTEGER NOT NULL, revision INTEGER NOT NULL);
                INSERT OR IGNORE INTO live_control VALUES (1, 1, 0);
                CREATE TABLE IF NOT EXISTS live_scores (id TEXT PRIMARY KEY, day TEXT NOT NULL, score INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS live_scores_day ON live_scores(day, score);
                INSERT OR IGNORE INTO live_scores SELECT id, day, score FROM live_runs;
                CREATE TABLE IF NOT EXISTS live_run_stages (
                    run_id TEXT PRIMARY KEY, version INTEGER NOT NULL,
                    passed INTEGER, failed INTEGER);
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

    def _invalidate_summary_cache(self):
        with self._summary_cache_lock:
            self._summary_cache_revision += 1
            self._summary_cache.clear()

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
        self.backfill_stats(run_id=run.id)

    def backfill_stats(self, run_id=None, limit=1000000):
        with self.connect() as db:
            rows = db.execute('''SELECT r.id,r.replay FROM live_runs r
                LEFT JOIN live_run_stages s ON s.run_id=r.id
                WHERE (s.run_id IS NULL OR s.version<>?) AND (? IS NULL OR r.id=?)
                LIMIT ?''', (VERSION, run_id, run_id, limit)).fetchall()
        for run_id, replay in rows:
            try:
                passed, failed = replay_stages(replay)
            except (ValueError, TypeError, IndexError):
                # Retain unknown coverage as NULL, never invent a failed attempt.
                passed = failed = None
            with self.connect() as db:
                db.execute('INSERT OR REPLACE INTO live_run_stages VALUES (?,?,?,?)',
                           (run_id, VERSION, passed, failed))
        if rows:
            self._invalidate_summary_cache()
        return len(rows)

    def history(self, page=1):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            db.execute('BEGIN')
            total = db.execute('SELECT count(*) FROM live_runs').fetchone()[0]
            pages = max(1, (total + 9) // 10)
            page = min(max(1, page), pages)
            rows = [dict(row) for row in db.execute(
                'SELECT id,score,max_tile,elapsed,ended FROM live_runs ORDER BY ended DESC,id DESC LIMIT 10 OFFSET ?',
                ((page - 1) * 10,))]
        return dict(history=rows, total=total, page=page, pages=pages)

    def summary(self, stats_range='all'):
        if stats_range not in {'all', '24h', 'recent100'}:
            stats_range = 'all'
        now = time.monotonic()
        ttl = 3.0 if stats_range == '24h' else 15.0
        with self._summary_cache_lock:
            revision = self._summary_cache_revision
            cached = self._summary_cache.get(stats_range)
            if (cached and cached[0] == revision
                    and now - cached[1] < ttl):
                return cached[2]
        result = self._summary_uncached(stats_range)
        with self._summary_cache_lock:
            if revision == self._summary_cache_revision:
                self._summary_cache[stats_range] = (revision, now, result)
        return result

    def _summary_uncached(self, stats_range):
        day = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        start, end = week_bounds()
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            db.execute('BEGIN')
            history = [dict(row) for row in db.execute('SELECT id,score,max_tile,elapsed,ended FROM live_runs ORDER BY ended DESC,id DESC LIMIT 10')]
            history_total = db.execute('SELECT count(*) FROM live_runs').fetchone()[0]
            daily = db.execute('SELECT * FROM live_days WHERE day=?', (day,)).fetchone()
            best, likes = db.execute('SELECT best,likes FROM live_totals WHERE id=1').fetchone()
            weekly = dict(db.execute('''SELECT coalesce(sum(games),0) AS games,
                coalesce(sum(score_sum),0) AS score_sum, coalesce(sum(tile32),0) AS tile32,
                coalesce(sum(tile64),0) AS tile64 FROM live_days WHERE day>=? AND day<?''', (start,end)).fetchone())
            scores = [row[0] for row in db.execute('SELECT score FROM live_scores WHERE day>=? AND day<?', (start,end))]
            if stats_range == '24h':
                run_filter, run_args = 'r.ended >= ?', (time.time() - 86400,)
            elif stats_range == 'recent100':
                run_filter, run_args = 'r.id IN (SELECT id FROM live_runs ORDER BY ended DESC,id DESC LIMIT 100)', ()
            else:
                run_filter, run_args = '1=1', ()
            all_scores = [row[0] for row in db.execute(
                f'SELECT r.score FROM live_runs r WHERE {run_filter}', run_args)]
            all_time = dict(db.execute(f'''SELECT count(*) AS games,
                coalesce(sum(score),0) AS score_sum,
                coalesce(sum(r.max_tile>=32768),0) AS tile32,
                coalesce(sum(r.max_tile>=65536),0) AS tile64 FROM live_runs r WHERE {run_filter}''', run_args).fetchone())
            coverage = dict(db.execute(f'''SELECT coalesce(sum(s.passed),0) AS passed,
                coalesce(sum(s.failed),0) AS failed, count(s.passed) AS analyzed_runs,
                count(*) AS processed_runs
                FROM live_runs r JOIN live_run_stages s ON s.run_id=r.id
                WHERE s.version=? AND {run_filter}''', (VERSION, *run_args)).fetchone())
        attempts = coverage['passed'] + coverage['failed']
        pending = all_time['games'] - coverage.pop('processed_runs')
        all_time.update(median_score=median(all_scores) if all_scores else None,
                        stage32=coverage, stage32_pending=pending,
                        stage32_rate=coverage['passed'] / attempts if attempts and not pending else None)
        weekly.update(start=start, end=end, median_score=median(scores) if scores and len(scores)==weekly['games'] else None)
        return dict(history=history, history_total=history_total, best=best, likes=likes,
                    week=weekly, all_time=all_time,
                    stats_range=stats_range,
                    today=dict(daily) if daily else dict(day=day, games=0, score_sum=0, tile32=0, tile64=0))

    def replay(self, run_id):
        with self.connect() as db:
            row = db.execute('SELECT replay FROM live_runs WHERE id=?', (run_id,)).fetchone()
        return row[0] if row else None

    def add_likes(self, count):
        if count:
            with self.connect() as db:
                db.execute('UPDATE live_totals SET likes=likes+? WHERE id=1', (count,))
