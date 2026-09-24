import json
from .store import LiveStore
from .protocol import LiveRun


class MultiLiveStore(LiveStore):
    def __init__(self, path=None):
        super().__init__(path)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS live_active_runs (
                    lane INTEGER PRIMARY KEY, generation INTEGER NOT NULL, checkpoint TEXT);
                CREATE TABLE IF NOT EXISTS live_run_lanes (
                    run_id TEXT PRIMARY KEY, lane INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS live_batch_state (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS live_activity_outbox (
                    trigger_key TEXT PRIMARY KEY, batch_id TEXT NOT NULL, milestone INTEGER NOT NULL,
                    participant_id TEXT NOT NULL, run_id TEXT NOT NULL, content_seq INTEGER NOT NULL,
                    created REAL NOT NULL, delivered INTEGER NOT NULL DEFAULT 0);
            ''')

    def load_slots(self):
        with self.connect() as db:
            rows = db.execute('SELECT lane,generation,checkpoint FROM live_active_runs ORDER BY lane').fetchall()
        if rows:
            return [dict(lane=lane, generation=generation, run=json.loads(data) if data else None)
                    for lane, generation, data in rows]
        # Upgrade the existing unfinished game without changing its UUID or seed.
        legacy = self.load()
        slots = [dict(lane=i, generation=1 if legacy else 0,
                      run=(legacy if i == 0 else LiveRun().checkpoint()) if legacy else None) for i in range(3)]
        self.save_slots(slots)
        return slots

    def load_batch(self):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM live_batch_state WHERE id=1').fetchone()
        return json.loads(row[0]) if row else None

    def save_slots(self, slots, batch=None, events=()):
        with self.connect() as db:
            db.executemany('INSERT OR REPLACE INTO live_active_runs VALUES (?,?,?)',
                           [(s['lane'], s['generation'], json.dumps(s['run']) if s['run'] else None) for s in slots])
            if batch is not None:
                db.execute('INSERT OR REPLACE INTO live_batch_state VALUES(1,?)', (json.dumps(batch),))
            for event in events:
                db.execute('INSERT OR IGNORE INTO live_activity_outbox VALUES(?,?,?,?,?,?,?,0)',
                           tuple(event[k] for k in ('trigger_key','batch_id','milestone','participant_id','run_id','content_seq','created')))

    def pending_activities(self):
        with self.connect() as db:
            rows = db.execute('SELECT trigger_key,batch_id,milestone,created FROM live_activity_outbox WHERE delivered=0').fetchall()
        return [dict(zip(('trigger_key','batch_id','milestone','created'), row)) for row in rows]

    def activity_delivered(self, key):
        with self.connect() as db:
            db.execute('UPDATE live_activity_outbox SET delivered=1 WHERE trigger_key=?', (key,))

    def finish_lane(self, run, lane):
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO live_run_lanes VALUES (?,?)', (run.id, lane))
        # The base store deduplicates statistics and rewards by run UUID.
        self.finish(run)

    def _label_history(self, rows):
        if not rows:
            return []
        with self.connect() as db:
            lanes = dict(db.execute('SELECT run_id,lane FROM live_run_lanes WHERE run_id IN (%s)' %
                                   ','.join('?' for _ in rows), [r['id'] for r in rows]))
        return [dict(row, lane=lanes.get(row['id'], 0)) for row in rows]

    def history(self, page=1):
        result = super().history(page)
        return dict(result, history=self._label_history(result['history']))

    def summary(self, stats_range='all'):
        result = super().summary(stats_range)
        return dict(result, history=self._label_history(result.get('history', [])))
