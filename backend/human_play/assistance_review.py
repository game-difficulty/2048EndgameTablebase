"""Review signals only: a match never disqualifies an otherwise valid run."""
from bisect import bisect_left, bisect_right
import json
import logging
import time

from backend.assistance_evidence import candidates
from backend.lookup_mask import MASK_VERSION, play_lookup_key
from .wall_timeline import move_instants

logger = logging.getLogger(__name__)
TOLERANCE_MS = 2000


def init_schema(db):
    db.execute('''CREATE TABLE IF NOT EXISTS human_assistance_reviews (
        id INTEGER PRIMARY KEY, run_id TEXT NOT NULL UNIQUE REFERENCES human_runs(id),
        user_id INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
        requested_at REAL NOT NULL, updated_at REAL NOT NULL, details TEXT NOT NULL,
        review_note TEXT NOT NULL DEFAULT '', operator_id INTEGER
    )''')


class Matcher:
    def __init__(self, run, raw, timeline):
        from .engine import EVENT
        self.run = run
        self.raw = raw
        self.times = move_instants(timeline, (delta for _, delta in EVENT.iter_unpack(raw))) if timeline else []
        self.previous = None
        self.hits = []
        self.seen = set()
        self.start = timeline.get('started_at_ms') if timeline else None
        low, high = self.start, self.start
        for stamp in self.times:
            if stamp is not None:
                low = stamp if low is None else min(low, stamp)
                high = stamp if high is None else max(high, stamp)
        self.events = candidates(run['user_id'], low - TOLERANCE_MS, high + TOLERANCE_MS) if low is not None else []
        if not self.events:
            self.times = []
        self.event_times = [item['queried_at_ms'] for item in self.events]

    def observe(self, state):
        if not self.events:
            return
        seq = state['seq']
        if self.previous is not None and self.events:
            low = self.start if seq == 1 else self.times[seq - 2]
            high = self.times[seq - 1]
            # Clock reversals and incomplete legacy coverage cannot prove an interval.
            if low is not None and high is not None and low <= high:
                left = bisect_left(self.event_times, low - TOLERANCE_MS)
                right = bisect_right(self.event_times, high + TOLERANCE_MS)
                keys = {}
                for event in self.events[left:right]:
                    if event['variant'] != self.run['variant'] or (event['id'], seq) in self.seen:
                        continue
                    if event['kind'] == 'ai':
                        key = ','.join(str(tile.bit_length() - 1 if tile else 0) for tile in self.previous)
                    elif event['mask_version'] == MASK_VERSION:
                        pattern = event['full_pattern']
                        if pattern not in keys:
                            try:
                                keys[pattern] = play_lookup_key(self.previous, self.run['variant'], pattern)
                            except (ValueError, KeyError):
                                keys[pattern] = None
                        key = keys[pattern]
                    else:
                        continue
                    if key != event['board_key']:
                        continue
                    self.seen.add((event['id'], seq))
                    self.hits.append({**event, 'move_seq': seq, 'play_board': list(self.previous),
                        'appeared_at_ms': low, 'moved_at_ms': high,
                        'actual_direction': ('up', 'right', 'down', 'left')[self.raw[(seq - 1) * 5] & 3],
                        'time_relation': 'during' if low <= event['queried_at_ms'] <= high else 'boundary'})
        self.previous = list(state['board'])

    def persist(self, db):
        if not self.hits:
            return
        now = time.time()
        db.execute('''INSERT OR IGNORE INTO human_assistance_reviews
            (run_id,user_id,requested_at,updated_at,details) VALUES(?,?,?,?,?)''',
            (self.run['id'], self.run['user_id'], now, now, json.dumps({'hits': self.hits}, separators=(',', ':'))))


def decide(review_id, operator_id, confirmed, note):
    from .store import database
    from .service import RunError
    with database() as db:
        row = db.execute('SELECT * FROM human_assistance_reviews WHERE id=?', (review_id,)).fetchone()
        if not row:
            raise RunError('review_not_found', 404)
        if row['status'] != 'pending':
            raise RunError('review_already_decided', 409)
        db.execute('''UPDATE human_assistance_reviews SET status=?,review_note=?,operator_id=?,updated_at=?
            WHERE id=? AND status='pending' ''',
            ('confirmed' if confirmed else 'dismissed', note, operator_id, time.time(), review_id))
    return {'id': review_id, 'confirmed': confirmed}
