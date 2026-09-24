"""Isolated, reproducible capacity measurements. Never opens a live database.

Run from the repository root: python -m tools.measure_human_resources
Compression projections use synthetic event distributions, not claimed human data.
Verification timings use legal seeded games and are specific to the local CPU.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sqlite3
import statistics
import tempfile
import time
import uuid
from unittest.mock import patch

from backend.gamer_ranked.prng import Xoshiro128StarStar
from backend.human_play import engine, service
from backend.human_play.store import init_db, database

SEED = '00000001000000020000000300000004'


def metadata(index, state=None):
    rid = str(uuid.UUID(int=index + 1))
    return dict(id=rid, user_id=index // 4 + 1, browser=str(uuid.UUID(int=index + 10)),
                variant='4x4', request_id=str(uuid.UUID(int=index + 10000)), seed=SEED,
                threshold=360000, created=1750000000.123, writer=str(uuid.UUID(int=123456)),
                reason='restarted', state=json.dumps(state or engine.initial(rid, '4x4', SEED)))


def insert(db, item, archive=None, status='sealed'):
    db.execute('''INSERT INTO human_runs
        (id,user_id,browser,variant,request_id,seed,threshold,created,writer,reason,state,archive,status,ended)
        VALUES (:id,:user_id,:browser,:variant,:request_id,:seed,:threshold,:created,:writer,:reason,:state,:archive,:status,:created)''',
               {**item, 'archive': archive, 'status': status})


def measured_database(fill):
    with tempfile.TemporaryDirectory(prefix='human-capacity-') as root:
        path = Path(root) / 'measure.sqlite3'
        with patch.dict(os.environ, {'HUMAN_PLAY_DB': str(path)}):
            init_db()
            base = path.stat().st_size
            with database() as db:
                fill(db)
            with database() as db:
                db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            return path.stat().st_size - base


def synthetic(count, timing):
    rng = random.Random(20260923)
    return b''.join(engine.EVENT.pack(rng.randrange(64) | (64 if rng.random() < .1 else 0),
                        rng.randint(*timing) if i else 0) for i in range(count))


def legal_game(index):
    seed = f'{index + 1:08x}000000020000000300000004'
    state = engine.initial(f'bench-{index}', '4x4', seed)
    raw = bytearray()
    weights = [16, 15, 14, 13, 9, 10, 11, 12, 8, 7, 6, 5, 1, 2, 3, 4]
    for i in range(10000):
        choices = []
        for direction in range(4):
            board, gained = engine.move(state['board'], 4, 4, direction)
            if board == state['board']:
                continue
            logs = [math.log2(v) if v else 0 for v in board]
            utility = sum(a * b for a, b in zip(logs, weights)) + board.count(0) * 40 + gained * .005
            choices.append((utility, direction, board))
        if not choices:
            break
        _, direction, board = max(choices)
        rng = Xoshiro128StarStar(list(state['rng']))
        pos, value = engine.spawn(board, rng)
        data = engine.EVENT.pack(direction | pos << 2 | (64 if value == 4 else 0), 150 + (i * 137 % 2850) if i else 0)
        raw.extend(data)
        state = engine.advance(state, '4x4', data)
    return seed, bytes(raw), state


def main():
    result = {'platform': platform.platform(), 'python': platform.python_version(), 'sqlite': sqlite3.sqlite_version,
              'note': 'Local synthetic sizing/benchmark; not production concurrency capacity.'}
    games = [legal_game(i) for i in range(12)]
    seed, legal_raw, final = max(games, key=lambda item: len(item[1]))
    game_index = next(i for i, game in enumerate(games) if game[0] == seed)
    first = engine.initial(f'bench-{game_index}', '4x4', seed)
    sizes = [len(raw) // 5 for _, raw, _ in games]
    samples = []
    for _ in range(25):
        started = time.perf_counter(); engine.advance(first, '4x4', legal_raw); samples.append(time.perf_counter() - started)
    result['verification'] = {'legal_game_lengths': sizes, 'longest_moves': len(legal_raw) // 5,
        'median_ms': round(statistics.median(samples) * 1000, 3),
        'moves_per_second': round(len(legal_raw) / 5 / statistics.median(samples)),
        'compressed_legal_bytes_per_move': round(len(gzip.compress(legal_raw, compresslevel=6, mtime=0)) / (len(legal_raw) / 5), 3)}
    result['compression'] = []
    for timing in [(100, 1000), (150, 3000), (0, 0xffffffff)]:
        for count in (1000, 10000, 100000, 200000):
            raw = synthetic(count, timing)
            archive = gzip.compress(engine.replay_bytes(metadata(0), raw), compresslevel=6, mtime=0)
            result['compression'].append({'moves': count, 'delta_range_ms': timing, 'raw_bytes': len(raw),
                'gzip_file_bytes': len(archive), 'bytes_per_move': round(len(archive) / count, 3)})
    result['storage'] = {}
    for label, state, status in [('initial_binding', None, 'active'), ('final_metadata', final, 'sealed')]:
        def fill(db):
            for i in range(2000):
                insert(db, metadata(i, state), status=status)
        result['storage'][label + '_bytes_per_run'] = measured_database(fill) / 2000
    for count in (1000, 10000, 100000):
        raw = synthetic(count, (150, 3000))
        def fill(db):
            for i in range(100):
                item = metadata(i, final)
                insert(db, item, gzip.compress(engine.replay_bytes(item, raw), compresslevel=6, mtime=0))
        result['storage'][f'archived_{count}_moves_bytes_per_run'] = measured_database(fill) / 100
    for count in (1, 20, 32):
        data = synthetic(count, (150, 3000))
        def fill(db):
            item = metadata(0, final); insert(db, item, status='active')
            for i in range(2000):
                db.execute('INSERT INTO human_chunks VALUES(?,?,?,?,?,?,?)',
                           (item['id'], i * count, count, hashlib.sha256(data).digest(), bytes.fromhex('a' * 64), data, 1750000000.123))
        result['storage'][f'chunk_{count}_moves_bytes_per_chunk'] = measured_database(fill) / 2000
    # Real committed transactions in a disposable WAL/FULL database; no HTTP/auth included.
    with tempfile.TemporaryDirectory(prefix='human-write-bench-') as root:
        with patch.dict(os.environ, {'HUMAN_PLAY_DB': str(Path(root) / 'write.sqlite3')}):
            init_db(); item = metadata(0)
            with database() as db:
                insert(db, item, status='active')
                db.execute('UPDATE human_runs SET monitored=1,permit_until=?', (time.time() + 60,))
            times = []
            for _ in range(200):
                start = time.perf_counter()
                service.online_check(item['user_id'], item['browser'], item['id'], item['writer'], 1)
                times.append((time.perf_counter() - start) * 1000)
            result['heartbeat_transaction_ms'] = {'median': round(statistics.median(times), 3),
                'p95': round(sorted(times)[189], 3), 'max': round(max(times), 3)}
    result['capacity'] = []
    for budget in (12416847872, 6 * 1024**3):
        for count in (1000, 10000, 100000):
            per_run = result['storage'][f'archived_{count}_moves_bytes_per_run']
            runs = int(budget / per_run)
            result['capacity'].append({'budget_bytes': budget, 'moves_per_run': count, 'runs': runs,
                                       'total_moves': runs * count, 'ten_thousand_moves': runs * count / 10000})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
