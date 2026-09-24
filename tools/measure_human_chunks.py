"""Measure identical batches in old/new SQLite structures, without touching project databases."""
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import tempfile
import uuid


def measure(compact, batch_moves):
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / 'chunks.db'
        db = sqlite3.connect(path)
        db.execute('PRAGMA page_size=4096')
        db.execute('''CREATE TABLE human_chunks(run_id TEXT NOT NULL, start INTEGER NOT NULL,
            count INTEGER NOT NULL, digest %s NOT NULL, previous_hash %s NOT NULL,
            data BLOB NOT NULL, received REAL NOT NULL, PRIMARY KEY(run_id,start)) %s''' %
            ('BLOB' if compact else 'TEXT', 'BLOB' if compact else 'TEXT', 'WITHOUT ROWID' if compact else ''))
        db.commit(); base = path.stat().st_size
        rng = random.Random(20260924)
        for game in range(100):
            run_id = str(uuid.UUID(int=rng.getrandbits(128)))
            for batch in range(100):
                data = rng.randbytes(batch_moves * 5)
                digest = hashlib.sha256(data).digest(); prefix = rng.randbytes(32)
                db.execute('INSERT INTO human_chunks VALUES(?,?,?,?,?,?,?)', (run_id, batch * batch_moves, batch_moves,
                    digest if compact else digest.hex(), prefix if compact else prefix.hex(), data, 1790170000.123 + batch * 10))
        db.commit(); db.close()
        return (path.stat().st_size - base) / 10000


def main():
    rows = []
    for moves in (1, 20, 32):
        before, after = measure(False, moves), measure(True, moves)
        row = dict(batch_moves=moves, action_bytes=moves * 5,
                   before_bytes_per_batch=before, after_bytes_per_batch=after,
                   reduction_percent=round((1 - after / before) * 100, 2))
        rows.append(row)
        print(json.dumps(row))
    output = Path('output/human-chunks.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
