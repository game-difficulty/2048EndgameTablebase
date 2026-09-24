"""Deterministic synthetic response sizing; never accesses a database or server."""
import gzip
import json
import random
import uuid


def main():
    rng = random.Random(20260924)
    entries = []
    for index in range(100):
        entries.append(dict(id=str(uuid.UUID(int=rng.getrandbits(128))), user_id=index + 1,
            variant='4x4', score=480000 - index * 1000, max_tile=32768, moves=10000 + index,
            elapsed=2500000 + index * 50000,
            nodes={str(2 ** exponent): {'seq': 100 * exponent + index, 'elapsed': rng.randint(1000, 2500000)}
                   for exponent in range(5, 16)}, ended_at=1790170000 + index,
            reason='game_over', eligibility='eligible', rank=index + 1, display_name=f'Player {index + 1}'))
    def size(rows):
        data = json.dumps(dict(entries=rows, variant='4x4', period='all'), separators=(',', ':'), ensure_ascii=False).encode()
        return dict(json_bytes=len(data), gzip_bytes=len(gzip.compress(data, compresslevel=4, mtime=0)))
    overview = [{key: value for key, value in row.items()
                 if key in {'id', 'user_id', 'score', 'max_tile', 'rank', 'display_name'}} for row in entries]
    print(json.dumps({'synthetic': True, 'before_100_with_nodes': size(entries),
                      'overview_10': size(overview[:10]), 'full_100': size(overview)}, indent=2))


if __name__ == '__main__':
    main()
