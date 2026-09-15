import base64
import secrets
import struct
import time
import uuid
import zlib

from backend.gamer_ranked.rules import initial_board, simulate_move, random_spawn, legal_moves
from backend.gamer_ranked.prng import Xoshiro128StarStar

DIRECTIONS = ('up', 'right', 'down', 'left')
STEP = struct.Struct('<IIB')  # sequence, elapsed delta milliseconds, RPL1 move byte
RECORD = struct.Struct('<IB')
MAX_STEPS = 100_000


def uleb(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return result


class LiveRun:
    def __init__(self, seed=None, run_id=None, started=None):
        self.seed = seed or ''.join(f'{secrets.randbelow(0xffffffff) + 1:08x}' for _ in range(4))
        self.id = str(uuid.UUID(run_id)) if run_id else str(uuid.uuid4())
        self.started = float(started or time.time())
        self.board, self.initial, self.rng = initial_board(self.seed)
        self.score = self.elapsed = self.seq = 0
        self.records = bytearray()
        self.nodes = {}
        self.source = 'AI'
        self.ended = None
        self.restart_at = None

    def make_step(self, direction, delta_ms):
        moved, _ = simulate_move(self.board, direction)
        if moved == self.board:
            raise ValueError('invalid_move')
        # apply() owns the RNG; peek using a copy of the four-word state.
        rng = Xoshiro128StarStar(self.rng.state.copy())
        index, exponent = random_spawn(moved, rng)
        return STEP.pack(self.seq + 1, min(3_600_000, max(0, int(delta_ms))), DIRECTIONS.index(direction) | (index << 2) | ((exponent - 1) << 6))

    def apply(self, packet):
        seq, delta, change = STEP.unpack(packet)
        if self.ended or seq != self.seq + 1 or seq > MAX_STEPS or delta > 3_600_000 or change > 127:
            raise ValueError('invalid_step')
        moved, score = simulate_move(self.board, DIRECTIONS[change & 3])
        if moved == self.board:
            raise ValueError('invalid_move')
        rng = Xoshiro128StarStar(self.rng.state.copy())
        index, exponent = random_spawn(moved, rng)
        if (index, exponent) != ((change >> 2) & 15, (change >> 6) + 1):
            raise ValueError('invalid_spawn')
        moved[index] = 2 ** exponent
        self.rng = rng
        self.board = moved
        self.score += score
        self.elapsed += delta
        self.seq = seq
        self.records.extend(RECORD.pack(delta, change))
        maximum = max(moved)
        for exponent in range(9, 32):
            value = 2 ** exponent
            if value <= maximum:
                self.nodes.setdefault(str(value), self.elapsed)

    def end(self, restart_at=None):
        if legal_moves(self.board):
            raise ValueError('not_game_over')
        self.ended = self.ended or time.time()
        self.restart_at = restart_at or self.ended + 15

    def snapshot(self):
        return dict(run_id=self.id, board=self.board, score=self.score, seq=self.seq,
                    elapsed_ms=self.elapsed, nodes=self.nodes, source=self.source,
                    started_at=self.started, ended_at=self.ended, restart_at=self.restart_at)

    def checkpoint(self):
        return dict(run_id=self.id, seed=self.seed, started=self.started, source=self.source,
                    ended=self.ended, restart_at=self.restart_at,
                    records=base64.b64encode(self.records).decode('ascii'))

    @classmethod
    def restore(cls, data):
        encoded = data.get('records', '')
        if not isinstance(encoded, str) or len(encoded) > 700_000:
            raise ValueError('record_too_large')
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) % RECORD.size or len(raw) > MAX_STEPS * RECORD.size:
            raise ValueError('invalid_record')
        run = cls(data['seed'], data['run_id'], data['started'])
        for delta, change in RECORD.iter_unpack(raw):
            run.apply(STEP.pack(run.seq + 1, delta, change))
        run.source = str(data.get('source') or 'AI')[:80]
        if data.get('ended'):
            run.end()
            run.ended = float(data['ended'])
            run.restart_at = float(data.get('restart_at') or run.ended + 15)
        return run

    def replay(self):
        raw = bytearray(b'RPL1\x44\x00\x02')
        raw.extend(index | (bit << 4) for index, bit in self.initial)
        raw.extend(b'\x83\x02\x04pow2')
        for delta, change in RECORD.iter_unpack(self.records):
            raw.append(change)
            raw.extend(uleb(delta))
        raw.append(132)
        raw.extend(struct.pack('<I', zlib.crc32(raw)))
        return 'REPLAY_v1RPL_B64_' + base64.b64encode(raw).decode('ascii')
