"""Small batch-stable timing drift, independent of game/spawn randomness."""
from functools import lru_cache
import hashlib
import random


@lru_cache(maxsize=8)
def batch_means(batch_id):
    # Batch IDs are fresh UUID4s. Deriving the profile from that ID lets a worker
    # resume the same profile after reconnect/restart without wire/checkpoint fields.
    rng = random.Random('live-pacing-v1:' + batch_id)
    base = rng.uniform(.002, .003)
    gap1, gap2 = rng.uniform(.002, .0025), rng.uniform(.002, .0025)
    means = [base, base + gap1, base + gap1 + gap2]
    rng.shuffle(means)
    return tuple(means)


def step_drift(batch_id, lane, sequence):
    if not batch_id:
        return 0.0
    # Stateless per-step jitter: interrupted/uncommitted steps keep their delay.
    key = f'live-pacing-v1:{batch_id}:{lane}:{sequence}'.encode('utf-8')
    value = int.from_bytes(hashlib.blake2s(key, digest_size=8).digest(), 'big')
    jitter = (value / (2**64 - 1) * 2 - 1) * .001
    return batch_means(batch_id)[lane] + jitter
