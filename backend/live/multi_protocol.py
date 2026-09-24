"""Multi-run wire formats. Gameplay is still validated by LiveRun."""
import struct
from .protocol import STEP, uleb

PROTOCOL = 'classic-multi-v1'
PUBLISH_PROTOCOL = 'classic-multi-publish-v2'
UPLINK = struct.Struct('<BIIIB')
BATCH = struct.Struct('<BI')
MAX_RECORDS = 64
MAX_DELTA = 3_600_000


def encode_step(lane, packet):
    _, delta, move = STEP.unpack(packet)
    if lane not in range(3) or move > 127 or delta > MAX_DELTA:
        raise ValueError('invalid_step')
    control = lane | ((move & 3) << 2) | (((move >> 2) & 15) << 4)
    return bytes([control]) + uleb((delta << 1) | (move >> 6))


def encode_source(lane, source_id):
    if lane not in range(3) or not 0 <= source_id <= 65535:
        raise ValueError('invalid_source')
    return bytes([3, lane]) + uleb(source_id)


def pack_batch(first_seq, records):
    if not 0 < first_seq <= 0xffffffff or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError('invalid_batch')
    return BATCH.pack(0x21, first_seq) + b''.join(records)


def unpack_uplink(raw):
    if not raw or len(raw) % UPLINK.size or len(raw) > MAX_RECORDS * UPLINK.size:
        raise ValueError('invalid_packet')
    packets = []
    for lane, generation, seq, delta, move in UPLINK.iter_unpack(raw):
        if lane >= 3 or generation == 0:
            raise ValueError('invalid_lane')
        packets.append((lane, generation, STEP.pack(seq, delta, move)))
    return packets
