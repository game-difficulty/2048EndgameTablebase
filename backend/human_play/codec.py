"""Lossless transport/storage transforms; validation always sees the original 5-byte events."""
import zlib

MAX_EVENTS_BYTES = 1000000
MAX_REPLAY_BYTES = MAX_EVENTS_BYTES + 65544


def to_planes(raw):
    if len(raw) % 5 or len(raw) > MAX_EVENTS_BYTES:
        raise ValueError('invalid_binary_length')
    return b''.join(raw[index::5] for index in range(5))


def from_planes(raw):
    if len(raw) % 5 or len(raw) > MAX_EVENTS_BYTES:
        raise ValueError('invalid_binary_length')
    count = len(raw) // 5
    result = bytearray(len(raw))
    for index in range(5):
        result[index::5] = raw[index * count:(index + 1) * count]
    return bytes(result)


def gunzip_limited(data, limit=MAX_EVENTS_BYTES):
    decoder = zlib.decompressobj(31)
    try:
        raw = decoder.decompress(data, limit + 1)
        if len(raw) > limit or decoder.unconsumed_tail:
            raise ValueError('record_too_large')
        if not decoder.eof or decoder.unused_data:
            raise ValueError('invalid_gzip')
        return raw
    except zlib.error as exc:
        raise ValueError('invalid_gzip') from exc


def compact_receipt(receipt):
    flags = {'active': 0, 'pending_archive': 1, 'sealed': 2}.get(receipt.get('status', 'active'), 0)
    flags |= 4 if receipt['monitored'] else 0
    flags |= 8 if receipt['eligibility'] != 'eligible' else 0
    return [2, receipt['seq'], receipt['epoch'], flags, int(receipt['server_time'] * 1000), receipt['permit']]


def decode_upload(data, encoding, layout):
    raw = gunzip_limited(data) if encoding == 'gzip' else data
    return from_planes(raw) if layout == 'planes5' else raw
