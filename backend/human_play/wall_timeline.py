"""Optional HPR header extension; legacy event bytes and hashes remain unchanged."""
from __future__ import annotations

import json


def validate(value, seq):
    if value is None:
        return None
    if isinstance(value, str):
        if len(value) > 4096:
            raise ValueError('invalid_wall_timeline')
        value = json.loads(value)
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('invalid_wall_timeline')
    anchors = value.get('anchors')
    if not isinstance(anchors, list) or not 1 <= len(anchors) <= 64:
        raise ValueError('invalid_wall_timeline')
    previous = 0
    for pair in anchors:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('invalid_wall_timeline')
        step, stamp = pair
        if type(step) is not int or not previous < step <= seq or not valid_stamp(stamp):
            raise ValueError('invalid_wall_timeline')
        previous = step
    start = value.get('started_at_ms')
    end = value.get('truncated_at_seq')
    if start is not None and not valid_stamp(start):
        raise ValueError('invalid_wall_timeline')
    if end is not None and (type(end) is not int or not previous < end <= seq + 1):
        raise ValueError('invalid_wall_timeline')
    return {'version': 1, 'anchors': anchors, 'started_at_ms': start, 'truncated_at_seq': end}


def valid_stamp(value):
    return type(value) is int and 1394323200000 <= value <= 8640000000000000


def move_instants(timeline, deltas):
    if not timeline:
        return [None for _ in deltas]
    anchors = dict(timeline['anchors'])
    limit = timeline.get('truncated_at_seq')
    stamp = None
    result = []
    for seq, delta in enumerate(deltas, 1):
        stamp = anchors.get(seq, None if stamp is None else stamp + delta)
        result.append(stamp if not limit or seq < limit else None)
    return result
