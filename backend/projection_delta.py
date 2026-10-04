"""Ordered, connection-local competition relay deltas (not authoritative state)."""
from uuid import uuid4


def diff(old, new, exclude=()):
    return {'set': {k: v for k, v in new.items() if k not in exclude and (k not in old or old[k] != v)},
            'unset': [k for k in old if k not in new and k not in exclude]}


def apply(old, patch):
    result = {**old, **patch['set']}
    for key in patch['unset']:
        result.pop(key, None)
    return result


def pack_view(view):
    if not view:
        return view
    frames = view.get('frames') or []
    if frames and frames[-1].get('sequence') == view.get('sequence') and frames[-1].get('payload') == view.get('payload'):
        return {**{k: v for k, v in view.items() if k != 'payload'}, 'payload_from_last_frame': True}
    return view


def unpack_view(view):
    if not view or not view.get('payload_from_last_frame'):
        return view
    result = dict(view)
    del result['payload_from_last_frame']
    result['payload'] = result['frames'][-1]['payload']
    return result


class ProjectionEncoder:
    def __init__(self):
        self.epoch, self.sequence, self.previous = uuid4().hex, 0, None

    def encode(self, projection):
        previous = self.previous
        self.previous = projection
        base = self.sequence
        self.sequence += 1
        header = {'stream_epoch': self.epoch, 'stream_sequence': self.sequence}
        identity = lambda p: tuple(p.get(k) for k in ('match_public_key', 'generation', 'current_game', 'phase'))
        if previous is None or identity(previous) != identity(projection):
            return {'type': 'projection', 'projection': projection, **header}
        old, new = previous.get('project_public_views') or {}, projection.get('project_public_views') or {}
        return {'type': 'projection_delta', **header, 'base_sequence': base,
                'fields': diff(previous, projection, ('project_public_views',)),
                'views': {'set': {s: pack_view(v) for s, v in new.items() if s not in old or old[s] != v},
                          'unset': [s for s in old if s not in new]}}


class ProjectionDecoder:
    def __init__(self):
        self.epoch, self.sequence, self.previous = None, None, None

    def decode(self, message):
        if message.get('type') == 'projection':
            projection = message['projection']
        elif message.get('type') == 'projection_delta':
            if (self.previous is None or self.epoch is None or message.get('stream_epoch') != self.epoch
                    or message.get('base_sequence') != self.sequence or message.get('stream_sequence') != self.sequence + 1):
                raise ValueError('projection_delta_gap')
            projection = apply(self.previous, message['fields'])
            views = apply(self.previous.get('project_public_views') or {}, message['views'])
            projection['project_public_views'] = {s: unpack_view(v) for s, v in views.items()}
        else:
            return None
        self.epoch, self.sequence = message.get('stream_epoch'), message.get('stream_sequence')
        self.previous = projection
        return projection
