"""Versioned viewer-only deltas. Authoritative match/replay formats are unchanged."""
from uuid import uuid4

def changes(previous, current, exclude=()):
    return {
        'set': {k: v for k, v in current.items() if k not in exclude
                and (k not in previous or previous[k] != v)},
        'unset': [k for k in previous if k not in current and k not in exclude],
    }


def pack_view(view):
    if not view:
        return view
    frames = view.get('frames') or []
    # The final frame already contains the current board in ordinary updates.
    if frames and frames[-1].get('sequence') == view.get('sequence') and frames[-1].get('payload') == view.get('payload'):
        return {**{k: v for k, v in view.items() if k != 'payload'}, 'payload_from_last_frame': True}
    return view


class MatchDeltaEncoder:
    def __init__(self):
        self.epoch = uuid4().hex
        self.sequence = 0
        self.previous = None

    @staticmethod
    def identity(snapshot):
        match = snapshot.get('match') or {}
        return tuple(match.get(k) for k in ('match_public_key', 'generation', 'current_game', 'phase'))

    def full(self, snapshot):
        # Bootstrap callers already supply a bounded recovery snapshot. A normal
        # phase transition must retain EVERY new frame, including the final batch.
        return {**snapshot, 'stream_epoch': self.epoch, 'stream_sequence': self.sequence}

    def encode(self, snapshot):
        previous = self.previous
        base = self.sequence
        self.sequence += 1
        self.previous = snapshot
        if not previous or not snapshot.get('match') or not previous.get('match') or self.identity(previous) != self.identity(snapshot):
            return self.full(snapshot)
        old_match, match = previous['match'], snapshot['match']
        old_views, views = old_match.get('project_public_views') or {}, match.get('project_public_views') or {}
        return {
            'type': 'match_delta', 'room_id': snapshot['room_id'], 'protocol': snapshot['protocol'],
            'stream_epoch': self.epoch, 'base_sequence': base, 'stream_sequence': self.sequence,
            'fields': changes(previous, snapshot, ('match',)),
            'match': changes(old_match, match, ('project_public_views',)),
            'views': {'set': {side: pack_view(view) for side, view in views.items()
                              if side not in old_views or old_views[side] != view},
                      'unset': [side for side in old_views if side not in views]},
        }
