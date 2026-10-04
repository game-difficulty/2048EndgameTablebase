import json

from backend.stream_snapshots import compact_public_view


def test_recovery_window_preserves_short_history_and_does_not_mutate_cache():
    view = {'sequence': 100, 'payload': {'score': 500}, 'frame_start': 1,
            'frames': [{'sequence': i, 'payload': {'score': i}} for i in range(1, 101)]}
    compact = compact_public_view(view)
    assert compact['frame_start'] == 93
    assert [f['sequence'] for f in compact['frames']] == list(range(93, 101))
    assert compact['payload'] == view['payload']
    assert len(view['frames']) == 100


def test_recovery_history_has_a_byte_limit_even_with_large_special_tile_payloads():
    view = {'sequence': 8, 'payload': {'score': 500}, 'frames': [
        {'sequence': i, 'payload': {'label': '曹'*6000}} for i in range(1, 9)]}
    compact = compact_public_view(view)
    assert len(json.dumps(compact['frames'], ensure_ascii=False, separators=(',', ':')).encode()) <= 64*1024
    assert compact['frames'][-1]['sequence'] == 8
    assert compact['payload'] == view['payload']
