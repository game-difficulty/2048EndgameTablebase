"""Bounded public recovery windows; no game rules or private checkpoints."""
import json


def compact_public_view(view, *, max_frames=8, max_bytes=64 * 1024):
    if not view:
        return view
    frames, used = [], 2
    for frame in reversed((view.get('frames') or [])[-max_frames:]):
        size = len(json.dumps(frame, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) + 1
        if used + size > max_bytes:
            break
        frames.append(frame)
        used += size
    frames.reverse()
    # A short interruption can still play every retained step. New viewers
    # immediately use payload; genuinely missing history rebases at this floor.
    return {**view, 'frames': frames,
            'frame_start': frames[0]['sequence'] if frames else view.get('sequence', 0)}
