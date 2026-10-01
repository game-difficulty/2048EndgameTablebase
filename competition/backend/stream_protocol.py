"""Private recovery deltas. Never forwarded to viewers."""
from copy import deepcopy
from .errors import CompetitionError

PROTOCOL = 'project-stream-v2'

def expand_checkpoint(checkpoint, previous, accepted):
    if 'delta_base' not in checkpoint:
        return checkpoint
    if checkpoint['delta_base'] != accepted or not previous:
        raise CompetitionError('CHECKPOINT_BASE_MISMATCH', '重新同步存档。', 409)
    result = deepcopy(checkpoint)
    result.pop('delta_base')
    patches = result.pop('lists', {})
    for key in ['metric_history', 'undo', 'lookBackHistory']:
        source = previous if key == 'metric_history' else previous.get('state', {})
        target = result if key == 'metric_history' else result['state']
        old = source.get(key, [])
        patch = patches.get(key, {})
        keep, append = patch.get('keep'), patch.get('append')
        if type(keep) is not int or not 0 <= keep <= len(old) or not isinstance(append, list):
            raise CompetitionError('INVALID_CLIENT_STATE', '恢复数据增量无效。', 400)
        target[key] = deepcopy(old[:keep]) + append
    return result
