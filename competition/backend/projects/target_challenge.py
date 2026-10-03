"""Registered, versioned standard-board targets; no room or tournament policy."""
from backend.human_play import engine
from ..errors import CompetitionError

VERSION = 'standard-target-v1'
PROJECT_REF = 'standard-target'
VARIANTS = dict(engine.VARIANTS)


def configuration(variant, target_kind, target_value):
    if variant not in VARIANTS or target_kind not in ('tile', 'board_sum'):
        raise CompetitionError('INVALID_CHALLENGE', 'Unknown board or target type.')
    if type(target_value) is not int:
        raise CompetitionError('INVALID_CHALLENGE', 'Target must be an integer.')
    if target_kind == 'tile':
        valid = 8 <= target_value <= 2**31 and not target_value & (target_value - 1)
    else:
        valid = 10 <= target_value <= 2**31 and target_value % 2 == 0
    if not valid:
        raise CompetitionError('INVALID_CHALLENGE', 'Use a power of two (8–2147483648), or an even board sum (10–2147483648).')
    return dict(project_ref=PROJECT_REF, rules_version=VERSION, variant=variant,
                target_kind=target_kind, target_value=target_value)


def reached(config, board):
    return (max(board) >= config['target_value'] if config['target_kind'] == 'tile'
            else sum(board) == config['target_value'])


def compare_best(yellow, white):
    if yellow is None and white is None:
        return 'draw'
    if yellow is None:
        return 'white'
    if white is None:
        return 'yellow'
    return 'yellow' if yellow < white else 'white' if white < yellow else 'draw'


def pool(config):
    """Only this server-owned registry can manufacture this frozen descriptor."""
    descriptor = dict(project_ref=PROJECT_REF, rules_version=VERSION,
                      display_name='Standard target challenge', view_kind='2048-board',
                      view_protocol='2048-board-v2', test_only=False)
    return [dict(key=PROJECT_REF, name=descriptor['display_name'], description='',
                 project_ref=PROJECT_REF, adapter_rules_version=VERSION, rules_version=VERSION,
                 adapter_snapshot={**descriptor, 'configuration': config})]
