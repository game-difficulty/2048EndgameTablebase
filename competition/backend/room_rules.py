"""Versioned room rules independent of events, identities and game adapters."""
from collections import Counter
from copy import deepcopy

from .errors import CompetitionError

VERSION = 'room-rules-v1'
MAX_GAMES = 15
MAX_TEAM_SIZE = 16
PRESETS = {
    'bo3': [{'actor': 'first', 'bans': 1, 'picks': 1}, {'actor': 'second', 'bans': 1, 'picks': 1}],
    'bo5': [{'actor': 'first', 'bans': 1, 'picks': 1}, {'actor': 'second', 'bans': 1, 'picks': 2}, {'actor': 'first', 'bans': 0, 'picks': 1}],
    'bo7': [{'actor': 'first', 'bans': 1, 'picks': 1}, {'actor': 'second', 'bans': 2, 'picks': 2}, {'actor': 'first', 'bans': 1, 'picks': 2}, {'actor': 'second', 'bans': 0, 'picks': 1}],
}


def invalid(message):
    raise CompetitionError('INVALID_ROOM_RULES', message)


def normalize_rules(raw=None, *, pool_size=32, team_clock_ms=1800000, draft_seconds=60, lineup_seconds=180):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        invalid('比赛规则必须是对象。')
    allowed = {'preset', 'team_size', 'series_mode', 'lineup_policy', 'final_selection', 'steps', 'team_clock_seconds', 'draft_seconds', 'lineup_seconds'}
    if set(raw) - allowed:
        invalid('存在不支持的比赛规则字段。')
    preset = raw.get('preset', 'bo3')
    if not isinstance(preset, str) or preset not in {*PRESETS, 'custom'}:
        invalid('请选择有效的 BP 模板。')
    def integer(key, default, low, high):
        value = raw.get(key, default)
        if type(value) is not int or not low <= value <= high:
            invalid(f'{key} 必须是 {low} 至 {high} 的整数。')
        return value
    team_size = integer('team_size', 3, 1, MAX_TEAM_SIZE)
    selection = raw.get('final_selection', 'blind' if preset == 'bo3' else 'random')
    mode = raw.get('series_mode', 'all' if preset == 'bo3' else 'best_of')
    lineup = raw.get('lineup_policy', 'unique' if preset == 'bo3' else 'everyone')
    if selection not in ('blind', 'random') or mode not in ('all', 'best_of') or lineup not in ('unique', 'everyone', 'balanced', 'free'):
        invalid('结束方式、出场规则或抽签方式无效。')
    steps = deepcopy(raw.get('steps')) if preset == 'custom' else deepcopy(PRESETS[preset])
    if preset != 'custom' and 'steps' in raw:
        invalid('编辑 BP 步骤时请选择自定义模板。')
    if not isinstance(steps, list) or not 1 <= len(steps) <= 24:
        invalid('自定义 BP 须包含 1 至 24 个步骤。')
    for step in steps:
        if not isinstance(step, dict) or set(step) != {'actor', 'bans', 'picks'}:
            invalid('每个 BP 步骤须包含行动方、禁用数量和选择数量。')
        if step['actor'] not in ('first', 'second'):
            invalid('BP 行动方必须为先手或后手。')
        if any(type(step[k]) is not int or not 0 <= step[k] <= MAX_GAMES for k in ('bans', 'picks')) or step['bans'] + step['picks'] == 0:
            invalid('BP 步骤必须包含有效的选禁数量。')
    count = sum(s['picks'] for s in steps) + 1
    minimum = count + sum(s['bans'] for s in steps)
    if not 1 <= count <= MAX_GAMES or count % 2 != 1:
        invalid('总对局数须为 1 至 15 的奇数（包含最后抽签一局）。')
    if minimum > pool_size:
        invalid(f'当前规则至少需要 {minimum} 个项目，项目池只有 {pool_size} 个。')
    if lineup == 'unique' and count > team_size:
        invalid('每人最多出场一次时，每队人数不能少于对局数。')
    return dict(version=VERSION, preset=preset, team_size=team_size, game_count=count,
                game_keys=[chr(65+i) for i in range(count)], steps=steps, minimum_pool_size=minimum,
                series_mode=mode, wins_required=count//2+1 if mode=='best_of' else None,
                lineup_policy=lineup, final_selection=selection,
                team_clock_seconds=integer('team_clock_seconds', {'bo5': 3600, 'bo7': 4800}.get(preset, team_clock_ms//1000), 30, 86400),
                draft_seconds=integer('draft_seconds', draft_seconds, 5, 3600),
                lineup_seconds=integer('lineup_seconds', lineup_seconds, 5, 3600))


def validate_lineup(rules, assignments):
    if not isinstance(assignments, dict) or set(assignments) != set(rules['game_keys']):
        raise CompetitionError('INVALID_LINEUP', '请为每一局安排一名选手。')
    values = list(assignments.values())
    if any(type(v) is not int or not 1 <= v <= rules['team_size'] for v in values):
        raise CompetitionError('INVALID_LINEUP', '出战席位不在本房间队伍范围内。')
    counts = Counter(values)
    if rules['lineup_policy'] == 'unique' and len(counts) != len(values):
        raise CompetitionError('INVALID_LINEUP', '本房间要求每名选手最多出场一次。')
    if rules['lineup_policy'] == 'everyone' and len(counts) != min(rules['team_size'], len(values)):
        raise CompetitionError('INVALID_LINEUP', '本房间要求尽可能全员出场：局数足够时每人至少一局，否则每局安排不同选手。')
    if rules['lineup_policy'] == 'balanced':
        totals = [counts[i] for i in range(1, rules['team_size']+1)]
        if max(totals)-min(totals) > 1:
            raise CompetitionError('INVALID_LINEUP', '本房间要求出场次数相差不超过一次。')


def default_lineup(rules):
    return {key: i % rules['team_size'] + 1 for i, key in enumerate(rules['game_keys'])}


def series_complete(rules, game_key, yellow_wins, white_wins):
    return game_key == rules['game_keys'][-1] or (rules['series_mode'] == 'best_of' and max(yellow_wins, white_wins) >= rules['wins_required'])
