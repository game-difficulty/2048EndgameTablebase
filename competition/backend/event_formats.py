"""Event capabilities, separate from room state machines and project adapters."""
FORMATS = {
    'team-draft-v1': {
        'label': '三人团队 · 选 Ban 对战', 'team_size': 3,
        'capabilities': {'registration': True, 'teams': True, 'rooms': True, 'statistics': False},
    },
    'team-top5-3x3-v1': {
        'label': '3×3 · 四队积分统计', 'team_size': 5,
        'capabilities': {'registration': True, 'teams': True, 'rooms': False, 'statistics': True},
    },
}
