"""819984 final flow: small result/clock operations, never board simulation."""
VERSION = '819984-final-v1'


def points(control):
    return {'yellow': 2 * int(control['yellow_wins']) + int(control['draws']),
            'white': 2 * int(control['white_wins']) + int(control['draws'])}


def refund_decision(winner, loser):
    """Times are client active elapsed milliseconds since the shared game start."""
    if winner.outcome not in ('no_moves', 'tile_limit', 'target_reached') or loser.outcome == 'surrendered':
        return None
    threshold = int(loser.extra.get('result_value', loser.score))
    history = (winner.extra.get('checkpoint') or {}).get('metric_history', [])
    end, other_end = int(winner.elapsed_ms), int(loser.elapsed_ms)
    # Refunds are settled only after normal completion, never borrowed to play
    # beyond the team's available time or used to reverse a timeout forfeiture.
    budget = winner.extra.get('project_clock_start_ms')
    if budget is not None and end >= int(budget):
        return None
    if end < other_end:
        return None
    current = 0
    decisive = None
    for elapsed, value in history:
        if elapsed <= other_end:
            current = value
        elif decisive is None and value > threshold:
            decisive = int(elapsed)
            break
    if current > threshold:
        decisive = other_end
    if decisive is None or decisive > end:
        return None
    return decisive, min(300000, max(0, end - decisive))
