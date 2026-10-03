"""Outcome contracts and conservation-preserving public-room payouts."""
import math

AMM = 'competition-fpmm-v1'
POOL = 'competition-pool-v1'


def score_options(wins):
    return tuple([f'{wins}:{n}' for n in range(wins)] +
                 [f'{n}:{wins}' for n in reversed(range(wins))])


OPTIONS = {'winner': ('yellow', 'white'), 'first_two': ('2:0', '1:1', '0:2'),
           'clinch_3': score_options(3), 'clinch_4': score_options(4)}


def market_contract(facts):
    kind = facts.get('room_kind', 'competition')
    count = (facts.get('rules') or {}).get('game_count', 3)
    if kind == 'time_attack':
        return POOL, ('winner',)
    if kind == 'duel':
        return POOL, ('winner', 'first_two') if count >= 2 else ('winner',)
    return AMM, ('winner', {5: 'clinch_3', 7: 'clinch_4'}.get(count, 'first_two'))


def initial_reserves(kind, initial):
    if kind == 'first_two':
        return {'2:0': initial, '1:1': initial // 2, '0:2': initial}
    if kind.startswith('clinch_'):
        wins = int(kind[-1])
        # P(k:n) = C(k+n-1,n) / 2**(k+n), for either winner.
        # FPMM probabilities are proportional to reciprocal reserves.
        from fractions import Fraction
        probs = [Fraction(math.comb(wins+n-1, n), 2**(wins+n)) for n in range(wins)]
        ratios = [probs[0]/p for p in probs]
        scale = math.lcm(*(r.denominator for r in ratios))
        # Keep the existing liquidity ceiling while preserving exact ratios.
        base = max(scale, initial // scale * scale)
        reserves = [base * r.numerator // r.denominator for r in ratios]
        return dict(zip(OPTIONS[kind], reserves + list(reversed(reserves))))
    return dict.fromkeys(OPTIONS[kind], initial)


def outcome(facts, kind):
    if facts['phase'] == 'CANCELLED':
        return None, 'cancelled'
    result = facts.get('public_result') or {}
    if kind == 'winner':
        side = result.get('winner_side')
        return (side, 'match_winner') if side in OPTIONS[kind] else (None, 'draw_or_unresolved')
    games = {g['game_key']: g for g in result.get('games', [])}
    count = 2 if kind == 'first_two' else int(kind[-1])*2-1
    yellow = white = 0
    for i in range(count):
        side = games.get(chr(65+i), {}).get('winner_side')
        if side not in ('yellow', 'white'):
            return None, 'draw_or_unresolved'
        yellow += side == 'yellow'
        white += side == 'white'
        if kind != 'first_two' and max(yellow, white) == int(kind[-1]):
            return f'{yellow}:{white}', 'clinching_score'
    return f'{yellow}:{white}', 'first_two_score'


def pool_payouts(positions, winner):
    total = sum(p['stake'] for p in positions)
    winning = [p for p in positions if p['option_id'] == winner]
    denominator = sum(p['stake'] for p in winning)
    if winner is None or not denominator:
        return {p['user_id']: p['stake'] for p in positions}, True
    payouts = {p['user_id']: 0 for p in positions}
    for p in winning:
        payouts[p['user_id']] = total*p['stake']//denominator
    # Largest remainders, stable user-id tie break; no units minted or discarded.
    remainder = total-sum(payouts.values())
    for p in sorted(winning, key=lambda p: (-(total*p['stake'] % denominator), p['user_id']))[:remainder]:
        payouts[p['user_id']] += 1
    return payouts, False
