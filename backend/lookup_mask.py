"""Logical lookup mask shared by readers and assistance review (not disk symmetry)."""
from __future__ import annotations

from engine_core.GoalSpec import GoalSpec

MASK_VERSION = 1


def replace_largest_tiles(board_encoded, n, target):
    board = int(board_encoded)
    if not n:
        return board
    tiles = [(board >> (4 * i)) & 15 for i in range(16)]
    threshold = sorted(tiles, reverse=True)[n - 1]
    if 2 ** threshold < int(target):
        return board
    count = 0
    # Preserve the reader's low-nibble-first tie breaking.
    for i, tile in enumerate(tiles):
        if count < n and tile >= threshold:
            tiles[i] = 15
            count += 1
    return sum(tile << (4 * i) for i, tile in enumerate(tiles))


def replace_variant_large_tiles(board_encoded, pattern, target):
    from Config import pattern_catalog
    seeds = pattern_catalog.get(pattern, {}).get('seed_boards', ())
    if not len(seeds):
        return int(board_encoded)
    goal = GoalSpec.parse(target)
    rank = goal.encoding_rank
    board, walls = int(board_encoded), int(seeds[0])
    result = 0
    for i in range(16):
        tile = (board >> (4 * i)) & 15
        if (walls >> (4 * i)) & 15 == 15:
            tile = 15
        elif goal.kind == "tile" and tile >= rank:
            tile = 14
        result |= tile << (4 * i)
    return result


def replace_board_for_lookup(board_encoded, pattern, n, target, use_variant):
    if str(target).startswith("sum-") and not use_variant:
        GoalSpec.parse(target)
        return int(board_encoded)
    return (replace_variant_large_tiles(board_encoded, pattern, target) if use_variant
            else replace_largest_tiles(board_encoded, n, target))


def descriptor(full_pattern):
    from Config import pattern_catalog, pattern_32k_tiles_map, category_info
    pattern, target = full_pattern.rsplit('_', 1)
    GoalSpec.parse(target)
    if pattern not in pattern_catalog:
        raise ValueError('invalid_evidence_pattern')
    variant = next((v for v in ('2x4', '3x3', '3x4') if pattern.startswith(v)), '4x4')
    return pattern, target, pattern_32k_tiles_map.get(pattern, [0])[0], pattern in category_info.get('variant', []), variant


def lookup_key(board_encoded, full_pattern):
    pattern, target, count, variant, _ = descriptor(full_pattern)
    return f'{replace_board_for_lookup(board_encoded, pattern, count, target, variant):016x}'


def play_lookup_key(board, variant, full_pattern):
    from Config import pattern_catalog
    pattern, _, _, is_variant, expected = descriptor(full_pattern)
    if variant != expected:
        return None
    ranks = [0 if not tile else min(15, int(tile).bit_length() - 1) for tile in board]
    if is_variant:
        walls = f'{int(pattern_catalog[pattern]["seed_boards"][0]):016x}'
        positions = [i for i, value in enumerate(walls) if value != 'f']
        if len(positions) != len(ranks):
            return None
        padded = [15] * 16
        for i, rank in zip(positions, ranks):
            padded[i] = rank
        ranks = padded
    encoded = int(''.join(format(rank, 'x') for rank in ranks), 16)
    return lookup_key(encoded, full_pattern)
