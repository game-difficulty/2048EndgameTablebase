def allows_five_tiler_relaxation(readers):
    patterns = {reader[3] for group in readers.values() for reader in group}
    return bool(patterns & {'free12', 'free11', 'free13', '4442', '4442f', '4442ff'})


def merge_urgency_for_readers(readers):
    """Use only tables admitted by the dispatcher's availability checks."""
    patterns = {reader[3] for group in readers.values() for reader in group}
    if 'free12' in patterns:
        return 1.5
    return 1.0 if patterns & {'free11', 'LL', '4442', '4442f', '4442ff'} else 0.0
