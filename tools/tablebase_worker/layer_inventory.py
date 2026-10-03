"""Conservative layer presence hints; never read table payloads on the move path."""
import os
import re

from backend.remote_workers.layers import MAX_LAYER_RANGES, MAX_LAYER, MIN_LAYER


class LayerInventory:
    def __init__(self):
        self._cache = {}

    def get(self, table):
        try:
            signature = tuple((str(path), path.stat().st_mtime_ns) for path in table.paths)
            cached = self._cache.get(table.table_id)
            if cached is not None and cached[0] == signature:
                return cached[1]
            value = self._scan(table)
            # Do not publish a partial scan when files changed during enumeration.
            if signature != tuple((str(path), path.stat().st_mtime_ns) for path in table.paths):
                return None
            self._cache[table.table_id] = (signature, value)
            return value
        except OSError:
            self._cache.pop(table.table_id, None)
            return None

    @staticmethod
    def _scan(table):
        expression = re.compile(rf'^{re.escape(table.table_id)}_(-?\d+)(b|\.z|\.book|\.zbook|\.exzbook|\.exadbook|\.exadzbook|\.bccmp|\.bcraw|\.bcpos|\.bcsuc)$')
        layers, positions, successes = set(), set(), set()
        for path in table.paths:
            with os.scandir(path) as entries:
                for entry in entries:
                    match = expression.fullmatch(entry.name)
                    if not match:
                        continue
                    ordinal, suffix = int(match[1]), match[2]
                    if not MIN_LAYER <= ordinal <= MAX_LAYER:
                        return None
                    if suffix in ('b', '.z'):
                        if entry.is_dir():
                            layers.add(ordinal)
                    elif entry.is_file():
                        if suffix == '.bcpos':
                            positions.add(ordinal)
                        elif suffix == '.bcsuc':
                            successes.add(ordinal)
                        else:
                            layers.add(ordinal)
        layers.update(positions & successes)
        ranges = []
        for ordinal in sorted(layers):
            if ranges and ordinal == ranges[-1][1] + 1:
                ranges[-1][1] = ordinal
            else:
                ranges.append([ordinal, ordinal])
        if len(ranges) > MAX_LAYER_RANGES:
            return None
        return {'version': 1, 'ranges': ranges}
