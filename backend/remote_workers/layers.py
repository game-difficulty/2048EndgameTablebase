"""Optional, bounded layer inventory in Worker advertisements."""

MAX_LAYER_RANGES = 4096
MAX_LAYER = 0xffffffff


def normalize_layer_inventory(value):
    if not isinstance(value, dict) or type(value.get('version')) is not int or value['version'] != 1:
        return None
    ranges = value.get('ranges')
    if not isinstance(ranges, list) or len(ranges) > MAX_LAYER_RANGES:
        return None
    result = []
    previous = -1
    for pair in ranges:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(v) is not int for v in pair)
                or not 0 <= pair[0] <= pair[1] <= MAX_LAYER or pair[0] <= previous):
            return None
        result.append(pair.copy())
        previous = pair[1]
    return {'version': 1, 'ranges': result}
