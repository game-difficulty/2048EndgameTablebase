import numpy as np


def sanitize_config(obj):
    if isinstance(obj, np.ndarray):  # type: ignore
        return [sanitize_config(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):  # type: ignore
        return obj.item()
    if isinstance(obj, tuple):
        return [sanitize_config(x) for x in obj]
    if isinstance(obj, set):
        return [sanitize_config(x) for x in obj]
    if isinstance(obj, list):
        return [sanitize_config(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_config(v) for k, v in obj.items()}
    return obj
