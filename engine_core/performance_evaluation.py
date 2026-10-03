from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_PERFECT_LABEL = "Perfect!"
DEFAULT_PERFECT_TOLERANCE = 3e-10
DEFAULT_PERFECT_TOLERANCES = {
    "uint32": 3e-10, "float32": 3e-10, "1-float32": 3e-10,
    "uint64": 1e-14, "float64": 1e-14, "1-float64": 1e-14,
}
DEFAULT_EVALUATIONS = (
    {"label": "Excellent!", "threshold": 0.999},
    {"label": "Nice try!", "threshold": 0.99},
    {"label": "Not bad!", "threshold": 0.975},
    {"label": "Mistake!", "threshold": 0.9},
    {"label": "Blunder!", "threshold": 0.75},
    {"label": "Terrible!", "threshold": float("-inf")},
)
CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs_and_configs"
    / "performance_evaluations.json"
)


def _dtype_name(dtype) -> str:
    name = str(dtype or "").strip().lower()
    return {"f32": "float32", "f64": "float64",
            "1-f32": "1-float32", "1-f64": "1-float64"}.get(name, name)


def _normalize_perfect_tolerances(raw) -> dict[str, float]:
    values = dict(DEFAULT_PERFECT_TOLERANCES)
    if isinstance(raw, dict):
        for dtype, value in raw.items():
            name = _dtype_name(dtype)
            if name not in values:
                continue
            try:
                tolerance = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(tolerance) and tolerance >= 0:
                values[name] = tolerance
    return values


def perfect_tolerance(dtype=None) -> float:
    return PERFORMANCE_PERFECT_TOLERANCES.get(
        _dtype_name(dtype), PERFORMANCE_PERFECT_TOLERANCE
    )


def _default_config() -> dict[str, Any]:
    return {
        "perfect_label": DEFAULT_PERFECT_LABEL,
        "perfect": {
            "tolerance": DEFAULT_PERFECT_TOLERANCE,
            "tolerance_by_dtype": dict(DEFAULT_PERFECT_TOLERANCES),
        },
        "evaluations": [dict(item) for item in DEFAULT_EVALUATIONS],
    }


def _normalize_evaluations(raw_items: Any) -> list[dict[str, float | str]]:
    normalized: list[dict[str, float | str]] = []
    if not isinstance(raw_items, list):
        return [dict(item) for item in DEFAULT_EVALUATIONS]

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("text") or "").strip()
        if not label:
            continue
        threshold_raw = item.get("threshold", item.get("value"))
        try:
            threshold = float(threshold_raw)
        except (TypeError, ValueError):
            continue
        normalized.append({"label": label, "threshold": threshold})

    if not normalized:
        return [dict(item) for item in DEFAULT_EVALUATIONS]

    normalized.sort(key=lambda item: float(item["threshold"]), reverse=True)
    return normalized


def load_performance_evaluation_config() -> dict[str, Any]:
    config = _default_config()
    if not CONFIG_PATH.exists():
        return config

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    except (OSError, json.JSONDecodeError):
        return config

    if isinstance(raw, dict):
        perfect_label = str(raw.get("perfect_label") or DEFAULT_PERFECT_LABEL).strip()
        config["perfect_label"] = perfect_label or DEFAULT_PERFECT_LABEL
        perfect = raw.get("perfect")
        if isinstance(perfect, dict):
            try:
                tolerance = float(perfect.get("tolerance", DEFAULT_PERFECT_TOLERANCE))
            except (TypeError, ValueError):
                tolerance = DEFAULT_PERFECT_TOLERANCE
            if math.isfinite(tolerance) and tolerance >= 0:
                config["perfect"]["tolerance"] = tolerance
            config["perfect"]["tolerance_by_dtype"] = _normalize_perfect_tolerances(
                perfect.get("tolerance_by_dtype")
            )
        config["evaluations"] = _normalize_evaluations(raw.get("evaluations"))
    return config


PERFORMANCE_EVALUATION_CONFIG = load_performance_evaluation_config()
PERFORMANCE_PERFECT_TOLERANCE = PERFORMANCE_EVALUATION_CONFIG["perfect"]["tolerance"]
PERFORMANCE_PERFECT_TOLERANCES = PERFORMANCE_EVALUATION_CONFIG["perfect"]["tolerance_by_dtype"]
PERFORMANCE_PERFECT_LABEL = str(
    PERFORMANCE_EVALUATION_CONFIG.get("perfect_label", DEFAULT_PERFECT_LABEL)
).strip() or DEFAULT_PERFECT_LABEL
PERFORMANCE_EVALUATIONS: tuple[dict[str, float | str], ...] = tuple(
    {
        "label": str(item["label"]),
        "threshold": float(item["threshold"]),
    }
    for item in PERFORMANCE_EVALUATION_CONFIG.get("evaluations", DEFAULT_EVALUATIONS)
)
PERFORMANCE_LABELS: tuple[str, ...] = (
    PERFORMANCE_PERFECT_LABEL,
    *[str(item["label"]) for item in PERFORMANCE_EVALUATIONS],
)


def is_perfect_result(selected_rate: float, best_rate: float, dtype=None) -> bool:
    try:
        selected = float(selected_rate)
        best = float(best_rate)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(selected) or not math.isfinite(best):
        return False
    return best - selected <= perfect_tolerance(dtype)


def markdown_label(label: str) -> str:
    stripped = str(label).strip()
    if stripped.startswith("**") and stripped.endswith("**"):
        return stripped
    return f"**{stripped}**"


def get_performance_labels(*, markdown: bool = False) -> tuple[str, ...]:
    if not markdown:
        return PERFORMANCE_LABELS
    return tuple(markdown_label(label) for label in PERFORMANCE_LABELS)


def build_performance_stats(*, markdown: bool = False) -> dict[str, int]:
    return {label: 0 for label in get_performance_labels(markdown=markdown)}


def evaluation_of_performance(loss: float, *, markdown: bool = False) -> str:
    try:
        numeric_loss = float(loss)
    except (TypeError, ValueError):
        numeric_loss = 0.0

    for item in PERFORMANCE_EVALUATIONS:
        if numeric_loss >= float(item["threshold"]):
            label = str(item["label"])
            return markdown_label(label) if markdown else label

    fallback = str(PERFORMANCE_EVALUATIONS[-1]["label"])
    return markdown_label(fallback) if markdown else fallback
