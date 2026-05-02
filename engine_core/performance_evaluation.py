from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_PERFECT_LABEL = "Perfect!"
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


def _default_config() -> dict[str, Any]:
    return {
        "perfect_label": DEFAULT_PERFECT_LABEL,
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
        config["evaluations"] = _normalize_evaluations(raw.get("evaluations"))
    return config


PERFORMANCE_EVALUATION_CONFIG = load_performance_evaluation_config()
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
