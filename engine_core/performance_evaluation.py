from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_PERFECT_LABEL = "Perfect!"
DEFAULT_PERFECT_COLOR = "#2e7d32"
DEFAULT_PERFECT_COMPARISON = "absolute_difference"
DEFAULT_PERFECT_TOLERANCE = 3e-10
DEFAULT_REPORT_DECIMAL_PLACES = 4
DEFAULT_ANALYSIS_START_STEP = 5
DEFAULT_REPORT_MIN_TEXT_LINES = 101
DEFAULT_EVALUATIONS = (
    {"label": "Excellent!", "threshold": 0.999, "color": "#7cb342"},
    {"label": "Nice try!", "threshold": 0.99, "color": "#c0ca33"},
    {"label": "Not bad!", "threshold": 0.975, "color": "#fb8c00"},
    {"label": "Mistake!", "threshold": 0.9, "color": "#f4511e"},
    {"label": "Blunder!", "threshold": 0.75, "color": "#e53935"},
    {"label": "Terrible!", "threshold": -1.0, "color": "#b71c1c"},
)
DEFAULT_RESULT_BAR_STOPS = (
    {"max_loss": 0.001, "color": "#2e7d32"},
    {"max_loss": 0.01, "color": "#8bc34a"},
    {"max_loss": 0.03, "color": "#ff9800"},
    {"max_loss": 0.1, "color": "#f44336"},
)
CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs_and_configs"
    / "performance_evaluations.json"
)


def _default_config() -> dict[str, Any]:
    return {
        "report_decimal_places": DEFAULT_REPORT_DECIMAL_PLACES,
        "perfect": {
            "label": DEFAULT_PERFECT_LABEL,
            "comparison": DEFAULT_PERFECT_COMPARISON,
            "tolerance": DEFAULT_PERFECT_TOLERANCE,
            "color": DEFAULT_PERFECT_COLOR,
        },
        "evaluations": [dict(item) for item in DEFAULT_EVALUATIONS],
        "result_bar": {"stops": [dict(item) for item in DEFAULT_RESULT_BAR_STOPS]},
        "analysis": {
            "start_step": DEFAULT_ANALYSIS_START_STEP,
            "report_min_text_lines": DEFAULT_REPORT_MIN_TEXT_LINES,
        },
    }


def _normalize_color(value: Any, fallback: str) -> str:
    color = str(value or "").strip()
    return color if color else fallback


def _normalize_nonnegative_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if math.isfinite(parsed) and parsed >= 0 else fallback


def _normalize_int(value: Any, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return min(maximum, max(minimum, parsed))


def _normalize_evaluations(raw_items: Any) -> list[dict[str, float | str]]:
    normalized: list[dict[str, float | str]] = []
    if not isinstance(raw_items, list):
        return [dict(item) for item in DEFAULT_EVALUATIONS]

    for index, item in enumerate(raw_items):
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
        if not math.isfinite(threshold) and threshold != float("-inf"):
            continue
        fallback_color = str(
            DEFAULT_EVALUATIONS[min(index, len(DEFAULT_EVALUATIONS) - 1)]["color"]
        )
        normalized.append(
            {
                "label": label,
                "threshold": threshold,
                "color": _normalize_color(item.get("color"), fallback_color),
            }
        )

    if not normalized:
        return [dict(item) for item in DEFAULT_EVALUATIONS]

    normalized.sort(key=lambda item: float(item["threshold"]), reverse=True)
    return normalized


def _normalize_result_bar(raw: Any) -> dict[str, list[dict[str, float | str]]]:
    raw_stops = raw.get("stops") if isinstance(raw, dict) else None
    if not isinstance(raw_stops, list):
        return {"stops": [dict(item) for item in DEFAULT_RESULT_BAR_STOPS]}

    stops: list[dict[str, float | str]] = []
    for index, item in enumerate(raw_stops):
        if not isinstance(item, dict):
            continue
        max_loss = _normalize_nonnegative_float(item.get("max_loss"), -1.0)
        if max_loss < 0 or max_loss > 1:
            continue
        fallback_color = str(
            DEFAULT_RESULT_BAR_STOPS[
                min(index, len(DEFAULT_RESULT_BAR_STOPS) - 1)
            ]["color"]
        )
        stops.append(
            {
                "max_loss": max_loss,
                "color": _normalize_color(item.get("color"), fallback_color),
            }
        )

    stops.sort(key=lambda item: float(item["max_loss"]))
    deduplicated = {float(item["max_loss"]): item for item in stops}
    normalized = [deduplicated[key] for key in sorted(deduplicated)]
    if not normalized or float(normalized[-1]["max_loss"]) <= 0:
        normalized = [dict(item) for item in DEFAULT_RESULT_BAR_STOPS]
    return {"stops": normalized}


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
        config["report_decimal_places"] = _normalize_int(
            raw.get("report_decimal_places"),
            DEFAULT_REPORT_DECIMAL_PLACES,
            0,
            15,
        )
        raw_perfect = raw.get("perfect") if isinstance(raw.get("perfect"), dict) else {}
        legacy_label = raw.get("perfect_label")
        comparison = str(
            raw_perfect.get("comparison") or DEFAULT_PERFECT_COMPARISON
        ).strip().lower()
        if comparison not in ("absolute_difference", "relative_ratio"):
            comparison = DEFAULT_PERFECT_COMPARISON
        perfect_label = str(
            raw_perfect.get("label") or legacy_label or DEFAULT_PERFECT_LABEL
        ).strip()
        config["perfect"] = {
            "label": perfect_label or DEFAULT_PERFECT_LABEL,
            "comparison": comparison,
            "tolerance": _normalize_nonnegative_float(
                raw_perfect.get("tolerance"), DEFAULT_PERFECT_TOLERANCE
            ),
            "color": _normalize_color(
                raw_perfect.get("color"), DEFAULT_PERFECT_COLOR
            ),
        }
        config["evaluations"] = _normalize_evaluations(raw.get("evaluations"))
        config["result_bar"] = _normalize_result_bar(raw.get("result_bar"))
        raw_analysis = raw.get("analysis") if isinstance(raw.get("analysis"), dict) else {}
        config["analysis"] = {
            "start_step": _normalize_int(
                raw_analysis.get("start_step"), DEFAULT_ANALYSIS_START_STEP, 0, 1000000
            ),
            "report_min_text_lines": _normalize_int(
                raw_analysis.get("report_min_text_lines"),
                DEFAULT_REPORT_MIN_TEXT_LINES,
                0,
                1000000,
            ),
        }
    return config


PERFORMANCE_EVALUATION_CONFIG = load_performance_evaluation_config()
PERFECT_CONFIG = PERFORMANCE_EVALUATION_CONFIG["perfect"]
PERFORMANCE_PERFECT_LABEL = str(PERFECT_CONFIG["label"])
PERFORMANCE_PERFECT_COLOR = str(PERFECT_CONFIG["color"])
PERFORMANCE_PERFECT_COMPARISON = str(PERFECT_CONFIG["comparison"])
PERFORMANCE_PERFECT_TOLERANCE = float(PERFECT_CONFIG["tolerance"])
REPORT_DECIMAL_PLACES = int(PERFORMANCE_EVALUATION_CONFIG["report_decimal_places"])
ANALYSIS_START_STEP = int(PERFORMANCE_EVALUATION_CONFIG["analysis"]["start_step"])
REPORT_MIN_TEXT_LINES = int(
    PERFORMANCE_EVALUATION_CONFIG["analysis"]["report_min_text_lines"]
)
PERFORMANCE_EVALUATIONS: tuple[dict[str, float | str], ...] = tuple(
    {
        "label": str(item["label"]),
        "threshold": float(item["threshold"]),
        "color": str(item["color"]),
    }
    for item in PERFORMANCE_EVALUATION_CONFIG.get("evaluations", DEFAULT_EVALUATIONS)
)
PERFORMANCE_LABELS: tuple[str, ...] = (
    PERFORMANCE_PERFECT_LABEL,
    *[str(item["label"]) for item in PERFORMANCE_EVALUATIONS],
)
RESULT_BAR_STOPS: tuple[dict[str, float | str], ...] = tuple(
    {
        "max_loss": float(item["max_loss"]),
        "color": str(item["color"]),
    }
    for item in PERFORMANCE_EVALUATION_CONFIG["result_bar"]["stops"]
)


def is_perfect_result(selected_rate: float, best_rate: float) -> bool:
    try:
        selected = float(selected_rate)
        best = float(best_rate)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(selected) or not math.isfinite(best):
        return False
    if PERFORMANCE_PERFECT_COMPARISON == "relative_ratio":
        if best <= 0:
            return abs(best - selected) <= PERFORMANCE_PERFECT_TOLERANCE
        return selected / best >= 1.0 - PERFORMANCE_PERFECT_TOLERANCE
    return best - selected <= PERFORMANCE_PERFECT_TOLERANCE


def public_performance_config() -> dict[str, Any]:
    return {
        "report_decimal_places": REPORT_DECIMAL_PLACES,
        "perfect": {
            "label": PERFORMANCE_PERFECT_LABEL,
            "comparison": PERFORMANCE_PERFECT_COMPARISON,
            "tolerance": PERFORMANCE_PERFECT_TOLERANCE,
            "color": PERFORMANCE_PERFECT_COLOR,
        },
        "evaluations": [dict(item) for item in PERFORMANCE_EVALUATIONS],
        "result_bar": {"stops": [dict(item) for item in RESULT_BAR_STOPS]},
        "analysis": {
            "start_step": ANALYSIS_START_STEP,
            "report_min_text_lines": REPORT_MIN_TEXT_LINES,
        },
    }


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
