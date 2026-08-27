from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import (
    exponent_to_value,
    normalize_hex_16,
    read_jsonl,
    value_to_exponent,
    write_json,
    write_jsonl,
)

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised only on missing deps
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


VISIBLE_VALUE_EXPONENTS = list(range(1, 16))
TEXT_ACCEPT_THRESHOLD = 0.60
COLOR_ACCEPT_THRESHOLD = 0.72
BOARD_ACCEPT_THRESHOLD = 0.78
ANNOTATION_REVIEW_THRESHOLD = 0.72
DEFAULT_PARSER_PROFILE = "standard"
VARIANT_3X4_PARSER_PROFILE = "variant-3x4"
PARSER_PROFILES = (DEFAULT_PARSER_PROFILE, VARIANT_3X4_PARSER_PROFILE)

# The Word guide uses a fixed screenshot palette which is close to, but not
# exactly the app theme. Empty cells and "2" cells intentionally share almost
# the same fill color, so exponent 1 must be decided by text presence.
DEFAULT_GUIDE_PALETTE: dict[int, list[int]] = {
    0: [255, 228, 227],
    1: [255, 228, 227],
    2: [254, 206, 206],
    3: [255, 192, 193],
    4: [253, 162, 161],
    5: [255, 141, 141],
    6: [253, 117, 119],
    7: [254, 156, 5],
    8: [253, 137, 2],
    9: [255, 198, 54],
    10: [255, 211, 100],
    11: [255, 222, 141],
    12: [244, 174, 210],
}


@dataclass
class TextMatch:
    exponent: int
    value: int
    score: float
    runner_up_score: float
    area_ratio: float


@dataclass
class CellProbe:
    row: int
    col: int
    bbox: list[int]
    background_rgb: list[int]
    text_match: TextMatch
    text_area_ratio: float


@dataclass
class BoardProbe:
    board_id: str
    bbox: list[int]
    visible_rows: int
    visible_cols: int
    fit_confidence: float
    cells: list[CellProbe]
    flags: list[str]


@dataclass
class BoardCandidate:
    bbox: list[int]
    visible_rows: int
    visible_cols: int
    fit_confidence: float
    cell_boxes: list[tuple[int, int, list[int]]]
    flags: list[str]
    source: str = "unknown"


@dataclass(frozen=True)
class BoardPadding:
    right: str = "f"
    bottom: str = "f"


def _padding_for_profile(profile: str) -> BoardPadding:
    if profile == VARIANT_3X4_PARSER_PROFILE:
        return BoardPadding(right="e", bottom="f")
    return BoardPadding()


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for v1 guide image parsing. Install "
            "opencv-python-headless in the offline tooling environment."
        ) from _CV2_IMPORT_ERROR
    return cv2


def _load_rgb(path: Path) -> np.ndarray:
    cv = _require_cv2()
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv.imdecode(data, cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def _odd_kernel(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _bbox_area(bbox: list[int]) -> int:
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def _bbox_intersection(left: list[int], right: list[int]) -> int:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x0 = max(lx, rx)
    y0 = max(ly, ry)
    x1 = min(lx + lw, rx + rw)
    y1 = min(ly + lh, ry + rh)
    return max(0, x1 - x0) * max(0, y1 - y0)


def _point_in_bbox(point: tuple[float, float], bbox: list[int], margin: float = 0) -> bool:
    x, y = point
    bx, by, bw, bh = bbox
    return bx - margin <= x <= bx + bw + margin and by - margin <= y <= by + bh + margin


def _center_of_bbox(bbox: list[int]) -> tuple[float, float]:
    return bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0


def _colored_mask(rgb: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    non_white = np.any(rgb < 248, axis=2)
    mask = ((saturation > 5) & (value < 254) & non_white).astype(np.uint8) * 255
    return mask


def _tile_palette_mask(rgb: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    palette = np.array(
        list({tuple(value) for value in DEFAULT_GUIDE_PALETTE.values()}),
        dtype=np.int16,
    )
    pixels = rgb.astype(np.int16)
    distances = np.linalg.norm(pixels[:, :, None, :] - palette[None, None, :, :], axis=3)
    nearest = distances.min(axis=2)
    # JPEG-compressed white gutters can be close to the pale tile color. A
    # loose non-white gate plus erosion below separates gutters while keeping
    # small, pale empty cells in multi-board figures.
    not_white = np.any(rgb < 248, axis=2)
    mask = ((nearest < 68) & not_white).astype(np.uint8) * 255
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    return cv.erode(mask, kernel, iterations=1)


def _tile_rect_bboxes(rgb: np.ndarray) -> list[list[int]]:
    cv = _require_cv2()
    mask = _tile_palette_mask(rgb)
    contours, _hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    raw: list[list[int]] = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w < 8 or h < 8:
            continue
        ratio = w / max(1, h)
        if ratio < 0.45 or ratio > 1.9:
            continue
        component = mask[y : y + h, x : x + w]
        fill_ratio = float((component > 0).sum()) / max(1, w * h)
        if fill_ratio < 0.25:
            continue
        raw.append([int(x), int(y), int(w), int(h)])

    if len(raw) < 4:
        return []

    sides = [min(bbox[2], bbox[3]) for bbox in raw]
    median_side = float(np.median(sides))
    filtered = [
        bbox
        for bbox in raw
        if min(bbox[2], bbox[3]) >= median_side * 0.45
        and max(bbox[2], bbox[3]) <= median_side * 1.75
    ]
    return sorted(filtered, key=lambda bbox: (bbox[1], bbox[0]))


def _sequence_strip_board_candidates(rgb: np.ndarray) -> list[BoardCandidate]:
    """Detect diagrams made of spatially separated 1x4 board states.

    Thin arrows and highlight frames connect adjacent tile masks in these
    diagrams. An extra erosion removes those strokes while preserving the
    substantially larger tile interiors.
    """
    cv = _require_cv2()
    image_height, image_width = rgb.shape[:2]
    if image_width / max(1, image_height) < 2.5:
        return []

    mask = _tile_palette_mask(rgb)
    separated = cv.erode(
        mask,
        cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    separated = cv.morphologyEx(
        separated,
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_RECT, (3, 9)),
        iterations=1,
    )
    raw_boxes: list[list[int]] = []
    for bbox in _contour_bboxes(separated):
        x, y, w, h = bbox
        if w < 12 or h < 12:
            continue
        ratio = w / max(1, h)
        if not 0.72 <= ratio <= 1.38:
            continue
        component = separated[y : y + h, x : x + w]
        fill_ratio = float((component > 0).sum()) / max(1, component.size)
        if fill_ratio >= 0.52:
            raw_boxes.append([int(x), int(y), int(w), int(h)])
    if len(raw_boxes) < 4:
        return []

    median_side = float(np.median([min(box[2], box[3]) for box in raw_boxes]))
    tile_boxes = [
        box
        for box in raw_boxes
        if median_side * 0.76 <= min(box[2], box[3]) <= median_side * 1.24
        and max(box[2], box[3]) <= median_side * 1.34
    ]
    if len(tile_boxes) < 4:
        return []

    y_positions = _cluster_positions([float(box[1]) for box in tile_boxes], median_side * 0.32)
    if not y_positions:
        return []
    y_gaps = np.diff(sorted(y_positions))
    if len(y_gaps) and float(np.min(y_gaps)) <= median_side * 1.46:
        return []

    median_w = float(np.median([box[2] for box in tile_boxes]))
    median_h = float(np.median([box[3] for box in tile_boxes]))
    expand = max(1, round(median_side * 0.055))
    candidates: list[BoardCandidate] = []
    for y_pos in y_positions:
        row_boxes = [box for box in tile_boxes if abs(box[1] - y_pos) <= median_side * 0.36]
        x_positions = _cluster_positions([float(box[0]) for box in row_boxes], median_side * 0.32)
        for x_group in _x_grid_groups(x_positions, median_side * 1.22):
            if len(x_group) != 4:
                continue
            evidence = sum(
                any(abs(box[0] - x_pos) <= median_side * 0.36 for box in row_boxes)
                for x_pos in x_group
            )
            if evidence < 3:
                continue
            cell_boxes = [
                (
                    0,
                    col,
                    [
                        max(0, round(x_pos) - expand),
                        max(0, round(y_pos) - expand),
                        max(1, round(median_w) + expand * 2),
                        max(1, round(median_h) + expand * 2),
                    ],
                )
                for col, x_pos in enumerate(x_group)
            ]
            xs = [cell[2][0] for cell in cell_boxes]
            ys = [cell[2][1] for cell in cell_boxes]
            rights = [cell[2][0] + cell[2][2] for cell in cell_boxes]
            bottoms = [cell[2][1] + cell[2][3] for cell in cell_boxes]
            bbox = [min(xs), min(ys), max(rights) - min(xs), max(bottoms) - min(ys)]
            if bbox[0] + bbox[2] > image_width or bbox[1] + bbox[3] > image_height:
                continue
            flags = ["partial_board_bottom_f_padding"]
            if evidence < 4:
                flags.append("sequence_strip_inferred_cell")
            candidates.append(
                BoardCandidate(
                    bbox=bbox,
                    visible_rows=1,
                    visible_cols=4,
                    fit_confidence=0.94 if evidence == 4 else 0.88,
                    cell_boxes=cell_boxes,
                    flags=flags,
                    source="sequence_strip",
                )
            )
    return _sort_board_candidates_spatial(candidates)


def _colored_cell_bboxes(rgb: np.ndarray) -> list[list[int]]:
    mask = _colored_mask(rgb)
    raw: list[list[int]] = []
    for bbox in _contour_bboxes(mask):
        x, y, w, h = bbox
        if w < 8 or h < 8:
            continue
        ratio = w / max(1, h)
        if ratio < 0.45 or ratio > 1.9:
            continue
        component = mask[y : y + h, x : x + w]
        fill_ratio = float((component > 0).sum()) / max(1, w * h)
        if fill_ratio < 0.25:
            continue
        raw.append([int(x), int(y), int(w), int(h)])
    if len(raw) < 4:
        return []

    sides = [min(bbox[2], bbox[3]) for bbox in raw]
    median_side = float(np.median(sides))
    filtered = [
        bbox
        for bbox in raw
        if min(bbox[2], bbox[3]) >= median_side * 0.58
        and max(bbox[2], bbox[3]) <= median_side * 1.65
    ]
    return sorted(filtered, key=lambda bbox: (bbox[1], bbox[0]))


def _cluster_positions(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - float(np.median(clusters[-1]))) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [float(np.median(cluster)) for cluster in clusters]


def _complete_missing_grid_positions(positions: list[float], step_hint: float) -> list[float]:
    if len(positions) < 2:
        return positions
    positions = sorted(positions)
    gaps = [positions[index + 1] - positions[index] for index in range(len(positions) - 1)]
    local_gaps = [gap for gap in gaps if step_hint * 0.55 <= gap <= step_hint * 1.55]
    step = float(np.median(local_gaps)) if local_gaps else step_hint
    completed = [positions[0]]
    segment_len = 1
    for index, gap in enumerate(gaps):
        next_value = positions[index + 1]
        if gap > step * 1.6 and segment_len >= 4:
            completed.append(next_value)
            segment_len = 1
            continue
        missing_count = int(round(gap / max(1.0, step))) - 1
        if 1 <= missing_count <= 2 and abs(gap / max(1.0, step) - (missing_count + 1)) <= 0.35:
            for missing_index in range(missing_count):
                completed.append(positions[index] + step * (missing_index + 1))
            segment_len += missing_count
        completed.append(next_value)
        segment_len += 1
    return _cluster_positions(completed, step * 0.25)


def _x_grid_groups(positions: list[float], step_hint: float) -> list[list[float]]:
    if len(positions) < 2:
        return []
    positions = sorted(positions)
    gaps = [positions[index + 1] - positions[index] for index in range(len(positions) - 1)]
    local_gaps = [gap for gap in gaps if step_hint * 0.55 <= gap <= step_hint * 1.55]
    step = float(np.median(local_gaps)) if local_gaps else step_hint
    groups: list[list[float]] = []
    current = [positions[0]]

    for next_value in positions[1:]:
        gap = next_value - current[-1]
        if gap > step * 1.6:
            missing_count = int(round(gap / max(1.0, step))) - 1
            if (
                1 <= missing_count
                and len(current) + missing_count + 1 <= 4
                and abs(gap / max(1.0, step) - (missing_count + 1)) <= 0.45
            ):
                for missing_index in range(missing_count):
                    current.append(current[-1] + step)
                current.append(next_value)
            else:
                while len(current) < 4:
                    current.append(current[-1] + step)
                groups.append(current[:4])
                current = [next_value]
        else:
            current.append(next_value)
            if len(current) > 4:
                groups.append(current[:4])
                current = current[4:]

    if current:
        while len(current) < 4:
            current.append(current[-1] + step)
        groups.append(current[:4])
    return [_cluster_positions(group, step * 0.25) for group in groups if len(group) >= 4]


def _split_position_groups(positions: list[float], min_len: int, max_len: int, step_hint: float) -> list[list[float]]:
    if not positions:
        return []
    positions = sorted(positions)
    gaps = [positions[index + 1] - positions[index] for index in range(len(positions) - 1)]
    local_gaps = [gap for gap in gaps if gap <= step_hint * 1.75]
    median_gap = float(np.median(local_gaps)) if local_gaps else step_hint
    segments: list[list[float]] = [[positions[0]]]
    for index, gap in enumerate(gaps):
        if gap > median_gap * 1.38:
            segments.append([positions[index + 1]])
        else:
            segments[-1].append(positions[index + 1])

    groups: list[list[float]] = []
    for segment in segments:
        if len(segment) < min_len:
            continue
        if len(segment) <= max_len:
            groups.append(segment)
            continue
        for start in range(0, len(segment), max_len):
            chunk = segment[start : start + max_len]
            if len(chunk) >= min_len:
                groups.append(chunk)
    return groups


def _cell_grid_y_groups(positions: list[float], median_side: float) -> list[list[float]]:
    if not positions:
        return []
    positions = sorted(positions)
    if len(positions) == 1:
        return []
    gaps = [positions[index + 1] - positions[index] for index in range(len(positions) - 1)]
    row_like_gaps = [gap for gap in gaps if gap <= median_side * 1.36]
    row_step = float(np.median(row_like_gaps)) if row_like_gaps else median_side * 1.12
    segments: list[list[float]] = [[positions[0]]]
    for index, gap in enumerate(gaps):
        if gap > row_step * 1.32:
            segments.append([positions[index + 1]])
        else:
            segments[-1].append(positions[index + 1])

    groups: list[list[float]] = []
    for segment in segments:
        if len(segment) < 2:
            continue
        if len(segment) <= 4:
            groups.append(segment)
            continue
        for start in range(0, len(segment), 4):
            chunk = segment[start : start + 4]
            if len(chunk) >= 2:
                groups.append(chunk)
    return groups


def _tile_board_candidates(rgb: np.ndarray) -> list[BoardCandidate]:
    tile_boxes = _tile_rect_bboxes(rgb)
    if len(tile_boxes) < 4:
        return []

    image_height, image_width = rgb.shape[:2]
    median_w = float(np.median([bbox[2] for bbox in tile_boxes]))
    median_h = float(np.median([bbox[3] for bbox in tile_boxes]))
    median_side = max(8.0, float(np.median([min(bbox[2], bbox[3]) for bbox in tile_boxes])))
    x_positions = _cluster_positions([float(bbox[0]) for bbox in tile_boxes], median_side * 0.65)
    y_positions = _cluster_positions([float(bbox[1]) for bbox in tile_boxes], median_side * 0.38)
    x_groups = _x_grid_groups(x_positions, median_side * 1.2)
    y_groups = _split_position_groups(y_positions, 1, 4, median_side * 1.25)

    candidates: list[BoardCandidate] = []
    expand = max(1, round(median_side * 0.07))
    for x_group in x_groups:
        if len(x_group) != 4:
            continue
        for y_group in y_groups:
            rows = len(y_group)
            if rows < 2 or rows > 4:
                continue
            expected = rows * 4
            evidence = 0
            for x_pos in x_group:
                for y_pos in y_group:
                    if any(
                        abs(box[0] - x_pos) <= median_side * 0.45
                        and abs(box[1] - y_pos) <= median_side * 0.45
                        for box in tile_boxes
                    ):
                        evidence += 1
            if evidence < 1:
                continue

            cell_boxes: list[tuple[int, int, list[int]]] = []
            for row, y_pos in enumerate(y_group):
                for col, x_pos in enumerate(x_group):
                    cell_boxes.append(
                        (
                            row,
                            col,
                            [
                                max(0, round(x_pos) - expand),
                                max(0, round(y_pos) - expand),
                                max(1, round(median_w) + expand * 2),
                                max(1, round(median_h) + expand * 2),
                            ],
                        )
                    )

            xs = [bbox[2][0] for bbox in cell_boxes]
            ys = [bbox[2][1] for bbox in cell_boxes]
            rights = [bbox[2][0] + bbox[2][2] for bbox in cell_boxes]
            bottoms = [bbox[2][1] + bbox[2][3] for bbox in cell_boxes]
            bbox = [min(xs), min(ys), max(rights) - min(xs), max(bottoms) - min(ys)]
            if bbox[0] < -median_side * 0.35 or bbox[1] < -median_side * 0.35:
                continue
            if bbox[0] + bbox[2] > image_width + median_side * 0.35:
                continue
            if bbox[1] + bbox[3] > image_height + median_side * 0.35:
                continue

            flags: list[str] = []
            if rows < 4:
                flags.append("partial_board_bottom_f_padding")
            evidence_ratio = evidence / max(1, expected)
            if evidence_ratio < 0.58:
                flags.append("low_grid_evidence")
            x_gaps = np.diff(sorted(x_group))
            y_gaps = np.diff(sorted(y_group)) if rows > 1 else np.array([median_side * 1.25])
            gap_variance = float(np.std(x_gaps) / max(1.0, np.mean(x_gaps)))
            if rows > 1:
                gap_variance += float(np.std(y_gaps) / max(1.0, np.mean(y_gaps)))
            fit_confidence = max(0.0, min(1.0, 0.58 + evidence_ratio * 0.34 - gap_variance * 0.2))
            if fit_confidence < 0.74:
                flags.append("low_grid_ratio_confidence")
            candidates.append(
                BoardCandidate(
                    bbox=bbox,
                    visible_rows=rows,
                    visible_cols=4,
                    fit_confidence=fit_confidence,
                    cell_boxes=cell_boxes,
                    flags=flags,
                    source="tile",
                )
            )

    return _dedupe_board_candidates(candidates)


def _grid_row_mask_evidence(
    mask: np.ndarray,
    x_group: list[float],
    y_pos: float,
    cell_w: float,
    cell_h: float,
) -> int:
    height, width = mask.shape[:2]
    evidence = 0
    for x_pos in x_group:
        x0 = max(0, int(round(x_pos)))
        y0 = max(0, int(round(y_pos)))
        x1 = min(width, int(round(x_pos + cell_w)))
        y1 = min(height, int(round(y_pos + cell_h)))
        if x1 <= x0 or y1 <= y0:
            continue
        slot = mask[y0:y1, x0:x1]
        if float((slot > 0).sum()) / max(1, slot.size) >= 0.16:
            evidence += 1
    return evidence


def _cell_grid_board_candidates(rgb: np.ndarray) -> list[BoardCandidate]:
    cell_boxes_source = _colored_cell_bboxes(rgb)
    if len(cell_boxes_source) < 4:
        return []

    tile_mask = _tile_palette_mask(rgb)
    image_height, image_width = rgb.shape[:2]
    median_w = float(np.median([bbox[2] for bbox in cell_boxes_source]))
    median_h = float(np.median([bbox[3] for bbox in cell_boxes_source]))
    median_side = max(8.0, float(np.median([min(bbox[2], bbox[3]) for bbox in cell_boxes_source])))
    x_positions = _cluster_positions([float(bbox[0]) for bbox in cell_boxes_source], median_side * 0.45)
    y_positions = _cluster_positions([float(bbox[1]) for bbox in cell_boxes_source], median_side * 0.45)
    x_groups = _x_grid_groups(x_positions, median_side * 1.15)
    y_groups = _cell_grid_y_groups(y_positions, median_side)

    candidates: list[BoardCandidate] = []
    expand = max(1, round(median_side * 0.08))
    for x_group in x_groups:
        if len(x_group) != 4:
            continue
        x_gaps = np.diff(sorted(x_group))
        if len(x_gaps) and float(np.std(x_gaps) / max(1.0, np.mean(x_gaps))) > 0.32:
            continue
        for y_group in y_groups:
            candidate_y_group = list(y_group)
            if len(candidate_y_group) in (2, 3):
                step_y = float(np.median(np.diff(candidate_y_group)))
                inferred_top_y = candidate_y_group[0] - step_y
                if inferred_top_y >= -median_side * 0.35:
                    nearby_foreign_row = any(
                        abs(existing_y - inferred_top_y) <= median_side * 0.82
                        and all(abs(existing_y - group_y) > median_side * 0.45 for group_y in candidate_y_group)
                        for existing_y in y_positions
                    )
                    if nearby_foreign_row:
                        top_evidence = 0
                    else:
                        top_evidence = _grid_row_mask_evidence(
                            tile_mask,
                            x_group,
                            inferred_top_y,
                            median_w,
                            median_h,
                        )
                    if top_evidence >= 2:
                        candidate_y_group = [inferred_top_y] + candidate_y_group

            rows = len(candidate_y_group)
            if rows < 2 or rows > 4:
                continue
            y_gaps = np.diff(sorted(candidate_y_group))
            if len(y_gaps) and float(np.std(y_gaps) / max(1.0, np.mean(y_gaps))) > 0.36:
                continue

            expected = rows * 4
            evidence = 0
            for x_pos in x_group:
                for y_pos in candidate_y_group:
                    if any(
                        abs(box[0] - x_pos) <= median_side * 0.55
                        and abs(box[1] - y_pos) <= median_side * 0.55
                        for box in cell_boxes_source
                    ):
                        evidence += 1
            min_evidence = max(3, int(math.ceil(expected * 0.34)))
            if evidence < min_evidence:
                continue

            cell_boxes: list[tuple[int, int, list[int]]] = []
            for row, y_pos in enumerate(candidate_y_group):
                for col, x_pos in enumerate(x_group):
                    cell_boxes.append(
                        (
                            row,
                            col,
                            [
                                max(0, round(x_pos) - expand),
                                max(0, round(y_pos) - expand),
                                max(1, round(median_w) + expand * 2),
                                max(1, round(median_h) + expand * 2),
                            ],
                        )
                    )

            xs = [bbox[2][0] for bbox in cell_boxes]
            ys = [bbox[2][1] for bbox in cell_boxes]
            rights = [bbox[2][0] + bbox[2][2] for bbox in cell_boxes]
            bottoms = [bbox[2][1] + bbox[2][3] for bbox in cell_boxes]
            bbox = [min(xs), min(ys), max(rights) - min(xs), max(bottoms) - min(ys)]
            if bbox[0] < -median_side * 0.35 or bbox[1] < -median_side * 0.35:
                continue
            if bbox[0] + bbox[2] > image_width + median_side * 0.35:
                continue
            if bbox[1] + bbox[3] > image_height + median_side * 0.35:
                continue

            flags: list[str] = []
            if rows < 4:
                flags.append("partial_board_bottom_f_padding")
            evidence_ratio = evidence / max(1, expected)
            if evidence_ratio < 0.60:
                flags.append("low_grid_evidence")
            gap_variance = 0.0
            if len(x_gaps):
                gap_variance += float(np.std(x_gaps) / max(1.0, np.mean(x_gaps)))
            if len(y_gaps):
                gap_variance += float(np.std(y_gaps) / max(1.0, np.mean(y_gaps)))
            fit_confidence = max(0.0, min(1.0, 0.62 + evidence_ratio * 0.34 - gap_variance * 0.24))
            if fit_confidence < 0.74:
                flags.append("low_grid_ratio_confidence")
            candidates.append(
                BoardCandidate(
                    bbox=bbox,
                    visible_rows=rows,
                    visible_cols=4,
                    fit_confidence=fit_confidence,
                    cell_boxes=cell_boxes,
                    flags=flags,
                    source="cell_grid",
                )
            )

    return _dedupe_board_candidates(candidates)


def _slot_mask_fill_ratios(mask: np.ndarray, cell_boxes: list[tuple[int, int, list[int]]]) -> list[float]:
    height, width = mask.shape[:2]
    ratios: list[float] = []
    for _row, _col, bbox in cell_boxes:
        x, y, w, h = bbox
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(width, int(x + w))
        y1 = min(height, int(y + h))
        if x1 <= x0 or y1 <= y0:
            ratios.append(0.0)
            continue
        slot = mask[y0:y1, x0:x1]
        ratios.append(float((slot > 0).sum()) / max(1, slot.size))
    return ratios


def _repair_edge_shifted_cell_grids(
    rgb: np.ndarray,
    candidates: list[BoardCandidate],
) -> list[BoardCandidate]:
    tile_mask = _tile_palette_mask(rgb)
    repaired: list[BoardCandidate] = []
    for candidate in candidates:
        ratios = _slot_mask_fill_ratios(tile_mask, candidate.cell_boxes)
        rows = candidate.visible_rows
        if len(ratios) != rows * 4 or rows < 2:
            repaired.append(candidate)
            continue

        ratios_by_col = [ratios[col::4] for col in range(4)]
        left_missing = all(ratio < 0.18 for ratio in ratios_by_col[0])
        right_missing = all(ratio < 0.18 for ratio in ratios_by_col[3])
        if left_missing == right_missing:
            repaired.append(candidate)
            continue

        x_positions = sorted({bbox[0] for _row, _col, bbox in candidate.cell_boxes})
        if len(x_positions) != 4:
            repaired.append(candidate)
            continue
        pitch = int(round(float(np.median(np.diff(x_positions)))))
        if pitch <= 0:
            repaired.append(candidate)
            continue

        dx = pitch if left_missing else -pitch
        shifted_boxes = [
            (row, col, [bbox[0] + dx, bbox[1], bbox[2], bbox[3]])
            for row, col, bbox in candidate.cell_boxes
        ]
        shifted_bbox = [
            candidate.bbox[0] + dx,
            candidate.bbox[1],
            candidate.bbox[2],
            candidate.bbox[3],
        ]
        if shifted_bbox[0] < 0 or shifted_bbox[0] + shifted_bbox[2] > rgb.shape[1]:
            repaired.append(candidate)
            continue

        shifted_ratios = _slot_mask_fill_ratios(tile_mask, shifted_boxes)
        original_support = sum(ratio >= 0.30 for ratio in ratios)
        shifted_support = sum(ratio >= 0.30 for ratio in shifted_ratios)
        if shifted_support < rows * 4 or shifted_support < original_support + rows:
            repaired.append(candidate)
            continue
        if min(shifted_ratios, default=0.0) < 0.30:
            repaired.append(candidate)
            continue

        median_fill = float(np.median(shifted_ratios))
        repaired.append(
            BoardCandidate(
                bbox=shifted_bbox,
                visible_rows=candidate.visible_rows,
                visible_cols=candidate.visible_cols,
                fit_confidence=max(candidate.fit_confidence, min(0.92, 0.76 + median_fill * 0.22)),
                cell_boxes=shifted_boxes,
                flags=list(candidate.flags) + ["edge_column_shift_recovered"],
                source=candidate.source,
            )
        )
    return repaired


def _mask_lattice_board_candidates(
    rgb: np.ndarray,
    seed_candidates: list[BoardCandidate],
    occupied_candidates: list[BoardCandidate] | None = None,
) -> list[BoardCandidate]:
    seeds = [candidate for candidate in seed_candidates if candidate.source == "cell_grid"]
    if len(seeds) < 3:
        return []
    occupied = occupied_candidates if occupied_candidates is not None else seed_candidates

    median_cell = float(
        np.median(
            [
                min(cell_bbox[2], cell_bbox[3])
                for candidate in seeds
                for _row, _col, cell_bbox in candidate.cell_boxes
            ]
        )
    )
    x_anchors = _cluster_positions([float(candidate.bbox[0]) for candidate in seeds], median_cell * 0.55)
    y_anchors = _cluster_positions([float(candidate.bbox[1]) for candidate in seeds], median_cell * 0.55)
    if len(x_anchors) < 2 or len(y_anchors) < 2:
        return []

    tile_mask = _tile_palette_mask(rgb)
    inferred: list[BoardCandidate] = []
    for y_anchor in y_anchors:
        row_seeds = [
            candidate
            for candidate in seeds
            if abs(candidate.bbox[1] - y_anchor) <= median_cell * 0.55
        ]
        if not row_seeds:
            continue
        row_template = max(row_seeds, key=lambda candidate: (candidate.fit_confidence, -candidate.bbox[0]))
        for x_anchor in x_anchors:
            column_support = any(
                abs(candidate.bbox[0] - x_anchor) <= median_cell * 0.55
                and abs(candidate.bbox[1] - y_anchor) > median_cell * 0.55
                for candidate in seeds
            )
            if not column_support:
                continue
            if any(
                abs(candidate.bbox[0] - x_anchor) <= median_cell * 0.55
                and abs(candidate.bbox[1] - y_anchor) <= median_cell * 0.55
                for candidate in seeds
            ):
                continue

            dx = int(round(x_anchor - row_template.bbox[0]))
            dy = int(round(y_anchor - row_template.bbox[1]))
            cell_boxes = [
                (row, col, [bbox[0] + dx, bbox[1] + dy, bbox[2], bbox[3]])
                for row, col, bbox in row_template.cell_boxes
            ]
            bbox = [
                row_template.bbox[0] + dx,
                row_template.bbox[1] + dy,
                row_template.bbox[2],
                row_template.bbox[3],
            ]
            if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > rgb.shape[1] or bbox[1] + bbox[3] > rgb.shape[0]:
                continue
            if any(
                _bbox_intersection(bbox, candidate.bbox)
                / max(1, min(_bbox_area(bbox), _bbox_area(candidate.bbox)))
                >= 0.25
                for candidate in occupied
            ):
                continue

            fill_ratios = _slot_mask_fill_ratios(tile_mask, cell_boxes)
            rows = row_template.visible_rows
            if len(fill_ratios) != rows * 4:
                continue
            row_support = [
                sum(ratio >= 0.30 for ratio in fill_ratios[row * 4 : (row + 1) * 4])
                for row in range(rows)
            ]
            if min(row_support, default=0) < 3:
                continue
            median_fill = float(np.median(fill_ratios))
            if median_fill < 0.42 or sum(ratio >= 0.30 for ratio in fill_ratios) < rows * 4 - 1:
                continue

            flags = ["mask_lattice_inferred"]
            if rows < 4:
                flags.append("partial_board_bottom_f_padding")
            fit_confidence = max(0.74, min(0.94, 0.62 + median_fill * 0.42))
            inferred.append(
                BoardCandidate(
                    bbox=bbox,
                    visible_rows=rows,
                    visible_cols=4,
                    fit_confidence=fit_confidence,
                    cell_boxes=cell_boxes,
                    flags=flags,
                    source="mask_lattice",
                )
            )

    return _dedupe_board_candidates(inferred)


def _tile_region_cell_boxes(
    board_bbox: list[int],
    visible_rows: int,
    visible_cols: int,
    median_side: float,
) -> list[tuple[int, int, list[int]]]:
    x, y, w, h = board_bbox
    slot_w = w / max(1, visible_cols)
    slot_h = h / max(1, visible_rows)
    tile_w = max(1.0, min(slot_w * 0.92, median_side * 1.12))
    tile_h = max(1.0, min(slot_h * 0.92, median_side * 1.12))
    cells: list[tuple[int, int, list[int]]] = []
    for row in range(visible_rows):
        for col in range(visible_cols):
            x0 = x + col * slot_w + (slot_w - tile_w) / 2.0
            y0 = y + row * slot_h + (slot_h - tile_h) / 2.0
            cells.append(
                (
                    row,
                    col,
                    [
                        int(round(x0)),
                        int(round(y0)),
                        max(1, int(round(tile_w))),
                        max(1, int(round(tile_h))),
                    ],
                )
            )
    return cells


def _tile_region_board_candidates(rgb: np.ndarray) -> list[BoardCandidate]:
    cv = _require_cv2()
    raw_tile_boxes = _tile_rect_bboxes(rgb)
    if raw_tile_boxes:
        median_side = max(8.0, float(np.median([min(bbox[2], bbox[3]) for bbox in raw_tile_boxes])))
    else:
        tile_mask_probe = _tile_palette_mask(rgb)
        raw_components = [
            bbox
            for bbox in _contour_bboxes(tile_mask_probe)
            if bbox[2] >= 8 and bbox[3] >= 8
        ]
        if not raw_components:
            return []
        median_side = max(8.0, float(np.median([min(bbox[2], bbox[3]) for bbox in raw_components])))

    mask = _tile_palette_mask(rgb)
    kernel_size = _odd_kernel(max(3, round(median_side * 0.18)))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    step = max(1.0, median_side * 1.15)
    image_height, image_width = mask.shape
    candidates: list[BoardCandidate] = []

    for bbox in _contour_bboxes(closed):
        x, y, w, h = bbox
        if w < max(22, int(round(step * 1.9))) or h < max(18, int(round(step * 0.72))):
            continue
        ratio = w / max(1, h)
        if ratio < 0.62 or ratio > 5.4:
            continue
        component = closed[y : y + h, x : x + w]
        fill_ratio = float((component > 0).sum()) / max(1, w * h)
        if fill_ratio < 0.42:
            continue

        rows_by_height = int(round(h / step))
        rows_by_ratio, ratio_confidence = _infer_visible_rows([x, y, w, h])
        if 1 <= rows_by_height <= 4:
            rows = rows_by_height
        else:
            rows = rows_by_ratio
        rows = max(1, min(4, rows))
        if rows < 2:
            continue

        expected_width = step * 4
        expected_height = step * rows
        adjusted_x = float(x)
        adjusted_y = float(y)
        adjusted_w = float(w)
        adjusted_h = float(h)
        flags: list[str] = []

        if adjusted_w < expected_width * 0.82:
            missing = expected_width - adjusted_w
            adjusted_x -= missing / 2.0
            adjusted_w = expected_width
            flags.append("tile_region_width_inferred")
        if adjusted_h < expected_height * 0.82:
            missing = expected_height - adjusted_h
            adjusted_y -= missing / 2.0
            adjusted_h = expected_height
            flags.append("tile_region_height_inferred")

        outer_expand = max(1.0, median_side * 0.08)
        adjusted_x -= outer_expand
        adjusted_y -= outer_expand
        adjusted_w += outer_expand * 2.0
        adjusted_h += outer_expand * 2.0

        adjusted_x = max(0.0, min(adjusted_x, image_width - 1.0))
        adjusted_y = max(0.0, min(adjusted_y, image_height - 1.0))
        adjusted_w = max(1.0, min(adjusted_w, image_width - adjusted_x))
        adjusted_h = max(1.0, min(adjusted_h, image_height - adjusted_y))
        adjusted_bbox = [
            int(round(adjusted_x)),
            int(round(adjusted_y)),
            int(round(adjusted_w)),
            int(round(adjusted_h)),
        ]

        width_error = abs(math.log(max(0.01, adjusted_w / max(1.0, expected_width))))
        height_error = abs(math.log(max(0.01, adjusted_h / max(1.0, expected_height))))
        fit_confidence = max(
            0.0,
            min(1.0, 0.78 + min(0.16, fill_ratio * 0.16) - (width_error + height_error) * 0.18),
        )
        if ratio_confidence < 0.55:
            flags.append("low_grid_ratio_confidence")
            fit_confidence = min(fit_confidence, 0.72)
        if rows < 4:
            flags.append("partial_board_bottom_f_padding")

        candidates.append(
            BoardCandidate(
                bbox=adjusted_bbox,
                visible_rows=rows,
                visible_cols=4,
                fit_confidence=fit_confidence,
                cell_boxes=_tile_region_cell_boxes(adjusted_bbox, rows, 4, median_side),
                flags=flags,
                source="tile_region",
            )
        )

    return _dedupe_board_candidates(candidates)


def _red_mask(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    # Highlight frames are drawn with a saturated red stroke. Pink/red tile
    # fills are intentionally excluded here because they are broad filled
    # rectangles, not annotation strokes.
    mask = (
        (red > 120)
        & (green < 125)
        & (blue < 125)
        & ((red - np.maximum(green, blue)) > 38)
    )
    return mask.astype(np.uint8) * 255


def _line_feature_mask(rgb: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    red = _red_mask(rgb) > 0
    palette = np.array(list({tuple(value) for value in DEFAULT_GUIDE_PALETTE.values()}), dtype=np.int16)
    pixels = rgb.astype(np.int16)
    nearest_palette = np.linalg.norm(pixels[:, :, None, :] - palette[None, None, :, :], axis=3).min(axis=2)
    red_channel = pixels[:, :, 0]
    blue_channel = pixels[:, :, 2]
    # Arrows in the current guide are drawn as purple/blue-purple strokes.
    # Restricting hue and excluding palette-like tile fills avoids treating the
    # book's blank purple 32768 cells as enormous arrow components.
    purple = (hue >= 105) & (hue <= 168)
    bluish_stroke = blue_channel >= red_channel + 6
    away_from_tile_fill = nearest_palette > 62
    mask = (
        purple
        & (saturation > 42)
        & (value > 145)
        & (value < 250)
        & ~red
        & (bluish_stroke | away_from_tile_fill)
    ).astype(np.uint8) * 255
    return mask


def _contour_bboxes(mask: np.ndarray) -> list[list[int]]:
    cv = _require_cv2()
    contours, _hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes: list[list[int]] = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        boxes.append([int(x), int(y), int(w), int(h)])
    return boxes


def _dedupe_bboxes(boxes: list[list[int]], overlap_threshold: float = 0.84) -> list[list[int]]:
    boxes = sorted(boxes, key=lambda bbox: _bbox_area(bbox), reverse=True)
    kept: list[list[int]] = []
    for bbox in boxes:
        area = _bbox_area(bbox)
        if area <= 0:
            continue
        duplicate = False
        for existing in kept:
            if _bbox_intersection(bbox, existing) / min(area, _bbox_area(existing)) >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(bbox)
    return sorted(kept, key=lambda bbox: (bbox[1], bbox[0]))


def _sort_board_candidates_spatial(candidates: list[BoardCandidate]) -> list[BoardCandidate]:
    if not candidates:
        return []
    median_height = float(np.median([candidate.bbox[3] for candidate in candidates]))
    tolerance = max(4.0, median_height * 0.35)
    rows: list[list[BoardCandidate]] = []
    for candidate in sorted(candidates, key=lambda item: (item.bbox[1], item.bbox[0])):
        center_y = candidate.bbox[1] + candidate.bbox[3] / 2.0
        if not rows:
            rows.append([candidate])
            continue
        row_center = float(np.median([item.bbox[1] + item.bbox[3] / 2.0 for item in rows[-1]]))
        if abs(center_y - row_center) <= tolerance:
            rows[-1].append(candidate)
        else:
            rows.append([candidate])
    sorted_rows = sorted(rows, key=lambda row: float(np.median([item.bbox[1] + item.bbox[3] / 2.0 for item in row])))
    output: list[BoardCandidate] = []
    for row in sorted_rows:
        output.extend(sorted(row, key=lambda item: item.bbox[0]))
    return output


def _dedupe_board_candidates(candidates: list[BoardCandidate]) -> list[BoardCandidate]:
    source_priority = {
        "cell_grid": 0,
        "mask_lattice": 1,
        "tile_region": 2,
        "tile": 3,
        "rough": 4,
    }
    candidates = sorted(
        candidates,
        key=lambda candidate: (
            source_priority.get(candidate.source, 3),
            -candidate.visible_rows,
            _bbox_area(candidate.bbox),
            -candidate.fit_confidence,
        ),
    )
    kept: list[BoardCandidate] = []
    for candidate in candidates:
        duplicate = False
        area = _bbox_area(candidate.bbox)
        replacement_indexes: list[int] = []
        for existing_index, existing in enumerate(kept):
            overlap = _bbox_intersection(candidate.bbox, existing.bbox)
            if overlap / max(1, min(area, _bbox_area(existing.bbox))) >= 0.62:
                if (
                    candidate.visible_rows > existing.visible_rows
                    and area <= _bbox_area(existing.bbox) * 5.6
                    and overlap / max(1, _bbox_area(existing.bbox)) >= 0.72
                ):
                    replacement_indexes.append(existing_index)
                    continue
                duplicate = True
                break
        if not duplicate:
            for existing_index in reversed(replacement_indexes):
                kept.pop(existing_index)
            kept.append(candidate)
    return _sort_board_candidates_spatial(kept)


def _infer_visible_rows(bbox: list[int]) -> tuple[int, float]:
    width = max(1, bbox[2])
    height = max(1, bbox[3])
    ratio = width / height
    best_rows = 4
    best_error = float("inf")
    for rows in (1, 2, 3, 4):
        expected_ratio = 4 / rows
        error = abs(math.log(max(0.01, ratio / expected_ratio)))
        if error < best_error:
            best_error = error
            best_rows = rows
    confidence = max(0.0, min(1.0, 1.0 - best_error / 0.42))
    return best_rows, confidence


def _rough_board_bboxes(rgb: np.ndarray) -> list[tuple[list[int], int, float, list[str]]]:
    cv = _require_cv2()
    mask = _colored_mask(rgb)
    height, width = mask.shape
    # The source images use 3-5 px white gutters between cells. The close
    # kernel must bridge gutters inside a board while staying well below the
    # white space between separate boards in n-in-one figures.
    kernel_size = _odd_kernel(max(7, round(min(width, height) * 0.04)))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    # A small dilation bridges JPEG seams inside a board without joining separated boards.
    closed = cv.dilate(closed, kernel, iterations=1)

    raw_boxes = _contour_bboxes(closed)
    candidates: list[list[int]] = []
    image_area = width * height
    for bbox in raw_boxes:
        x, y, w, h = bbox
        if w < 24 or h < 18:
            continue
        area = w * h
        if area < max(220, image_area * 0.0015):
            continue
        ratio = w / max(1, h)
        if ratio < 0.68 or ratio > 5.2:
            continue
        candidates.append([x, y, w, h])

    refined: list[tuple[list[int], int, float, list[str]]] = []
    for bbox in _dedupe_bboxes(candidates):
        rows, ratio_confidence = _infer_visible_rows(bbox)
        if rows < 2:
            continue
        if ratio_confidence < 0.2:
            continue
        flags: list[str] = []
        if ratio_confidence < 0.74:
            flags.append("low_grid_ratio_confidence")
        if rows < 4:
            flags.append("partial_board_bottom_f_padding")
        refined.append((bbox, rows, ratio_confidence, flags))

    return refined


def _rough_board_candidates(rgb: np.ndarray) -> list[BoardCandidate]:
    candidates: list[BoardCandidate] = []
    for bbox, rows, fit_confidence, flags in _rough_board_bboxes(rgb):
        candidates.append(
            BoardCandidate(
                bbox=bbox,
                visible_rows=rows,
                visible_cols=4,
                fit_confidence=fit_confidence,
                cell_boxes=_cell_boxes(bbox, rows, 4),
                flags=list(flags),
                source="rough",
            )
        )
    return candidates


def _prefer_full_tile_over_split_regions(
    tile_region_candidates: list[BoardCandidate],
    tile_candidates: list[BoardCandidate],
) -> tuple[list[BoardCandidate], list[BoardCandidate]]:
    if not tile_region_candidates or not tile_candidates:
        return tile_region_candidates, tile_candidates
    kept_regions = list(tile_region_candidates)
    promoted_tiles: list[BoardCandidate] = []
    for tile in tile_candidates:
        if tile.visible_rows != 4:
            continue
        covered_regions = []
        for region in kept_regions:
            horizontal_overlap = _bbox_intersection(
                [tile.bbox[0], region.bbox[1], tile.bbox[2], region.bbox[3]],
                region.bbox,
            ) / max(1, _bbox_area(region.bbox))
            region_overlap = _bbox_intersection(tile.bbox, region.bbox) / max(1, _bbox_area(region.bbox))
            if horizontal_overlap >= 0.72 and region_overlap >= 0.72:
                covered_regions.append(region)
        if len(covered_regions) < 2:
            continue
        rows_total = sum(region.visible_rows for region in covered_regions)
        if rows_total != 4:
            continue
        union = _union_bbox([region.bbox for region in covered_regions])
        if _bbox_intersection(tile.bbox, union) / max(1, _bbox_area(union)) < 0.90:
            continue
        promoted = BoardCandidate(
            bbox=tile.bbox,
            visible_rows=tile.visible_rows,
            visible_cols=tile.visible_cols,
            fit_confidence=max(tile.fit_confidence, 0.82),
            cell_boxes=tile.cell_boxes,
            flags=[flag for flag in tile.flags if flag != "low_grid_evidence"] + ["merged_split_tile_regions"],
            source=tile.source,
        )
        promoted_tiles.append(promoted)
        kept_regions = [region for region in kept_regions if region not in covered_regions]
    return kept_regions, promoted_tiles + tile_candidates


def _filter_undersized_board_candidates(
    rgb: np.ndarray,
    candidates: list[BoardCandidate],
) -> list[BoardCandidate]:
    tile_boxes = _tile_rect_bboxes(rgb)
    if len(tile_boxes) < 4:
        return candidates
    median_tile_side = float(np.median([min(box[2], box[3]) for box in tile_boxes]))
    minimum_cell_side = median_tile_side * 0.25
    return [
        candidate
        for candidate in candidates
        if min(
            candidate.bbox[2] / max(1, candidate.visible_cols),
            candidate.bbox[3] / max(1, candidate.visible_rows),
        )
        >= minimum_cell_side
    ]


def _variant_3x4_board_candidates(rgb: np.ndarray) -> list[BoardCandidate]:
    """Detect this guide's isolated 3x4 and 3x3 screenshot media.

    The PDF stores every state as a separate, consistently framed raster. The
    explicit profile therefore uses the media aspect ratio as structural
    evidence and only accepts images with enough cells from the guide palette.
    This deliberately excludes coordinate legends and relationship diagrams.
    """
    image_height, image_width = rgb.shape[:2]
    visible_rows = 3
    visible_cols = 4 if image_width / max(1, image_height) >= 1.15 else 3
    expected_cells = visible_rows * visible_cols
    tile_boxes = _tile_rect_bboxes(rgb)
    if len(tile_boxes) < math.ceil(expected_cells * 0.5):
        return []

    palette = np.array(
        list({tuple(value) for value in DEFAULT_GUIDE_PALETTE.values()}),
        dtype=np.float32,
    )
    matching_tiles = 0
    for tile_bbox in tile_boxes:
        tile_rgb = _crop(rgb, tile_bbox)
        if tile_rgb.size == 0:
            continue
        background = np.array(_cell_background(tile_rgb), dtype=np.float32)
        if float(np.linalg.norm(palette - background, axis=1).min()) <= 48.0:
            matching_tiles += 1
    if matching_tiles < math.ceil(expected_cells * 0.5):
        return []

    # These screenshots use a narrow white frame. Deriving the grid from the
    # complete media frame remains stable when arrows obscure a cell edge.
    inset_x = max(1, round(image_width * 0.016))
    inset_y = max(1, round(image_height * 0.02))
    bbox = [
        inset_x,
        inset_y,
        max(1, image_width - inset_x * 2),
        max(1, image_height - inset_y * 2),
    ]
    evidence_ratio = min(1.0, matching_tiles / max(1, expected_cells))
    fit_confidence = min(0.96, 0.82 + evidence_ratio * 0.14)
    return [
        BoardCandidate(
            bbox=bbox,
            visible_rows=visible_rows,
            visible_cols=visible_cols,
            fit_confidence=fit_confidence,
            cell_boxes=_cell_boxes(bbox, visible_rows, visible_cols),
            flags=[],
            source=VARIANT_3X4_PARSER_PROFILE,
        )
    ]


def _find_board_candidates(
    rgb: np.ndarray,
    profile: str = DEFAULT_PARSER_PROFILE,
) -> list[BoardCandidate]:
    if profile == VARIANT_3X4_PARSER_PROFILE:
        return _variant_3x4_board_candidates(rgb)
    sequence_strip_candidates = _sequence_strip_board_candidates(rgb)
    if sequence_strip_candidates:
        return sequence_strip_candidates
    cell_grid_candidates = _repair_edge_shifted_cell_grids(
        rgb,
        _cell_grid_board_candidates(rgb),
    )
    tile_region_candidates = _tile_region_board_candidates(rgb)
    tile_candidates = _tile_board_candidates(rgb)
    tile_region_candidates, tile_candidates = _prefer_full_tile_over_split_regions(
        tile_region_candidates,
        tile_candidates,
    )
    if tile_region_candidates:
        filtered_tile_candidates: list[BoardCandidate] = []
        for candidate in tile_candidates:
            if "low_grid_evidence" in candidate.flags and any(
                _bbox_intersection(candidate.bbox, region.bbox)
                / max(1, min(_bbox_area(candidate.bbox), _bbox_area(region.bbox)))
                >= 0.18
                for region in tile_region_candidates
            ):
                continue
            filtered_tile_candidates.append(candidate)
        tile_candidates = filtered_tile_candidates
    rough_candidates = _rough_board_candidates(rgb)
    occupied_candidates = _dedupe_board_candidates(
        cell_grid_candidates
        + tile_region_candidates
        + tile_candidates
        + rough_candidates
    )
    mask_lattice_candidates = _mask_lattice_board_candidates(
        rgb,
        cell_grid_candidates,
        occupied_candidates,
    )
    all_candidates = occupied_candidates + mask_lattice_candidates
    if all_candidates:
        return _filter_undersized_board_candidates(
            rgb,
            _dedupe_board_candidates(all_candidates),
        )
    return _filter_undersized_board_candidates(rgb, rough_candidates)


def _find_board_bboxes(rgb: np.ndarray) -> list[tuple[list[int], int, float, list[str]]]:
    return [
        (candidate.bbox, candidate.visible_rows, candidate.fit_confidence, list(candidate.flags))
        for candidate in _find_board_candidates(rgb)
    ]


def _cell_boxes(board_bbox: list[int], visible_rows: int, visible_cols: int = 4) -> list[tuple[int, int, list[int]]]:
    x, y, w, h = board_bbox
    cells: list[tuple[int, int, list[int]]] = []
    for row in range(visible_rows):
        y0 = y + round(row * h / visible_rows)
        y1 = y + round((row + 1) * h / visible_rows)
        for col in range(visible_cols):
            x0 = x + round(col * w / visible_cols)
            x1 = x + round((col + 1) * w / visible_cols)
            cells.append((row, col, [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]))
    return cells


def _inset_bbox(bbox: list[int], ratio: float) -> list[int]:
    x, y, w, h = bbox
    dx = max(1, round(w * ratio))
    dy = max(1, round(h * ratio))
    return [x + dx, y + dy, max(1, w - 2 * dx), max(1, h - 2 * dy)]


def _crop(rgb: np.ndarray, bbox: list[int]) -> np.ndarray:
    x, y, w, h = bbox
    return rgb[max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)]


def _cell_background(cell_rgb: np.ndarray) -> list[int]:
    height, width = cell_rgb.shape[:2]
    pad_x = max(1, round(width * 0.12))
    pad_y = max(1, round(height * 0.12))
    corner_w = max(2, round(width * 0.18))
    corner_h = max(2, round(height * 0.18))
    samples = [
        cell_rgb[pad_y : pad_y + corner_h, pad_x : pad_x + corner_w],
        cell_rgb[pad_y : pad_y + corner_h, width - pad_x - corner_w : width - pad_x],
        cell_rgb[height - pad_y - corner_h : height - pad_y, pad_x : pad_x + corner_w],
        cell_rgb[
            height - pad_y - corner_h : height - pad_y,
            width - pad_x - corner_w : width - pad_x,
        ],
    ]
    sample_pixels = [sample.reshape(-1, 3) for sample in samples if sample.size]
    if sample_pixels:
        pixels = np.concatenate(sample_pixels, axis=0)
    else:
        pixels = cell_rgb.reshape(-1, 3)
    median = np.median(pixels, axis=0)
    return [int(round(v)) for v in median.tolist()]


def _text_mask(cell_rgb: np.ndarray, background_rgb: list[int]) -> np.ndarray:
    mask = _raw_text_mask(cell_rgb, background_rgb)
    annotation_mask = ((_red_mask(cell_rgb) > 0) | (_line_feature_mask(cell_rgb) > 0)).astype(np.uint8) * 255
    mask[annotation_mask > 0] = 0
    return _clean_text_mask(mask)


def _clean_text_mask(mask: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    mask = (mask > 0).astype(np.uint8) * 255
    component_count, labels, stats, _centroids = cv.connectedComponentsWithStats(mask, 8)
    output = np.zeros_like(mask)
    height, width = mask.shape[:2]
    min_area = max(2, int(round(mask.size * 0.0008)))
    for component_index in range(1, component_count):
        x, y, w, h, area = [int(value) for value in stats[component_index]]
        if area < min_area:
            continue
        touches_edge = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
        if touches_edge:
            continue
        output[labels == component_index] = 255
    return output


def _raw_text_mask(cell_rgb: np.ndarray, background_rgb: list[int]) -> np.ndarray:
    cv = _require_cv2()
    background = np.array(background_rgb, dtype=np.int16).reshape(1, 1, 3)
    diff = np.abs(cell_rgb.astype(np.int16) - background).max(axis=2)
    gray = cv.cvtColor(cell_rgb, cv.COLOR_RGB2GRAY)
    mask = ((diff > 22) & (gray < 248)).astype(np.uint8) * 255
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    return cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)


def _straight_stroke_core(mask: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    height, width = mask.shape[:2]
    horizontal = cv.getStructuringElement(
        cv.MORPH_RECT,
        (max(7, int(round(width * 0.42))), 1),
    )
    vertical = cv.getStructuringElement(
        cv.MORPH_RECT,
        (1, max(7, int(round(height * 0.42)))),
    )
    horizontal_core = cv.morphologyEx(mask, cv.MORPH_OPEN, horizontal)
    vertical_core = cv.morphologyEx(mask, cv.MORPH_OPEN, vertical)
    core = cv.bitwise_or(horizontal_core, vertical_core)
    return cv.dilate(core, np.ones((3, 3), dtype=np.uint8), iterations=1)


def _looks_like_line_occluded_two(
    cell_rgb: np.ndarray,
    background_rgb: list[int],
) -> tuple[bool, float, float]:
    height, width = cell_rgb.shape[:2]
    if height < 18 or width < 18:
        return False, 0.0, 0.0

    background = np.array(background_rgb, dtype=np.float32)
    pale_distance = min(
        float(np.linalg.norm(background - np.array(DEFAULT_GUIDE_PALETTE[0], dtype=np.float32))),
        float(np.linalg.norm(background - np.array(DEFAULT_GUIDE_PALETTE[1], dtype=np.float32))),
    )
    if pale_distance > 28:
        return False, 0.0, 0.0

    raw = _raw_text_mask(cell_rgb, background_rgb)
    line_features = _line_feature_mask(cell_rgb)
    red = _red_mask(cell_rgb)
    annotation_ratio = float(((line_features > 0) | (red > 0)).sum()) / max(1, raw.size)
    if annotation_ratio < 0.005:
        return False, 0.0, 0.0

    straight_core = _straight_stroke_core(raw)
    straight_core = np.maximum(straight_core, _straight_stroke_core(line_features))
    residual = raw.copy()
    residual[(straight_core > 0) | (red > 0)] = 0
    residual = _clean_text_mask(residual)

    residual_area = int((residual > 0).sum())
    residual_ratio = residual_area / max(1, residual.size)
    bbox = _mask_bbox(residual)
    if bbox is None or residual_ratio < 0.012:
        return False, 0.0, residual_ratio

    x, y, w, h = bbox
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    centered = width * 0.30 <= center_x <= width * 0.72 and height * 0.25 <= center_y <= height * 0.78
    digit_sized = w >= width * 0.16 and h >= height * 0.28
    central_pixels = int(
        (
            residual[
                int(round(height * 0.22)) : int(round(height * 0.82)),
                int(round(width * 0.24)) : int(round(width * 0.76)),
            ]
            > 0
        ).sum()
    )
    if not centered or not digit_sized or central_pixels < max(5, int(round(residual.size * 0.008))):
        return False, 0.0, residual_ratio

    # Keep reconstructed glyphs below the general text-confidence threshold so
    # they must also pass the pale 2/empty color compatibility gate.
    score = max(0.56, min(0.59, 0.54 + residual_ratio * 2.2))
    return True, score, max(0.0, score - 0.065)


def _looks_like_occluded_two(cell_rgb: np.ndarray, background_rgb: list[int]) -> tuple[bool, float, float]:
    cv = _require_cv2()
    height, width = cell_rgb.shape[:2]
    if height < 18 or width < 18:
        return False, 0.0, 0.0

    background = np.array(background_rgb, dtype=np.float32)
    pale_distance = min(
        float(np.linalg.norm(background - np.array(DEFAULT_GUIDE_PALETTE[0], dtype=np.float32))),
        float(np.linalg.norm(background - np.array(DEFAULT_GUIDE_PALETTE[1], dtype=np.float32))),
    )
    if pale_distance > 25:
        return False, 0.0, 0.0

    raw = _raw_text_mask(cell_rgb, background_rgb)
    annotation_mask = ((_red_mask(cell_rgb) > 0) | (_line_feature_mask(cell_rgb) > 0)).astype(np.uint8) * 255
    component_count, _labels, stats, _centroids = cv.connectedComponentsWithStats(annotation_mask, 8)
    has_crossing_line = False
    for component_index in range(1, component_count):
        x, y, w, h, area = [int(value) for value in stats[component_index]]
        if w >= width * 0.55 and h <= height * 0.35 and area >= width:
            if height * 0.25 <= y + h / 2 <= height * 0.72:
                has_crossing_line = True
                break
    if not has_crossing_line:
        return False, 0.0, 0.0

    split = raw.copy()
    split[annotation_mask > 0] = 0
    component_count, labels, stats, _centroids = cv.connectedComponentsWithStats(split, 8)
    kept = np.zeros_like(split)
    kept_components = 0
    for component_index in range(1, component_count):
        x, y, w, h, area = [int(value) for value in stats[component_index]]
        if area < max(2, int(round(split.size * 0.002))):
            continue
        if min(w, h) <= 1:
            continue
        if area / max(1, w * h) < 0.18:
            continue
        kept[labels == component_index] = 255
        kept_components += 1

    area = int((kept > 0).sum())
    area_ratio = area / max(1, height * width)
    bbox = _mask_bbox(kept)
    if bbox is None:
        return False, 0.0, area_ratio
    x, y, w, h = bbox
    centered_enough = x <= width * 0.58 and x + w >= width * 0.38
    shaped_enough = w >= width * 0.35 and h >= height * 0.22
    area_enough = 0.035 <= area_ratio <= 0.18
    if kept_components >= 1 and centered_enough and shaped_enough and area_enough:
        score = max(0.52, min(0.62, 0.48 + area_ratio * 1.1))
        runner_up = max(0.0, score - 0.055)
        return True, score, runner_up
    return False, 0.0, area_ratio


def _looks_like_marked_two(cell_rgb: np.ndarray, background_rgb: list[int]) -> tuple[bool, float, float]:
    cv = _require_cv2()
    height, width = cell_rgb.shape[:2]
    if height < 18 or width < 18:
        return False, 0.0, 0.0

    background = np.array(background_rgb, dtype=np.float32)
    pale_distance = min(
        float(np.linalg.norm(background - np.array(DEFAULT_GUIDE_PALETTE[0], dtype=np.float32))),
        float(np.linalg.norm(background - np.array(DEFAULT_GUIDE_PALETTE[1], dtype=np.float32))),
    )
    if pale_distance > 28:
        return False, 0.0, 0.0

    raw = _raw_text_mask(cell_rgb, background_rgb)
    annotation_mask = ((_red_mask(cell_rgb) > 0) | (_line_feature_mask(cell_rgb) > 0)).astype(np.uint8) * 255
    annotation_ratio = float((annotation_mask > 0).sum()) / max(1, height * width)
    raw_ratio = float((raw > 0).sum()) / max(1, height * width)
    if annotation_ratio < 0.01 or raw_ratio < 0.045:
        return False, 0.0, raw_ratio

    split = raw.copy()
    split[annotation_mask > 0] = 0
    component_count, labels, stats, _centroids = cv.connectedComponentsWithStats(split, 8)
    best_score = 0.0
    best_area_ratio = 0.0
    for component_index in range(1, component_count):
        x, y, w, h, area = [int(value) for value in stats[component_index]]
        if area < max(4, int(round(split.size * 0.012))):
            continue
        area_ratio = area / max(1, height * width)
        if area_ratio > 0.20:
            continue
        if h < height * 0.18 or w < width * 0.16:
            continue
        if h > height * 0.46:
            continue
        if w > width * 0.88 and h > height * 0.34:
            continue
        center_x = x + w / 2.0
        centered = width * 0.20 <= center_x <= width * 0.82
        not_just_line = h >= height * 0.24 or area / max(1, w * h) >= 0.32
        if not centered or not not_just_line:
            continue
        score = 0.50 + min(0.12, area_ratio * 0.8) + min(0.06, annotation_ratio * 0.7)
        if score > best_score:
            best_score = score
            best_area_ratio = area_ratio

    if best_score >= 0.54:
        return True, min(0.64, best_score), max(0.0, best_score - 0.06)
    return False, 0.0, best_area_ratio


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _normalize_binary_mask(mask: np.ndarray, size: int = 64) -> np.ndarray | None:
    cv = _require_cv2()
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    x, y, w, h = bbox
    if w < 2 or h < 2:
        return None
    crop = mask[y : y + h, x : x + w]
    scale = min((size - 8) / max(1, w), (size - 8) / max(1, h))
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    resized = cv.resize(crop, (resized_w, resized_h), interpolation=cv.INTER_NEAREST)
    output = np.zeros((size, size), dtype=np.uint8)
    ox = (size - resized_w) // 2
    oy = (size - resized_h) // 2
    output[oy : oy + resized_h, ox : ox + resized_w] = resized
    return output > 0


class TemplateMatcher:
    def __init__(self, repo_root: Path | None = None):
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.templates = self._build_templates()

    def _font(self, size: int):
        font_path = self.repo_root / "font" / "ClearSans" / "ClearSans-Bold.ttf"
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
        return ImageFont.load_default()

    def _render_template(self, value: int, canvas_size: int = 64) -> np.ndarray:
        text = str(value)
        target_w = canvas_size - 8
        target_h = canvas_size - 14
        font_size = 44
        while font_size > 10:
            font = self._font(font_size)
            probe = Image.new("L", (canvas_size, canvas_size), 0)
            draw = ImageDraw.Draw(probe)
            bbox = draw.textbbox((0, 0), text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width <= target_w and height <= target_h:
                break
            font_size -= 1

        image = Image.new("L", (canvas_size, canvas_size), 0)
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        x = (canvas_size - width) / 2 - bbox[0]
        y = (canvas_size - height) / 2 - bbox[1]
        draw.text((x, y), text, fill=255, font=font)
        return _normalize_binary_mask(np.array(image))  # type: ignore[return-value]

    def _build_templates(self) -> dict[int, np.ndarray]:
        templates: dict[int, np.ndarray] = {}
        for exponent in VISIBLE_VALUE_EXPONENTS:
            templates[exponent] = self._render_template(exponent_to_value(exponent))
        return templates

    @staticmethod
    def _score(observed: np.ndarray, template: np.ndarray) -> float:
        intersection = np.logical_and(observed, template).sum()
        union = np.logical_or(observed, template).sum()
        if union == 0:
            return 0.0
        iou = intersection / union
        agreement = (observed == template).mean()
        return float(iou * 0.82 + agreement * 0.18)

    def match(self, mask: np.ndarray, cell_area: int) -> TextMatch:
        area = int((mask > 0).sum())
        area_ratio = area / max(1, cell_area)
        normalized = _normalize_binary_mask(mask)
        if normalized is None or area_ratio < 0.0035:
            return TextMatch(0, 0, 0.0, 0.0, float(area_ratio))

        scores = [
            (exponent, self._score(normalized, template))
            for exponent, template in self.templates.items()
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        exponent, score = scores[0]
        runner_up = scores[1][1] if len(scores) > 1 else 0.0
        return TextMatch(
            exponent=exponent,
            value=exponent_to_value(exponent),
            score=float(score),
            runner_up_score=float(runner_up),
            area_ratio=float(area_ratio),
        )


def _probe_boards(
    image_id: str,
    rgb: np.ndarray,
    matcher: TemplateMatcher,
    profile: str = DEFAULT_PARSER_PROFILE,
) -> list[BoardProbe]:
    probes: list[BoardProbe] = []
    for board_index, candidate in enumerate(_find_board_candidates(rgb, profile)):
        board_id = f"{image_id}_b{board_index:02d}"
        cells: list[CellProbe] = []
        for row, col, cell_bbox in candidate.cell_boxes:
            inner_bbox = _inset_bbox(cell_bbox, 0.04)
            cell_rgb = _crop(rgb, inner_bbox)
            if cell_rgb.size == 0:
                continue
            background = _cell_background(cell_rgb)
            mask = _text_mask(cell_rgb, background)
            text_match = matcher.match(mask, cell_rgb.shape[0] * cell_rgb.shape[1])
            line_occluded_two, marked_score, marked_runner_up = _looks_like_line_occluded_two(
                cell_rgb,
                background,
            )
            marked_two, fallback_score, fallback_runner_up = _looks_like_marked_two(cell_rgb, background)
            if line_occluded_two:
                text_match = TextMatch(
                    exponent=1,
                    value=2,
                    score=marked_score,
                    runner_up_score=marked_runner_up,
                    area_ratio=max(text_match.area_ratio, 0.04),
                )
            elif marked_two and (text_match.exponent != 1 or text_match.score < fallback_score):
                text_match = TextMatch(
                    exponent=1,
                    value=2,
                    score=fallback_score,
                    runner_up_score=fallback_runner_up,
                    area_ratio=max(text_match.area_ratio, 0.04),
                )
            elif text_match.exponent != 1:
                occluded_two, score, runner_up = _looks_like_occluded_two(cell_rgb, background)
                if occluded_two:
                    text_match = TextMatch(
                        exponent=1,
                        value=2,
                        score=score,
                        runner_up_score=runner_up,
                        area_ratio=max(text_match.area_ratio, 0.04),
                    )
            cells.append(
                CellProbe(
                    row=row,
                    col=col,
                    bbox=cell_bbox,
                    background_rgb=background,
                    text_match=text_match,
                    text_area_ratio=text_match.area_ratio,
                )
            )
        if len(cells) >= candidate.visible_rows * candidate.visible_cols:
            probes.append(
                BoardProbe(
                    board_id,
                    candidate.bbox,
                    candidate.visible_rows,
                    candidate.visible_cols,
                    candidate.fit_confidence,
                    cells,
                    list(candidate.flags),
                )
            )
    return probes


def _median_rgb(colors: list[list[int]]) -> list[int]:
    if not colors:
        return [0, 0, 0]
    return [int(round(v)) for v in np.median(np.array(colors, dtype=np.float32), axis=0)]


def _build_palette(all_boards: list[BoardProbe]) -> dict[int, list[int]]:
    samples: dict[int, list[list[int]]] = {exponent: [] for exponent in range(16)}
    for board in all_boards:
        for cell in board.cells:
            match = cell.text_match
            if match.exponent > 0 and match.score >= TEXT_ACCEPT_THRESHOLD and (match.score - match.runner_up_score) >= 0.03:
                samples[match.exponent].append(cell.background_rgb)
            elif match.area_ratio < 0.0035:
                samples[0].append(cell.background_rgb)

    palette = {exponent: list(rgb) for exponent, rgb in DEFAULT_GUIDE_PALETTE.items()}
    for exponent, colors in samples.items():
        if exponent in (0, 1):
            continue
        if not colors:
            continue
        median = _median_rgb(colors)
        default = palette.get(exponent)
        if default is None:
            continue
        distance = float(np.linalg.norm(np.array(median, dtype=np.float32) - np.array(default, dtype=np.float32)))
        if len(colors) >= 3 and distance <= 45:
            palette[exponent] = median
    return palette


def _classify_color(background_rgb: list[int], palette: dict[int, list[int]]) -> tuple[int, float, float]:
    if not palette:
        return 0, 0.0, 0.0
    color = np.array(background_rgb, dtype=np.float32)
    distances = []
    for exponent, palette_rgb in palette.items():
        distance = float(np.linalg.norm(color - np.array(palette_rgb, dtype=np.float32)))
        distances.append((exponent, distance))
    distances.sort(key=lambda item: item[1])
    exponent, distance = distances[0]
    second = distances[1][1] if len(distances) > 1 else distance + 80
    confidence = max(0.0, min(1.0, 1.0 - distance / 82.0))
    margin = max(0.0, min(1.0, (second - distance) / 70.0))
    return int(exponent), float(confidence), float(margin)


def _cell_record(cell: CellProbe, palette: dict[int, list[int]]) -> tuple[dict[str, Any], bool]:
    text = cell.text_match
    color_exp, color_confidence, color_margin = _classify_color(cell.background_rgb, palette)
    flags: list[str] = []
    needs_review = False

    text_confident = (
        text.exponent > 0
        and text.score >= TEXT_ACCEPT_THRESHOLD
        and (text.score - text.runner_up_score) >= 0.025
        and text.area_ratio >= 0.018
    )
    low_two_color_compatible = color_exp in (0, 1) or (
        color_exp == 2 and color_margin < 0.03 and color_confidence < 0.84
    )
    low_two_text_confident = (
        text.exponent == 1
        and low_two_color_compatible
        and text.score >= 0.34
        and (text.score - text.runner_up_score) >= 0.045
        and text.area_ratio >= 0.015
    )
    fallback_text_confident = (
        text.exponent > 1
        and text.score >= 0.54
        and (text.score - text.runner_up_score) >= 0.018
        and text.area_ratio >= 0.018
        and color_confidence < 0.25
    )
    color_confident = color_exp > 1 and color_confidence >= COLOR_ACCEPT_THRESHOLD and (
        color_margin >= 0.035 or color_confidence >= 0.92
    )
    empty_confident = text.area_ratio < 0.0035 and (
        not palette or color_exp in (0, 1) or color_confidence < 0.58
    )
    actual_4096_text = (
        text.exponent == 12
        and text.score >= 0.45
        and text.area_ratio >= 0.014
        and (text.score - text.runner_up_score) >= 0.018
    )
    blank_purple_32768 = (
        color_exp == 12
        and color_confidence >= 0.58
        and text.area_ratio < 0.016
        and not actual_4096_text
    )

    if text_confident or low_two_text_confident or fallback_text_confident:
        exponent = text.exponent
        confidence = min(1.0, text.score + (0.18 if text_confident else 0.24))
        if low_two_text_confident and not text_confident:
            flags.append("low_two_text")
        if fallback_text_confident and not text_confident:
            flags.append("low_text_unknown_color")
            needs_review = True
        if color_confident and color_exp not in (0, exponent):
            flags.append("text_color_conflict")
            needs_review = True
            confidence = min(confidence, 0.64)
        elif not color_confident:
            flags.append("text_only")
            confidence = min(confidence, 0.78)
    elif blank_purple_32768:
        exponent = 15
        confidence = max(0.82, color_confidence)
        flags.append("blank_purple_32768")
    elif empty_confident:
        exponent = 0
        confidence = max(0.78, color_confidence if color_exp == 0 else 0.78)
    elif color_confident and color_exp > 0:
        exponent = color_exp
        confidence = color_confidence
        flags.append("color_only")
        needs_review = True
    else:
        exponent = color_exp if color_confident else 0
        confidence = max(text.score, color_confidence)
        flags.append("low_value_confidence")
        needs_review = True

    return (
        {
            "row": cell.row,
            "col": cell.col,
            "bbox": cell.bbox,
            "exponent": int(exponent),
            "value": exponent_to_value(int(exponent)),
            "hex_digit": format(int(exponent), "x"),
            "confidence": round(float(confidence), 4),
            "background_rgb": cell.background_rgb,
            "text": {
                "exponent": text.exponent,
                "value": text.value,
                "score": round(text.score, 4),
                "runner_up_score": round(text.runner_up_score, 4),
                "area_ratio": round(text.area_ratio, 6),
            },
            "color": {
                "exponent": color_exp,
                "value": exponent_to_value(color_exp),
                "confidence": round(color_confidence, 4),
                "margin": round(color_margin, 4),
            },
            "flags": flags,
        },
        needs_review,
    )


def _padding_cell(row: int, col: int, digit: str, edge: str) -> dict[str, Any]:
    exponent = int(digit, 16)
    return {
        "row": row,
        "col": col,
        "bbox": None,
        "exponent": exponent,
        "value": exponent_to_value(exponent),
        "hex_digit": digit,
        "confidence": 1.0,
        "background_rgb": None,
        "flags": [f"padding_{edge}_{digit}"],
    }


def _finalize_board(
    board: BoardProbe,
    palette: dict[int, list[int]],
    profile: str = DEFAULT_PARSER_PROFILE,
) -> tuple[dict[str, Any], bool]:
    cells_by_position: dict[tuple[int, int], dict[str, Any]] = {}
    cell_records: list[dict[str, Any]] = []
    needs_review = board.fit_confidence < 0.74
    min_confidence = board.fit_confidence
    flags = list(board.flags)

    for cell in board.cells:
        record, cell_needs_review = _cell_record(cell, palette)
        cells_by_position[(cell.row, cell.col)] = record
        cell_records.append(record)
        needs_review = needs_review or cell_needs_review
        min_confidence = min(min_confidence, float(record["confidence"]))

    padding = _padding_for_profile(profile)
    digits: list[str] = []
    for row in range(4):
        for col in range(4):
            if row < board.visible_rows and col < board.visible_cols:
                record = cells_by_position.get((row, col))
                if record is None:
                    digits.append("0")
                    flags.append("missing_visible_cell")
                    needs_review = True
                else:
                    digits.append(str(record["hex_digit"]))
            elif row < board.visible_rows:
                digits.append(padding.right)
                cell_records.append(_padding_cell(row, col, padding.right, "right"))
            else:
                digits.append(padding.bottom)
                cell_records.append(_padding_cell(row, col, padding.bottom, "bottom"))

    if min_confidence < BOARD_ACCEPT_THRESHOLD:
        needs_review = True
        flags.append("low_board_confidence")

    return (
        {
            "board_id": board.board_id,
            "bbox": board.bbox,
            "visible_rows": board.visible_rows,
            "visible_cols": board.visible_cols,
            "padding": {
                "right": padding.right,
                "bottom": padding.bottom,
            },
            "hex": normalize_hex_16("".join(digits)),
            "cells": sorted(cell_records, key=lambda item: (item["row"], item["col"])),
            "confidence": round(float(min_confidence), 4),
            "flags": sorted(set(flags)),
        },
        needs_review,
    )


def _nearest_board(point: tuple[float, float], boards: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not boards:
        return None
    px, py = point
    best: tuple[float, dict[str, Any]] | None = None
    for board in boards:
        cx, cy = _center_of_bbox(board["bbox"])
        distance = (px - cx) ** 2 + (py - cy) ** 2
        if best is None or distance < best[0]:
            best = (distance, board)
    return best[1] if best else None


def _nearest_board_by_edge(point: tuple[float, float], boards: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not boards:
        return None
    px, py = point
    best: tuple[float, dict[str, Any]] | None = None
    for board in boards:
        x, y, w, h = board["bbox"]
        dx = max(x - px, 0.0, px - (x + w))
        dy = max(y - py, 0.0, py - (y + h))
        distance = dx * dx + dy * dy
        if best is None or distance < best[0]:
            best = (distance, board)
    return best[1] if best else None


def _cell_for_point(point: tuple[float, float], board: dict[str, Any], margin: float = 0) -> tuple[int, int] | None:
    candidates: list[tuple[float, tuple[int, int]]] = []
    for cell in board.get("cells", []):
        bbox = cell.get("bbox")
        if bbox and _point_in_bbox(point, bbox, margin):
            cx, cy = _center_of_bbox(bbox)
            distance = (point[0] - cx) ** 2 + (point[1] - cy) ** 2
            candidates.append((distance, (int(cell["row"]), int(cell["col"]))))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _union_bbox(boxes: list[list[int]], expand: int = 0) -> list[int]:
    xs = [bbox[0] for bbox in boxes]
    ys = [bbox[1] for bbox in boxes]
    rights = [bbox[0] + bbox[2] for bbox in boxes]
    bottoms = [bbox[1] + bbox[3] for bbox in boxes]
    x0 = min(xs) - expand
    y0 = min(ys) - expand
    x1 = max(rights) + expand
    y1 = max(bottoms) + expand
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def _edge_coverages(mask: np.ndarray, bbox: list[int], thickness: int) -> tuple[float, float, float, float]:
    height, width = mask.shape[:2]
    x, y, w, h = bbox
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0, 0.0, 0.0
    band = max(1, thickness)

    top = mask[max(0, y0 - band) : min(height, y0 + band + 1), x0:x1]
    bottom = mask[max(0, y1 - band - 1) : min(height, y1 + band), x0:x1]
    left = mask[y0:y1, max(0, x0 - band) : min(width, x0 + band + 1)]
    right = mask[y0:y1, max(0, x1 - band - 1) : min(width, x1 + band)]

    top_cov = float(np.any(top > 0, axis=0).mean()) if top.size else 0.0
    bottom_cov = float(np.any(bottom > 0, axis=0).mean()) if bottom.size else 0.0
    left_cov = float(np.any(left > 0, axis=1).mean()) if left.size else 0.0
    right_cov = float(np.any(right > 0, axis=1).mean()) if right.size else 0.0
    return top_cov, bottom_cov, left_cov, right_cov


def _rectangle_edges_connected(mask: np.ndarray, bbox: list[int], thickness: int) -> bool:
    cv = _require_cv2()
    height, width = mask.shape[:2]
    x, y, w, h = bbox
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return False
    band = max(1, thickness)
    crop = mask[y0:y1, x0:x1]
    outline = np.zeros_like(crop, dtype=np.uint8)
    outline[: min(crop.shape[0], band + 1), :] = 255
    outline[max(0, crop.shape[0] - band - 1) :, :] = 255
    outline[:, : min(crop.shape[1], band + 1)] = 255
    outline[:, max(0, crop.shape[1] - band - 1) :] = 255
    edge = ((crop > 0) & (outline > 0)).astype(np.uint8) * 255
    total = int((edge > 0).sum())
    if total < max(8, int(round((w + h) * 0.18))):
        return False
    component_count, labels, stats, _centroids = cv.connectedComponentsWithStats(edge, 8)
    for component_index in range(1, component_count):
        area = int(stats[component_index, cv.CC_STAT_AREA])
        if area / max(1, total) < 0.72:
            continue
        component = labels == component_index
        top = component[: min(component.shape[0], band + 1), :]
        bottom = component[max(0, component.shape[0] - band - 1) :, :]
        left = component[:, : min(component.shape[1], band + 1)]
        right = component[:, max(0, component.shape[1] - band - 1) :]
        top_cov = float(np.any(top, axis=0).mean()) if top.size else 0.0
        bottom_cov = float(np.any(bottom, axis=0).mean()) if bottom.size else 0.0
        left_cov = float(np.any(left, axis=1).mean()) if left.size else 0.0
        right_cov = float(np.any(right, axis=1).mean()) if right.size else 0.0
        if min(top_cov, bottom_cov, left_cov, right_cov) >= 0.30:
            return True
    return False


def _cells_to_range(cells: list[list[int]]) -> tuple[int, int, int, int] | None:
    if not cells:
        return None
    rows = [int(cell[0]) for cell in cells]
    cols = [int(cell[1]) for cell in cells]
    row0, row1 = min(rows), max(rows)
    col0, col1 = min(cols), max(cols)
    if len(cells) != (row1 - row0 + 1) * (col1 - col0 + 1):
        return None
    return row0, row1, col0, col1


def _line_span_coverage(mask: np.ndarray, bbox: list[int], horizontal: bool, thickness: int) -> float:
    x, y, w, h = bbox
    height, width = mask.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    band = max(1, thickness)
    if horizontal:
        cy = (y0 + y1) // 2
        crop = mask[max(0, cy - band) : min(height, cy + band + 1), x0:x1]
        return float(np.any(crop > 0, axis=0).mean()) if crop.size else 0.0
    cx = (x0 + x1) // 2
    crop = mask[y0:y1, max(0, cx - band) : min(width, cx + band + 1)]
    return float(np.any(crop > 0, axis=1).mean()) if crop.size else 0.0


def _internal_row_separator_coverage(
    mask: np.ndarray,
    cells_by_position: dict[tuple[int, int], dict[str, Any]],
    row: int,
    col0: int,
    col1: int,
    thickness: int,
) -> float:
    top_boxes = [cells_by_position[(row, col)]["bbox"] for col in range(col0, col1 + 1)]
    bottom_boxes = [cells_by_position[(row + 1, col)]["bbox"] for col in range(col0, col1 + 1)]
    x0 = min(box[0] for box in top_boxes + bottom_boxes)
    x1 = max(box[0] + box[2] for box in top_boxes + bottom_boxes)
    y_top = max(box[1] + box[3] for box in top_boxes)
    y_bottom = min(box[1] for box in bottom_boxes)
    y = int(round((y_top + y_bottom) / 2.0))
    return _line_span_coverage(mask, [x0, y - thickness, x1 - x0, thickness * 2 + 1], True, thickness)


def _internal_col_separator_coverage(
    mask: np.ndarray,
    cells_by_position: dict[tuple[int, int], dict[str, Any]],
    row0: int,
    row1: int,
    col: int,
    thickness: int,
) -> float:
    left_boxes = [cells_by_position[(row, col)]["bbox"] for row in range(row0, row1 + 1)]
    right_boxes = [cells_by_position[(row, col + 1)]["bbox"] for row in range(row0, row1 + 1)]
    y0 = min(box[1] for box in left_boxes + right_boxes)
    y1 = max(box[1] + box[3] for box in left_boxes + right_boxes)
    x_left = max(box[0] + box[2] for box in left_boxes)
    x_right = min(box[0] for box in right_boxes)
    x = int(round((x_left + x_right) / 2.0))
    return _line_span_coverage(mask, [x - thickness, y0, thickness * 2 + 1, y1 - y0], False, thickness)


def _split_highlight_by_internal_separators(
    annotation: dict[str, Any],
    mask: np.ndarray,
    cells_by_position: dict[tuple[int, int], dict[str, Any]],
    thickness: int,
) -> list[dict[str, Any]]:
    cell_range = _cells_to_range(annotation.get("cells") or [])
    if cell_range is None:
        return [annotation]
    row0, row1, col0, col1 = cell_range
    row_separators = [
        _internal_row_separator_coverage(mask, cells_by_position, row, col0, col1, thickness)
        for row in range(row0, row1)
    ]
    col_separators = [
        _internal_col_separator_coverage(mask, cells_by_position, row0, row1, col, thickness)
        for col in range(col0, col1)
    ]
    has_full_row_separators = bool(row_separators) and min(row_separators) >= 0.72
    has_full_col_separators = bool(col_separators) and min(col_separators) >= 0.72
    if has_full_row_separators and not has_full_col_separators:
        output = []
        for row in range(row0, row1 + 1):
            cells = [[row, col] for col in range(col0, col1 + 1)]
            item = dict(annotation)
            item["annotation_id"] = f"{annotation.get('annotation_id', 'highlight')}_row_{row}"
            item["cells"] = cells
            item["bbox"] = _union_bbox([cells_by_position[(row, col)]["bbox"] for col in range(col0, col1 + 1)])
            item["flags"] = sorted(set((item.get("flags") or []) + ["split_by_internal_row_separators"]))
            output.append(item)
        return output
    if has_full_col_separators and not has_full_row_separators:
        output = []
        for col in range(col0, col1 + 1):
            cells = [[row, col] for row in range(row0, row1 + 1)]
            item = dict(annotation)
            item["annotation_id"] = f"{annotation.get('annotation_id', 'highlight')}_col_{col}"
            item["cells"] = cells
            item["bbox"] = _union_bbox([cells_by_position[(row, col)]["bbox"] for row in range(row0, row1 + 1)])
            item["flags"] = sorted(set((item.get("flags") or []) + ["split_by_internal_col_separators"]))
            output.append(item)
        return output
    return [annotation]


def _dedupe_highlights(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotations = sorted(
        annotations,
        key=lambda item: (
            -len(item.get("cells") or []),
            -float(item.get("confidence", 0.0)),
            item.get("board_id") or "",
        ),
    )
    kept: list[dict[str, Any]] = []
    for annotation in annotations:
        cells = {tuple(cell) for cell in annotation.get("cells") or []}
        duplicate = False
        for existing in kept:
            if annotation.get("board_id") != existing.get("board_id"):
                continue
            existing_cells = {tuple(cell) for cell in existing.get("cells") or []}
            if cells and existing_cells:
                if cells < existing_cells:
                    duplicate = True
                    break
                jaccard = len(cells & existing_cells) / max(1, len(cells | existing_cells))
                if jaccard >= 0.86:
                    duplicate = True
                    break
            if annotation.get("bbox") and existing.get("bbox"):
                overlap = _bbox_intersection(annotation["bbox"], existing["bbox"])
                if overlap / max(1, min(_bbox_area(annotation["bbox"]), _bbox_area(existing["bbox"]))) >= 0.88:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(annotation)
    return sorted(kept, key=lambda item: (item.get("bbox", [0, 0, 0, 0])[1], item.get("bbox", [0, 0, 0, 0])[0]))


def _detect_grid_highlights(mask: np.ndarray, boards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cv = _require_cv2()
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    edge_mask = cv.dilate(mask, kernel, iterations=1)
    annotations: list[dict[str, Any]] = []
    for board in boards:
        cells_by_position = {
            (int(cell["row"]), int(cell["col"])): cell
            for cell in board.get("cells", [])
            if cell.get("bbox") is not None
        }
        if not cells_by_position:
            continue
        rows = int(board.get("visible_rows") or 4)
        cols = int(board.get("visible_cols") or 4)
        cell_boxes = [cell["bbox"] for cell in cells_by_position.values()]
        median_cell = float(np.median([min(bbox[2], bbox[3]) for bbox in cell_boxes]))
        thickness = max(2, round(median_cell * 0.06))
        expand = max(1, round(median_cell * 0.04))

        for row0 in range(rows):
            for row1 in range(row0, rows):
                for col0 in range(cols):
                    for col1 in range(col0, cols):
                        range_cells = [
                            [row, col]
                            for row in range(row0, row1 + 1)
                            for col in range(col0, col1 + 1)
                            if (row, col) in cells_by_position
                        ]
                        if len(range_cells) != (row1 - row0 + 1) * (col1 - col0 + 1):
                            continue
                        bbox = _union_bbox(
                            [cells_by_position[(row, col)]["bbox"] for row, col in range_cells],
                            expand=expand,
                        )
                        top, bottom, left, right = _edge_coverages(edge_mask, bbox, thickness)
                        coverages = [top, bottom, left, right]
                        strong_edges = sum(coverage >= 0.38 for coverage in coverages)
                        if strong_edges < 4:
                            continue
                        if not _rectangle_edges_connected(edge_mask, bbox, thickness):
                            continue
                        confidence = max(0.0, min(1.0, sum(coverages) / 4.0))
                        if confidence < 0.42:
                            continue
                        annotation = {
                            "type": "highlight",
                            "annotation_id": f"highlight_grid_{len(annotations):02d}",
                            "bbox": bbox,
                            "board_id": board.get("board_id"),
                            "cells": sorted(range_cells),
                            "confidence": round(max(0.6, confidence), 4),
                            "flags": [],
                        }
                        annotations.extend(
                            _split_highlight_by_internal_separators(
                                annotation,
                                edge_mask,
                                cells_by_position,
                                thickness,
                            )
                        )

    return _dedupe_highlights(annotations)


def _detect_highlights(rgb: np.ndarray, boards: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    cv = _require_cv2()
    mask = _red_mask(rgb)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    annotations: list[dict[str, Any]] = _detect_grid_highlights(closed, boards)
    residual_components = 0

    for bbox in _contour_bboxes(closed):
        x, y, w, h = bbox
        if w < 8 or h < 8:
            continue
        if any(
            annotation.get("bbox")
            and _bbox_intersection(bbox, annotation["bbox"]) / max(1, _bbox_area(bbox)) >= 0.35
            for annotation in annotations
        ):
            continue
        component = closed[y : y + h, x : x + w]
        fill_ratio = float((component > 0).sum()) / max(1, w * h)
        # Solid red/orange tiles are not highlights. Rectangular borders have
        # relatively low fill ratio but visible edge coverage.
        if fill_ratio > 0.42:
            continue
        small_cell_mark = False
        for board in boards:
            for cell in board.get("cells", []):
                cell_bbox = cell.get("bbox")
                if not cell_bbox or _bbox_intersection(bbox, cell_bbox) <= 0:
                    continue
                if w < cell_bbox[2] * 0.62 and h < cell_bbox[3] * 0.62:
                    small_cell_mark = True
                    break
            if small_cell_mark:
                break
        if small_cell_mark:
            continue
        matched_cells: list[list[int]] = []
        matched_board = None
        for board in boards:
            if _bbox_intersection(bbox, board["bbox"]) <= 0:
                continue
            for cell in board.get("cells", []):
                cell_bbox = cell.get("bbox")
                if not cell_bbox:
                    continue
                overlap = _bbox_intersection(bbox, cell_bbox)
                if overlap / max(1, _bbox_area(cell_bbox)) > 0.12 or _point_in_bbox(
                    _center_of_bbox(cell_bbox), bbox, margin=3
                ):
                    matched_cells.append([int(cell["row"]), int(cell["col"])])
                    matched_board = board
        if matched_cells:
            if matched_board:
                visible_cell_boxes = [
                    cell["bbox"]
                    for cell in matched_board.get("cells", [])
                    if cell.get("bbox")
                ]
                if visible_cell_boxes:
                    median_cell_w = float(np.median([cell_bbox[2] for cell_bbox in visible_cell_boxes]))
                    median_cell_h = float(np.median([cell_bbox[3] for cell_bbox in visible_cell_boxes]))
                    if w < median_cell_w * 0.45 and h < median_cell_h * 0.45:
                        continue
                    unique_cells = sorted(set(tuple(cell) for cell in matched_cells))
                    if len(unique_cells) == 1:
                        cell_bbox = next(
                            (
                                cell.get("bbox")
                                for cell in matched_board.get("cells", [])
                                if [int(cell["row"]), int(cell["col"])] == list(unique_cells[0])
                            ),
                            None,
                        )
                        if cell_bbox:
                            cell_w = int(cell_bbox[2])
                            cell_h = int(cell_bbox[3])
                            # Text remnants inside orange/red tiles can satisfy the red mask.
                            # A single-cell frame must reach most of the tile extent.
                            if w < cell_w * 0.74 or h < cell_h * 0.74:
                                continue
                    visible_cell_count = int(matched_board.get("visible_rows", 4)) * int(
                        matched_board.get("visible_cols", 4)
                    )
                    if len(set(tuple(cell) for cell in matched_cells)) >= max(1, int(visible_cell_count * 0.75)):
                        continue
            annotations.append(
                {
                    "type": "highlight",
                    "annotation_id": f"highlight_contour_{len(annotations):02d}",
                    "bbox": bbox,
                    "board_id": matched_board["board_id"] if matched_board else None,
                    "cells": sorted(matched_cells),
                    "confidence": round(max(0.55, min(1.0, 1.0 - fill_ratio)), 4),
                    "flags": [],
                }
            )
        else:
            residual_components += 1

    return _dedupe_highlights(annotations), residual_components


def _line_key(line: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = line
    if (x2, y2) < (x1, y1):
        return (x2, y2, x1, y1)
    return (x1, y1, x2, y2)


def _dedupe_lines(lines: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for line in sorted(lines, key=lambda item: -math.hypot(item[2] - item[0], item[3] - item[1])):
        duplicate = False
        x1, y1, x2, y2 = _line_key(line)
        for existing in kept:
            ex1, ey1, ex2, ey2 = _line_key(existing)
            if max(abs(x1 - ex1), abs(y1 - ey1), abs(x2 - ex2), abs(y2 - ey2)) <= 4:
                duplicate = True
                break
        if not duplicate:
            kept.append(line)
    return kept


def _axis_endpoints_from_bbox(bbox: list[int]) -> tuple[tuple[float, float], tuple[float, float]]:
    x, y, w, h = bbox
    if w >= h:
        return (float(x), float(y + h / 2.0)), (float(x + w), float(y + h / 2.0))
    return (float(x + w / 2.0), float(y)), (float(x + w / 2.0), float(y + h))


def _endpoint_arrow_score(
    mask: np.ndarray,
    endpoint: tuple[float, float],
    other: tuple[float, float],
    radius: float,
) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    endpoint_vector = np.array(endpoint, dtype=np.float32)
    other_vector = np.array(other, dtype=np.float32)
    direction = other_vector - endpoint_vector
    length = float(np.linalg.norm(direction))
    if length <= 0:
        return 0.0
    direction /= length
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    relative = points - endpoint_vector
    along = relative @ direction
    perp = np.abs(relative @ perpendicular)
    tip_limit = min(6.5, radius * 0.48)
    tip = (along >= -2.0) & (along <= tip_limit) & (perp <= radius * 1.15)
    cap = (along >= -radius * 0.18) & (along <= radius * 0.85) & (perp <= radius * 0.90)
    if not np.any(tip) and not np.any(cap):
        return 0.0
    tip_perp = perp[tip]
    cap_perp = perp[cap]
    tip_area = float(tip.sum())
    tip_spread = float(np.percentile(tip_perp, 92)) if len(tip_perp) else 0.0
    broad_tip = float((tip_perp >= radius * 0.16).sum()) if len(tip_perp) else 0.0
    cap_area = float(cap.sum())
    cap_spread = float(np.percentile(cap_perp, 92)) if len(cap_perp) else 0.0
    return tip_area * 1.65 + tip_spread * 6.0 + broad_tip * 3.2 + cap_area * 0.35 + cap_spread * 1.2


def _orient_segment_by_arrowhead(
    mask: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    min_cell: int,
) -> tuple[tuple[float, float], tuple[float, float], list[str]]:
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    radius = max(5.0, min(float(min_cell) * 0.34, length * 0.32))
    start_score = _endpoint_arrow_score(mask, start, end, radius)
    end_score = _endpoint_arrow_score(mask, end, start, radius)
    flags: list[str] = []
    if start_score > end_score * 1.10 and start_score - end_score >= 6.0:
        return end, start, flags
    if end_score > start_score * 1.10 and end_score - start_score >= 6.0:
        return start, end, flags
    flags.append("direction_inferred_from_geometry")
    return start, end, flags


def _line_annotation_duplicate(annotation: dict[str, Any], annotations: list[dict[str, Any]]) -> bool:
    bbox = annotation.get("bbox")
    if not bbox:
        return False
    for existing in annotations:
        if annotation.get("type") == "arrow" and existing.get("type") == "arrow":
            if annotation.get("board_id") == existing.get("board_id"):
                annotation_cells = {tuple(annotation.get("from_cell") or []), tuple(annotation.get("to_cell") or [])}
                existing_cells = {tuple(existing.get("from_cell") or []), tuple(existing.get("to_cell") or [])}
                if annotation_cells == existing_cells:
                    return True
        if annotation.get("type") == "connector" and existing.get("type") == "connector":
            annotation_boards = {annotation.get("from_board"), annotation.get("to_board")}
            existing_boards = {existing.get("from_board"), existing.get("to_board")}
            if annotation_boards == existing_boards:
                return True
        existing_bbox = existing.get("bbox")
        if not existing_bbox:
            continue
        overlap = _bbox_intersection(bbox, existing_bbox)
        if overlap / max(1, min(_bbox_area(bbox), _bbox_area(existing_bbox))) >= 0.58:
            if annotation.get("type") == existing.get("type") or {
                annotation.get("type"),
                existing.get("type"),
            } == {"arrow", "connector"}:
                return True
    return False


def _dedupe_line_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for annotation in sorted(
        annotations,
        key=lambda item: (
            -float(item.get("confidence", 0.0)),
            len(item.get("flags") or []),
            -_bbox_area(item.get("bbox", [0, 0, 0, 0])),
            item.get("bbox", [0, 0, 0, 0])[1],
        ),
    ):
        if not _line_annotation_duplicate(annotation, kept):
            kept.append(annotation)
    kept = _merge_continuous_arrows(kept)
    kept = _harmonize_parallel_arrow_groups(kept)
    return sorted(kept, key=lambda item: (item.get("bbox", [0, 0, 0, 0])[1], item.get("bbox", [0, 0, 0, 0])[0]))


def _arrow_axis(annotation: dict[str, Any]) -> tuple[str, str, int, int, int, int] | None:
    if annotation.get("type") != "arrow":
        return None
    board_id = annotation.get("board_id")
    from_cell = annotation.get("from_cell")
    to_cell = annotation.get("to_cell")
    if not board_id or not isinstance(from_cell, list) or not isinstance(to_cell, list):
        return None
    if len(from_cell) != 2 or len(to_cell) != 2:
        return None
    fr, fc = int(from_cell[0]), int(from_cell[1])
    tr, tc = int(to_cell[0]), int(to_cell[1])
    if fr == tr and fc != tc:
        sign = 1 if tc > fc else -1
        return str(board_id), "row", fr, sign, min(fc, tc), max(fc, tc)
    if fc == tc and fr != tr:
        sign = 1 if tr > fr else -1
        return str(board_id), "col", fc, sign, min(fr, tr), max(fr, tr)
    return None


def _arrow_from_axis(
    template: dict[str, Any],
    axis: tuple[str, str, int, int],
    start: int,
    end: int,
    bbox: list[int],
    confidence: float,
    flags: list[str],
    index: int,
) -> dict[str, Any]:
    board_id, axis_kind, lane, sign = axis
    if axis_kind == "row":
        from_cell = [lane, start] if sign > 0 else [lane, end]
        to_cell = [lane, end] if sign > 0 else [lane, start]
    else:
        from_cell = [start, lane] if sign > 0 else [end, lane]
        to_cell = [end, lane] if sign > 0 else [start, lane]
    output = dict(template)
    output.update(
        {
            "type": "arrow",
            "annotation_id": f"merged_arrow_{index:02d}",
            "board_id": board_id,
            "from_cell": from_cell,
            "to_cell": to_cell,
            "bbox": bbox,
            "confidence": round(float(confidence), 4),
            "flags": sorted(set(flags)),
        }
    )
    return output


def _merge_continuous_arrows(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arrows_by_axis: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for annotation in annotations:
        axis = _arrow_axis(annotation)
        if axis is None:
            passthrough.append(annotation)
            continue
        board_id, axis_kind, lane, sign, start, end = axis
        item = dict(annotation)
        item["_axis_interval"] = [start, end]
        item["_axis_sign"] = sign
        arrows_by_axis.setdefault((board_id, axis_kind, lane), []).append(item)

    merged: list[dict[str, Any]] = []
    merge_index = 0
    for axis, items in arrows_by_axis.items():
        items = sorted(items, key=lambda item: (item["_axis_interval"][0], item["_axis_interval"][1]))
        current: list[dict[str, Any]] = []
        current_start = 0
        current_end = -1
        for item in items:
            start, end = [int(value) for value in item["_axis_interval"]]
            if not current:
                current = [item]
                current_start = start
                current_end = end
                continue
            if start <= current_end + 1:
                current.append(item)
                current_end = max(current_end, end)
            else:
                merged.append(
                    _merged_arrow_from_items(axis, current, current_start, current_end, merge_index)
                )
                merge_index += 1
                current = [item]
                current_start = start
                current_end = end
        if current:
            merged.append(_merged_arrow_from_items(axis, current, current_start, current_end, merge_index))
            merge_index += 1

    merged = _suppress_contained_reverse_arrows(merged)
    return passthrough + merged


def _merged_arrow_from_items(
    axis: tuple[str, str, int],
    items: list[dict[str, Any]],
    start: int,
    end: int,
    merge_index: int,
) -> dict[str, Any]:
    sign_scores = {1: 0.0, -1: 0.0}
    sign_counts = {1: 0, -1: 0}
    for item in items:
        interval_start, interval_end = [int(value) for value in item["_axis_interval"]]
        length = max(1, interval_end - interval_start)
        sign = int(item["_axis_sign"])
        confidence = float(item.get("confidence", 0.0))
        sign_scores[sign] += length * confidence
        sign_counts[sign] += 1
    if sign_scores[1] == sign_scores[-1]:
        chosen_sign = int(max(items, key=lambda item: float(item.get("confidence", 0.0)))["_axis_sign"])
    else:
        chosen_sign = 1 if sign_scores[1] > sign_scores[-1] else -1

    bbox = _union_bbox([item.get("bbox", [0, 0, 0, 0]) for item in items], expand=0)
    confidence = max(float(item.get("confidence", 0.0)) for item in items)
    flags: list[str] = []
    for item in items:
        flags.extend(str(flag) for flag in (item.get("flags") or []))
    if sign_counts[1] and sign_counts[-1]:
        flags.append("merged_opposite_direction_fragments")
    template = max(
        [item for item in items if int(item["_axis_sign"]) == chosen_sign],
        key=lambda item: float(item.get("confidence", 0.0)),
    )
    output = _arrow_from_axis(
        template,
        (axis[0], axis[1], axis[2], chosen_sign),
        start,
        end,
        bbox,
        confidence,
        flags,
        merge_index,
    )
    output.pop("_axis_interval", None)
    output.pop("_axis_sign", None)
    return output


def _suppress_contained_reverse_arrows(arrows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    axis_cache = {id(arrow): _arrow_axis(arrow) for arrow in arrows}
    for arrow in arrows:
        axis = axis_cache[id(arrow)]
        if axis is None:
            kept.append(arrow)
            continue
        board_id, axis_kind, lane, sign, start, end = axis
        length = end - start
        suppress = False
        for other in arrows:
            if other is arrow:
                continue
            other_axis = axis_cache[id(other)]
            if other_axis is None:
                continue
            other_board, other_axis_kind, other_lane, other_sign, other_start, other_end = other_axis
            if (board_id, axis_kind, lane) != (other_board, other_axis_kind, other_lane):
                continue
            other_length = other_end - other_start
            if other_sign == sign or other_length <= length:
                continue
            if start >= other_start and end <= other_end and (
                "direction_inferred_from_geometry" in (arrow.get("flags") or [])
                or other_length >= length + 1
                or float(other.get("confidence", 0.0)) >= float(arrow.get("confidence", 0.0))
            ):
                suppress = True
                break
        if not suppress:
            kept.append(arrow)
    return kept


def _arrow_visual_weight(annotation: dict[str, Any]) -> float:
    bbox = annotation.get("bbox") or [0, 0, 0, 0]
    length = float(max(int(bbox[2] or 0), int(bbox[3] or 0), 1))
    confidence = max(0.0, min(1.0, float(annotation.get("confidence", 0.0))))
    weight = length * (0.55 + confidence)
    if "direction_inferred_from_geometry" in (annotation.get("flags") or []):
        weight *= 0.82
    return weight


def _flip_arrow_direction(annotation: dict[str, Any]) -> dict[str, Any]:
    output = dict(annotation)
    output["from_cell"], output["to_cell"] = annotation.get("to_cell"), annotation.get("from_cell")
    flags = set(str(flag) for flag in (annotation.get("flags") or []))
    flags.add("direction_harmonized_with_parallel_group")
    output["flags"] = sorted(flags)
    return output


def _harmonize_parallel_arrow_groups(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], list[tuple[dict[str, Any], tuple[str, str, int, int, int, int]]]] = {}
    passthrough: list[dict[str, Any]] = []
    for annotation in annotations:
        axis = _arrow_axis(annotation)
        if axis is None:
            passthrough.append(annotation)
            continue
        board_id, axis_kind, _lane, _sign, start, end = axis
        groups.setdefault((board_id, axis_kind, start, end), []).append((annotation, axis))

    harmonized: list[dict[str, Any]] = []
    for items in groups.values():
        if len(items) < 2:
            harmonized.extend(annotation for annotation, _axis in items)
            continue

        sign_scores = {1: 0.0, -1: 0.0}
        sign_counts = {1: 0, -1: 0}
        for annotation, axis in items:
            sign = int(axis[3])
            sign_scores[sign] += _arrow_visual_weight(annotation)
            sign_counts[sign] += 1

        if not sign_counts[1] or not sign_counts[-1]:
            harmonized.extend(annotation for annotation, _axis in items)
            continue

        positive = sign_scores[1]
        negative = sign_scores[-1]
        if positive == 0 and negative == 0:
            harmonized.extend(annotation for annotation, _axis in items)
            continue

        if sign_counts[1] != sign_counts[-1]:
            chosen_sign = 1 if sign_counts[1] > sign_counts[-1] else -1
        else:
            dominant = max(positive, negative)
            weaker = max(1e-6, min(positive, negative))
            if dominant / weaker < 1.12:
                harmonized.extend(annotation for annotation, _axis in items)
                continue
            chosen_sign = 1 if positive > negative else -1

        for annotation, axis in items:
            if int(axis[3]) == chosen_sign:
                harmonized.append(annotation)
            else:
                harmonized.append(_flip_arrow_direction(annotation))
    return passthrough + harmonized


def _detect_connector_components(
    mask: np.ndarray,
    boards: list[dict[str, Any]],
    min_cell: int,
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for index, bbox in enumerate(_contour_bboxes(mask)):
        x, y, w, h = bbox
        longest = max(w, h)
        shortest = max(1, min(w, h))
        if longest < min_cell * 0.45 or longest / shortest < 2.0:
            continue
        center = (x + w / 2.0, y + h / 2.0)
        if any(_point_in_bbox(center, board["bbox"], margin=2) for board in boards):
            continue
        start, end = _axis_endpoints_from_bbox(bbox)
        local_x0 = max(0, x - 2)
        local_y0 = max(0, y - 2)
        local = mask[local_y0 : y + h + 2, local_x0 : x + w + 2]
        from_point, to_point, flags = _orient_segment_by_arrowhead(
            local,
            (start[0] - local_x0, start[1] - local_y0),
            (end[0] - local_x0, end[1] - local_y0),
            min_cell,
        )
        from_point = (from_point[0] + local_x0, from_point[1] + local_y0)
        to_point = (to_point[0] + local_x0, to_point[1] + local_y0)
        nearest_start = _nearest_board_by_edge(from_point, boards)
        nearest_end = _nearest_board_by_edge(to_point, boards)
        if not nearest_start or not nearest_end or nearest_start["board_id"] == nearest_end["board_id"]:
            continue
        annotations.append(
            {
                "type": "connector",
                "annotation_id": f"connector_component_{index:02d}",
                "bbox": bbox,
                "from_board": nearest_start["board_id"],
                "to_board": nearest_end["board_id"],
                "confidence": round(min(1.0, max(0.58, longest / max(1.0, min_cell * 2.0))), 4),
                "flags": flags,
            }
        )
    return annotations


def _detect_line_components(
    mask: np.ndarray,
    boards: list[dict[str, Any]],
    min_cell: int,
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for index, bbox in enumerate(_contour_bboxes(mask)):
        x, y, w, h = bbox
        longest = max(w, h)
        shortest = max(1, min(w, h))
        if longest < min_cell * 0.5 or longest / shortest < 1.65:
            continue
        fill_ratio = float((mask[y : y + h, x : x + w] > 0).sum()) / max(1, w * h)
        if fill_ratio > 0.72:
            continue
        dx = w
        dy = h
        is_axis_aligned = dx <= max(4, dy * 0.34) or dy <= max(4, dx * 0.34)
        if not is_axis_aligned:
            continue

        start, end = _axis_endpoints_from_bbox(bbox)
        local_x0 = max(0, x - 2)
        local_y0 = max(0, y - 2)
        local = mask[local_y0 : y + h + 2, local_x0 : x + w + 2]
        from_point, to_point, flags = _orient_segment_by_arrowhead(
            local,
            (start[0] - local_x0, start[1] - local_y0),
            (end[0] - local_x0, end[1] - local_y0),
            min_cell,
        )
        from_point = (from_point[0] + local_x0, from_point[1] + local_y0)
        to_point = (to_point[0] + local_x0, to_point[1] + local_y0)

        overlaps = [
            (_bbox_intersection(bbox, board["bbox"]) / max(1, _bbox_area(bbox)), board)
            for board in boards
            if _bbox_intersection(bbox, board["bbox"]) > 0
        ]
        overlaps.sort(key=lambda item: item[0], reverse=True)
        midpoint = ((from_point[0] + to_point[0]) / 2.0, (from_point[1] + to_point[1]) / 2.0)
        midpoint_board = next((board for board in boards if _point_in_bbox(midpoint, board["bbox"], 2)), None)
        dominant_board = overlaps[0][1] if overlaps and overlaps[0][0] >= 0.58 else midpoint_board

        if dominant_board is not None:
            from_cell = _cell_for_point(from_point, dominant_board, margin=min_cell * 0.32)
            to_cell = _cell_for_point(to_point, dominant_board, margin=min_cell * 0.32)
            if from_cell and to_cell and from_cell != to_cell:
                annotation = {
                    "type": "arrow",
                    "annotation_id": f"component_line_{index:02d}",
                    "bbox": bbox,
                    "board_id": dominant_board["board_id"],
                    "from_cell": list(from_cell),
                    "to_cell": list(to_cell),
                    "confidence": round(min(1.0, max(0.62, longest / max(1.0, min_cell * 2.0))), 4),
                    "flags": flags,
                }
                if not _line_annotation_duplicate(annotation, annotations):
                    annotations.append(annotation)
                continue

        nearest_start = _nearest_board_by_edge(from_point, boards)
        nearest_end = _nearest_board_by_edge(to_point, boards)
        if nearest_start and nearest_end and nearest_start["board_id"] != nearest_end["board_id"]:
            annotation = {
                "type": "connector",
                "annotation_id": f"component_line_{index:02d}",
                "bbox": bbox,
                "from_board": nearest_start["board_id"],
                "to_board": nearest_end["board_id"],
                "confidence": round(min(1.0, max(0.58, longest / max(1.0, min_cell * 2.0))), 4),
                "flags": flags,
            }
            if not _line_annotation_duplicate(annotation, annotations):
                annotations.append(annotation)
    return annotations


def _projection_runs(projection: np.ndarray, max_gap: int) -> list[tuple[int, int, float, int]]:
    cv = _require_cv2()
    if projection.size == 0:
        return []
    gap = max(1, int(max_gap))
    binary = projection.astype(np.uint8).reshape(1, -1) * 255
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (gap + 1, 1))
    closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=1).reshape(-1) > 0
    runs: list[tuple[int, int, float, int]] = []
    index = 0
    while index < len(closed):
        if not closed[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(closed) and closed[index + 1]:
            index += 1
        end = index
        raw = projection[start : end + 1]
        raw_count = int(raw.sum())
        raw_coverage = float(raw.mean()) if raw.size else 0.0
        runs.append((start, end, raw_coverage, raw_count))
        index += 1
    return runs


def _axis_run_line_seed_count(
    line_boxes: list[list[int]],
    board_bbox: list[int],
    horizontal: bool,
    lane_position: int,
    run_start: int,
    run_end: int,
    thickness: int,
    min_cell: int,
) -> int:
    count = 0
    lane_tolerance = thickness * 2 + 1
    for bbox in line_boxes:
        x, y, w, h = bbox
        if _bbox_intersection(bbox, board_bbox) <= 0:
            continue
        cx = x + w / 2.0
        cy = y + h / 2.0
        if horizontal:
            if abs(cy - lane_position) > lane_tolerance:
                continue
            if x + w < run_start or x > run_end:
                continue
            if w >= min_cell * 0.28 and w / max(1, h) >= 1.75:
                count += 1
        else:
            if abs(cx - lane_position) > lane_tolerance:
                continue
            if y + h < run_start or y > run_end:
                continue
            if h >= min_cell * 0.28 and h / max(1, w) >= 1.75:
                count += 1
    return count


def _fragmented_axis_run_allowed(
    visible_rows: int,
    horizontal: bool,
    run_length: int,
    raw_coverage: float,
    raw_count: int,
    min_cell: int,
) -> bool:
    if horizontal:
        return False
    if visible_rows > 2:
        return False
    return (
        run_length >= min_cell * 1.22
        and raw_count >= max(8, round(min_cell * 0.50))
        and raw_coverage >= 0.30
    )


def _detect_axis_run_arrows(
    mask: np.ndarray,
    boards: list[dict[str, Any]],
    min_cell: int,
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    line_boxes = _contour_bboxes(mask)
    for board in boards:
        cells_by_position = {
            (int(cell["row"]), int(cell["col"])): cell
            for cell in board.get("cells", [])
            if cell.get("bbox") is not None
        }
        if not cells_by_position:
            continue
        rows = int(board.get("visible_rows") or 4)
        cols = int(board.get("visible_cols") or 4)
        cell_boxes = [cell["bbox"] for cell in cells_by_position.values()]
        board_min_cell = max(1, int(round(np.median([min(bbox[2], bbox[3]) for bbox in cell_boxes]))))
        thickness = max(2, round(board_min_cell * 0.10))
        max_gap = max(3, round(board_min_cell * 0.82))
        min_run_length = max(8, round(board_min_cell * 0.78))
        min_raw_count = max(5, round(board_min_cell * 0.42))
        min_seed_count = 1 if rows <= 2 else 2
        min_axis_coverage = 0.20 if rows <= 2 else 0.45

        for row in range(rows):
            row_boxes = [cells_by_position[(row, col)]["bbox"] for col in range(cols) if (row, col) in cells_by_position]
            if len(row_boxes) < 2:
                continue
            y = int(round(np.median([bbox[1] + bbox[3] / 2.0 for bbox in row_boxes])))
            x0 = min(bbox[0] for bbox in row_boxes)
            x1 = max(bbox[0] + bbox[2] for bbox in row_boxes)
            crop = mask[max(0, y - thickness) : y + thickness + 1, x0:x1]
            if crop.size == 0:
                continue
            projection = np.any(crop > 0, axis=0)
            for start, end, raw_coverage, raw_count in _projection_runs(projection, max_gap):
                run_length = end - start + 1
                if run_length < min_run_length or raw_count < min_raw_count or raw_coverage < min_axis_coverage:
                    continue
                seed_count = _axis_run_line_seed_count(
                    line_boxes,
                    board["bbox"],
                    True,
                    y,
                    x0 + start,
                    x0 + end,
                    thickness,
                    board_min_cell,
                )
                fragmented_flags: list[str] = []
                if seed_count < min_seed_count:
                    if not _fragmented_axis_run_allowed(
                        rows,
                        True,
                        run_length,
                        raw_coverage,
                        raw_count,
                        board_min_cell,
                    ):
                        continue
                    fragmented_flags.append("fragmented_axis_run")
                start_point = (float(x0 + start), float(y))
                end_point = (float(x0 + end), float(y))
                local_x0 = max(0, int(start_point[0]) - thickness - 2)
                local_y0 = max(0, y - thickness - 2)
                local_x1 = min(mask.shape[1], int(end_point[0]) + thickness + 3)
                local_y1 = min(mask.shape[0], y + thickness + 3)
                local = mask[local_y0:local_y1, local_x0:local_x1]
                from_point, to_point, flags = _orient_segment_by_arrowhead(
                    local,
                    (start_point[0] - local_x0, start_point[1] - local_y0),
                    (end_point[0] - local_x0, end_point[1] - local_y0),
                    board_min_cell,
                )
                from_point = (from_point[0] + local_x0, from_point[1] + local_y0)
                to_point = (to_point[0] + local_x0, to_point[1] + local_y0)
                from_cell = _cell_for_point(from_point, board, margin=board_min_cell * 0.45)
                to_cell = _cell_for_point(to_point, board, margin=board_min_cell * 0.45)
                if not from_cell or not to_cell or from_cell == to_cell:
                    continue
                annotations.append(
                    {
                        "type": "arrow",
                        "annotation_id": f"axis_run_{len(annotations):02d}",
                        "bbox": [int(x0 + start), int(y - thickness), int(run_length), int(thickness * 2 + 1)],
                        "board_id": board["board_id"],
                        "from_cell": list(from_cell),
                        "to_cell": list(to_cell),
                        "confidence": round(min(1.0, max(0.56, run_length / max(1.0, board_min_cell * 2.4), raw_coverage)), 4),
                        "flags": sorted(set(flags + fragmented_flags)),
                    }
                )

        for col in range(cols):
            col_boxes = [cells_by_position[(row, col)]["bbox"] for row in range(rows) if (row, col) in cells_by_position]
            if len(col_boxes) < 2:
                continue
            x = int(round(np.median([bbox[0] + bbox[2] / 2.0 for bbox in col_boxes])))
            y0 = min(bbox[1] for bbox in col_boxes)
            y1 = max(bbox[1] + bbox[3] for bbox in col_boxes)
            crop = mask[y0:y1, max(0, x - thickness) : x + thickness + 1]
            if crop.size == 0:
                continue
            projection = np.any(crop > 0, axis=1)
            for start, end, raw_coverage, raw_count in _projection_runs(projection, max_gap):
                run_length = end - start + 1
                if run_length < min_run_length or raw_count < min_raw_count or raw_coverage < min_axis_coverage:
                    continue
                seed_count = _axis_run_line_seed_count(
                    line_boxes,
                    board["bbox"],
                    False,
                    x,
                    y0 + start,
                    y0 + end,
                    thickness,
                    board_min_cell,
                )
                fragmented_flags = []
                if seed_count < min_seed_count:
                    if not _fragmented_axis_run_allowed(
                        rows,
                        False,
                        run_length,
                        raw_coverage,
                        raw_count,
                        board_min_cell,
                    ):
                        continue
                    fragmented_flags.append("fragmented_axis_run")
                start_point = (float(x), float(y0 + start))
                end_point = (float(x), float(y0 + end))
                local_x0 = max(0, x - thickness - 2)
                local_y0 = max(0, int(start_point[1]) - thickness - 2)
                local_x1 = min(mask.shape[1], x + thickness + 3)
                local_y1 = min(mask.shape[0], int(end_point[1]) + thickness + 3)
                local = mask[local_y0:local_y1, local_x0:local_x1]
                from_point, to_point, flags = _orient_segment_by_arrowhead(
                    local,
                    (start_point[0] - local_x0, start_point[1] - local_y0),
                    (end_point[0] - local_x0, end_point[1] - local_y0),
                    board_min_cell,
                )
                from_point = (from_point[0] + local_x0, from_point[1] + local_y0)
                to_point = (to_point[0] + local_x0, to_point[1] + local_y0)
                from_cell = _cell_for_point(from_point, board, margin=board_min_cell * 0.45)
                to_cell = _cell_for_point(to_point, board, margin=board_min_cell * 0.45)
                if not from_cell or not to_cell or from_cell == to_cell:
                    continue
                annotations.append(
                    {
                        "type": "arrow",
                        "annotation_id": f"axis_run_{len(annotations):02d}",
                        "bbox": [int(x - thickness), int(y0 + start), int(thickness * 2 + 1), int(run_length)],
                        "board_id": board["board_id"],
                        "from_cell": list(from_cell),
                        "to_cell": list(to_cell),
                        "confidence": round(min(1.0, max(0.56, run_length / max(1.0, board_min_cell * 2.4), raw_coverage)), 4),
                        "flags": sorted(set(flags + fragmented_flags)),
                    }
                )
    return annotations


def _reorient_inferred_arrows_from_full_bbox(
    mask: np.ndarray,
    annotations: list[dict[str, Any]],
    boards: list[dict[str, Any]],
    min_cell: int,
) -> list[dict[str, Any]]:
    boards_by_id = {board.get("board_id"): board for board in boards}
    refined: list[dict[str, Any]] = []
    for annotation in annotations:
        if annotation.get("type") != "arrow":
            refined.append(annotation)
            continue
        if "direction_inferred_from_geometry" not in (annotation.get("flags") or []):
            refined.append(annotation)
            continue
        bbox = annotation.get("bbox")
        board = boards_by_id.get(annotation.get("board_id"))
        if not bbox or board is None:
            refined.append(annotation)
            continue

        x, y, w, h = [int(value) for value in bbox]
        if max(w, h) < min_cell * 0.45:
            refined.append(annotation)
            continue
        if not (w <= max(4, h * 0.34) or h <= max(4, w * 0.34)):
            refined.append(annotation)
            continue

        start, end = _axis_endpoints_from_bbox([x, y, w, h])
        local_x0 = max(0, x - 4)
        local_y0 = max(0, y - 4)
        local = mask[local_y0 : y + h + 4, local_x0 : x + w + 4]
        from_point, to_point, orientation_flags = _orient_segment_by_arrowhead(
            local,
            (start[0] - local_x0, start[1] - local_y0),
            (end[0] - local_x0, end[1] - local_y0),
            min_cell,
        )
        if "direction_inferred_from_geometry" in orientation_flags:
            refined.append(annotation)
            continue

        from_point = (from_point[0] + local_x0, from_point[1] + local_y0)
        to_point = (to_point[0] + local_x0, to_point[1] + local_y0)
        from_cell = _cell_for_point(from_point, board, margin=min_cell * 0.45)
        to_cell = _cell_for_point(to_point, board, margin=min_cell * 0.45)
        if not from_cell or not to_cell or from_cell == to_cell:
            refined.append(annotation)
            continue

        output = dict(annotation)
        output["from_cell"] = list(from_cell)
        output["to_cell"] = list(to_cell)
        flags = set(str(flag) for flag in (annotation.get("flags") or []))
        flags.discard("direction_inferred_from_geometry")
        flags.add("direction_refined_from_full_bbox")
        output["flags"] = sorted(flags)
        refined.append(output)
    return refined


def _detect_lines(rgb: np.ndarray, boards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cv = _require_cv2()
    if not boards:
        return []
    raw_mask = _line_feature_mask(rgb)
    if int((raw_mask > 0).sum()) == 0:
        return []
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask = cv.morphologyEx(raw_mask, cv.MORPH_CLOSE, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=1)
    edges = cv.Canny(mask, 50, 150)
    min_cell = min(
        max(1, cell["bbox"][2])
        for board in boards
        for cell in board.get("cells", [])
        if cell.get("bbox")
    )
    annotations: list[dict[str, Any]] = _detect_line_components(mask, boards, min_cell)
    annotations.extend(_detect_axis_run_arrows(mask, boards, min_cell))
    annotations.extend(_detect_connector_components(mask, boards, min_cell))

    raw_lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(6, round(min_cell * 0.15)),
        minLineLength=max(8, round(min_cell * 0.42)),
        maxLineGap=max(2, round(min_cell * 0.12)),
    )
    if raw_lines is None:
        return _reorient_inferred_arrows_from_full_bbox(
            mask,
            _dedupe_line_annotations(annotations),
            boards,
            min_cell,
        )

    normalized_lines = np.asarray(raw_lines).reshape(-1, 4)
    lines = _dedupe_lines([tuple(map(int, line)) for line in normalized_lines])
    for index, (x1, y1, x2, y2) in enumerate(lines):
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_cell * 0.35:
            continue
        start = (float(x1), float(y1))
        end = (float(x2), float(y2))
        line_bbox = [
            min(x1, x2),
            min(y1, y2),
            abs(x2 - x1) + 1,
            abs(y2 - y1) + 1,
        ]
        lx, ly, lw, lh = line_bbox
        local_x0 = max(0, lx - 3)
        local_y0 = max(0, ly - 3)
        local = mask[local_y0 : ly + lh + 3, local_x0 : lx + lw + 3]
        from_point, to_point, flags = _orient_segment_by_arrowhead(
            local,
            (start[0] - local_x0, start[1] - local_y0),
            (end[0] - local_x0, end[1] - local_y0),
            min_cell,
        )
        start = (from_point[0] + local_x0, from_point[1] + local_y0)
        end = (to_point[0] + local_x0, to_point[1] + local_y0)
        midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        start_board = next((board for board in boards if _point_in_bbox(start, board["bbox"], 2)), None)
        end_board = next((board for board in boards if _point_in_bbox(end, board["bbox"], 2)), None)
        midpoint_board = next((board for board in boards if _point_in_bbox(midpoint, board["bbox"], 2)), None)
        confidence = min(1.0, max(0.5, length / max(1.0, min_cell * 2.2)))

        if start_board and end_board and start_board["board_id"] == end_board["board_id"]:
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            is_axis_aligned = dx <= max(3, dy * 0.28) or dy <= max(3, dx * 0.28)
            if length < min_cell * 0.9 or not is_axis_aligned:
                continue
            from_cell = _cell_for_point(start, start_board, margin=min_cell * 0.28)
            to_cell = _cell_for_point(end, end_board, margin=min_cell * 0.28)
            if not from_cell or not to_cell or from_cell == to_cell:
                continue
            annotation = {
                "type": "arrow",
                "annotation_id": f"line_{index:02d}",
                "bbox": line_bbox,
                "board_id": start_board["board_id"],
                "from_cell": list(from_cell),
                "to_cell": list(to_cell),
                "confidence": round(confidence, 4),
                "flags": flags,
            }
            if not _line_annotation_duplicate(annotation, annotations):
                annotations.append(annotation)
        else:
            if midpoint_board is not None or length < min_cell * 0.42:
                continue
            nearest_start = start_board or _nearest_board_by_edge(start, boards)
            nearest_end = end_board or _nearest_board_by_edge(end, boards)
            if nearest_start and nearest_end and nearest_start["board_id"] != nearest_end["board_id"]:
                if any(
                    item.get("type") == "connector"
                    and _bbox_intersection(item["bbox"], line_bbox) / max(1, min(_bbox_area(item["bbox"]), _bbox_area(line_bbox))) > 0.45
                    for item in annotations
                ):
                    continue
                annotation = {
                    "type": "connector",
                    "annotation_id": f"line_{index:02d}",
                    "bbox": line_bbox,
                    "from_board": nearest_start["board_id"],
                    "to_board": nearest_end["board_id"],
                    "confidence": round(confidence, 4),
                    "flags": flags,
                }
                if not _line_annotation_duplicate(annotation, annotations):
                    annotations.append(annotation)
    return _reorient_inferred_arrows_from_full_bbox(
        mask,
        _dedupe_line_annotations(annotations),
        boards,
        min_cell,
    )


def _detect_annotations(rgb: np.ndarray, boards: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    highlights, red_residual = _detect_highlights(rgb, boards)
    lines = _detect_lines(rgb, boards)
    annotations = highlights + lines
    low_confidence = [
        annotation
        for annotation in annotations
        if float(annotation.get("confidence", 0.0)) < ANNOTATION_REVIEW_THRESHOLD
    ]
    return (
        annotations,
        {
            "red_residual_components": red_residual,
            "annotation_count": len(annotations),
            "low_confidence_annotations": len(low_confidence),
        },
    )


def _load_overrides(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _apply_override(record: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    override = overrides.get(record.get("image_id")) or overrides.get(record.get("sha256"))
    if not isinstance(override, dict):
        return record
    next_record = dict(record)
    for key in ("status", "boards", "annotations"):
        if key in override:
            next_record[key] = override[key]
    board_overrides = override.get("board_overrides")
    if isinstance(board_overrides, list) and "boards" not in override:
        next_boards = [dict(board) for board in next_record.get("boards", [])]
        for board_override in board_overrides:
            if not isinstance(board_override, dict):
                continue
            board_index = board_override.get("board_index")
            board_id = str(board_override.get("board_id") or "")
            matched_index: int | None = None
            if isinstance(board_index, int) and 0 <= board_index < len(next_boards):
                matched_index = board_index
            elif board_id:
                matched_index = next(
                    (
                        index
                        for index, board in enumerate(next_boards)
                        if str(board.get("board_id") or "") == board_id
                    ),
                    None,
                )
            if matched_index is None:
                continue
            patch = {
                key: value
                for key, value in board_override.items()
                if key not in {"board_id", "board_index"}
            }
            next_boards[matched_index] = {**next_boards[matched_index], **patch}
        next_record["boards"] = next_boards
    if "quality" not in next_record or not isinstance(next_record["quality"], dict):
        next_record["quality"] = {}
    next_record["quality"] = dict(next_record["quality"])
    next_record["quality"]["used_override"] = True
    flags = set(next_record["quality"].get("flags") or [])
    flags.add("manual_override")
    next_record["quality"]["flags"] = sorted(flags)
    return next_record


def _status_for(boards: list[dict[str, Any]], annotations: list[dict[str, Any]], quality_flags: list[str]) -> str:
    if not boards and not annotations:
        return "ignored"
    if quality_flags:
        return "review"
    for board in boards:
        if board.get("flags") or float(board.get("confidence", 0.0)) < BOARD_ACCEPT_THRESHOLD:
            return "review"
        for cell in board.get("cells", []):
            cell_flags = set(cell.get("flags") or [])
            if cell_flags and not all(flag.startswith("padding_") for flag in cell_flags):
                return "review"
    for annotation in annotations:
        if annotation.get("flags") or float(annotation.get("confidence", 0.0)) < ANNOTATION_REVIEW_THRESHOLD:
            return "review"
    return "accepted"


def _parse_entry(
    manifest_entry: dict[str, Any],
    probes: list[BoardProbe],
    palette: dict[int, list[int]],
    overrides: dict[str, Any],
    profile: str = DEFAULT_PARSER_PROFILE,
) -> dict[str, Any]:
    image_path = Path(manifest_entry["path"])
    rgb = _load_rgb(image_path)
    boards: list[dict[str, Any]] = []
    quality_flags: list[str] = []
    for probe in probes:
        board, needs_review = _finalize_board(probe, palette, profile)
        boards.append(board)
        if needs_review:
            quality_flags.append(f"{board['board_id']}:review")

    annotations, annotation_quality = _detect_annotations(rgb, boards)
    if annotation_quality["red_residual_components"]:
        quality_flags.append("red_annotation_residual")
    if annotation_quality["low_confidence_annotations"]:
        quality_flags.append("low_confidence_annotation")

    min_board_confidence = min(
        [float(board.get("confidence", 1.0)) for board in boards],
        default=1.0,
    )
    record = {
        "image_id": manifest_entry["image_id"],
        "sha256": manifest_entry["sha256"],
        "source_name": manifest_entry.get("source_name"),
        "path": manifest_entry["path"],
        "relative_path": manifest_entry.get("relative_path"),
        "doc_index": manifest_entry.get("doc_index"),
        "width": manifest_entry.get("width"),
        "height": manifest_entry.get("height"),
        "duplicate_of": manifest_entry.get("duplicate_of"),
        "status": _status_for(boards, annotations, quality_flags),
        "boards": boards,
        "annotations": annotations,
        "quality": {
            "parser_profile": profile,
            "min_board_confidence": round(min_board_confidence, 4),
            "palette_size": len(palette),
            "flags": sorted(set(quality_flags)),
            "used_override": False,
            **annotation_quality,
        },
    }
    return _apply_override(record, overrides)


def parse_manifest(
    manifest_path: Path,
    out_path: Path,
    qa_dir: Path | None = None,
    overrides_path: Path | None = None,
    limit: int | None = None,
    profile: str = DEFAULT_PARSER_PROFILE,
) -> list[dict[str, Any]]:
    if profile not in PARSER_PROFILES:
        raise ValueError(f"Unknown parser profile: {profile}")
    manifest = read_jsonl(manifest_path)
    if limit is not None:
        manifest = manifest[:limit]
    matcher = TemplateMatcher()
    all_probes_by_image: dict[str, list[BoardProbe]] = {}
    all_boards: list[BoardProbe] = []

    for entry in manifest:
        image_path = Path(entry["path"])
        rgb = _load_rgb(image_path)
        probes = _probe_boards(entry["image_id"], rgb, matcher, profile)
        all_probes_by_image[entry["image_id"]] = probes
        all_boards.extend(probes)

    palette = _build_palette(all_boards)
    overrides = _load_overrides(overrides_path)
    parsed = [
        _parse_entry(
            entry,
            all_probes_by_image.get(entry["image_id"], []),
            palette,
            overrides,
            profile,
        )
        for entry in manifest
    ]

    write_jsonl(out_path, parsed)
    write_json(
        out_path.with_name("guide_boards.json"),
        {
            "source_manifest": str(manifest_path),
            "palette": {str(exponent): rgb for exponent, rgb in sorted(palette.items())},
            "boards": [
                {
                    "image_id": record["image_id"],
                    "board_id": board["board_id"],
                    "status": record["status"],
                    "hex": board["hex"],
                    "visible_rows": board["visible_rows"],
                    "visible_cols": board["visible_cols"],
                    "padding": board.get("padding"),
                    "confidence": board["confidence"],
                    "flags": board["flags"],
                    "doc_index": record.get("doc_index"),
                }
                for record in parsed
                for board in record.get("boards", [])
            ],
        },
    )
    if qa_dir:
        qa_dir.mkdir(parents=True, exist_ok=True)
        write_json(qa_dir / "palette.json", {str(k): v for k, v in sorted(palette.items())})
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse guide media images into board data.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--qa-dir", type=Path, default=None)
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--profile", choices=PARSER_PROFILES, default=DEFAULT_PARSER_PROFILE)
    args = parser.parse_args()

    parsed = parse_manifest(
        args.manifest,
        args.out,
        qa_dir=args.qa_dir,
        overrides_path=args.overrides,
        limit=args.limit,
        profile=args.profile,
    )
    counts: dict[str, int] = {}
    for record in parsed:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    print(json.dumps({"images": len(parsed), "status": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
