from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from .common import read_jsonl
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.guide_image_parser.common import read_jsonl


def _load_expectations(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("gold sample file must contain a top-level samples list")
    return samples


def _board_key(board: dict[str, Any]) -> tuple[int, int]:
    bbox = board.get("bbox") or [0, 0, 0, 0]
    return int(bbox[1]), int(bbox[0])


def _sort_boards_spatial(boards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not boards:
        return []
    heights = [int((board.get("bbox") or [0, 0, 0, 0])[3]) for board in boards]
    median_height = float(np.median(heights)) if heights else 0.0
    tolerance = max(4.0, median_height * 0.35)
    rows: list[list[dict[str, Any]]] = []
    for board in sorted(boards, key=_board_key):
        bbox = board.get("bbox") or [0, 0, 0, 0]
        center_y = float(bbox[1]) + float(bbox[3]) / 2.0
        if not rows:
            rows.append([board])
            continue
        row_center = float(np.median([float((item.get("bbox") or [0, 0, 0, 0])[1]) + float((item.get("bbox") or [0, 0, 0, 0])[3]) / 2.0 for item in rows[-1]]))
        if abs(center_y - row_center) <= tolerance:
            rows[-1].append(board)
        else:
            rows.append([board])
    output: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda items: float(np.median([float((item.get("bbox") or [0, 0, 0, 0])[1]) + float((item.get("bbox") or [0, 0, 0, 0])[3]) / 2.0 for item in items]))):
        output.extend(sorted(row, key=lambda item: int((item.get("bbox") or [0, 0, 0, 0])[0])))
    return output


def _highlight_cells(record: dict[str, Any], board_index: int) -> set[tuple[int, int]]:
    boards = _sort_boards_spatial(record.get("boards", []))
    if board_index >= len(boards):
        return set()
    board_id = boards[board_index].get("board_id")
    cells: set[tuple[int, int]] = set()
    for annotation in record.get("annotations", []):
        if annotation.get("type") != "highlight" or annotation.get("board_id") != board_id:
            continue
        for cell in annotation.get("cells") or []:
            if isinstance(cell, list) and len(cell) == 2:
                cells.add((int(cell[0]), int(cell[1])))
    return cells


def validate_gold_samples(parsed_path: Path, expectations_path: Path) -> dict[str, Any]:
    records = {record["image_id"]: record for record in read_jsonl(parsed_path)}
    samples = _load_expectations(expectations_path)
    failures: list[dict[str, Any]] = []
    board_total = 0
    board_passed = 0
    highlight_total = 0
    highlight_passed = 0

    for sample in samples:
        image_id = sample["image_id"]
        record = records.get(image_id)
        if record is None:
            failures.append({"image_id": image_id, "type": "missing_record"})
            continue

        actual_boards = _sort_boards_spatial(record.get("boards", []))
        expected_boards = sample.get("boards") or []
        if len(actual_boards) != len(expected_boards):
            failures.append(
                {
                    "image_id": image_id,
                    "type": "board_count",
                    "expected": len(expected_boards),
                    "actual": len(actual_boards),
                }
            )

        for index, expected in enumerate(expected_boards):
            board_total += 1
            actual_hex = actual_boards[index].get("hex") if index < len(actual_boards) else None
            expected_hex = expected.get("hex")
            if actual_hex == expected_hex:
                board_passed += 1
            else:
                failures.append(
                    {
                        "image_id": image_id,
                        "type": "board_hex",
                        "board_index": index,
                        "expected": expected_hex,
                        "actual": actual_hex,
                    }
                )

        for expected in sample.get("highlights") or []:
            highlight_total += 1
            board_index = int(expected.get("board_index", 0))
            expected_cells = {tuple(cell) for cell in expected.get("cells") or []}
            actual_cells = _highlight_cells(record, board_index)
            if expected_cells.issubset(actual_cells):
                highlight_passed += 1
            else:
                failures.append(
                    {
                        "image_id": image_id,
                        "type": "highlight_cells",
                        "board_index": board_index,
                        "expected": sorted(expected_cells),
                        "actual": sorted(actual_cells),
                    }
                )

    return {
        "samples": len(samples),
        "boards": {"passed": board_passed, "total": board_total},
        "highlights": {"passed": highlight_passed, "total": highlight_total},
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate parsed guide images against manually checked gold samples.")
    parser.add_argument("--parsed", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    args = parser.parse_args()
    result = validate_gold_samples(args.parsed, args.gold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
