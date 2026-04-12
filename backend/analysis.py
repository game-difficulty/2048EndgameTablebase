from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np

from .analysis_core import Analyzer


AnalysisUpdateCallback = Callable[[dict[str, Any]], None]


def resolve_analysis_inputs(paths: list[str]) -> list[str]:
    file_list: list[str] = []
    seen: set[str] = set()

    for raw_path in paths:
        path = str(raw_path or "").strip()
        if not path or not os.path.exists(path):
            continue

        if os.path.isdir(path):
            folder_files = [
                os.path.join(path, filename)
                for filename in os.listdir(path)
                if (
                    filename.lower().endswith(".txt")
                    or filename.lower().endswith(".vrs")
                )
                and os.path.isfile(os.path.join(path, filename))
            ]
            for file_path in folder_files:
                normalized = os.path.normpath(file_path)
                if normalized not in seen:
                    seen.add(normalized)
                    file_list.append(normalized)
            continue

        if os.path.isfile(path) and (
            path.lower().endswith(".txt")
            or path.lower().endswith(".vrs")
            or "." not in os.path.basename(path)
        ):
            normalized = os.path.normpath(path)
            if normalized not in seen:
                seen.add(normalized)
                file_list.append(normalized)

    return file_list


def run_analysis_file(
    file_path: str,
    pattern: str,
    target_value: int,
    full_pattern: str,
) -> dict[str, Any]:
    analyzer = Analyzer(
        file_path=file_path,
        pattern=pattern,
        target=target_value,
        full_pattern=full_pattern,
        target_path=os.path.dirname(file_path),
    )
    analyzer.generate_reports()
    return {
        "path": os.path.normpath(file_path),
        "status": "done",
        "output_dir": os.path.normpath(os.path.dirname(file_path)),
    }


def run_batch_analysis(
    file_list: list[str],
    pattern: str,
    target_value: int,
    full_pattern: str,
    on_progress: AnalysisUpdateCallback,
) -> list[dict[str, Any]]:
    total = len(file_list)
    if total <= 0:
        return []

    max_workers = min(total, max(1, multiprocessing.cpu_count() - 1))
    completed = 0
    entries: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_analysis_file, file_path, pattern, target_value, full_pattern
            ): file_path
            for file_path in file_list
        }

        for future in as_completed(futures):
            source_path = os.path.normpath(futures[future])
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "path": source_path,
                    "status": "failed",
                    "message": str(exc),
                    "output_dir": os.path.normpath(os.path.dirname(source_path)),
                }

            completed += 1
            entries.append(result)
            on_progress(
                {
                    "completed": completed,
                    "total": total,
                    "current_file": result["path"],
                    "entry": result,
                    "entries": entries[-8:],
                    "done": sum(1 for item in entries if item["status"] == "done"),
                    "failed": sum(
                        1 for item in entries if item["status"] == "failed"
                    ),
                }
            )

    return entries


def normalize_target_value(target: Any) -> tuple[str, int, str]:
    target_tile = str(target or "").strip()
    if not target_tile:
        raise ValueError("Missing target tile")

    try:
        numeric_target = int(target_tile)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid target tile: {target}") from exc

    if numeric_target <= 0:
        raise ValueError(f"Invalid target tile: {target}")

    return target_tile, int(np.log2(numeric_target)), f"{numeric_target}"
