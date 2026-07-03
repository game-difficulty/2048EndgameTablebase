from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _coerce_path_items(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value]
    return [str(value).strip()]


def normalize_build_folder_paths(
    folder_paths=None,
    *,
    folder_path: str | None = None,
    pathname: str | None = None,
) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()

    candidates = _coerce_path_items(folder_paths)
    if folder_path:
        candidates.append(str(folder_path).strip())
    if not candidates and pathname:
        candidates.append(os.path.dirname(str(pathname).rstrip("\\/")))

    for raw_path in candidates:
        if not raw_path:
            continue
        normalized = os.path.normpath(raw_path)
        key = os.path.normcase(os.path.abspath(normalized))
        if key in seen:
            continue
        seen.add(key)
        paths.append(normalized)
    return paths


@dataclass(frozen=True)
class StoragePathPlan:
    hot_root: Path
    cold_roots: tuple[Path, ...]

    @classmethod
    def from_folder_paths(cls, folder_paths: list[str]) -> "StoragePathPlan":
        if not folder_paths:
            raise ValueError("At least one build folder is required")
        hot_root = Path(folder_paths[0])
        cold_roots = tuple(Path(path) for path in folder_paths[1:])
        return cls(hot_root=hot_root, cold_roots=cold_roots)

    @property
    def roots(self) -> tuple[Path, ...]:
        return (self.hot_root, *self.cold_roots)

    @property
    def primary_cold_root(self) -> Path:
        return self.cold_roots[0] if self.cold_roots else self.hot_root

    def validate_existing_dirs(self) -> None:
        for path in self.roots:
            if not path.is_dir():
                raise ValueError(f"Invalid build folder: {path}")

    def pathname_for(self, root: Path, pattern: str, target_tile: str | int) -> str:
        return str(root / f"{pattern}_{target_tile}_")

    def hot_pathname(self, pattern: str, target_tile: str | int) -> str:
        return self.pathname_for(self.hot_root, pattern, target_tile)

    def cold_pathnames(self, pattern: str, target_tile: str | int) -> list[str]:
        return [self.pathname_for(root, pattern, target_tile) for root in self.cold_roots]

    def filepath_map_entries(self, success_rate_dtype: str) -> list[tuple[str, str]]:
        return [(str(path), success_rate_dtype) for path in self.roots]
