"""Discover result prefixes without reading layer data or changing result files."""
from pathlib import Path
import json
import math
import re

from Config import SingletonConfig, pattern_catalog, DTYPE_CONFIG
from engine_core.GoalSpec import GoalSpec, available_target_tokens

RESULT = re.compile(r"^(.+)_(-?\d+)(\.book|\.z|b|\.zbook|\.exzbook|\.exadbook|\.exadzbook|\.bccmp|\.bcraw|\.bcpos|\.bcsuc)$")


def catalog_snapshot():
    return {"available_tables": SingletonConfig.get_available_pattern_targets(),
            "catalog_spawn_rate": float(SingletonConfig().config.get("4_spawn_rate", .1)),
            "target_tiles": available_target_tokens()}


def scan_tables(folder):
    if not str(folder).strip():
        raise ValueError("Select a table directory")
    root = Path(folder).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Select a table directory")
    candidates = []
    queue = [(root, 0)]
    visited = 0
    while queue:
        directory, depth = queue.pop(0)
        visited += 1
        if visited > 256:
            raise ValueError("Too many directories; select a more specific folder")
        entries = list(directory.iterdir())
        names = {item.name for item in entries}
        prefixes = set()
        for name in names:
            match = RESULT.fullmatch(name)
            if not match:
                continue
            prefix, layer, suffix = match.groups()
            if suffix in (".bcpos", ".bcsuc") and not all(
                    prefix + "_" + layer + ext in names for ext in (".bcpos", ".bcsuc")):
                continue
            prefixes.add(prefix)
        for prefix in sorted(prefixes):
            pattern, _, token = prefix.rpartition("_")
            if pattern not in pattern_catalog:
                continue
            try:
                goal = GoalSpec.parse(token)
            except ValueError:
                continue
            if goal.token != token:
                continue
            metadata = directory / (prefix + "_goal.json")
            if metadata.exists():
                meta = json.loads(metadata.read_text(encoding="utf-8"))
                if (meta.get("kind"), meta.get("value")) != (goal.kind, goal.value):
                    raise ValueError(f"Goal metadata disagrees with filename: {metadata}")
                if goal.kind == "sum" and meta.get("semantics") != "post_move_mod16384_target_minus_2":
                    raise ValueError(f"Unsupported goal semantics: {metadata}")
            config_path = directory / (prefix + "_config.txt")
            fields = {}
            if config_path.exists():
                fields = dict(line.split(":", 1) for line in config_path.read_text(encoding="utf-8-sig").splitlines() if ":" in line)
            dtype = fields.get("success_rate_dtype", "uint32").strip()
            rate_text = fields.get("4_spawn_rate")
            rate = float(rate_text) if rate_text is not None else float(SingletonConfig().config.get("4_spawn_rate", 0.1))
            if dtype not in DTYPE_CONFIG or not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError(f"Invalid table configuration: {config_path}")
            candidates.append(dict(path=str(directory), pattern=pattern, target=goal.token,
                                   dtype=dtype, spawn_rate=rate, assumed_rate=rate_text is None))
        if depth < 2:
            queue.extend((item, depth + 1) for item in entries
                         if item.is_dir() and not item.is_symlink() and item.resolve().is_relative_to(root)
                         and not RESULT.fullmatch(item.name))
    return candidates


def import_tables(selected):
    # Rescan selected directories: never trust client-supplied dtype or probability.
    verified = []
    scans = {}
    for item in selected:
        path = str(Path(item["path"]).resolve(strict=True))
        if path not in scans:
            scans[path] = scan_tables(path)
        match = next((row for row in scans[path] if row["path"] == path
                      and row["pattern"] == item["pattern"] and row["target"] == item["target"]), None)
        if match is None:
            raise ValueError("Selected table no longer exists; scan again")
        verified.append(match)
    config = SingletonConfig().config
    for row in verified:
        key = SingletonConfig.get_pattern_key(f'{row["pattern"]}_{row["target"]}', row["spawn_rate"])
        paths = config.setdefault("filepath_map", {}).setdefault(key, [])
        entry = (row["path"], row["dtype"])
        paths[:] = [old for old in paths if Path(old[0]).resolve() != Path(row["path"])]
        paths.append(entry)
    SingletonConfig().save_config(config)
    return verified
