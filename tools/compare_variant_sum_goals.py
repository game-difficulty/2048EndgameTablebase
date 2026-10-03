"""Serial, resumable full builds and layer-zero comparison for sum goals."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CASES = [(p, t, a) for p, t in (("2x4", 900), ("3x3", 1800))
         for a in ("classic", "ex", "bc")]
TOLERANCE = 6e-9


def save(path, value):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    temp.replace(path)


def case_dir(root, case):
    pattern, target, algorithm = case
    return root / f"{pattern}_sum-{target}_{algorithm}"


def load_native(root):
    import native_core
    global _dll
    _dll = os.add_dll_directory("C:/Apps/mingw64/bin") if os.name == "nt" else None
    module_file = next((root / "runtime").glob("formation_core*.pyd"))
    spec = importlib.util.spec_from_file_location("native_core.formation_core", module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    native_core.formation_core = module
    return module


def worker(root, index):
    native = load_native(root)
    import numpy as np
    import Config
    from Config import SingletonConfig, pattern_catalog
    from engine_core import BookBuilder as builder
    from engine_core.EXPhysicalPattern import resolve_ex_physical_pattern
    pattern, target, algorithm = CASES[index]
    folder = case_dir(root, CASES[index])
    folder.mkdir(exist_ok=True)
    prefix = str(folder / f"{pattern}_sum-{target}_")
    result = dict(pattern=pattern, target=target, algorithm=algorithm,
                  status="running", started=time.time(), layer=0)
    save(folder / "result.json", result)
    if algorithm == "bc":
        result.update(status="unsupported", reason="BC has no variant-mover implementation; v_start_build rejects variant BC")
        save(folder / "result.json", result)
        return 0
    signal = str(folder / "threshold.txt")
    Config.RUNTIME_DELETION_THRESHOLD_SIGNAL_PATH = signal
    builder.RUNTIME_DELETION_THRESHOLD_SIGNAL_PATH = signal
    settings = dict(algorithm_mode=algorithm, advanced_algo=False, zmask_algo=algorithm == "ex",
        compress=False, compress_temp_files=False, optimal_branch_only=False, chunked_solve=False,
        deletion_threshold=0.0, deletion_threshold_mode="off", success_rate_dtype="float64",
        direct_io=False, SmallTileSumLimit=96, **{"4_spawn_rate": 0.1})
    SingletonConfig().config.update(settings)
    result["settings"] = settings
    # Limit this independent job without changing any other process's thread count.
    original_options = builder._build_native_run_options
    def run_options(*args, **kwargs):
        options = original_options(*args, **kwargs)
        options.num_threads = 8
        return options
    builder._build_native_run_options = run_options
    result["threads"] = 8
    save(folder / "result.json", result)
    try:
        builder.v_start_build(pattern, f"sum-{target}", prefix)
        result["build_seconds"] = time.time() - result["started"]
        seeds = np.asarray(pattern_catalog[pattern]["seed_boards"], dtype=np.uint64)
        rows = {}
        if algorithm == "classic":
            path = Path(prefix + "0.book")
            if not path.exists():
                raise FileNotFoundError(path)
            rows = {f"{board:016x}": value for board, value in struct.iter_unpack("<Qd", path.read_bytes())}
            result["stored_rows"] = len(rows)
        else:
            resolution = resolve_ex_physical_pattern(pattern, seeds, target.bit_length()-1, 96, advanced=False)
            result["physical_transform"] = int(resolution.transform_id)
            for seed in seeds:
                physical = native.apply_sym_like(int(seed), int(resolution.transform_id))
                lookup = native.lookup_ex_zbook_cold(prefix + "0.zbook", prefix + ".zlut", physical)
                if not lookup["found"]:
                    raise RuntimeError(f"Initial state missing from EX layer zero: {int(seed):016x}")
                rows[f"{int(seed):016x}"] = lookup["numeric_value"]
        expected = {f"{int(board):016x}" for board in seeds}
        if set(rows) != expected:
            raise RuntimeError(f"Initial layer coverage mismatch: expected {expected}, actual {set(rows)}")
        result.update(status="complete", initial_values=rows, finished=time.time())
    except BaseException:
        result.update(status="error", error=traceback.format_exc(), finished=time.time())
        traceback.print_exc()
    save(folder / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "complete" else 1


def report(root):
    results = []
    for case in CASES:
        path = case_dir(root, case) / "result.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf8")))
    comparisons = []
    for pattern in ("2x4", "3x3"):
        group = [r for r in results if r["pattern"] == pattern and r["status"] == "complete"]
        for i, a in enumerate(group):
            for b in group[i+1:]:
                av, bv = a["initial_values"], b["initial_values"]
                delta = {key: abs(av[key]-bv[key]) for key in av.keys() & bv.keys()}
                comparisons.append(dict(pattern=pattern, algorithms=[a["algorithm"], b["algorithm"]],
                    differences=delta, passed=set(av)==set(bv) and bool(delta) and max(delta.values()) <= TOLERANCE))
    result = dict(tolerance=TOLERANCE, results=results, comparisons=comparisons)
    save(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker is not None:
        return worker(root, args.worker)
    root.mkdir(parents=True, exist_ok=True)
    runtime = root / "runtime"
    runtime.mkdir(exist_ok=True)
    source = REPO / "native_core/formation_core.cp312-win_amd64.pyd"
    copied = runtime / source.name
    if not copied.exists():
        shutil.copy2(source, copied)
    save(root / "plan.json", dict(cases=CASES, tolerance=TOLERANCE,
        native_sha256=hashlib.sha256(copied.read_bytes()).hexdigest(), pid=os.getpid()))
    for i, case in enumerate(CASES):
        folder = case_dir(root, case)
        folder.mkdir(exist_ok=True)
        previous = folder / "result.json"
        if previous.exists() and json.loads(previous.read_text(encoding="utf8"))["status"] in ("complete", "unsupported"):
            continue
        print(f"Starting {case}", flush=True)
        with (folder / "worker.log").open("a", encoding="utf8") as log:
            proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                "--root", str(root), "--worker", str(i)], stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            save(root / "current.json", dict(case=case, pid=proc.pid, started=time.time()))
            code = proc.wait()
        if code and not previous.exists():
            save(previous, dict(pattern=case[0], target=case[1], algorithm=case[2], status="error", exit_code=code))
        report(root)
    result = report(root)
    save(root / "current.json", dict(status="finished", finished=time.time()))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
