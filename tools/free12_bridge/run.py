"""Plan or run the bounded free12-4096 bridge; default command is read-only."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE11, BASE12 = 5 * 32768 + 20, 4 * 32768 + 22
SOLVED = struct.Struct("<8s6I2BH4x5Q")
TEMP = struct.Struct("<8s4I2BH4x4Q")
SLOT_SOLVED, SLOT_TEMP = struct.Struct("<8Q2I"), struct.Struct("<6Q")


def header(path: Path, *, solved: bool = True) -> dict:
    """Read directories only, but check the exact expected file length."""
    fmt, slot_fmt = (SOLVED, SLOT_SOLVED) if solved else (TEMP, SLOT_TEMP)
    with path.open("rb") as stream:
        raw = stream.read(fmt.size)
        if len(raw) != fmt.size:
            raise ValueError(f"Incomplete header: {path}")
        h = fmt.unpack(raw)
        if h[0] != (b"EXAD7SLV" if solved else b"EXAD7TMP") or h[1] != 2:
            raise ValueError(f"Unsupported EXAD file: {path}")
        if solved:
            _, _, dtype, value_size, total, threshold, slots, transform, inverse, _, lut, logical, physical, rows, values = h
            if (dtype, value_size) != (1, 4):
                raise ValueError(f"Expected uint32 success: {path}")
        else:
            _, _, total, threshold, slots, transform, inverse, _, lut, logical, physical, rows = h
            values = value_size = 0
        if slots != 48 or transform != 0 or inverse != 0:
            raise ValueError(f"Expected 48 slots and identity physical transform: {path}")
        tables = [slot_fmt.unpack(stream.read(slot_fmt.size)) for _ in range(slots)]
    size = fmt.size + slots * slot_fmt.size + values * value_size
    size += sum(b[0] * 16 + b[1] + b[2] * 8 for b in tables)
    if size != path.stat().st_size or sum(b[3] for b in tables) != rows:
        raise ValueError(f"File length/row counts do not match its header: {path}")
    if solved:
        row_base = value_base = 0
        for b in tables:
            if b[6:8] != (row_base, value_base):
                raise ValueError(f"Invalid AD row/value offsets: {path}")
            row_base += b[3]
            value_base += b[3] * b[8]
        if value_base != values:
            raise ValueError(f"Invalid AD value count: {path}")
    return dict(sum=total, rows=rows, values=values, lut=lut, logical=logical,
                physical=physical, threshold=threshold,
                active_slots=[dict(key=i - 16, rows=b[3], width=b[8] if solved else None)
                              for i, b in enumerate(tables) if b[3]])


def read_config(path: Path) -> dict:
    return dict(line.strip().split(": ", 1) for line in path.read_text().splitlines() if ": " in line)


def recover_stsl(config: dict, target: int) -> int:
    # Old config.txt omitted STSL, but the persisted FNV signature includes it.
    from Config import pattern_catalog
    from engine_core.EXPhysicalPattern import _signature
    matches = [s for s in range(1025) if _signature(
        "free11", "full", [], pattern_catalog["free11"]["success_shifts"], [],
        target, 5, s, 0) == int(config["ex_logical_pattern_signature"])]
    if len(matches) != 1:
        raise ValueError("Could not uniquely recover source STSL from its saved signature")
    return matches[0]


def plan(args) -> dict:
    from Config import pattern_catalog
    from engine_core.EXPhysicalPattern import resolve_ex_physical_pattern
    source1 = args.source1.resolve() / "free11_1024_"
    source2 = args.source2.resolve() / "free11_2048_"
    configs = [read_config(Path(f"{p}config.txt")) for p in (source1, source2)]
    for config in configs:
        if config.get("algorithm_mode") != "exad" or config["success_rate_dtype"] != "uint32":
            raise ValueError("Both sources must be EXAD uint32")
        if float(config["4_spawn_rate"]) != 0.1:
            raise ValueError("Both sources must use spawn4=0.1")
    stsls = [recover_stsl(c, t) for c, t in zip(configs, (10, 11))]
    stsl = min(stsls)
    resolution = resolve_ex_physical_pattern("free12", pattern_catalog["free12"]["seed_boards"],
                                             12, stsl, advanced=True)
    if resolution.transform_id != 0:
        raise ValueError("This bridge requires identity physical transform")
    seeds = []
    for step in (469, 470):
        path = Path(f"{source1}{step}.exadbook")
        h = header(path)
        if h["sum"] != BASE11 + 2 * step or h["logical"] != int(configs[0]["ex_logical_pattern_signature"]):
            raise ValueError(f"Source seed metadata mismatch: {path}")
        total = h["sum"] - 32768 + 1024
        seeds.append(dict(source_step=step, target_step=(total - BASE12) // 2,
                          target_sum=total, path=str(path), rows=h["rows"],
                          bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
    last = seeds[-1]["target_step"] + args.new_layers
    boundaries = []
    for step in (last - 1, last):
        total = BASE12 + 2 * step
        source_sum = total + 32768 - 2048
        source_step = (source_sum - BASE11) // 2
        if source_step < 0:
            raise ValueError("Window ends before the free11-2k seed sum")
        path = Path(f"{source2}{source_step}.exadbook")
        h = header(path)
        if h["sum"] != source_sum or h["logical"] != int(configs[1]["ex_logical_pattern_signature"]):
            raise ValueError(f"Boundary source metadata mismatch: {path}")
        if any(s["key"] != 5 or s["width"] != 1 for s in h["active_slots"]):
            raise ValueError("Boundary source has derived lanes; shorten the window")
        boundaries.append(dict(target_step=step, target_sum=total, source_step=source_step,
                               source_sum=source_sum, path=str(path), rows=h["rows"],
                               bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
    out = args.output.resolve()
    for source in (args.source1.resolve(), args.source2.resolve()):
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("Output and source directories must be disjoint")
    return dict(version=1, output=str(out), prefix=str(out / "free12_4096_"),
                source1=str(source1), source2=str(source2), source_stsl=stsls, stsl=stsl,
                signature=int(resolution.logical_pattern_signature), seed_threshold=0.4,
                deletion_threshold=0.05, relative_deletion_threshold=0.0, compress=False,
                dtype="uint32", spawn4=0.1, new_layers=args.new_layers,
                seeds=seeds, boundaries=boundaries, solver_steps=last + 3)


def save_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".writing")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def artifact(p, step, suffix):
    return Path(f"{p['prefix']}{step}{suffix}")


def existing_artifact(p, step, suffix):
    hot = artifact(p, step, suffix)
    if hot.exists() or not p.get("cold_prefix"):
        return hot
    return Path(f"{p['cold_prefix']}{step}{suffix}")


def check_target(p, step, solved):
    suffix = ".exadbook" if solved else ".exadtmp"
    path = existing_artifact(p, step, suffix)
    try:
        h = header(path, solved=solved)
    except FileNotFoundError:
        # The asynchronous archiver publishes cold before removing hot.
        h = header(existing_artifact(p, step, suffix), solved=solved)
    if h["sum"] != BASE12 + step * 2 or h["logical"] != p["signature"] or h["physical"] != p["signature"]:
        raise ValueError(f"Target metadata mismatch: {path}")
    with Path(f"{p['prefix']}.exadlut").open("rb") as stream:
        lut_header = stream.read(48)
    magic, version, _, transform, inverse, _, signature, logical, physical = struct.unpack("<8s2I2BH4x3Q", lut_header)
    if (magic, version, transform, inverse, signature, logical, physical) != (
            b"EXAD7LUT", 2, 0, 0, h["lut"], p["signature"], p["signature"]):
        raise ValueError(f"Target LUT does not match layer: {path}")
    return h


def native(args, p, mode, step, *extra):
    exe = args.native.resolve()
    if not exe.is_file():
        raise FileNotFoundError(f"Build the native helper first: {exe}")
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(ROOT / "native_core"), str(Path(sys.executable).parent), env.get("PATH", "")])
    command = [str(exe), mode, p["prefix"], str(step), str(p["stsl"]), str(p["signature"]),
               str(args.threads), *map(str, extra)]
    print(f"[{mode}] step={step}", flush=True)
    with Path(p["prefix"]).parent.joinpath("bridge-native.log").open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {mode} step={step}\n")
        log.flush()
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code, command)


@contextlib.contextmanager
def lock_output(out):
    # OS advisory lock releases automatically on process death; the tiny file remains.
    stream = (out / "bridge.lock").open("a+b")
    stream.seek(0)
    stream.write(b"0")
    stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def prepare(args, p):
    for seed in p["seeds"]:
        step = seed["target_step"]
        if artifact(p, step, ".exadbook").exists():
            check_target(p, step, True)
        elif artifact(p, step, ".exadtmp").exists():
            check_target(p, step, False)
        else:
            native(args, p, "prepare", step, seed["path"], f"{p['source1']}.exadlut")
            check_target(p, step, False)


def generate(args, p):
    first, last = p["seeds"][0]["target_step"], p["boundaries"][-1]["target_step"]
    if any(artifact(p, s, ".exadbook").exists() for s in range(first, last - 1)):
        raise ValueError("Backward solve has started; resume solve, not generation")
    # Never let an unprepared invocation fall back to the real layer0 seed.
    for seed in p["seeds"]:
        check_target(p, seed["target_step"], False)
    if check_target(p, 0, False)["rows"] != 0:
        raise ValueError("Expected the bridge's empty generation layer0")
    gap = False
    for step in range(first, last + 1):
        if artifact(p, step, ".exadtmp").exists():
            if gap:
                raise ValueError(f"Generation checkpoint gap before layer {step}")
            check_target(p, step, False)
        else:
            gap = True
    native(args, p, "generate", last)
    for step in range(first, last + 1):
        check_target(p, step, False)


def boundaries(args, p):
    for b in p["boundaries"]:
        step = b["target_step"]
        if artifact(p, step, ".exadbook").exists():
            check_target(p, step, True)
            continue
        check_target(p, step, False)
        native(args, p, "boundary", step, b["path"], f"{p['source2']}.exadlut")
        check_target(p, step, True)


def solve(args, p):
    import numpy as np
    from Config import pattern_catalog
    from native_core import formation_core as f
    first, last = p["seeds"][0]["target_step"], p["boundaries"][-1]["target_step"]
    for step in range(first, last + 1):
        check_target(p, step, existing_artifact(p, step, ".exadbook").exists())
    for b in p["boundaries"]:
        check_target(p, b["target_step"], True)
    spec = f.AdvancedPatternSpec()
    spec.name, spec.target, spec.num_free_32k = "free12", 12, 4
    spec.small_tile_sum_limit = p["stsl"]
    spec.symm_mode = 1
    spec.success_shifts = list(range(0, 64, 4))
    spec.logical_pattern_signature = spec.physical_pattern_signature = p["signature"]
    options = f.RunOptions()
    options.target, options.steps, options.pathname = 12, p["solver_steps"], p["prefix"]
    if p.get("cold_prefix"):
        options.cold_pathnames = [p["cold_prefix"]]
    options.docheck_step = 2048 - (BASE12 % 4096) // 2
    options.is_free, options.num_threads = True, args.threads
    options.spawn_rate4, options.success_rate_dtype = 0.1, "uint32"
    options.deletion_threshold, options.relative_deletion_threshold = 0.05, 0.0
    options.compress = options.compress_temp_files = False
    options.chunked_solve = args.chunked_solve
    options.direct_io = args.direct_io
    placeholders = [artifact(p, s, ".exadbook") for s in range(first)]
    # The scope is owned by bridge-plan.json. Real data below the interval is an error.
    for path in placeholders:
        if path.exists() and path.stat().st_size != 0:
            raise ValueError(f"Unexpected real layer outside bridge interval: {path}")
    try:
        for path in placeholders:
            path.touch(exist_ok=True)
        print(f"[solve] {last - 2} -> {first}; steps={options.steps}", flush=True)
        f.run_pattern_solve_exad(np.asarray(pattern_catalog["free12"]["seed_boards"], dtype=np.uint64), spec, options)
    finally:
        for path in placeholders:
            if path.exists() and path.stat().st_size == 0:
                path.unlink()
    # The normal solver retires i+2. Its two lowest layers have no lower caller
    # to retire them because of our placeholders, so finish their 0.05 pruning.
    for step in (first, first + 1):
        native(args, p, "prune", step)
    for step in range(first, last + 1):
        check_target(p, step, True)


def execute(args, p):
    out = Path(p["output"])
    out.mkdir(parents=True, exist_ok=True)
    with lock_output(out):
        manifest = out / "bridge-plan.json"
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != p:
                raise ValueError("Existing plan differs from this invocation or source files changed")
        else:
            if any(out.glob("free12_4096_*")):
                raise ValueError("Output contains unmanaged tablebase files")
            save_json(manifest, p)
        cfg = dict(compress=False, compress_temp_files=False, optimal_branch_only=False,
                   algorithm_mode="exad", advanced_algo=True, zmask_algo=True,
                   SmallTileSumLimit=p["stsl"], deletion_threshold=0.05,
                   deletion_threshold_mode="absolute", success_rate_dtype="uint32",
                   ex_physical_transform=0, ex_inverse_physical_transform=0,
                   ex_physical_canonical_mode="full", ex_logical_pattern_signature=p["signature"],
                   ex_physical_pattern_signature=p["signature"], **{"4_spawn_rate": 0.1})
        Path(f"{p['prefix']}config.txt").write_text("".join(f"{k}: {v}\n" for k, v in cfg.items()), encoding="utf-8")
        progress_file = out / "bridge-progress.json"
        progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        phases = dict(prepare=prepare, generate=generate, boundary=boundaries, solve=solve)
        for phase in phases if args.command == "run" else [args.command]:
            if args.command == "run" and progress.get(phase):
                print(f"[{phase}] previously completed", flush=True)
                continue
            phases[phase](args, p)
            progress[phase] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_json(progress_file, progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "prepare", "generate", "boundary", "solve", "run"], nargs="?", default="plan")
    parser.add_argument("--source1", type=Path, default=Path("D:/free11-1k"))
    parser.add_argument("--source2", type=Path, default=Path("G:/free11-2k"))
    parser.add_argument("--output", type=Path, default=Path("D:/free12-4k"))
    parser.add_argument("--new-layers", type=int, default=64, help="Complete new layers after the two seeds (default 64, total 66)")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--native", type=Path, default=ROOT / "native_core/build-free12-bridge/free12_bridge.exe")
    parser.add_argument("--chunked-solve", action="store_true", help="Use the existing EXAD chunked solver")
    parser.add_argument("--direct-io", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0 or args.new_layers < 2:
        parser.error("threads must be positive and new-layers must be at least 2")
    p = plan(args)
    if args.command == "plan":
        print(json.dumps(p, ensure_ascii=False, indent=2))
    else:
        execute(args, p)


if __name__ == "__main__":
    main()
