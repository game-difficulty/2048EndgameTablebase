"""Resume the bridge with an SSD hot prefix and asynchronous HDD retirement.

The default plan command is read-only. solve takes the original output lock;
it cannot run alongside the original bridge process.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import run

GIB = 1024 ** 3


def bounds(p):
    return p["seeds"][0]["target_step"], p["boundaries"][-1]["target_step"]


def hot_plan(p, directory):
    directory = directory.resolve()
    for forbidden in (Path(p["output"]), Path(p["source1"]).parent,
                      Path(p["source2"]).parent):
        forbidden = forbidden.resolve()
        if directory == forbidden or directory in forbidden.parents or forbidden in directory.parents:
            raise ValueError("SSD directory must be disjoint from output/source directories")
    return dict(p, prefix=str(directory / "free12_4096_"), cold_prefix=p["prefix"])


def inspect(p, hp):
    """Validate the full checkpoint and require a contiguous solved suffix."""
    first, last = bounds(p)
    frontier = None
    found_solved = False
    for step in range(first, last + 1):
        solved = run.existing_artifact(hp, step, ".exadbook").exists()
        if solved:
            found_solved = True
        elif found_solved:
            raise ValueError(f"Gap in solved suffix at layer {step}")
        else:
            frontier = step
        # Validate against the authoritative LUT on HDD, including before staging.
        path = run.existing_artifact(hp, step, ".exadbook" if solved else ".exadtmp")
        h = run.header(path, solved=solved)
        expected = run.check_target(dict(p, prefix=p["prefix"]), step, solved) if (
            path == run.artifact(p, step, ".exadbook" if solved else ".exadtmp")) else None
        if expected is None:
            if h["sum"] != run.BASE12 + step * 2 or h["logical"] != p["signature"] or h["physical"] != p["signature"]:
                raise ValueError(f"Invalid hot checkpoint: {path}")
            with Path(f"{p['prefix']}.exadlut").open("rb") as stream:
                lut = run.struct.unpack("<8s2I2BH4x3Q", stream.read(48))
            if h["lut"] != lut[6]:
                raise ValueError(f"Hot checkpoint LUT mismatch: {path}")
    if frontier is not None and frontier > last - 2:
        raise ValueError("Both boundary layers must already be finalized")
    return frontier


def copy_publish(source, destination, *, solved=False, remove_source=False, reserve_bytes=0):
    """Stream copy, flush, validate, atomically publish, then optionally retire.

    No recursive operations, and no deletion before successful publication.
    Header validation checks layout/length, not a full payload checksum.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Copy source and destination must differ")
    before = source.stat()
    source_header = run.header(source) if solved else None
    if shutil.disk_usage(destination.parent).free < before.st_size + reserve_bytes:
        raise OSError(f"Insufficient free space to publish {destination}")
    temporary = destination.with_name(destination.name + ".ssd-copying")
    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while copying: {source}")
    if temporary.stat().st_size != before.st_size:
        raise RuntimeError(f"Truncated copy: {temporary}")
    if solved and run.header(temporary) != source_header:
        raise RuntimeError(f"Copied EXAD metadata mismatch: {temporary}")
    os.replace(temporary, destination)
    if remove_source:
        source.unlink()


def completed_step(hp):
    path = Path(f"{hp['prefix']}exad_solve_stats.csv")
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8")
    # A native append can be in progress; only consume complete records.
    content = content[:content.rfind("\n") + 1]
    steps = [int(row["step"]) for row in csv.DictReader(io.StringIO(content))
             if row.get("stage") == "chunked_solve" and row.get("time")]
    return min(steps) if steps else None


class Archiver:
    def __init__(self, p, hp, frontier, reserve_bytes=GIB):
        self.p, self.hp, self.frontier = p, hp, frontier
        self.reserve_bytes = reserve_bytes
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.work, name="bridge-archive", daemon=False)

    def retire(self, minimum):
        first, last = bounds(self.p)
        for step in range(last, max(first, minimum) - 1, -1):
            source = run.artifact(self.hp, step, ".exadbook")
            if not source.exists():
                continue
            destination = run.artifact(self.p, step, ".exadbook")
            print(f"[archive] layer={step} bytes={source.stat().st_size}", flush=True)
            copy_publish(source, destination, solved=True, remove_source=True,
                         reserve_bytes=self.reserve_bytes)
            print(f"[archived] layer={step}", flush=True)

    def work(self):
        try:
            while not self.stop.is_set():
                completed = completed_step(self.hp)
                # On resume, layers strictly above the two active futures are retired.
                minimum = self.frontier + 3 if self.frontier is not None else bounds(self.p)[1] + 1
                if completed is not None:
                    minimum = min(minimum, completed + 2)
                # Lowest two are explicitly pruned by the worker after native return.
                self.retire(max(bounds(self.p)[0] + 2, minimum))
                self.stop.wait(2)
        except BaseException as exc:
            self.error = exc


def stage(p, hp, frontier, reserve_bytes):
    directory = Path(hp["prefix"]).parent
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "bridge-ssd-plan.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != hp:
            raise ValueError("SSD directory belongs to a different bridge plan")
    elif any(directory.iterdir()):
        raise ValueError("Use an empty dedicated SSD directory")
    else:
        run.save_json(manifest, hp)
    for suffix in (".exadlut", "config.txt"):
        source = Path(f"{p['prefix']}{suffix}")
        if source.exists():
            copy_publish(source, Path(f"{hp['prefix']}{suffix}"), reserve_bytes=reserve_bytes)
    futures = (frontier + 1, frontier + 2) if frontier is not None else (bounds(p)[0], bounds(p)[0] + 1)
    for step in futures:
        destination = run.artifact(hp, step, ".exadbook")
        if not destination.exists():
            print(f"[stage] future layer={step}", flush=True)
            copy_publish(run.artifact(p, step, ".exadbook"), destination,
                         solved=True, reserve_bytes=reserve_bytes)
        run.check_target(hp, step, True)


def worker(manifest, threads, native, direct_io):
    hp = json.loads(Path(manifest).read_text(encoding="utf-8"))
    args = argparse.Namespace(threads=threads, native=Path(native),
                              chunked_solve=True, direct_io=direct_io)
    run.solve(args, hp)


def solve_ssd(args, p, hp):
    # The existing process owns this exact lock. We never stop that process.
    with run.lock_output(Path(p["output"])):
        frontier = inspect(p, hp)
        reserve = int(args.reserve_gib * GIB)
        stage(p, hp, frontier, reserve)
        archiver = Archiver(p, hp, frontier)
        command = [sys.executable, "-u", "-c",
                   "import sys; sys.path.insert(0, sys.argv[1]); import ssd; "
                   "ssd.worker(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]=='1')",
                   str(Path(__file__).parent), str(Path(hp["prefix"]).parent / "bridge-ssd-plan.json"),
                   str(args.threads), str(args.native.resolve()), str(int(args.direct_io))]
        process = subprocess.Popen(command, cwd=run.ROOT)
        archiver.thread.start()
        try:
            while process.poll() is None:
                if archiver.error:
                    raise RuntimeError("SSD archive failed; retaining unarchived hot files") from archiver.error
                if shutil.disk_usage(Path(hp["prefix"]).parent).free < reserve:
                    raise OSError("SSD reserve reached; stopping this worker for resumable recovery")
                time.sleep(2)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            # Only the child started above is ever terminated, never an existing job.
            if process.poll() is None:
                process.terminate()
                process.wait()
            archiver.stop.set()
            archiver.thread.join()
        if archiver.error:
            raise RuntimeError("Archive failed; hot checkpoints retained") from archiver.error
        archiver.retire(bounds(p)[0])
        for step in range(bounds(p)[0], bounds(p)[1] + 1):
            run.check_target(p, step, True)
        progress_file = Path(p["output"]) / "bridge-progress.json"
        progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        progress["solve"] = time.strftime("%Y-%m-%d %H:%M:%S")
        run.save_json(progress_file, progress)
        print("[complete] All solved layers archived and validated on HDD", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "solve"), nargs="?", default="plan")
    parser.add_argument("--output", type=Path, default=Path("D:/free12-4k"))
    parser.add_argument("--hot-directory", type=Path, default=Path("C:/2048_tables/tmp/free12-4k"))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--reserve-gib", type=float, default=64)
    parser.add_argument("--native", type=Path, default=run.ROOT / "native_core/build-free12-bridge/free12_bridge.exe")
    parser.add_argument("--direct-io", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0 or args.reserve_gib < 1:
        parser.error("threads must be positive and reserve-gib must be at least 1")
    p = json.loads((args.output / "bridge-plan.json").read_text(encoding="utf-8"))
    if Path(p["output"]).resolve() != args.output.resolve():
        raise ValueError("Output does not match stored bridge plan")
    hp = hot_plan(p, args.hot_directory)
    if args.command == "plan":
        frontier = inspect(p, hp)
        print(json.dumps(dict(next_layer=frontier, hot_prefix=hp["prefix"],
                              cold_prefix=p["prefix"], reserve_gib=args.reserve_gib,
                              note="Read-only snapshot; revalidated under lock before solve"), indent=2))
    else:
        solve_ssd(args, p, hp)


if __name__ == "__main__":
    main()
