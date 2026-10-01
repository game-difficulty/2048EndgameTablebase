"""Generate 955..984 in isolation, extend the existing plan, then resume SSD solve."""
import argparse
import json
from pathlib import Path
import time

import run
import ssd


def extend(args):
    out = args.output.resolve()
    with run.lock_output(out):
        manifest = out / "bridge-plan.json"
        backup = out / "bridge-before-955-plan.json"
        current = json.loads(manifest.read_text(encoding="utf-8"))
        if not backup.exists():
            if current["seeds"][0]["target_step"] != 980:
                raise ValueError("Expected the original 980 bridge plan")
            run.save_json(backup, current)
        original = json.loads(backup.read_text(encoding="utf-8"))
        updated = dict(original, new_layers=89)
        seeds = []
        for source_step in (444, 445):
            path = Path(f"{original['source1']}{source_step}.exadbook")
            h = run.header(path)
            target_step = source_step + 511
            if h["sum"] - 32768 + 1024 != run.BASE12 + 2 * target_step:
                raise ValueError("Seed sum mismatch")
            seeds.append(dict(source_step=source_step, target_step=target_step,
                              target_sum=run.BASE12 + 2 * target_step, path=str(path),
                              rows=h["rows"], bytes=path.stat().st_size,
                              mtime_ns=path.stat().st_mtime_ns))
        updated["seeds"] = seeds
        updated["extension"] = dict(original_first_layer=980, regenerated_through=984,
                                    preserved_generation_from=985)
        if current not in (original, updated):
            raise ValueError("Main plan changed unexpectedly")
        hp = ssd.hot_plan(updated, args.hot_directory)
        old_hp = ssd.hot_plan(original, args.hot_directory)
        hot_manifest = args.hot_directory / "bridge-ssd-plan.json"
        if json.loads(hot_manifest.read_text(encoding="utf-8")) not in (old_hp, hp):
            raise ValueError("Hot plan changed unexpectedly")
        # Never regenerate a layer whose probabilities already exist.
        for step in range(955, 985):
            for prefix in (updated["prefix"], hp["prefix"]):
                path = Path(f"{prefix}{step}.exadbook")
                if path.exists() and path.stat().st_size:
                    raise ValueError(f"Would overlap a solved layer: {path}")

        def solved_snapshot():
            return {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                    for directory in (out, args.hot_directory)
                    for path in directory.glob("free12_4096_*.exadbook")
                    if path.stat().st_size}

        before = solved_snapshot()
        directory = args.generation_directory.resolve()
        if directory in (out, args.hot_directory.resolve()) or out in directory.parents or args.hot_directory.resolve() in directory.parents:
            raise ValueError("Generation directory must be independent")
        directory.mkdir(parents=True, exist_ok=True)
        gp = dict(updated, prefix=str(directory / "free12_4096_"))
        generation_manifest = directory / "extension-plan.json"
        if generation_manifest.exists():
            if json.loads(generation_manifest.read_text(encoding="utf-8")) != gp:
                raise ValueError("Generation directory plan differs")
        elif any(directory.iterdir()):
            raise ValueError("Generation directory is not empty")
        else:
            run.save_json(generation_manifest, gp)
        for seed in seeds:
            step = seed["target_step"]
            if not run.artifact(gp, step, ".exadtmp").exists():
                run.native(args, gp, "prepare", step, seed["path"], f"{original['source1']}.exadlut")
            run.check_target(gp, step, False)
        generated = directory / "generation-complete.json"
        if not generated.exists():
            print("[extension] generating 955..984; existing solved layers untouched", flush=True)
            run.native(args, gp, "generate", 984)
        for step in range(955, 985):
            run.check_target(gp, step, False)
        if Path(f"{gp['prefix']}.exadlut").read_bytes() != Path(f"{updated['prefix']}.exadlut").read_bytes():
            raise ValueError("Generated LUT differs from the existing target LUT")
        run.save_json(generated, dict(last=984, time=time.strftime("%Y-%m-%d %H:%M:%S")))
        # Explicitly requested replacement, only after the new files have validated.
        for step in range(980, 985):
            path = run.artifact(updated, step, ".exadtmp")
            if path.exists():
                path.unlink()
        for step in range(955, 985):
            source = run.artifact(gp, step, ".exadtmp")
            destination = run.artifact(updated, step, ".exadtmp")
            ssd.copy_publish(source, destination, reserve_bytes=ssd.GIB)
            run.check_target(updated, step, False)
            print(f"[installed] generation layer={step}", flush=True)
        # Old force-terminated runs may have left the old lower-range sentinels.
        for step in range(955, 980):
            for prefix in (updated["prefix"], hp["prefix"]):
                path = Path(f"{prefix}{step}.exadbook")
                if path.exists():
                    if path.stat().st_size:
                        raise ValueError(f"Unexpected real solved layer: {path}")
                    path.unlink()
        if before != solved_snapshot():
            raise RuntimeError("An existing solved checkpoint changed during generation")
        # Validate the combined old and new interval before changing either plan.
        frontier = ssd.inspect(updated, hp)
        run.save_json(hot_manifest, hp)
        run.save_json(manifest, updated)
        progress_path = out / "bridge-progress.json"
        progress = json.loads(progress_path.read_text())
        progress.pop("solve", None)
        progress["extension_955_through_984"] = time.strftime("%Y-%m-%d %H:%M:%S")
        run.save_json(progress_path, progress)
        print(f"[extension-ready] resume={frontier}; lower=955; solved checkpoints preserved={len(before)}", flush=True)
    ssd.solve_ssd(args, updated, hp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("D:/free12-4k"))
    parser.add_argument("--hot-directory", type=Path, default=Path("C:/2048_tables/tmp/free12-4k"))
    parser.add_argument("--generation-directory", type=Path, default=Path("C:/2048_tables/tmp/free12-extension-955"))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--reserve-gib", type=float, default=64)
    parser.add_argument("--native", type=Path, default=run.ROOT / "native_core/build-free12-bridge/free12_bridge.exe")
    parser.add_argument("--direct-io", action="store_true")
    extend(parser.parse_args())
