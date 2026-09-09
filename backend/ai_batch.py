from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from Config import SingletonConfig
from backend.ai_decision import choose_full_ai_move
from engine_core.AIPlayer import CoreAILogic, Dispatcher
from engine_core.BoardMover import decode_board, encode_board, s_move_board
from native_core import ai_core


VRS_DTYPE = np.dtype([("board", "<u8"), ("score", "<u4"), ("move", "u1")])
DIRECTION_CODES = {"left": 1, "right": 2, "up": 3, "down": 4}
MAX_ENCODED_SCORE = int(np.iinfo(np.uint32).max)


@dataclass(frozen=True)
class GameJob:
    index: int
    seed: int
    start_board: int
    max_steps: int
    spawn_rate4: float
    time_ratio: float
    threads_per_game: int
    replay_dir: str
    decision_dir: str


@dataclass
class GameResult:
    index: int
    score: int = 0
    steps: int = 0
    final_board: str = "0000000000000000"
    max_tile: int = 0
    stop_reason: str = ""
    replay_path: str = ""
    decision_path: str = ""
    source_counts: dict[str, int] | None = None
    error: str = ""


class BatchAIRuntime:
    def __init__(
        self,
        board_encoded: int,
        board: np.ndarray,
        *,
        threads_per_game: int,
        spawn_rate4: float,
        time_ratio: float,
    ) -> None:
        self.dispatcher = Dispatcher(board, np.uint64(board_encoded))
        self.player = ai_core.AIPlayer(int(board_encoded))
        self.player.max_threads = int(threads_per_game)
        self.logic = CoreAILogic()
        self.spawn_rate4 = float(spawn_rate4)
        self.time_ratio = float(time_ratio)

    def fallback_provider(self, board_encoded: int, spawn_rate4: float, time_ratio: float):
        self.player.reset_board(int(board_encoded))
        self.player.update_spawn_rate(float(spawn_rate4))
        self.logic.time_limit_ratio = float(time_ratio)
        return self.player, self.logic

    def choose(self, board_encoded: int, board: np.ndarray, allow_resolve_32768: bool):
        return choose_full_ai_move(
            board_encoded=board_encoded,
            board=board,
            dispatcher=self.dispatcher,
            fallback_provider=self.fallback_provider,
            spawn_rate4=self.spawn_rate4,
            time_limit_ratio=self.time_ratio,
            allow_resolve_32768=allow_resolve_32768,
        )


def _spawn_tile(board_encoded: int, spawn_rate4: float, rng: random.Random) -> int:
    empty_positions = [
        position
        for position in range(16)
        if not (board_encoded >> (4 * position)) & 0xF
    ]
    if not empty_positions:
        return board_encoded
    position = rng.choice(empty_positions)
    value = 2 if rng.random() < spawn_rate4 else 1
    return board_encoded | (value << (4 * position))


def _new_game(
    start_board: int, spawn_rate4: float, rng: random.Random
) -> tuple[int, np.ndarray]:
    board_encoded = int(start_board)
    if board_encoded == 0:
        board_encoded = _spawn_tile(0, spawn_rate4, rng)
        board_encoded = _spawn_tile(board_encoded, spawn_rate4, rng)
    return board_encoded, decode_board(np.uint64(board_encoded))


def _clear_line_32k_pair(board: np.ndarray) -> tuple[str, tuple[int, int], tuple[int, int]] | None:
    positions = np.argwhere(board == 32768)
    if len(positions) != 2:
        return None
    first = (int(positions[0][0]), int(positions[0][1]))
    second = (int(positions[1][0]), int(positions[1][1]))
    if first[0] == second[0]:
        row = first[0]
        left, right = sorted((first[1], second[1]))
        if np.all(board[row, left + 1 : right] == 0):
            return "right", first, second
    if first[1] == second[1]:
        column = first[1]
        top, bottom = sorted((first[0], second[0]))
        if np.all(board[top + 1 : bottom, column] == 0):
            return "down", first, second
    return None


def _write_game_files(
    job: GameJob,
    history: list[tuple[int, int, int]],
    decisions: list[tuple[int, str]],
    score: int,
) -> tuple[str, str]:
    steps = len(history) - 1
    stem = f"game_{job.index:06d}_score_{score}_steps_{steps}"
    replay_path = Path(job.replay_dir) / f"{stem}.vrs"
    decision_path = Path(job.decision_dir) / f"{stem}.txt"

    replay = np.empty(len(history), dtype=VRS_DTYPE)
    for index, (board_encoded, board_score, move_code) in enumerate(history):
        replay[index] = (
            np.uint64(board_encoded),
            np.uint32(min(board_score, MAX_ENCODED_SCORE)),
            np.uint8(move_code),
        )

    replay_tmp = replay_path.with_suffix(replay_path.suffix + ".tmp")
    with replay_tmp.open("wb") as file:
        replay.tofile(file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(replay_tmp, replay_path)

    decision_tmp = decision_path.with_suffix(decision_path.suffix + ".tmp")
    with decision_tmp.open("w", encoding="utf-8", newline="\n") as file:
        for board_encoded, source in decisions:
            normalized_source = "_".join(str(source or "AI").split())
            file.write(f"{board_encoded:016x} {normalized_source}\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(decision_tmp, decision_path)
    return str(replay_path), str(decision_path)


def _run_game(job: GameJob) -> GameResult:
    try:
        rng = random.Random(job.seed)
        board_encoded, board = _new_game(job.start_board, job.spawn_rate4, rng)
        runtime = BatchAIRuntime(
            board_encoded,
            board,
            threads_per_game=job.threads_per_game,
            spawn_rate4=job.spawn_rate4,
            time_ratio=job.time_ratio,
        )
        history: list[tuple[int, int, int]] = [(board_encoded, 0, 0)]
        decisions: list[tuple[int, str]] = []
        source_counts: Counter[str] = Counter()
        score = 0
        ai_steps = 0
        has_65k = False
        initial_sum = int(np.sum(board))
        stop_reason = "game_over"

        while ai_steps < job.max_steps:
            decision_board = board_encoded
            decision = runtime.choose(
                board_encoded,
                board,
                allow_resolve_32768=not has_65k,
            )
            if decision.direction not in DIRECTION_CODES:
                stop_reason = "no_move"
                break

            move_code = DIRECTION_CODES[decision.direction]
            moved_board, move_score = s_move_board(np.uint64(board_encoded), move_code)
            moved_board = int(moved_board)
            if moved_board == board_encoded:
                stop_reason = "invalid_ai_move"
                break

            board_encoded = _spawn_tile(moved_board, job.spawn_rate4, rng)
            score += int(move_score)
            board = decode_board(np.uint64(board_encoded))
            decisions.append((decision_board, decision.source))
            source_counts[decision.source] += 1
            history.append((board_encoded, score, move_code))
            ai_steps += 1

            if initial_sum + 2.3 * ai_steps > 65536 and not has_65k:
                merge = _clear_line_32k_pair(board)
                if merge is not None:
                    direction, first, second = merge
                    special_source_board = board_encoded
                    board[first] = 16384
                    board[second] = 16384
                    board_encoded = int(encode_board(board))
                    score += 32768
                    special_move = DIRECTION_CODES[direction]
                    moved_board, move_score = s_move_board(
                        np.uint64(board_encoded), special_move
                    )
                    board_encoded = _spawn_tile(
                        int(moved_board), job.spawn_rate4, rng
                    )
                    score += int(move_score)
                    board = decode_board(np.uint64(board_encoded))
                    decisions.append((special_source_board, "AI"))
                    source_counts["AI"] += 1
                    history.append((board_encoded, score, special_move))
                    has_65k = True

        else:
            stop_reason = "max_steps"

        replay_path, decision_path = _write_game_files(
            job, history, decisions, score
        )
        max_tile = int(np.max(board)) if board.size else 0
        if has_65k:
            max_tile = max(max_tile, 65536)
        return GameResult(
            index=job.index,
            score=score,
            steps=len(history) - 1,
            final_board=f"{board_encoded:016x}",
            max_tile=max_tile,
            stop_reason=stop_reason,
            replay_path=replay_path,
            decision_path=decision_path,
            source_counts=dict(source_counts),
        )
    except Exception as exc:
        return GameResult(index=job.index, stop_reason="error", error=repr(exc))


def _atomic_write_json(path: Path, payload: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _worker_count(value: str) -> str | int:
    return "auto" if value.lower() == "auto" else _positive_int(value)


def _parse_board(value: str) -> int:
    try:
        board = int(value, 16)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("start board must be hexadecimal") from exc
    if not 0 <= board <= 0xFFFFFFFFFFFFFFFF:
        raise argparse.ArgumentTypeError("start board must fit in 64 bits")
    return board


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="2048EndgameTablebase --ai-test",
        description="Run the complete tablebase-first AI in batch mode.",
    )
    parser.add_argument("--games", type=_positive_int, default=100)
    parser.add_argument("--output", type=Path, default=Path.cwd() / "ai_test_results")
    parser.add_argument("--workers", type=_worker_count, default="auto")
    parser.add_argument("--threads-per-game", type=_positive_int, default=4)
    parser.add_argument("--time-ratio", type=float, default=2.1)
    parser.add_argument("--max-steps", type=_positive_int, default=65536)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-board", type=_parse_board, default=0)
    parser.add_argument("--spawn-rate4", type=float, default=None)
    return parser


def _new_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = output_root / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"run_{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _print(message: str) -> None:
    stream = getattr(sys, "stdout", None)
    if stream is not None:
        print(message, flush=True)


def _attach_parent_console() -> None:
    if os.name != "nt" or getattr(sys, "stdout", None) is not None:
        return
    try:
        import ctypes

        if ctypes.windll.kernel32.AttachConsole(-1):
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
            sys.stderr = open("CONOUT$", "w", encoding="utf-8", buffering=1)
    except (OSError, AttributeError):
        return


def run_batch(args: argparse.Namespace) -> int:
    config = SingletonConfig().config
    spawn_rate4 = (
        float(config.get("4_spawn_rate", 0.1))
        if args.spawn_rate4 is None
        else float(args.spawn_rate4)
    )
    if not 0.0 <= spawn_rate4 <= 1.0:
        raise ValueError("spawn-rate4 must be between 0 and 1")
    if args.time_ratio <= 0:
        raise ValueError("time-ratio must be greater than zero")

    cpu_count = os.cpu_count() or 1
    workers = (
        max(1, cpu_count // args.threads_per_game)
        if args.workers == "auto"
        else int(args.workers)
    )
    workers = min(workers, args.games)
    base_seed = int(args.seed if args.seed is not None else time.time_ns() & 0xFFFFFFFF)

    run_dir = _new_run_directory(args.output.expanduser().resolve())
    replay_dir = run_dir / "replays"
    decision_dir = run_dir / "decisions"
    replay_dir.mkdir()
    decision_dir.mkdir()

    run_info = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "games": args.games,
        "workers": workers,
        "threads_per_game": args.threads_per_game,
        "time_ratio": args.time_ratio,
        "max_steps": args.max_steps,
        "seed": base_seed,
        "start_board": f"{args.start_board:016x}",
        "spawn_rate4": spawn_rate4,
        "algorithm_mode": config.get("algorithm_mode", ""),
        "configured_table_count": len(config.get("filepath_map", {})),
    }
    _atomic_write_json(run_dir / "run.json", run_info)
    _atomic_write_json(
        run_dir / "progress.json",
        {"status": "running", "completed": 0, "failed": 0, "total": args.games},
    )

    jobs = [
        GameJob(
            index=index,
            seed=(base_seed + index * 0x9E3779B1) & 0xFFFFFFFF,
            start_board=args.start_board,
            max_steps=args.max_steps,
            spawn_rate4=spawn_rate4,
            time_ratio=args.time_ratio,
            threads_per_game=args.threads_per_game,
            replay_dir=str(replay_dir),
            decision_dir=str(decision_dir),
        )
        for index in range(1, args.games + 1)
    ]

    results: list[GameResult] = []
    failed = 0
    _print(f"AI batch output: {run_dir}")
    context = multiprocessing.get_context("spawn")
    pool = context.Pool(processes=workers)
    try:
        for result in pool.imap_unordered(_run_game, jobs, chunksize=1):
            results.append(result)
            failed += bool(result.error)
            completed = len(results)
            _atomic_write_json(
                run_dir / "progress.json",
                {
                    "status": "running",
                    "completed": completed,
                    "failed": failed,
                    "total": args.games,
                    "last_game": result.index,
                },
            )
            _print(f"Completed {completed}/{args.games}: game {result.index}")
        pool.close()
    except KeyboardInterrupt:
        pool.terminate()
        _atomic_write_json(
            run_dir / "progress.json",
            {
                "status": "cancelled",
                "completed": len(results),
                "failed": failed,
                "total": args.games,
            },
        )
        return 130
    finally:
        pool.join()

    successful = [result for result in results if not result.error]
    source_counts: Counter[str] = Counter()
    for result in successful:
        source_counts.update(result.source_counts or {})
    scores = [result.score for result in successful]
    summary = {
        "status": "complete" if not failed else "complete_with_errors",
        "finished_at": datetime.now().astimezone().isoformat(),
        "games": args.games,
        "successful": len(successful),
        "failed": failed,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_average": sum(scores) / len(scores) if scores else None,
        "total_steps": sum(result.steps for result in successful),
        "decision_sources": dict(source_counts),
        "results": [asdict(result) for result in sorted(results, key=lambda item: item.index)],
    }
    _atomic_write_json(run_dir / "summary.json", summary)
    _atomic_write_json(
        run_dir / "progress.json",
        {
            "status": summary["status"],
            "completed": len(results),
            "failed": failed,
            "total": args.games,
        },
    )
    run_info["status"] = summary["status"]
    run_info["finished_at"] = summary["finished_at"]
    _atomic_write_json(run_dir / "run.json", run_info)
    _print(f"AI batch finished: {run_dir}")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    _attach_parent_console()
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        return run_batch(args)
    except Exception as exc:
        parser.exit(2, f"AI batch failed: {exc}\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
