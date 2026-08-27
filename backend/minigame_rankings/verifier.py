from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import uuid
from typing import Any

from datetime import datetime, timedelta, timezone

from backend.auth.db import auth_db


VERIFY_TIMEOUT_SECONDS = float(os.getenv("MINIGAME_VERIFY_TIMEOUT_SECONDS", "90"))
VERIFY_RETRY_DELAY_SECONDS = float(os.getenv("MINIGAME_VERIFY_RETRY_DELAY_SECONDS", "30"))


class MinigameVerifierError(RuntimeError):
    pass


class MinigameVerifierUnavailable(MinigameVerifierError):
    pass


class _NodeVerifier:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None

    @staticmethod
    def _command() -> tuple[list[str], Path]:
        root = Path(__file__).resolve().parents[2]
        script = root / "frontend" / "scripts" / "minigameVerifier.mjs"
        node = str(os.getenv("MINIGAME_NODE_BINARY") or "node")
        return [node, str(script)], root / "frontend"

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                try:
                    self._responses.put(json.loads(line))
                except (TypeError, ValueError):
                    self._responses.put({"ok": False, "error": "invalid_verifier_response"})
        finally:
            self._responses.put(None)

    def _start(self) -> None:
        self._stop()
        command, cwd = self._command()
        self._process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            close_fds=os.name != "nt",
        )
        self._reader = threading.Thread(
            target=self._read_stdout,
            args=(self._process,),
            name="minigame-node-verifier-reader",
            daemon=True,
        )
        self._reader.start()

    def _stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        while True:
            try:
                self._responses.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        with self._lock:
            self._stop()

    def verify(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        request = {"request_id": request_id, **payload}
        with self._lock:
            for attempt in range(2):
                process = self._process
                if process is None or process.poll() is not None:
                    try:
                        self._start()
                    except OSError as exc:
                        raise MinigameVerifierUnavailable("verifier_start_failed") from exc
                    process = self._process
                try:
                    assert process is not None and process.stdin is not None
                    process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                    process.stdin.flush()
                    response = self._responses.get(timeout=VERIFY_TIMEOUT_SECONDS)
                except (BrokenPipeError, OSError, queue.Empty):
                    self._stop()
                    if attempt == 0:
                        continue
                    raise MinigameVerifierUnavailable("verifier_unavailable")
                if response is None:
                    self._stop()
                    if attempt == 0:
                        continue
                    raise MinigameVerifierUnavailable("verifier_stopped")
                if str(response.get("request_id")) != request_id:
                    self._stop()
                    raise MinigameVerifierUnavailable("verifier_response_mismatch")
                if response.get("error") == "invalid_verifier_response":
                    self._stop()
                    raise MinigameVerifierUnavailable("invalid_verifier_response")
                if not response.get("ok"):
                    raise MinigameVerifierError(str(response.get("error") or "verification_failed"))
                result = response.get("result")
                if not isinstance(result, dict):
                    raise MinigameVerifierUnavailable("invalid_verifier_result")
                return result
        raise MinigameVerifierUnavailable("verifier_unavailable")


_verifier = _NodeVerifier()
_retry_after_monotonic = 0.0


def verify_ranked_run(run: dict[str, Any]) -> dict[str, Any]:
    from .service import _derive_seed_hex

    derived_seed = _derive_seed_hex(
        run_id=str(run["run_id"]),
        user_id=int(run["user_id"]),
        game_id=str(run["game_id"]),
        difficulty=int(run["difficulty"]),
        rules_version=int(run["rules_version"]),
        salt_hex=str(run["seed_salt_hex"]),
        started_at=str(run["started_at"]),
        ip_address=str(run.get("start_ip") or ""),
    )
    if derived_seed != str(run["seed_hex"]):
        raise MinigameVerifierError("run_seed_mismatch")
    summary = dict(run.get("claimed_summary") or {})
    expected = {
        "run_id": str(run["run_id"]),
        "seed_hex": str(run["seed_hex"]),
        "rules_version": int(run["rules_version"]),
        "game_id": str(run["game_id"]),
        "difficulty": int(run["difficulty"]),
        **summary,
    }
    return _verifier.verify(
        {
            "record_encoding": str(run.get("pending_record") or ""),
            "expected": expected,
        }
    )


def close_verifier() -> None:
    _verifier.close()


def _error_code(error: BaseException) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(error or "verification_failed"))
    return (value.strip("_") or "verification_failed")[:64]


def process_one_pending_run() -> bool:
    global _retry_after_monotonic

    from .service import claim_pending, finish_rejected, finish_verified

    if time.monotonic() < _retry_after_monotonic:
        return False
    run = claim_pending()
    if run is None:
        return False
    try:
        result = verify_ranked_run(run)
        finish_verified(
            str(run["run_id"]),
            score=int(result["score"]),
            trophy_tier=int(result["trophy_tier"]),
            highest_tile_exp=int(result["highest_tile_exp"]),
            final_board=[int(value) for value in result["final_board"]],
            board_rows=int(result["board_rows"]),
            board_cols=int(result["board_cols"]),
            action_count=int(result["action_count"]),
            elapsed_ms=int(result["elapsed_ms"]),
        )
    except MinigameVerifierUnavailable:
        _retry_after_monotonic = time.monotonic() + VERIFY_RETRY_DELAY_SECONDS
        with auth_db() as db:
            db.execute(
                """
                UPDATE minigame_ranked_runs
                SET status = 'pending', validation_started_at = NULL
                WHERE run_id = ? AND status = 'validating'
                """,
                (str(run["run_id"]),),
            )
        return False
    except MinigameVerifierError as exc:
        finish_rejected(str(run["run_id"]), _error_code(exc))
    except Exception:
        _retry_after_monotonic = time.monotonic() + VERIFY_RETRY_DELAY_SECONDS
        with auth_db() as db:
            db.execute(
                """
                UPDATE minigame_ranked_runs
                SET status = 'pending', validation_started_at = NULL
                WHERE run_id = ? AND status = 'validating'
                """,
                (str(run["run_id"]),),
            )
        return False
    _retry_after_monotonic = 0.0
    return True


def cleanup_stale_ranked_runs() -> None:
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=30)).isoformat()
    with auth_db() as db:
        db.execute(
            """
            UPDATE minigame_ranked_runs
            SET status = 'expired', completed_at = ?, pending_record = NULL
            WHERE status IN ('active', 'qualified') AND expires_at <= ?
            """,
            (now.isoformat(), now.isoformat()),
        )
        db.execute(
            """
            DELETE FROM minigame_ranked_runs
            WHERE status IN ('verified', 'rejected', 'expired', 'not_candidate')
              AND COALESCE(completed_at, started_at) < ?
            """,
            (cutoff,),
        )


def prepare_validation_queue() -> None:
    with auth_db() as db:
        db.execute(
            """
            UPDATE minigame_ranked_runs
            SET status = 'pending', validation_started_at = NULL
            WHERE status = 'validating'
            """
        )
    cleanup_stale_ranked_runs()


__all__ = [
    "MinigameVerifierError",
    "MinigameVerifierUnavailable",
    "cleanup_stale_ranked_runs",
    "close_verifier",
    "prepare_validation_queue",
    "process_one_pending_run",
    "verify_ranked_run",
]
