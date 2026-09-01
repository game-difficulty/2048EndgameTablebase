from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from typing import Any
import uuid

from backend.auth.db import auth_db, get_auth_db_path

from .catalog import MINIGAME_BY_ID, minigame_catalog
from .mgo1 import parse_mgo1_envelope


LEADERBOARD_LIMIT = 100
RANKED_RULES_VERSION = 1
RUN_LIFETIME = timedelta(hours=24)
SUBMISSION_TOKEN_LIFETIME = timedelta(minutes=10)
LEASE_LIFETIME = timedelta(seconds=60)
MAX_PENDING_RECORD_BYTES = 256 * 1024
MAX_PENDING_GLOBAL = 32
MAX_PENDING_PER_USER = 4
MAX_CHECKPOINTS_PER_RUN = 64
MGO_RECORD_PREFIX = "MINIGAME_v1MGO_B64_"


class RunTokenError(ValueError):
    pass


class RunTokenExpired(RunTokenError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _token_secret() -> bytes:
    configured = str(os.getenv("MINIGAME_RANKING_SECRET") or "").strip()
    if configured:
        return configured.encode("utf-8")

    secret_path = get_auth_db_path().with_name("minigame_rankings.secret")
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return bytes.fromhex(secret_path.read_text(encoding="ascii").strip())
    except (FileNotFoundError, ValueError):
        generated = secrets.token_bytes(32)
        try:
            descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return bytes.fromhex(secret_path.read_text(encoding="ascii").strip())
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(generated.hex())
        return generated


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _sign_token(kind: str, claims: dict[str, Any], expires_at: datetime) -> str:
    payload = {
        "v": 1,
        "kind": str(kind),
        **claims,
        "exp": int(expires_at.timestamp()),
    }
    encoded_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hmac.new(_token_secret(), encoded_payload, hashlib.sha256).digest()
    return f"{_b64url_encode(encoded_payload)}.{_b64url_encode(signature)}"


def _verify_token(token: str, *, kind: str, now: datetime) -> dict[str, Any]:
    try:
        encoded_payload, encoded_signature = str(token or "").strip().split(".", 1)
        payload_bytes = _b64url_decode(encoded_payload)
        signature = _b64url_decode(encoded_signature)
        expected = hmac.new(_token_secret(), payload_bytes, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise RunTokenError("invalid_token")
        payload = json.loads(payload_bytes.decode("utf-8"))
        if payload.get("v") != 1 or payload.get("kind") != kind:
            raise RunTokenError("invalid_token")
        if int(payload.get("exp", 0)) <= int(now.timestamp()):
            raise RunTokenExpired("token_expired")
        return payload
    except RunTokenError:
        raise
    except Exception as exc:
        raise RunTokenError("invalid_token") from exc


def _derive_seed_hex(
    *,
    run_id: str,
    user_id: int,
    game_id: str,
    difficulty: int,
    rules_version: int,
    salt_hex: str,
    started_at: str,
    ip_address: str,
) -> str:
    message = "|".join(
        (
            "minigame-seed-v1",
            str(rules_version),
            str(run_id),
            str(int(user_id)),
            str(game_id),
            str(int(difficulty)),
            str(started_at),
            str(ip_address or ""),
            str(salt_hex),
        )
    ).encode("utf-8")
    return hmac.new(_token_secret(), message, hashlib.sha256).digest()[:16].hex()


def _summary_hash(summary: dict[str, Any]) -> str:
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lease_hash(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def _lease_matches(row: sqlite3.Row | dict[str, Any], token: str) -> bool:
    expected = str(row["lease_token_hash"] or "")
    supplied = _lease_hash(token)
    return bool(expected) and secrets.compare_digest(expected, supplied)


def _new_lease(now: datetime, token: str) -> tuple[str, str, str]:
    normalized = str(token or "").strip()
    if len(normalized) < 16 or len(normalized) > 256:
        raise ValueError("invalid_lease_token")
    return normalized, _lease_hash(normalized), _iso(now + LEASE_LIFETIME)


def _lease_expired(row: sqlite3.Row | dict[str, Any], now: datetime) -> bool:
    raw_expiry = row["lease_expires_at"]
    if not raw_expiry:
        return True
    try:
        return datetime.fromisoformat(str(raw_expiry)) <= now
    except ValueError:
        return True


def _assert_live_lease(
    row: sqlite3.Row | dict[str, Any],
    lease_token: str,
    now: datetime,
) -> None:
    if not _lease_matches(row, lease_token):
        raise PermissionError("lease_mismatch")
    if _lease_expired(row, now):
        raise PermissionError("lease_expired")


def _entry_key(kind: str, user_id: int, game_id: str, difficulty: int) -> str:
    raw = f"minigame:{kind}:{user_id}:{game_id}:{difficulty}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _avatar_url(avatar_key: str | None) -> str | None:
    return f"/media/avatars/{avatar_key}" if avatar_key else None


def _run_token_for(row: sqlite3.Row | dict[str, Any]) -> str:
    return _sign_token(
        "run",
        {
            "rid": str(row["run_id"]),
            "uid": int(row["user_id"]),
            "game": str(row["game_id"]),
            "difficulty": int(row["difficulty"]),
            "rules": int(row["rules_version"]),
        },
        datetime.fromisoformat(str(row["expires_at"])),
    )


def _public_run(
    row: sqlite3.Row | dict[str, Any],
    *,
    include_run_token: bool = False,
) -> dict[str, Any]:
    payload = {
        "run_id": str(row["run_id"]),
        "game_id": str(row["game_id"]),
        "difficulty": int(row["difficulty"]),
        "rules_version": int(row["rules_version"]),
        "seed_hex": str(row["seed_hex"]),
        "status": str(row["status"]),
        "started_at": str(row["started_at"]),
        "expires_at": str(row["expires_at"]),
        "qualified_at": row["qualified_at"],
        "submitted_at": row["submitted_at"],
        "completed_at": row["completed_at"],
        "error_code": row["error_code"],
        "lease_expires_at": row["lease_expires_at"],
        "lease_generation": int(row["lease_generation"] or 1),
    }
    if include_run_token:
        payload["run_token"] = _run_token_for(row)
    return payload


def _public_checkpoint(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_id": int(row["id"]),
        "run_id": str(row["run_id"]),
        "revision": int(row["revision"]),
        "status": str(row["status"]),
        "action_count": int(row["action_count"] or 0),
        "submitted_at": str(row["submitted_at"]),
        "completed_at": row["completed_at"],
        "error_code": row["error_code"],
    }


def _normalize_game(game_id: str, difficulty: int) -> tuple[str, int]:
    normalized_game_id = str(game_id or "").strip()
    if normalized_game_id not in MINIGAME_BY_ID:
        raise ValueError("unknown_game")
    try:
        normalized_difficulty = int(difficulty)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_difficulty") from exc
    if normalized_difficulty not in (0, 1):
        raise ValueError("invalid_difficulty")
    return normalized_game_id, normalized_difficulty


def _normalize_summary(
    *,
    score: int,
    trophy_tier: int,
    highest_tile_exp: int,
    final_board: list[int],
    board_rows: int,
    board_cols: int,
    action_count: int,
    elapsed_ms: int,
) -> dict[str, Any]:
    normalized_score = int(score)
    normalized_trophy = int(trophy_tier)
    normalized_highest = int(highest_tile_exp)
    rows = int(board_rows)
    cols = int(board_cols)
    actions = int(action_count)
    elapsed = int(elapsed_ms)
    board = [int(value) for value in final_board]
    if normalized_score < 0 or normalized_score > 2**63 - 1:
        raise ValueError("invalid_score")
    if normalized_trophy < 0 or normalized_trophy > 4:
        raise ValueError("invalid_trophy_tier")
    if normalized_highest < 0 or normalized_highest > 63:
        raise ValueError("invalid_highest_tile")
    if rows < 1 or rows > 8 or cols < 1 or cols > 8 or rows * cols != len(board):
        raise ValueError("invalid_board_shape")
    if len(board) > 64 or any(value < -1 or value > 63 for value in board):
        raise ValueError("invalid_board_data")
    if actions < 0 or actions > 50_000:
        raise ValueError("invalid_action_count")
    if elapsed < 0 or elapsed > 7 * 24 * 60 * 60 * 1000:
        raise ValueError("invalid_elapsed_ms")
    return {
        "score": normalized_score,
        "trophy_tier": normalized_trophy,
        "highest_tile_exp": normalized_highest,
        "final_board": board,
        "board_rows": rows,
        "board_cols": cols,
        "action_count": actions,
        "elapsed_ms": elapsed,
    }


def create_ranked_run(
    *,
    user_id: int,
    request_id: str,
    game_id: str,
    difficulty: int,
    ip_address: str,
    lease_token: str,
    replace_run_id: str | None = None,
    replace_lease_token: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    normalized_request = str(request_id or "").strip()
    if not normalized_request or len(normalized_request) > 128:
        raise ValueError("invalid_request_id")
    normalized_game_id, normalized_difficulty = _normalize_game(game_id, difficulty)
    current = now or _utc_now()
    expires = current + RUN_LIFETIME
    lease_token, lease_hash, lease_expires_at = _new_lease(current, lease_token)
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE user_id = ? AND request_id = ?",
            (int(user_id), normalized_request),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["game_id"]) != normalized_game_id
                or int(existing["difficulty"]) != normalized_difficulty
            ):
                raise ValueError("request_id_conflict")
            if str(existing["status"]) not in {"active", "qualified"}:
                return _public_run(existing)
            if not _lease_matches(existing, lease_token):
                raise RuntimeError("active_run_exists")
            db.execute(
                """
                UPDATE minigame_ranked_runs
                SET lease_expires_at = ?, lease_last_seen_at = ?
                WHERE run_id = ? AND status IN ('active', 'qualified')
                """,
                (lease_expires_at, _iso(current), str(existing["run_id"])),
            )
            existing = db.execute(
                "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
                (str(existing["run_id"]),),
            ).fetchone()
            payload = _public_run(existing, include_run_token=True)
            payload["lease_token"] = lease_token
            return payload

        active_rows = db.execute(
            """
            SELECT * FROM minigame_ranked_runs
            WHERE user_id = ? AND game_id = ? AND difficulty = ?
              AND status IN ('active', 'qualified')
            ORDER BY started_at ASC
            """,
            (int(user_id), normalized_game_id, normalized_difficulty),
        ).fetchall()
        for active in active_rows:
            if _lease_expired(active, current):
                db.execute(
                    """
                    UPDATE minigame_ranked_runs
                    SET status = 'expired', completed_at = ?, error_code = 'lease_expired',
                        lease_token_hash = NULL, lease_expires_at = NULL
                    WHERE run_id = ? AND status IN ('active', 'qualified')
                    """,
                    (_iso(current), str(active["run_id"])),
                )
        active_rows = db.execute(
            """
            SELECT * FROM minigame_ranked_runs
            WHERE user_id = ? AND game_id = ? AND difficulty = ?
              AND status IN ('active', 'qualified')
            ORDER BY started_at ASC
            """,
            (int(user_id), normalized_game_id, normalized_difficulty),
        ).fetchall()
        if active_rows:
            active = active_rows[0]
            if str(replace_run_id or "") != str(active["run_id"]):
                raise RuntimeError("active_run_exists")
            if not _lease_matches(active, str(replace_lease_token or "")):
                raise PermissionError("lease_mismatch")
            db.execute(
                """
                UPDATE minigame_ranked_runs
                SET status = 'expired', completed_at = ?, error_code = 'replaced',
                    lease_token_hash = NULL, lease_expires_at = NULL
                WHERE run_id = ? AND status IN ('active', 'qualified')
                """,
                (_iso(current), str(active["run_id"])),
            )

        run_id = str(uuid.uuid4())
        salt_hex = secrets.token_hex(16)
        started_at = _iso(current)
        start_ip = str(ip_address or "")[:128]
        seed_hex = _derive_seed_hex(
            run_id=run_id,
            user_id=int(user_id),
            game_id=normalized_game_id,
            difficulty=normalized_difficulty,
            rules_version=RANKED_RULES_VERSION,
            salt_hex=salt_hex,
            started_at=started_at,
            ip_address=start_ip,
        )
        db.execute(
            """
            INSERT INTO minigame_ranked_runs
            (run_id, user_id, request_id, game_id, difficulty, rules_version,
             seed_salt_hex, seed_hex, lease_token_hash, lease_expires_at,
             lease_last_seen_at, lease_generation, status, started_at, expires_at, start_ip)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'active', ?, ?, ?)
            """,
            (
                run_id,
                int(user_id),
                normalized_request,
                normalized_game_id,
                normalized_difficulty,
                RANKED_RULES_VERSION,
                salt_hex,
                seed_hex,
                lease_hash,
                lease_expires_at,
                _iso(current),
                started_at,
                _iso(expires),
                start_ip,
            ),
        )
        row = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    payload = _public_run(row, include_run_token=True)
    payload["lease_token"] = lease_token
    return payload


def heartbeat_ranked_run(
    *,
    run_id: str,
    user_id: int,
    lease_token: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or _utc_now()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if row is None:
            raise LookupError("run_not_found")
        if int(row["user_id"]) != int(user_id):
            raise PermissionError("run_owner_mismatch")
        if str(row["status"]) not in {"active", "qualified"}:
            raise RuntimeError("run_not_active")
        if not _lease_matches(row, lease_token):
            raise PermissionError("lease_mismatch")
        if datetime.fromisoformat(str(row["expires_at"])) <= current:
            db.execute(
                """
                UPDATE minigame_ranked_runs
                SET status = 'expired', completed_at = ?, error_code = 'run_expired',
                    lease_token_hash = NULL, lease_expires_at = NULL
                WHERE run_id = ? AND status IN ('active', 'qualified')
                """,
                (_iso(current), str(run_id)),
            )
            raise RunTokenExpired("run_expired")
        lease_expires_at = _iso(current + LEASE_LIFETIME)
        db.execute(
            """
            UPDATE minigame_ranked_runs
            SET lease_expires_at = ?, lease_last_seen_at = ?
            WHERE run_id = ? AND status IN ('active', 'qualified')
            """,
            (lease_expires_at, _iso(current), str(run_id)),
        )
        saved = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
    return _public_run(saved)


def claim_ranked_run(
    *,
    run_id: str,
    user_id: int,
    lease_token: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or _utc_now()
    lease_token, lease_hash, lease_expires_at = _new_lease(current, lease_token)
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if row is None:
            raise LookupError("run_not_found")
        if int(row["user_id"]) != int(user_id):
            raise PermissionError("run_owner_mismatch")
        if str(row["status"]) not in {"active", "qualified"}:
            raise RuntimeError("run_not_active")
        if datetime.fromisoformat(str(row["expires_at"])) <= current:
            raise RunTokenExpired("run_expired")
        if not _lease_expired(row, current):
            if not _lease_matches(row, lease_token):
                raise RuntimeError("lease_active")
            next_generation = int(row["lease_generation"] or 1)
        else:
            next_generation = int(row["lease_generation"] or 1) + 1
        db.execute(
            """
            UPDATE minigame_ranked_runs
            SET lease_token_hash = ?, lease_expires_at = ?, lease_last_seen_at = ?,
                lease_generation = ?
            WHERE run_id = ? AND status IN ('active', 'qualified')
            """,
            (lease_hash, lease_expires_at, _iso(current), next_generation, str(run_id)),
        )
        saved = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
    payload = _public_run(saved, include_run_token=True)
    payload["lease_token"] = lease_token
    return payload


def abandon_ranked_run(
    *,
    run_id: str,
    user_id: int,
    lease_token: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or _utc_now()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if row is None:
            raise LookupError("run_not_found")
        if int(row["user_id"]) != int(user_id):
            raise PermissionError("run_owner_mismatch")
        if str(row["status"]) not in {"active", "qualified"}:
            return _public_run(row)
        if not _lease_matches(row, lease_token):
            raise PermissionError("lease_mismatch")
        db.execute(
            """
            UPDATE minigame_ranked_runs
            SET status = 'expired', completed_at = ?, error_code = 'abandoned',
                lease_token_hash = NULL, lease_expires_at = NULL
            WHERE run_id = ? AND status IN ('active', 'qualified')
            """,
            (_iso(current), str(run_id)),
        )
        saved = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
    return _public_run(saved)


def _assert_token_claims(
    payload: dict[str, Any],
    row: sqlite3.Row | dict[str, Any],
) -> None:
    expected = {
        "rid": str(row["run_id"]),
        "uid": int(row["user_id"]),
        "game": str(row["game_id"]),
        "difficulty": int(row["difficulty"]),
        "rules": int(row["rules_version"]),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RunTokenError("token_claim_mismatch")


def _submission_token_for(
    row: sqlite3.Row | dict[str, Any],
    summary: dict[str, Any],
    expires_at: datetime,
) -> str:
    summary_digest = _summary_hash(summary)
    token_id = hmac.new(
        _token_secret(),
        f"minigame-submission-v1|{row['run_id']}|{summary_digest}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]
    return _sign_token(
        "submission",
        {
            "rid": str(row["run_id"]),
            "uid": int(row["user_id"]),
            "game": str(row["game_id"]),
            "difficulty": int(row["difficulty"]),
            "rules": int(row["rules_version"]),
            "jti": token_id,
            "summary": summary_digest,
        },
        expires_at,
    )


def _top_100_cutoff(
    db: sqlite3.Connection,
    *,
    game_id: str,
    difficulty: int,
) -> int | None:
    row = db.execute(
        """
        SELECT scores.best_score
        FROM minigame_high_scores AS scores
        JOIN users ON users.id = scores.user_id
        WHERE scores.game_id = ?
          AND scores.difficulty = ?
          AND scores.verification_level = 'verified'
          AND users.status = 'active'
          AND TRIM(COALESCE(users.display_name, '')) <> ''
        ORDER BY scores.best_score DESC, scores.score_achieved_at ASC, scores.user_id ASC
        LIMIT 1 OFFSET 99
        """,
        (game_id, difficulty),
    ).fetchone()
    return None if row is None else int(row["best_score"])


def submit_ranked_checkpoint(
    *,
    run_id: str,
    user_id: int,
    run_token: str,
    lease_token: str,
    revision: int,
    score: int,
    trophy_tier: int,
    highest_tile_exp: int,
    final_board: list[int],
    board_rows: int,
    board_cols: int,
    action_count: int,
    elapsed_ms: int,
    record_encoding: str,
    ip_address: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    normalized_revision = int(revision)
    if normalized_revision < 1 or normalized_revision > MAX_CHECKPOINTS_PER_RUN:
        raise ValueError("invalid_checkpoint_revision")
    summary = _normalize_summary(
        score=score,
        trophy_tier=trophy_tier,
        highest_tile_exp=highest_tile_exp,
        final_board=final_board,
        board_rows=board_rows,
        board_cols=board_cols,
        action_count=action_count,
        elapsed_ms=elapsed_ms,
    )
    summary_json = json.dumps(summary, separators=(",", ":"), sort_keys=True)
    envelope = parse_mgo1_envelope(record_encoding)
    normalized_record = str(record_encoding).strip()
    record_hash = hashlib.sha256(normalized_record.encode("utf-8")).hexdigest()
    current = now or _utc_now()

    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        run = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if run is None:
            raise LookupError("run_not_found")
        if int(run["user_id"]) != int(user_id):
            raise PermissionError("run_owner_mismatch")
        if str(run["status"]) != "active":
            raise ValueError("run_not_active")
        token_payload = _verify_token(run_token, kind="run", now=current)
        _assert_token_claims(token_payload, run)
        if datetime.fromisoformat(str(run["expires_at"])) <= current:
            raise RunTokenExpired("run_expired")
        _assert_live_lease(run, lease_token, current)

        if (
            envelope.run_id != str(run["run_id"])
            or envelope.game_id != str(run["game_id"])
            or envelope.difficulty != int(run["difficulty"])
            or envelope.rules_version != int(run["rules_version"])
            or not hmac.compare_digest(envelope.seed_hex, str(run["seed_hex"]))
            or envelope.action_count != summary["action_count"]
            or envelope.elapsed_ms != summary["elapsed_ms"]
        ):
            raise ValueError("record_claim_mismatch")

        duplicate = db.execute(
            "SELECT * FROM minigame_ranked_checkpoints WHERE run_id = ? AND record_hash = ?",
            (str(run_id), record_hash),
        ).fetchone()
        if duplicate is not None:
            return _public_checkpoint(duplicate)
        same_revision = db.execute(
            "SELECT * FROM minigame_ranked_checkpoints WHERE run_id = ? AND revision = ?",
            (str(run_id), normalized_revision),
        ).fetchone()
        if same_revision is not None:
            raise ValueError("checkpoint_revision_conflict")
        latest_revision = int(
            db.execute(
                "SELECT COALESCE(MAX(revision), 0) AS revision FROM minigame_ranked_checkpoints WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()["revision"]
        )
        if normalized_revision != latest_revision + 1:
            raise ValueError("checkpoint_revision_gap")

        user_pending = int(
            db.execute(
                """
                SELECT COUNT(*) AS count
                FROM minigame_ranked_checkpoints AS checkpoint
                JOIN minigame_ranked_runs AS parent ON parent.run_id = checkpoint.run_id
                WHERE parent.user_id = ? AND checkpoint.status IN ('pending', 'validating')
                """,
                (int(user_id),),
            ).fetchone()["count"]
        )
        global_pending = int(
            db.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM minigame_ranked_checkpoints WHERE status IN ('pending', 'validating'))
                  + (SELECT COUNT(*) FROM minigame_ranked_runs WHERE status IN ('pending', 'validating'))
                  AS count
                """
            ).fetchone()["count"]
        )
        if user_pending >= MAX_PENDING_PER_USER:
            raise RuntimeError("user_pending_limit")
        if global_pending >= MAX_PENDING_GLOBAL:
            raise RuntimeError("queue_full")

        # A newer cumulative checkpoint makes older queued copies redundant.
        db.execute(
            """
            UPDATE minigame_ranked_checkpoints
            SET status = 'superseded', pending_record = NULL, completed_at = ?
            WHERE run_id = ? AND status = 'pending'
            """,
            (_iso(current), str(run_id)),
        )
        cursor = db.execute(
            """
            INSERT INTO minigame_ranked_checkpoints
            (run_id, revision, status, claimed_summary_json, pending_record,
             record_hash, action_count, submitted_at, submit_ip)
            VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                normalized_revision,
                summary_json,
                normalized_record,
                record_hash,
                summary["action_count"],
                _iso(current),
                str(ip_address or "")[:128],
            ),
        )
        saved = db.execute(
            "SELECT * FROM minigame_ranked_checkpoints WHERE id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
    return _public_checkpoint(saved)


def get_ranked_checkpoint(
    *, run_id: str, revision: int, user_id: int
) -> dict[str, Any]:
    with auth_db() as db:
        row = db.execute(
            """
            SELECT checkpoint.*
            FROM minigame_ranked_checkpoints AS checkpoint
            JOIN minigame_ranked_runs AS parent ON parent.run_id = checkpoint.run_id
            WHERE checkpoint.run_id = ? AND checkpoint.revision = ? AND parent.user_id = ?
            """,
            (str(run_id), int(revision), int(user_id)),
        ).fetchone()
    if row is None:
        raise LookupError("checkpoint_not_found")
    return _public_checkpoint(row)


def qualify_ranked_run(
    *,
    run_id: str,
    user_id: int,
    run_token: str,
    lease_token: str,
    score: int,
    trophy_tier: int,
    highest_tile_exp: int,
    final_board: list[int],
    board_rows: int,
    board_cols: int,
    action_count: int,
    elapsed_ms: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    summary = _normalize_summary(
        score=score,
        trophy_tier=trophy_tier,
        highest_tile_exp=highest_tile_exp,
        final_board=final_board,
        board_rows=board_rows,
        board_cols=board_cols,
        action_count=action_count,
        elapsed_ms=elapsed_ms,
    )
    summary_json = json.dumps(summary, separators=(",", ":"), sort_keys=True)
    current = now or _utc_now()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if row is None:
            raise LookupError("run_not_found")
        if int(row["user_id"]) != int(user_id):
            raise PermissionError("run_owner_mismatch")
        token_payload = _verify_token(run_token, kind="run", now=current)
        _assert_token_claims(token_payload, row)
        if datetime.fromisoformat(str(row["expires_at"])) <= current:
            db.execute(
                """
                UPDATE minigame_ranked_runs
                SET status = 'expired', completed_at = ?
                WHERE run_id = ? AND status IN ('active', 'qualified')
                """,
                (_iso(current), str(run_id)),
            )
            raise RunTokenExpired("run_expired")
        if str(row["status"]) == "not_candidate":
            if str(row["claimed_summary_json"] or "") != summary_json:
                raise ValueError("qualification_conflict")
            return {"candidate": False, **_public_run(row)}
        if str(row["status"]) not in {"active", "qualified"}:
            return {"candidate": True, **_public_run(row)}
        _assert_live_lease(row, lease_token, current)
        if str(row["status"]) == "qualified":
            if str(row["claimed_summary_json"] or "") != summary_json:
                raise ValueError("qualification_conflict")
            token_expires = datetime.fromisoformat(str(row["qualification_expires_at"]))
            if token_expires <= current:
                raise RunTokenExpired("submission_token_expired")
            submission_token = _submission_token_for(row, summary, token_expires)
            stored_hash = str(row["submission_token_hash"] or "")
            supplied_hash = hashlib.sha256(submission_token.encode("utf-8")).hexdigest()
            if not stored_hash or not hmac.compare_digest(stored_hash, supplied_hash):
                raise RunTokenError("submission_token_replaced")
            return {
                "candidate": True,
                "reasons": [],
                "submission_token": submission_token,
                "submission_expires_at": _iso(token_expires),
                **_public_run(row),
            }

        personal_best = db.execute(
            """
            SELECT best_score, trophy_tier, verification_level
            FROM minigame_high_scores
            WHERE user_id = ? AND game_id = ? AND difficulty = ?
            """,
            (int(user_id), str(row["game_id"]), int(row["difficulty"])),
        ).fetchone()
        if personal_best is not None and str(personal_best["verification_level"]) != "verified":
            personal_best = None
        reasons: list[str] = []
        improves_personal_best = (
            personal_best is None
            or summary["score"] > int(personal_best["best_score"])
        )
        if improves_personal_best:
            reasons.append("personal_best")
        if personal_best is None:
            if summary["trophy_tier"] > 0:
                reasons.append("trophy_improvement")
        elif summary["trophy_tier"] > int(personal_best["trophy_tier"]):
            reasons.append("trophy_improvement")
        cutoff = _top_100_cutoff(
            db,
            game_id=str(row["game_id"]),
            difficulty=int(row["difficulty"]),
        )
        # One leaderboard row is kept per user. A score below that user's PB
        # cannot change the top 100 even when it exceeds the global cutoff.
        if improves_personal_best and (cutoff is None or summary["score"] > cutoff):
            reasons.append("top_100")

        if not reasons:
            db.execute(
                """
                UPDATE minigame_ranked_runs
                SET status = 'not_candidate', claimed_summary_json = ?,
                    qualified_at = ?, completed_at = ?, error_code = NULL,
                    lease_token_hash = NULL, lease_expires_at = NULL
                WHERE run_id = ?
                """,
                (summary_json, _iso(current), _iso(current), str(run_id)),
            )
            saved = db.execute(
                "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
            return {"candidate": False, "reasons": [], **_public_run(saved)}

        token_expires = min(current + SUBMISSION_TOKEN_LIFETIME, datetime.fromisoformat(str(row["expires_at"])))
        submission_token = _submission_token_for(row, summary, token_expires)
        db.execute(
            """
            UPDATE minigame_ranked_runs
            SET status = 'qualified', qualified_at = ?, qualification_expires_at = ?,
                submission_token_hash = ?, submission_token_consumed_at = NULL,
                claimed_summary_json = ?, completed_at = NULL, error_code = NULL
            WHERE run_id = ?
            """,
            (
                _iso(current),
                _iso(token_expires),
                hashlib.sha256(submission_token.encode("utf-8")).hexdigest(),
                summary_json,
                str(run_id),
            ),
        )
        saved = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
    return {
        "candidate": True,
        "reasons": reasons,
        "submission_token": submission_token,
        "submission_expires_at": _iso(token_expires),
        **_public_run(saved),
    }


def submit_ranked_run(
    *,
    run_id: str,
    user_id: int,
    submission_token: str,
    lease_token: str,
    record_encoding: str,
    ip_address: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    envelope = parse_mgo1_envelope(record_encoding)
    normalized_record = str(record_encoding).strip()
    record_hash = hashlib.sha256(normalized_record.encode("utf-8")).hexdigest()
    current = now or _utc_now()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if row is None:
            raise LookupError("run_not_found")
        if int(row["user_id"]) != int(user_id):
            raise PermissionError("run_owner_mismatch")
        if str(row["status"]) in {"pending", "validating", "verified", "rejected"}:
            if row["record_hash"] and not hmac.compare_digest(str(row["record_hash"]), record_hash):
                raise ValueError("submission_already_used")
            return _public_run(row)
        if str(row["status"]) != "qualified":
            raise ValueError("run_not_qualified")

        summary = json.loads(str(row["claimed_summary_json"] or "{}"))
        if (
            envelope.run_id != str(row["run_id"])
            or envelope.game_id != str(row["game_id"])
            or envelope.difficulty != int(row["difficulty"])
            or envelope.rules_version != int(row["rules_version"])
            or not hmac.compare_digest(envelope.seed_hex, str(row["seed_hex"]))
            or envelope.action_count != int(summary.get("action_count", -1))
            or envelope.elapsed_ms != int(summary.get("elapsed_ms", -1))
        ):
            raise ValueError("record_claim_mismatch")

        token_payload = _verify_token(submission_token, kind="submission", now=current)
        _assert_token_claims(token_payload, row)
        if token_payload.get("summary") != _summary_hash(summary):
            raise RunTokenError("summary_claim_mismatch")
        _assert_live_lease(row, lease_token, current)
        stored_token_hash = str(row["submission_token_hash"] or "")
        supplied_token_hash = hashlib.sha256(str(submission_token).encode("utf-8")).hexdigest()
        if not stored_token_hash or not hmac.compare_digest(stored_token_hash, supplied_token_hash):
            raise RunTokenError("submission_token_replaced")
        if row["submission_token_consumed_at"] is not None:
            raise ValueError("submission_already_used")

        user_pending = int(
            db.execute(
                """
                SELECT COUNT(*) AS count FROM minigame_ranked_runs
                WHERE user_id = ? AND status IN ('pending', 'validating')
                """,
                (int(user_id),),
            ).fetchone()["count"]
        )
        global_pending = int(
            db.execute(
                """
                SELECT COUNT(*) AS count FROM minigame_ranked_runs
                WHERE status IN ('pending', 'validating')
                """
            ).fetchone()["count"]
        )
        if user_pending >= 1:
            raise RuntimeError("user_pending_limit")
        if global_pending >= MAX_PENDING_GLOBAL:
            raise RuntimeError("queue_full")

        changed = db.execute(
            """
            UPDATE minigame_ranked_runs
            SET status = 'pending', pending_record = ?, record_hash = ?,
                action_count = ?, submitted_at = ?, submit_ip = ?,
                submission_token_consumed_at = ?, error_code = NULL,
                lease_token_hash = NULL, lease_expires_at = NULL
            WHERE run_id = ? AND status = 'qualified'
              AND submission_token_consumed_at IS NULL
            """,
            (
                normalized_record,
                record_hash,
                int(summary["action_count"]),
                _iso(current),
                str(ip_address or "")[:128],
                _iso(current),
                str(run_id),
            ),
        ).rowcount
        if changed != 1:
            raise ValueError("submission_already_used")
        saved = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
    return _public_run(saved)


def get_ranked_run(*, run_id: str, user_id: int, now: datetime | None = None) -> dict[str, Any]:
    current = now or _utc_now()
    with auth_db() as db:
        row = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if row is None:
            raise LookupError("run_not_found")
        if int(row["user_id"]) != int(user_id):
            raise PermissionError("run_owner_mismatch")
        if str(row["status"]) in {"active", "qualified"} and datetime.fromisoformat(str(row["expires_at"])) <= current:
            db.execute(
                """
                UPDATE minigame_ranked_runs
                SET status = 'expired', completed_at = ?
                WHERE run_id = ?
                """,
                (_iso(current), str(run_id)),
            )
            row = db.execute(
                "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
    return _public_run(row)


def claim_pending_checkpoint() -> dict[str, Any] | None:
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        checkpoint = db.execute(
            """
            SELECT * FROM minigame_ranked_checkpoints
            WHERE status = 'pending'
            ORDER BY submitted_at ASC, id ASC
            LIMIT 1
            """
        ).fetchone()
        if checkpoint is None:
            return None
        claimed_at = _iso(_utc_now())
        changed = db.execute(
            """
            UPDATE minigame_ranked_checkpoints
            SET status = 'validating', validation_started_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (claimed_at, int(checkpoint["id"])),
        ).rowcount
        if changed != 1:
            return None
        claimed = db.execute(
            "SELECT * FROM minigame_ranked_checkpoints WHERE id = ?",
            (int(checkpoint["id"]),),
        ).fetchone()
        run = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(claimed["run_id"]),),
        ).fetchone()
        if run is None:
            db.execute(
                "UPDATE minigame_ranked_checkpoints SET status = 'rejected', error_code = 'run_not_found' WHERE id = ?",
                (int(claimed["id"]),),
            )
            return None
    payload = dict(run)
    payload.update({
        "checkpoint_id": int(claimed["id"]),
        "revision": int(claimed["revision"]),
        "pending_record": str(claimed["pending_record"] or ""),
        "record_hash": str(claimed["record_hash"] or ""),
        "claimed_summary": json.loads(str(claimed["claimed_summary_json"] or "{}")),
    })
    return payload


def finish_checkpoint_rejected(checkpoint_id: int, error_code: str) -> dict[str, Any]:
    now = _iso(_utc_now())
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM minigame_ranked_checkpoints WHERE id = ?",
            (int(checkpoint_id),),
        ).fetchone()
        if row is None:
            raise LookupError("checkpoint_not_found")
        if str(row["status"]) == "rejected":
            return _public_checkpoint(row)
        if str(row["status"]) != "validating":
            raise ValueError("checkpoint_not_validating")
        db.execute(
            """
            UPDATE minigame_ranked_checkpoints
            SET status = 'rejected', error_code = ?, completed_at = ?, pending_record = NULL
            WHERE id = ? AND status = 'validating'
            """,
            (str(error_code or "validation_failed")[:64], now, int(checkpoint_id)),
        )
        saved = db.execute(
            "SELECT * FROM minigame_ranked_checkpoints WHERE id = ?",
            (int(checkpoint_id),),
        ).fetchone()
    return _public_checkpoint(saved)


def finish_checkpoint_verified(
    checkpoint_id: int,
    *,
    score: int,
    trophy_tier: int,
    highest_tile_exp: int,
    final_board: list[int],
    board_rows: int,
    board_cols: int,
    action_count: int,
    elapsed_ms: int,
) -> dict[str, Any]:
    verified = _normalize_summary(
        score=score,
        trophy_tier=trophy_tier,
        highest_tile_exp=highest_tile_exp,
        final_board=final_board,
        board_rows=board_rows,
        board_cols=board_cols,
        action_count=action_count,
        elapsed_ms=elapsed_ms,
    )
    now = _iso(_utc_now())
    verified_json = json.dumps(verified, sort_keys=True, separators=(",", ":"))
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        checkpoint = db.execute(
            "SELECT * FROM minigame_ranked_checkpoints WHERE id = ?",
            (int(checkpoint_id),),
        ).fetchone()
        if checkpoint is None:
            raise LookupError("checkpoint_not_found")
        if str(checkpoint["status"]) == "verified":
            return _public_checkpoint(checkpoint)
        if str(checkpoint["status"]) != "validating":
            raise ValueError("checkpoint_not_validating")
        run = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(checkpoint["run_id"]),),
        ).fetchone()
        if run is None:
            raise LookupError("run_not_found")
        score_updated, trophy_updated = _apply_verified_result(
            db,
            run=run,
            verified=verified,
            record_hash=str(checkpoint["record_hash"] or ""),
            record_blob=str(checkpoint["pending_record"] or ""),
            verified_at=now,
        )
        db.execute(
            """
            UPDATE minigame_ranked_checkpoints
            SET status = 'verified', completed_at = ?, verified_summary_json = ?,
                pending_record = NULL, error_code = NULL
            WHERE id = ? AND status = 'validating'
            """,
            (now, verified_json, int(checkpoint_id)),
        )
        saved = db.execute(
            "SELECT * FROM minigame_ranked_checkpoints WHERE id = ?",
            (int(checkpoint_id),),
        ).fetchone()
    return {
        **_public_checkpoint(saved),
        "score_updated": score_updated,
        "trophy_updated": trophy_updated,
    }


def claim_pending() -> dict[str, Any] | None:
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            """
            SELECT * FROM minigame_ranked_runs
            WHERE status = 'pending'
            ORDER BY submitted_at ASC, run_id ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        claimed_at = _iso(_utc_now())
        changed = db.execute(
            """
            UPDATE minigame_ranked_runs
            SET status = 'validating', validation_started_at = ?
            WHERE run_id = ? AND status = 'pending'
            """,
            (claimed_at, str(row["run_id"])),
        ).rowcount
        if changed != 1:
            return None
        claimed = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(row["run_id"]),),
        ).fetchone()
    payload = dict(claimed)
    payload["claimed_summary"] = json.loads(str(claimed["claimed_summary_json"] or "{}"))
    return payload


def finish_rejected(run_id: str, error_code: str) -> dict[str, Any]:
    now = _iso(_utc_now())
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if row is None:
            raise LookupError("run_not_found")
        if str(row["status"]) == "rejected":
            return _public_run(row)
        if str(row["status"]) != "validating":
            raise ValueError("run_not_validating")
        db.execute(
            """
            UPDATE minigame_ranked_runs
            SET status = 'rejected', error_code = ?, completed_at = ?,
                pending_record = NULL
            WHERE run_id = ? AND status = 'validating'
            """,
            (str(error_code or "validation_failed")[:64], now, str(run_id)),
        )
        saved = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
    return _public_run(saved)


def _apply_verified_result(
    db: sqlite3.Connection,
    *,
    run: sqlite3.Row | dict[str, Any],
    verified: dict[str, Any],
    record_hash: str,
    record_blob: str,
    verified_at: str,
) -> tuple[bool, bool]:
    board_json = json.dumps(verified["final_board"], separators=(",", ":"))
    existing = db.execute(
        """
        SELECT best_score, trophy_tier, verification_level
        FROM minigame_high_scores
        WHERE user_id = ? AND game_id = ? AND difficulty = ?
        """,
        (int(run["user_id"]), str(run["game_id"]), int(run["difficulty"])),
    ).fetchone()
    existing_is_verified = (
        existing is not None and str(existing["verification_level"]) == "verified"
    )
    score_updated = (
        not existing_is_verified or verified["score"] > int(existing["best_score"])
    )
    trophy_updated = (
        verified["trophy_tier"] > 0
        and (
            not existing_is_verified
            or verified["trophy_tier"] > int(existing["trophy_tier"])
        )
    )

    if existing is None:
        db.execute(
            """
            INSERT INTO minigame_high_scores
            (user_id, game_id, difficulty, best_score, trophy_tier,
             highest_tile_exp, final_board_json, board_rows, board_cols,
             score_achieved_at, trophy_achieved_at, score_run_id, trophy_run_id,
             verification_level, score_verified_at, trophy_verified_at,
             record_hash, record_blob, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'verified', ?, ?, ?, ?, ?)
            """,
            (
                int(run["user_id"]), str(run["game_id"]), int(run["difficulty"]),
                verified["score"], verified["trophy_tier"], verified["highest_tile_exp"],
                board_json, verified["board_rows"], verified["board_cols"], verified_at,
                verified_at if verified["trophy_tier"] > 0 else None,
                str(run["run_id"]), str(run["run_id"]) if verified["trophy_tier"] > 0 else None,
                verified_at, verified_at if verified["trophy_tier"] > 0 else None,
                str(record_hash or ""), str(record_blob or ""), verified_at,
            ),
        )
    elif score_updated or trophy_updated:
        db.execute(
            """
            UPDATE minigame_high_scores
            SET best_score = CASE WHEN ? THEN ? ELSE best_score END,
                highest_tile_exp = CASE WHEN ? THEN ? ELSE highest_tile_exp END,
                final_board_json = CASE WHEN ? THEN ? ELSE final_board_json END,
                board_rows = CASE WHEN ? THEN ? ELSE board_rows END,
                board_cols = CASE WHEN ? THEN ? ELSE board_cols END,
                score_achieved_at = CASE WHEN ? THEN ? ELSE score_achieved_at END,
                score_run_id = CASE WHEN ? THEN ? ELSE score_run_id END,
                verification_level = CASE WHEN ? THEN 'verified' ELSE verification_level END,
                score_verified_at = CASE WHEN ? THEN ? ELSE score_verified_at END,
                record_hash = CASE WHEN ? THEN ? ELSE record_hash END,
                record_blob = CASE WHEN ? THEN ? ELSE record_blob END,
                trophy_tier = CASE WHEN ? THEN ? ELSE trophy_tier END,
                trophy_achieved_at = CASE WHEN ? THEN ? ELSE trophy_achieved_at END,
                trophy_run_id = CASE WHEN ? THEN ? ELSE trophy_run_id END,
                trophy_verified_at = CASE WHEN ? THEN ? ELSE trophy_verified_at END,
                updated_at = ?
            WHERE user_id = ? AND game_id = ? AND difficulty = ?
            """,
            (
                score_updated, verified["score"],
                score_updated, verified["highest_tile_exp"],
                score_updated, board_json,
                score_updated, verified["board_rows"],
                score_updated, verified["board_cols"],
                score_updated, verified_at,
                score_updated, str(run["run_id"]),
                score_updated,
                score_updated, verified_at,
                score_updated, str(record_hash or ""),
                score_updated, str(record_blob or ""),
                trophy_updated, verified["trophy_tier"],
                trophy_updated, verified_at,
                trophy_updated, str(run["run_id"]),
                trophy_updated, verified_at,
                verified_at,
                int(run["user_id"]), str(run["game_id"]), int(run["difficulty"]),
            ),
        )
    return bool(score_updated), bool(trophy_updated)


def finish_verified(
    run_id: str,
    *,
    score: int,
    trophy_tier: int,
    highest_tile_exp: int,
    final_board: list[int],
    board_rows: int,
    board_cols: int,
    action_count: int,
    elapsed_ms: int,
) -> dict[str, Any]:
    verified = _normalize_summary(
        score=score,
        trophy_tier=trophy_tier,
        highest_tile_exp=highest_tile_exp,
        final_board=final_board,
        board_rows=board_rows,
        board_cols=board_cols,
        action_count=action_count,
        elapsed_ms=elapsed_ms,
    )
    now = _iso(_utc_now())
    verified_json = json.dumps(verified, sort_keys=True, separators=(",", ":"))
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        run = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if run is None:
            raise LookupError("run_not_found")
        if str(run["status"]) == "verified":
            return _public_run(run)
        if str(run["status"]) != "validating":
            raise ValueError("run_not_validating")

        score_updated, trophy_updated = _apply_verified_result(
            db,
            run=run,
            verified=verified,
            record_hash=str(run["record_hash"] or ""),
            record_blob=str(run["pending_record"] or ""),
            verified_at=now,
        )

        db.execute(
            """
            UPDATE minigame_ranked_runs
            SET status = 'verified', completed_at = ?, verified_summary_json = ?,
                pending_record = NULL, error_code = NULL
            WHERE run_id = ? AND status = 'validating'
            """,
            (now, verified_json, str(run_id)),
        )
        saved = db.execute(
            "SELECT * FROM minigame_ranked_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
    return {
        **_public_run(saved),
        "score_updated": bool(score_updated),
        "trophy_updated": bool(trophy_updated),
    }


def submit_score(
    *,
    user_id: int,
    game_id: str,
    difficulty: int,
    score: int,
    trophy_tier: int,
    highest_tile_exp: int,
    final_board: list[int],
    board_rows: int,
    board_cols: int,
) -> dict[str, Any]:
    normalized_game_id = str(game_id or "").strip()
    if normalized_game_id not in MINIGAME_BY_ID:
        raise ValueError("Unknown minigame.")
    normalized_difficulty = 1 if int(difficulty) else 0
    normalized_score = int(score)
    normalized_trophy = int(trophy_tier)
    normalized_highest = int(highest_tile_exp)
    rows = int(board_rows)
    cols = int(board_cols)
    board = [int(value) for value in final_board]
    if normalized_score < 0 or normalized_score > 2_147_483_647:
        raise ValueError("Invalid score.")
    if normalized_trophy < 0 or normalized_trophy > 4:
        raise ValueError("Invalid trophy tier.")
    if normalized_highest < 0 or normalized_highest > 63:
        raise ValueError("Invalid highest tile.")
    if rows < 1 or rows > 8 or cols < 1 or cols > 8 or rows * cols != len(board):
        raise ValueError("Invalid board shape.")
    if len(board) > 64 or any(value < -1 or value > 63 for value in board):
        raise ValueError("Invalid board data.")

    now = _now_iso()
    board_json = json.dumps(board, separators=(",", ":"))
    score_updated = False
    trophy_updated = False
    with auth_db() as db:
        existing = db.execute(
            """
            SELECT best_score, trophy_tier
            FROM minigame_high_scores
            WHERE user_id = ? AND game_id = ? AND difficulty = ?
            """,
            (int(user_id), normalized_game_id, normalized_difficulty),
        ).fetchone()
        if existing is None:
            db.execute(
                """
                INSERT INTO minigame_high_scores
                (user_id, game_id, difficulty, best_score, trophy_tier,
                 highest_tile_exp, final_board_json, board_rows, board_cols,
                 score_achieved_at, trophy_achieved_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(user_id),
                    normalized_game_id,
                    normalized_difficulty,
                    normalized_score,
                    normalized_trophy,
                    normalized_highest,
                    board_json,
                    rows,
                    cols,
                    now,
                    now if normalized_trophy > 0 else None,
                    now,
                ),
            )
            score_updated = True
            trophy_updated = normalized_trophy > 0
        else:
            score_updated = normalized_score > int(existing["best_score"])
            trophy_updated = normalized_trophy > int(existing["trophy_tier"])
            if score_updated or trophy_updated:
                db.execute(
                    """
                    UPDATE minigame_high_scores
                    SET best_score = CASE WHEN ? THEN ? ELSE best_score END,
                        highest_tile_exp = CASE WHEN ? THEN ? ELSE highest_tile_exp END,
                        final_board_json = CASE WHEN ? THEN ? ELSE final_board_json END,
                        board_rows = CASE WHEN ? THEN ? ELSE board_rows END,
                        board_cols = CASE WHEN ? THEN ? ELSE board_cols END,
                        score_achieved_at = CASE WHEN ? THEN ? ELSE score_achieved_at END,
                        trophy_tier = CASE WHEN ? THEN ? ELSE trophy_tier END,
                        trophy_achieved_at = CASE WHEN ? THEN ? ELSE trophy_achieved_at END,
                        updated_at = ?
                    WHERE user_id = ? AND game_id = ? AND difficulty = ?
                    """,
                    (
                        score_updated, normalized_score,
                        score_updated, normalized_highest,
                        score_updated, board_json,
                        score_updated, rows,
                        score_updated, cols,
                        score_updated, now,
                        trophy_updated, normalized_trophy,
                        trophy_updated, now,
                        now,
                        int(user_id), normalized_game_id, normalized_difficulty,
                    ),
                )
        saved = db.execute(
            """
            SELECT best_score, trophy_tier, highest_tile_exp, score_achieved_at
            FROM minigame_high_scores
            WHERE user_id = ? AND game_id = ? AND difficulty = ?
            """,
            (int(user_id), normalized_game_id, normalized_difficulty),
        ).fetchone()
    return {
        "accepted": True,
        "score_updated": score_updated,
        "trophy_updated": trophy_updated,
        "personal_best": int(saved["best_score"]),
        "trophy_tier": int(saved["trophy_tier"]),
        "highest_tile_exp": int(saved["highest_tile_exp"]),
        "achieved_at": str(saved["score_achieved_at"]),
    }


def game_leaderboard(game_id: str, *, difficulty: int, limit: int) -> dict[str, Any]:
    normalized_game_id = str(game_id or "").strip()
    if normalized_game_id not in MINIGAME_BY_ID:
        raise KeyError(normalized_game_id)
    normalized_difficulty = 1 if int(difficulty) else 0
    row_limit = max(1, min(int(limit), LEADERBOARD_LIMIT))
    with auth_db() as db:
        rows = db.execute(
            """
            SELECT
              scores.user_id,
              scores.best_score,
              scores.trophy_tier,
              scores.highest_tile_exp,
              scores.score_achieved_at,
              users.display_name,
              profiles.avatar_key,
              CASE WHEN entitlements.tier = 'supporter' THEN 1 ELSE 0 END AS is_supporter
            FROM minigame_high_scores AS scores
            JOIN users ON users.id = scores.user_id
            LEFT JOIN user_profiles AS profiles ON profiles.user_id = scores.user_id
            LEFT JOIN user_entitlements AS entitlements ON entitlements.user_id = scores.user_id
            WHERE scores.game_id = ?
              AND scores.difficulty = ?
              AND scores.best_score > 0
              AND users.status = 'active'
              AND TRIM(COALESCE(users.display_name, '')) <> ''
            ORDER BY scores.best_score DESC, scores.score_achieved_at ASC, scores.user_id ASC
            LIMIT ?
            """,
            (normalized_game_id, normalized_difficulty, row_limit),
        ).fetchall()
    entries = [
        {
            "entry_key": _entry_key("score", int(row["user_id"]), normalized_game_id, normalized_difficulty),
            "rank": index,
            "display_name": str(row["display_name"]),
            "avatar_url": _avatar_url(row["avatar_key"]),
            "is_supporter": bool(row["is_supporter"]),
            "score": int(row["best_score"]),
            "trophy_tier": int(row["trophy_tier"]),
            "highest_tile": 2 ** int(row["highest_tile_exp"]) if int(row["highest_tile_exp"]) > 0 else 0,
            "achieved_at": str(row["score_achieved_at"]),
        }
        for index, row in enumerate(rows, start=1)
    ]
    return {
        "kind": "game",
        "game_id": normalized_game_id,
        "title": MINIGAME_BY_ID[normalized_game_id],
        "difficulty": normalized_difficulty,
        "generated_at": _now_iso(),
        "entry_count": len(entries),
        "entries": entries,
    }


def trophy_leaderboard(*, difficulty: int, limit: int) -> dict[str, Any]:
    normalized_difficulty = 1 if int(difficulty) else 0
    row_limit = max(1, min(int(limit), LEADERBOARD_LIMIT))
    with auth_db() as db:
        rows = db.execute(
            """
            SELECT
              scores.user_id,
              users.display_name,
              profiles.avatar_key,
              CASE WHEN entitlements.tier = 'supporter' THEN 1 ELSE 0 END AS is_supporter,
              SUM(CASE WHEN scores.trophy_tier >= 4 THEN 1 ELSE 0 END) AS grand_count,
              SUM(CASE WHEN scores.trophy_tier >= 3 THEN 1 ELSE 0 END) AS gold_count,
              SUM(CASE WHEN scores.trophy_tier >= 2 THEN 1 ELSE 0 END) AS silver_count,
              SUM(CASE WHEN scores.trophy_tier >= 1 THEN 1 ELSE 0 END) AS bronze_count,
              MAX(COALESCE(scores.trophy_achieved_at, scores.score_achieved_at)) AS achieved_at
            FROM minigame_high_scores AS scores
            JOIN users ON users.id = scores.user_id
            LEFT JOIN user_profiles AS profiles ON profiles.user_id = scores.user_id
            LEFT JOIN user_entitlements AS entitlements ON entitlements.user_id = scores.user_id
            WHERE scores.difficulty = ?
              AND scores.trophy_tier > 0
              AND users.status = 'active'
              AND TRIM(COALESCE(users.display_name, '')) <> ''
            GROUP BY scores.user_id, users.display_name, profiles.avatar_key, entitlements.tier
            ORDER BY grand_count DESC, gold_count DESC, silver_count DESC,
                     bronze_count DESC, achieved_at ASC, scores.user_id ASC
            LIMIT ?
            """,
            (normalized_difficulty, row_limit),
        ).fetchall()
    entries = [
        {
            "entry_key": _entry_key("trophy", int(row["user_id"]), "overall", normalized_difficulty),
            "rank": index,
            "display_name": str(row["display_name"]),
            "avatar_url": _avatar_url(row["avatar_key"]),
            "is_supporter": bool(row["is_supporter"]),
            "trophies": {
                "grand": int(row["grand_count"] or 0),
                "gold": int(row["gold_count"] or 0),
                "silver": int(row["silver_count"] or 0),
                "bronze": int(row["bronze_count"] or 0),
            },
            "achieved_at": str(row["achieved_at"]),
        }
        for index, row in enumerate(rows, start=1)
    ]
    return {
        "kind": "overall",
        "difficulty": normalized_difficulty,
        "generated_at": _now_iso(),
        "entry_count": len(entries),
        "game_count": len(MINIGAME_BY_ID),
        "entries": entries,
    }


__all__ = [
    "LEADERBOARD_LIMIT",
    "MAX_PENDING_GLOBAL",
    "MAX_PENDING_RECORD_BYTES",
    "MGO_RECORD_PREFIX",
    "RANKED_RULES_VERSION",
    "RunTokenError",
    "RunTokenExpired",
    "claim_pending",
    "claim_pending_checkpoint",
    "create_ranked_run",
    "finish_checkpoint_rejected",
    "finish_checkpoint_verified",
    "finish_rejected",
    "finish_verified",
    "game_leaderboard",
    "get_ranked_run",
    "get_ranked_checkpoint",
    "minigame_catalog",
    "qualify_ranked_run",
    "submit_score",
    "submit_ranked_run",
    "submit_ranked_checkpoint",
    "trophy_leaderboard",
]
