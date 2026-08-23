from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from backend.auth.db import auth_db

from .catalog import MINIGAME_BY_ID, minigame_catalog


LEADERBOARD_LIMIT = 100


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry_key(kind: str, user_id: int, game_id: str, difficulty: int) -> str:
    raw = f"minigame:{kind}:{user_id}:{game_id}:{difficulty}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _avatar_url(avatar_key: str | None) -> str | None:
    return f"/media/avatars/{avatar_key}" if avatar_key else None


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
    "game_leaderboard",
    "minigame_catalog",
    "submit_score",
    "trophy_leaderboard",
]
