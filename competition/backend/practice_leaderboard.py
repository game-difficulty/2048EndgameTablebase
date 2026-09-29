"""Small, client-reported practice leaderboard; never used for match adjudication."""

from __future__ import annotations

from datetime import datetime, timezone

from .db import CompetitionDatabase
from .domain import Principal
from .errors import CompetitionError


# Bump a project's version when its practice rules change. Old results remain stored,
# but are not mixed with the current leaderboard.
PROJECT_RULES = {
    "tournament-cargo-transport-4x4": ("deliveries", 1),
    "tournament-spawn4-50-3x3": ("score", 1),
    "tournament-evil-spawn-4x4": ("score", 1),
    "tournament-pure2-full-race-3x3": ("time", 1),
    "tournament-grand-full-undo-race-3x3": ("time", 1),
    "tournament-dice-wall-3x3": ("board_sum", 1),
    "tournament-mirror-64x10-race-4x4": ("time", 1),
    "tournament-256-brick-5x5": ("score", 1),
    "tournament-isolated-island-hard-4x4": ("score", 1),
    "tournament-shape-shifter-hard-12": ("score", 2),
    "practice-hundred-step-seal-4x4": ("score", 1),
    "practice-growing-tiles-4x4": ("score", 2),
    "practice-pair-bond-4x4": ("score", 2),
    "practice-chemical-reaction-4x4": ("score", 2),
    "practice-timed-bomb-4x4": ("score", 2),
}


class PracticeLeaderboard:
    def __init__(self, database: CompetitionDatabase):
        self.database = database

    @staticmethod
    def _rules(project_id: str) -> tuple[str, int]:
        rules = PROJECT_RULES.get(project_id)
        if rules is None:
            raise CompetitionError("UNKNOWN_PRACTICE_PROJECT", "Unknown practice project.", 404)
        return rules

    def list(self, project_id: str, principal: Principal | None = None) -> dict:
        metric, version = self._rules(project_id)
        order = "result_value ASC, elapsed_ms ASC" if metric == "time" else "result_value DESC, elapsed_ms ASC"
        with self.database.transaction() as db:
            rows = db.execute(
                f"""SELECT user_id, display_name, result_value, elapsed_ms, achieved_at
                    FROM practice_bests WHERE project_id = ? AND rules_version = ?
                    ORDER BY {order}, achieved_at ASC, user_id ASC LIMIT 10""",
                (project_id, version),
            ).fetchall()
            mine = db.execute(
                """SELECT user_id, display_name, result_value, elapsed_ms, achieved_at
                    FROM practice_bests WHERE project_id = ? AND rules_version = ? AND user_id = ?""",
                (project_id, version, principal.user_id),
            ).fetchone() if principal else None
        return {
            "project_id": project_id,
            "rules_version": version,
            "metric": metric,
            "signed_in": principal is not None,
            "top": [dict(row) for row in rows],
            "my_best": dict(mine) if mine else None,
        }

    def submit(
        self, project_id: str, principal: Principal, *, score: int,
        board_sum: int, elapsed_ms: int, outcome: str,
    ) -> dict:
        metric, version = self._rules(project_id)
        if any(type(value) is not int or value < 0 or value > 10**12 for value in (score, board_sum, elapsed_ms)):
            raise CompetitionError("INVALID_PRACTICE_RESULT", "Invalid practice result.")
        allowed = ("time_limit", "no_moves") if metric == "deliveries" else ("target_reached",) if metric == "time" else ("no_moves",)
        if outcome not in allowed or (metric == "time" and elapsed_ms == 0):
            raise CompetitionError("INVALID_PRACTICE_RESULT", "This run has not reached a recordable ending.")
        value = board_sum if metric == "board_sum" else elapsed_ms if metric == "time" else score
        improved = False
        with self.database.transaction(immediate=True) as db:
            previous = db.execute(
                """SELECT result_value, elapsed_ms FROM practice_bests
                    WHERE project_id = ? AND rules_version = ? AND user_id = ?""",
                (project_id, version, principal.user_id),
            ).fetchone()
            improved = previous is None or (
                (value < previous["result_value"] or (value == previous["result_value"] and elapsed_ms < previous["elapsed_ms"]))
                if metric == "time" else
                (value > previous["result_value"] or (value == previous["result_value"] and elapsed_ms < previous["elapsed_ms"]))
            )
            if improved:
                db.execute(
                    """INSERT INTO practice_bests
                        (project_id, rules_version, user_id, display_name, result_value, elapsed_ms, outcome, achieved_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(project_id, rules_version, user_id) DO UPDATE SET
                        display_name = excluded.display_name,
                        result_value = excluded.result_value,
                        elapsed_ms = excluded.elapsed_ms,
                        outcome = excluded.outcome,
                        achieved_at = excluded.achieved_at""",
                    (project_id, version, principal.user_id, principal.display_name, value,
                     elapsed_ms, outcome, datetime.now(timezone.utc).isoformat()),
                )
        return {**self.list(project_id, principal), "improved": improved}
