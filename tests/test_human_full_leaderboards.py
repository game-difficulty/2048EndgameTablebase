import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from backend.auth.db import auth_db, init_auth_db
from backend.auth.service import iso
from backend.human_play import engine, leaderboards
from backend.human_play.analysis_summary import build_summary, save_summary
from backend.human_play.store import database, init_db


SEED = "00000001000000020000000300000004"


def add_user(uid: int):
    with auth_db() as db:
        db.execute("""INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at)
            VALUES(?,?,?,?,?,?)""", (uid, f"{uid}@test.invalid", "!", f"Player {uid}", iso(), iso()))


def add_run(db, run_id: str, user_id: int, *, score: int, ended: float,
            board=None, visible: int = 1):
    state = engine.initial(run_id, "4x4", SEED)
    state.update({"score": score, "board": board or [32768, 16384] + [0] * 14})
    db.execute("""INSERT INTO human_runs
        (id,user_id,browser,variant,request_id,seed,threshold,status,created,ended,
         reason,writer,state,archive,visible,has_replay,source)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, user_id, f"browser-{user_id}", "4x4", f"request-{run_id}", SEED, 0,
         "sealed", ended - 1, ended, "game_over", "writer", json.dumps(state),
         b"archive", visible, 1, "native"))


def graded_summary(fit: float = .99, timing: int = 1000):
    segment = {"start_index": 0, "end_index": 20, "evaluated_moves": 20,
               "goodness_of_fit": fit, "max_combo": 20,
               "performance_counts": {"Perfect!": 20}}
    summary = build_summary([segment], [timing] * 20)
    summary["goal_tile"] = 32768
    return summary


def test_full_score_pagination_and_rate_minimum_are_indexed_facts():
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "CLOUD_AUTH_DB": str(Path(temp) / "auth.sqlite3"),
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
    }):
        init_auth_db(); init_db()
        for uid in range(1, 52):
            add_user(uid)
        with database() as db:
            for uid in range(1, 52):
                run_id = f"run-{uid}"
                add_run(db, run_id, uid, score=uid * 100, ended=float(uid))
                db.execute("""INSERT INTO human_player_ratings
                    (user_id,variant,pb_score,pb_run_id,pb_ended_at,rating,game_count,updated_at)
                    VALUES(?,?,?,?,?,?,?,?)""",
                    (uid, "4x4", uid * 100, run_id, float(uid), 1000 + uid, 10, time.time()))
            for uid, candidates in ((1, 9), (2, 10), (3, 12)):
                db.execute("""INSERT INTO human_player_statistics
                    (user_id,variant,game_count,pb_score,b10_score,b10_rating,
                     primary_achievement_count,rate_32k_value,features_json,
                     rate_32k_passed,rate_32k_total,rate_32k_covered_games,
                     rate_32k_candidate_games,stats_version,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (uid, "4x4", candidates, 1, 1, 1, candidates,
                     .9 if uid == 2 else .8, "{}", 9 if uid == 2 else 8, 10,
                     candidates, candidates, 1, time.time()))
            second = leaderboards.full_page(db, kind="score", variant="4x4", page=2)
            rate = leaderboards.full_page(db, kind="rate32k", variant="4x4")
            rating_board = leaderboards.full_page(db, kind="rating", variant="4x4")
        assert second["page_size"] == 50 and second["total"] == 51
        assert len(second["entries"]) == 1 and second["entries"][0]["rank"] == 51
        assert [entry["user_id"] for entry in rate["entries"]] == [2, 3]
        assert rate["minimum_32k_games"] == 10
        assert next(entry for entry in rating_board["entries"] if entry["user_id"] == 2)["b10_score"] == 1


def test_strength_board_keeps_one_best_result_per_player_and_hides_weighted_score():
    now = time.time()
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "CLOUD_AUTH_DB": str(Path(temp) / "auth.sqlite3"),
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
    }):
        init_auth_db(); init_db(); add_user(1); add_user(2)
        with database() as db:
            add_run(db, "p1-old", 1, score=800000, ended=now - 9 * 86400)
            add_run(db, "p1-new", 1, score=820000, ended=now - 3600)
            add_run(db, "p2-new", 2, score=810000, ended=now - 1800)
        save_summary(run_id="p1-old", user_id=1, pattern="free10", target="512",
                     job_id="job-1", summary=graded_summary(.99, 1000))
        save_summary(run_id="p1-new", user_id=1, pattern="free10", target="512",
                     job_id="job-2", summary=graded_summary(.95, 2000))
        save_summary(run_id="p2-new", user_id=2, pattern="free10", target="512",
                     job_id="job-3", summary=graded_summary(.98, 1500))
        with database() as db:
            all_time = leaderboards.full_page(db, kind="strength", variant="4x4",
                period="all", pattern="free10", target="512")
            weekly = leaderboards.full_page(db, kind="strength", variant="4x4",
                period="week", pattern="free10", target="512")
            db.execute("UPDATE human_analysis_week_state SET next_expiry_at=0")
            maintained = leaderboards.maintain_due_analysis_weeks(db, now=now)
            catalog = leaderboards.catalog(db)
        assert len(all_time["entries"]) == 2
        assert len({entry["user_id"] for entry in all_time["entries"]}) == 2
        assert all("weighted_score" not in entry for entry in all_time["entries"])
        assert all(entry["grade"] in {"SSS", "SS", "S", "A", "B", "C", "D", "E", "F"}
                   for entry in all_time["entries"])
        assert {entry["run_id"] for entry in weekly["entries"]} == {"p1-new", "p2-new"}
        assert maintained == 1
        assert catalog["strength"] == [{"variant": "4x4", "pattern": "free10",
                                        "target": "512", "results": 3}]


def test_strength_catalog_only_keeps_formations_with_results_and_removes_revoked_last_result():
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "CLOUD_AUTH_DB": str(Path(temp) / "auth.sqlite3"),
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
    }):
        init_auth_db(); init_db(); add_user(1)
        with database() as db:
            assert leaderboards.catalog(db)["strength"] == []
            add_run(db, "catalog-run", 1, score=800000, ended=time.time())
        save_summary(run_id="catalog-run", user_id=1, pattern="free10", target="512",
                     job_id="catalog-job", summary=graded_summary())
        with database() as db:
            assert leaderboards.catalog(db)["strength"] == [{
                "variant": "4x4", "pattern": "free10", "target": "512", "results": 1}]
            db.execute("UPDATE human_runs SET visible=0 WHERE id='catalog-run'")
            leaderboards.refresh_run(db, "catalog-run")
            assert leaderboards.catalog(db)["strength"] == []
