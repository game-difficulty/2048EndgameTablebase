import gzip
import base64
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from backend.auth.db import init_auth_db, auth_db
from backend.cloud_analysis_jobs import (
    AnalysisJob, AnalysisQueueFull, AnalysisWorkItem, _admit_job, _claim_job,
    analysis_download_filename,
)
from backend.cloud_files import build_file_download_response
from backend.quota.errors import InsufficientTokens
from backend.quota.service import reserve_operation_tokens_many
from backend.gamer_ranked.prng import Xoshiro128StarStar
from backend.human_play import engine, service
from backend.human_play.analysis_bridge import archived_to_analysis_text
from backend.human_play.analysis_summary import build_summary, get_summary, intervals_from_analysis_input, list_summaries, poster_goal_tile, save_summary
from backend.human_play.store import database, init_db
from backend.replay_2048next import MoveRecord, decode_2048next_replay


SEED = "00000001000000020000000300000004"


def test_archived_analysis_download_name_uses_player_game_time_and_score():
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
    }):
        init_db()
        ended = datetime(2026, 9, 12, 21, 1, 45,
                         tzinfo=timezone(timedelta(hours=8))).timestamp()
        state = engine.initial("named-run", "4x4", SEED)
        state["score"] = 830440
        with database() as db:
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,created,ended,reason,writer,state,archive,visible)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("named-run", 7, "browser", "4x4", "request", SEED, 0,
                 "sealed", ended - 3600, ended, "game_over", "writer",
                 json.dumps(state), b"archive", 1))
        archive = Path(temp) / "analysis_free10_01234567.zip"
        archive.write_bytes(b"PK")
        work_items = [
            AnalysisWorkItem(Path(temp) / "input.vrs", "named-run.vrs", pattern, target,
                             source_run_id="named-run")
            for pattern, target in (("free10", "256"), ("free9", "512"))
        ]
        job = AnalysisJob(
            job_id="0123456789abcdef0123456789abcdef", user_id=7, session_id=None,
            pattern="free10", target="256", target_value=256, full_pattern="free10_256",
            input_paths=[], input_names={}, output_dir=Path(temp), total=2,
            status="finished", zip_path=archive, work_items=work_items,
        )
        filename = analysis_download_filename(job, {"display_name": "游戏/难度"})
        assert filename == "游戏/难度-2026-09-12_21-01-45-4x4-830440分.zip"
        response = build_file_download_response(
            archive, filename=filename, root=Path(temp), preserve_unicode_filename=True)
        disposition = response.headers["content-disposition"]
        assert "filename*=utf-8''" in disposition
        assert "%E6%B8%B8%E6%88%8F_%E9%9A%BE%E5%BA%A6" in disposition
        assert "free10" not in disposition and "01234567" not in disposition


def test_stage_summary_averages_whole_endgames_and_filters_short_or_zero_segments():
    assert poster_goal_tile("free10", "512", "4x4") == 32768
    assert poster_goal_tile("free9", "512", "4x4") == 65536
    assert poster_goal_tile("3x4", "512", "3x4") is None
    def segment(start, end, fit, perfect, combo):
        return {"start_index": start, "end_index": end, "total_moves": end - start,
                "evaluated_moves": end - start, "goodness_of_fit": fit,
                "max_combo": combo, "performance_counts": {"Perfect!": perfect,
                "Mistake!": end - start - perfect},
                "report_file": f"free10_512_{end}_{fit:.4f}.txt",
                "replay_file": f"free10_512_{end}_{fit:.4f}.rpl"}
    intervals = [1000] * 89
    intervals[20] = None
    intervals[40] = 1_200_001
    result = build_summary([
        segment(0, 19, .9, 19, 19),
        segment(19, 39, .8, 10, 8),
        segment(39, 69, .4, 12, 11),
        segment(69, 89, 0, 20, 20),
    ], intervals)
    aggregate = result["aggregate"]
    assert [item["included"] for item in result["segments"]] == [False, True, True, False]
    assert aggregate["stage_count"] == 2
    assert aggregate["mean_goodness_of_fit"] == pytest.approx(.6)
    assert aggregate["performance_counts"] == {"Perfect!": 22, "Mistake!": 28}
    assert aggregate["max_combo"] == 11
    assert aggregate["total_moves"] == 50
    assert aggregate["timed_moves"] == 48
    assert aggregate["valid_timing_ms"] == 48_000
    assert aggregate["mean_ms_per_timed_move"] == 1000
    lossy = build_summary([segment(0, 20, .5, 10, 3)], intervals[:20],
                          source="verse", timing_lossy=True)
    assert lossy["timing_lossy"] is True
    assert lossy["aggregate"]["speed_grade_eligible"] is True


def test_stage_summary_is_persistent_but_hidden_run_stays_hidden_to_owner():
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
    }):
        init_db()
        state = engine.initial("summary-run", "4x4", SEED)
        state["score"] = 500
        previous = engine.initial("prior-run", "4x4", SEED)
        previous["score"] = 400
        with database() as db:
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,created,ended,reason,writer,state,archive,visible)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("prior-run", 7, "browser", "4x4", "prior-request", SEED, 0,
                 "sealed", 1, 2, "game_over", "writer", json.dumps(previous), b"archive", 1))
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,created,ended,reason,writer,state,archive,visible)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("summary-run", 7, "browser", "4x4", "request", SEED, 0,
                 "sealed", 1, 3, "game_over", "writer", json.dumps(state), b"archive", 1))
        summary = build_summary([], [])
        summary_id = save_summary(run_id="summary-run", user_id=7, pattern="free10",
                                  target="512", job_id="job", summary=summary)
        listed = list_summaries("summary-run", 7)[0]
        assert listed["id"] == summary_id
        assert "segments" not in listed
        assert get_summary(summary_id, 7)["segments"] == []
        assert get_summary(summary_id, 7)["run"]["pb"] == {
            "new_best": True, "previous_best": 400, "delta": 100}
        assert list_summaries("summary-run", 8)[0]["id"] == summary_id
        with database() as db:
            db.execute("UPDATE human_runs SET visible=0 WHERE id='summary-run'")
        assert list_summaries("summary-run", 7) == []
        assert get_summary(summary_id, 7) is None


def test_public_run_can_be_analyzed_by_helper_but_result_belongs_to_owner():
    from backend.human_play.analysis import _run_metadata
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "CLOUD_AUTH_DB": str(Path(temp) / "auth.sqlite3"),
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
    }):
        init_auth_db(); init_db()
        with auth_db() as db:
            for user_id, name in ((1, "Owner"), (2, "Helper")):
                db.execute("""INSERT INTO users(id,email,password_hash,display_name,status,created_at,updated_at)
                    VALUES(?,?, '!',?,'active','2026-01-01','2026-01-01')""",
                    (user_id, f"{user_id}@example.invalid", name))
        state = engine.initial("public-run", "4x4", SEED)
        with database() as db:
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,created,ended,
                 reason,writer,state,archive,visible,eligibility,has_replay)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("public-run", 1, "browser", "4x4", "request", SEED, 0, "sealed", 1, 2,
                 "game_over", "writer", json.dumps(state), b"archive", 1, "eligible", 1))
        metadata = _run_metadata("public-run", 2)
        assert metadata["subject"]["id"] == 1
        assert metadata["subject"]["display_name"] == "Owner"
        with database() as db:
            db.execute("UPDATE human_runs SET eligibility='ineligible' WHERE id='public-run'")
        with pytest.raises(service.RunError, match="analysis_run_not_found"):
            _run_metadata("public-run", 2)
        assert _run_metadata("public-run", 1)["subject"]["id"] == 1


def test_public_listing_is_monotonic_after_first_public_analysis():
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
    }):
        init_db()
        state = engine.initial("listed-run", "4x4", SEED)
        with database() as db:
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,created,ended,
                 reason,writer,state,archive,visible,eligibility,has_replay)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("listed-run", 7, "browser", "4x4", "request", SEED, 0, "sealed", 1, 2,
                 "game_over", "writer", json.dumps(state), b"archive", 1, "eligible", 1))
        summary = build_summary([], [])
        summary_id = save_summary(run_id="listed-run", user_id=7, pattern="free10",
                                  target="512", job_id="first", summary=summary, listed=True)
        save_summary(run_id="listed-run", user_id=7, pattern="free10",
                     target="512", job_id="second", summary=summary, listed=False)
        with database() as db:
            row = db.execute("SELECT listed,job_id FROM human_analysis_summaries WHERE id=?",
                             (summary_id,)).fetchone()
        assert dict(row) == {"listed": 1, "job_id": "second"}


def test_unlisted_result_is_temporarily_visible_to_payer_only():
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "CLOUD_AUTH_DB": str(Path(temp) / "auth.sqlite3"),
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
    }):
        init_auth_db(); init_db()
        with auth_db() as db:
            for user_id, name in ((1, "Owner"), (2, "Helper"), (3, "Other")):
                db.execute("""INSERT INTO users(id,email,password_hash,display_name,status,created_at,updated_at)
                    VALUES(?,?, '!',?,'active','2026-01-01','2026-01-01')""",
                    (user_id, f"{user_id}@example.invalid", name))
            future = datetime.now(timezone.utc) + timedelta(hours=1)
            db.execute("""INSERT INTO analysis_jobs
                (job_id,user_id,pattern,target,status,total,done,failed,created_at,updated_at,expires_at)
                VALUES('helper-job',2,'free10','512','finished',1,1,0,?,?,?)""",
                (datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat(),
                 future.isoformat()))
        state = engine.initial("private-result", "4x4", SEED)
        with database() as db:
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,created,ended,
                 reason,writer,state,archive,visible,eligibility,has_replay)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("private-result", 1, "browser", "4x4", "request", SEED, 0, "sealed", 1, 2,
                 "game_over", "writer", json.dumps(state), b"archive", 1, "eligible", 1))
        summary_id = save_summary(run_id="private-result", user_id=1, pattern="free10",
                                  target="512", job_id="helper-job",
                                  summary=build_summary([], []), listed=False)
        assert get_summary(summary_id, 2)["subject"]["display_name"] == "Owner"
        assert get_summary(summary_id, 3) is None
        with auth_db() as db:
            db.execute("UPDATE analysis_jobs SET expires_at='2000-01-01T00:00:00+00:00' WHERE job_id='helper-job'")
        assert get_summary(summary_id, 2) is None


def test_public_analysis_library_orders_by_game_time_not_rerun_time():
    from backend.human_play.analysis_library import list_entries
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "CLOUD_AUTH_DB": str(Path(temp) / "auth.sqlite3"),
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
    }):
        init_auth_db(); init_db()
        with auth_db() as db:
            db.execute("""INSERT INTO users(id,email,password_hash,display_name,status,created_at,updated_at)
                VALUES(1,'owner@example.invalid','!','Owner','active','2026-01-01','2026-01-01')""")
        for run_id, ended, score in (("older", 10, 900), ("newer", 20, 800)):
            state = engine.initial(run_id, "4x4", SEED); state["score"] = score
            with database() as db:
                db.execute("""INSERT INTO human_runs
                    (id,user_id,browser,variant,request_id,seed,threshold,status,created,ended,
                     reason,writer,state,archive,visible,eligibility,has_replay)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (run_id, 1, "browser", "4x4", f"request-{run_id}", SEED, 0, "sealed", 1,
                     ended, "game_over", "writer", json.dumps(state), b"archive", 1, "eligible", 1))
            save_summary(run_id=run_id, user_id=1, pattern="free10", target="512",
                         job_id=f"job-{run_id}", summary=build_summary([], []), listed=True)
        # A newer rerun timestamp for the older game must not affect discovery order.
        save_summary(run_id="older", user_id=1, pattern="free10", target="512",
                     job_id="rerun", summary=build_summary([], []), listed=True)
        # Pre-library summaries can be newer while having no retained stage
        # artifacts. They must not consume a result page and hide older,
        # actually playable library entries.
        for index in range(25):
            run_id = f"unbacked-{index}"
            state = engine.initial(run_id, "4x4", SEED)
            with database() as db:
                db.execute("""INSERT INTO human_runs
                    (id,user_id,browser,variant,request_id,seed,threshold,status,created,ended,
                     reason,writer,state,archive,visible,eligibility,has_replay)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (run_id, 1, "browser", "4x4", f"request-{run_id}", SEED, 0, "sealed", 1,
                     100 + index, "game_over", "writer", json.dumps(state), b"archive", 1,
                     "eligible", 1))
            save_summary(run_id=run_id, user_id=1, pattern="free10", target="512",
                         job_id=f"job-{run_id}", summary=build_summary([], []), listed=True)
        with database() as db:
            summary_ids = {row["run_id"]: int(row["id"]) for row in db.execute(
                "SELECT id,run_id FROM human_analysis_summaries WHERE run_id IN ('older','newer')")}
        retained = {summary_id: [{"available": True}]
                    for summary_id in summary_ids.values()}
        with patch("backend.analysis_history.library_artifacts", return_value=retained):
            result = list_entries(limit=20)
            assert [item["run_id"] for item in result["items"]] == ["newer", "older"]
            assert [item["run_id"] for item in list_entries(grade="unrated")["items"]] == [
                "newer", "older"]
            with database() as db:
                db.executemany("""INSERT INTO human_analysis_results
                    (summary_id,run_id,user_id,variant,pattern,target,run_ended_at,
                     result_created_at,metric_version,grade_version,weighted_score,grade,
                     mean_goodness_of_fit,mean_ms_per_timed_move,max_combo,stage_count,
                     evaluated_moves,final_score,active)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""", (
                    (summary_ids["older"], "older", 1, "4x4", "free10", "512", 10,
                     1, 1, 2, 75, "A", .9, 1000, 10, 1, 20, 900),
                    (summary_ids["newer"], "newer", 1, "4x4", "free10", "512", 20,
                     1, 1, 2, 80, "S", .95, 1000, 12, 1, 20, 800),
                ))
            assert [item["run_id"] for item in list_entries(grade="S")["items"]] == ["newer"]
            assert [item["run_id"] for item in list_entries(grade="A")["items"]] == ["older"]
            assert list_entries(grade="unrated")["items"] == []
        with pytest.raises(service.RunError, match="invalid_analysis_grade"):
            list_entries(grade="Z")


def test_result_grade_is_persisted_and_prior_summary_can_be_graded_without_reanalysis():
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
    }):
        init_db()
        state = engine.initial("graded-run", "4x4", SEED)
        state["score"] = 850000
        with database() as db:
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,created,ended,
                 reason,writer,state,archive,visible)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("graded-run", 7, "browser", "4x4", "graded-request", SEED, 0,
                 "sealed", 1, 2, "game_over", "writer", json.dumps(state), b"archive", 1))
        segment = {"start_index": 0, "end_index": 20, "evaluated_moves": 20,
                   "goodness_of_fit": .99, "max_combo": 20,
                   "performance_counts": {"Perfect!": 20}}
        summary = build_summary([segment], [1000] * 20)
        summary["goal_tile"] = 32768
        summary["aggregate"]["poster_eligible"] = True
        summary_id = save_summary(run_id="graded-run", user_id=7, pattern="free10",
                                  target="512", job_id="job", summary=summary)
        assert get_summary(summary_id, 7)["grade"] == "SSS"
        with database() as db:
            stored = json.loads(db.execute("SELECT summary_json FROM human_analysis_summaries WHERE id=?",
                                           (summary_id,)).fetchone()["summary_json"])
            assert stored["grade"] == "SSS"
            stored["grade"] = None
            stored["grade_version"] = 1
            stored["timing_lossy"] = True
            stored["aggregate"]["speed_grade_eligible"] = False
            db.execute("""UPDATE human_analysis_summaries
                SET summary_json=?,aggregate_json=? WHERE id=?""",
                (json.dumps(stored), json.dumps(stored["aggregate"]), summary_id))
        assert get_summary(summary_id, 7)["grade"] == "SSS"
        with database() as db:
            upgraded = json.loads(db.execute(
                "SELECT summary_json FROM human_analysis_summaries WHERE id=?",
                (summary_id,)).fetchone()["summary_json"])
        from backend.human_play.analysis_grade import GRADE_VERSION
        assert upgraded["grade_version"] == GRADE_VERSION
        assert upgraded["aggregate"]["speed_grade_eligible"] is True


def test_analysis_summary_reports_current_personal_game_rank():
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
    }):
        init_db()
        runs = []
        for run_id, score, ended in (("best", 900, 1), ("analyzed", 800, 2)):
            state = engine.initial(run_id, "4x4", SEED)
            state["score"] = score
            runs.append((run_id, 7, "browser", "4x4", f"request-{run_id}", SEED, 0,
                         "sealed", 0, ended, "game_over", "writer", json.dumps(state),
                         b"archive", 1))
        with database() as db:
            db.executemany("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,created,ended,
                 reason,writer,state,archive,visible) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", runs)
        summary_id = save_summary(run_id="analyzed", user_id=7, pattern="free10",
                                  target="512", job_id="job", summary=build_summary([], []))
        result = get_summary(summary_id, 7)
        assert result["run"]["pb"]["new_best"] is False
        assert result["run"]["personal_rank"] == 2

        later = engine.initial("later", "4x4", SEED)
        later["score"] = 850
        with database() as db:
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,created,ended,
                 reason,writer,state,archive,visible) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("later", 7, "browser", "4x4", "request-later", SEED, 0, "sealed", 0,
                 3, "game_over", "writer", json.dumps(later), b"archive", 1))
        assert get_summary(summary_id, 7)["run"]["personal_rank"] == 3


def test_analyzer_captures_same_unrounded_fit_as_written_stage_files():
    from backend.analysis_core import Analyzer
    analyzer = Analyzer.__new__(Analyzer)
    from engine_core.GoalSpec import GoalSpec
    analyzer.goal = GoalSpec.parse(512)
    analyzer.sum_goal_completed = False
    analyzer.segment_summaries = []
    analyzer.segment_start_index = 5
    analyzer.text_list = ["line"] * 101
    analyzer.goodness_of_fit = .7654321
    analyzer.step_count = 17
    analyzer.max_combo = 9
    analyzer.performance_stats = {"**Perfect!**": 12, "**Mistake!**": 5}
    analyzer.write_analysis = lambda step: f"free10_512_{step}_0.7654.txt"
    analyzer.save_rec_to_file = lambda step: f"free10_512_{step}_0.7654.rpl"
    analyzer._flush_segment(25, 25)
    assert analyzer.segment_summaries[0]["start_index"] == 5
    assert analyzer.segment_summaries[0]["total_moves"] == 20
    assert analyzer.segment_summaries[0]["goodness_of_fit"] == .7654321
    assert analyzer.segment_summaries[0]["performance_counts"]["Perfect!"] == 12


def test_analyzer_flushes_previous_stage_before_analyzing_new_board():
    from backend.analysis_core import Analyzer
    analyzer = Analyzer.__new__(Analyzer)
    from engine_core.GoalSpec import GoalSpec
    analyzer.goal = GoalSpec.parse(512)
    analyzer.sum_goal_completed = False
    analyzer.pattern = "free10"
    analyzer.target = 9
    analyzer.large_tile_sum = 0
    analyzer.record_list = [(0, 0, 1, 1, 0), (1, 0, 1, 1, 0)]
    analyzer.bm = SimpleNamespace(decode_board=lambda encoded: np.array([int(encoded)]))
    analyzer.check_nth_largest = lambda encoded: True
    analyzer.mask_large_tiles = lambda board: board + (1 if int(board[0]) == 0 else 2)
    calls = []
    analyzer._flush_segment = lambda report_step, end_index: calls.append(("flush", end_index))
    analyzer._analyze_one_step = lambda *args: calls.append(("move", int(args[0][0]))) or True
    assert analyzer.analyze_one_step(0) is True
    assert analyzer.analyze_one_step(1) is True
    assert calls == [("flush", 0), ("move", 0), ("flush", 1), ("move", 1)]


def test_analyzer_does_not_count_unmatched_move_in_next_stage():
    from backend.analysis_core import Analyzer
    analyzer = Analyzer.__new__(Analyzer)
    from engine_core.GoalSpec import GoalSpec
    analyzer.goal = GoalSpec.parse(512)
    analyzer.sum_goal_completed = False
    analyzer.pattern = "free10"
    analyzer.large_tile_sum = 7
    analyzer.segment_start_index = 0
    analyzer.record_list = [(0, 0, 1, 1, 0)]
    analyzer.bm = SimpleNamespace(decode_board=lambda encoded: np.array([0]))
    analyzer.check_nth_largest = lambda encoded: False
    analyzer._flush_segment = lambda report_step, end_index: None
    assert analyzer.analyze_one_step(0) is False
    assert analyzer.segment_start_index == 1
    assert analyzer.large_tile_sum == 0


def test_verse_unknown_interval_remains_unknown_after_normalization(tmp_path):
    from backend.human_play import verse_replay
    replay = (Path(__file__).resolve().parents[1] / "frontend" / "tests" / "fixtures" /
              "verse-replay" / "Blueawa_3x4_2026-09-20_71356.vrs").read_bytes()
    variant, initial, moves = verse_replay._legacy(replay.decode("latin-1"))
    assert any(move[3] == verse_replay.UNKNOWN_TIMING_MS for move in moves)
    normalized = verse_replay._encode(variant, initial, moves)
    path = tmp_path / "input.vrs"
    path.write_text(verse_replay.PREFIX + base64.b64encode(normalized).decode("ascii"))
    intervals = intervals_from_analysis_input(path)
    assert intervals.count(None) == 1
    assert len(intervals) == len(moves)


@pytest.mark.parametrize("variant", list(engine.VARIANTS))
def test_archived_replay_converts_to_analyzer_format(variant):
    run_id = "analysis-test"
    state = engine.initial(run_id, variant, SEED)
    for direction in range(4):
        board, _ = engine.move(state["board"], *engine.VARIANTS[variant], direction)
        if board != state["board"]:
            rng = Xoshiro128StarStar(list(state["rng"]))
            position, value = engine.spawn(board, rng)
            event = engine.EVENT.pack(direction | position << 2 | (64 if value == 4 else 0), 2345)
            break
    run = {"id": run_id, "variant": variant, "seed": SEED, "reason": "game_over", "created": 1}
    archive = gzip.compress(engine.replay_bytes(run, event, version=2))
    replay = decode_2048next_replay(archived_to_analysis_text(
        archive, expected_run_id=run_id, expected_variant=variant, expected_seq=1))
    assert (replay.height, replay.width) == engine.VARIANTS[variant]
    moves = [record for record in replay.records if isinstance(record, MoveRecord)]
    assert len(moves) == 1
    assert (moves[0].direction, moves[0].spawn_index, moves[0].delta_ms) == (direction, position, 2345)
    with pytest.raises(ValueError, match="archive_mismatch"):
        archived_to_analysis_text(archive, expected_run_id=run_id, expected_variant=variant, expected_seq=2)


def test_hidden_archived_run_is_unavailable_even_to_owner():
    from backend.human_play.analysis import _run_metadata
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "CLOUD_AUTH_DB": str(Path(temp) / "auth.sqlite3"),
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
    }):
        init_auth_db(); init_db()
        state = engine.initial("hidden-run", "4x4", SEED)
        with database() as db:
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,created,writer,state,archive,visible)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("hidden-run", 7, "browser", "4x4", "request", SEED, 0, "sealed", 1,
                 "writer", json.dumps(state), b"archive", 0))
        with pytest.raises(service.RunError, match="analysis_run_not_found"):
            _run_metadata("hidden-run", 7)
        with database() as db:
            db.execute("UPDATE human_runs SET visible=1 WHERE id='hidden-run'")
        assert _run_metadata("hidden-run", 7)["moves"] == 0


def test_shared_queue_waiting_limit_is_60_items():
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "CLOUD_AUTH_DB": str(Path(temp) / "auth.sqlite3"),
    }):
        init_auth_db()
        def job(name, user, count):
            return AnalysisJob(name, user, None, "p", "512", 9, "p_512", [], {},
                               Path(temp) / name, count)
        _admit_job(job("first", 1, 60))
        with pytest.raises(AnalysisQueueFull, match="analysis_queue_full"):
            _admit_job(job("overflow", 2, 1))
        with pytest.raises(AnalysisQueueFull, match="user_analysis_queue_full"):
            _admit_job(job("same-user", 1, 1))
        assert _claim_job() == "first"
        assert _claim_job() is None  # One global execution lease, even with another worker.


def test_batch_reservation_is_atomic_when_second_item_exceeds_balance():
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "CLOUD_AUTH_DB": str(Path(temp) / "auth.sqlite3"),
    }):
        init_auth_db()
        with auth_db() as db:
            db.execute("""INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at)
                VALUES(1,'test@example.invalid','!','Test','2026-01-01','2026-01-01')""")
            db.execute("""INSERT INTO token_accounts(user_id,bonus_balance_units,paid_balance_units,created_at,updated_at)
                VALUES(1,150000,0,'2026-01-01','2026-01-01')""")
        with pytest.raises(InsufficientTokens):
            reserve_operation_tokens_many(user_id=1, session_id=None, operation_key="analysis_per_replay",
                                          full_patterns=["L3_512", "L3_1024"])
        with auth_db() as db:
            assert db.execute("SELECT bonus_balance_units FROM token_accounts WHERE user_id=1").fetchone()[0] == 150000
            assert db.execute("SELECT count(*) FROM token_ledger WHERE event_type='reserve'").fetchone()[0] == 0


def test_archive_id_creates_task_without_client_reupload():
    from backend.human_play import analysis as human_analysis
    from backend import cloud_analysis_jobs as jobs
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
        "CLOUD_AUTH_DB": str(Path(temp) / "auth.sqlite3"),
        "HUMAN_PLAY_DB": str(Path(temp) / "human.sqlite3"),
        "CLOUD_UPLOAD_ROOT": str(Path(temp) / "uploads"),
    }):
        init_auth_db(); init_db()
        with auth_db() as db:
            db.execute("""INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at)
                VALUES(1,'analysis@example.invalid','!','Analyst','2026-01-01','2026-01-01')""")
            db.execute("""INSERT INTO token_accounts(user_id,bonus_balance_units,paid_balance_units,created_at,updated_at)
                VALUES(1,200000,0,'2026-01-01','2026-01-01')""")
        state = engine.initial("archived-run", "4x4", SEED)
        archive = gzip.compress(engine.replay_bytes({"id": "archived-run", "variant": "4x4",
            "seed": SEED, "reason": "game_over", "created": 1}, b"", version=2))
        with database() as db:
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,created,writer,state,archive,visible)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("archived-run", 1, "browser", "4x4", "request", SEED, 0, "sealed", 1,
                 "writer", json.dumps(state), archive, 1))

        class FakeAnalyzer:
            def __init__(self, **kwargs):
                self.path = Path(kwargs["file_path"])
                self.output = Path(kwargs["target_path"])
                self.segment_summaries = []
            def generate_reports(self):
                assert self.path.read_text(encoding="ascii").startswith("REPLAY_v1RPL_B64_")
                (self.output / "report.txt").write_text("analyzed", encoding="utf-8")

        with patch.object(human_analysis, "resolve_configured_tablebase", return_value={"_provider": "local"}), \
             patch.object(human_analysis, "get_catalog_version", return_value="test-catalog"), \
             patch.object(jobs, "start_analysis_worker"), patch.object(jobs, "Analyzer", FakeAnalyzer):
            selection = [{"pattern": "L3", "target": "512"}, {"pattern": "L3", "target": "1024"}]
            estimate = human_analysis.quote("archived-run", 1, selection)
            task = human_analysis.create("archived-run", 1, None, selection,
                                         estimate["total_cost_units"], "test-catalog",
                                         "request-id-00000001")
            assert task.total == 2
            with auth_db() as db:
                assert db.execute("SELECT job_id FROM human_analysis_requests WHERE user_id=1 AND request_id=?",
                                  ("request-id-00000001",)).fetchone()[0] == task.job_id
                assert db.execute("SELECT count(*) FROM analysis_queue WHERE job_id=?", (task.job_id,)).fetchone()[0] == 1
            with patch.object(human_analysis, "reserve_operation_tokens_many", side_effect=AssertionError("charged twice")):
                retried = human_analysis.create("archived-run", 1, None, selection,
                                                estimate["total_cost_units"], "test-catalog",
                                                "request-id-00000001")
                assert retried.job_id == task.job_id
            assert len(task.input_paths) == 1
            assert task.input_paths[0].exists()
            assert jobs._claim_job() == task.job_id
            jobs._run_job(task.job_id)
            assert task.status == "queued" and task.completed == 1
            with auth_db() as db:
                db.execute("UPDATE analysis_queue SET status='queued',lease_owner=NULL,lease_until=0 WHERE job_id=?", (task.job_id,))
            assert jobs._claim_job() == task.job_id
            jobs._run_job(task.job_id)
            with auth_db() as db:
                db.execute("UPDATE analysis_queue SET status='finished',pending=0 WHERE job_id=?", (task.job_id,))
            result = jobs.get_analysis_job(task.job_id, user_id=1)
            assert result.status == "finished"
            assert result.zip_path is not None and result.zip_path.exists()
            assert result.done == 2
            assert [item["status"] for item in jobs.analysis_job_payload(result)["items"]] == ["done", "done"]
            assert len(list_summaries("archived-run", 1)) == 2
            assert all(item["summary_id"] for item in result.entries)
