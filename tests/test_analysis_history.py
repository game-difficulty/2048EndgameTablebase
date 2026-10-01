import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.analysis_history import (
    analysis_replay_viewer_url, artifact_root, enforce_limits, get_history, init_schema,
    publish_segments,
)
from backend.auth.db import auth_db, init_auth_db
from backend.cloud_files import SavedUpload, delete_upload, register_upload
from engine_core.replay_utils import REPLAY_DTYPE, replay_sentinel


def _user(db, user_id=1):
    db.execute("""INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at)
        VALUES(?,?, '!','Player','2026-01-01','2026-01-01')""",
               (user_id, f"player-{user_id}@example.invalid"))


def test_upload_is_physically_deleted_by_id(tmp_path):
    with patch.dict(os.environ, {"CLOUD_AUTH_DB": str(tmp_path / "auth.sqlite3"),
                                 "CLOUD_UPLOAD_ROOT": str(tmp_path / "uploads")}):
        init_auth_db()
        with auth_db() as db:
            _user(db)
        path = tmp_path / "uploads" / "source.txt"
        path.parent.mkdir(); path.write_text("source", encoding="utf-8")
        record = register_upload(SavedUpload(path, "source.txt", "source.txt", 6, "text/plain"),
                                 kind="analysis", user_id=1)
        assert delete_upload(record.upload_id, 1)
        assert not path.exists()
        with auth_db() as db:
            assert db.execute("SELECT 1 FROM uploads WHERE upload_id=?", (record.upload_id,)).fetchone() is None


def test_publish_replay_and_keep_history_when_artifact_is_pruned(tmp_path):
    with patch.dict(os.environ, {"CLOUD_AUTH_DB": str(tmp_path / "auth.sqlite3"),
                                 "CLOUD_UPLOAD_ROOT": str(tmp_path / "uploads")}):
        init_auth_db()
        with auth_db() as db:
            _user(db)
            init_schema(db)
            db.execute("""INSERT INTO analysis_history_jobs
                (job_id,user_id,origin,status,total,done,failed,metadata_json,created_at)
                VALUES('job',1,'main_upload','running',1,0,0,'{}',?)""", (time.time(),))
            db.execute("""INSERT INTO analysis_history_items
                (id,job_id,work_index,source_filename,pattern,target,status)
                VALUES(1,'job',0,'source.vrs','free10','512','running')""")
        replay = np.zeros(2, dtype=REPLAY_DTYPE)
        replay[-1] = replay_sentinel()
        replay_path = tmp_path / "stage.rpl"; replay.tofile(replay_path)
        analyzer = SimpleNamespace(variant="4x4", segment_summaries=[{
            "start_index": 10, "end_index": 11, "evaluated_moves": 1,
            "goodness_of_fit": 1.0, "max_combo": 1,
            "performance_counts": {"Perfect!": 1}, "replay_path": str(replay_path),
        }])
        published = publish_segments(item_id=1, analyzer=analyzer, pattern="free10", target="512")
        assert len(published) == 1
        artifact_id = published[0]["artifact_id"]
        detail = get_history("job", 1)
        assert detail["items"][0]["artifacts"][0]["available"] is True
        assert detail["items"][0]["artifacts"][0]["goodness_of_fit"] == 1.0
        # Old manifests contain IDs and positions but no fit: polling must hydrate
        # persisted metadata rather than requiring the player to pay for a rerun.
        from backend.cloud_analysis_jobs import analysis_job_payload
        job = SimpleNamespace(job_id="job", user_id=1, status="running", zip_path=None,
            pattern="free10", target="512", completed=1, total=1, done=1, failed=0,
            current_file="", error="", work_items=[SimpleNamespace(pattern="free10", target="512")],
            entries=[{"status": "done", "artifacts": [{"artifact_id": artifact_id}]}])
        payload = analysis_job_payload(job)
        assert payload["items"][0]["artifacts"][0]["goodness_of_fit"] == 1.0
        assert payload["entries"][0]["artifacts"][0]["goodness_of_fit"] == 1.0
        assert "goodness_of_fit" not in job.entries[0]["artifacts"][0]

        # Replay titles use score and stage position, not an internal filename
        # or a display name that may belong to a different account.
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from urllib.parse import unquote
        from backend.analysis_history import router
        app = FastAPI(); app.include_router(router)
        with auth_db() as db:
            _user(db, 2)
            db.execute("UPDATE users SET display_name='游戏/难度' WHERE id=2")
            db.execute("UPDATE analysis_history_jobs SET subject_user_id=2 WHERE job_id='job'")
            db.execute("UPDATE analysis_history_items SET score=7357 WHERE id=1")
            db.execute("UPDATE analysis_history_items SET source_run_id='missing-source' WHERE id=1")
        payload = analysis_job_payload(job)
        assert payload["entries"][0]["score"] == 7357
        assert payload["entries"][0]["variant"] == "4x4"
        from backend.analysis_history import job_item_metadata
        assert job_item_metadata("job", 2) == {}
        with TestClient(app) as client, patch("backend.analysis_history.current_user_from_request", return_value={"id": 1}):
            for fit, expected in ((1.0, "100.0%"), (0.0, "0.0%"), (None, "—")):
                with auth_db() as db:
                    db.execute("UPDATE analysis_replay_artifacts SET goodness_of_fit=? WHERE artifact_id=?", (fit, artifact_id))
                response = client.get(f"/api/analysis/replays/{artifact_id}")
                assert response.status_code == 200
                assert response.content == replay_path.read_bytes()
                assert unquote(response.headers["X-Replay-Title"]) == f"7,357 分 · 4×4 · 第 11 步 · free10-512 · {expected}"
                assert analysis_job_payload(job)["items"][0]["artifacts"][0]["goodness_of_fit"] == fit
        with auth_db() as db:
            row = db.execute("SELECT relative_path FROM analysis_replay_artifacts WHERE artifact_id=?",
                             (artifact_id,)).fetchone()
        (artifact_root() / row["relative_path"]).unlink()
        detail = get_history("job", 1)
        assert detail["items"][0]["artifacts"][0]["available"] is False
        assert analysis_job_payload(job)["items"][0]["artifacts"][0]["available"] is False


def test_archive_history_uses_original_game_score(tmp_path):
    from backend.human_play.store import database, init_db
    from backend.analysis_history import _source_run_details
    with patch.dict(os.environ, {"CLOUD_AUTH_DB": str(tmp_path / "auth.sqlite3"),
                                 "HUMAN_PLAY_DB": str(tmp_path / "human.sqlite3")}):
        init_db()
        with database() as db:
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,created,
                 writer,state) VALUES('run-1',1,'browser','3x4','request','seed',0,
                 'sealed',1,'writer','{"score":133560}')""")
        assert _source_run_details(["run-1"]) == {
            "run-1": {"score": 133560, "variant": "3x4"}
        }


def test_source_metadata_missing_database_is_optional_and_never_created(tmp_path, caplog):
    from backend.analysis_history import _source_run_details
    source = tmp_path / "missing # source.sqlite3"
    with patch.dict(os.environ, {"HUMAN_PLAY_DB": str(source)}):
        assert _source_run_details(["run-1"]) == {}
    assert not source.exists()
    assert "Analysis source metadata unavailable" in caplog.text


def test_source_metadata_missing_table_or_corrupt_database_is_optional(tmp_path):
    import sqlite3
    from backend.analysis_history import _source_run_details
    source = tmp_path / "source.sqlite3"
    with sqlite3.connect(source):
        pass
    with patch.dict(os.environ, {"HUMAN_PLAY_DB": str(source)}):
        assert _source_run_details(["run-1"]) == {}
        source.write_bytes(b"not a sqlite database")
        assert _source_run_details(["run-1"]) == {}


def test_source_metadata_missing_record_and_invalid_state(tmp_path):
    import sqlite3
    from backend.analysis_history import _source_run_details
    source = tmp_path / "source.sqlite3"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE human_runs(id TEXT,variant TEXT,state TEXT)")
        db.execute("INSERT INTO human_runs VALUES('bad','4x4','null')")
    with patch.dict(os.environ, {"HUMAN_PLAY_DB": str(source)}):
        assert _source_run_details(["missing", "bad"]) == {"bad": {"score": None, "variant": "4x4"}}


def test_free_account_keeps_newest_fifty_replays(tmp_path):
    with patch.dict(os.environ, {"CLOUD_AUTH_DB": str(tmp_path / "auth.sqlite3"),
                                 "CLOUD_UPLOAD_ROOT": str(tmp_path / "uploads")}):
        init_auth_db(); root = artifact_root()
        with auth_db() as db:
            _user(db); init_schema(db)
            db.execute("""INSERT INTO analysis_history_jobs
                (job_id,user_id,origin,status,total,done,failed,metadata_json,created_at)
                VALUES('job',1,'main_upload','finished',51,51,0,'{}',1)""")
            for index in range(51):
                db.execute("""INSERT INTO analysis_history_items
                    (id,job_id,work_index,source_filename,pattern,target,status)
                    VALUES(?, 'job', ?, 'x','free10','512','done')""", (index + 1, index))
                relative = f"test/{index}.rpl"; path = root / relative
                path.parent.mkdir(exist_ok=True); path.write_bytes(b"x")
                db.execute("""INSERT INTO analysis_replay_artifacts
                    (artifact_id,history_item_id,segment_index,relative_path,byte_size,
                     source_start_index,source_end_index,replay_move_count,evaluated_moves,
                     max_combo,performance_counts_json,pattern,target,use_variant,format_version,created_at)
                    VALUES(?,?,?,?,1,0,1,1,1,0,'{}','free10','512',0,1,?)""",
                    (f"a{index}", index + 1, 0, relative, index))
        assert enforce_limits(1) == 1
        with auth_db() as db:
            active = db.execute("SELECT COUNT(*) FROM analysis_replay_artifacts WHERE deleted_at IS NULL").fetchone()[0]
            oldest = db.execute("SELECT delete_reason FROM analysis_replay_artifacts WHERE artifact_id='a0'").fetchone()[0]
        assert active == 50
        assert oldest == "user_limit"


def test_analysis_replay_viewer_url_busts_cached_entry_document():
    url = analysis_replay_viewer_url("artifact-id", "signed-token")
    assert url == (
        "https://2048tables.online/?tab=replay&analysis_replay_v=4"
        "#analysisReplay=artifact-id&token=signed-token"
    )
