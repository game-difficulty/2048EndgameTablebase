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
        with auth_db() as db:
            row = db.execute("SELECT relative_path FROM analysis_replay_artifacts WHERE artifact_id=?",
                             (artifact_id,)).fetchone()
        (artifact_root() / row["relative_path"]).unlink()
        detail = get_history("job", 1)
        assert detail["items"][0]["artifacts"][0]["available"] is False


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
        "https://2048tables.online/?tab=replay&analysis_replay_v=2"
        "#analysisReplay=artifact-id&token=signed-token"
    )
