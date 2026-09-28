from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.auth.dependencies import require_user
from backend.human_play.production_app import app
from backend.quota.errors import InsufficientTokens


def _post_analysis():
    client = TestClient(app, raise_server_exceptions=False)
    try:
        return client.post(
            "/api/analysis/jobs",
            data={"pattern": "free10", "target": "512"},
            files={"files": ("sample.rpl", b"replay", "application/octet-stream")},
        )
    finally:
        client.close()


def test_play_upload_creates_history_backed_job():
    app.dependency_overrides[require_user] = lambda: {"id": 7, "session_id": 9}
    upload = SimpleNamespace(upload_id="upload-1", filename="sample.rpl")
    try:
        with (
            patch("backend.human_play.production_app.resolve_configured_tablebase", return_value={"_provider": "local"}),
            patch("backend.human_play.production_app.check_analysis_capacity"),
            patch("backend.human_play.production_app.save_upload_file", new_callable=AsyncMock) as save_file,
            patch("backend.human_play.production_app.register_upload", return_value=upload),
            patch("backend.human_play.production_app.reserve_operation_tokens_many", return_value=["reservation"]) as reserve,
            patch("backend.human_play.production_app.create_analysis_job", return_value=SimpleNamespace(job_id="job-1", total=1)) as create,
            patch("backend.human_play.production_app.record_usage"),
            patch("backend.human_play.production_app.get_token_balance", return_value=1200),
        ):
            response = _post_analysis()
        assert response.status_code == 200
        assert response.json() == {"job_id": "job-1", "total": 1, "token_balance": 1200}
        assert save_file.await_count == 1
        assert reserve.call_args.kwargs["full_patterns"] == ["free10_512"]
        assert create.call_args.kwargs["quota_reservations"] == ["reservation"]
    finally:
        app.dependency_overrides.clear()


def test_play_upload_releases_file_when_balance_is_insufficient():
    app.dependency_overrides[require_user] = lambda: {"id": 7, "session_id": 9}
    upload = SimpleNamespace(upload_id="upload-1", filename="sample.rpl")
    try:
        with (
            patch("backend.human_play.production_app.resolve_configured_tablebase", return_value={"_provider": "local"}),
            patch("backend.human_play.production_app.check_analysis_capacity"),
            patch("backend.human_play.production_app.save_upload_file", new_callable=AsyncMock),
            patch("backend.human_play.production_app.register_upload", return_value=upload),
            patch("backend.human_play.production_app.reserve_operation_tokens_many", side_effect=InsufficientTokens(required_units=2000, balance_units=0)),
            patch("backend.human_play.production_app.delete_upload") as delete_upload,
        ):
            response = _post_analysis()
        assert response.status_code == 402
        assert response.json()["detail"]["code"] == "INSUFFICIENT_TOKENS"
        delete_upload.assert_called_once_with("upload-1", 7)
    finally:
        app.dependency_overrides.clear()


def test_play_upload_rejects_offline_remote_before_saving():
    app.dependency_overrides[require_user] = lambda: {"id": 7, "session_id": 9}
    try:
        with (
            patch("backend.human_play.production_app.resolve_configured_tablebase", return_value={"_provider": "remote", "_available": False}),
            patch("backend.human_play.production_app.save_upload_file", new_callable=AsyncMock) as save_file,
        ):
            response = _post_analysis()
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "REMOTE_TABLEBASE_OFFLINE"
        save_file.assert_not_awaited()
    finally:
        app.dependency_overrides.clear()
