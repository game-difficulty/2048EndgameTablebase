from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path
import sys

import websockets


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BLOCKED_ACTIONS = (
    "START_BUILD",
    "REPLAY_LOAD_FILE",
    "TESTER_SAVE_LOG",
    "NOTEBOOK_GET_INIT",
)


def _http_get_json(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _multipart_body(
    *,
    field_name: str,
    filename: str,
    content: bytes,
    fields: dict[str, str],
) -> tuple[bytes, str]:
    boundary = f"----2048tables{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("ascii"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("ascii"),
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("ascii"),
            b"Content-Type: application/octet-stream\r\n\r\n",
            content,
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    return b"".join(chunks), boundary


def _http_upload(base_url: str, filename: str, content: bytes, kind: str) -> tuple[int, str]:
    body, boundary = _multipart_body(
        field_name="files",
        filename=filename,
        content=content,
        fields={"kind": kind},
    )
    request = urllib.request.Request(
        f"{base_url}/api/uploads",
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def assert_catalog_safe(base_url: str) -> None:
    payload = _http_get_json(base_url, "/api/tablebases")
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise AssertionError("/api/tablebases did not return a tables list")
    forbidden_keys = {"root", "relative_path", "path", "filepath", "absolute_path"}
    for table in tables:
        leaked = forbidden_keys.intersection(table.keys())
        if leaked:
            raise AssertionError(f"catalog leaks server path fields: {sorted(leaked)}")
    print(f"PASS catalog_safe tables={len(tables)}")


def assert_upload_policy(base_url: str) -> None:
    status, body = _http_upload(base_url, "sample.rpl", b"", "replay")
    if status != 200:
        raise AssertionError(f"valid replay upload expected 200, got {status}: {body}")
    payload = json.loads(body)
    uploads = payload.get("uploads")
    if not isinstance(uploads, list) or len(uploads) != 1:
        raise AssertionError(f"upload response malformed: {payload}")
    upload = uploads[0]
    if set(upload.keys()) - {"upload_id", "filename", "size", "content_type", "kind"}:
        raise AssertionError(f"upload response has unexpected keys: {upload}")

    status, _body = _http_upload(base_url, "bad.exe", b"bad", "replay")
    if status != 400:
        raise AssertionError(f"invalid replay extension expected 400, got {status}")
    print("PASS upload_policy")


def assert_download_registry_guard() -> None:
    from backend.cloud_files import get_download_record, register_download_path

    temp_root = Path(tempfile.mkdtemp(prefix="2048tables-smoke-"))
    download_root = temp_root / "downloads"
    download_root.mkdir(parents=True)
    original_root = os.environ.get("CLOUD_UPLOAD_ROOT")
    os.environ["CLOUD_UPLOAD_ROOT"] = str(temp_root / "uploads")
    try:
        safe_file = download_root / "safe.txt"
        safe_file.write_text("ok", encoding="utf-8")
        record = register_download_path(safe_file, filename="../safe.txt")
        if record.filename != "safe.txt":
            raise AssertionError(f"download filename was not sanitized: {record.filename}")
        resolved = get_download_record(record.download_id)
        if resolved.path != safe_file.resolve():
            raise AssertionError("download registry returned the wrong path")
    finally:
        if original_root is None:
            os.environ.pop("CLOUD_UPLOAD_ROOT", None)
        else:
            os.environ["CLOUD_UPLOAD_ROOT"] = original_root
    print("PASS download_registry_guard")


async def assert_ws_denylist(ws_url: str) -> None:
    async with websockets.connect(f"{ws_url}/ws/smoke_{uuid.uuid4().hex[:8]}") as ws:
        for action in BLOCKED_ACTIONS:
            await ws.send(json.dumps({"action": action, "data": {"path": "/etc/passwd"}}))
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            payload = json.loads(raw)
            if payload.get("action") != "ERROR":
                raise AssertionError(f"{action} was not rejected: {payload}")
            message = payload.get("data", {}).get("message", "")
            if "/etc/passwd" in message:
                raise AssertionError(f"{action} error leaked supplied path: {message}")
    print("PASS ws_denylist")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the cloud runtime boundary.")
    parser.add_argument("--base-url", default=os.getenv("CLOUD_SMOKE_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--ws-url", default=os.getenv("CLOUD_SMOKE_WS_URL", "ws://127.0.0.1:8000"))
    args = parser.parse_args()

    assert_catalog_safe(args.base_url.rstrip("/"))
    assert_upload_policy(args.base_url.rstrip("/"))
    assert_download_registry_guard()
    asyncio.run(assert_ws_denylist(args.ws_url.rstrip("/")))
    print("PASS cloud_smoke_test")


if __name__ == "__main__":
    main()
