"""Grant user 89 room creation without granting platform-wide organizer powers.

Run on the tournament host as root. The source release and four file digests are
pinned so concurrent deployments cannot silently become part of this change.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time


ROOT = Path("/opt/2048tables/tournament-app")
OLD = ROOT / "releases/20260929-client-runtime-r1"
NEW = ROOT / "releases/20260929-room-creator-89-r1"
SERVICE = "2048tables-competition-test.service"
ENV_FILE = Path("/etc/2048tables/competition-test.env")
AUTH_DB = Path("/var/lib/2048tables/auth.sqlite3")
COMPETITION_DB = Path("/var/lib/2048tables/competition-test/competition.sqlite3")
EXPECTED_SHA256 = {
    "config.py": "7daddb08c3804a23e8a1e8732606b305fc9bdf62cbf99b986d0c850554fc062e",
    "app.py": "442360de2e228b64b802baa302e756c40411e3f6996fa577e1614af2f1c5c586",
    "routes.py": "b596c6f5738e36dfbd65ffa3587792545985b81e8c8c56d7a490f842b06c2f7f",
    "service.py": "40069fc9073824a2663d5d9144f239184d70fc6887606ef603e7cfa26e0eab5c",
}


def replace_once(content: str, original: str, replacement: str, label: str) -> str:
    count = content.count(original)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return content.replace(original, replacement, 1)


def patch_source(relative: str, changes: list[tuple[str, str]]) -> None:
    source = OLD / "competition/backend" / relative
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256[relative]:
        raise RuntimeError(f"source changed: {relative}")
    content = raw.decode("utf-8").replace("\r\n", "\n")
    for before, after in changes:
        content = replace_once(content, before, after, relative)
    (NEW / "competition/backend" / relative).write_text(content, encoding="utf-8")


def switch(target: Path) -> None:
    temporary = ROOT / "current.next-room-creator-89-r1"
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary symlink exists: {temporary}")
    temporary.symlink_to(target)
    os.replace(temporary, ROOT / "current")


def check_status() -> None:
    with sqlite3.connect(f"file:{COMPETITION_DB}?mode=ro", uri=True) as db:
        statuses = db.execute("SELECT status, count(*) FROM competitions GROUP BY status").fetchall()
    print("room statuses:", statuses, flush=True)
    if any(status not in {"SEATING", "FINISHED", "CANCELLED"} for status, _ in statuses):
        raise RuntimeError("active match present; service restart postponed")


def check_user() -> None:
    with sqlite3.connect(f"file:{AUTH_DB}?mode=ro", uri=True) as db:
        row = db.execute("SELECT display_name, email, role, status FROM users WHERE id = 89").fetchone()
    if row != ("sumeka", "suwenka@outlook.com", "user", "active"):
        raise RuntimeError("user 89 identity does not match the requested account")


def check_service() -> None:
    for _ in range(20):
        result = subprocess.run(["curl", "-ksS", "--resolve", "tournament.2048tables.online:443:127.0.0.1",
                                 "https://tournament.2048tables.online/api/health"],
                                capture_output=True, text=True)
        if result.returncode == 0 and '"ok":true' in result.stdout:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("tournament API health check failed")
    subprocess.run(["systemctl", "is-active", "--quiet", SERVICE], check=True)
    pid = subprocess.run(["systemctl", "show", "-p", "MainPID", "--value", SERVICE],
                         capture_output=True, text=True, check=True).stdout.strip()
    environment = Path(f"/proc/{int(pid)}/environ").read_bytes().split(b"\0")
    if b"COMPETITION_ROOM_CREATOR_IDS=89" not in environment:
        raise RuntimeError("running service did not receive room creator setting")


def main() -> None:
    if os.geteuid() != 0:
        raise RuntimeError("run as root on the tournament host")
    current = ROOT / "current"
    if not current.is_symlink() or current.resolve(strict=True) != OLD:
        raise RuntimeError("current release changed")
    if NEW.exists():
        # A previous preparation may have stopped before changing any files.
        # Only resume when every target file is still an exact baseline copy.
        for name, digest in EXPECTED_SHA256.items():
            candidate = NEW / "competition/backend" / name
            if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
                raise RuntimeError("target release exists with changes; refusing to overwrite")
    check_user()
    check_status()
    original_env = ENV_FILE.read_bytes()
    if any(line.startswith(b"COMPETITION_ROOM_CREATOR_IDS=") for line in original_env.splitlines()):
        raise RuntimeError("creator allowlist already configured")
    shutil.copytree(OLD, NEW, dirs_exist_ok=True)
    patch_source("config.py", [
        ("    bootstrap_organizer_ids: frozenset[int]\n", "    bootstrap_organizer_ids: frozenset[int]\n    room_creator_ids: frozenset[int]\n"),
        ("    organizer_ids = frozenset(int(item) for item in raw_ids if item.isdigit())\n",
         "    organizer_ids = frozenset(int(item) for item in raw_ids if item.isdigit())\n"
         "    raw_creator_ids = _csv(\"COMPETITION_ROOM_CREATOR_IDS\")\n"
         "    room_creator_ids = frozenset(int(item) for item in raw_creator_ids if item.isdigit())\n"),
        ("        bootstrap_organizer_ids=organizer_ids,\n",
         "        bootstrap_organizer_ids=organizer_ids,\n        room_creator_ids=room_creator_ids,\n"),
    ])
    patch_source("app.py", [
        ("        bootstrap_organizer_ids=settings.bootstrap_organizer_ids,\n",
         "        bootstrap_organizer_ids=settings.bootstrap_organizer_ids,\n"
         "        room_creator_ids=settings.room_creator_ids,\n"),
    ])
    patch_source("routes.py", [
        ('"can_create_competition": service._is_platform_organizer(principal)',
         '"can_create_competition": service._can_create_competition(principal)'),
    ])
    patch_source("service.py", [
        ("        bootstrap_organizer_ids: frozenset[int] = frozenset(),\n",
         "        bootstrap_organizer_ids: frozenset[int] = frozenset(),\n"
         "        room_creator_ids: frozenset[int] = frozenset(),\n"),
        ("        self.bootstrap_organizer_ids = bootstrap_organizer_ids\n",
         "        self.bootstrap_organizer_ids = bootstrap_organizer_ids\n"
         "        self.room_creator_ids = room_creator_ids\n"),
        ("    def _new_room_code(self) -> str:\n",
         "    def _can_create_competition(self, principal: Principal) -> bool:\n"
         "        return self._is_platform_organizer(principal) or principal.user_id in self.room_creator_ids\n\n"
         "    def _new_room_code(self) -> str:\n"),
        ("        if not self._is_platform_organizer(principal):\n            raise CompetitionError(\n"
         "                \"ORGANIZER_REQUIRED\",\n"
         "                \"Only a platform organizer can create a competition room.\",\n",
         "        if not self._can_create_competition(principal):\n            raise CompetitionError(\n"
         "                \"ORGANIZER_REQUIRED\",\n"
         "                \"Only a platform organizer can create a competition room.\",\n"),
    ])
    subprocess.run(["/opt/2048tables/venv/bin/python", "-m", "compileall", "-q",
                    str(NEW / "competition/backend")], check=True)
    updated_env = original_env + (b"" if original_env.endswith(b"\n") else b"\n") + b"COMPETITION_ROOM_CREATOR_IDS=89\n"
    temporary_env = ENV_FILE.with_name("competition-test.env.next-room-creator-89-r1")
    if temporary_env.exists():
        raise RuntimeError(f"temporary environment file exists: {temporary_env}")
    try:
        temporary_env.write_bytes(updated_env)
        shutil.copymode(ENV_FILE, temporary_env)
        os.replace(temporary_env, ENV_FILE)
        switch(NEW)
        subprocess.run(["systemctl", "restart", SERVICE], check=True)
        check_service()
    except BaseException:
        if current.resolve() == NEW:
            switch(OLD)
        ENV_FILE.write_bytes(original_env)
        if current.resolve() == OLD:
            subprocess.run(["systemctl", "restart", SERVICE], check=True)
        raise
    print(f"granted room creation to user 89 via {NEW}; platform organizer list unchanged", flush=True)


if __name__ == "__main__":
    main()
