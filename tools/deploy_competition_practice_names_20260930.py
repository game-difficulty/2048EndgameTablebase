"""Release final project names and practice UI/metric updates; keep all 20 projects.

Upload this file, the frontend dist/ and the five backend modules to STAGE,
then run it with sudo on the tournament host. Existing releases are retained.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time
from urllib.request import urlopen


ROOT = Path("/opt/2048tables")
LABEL = "20260930-practice-names-r1"
STAGE = Path("/home/ubuntu/competition-release-20260930-practice-names-r1")
FRONTEND = ROOT / "tournament"
BACKEND = ROOT / "tournament-app"
OLD_FRONTEND = FRONTEND / "releases/20260930-projects-20-r1"
OLD_BACKEND = BACKEND / "releases/20260930-projects-20-r1"
NEW_FRONTEND = FRONTEND / "releases" / LABEL
NEW_BACKEND = BACKEND / "releases" / LABEL
SERVICE = "2048tables-competition-test.service"
DATABASE = Path("/var/lib/2048tables/competition-test/competition.sqlite3")
BACKEND_FILES = (
    "competition/backend/practice_leaderboard.py",
    "competition/backend/projects/__init__.py",
    "competition/backend/projects/tournament_variants.py",
    "competition/backend/projects/client_variants.py",
    "competition/backend/projects/cargo_transport.py",
)


def current(root: Path) -> Path:
    link = root / "current"
    if not link.is_symlink():
        raise RuntimeError(f"not a release symlink: {link}")
    return link.resolve(strict=True)


def switch(root: Path, target: Path) -> None:
    temporary = root / f"current.next-{LABEL}"
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary symlink exists: {temporary}")
    temporary.symlink_to(target)
    os.replace(temporary, root / "current")


def check_health() -> bool:
    for _ in range(30):
        time.sleep(0.4)
        try:
            with urlopen("http://127.0.0.1:8770/api/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
    return False


def check_path(path: str) -> None:
    status = ""
    for _ in range(24):
        response = subprocess.run(
            ["curl", "-ksS", "--resolve", "tournament.2048tables.online:443:127.0.0.1",
             "-o", "/dev/null", "-w", "%{http_code}", f"https://tournament.2048tables.online{path}"],
            capture_output=True, text=True, check=True,
        )
        status = response.stdout
        if status == "200":
            return
        time.sleep(0.25)
    raise RuntimeError(f"{path}: HTTP {status}")


def main() -> None:
    if os.geteuid() != 0:
        raise RuntimeError("run with sudo")
    if current(FRONTEND) != OLD_FRONTEND or current(BACKEND) != OLD_BACKEND:
        raise RuntimeError("current release changed; refusing to replace it")
    if NEW_FRONTEND.exists() or NEW_BACKEND.exists():
        raise RuntimeError("release or backup already exists; refusing to overwrite")
    if not (STAGE / "dist/index.html").is_file() or any(not (STAGE / path).is_file() for path in BACKEND_FILES):
        raise RuntimeError("staged frontend or backend module missing")
    if len(list((STAGE / "dist/assets").iterdir())) < 20:
        raise RuntimeError("staged frontend assets appear incomplete")

    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as db:
        statuses = db.execute("SELECT status,count(*) FROM competitions GROUP BY status").fetchall()
        playing = db.execute("SELECT count(*) FROM competition_game_sessions WHERE state='playing'").fetchone()[0]
        running = db.execute("SELECT count(*) FROM competition_team_clocks WHERE running_since IS NOT NULL").fetchone()[0]
    print("room statuses:", statuses, "playing sessions:", playing, "running clocks:", running, flush=True)
    safe_statuses = {"SEATING", "GAME_A_READY", "GAME_B_READY", "GAME_C_READY", "FINISHED", "CANCELLED"}
    if playing or running or any(status not in safe_statuses for status, _ in statuses):
        raise RuntimeError("active game or deadline present; deployment postponed")

    shutil.copytree(OLD_FRONTEND, NEW_FRONTEND)
    shutil.copytree(STAGE / "dist", NEW_FRONTEND / "dist", dirs_exist_ok=True)
    # Windows SCP preserves restrictive source modes; Nginx needs traversal
    # and read access to every staged static asset.
    for directory, directories, files in os.walk(NEW_FRONTEND / "dist"):
        os.chmod(directory, 0o755)
        for name in directories:
            os.chmod(Path(directory) / name, 0o755)
        for name in files:
            os.chmod(Path(directory) / name, 0o644)
    shutil.copytree(OLD_BACKEND, NEW_BACKEND)
    for path in BACKEND_FILES:
        shutil.copy2(STAGE / path, NEW_BACKEND / path)
        os.chmod(NEW_BACKEND / path, 0o644)
    index = (NEW_FRONTEND / "dist/index.html").read_text()
    for asset in re.findall(r'["\'](/assets/[^"\']+)["\']', index):
        if not (NEW_FRONTEND / "dist" / asset.lstrip("/")).is_file():
            raise RuntimeError(f"missing built asset: {asset}")
    smoke = """
from competition.backend.projects import ProjectRegistry, TOURNAMENT_ADAPTER_FACTORIES, tournament_project_catalog
from competition.backend.practice_leaderboard import PROJECT_RULES
registry = ProjectRegistry()
for factory in TOURNAMENT_ADAPTER_FACTORIES:
    registry.register(factory)
catalog = tournament_project_catalog()
assert len(catalog) == 20, len(catalog)
for project in catalog:
    registry.descriptor(project['project_ref'], project['adapter_rules_version'])
assert PROJECT_RULES['practice-fission-4x4'] == ('board_sum', 2)
assert PROJECT_RULES['practice-look-back-3x4'] == ('time', 2)
assert catalog[2]['name'].startswith('寸步难行')
print('20 project descriptors, new names and metric versions: OK')
"""
    env = dict(os.environ, PYTHONPATH=f"{NEW_BACKEND}:{ROOT / 'app'}")
    subprocess.run([str(ROOT / "venv/bin/python"), "-c", smoke], cwd=NEW_BACKEND, env=env, check=True)

    names = {
        "tournament-cargo-transport-4x4": "真·华容道（4×4）",
        "tournament-evil-spawn-4x4": "寸步难行（4×4）",
        "tournament-pure2-full-race-3x3": "极限速通（3×3）",
        "tournament-mirror-64x10-race-4x4": "镜面领域（4×4）",
        "tournament-isolated-island-hard-4x4": "孤岛（4×4）",
    }
    with sqlite3.connect(DATABASE) as db:
        old_names = [row for row in db.execute("SELECT competition_id, project_key, name, adapter_snapshot_json, project_ref FROM competition_projects") if row[4] in names]
        backup_path = DATABASE.with_name(DATABASE.name + f".before-{LABEL}")
        if backup_path.exists():
            raise RuntimeError("database backup already exists")
        with sqlite3.connect(backup_path) as backup:
            db.backup(backup)
    names_changed = False
    switched_frontend = False
    switched_backend = False
    try:
        with sqlite3.connect(DATABASE) as db:
            for competition_id, project_key, name, raw, ref in old_names:
                snapshot = json.loads(raw)
                snapshot["display_name"] = names[ref]
                db.execute("UPDATE competition_projects SET name=?, adapter_snapshot_json=? WHERE competition_id=? AND project_key=?", (names[ref], json.dumps(snapshot, ensure_ascii=False), competition_id, project_key))
        names_changed = True
        switch(FRONTEND, NEW_FRONTEND)
        switched_frontend = True
        switch(BACKEND, NEW_BACKEND)
        switched_backend = True
        subprocess.run(["systemctl", "restart", SERVICE], check=True)
        if not check_health():
            raise RuntimeError("competition service failed health check")
        for path in ("/practice", "/practice/19", "/practice/20", "/test",
                     "/api/practice/practice-aftershock-4x4/leaderboard",
                     "/api/practice/practice-look-back-3x4/leaderboard",
                     "/api/practice/practice-fission-4x4/leaderboard", "/api/health"):
            check_path(path)
    except BaseException:
        if names_changed:
            with sqlite3.connect(DATABASE) as db:
                for competition_id, project_key, name, raw, ref in old_names:
                    db.execute("UPDATE competition_projects SET name=?, adapter_snapshot_json=? WHERE competition_id=? AND project_key=?", (name, raw, competition_id, project_key))
        if switched_frontend and current(FRONTEND) == NEW_FRONTEND:
            switch(FRONTEND, OLD_FRONTEND)
        if switched_backend and current(BACKEND) == NEW_BACKEND:
            switch(BACKEND, OLD_BACKEND)
        if switched_backend:
            subprocess.run(["systemctl", "restart", SERVICE], check=False)
        raise
    print("deployed:", NEW_FRONTEND, NEW_BACKEND, flush=True)


if __name__ == "__main__":
    main()
