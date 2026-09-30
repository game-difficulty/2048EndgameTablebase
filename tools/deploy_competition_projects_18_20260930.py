"""Release practice pages 16–18 and the selectable 18-project match catalog.

Upload this file, the frontend dist/ and the five backend modules to STAGE,
then run it with sudo on the tournament host. Existing releases are retained.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time
from urllib.request import urlopen


ROOT = Path("/opt/2048tables")
LABEL = "20260930-projects-18-r3"
STAGE = Path("/home/ubuntu/competition-release-20260930-projects-18-r1")
FRONTEND = ROOT / "tournament"
BACKEND = ROOT / "tournament-app"
OLD_FRONTEND = FRONTEND / "releases/20260930-events-schedule-r1"
OLD_BACKEND = BACKEND / "releases/20260930-events-schedule-r1"
NEW_FRONTEND = FRONTEND / "releases" / LABEL
NEW_BACKEND = BACKEND / "releases" / LABEL
NGINX = Path("/etc/nginx/sites-available/tournament.2048tables.online")
NGINX_BACKUP = NGINX.with_name(NGINX.name + f".before-{LABEL}")
OLD_ROUTE = "location ~ ^/practice/(?:[1-9]|1[0-5])/?$"
NEW_ROUTE = "location ~ ^/practice/(?:[1-9]|1[0-8])/?$"
OLD_OUTCOMES = '"target_reached", "no_moves", "time_limit", "opponent_finished", "surrendered"'
NEW_OUTCOMES = '"target_reached", "no_moves", "tile_limit", "time_limit", "opponent_finished", "surrendered"'
SERVICE = "2048tables-competition-test.service"
DATABASE = Path("/var/lib/2048tables/competition-test/competition.sqlite3")
BACKEND_FILES = (
    "competition/backend/practice_leaderboard.py",
    "competition/backend/projects/__init__.py",
    "competition/backend/projects/tournament_variants.py",
    "competition/backend/projects/client_variants.py",
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
    if NEW_FRONTEND.exists() or NEW_BACKEND.exists() or NGINX_BACKUP.exists():
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

    nginx_original = NGINX.read_text()
    if nginx_original.count(OLD_ROUTE) != 1 or NEW_ROUTE in nginx_original:
        raise RuntimeError("Nginx practice route differs from expected version")
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
    service_file = NEW_BACKEND / "competition/backend/service.py"
    service_source = service_file.read_text()
    if service_source.count(OLD_OUTCOMES) != 1:
        raise RuntimeError("server outcome validation changed; refusing to patch")
    service_file.write_text(service_source.replace(OLD_OUTCOMES, NEW_OUTCOMES))
    shutil.copy2(NGINX, NGINX_BACKUP)

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
assert len(catalog) == 18, len(catalog)
for project in catalog:
    registry.descriptor(project['project_ref'], project['adapter_rules_version'])
for project in catalog[-6:]:
    assert PROJECT_RULES[project['project_ref']][0] == 'score'
print('18 project descriptors and 6 new leaderboards: OK')
"""
    env = dict(os.environ, PYTHONPATH=f"{NEW_BACKEND}:{ROOT / 'app'}")
    subprocess.run([str(ROOT / "venv/bin/python"), "-c", smoke], cwd=NEW_BACKEND, env=env, check=True)

    switched_frontend = False
    switched_backend = False
    nginx_changed = False
    try:
        switch(FRONTEND, NEW_FRONTEND)
        switched_frontend = True
        switch(BACKEND, NEW_BACKEND)
        switched_backend = True
        subprocess.run(["systemctl", "restart", SERVICE], check=True)
        if not check_health():
            raise RuntimeError("competition service failed health check")
        nginx_changed = True
        NGINX.write_text(nginx_original.replace(OLD_ROUTE, NEW_ROUTE))
        subprocess.run(["nginx", "-t"], check=True)
        subprocess.run(["systemctl", "reload", "nginx"], check=True)
        for path in ("/practice", "/practice/16", "/practice/17", "/practice/18", "/test",
                     "/api/practice/practice-full-load-4x4/leaderboard",
                     "/api/practice/practice-heavy-tiles-4x4/leaderboard",
                     "/api/practice/practice-fission-4x4/leaderboard", "/api/health"):
            check_path(path)
    except BaseException:
        if nginx_changed:
            shutil.copy2(NGINX_BACKUP, NGINX)
            subprocess.run(["nginx", "-t"], check=False)
            subprocess.run(["systemctl", "reload", "nginx"], check=False)
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
