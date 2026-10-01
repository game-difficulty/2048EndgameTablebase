"""Scoped release for practice projects 13–15 on tournament.2048tables.online.

Run as root on the tournament host after uploading dist/ and
practice_leaderboard.py into STAGE. Old releases and Nginx configuration are
kept for rollback. No formal competition project adapters are changed.
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
LABEL = "20260929-special-practice-r3"
STAGE = ROOT / "deploy-stage" / LABEL
FRONTEND_ROOT = ROOT / "tournament"
BACKEND_ROOT = ROOT / "tournament-app"
OLD_FRONTEND = FRONTEND_ROOT / "releases/20260929-geometry-r1"
OLD_BACKEND = BACKEND_ROOT / "releases/20260929-evilgen-seeded-r1"
NEW_FRONTEND = FRONTEND_ROOT / "releases" / LABEL
NEW_BACKEND = BACKEND_ROOT / "releases" / LABEL
NGINX = Path("/etc/nginx/sites-available/tournament.2048tables.online")
NGINX_BACKUP = NGINX.with_name(NGINX.name + f".before-{LABEL}")
OLD_ROUTE = "location ~ ^/practice/(?:[1-9]|1[0-2])/?$"
NEW_ROUTE = "location ~ ^/practice/(?:[1-9]|1[0-5])/?$"
SERVICE = "2048tables-competition-test.service"
DATABASE = Path("/var/lib/2048tables/competition-test/competition.sqlite3")


def current(root: Path) -> Path:
    link = root / "current"
    if not link.is_symlink():
        raise RuntimeError(f"not a release symlink: {link}")
    return link.resolve(strict=True)


def switch(root: Path, target: Path) -> None:
    temporary = root / f"current.next-{LABEL}"
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary symlink already exists: {temporary}")
    temporary.symlink_to(target)
    os.replace(temporary, root / "current")


def healthy() -> bool:
    for _ in range(30):
        time.sleep(0.4)
        try:
            with urlopen("http://127.0.0.1:8770/api/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
    return False


def main() -> None:
    if os.geteuid() != 0:
        raise RuntimeError("run with sudo")
    if current(FRONTEND_ROOT) != OLD_FRONTEND or current(BACKEND_ROOT) != OLD_BACKEND:
        raise RuntimeError("current releases differ from preflight; refusing to replace them")
    if NEW_FRONTEND.exists() or NEW_BACKEND.exists() or NGINX_BACKUP.exists():
        raise RuntimeError("release or backup already exists; refusing to overwrite")
    if not (STAGE / "dist/index.html").is_file() or not (STAGE / "practice_leaderboard.py").is_file():
        raise RuntimeError("staged frontend or leaderboard module is missing")
    if len(list((STAGE / "dist/assets").iterdir())) < 20:
        raise RuntimeError("staged frontend assets appear incomplete")
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as db:
        statuses = db.execute("SELECT status, count(*) FROM competitions GROUP BY status").fetchall()
    print("room statuses:", statuses, flush=True)
    if any(status not in {"SEATING", "FINISHED", "CANCELLED"} for status, _ in statuses):
        raise RuntimeError("active competition present; deployment postponed")

    nginx_original = NGINX.read_text()
    if nginx_original.count(OLD_ROUTE) != 1 or NEW_ROUTE in nginx_original:
        raise RuntimeError("Nginx practice route differs from expected version")
    shutil.copytree(OLD_FRONTEND, NEW_FRONTEND)
    shutil.copytree(STAGE / "dist", NEW_FRONTEND / "dist", dirs_exist_ok=True)
    shutil.copytree(OLD_BACKEND, NEW_BACKEND)
    shutil.copy2(STAGE / "practice_leaderboard.py", NEW_BACKEND / "competition/backend/practice_leaderboard.py")
    shutil.copy2(NGINX, NGINX_BACKUP)

    index = (NEW_FRONTEND / "dist/index.html").read_text()
    for asset in re.findall(r'["\'](/assets/[^"\']+)["\']', index):
        if not (NEW_FRONTEND / "dist" / asset.lstrip("/")).is_file():
            raise RuntimeError(f"built HTML references missing asset: {asset}")
    smoke = """
from competition.backend.practice_leaderboard import PROJECT_RULES
for project in ('practice-pair-bond-4x4', 'practice-chemical-reaction-4x4', 'practice-timed-bomb-4x4'):
    assert PROJECT_RULES[project] == ('score', 1), project
print('practice leaderboard projects: OK')
"""
    env = dict(os.environ, PYTHONPATH=f"{NEW_BACKEND}:{ROOT / 'app'}")
    subprocess.run([str(ROOT / "venv/bin/python"), "-c", smoke], cwd=NEW_BACKEND, env=env, check=True)

    switched_frontend = False
    switched_backend = False
    nginx_changed = False
    try:
        switch(FRONTEND_ROOT, NEW_FRONTEND)
        switched_frontend = True
        switch(BACKEND_ROOT, NEW_BACKEND)
        switched_backend = True
        subprocess.run(["systemctl", "restart", SERVICE], check=True)
        if not healthy():
            raise RuntimeError("competition service failed health check")
        nginx_changed = True
        NGINX.write_text(nginx_original.replace(OLD_ROUTE, NEW_ROUTE))
        subprocess.run(["nginx", "-t"], check=True)
        subprocess.run(["systemctl", "reload", "nginx"], check=True)
        for path in ("/practice/13", "/practice/14", "/practice/15"):
            status = ""
            for _ in range(20):
                response = subprocess.run(
                    ["curl", "-ksS", "--resolve", "tournament.2048tables.online:443:127.0.0.1",
                     "-o", "/dev/null", "-w", "%{http_code}",
                     f"https://tournament.2048tables.online{path}"],
                    capture_output=True, text=True, check=True,
                )
                status = response.stdout
                if status == "200":
                    break
                time.sleep(0.25)
            if status != "200":
                raise RuntimeError(f"{path}: HTTP {status}")
    except BaseException:
        if nginx_changed:
            shutil.copy2(NGINX_BACKUP, NGINX)
            subprocess.run(["nginx", "-t"], check=False)
            subprocess.run(["systemctl", "reload", "nginx"], check=False)
        if switched_frontend or switched_backend:
            if switched_frontend and current(FRONTEND_ROOT) == NEW_FRONTEND:
                switch(FRONTEND_ROOT, OLD_FRONTEND)
            if switched_backend and current(BACKEND_ROOT) == NEW_BACKEND:
                switch(BACKEND_ROOT, OLD_BACKEND)
            subprocess.run(["systemctl", "restart", SERVICE], check=False)
        raise
    print("deployed:", NEW_FRONTEND, NEW_BACKEND, flush=True)


if __name__ == "__main__":
    main()
