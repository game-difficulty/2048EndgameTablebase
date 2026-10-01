"""Deploy the 6x6 shape / 4x5 growing rules and scoped live renderer changes.

Run as root on the tournament host after staging the matching local builds.
Existing releases and the previous Live entry are retained for rollback.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time
from urllib.request import urlopen


ROOT = Path("/opt/2048tables")
STAGE = ROOT / "deploy-stage/competition-geometry-20260929-r1"
BACKEND_ROOT = ROOT / "tournament-app"
FRONTEND_ROOT = ROOT / "tournament"
OLD_BACKEND = BACKEND_ROOT / "releases/20260929-projects-11-12-r1"
OLD_FRONTEND = FRONTEND_ROOT / "releases/20260929-projects-11-12-r1"
NEW_BACKEND = BACKEND_ROOT / "releases/20260929-geometry-r1"
NEW_FRONTEND = FRONTEND_ROOT / "releases/20260929-geometry-r1"
LIVE_DIST = ROOT / "app/frontend/dist"
LIVE_HTML = LIVE_DIST / "live/index.html"
EXPECTED_LIVE_SHA256 = "02e455235831a972b89f4db2b75837cc31ce17445a7db1aae98e07b92ffed6bf"
BACKUP = Path("/var/lib/2048tables/backups/competition-geometry-20260929-r1")
DATABASE = Path("/var/lib/2048tables/competition-test/competition.sqlite3")
SERVICE = "2048tables-competition-test.service"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target(root: Path) -> Path:
    link = root / "current"
    if not link.is_symlink():
        raise RuntimeError(f"expected current symlink: {link}")
    return link.resolve(strict=True)


def switch(root: Path, destination: Path) -> None:
    temporary = root / "current.next-geometry-r1"
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary symlink already exists: {temporary}")
    temporary.symlink_to(destination)
    os.replace(temporary, root / "current")


def healthy() -> bool:
    for _ in range(25):
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
    if target(BACKEND_ROOT) != OLD_BACKEND or target(FRONTEND_ROOT) != OLD_FRONTEND:
        raise RuntimeError("current competition releases changed")
    if digest(LIVE_HTML) != EXPECTED_LIVE_SHA256:
        raise RuntimeError("Live entry changed since preflight")
    if NEW_BACKEND.exists() != NEW_FRONTEND.exists() or BACKUP.exists():
        raise RuntimeError("incomplete release pair or backup already exists")

    source_files = {
        "tournament_variants.py": "competition/backend/projects/tournament_variants.py",
        "projects_init.py": "competition/backend/projects/__init__.py",
        "practice_variants.py": "competition/backend/projects/practice_variants.py",
        "practice_leaderboard.py": "competition/backend/practice_leaderboard.py",
    }
    required = [*(STAGE / "backend" / name for name in source_files),
                STAGE / "competition-dist/index.html", STAGE / "live/index.html"]
    if any(not path.is_file() for path in required):
        raise RuntimeError("staged artifact missing")
    if not (STAGE / "live/assets").is_dir():
        raise RuntimeError("staged Live assets missing")
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as db:
        statuses = db.execute("SELECT status, count(*) FROM competitions GROUP BY status").fetchall()
    print("room statuses:", statuses, flush=True)
    if any(status not in {"SEATING", "FINISHED", "CANCELLED"} for status, _ in statuses):
        raise RuntimeError("active competition exists; aborting")

    if not NEW_BACKEND.exists():
        shutil.copytree(OLD_BACKEND, NEW_BACKEND)
        shutil.copytree(OLD_FRONTEND, NEW_FRONTEND)
    for staged_name, relative in source_files.items():
        installed = NEW_BACKEND / relative
        if not installed.is_file():
            raise RuntimeError(f"backend release file missing: {relative}")
        if digest(installed) != digest(STAGE / "backend" / staged_name):
            raise RuntimeError(f"backend copy mismatch: {relative}")
    if not (NEW_FRONTEND / "dist/index.html").is_file():
        raise RuntimeError("competition frontend release missing")
    if digest(NEW_FRONTEND / "dist/index.html") != digest(STAGE / "competition-dist/index.html"):
        raise RuntimeError("competition frontend copy mismatch")

    smoke = """
from competition.backend.projects import ProjectRegistry, TOURNAMENT_ADAPTER_FACTORIES, tournament_project_catalog
from competition.backend.practice_leaderboard import PROJECT_RULES
registry = ProjectRegistry()
for factory in TOURNAMENT_ADAPTER_FACTORIES:
    registry.register(factory)
catalog = tournament_project_catalog()
assert len(catalog) == 12
for entry in catalog:
    registry.resolve(entry['project_ref'], entry['rules_version'])
shape = next(entry for entry in catalog if entry['key'] == 'project-10')
growing = next(entry for entry in catalog if entry['key'] == 'project-12')
assert shape['rules_version'] == growing['rules_version'] == 'tournament-v3'
assert registry.resolve(shape['project_ref'], shape['rules_version']).public_payload(
    registry.resolve(shape['project_ref'], shape['rules_version']).initial_state(seed='31' * 32)
)['cols'] <= 6
assert registry.resolve(growing['project_ref'], growing['rules_version']).public_payload(
    registry.resolve(growing['project_ref'], growing['rules_version']).initial_state(seed='31' * 32)
)['cols'] == 5
assert PROJECT_RULES[shape['project_ref']][1] == PROJECT_RULES[growing['project_ref']][1] == 2
print('catalog and v3 adapters: OK')
"""
    env = dict(os.environ, PYTHONPATH=f"{NEW_BACKEND}:{ROOT / 'app'}")
    subprocess.run([str(ROOT / "venv/bin/python"), "-c", smoke], cwd=NEW_BACKEND,
                   env=env, check=True)

    # Content-hashed assets may be shared by other entry points; never overwrite them.
    asset_count = 0
    for source in (STAGE / "live/assets").iterdir():
        if not source.is_file():
            continue
        destination = LIVE_DIST / "assets" / source.name
        if destination.exists():
            if digest(source) != digest(destination):
                raise RuntimeError(f"Live asset hash collision: {source.name}")
        else:
            shutil.copy2(source, destination)
            asset_count += 1
    new_html = (STAGE / "live/index.html").read_text(encoding="utf-8")
    for asset in re.findall(r"/assets/([^\"']+)", new_html):
        if not (LIVE_DIST / "assets" / asset).is_file():
            raise RuntimeError(f"referenced Live asset missing: {asset}")

    BACKUP.mkdir(parents=True)
    shutil.copy2(LIVE_HTML, BACKUP / "live-index.html")
    try:
        switch(BACKEND_ROOT, NEW_BACKEND)
        switch(FRONTEND_ROOT, NEW_FRONTEND)
        subprocess.run(["systemctl", "restart", SERVICE], check=True)
        if not healthy():
            raise RuntimeError("competition health check failed")
        temporary = LIVE_DIST / "live/index.next-geometry-r1.html"
        if temporary.exists():
            raise RuntimeError(f"temporary Live entry already exists: {temporary}")
        shutil.copy2(STAGE / "live/index.html", temporary)
        os.replace(temporary, LIVE_HTML)
    except BaseException:
        if target(BACKEND_ROOT) == NEW_BACKEND:
            switch(BACKEND_ROOT, OLD_BACKEND)
        if target(FRONTEND_ROOT) == NEW_FRONTEND:
            switch(FRONTEND_ROOT, OLD_FRONTEND)
        if digest(LIVE_HTML) != EXPECTED_LIVE_SHA256:
            shutil.copy2(BACKUP / "live-index.html", LIVE_HTML)
        subprocess.run(["systemctl", "restart", SERVICE], check=False)
        raise
    print("deployed:", NEW_BACKEND, NEW_FRONTEND, flush=True)
    print("Live assets added:", asset_count, "entry SHA-256:", digest(LIVE_HTML), flush=True)


if __name__ == "__main__":
    main()
