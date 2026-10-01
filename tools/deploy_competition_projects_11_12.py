"""One-off scoped release for competition projects 11 and 12.

Run on the tournament host as root after uploading the four staged artifacts.
The prior releases remain intact; a failed service health check restores both
current symlinks and restarts the prior backend.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import hashlib
import sqlite3
import subprocess
import time
from urllib.request import urlopen


ROOT = Path("/opt/2048tables")
STAGE = ROOT / "deploy-stage/competition-projects-11-12-20260929-r1"
BACKEND_ROOT = ROOT / "tournament-app"
FRONTEND_ROOT = ROOT / "tournament"
OLD_BACKEND = BACKEND_ROOT / "releases/20260929-competition-live-avatars-icons-r1"
OLD_FRONTEND = FRONTEND_ROOT / "releases/20260929-practice-appearance-c0fa4cc"
NEW_BACKEND = BACKEND_ROOT / "releases/20260929-projects-11-12-r1"
NEW_FRONTEND = FRONTEND_ROOT / "releases/20260929-projects-11-12-r1"
DATABASE = Path("/var/lib/2048tables/competition-test/competition.sqlite3")
SERVICE = "2048tables-competition-test.service"


def current_target(root: Path) -> Path:
    link = root / "current"
    if not link.is_symlink():
        raise RuntimeError(f"not a symlink: {link}")
    return link.resolve(strict=True)


def switch(root: Path, target: Path) -> None:
    temporary = root / "current.next-projects-11-12-r1"
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary symlink already exists: {temporary}")
    temporary.symlink_to(target)
    os.replace(temporary, root / "current")


def service_healthy() -> bool:
    for _ in range(25):
        time.sleep(0.4)
        try:
            with urlopen("http://127.0.0.1:8770/api/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
    return False


def same_contents(left: Path, right: Path) -> bool:
    def digest(path: Path) -> bytes:
        return hashlib.sha256(path.read_bytes()).digest()
    return left.is_file() and right.is_file() and digest(left) == digest(right)


def main() -> None:
    if os.geteuid() != 0:
        raise RuntimeError("run with sudo")
    if current_target(BACKEND_ROOT) != OLD_BACKEND:
        raise RuntimeError("backend current release changed; refusing to overwrite")
    if current_target(FRONTEND_ROOT) != OLD_FRONTEND:
        raise RuntimeError("frontend current release changed; refusing to overwrite")
    if NEW_BACKEND.exists() != NEW_FRONTEND.exists():
        raise RuntimeError("only one release directory exists; aborting")
    for source in (
        STAGE / "tournament_variants.py",
        STAGE / "projects_init.py",
        STAGE / "practice_variants.py",
        STAGE / "dist/index.html",
    ):
        if not source.is_file():
            raise RuntimeError(f"missing staged artifact: {source}")
    if len(list((STAGE / "dist").rglob("*.*"))) < 30:
        raise RuntimeError("staged frontend build is incomplete")

    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as db:
        statuses = db.execute("SELECT status, count(*) FROM competitions GROUP BY status").fetchall()
    print("room statuses:", statuses, flush=True)
    if any(status not in {"SEATING", "FINISHED"} for status, _ in statuses):
        raise RuntimeError("an active competition may be interrupted; aborting")

    if not NEW_BACKEND.exists():
        shutil.copytree(OLD_BACKEND, NEW_BACKEND)
        project_dir = NEW_BACKEND / "competition/backend/projects"
        shutil.copy2(STAGE / "tournament_variants.py", project_dir / "tournament_variants.py")
        shutil.copy2(STAGE / "projects_init.py", project_dir / "__init__.py")
        shutil.copy2(STAGE / "practice_variants.py", project_dir / "practice_variants.py")
        shutil.copytree(OLD_FRONTEND, NEW_FRONTEND)
        shutil.copytree(STAGE / "dist", NEW_FRONTEND / "dist", dirs_exist_ok=True)

    project_dir = NEW_BACKEND / "competition/backend/projects"
    for staged, installed in (
        (STAGE / "tournament_variants.py", project_dir / "tournament_variants.py"),
        (STAGE / "projects_init.py", project_dir / "__init__.py"),
        (STAGE / "practice_variants.py", project_dir / "practice_variants.py"),
        (STAGE / "dist/index.html", NEW_FRONTEND / "dist/index.html"),
    ):
        if not same_contents(staged, installed):
            raise RuntimeError(f"prepared artifact mismatch: {installed}")

    smoke = """
from competition.backend.projects import ProjectRegistry, TOURNAMENT_ADAPTER_FACTORIES, tournament_project_catalog
catalog = tournament_project_catalog()
assert len(catalog) == 12, len(catalog)
registry = ProjectRegistry()
for factory in TOURNAMENT_ADAPTER_FACTORIES:
    registry.register(factory)
for entry in catalog:
    registry.resolve(entry['project_ref'], entry['adapter_rules_version'])
for key in ('project-11', 'project-12'):
    entry = next(item for item in catalog if item['key'] == key)
    adapter = registry.resolve(entry['project_ref'], entry['adapter_rules_version'])
    state = adapter.initial_state(seed='00' * 32)
    assert adapter.public_view(state).view_kind
print('registered projects:', len(catalog), 'new adapters: OK')
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{NEW_BACKEND}:{ROOT / 'app'}"
    subprocess.run(
        [str(ROOT / "venv/bin/python"), "-c", smoke],
        cwd=NEW_BACKEND, env=env, check=True,
    )

    try:
        switch(BACKEND_ROOT, NEW_BACKEND)
        switch(FRONTEND_ROOT, NEW_FRONTEND)
        subprocess.run(["systemctl", "restart", SERVICE], check=True)
        if not service_healthy():
            raise RuntimeError("new competition service failed health check")
    except BaseException:
        if current_target(BACKEND_ROOT) == NEW_BACKEND:
            switch(BACKEND_ROOT, OLD_BACKEND)
        if current_target(FRONTEND_ROOT) == NEW_FRONTEND:
            switch(FRONTEND_ROOT, OLD_FRONTEND)
        subprocess.run(["systemctl", "restart", SERVICE], check=False)
        raise
    print("deployed:", NEW_BACKEND, NEW_FRONTEND, flush=True)


if __name__ == "__main__":
    main()
