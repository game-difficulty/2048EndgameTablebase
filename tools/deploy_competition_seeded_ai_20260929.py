"""Install seeded EvilGen only for the competition backend, with rollback.

Run on the tournament host as root after building ai_core in the staged
native_core tree. The main site's native extension remains untouched.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from urllib.request import urlopen


ROOT = Path("/opt/2048tables")
STAGE = ROOT / "deploy-stage/evilgen-seeded-20260929-r1/native_core"
BACKEND_ROOT = ROOT / "tournament-app"
OLD_BACKEND = BACKEND_ROOT / "releases/20260929-geometry-r1"
NEW_BACKEND = BACKEND_ROOT / "releases/20260929-evilgen-seeded-r1"
DATABASE = Path("/var/lib/2048tables/competition-test/competition.sqlite3")
SERVICE = "2048tables-competition-test.service"
MODULE_NAME = "ai_core.cpython-310-x86_64-linux-gnu.so"
EXPECTED_MODULE_SHA256 = "387c5695f50f0a1c692a04292e425c4031de84fee60dd120f9fb3be7eba78d77"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_target() -> Path:
    link = BACKEND_ROOT / "current"
    if not link.is_symlink():
        raise RuntimeError("competition current is not a symlink")
    return link.resolve(strict=True)


def switch(destination: Path) -> None:
    temporary = BACKEND_ROOT / "current.next-evilgen-r1"
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary symlink already exists: {temporary}")
    temporary.symlink_to(destination)
    os.replace(temporary, BACKEND_ROOT / "current")


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
    if current_target() != OLD_BACKEND:
        raise RuntimeError("competition backend release changed")
    source = STAGE / MODULE_NAME
    if not source.is_file() or digest(source) != EXPECTED_MODULE_SHA256:
        raise RuntimeError("compiled seeded ai_core does not match verified build")
    if NEW_BACKEND.exists():
        raise RuntimeError("new backend release already exists")
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as db:
        statuses = db.execute("SELECT status, count(*) FROM competitions GROUP BY status").fetchall()
    print("room statuses:", statuses, flush=True)
    if any(status not in {"SEATING", "FINISHED", "CANCELLED"} for status, _ in statuses):
        raise RuntimeError("active competition exists; aborting")

    shutil.copytree(OLD_BACKEND, NEW_BACKEND)
    package = NEW_BACKEND / "native_core"
    package.mkdir()
    shutil.copy2(STAGE / "__init__.py", package / "__init__.py")
    shutil.copy2(source, package / MODULE_NAME)
    if digest(package / MODULE_NAME) != EXPECTED_MODULE_SHA256:
        raise RuntimeError("installed ai_core differs from verified build")

    smoke = """
from pathlib import Path
import native_core.ai_core as native
from competition.backend.projects import ProjectRegistry, TOURNAMENT_ADAPTER_FACTORIES
from competition.backend.projects.tournament_variants import TOURNAMENT_RULES
root = Path(__import__('os').environ['PYTHONPATH'].split(':')[0]).resolve()
assert Path(native.__file__).resolve().is_relative_to(root)
assert hasattr(native.EvilGen, 'gen_new_num_seeded')
board = 0x123456789abcde0f
assert tuple(native.EvilGen(board).gen_new_num_seeded(5, 1)[1:]) == (14, 2)
assert tuple(native.EvilGen(board).gen_new_num_seeded(5, 2)[1:]) == (14, 1)
assert tuple(native.EvilGen(board).gen_new_num(5)[1:]) == (14, 1)
registry = ProjectRegistry()
for factory in TOURNAMENT_ADAPTER_FACTORIES:
    registry.register(factory)
adapter = registry.resolve('tournament-evil-spawn-4x4', 'tournament-v2')
yellow = adapter.initial_state_for_side(seed='31' * 32, side='yellow')
white = adapter.initial_state_for_side(seed='31' * 32, side='white')
assert yellow.board == white.board
assert yellow.rng_counter == white.rng_counter == 2
assert adapter.public_view(yellow).view_kind == '2048-board'
print('seeded native module and competition adapter: OK')
"""
    env = dict(os.environ, PYTHONPATH=f"{NEW_BACKEND}:{ROOT / 'app'}")
    subprocess.run([str(ROOT / "venv/bin/python"), "-c", smoke], cwd=NEW_BACKEND,
                   env=env, check=True)

    try:
        switch(NEW_BACKEND)
        subprocess.run(["systemctl", "restart", SERVICE], check=True)
        if not healthy():
            raise RuntimeError("competition health check failed")
    except BaseException:
        if current_target() == NEW_BACKEND:
            switch(OLD_BACKEND)
        subprocess.run(["systemctl", "restart", SERVICE], check=False)
        raise
    print("deployed:", NEW_BACKEND, flush=True)
    print("competition ai_core SHA-256:", digest(package / MODULE_NAME), flush=True)


if __name__ == "__main__":
    main()
