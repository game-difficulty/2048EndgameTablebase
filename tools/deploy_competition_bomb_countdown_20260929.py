"""Frontend-only tournament practice release: bomb countdown 12–32."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time


ROOT = Path("/opt/2048tables")
FRONTEND = ROOT / "tournament"
BACKEND = ROOT / "tournament-app"
OLD = FRONTEND / "releases/20260929-special-practice-r3"
NEW = FRONTEND / "releases/20260929-bomb-countdown-12-32-r1"
STAGE = ROOT / "deploy-stage/20260929-bomb-countdown-12-32-r1/dist"


def current(root: Path) -> Path:
    link = root / "current"
    if not link.is_symlink():
        raise RuntimeError(f"not a release symlink: {link}")
    return link.resolve(strict=True)


def switch(target: Path) -> None:
    temporary = FRONTEND / "current.next-bomb-countdown-12-32-r1"
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary symlink exists: {temporary}")
    temporary.symlink_to(target)
    os.replace(temporary, FRONTEND / "current")


def check_frontend() -> None:
    for _ in range(20):
        result = subprocess.run(
            ["curl", "-ksS", "--resolve", "tournament.2048tables.online:443:127.0.0.1",
             "https://tournament.2048tables.online/practice/15"],
            capture_output=True, text=True, check=True,
        )
        if 'index-fHLqNpnp.js' in result.stdout:
            return
        time.sleep(0.25)
    raise RuntimeError("practice page did not serve the new build")


def main() -> None:
    if os.geteuid() != 0:
        raise RuntimeError("run with sudo")
    if current(FRONTEND) != OLD or current(BACKEND) != BACKEND / "releases/20260929-special-practice-r3":
        raise RuntimeError("current releases changed; refusing to deploy")
    if NEW.exists() or not (STAGE / "index.html").is_file():
        raise RuntimeError("new release exists or staged build is missing")
    if "index-fHLqNpnp.js" not in (STAGE / "index.html").read_text():
        raise RuntimeError("unexpected staged frontend build")
    with sqlite3.connect("file:/var/lib/2048tables/competition-test/competition.sqlite3?mode=ro", uri=True) as db:
        statuses = db.execute("SELECT status, count(*) FROM competitions GROUP BY status").fetchall()
    print("room statuses:", statuses, flush=True)
    if any(status not in {"SEATING", "FINISHED", "CANCELLED"} for status, _ in statuses):
        raise RuntimeError("active competition present; deployment postponed")

    shutil.copytree(OLD, NEW)
    shutil.copytree(STAGE, NEW / "dist", dirs_exist_ok=True)
    if not (NEW / "dist/assets/index-fHLqNpnp.js").is_file():
        raise RuntimeError("new JS asset missing")
    try:
        switch(NEW)
        check_frontend()
    except BaseException:
        if current(FRONTEND) == NEW:
            switch(OLD)
        raise
    print("deployed:", NEW, flush=True)


if __name__ == "__main__":
    main()
