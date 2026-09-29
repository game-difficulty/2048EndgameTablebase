"""Pinpoint release on top of the published tournament assets.

The working tree also contains an unfinished match runtime. This script only
patches the SHA-pinned published bundle and the practice leaderboard versions;
it intentionally does not build or publish the rest of that working tree.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time


ROOT = Path("/opt/2048tables")
FRONTEND = ROOT / "tournament"
BACKEND = ROOT / "tournament-app"
OLD_FRONTEND = FRONTEND / "releases/20260929-bomb-countdown-12-32-r1"
OLD_BACKEND = BACKEND / "releases/20260929-special-practice-r3"
NEW_FRONTEND = FRONTEND / "releases/20260929-zh-special-r1"
NEW_BACKEND = BACKEND / "releases/20260929-zh-special-r1"
OLD_ASSET = "index-fHLqNpnp.js"
OLD_SHA256 = "e4342dd34a5d8a4e5386338196bfd46acaa132b39a5da00092f05880a4d8565b"
SERVICE = "2048tables-competition-test.service"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def patched_bundle(bundle: str, errors_source: str) -> str:
    mapping = re.search(r"export const ERROR_MESSAGES = Object\.freeze\((\{.*?\})\);", errors_source, re.S)
    function = re.search(r"export function userFacingError\(cause\) \{.*?\n\}", errors_source, re.S)
    if not mapping or not function:
        raise RuntimeError("cannot extract Chinese error formatter")
    formatter = function.group().removeprefix("export ")
    replacement = f"const COMPETITION_ZH_ERRORS=Object.freeze({mapping.group(1)});"
    formatter = formatter.replace("ERROR_MESSAGES", "COMPETITION_ZH_ERRORS")
    replacement += formatter + "function Ge(e){d.value=userFacingError(e)}"
    bundle = replace_once(bundle,
        "function Ge(e){d.value=e instanceof Error?e.message:String(e||`发生未知错误`)}",
        replacement, "frontend error formatter")
    bundle = replace_once(bundle, "Ge(n.error?.message||`房间连接失败`)",
                          "Ge(n.error||`房间连接失败`)", "socket errors")
    bundle = replace_once(bundle, "异色相邻变墙", "异色相撞合成一格墙", "chemical description")
    bundle = replace_once(bundle,
        "e.kind===`bomb`&&t.kind===`bomb`||e.kind===t.kind&&(e.kind===`chemical-a`||e.kind===`chemical-b`)",
        "e.kind===`bomb`&&t.kind===`bomb`||(e.kind===`chemical-a`||e.kind===`chemical-b`)&&(t.kind===`chemical-a`||t.kind===`chemical-b`)",
        "chemical merge eligibility")
    bundle = replace_once(bundle, "else c.push(g.id,e.id);p=d=!0;break}",
        "else if(e.kind!==g.kind){let t={id:_,kind:`wall`,value:-1,cells:g.cells.slice(),merged:!0};l(t),s.push({tile:yc(t),sources:[g.id,e.id]})}else c.push(g.id,e.id);p=d=!0;break}",
        "chemical collision result")
    bundle = replace_once(bundle, "nextTicket(){return this.randomState=Vs(this.randomState),this.randomState}spawn(){",
        "nextTicket(){return this.randomState=Vs(this.randomState),this.randomState}"
        "specialSpawnChance(){let e=this.tiles.filter(e=>this.project.specialRule===`pair`?e.kind===`pair-single`:this.project.specialRule===`chemical`?e.kind===`chemical-a`||e.kind===`chemical-b`:e.kind===`bomb`).length;return Math.max(0,Number(this.project.specialSpawnRate??.05)-e*.02)}spawn(){",
        "dynamic special spawn chance")
    bundle = replace_once(bundle,
        "Us(e,`special`)<Number(this.project.specialSpawnRate??.05)",
        "Us(e,`special`)<this.specialSpawnChance()", "special spawn comparison")
    chemical_start = "}else if(this.project.specialRule===`chemical`){"
    chemical_end = "}}reset(e=!0)"
    start = bundle.find(chemical_start)
    end = bundle.find(chemical_end, start)
    if start < 0 or end < 0 or bundle.count(chemical_start) != 1:
        raise RuntimeError("cannot isolate old chemical adjacency reaction")
    bundle = bundle[:start] + "}}reset(e=!0)" + bundle[end + len(chemical_end):]
    if "异色相邻变墙" in bundle or chemical_start in bundle:
        raise RuntimeError("old chemical behavior remains")
    return bundle


def checked_current(root: Path, expected: Path) -> None:
    link = root / "current"
    if not link.is_symlink() or link.resolve(strict=True) != expected:
        raise RuntimeError(f"release changed: {link}")


def check_js_syntax(bundle: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mjs", encoding="utf-8", delete=False) as temporary:
        temporary.write(bundle)
        path = Path(temporary.name)
    try:
        subprocess.run(["node", "--check", str(path)], check=True)
    finally:
        path.unlink(missing_ok=True)


def switch(root: Path, target: Path) -> None:
    temporary = root / "current.next-zh-special-r1"
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary symlink exists: {temporary}")
    temporary.symlink_to(target)
    os.replace(temporary, root / "current")


def active_room_check() -> None:
    with sqlite3.connect("file:/var/lib/2048tables/competition-test/competition.sqlite3?mode=ro", uri=True) as db:
        statuses = db.execute("SELECT status, count(*) FROM competitions GROUP BY status").fetchall()
    print("room statuses:", statuses, flush=True)
    if any(status not in {"SEATING", "FINISHED", "CANCELLED"} for status, _ in statuses):
        raise RuntimeError("active competition present; deployment postponed")


def check_http(asset: str) -> None:
    for _ in range(20):
        page = subprocess.run(["curl", "-ksS", "--resolve", "tournament.2048tables.online:443:127.0.0.1",
                               "https://tournament.2048tables.online/test"],
                              capture_output=True, text=True, check=True)
        health = subprocess.run(["curl", "-ksS", "--resolve", "tournament.2048tables.online:443:127.0.0.1",
                                 "https://tournament.2048tables.online/api/health"],
                                capture_output=True, text=True, check=True)
        if asset in page.stdout and '"ok":true' in health.stdout:
            return
        time.sleep(0.5)
    raise RuntimeError("frontend or backend health check failed")


def deploy(errors_path: Path) -> None:
    if os.geteuid() != 0:
        raise RuntimeError("run as root on the tournament server")
    checked_current(FRONTEND, OLD_FRONTEND)
    checked_current(BACKEND, OLD_BACKEND)
    if NEW_FRONTEND.exists() or NEW_BACKEND.exists():
        raise RuntimeError("target release already exists")
    active_room_check()
    original = (OLD_FRONTEND / "dist/assets" / OLD_ASSET).read_bytes()
    if hashlib.sha256(original).hexdigest() != OLD_SHA256:
        raise RuntimeError("published JS digest changed")
    patched = patched_bundle(original.decode("utf-8"), errors_path.read_text(encoding="utf-8")).encode("utf-8")
    check_js_syntax(patched.decode("utf-8"))
    digest = hashlib.sha256(patched).hexdigest()
    asset = f"index-zh-special-{digest[:12]}.js"
    shutil.copytree(OLD_FRONTEND, NEW_FRONTEND)
    shutil.copytree(OLD_BACKEND, NEW_BACKEND)
    (NEW_FRONTEND / "dist/assets" / asset).write_bytes(patched)
    page = NEW_FRONTEND / "dist/index.html"
    page.write_text(replace_once(page.read_text(encoding="utf-8"), OLD_ASSET, asset, "index asset"), encoding="utf-8")
    leaderboard = NEW_BACKEND / "competition/backend/practice_leaderboard.py"
    contents = leaderboard.read_text(encoding="utf-8")
    for project in ("practice-pair-bond-4x4", "practice-chemical-reaction-4x4", "practice-timed-bomb-4x4"):
        contents = replace_once(contents, f'"{project}": ("score", 1)', f'"{project}": ("score", 2)', project)
    leaderboard.write_text(contents, encoding="utf-8")
    subprocess.run(["/opt/2048tables/venv/bin/python", "-m", "py_compile", str(leaderboard)], check=True)
    try:
        switch(FRONTEND, NEW_FRONTEND)
        switch(BACKEND, NEW_BACKEND)
        subprocess.run(["systemctl", "restart", SERVICE], check=True)
        check_http(asset)
        subprocess.run(["systemctl", "is-active", "--quiet", SERVICE], check=True)
    except BaseException:
        if (FRONTEND / "current").resolve() == NEW_FRONTEND:
            switch(FRONTEND, OLD_FRONTEND)
        if (BACKEND / "current").resolve() == NEW_BACKEND:
            switch(BACKEND, OLD_BACKEND)
            subprocess.run(["systemctl", "restart", SERVICE], check=True)
        raise
    print(f"deployed {NEW_FRONTEND} and {NEW_BACKEND}; JS {asset}; sha256 {digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        original = (OLD_FRONTEND / "dist/assets" / OLD_ASSET).read_bytes()
        if hashlib.sha256(original).hexdigest() != OLD_SHA256:
            raise RuntimeError("published JS digest changed")
        result = patched_bundle(original.decode("utf-8"), args.errors.read_text(encoding="utf-8"))
        check_js_syntax(result)
        print("patched JS sha256:", hashlib.sha256(result.encode("utf-8")).hexdigest())
    else:
        deploy(args.errors)
