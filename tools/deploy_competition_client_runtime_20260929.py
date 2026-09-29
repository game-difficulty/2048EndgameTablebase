"""Deploy only the tournament client-runtime release; preserve other sites.

Run on the host after uploading a staged competition/ tree and frontend dist.
The live bundle already contains the required receiver; pin its verified hashes
instead of replacing its independently released entry or other site assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from urllib.request import urlopen

ROOT = Path('/opt/2048tables')
RELEASE = '20260929-client-runtime-r1'
STAGE = ROOT / 'deploy-stage' / RELEASE
FRONTEND = ROOT / 'tournament'
BACKEND = ROOT / 'tournament-app'
OLD_RELEASE = '20260929-zh-special-r1'
SERVICE = '2048tables-competition-test.service'
DATABASE = Path('/var/lib/2048tables/competition-test/competition.sqlite3')
BACKUP = Path('/var/lib/2048tables/backups') / RELEASE
LIVE_HASHES = {
    'live-Bm4cqBte.js': 'e32b8c2ec0f7156ddce19a9d6e86a6700250dbb9342d760bfd4769b60b91a49f',
    'live-B3heurOu.css': '40ef4bd6f128f8ddaaf6dec1da680d66162aa0c0af63d30be0be1f391ce4897c',
}
WASM_HASHES = {
    'evil_core.js': '7eab4108e1b5b2804884dd698f71bca04d3ded5e8396ef0b87d7f8ecb1c9c270',
    'evil_core.wasm': '95fbedfb9c4722f1c50d08349853b73bd4eb46521a102ebdf1e838e03c22e813',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target(root):
    link = root / 'current'
    if not link.is_symlink():
        raise RuntimeError(f'Expected release symlink: {link}')
    return link.resolve(strict=True)


def switch(root, destination):
    pending = root / f'current.next-{RELEASE}'
    if pending.exists() or pending.is_symlink():
        raise RuntimeError(f'Temporary link already exists: {pending}')
    pending.symlink_to(destination)
    os.replace(pending, root / 'current')


def room_check():
    with sqlite3.connect(f'file:{DATABASE}?mode=ro', uri=True) as db:
        statuses = db.execute('SELECT status, count(*) FROM competitions GROUP BY status').fetchall()
    print('Room statuses:', statuses, flush=True)
    if any(status not in {'SEATING', 'FINISHED', 'CANCELLED'} for status, _ in statuses):
        raise RuntimeError('Active competition present; abort without switching releases')


def preflight():
    for root in (FRONTEND, BACKEND):
        if target(root) != root / 'releases' / OLD_RELEASE:
            raise RuntimeError(f'Current release changed: {root}')
        if (root / 'releases' / RELEASE).exists():
            raise RuntimeError(f'Target release already exists: {root}')
    live = ROOT / 'app/frontend/dist'
    entry = (live / 'live/index.html').read_text()
    for name, expected in LIVE_HASHES.items():
        if name not in entry or digest(live / 'assets' / name) != expected:
            raise RuntimeError('Live release changed; recheck receiver before deploying')
    for name, expected in WASM_HASHES.items():
        if digest(FRONTEND / 'current/dist/wasm' / name) != expected:
            raise RuntimeError(f'Seeded WASM mismatch: {name}')
    manifest = json.loads((STAGE / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = (STAGE / relative).resolve()
        if not path.is_relative_to(STAGE) or digest(path) != expected:
            raise RuntimeError(f'Staged artifact mismatch: {relative}')
    room_check()
    print('Preflight OK; revision:', manifest['revision'], flush=True)
    return manifest


def verify_http():
    last_error = None
    for _ in range(25):
        try:
            with urlopen('http://127.0.0.1:8770/api/health', timeout=2) as response:
                assert json.load(response)['ok']
            with urlopen('http://127.0.0.1:8770/openapi.json', timeout=2) as response:
                paths = json.load(response)['paths']
            assert '/api/competitions/{room_code}/games/current/state' in paths
            assert '/api/competitions/{room_code}/games/current/move' not in paths
            assert '/api/competitions/{room_code}/games/current/action' not in paths
            return
        except (OSError, AssertionError) as error:
            last_error = error
            time.sleep(.4)
    raise RuntimeError(f'Health/protocol verification failed: {last_error}')


def deploy():
    manifest = preflight()
    if BACKUP.exists():
        raise RuntimeError('Backup already exists')
    new_backend, new_frontend = (root / 'releases' / RELEASE for root in (BACKEND, FRONTEND))
    old_backend, old_frontend = target(BACKEND), target(FRONTEND)
    shutil.copytree(old_backend, new_backend)
    shutil.copytree(old_frontend, new_frontend)
    shutil.copytree(STAGE / 'competition/backend', new_backend / 'competition/backend', dirs_exist_ok=True)
    # Keep old hashed assets and the existing seeded WASM available to open tabs.
    shutil.copytree(STAGE / 'competition-dist', new_frontend / 'dist', dirs_exist_ok=True)
    import gzip
    entry = new_frontend / 'dist/index.html'
    if entry.with_suffix('.html.gz').exists():
        entry.with_suffix('.html.gz').write_bytes(gzip.compress(entry.read_bytes()))
    smoke = '''
from competition.backend.routes import router
from competition.backend.service import CompetitionService
from competition.backend.projects import tournament_project_catalog
from competition.backend.client_runtime import PROTOCOL
paths = {route.path for route in router.routes}
assert '/api/competitions/{room_code}/games/current/state' in paths
assert '/api/competitions/{room_code}/games/current/move' not in paths
assert '/api/competitions/{room_code}/games/current/action' not in paths
assert not hasattr(CompetitionService, 'move_current_game')
assert len(tournament_project_catalog()) == 12
assert PROTOCOL == 'client-runtime-v1'
print('Remote imports, catalog and client-only routes: OK')
'''
    subprocess.run([str(ROOT / 'venv/bin/python'), '-c', smoke], cwd=new_backend,
                   env=dict(os.environ, PYTHONPATH=f'{new_backend}:{ROOT / "app"}'), check=True)
    room_check()
    BACKUP.mkdir(parents=True)
    with sqlite3.connect(f'file:{DATABASE}?mode=ro', uri=True) as source:
        with sqlite3.connect(BACKUP / 'competition.sqlite3') as destination:
            source.backup(destination)
    (BACKUP / 'release.json').write_text(json.dumps({
        'old_backend': str(old_backend), 'old_frontend': str(old_frontend),
        'new_backend': str(new_backend), 'new_frontend': str(new_frontend),
        'revision': manifest['revision'], 'live_unchanged': True,
    }, indent=2) + '\n')
    try:
        switch(BACKEND, new_backend)
        switch(FRONTEND, new_frontend)
        subprocess.run(['systemctl', 'restart', SERVICE], check=True)
        verify_http()
        subprocess.run(['systemctl', 'is-active', '--quiet', SERVICE], check=True)
    except BaseException:
        if target(BACKEND) == new_backend:
            switch(BACKEND, old_backend)
        if target(FRONTEND) == new_frontend:
            switch(FRONTEND, old_frontend)
        subprocess.run(['systemctl', 'restart', SERVICE], check=False)
        raise
    print('Deployed:', new_backend, new_frontend, flush=True)
    print('Live entry/assets and main/play services unchanged; backup:', BACKUP, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise RuntimeError('Run as root on the tournament host')
    preflight() if args.check_only else deploy()
