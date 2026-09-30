"""Scoped event/enrollment/schedule release. Run on host; --check-only is read-only."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from urllib.request import urlopen

RELEASE = '20260930-events-schedule-r1'
ROOT = Path('/opt/2048tables')
STAGE = ROOT / 'deploy-stage' / RELEASE
BACK = ROOT / 'tournament-app'
FRONT = ROOT / 'tournament'
BACKUP = Path('/var/lib/2048tables/backups') / RELEASE
DB = Path('/var/lib/2048tables/competition-test/competition.sqlite3')
NGINX = Path('/etc/nginx/sites-enabled/tournament.2048tables.online').resolve()
SERVICE = '2048tables-competition-test.service'
ENV = Path('/etc/2048tables/competition-test.env')
PLAY_DB = Path('/var/lib/2048tables/play/human.sqlite3')
OLD = '20260929-competition-predictions-r1'

def run(*args):
    subprocess.run(args, check=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rooms(allow_waiting):
    with sqlite3.connect(f'file:{DB}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT status,count(*) FROM competitions GROUP BY status').fetchall()
        print('Room states', rows, flush=True)
        safe = {'SEATING','FINISHED','CANCELLED'}
        if allow_waiting:
            safe |= {'READY_CHECK','GAME_A_READY','GAME_B_READY','GAME_C_READY'}
        if any(status not in safe for status,count in rows):
            raise RuntimeError('Active match; deployment stopped')
        if db.execute("SELECT 1 FROM competition_team_clocks WHERE state='running' LIMIT 1").fetchone():
            raise RuntimeError('Running clock; deployment stopped')

def switch(root, destination):
    link = root / (RELEASE + '.next')
    link.symlink_to(destination)
    os.replace(link, root / 'current')

def health():
    for _ in range(30):
        try:
            with urlopen('http://127.0.0.1:8770/api/health', timeout=2) as response:
                assert json.load(response)['schema_version'] == 14
            with urlopen('http://127.0.0.1:8770/api/events', timeout=2) as response:
                assert len(json.load(response)['events']) >= 2
            return
        except Exception:
            time.sleep(.5)
    raise RuntimeError('New service health failed')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--allow-waiting',action='store_true',help='Only after operator approves restarting a waiting match')
    args=parser.parse_args()
    manifest=json.loads((STAGE/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        path=(STAGE/name).resolve()
        assert path.is_relative_to(STAGE) and sha(path)==digest, name
    oldback,oldfront=(BACK/'current').resolve(),(FRONT/'current').resolve()
    assert oldback.name==OLD and oldfront.name==OLD
    assert not BACKUP.exists()
    rooms(args.allow_waiting)
    assert PLAY_DB.is_file()
    assert NGINX.read_text().count('location = /test/ { return 308 /test; }')==1
    if args.check_only:
        print('Preflight OK');return
    BACKUP.mkdir(parents=True)
    shutil.copy2(NGINX,BACKUP/'nginx.conf')
    shutil.copy2(ENV,BACKUP/'competition-test.env')
    newback,newfront=BACK/'releases'/RELEASE,FRONT/'releases'/RELEASE
    shutil.copytree(oldback,newback)
    shutil.copytree(oldfront,newfront)
    shutil.copytree(STAGE/'competition/backend',newback/'competition/backend',dirs_exist_ok=True)
    shutil.copytree(STAGE/'dist',newfront/'dist',dirs_exist_ok=True)
    html=newfront/'dist/index.html'
    html.with_suffix('.html.gz').write_bytes(gzip.compress(html.read_bytes()))
    run(str(ROOT/'venv/bin/python'),'-m','compileall','-q',str(newback/'competition/backend'))
    (BACKUP/'release.json').write_text(json.dumps({'old_backend':str(oldback),'old_frontend':str(oldfront),'commit':manifest['commit']}))
    run('systemctl','stop',SERVICE)
    try:
        rooms(args.allow_waiting)
        with sqlite3.connect(f'file:{DB}?mode=ro',uri=True) as source:
            with sqlite3.connect(BACKUP/'competition.sqlite3') as target:
                source.backup(target)
        switch(BACK,newback)
        env = ENV.read_text()
        lines = [line for line in env.splitlines() if not line.startswith('HUMAN_PLAY_DB=')]
        ENV.write_text('\n'.join(lines) + '\nHUMAN_PLAY_DB=' + str(PLAY_DB) + '\n')
        run('systemctl','start',SERVICE)
        health()
        switch(FRONT,newfront)
        text=NGINX.read_text()
        text=text.replace('location = /test/ { return 308 /test; }', 'location = /test/ { return 308 /test; }\n    location ~ "^/events(?:/[a-z0-9]+(?:-[a-z0-9]+)*)?/?$" { try_files /index.html =404; add_header Cache-Control no-cache; }')
        NGINX.write_text(text)
        run('nginx','-t')
        run('systemctl','reload','nginx')
        health()
    except BaseException:
        switch(BACK,oldback);switch(FRONT,oldfront)
        shutil.copy2(BACKUP/'nginx.conf',NGINX)
        shutil.copy2(BACKUP/'competition-test.env',ENV)
        run('nginx','-t');run('systemctl','reload','nginx')
        run('systemctl','restart',SERVICE)
        # Additive schema changes are retained; never overwrite live data during rollback.
        raise
    print('DEPLOYED',RELEASE, 'backup', BACKUP, flush=True)

if __name__=='__main__':
    main()
