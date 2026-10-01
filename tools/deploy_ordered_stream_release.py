"""Publish verified ordered-stream overlay. Run on production with sudo."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tarfile
import time
from urllib.request import urlopen

ROOT = Path('/opt/2048tables')
APP = ROOT / 'app'
RELEASE = '20261001-ordered-stream-320662e'
STAGE = Path('/tmp') / RELEASE
BACKUP = ROOT / 'backups' / RELEASE

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def switch(base, target):
    temporary = base / 'current.next-stream'
    temporary.symlink_to(target)
    os.replace(temporary, base / 'current')

def idle():
    with sqlite3.connect('file:/var/lib/2048tables/competition-test/competition.sqlite3?mode=ro', uri=True) as db:
        rows = db.execute("SELECT room_code,status FROM competitions WHERE status IN ('GAME_A_PLAYING','GAME_B_PLAYING','GAME_C_PLAYING')").fetchall()
    if rows:
        raise RuntimeError(f'Active competitions: {rows}')

def services(action):
    subprocess.run(['systemctl', action, '2048tables-cloud', '2048tables-competition-test'], check=True)

def health():
    for _ in range(25):
        try:
            for url in ['http://127.0.0.1:8000/api/live/lobby','http://127.0.0.1:8770/api/health']:
                with urlopen(url, timeout=3) as r:
                    assert r.status == 200
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError('health failed')

def main():
    manifest = json.loads((STAGE/'manifest.json').read_text())
    for path, value in manifest['sha256'].items():
        assert sha(STAGE/path) == value, path
    for path, value in manifest['productionHashes'].items():
        assert sha(Path(path)) == value, 'Production changed: ' + path
    idle()
    old_comp = (ROOT/'tournament-app/current').resolve()
    old_front = (ROOT/'tournament/current').resolve()
    old_play = (ROOT/'play/current').resolve()
    unchanged = {str(p):sha(p) for p in [APP/'backend/app.py', APP/'frontend/dist/human/index.html', APP/'backend/room_activities/competition_predictions.py']}
    new_comp = ROOT/'tournament-app/releases'/RELEASE
    new_front = ROOT/'tournament/releases'/RELEASE
    assert not new_comp.exists() and not BACKUP.exists()
    BACKUP.mkdir(parents=True)
    shutil.copy2(STAGE/'manifest.json', BACKUP/'manifest.json')
    with tarfile.open(BACKUP/'previous.tar.gz', 'w:gz') as package:
        package.add(APP/'frontend/dist', arcname='frontend/dist')
        for path in manifest['paths']:
            if path.startswith('backend/'):
                package.add(APP/path, arcname=path)
    shutil.copytree(old_comp,new_comp)
    shutil.copytree(old_front,new_front)
    for path in manifest['paths']:
        if path.startswith('competition/backend/'):
            shutil.copy2(STAGE/path,new_comp/path)
    shutil.copytree(STAGE/'competition/frontend/dist', new_front/'dist',dirs_exist_ok=True)
    (new_front/'dist/index.html.gz').write_bytes(gzip.compress((new_front/'dist/index.html').read_bytes()))
    for base in [new_comp,new_front]:
        (base/'release.json').write_text(json.dumps(dict(manifest,deployedAt=time.time())))
        subprocess.run(['chown','-R','ubuntu:ubuntu',str(base)],check=True)
    # Preserve an exact effective overlay and verify syntax before downtime.
    for path in manifest['paths']:
        if path.startswith('backend/') or path.startswith('competition/backend/'):
            compile((STAGE/path).read_text(),path,'exec')
    with tarfile.open(BACKUP/'overlay.tar.gz','w:gz') as package:
        for path in manifest['paths']:
            package.add(STAGE/path,arcname=path)
    try:
        idle()
        services('stop')
        idle()
        for path in manifest['paths']:
            if path.startswith('backend/'):
                shutil.copy2(STAGE/path,APP/path)
        for path in (STAGE/'frontend/dist').rglob('*'):
            rel=path.relative_to(STAGE/'frontend/dist')
            if path.is_dir() or rel.parts[0]=='human':
                continue
            target=APP/'frontend/dist'/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            if rel.parts[0]=='assets' and target.exists():
                assert sha(target)==sha(path),str(rel)
            temp=target.with_name(target.name+'.next-stream')
            shutil.copy2(path,temp)
            os.replace(temp,target)
        (APP/'frontend/dist/main-source-manifest.json').write_text(json.dumps(manifest,indent=2))
        switch(ROOT/'tournament-app',new_comp)
        switch(ROOT/'tournament',new_front)
        services('start')
        health()
        for path,value in unchanged.items():
            assert sha(Path(path))==value,path
        assert (ROOT/'play/current').resolve()==old_play
        for host,folder,entry,route in [('2048tables.online',APP/'frontend/dist','index.html','/'),('live.2048tables.online',APP/'frontend/dist','live/index.html','/'),('tournament.2048tables.online',new_front/'dist','index.html','/practice')]:
            body=subprocess.check_output(['curl','-fksS','--max-time','10','--resolve',host+':443:127.0.0.1','https://'+host+route])
            assert body==(folder/entry).read_bytes(),host
        result={'release':RELEASE,'verified':True,'previousBackend':str(old_comp),'previousFrontend':str(old_front),'playUnchanged':str(old_play),'finishedAt':time.time()}
        (BACKUP/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    except BaseException:
        switch(ROOT/'tournament-app',old_comp)
        switch(ROOT/'tournament',old_front)
        with tarfile.open(BACKUP/'previous.tar.gz') as package:
            package.extractall(APP)
        services('restart')
        raise

if __name__=='__main__':
    main()
