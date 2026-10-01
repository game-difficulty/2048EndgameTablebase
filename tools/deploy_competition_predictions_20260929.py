"""Scoped release packer / remote installer. No wallet transactions in smoke checks."""
import argparse
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

RELEASE='20260929-competition-predictions-r1'
ROOT=Path('/opt/2048tables')
APP=ROOT/'app'
STAGE=ROOT/'deploy-stage'/RELEASE
BACKUP=Path('/var/lib/2048tables/backups')/RELEASE
COMP=ROOT/'tournament-app'
FRONT=ROOT/'tournament'
CS='2048tables-competition-test.service'
LS='2048tables-cloud.service'
DB=Path('/var/lib/2048tables/competition-test/competition.sqlite3')
BACKEND_FILES=['backend/live/routes.py','backend/live/dynamic_rooms.py','backend/room_activities/runtime.py','backend/room_activities/competition_predictions.py']
COMP_FILES=['competition/backend/db.py','competition/backend/routes.py','competition/backend/service.py']

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def pack():
    repo=Path(__file__).resolve().parents[1]
    base=repo/'output/deploy-competition-predictions'
    stage=base/'stage'
    build=repo/'output/competition-predictions-build'
    manifest=json.loads((build/'.vite/manifest.json').read_text())
    seen=set()
    def visit(key):
        if key in seen:return
        seen.add(key)
        item=manifest[key]
        for file in [item['file'],*item.get('css',[]),*item.get('assets',[])]:
            dest=stage/'live-dist'/file
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(build/file,dest)
        for dependency in item.get('imports',[])+item.get('dynamicImports',[]):visit(dependency)
    visit('live/index.html')
    (stage/'live-dist/live').mkdir(parents=True,exist_ok=True)
    shutil.copy2(build/'live/index.html',stage/'live-dist/live/index.html')
    expected={file:sha(base/file) for file in BACKEND_FILES+COMP_FILES if (base/file).exists()}
    expected['frontend/dist/live/index.html']=sha(base/'frontend/dist/live/index.html')
    data={'release':RELEASE,'expected':expected,'files':{p.relative_to(stage).as_posix():sha(p) for p in stage.rglob('*') if p.is_file() and p.name!='manifest.json'}}
    (stage/'manifest.json').write_text(json.dumps(data,indent=2))
    with tarfile.open(base/'release.tgz','w:gz') as archive:
        for p in stage.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(stage).as_posix())
    print('Packed',len(data['files']),'files; SHA256',sha(base/'release.tgz'))

def rooms():
    with sqlite3.connect(f'file:{DB}?mode=ro',uri=True) as db:
        rows=db.execute('SELECT status,count(*) FROM competitions GROUP BY status').fetchall()
    print('Room states:',rows,flush=True)
    if any(status not in ('SEATING','FINISHED','CANCELLED') for status,_ in rows):raise RuntimeError('Active match: stop deployment')

def current(root):return (root/'current').resolve(strict=True)

def switch(root,dest):
    link=root/(RELEASE+'.next')
    link.symlink_to(dest)
    os.replace(link,root/'current')

def http(url):
    with urlopen(url,timeout=3) as response:return json.load(response)

def health():
    error=None
    for _ in range(30):
        try:
            assert http('http://127.0.0.1:8770/api/health')['ok']
            assert '/api/internal/live/settlement/{public_key}' in http('http://127.0.0.1:8770/openapi.json')['paths']
            assert isinstance(http('http://127.0.0.1:8000/api/live/lobby')['rooms'],list)
            for service in (CS,LS):subprocess.run(['systemctl','is-active','--quiet',service],check=True)
            return
        except Exception as exc:error=exc;time.sleep(.5)
    raise RuntimeError(f'Health failed: {error}')

def deploy(check=False):
    data=json.loads((STAGE/'manifest.json').read_text())
    oldcomp,oldfront=current(COMP),current(FRONT)
    for root in (COMP,FRONT):
        expected_release='20260929-room-creator-89-r1' if root==COMP else '20260929-client-runtime-r1'
        if current(root)!=root/'releases'/expected_release:raise RuntimeError('Release changed')
        if (root/'releases'/RELEASE).exists():raise RuntimeError('Release already exists')
    if BACKUP.exists():raise RuntimeError('Backup exists')
    for file,digest in data['files'].items():
        path=(STAGE/file).resolve()
        if not path.is_relative_to(STAGE) or sha(path)!=digest:raise RuntimeError('Invalid stage: '+file)
    for file,digest in data['expected'].items():
        root=oldcomp if file.startswith('competition/') else APP
        if sha(root/file)!=digest:raise RuntimeError('Published file changed: '+file)
    if (APP/BACKEND_FILES[-1]).exists():raise RuntimeError('Prediction module already deployed')
    rooms()
    protected={p:sha(p) for p in [APP/'frontend/dist/index.html',APP/'frontend/dist/human/index.html'] if p.exists()}
    if check:print('Preflight OK');return
    BACKUP.mkdir(parents=True)
    for source,name in [(DB,'competition.sqlite3'),(Path('/var/lib/2048tables/auth.sqlite3'),'auth.sqlite3')]:
        with sqlite3.connect(f'file:{source}?mode=ro',uri=True) as db:
            with sqlite3.connect(BACKUP/name) as dest:db.backup(dest)
    for file in BACKEND_FILES+['frontend/dist/live/index.html']:
        path=APP/file
        if path.exists():
            dest=BACKUP/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    newcomp,newfront=COMP/'releases'/RELEASE,FRONT/'releases'/RELEASE
    shutil.copytree(oldcomp,newcomp)
    shutil.copytree(oldfront,newfront)
    for file in COMP_FILES:shutil.copy2(STAGE/file,newcomp/file)
    shutil.copytree(STAGE/'competition-dist',newfront/'dist',dirs_exist_ok=True)
    entry=newfront/'dist/index.html'
    entry.with_suffix('.html.gz').write_bytes(gzip.compress(entry.read_bytes()))
    subprocess.run([str(ROOT/'venv/bin/python'),'-m','compileall','-q',str(newcomp/'competition/backend')],check=True)
    (BACKUP/'release.json').write_text(json.dumps(dict(old_comp=str(oldcomp),old_front=str(oldfront),release=RELEASE),indent=2))
    try:
        subprocess.run(['systemctl','stop',CS],check=True)
        rooms()
        switch(COMP,newcomp);switch(FRONT,newfront)
        subprocess.run(['systemctl','start',CS],check=True)
        for _ in range(20):
            try:
                assert '/api/internal/live/settlement/{public_key}' in http('http://127.0.0.1:8770/openapi.json')['paths'];break
            except Exception:time.sleep(.5)
        else:raise RuntimeError('New competition endpoint unavailable')
        for file in BACKEND_FILES:
            dest=APP/file;temp=dest.with_suffix('.py.next');shutil.copy2(STAGE/file,temp);os.replace(temp,dest)
        subprocess.run(['systemctl','restart',LS],check=True)
        health()
        assets=APP/'frontend/dist/assets'
        for source in (STAGE/'live-dist/assets').iterdir():
            dest=assets/source.name
            if dest.exists() and sha(dest)!=sha(source):raise RuntimeError('Hashed asset collision')
            if not dest.exists():shutil.copy2(source,dest)
        entry=APP/'frontend/dist/live/index.html'
        temp=entry.with_suffix('.html.next');shutil.copy2(STAGE/'live-dist/live/index.html',temp);os.replace(temp,entry)
        entry.with_suffix('.html.gz').write_bytes(gzip.compress(entry.read_bytes()))
        for path,digest in protected.items():assert sha(path)==digest
        health()
    except BaseException:
        switch(COMP,oldcomp);switch(FRONT,oldfront)
        for file in BACKEND_FILES+['frontend/dist/live/index.html']:
            if (BACKUP/file).exists():shutil.copy2(BACKUP/file,APP/file)
        entry=APP/'frontend/dist/live/index.html';entry.with_suffix('.html.gz').write_bytes(gzip.compress(entry.read_bytes()))
        subprocess.run(['systemctl','restart',CS,LS],check=False)
        raise
    print('DEPLOYED',RELEASE,'backup',BACKUP,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pack',action='store_true');parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.pack:pack()
    else:deploy(args.check_only)
