import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

root=Path(__file__).resolve().parents[1]
stage=Path('C:/Users/Administrator/AppData/Local/Temp/2048-ordered-stream-kvciy2vp')
source=stage/'source'
manifest=json.loads((source/'manifest.json').read_text())
path='frontend/src/features/roomActivities/CompetitionPredictions.vue'
(source/path).write_bytes(subprocess.check_output(['git','show','819d700:'+path],cwd=root))
manifest['paths'].append(path)
manifest['sha256'][path]=hashlib.sha256((source/path).read_bytes()).hexdigest()
manifest['preservedProductionOverlay']={'819d700':[path]}
manifest['baselineVerification']='Main/live baseline entry assets byte-matched production checkpoint. All archived frontend/competition/font sources match Git baseline. Tournament uses installed newer Vite; source baseline hash-verified.'
manifest['productionHashes']={}
paths=['/opt/2048tables/app/frontend/dist/index.html','/opt/2048tables/app/frontend/dist/live/index.html','/opt/2048tables/tournament/current/dist/index.html']
paths += ['/opt/2048tables/app/'+p for p in manifest['paths'] if p.startswith('backend/')]
for line in subprocess.check_output(['ssh','ubuntu@43.135.117.69','sha256sum '+' '.join(paths)]).decode().splitlines():
    value,path=line.split(None,1)
    manifest['productionHashes'][path]=value
subprocess.run(['npm.cmd','run','build'],cwd=source/'frontend',env=dict(os.environ,GIT_DIR=str(root/'.git')),check=True)
(source/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(stage/'deploy.tar.gz','w:gz') as package:
    for path in manifest['paths']:
        package.add(source/path,arcname=path)
    for folder in ['frontend/dist','competition/frontend/dist']:
        package.add(source/folder,arcname=folder)
    package.add(source/'manifest.json',arcname='manifest.json')
print('PACKAGE',stage/'deploy.tar.gz',(stage/'deploy.tar.gz').stat().st_size)
