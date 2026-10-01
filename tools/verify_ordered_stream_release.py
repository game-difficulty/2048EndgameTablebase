"""Verify publication and prune only explicitly reviewed release artifacts."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.request import urlopen, Request

root=Path('/opt/2048tables')
release='20261001-ordered-stream-320662e'
backup=root/'backups'/release
assert json.loads((backup/'result.json').read_text())['verified']
verified=[]
def fetch(host,path):
    return subprocess.check_output(['curl','-fksS','--max-time','15','--resolve',host+':443:127.0.0.1','https://'+host+path])
for host,folder,route,entry in [('2048tables.online',root/'app/frontend/dist','/','index.html'),('live.2048tables.online',root/'app/frontend/dist','/','live/index.html'),('tournament.2048tables.online',root/'tournament/current/dist','/practice','index.html')]:
    body=fetch(host,route)
    assert body==(folder/entry).read_bytes(),host
    for path in re.findall(r'(?:src|href)="(/assets/[^"?]+)',body.decode()):
        assert fetch(host,path)==(folder/path.lstrip('/')).read_bytes(),path
    verified.append(host)
processes={}
for service in ['2048tables-cloud','2048tables-competition-test','2048tables-play']:
    pid=subprocess.check_output(['systemctl','show',service,'-p','MainPID','--value']).decode().strip()
    processes[service]={'pid':pid,'cwd':str((Path('/proc')/pid/'cwd').resolve())}
assert processes['2048tables-competition-test']['cwd']==str(root/'tournament-app/releases'/release)
candidate=root/'tournament/releases/20261001-cargo-rigid-r1'
assert candidate.resolve().parent==root/'tournament/releases' and not candidate.is_symlink()
assert (root/'tournament/current').resolve()!=candidate
# Check process mappings/open files, symlinks, and configured rollback pins.
protected=[]
for process in Path('/proc').iterdir():
    if not process.name.isdigit(): continue
    for ref in [process/'cwd',process/'exe',*list((process/'fd').glob('*'))]:
        try:
            if str(ref.resolve()).startswith(str(candidate)): protected.append(str(ref))
        except OSError: pass
    try:
        if str(candidate) in (process/'maps').read_text(): protected.append(str(process/'maps'))
    except OSError: pass
for base in [root/'tournament',root/'tournament-app',Path('/etc/nginx'),Path('/etc/systemd/system')]:
    for path in base.rglob('*'):
        if path.is_symlink() and str(path.resolve()).startswith(str(candidate)):
            protected.append(str(path))
        if path.is_file() and path.suffix in {'.conf','.service','.json','.pin'}:
            try:
                # Previous-release history is provenance, not a rollback pin.
                if path.name=='release.json': continue
                if str(candidate) in path.read_text(): protected.append(str(path))
            except (OSError,UnicodeError): pass
removed=[]
reclaimed=0
if not protected:
    reclaimed += sum(p.stat().st_size for p in candidate.rglob('*') if p.is_file())
    shutil.rmtree(candidate)
    removed.append(str(candidate))
shutil.copy2('/tmp/deploy_ordered_stream_release.py',backup/'deploy.py')
for target in [Path('/tmp')/release,Path('/tmp/deploy.tar.gz'),Path('/tmp/deploy_ordered_stream_release.py')]:
    assert target.resolve().parent==Path('/tmp') and not target.is_symlink()
    reclaimed+=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()) if target.is_dir() else target.stat().st_size
    shutil.rmtree(target) if target.is_dir() else target.unlink()
    removed.append(str(target))
usage=shutil.disk_usage('/')
report={'originVerifiedHosts':verified,'processes':processes,'removed':removed,'reclaimedBytes':reclaimed,
        'retainedFrontendReleases':sorted(p.name for p in (root/'tournament/releases').iterdir()),
        'skippedProtected':protected,'skipped':'Older backend releases and mixed backup artifacts lack reliable retention provenance; existing database backups retained.',
        'diskUsedPercent':round(100*usage.used/usage.total,1),'diskFreeBytes':usage.free}
(backup/'verification-retention.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
