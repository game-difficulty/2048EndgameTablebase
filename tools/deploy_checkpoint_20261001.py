"""Deploy the audited checkpoint without replacing the split Play dispatcher."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tarfile
import time
from urllib.request import urlopen

ROOT = Path('/opt/2048tables')
APP = ROOT / 'app'
STAGE = Path('/tmp/2048-checkpoint-2a27eef')
RELEASE = '20261001-checkpoint-2a27eef'
BACKUP = ROOT / 'backups' / RELEASE
COMP = ROOT / 'tournament-app'
FRONT = ROOT / 'tournament'
PYTHON = ROOT / 'venv/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def switch(root, target):
    temporary = root / 'current.next-checkpoint'
    assert not temporary.exists() and not temporary.is_symlink()
    temporary.symlink_to(target)
    os.replace(temporary, root / 'current')


def restart(service):
    subprocess.run(['systemctl', 'restart', service], check=True)


def get(url):
    with urlopen(url, timeout=10) as response:
        return response.read()


def health():
    for attempt in range(30):
        try:
            get('http://127.0.0.1:8000/api/live/lobby')
            get('http://127.0.0.1:8770/api/health')
            get('http://127.0.0.1:8766/api/human/health')
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError('health check failed')


def no_active_competition():
    with sqlite3.connect('file:/var/lib/2048tables/competition-test/competition.sqlite3?mode=ro', uri=True) as db:
        rows = db.execute('SELECT status,count(*) FROM competitions GROUP BY status').fetchall()
    if any(status not in {'SEATING', 'FINISHED', 'CANCELLED'} for status, _ in rows):
        raise RuntimeError(f'active competition: {rows}')
    return rows


def main():
    assert os.geteuid() == 0
    manifest = json.loads((STAGE / 'manifest.json').read_text())
    old_comp = (COMP / 'current').resolve()
    old_front = (FRONT / 'current').resolve()
    old_play = (ROOT / 'play/current').resolve()
    assert old_comp.name == '20261001-daily-activity-f446989'
    assert old_front.name == '20261001-cargo-weights-r1'
    assert not (BACKUP / 'deployment-result.json').exists()
    old_dispatcher = digest(APP / 'backend/app.py')
    old_human = digest(APP / 'frontend/dist/human/index.html')
    old_entries = {path: digest(APP / 'frontend/dist' / path) for path in ['index.html', 'live/index.html']}
    for path, expected in manifest['sha256'].items():
        assert digest(STAGE / path) == expected, path
    preflight = STAGE / 'preflight'
    shutil.copytree(APP / 'backend', preflight / 'backend', dirs_exist_ok=True)
    if not (preflight / 'docs_and_configs').exists():
        (preflight / 'docs_and_configs').symlink_to(APP / 'docs_and_configs')
    for path in manifest['paths']:
        if path.startswith('backend/'):
            target = preflight / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(STAGE / path, target)
    env = dict(os.environ, PYTHONPATH=str(preflight) + ':' + str(APP))
    smoke = "import backend.app as m; assert '/api/live/state' in m.app.openapi()['paths']; print('production dispatcher plus checkpoint overlay: import OK')"
    subprocess.run([str(PYTHON), '-c', smoke], cwd=preflight, env=env, check=True)
    print('competition statuses:', no_active_competition(), flush=True)
    BACKUP.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STAGE / 'verified-source.tar', BACKUP / 'verified-source.tar')
    shutil.copy2(STAGE / 'manifest.json', BACKUP / 'deployment-input.json')
    if not (BACKUP / 'previous-main.tar.gz').exists():
        with tarfile.open(BACKUP / 'previous-main.tar.gz', 'w:gz') as out:
            out.add(APP / 'frontend/dist', arcname='frontend/dist')
            for path in manifest['paths']:
                if not path.startswith('backend/') and not path.startswith('docs_and_configs/'):
                    continue
                if (APP / path).exists():
                    out.add(APP / path, arcname=path)
    for name, source in [('auth', '/var/lib/2048tables/auth.sqlite3'),
                         ('competition', '/var/lib/2048tables/competition-test/competition.sqlite3')]:
        destination = Path('/var/lib/2048tables/backups/db') / name / (RELEASE + '.sqlite3')
        if not destination.exists():
            subprocess.run([str(PYTHON), str(STAGE / 'tools/backup_sqlite.py'), source, str(destination)], check=True)
            destination.with_suffix('.json').write_text(json.dumps({
                'source': source, 'release': RELEASE, 'created': time.time(), 'quick_check': 'ok'}))
        else:
            with sqlite3.connect(f'file:{destination}?mode=ro', uri=True) as db:
                assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    new_comp = COMP / 'releases' / RELEASE
    new_front = FRONT / 'releases' / RELEASE
    assert new_comp != old_comp and new_front != old_front
    shutil.copytree(old_comp, new_comp, dirs_exist_ok=True)
    shutil.copytree(old_front, new_front, dirs_exist_ok=True)
    for path in manifest['paths']:
        if path.startswith('competition/backend/'):
            shutil.copy2(STAGE / path, new_comp / path)
    # Keep old hashed assets so tabs opened before the release can lazy-load.
    shutil.copytree(STAGE / 'competition-dist', new_front / 'dist', dirs_exist_ok=True)
    html = (new_front / 'dist/index.html').read_bytes()
    (new_front / 'dist/index.html.gz').write_bytes(gzip.compress(html))
    (new_comp / 'release.json').write_text(json.dumps({'revision': manifest['revision'], 'previous': str(old_comp)}))
    (new_front / 'release.json').write_text(json.dumps({'revision': manifest['revision'], 'previous': str(old_front)}))
    try:
        no_active_competition()
        subprocess.run(['systemctl', 'stop', '2048tables-competition-test.service', '2048tables-cloud.service'], check=True)
        no_active_competition()
        for path in manifest['paths']:
            if path.startswith('backend/') or path.startswith('docs_and_configs/'):
                target = APP / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(STAGE / path, target)
        for path in (STAGE / 'main-dist').rglob('*'):
            relative = path.relative_to(STAGE / 'main-dist')
            if path.is_dir() or relative.parts[0] == 'human':
                continue
            target = APP / 'frontend/dist' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative.parts[0] == 'assets' and target.exists():
                assert digest(target) == digest(path), f'hashed asset collision: {relative}'
            temporary = target.with_name(target.name + '.next-checkpoint')
            shutil.copy2(path, temporary)
            os.replace(temporary, target)
        for folder in ['frontend/src', 'frontend/scripts']:
            shutil.copytree(STAGE / folder, APP / folder, dirs_exist_ok=True)
        build = json.loads((STAGE / 'main-dist/release.json').read_text())
        provenance = dict(manifest, release=RELEASE, build=build, previousEntries=old_entries,
                          sourceArchive=str(BACKUP / 'verified-source.tar'), deployedAt=time.time(),
                          backendOverlay='audited changed modules; production backend/app.py preserved')
        (APP / 'frontend/dist/main-source-manifest.json').write_text(json.dumps(provenance, indent=2))
        switch(COMP, new_comp)
        switch(FRONT, new_front)
        for base in [new_comp, new_front]:
            subprocess.run(['chown', '-R', 'ubuntu:ubuntu', str(base)], check=True)
        restart('2048tables-cloud.service')
        restart('2048tables-competition-test.service')
        health()
        assert digest(APP / 'backend/app.py') == old_dispatcher
        assert digest(APP / 'frontend/dist/human/index.html') == old_human
        assert (ROOT / 'play/current').resolve() == old_play
        for domain, dist, entry in [('2048tables.online', APP / 'frontend/dist', 'index.html'),
                                    ('live.2048tables.online', APP / 'frontend/dist', 'live/index.html'),
                                    ('tournament.2048tables.online', new_front / 'dist', 'index.html')]:
            route = '/practice' if domain.startswith('tournament.') else '/'
            published = subprocess.check_output(['curl', '-fksS', '--max-time', '20', '--resolve',
                domain + ':443:127.0.0.1', 'https://' + domain + route])
            assert published == (dist / entry).read_bytes(), domain
            for asset in re.findall(r'(?:src|href)="(/assets/[^"?]+)', published.decode()):
                data = subprocess.check_output(['curl', '-fksS', '--max-time', '20', '--resolve',
                    domain + ':443:127.0.0.1', 'https://' + domain + asset])
                assert data == (dist / asset.lstrip('/')).read_bytes(), asset
        report = {'release': RELEASE, 'revision': manifest['revision'], 'build': build,
                  'previous': {'competitionBackend': str(old_comp), 'competitionFrontend': str(old_front),
                               'play': str(old_play), 'mainEntries': old_entries}, 'verified': True}
        (BACKUP / 'deployment-result.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    except BaseException:
        switch(COMP, old_comp)
        switch(FRONT, old_front)
        with tarfile.open(BACKUP / 'previous-main.tar.gz') as old:
            old.extractall(APP)
        restart('2048tables-cloud.service')
        restart('2048tables-competition-test.service')
        raise


if __name__ == '__main__':
    main()
