"""Scoped static release; preserve the newer live bundle's unrelated changes."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

LABEL = '20261001-cargo-rigid-r1'
STAGE = Path('/home/ubuntu') / LABEL
ROOT = Path('/opt/2048tables/tournament')
OLD = ROOT / 'releases/20260930-live-i18n-r1'
NEW = ROOT / 'releases' / LABEL
LIVE = Path('/opt/2048tables/app/frontend/dist')
HTML = LIVE / 'live/index.html'
BUNDLE = LIVE / 'assets/live-CWmSLUPI.js'
BACKUP = Path('/var/lib/2048tables/backups/live-static') / LABEL


def sha(data):
    return hashlib.sha256(data).hexdigest()


def switch(target):
    temporary = ROOT / (LABEL + '.next')
    temporary.symlink_to(target)
    os.replace(temporary, ROOT / 'current')


def live_patch(source, shapes):
    before = '[[[0,0],[0,1],[1,0],[1,1]],[[0,0],[0,1],[1,0]],[[0,0],[0,1],[1,1]],[[0,1],[1,0],[1,1]],[[0,0],[1,0],[1,1]]]'
    label = 'return n?{left:`${(n[1]*F+P/2)/L*100}%`,top:`${(n[0]*F+P/2)/L*100}%`}:null}'
    assert source.count(before) == source.count(label) == 1
    source = source.replace(before, json.dumps(shapes, separators=(',', ':')))
    return source.replace(label, 'n||=t.length?[t.reduce((s,c)=>s+c[0],0)/t.length,t.reduce((s,c)=>s+c[1],0)/t.length]:null;' + label)


def main():
    assert os.geteuid() == 0
    assert (ROOT / 'current').resolve() == OLD and not NEW.exists()
    old_html = HTML.read_bytes()
    assert sha(old_html) == 'a693024f929c5891a4039ad3ed18514809ec43b4839f9eb17e8767961517e62c'
    assert sha(BUNDLE.read_bytes()) == 'd17b26d5ae85bd338fb6c024b5146b4970e0f2f7a7900195994b6d7cde417453'
    shapes = [json.loads(value) for value in re.findall(r'cells: (\[\[.*?\]\])', (STAGE / 'cargoShapes.mjs').read_text())]
    assert len(shapes) == 6
    source = live_patch(BUNDLE.read_text(), shapes).encode()
    asset = LIVE / 'assets' / ('live-cargo-' + sha(source)[:16] + '.js')
    assert not asset.exists()
    new_html = old_html.replace(BUNDLE.name.encode(), asset.name.encode())
    assert new_html != old_html
    shutil.copytree(OLD, NEW)
    shutil.copytree(STAGE / 'dist', NEW / 'dist', dirs_exist_ok=True)
    (NEW / 'dist/index.html.gz').write_bytes(gzip.compress((NEW / 'dist/index.html').read_bytes()))
    (NEW / 'release.json').write_text(json.dumps({'label': LABEL, 'previous': str(OLD)}))
    for p in NEW.rglob('*'):
        os.chmod(p, 0o755 if p.is_dir() else 0o644)
    BACKUP.mkdir()
    (BACKUP / 'live-index.html').write_bytes(old_html)
    (BACKUP / 'manifest.json').write_text(json.dumps({'previous_bundle': str(BUNDLE), 'new_bundle': str(asset), 'previous_sha': sha(old_html)}))
    asset.write_bytes(source)
    os.chmod(asset, 0o644)
    asset.with_suffix('.js.gz').write_bytes(gzip.compress(source))

    def write_html(data):
        temporary = HTML.with_name('index.' + LABEL + '.html')
        temporary.write_bytes(data)
        os.chmod(temporary, 0o644)
        os.replace(temporary, HTML)
        HTML.with_suffix('.html.gz').write_bytes(gzip.compress(data))

    def fetch(host, path):
        return subprocess.check_output(['curl', '-fksS', '--max-time', '20', '--resolve', host + ':443:127.0.0.1', 'https://' + host + path])

    try:
        write_html(new_html)
        switch(NEW)
        for path in ('/practice/1', '/practice/12', '/test'):
            assert fetch('tournament.2048tables.online', path) == (NEW / 'dist/index.html').read_bytes()
        fetch('tournament.2048tables.online', '/api/health')
        assert fetch('live.2048tables.online', '/') == new_html
        assert fetch('live.2048tables.online', '/assets/' + asset.name) == source
        for path in re.findall(r'(?:src|href)="(/assets/[^"?]+)', (NEW / 'dist/index.html').read_text()):
            assert fetch('tournament.2048tables.online', path) == (NEW / 'dist' / path.lstrip('/')).read_bytes()
    except BaseException:
        switch(OLD)
        write_html(old_html)
        raise
    print(json.dumps({'release': str(NEW), 'live_asset': asset.name, 'backup': str(BACKUP), 'verified': True}))


if __name__ == '__main__':
    main()
