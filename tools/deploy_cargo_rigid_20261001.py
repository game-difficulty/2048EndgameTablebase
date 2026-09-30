"""Tournament static release; verify the concurrent live release has cargo support."""
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


def switch(target):
    temporary = ROOT / (LABEL + '.next')
    temporary.symlink_to(target)
    os.replace(temporary, ROOT / 'current')


def fetch(host, path):
    return subprocess.check_output(['curl', '-fksS', '--max-time', '20', '--resolve', host + ':443:127.0.0.1', 'https://' + host + path])


def main():
    assert os.geteuid() == 0
    assert (ROOT / 'current').resolve() == OLD and not NEW.exists()
    # The concurrent chat deployment already included the committed shared
    # cargo shapes and renderer. Never overwrite it with an older live build.
    live_html = (LIVE / 'live/index.html').read_bytes()
    assert hashlib.sha256(live_html).hexdigest() == '085786818663f765f4d2fa38704d8a2553ad2510fe4e6d984dab4ccd0cea3db4'
    live_bundle = LIVE / 'assets/live-CptWFNve.js'
    assert hashlib.sha256(live_bundle.read_bytes()).hexdigest() == 'c196a960652c3a9480cd054dd47f495fbd261e63e7f4a4be6c3b7e114077da7b'
    assert fetch('live.2048tables.online', '/') == live_html
    assert fetch('live.2048tables.online', '/assets/' + live_bundle.name) == live_bundle.read_bytes()
    shutil.copytree(OLD, NEW)
    shutil.copytree(STAGE / 'dist', NEW / 'dist', dirs_exist_ok=True)
    (NEW / 'dist/index.html.gz').write_bytes(gzip.compress((NEW / 'dist/index.html').read_bytes()))
    (NEW / 'release.json').write_text(json.dumps({'label': LABEL, 'previous': str(OLD)}))
    for p in NEW.rglob('*'):
        os.chmod(p, 0o755 if p.is_dir() else 0o644)
    try:
        switch(NEW)
        for path in ('/practice/1', '/practice/12', '/test'):
            assert fetch('tournament.2048tables.online', path) == (NEW / 'dist/index.html').read_bytes()
        fetch('tournament.2048tables.online', '/api/health')
        for path in re.findall(r'(?:src|href)="(/assets/[^"?]+)', (NEW / 'dist/index.html').read_text()):
            assert fetch('tournament.2048tables.online', path) == (NEW / 'dist' / path.lstrip('/')).read_bytes()
    except BaseException:
        switch(OLD)
        raise
    print(json.dumps({'release': str(NEW), 'live_asset': live_bundle.name, 'verified': True}))


if __name__ == '__main__':
    main()
