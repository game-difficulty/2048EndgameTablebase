"""Frontend-only cargo probability update, without backend/database changes."""
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path('/opt/2048tables/tournament')
OLD = ROOT / 'releases/20261001-cargo-rigid-r1'
NEW = ROOT / 'releases/20261001-cargo-weights-r1'
STAGE = Path('/home/ubuntu/20261001-cargo-weights-r1/dist')


def switch(target):
    temp = ROOT / 'current.next-cargo-weights'
    temp.symlink_to(target)
    os.replace(temp, ROOT / 'current')


def main():
    assert (ROOT / 'current').resolve() == OLD and not NEW.exists()
    shutil.copytree(OLD, NEW)
    shutil.copytree(STAGE, NEW / 'dist', dirs_exist_ok=True)
    html = (NEW / 'dist/index.html').read_bytes()
    (NEW / 'dist/index.html.gz').write_bytes(gzip.compress(html))
    (NEW / 'release.json').write_text(json.dumps({'previous': str(OLD), 'change': 'cargo family weights 30/30/30/10'}))
    for p in NEW.rglob('*'):
        os.chmod(p, 0o755 if p.is_dir() else 0o644)
    try:
        switch(NEW)
        assets = re.findall(r'(?:src|href)="(/assets/[^"?]+)', html.decode())
        for path in ['/practice/1', '/test', '/api/health', *assets]:
            data = subprocess.check_output(['curl', '-fksS', '--max-time', '20', '--resolve', 'tournament.2048tables.online:443:127.0.0.1', 'https://tournament.2048tables.online' + path])
            if path in assets:
                assert data == (NEW / 'dist' / path.lstrip('/')).read_bytes()
            elif path != '/api/health':
                assert data == html
    except BaseException:
        switch(OLD)
        raise
    print('Verified:', NEW)


if __name__ == '__main__':
    main()
