"""Frontend-only numbering release; retain the previous release for rollback."""
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/opt/2048tables/tournament')
OLD = ROOT / 'releases/20260930-appendix-tags-r1'
NEW = ROOT / 'releases/20260930-numbering-r1'
STAGE = Path('/home/ubuntu/competition-release-20260930-numbering-r1/dist')


def switch(target):
    temporary = ROOT / 'current.next-numbering-r1'
    temporary.symlink_to(target)
    os.replace(temporary, ROOT / 'current')


def main():
    if (ROOT / 'current').resolve() != OLD or NEW.exists():
        raise RuntimeError('Unexpected current release or existing release directory')
    if not (STAGE / 'index.html').is_file():
        raise RuntimeError('Missing staged frontend')
    shutil.copytree(OLD, NEW)
    shutil.copytree(STAGE, NEW / 'dist', dirs_exist_ok=True)
    for directory, _, files in os.walk(NEW / 'dist'):
        os.chmod(directory, 0o755)
        for name in files:
            os.chmod(Path(directory) / name, 0o644)
    switch(NEW)
    try:
        for path in ('/practice', '/practice/5', '/practice/19', '/api/health'):
            result = subprocess.check_output([
                'curl', '-fksS', '--resolve', 'tournament.2048tables.online:443:127.0.0.1',
                'https://tournament.2048tables.online' + path,
            ])
            if path.startswith('/practice') and result != (NEW / 'dist/index.html').read_bytes():
                raise RuntimeError('Unexpected served frontend at ' + path)
    except BaseException:
        if (ROOT / 'current').resolve() == NEW:
            switch(OLD)
        raise
    print('Deployed:', NEW)


if __name__ == '__main__':
    main()
