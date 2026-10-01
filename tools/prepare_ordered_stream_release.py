"""Build an explicit committed overlay on the verified production snapshot."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = '2a27eefd9874d44bd131a5147e0071f1635a851b'
OVERLAYS = ['3681896', '130c487', '320662e']

def run(args, **kwargs):
    return subprocess.check_output(args, cwd=ROOT, **kwargs).decode().strip()

def main():
    stage = Path(tempfile.mkdtemp(prefix='2048-ordered-stream-'))
    archive = stage / 'source.tar'
    run(['git', 'archive', '--format=tar', '-o', str(archive), BASE])
    source = stage / 'source'
    source.mkdir()
    with tarfile.open(archive) as package:
        package.extractall(source, filter='data')
    for folder in ['frontend', 'competition/frontend']:
        os.symlink(ROOT / folder / 'node_modules', source / folder / 'node_modules', target_is_directory=True)
    env = dict(os.environ, GIT_DIR=str(ROOT / '.git'))
    for folder in ['frontend', 'competition/frontend']:
        subprocess.run(['npm.cmd', 'run', 'build'], cwd=source / folder, env=env, check=True)
    # Preserve baseline builds for comparison with the actual deployed bytes.
    (source / 'frontend/dist').rename(stage / 'baseline-main')
    (source / 'competition/frontend/dist').rename(stage / 'baseline-tournament')
    paths = set()
    for commit in OVERLAYS:
        paths.update(run(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit]).splitlines())
    for path in sorted(paths):
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(['git', 'show', '320662e:' + path], cwd=ROOT))
    for folder in ['frontend', 'competition/frontend']:
        subprocess.run(['npm.cmd', 'run', 'build'], cwd=source / folder, env=env, check=True)
    manifest = {'release': '20261001-ordered-stream-320662e', 'baseline': BASE,
                'revision': run(['git', 'rev-parse', '320662e']), 'overlays': OVERLAYS,
                'paths': sorted(paths), 'sha256': {p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in paths}}
    (source / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    with tarfile.open(stage / 'release.tar.gz', 'w:gz') as package:
        for path in sorted(paths):
            package.add(source / path, arcname=path)
        for folder in ['frontend/dist', 'competition/frontend/dist']:
            package.add(source / folder, arcname=folder)
        package.add(source / 'manifest.json', arcname='manifest.json')
        package.add(archive, arcname='baseline-source.tar')
    print('STAGE=' + str(stage), flush=True)

if __name__ == '__main__':
    main()
