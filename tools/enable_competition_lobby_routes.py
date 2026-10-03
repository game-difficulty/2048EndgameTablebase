"""Enable public competition lobby deep links in the strict production vhost."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


def add_lobby_routes(config: str) -> str:
    anchor = '    location = /practice { try_files /index.html =404; add_header Cache-Control no-cache; }'
    if config.count(anchor) != 1:
        raise ValueError('Expected exactly one practice entry in the tournament vhost')
    blocks = []
    for route in ('duels', 'time-attacks'):
        for suffix, body in (('', 'try_files /index.html =404; add_header Cache-Control no-cache;'),
                             ('/', f'return 308 /{route};')):
            prefix = f'location = /{route}{suffix} '
            if prefix not in config:
                blocks.append(f'    {prefix}{{ {body} }}')
    return config.replace(anchor, '\n'.join(blocks + [anchor])) if blocks else config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path('/etc/nginx/sites-available/tournament.2048tables.online'))
    parser.add_argument('--backup-dir', type=Path, required=True)
    args = parser.parse_args()
    before = args.config.read_text()
    after = add_lobby_routes(before)
    if before == after:
        print('Lobby routes already enabled')
        return
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir / 'tournament.nginx.previous'
    if backup.exists():
        raise RuntimeError('Refusing to overwrite an existing rollback configuration')
    shutil.copy2(args.config, backup)
    try:
        args.config.write_text(after)
        subprocess.run(['nginx', '-t'], check=True)
        subprocess.run(['systemctl', 'reload', 'nginx'], check=True)
    except BaseException:
        shutil.copy2(backup, args.config)
        subprocess.run(['nginx', '-t'], check=True)
        subprocess.run(['systemctl', 'reload', 'nginx'], check=True)
        raise
    print('Enabled /duels and /time-attacks, including trailing-slash redirects')


if __name__ == '__main__':
    main()
