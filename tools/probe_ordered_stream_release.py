import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tarfile

root = Path('/opt/2048tables')
result = {}
for name, folder, entries in [('main', root/'app/frontend/dist', ['index.html', 'live/index.html']),
                               ('tournament', root/'tournament/current/dist', ['index.html'])]:
    files = {}
    for entry in entries:
        files[entry] = hashlib.sha256((folder/entry).read_bytes()).hexdigest()
        for path in re.findall(r'(?:src|href)="(/assets/[^"?]+)', (folder/entry).read_text()):
            files[path] = hashlib.sha256((folder/path.lstrip('/')).read_bytes()).hexdigest()
    result[name] = files
result['backend_hashes'] = {}
for p in ['backend/live/competition_content.py', 'backend/live/dynamic_rooms.py', 'backend/live/routes.py']:
    result['backend_hashes'][p] = hashlib.sha256((root/'app'/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
with sqlite3.connect('file:/var/lib/2048tables/competition-test/competition.sqlite3?mode=ro', uri=True) as db:
    result['statuses'] = db.execute('SELECT status,count(*) FROM competitions GROUP BY status').fetchall()
    result['rooms'] = db.execute('SELECT room_code,status,updated_at FROM competitions').fetchall()
result['archive_sources'] = {}
with tarfile.open(root/'backups/20261001-checkpoint-2a27eef/verified-source.tar') as archive:
    for entry in archive:
        if entry.isfile() and (entry.name.startswith('frontend/') or entry.name.startswith('competition/') or entry.name.startswith('font/')):
            result['archive_sources'][entry.name] = hashlib.sha256(archive.extractfile(entry).read().replace(b'\r\n', b'\n')).hexdigest()
print(json.dumps(result))
