import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from build import EXE, HERE, ROOT, build, environment

parser = argparse.ArgumentParser()
parser.add_argument('--depths', nargs='+', type=int, default=[3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--seconds', type=float, default=5)
parser.add_argument('--nodes', type=int, default=20000000)
parser.add_argument('--board', default='330043598671da10')
parser.add_argument('--target', type=int, default=2048)
parser.add_argument('--cache-mib', type=int, default=32)
parser.add_argument('--label', default='baseline')
args = parser.parse_args()
if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.label):
    parser.error('label must contain only letters, digits, underscores and hyphens')
command = build()
out = HERE / 'results' / args.label
out.mkdir(parents=True, exist_ok=True)
metadata = dict(board=args.board, target=args.target, seconds_per_direction=args.seconds,
                nodes_per_direction=args.nodes, cache_mib=args.cache_mib, compiler=command, hashes={})
for path in [HERE / 'merge_probe.cpp', ROOT / 'native_core/include/BoardMover.h', ROOT / 'native_core/include/CommonMover.h']:
    metadata['hashes'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
(out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
with (out / 'raw.jsonl').open('w', encoding='utf-8') as stream:
    for depth in args.depths:
        result = subprocess.run([str(EXE), '--board', args.board, '--target', str(args.target),
                                 '--depth', str(depth), '--seconds', str(args.seconds), '--nodes', str(args.nodes),
                                 '--cache-mib', str(args.cache_mib)],
                                env=environment()[1], capture_output=True, text=True, check=True,
                                timeout=args.seconds * 4 + 60)
        row = json.loads(result.stdout)
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps(row), flush=True)
