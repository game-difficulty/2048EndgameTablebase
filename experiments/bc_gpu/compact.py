"""Generate zero-pruned future fixtures using the production CPU BC compactor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from run import HERE

p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);args=p.parse_args()
root=args.dataset.resolve();meta=json.loads((root/'manifest.json').read_text());results=[]
env=os.environ.copy();env['PATH']='C:/Apps/mingw64/bin;'+env['PATH']
for s in range(meta['min_sum'],meta['max_sum']+5,2):
    cmd=[str(HERE/'build/native_fixture.exe'),'--compact',str(root),str(s),str(meta['target_rank']),str(meta['threads'])]
    result=json.loads(subprocess.check_output(cmd,env=env,text=True));results.append(result);print(result,flush=True)
(root/'compaction.json').write_text(json.dumps(results,indent=2))
