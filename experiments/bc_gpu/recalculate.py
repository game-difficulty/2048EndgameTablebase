"""Reuse our generated positions; independently backsolve a new spawn probability.

Hard links are used only for immutable position/index inputs. Values and reports
are newly written in the destination, and an existing destination is refused.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--p4',type=float,default=.1);p.add_argument('--threads',type=int,default=16)
    args=p.parse_args();source=args.source.resolve();output=args.output.resolve()
    output.mkdir(exist_ok=False)
    meta=json.loads((source/'manifest.json').read_text());meta.update(p4=args.p4,threads=args.threads)
    for f in source.glob('*.bin'):os.link(f,output/f.name)
    for s in range(meta['min_sum'],meta['max_sum']+5,2):
        d=output/str(s);d.mkdir()
        for f in (source/str(s)).iterdir():
            if f.name!='values.bin':os.link(f,d/f.name)
        if s>meta['max_sum']:(d/'values.bin').write_bytes(b'')
    with (source/'cpu_layers.csv').open() as f:rows={int(r['small_sum']):r for r in csv.DictReader(f)}
    env=os.environ.copy();env['PATH']='C:/Apps/mingw64/bin;'+env['PATH']
    results=[];start=time.perf_counter()
    for s in range(meta['max_sum'],meta['min_sum']-1,-2):
        cmd=[str(HERE/'build/native_fixture.exe'),'--layer',str(output),str(s),str(meta['target_rank']),
             str(args.threads),str(args.p4),str(meta['modulus']),'1',str(output/str(s)/'values.bin')]
        result=json.loads(subprocess.check_output(cmd,env=env,text=True))
        vals=np.fromfile(output/str(s)/'values.bin',dtype=np.uint32)
        row=rows[s];row.update(solve_seconds=result['median_seconds'],index_seconds=result['load_index_seconds'],
            nonzero=int(np.count_nonzero(vals)),value_min=int(vals.min()) if len(vals) else 0,
            value_max=int(vals.max(initial=0)))
        results.append(row);print(s,result,flush=True)
    with (output/'cpu_layers.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
    meta['recalculation_seconds']=time.perf_counter()-start;meta['position_source']=str(source)
    (output/'manifest.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
