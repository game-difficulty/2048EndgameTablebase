"""Synthetic future-table replication to distinguish L2-fit from large-working-set behavior."""
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np
import cupy as cp
from run import HERE,timed_kernel

p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--sum',type=int,default=52)
p.add_argument('--repeats',type=int,default=9);args=p.parse_args();root=args.dataset.resolve()
meta=json.loads((root/'manifest.json').read_text());s=args.sum
boards=cp.asarray(np.fromfile(root/str(s)/'boards.bin',np.uint64));n=boards.size
reference=np.fromfile(root/str(s)/'values.bin',np.uint32)
lut=cp.asarray(np.fromfile(root/'lut.bin',np.uint32));moves=cp.asarray(np.fromfile(root/'moves.bin',np.uint32))
host=[];strides=[]
for fs in (s+2,s+4):
    for k,d in [('hash',np.uint8),('cells',np.uint32),('words',np.uint64),('bases',np.uint32),('values',np.uint32)]:
        a=np.fromfile(root/'solved'/str(fs)/(k+'.bin'),d);host.append(a);strides.append(a.size//16 if k=='hash' else a.size)
strides=cp.asarray(np.asarray(strides,np.uint64))
module=cp.RawModule(code=(HERE/'kernels.cu').read_text(),options=('--std=c++17','--fmad=false'))
kernel=module.get_function('solve_capacity');out=cp.empty(n,np.uint32)
num,den=float(meta['p4']).as_integer_ratio();bits=den.bit_length()-1;results=[]
for copies in (1,2,4,8,16,32,64,128):
    device=[cp.tile(cp.asarray(a),copies) for a in host];cp.cuda.Stream.null.synchronize()
    call=(boards,np.uint32(n),lut,moves,*device,np.uint32(meta['modulus']),np.uint32(meta['target_rank']),
          np.uint32(meta['terminal_value']),np.uint64(num),np.uint32(bits),strides,np.uint32(copies),out)
    timed_kernel(kernel,call,n)
    ts=[];t=time.perf_counter()
    for _ in range(args.repeats):ts.append(timed_kernel(kernel,call,n))
    wall=(time.perf_counter()-t)/args.repeats
    assert np.array_equal(out.get(),reference)
    results.append(dict(replicas=copies,future_bytes=sum(a.nbytes for a in device),
                        solve_seconds=statistics.median(ts),launch_sync_wall_seconds=wall,
                        mrows_s=n/statistics.median(ts)/1e6,exact=True))
    print(results[-1],flush=True);del device,call;cp.get_default_memory_pool().free_all_blocks()
report=dict(scope='synthetic independently replicated future tables, not new states or a free12 measurement',
            dataset=str(root),small_sum=s,rows=n,repeats=args.repeats,l2_bytes=cp.cuda.runtime.getDeviceProperties(0)['l2CacheSize'],results=results)
(root/'capacity.json').write_text(json.dumps(report,indent=2))
