"""Reproducible large-table route/memory and launch-geometry measurements.

Input is an immutable snapshot of GPU-generated positions and GPU future values.
This program never computes a CPU reference solution.
"""
import argparse,hashlib,json,statistics,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cupy as cp
from stream import Runtime,Position,U32,U64,launch,prefix,sync
from family import Window,targets,load_meta,solve_layer,HostCellCache,GIB

def fingerprint(path):
    meta=load_meta(path);h=hashlib.sha256()
    for cid,m in enumerate(meta['cells']):
        h.update(json.dumps(m,sort_keys=True).encode())
        if m['rows']:
            with (path/(str(cid)+'.cell')).open('rb') as f:
                while data:=f.read(8*1024**2):h.update(data)
    return h.hexdigest()

def kernels(root,step,rt):
    meta=load_meta(root/'generated'/str(step));mod=rt.mod
    fid=max(range(mod),key=lambda i:meta['row_bounds'][i*mod+i+1]-meta['row_bounds'][i*mod+i])
    cid=fid*mod+fid;pos=Position.load(rt,root/'generated'/str(step),cid)
    boards=cp.concatenate([b for _,b in pos.batches()]);out=cp.empty(boards.size,U32)
    counts=cp.empty(boards.size,U32);launch(rt.module.get_function('empty_counts'),boards.size,(boards,U32(boards.size),counts))
    active_rows=int(cp.count_nonzero(counts).item());mean_empty=float(cp.mean(counts).item())
    eo=prefix(counts);part=cp.empty(0,U32);num,den=float(.1).as_integer_ratio();bits=den.bit_length()-1
    reference=None;results=[]
    for factor in (2.,3.2):
        windows={s:Window(rt,root/'solved'/str(step+s),factor) for s in (1,2)}
        half=6*16384+9+step
        for s in (1,2):windows[s].ensure(targets(fid,s,half,mod),3.5*GIB)
        missing=cp.zeros(1,U32)
        for name in ('resident_cells','family_fused'):
            fn=rt.module.get_function(name)
            if name=='resident_cells':args=(boards,U32(boards.size),rt.lut,rt.moves,windows[1].desc,windows[2].desc,
                U32(mod),U32(9),U64(num),U32(bits),out,missing)
            else:args=(boards,U32(boards.size),rt.lut,rt.moves,windows[1].desc,windows[2].desc,
                U32(mod),U32(9),U32(3),U32(2),eo,part,out,U64(num),U32(bits),missing)
            for block in (64,128,256):
                # A single short launch was insufficient to stabilize Windows GPU
                # power/residency behavior after disk I/O. Warm by elapsed time.
                warm_start=time.perf_counter();warm_launches=0
                while time.perf_counter()-warm_start<.75:
                    launch(fn,boards.size,args,block);sync();warm_launches+=1
                warm_seconds=time.perf_counter()-warm_start;times=[];wall=time.perf_counter()
                for _ in range(9):
                    begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record();launch(fn,boards.size,args,block)
                    end.record();end.synchronize();times.append(cp.cuda.get_elapsed_time(begin,end)/1000)
                wall=(time.perf_counter()-wall)/9;actual=out.get()
                if reference is None:reference=actual
                assert np.array_equal(actual,reference);assert not int(missing[0].item())
                rec=dict(kernel=name,hash_factor=factor,block=block,rows=boards.size,cid=cid,repeats=9,
                    rows_with_empty=active_rows,mean_empty_count=mean_empty,
                    warmup_seconds=warm_seconds,warmup_launches=warm_launches,
                    median_seconds=statistics.median(times),mean_launch_sync_seconds=wall,
                    future_live_bytes=sum(w.used for w in windows.values()),kernel_attributes=fn.attributes,
                    values_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
                results.append(rec);print(json.dumps(rec),flush=True)
        del windows,args
    (root/'kernel_profile.json').write_text(json.dumps(results,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--step',type=int,default=166)
    p.add_argument('--kernels-only',action='store_true');p.add_argument('--routes-only',action='store_true');a=p.parse_args();root=a.root.resolve()
    cp.get_default_memory_pool().set_limit(size=10*GIB);rt=Runtime(root,13)
    if not a.routes_only:kernels(root,a.step,rt);cp.get_default_memory_pool().free_all_blocks()
    if a.kernels_only:return
    results=[];expected=None
    for budget,split,asynchronous in ((10.,False,False),(8.,False,False),(8.,True,False),(8.,False,True)):
        cp.get_default_memory_pool().free_all_blocks();cp.get_default_memory_pool().set_limit(size=int(budget*GIB))
        args=SimpleNamespace(hash_factor=2.,temp_gib=4.,free=10,p4=.1,device_gib=budget,reserve_gib=1.5,
            split_spawn=split,word_batch=16384,block=128,target=9,gpu_temp_gib=4.,cache_current=True,async_writes=asynchronous)
        cache=HostCellCache(6*GIB);started=time.time();rec=solve_layer(rt,root,a.step,args,hostcache=cache)
        rec.update(timestamp_start=started,timestamp_end=time.time())
        digest=fingerprint(root/'solved'/str(a.step));rec['output_sha256']=digest
        if expected is None:expected=digest
        assert digest==expected,'route changed exact positions or values'
        results.append(rec);print(json.dumps(rec),flush=True);del cache
    (root/'route_profile.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
