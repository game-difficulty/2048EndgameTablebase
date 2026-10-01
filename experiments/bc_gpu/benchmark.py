"""Repeated CPU production BC vs CUDA resident and prepared-index cold-load baselines.

Run without the generator or other benchmark jobs concurrently. Cold GPU paths
read our generated exported device layout; CPU builds its production index from
.bcpos. These are explicitly different preparation paths, not whole-job speedups.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
import cupy as cp
from run import HERE,timed_kernel

def array(path,dtype):return np.fromfile(path,dtype=dtype)
def future(root,s):
    return {k:cp.asarray(array(root/str(s)/(k+'.bin'),d)) for k,d in
            [('hash',np.uint8),('cells',np.uint32),('words',np.uint64),('bases',np.uint32),('values',np.uint32)]}

def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path)
    p.add_argument('--sums',type=int,nargs='+',default=[38,48,52,62]);p.add_argument('--repeats',type=int,default=9)
    p.add_argument('--compact-futures',action='store_true')
    args=p.parse_args();root=args.dataset.resolve();meta=json.loads((root/'manifest.json').read_text())
    env=os.environ.copy();env['PATH']='C:/Apps/mingw64/bin;'+env['PATH']
    future_root=root/'solved' if args.compact_futures else root
    results=[]
    for s in args.sums:
        reference=array(root/str(s)/'values.bin',np.uint32)
        row=dict(small_sum=s,rows=len(reference),cpu={})
        for threads in (1,16,32):
            out=HERE/'build/benchmark_values.bin'
            cmd=[str(HERE/'build/native_fixture.exe'),'--layer-sparse' if args.compact_futures else '--layer',str(root),str(s),str(meta['target_rank']),
                 str(threads),str(meta['p4']),str(meta['modulus']),str(args.repeats),str(out)]
            result=json.loads(subprocess.check_output(cmd,env=env,text=True))
            assert np.array_equal(array(out,np.uint32),reference)
            row['cpu'][str(threads)]=result
            print('cpu',s,threads,result,flush=True)
        results.append(row)
    module=cp.RawModule(code=(HERE/'kernels.cu').read_text(),options=('--std=c++17','--fmad=false'))
    decode,solve=module.get_function('decode'),module.get_function('solve')
    common={k:cp.asarray(array(root/(k+'.bin'),d)) for k,d in
            [('lut',np.uint32),('moves',np.uint32),('unrank_offsets',np.uint32),('unrank',np.uint16)]}
    num,den=float(meta['p4']).as_integer_ratio();bits=den.bit_length()-1
    for row in results:
        s=row['small_sum'];n=row['rows'];reference=array(root/str(s)/'values.bin',np.uint32)
        expected_boards=array(root/str(s)/'boards.bin',np.uint64)
        def prepare():
            tasks=cp.asarray(array(root/str(s)/'tasks.bin',np.uint8));nt=tasks.size//32
            f2,f4=future(future_root,s+2),future(future_root,s+4)
            boards=cp.empty(n,np.uint64);out=cp.full(n,0xdeadbeef,np.uint32)
            da=(tasks,np.uint32(nt),common['unrank_offsets'],common['unrank'],boards)
            sa=(boards,np.uint32(n),common['lut'],common['moves'],
                *(f2[k] for k in ('hash','cells','words','bases','values')),
                *(f4[k] for k in ('hash','cells','words','bases','values')),
                np.uint32(meta['modulus']),np.uint32(meta['target_rank']),np.uint32(meta['terminal_value']),
                np.float64(meta['p4']),np.uint64(num),np.uint32(bits),np.uint32(1),out)
            return da,sa,boards,out,nt
        da,sa,boards,out,nt=prepare();cp.cuda.Stream.null.synchronize()
        row['gpu']={}
        for block in (64,128,256):
            timed_kernel(decode,da,nt,block);timed_kernel(solve,sa,n,block)
            ds=[];ss=[];wall=[]
            for _ in range(args.repeats):
                t=time.perf_counter();ds.append(timed_kernel(decode,da,nt,block));ss.append(timed_kernel(solve,sa,n,block));wall.append(time.perf_counter()-t)
            assert np.array_equal(boards.get(),expected_boards)
            assert np.array_equal(out.get(),reference)
            row['gpu'][str(block)]=dict(decode_seconds=statistics.median(ds),solve_seconds=statistics.median(ss),
                                      launch_sync_wall_seconds=statistics.median(wall),exact=True)
        del da,sa,boards,out
        cold=[]
        for _ in range(3):
            cp.get_default_memory_pool().free_all_blocks()
            t=time.perf_counter();da,sa,boards,out,nt=prepare();cp.cuda.Stream.null.synchronize()
            prepared=time.perf_counter()-t
            timed_kernel(decode,da,nt,128);timed_kernel(solve,sa,n,128);actual=out.get()
            cold.append(dict(prepared_disk_upload_seconds=prepared,total_seconds=time.perf_counter()-t))
            assert np.array_equal(actual,reference)
            del da,sa,boards,out,actual
        row['gpu_prepared_cold_repeats']=cold
        print('gpu',s,row['gpu'],cold,flush=True)
    report=dict(dataset=str(root),gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        repeats=args.repeats,rounding='cpu80',compact_futures=args.compact_futures,
        timing_scope='CPU production resident raw solve; GPU decode+resident solve; cold GPU uses prebuilt exported indexes and warm OS file cache',layers=results)
    (root/('benchmark_sparse.json' if args.compact_futures else 'benchmark.json')).write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
