"""Replay self-generated BC layers on CUDA, checking every value and decode.

Timings explicitly separate CUDA events, transfer wall time, and disk loading.
This first resident baseline exports static indexes built by the fixture tool.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
import cupy as cp

HERE = Path(__file__).resolve().parent
DTYPES = dict(hash=np.uint8, cells=np.uint32, words=np.uint64, bases=np.uint32,
              tasks=np.uint8, boards=np.uint64, values=np.uint32)

def timed_kernel(fn, args, n, block=128):
    if not n:
        return 0.0
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    begin.record()
    fn(((n + block - 1)//block,), (block,), args)
    end.record(); end.synchronize()
    return cp.cuda.get_elapsed_time(begin, end)/1000

def load(root, s):
    t = time.perf_counter()
    host = {k: np.fromfile(root/str(s)/(k+'.bin'), dtype=d) for k, d in DTYPES.items() if k not in ('boards','values')}
    disk = time.perf_counter()-t
    host.update({k:np.fromfile(root/str(s)/(k+'.bin'),dtype=DTYPES[k]) for k in ('boards','values')})
    t = time.perf_counter()
    device = {k: cp.asarray(a) for k, a in host.items() if k not in ('boards', 'values')}
    cp.cuda.Stream.null.synchronize()
    return host, device, disk, time.perf_counter()-t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--block', type=int, default=128)
    parser.add_argument('--rounding',choices=['cpu80','fp64'],default='cpu80')
    parser.add_argument('--compact-futures',action='store_true')
    args = parser.parse_args()
    root = args.dataset.resolve()
    meta = json.loads((root/'manifest.json').read_text())
    numerator,denominator=float(meta['p4']).as_integer_ratio()
    bits=denominator.bit_length()-1
    if not 0<=meta['p4']<=1 or bits>59:
        raise ValueError('Experimental CPU80 mode supports probability dyadics with denominator exponent <=59')
    with (root/'cpu_layers.csv').open() as f:
        cpu_layers = {int(row['small_sum']): row for row in csv.DictReader(f)}
    module = cp.RawModule(code=(HERE/'kernels.cu').read_text(), options=('--std=c++17', '--fmad=false'))
    decode, solve = module.get_function('decode'), module.get_function('solve')
    common = {k:cp.asarray(np.fromfile(root/(k+'.bin'),dtype=d)) for k,d in
              [('lut',np.uint32),('moves',np.uint32),('unrank_offsets',np.uint32),('unrank',np.uint16)]}
    cp.cuda.Stream.null.synchronize()
    futures = {}
    future_root=root/'solved' if args.compact_futures else root
    for s in (meta['max_sum']+2,meta['max_sum']+4):
        _,device,_,_=load(future_root,s);device.pop('tasks');device['values']=cp.empty(0,np.uint32);futures[s]=device
    results=[];mismatches=0;total_rows=0;peak_live=0
    for s in range(meta['max_sum'],meta['min_sum']-1,-2):
        host,dev,disk,upload=load(root,s)
        n=len(host['values']);nt=len(host['tasks'])//32
        boards=cp.empty(n,np.uint64);out=cp.empty(n,np.uint32)
        dargs=(dev['tasks'],np.uint32(nt),common['unrank_offsets'],common['unrank'],boards)
        decode_seconds=timed_kernel(decode,dargs,nt,args.block)
        actual_boards=boards.get()
        if not np.array_equal(actual_boards,host['boards']):
            raise AssertionError(f'GPU BC decode mismatch sum={s}')
        f2,f4=futures[s+2],futures[s+4]
        call_args=(boards,np.uint32(n),common['lut'],common['moves'],
                   *(f2[k] for k in ('hash','cells','words','bases','values')),
                   *(f4[k] for k in ('hash','cells','words','bases','values')),
                   np.uint32(meta['modulus']),np.uint32(meta['target_rank']),
                   np.uint32(meta['terminal_value']),np.float64(meta['p4']),np.uint64(numerator),
                   np.uint32(bits),np.uint32(args.rounding=='cpu80'),out)
        out.fill(np.uint32(0xdeadbeef))
        timed_kernel(solve,call_args,n,args.block)  # warm-up, excluded
        wall_start=time.perf_counter()
        timings=[timed_kernel(solve,call_args,n,args.block) for _ in range(args.repeats)]
        launch_sync_wall=(time.perf_counter()-wall_start)/args.repeats
        t=time.perf_counter();actual=out.get();download=time.perf_counter()-t
        delta=np.abs(actual.astype(np.int64)-host['values'].astype(np.int64))
        bad=int(np.count_nonzero(delta));mismatches+=bad;total_rows+=n
        nonterminal=n-int(cpu_layers[s]['terminal'])
        kernel=statistics.median(timings)
        live=sum(a.nbytes for a in common.values())+sum(a.nbytes for a in dev.values())+boards.nbytes+out.nbytes
        live+=sum(a.nbytes for f in (f2,f4) for a in f.values())
        peak_live=max(peak_live,live)
        results.append(dict(small_sum=s,rows=n,nonterminal_rows=nonterminal,
            mismatches=bad,max_abs_delta=int(delta.max(initial=0)),decode_seconds=decode_seconds,
            solve_seconds=kernel,upload_seconds=upload,download_seconds=download,disk_read_seconds=disk,
            launch_sync_wall_seconds=launch_sync_wall,
            resident_mrows_s=n/kernel/1e6 if kernel else 0,
            rolling_transfer_mrows_s=n/(upload+decode_seconds+kernel+download)/1e6 if n else 0,
            live_device_bytes=live,values_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
        print(f'sum={s} rows={n} mismatches={bad} max_delta={delta.max(initial=0)} '
              f'kernel_Mrows/s={results[-1]["resident_mrows_s"]:.3f}',flush=True)
        # Chained GPU values are the ONLY values for the next layer, never CPU oracle values.
        if args.compact_futures:
            compact_host,compact_dev,_,_=load(future_root,s);compact_dev.pop('tasks')
            keep=out!=0;compact_values=out[keep]
            assert np.array_equal(compact_values.get(),compact_host['values']),f'compact values mismatch sum={s}'
            assert np.array_equal(boards[keep].get(),compact_host['boards']),f'compact board set/order mismatch sum={s}'
            compact_dev['values']=compact_values;futures[s]=compact_dev
            del compact_host,compact_dev,compact_values,keep
        else:
            dev.pop('tasks');dev['values']=out;futures[s]=dev
        del futures[s+4]
        del f2,f4,boards,call_args,dargs,out,dev,host,actual_boards,actual
    prop=cp.cuda.runtime.getDeviceProperties(0)
    report=dict(dataset=str(root),fixture=meta,gpu=prop['name'].decode(),cupy=cp.__version__,
                block=args.block,repeats=args.repeats,rounding=args.rounding,rows=total_rows,mismatches=mismatches,
                peak_live_device_bytes=peak_live,all_values_exact=(mismatches==0),
                compact_futures=args.compact_futures,
                scope='resident UInt32, generated full reachable layers, GPU chained backsolve; sparse mode verifies GPU keep/value stream against CPU compact positions; no GPU position compaction or family streaming',
                layers=results)
    suffix='_sparse' if args.compact_futures else ''
    output=root/f'gpu_{args.rounding}_block{args.block}{suffix}.json';output.write_text(json.dumps(report,indent=2))
    with (root/f'gpu_{args.rounding}_block{args.block}{suffix}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
    if mismatches:
        raise SystemExit(f'{mismatches} mismatches; see {output}')
    print(f'PASS: {total_rows} chained values exact; report={output}')

if __name__=='__main__':main()
