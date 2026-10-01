"""Independent, sampled Bellman consistency check of the existing EX layer 0.

Reads old layer 1/2 values; does not build or solve a CPU table. Moves and D4
are written directly in Python, independently of both CUDA kernels and LUTs.
"""
import argparse,functools,json,struct,sys,time
from pathlib import Path
import numpy as np
from compare_samples import read_cell,encode,lookup

def move(board,axis,reverse):
    out=0
    for line in range(4):
        positions=[line*4+j if axis==0 else j*4+line for j in range(4)]
        if reverse:positions.reverse()
        values=[(board>>(4*p))&15 for p in positions];values=[v for v in values if v]
        merged=[];i=0
        while i<len(values):
            if i+1<len(values) and values[i]==values[i+1] and values[i]!=15:
                merged.append(values[i]+1);i+=2
            else:merged.append(values[i]);i+=1
        for p,v in zip(positions,merged):out|=v<<(4*p)
    return out

@functools.lru_cache(maxsize=200000)
def canonical(board):
    variants=[]
    for trans in (False,True):
        for flipr in (False,True):
            for flipc in (False,True):
                b=0
                for r in range(4):
                    for c in range(4):
                        x,y=(c,r) if trans else (r,c)
                        if flipr:x=3-x
                        if flipc:y=3-y
                        b|=((board>>(4*(r*4+c)))&15)<<(4*(x*4+y))
                variants.append(b)
    return min(variants)

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--count',type=int,default=128);a=p.parse_args();root=a.root.resolve()
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'native_core'));import formation_core
    old=Path('C:/2048_tables/free10-512');reference=json.loads((root/'old_ex_samples.json').read_text())
    samples=reference['layers'][0]['samples'][:a.count];lut=np.fromfile(root/'lut.bin',np.uint32).reshape(-1,4)
    meta=json.loads((root/'solved'/'0'/'meta.json').read_text());cache={}
    @functools.lru_cache(maxsize=200000)
    def future(board,step):
        v=formation_core.lookup_ex_zbook_cold(str(old/f'free10_512_{step}.zbook'),str(old/'free10_512_.zlut'),board)
        return v['raw_value_bits'] if v['found'] else None
    num,unit=float(.1).as_integer_ratio();results=[];t=time.perf_counter()
    for sample in samples:
        board=int(sample['board'],16);sums=[0,0];upper=[0,0];empties=[p for p in range(16) if not ((board>>(4*p))&15)]
        for p in empties:
            for spawn in (1,2):
                spawned=board|(spawn<<(4*p));best=0;unknown=False
                for axis in (0,1):
                    for reverse in (False,True):
                        moved=move(spawned,axis,reverse)
                        if moved!=spawned:
                            v=future(canonical(moved),spawn)
                            if v is None:unknown=True
                            else:best=max(best,v)
                sums[spawn-1]+=best
                upper[spawn-1]+=max(best,400000000 if unknown else 0)
        expected=(sums[0]*(unit-num)+sums[1]*num)//(unit*len(empties)) if empties else 0
        expected_hi=(upper[0]*(unit-num)+upper[1]*num)//(unit*len(empties)) if empties else 0
        cid,key,rank=encode(board,lut,13)
        if cid not in cache:cache[cid]=read_cell(root/'solved'/'0',cid,meta)
        gpu=lookup(cache[cid],key,rank)
        results.append(dict(board=sample['board'],old_layer0=sample['raw_value'],old_future_bellman=expected,
                            old_future_bellman_upper=expected_hi,gpu=gpu,gpu_error=max(0,expected-gpu,gpu-expected_hi),
                            old_layer0_error=max(0,expected-sample['raw_value'],sample['raw_value']-expected_hi)))
    # Validate native cold reads against bytes from the actual stored value section.
    path=old/'free10_512_0.zbook';raw=path.read_bytes();h=struct.unpack('<8s6I9Q',raw[:104]);nb,small,words,n=h[9:13]
    offset=104+16*nb+small+8*words;values=np.frombuffer(raw,np.uint32,int(n),offset)
    for sample in samples:
        v=formation_core.lookup_ex_zbook_cold(str(path),str(old/'free10_512_.zlut'),int(sample['board'],16))
        assert int(values[v['global_dense_index']])==sample['raw_value']
    report=dict(samples=len(results),seconds=time.perf_counter()-t,stored_rows=len(values),stored_distinct_values=len(np.unique(values)),
        native_reader_matches_raw_value_section=True,max_gpu_bellman_error=max(x['gpu_error'] for x in results),
        inexact_bellman_intervals=sum(x['old_future_bellman']!=x['old_future_bellman_upper'] for x in results),
        max_old_layer0_bellman_error=max(x['old_layer0_error'] for x in results),
        old_layer0_inconsistent_samples=sum(x['old_layer0_error']>100 for x in results),rows=results,
        scope='Independent sampled one-step consistency check using old EX future values, not a CPU table recalculation. Missing archived future values are bounded by 0..0.1; a larger known direction max makes this interval exact.')
    (root/'initial_diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    assert report['max_gpu_bellman_error']<=32,report
if __name__=='__main__':main()
