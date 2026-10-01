"""Validate new GPU index/compactor/family pipeline against existing CPU fixtures.

Does not calculate a new CPU solution. Uses the previously audited small data.
"""
import argparse,json,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cupy as cp
from stream import Runtime,Mutable,Position,U32,U64
from family import solve_layer,save_cell,load_meta,read_cell_host,HostCellCache,GIB

def main():
    p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);p.add_argument('--sums',type=int,nargs='+',default=[38,48,52,62])
    p.add_argument('--gpu-temp-gib',type=float,default=0.);p.add_argument('--cache-current',action='store_true')
    p.add_argument('--host-cache-gib',type=float,default=0.);p.add_argument('--temp-gib',type=float,default=0.)
    p.add_argument('--async-writes',action='store_true');a=p.parse_args()
    original=a.fixture.resolve();meta=load_meta(original) if (original/'meta.json').exists() else json.loads((original/'manifest.json').read_text())
    root=original/'stream_validation';root.mkdir(exist_ok=True)
    for name in ('lut.bin','moves.bin','unrank.bin','unrank_offsets.bin'):shutil.copyfile(original/name,root/name)
    rt=Runtime(root,meta['modulus']);results=[];hostcache=HostCellCache(int(a.host_cache_gib*GIB))
    for s in a.sums:
        step=(s-meta['min_sum'])//2
        for ss in (s,s+2,s+4):
            st=(ss-meta['min_sum'])//2;host_boards=np.fromfile(original/str(ss)/'boards.bin',U64)
            host_values=np.fromfile(original/str(ss)/'values.bin',U32)
            mutable=Mutable(rt);mutable.insert(cp.asarray(host_boards));pos=mutable.freeze();del mutable
            decoded=np.concatenate([b.get() for _,b in pos.batches()]) if pos.n else np.empty(0,U64)
            assert np.array_equal(decoded,host_boards),f'key/rank/decode ordering changed at {ss}'
            m=pos.save(root/'generated'/str(st));del pos
            if ss>s:
                dst=root/'solved'/str(st);dst.mkdir(parents=True,exist_ok=True);cells=[]
                for cid in range(rt.mod**2):
                    cell=Position.load(rt,root/'generated'/str(st),cid);v=cp.asarray(host_values[slice(*m['row_bounds'][cid:cid+2])])
                    compact,cv=cell.compact(v)
                    # Independent set/order validation of GPU position compaction.
                    actual=np.concatenate([b.get() for _,b in compact.batches()]) if compact.n else np.empty(0,U64)
                    lo,hi=m['row_bounds'][cid:cid+2]
                    assert np.array_equal(actual,host_boards[lo:hi][host_values[lo:hi]!=0])
                    assert np.array_equal(cv.get(),host_values[lo:hi][host_values[lo:hi]!=0])
                    cells.append(save_cell(dst,cid,compact,cv))
                (dst/'meta.json').write_text(json.dumps(dict(cells=cells,rows=sum(c['rows'] for c in cells))))
                del cell,v,compact,cv
        for split in (False,True):
            args=SimpleNamespace(hash_factor=2.,temp_gib=a.temp_gib,free=meta['free_cells'],p4=meta['p4'],device_gib=3.,reserve_gib=1.,
                split_spawn=split,word_batch=1024,block=128,target=meta['target_rank'],gpu_temp_gib=a.gpu_temp_gib,
                cache_current=a.cache_current,async_writes=a.async_writes)
            report=solve_layer(rt,root,step,args,hostcache=hostcache)
            refv=np.fromfile(original/str(s)/'values.bin',U32);refb=np.fromfile(original/str(s)/'boards.bin',U64)
            actualb=[];actualv=[]
            solved_meta=load_meta(root/'solved'/str(step))
            for cid in range(rt.mod**2):
                if not solved_meta['cells'][cid]['rows']:continue
                z=read_cell_host(root/'solved'/str(step),cid,solved_meta['cells'][cid])
                pos=Position(rt,cp.asarray(z['keys']),cp.asarray(z['starts']),cp.asarray(z['words']))
                actualb.extend(b.get() for _,b in pos.batches());actualv.append(z['values'])
            ab=np.concatenate(actualb) if actualb else np.empty(0,U64);av=np.concatenate(actualv) if actualv else np.empty(0,U32)
            assert np.array_equal(ab,refb[refv!=0]),f'family compact position differs sum={s} split={split}'
            delta=np.abs(av.astype(np.int64)-refv[refv!=0].astype(np.int64))
            assert not np.any(delta),f'family values differ sum={s} split={split}: {delta.max()}'
            report.update(sum=s,mismatches=0,max_abs_delta=0,positions_exact=True,forced_disk_temporaries=a.gpu_temp_gib==0)
            results.append(report);print(json.dumps(report),flush=True)
        (root/('validation_cached.json' if a.cache_current else 'validation.json')).write_text(json.dumps(results,indent=2))
    print('PASS: GPU index, compact, split scratch and fused partial chains match audited CPU values')
if __name__=='__main__':main()
