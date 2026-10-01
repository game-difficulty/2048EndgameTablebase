"""Read-only deterministic sampling of existing EX results; never solve on CPU."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np

def board_at(key,rank,offsets,unrank,groups):
    nw=int(key)>>48;ne=(int(key)>>32)&65535;sw=(int(key)>>16)&65535;se=int(key)&65535
    nsw=int(groups[sw]);nse=int(groups[se]);q,rs=divmod(rank,nse);rn,rw=divmod(q,nsw)
    words=[nw,int(unrank[int(offsets[ne])+rn]),int(unrank[int(offsets[sw])+rw]),int(unrank[int(offsets[se])+rs])]
    out=0
    for w,positions in zip(words,[(60,56,44,40),(52,48,36,32),(28,24,12,8),(20,16,4,0)]):
        for j,shift in enumerate(positions):out|=((w>>(4*j))&15)<<shift
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--count',type=int,default=1024)
    p.add_argument('--old',type=Path,default=Path('C:/2048_tables/free10-512'))
    p.add_argument('--steps',type=int,nargs='+',default=list(range(6)));p.add_argument('--output',default='old_ex_samples.json')
    a=p.parse_args();root=a.root.resolve()
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'native_core'));import formation_core
    lut=np.fromfile(root/'lut.bin',np.uint32).reshape(-1,4);groups=np.zeros(65536,np.uint32)
    valid=lut[:,3]!=0;groups[lut[valid,1]]=lut[valid,3]
    offsets=np.fromfile(root/'unrank_offsets.bin',np.uint32);unrank=np.fromfile(root/'unrank.bin',np.uint16)
    rng=np.random.default_rng(20260923);report=[];t=time.perf_counter()
    for step in a.steps:
        path=root/'generated'/str(step);keys=np.load(path/'keys.npy');starts=np.load(path/'starts.npy');words=np.load(path/'words.npy')
        counts=np.fromiter((int(w).bit_count() for w in words),dtype=np.uint32,count=len(words))
        rows=np.r_[np.uint64(0),np.cumsum(counts,dtype=np.uint64)];n=int(rows[-1]);samples=[]
        for row in sorted(rng.choice(n,min(a.count,n),replace=False)):
            wi=int(np.searchsorted(rows,np.uint64(row),side='right')-1)
            bi=int(np.searchsorted(starts,np.uint32(wi),side='right')-1)
            bits=int(words[wi]);ordinal=int(row-rows[wi])
            for _ in range(ordinal):bits&=bits-1
            rank=(wi-int(starts[bi]))*64+(bits&-bits).bit_length()-1
            board=board_at(keys[bi],rank,offsets,unrank,groups)
            found=formation_core.lookup_ex_zbook_cold(str(a.old/f'free10_512_{step}.zbook'),str(a.old/'free10_512_.zlut'),board)
            samples.append(dict(board=f'{board:016x}',generated_row=int(row),found=found['found'],raw_value=found['raw_value_bits']))
        report.append(dict(step=step,generated_rows=n,samples=samples))
        print(step,'samples',len(samples),'old_found',sum(x['found'] for x in samples),flush=True)
    (root/a.output).write_text(json.dumps(dict(seed=20260923,old_directory=str(a.old),threshold=.1,
        seconds=time.perf_counter()-t,layers=report),indent=2))
if __name__=='__main__':main()
