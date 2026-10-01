"""Independently read GPU cell files and compare deterministic old EX samples."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def read_cell(path,cid,meta):
    m=meta['cells'][cid]
    if not m['rows']:return None
    if m.get('layout')=='flat_cell_v1':
        with (path/f'{cid}.cell').open('rb') as f:
            keys=np.fromfile(f,np.uint64,m['buckets']);starts=np.fromfile(f,np.uint32,m['buckets']+1)
            words=np.fromfile(f,np.uint64,m['words']);values=np.fromfile(f,np.uint32,m['rows'])
    else:
        with np.load(path/f'{cid}.npz') as z:keys,starts,words,values=(z[k] for k in ('keys','starts','words','values'))
    rows=np.r_[np.uint64(0),np.cumsum(np.fromiter((int(w).bit_count() for w in words),dtype=np.uint32),dtype=np.uint64)]
    return keys,starts,words,values,rows

def encode(b,lut,mod):
    q=[sum(((b>>shift)&15)<<(4*j) for j,shift in enumerate(p)) for p in
       ((60,56,44,40),(52,48,36,32),(28,24,12,8),(20,16,4,0))]
    a,c,d,e=(list(map(int,lut[w])) for w in q)
    cid=((min(a[0]+c[0],d[0]+e[0])//2)%mod)*mod+(min(a[0]+d[0],c[0]+e[0])//2)%mod
    key=(q[0]<<48)|(c[1]<<32)|(d[1]<<16)|e[1];rank=(c[2]*d[3]+d[2])*e[3]+e[2]
    return cid,key,rank

def lookup(cell,key,rank):
    if cell is None:return 0
    keys,starts,words,values,rows=cell;bi=int(np.searchsorted(keys,np.uint64(key)))
    if bi==len(keys) or int(keys[bi])!=key:return 0
    wi=int(starts[bi])+rank//64;bit=rank%64;bits=int(words[wi])
    if not ((bits>>bit)&1):return 0
    ri=int(rows[wi])+(bits&((1<<bit)-1)).bit_count();return int(values[ri])

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--modulus',type=int,default=13)
    p.add_argument('--tolerance-raw',type=int,default=4000);p.add_argument('--reference',type=Path)
    p.add_argument('--output',default='sample_comparison.json');a=p.parse_args();root=a.root.resolve()
    reference=json.loads((a.reference or root/'old_ex_samples.json').read_text());lut=np.fromfile(root/'lut.bin',np.uint32).reshape(-1,4)
    layers=[];bad=0;worst=[];deltas=[];pairs=[]
    for layer in reference['layers']:
        step=layer['step'];path=root/'solved'/str(step);meta=json.loads((path/'meta.json').read_text());cache={};ds=[];absent=0
        for sample in layer['samples']:
            cid,key,rank=encode(int(sample['board'],16),lut,a.modulus)
            if cid not in cache:cache[cid]=read_cell(path,cid,meta)
            value=lookup(cache[cid],key,rank)
            pairs.append(dict(step=step,board=sample['board'],old_found=sample['found'],old_raw=sample['raw_value'],
                gpu_raw=value,abs_raw_delta=abs(value-sample['raw_value']) if sample['found'] else ''))
            if sample['found']:
                delta=abs(value-sample['raw_value']);ds.append(delta);deltas.append(delta)
                bad+=delta>a.tolerance_raw;worst.append(dict(step=step,board=sample['board'],gpu=value,old=sample['raw_value'],delta=delta))
            else:
                absent+=1;bad+=value>reference['threshold']*4000000000+a.tolerance_raw
        rec=dict(step=step,samples=len(layer['samples']),old_found=len(ds),old_threshold_absent=absent,
                 max_abs_raw=max(ds,default=0),mean_abs_raw=float(np.mean(ds)),p99_abs_raw=float(np.percentile(ds,99)))
        layers.append(rec);print(json.dumps(rec),flush=True)
    report=dict(sample_count=sum(x['samples'] for x in layers),old_found=len(deltas),bad_samples=bad,
        tolerance_raw=a.tolerance_raw,tolerance_probability=a.tolerance_raw/4000000000,
        max_abs_raw=max(deltas,default=0),max_abs_probability=max(deltas,default=0)/4000000000,
        layers=layers,worst=sorted(worst,key=lambda x:x['delta'],reverse=True)[:20],
        note='Old absence is checked against its archival 0.1 threshold; exact GPU futures only prune zero values.')
    (root/a.output).write_text(json.dumps(report,indent=2))
    with (root/(Path(a.output).stem+'_pairs.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    assert not bad,report
    print('PASS: independently decoded GPU values agree with the old CPU EX samples within the stated tolerance')
if __name__=='__main__':main()
