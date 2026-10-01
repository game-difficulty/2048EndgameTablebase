"""FamilyChain GPU solve using exact compressed future cells and bounded temporaries.

Ascending family order visits each off-diagonal cell twice. The first visit
saves per-empty maxima; the second merges the other direction before reducing.
Two future windows may be fused; --split-spawn uses explicit partial/scratch.
"""
from __future__ import annotations
import argparse, collections, json, shutil, time
from pathlib import Path
import numpy as np
import cupy as cp
from stream import Runtime, Position, prefix, launch, scalar, sync, U32, U64

GIB=1024**3
def elapsed(t): sync();return time.perf_counter()-t
def load_meta(path):return json.loads((Path(path)/'meta.json').read_text())

def process_memory():
    import ctypes,sys
    if sys.platform!='win32':return {}
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
            ('peak_working_set','working_set','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
    c=Counters();c.cb=ctypes.sizeof(c);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    handle=ctypes.windll.kernel32.GetCurrentProcess()
    ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle),ctypes.byref(c),c.cb)
    return dict(host_working_set_bytes=c.working_set,host_peak_working_set_bytes=c.peak_working_set,
                host_private_commit_bytes=c.private,host_peak_commit_bytes=c.peak_pagefile) if ok else {}

class HostCellCache:
    """Bounded exact CPU-side staging cache, never a CPU calculation."""
    def __init__(self,budget):
        self.budget=budget;self.used=0;self.peak=0;self.data=collections.OrderedDict();self.hits=0
    def get(self,key):
        if key not in self.data:return None
        self.data.move_to_end(key);self.hits+=1;return self.data[key]
    def put(self,key,arrays):
        size=sum(a.nbytes for a in arrays.values())
        if size>self.budget:return
        if key in self.data:self.used-=sum(a.nbytes for a in self.data.pop(key).values())
        while self.used+size>self.budget:
            _,old=self.data.popitem(last=False);self.used-=sum(a.nbytes for a in old.values())
        for a in arrays.values():a.flags.writeable=False
        self.data[key]=arrays;self.used+=size;self.peak=max(self.peak,self.used)
    def prune(self,first,last):
        for key in list(self.data):
            if not first<=key[0]<=last:self.used-=sum(a.nbytes for a in self.data.pop(key).values())

class CellWriter:
    """At most two host cell buffers queued for disk, overlapping GPU computation."""
    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor
        self.executor=ThreadPoolExecutor(max_workers=1);self.pending=[];self.worker_seconds=0.;self.bytes=0
    def wait_slot(self):
        if len(self.pending)>=2:self.pending.pop(0).result()
    def _write(self,path,arrays):
        t=time.perf_counter()
        with path.open('wb') as f:
            for a in arrays.values():a.tofile(f)
        self.worker_seconds+=time.perf_counter()-t;self.bytes+=sum(a.nbytes for a in arrays.values())
    def submit(self,path,arrays):self.pending.append(self.executor.submit(self._write,path,arrays))
    def close(self):
        for future in self.pending:future.result()
        self.pending.clear();self.executor.shutdown(wait=True)

class TempStore:
    def __init__(self,path,budget,gpu_budget=0):
        self.path=Path(path);self.path.mkdir(exist_ok=True,parents=True);self.budget=budget
        self.hot={};self.used=0;self.peak=0;self.disk_written=0;self.disk_read=0;self.transfer=0
        self.device={};self.gpu_used=0;self.gpu_peak=0;self.gpu_budget=gpu_budget
    def put(self,key,array):
        if self.gpu_used+array.nbytes<=self.gpu_budget:
            self.device[key]=array;self.gpu_used+=array.nbytes;self.gpu_peak=max(self.gpu_peak,self.gpu_used);return
        host=array.get();self.transfer+=host.nbytes
        if self.used+host.nbytes<=self.budget:
            self.hot[key]=host;self.used+=host.nbytes;self.peak=max(self.peak,self.used)
        else:host.tofile(self.path/(key+'.bin'));self.disk_written+=host.nbytes
    def get(self,key,dtype,count):
        if key in self.device:
            out=self.device.pop(key);self.gpu_used-=out.nbytes;assert out.size==count;return out
        if key in self.hot:host=self.hot.pop(key);self.used-=host.nbytes
        else:
            p=self.path/(key+'.bin');host=np.fromfile(p,dtype);p.unlink();self.disk_read+=host.nbytes
        assert host.size==count,(key,host.size,count)
        self.transfer+=host.nbytes;return cp.asarray(host)
    def assert_empty(self):
        assert not self.hot and not self.device and not list(self.path.glob('*.bin')),'unconsumed temporary results'

class Cell:
    def __init__(self,rt,pos,values,hash_factor):
        self.pos=pos;self.values=values
        cap=1<<(max(4,int(pos.keys.size*hash_factor)+1)-1).bit_length()
        self.hash=cp.full((cap,2),U64(0xffffffffffffffff),U64)
        self.bases=cp.empty(pos.words.size,np.uint16)
        if pos.keys.size:rt.module.get_function('build_cell_index')((pos.keys.size,), (128,),
            (pos.keys,pos.starts,pos.rows,U32(pos.keys.size),self.hash,U32(cap-1),self.bases))
        self.desc=[self.hash.data.ptr,pos.words.data.ptr,self.bases.data.ptr,values.data.ptr,cap-1,pos.n]
    @property
    def nbytes(self):return self.pos.nbytes+self.values.nbytes+self.hash.nbytes+self.bases.nbytes

def save_cell(path,cid,pos,values,hostcache=None,writer=None):
    # Flat cell avoids ZIP CRC and redundant full-array copies in the hot I/O path.
    # Sizes are in the committed layer manifest; this is an experiment-only format.
    if writer is not None:writer.wait_slot()
    arrays={k:a.get() for k,a in zip(('keys','starts','words','values'),(pos.keys,pos.starts,pos.words,values))}
    target=Path(path)/(str(cid)+'.cell')
    if writer is not None:writer.submit(target,arrays)
    else:
        with target.open('wb') as f:
            for a in arrays.values():a.tofile(f)
    if hostcache is not None:hostcache.put((int(Path(path).name),cid),arrays)
    return dict(rows=pos.n,buckets=pos.keys.size,words=pos.words.size,layout='flat_cell_v1')

def read_cell_host(path,cid,meta):
    if meta.get('layout')=='flat_cell_v1':
        with (Path(path)/(str(cid)+'.cell')).open('rb') as f:
            arrays={k:np.fromfile(f,dtype,count) for k,dtype,count in (
                ('keys',U64,meta['buckets']),('starts',U32,meta['buckets']+1),
                ('words',U64,meta['words']),('values',U32,meta['rows']))}
        expected=8*meta['buckets']+4*(meta['buckets']+1)+8*meta['words']+4*meta['rows']
        assert sum(a.nbytes for a in arrays.values())==expected,'truncated experimental cell'
        return arrays
    with np.load(Path(path)/(str(cid)+'.npz')) as z:return {k:z[k] for k in z.files}

class Window:
    def __init__(self,rt,path,hash_factor=2.,hostcache=None):
        self.rt=rt;self.path=Path(path);self.meta=load_meta(path);self.factor=hash_factor
        self.cache=collections.OrderedDict();self.used=0;self.peak=0;self.loaded_bytes=0;self.loads=0;self.hits=0
        self.desc=cp.zeros((rt.mod**2,6),U64);self.seconds=0.
        self.hostcache=hostcache;self.disk_loaded_bytes=0;self.host_hits=0
    def estimate(self,cid):
        m=self.meta['cells'][cid];b=m['buckets'];w=m['words'];n=m['rows']
        cap=1<<(max(4,int(b*self.factor)+1)-1).bit_length()
        return 16*b+14*w+4*n+16*cap+16
    def need_bytes(self,ids):return sum(self.estimate(cid) for cid in ids)
    def trim(self,ids,budget):
        ids=set(ids);pending=sum(self.estimate(cid) for cid in ids if cid not in self.cache)
        for cid in list(self.cache):
            if self.used+pending<=budget:break
            if cid not in ids:self.used-=self.cache.pop(cid).nbytes
    def reset_stats(self):
        self.peak=self.used;self.loaded_bytes=0;self.loads=0;self.hits=0;self.seconds=0.
        self.disk_loaded_bytes=0;self.host_hits=0
    def ensure(self,ids,budget):
        t=time.perf_counter();ids=set(ids);needed=self.need_bytes(ids)
        if needed>budget:raise MemoryError(f'family future window requires {needed/GIB:.3f} GiB, budget {budget/GIB:.3f}')
        # Keep useful old cells while budget permits; never evict a required dependency.
        missing=[cid for cid in sorted(ids) if cid not in self.cache]
        pending=sum(self.estimate(cid) for cid in missing)
        self.trim(ids,budget)
        for cid in sorted(ids):
            if cid in self.cache:self.hits+=1;self.cache.move_to_end(cid);continue
            if self.meta['cells'][cid]['rows']:
                cachekey=(int(self.path.name),cid)
                arrays=self.hostcache.get(cachekey) if self.hostcache is not None else None
                if arrays is None:
                    arrays=read_cell_host(self.path,cid,self.meta['cells'][cid]);self.disk_loaded_bytes+=sum(v.nbytes for v in arrays.values())
                    if self.hostcache is not None:self.hostcache.put(cachekey,arrays)
                else:self.host_hits+=1
                pos=Position(self.rt,cp.asarray(arrays['keys']),cp.asarray(arrays['starts']),cp.asarray(arrays['words']))
                values=cp.asarray(arrays['values']);self.loaded_bytes+=sum(v.nbytes for v in arrays.values())
                del arrays
            else:
                pos=Position(self.rt,cp.empty(0,U64),cp.zeros(1,U32),cp.empty(0,U64));values=cp.empty(0,U32)
            cell=Cell(self.rt,pos,values,self.factor);self.cache[cid]=cell;self.used+=cell.nbytes;self.loads+=1
        self.peak=max(self.peak,self.used)
        d=np.zeros((self.rt.mod**2,6),U64)
        for cid in ids:d[cid]=self.cache[cid].desc
        self.desc.set(d);self.seconds+=elapsed(t)
    def drop(self):self.cache.clear();self.used=0

def targets(fid,spawn,half_sum,mod):
    families={fid,(fid+spawn)%mod,(half_sum-fid)%mod}
    return {r*mod+c for r in range(mod) for c in range(mod) if r in families or c in families}

def safe_remove(path,base):
    path=Path(path).resolve();base=Path(base).resolve()
    if path.parent!=base or not path.name.isdigit():raise ValueError('invalid experimental cleanup path')
    if path.exists():shutil.rmtree(path)

def init_terminal(rt,root,step):
    dst=root/'solved'/str(step);dst.mkdir(parents=True,exist_ok=True);cells=[]
    for cid in range(rt.mod**2):
        pos=Position.load(rt,root/'generated'/str(step),cid)
        if pos.n:
            values=cp.full(pos.n,U32(4000000000),U32);cells.append(save_cell(dst,cid,pos,values))
        else:cells.append(dict(rows=0,buckets=0,words=0))
    (dst/'meta.json').write_text(json.dumps(dict(cells=cells,rows=sum(c['rows'] for c in cells),terminal=True)))

def solve_layer(rt,root,step,args,rolling=None,hostcache=None):
    begin=time.perf_counter();source=root/'generated'/str(step);source_meta=load_meta(source)
    dst=root/'solved'/str(step);dst.mkdir(parents=True,exist_ok=True)
    windows={spawn:(rolling[step+spawn] if rolling is not None and step+spawn in rolling else
        Window(rt,root/'solved'/str(step+spawn),args.hash_factor,hostcache)) for spawn in (1,2)}
    retained_bytes=sum(w.used for w in windows.values())
    for w in windows.values():w.reset_stats()
    m=rt.mod
    half_sum=(16-args.free)*16384+args.free-1+step
    numerator,denom=float(args.p4).as_integer_ratio();bits=denom.bit_length()-1
    if bits>59:raise ValueError('unsupported probability')
    missing=cp.zeros(1,U32);cells=[None]*(m*m)
    writer=CellWriter() if getattr(args,'async_writes',False) else None
    fused=rt.module.get_function('family_fused');split=rt.module.get_function('family_pass')
    empties=rt.module.get_function('empty_counts');kernel_s=0.;read_s=0.;compact_s=0.;write_s=0.;temp_s=0.
    peak=0;reserved_peak=0;passes=0;partial_rows=0
    # Reserve for current cell, decoded batch, compact, temporary and CUDA context.
    cache_budget=int((args.device_gib-args.reserve_gib)*GIB)
    max_window=max(sum(windows[s].need_bytes(targets(fid,s,half_sum,m)) for s in (1,2)) for fid in range(m))
    if args.split_spawn:max_window=max(windows[s].need_bytes(targets(fid,s,half_sum,m)) for fid in range(m) for s in (1,2))
    gpu_temp_budget=min(int(getattr(args,'gpu_temp_gib',0)*GIB),max(0,cache_budget-max_window))
    cache_budget-=gpu_temp_budget
    temp=TempStore(root/'temp',int(args.temp_gib*GIB),gpu_temp_budget)
    source_bytes=16*source_meta['buckets']+12*source_meta['bitmap_words']+8
    t=time.perf_counter()
    source_cached=Position.load(rt,source) if getattr(args,'cache_current',False) and source_bytes<(args.reserve_gib*GIB-256*1024**2) else None
    read_s+=elapsed(t)
    phases=(2,1) if args.split_spawn else (0,)
    for phase in phases:
        if phase==1:windows[2].drop()
        for fid in range(m):
            ids={s:targets(fid,s,half_sum,m) for s in ((phase,) if phase else (1,2))}
            if phase:windows[phase].ensure(ids[phase],cache_budget)
            else:
                needs=[windows[s].need_bytes(ids[s]) for s in (1,2)]
                if sum(needs)>cache_budget:raise MemoryError('Both windows exceed budget; rerun this layer with --split-spawn')
                spare=(cache_budget-sum(needs))//2
                for s in (1,2):windows[s].trim(ids[s],needs[s-1]+spare)
                for s in (1,2):windows[s].ensure(ids[s],needs[s-1]+spare)
            source_ids=sorted({fid*m+c for c in range(m)}|{r*m+fid for r in range(m)})
            for cid in source_ids:
                a,b=source_meta['bucket_bounds'][cid:cid+2]
                if a==b:
                    cells[cid]=dict(rows=0,buckets=0,words=0);continue
                row,col=divmod(cid,m);mode=2 if row==col else (0 if fid==min(row,col) else 1)
                directions=3 if mode==2 else (1 if row==fid else 2)
                t=time.perf_counter();pos=source_cached.cell_view(cid,source_meta) if source_cached is not None else Position.load(rt,source,cid);read_s+=elapsed(t)
                completed=(phase in (0,1) and mode!=0)
                values=cp.empty(pos.n,U32) if completed else cp.empty(0,U32)
                for batch,(offset,boards) in enumerate(pos.batches(args.word_batch)):
                    counts=cp.empty(boards.size,U32);launch(empties,boards.size,(boards,U32(boards.size),counts));eo=prefix(counts)
                    count=scalar(eo[-1]);tag=f'{phase}_{cid}_{batch}'
                    t=time.perf_counter()
                    part=temp.get(tag,U32,count*(1 if phase else 2)) if mode==1 else cp.empty(count*(1 if phase else 2) if mode==0 else 0,U32)
                    if phase:
                        scratch=temp.get(f's_{cid}_{batch}',U64,boards.size) if phase==1 and mode else cp.empty(boards.size,U64)
                    out=values[offset:offset+boards.size] if completed else cp.empty(0,U32)
                    temp_s+=elapsed(t)
                    e0,e1=cp.cuda.Event(),cp.cuda.Event();e0.record()
                    if phase:
                        call=(boards,U32(boards.size),rt.lut,rt.moves,windows[phase].desc,U32(m),U32(args.target),U32(phase),
                            U32(directions),U32(mode),eo,part,scratch,out,U64(numerator),U32(bits),missing)
                        launch(split,boards.size,call,args.block)
                    else:
                        call=(boards,U32(boards.size),rt.lut,rt.moves,windows[1].desc,windows[2].desc,U32(m),U32(args.target),
                            U32(directions),U32(mode),eo,part,out,U64(numerator),U32(bits),missing)
                        launch(fused,boards.size,call,args.block)
                    e1.record();e1.synchronize();kernel_s+=cp.cuda.get_elapsed_time(e0,e1)/1000
                    t=time.perf_counter()
                    if mode==0:temp.put(tag,part);partial_rows+=boards.size
                    if phase==2 and mode:temp.put(f's_{cid}_{batch}',scratch)
                    temp_s+=elapsed(t);peak=max(peak,cp.get_default_memory_pool().used_bytes())
                    reserved_peak=max(reserved_peak,cp.get_default_memory_pool().total_bytes())
                    del call,part,out,counts,eo
                    if phase:del scratch
                if pos.n:del boards
                passes+=1
                if completed:
                    t=time.perf_counter();compact,v=pos.compact(values);compact_s+=elapsed(t)
                    t=time.perf_counter();cells[cid]=save_cell(dst,cid,compact,v,hostcache,writer);write_s+=time.perf_counter()-t
                    del compact,v
                del values,pos
            bad=scalar(missing[0])
            if bad:raise AssertionError(f'{bad} lookups escaped loaded family window step={step} fid={fid} phase={phase}')
            if cp.get_default_memory_pool().used_bytes()>args.device_gib*GIB:raise MemoryError('device live allocation budget exceeded')
    if writer is not None:
        t=time.perf_counter();writer.close();write_s+=time.perf_counter()-t
    temp.assert_empty();assert all(c is not None for c in cells)
    (dst/'meta.json').write_text(json.dumps(dict(cells=cells,rows=sum(c['rows'] for c in cells),step=step)))
    rec=dict(step=step,route='family_split' if args.split_spawn else 'family_fused',input_rows=source_meta['rows'],
        output_rows=sum(c['rows'] for c in cells),total_seconds=time.perf_counter()-begin,kernel_seconds=kernel_s,
        current_read_seconds=read_s,compact_seconds=compact_s,write_seconds=write_s,temp_seconds=temp_s,
        future_prepare_seconds=sum(w.seconds for w in windows.values()),future_loaded_bytes=sum(w.loaded_bytes for w in windows.values()),
        future_disk_loaded_bytes=sum(w.disk_loaded_bytes for w in windows.values()),future_host_cache_hits=sum(w.host_hits for w in windows.values()),
        future_cell_loads=sum(w.loads for w in windows.values()),future_cell_hits=sum(w.hits for w in windows.values()),
        partial_rows=partial_rows,temp_transfer_bytes=temp.transfer,temp_disk_write_bytes=temp.disk_written,temp_disk_read_bytes=temp.disk_read,
        temp_host_peak_bytes=temp.peak,temp_gpu_peak_bytes=temp.gpu_peak,gpu_temp_budget_bytes=gpu_temp_budget,
        current_cached=source_cached is not None,retained_future_bytes=retained_bytes,host_cell_cache_peak_bytes=hostcache.peak if hostcache else 0,
        device_peak_live_bytes=peak,device_budget_gib=args.device_gib,window_misses=scalar(missing[0]),passes=passes)
    rec.update(device_pool_reserved_peak_bytes=reserved_peak,device_pool_limit_bytes=cp.get_default_memory_pool().get_limit(),**process_memory())
    rec.update(async_writes=writer is not None,write_worker_seconds=writer.worker_seconds if writer else 0.)
    if rolling is not None:
        rolling.clear();rolling[step+1]=windows[1]
    return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--modulus',type=int,default=13)
    p.add_argument('--free',type=int,default=10);p.add_argument('--target',type=int,default=9);p.add_argument('--p4',type=float,default=.1)
    p.add_argument('--start',type=int,default=289);p.add_argument('--stop',type=int,default=0);p.add_argument('--init-terminal',action='store_true')
    p.add_argument('--device-gib',type=float,default=10.);p.add_argument('--reserve-gib',type=float,default=1.5)
    p.add_argument('--temp-gib',type=float,default=4.);p.add_argument('--split-spawn',action='store_true');p.add_argument('--keep-all',action='store_true')
    p.add_argument('--gpu-temp-gib',type=float,default=4.);p.add_argument('--cache-current',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--host-cache-gib',type=float,default=6.);p.add_argument('--clean-temp',action='store_true')
    p.add_argument('--async-writes',action='store_true')
    p.add_argument('--word-batch',type=int,default=16384);p.add_argument('--block',type=int,default=128);p.add_argument('--hash-factor',type=float,default=2.)
    args=p.parse_args();root=args.root.resolve()
    cp.get_default_memory_pool().set_limit(size=int(args.device_gib*GIB))
    rt=Runtime(root,args.modulus)
    if args.clean_temp:
        import re
        for f in (root/'temp').glob('*.bin'):
            if re.fullmatch(r'[012s]_\d+_\d+\.bin',f.name):f.unlink()
    if args.init_terminal:
        for step in (args.start+1,args.start+2):init_terminal(rt,root,step)
    report=root/'gpu_family.json';reports=json.loads(report.read_text()) if report.exists() else []
    rolling={};hostcache=HostCellCache(int(args.host_cache_gib*GIB))
    for step in range(args.start,args.stop-1,-1):
        rec=solve_layer(rt,root,step,args,rolling,hostcache);reports=[r for r in reports if r['step']!=step]+[rec]
        report.write_text(json.dumps(reports,indent=2));print(json.dumps(rec),flush=True)
        hostcache.prune(step,step+1)
        if not args.keep_all and step+2>5:safe_remove(root/'solved'/str(step+2),root/'solved')
        cp.get_default_memory_pool().free_all_blocks()
if __name__=='__main__':main()
