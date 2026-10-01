"""Bounded BC CUDA generation, compact positions and family window primitives.

Experimental container: keys.npy, starts.npy, words.npy, meta.json. BC key/rank
and modulo family semantics are unchanged; this is NOT a .bcpos file writer.
No CPU generation/solve is used after initial seed setup.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import cupy as cp

HERE=Path(__file__).resolve().parent
U32=np.uint32; U64=np.uint64
MAXKEY=U64(0xffffffffffffffff)

def sync(): cp.cuda.Stream.null.synchronize()
def launch(fn,n,args,block=128):
    if n: fn(((n+block-1)//block,), (block,), args)
def prefix(a):
    out=cp.empty(a.size+1,U32);out[0]=0
    cp.cumsum(a,dtype=U32,out=out[1:]);return out
def scalar(a): return int(a.item())

class Runtime:
    def __init__(self,root,mod=13):
        self.root=Path(root);self.mod=mod
        self.module=cp.RawModule(code=(HERE/'kernels.cu').read_text()+'\n'+(HERE/'stream.cu').read_text(),
                                 options=('--std=c++17','--fmad=false'))
        self.k={s:self.module.get_function(s) for s in ('insert_boards','generate_mutable','key_metadata',
            'gather_words','rehash_mutable','word_popcount','decode_words','compact_words','bucket_live')}
        host=np.fromfile(self.root/'lut.bin',U32).reshape(-1,4)
        self.lut=cp.asarray(host.ravel());g=np.zeros((65536,2),U32)
        valid=host[:,3]!=0;g[host[valid,1],0]=host[valid,0];g[host[valid,1],1]=host[valid,3]
        self.groups=cp.asarray(g.ravel())
        self.moves=cp.asarray(np.fromfile(self.root/'moves.bin',U32))
        self.offsets=cp.asarray(np.fromfile(self.root/'unrank_offsets.bin',U32))
        self.unrank=cp.asarray(np.fromfile(self.root/'unrank.bin',np.uint16))
    def metadata(self,keys):
        lens=cp.empty(keys.size,U32);cids=cp.empty_like(lens)
        launch(self.k['key_metadata'],keys.size,(keys,U32(keys.size),self.lut,self.groups,U32(self.mod),lens,cids))
        return lens,cids
    def rows(self,words):
        counts=cp.empty(words.size,U32)
        launch(self.k['word_popcount'],words.size,(words,U32(words.size),counts))
        if scalar(cp.sum(counts,dtype=U64))>=2**32:
            raise OverflowError('Experimental per-position row offsets are uint32; split this position into cells before loading')
        return prefix(counts)

class Position:
    def __init__(self,rt,keys,starts,words,cids=None):
        self.rt=rt;self.keys=keys;self.starts=starts;self.words=words
        self.rows=rt.rows(words);self.n=scalar(self.rows[-1])
        self.cids=rt.metadata(keys)[1] if cids is None else cids
    @property
    def nbytes(self):return sum(x.nbytes for x in (self.keys,self.starts,self.words,self.rows,self.cids))
    def cell_view(self,cid,meta):
        a,b=meta['bucket_bounds'][cid:cid+2]
        # Only small scalar metadata crosses to the host; bitmap and keys remain resident.
        w0,w1=map(int,self.starts[cp.asarray([a,b])].get())
        rb=meta['row_bounds'][cid];obj=object.__new__(Position)
        obj.rt=self.rt;obj.keys=self.keys[a:b];obj.starts=self.starts[a:b+1]-U32(w0)
        obj.words=self.words[w0:w1];obj.rows=self.rows[w0:w1+1]-U32(rb)
        obj.n=meta['row_bounds'][cid+1]-rb;obj.cids=self.cids[a:b]
        return obj
    def batches(self,max_words=16384):
        # At most max_words*64 boards, regardless of bitmap density.
        for first in range(0,self.words.size,max_words):
            nw=min(max_words,self.words.size-first)
            bounds=self.rows[cp.asarray([first,first+nw])].get();n=int(bounds[1]-bounds[0])
            if not n:continue
            boards=cp.empty(n,U64)
            launch(self.rt.k['decode_words'],nw,(self.keys,self.starts,U32(self.keys.size),self.words,
                self.rows,U32(first),U32(nw),self.rt.groups,self.rt.offsets,self.rt.unrank,boards))
            yield int(bounds[0]),boards
    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        # This is the host I/O boundary, not a full-board expansion.
        keys=self.keys.get();starts=self.starts.get();words=self.words.get();cids=self.cids.get()
        bounds=np.searchsorted(cids,np.arange(self.rt.mod**2+1)).astype(np.uint32)
        rowbounds=self.rows[cp.asarray(starts[bounds])].get()
        meta=dict(rows=self.n,buckets=len(keys),bitmap_words=len(words),modulus=self.rt.mod,
                  bucket_bounds=bounds.tolist(),row_bounds=rowbounds.tolist(),format='bc-gpu-experiment-v1')
        for name,a in [('keys',keys),('starts',starts),('words',words)]:np.save(path/(name+'.npy'),a)
        (path/'meta.json').write_text(json.dumps(meta))
        return meta
    @classmethod
    def load(cls,rt,path,cid=None):
        path=Path(path);meta=json.loads((path/'meta.json').read_text())
        keys=np.load(path/'keys.npy',mmap_mode='r');starts=np.load(path/'starts.npy',mmap_mode='r');words=np.load(path/'words.npy',mmap_mode='r')
        a,b=(0,len(keys)) if cid is None else meta['bucket_bounds'][cid:cid+2]
        w0,w1=int(starts[a]),int(starts[b])
        return cls(rt,cp.asarray(keys[a:b]),cp.asarray(starts[a:b+1]-w0),cp.asarray(words[w0:w1]))
    def compact(self,values):
        assert values.size==self.n
        words=cp.empty_like(self.words)
        launch(self.rt.k['compact_words'],words.size,(self.words,self.rows,values,U32(words.size),words))
        rows=self.rt.rows(words);live=cp.empty(self.keys.size,U32)
        launch(self.rt.k['bucket_live'],live.size,(self.starts,rows,U32(live.size),live))
        take=cp.nonzero(live)[0];lens=(self.starts[1:]-self.starts[:-1])[take];starts=prefix(lens)
        out=cp.empty(scalar(starts[-1]),U64)
        if take.size:self.rt.k['gather_words']((take.size,), (128,),
            (self.starts[take],words,starts,lens,U32(take.size),out))
        return Position(self.rt,self.keys[take],starts,out,self.cids[take]),values[values!=0]

class Mutable:
    def __init__(self,rt,hashcap=1<<18,wordcap=1<<20):
        self.rt=rt;self.hashcap=1<<(max(4,int(hashcap))-1).bit_length();self.wordcap=int(wordcap)
        self.keys=cp.full(self.hashcap,MAXKEY,U64);self.ptr=cp.full(self.hashcap,U32(0xffffffff),U32)
        self.words=cp.zeros(self.wordcap,U64);self.counters=cp.zeros(3,U32);self.retries=0
    @property
    def nbytes(self):return self.keys.nbytes+self.ptr.nbytes+self.words.nbytes
    def args(self):return (self.keys,self.ptr,self.words,self.counters,U32(self.hashcap-1),U32(self.wordcap))
    def grow(self,hashcap,wordcap):
        # Rebuild keys after an overflow, dropping incomplete allocations only.
        # Already inserted bits survive; callers replay the complete failed batch.
        oldkeys,oldptr=self.keys,self.ptr;oldcap=self.hashcap
        self.hashcap=int(hashcap);self.wordcap=int(wordcap)
        keys=cp.full(self.hashcap,MAXKEY,U64);ptr=cp.full(self.hashcap,U32(0xffffffff),U32)
        launch(self.rt.k['rehash_mutable'],oldcap,(oldkeys,oldptr,U32(oldcap),keys,ptr,U32(self.hashcap-1)))
        if self.words.size<wordcap:
            words=cp.zeros(wordcap,U64);words[:self.words.size]=self.words;self.words=words
        self.keys,self.ptr=keys,ptr
        valid=(oldkeys!=MAXKEY)&(oldptr<U32(0xfffffffe))
        # Failed allocations are at the end of the monotonic bitmap arena.
        self.counters[0]=cp.count_nonzero(valid).astype(U32)
        if scalar(self.counters[1])>self.words.size:raise RuntimeError('growth did not cover attempted allocations')
        self.counters[2]=0
    def insert(self,boards,spawn=0,target=9,check=True,only_terminal=False):
        kernel=self.rt.k['generate_mutable' if spawn else 'insert_boards']
        while True:
            if spawn:
                args=(boards,U32(boards.size),self.rt.lut,self.rt.moves,*self.args(),U32(spawn),U32(target),U32(check),U32(only_terminal))
            else:args=(boards,U32(boards.size),self.rt.lut,*self.args())
            launch(kernel,boards.size,args);counts=self.counters.get();nb,nw,error=map(int,counts)
            if error or nb>self.hashcap*.65 or nw>self.wordcap*.9:
                hc=self.hashcap*2 if (error&1 or nb>self.hashcap*.65) else self.hashcap
                wc=max(self.wordcap*2,nw*2) if (error&2 or nw>self.wordcap*.9) else self.wordcap
                self.grow(hc,wc)
                if error:self.retries+=1;continue
            return
    def freeze(self):
        take=cp.nonzero((self.keys!=MAXKEY)&(self.ptr<U32(0xfffffffe)))[0]
        keys=self.keys[take];lens,cids=self.rt.metadata(keys)
        order=cp.lexsort(cp.stack((keys,cids.astype(U64))));keys=keys[order];cids=cids[order];lens=lens[order]
        src=self.ptr[take[order]];starts=prefix(lens);words=cp.empty(scalar(starts[-1]),U64)
        if keys.size:self.rt.k['gather_words']((keys.size,), (128,), (src,self.words,starts,lens,U32(keys.size),words))
        return Position(self.rt,keys,starts,words,cids)

def generate(args):
    root=args.root.resolve();rt=Runtime(root,args.modulus);out=root/'generated';out.mkdir(exist_ok=True)
    rows=list(csv.DictReader(args.reference.open())) if args.reference else []
    expected={int(x['step']):int(x['primary_live']) for x in rows if x['stage'] in ('init','forward','forward_terminal')}
    report_path=root/'gpu_generate.json';reports=json.loads(report_path.read_text()) if report_path.exists() else []
    start=args.resume
    if start:
        current=Position.load(rt,out/str(start));carry=Mutable(rt)
        previous=Position.load(rt,out/str(start-1))
        for _,boards in previous.batches(args.word_batch):carry.insert(boards,2,args.target,start-1>args.docheck,start+1>=args.final_step)
        del previous,boards
    else:
        seed=Mutable(rt);seed.insert(cp.asarray(np.fromfile(root/'initial.bin',U64)))
        current=seed.freeze();del seed;current.save(out/'0');carry=Mutable(rt)
        if 0 in expected:assert current.n==expected[0],(current.n,expected[0])
    for step in range(start,min(args.final_step,args.stop_step)):
        begin=time.perf_counter();n=current.n;decode=0.;work=0.
        after=Mutable(rt)
        # Carry already contains the Spawn4 contribution from the preceding layer.
        for _,boards in current.batches(args.word_batch):
            sync();t=time.perf_counter()
            carry.insert(boards,1,args.target,step>args.docheck,step+1>=args.final_step)
            after.insert(boards,2,args.target,step>args.docheck,step+2>=args.final_step)
            work+=time.perf_counter()-t
        if n:del boards
        live=cp.get_default_memory_pool().used_bytes();t=time.perf_counter()
        nxt=carry.freeze();sync();freeze=time.perf_counter()-t
        del current,carry;current=nxt;carry=after;del nxt,after
        t=time.perf_counter();meta=current.save(out/str(step+1));write=time.perf_counter()-t
        total=time.perf_counter()-begin
        rec=dict(step=step+1,input_rows=n,rows=current.n,buckets=current.keys.size,
            bitmap_bytes=current.words.nbytes,work_seconds=work,freeze_seconds=freeze,
            write_seconds=write,total_seconds=total,live_device_bytes=live,
            device_reserved_bytes=cp.get_default_memory_pool().total_bytes(),expected_rows=expected.get(step+1))
        reports=[r for r in reports if r['step']!=step+1]+[rec]
        report_path.write_text(json.dumps(reports,indent=2))
        print(json.dumps(rec),flush=True)
        if step+1 in expected:assert current.n==expected[step+1],f'generation count differs at {step+1}: {current.n} != {expected[step+1]}'
        if step+1==args.final_step:
            tail=carry.freeze();tail.save(out/str(step+2));print('terminal secondary rows',tail.n,flush=True)
        cp.get_default_memory_pool().free_all_blocks()

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--modulus',type=int,default=13)
    p.add_argument('--target',type=int,default=9);p.add_argument('--docheck',type=int,default=247)
    p.add_argument('--final-step',type=int,default=290);p.add_argument('--stop-step',type=int,default=290)
    p.add_argument('--resume',type=int,default=0);p.add_argument('--word-batch',type=int,default=16384)
    p.add_argument('--reference',type=Path,default=Path('C:/2048_tables/free10-512/free10_512_zmask_generate_stats.csv'))
    generate(p.parse_args())
if __name__=='__main__':main()
