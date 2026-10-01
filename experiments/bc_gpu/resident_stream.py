"""Full future residency with bounded current-cell output, for route comparison.

When two full future indexes fit, retaining them avoids FamilyChain partial I/O.
Current boards and values remain batched; no whole-layer decoded board array.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cupy as cp
from stream import Runtime,Position,launch,sync,U32,U64
from family import Window,HostCellCache,CellWriter,save_cell,load_meta,process_memory,GIB
from profile_family import fingerprint

def solve(rt,root,step,args):
    begin=time.perf_counter();m=rt.mod;meta=load_meta(root/'generated'/str(step));path=root/'solved'/str(step);path.mkdir(exist_ok=True,parents=True)
    hostcache=HostCellCache(6*GIB);windows={s:Window(rt,root/'solved'/str(step+s),2.,hostcache) for s in (1,2)}
    ids=set(range(m*m));sizes={s:windows[s].need_bytes(ids) for s in (1,2)}
    if sum(sizes.values())>(args.device_gib-1.5)*GIB:raise MemoryError('Full future residency does not fit the reserved budget; use family route')
    for s in (1,2):windows[s].ensure(ids,sizes[s])
    t=time.perf_counter();current=Position.load(rt,root/'generated'/str(step));sync();read_s=time.perf_counter()-t
    if current.nbytes>1.25*GIB:raise MemoryError('This resident comparison requires a compressed current layer below 1.25 GiB')
    writer=CellWriter() if args.async_writes else None;num,den=float(.1).as_integer_ratio();bits=den.bit_length()-1
    fn=rt.module.get_function('resident_cells');missing=cp.zeros(1,U32);cells=[];kernel_s=compact_s=write_s=0.;peak=reserved=0
    for cid in range(m*m):
        pos=current.cell_view(cid,meta)
        if not pos.n:cells.append(dict(rows=0,buckets=0,words=0));continue
        values=cp.empty(pos.n,U32)
        for offset,boards in pos.batches(16384):
            e0,e1=cp.cuda.Event(),cp.cuda.Event();e0.record()
            launch(fn,boards.size,(boards,U32(boards.size),rt.lut,rt.moves,windows[1].desc,windows[2].desc,
                U32(m),U32(9),U64(num),U32(bits),values[offset:offset+boards.size],missing),128)
            e1.record();e1.synchronize();kernel_s+=cp.cuda.get_elapsed_time(e0,e1)/1000
            peak=max(peak,cp.get_default_memory_pool().used_bytes());reserved=max(reserved,cp.get_default_memory_pool().total_bytes())
        t=time.perf_counter();compact,v=pos.compact(values);sync();compact_s+=time.perf_counter()-t
        t=time.perf_counter();cells.append(save_cell(path,cid,compact,v,hostcache,writer));write_s+=time.perf_counter()-t
        del boards,values,compact,v
    if writer is not None:
        t=time.perf_counter();writer.close();write_s+=time.perf_counter()-t
    assert int(missing[0].item())==0
    (path/'meta.json').write_text(json.dumps(dict(cells=cells,rows=sum(x['rows'] for x in cells),step=step)))
    return dict(step=step,route='full_futures_streamed_current',input_rows=meta['rows'],output_rows=sum(x['rows'] for x in cells),
        total_seconds=time.perf_counter()-begin,kernel_seconds=kernel_s,current_read_seconds=read_s,compact_seconds=compact_s,
        write_seconds=write_s,future_prepare_seconds=sum(w.seconds for w in windows.values()),future_loaded_bytes=sum(w.loaded_bytes for w in windows.values()),
        future_disk_loaded_bytes=sum(w.disk_loaded_bytes for w in windows.values()),full_future_device_bytes=sum(w.used for w in windows.values()),
        partial_transfer_bytes=0,window_misses=0,device_peak_live_bytes=peak,device_pool_reserved_peak_bytes=reserved,
        device_pool_limit_bytes=cp.get_default_memory_pool().get_limit(),async_writes=writer is not None,
        write_worker_seconds=writer.worker_seconds if writer else 0.,**process_memory())

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--step',type=int,default=166)
    p.add_argument('--device-gib',type=float,default=10.);p.add_argument('--async-writes',action='store_true');args=p.parse_args();root=args.root.resolve()
    cp.get_default_memory_pool().set_limit(size=int(args.device_gib*GIB));rt=Runtime(root,13)
    start=time.time();result=solve(rt,root,args.step,args);result.update(timestamp_start=start,timestamp_end=time.time())
    result['output_sha256']=fingerprint(root/'solved'/str(args.step))
    reference=json.loads((root/'route_profile.json').read_text())[0]['output_sha256'];assert result['output_sha256']==reference
    (root/'resident_profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
