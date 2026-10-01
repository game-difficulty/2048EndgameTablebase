"""Preserve immutable completed GPU futures with hard links before rolling cleanup."""
import argparse,json,os,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--step',type=int,default=240)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose an empty snapshot directory; immutable snapshots are never overwritten')
    out.mkdir(exist_ok=True,parents=True)
    deadline=time.monotonic()+1800
    while not (root/'solved'/str(a.step+1)/'meta.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('GPU layer did not finish within 30 minutes')
        time.sleep(.25)
    def link_file(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():os.link(src,dst)
    for name in ('lut.bin','moves.bin','unrank.bin','unrank_offsets.bin'):link_file(root/name,out/name)
    for relative in (Path('generated')/str(a.step),Path('solved')/str(a.step+1),Path('solved')/str(a.step+2)):
        for f in (root/relative).iterdir():
            if f.is_file():link_file(f,out/relative/f.name)
    meta=dict(source=str(root),step=a.step,created=time.time(),immutable_hardlinks=True)
    (out/'snapshot.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
