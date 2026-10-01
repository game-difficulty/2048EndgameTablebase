"""Independent CPU primitive and rounding references, including boundary cases."""
import json
from pathlib import Path
import numpy as np
import cupy as cp
from run import HERE,timed_kernel

root=HERE/'data/probes'
fixture=HERE/'data/free8_32_p010'
module=cp.RawModule(code=(HERE/'kernels.cu').read_text(),options=('--std=c++17','--fmad=false'))
boards=np.fromfile(root/'primitive_inputs.bin',np.uint64)
expected=np.fromfile(root/'primitive_expected.bin',np.uint64)
actual=cp.empty(expected.size,np.uint64)
timed_kernel(module.get_function('primitives'),(cp.asarray(boards),np.uint32(len(boards)),
    cp.asarray(np.fromfile(fixture/'moves.bin',np.uint32)),actual),len(boards))
primitive_bad=int(np.count_nonzero(actual.get()!=expected))
assert primitive_bad==0,primitive_bad
s2=cp.asarray(np.fromfile(root/'s2.bin',np.uint64));s4=cp.asarray(np.fromfile(root/'s4.bin',np.uint64))
counts=cp.asarray(np.fromfile(root/'counts.bin',np.uint32));actual=cp.empty(len(counts),np.uint32)
probabilities=np.fromfile(root/'probabilities.bin',np.float64);results=[]
for k,p in enumerate(probabilities):
    num,den=float(p).as_integer_ratio();bits=den.bit_length()-1
    timed_kernel(module.get_function('rounding_probe'),(s2,s4,counts,np.uint32(len(counts)),
        np.uint64(num),np.uint32(bits),actual),len(counts))
    expected=np.fromfile(root/f'rounding_{k}.bin',np.uint32)
    bad=int(np.count_nonzero(actual.get()!=expected))
    results.append(dict(p4=float(p),cases=len(counts),mismatches=bad))
    print(results[-1],flush=True)
report=dict(primitive_boards=len(boards),primitive_outputs=len(boards)*5,
            primitive_mismatches=primitive_bad,rounding=results)
(root/'validation.json').write_text(json.dumps(report,indent=2))
assert all(r['mismatches']==0 for r in results)
print('PASS primitives and CPU80 rounding')
