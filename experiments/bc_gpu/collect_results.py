"""Audit generated evidence and publish small, versionable result artifacts."""
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import cupy as cp
from run import HERE

out=HERE/'results';out.mkdir(exist_ok=True)
cases=[('free6_8_p025','gpu_cpu80_block128.json'),
       ('free8_32_p025','gpu_cpu80_block128.json'),
       ('free8_32_p010','gpu_cpu80_block128.json'),
       ('free8_32_p010','gpu_cpu80_block128_sparse.json'),
       ('free9_16_p010','gpu_cpu80_block128.json')]
checks=[]
for case,name in cases:
    root=HERE/'data'/case
    report=json.loads((root/name).read_text());meta=json.loads((root/'manifest.json').read_text())
    assert report['all_values_exact'] and report['mismatches']==0
    assert report['rows']==meta['generated_rows']==sum(x['rows'] for x in report['layers'])
    assert {x['small_sum'] for x in report['layers']}==set(range(meta['min_sum'],meta['max_sum']+1,2))
    for layer in report['layers']:
        assert layer['mismatches']==layer['max_abs_delta']==0
        values=(root/str(layer['small_sum'])/'values.bin').read_bytes()
        assert len(values)==layer['rows']*4
        assert hashlib.sha256(values).hexdigest()==layer['values_sha256']
    checks.append(dict(case=case,report=name,rows=report['rows'],all_cpu_value_hashes_match=True))
    shutil.copyfile(root/name,out/f'{case}_{name}')
    shutil.copyfile(root/'manifest.json',out/f'{case}_manifest.json')
    shutil.copyfile(root/'cpu_layers.csv',out/f'{case}_cpu_layers.csv')
for case,name in [('free8_32_p010','benchmark.json'),('free8_32_p010','benchmark_sparse.json'),('free9_16_p010','benchmark.json')]:
    root=HERE/'data'/case;report=json.loads((root/name).read_text())
    assert report['repeats']>=9
    assert all(g['exact'] for r in report['layers'] for g in r['gpu'].values())
    shutil.copyfile(root/name,out/f'{case}_{name}')
probe=json.loads((HERE/'data/probes/validation.json').read_text())
assert probe['primitive_mismatches']==0 and all(r['mismatches']==0 for r in probe['rounding'])
shutil.copyfile(HERE/'data/probes/validation.json',out/'primitives_and_rounding.json')
capacity=json.loads((HERE/'data/free8_32_p010/capacity.json').read_text())
assert all(r['exact'] for r in capacity['results'])
assert capacity['results'][-1]['future_bytes']>4*1024**3
shutil.copyfile(HERE/'data/free8_32_p010/capacity.json',out/'capacity.json')
repo=HERE.parent.parent
sources=list(HERE.glob('*.py'))+list(HERE.glob('*.cpp'))+list(HERE.glob('*.cu'))
sources += [repo/'native_core/include'/s for s in ('BCResidentSolve.h','BCFutureSuccessLookup.h','BCSolveEdgeKernel.h',
    'BCKeyRank.h','BCCellBuilder.h','BCBoardCodec.h','BCLut.h','BoardMover.h','Calculator.h','BCPositionFile.h')]
sources.append(repo/'native_core/src/CanonicalBatch.cpp')
source_hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
environment=dict(created_utc=datetime.now(timezone.utc).isoformat(),gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
    driver_version=cp.cuda.runtime.driverGetVersion(),runtime_version=cp.cuda.runtime.runtimeGetVersion(),
    cupy=cp.__version__,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
    source_sha256=source_hashes,checks=checks,primitive_and_rounding_pass=True,
    cpu='AMD Ryzen 9 9950X, 16 cores / 32 logical processors',
    caveats=['resident scalar UInt32 only','no GPU generation or family streaming',
             'prepared-index cold path uses OS file cache; not an SSD throughput test',
             'CPU80 arithmetic assumes 64-bit long-double significand and p4 denominator exponent <=59',
             'memory estimate is core live buffers, not whole CUDA context or allocator reserve'])
(out/'audit.json').write_text(json.dumps(environment,indent=2))
print('PASS: full-layer coverage, all CPU value hashes, benchmark validations, primitive and rounding evidence')
