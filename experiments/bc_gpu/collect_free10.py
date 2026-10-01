"""Audit full large-table coverage and publish compact experiment evidence."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve()
    old=Path('C:/2048_tables/free10-512');gen=read(root/'gpu_generate.json');solve=read(root/'gpu_family.json')
    samples=read(root/'sample_comparison.json');initial=read(root/'initial_diagnostic.json')
    profile_root=root.parent/'free10_profile166'
    routes=read(profile_root/'route_profile.json');resident=read(profile_root/'resident_profile.json');kernels=read(profile_root/'kernel_profile.json')
    assert len(routes)==4 and len(kernels)==12
    assert len({x['output_sha256'] for x in routes}|{resident['output_sha256']})==1
    assert len({x['values_sha256'] for x in kernels})==1
    assert all(x['warmup_seconds']>=.75 for x in kernels)
    assert all(x['device_pool_reserved_peak_bytes']<=x['device_pool_limit_bytes'] for x in routes+[resident])
    # Preserve direct-comparison failures; independently establish why layer 0
    # cannot be used as a numeric oracle. Never relabel its old values as matching.
    assert all(x['max_abs_raw']<=samples['tolerance_raw'] for x in samples['layers'] if x['step']!=0)
    assert initial['samples']==samples['layers'][0]['samples']==1024
    assert initial['max_gpu_bellman_error']<=32 and initial['inexact_bellman_intervals']==0
    assert initial['native_reader_matches_raw_value_section']
    assert initial['old_layer0_inconsistent_samples']==samples['bad_samples']
    assert sorted(x['step'] for x in gen)==list(range(1,291))
    assert sorted(x['step'] for x in solve)==list(range(290))
    with (old/'free10_512_zmask_generate_stats.csv').open() as f:cg=list(csv.DictReader(f))
    with (old/'free10_512_zmask_solve_stats.csv').open() as f:cs=list(csv.DictReader(f))
    gt=next(x for x in cg if x['stage']=='_total');st=next(x for x in cs if x['stage']=='_total')
    oldsolve={int(x['step']):x for x in cs if x['stage']=='solve'}
    oldgen={int(x['step']):x for x in cg if x['stage'] in ('init','forward','forward_terminal')}
    for x in gen:assert x['rows']==int(oldgen[x['step']]['primary_live'])
    assert read(root/'generated'/'0'/'meta.json')['rows']==int(oldgen[0]['primary_live'])
    assert read(root/'generated'/'291'/'meta.json')['rows']==int(oldgen[290]['secondary_live'])
    for x in solve:
        assert x['input_rows']==int(oldsolve[x['step']]['input_live'])
        assert x['window_misses']==0
        if x.get('device_pool_limit_bytes',0):assert x['device_pool_reserved_peak_bytes']<=x['device_pool_limit_bytes']
    generation_seconds=sum(x['total_seconds'] for x in gen);solve_seconds=sum(x['total_seconds'] for x in solve)
    changes=[dict(step=x['step'],gpu=x['output_rows'],old=int(oldsolve[x['step']]['output_live']),
        difference=x['output_rows']-int(oldsolve[x['step']]['output_live'])) for x in solve if x['output_rows']!=int(oldsolve[x['step']]['output_live'])]
    summary=dict(dataset='free10-512',initial_rows=17925,position_layers=292,solve_layers=290,
        input_rows=sum(x['input_rows'] for x in solve),generated_counts_all_match=True,
        gpu_generation_seconds=generation_seconds,old_generation_seconds=float(gt['total_seconds']),
        gpu_solve_seconds=solve_seconds,old_solve_seconds=float(st['total_seconds']),
        generation_speedup=float(gt['total_seconds'])/generation_seconds,solve_speedup=float(st['total_seconds'])/solve_seconds,
        combined_speedup=(float(gt['total_seconds'])+float(st['total_seconds']))/(generation_seconds+solve_seconds),
        solve_kernel_seconds=sum(x['kernel_seconds'] for x in solve),samples=samples,initial_diagnostic=initial,
        validation_status='Old layer 0 is inconsistent with its own future layers; layers 1..5 agree, and independent sampled layer-0 recurrence agrees with GPU.',
        generation_peak_live_bytes=max(x['live_device_bytes'] for x in gen),
        generation_peak_reserved_bytes=max(x['device_reserved_bytes'] for x in gen),
        solve_peak_live_bytes=max(x['device_peak_live_bytes'] for x in solve),
        solve_peak_pool_reserved_bytes=max(x.get('device_pool_reserved_peak_bytes',0) for x in solve),
        hard_pool_limited_layers=sum(bool(x.get('device_pool_limit_bytes',0)) for x in solve),
        host_peak_working_set_bytes=max(x.get('host_peak_working_set_bytes',0) for x in solve),
        host_peak_commit_bytes=max(x.get('host_peak_commit_bytes',0) for x in solve),
        nonzero_count_difference_layers=len(changes),nonzero_count_differences=changes,
        source_hashes_at_collection={f:digest(HERE/f) for f in ('native_fixture.cpp','kernels.cu','stream.cu','stream.py','family.py',
            'resident_stream.py','profile_family.py','validate_stream.py','sample_reference.py','compare_samples.py','diagnose_initial.py','collect_free10.py')},
        old_input_hashes={f:digest(old/f) for f in ('free10_512_config.txt','free10_512_zmask_generate_stats.csv','free10_512_zmask_solve_stats.csv')},
        limitations=['Staged development run; sums of completed step wall times, not one cold run of a frozen final version.',
                    'Buffered I/O versus old EX direct I/O; other tablebase jobs were running concurrently.',
                    'Full long-term compressed archive export excluded; only exact rolling futures, early layers and snapshots retained.',
                    'The final source includes optional features profiled separately; source hashes are collection-time provenance.'])
    dest=HERE/'results';dest.mkdir(exist_ok=True)
    (dest/'free10_summary.json').write_text(json.dumps(summary,indent=2))
    for src,name in [(root/'gpu_generate.json','free10_generate.json'),(root/'gpu_family.json','free10_solve.json'),
                     (root/'sample_comparison.json','free10_sample_comparison.json'),
                     (root/'sample_comparison_pairs.csv','free10_sample_pairs.csv'),
                     (root/'initial_diagnostic.json','free10_initial_diagnostic.json')]:shutil.copyfile(src,dest/name)
    for profile in ('166','240'):
        base=root.parent/f'free10_profile{profile}'
        for name in ('kernel_profile.json','route_profile.json','resident_profile.json','sample_comparison.json','gpu_monitor.csv'):
            if (base/name).exists():shutil.copyfile(base/name,dest/f'free10_profile{profile}_{name}')
    small=root.parent/'free8_32_p010'/'stream_validation'
    for name in ('validation.json','validation_cached.json'):
        if (small/name).exists():shutil.copyfile(small/name,dest/f'free10_{name}')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('samples','initial_diagnostic','nonzero_count_differences','source_hashes_at_collection','old_input_hashes')},indent=2))
if __name__=='__main__':main()
