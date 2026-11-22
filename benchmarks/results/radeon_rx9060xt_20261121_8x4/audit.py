#!/usr/bin/env python3
"""Provenance capture and final audit for the 2026-11-21 campaign."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv, gzip, hashlib, itertools, json, math, re, shutil, subprocess, sys

R=Path('/home/kawai/openmx4.0_gpu')
W=R/'work/codex_benchmark_20261121_8x4'
O=R/'benchmarks/results/radeon_rx9060xt_20261121_8x4'
U=Path('/home/kawai/openmx4.0')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def now():return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
sources=sorted((U/'work/input_example').glob('*.dat'))+sorted((U/'work/large_example').glob('*.dat'))+[R/'benchmarks/inputs/cghelgeo.dat',R/'benchmarks/inputs/sidia333_nc_cluster.dat']
files=set()
for p in sources:
    m=re.search(r'<Definition.of.Atomic.Species\s*\n(.*?)Definition.of.Atomic.Species>',p.read_text(),re.I|re.S);assert m,p
    for line in m[1].splitlines():
        v=line.split('#',1)[0].split()
        if len(v)>=3:
            files.add(U/'DFT_DATA19/PAO'/(v[1].split('-',1)[0]+'.pao'))
            files.add(U/'DFT_DATA19/VPS'/(v[2]+'.vps'))
data={str(p.relative_to(U/'DFT_DATA19')):sha(p) for p in sorted(files)}
if '--prepare' in sys.argv:
    save(O/'data-files.json',data)
    save(O/'original-inputs.json',[{'case':p.stem,'source':str(p),'sha256':sha(p)} for p in sources])
    (O/'original-inputs').mkdir(exist_ok=True)
    for p in sources:shutil.copy2(p,O/'original-inputs'/p.name)
    machine={}
    for name,cmd in [('cpu',['lscpu']),('memory',['free','-b']),('kernel',['uname','-a'])]:
        result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        machine[name]=result.stdout
    gpu=[]
    for card in Path('/sys/class/drm').glob('card[0-9]'):
        dev=card/'device'
        if not (dev/'vendor').exists():continue
        gpu.append({n:(dev/n).read_text().strip() for n in ['vendor','device','subsystem_vendor','subsystem_device','mem_info_vram_total'] if (dev/n).exists()})
    machine['gpu']=gpu;save(O/'machine.json',machine)
    shutil.copy2(__file__,O/'audit.py')
    print('Captured 32 original inputs and',len(data),'basis/pseudopotential hashes.')
    raise SystemExit(0)

queue=json.loads((W/'queue.json').read_text());assert 'controller_finished' in queue
jobs=queue['jobs'];assert len(jobs)==12 and all(j['status']=='completed' for j in jobs),[(j['test'],j['variant'],j['status']) for j in jobs]
runs=json.loads((O/'runs.json').read_text());assert len(runs)==96 and all(x['status']=='completed' for x in runs)
assert data==json.loads((O/'data-files.json').read_text()),'DFT data changed during measurement'
for src in json.loads((O/'original-inputs.json').read_text()):assert sha(Path(src['source']))==src['sha256']
def scientific(text):
    return '\n'.join(l for l in text.splitlines() if not re.match(r'^\s*(DATA\.PATH|scf\.eigen\.lib)\s+',l,re.I)).strip()
input_audit=[];execution=[];bundled=[];fallback=[]
for job in jobs:
    folder=Path(job['directory']);cmd=json.loads((folder/'command.json').read_text());binding=json.loads((folder/'rank-environment.json').read_text())
    assert binding['pass'] and len(binding['ranks'])==8
    assert job['rank_environment_verified'] and job['completed']==job['expected'] and job['normal_finish_markers']==job['expected']
    for item in cmd['inputs']:
        actual=folder/({'S':'input_example','L':'large_example'}[job['test']])/Path(item['source']).name if job['test'] in ('S','L') else folder/'input.dat'
        source=Path(item['source']);assert sha(source)==item['original_sha256'] and sha(actual)==item['input_sha256']
        assert scientific(source.read_text())==scientific(actual.read_text()),actual
        input_audit.append({'test':job['test'],'variant':job['variant'],'case':item['case'],'scientific_input_unchanged':True,'original_sha256':item['original_sha256'],'measurement_sha256':item['input_sha256']})
    libs=json.loads((folder/'runtime-libraries.json').read_text())
    if job['variant'].startswith('fork'):
        for library in ['libopenmx_core.so','libopenmx_host_simd.so']:
            assert any(str(W/'frozen/fork/lib'/library)==p for p in libs),(folder,library)
    execution.append({'test':job['test'],'variant':job['variant'],'bindings':binding,'libraries':libs})
    sections=(folder/'run.log').read_text(errors='replace').split('The calculation was normally finished.')
    for index,item in enumerate(cmd['inputs']):
        row=next(x for x in runs if x['test']==job['test'] and x['variant']==job['variant'] and x['case']==item['case'])
        history=re.findall(r'MD=\s*(\d+)\s+SCF=\s*(\d+)',sections[index])
        row['md_steps']=len({int(md) for md,scf in history})
        row['final_scf_iteration']=int(history[-1][1]) if history else None
    for row in [x for x in runs if x['test']==job['test'] and x['variant']==job['variant']]:
        assert math.isfinite(row['energy_Ha']) and all(map(math.isfinite,row['forces_Ha_bohr'])) and len(row['forces_Ha_bohr'])==3*len(row['atoms'])
        if 'bundled_reference' in row:bundled.append({'test':row['test'],'variant':row['variant'],'case':row['case'],**row['bundled_reference']})
    folder_out=O/'logs'/(job['test']+'-'+job['variant']);folder_out.mkdir(parents=True,exist_ok=True)
    for p in [folder/'run.log',*[folder/(x['system']+'.out') for x in cmd['inputs']]]:
        (folder_out/(p.name+'.gz')).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    for n in ['command.json','rank-environment.json','runtime-libraries.json']:
        shutil.copy2(folder/n,folder_out/n)
    if job['variant']=='fork-gpu':
        lines=(folder/'run.log').read_text(errors='replace').splitlines()
        candidates=[l.strip() for l in lines if re.search(r'fallback|falling back|using ELPA2',l,re.I)]
        dense_cpu=[l for l in candidates if ('elpa' in l.lower() or 'scalapack' in l.lower() or ('cpu' in l.lower() and re.search(r'dense|diagonal|eigensolver',l,re.I)))]
        construction=[l for l in candidates if l not in dense_cpu and 'cpu' in l.lower() and re.search(r'gemm|blas|matrix|projector|overlap|hamiltonian',l,re.I)]
        fp64_gpu=[l for l in candidates if l not in dense_cpu and 'cpu' not in l.lower() and 'fp64' in l.lower()]
        job['fallback_log_candidates']=candidates[:80]
        job['dense_cpu_fallback_lines']=dense_cpu[:80]
        job['dense_cpu_fallback_message_count']=len(dense_cpu)
        fallback.append({'test':job['test'],'gpu_accelerated_markers':job['gpu_accelerated_markers'],
                         'dense_cpu_fallback_message_count':len(dense_cpu),
                         'matrix_or_gemm_cpu_fallback_lines':sorted(set(construction)),
                         'gpu_fp64_fallback_lines':sorted(set(fp64_gpu)),
                         'dense_cpu_fallback_lines':job['dense_cpu_fallback_lines'],
                         'gpu_eigensolver_messages':sorted(set(l.strip() for l in lines if 'Solving the eigenvalue problem (GPU-accelerated,' in l)),
                         'gpu_eigensolver_message_count':sum('Solving the eigenvalue problem (GPU-accelerated,' in l for l in lines),
                         'fallback_messages':sorted(set(l.strip() for l in lines if 'fallback' in l.lower()))[:80]})
save(O/'run-results.json',jobs);save(W/'queue.json',queue)
save(O/'runs.json',runs)
with (O/'case-results.csv').open('w',newline='') as f:
    fields=['test','case','variant','status','seconds','total_computational_seconds','scf_steps','md_steps','final_scf_iteration','energy_Ha','scf_criterion','scf_maxIter']
    writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(runs)
save(O/'input-audit.json',input_audit);save(O/'execution-audit.json',execution);save(O/'bundled-reference-comparisons.json',bundled);save(O/'gpu-path-audit.json',fallback)
comparisons=json.loads((O/'comparisons.json').read_text());assert len(comparisons)==96
manifest=json.loads((O/'manifest.json').read_text());assert all(sha(W/'frozen'/p)==h for p,h in manifest['binaries'].items())
validation=json.loads((O/'validation.json').read_text());validation.update(final_audit_finished=now(),input_audit_pass=True,rank_and_library_audit_pass=True,
      unchanged_data_files=len(data),finite_force_components=sum(len(x['forces_Ha_bohr']) for x in runs),
      bundled_reference_passes=sum(x['pass'] for x in bundled),bundled_reference_comparisons=len(bundled))
save(O/'validation.json',validation)
names=['upstream-cpu','fork-cpu','fork-gpu']
titles={'S':'runtest (14 cases)','L':'runtestL (16 cases)','cghelgeo':'cghelgeo.dat','sidia333_nc_cluster':'sidia333_nc_cluster.dat'}
text=['# Radeon RX 9060 XT / Threadripper 3970X — 2026-11-21, 8 MPI × 4 OpenMP','',
      f"Measured {queue['started']} – {queue['controller_finished']} JST; revision `{manifest['revision']}`. One measurement per route and test; jobs ran sequentially. All eight MPI ranks were audited with four disjoint physical cores per rank, four OpenMP threads and four math-library threads (32 cores total, SMT disabled).",'',
      '| Route | Build / runtime |','|---|---|',
      '| Upstream CPU | Existing OpenMX 4.0 + official 4.0.1 patch; Intel icx/ifx 2026.0, Intel MPI 2021.18, threaded LP64 MKL, ELPA2. Executable hash matches the previously verified upstream build. |',
      '| Fork CPU | Release HIP build; ROCm core-10.1 Clang/Flang, Open MPI 5.0.9, AOCL 5.2 OpenMP, ELPA2; `OPENMX_GPU=0`. Host SIMD selects AVX2 on this CPU. |',
      '| Fork HIP | Same frozen executable/core/SIMD libraries; `OPENMX_GPU=1`, `scf.eigen.lib gpusolver`, `HSA_ENABLE_SDMA=0`; one gfx1200 Radeon RX 9060 XT, 16 GiB. NVIDIA MPS does not apply. |','',
      'Original geometry, grids, precision options, stdout, SCF/MD limits and convergence criteria are retained. Only `DATA.PATH` and `scf.eigen.lib` differ in measurement copies. Every job starts in a fresh directory without copied restart. All 44 basis/pseudopotential hashes stayed unchanged. Compiler, MPI and numerical-library differences are included in upstream comparisons.','',
      'Suite totals sum native per-case elapsed times; named inputs use maximum-rank `Total Computational Time`. These exclude MPI launch/shutdown; wall times are separately recorded. GPU-route totals include CPU work and any CPU fallback selected by the application.','',
      '| Test | Upstream CPU (s) | Fork CPU (s) | Fork HIP (s) | Upstream / CPU | Upstream / HIP | CPU / HIP |',
      '|---|---:|---:|---:|---:|---:|---:|']
for test in ['S','L','cghelgeo','sidia333_nc_cluster']:
    a,b,c=[next(j['seconds'] for j in jobs if j['test']==test and j['variant']==n) for n in names]
    text.append(f'| {titles[test]} | {a:.3f} | {b:.3f} | {c:.3f} | {a/b:.3f} | {a/c:.3f} | {b/c:.3f} |')
text+=['','All timings are single measurements. Ratios greater than one mean the denominator route is faster. Different SCF/MD counts can affect whole-input times. Case tables show total SCF steps [actual MD steps].','']
for test in ['S','L','cghelgeo','sidia333_nc_cluster']:
    text+=['## '+titles[test]+' case details','','| Input | Upstream CPU (s) | Fork CPU (s) | Fork HIP (s) | CPU / HIP | SCF [MD] steps (upstream / CPU / HIP) | Max pairwise ΔE (Ha) | Max pairwise ΔF (Ha/bohr) | Agreement |',
           '|---|---:|---:|---:|---:|---|---:|---:|---|']
    cases=sorted({x['case'] for x in runs if x['test']==test})
    for case in cases:
        rows=[next(x for x in runs if x['test']==test and x['case']==case and x['variant']==n) for n in names]
        a,b,c=[x['seconds'] for x in rows];cmp=[x for x in comparisons if x['test']==test and x['case']==case]
        de=max(x['delta_energy_Ha'] for x in cmp);df=max(x['max_delta_force_Ha_bohr'] for x in cmp)
        steps=' / '.join(str(x['scf_steps'])+' ['+str(x['md_steps'])+']' for x in rows)
        text.append(f"| {case} | {a:.3f} | {b:.3f} | {c:.3f} | {b/c:.3f} | {steps} | {de:.3e} | {df:.3e} | {'PASS' if all(x['pass'] for x in cmp) else 'FAIL'} |")
    text.append('')
text+=['## Numerical checks','',
       f"All **96 completed case outputs** contain finite energies and all **{validation['finite_force_components']:,} force components**. Rank binding, runtime libraries, input integrity and frozen-binary hashes passed. Pairwise comparisons: **{validation['comparison_passes']}/{validation['comparisons']} PASS**. Bundled S/L reference comparisons: **{validation['bundled_reference_passes']}/{validation['bundled_reference_comparisons']} PASS**.",'',
       'Agreement requires the same geometry/grids, absolute total-energy difference ≤1e-7 Ha and maximum force-component difference ≤1e-7 Ha/bohr. Completion markers alone do not establish PASS. All failed comparisons remain in the records. The original cghelgeo criterion is 1e-7; sidia333 retains criterion 1e-13 and its 25-SCF cap, so its capped timing does not establish convergence.','']
failed=[x for x in comparisons if not x['pass']]
if failed:
    text+=['Failed route-pair comparisons:','']
    for x in failed:text.append(f"- {x['test']} / {x['case']}, {x['left']} vs {x['right']}: |ΔE|={x['delta_energy_Ha']:.3e} Ha, max |ΔF|={x['max_delta_force_Ha_bohr']:.3e} Ha/bohr.")
    text.append('')
text+=['## Records','',
       '- `case-results.csv`, `run-results.csv`: all per-case computational/native times, MPI-job wall times, SCF counts and memory use.',
       '- `comparisons.csv` / JSON and `bundled-reference-comparisons.json`: route-pair and reference checks.',
       '- `manifest.json`, `original-inputs.json`, `data-files.json`, `input-audit.json`, `execution-audit.json`: source/build, inputs/data, environments, rank affinity and loaded libraries.',
       '- `gpu-path-audit.json`: GPU acceleration and fallback messages; matrix construction and dense diagonalization must be interpreted separately.',
       '- `logs/`: compressed timing logs and all final outputs, commands and execution audits.',
       '- `benchmark.py`, `audit.py`: reproduction and final-audit scripts.',
       '- Full raw calculations, restart files, frozen binaries and CPU/memory samples: `'+str(W)+'`.','']
(O/'README.md').write_text('\n'.join(text))
shutil.copy2(__file__,O/'audit.py')
print('Final audit PASS:',len(runs),'outputs,',validation['finite_force_components'],'finite force components.')
print('Numerical route pairs:',validation['comparison_passes'],'/',validation['comparisons'],'PASS')
for test in titles:
    print(test,[(j['variant'],round(j['seconds'],3)) for j in jobs if j['test']==test])
