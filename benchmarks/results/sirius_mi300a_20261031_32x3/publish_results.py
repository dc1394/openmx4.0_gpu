#!/usr/bin/env python3
"""Retain compact, reviewable results only after all requested runs finish."""
import csv
import json
import re
from pathlib import Path
import shutil
import subprocess

B=Path(__file__).resolve().parent
ROOT=B.parents[1]
subprocess.run(['python3',str(B/'collect.py')],check=True)
summary=json.loads((B/'summary.json').read_text())
assert summary['completed_measurements']==12, 'All 12 measurements must finish before publication'
assert len(summary['comparisons'])==96, 'All 32 cases need all three pairwise comparisons'
dest=ROOT/'benchmarks/results/sirius_mi300a_20261031_32x3'
dest.mkdir(parents=True,exist_ok=True)
manifest=json.loads((B/'manifest.json').read_text())
local_budget_records={name for name in manifest.get('source_artifacts',{})
    if (name.startswith('budget-') and name.endswith('.json')) or name=='resume-estimate-32x3.json'}
public_manifest=json.loads(json.dumps(manifest))
public_manifest.pop('budget_policy',None)
public_manifest['source_artifacts']={k:v for k,v in manifest.get('source_artifacts',{}).items()
    if k not in local_budget_records}
(dest/'manifest.json').write_text(json.dumps(public_manifest,indent=2)+'\n')
for name in ['summary.json','timings.csv','comparisons.csv','prepare.py','run.py','collect.py','publish_results.py','suite-mpirun','compare_gpu_suites.py','blas_probe.py','budget_guard.py']:
    shutil.copy2(B/name,dest/name)
shutil.copy2(B/'harness_upstream_cpu/tools/run_gpu_suites.py',dest/'run_gpu_suites.py')
job_scripts=json.loads((B/'job_scripts.json').read_text())
accepted_scripts=set()
for measurement in summary['timings']:
    result=json.loads((B/'runs'/measurement['route']/(measurement['workload']+'.status.json')).read_text())
    accepted_scripts.add(job_scripts[result['jobid']])
for name in sorted(accepted_scripts):
    shutil.copy2(B/name,dest/name)
shutil.copy2(B/'job_scripts.json',dest/'job_scripts.json')
shutil.copy2(B/'jobs.json',dest/'jobs.json')
for name in public_manifest.get('source_artifacts',{}):
    shutil.copy2(B/name,dest/Path(name).name)
for route in manifest['binaries']:
    if route=='upstream_cpu':
        shutil.copy2(B/'binaries'/route/'build-manifest.json',dest/'upstream-build-manifest.json')
rows=[]
outputs=[]
phase_rows=[]
launch_records=[]
for measurement in summary['timings']:
    route,w=measurement['route'],measurement['workload']
    result=json.loads((B/'runs'/route/(w+'.status.json')).read_text())
    folder=B/'runs'/route/w
    launch=dict(route=route,workload=w,jobid=result['jobid'],binary_sha256=result['binary_sha256'],
        parent_environment=result['environment'],
        ranks=[json.loads(p.read_text()) for p in sorted((folder/'affinity').glob('rank.*.json'))])
    if w in ('S','L','L2','L3'):
        launch['helper_environment']=json.loads((folder/'command.json').read_text())['environment']
        launch['mpi_environment']=json.loads((folder/'launch-environment.json').read_text())
    launch_records.append(launch)
    for case in result['cases']:
        record=next(x for x in manifest['inputs'] if x['workload']==w and x['name']==case['name'])
        outputdir=B/'runs'/route/w
        if w in ('S','L','L2','L3'): outputdir/='together'
        phase_text=(outputdir/(record['system']+'.out')).read_text()
        for match in re.finditer(r'^\s*(.*?)\s*=\s+(\d+)\s+([0-9.]+)\s+(\d+)\s+([0-9.]+)\s*$',phase_text,re.M):
            phase_rows.append(dict(route=route,workload=w,case=case['name'],phase=match[1].strip(),
                min_rank=int(match[2]),min_seconds=float(match[3]),max_rank=int(match[4]),max_seconds=float(match[5])))
        native={r['case']:r['elapsed_seconds'] for r in result.get('native_results',[])}
        conv=next(c for c in summary['convergence'] if c['route']==route and c['workload']==w and c['case']==case['name'])
        rows.append(dict(route=route,workload=w,case=case['name'],nodes=result['nodes'],
            ranks_per_node=32,threads=3,elapsed_seconds=native.get(case['name'],case['elapsed_seconds']),
            energy_hartree=case['energy_hartree'],scf_iterations=case['scf_iterations'],atoms=len(case['atoms']),
            converged_all_md=conv['converged_all_md'],unconverged_md_steps=';'.join(str(x['md_step']) for x in conv['md_steps'] if not x['scf_converged']),
            grid1=case['grids'][0],grid2=case['grids'][1],grid3=case['grids'][2],
            binary_sha256=result['binary_sha256'],original_input_sha256=case['original_input_sha256'],runtime_input_sha256=case['runtime_input_sha256'],jobid=result['jobid']))
        outputs.append(dict(route=route,workload=w,**case,converged_all_md=conv['converged_all_md'],convergence_by_md=conv['md_steps']))
    metadata={k:v for k,v in result.items() if k not in ('cases','native_results')}
    (dest/(route+'_'+w+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')
with (dest/'cases.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
with (dest/'phases.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(phase_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(phase_rows)
(dest/'outputs.json').write_text(json.dumps(outputs,separators=(',',':'))+'\n')
(dest/'launch-environments.json').write_text(json.dumps(launch_records,separators=(',',':'))+'\n')
one_node_phases=[]
for row in rows:
    if row['nodes']!=1:
        continue
    one_node_phases.append(dict(route=row['route'],workload=row['workload'],case=row['case'],
        phases_max_seconds={p['phase']:p['max_seconds'] for p in phase_rows
            if (p['route'],p['workload'],p['case'])==(row['route'],row['workload'],row['case'])}))
(dest/'one-node-phases.json').write_text(json.dumps(one_node_phases,indent=2)+'\n')
def phase_sum(route,workload,phase):
    return sum(p['max_seconds'] for p in phase_rows
        if p['route']==route and p['workload']==workload and p['phase']==phase)
passed=sum(x['status']=='passed' for x in summary['comparisons'])
lines=[
    'Sirius MI300A benchmarks, 2026-10-31, 32 MPI x 3 OpenMP per physical node',
    '='*73,'',
    'Scope: native -runtest, -runtestL and the versioned sidia333_nc_cluster.dat',
    'and cghelgeo.dat decks. Every measurement uses one physical four-APU node',
    '(32 ranks, 96 CPU cores, four GPUs). All ranks have three disjoint CPU places.',
    'All 12 route/workload measurements and 96 calculations finished.',
    'The user removed L2/L3 after the project-budget stop. Matching completed',
    'CPU measurements and the final GPU S measurement were retained; only',
    'the three missing GPU measurements were run after the scope change.',
    'No separate gpusolver2 or gpusolver3 benchmark was requested or run.',
    '',
    'Three routes: official OpenMX 4.0 plus official 4.0.1 fixes (CPU elpa2),',
    'the unified GPU fork with CPU elpa2/OPENMX_GPU=0, and the fork with',
    'ordinary gpusolver/OPENMX_GPU=1. Both fork routes use frozen executables',
    'built after the ELPA OpenMP fix and remote merge, with identical shared DSOs.',
    'All GPU measurements include remote 73c4a2c0, which guards',
    'GPU ELPA handle destruction without changing numerical operations.',
    'GPU S was repeated with that guard at the user request; its older result is excluded.',
    'Exact source and executable versions are retained for each workload.',
    'All use AMD ROCm 7.2.0 Clang/flang,',
    'OpenMPI 5.0.10 and AOCL 5.1.0 (OpenMP BLIS preloaded first, three threads).',
    'The official code retains ELPA 2018.05.001;',
    'the fork uses ELPA 2026.02.002. Both ELPA libraries enable OpenMP.',
    'Fork CPU handles explicitly use three threads; GPU ELPA host teams',
    'remain one thread. The fork initializes MPI with MPI_THREAD_SERIALIZED.',
    'Exact compiler/link flags,',
    'source revision and executable/DSO hashes are in the manifests.',
    'The Krylov fix binds each OpenMP worker to its rank GPU and initializes',
    'rocBLAS on the calling thread before workers start their first GEMM.',
    'A 32-rank, three-thread native BLAS probe reproduced stale HIP 209 with',
    'numerically exact products; upfront rocBLAS initialization eliminated it.',
    'All eight fork route/workload measurements were rerun after enabling',
    'ELPA OpenMP. Earlier fork measurements are excluded, including the old',
    'GPU-only Krylov rerun. All earlier native-suite measurements, including',
    'official CPU, are also excluded: the helper overwrote BLIS_NUM_THREADS',
    'with 1 even when its parent requested 3. Earlier direct BLAS diagnostics',
    'observed three workers but did not test the suite helper. Only the four',
    'direct named CPU measurements are retained from before the helper fix.',
    'The corrected helper preserves explicit BLAS settings. Every accepted',
    'native suite records three BLAS threads before MPI and on every rank.',
    'A separate native C60/GaAs sequence through that helper observed three',
    'actual BLIS workers on all 32 ranks in all three routes (PBS 48495).',
    'AOCL selection had also preferred serial BLIS at link time; the corrected',
    'build selects MT BLIS directly. An untimed direct audit confirms three',
    'workers without LD_PRELOAD. Production keeps the common MT preload.',
    'The production patch and validation records are retained alongside results.',
    '',
    'Every physical input control, including SCF/MD limits, convergence criteria,',
    'grids, geometry and basis, is preserved. Only deployment paths and the',
    'CPU/GPU eigensolver selector differ. The 27 PAO and 20 VPS files have',
    'identical SHA-256 values across all routes. The collector checks all atom',
    'identities, coordinates, grids, energies and every force component.',
    '',
    'Timing: sum of native per-case runtest result rows for each whole suite;',
    'Elapsed.Time. in the output for named inputs. Launcher wall times are',
    'retained separately. Each suite uses its native flag in one persistent',
    'MPI process set. Prefixes of failed suites are never accepted as complete.',
    'These are single measurements on allocated, exclusive physical nodes.',
    'phases.csv retains the native minimum/maximum MPI-rank stage times; nested',
    'stages and maxima from different ranks must not be added as a wall time.',
    'OPENMX_GPU_VERBOSE=1 retains solver dispatch/timing diagnostics. Extra',
    'Krylov stage profiling, GEMMul8 profiling and allocation tracing are off.',
    '',
    'Additional measurement resource use is recorded separately in',
    'additional_measurement_resources.json; the reused nine measurements',
    'did not need another allocation.',
    '',
    'Native times (seconds):',
    'workload                  nodes  upstream CPU    fork CPU    fork GPU',
]
for w,n in manifest['workloads'].items():
    times={x['route']:x['benchmark_seconds'] for x in summary['timings'] if x['workload']==w}
    lines.append(f'{w:26s} {n:2d} {times["upstream_cpu"]:13.2f} {times["fork_cpu"]:11.2f} {times["fork_gpu"]:11.2f}')
lines += ['',f'Numerical comparisons within 1e-7 Ha and 1e-7 Ha/bohr: {passed}/96.',
          'Completion and numerical acceptance are reported separately. Fixed-SCF',
          'decks, including sidia333 (25 SCF steps),',
          'are not described as converged. Native stdout convergence statements',
          'are retained for every MD step in summary.json and outputs.json.',
          'SCF counts and every mismatch remain in cases.csv/comparisons.csv.',
          '',
          'Observed one-node phase times (seconds; sums of native MPI-rank maxima):',
          f'  S Hamiltonian: fork CPU {phase_sum("fork_cpu","S","Set_Hamiltonian"):.3f}, GPU {phase_sum("fork_gpu","S","Set_Hamiltonian"):.3f}.',
          f'  S density grid: fork CPU {phase_sum("fork_cpu","S","Set_Density_Grid"):.3f}, GPU {phase_sum("fork_gpu","S","Set_Density_Grid"):.3f}.',
          *[f'  {w} diagonalization: fork CPU {phase_sum("fork_cpu",w,"Diagonalization"):.3f}, GPU {phase_sum("fork_gpu",w,"Diagonalization"):.3f}.' for w in ('L','sidia333_nc_cluster','cghelgeo')],
          '  These measurements identify the expensive phases. They do not alone',
          '  isolate kernel launch, memory movement, queue contention or communication.',
          '  one-node-phases.json retains phase maxima for all 32 one-node cases per route.',
          '',
          'Runtime settings and stability:',
          '  OMP_NUM_THREADS=3, OMP_MAX_ACTIVE_LEVELS=1, OMP_DYNAMIC=FALSE',
          '  BLIS_NUM_THREADS=3 (AOCL OpenMP BLIS), OPENBLAS_NUM_THREADS=1, MAGMA_NUM_THREADS=1',
          '  HSA_XNACK=1, HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0',
          '  HSA_ENABLE_SDMA=1, HSA_NO_SCRATCH_RECLAIM=1',
          '  LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD=0',
          '  UCX_TLS=self,cma,rc, UCX_RCACHE_ENABLE=n; default memory-type cache',
          '',
          'The final per-measurement JSON files give the actual environment and',
          'PBS allocation identities. Earlier UCX MM endpoint assertions occurred',
          'in the official CPU code and the fork CPU/GPU code. Those attempts',
          'were excluded. A separate GPU fault surfaced while waiting for native',
          'on-demand orbital/density kernels; changing SDMA did not fix it.',
          'The final GPU runs use HSA_NO_SCRATCH_RECLAIM=1. This retains scratch',
          'on its queue; it is a tested runtime setting, not proof of fault cause.',
          'ROCr reference: https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-7.0.0/api-reference/environment_variables.html',
          '',
          'GPU-enabled execution can select CPU eigensolvers for small matrices',
          'or according to the normal capacity/sharing policy. Actual dispatch',
          'must be read from retained run logs; no threshold or memory override',
          'was used to force a benchmark route.',
          '',
          'Full logs, PBS scripts, output files and interrupted attempts:',str(B),
          'prepare.py records the initial input snapshot; later build/launcher',
          'updates are recorded by their exact source and binary versions in',
          'manifest.json. fork-build-manifest.json records the final HIP toolchain',
          'and upstream-build-manifest.json records the official CPU build.',
          'The submitted PBS scripts retain the original local campaign paths.',
          'Raw input locations and all data-file hashes are in manifest.json.',
          'The original CPU scripts also ran superseded suite measurements;',
          'only their direct named CPU measurements are accepted. job_scripts.json',
          'maps every accepted measurement to its submitted PBS script.',
          'The RSS sampling in suite helpers covers processes on the launcher node;',
          'it is not a sum of remote-node memory usage.']
(dest/'README.txt').write_text('\n'.join(lines)+'\n')
print(str(dest))
