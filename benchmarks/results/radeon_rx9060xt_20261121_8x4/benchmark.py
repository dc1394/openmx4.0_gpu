#!/usr/bin/env python3
"""Measure all three OpenMX routes sequentially at 8 MPI x 4 OpenMP."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv, fcntl, hashlib, itertools, json, math, os, re, shutil, signal, subprocess, sys, time

REPO = Path('/home/kawai/openmx4.0_gpu')
UPSTREAM = Path('/home/kawai/openmx4.0')
ROOT = REPO / 'work/codex_benchmark_20261121_8x4'
OUT = REPO / 'benchmarks/results/radeon_rx9060xt_20261121_8x4'
BUILD = REPO / 'build/hip-application-20261025/openmpi-aocl'
SDK = REPO / 'work/codex_rocm_upgrade_20261025/sdk/opt/rocm/core-10.1'
sys.path.insert(0, str(REPO / 'tools'))
from compare_gpu_suites import read_output
from run_gpu_suites import available_bytes, group_rss, stop_job, read_result_rows

def now(): return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)
def keyword(text, key):
    match = re.search(r'^\s*' + re.escape(key) + r'\s+(\S+)', text, re.M | re.I)
    return match[1] if match else None
def setkey(text, key, value):
    pat = r'^(\s*' + re.escape(key) + r'\s+)\S+'
    if re.search(pat, text, re.M | re.I):
        return re.sub(pat, lambda m: m[1] + value, text, flags=re.M | re.I)
    return text + '\n' + key + ' ' + value + '\n'
def physical(path):
    energy, atoms, forces, grids = read_output(path)
    assert all(math.isfinite(x) for a in atoms for x in a[2:])
    return {'energy_Ha': energy, 'atoms': atoms, 'forces_Ha_bohr': forces, 'grids': grids}
def compare(a, b):
    same = (a['grids'] == b['grids'] and len(a['atoms']) == len(b['atoms'])
            and len(a['forces_Ha_bohr']) == len(b['forces_Ha_bohr'])
            and all(x[:2] == y[:2] and max(abs(c-d) for c,d in zip(x[2:], y[2:])) <= 1.000001e-5
                    for x,y in zip(a['atoms'], b['atoms'])))
    de = abs(a['energy_Ha'] - b['energy_Ha'])
    df = max((abs(x-y) for x,y in zip(a['forces_Ha_bohr'], b['forces_Ha_bohr'])), default=0)
    return {'pass': same and de <= 1e-7 and df <= 1e-7,
            'delta_energy_Ha': de, 'max_delta_force_Ha_bohr': df, 'same_geometry_and_grids': same}

ROOT.mkdir(exist_ok=True)
lock = (ROOT / 'benchmark.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (ROOT / 'queue.json').exists(), 'Existing campaign: inspect before resuming; never duplicate jobs'
assert json.loads((ROOT / 'build.json').read_text())['returncode'] == 0
OUT.mkdir(parents=True, exist_ok=False)
(OUT / '.gitignore').write_text('!README.md\n')
shutil.copy2(__file__, OUT / 'benchmark.py')
common = {'OMP_NUM_THREADS': '4', 'OMP_DYNAMIC': 'FALSE', 'OMP_PROC_BIND': 'close',
          'OMP_PLACES': 'cores', 'OMP_MAX_ACTIVE_LEVELS': '1', 'OPENBLAS_NUM_THREADS': '1'}
clean = {k:v for k,v in os.environ.items()
         if not k.startswith(('OPENMX_', 'GEMMUL8_', 'OMP_', 'OMPI_', 'I_MPI_', 'MKL_', 'BLIS_',
                              'OPENBLAS_', 'HIP_', 'ROCM_', 'ROCR_', 'HSA_', 'LIBOMPTARGET_', 'MAGMA_', 'KMP_'))
         and k not in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LIBRARY_PATH', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH')}
hipenv = {**common, 'BLIS_NUM_THREADS': '4', 'MAGMA_NUM_THREADS': '4', 'MKL_NUM_THREADS': '1',
          'LIBOMPTARGET_HEAP_SIZE': '67108864', 'HSA_OVERRIDE_CPU_AFFINITY_DEBUG': '0',
          'HIP_VISIBLE_DEVICES': '0', 'ROCR_VISIBLE_DEVICES': '0',
          'LD_LIBRARY_PATH': ':'.join(map(str, [SDK/'llvm/lib', SDK/'lib',
             Path('/home/kawai/opt/openmpi-rocmclang/lib'), Path('/opt/AMD/aocl/aocl-linux-aocc-5.2.0/aocc/lib'),
             Path('/opt/AMD/aocc-compiler-5.1.0/lib')]))}
intelenv = {**common, 'BLIS_NUM_THREADS': '1', 'OPENMX_GPU': '0', 'MKL_NUM_THREADS': '4',
            'MKL_DYNAMIC': 'FALSE', 'I_MPI_PIN': '1', 'I_MPI_PIN_DOMAIN': 'omp',
            'I_MPI_PIN_ORDER': 'compact', 'I_MPI_FABRICS': 'shm',
            'LD_LIBRARY_PATH': '/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/mpi/2021.18/lib:/opt/intel/oneapi/mkl/latest/lib'}
freeze = ROOT / 'frozen'
(freeze / 'fork/lib').mkdir(parents=True)
(freeze / 'upstream').mkdir()
for src, dest in [(BUILD/'openmx', freeze/'fork/openmx'),
                  (BUILD/'core/libopenmx_core.so', freeze/'fork/lib/libopenmx_core.so'),
                  (BUILD/'core/host-simd/lib/libopenmx_host_simd.so', freeze/'fork/lib/libopenmx_host_simd.so'),
                  (UPSTREAM/'work/openmx', freeze/'upstream/openmx')]:
    shutil.copy2(src, dest)
wrapper = ROOT / 'run-fork-rank.sh'
wrapper.write_text('#!/bin/sh\nexport LD_PRELOAD="$BENCH_PRELOAD"\nexec "$@"\n')
wrapper.chmod(0o755)
hipenv['BENCH_PRELOAD'] = ':'.join(str(freeze/'fork/lib'/n) for n in ['libopenmx_core.so', 'libopenmx_host_simd.so'])
launcher = ['/home/kawai/opt/openmpi-rocmclang/bin/mpirun', '-np', '8', '--map-by', 'slot:PE=4', '--bind-to', 'core']
variants = {
 'upstream-cpu': {'binary': freeze/'upstream/openmx', 'launcher': ['/opt/intel/oneapi/mpi/2021.18/bin/mpiexec', '-np', '8'], 'controls': intelenv, 'gpu': False},
 'fork-cpu': {'binary': freeze/'fork/openmx', 'launcher': launcher + [str(wrapper)], 'controls': {**hipenv, 'OPENMX_GPU': '0'}, 'gpu': False},
 'fork-gpu': {'binary': freeze/'fork/openmx', 'launcher': launcher + [str(wrapper)], 'controls': {**hipenv, 'OPENMX_GPU': '1', 'HSA_ENABLE_SDMA': '0'}, 'gpu': True}}
for name, v in variants.items():
    probeenv = {**clean, **v['controls']}
    if name.startswith('fork'): probeenv['LD_PRELOAD'] = hipenv['BENCH_PRELOAD']
    p = subprocess.run(['ldd', str(v['binary'])], env=probeenv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (ROOT/(name+'-ldd.log')).write_text(p.stdout)
    assert p.returncode == 0 and 'not found' not in p.stdout, (name, p.stdout)
queue = [{'test':test, 'variant':name, 'status':'pending'}
         for test in ['S', 'sidia333_nc_cluster', 'cghelgeo', 'L'] for name in variants]
meta = {'started':now(), 'pid':os.getpid(), 'revision':subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
        'ranks':8, 'threads':4, 'physical_cores':32, 'host_memory_reserve_gib':32,
        'measurements_per_route_and_test':1, 'input_policy':'Original geometry, SCF/MD limits, criterion, stdout and precision; only DATA.PATH and scf.eigen.lib changed.',
        'timing_policy':'Suite totals sum native case times; named inputs use maximum-rank Total Computational Time. Wall time saved separately.',
        'comparison_policy':'Finite energy and complete finite forces; geometry/grid agreement; abs energy <=1e-7 Ha and max force component <=1e-7 Ha/bohr. Completion alone is not PASS.',
        'nvidia_mps':'Not applicable: AMD Radeon HIP.', 'restart_policy':'Independent fresh working directory for every route; no copied restart.',
        'working_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff'], cwd=REPO)).hexdigest(),
        'binaries':{str(p.relative_to(freeze)):sha(p) for p in freeze.rglob('*') if p.is_file()},
        'jobs':queue}
save(ROOT/'queue.json', meta)
save(OUT/'manifest.json', {**{k:v for k,v in meta.items() if k!='jobs'}, 'build':json.loads((ROOT/'build.json').read_text()),
                          'variant_environment':{k:v['controls'] for k,v in variants.items()}})
stop_requested = False
def request_stop(signum, frame):
    global stop_requested
    stop_requested = True
signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
proc = None

def audit_ranks(sid, dest, mpi):
    rows = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal(): continue
        try:
            if (entry/'comm').read_text().strip() != 'openmx': continue
            fields = (entry/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[3]) != sid: continue
            env = dict(x.split(b'=',1) for x in (entry/'environ').read_bytes().split(b'\0') if b'=' in x)
            rank = int(env[b'OMPI_COMM_WORLD_RANK'] if mpi=='openmpi' else env[b'PMI_RANK'])
            masks = [set(os.sched_getaffinity(int(t.name))) for t in (entry/'task').iterdir()]
            cpus = set().union(*masks)
            rows.append({'rank':rank, 'pid':int(entry.name), 'OMP_NUM_THREADS':env.get(b'OMP_NUM_THREADS',b'').decode(),
                         'math_threads':env.get(b'BLIS_NUM_THREADS' if mpi=='openmpi' else b'MKL_NUM_THREADS',b'').decode(),
                         'allowed_cpus':sorted(cpus), 'os_threads':len(masks)})
            if rank == 0:
                libs = sorted(set(l.split()[-1] for l in (entry/'maps').read_text().splitlines() if '.so' in l and '/' in l))
                save(dest.with_name('runtime-libraries.json'), libs)
        except (OSError, ValueError, KeyError, ProcessLookupError): continue
    if len(rows)!=8 or set(x['rank'] for x in rows)!=set(range(8)): return False
    valid = all(x['OMP_NUM_THREADS']=='4' and x['math_threads']=='4' and len(x['allowed_cpus'])==4 for x in rows)
    sets = [set(x['allowed_cpus']) for x in rows]
    valid = valid and all(not(a & b) for a,b in itertools.combinations(sets,2)) and len(set.union(*sets))==32
    save(dest, {'pass':valid, 'mpi':mpi, 'ranks':sorted(rows,key=lambda x:x['rank'])})
    return valid

all_results = []
try:
 for job in queue:
    if stop_requested: break
    name, test = job['variant'], job['test'];v = variants[name];suite = test in ('S','L')
    folder = ROOT/'runs'/(test+'-'+name);folder.mkdir(parents=True)
    directory = {'S':'input_example','L':'large_example'}.get(test)
    sources = sorted((UPSTREAM/'work'/directory).glob('*.dat')) if suite else [REPO/'benchmarks/inputs'/(test+'.dat')]
    cases = []
    for source in sources:
        original = source.read_text();system = keyword(original,'System.Name');assert system
        text = setkey(original,'DATA.PATH',str(UPSTREAM/'DFT_DATA19'))
        text = setkey(text,'scf.eigen.lib','gpusolver' if v['gpu'] else 'elpa2')
        dest = folder/directory/source.name if suite else folder/'input.dat';dest.parent.mkdir(exist_ok=True);dest.write_text(text)
        if suite: shutil.copy2(source.parent/(system+'.out'), dest.parent/(system+'.out'))
        cases.append({'case':source.stem, 'system':system, 'source':str(source), 'original_sha256':sha(source), 'input_sha256':sha(dest),
                      'scf_criterion':keyword(original,'scf.criterion'), 'scf_maxIter':keyword(original,'scf.maxIter'),
                      'MD_maxIter':keyword(original,'MD.maxIter'), 'status':'pending'})
    args = v['launcher']+[str(v['binary']), '-runtest'+('' if test=='S' else 'L') if suite else 'input.dat', '-nt','4']
    save(folder/'command.json', {'command':args, 'controls':v['controls'], 'inputs':cases, 'binary_sha256':sha(v['binary'])})
    save(folder/'results.json', cases)
    job.update(status='running', started=now(), directory=str(folder));save(ROOT/'queue.json',meta)
    print(now(),'START',test,name,flush=True)
    begin = time.monotonic();peak = 0;minimum = available_bytes();last_sample = 0;audited = False;reason = None
    marker = folder/('runtest'+('' if test=='S' else 'L')+'.result') if suite else None
    def collect():
        native = read_result_rows(marker) if suite else {}
        for i, item in enumerate(cases,1):
            if item['status'] != 'pending' or (suite and i not in native): continue
            try:
                obs = physical(folder/(item['system']+'.out'))
                item.update(status='completed', **obs)
                if suite:
                    row = native[i]
                    assert row and len(row)==3 and all(math.isfinite(x) for x in row)
                    item.update(seconds=row[0], native_delta_energy=row[1], native_force_norm=row[2],
                                bundled_reference=compare(obs,physical(folder/directory/(item['system']+'.out'))))
            except Exception as exc: item.update(status='invalid_output', error=str(exc))
        save(folder/'results.json', cases)
    with (folder/'run.log').open('w') as log:
        proc = subprocess.Popen(args,cwd=folder,env={**clean,**v['controls']},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(folder/'process.json',{'pid':proc.pid,'started':now(),'command':args})
        while proc.poll() is None:
            free = available_bytes();minimum=min(minimum,free);peak=max(peak,group_rss(proc.pid))
            if not audited: audited=audit_ranks(proc.pid,folder/'rank-environment.json','intel' if name=='upstream-cpu' else 'openmpi')
            if stop_requested: reason='interrupted'
            elif free < 32*1024**3: reason='host_memory_guard'
            elif time.monotonic()-begin > 4*3600: reason='timeout_4_hours'
            elif not audited and time.monotonic()-begin > 60: reason='rank_affinity_audit_failed'
            if reason: stop_job(proc);break
            if time.monotonic()-last_sample >= 10:
                if suite: collect()
                with (folder/'run.log').open('rb') as f:
                    f.seek(0,2);f.seek(max(0,f.tell()-131072));tail=f.read().decode(errors='replace')
                scf = re.findall(r'MD=\s*(\d+)\s+SCF=\s*(\d+)',tail)
                sample={'time':now(),'test':test,'variant':name,'completed_cases':sum(x['status']=='completed' for x in cases),
                        'total_cases':len(cases),'case':next((x['case'] for x in cases if x['status']=='pending'),None),
                        'MD':int(scf[-1][0]) if scf else None,'SCF':int(scf[-1][1]) if scf else None,
                        'wall_s':time.monotonic()-begin,'available_gib':free/1024**3,'peak_rss_gib':peak/1024**3,
                        'load_average':os.getloadavg(),'rank_environment_verified':audited}
                save(ROOT/'current.json',sample)
                with (ROOT/'cpu-load.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
                last_sample=time.monotonic()
            time.sleep(1)
    text=(folder/'run.log').read_text(errors='replace')
    if suite or (proc.returncode==0 and 'The calculation was normally finished.' in text): collect()
    times=[float(x) for x in re.findall(r'Total Computational Time\s*=\s*\d+\s+[\d.]+\s+\d+\s+([\d.]+)',text)]
    sections=text.split('The calculation was normally finished.')
    for idx,item in enumerate(cases):
        if idx<len(times):item['total_computational_seconds']=times[idx]
        if idx<len(sections):
            item['scf_steps']=len(re.findall(r'MD=\s*\d+\s+SCF=',sections[idx]))
            item['scf_limit_message_count']=len(re.findall(r'(?:not.*converg|converg.*not|maximum.*SCF|SCF.*maximum)',sections[idx],re.I))
        if not suite and times:item['seconds']=times[-1]
    save(folder/'results.json',cases)
    complete=sum(x['status']=='completed' for x in cases)
    valid=(proc.returncode==0 and complete==len(cases) and len(times)==len(cases) and audited
           and text.count('The calculation was normally finished.')==len(cases))
    job.update(status=reason or ('completed' if valid else 'invalid_run'), ended=now(), returncode=proc.returncode,
               wall_seconds=time.monotonic()-begin,peak_rss_gib=peak/1024**3,min_available_gib=minimum/1024**3,
               completed=complete,expected=len(cases),normal_finish_markers=text.count('The calculation was normally finished.'),
               rank_environment_verified=audited,seconds=sum(x.get('seconds',0) for x in cases),
               gpu_accelerated_markers=text.count('(GPU-accelerated)'),
               dense_cpu_fallback_lines=[l for l in text.splitlines() if 'fallback' in l.lower() and any(k in l.lower() for k in ['elpa','diagonal','dense','eigen'])][:40])
    save(folder/'process.json',{'pid':proc.pid,'finished':now(),'returncode':proc.returncode,'status':job['status']})
    save(ROOT/'queue.json',meta)
    all_results.extend({'test':test,'variant':name,**x} for x in cases)
    save(OUT/'runs.json',all_results);save(OUT/'run-results.json',queue)
    print(now(),'END',test,name,job['status'],'seconds',round(job['seconds'],3),'wall',round(job['wall_seconds'],3),flush=True)
    proc=None
    if reason:break
finally:
    if proc is not None and proc.poll() is None:stop_job(proc)
    meta['controller_finished']=now();meta['stopped_by_user']=stop_requested;save(ROOT/'queue.json',meta)
    save(OUT/'run-results.json',queue)

comparisons=[]
for test in ['S','L','cghelgeo','sidia333_nc_cluster']:
    grouped={name:{x['case']:x for x in all_results if x['test']==test and x['variant']==name and x['status']=='completed'} for name in variants}
    for a,b in itertools.combinations(variants,2):
        for case in sorted(grouped[a].keys() & grouped[b].keys()):
            comparisons.append({'test':test,'case':case,'left':a,'right':b,**compare(grouped[a][case],grouped[b][case])})
save(OUT/'comparisons.json',comparisons)
for filename,rows,fields in [
 ('case-results.csv',all_results,['test','case','variant','status','seconds','total_computational_seconds','scf_steps','energy_Ha','scf_criterion','scf_maxIter']),
 ('run-results.csv',queue,['test','variant','status','seconds','wall_seconds','completed','expected','peak_rss_gib','min_available_gib','rank_environment_verified']),
 ('comparisons.csv',comparisons,['test','case','left','right','pass','delta_energy_Ha','max_delta_force_Ha_bohr','same_geometry_and_grids'])]:
    with (OUT/filename).open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
summary=['# Radeon RX 9060 XT benchmark — 2026-11-21, 8 MPI × 4 OpenMP','',
         'All jobs run sequentially on 32 physical Threadripper 3970X cores. One measurement per route and test. Original SCF/MD settings, no copied restart. Suite totals sum native per-case times; named inputs use maximum-rank computational times. MPI-job wall times are separate. GPU-route timings include application-selected CPU fallbacks.','',
         '| Test | Upstream CPU (s) | Fork CPU (s) | Fork HIP (s) |','|---|---:|---:|---:|']
for test in ['S','L','cghelgeo','sidia333_nc_cluster']:
    jobs={j['variant']:j for j in queue if j['test']==test}
    summary.append('| '+test+' | '+' | '.join(f"{jobs[n]['seconds']:.3f}" if jobs[n].get('status')=='completed' else jobs[n]['status'] for n in variants)+' |')
summary += ['','Finite output/completion and numerical agreement are separate checks. Numerical PASS requires matching geometry/grids, |ΔE| ≤1e-7 Ha and max |ΔF| ≤1e-7 Ha/bohr. Failed comparisons are retained. The original sidia333 25-SCF cap does not establish convergence.','',
            f"Completed valid executions: {sum(x['status']=='completed' for x in all_results)} / 96. Pairwise numerical PASS: {sum(x['pass'] for x in comparisons)} / {len(comparisons)}.",'',
            'Per-case timings, SCF counts, energy/force comparisons and runtime configuration are in CSV/JSON. Raw outputs, CPU/memory samples, frozen binaries and all rank-affinity/library audits: `'+str(ROOT)+'`.','']
(OUT/'README.md').write_text('\n'.join(summary))
save(OUT/'validation.json',{'all_jobs_completed':all(j['status']=='completed' for j in queue),'completed_outputs':sum(x['status']=='completed' for x in all_results),
                          'expected_outputs':96,'comparison_passes':sum(x['pass'] for x in comparisons),'comparisons':len(comparisons),
                          'failed_comparisons':[x for x in comparisons if not x['pass']],
                          'frozen_binaries_unchanged':all(sha(freeze/p)==h for p,h in meta['binaries'].items()),'finished':now()})
print(now(),'CAMPAIGN FINISHED',flush=True)
