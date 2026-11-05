#!/usr/bin/env python3
"""Isolated, auditable sidia333 runs on the local 32-core SVE256 machine."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'benchmarks/results/graviton3_sve256_20261105'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(binary):
    vl = ctypes.CDLL(None).prctl(51, 0, 0, 0, 0)
    if vl < 0 or vl & 65535 != 32:
        raise SystemExit(f'Require SVE256, got prctl={vl}')
    core = binary.parent.parent / 'lib/libopenmx_core.so'
    lib = ctypes.CDLL(str(core))
    lib.openmx_core_host_simd_mode.argtypes = [ctypes.c_char_p]
    lib.openmx_core_host_simd_name.argtypes = [ctypes.c_int]
    lib.openmx_core_host_simd_name.restype = ctypes.c_char_p
    mode = lib.openmx_core_host_simd_mode(os.environ['OPENMX_HOST_SIMD'].encode())
    selected = lib.openmx_core_host_simd_name(mode).decode()
    if os.environ['OPENMX_HOST_SIMD'] in ('auto', 'sve256') and selected != 'SVE256':
        raise SystemExit(f'Unexpected host SIMD: {selected}')
    audit = dict(rank=int(os.environ['OMPI_COMM_WORLD_RANK']), pid=os.getpid(),
                 affinity=sorted(os.sched_getaffinity(0)), sve_bits=(vl & 65535)*8,
                 selected_simd=selected, threads=os.environ['OMP_NUM_THREADS'],
                 core=str(core), binary=str(binary))
    metadata=json.loads(Path('command.json').read_text())
    blas_path=metadata.get('blas_library')
    if blas_path:
        blas=ctypes.CDLL(blas_path)
        if metadata['blas_kind'] == 'OpenBLAS':
            blas.openblas_get_config.restype=ctypes.c_char_p
            blas.openblas_get_corename.restype=ctypes.c_char_p
            audit.update(blas_library=blas_path, blas_config=blas.openblas_get_config().decode(),
                         blas_core=blas.openblas_get_corename().decode(),
                         blas_threads=blas.openblas_get_num_threads(),
                         blas_parallel=blas.openblas_get_parallel())
        else:
            version=[ctypes.c_int() for _ in range(3)]
            tag=ctypes.c_char_p()
            blas.armplversion(*(ctypes.byref(v) for v in version),ctypes.byref(tag))
            audit.update(blas_library=blas_path, blas_core='ARMPL',
                         blas_config='.'.join(str(v.value) for v in version)+' '+tag.value.decode(),
                         blas_threads=int(os.environ['OMP_NUM_THREADS']) if '_mp.' in blas_path else 1)
        # Verify the actual owner of the standard BLAS entry point, too.
        class DlInfo(ctypes.Structure):
            _fields_=[('filename',ctypes.c_char_p),('base',ctypes.c_void_p),
                      ('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
        owner=DlInfo()
        ctypes.CDLL(None).dladdr(ctypes.cast(blas.zgemm_,ctypes.c_void_p),ctypes.byref(owner))
        audit['zgemm_library']=str(Path(owner.filename.decode()).resolve())
        if audit['zgemm_library'] != str(Path(blas_path).resolve()):
            raise SystemExit(f"Unexpected zgemm_ owner: {audit['zgemm_library']}")
        if (metadata.get('expected_blas_core') and
                audit['blas_core'].casefold()!=metadata['expected_blas_core'].casefold()):
            raise SystemExit(f"Wrong BLAS loaded: {audit['blas_core']}; expected {metadata['expected_blas_core']}")
    # Loading an OpenMP BLAS can bind the probing Python thread to one core.
    # exec must inherit the entire MPI rank's allocation, so the application
    # discovers all of its OpenMP places afresh.
    audit['affinity_after_blas_probe']=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,audit['affinity'])
    audit['affinity_before_exec']=sorted(os.sched_getaffinity(0))
    Path(f'rank-{audit["rank"]:02}.json').write_text(json.dumps(audit, indent=2)+'\n')
    os.execv(str(binary), [str(binary), 'input.dat', '-nt', os.environ['OMP_NUM_THREADS']])


def run(args):
    stop = RESULTS / 'STOP_BENCHMARKS'
    if stop.exists():
        raise SystemExit('Benchmark queue stopped before '+args.name+': '+stop.read_text().strip())
    folder = RESULTS / args.name
    folder.mkdir(parents=True, exist_ok=False)
    binary = Path(args.binary).resolve()
    original = ROOT / 'benchmarks/inputs/sidia333_nc_cluster.dat'
    text = original.read_text()
    overrides = {'DATA.PATH': str(ROOT/'DFT_DATA19'), 'level.of.fileout': '1',
                 'scf.eigen.lib': 'elpa2', 'scf.restart': 'off'}
    if args.steps != 25:
        overrides['scf.maxIter'] = str(args.steps)
    for key, value in overrides.items():
        pat = r'^' + re.escape(key) + r'\s+.*$'
        if re.search(pat, text, re.M | re.I):
            text = re.sub(pat, key+' '+value, text, flags=re.M | re.I)
        else:
            text += '\n'+key+' '+value+'\n'
    (folder/'input.dat').write_text(text)
    env = {k:v for k,v in os.environ.items()
           if not k.startswith(('OPENMX_', 'OMP_', 'OPENBLAS_', 'GOTO_', 'ELPA_', 'ARMPL_', 'LD_'))}
    env.update(OPENMX_HOST_SIMD=args.mode, OPENMX_GPU='0',
               OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC='FALSE',
               OMP_MAX_ACTIVE_LEVELS='1', OMP_PROC_BIND='close', OMP_PLACES='cores',
               OPENBLAS_NUM_THREADS=str(args.blas_threads), GOTO_NUM_THREADS=str(args.blas_threads))
    for item in args.env:
        key, value = item.split('=', 1)
        env[key] = value
    command = ['/usr/lib64/openmpi/bin/mpirun', '-np', str(args.ranks),
               '--bind-to', 'core', '--map-by', f'slot:PE={args.threads}',
               '/usr/bin/python3', str(Path(__file__).resolve()), '--rank', str(binary)]
    metadata = dict(command=command, overrides=overrides, original_sha256=sha(original),
                    effective_sha256=sha(folder/'input.dat'), binary_sha256=sha(binary),
                    harness_sha256=sha(Path(__file__).resolve()),
                    libraries={p.name:sha(p) for p in (binary.parent.parent/'lib').glob('libopenmx*.so')},
                    ranks=args.ranks, threads=args.threads, openblas_threads_environment=args.blas_threads,
                    steps=args.steps, mode=args.mode,
                    purpose=args.purpose or ('benchmark' if args.steps==25 else 'pilot'),
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    expected_blas_core=args.expect_blas_core,
                    environment={k:v for k,v in env.items() if k.startswith(('OPENMX_', 'OMP_', 'OPENBLAS_', 'ELPA_', 'ARMPL_', 'GOTO_'))})
    build_info=binary.parent.parent/'build-info.json'
    if build_info.exists():
        metadata['build']=json.loads(build_info.read_text())
    linked=subprocess.check_output(['ldd',str(binary)],env=env,text=True)
    (folder/'linked-libraries.txt').write_text(linked)
    blas_match=re.search(r'(libopenblas\S*|libarmpl\S*)\s+=>\s+(\S+)',linked)
    if blas_match:
        blas_path=Path(blas_match[2]).resolve()
        metadata.update(blas_library=str(blas_path),blas_sha256=sha(blas_path),
                        blas_kind='OpenBLAS' if 'openblas' in blas_match[1] else 'ArmPL')
    else:
        raise SystemExit('No supported BLAS library found in ldd output')
    (folder/'command.json').write_text(json.dumps(metadata, indent=2)+'\n')
    start = time.monotonic()
    with (folder/'run.log').open('w') as log:
        process = subprocess.Popen(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
        code = process.wait()
    wall = time.monotonic()-start
    log = (folder/'run.log').read_text(errors='replace')
    tail = log.rsplit('Computational Time (second)',1)[-1]
    pattern = r'^\s*(.+?)\s+=\s+(\d+)\s+([\d.eE+-]+)\s+(\d+)\s+([\d.eE+-]+)\s*$'
    phases = {m[0].strip():float(m[4]) for m in re.findall(pattern,tail,re.M)}
    result = dict(returncode=code, wall_seconds=wall, phases=phases,
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  normal_finish='The calculation was normally finished.' in log,
                  scf_steps=len(set(re.findall(r'\*+ MD=\s*(\d+)\s+SCF=\s*(\d+) \*+',log))),
                  scf_not_converged='The SCF was not fully achieved' in log,
                  scf_converged='The SCF was achieved at MD=' in log,
                  scf_history=[dict(uele=float(u),delta_uele=float(d),normrd=float(n))
                               for u,d,n in re.findall(r'<DFT>\s+Uele\s*=\s*(\S+)\s+dUele\s*=\s*(\S+)\s*\n<DFT>\s+NormRD\s*=\s*(\S+)',log)])
    output=folder/'sidia333_nc_cluster.out'
    if output.exists():
        from compare_gpu_suites import read_output
        energy, atoms, forces, grids=read_output(output)
        result.update(energy=energy,atoms=atoms,forces=forces,grids=grids)
        result['output_sha256']=sha(output)
    result['log_sha256']=sha(folder/'run.log')
    ranks=[json.loads(p.read_text()) for p in sorted(folder.glob('rank-*.json'))]
    affinity=[core for r in ranks for core in r['affinity']]
    result['rank_audit_passed']=(len(ranks)==args.ranks and
        all(len(r['affinity'])==args.threads and r['sve_bits']==256 for r in ranks) and
        len(set(affinity))==len(affinity))
    result['actual_blas_threads']=sorted(set(r['blas_threads'] for r in ranks if 'blas_threads' in r))
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('atoms','forces','scf_history')},indent=2),flush=True)
    return 0 if code==0 and result['normal_finish'] and 'energy' in result and result['rank_audit_passed'] else 1


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--rank':
        rank(Path(sys.argv[2]))
    else:
        p=argparse.ArgumentParser(description=__doc__)
        p.add_argument('name')
        p.add_argument('--binary',required=True)
        p.add_argument('--ranks',type=int,default=32)
        p.add_argument('--threads',type=int,default=1)
        p.add_argument('--blas-threads',type=int,default=1,
                       help='OPENBLAS_NUM_THREADS; OpenMP OpenBLAS instead follows --threads')
        p.add_argument('--mode',default='sve256')
        p.add_argument('--expect-blas-core',help='Fail before running if a different BLAS implementation is loaded')
        p.add_argument('--steps',type=int,default=25)
        p.add_argument('--purpose',choices=('benchmark','validation','pilot','profile'))
        p.add_argument('--env',action='append',default=[])
        sys.exit(run(p.parse_args()))
