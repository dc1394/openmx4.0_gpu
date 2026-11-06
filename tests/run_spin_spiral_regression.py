#!/usr/bin/env python3
"""Run small spin-spiral band regressions using the installed OpenMX executable.

Requires MPI and the repository's Fe4.0H/Fe_CA19H data. The original input is
copied into fresh directories. Three SCF steps test MPI decomposition, odd
mesh equivalence and the q=0 limit against the ordinary noncollinear solver.
These shortened cases test numerical agreement, not SCF convergence.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from compare_gpu_suites import read_output
from run_gpu_suites import stop_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'work/openmx')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launcher', default='mpirun --bind-to none')
    parser.add_argument('--gpu-check', action='store_true',
                        help='also compare OPENMX_GPU=1 using the CUDA/HIP build')
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    template = (ROOT/'work/Fefcc-SpinSpiral.dat').read_text()
    cases = [
        ('gamma-serial', 1, '2 1 1', None, 'on', '0.0 0.5 0.5', False),
        ('gamma-parallel', 4, '2 1 1', None, 'on', '0.0 0.5 0.5', False),
        ('gamma-odd', 1, '3 1 1', 'GCenter', 'on', '0.0 0.5 0.5', False),
        ('regular-odd', 1, '3 1 1', 'REGULAR', 'on', '0.0 0.5 0.5', False),
        ('gamma-zero', 1, '2 1 1', None, 'on', '0.0 0.0 0.0', False),
        ('noncollinear-zero', 1, '2 1 1', None, 'off', '0.0 0.0 0.0', False),
    ]
    if args.gpu_check:
        cases.append(('gamma-gpu-enabled', 1, '2 1 1', None, 'on', '0.0 0.5 0.5', True))
    env = {k:v for k,v in os.environ.items()
           if not k.startswith(('OPENMX_', 'GEMMUL8_', 'OMP_', 'MKL_', 'KMP_'))}
    env.update(OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1', OMP_STACKSIZE='64M')
    results = {}
    for name, ranks, mesh, generation, generalized, q, gpu in cases:
        folder = out/name
        folder.mkdir()
        overrides = {'DATA.PATH':str(ROOT/'DFT_DATA19'), 'System.CurrrentDirectory':'./',
                     'System.Name':'result', 'level.of.fileout':'1', 'scf.maxIter':'3',
                     'scf.restart':'off', 'scf.Kgrid':mesh, 'scf.Generalized.Bloch':generalized,
                     'Spin.Spiral.Vector':q, 'scf.eigen.lib':'elpa2'}
        if generation:
            overrides['scf.Generation.Kpoint'] = generation
        keys = {k.lower() for k in overrides}
        lines = [line for line in template.splitlines()
                 if not line.split() or line.split()[0].lower() not in keys]
        lines.extend(f'{key} {value}' for key,value in overrides.items())
        (folder/'input.dat').write_text('\n'.join(lines)+'\n')
        command = shlex.split(args.launcher)+['-np',str(ranks),str(binary),'input.dat','-nt','1']
        environment = dict(env, OPENMX_GPU=str(int(gpu)))
        (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        start = time.monotonic()
        with (folder/'run.log').open('w') as log:
            process = subprocess.Popen(command,cwd=folder,env=environment,stdout=log,
                                       stderr=subprocess.STDOUT,start_new_session=True)
            try:
                process.wait(timeout=300)
            except BaseException:
                stop_job(process)
                raise
        stdout = (folder/'run.log').read_text()
        assert process.returncode == 0, (name,process.returncode,folder/'run.log')
        assert stdout.count('The calculation was normally finished.') == 1, name
        assert not re.search(r'(?i)segmentation|invalid pointer|illegal value|parameter.*incorrect',stdout), name
        steps = re.findall(r'MD=\s*1\s+SCF=\s*(\d+)',stdout)
        assert steps == ['1','2','3'], (name,steps)
        assert ('Generalized Bloch Theorem is included' in stdout) == (generalized=='on'), name
        energy, atoms, forces, grids = read_output(folder/'result.out')
        results[name] = dict(energy=energy, atoms=atoms, forces=forces, grids=grids,
                             seconds=time.monotonic()-start, MPI_ranks=ranks)
        (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print('PASS completed:',name,flush=True)
    pairs = [('gamma-serial','gamma-parallel'), ('gamma-odd','regular-odd'),
             ('gamma-zero','noncollinear-zero')]
    if args.gpu_check:
        pairs.append(('gamma-serial','gamma-gpu-enabled'))
    comparisons = []
    for a,b in pairs:
        x,y = results[a],results[b]
        assert x['atoms'] == y['atoms'] and x['grids'] == y['grids'], (a,b)
        de = abs(x['energy']-y['energy'])
        df = max(abs(u-v) for u,v in zip(x['forces'],y['forces']))
        comparisons.append(dict(baseline=a,candidate=b,energy_difference=de,
                                max_force_difference=df,passed=de<=1e-7 and df<=1e-7))
        print(f'{a} / {b}: dE={de:.3g} Ha, max dF={df:.3g} Ha/bohr',flush=True)
    (out/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    assert all(row['passed'] for row in comparisons), comparisons
    print('PASS: all spin-spiral numerical comparisons',flush=True)


if __name__ == '__main__':
    main()
