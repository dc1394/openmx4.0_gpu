#!/usr/bin/env python3
"""Run the existing distributed ELPA eigenpair checks against the SVE256 build."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build/cpu-sve256'
OUT = ROOT / 'build/sve256-logs'
spec = importlib.util.spec_from_file_location('elpa_checks', ROOT/'tests/run_elpa_cpu_mpi.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'elpa-driver.c').write_text(checks.DRIVER)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--blas',choices=('openblas','armpl'),default='openblas')
p.add_argument('--complex-kernel',choices=('auto','block1','block2'),default='auto')
args=p.parse_args()
if args.blas=='openblas':
    blas = Path.home()/'.local/opt/openblas-0.3.34-sve256/lib'
    blas_name='openblas'
else:
    blas = Path.home()/'.local/opt/armpl-26.07/armpl_26.07_gcc/lib'
    blas_name='armpl_lp64'
tag=f'elpa-{args.blas}-{args.complex_kernel}'
command = ['/usr/lib64/openmpi/bin/mpicc', '-O2', '-fopenmp',
           '-mcpu=neoverse-v1', '-msve-vector-bits=256',
           '-DOPENMX_TEST_ELPA_WORKERS', '-Wl,--wrap=omp_get_thread_num',
           '-I'+str(ROOT/'source'), '-I'+str(BUILD/'cpu-elpa/include/elpa_openmp-2026.02.002'),
           str(OUT/'elpa-driver.c'), str(ROOT/'source/openmx_elpa_cpu.c'),
           str(ROOT/'source/openmx_env.c'),
           str(BUILD/'cpu-elpa/lib/libelpa_openmp.a'), str(BUILD/'cpu-scalapack/lib/libscalapack.a'),
           '-L'+str(blas), '-Wl,-rpath,'+str(blas), '-l'+blas_name,
           '-lmpi_mpifh', '-lgfortran', '-lm', '-ldl', '-o', str(OUT/(tag+'-check'))]
env = os.environ.copy()
env.update(OMPI_CC='/usr/bin/gcc14-gcc', OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1',
           OPENBLAS_NUM_THREADS='1', OPENMX_GPU='0')
env.pop('ELPA_DEFAULT_complex_kernel',None)
if args.complex_kernel!='auto':
    env['ELPA_DEFAULT_complex_kernel']='ELPA_2STAGE_COMPLEX_SVE256_'+args.complex_kernel.upper()
subprocess.run(command, env=env, check=True)
for ranks in (6, 8):
    for threads in (1, 2):
        env['OMP_NUM_THREADS'] = str(threads)
        with (OUT/f'{tag}-{ranks}x{threads}.log').open('w') as log:
            subprocess.run(['/usr/lib64/openmpi/bin/mpirun', '-np', str(ranks),
                            '--map-by', f'slot:PE={threads}', '--bind-to', 'core',
                            str(OUT/(tag+'-check'))], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=180)
        print(f'ELPA {ranks} ranks x {threads} threads: PASS', flush=True)
