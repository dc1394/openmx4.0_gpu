#!/usr/bin/env python3
"""Check production stress halo exchange and kinetic strain with real C++ ownership.

Without MPI arguments, collectives and one logical worker are modeled.
With MPI arguments, four MPI ranks and available OpenMP workers exercise the same
production blocks, including an empty rank, actual ring exchange and barriers.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command
from run_cwf_band_helpers_cpu import MPI

ROOT = Path(__file__).resolve().parents[1]


def blocks(source):
    first = source.index('    CDM to CDM0')
    halo_start = source.index('  CDM0=halo->density_rows;')
    assert halo_start < first
    halo_end = source.index('  /* Initialization of Stress_Tensor */', first)
    kinetic_start = source.index('  dtime(&stime);', source.index('#2 of Stress'))
    kinetic_end = source.rindex('/***', kinetic_start, source.index('#3 of Stress', kinetic_start))
    return source[halo_start:halo_end], source[kinetic_start:kinetic_end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--mpi-cxx')
    parser.add_argument('--mpiexec')
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    if bool(args.mpi_cxx) != bool(args.mpiexec):
        parser.error('--mpi-cxx and --mpiexec must be supplied together')
    library = args.core_library.resolve(strict=True)
    halo, kinetic = blocks((ROOT/'source/Stress.c').read_text())
    with tempfile.TemporaryDirectory(prefix='openmx-stress-halo-') as directory:
        folder = Path(directory)
        openmp = False
        if args.mpi_cxx:
            probe = folder/'openmp_probe.c'
            probe.write_text('#include <omp.h>\nint main(void){return omp_get_max_threads()<1;}\n')
            command = compiler_command('c', args.mpi_cxx)+['-fopenmp', '-x', 'c', str(probe), '-o', str(folder/'openmp-probe')]
            checked = subprocess.run(command, capture_output=True, text=True)
            openmp = checked.returncode == 0
        if not args.mpi_cxx:
            (folder/'mpi.h').write_text(MPI.replace('#endif',
                'int MPI_Isend(const void*,int,int,int,int,MPI_Comm,MPI_Request*);\n'
                'int MPI_Recv(void*,int,int,int,int,MPI_Comm,MPI_Status*);\n'
                'int MPI_Wait(MPI_Request*,MPI_Status*);\nint MPI_Finalize(void);\n#endif'))
        if not openmp:
            (folder/'omp.h').write_text('static inline int omp_get_max_threads(void){return 1;}\n'
                'static inline int omp_get_thread_num(void){return 0;}\n'
                'static inline int omp_get_num_threads(void){return 1;}\n'
                'static inline int omp_get_num_procs(void){return 1;}\n')
        env = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))
               and k not in ['LD_PRELOAD', 'LD_LIBRARY_PATH']}
        env.update(OMP_NUM_THREADS='2', OMP_DYNAMIC='FALSE', OMP_PROC_BIND='false')
        if args.mpi_cxx:
            env['ASAN_OPTIONS'] = 'detect_leaks=0'  # MPI keeps process runtime allocations.

        def build(name, halo_code, kinetic_code):
            (folder/'stress_halo_native.inc').write_text(halo_code)
            (folder/'stress_kinetic_native.inc').write_text(kinetic_code)
            exe = folder/name
            command = compiler_command('c', args.mpi_cxx)
            if args.mpi_cxx:
                command += ['-DSTRESS_REAL_MPI=1', '-DSTRESS_REAL_OPENMP='+str(int(openmp))]
                if openmp:
                    command += ['-fopenmp']
            command += ['-I'+str(folder), '-I'+str(ROOT/'source'), '-I'+str(ROOT/'core/include'),
                '-x', 'c', str(ROOT/'tests/stress_halo_harness.c'), '-x', 'none', str(library),
                '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(exe)]
            subprocess.run(command, check=True)
            return exe

        def run(exe, *extra, **kwargs):
            command = [str(exe), *extra]
            if args.mpiexec:
                command = [args.mpiexec, '-np', '4', *command]
            return subprocess.run(command, cwd=folder, env=env, timeout=120,
                                  preexec_fn=bounded_stack, **kwargs)

        exe = build('stress-halo', halo, kinetic)
        run(exe, check=True)
        for mode in ['count', 'orbitals']:
            failed = run(exe, mode, capture_output=True, text=True)
            assert failed.returncode != 0 and 'invalid stress' in failed.stderr, failed
        if args.negative_control:
            key = 'CDM[spin][Mc_AN][h_AN][i][j]'
            assert halo.count(key) == 2
            for name, h, k in [
                ('density-spin', halo.replace(key, 'CDM[0][Mc_AN][h_AN][i][j]'), kinetic),
                ('kinetic-alias', halo, kinetic.replace('"stress:kinetic:z"', '"stress:kinetic:y"'))]:
                failed = run(build(name, h, k), capture_output=True, text=True)
                assert failed.returncode != 0 and 'FAIL line' in failed.stderr, failed
    print('PASS: production halo/kinetic blocks, ownership, strain derivatives and two negative controls')


if __name__ == '__main__':
    main()
