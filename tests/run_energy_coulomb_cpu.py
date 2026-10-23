#!/usr/bin/env python3
"""Check production nuclear/neutral-atom energy and force assembly.

The actual C++ owner and production C control/reductions are used. MPI and
one-worker OpenMP are modeled. A prescribed two-center kernel lets independent
ordered-pair sums check energies, forces, decomposition and flat batch packing;
MD initialization also exercises the production cutoff quadrature.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command
from run_cwf_band_helpers_cpu import MPI

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    source = (ROOT/'source/Total_Energy.c').read_text()
    start = source.index('\ndouble Calc_Ecore(')
    end = source.index('\n#pragma optimization_level 1\nvoid Calc_Edc(', start)
    cutoff = source.index('\nvoid EH0_TwoCenter_at_Cutoff(')
    native = source[start:end]+source[cutoff:source.index('\ndouble Calc_Ehub(', cutoff)]
    with tempfile.TemporaryDirectory(prefix='openmx-energy-coulomb-') as directory:
        folder = Path(directory)
        (folder/'mpi.h').write_text(MPI)
        (folder/'omp.h').write_text('static inline int omp_get_max_threads(void){return 1;}\n'
                                  'static inline int omp_get_thread_num(void){return 0;}\n'
                                  'static inline int omp_get_num_threads(void){return 1;}\n'
                                  'static inline int omp_get_num_procs(void){return 1;}\n')

        def build(name, backend, code=native):
            (folder/'coulomb_native.inc').write_text(code)
            executable = folder/name
            subprocess.run(compiler_command('c')+['-DOPENMX_GPU_HIP='+str(backend),
                '-I'+str(folder), '-I'+str(ROOT/'source'), '-I'+str(ROOT/'core/include'),
                '-x', 'c', str(ROOT/'tests/energy_coulomb_harness.c'), '-x', 'none', str(library),
                '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        for backend in [0, 1]:
            executable = build('coulomb-'+str(backend), backend)
            subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
        if args.negative_control:
            original = 'out3=openmx_solver_real_matrix_row(&batch.values,7);'
            assert native.count(original) == 1
            broken = native.replace(original, 'out3=openmx_solver_real_matrix_row(&batch.values,6);')
            executable = build('aliased-force-component', 0, broken)
            failed = subprocess.run([str(executable)], cwd=folder, capture_output=True,
                                    text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed
    print('PASS: 96 Coulomb cases and a force-component alias negative control')


if __name__ == '__main__':
    main()
