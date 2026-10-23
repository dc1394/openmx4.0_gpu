#!/usr/bin/env python3
"""Check the production VNA derivative with C++ owned outputs.

Fixed input row tables, damping and sequential logical workers are modeled.
The full application regressions exercise native MPI/OpenMP and damping.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command
from run_cwf_band_helpers_cpu import function, MPI

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    native = function((ROOT/'source/Stress.c').read_text(), 'dHVNA')
    with tempfile.TemporaryDirectory(prefix='openmx-stress-vna-') as directory:
        folder = Path(directory)
        (folder/'mpi.h').write_text(MPI)
        (folder/'omp.h').write_text('extern int fixture_worker;\n'
            'static inline int omp_get_max_threads(void){return 4;}\n'
            'static inline int omp_get_thread_num(void){return fixture_worker;}\n')

        def build(name, code):
            (folder/'stress_vna_native.inc').write_text(code)
            executable = folder/name
            subprocess.run(compiler_command('c')+['-I'+str(folder), '-I'+str(ROOT/'source'),
                '-I'+str(ROOT/'core/include'), '-x', 'c', str(ROOT/'tests/stress_vna_harness.c'),
                '-x', 'none', str(library), '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        executable = build('stress-vna', native)
        subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
        for mode in ['radial', 'angular', 'orbitals']:
            failed = subprocess.run([str(executable), mode], cwd=folder, capture_output=True,
                                    text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 77 and 'invalid stress VNA' in failed.stderr, failed
        if args.negative_control:
            for name, original, replacement in [
                ('axis-alias', 'openmx_solver_real_matrix_row(Hz,', 'openmx_solver_real_matrix_row(Hy,'),
                ('strain-transpose', 'OpenMX_Stress_VnaStrainRow(H_sts,1,2,m)',
                 'OpenMX_Stress_VnaStrainRow(H_sts,2,1,m)')]:
                assert original in native
                failed = subprocess.run([str(build(name, native.replace(original, replacement)))], cwd=folder,
                                        capture_output=True, text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed
    print('PASS: VNA branches, rectangular shapes, ownership reuse, independent strain and two negative controls')


if __name__ == '__main__':
    main()
