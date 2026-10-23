#!/usr/bin/env python3
"""Check production Stress3 with actual ownership and independent strain energies.

One logical OpenMP worker, MPI, orbitals and XC producers are modeled.
The application regressions separately exercise native MPI/OpenMP concurrency.
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
    native = function((ROOT/'source/Stress.c').read_text(), 'Stress3')
    with tempfile.TemporaryDirectory(prefix='openmx-stress-grid-') as directory:
        folder = Path(directory)
        (folder/'mpi.h').write_text(MPI)
        (folder/'omp.h').write_text('static inline int omp_get_max_threads(void){return 1;}\n'
            'static inline int omp_get_thread_num(void){return 0;}\n')

        def build(name, code):
            (folder/'stress_grid_native.inc').write_text(code)
            executable = folder/name
            subprocess.run(compiler_command('c')+['-I'+str(folder), '-I'+str(ROOT/'source'),
                '-I'+str(ROOT/'core/include'), '-x', 'c', str(ROOT/'tests/stress_grid_harness.c'),
                '-x', 'none', str(library), '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        executable = build('stress-grid', native)
        subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
        for mode in ['spin', 'points', 'orbitals', 'gradient']:
            failed = subprocess.run([str(executable), mode], cwd=folder, capture_output=True,
                                    text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 77 and 'invalid stress' in failed.stderr, failed
        if args.negative_control:
            for name, original, replacement in [
                ('derivative-axis', 'tmpz += ai_dorbs0[3*i+2] * tmp0;',
                 'tmpz += ai_dorbs0[3*i+1] * tmp0;'),
                ('gradient-component', 'OpenMX_Stress_GradientRow(&gga_b,0,spin,2)[BN]',
                 'OpenMX_Stress_GradientRow(&gga_b,0,spin,1)[BN]')]:
                assert original in native
                failed = subprocess.run([str(build(name, native.replace(original, replacement)))], cwd=folder,
                                        capture_output=True, text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed
    print('PASS: grid/orbital/GGA stress, invalid shapes and two derivative-component negative controls')


if __name__ == '__main__':
    main()
