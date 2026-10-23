#!/usr/bin/env python3
"""Check production nuclear/Hartree stress helpers and real scratch ownership.

MPI, sequential logical OpenMP teams and FFT coefficients are modeled. Nuclear
and fixed-prefactor reciprocal energies have independent strain differences.
"""
import argparse
from pathlib import Path
import re
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
    source = (ROOT/'source/Stress.c').read_text()
    native = function(source, 'Stress_core')+'\n'+function(source, 'Stress_H1')
    # Execute the real worker body sequentially for every logical worker.
    native, count = re.subn(r'^#pragma omp parallel[^\n]*',
                           'for(logical_worker=0;logical_worker<logical_team;logical_worker++)', native, flags=re.M)
    assert count == 1
    with tempfile.TemporaryDirectory(prefix='openmx-stress-workspace-') as directory:
        folder = Path(directory)
        (folder/'mpi.h').write_text(MPI)
        (folder/'omp.h').write_text('static inline int omp_get_max_threads(void){return 4;}\n'
            'static inline int omp_get_thread_num(void){return logical_worker%logical_team;}\n'
            'static inline int omp_get_num_threads(void){return logical_team;}\n'
            'static inline int omp_get_num_procs(void){return 4;}\n')

        def build(name, code):
            (folder/'stress_native.inc').write_text(code)
            executable = folder/name
            subprocess.run(compiler_command('c')+['-I'+str(folder), '-I'+str(ROOT/'source'),
                '-I'+str(ROOT/'core/include'), '-x', 'c', str(ROOT/'tests/stress_workspace_harness.c'),
                '-x', 'none', str(library), '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        executable = build('stress', native)
        subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
        for mode, diagnostic in [('workers', 'invalid stress worker count'), ('grid', 'invalid stress reciprocal grid count')]:
            failed = subprocess.run([str(executable), mode], cwd=folder, capture_output=True,
                                    text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 77 and diagnostic in failed.stderr, failed
        if args.negative_control:
            original = 'ImRhok = openmx_solver_real_matrix_row(&reciprocal,1);'
            assert native.count(original) == 1
            broken = native.replace(original, 'ImRhok = openmx_solver_real_matrix_row(&reciprocal,0);')
            failed = subprocess.run([str(build('aliased-fft-components', broken))], cwd=folder,
                                    capture_output=True, text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed
    print('PASS: production stress, padded/reused rows, strain derivatives and an FFT alias negative control')


if __name__ == '__main__':
    main()
