#!/usr/bin/env python3
"""Check production scalar nonlocal stress and owned complex buffer layout.

MPI/logical workers/damping are modeled; the C++ owner and native derivative
are real. Scalar projector bilinears and Gaussian energy strain are independent
oracles. Complex storage checks cover all spin components; they do not assert
physical support for the application's rejected noncollinear-stress mode.
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
    source = (ROOT/'source/Stress.c').read_text()
    native = function(source, 'dHNL_SO')+'\n'+function(source, 'dHNL')
    header = (ROOT/'source/openmx_stress_workspace.h').read_text()
    with tempfile.TemporaryDirectory(prefix='openmx-stress-nonlocal-') as directory:
        folder = Path(directory)
        (folder/'mpi.h').write_text(MPI)
        (folder/'omp.h').write_text('extern int fixture_worker;\n'
            'static inline int omp_get_max_threads(void){return 4;}\n'
            'static inline int omp_get_thread_num(void){return fixture_worker;}\n')

        def build(name, code, workspace=header):
            (folder/'stress_nonlocal_native.inc').write_text(code)
            (folder/'openmx_stress_workspace.h').write_text(workspace)
            executable = folder/name
            subprocess.run(compiler_command('c')+['-I'+str(folder), '-I'+str(ROOT/'source'),
                '-I'+str(ROOT/'core/include'), '-x', 'c', str(ROOT/'tests/stress_nonlocal_harness.c'),
                '-x', 'none', str(library), '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        executable = build('stress-nonlocal', native)
        subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
        for mode in ['0', '-1']:
            failed = subprocess.run([str(executable), mode], cwd=folder, capture_output=True,
                                    text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 77 and 'invalid stress nonlocal' in failed.stderr, failed
        if args.negative_control:
            for name, code, workspace in [
                ('force-axis', native.replace('openmx_solver_complex_plane_row(Hz,',
                                             'openmx_solver_complex_plane_row(Hy,'), header),
                ('spin-stride', native, header.replace('((spin*4+axis)*view->extent[1]+derivative)',
                                                      '((spin*3+axis)*view->extent[1]+derivative)'))]:
                assert code != native or workspace != header
                failed = subprocess.run([str(build(name, code, workspace))], cwd=folder,
                                        capture_output=True, text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed
    print('PASS: native scalar nonlocal force/stress, Gaussian strain, complex ownership and negative controls')


if __name__ == '__main__':
    main()
