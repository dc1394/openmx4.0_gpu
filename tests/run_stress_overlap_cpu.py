#!/usr/bin/env python3
"""Check production overlap exchange with actual C++ owned messages.

The serial variant checks ownership and the no-peer path. The optional real
four-rank variant checks rectangular primitive-orbital blocks, four derivative
components, an empty rank and messages larger than 64 KiB.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command
from run_cwf_band_helpers_cpu import function, MPI

ROOT = Path(__file__).resolve().parents[1]


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
    native = function((ROOT/'source/Stress.c').read_text(), 'MPI_OLP')
    header = (ROOT/'source/openmx_stress_workspace.h').read_text()
    with tempfile.TemporaryDirectory(prefix='openmx-stress-overlap-') as directory:
        folder = Path(directory)
        if not args.mpi_cxx:
            (folder/'mpi.h').write_text(MPI.replace('#endif',
                'int MPI_Isend(const void*,int,int,int,int,MPI_Comm,MPI_Request*);\n'
                'int MPI_Recv(void*,int,int,int,int,MPI_Comm,MPI_Status*);\n'
                'int MPI_Wait(MPI_Request*,MPI_Status*);\n#endif'))
        (folder/'omp.h').write_text('static inline int omp_get_max_threads(void){return 1;}\n'
                                  'static inline int omp_get_thread_num(void){return 0;}\n')
        env = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))
               and k not in ['LD_PRELOAD', 'LD_LIBRARY_PATH']}
        if args.mpi_cxx:
            env['ASAN_OPTIONS'] = 'detect_leaks=0'  # MPI retains process-runtime allocations.

        def build(name, code, workspace=header):
            (folder/'stress_overlap_native.inc').write_text(code)
            (folder/'openmx_stress_workspace.h').write_text(workspace)
            executable = folder/name
            command = compiler_command('c', args.mpi_cxx)
            if args.mpi_cxx:
                command += ['-DSTRESS_REAL_MPI=1']
            subprocess.run(command+['-I'+str(folder), '-I'+str(ROOT/'source'),
                '-I'+str(ROOT/'core/include'), '-x', 'c', str(ROOT/'tests/stress_overlap_harness.c'),
                '-x', 'none', str(library), '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        def run(executable, **kwargs):
            command = [str(executable)]
            if args.mpiexec:
                command = [args.mpiexec, '-np', '4', *command]
            return subprocess.run(command, cwd=folder, env=env, timeout=120,
                                  preexec_fn=bounded_stack, **kwargs)

        run(build('stress-overlap', native), check=True)
        if args.negative_control:
            bad_header = header.replace('"stress:overlap:receive-sizes"', '"stress:overlap:send-sizes"')
            assert bad_header != header
            failed = run(build('counts-alias', native, bad_header), capture_output=True, text=True)
            assert failed.returncode != 0 and 'FAIL line' in failed.stderr, failed
            if args.mpi_cxx:
                original = 'tmp_array[num] = OLP1[k][Mc_AN][h_AN][i][j];'
                assert native.count(original) == 1
                failed = run(build('wrong-component', native.replace(original,
                    'tmp_array[num] = OLP1[0][Mc_AN][h_AN][i][j];')), capture_output=True, text=True)
                assert failed.returncode != 0 and 'FAIL line' in failed.stderr, failed
    print('PASS: overlap ownership, primitive-orbital exchange and negative controls')


if __name__ == '__main__':
    main()
