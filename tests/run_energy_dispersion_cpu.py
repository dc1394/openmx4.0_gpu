#!/usr/bin/env python3
"""Check production D3 energy/gradients/stress with actual C++ ownership.

MPI is modeled as one rank. Prescribed symmetric coefficient tables feed an
independent long-double energy oracle and coordinate/strain finite differences.
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
    start = source.index('\ndouble Calc_EdftD3(')
    native = source[start:source.index('\nvoid Energy_Decomposition(', start)]
    with tempfile.TemporaryDirectory(prefix='openmx-energy-dispersion-') as directory:
        folder = Path(directory)
        (folder/'mpi.h').write_text(MPI)
        (folder/'omp.h').write_text('static inline int omp_get_max_threads(void){return 1;}\n'
                                  'static inline int omp_get_thread_num(void){return 0;}\n')

        def build(name, code):
            (folder/'dispersion_native.inc').write_text(code)
            executable = folder/name
            subprocess.run(compiler_command('c')+['-I'+str(folder), '-I'+str(ROOT/'source'),
                '-I'+str(ROOT/'core/include'), '-x', 'c', str(ROOT/'tests/energy_dispersion_harness.c'),
                '-x', 'none', str(library), '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        subprocess.run([str(build('dispersion', native))], cwd=folder, check=True, preexec_fn=bounded_stack)
        if args.negative_control:
            original = '=-fdamp*fdamp*expval*k1_dftD*rcovab_dftD[wanA][wanB]/dist2;'
            assert native.count(original) == 1
            broken = native.replace(original, '=0.0;')
            executable = build('missing-coordination-gradient', broken)
            failed = subprocess.run([str(executable)], cwd=folder, capture_output=True,
                                    text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed
    print('PASS: production D3 energy, coordinate/strain derivatives and missing-gradient negative control')


if __name__ == '__main__':
    main()
