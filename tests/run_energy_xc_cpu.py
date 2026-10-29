#!/usr/bin/env python3
"""Check production fine-XC assembly, flat scratch and the real Lebedev tables.

The actual C++ owner is linked. Fine-XC tests model MPI/GPU calls, radial/XC
kernels and sequential logical OpenMP teams. An independent
scalar integral and coordinate differences check the native gradients. A
separate executable tests all three actual angular tables and their moments.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command
from run_cwf_band_helpers_cpu import function, MPI

ROOT = Path(__file__).resolve().parents[1]


def closing(source, start):
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*|[{}]', re.S)
    depth = 0
    for token in tokens.finditer(source, start):
        if token[0] == '{':
            depth += 1
        elif token[0] == '}':
            depth -= 1
            if not depth:
                return token.end()
    raise ValueError('unclosed production block')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    source = function((ROOT/'source/Total_Energy.c').read_text(), 'Calc_EXC_EH1')
    marker = source.index('    OpenMXLebedevRuleData lebedev;')
    start = source.rindex('if (Exc0_correction_flag==1){', 0, marker)
    native = source[start:closing(source, source.index('{', start))]
    pragma = re.search(r'^#pragma omp parallel[^\n]*', native, re.M)
    opening = native.index('{', pragma.end())
    end = closing(native, opening)
    # Execute the untouched worker body once for each logical worker. This
    # checks the actual partition/worker IDs without a toolchain-specific OMP
    # dependency or mixing sanitizer runtimes across the core boundary.
    native = (native[:pragma.start()] +
              'for(int energy_fixture_size=next_team(), energy_fixture_worker=0; '
              'energy_fixture_worker<energy_fixture_size; energy_fixture_worker++)' +
              native[pragma.end():opening+1]+'\nenergy_fixture_thread=energy_fixture_worker;\n'+
              native[opening+1:end]+'\nenergy_fixture_thread=0;energy_fixture_team=1;\n'+native[end:])
    with tempfile.TemporaryDirectory(prefix='openmx-energy-xc-') as directory:
        folder = Path(directory)
        (folder/'mpi.h').write_text(MPI)
        (folder/'omp.h').write_text('extern int energy_fixture_thread,energy_fixture_team;\n'
                                  'static inline int omp_get_max_threads(void){return 4;}\n'
                                  'static inline int omp_get_thread_num(void){return energy_fixture_thread;}\n'
                                  'static inline int omp_get_num_threads(void){return energy_fixture_team;}\n'
                                  'static inline int omp_get_num_procs(void){return 4;}\n')

        def build(name, backend=0, code=native, production_grid=False):
            (folder/'xc_native.inc').write_text(code)
            executable = folder/name
            flags = ['-DOPENMX_GPU_HIP='+str(backend)]
            if production_grid:
                flags += ['-DENERGY_GRID_PRODUCTION=1']
            subprocess.run(compiler_command('c')+flags+['-I'+str(folder),
                '-I'+str(ROOT/'source'), '-I'+str(ROOT/'core/include'), '-x', 'c',
                str(ROOT/'tests/energy_xc_harness.c'), '-x', 'none', str(library),
                '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        for backend in [0, 1]:
            executable = build('xc-'+str(backend), backend)
            subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
            for invalid in ['0', '-1', '591']:
                failed = subprocess.run([str(executable), invalid], cwd=folder, capture_output=True,
                                        text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 77 and 'unsupported Lebedev grid' in failed.stderr, failed
        executable = build('actual-angular-tables', production_grid=True)
        subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
        if args.negative_control:
            original = 'partial_z=OpenMX_Energy_XCForceRow(&xc_partials,2,OMPID);'
            assert native.count(original) == 1
            broken = native.replace(original, 'partial_z=OpenMX_Energy_XCForceRow(&xc_partials,1,OMPID);')
            executable = build('aliased-gradient-component', code=broken)
            failed = subprocess.run([str(executable)], cwd=folder, capture_output=True,
                                    text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed
    print('PASS: 32 fine-XC cases, finite-difference gradients, actual angular moments and an alias negative control')


if __name__ == '__main__':
    main()
