#!/usr/bin/env python3
"""Check production fine-XC stress with real C++ ownership and scalar oracles.

MPI, radial/XC functions and sequential logical OpenMP teams are modeled.
Independent coordinate and center/image strain differences check the outputs.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command
from run_cwf_band_helpers_cpu import MPI
from run_energy_xc_cpu import closing

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    source = (ROOT/'source/Stress.c').read_text()
    marker = source.index('    OpenMXLebedevRuleData lebedev;')
    start = source.rindex('if (Exc0_correction_flag==1){', 0, marker)
    native = source[start:closing(source, source.index('{', start))]
    pragma = re.search(r'^#pragma omp parallel[^\n]*', native, re.M)
    opening = native.index('{', pragma.end())
    end = closing(native, opening)
    native = (native[:pragma.start()] +
              'for(int energy_fixture_size=next_team(), energy_fixture_worker=0; '
              'energy_fixture_worker<energy_fixture_size; energy_fixture_worker++)' +
              native[pragma.end():opening+1]+'\nenergy_fixture_thread=energy_fixture_worker;\n'+
              native[opening+1:end]+'\nenergy_fixture_thread=0;energy_fixture_team=1;\n'+native[end:])
    with tempfile.TemporaryDirectory(prefix='openmx-stress-xc-') as directory:
        folder = Path(directory)
        (folder/'mpi.h').write_text(MPI)
        (folder/'omp.h').write_text('extern int energy_fixture_thread,energy_fixture_team;\n'
            'static inline int omp_get_max_threads(void){return 4;}\n'
            'static inline int omp_get_thread_num(void){return energy_fixture_thread;}\n'
            'static inline int omp_get_num_threads(void){return energy_fixture_team;}\n'
            'static inline int omp_get_num_procs(void){return 4;}\n')

        def build(name, code):
            (folder/'stress_xc_native.inc').write_text(code)
            executable = folder/name
            subprocess.run(compiler_command('c')+['-I'+str(folder), '-I'+str(ROOT/'source'),
                '-I'+str(ROOT/'core/include'), '-x', 'c', str(ROOT/'tests/stress_xc_harness.c'),
                '-x', 'none', str(library), '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        executable = build('stress-xc', native)
        subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
        for points in ['0', '-1', '591']:
            failed = subprocess.run([str(executable), points], cwd=folder, capture_output=True,
                                    text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 77 and 'unsupported Lebedev grid' in failed.stderr, failed
        if args.negative_control:
            for name, original, replacement in [
                ('partial-alias', 'partial_z=OpenMX_XC_ForceRow(&xc_partials,2,OMPID);',
                 'partial_z=OpenMX_XC_ForceRow(&xc_partials,1,OMPID);'),
                ('strain-component', 'sumt1[8] += dz1*dz2;', 'sumt1[8] += dz1*dy2;')]:
                assert native.count(original) == 1
                failed = subprocess.run([str(build(name, native.replace(original, replacement)))], cwd=folder,
                                        capture_output=True, text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed
    print('PASS: fine stress, coordinate/strain oracles and two component negative controls')


if __name__ == '__main__':
    main()
