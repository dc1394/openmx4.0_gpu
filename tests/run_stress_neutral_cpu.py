#!/usr/bin/env python3
"""Check native neutral-grid/cutoff/stress with real C++ scratch ownership.

MPI, sequential logical OpenMP teams and radial/two-center producers are
modeled. Grid coordinates, polynomial potential moments and independent
ordered-pair energy strain check the production control and reductions.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command
from run_cwf_band_helpers_cpu import function, MPI
from run_energy_xc_cpu import closing

ROOT = Path(__file__).resolve().parents[1]


def logical_workers(code):
    for pragma in reversed(list(re.finditer(r'^#pragma omp parallel[^\n]*', code, re.M))):
        opening = code.index('{', pragma.end())
        end = closing(code, opening)
        assert code[end-1] == '}'
        code = (code[:pragma.start()] +
                'for(int neutral_team=fixture_next_team(), neutral_worker=0; '
                'neutral_worker<neutral_team; neutral_worker++)' +
                code[pragma.end():opening+1]+'\nfixture_thread=neutral_worker;\n'+
                code[opening+1:end]+'\nfixture_thread=0;fixture_team=1;\n'+code[end:])
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    source = (ROOT/'source/Stress.c').read_text()
    neutral_header = (ROOT/'source/openmx_neutral_atom.h').read_text()
    native = logical_workers(function(source, 'EH0_TwoCenter_at_Cutoff')+'\n'+function(source, 'Stress_H0'))
    with tempfile.TemporaryDirectory(prefix='openmx-stress-neutral-') as directory:
        folder = Path(directory)
        (folder/'mpi.h').write_text(MPI)
        (folder/'omp.h').write_text('extern int fixture_thread,fixture_team;\n'
            'static inline int omp_get_max_threads(void){return 4;}\n'
            'static inline int omp_get_thread_num(void){return fixture_thread;}\n'
            'static inline int omp_get_num_threads(void){return fixture_team;}\n'
            'static inline int omp_get_num_procs(void){return 4;}\n')

        def build(name, code, neutral=neutral_header, reference64=False):
            (folder/'stress_neutral_native.inc').write_text(code)
            (folder/'openmx_neutral_atom.h').write_text(neutral)
            executable = folder/name
            flags = ['-DOPENMX_TEST_REFERENCE_BINARY64=1'] if reference64 else []
            subprocess.run(compiler_command('c')+flags+['-I'+str(folder), '-I'+str(ROOT/'source'),
                '-I'+str(ROOT/'core/include'), '-x', 'c', str(ROOT/'tests/stress_neutral_harness.c'),
                '-x', 'none', str(library), '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        executable = build('stress-neutral', native)
        subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
        subprocess.run([str(build('stress-neutral-binary64', native, reference64=True))],
                       cwd=folder, check=True, preexec_fn=bounded_stack)
        for mode in ['axis', 'workers', 'peers']:
            failed = subprocess.run([str(executable), mode], cwd=folder, capture_output=True,
                                    text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 77 and 'invalid stress' in failed.stderr, failed
        if args.negative_control:
            for name, original, replacement, location in [
                ('grid-axis', 'z = axis[n3];', 'z = axis[0];', 'header'),
                ('cutoff-distance', 'z2 = z - rcut;', 'z2 = z - 0.5*rcut;', 'header')]:
                text = native if location == 'native' else neutral_header
                assert text.count(original) == 1
                changed = text.replace(original, replacement)
                executable = (build(name, changed) if location == 'native' else
                              build(name, native, changed))
                failed = subprocess.run([str(executable)], cwd=folder,
                                        capture_output=True, text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed
    print('PASS: neutral grid/cutoff/stress, independent moments/strain, ownership and negative controls')


if __name__ == '__main__':
    main()
