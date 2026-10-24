#!/usr/bin/env python3
"""Check actual native grid allocation/free adapters with the real C++ owner.

The production case-4 and final-release blocks are compiled as C11/C++17;
only MPI abort is modeled. Optional orbital allocations exercise their legacy
lifecycle. Wrong-shape and missing-release negative controls check the boundary.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    allocation = (ROOT/'source/Allocate_Arrays.c').read_text()
    begin = allocation.index('    case 4:')
    allocation = allocation[begin:allocation.index('    case 5:', begin)]
    release = (ROOT/'source/Free_Arrays.c').read_text()
    begin = release.index('  /* Allocation_Arrays(4) */')
    release = release[begin:release.index('  if ( CWF_Calc==1', begin)]
    native = ('static void native_allocate(OpenMXCoreContext *core){int i,k,l;switch(4){\n'+
              allocation+'\n}}\nstatic void native_free(OpenMXCoreContext *core){int i,k,l;\n'+release+'\n}\n')
    with tempfile.TemporaryDirectory(prefix='openmx-neutral-grid-native-') as directory:
        folder = Path(directory)

        def build(name, language, code=native):
            (folder/'neutral_grid_native.inc').write_text(code)
            executable = folder/name
            subprocess.run(compiler_command(language)+['-I'+str(folder), '-I'+str(ROOT/'source'),
                '-I'+str(ROOT/'core/include'), '-x', language,
                str(ROOT/'tests/neutral_grid_native_harness.c'), '-x', 'none', str(library),
                '-Wl,-rpath,'+str(library.parent), '-o', str(executable)], check=True)
            return executable

        for language in ['c', 'c++']:
            executable = build('native-'+language, language)
            subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
            for invalid in ['context', 'shape', 'overflow']:
                failed = subprocess.run([str(executable), invalid], cwd=folder, capture_output=True,
                                        text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 77 and 'Allocate_Arrays: neutral grid:' in failed.stderr, failed
        if args.negative_control:
            for name, old, new in [
                ('wrong-shape', 'shape={SpeciesNum,Max_TGN_EH0};', 'shape={SpeciesNum,Max_TGN_EH0+1};'),
                ('missing-release', 'openmx_core_neutral_grid_release(core);', '/* missing release */')]:
                assert native.count(old) == 1
                failed = subprocess.run([str(build(name, 'c', native.replace(old, new)))], cwd=folder,
                    capture_output=True, text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed
    print('PASS: actual grid allocation/free C11/C++17 adapters, real C++ owner and negative controls')


if __name__ == '__main__':
    main()
