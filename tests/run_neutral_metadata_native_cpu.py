#!/usr/bin/env python3
"""Compile actual native metadata allocation/release as C11/C++17.

Use the real C++ owner, including preserved raw AtomSpecies values and main
numeric-grid lifetime independence. Other case-2 arrays and MPI are modeled.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command
from run_cwf_band_helpers_cpu import function

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    allocation = (ROOT/'source/Allocate_Arrays.c').read_text()
    begin = allocation.index('    case 2:')
    allocation = allocation[begin:allocation.index('      FNAN =', begin)]
    release = (ROOT/'source/Free_Arrays.c').read_text()
    begin = release.index('  openmx_core_neutral_grid_metadata_release(core);')
    release = release[begin:release.index('  free(FNAN);', begin)]
    native = ('static void native_allocate(OpenMXCoreContext *core){switch(2){\n'+
              allocation+'\nbreak;}}\nstatic void native_free(OpenMXCoreContext *core){\n'+release+'\n}\n')
    with tempfile.TemporaryDirectory(prefix='openmx-neutral-metadata-') as directory:
        folder = Path(directory)

        def build(name, language, code=native):
            (folder/'neutral_metadata_native.inc').write_text(code)
            executable = folder/name
            subprocess.run(compiler_command(language)+['-I'+str(folder), '-I'+str(ROOT/'source'),
                '-I'+str(ROOT/'core/include'), '-x', language,
                str(ROOT/'tests/neutral_metadata_native_harness.c'), '-x', 'none', str(library),
                '-Wl,-rpath,'+str(library.parent), '-o', str(executable)], check=True)
            return executable

        for language in ['c', 'c++']:
            executable = build('native-'+language, language)
            subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
            for invalid in ['context', 'shape', 'range']:
                failed = subprocess.run([str(executable), invalid], cwd=folder, capture_output=True,
                                        text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 77 and 'Allocate_Arrays: neutral grid metadata:' in failed.stderr, failed
        if args.negative_control:
            for name, old, new in [
                ('wrong-shape', 'shape={(size_t)SpeciesNum};', 'shape={(size_t)SpeciesNum+1};'),
                ('missing-release', 'openmx_core_neutral_grid_metadata_release(core);', '/* missing release */')]:
                assert native.count(old) == 1
                failed = subprocess.run([str(build(name, 'c', native.replace(old, new)))], cwd=folder,
                    capture_output=True, text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed

        # Compile the production input-validation and raw AtomSpecies blocks.
        # Use the current callback boundary and real RAII PAO/VPS readers.
        # Only the publication of the atomic view is modeled in this fixture.
        source = (ROOT/'source/SetPara_DFT.c').read_text()
        start = source.index('  OpenMXNeutralGridMetadataView metadata;',
                             source.rindex('\nstatic int ReadPara_DFT('))
        boundary = source[start:source.index('\n  int spe,po;', start)]
        blocks = ['static int metadata_input_boundary(OpenMXCoreContext *core,int SpeciesNum,\n'
                  '    char *error,size_t error_capacity){\n'+boundary+'\nreturn 1;\n}',
                  function(source, 'Read_PAO')]
        start = source.rindex('\nstatic int Read_VPS(')
        # Retain the actual raw-species check, capacity guards and ghost-density
        # reset. Later projector/Hartree assembly has separate native tests.
        blocks.append(source[start:source.index('  atomic->species.core_charge[spe]=', start)]+
                      '\n  return 1;\n}')
        input_native = '\n'.join(blocks)

        def build_input(name, language, code=input_native):
            (folder/'neutral_metadata_input.inc').write_text(code)
            executable = folder/name
            subprocess.run(compiler_command(language)+['-I'+str(folder), '-I'+str(ROOT/'core/include'),
                '-x', language, str(ROOT/'tests/neutral_metadata_input_harness.c'), '-x', 'none', str(library),
                '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        for language in ['c', 'c++']:
            executable = build_input('input-'+language, language)
            subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
            for mode in ['boundary-context', 'boundary-empty', 'boundary-shape']:
                failed = subprocess.run([str(executable), mode], cwd=folder, capture_output=True,
                                        text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 0 and 'neutral grid metadata unavailable or shape mismatch' in failed.stdout, failed
            for pao, vps in [('.25', '.75'), ('14', '6'), ('0', '14'), ('.25000000000000006', '.25')]:
                failed = subprocess.run([str(executable), 'mismatch', pao, vps], cwd=folder,
                                        capture_output=True, text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 0 and 'Not the same atom in PAO and VPS' in failed.stdout, failed
        if args.negative_control:
            old = 'metadata->spacing[spe]!=data->atom_species && data->atomic_number!=0'
            assert input_native.count(old) == 1
            changed = input_native.replace(old, '(int)metadata->spacing[spe]!=(int)data->atom_species && data->atomic_number!=0')
            executable = build_input('lost-fractional-species', 'c', changed)
            failed = subprocess.run([str(executable), 'mismatch', '.25', '.75'], cwd=folder,
                                    capture_output=True, text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed
    print('PASS: native metadata C11/C++17 adapters, actual C++ owner and negative controls')


if __name__ == '__main__':
    main()
