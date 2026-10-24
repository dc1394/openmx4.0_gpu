#!/usr/bin/env python3
"""Check the native owned-transfer adapter with real C++ resources/exchanges.

MPI abort is modeled. Instrumented calls forward to the actual destroy/adopt
APIs, so destructor counts verify normal, callback failure and exception exits.
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
    header = (ROOT/'source/openmx_projector_exchange.h').read_text()
    with tempfile.TemporaryDirectory(prefix='openmx-projector-scope-') as directory:
        folder = Path(directory)
        (folder/'mpi.h').write_text(MPI)

        def build(name, code):
            (folder/'openmx_projector_exchange.h').write_text(code)
            executable = folder/name
            subprocess.run(compiler_command('c++')+['-I'+str(folder), '-I'+str(ROOT/'source'),
                '-I'+str(ROOT/'core/include'), str(ROOT/'tests/projector_scope_harness.cpp'),
                str(library), '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        executable = build('projector-scope', header)
        subprocess.run([str(executable)], cwd=folder, check=True, preexec_fn=bounded_stack)
        for mode in ['adopt', 'create', 'resources', 'output']:
            failed = subprocess.run([str(executable), mode], cwd=folder, capture_output=True,
                                    text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 77 and 'Projector exchange:' in failed.stderr, failed
        if args.negative_control:
            original = 'openmx_core_native_adopt(resources,OPENMX_EXCHANGE_RESOURCE_HANDLE'
            assert header.count(original) == 1
            failed = subprocess.run([str(build('unowned-handle', header.replace(original,
                'discard_adoption(resources,OPENMX_EXCHANGE_RESOURCE_HANDLE')))], cwd=folder,
                capture_output=True, text=True, preexec_fn=bounded_stack)
            assert failed.returncode != 0 and 'FAIL line' in failed.stderr, failed
    print('PASS: real owned projector resources, normal/failure/exception cleanup and unowned-handle negative control')


if __name__ == '__main__':
    main()
