#!/usr/bin/env python3
"""Exercise portable density-mixing state/control with the actual shared core.

Compile production constructors, cache ownership and the unmodified preflight
and memory-log blocks. MPI abort is modeled; no GPU or BLAS SDK is needed.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import (bounded_stack, compiler_command,
                                       allocation_fault_source, allocation_fault_environment)
from run_cluster_col_state_cpu import function

ROOT = Path(__file__).resolve().parents[1]


def block(text, start):
    opening = text.index('{', start)
    end, depth = opening + 1, 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    text = (ROOT / 'source/DIIS_Mixing_Rhok.c').read_text()
    kerker = (ROOT / 'source/Kerker_Mixing_Rhok.c').read_text()
    names = ['Mix_Fast_Enabled', 'MixRhok_Fast_Invalidate', 'MixRhok_FastResize',
             'MixRhok_RotateSlots', 'MixRhok_Reset', 'Mixing_Rhok_Bind_Resources']
    bodies = [function(text, name) for name in names]
    generated = '\n'.join(body[:body.index('{')].rstrip() + ';' for body in bodies)
    generated += '\n' + '\n'.join(bodies)
    normal = function(text, 'DIIS_Mixing_Rhok_Normal')
    begin = normal.index('int use_fast = Mix_Fast_Enabled();')
    end = normal.index('if (incremental){', begin)
    generated += '\nstatic int preflight(MixingRhokState *state,int SCF_iter,int NumMix,int spinmax) {\n'
    generated += normal[begin:end] + '\nreturn incremental;\n}\n'
    # Keep the actual logging condition/statements for all four paths. The
    # fixture invokes these blocks independently of their large grid loops.
    for source, prefix in [(text, 'diis'), (kerker, 'kerker')]:
        for variant in ['normal', 'negf']:
            suffix = 'Normal' if variant == 'normal' else 'NEGF'
            fragment = function(source, ('DIIS' if prefix == 'diis' else 'Kerker') + '_Mixing_Rhok_' + suffix)
            field = prefix + '_first_' + variant
            start = fragment.index('if (state->' + field + ')')
            if prefix == 'diis':
                selected = block(fragment, start)
            else:
                end = fragment.index('state->' + field + '=0;', start) + len('state->' + field + '=0;')
                selected = fragment[start:end]
            generated += '\nstatic void log_' + prefix + '_' + variant + '(MixingRhokState *state) {\n'
            generated += 'const int size_Re_OptRhok=8,size_Kerker_weight=4;\n' + selected + '\n}\n'
    with tempfile.TemporaryDirectory(prefix='openmx-mixing-rhok-state-') as directory:
        folder = Path(directory)
        (folder / 'mixing_rhok_state.inc').write_text(generated)
        executable = folder / 'check'
        command = compiler_command('c++') + [
            '-I' + str(folder), '-I' + str(ROOT / 'source'), '-I' + str(ROOT / 'core/include'),
            '-I' + str(ROOT / 'core/tests'), str(ROOT / 'tests/mixing_rhok_state_harness.cpp'),
            allocation_fault_source(folder), str(ROOT / 'source/openmx_env.c'), str(library),
            '-Wl,-rpath,' + str(library.parent), '-o', str(executable)]
        subprocess.run(command, check=True)
        environment = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
        environment.update(allocation_fault_environment(folder))
        subprocess.run([str(executable)], env=environment, check=True,
                       timeout=120, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()
