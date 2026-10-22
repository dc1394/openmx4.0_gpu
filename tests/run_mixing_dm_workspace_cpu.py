#!/usr/bin/env python3
"""Check real-space mixing ownership and production inverses with the actual core.

Compile production state constructors, memory-log blocks and entry/inverse
adapters. The actual shared C++ core performs the inverse arithmetic. Large
numerical callbacks and MPI abort are modeled; no GPU/BLAS SDK is used.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import (bounded_stack, compiler_command,
                                       allocation_fault_source, allocation_fault_environment)
from run_cluster_col_state_cpu import function
from run_mixing_rhok_state_cpu import block

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    sources = {name: (ROOT / 'source' / (name + '.c')).read_text()
               for name in ['DIIS_Mixing_DM', 'ADIIS_Mixing_DM', 'GR_Pulay_DM']}
    generated = '\n'.join(function(sources['DIIS_Mixing_DM'], name)
                          for name in ['MixDM_Reset', 'Mixing_DM_Bind_Resources'])
    for name, namespace in [('DIIS_Mixing_DM', 'diis'), ('ADIIS_Mixing_DM', 'adiis'), ('GR_Pulay_DM', 'gr')]:
        text = sources[name]
        generated += '\n' + function(text, name) + '\n'
        generated += '\nnamespace ' + namespace + ' {\n' + function(text, 'Inverse') + '\n}\n'
        if namespace != 'adiis':
            body = function(text, name + '_Compute')
            field = 'diis_first' if namespace == 'diis' else 'gr_first'
            selected = block(body, body.index('if (state->' + field + ')'))
            generated += '\nstatic void log_' + namespace + '(OpenMXMixingDMState *state) {\n'
            generated += 'const int size_OptRDM=26,size_iOptRDM=26;\n' + selected + '\n}\n'
    with tempfile.TemporaryDirectory(prefix='openmx-mixing-dm-workspace-') as directory:
        folder = Path(directory)
        (folder / 'mpi.h').write_text('#pragma once\ntypedef int MPI_Comm;\n')
        (folder / 'omp.h').write_text('#pragma once\nstatic int omp_get_thread_num(){return 0;}\nstatic int omp_get_max_threads(){return 2;}\n')
        (folder / 'mixing_dm_workspace.inc').write_text(generated)
        executable = folder / 'check'
        command = compiler_command('c++') + [
            '-I' + str(folder), '-I' + str(ROOT / 'source'), '-I' + str(ROOT / 'core/include'),
            '-I' + str(ROOT / 'core/tests'), str(ROOT / 'tests/mixing_dm_workspace_harness.cpp'),
            allocation_fault_source(folder), str(library), '-Wl,-rpath,' + str(library.parent),
            '-o', str(executable)]
        subprocess.run(command, check=True)
        environment = os.environ.copy()
        environment.update(allocation_fault_environment(folder))
        subprocess.run([str(executable)], env=environment, check=True,
                       timeout=120, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()
