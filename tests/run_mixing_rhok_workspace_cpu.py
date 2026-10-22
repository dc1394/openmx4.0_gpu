#!/usr/bin/env python3
"""Check production density-mixing dispatch/inverses with the actual C++ owner.

MPI abort and LAPACK/eigensolver calls are modeled. Native inverse loops,
matrix layouts, scratch acquisition and dispatch are compiled unchanged.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    rho = (ROOT / 'source/DIIS_Mixing_Rhok.c').read_text()
    kerker = (ROOT / 'source/Kerker_Mixing_Rhok.c').read_text()
    names = ['Inverse', 'Inverse2', 'Complex_Inverse', 'DIIS_Mixing_Rhok']
    bodies = [function(rho, name) for name in names]
    bodies.append(function(kerker, 'Kerker_Mixing_Rhok'))
    generated = '\n'.join(body[:body.index('{')].rstrip() + ';' for body in bodies)
    generated += '\n' + '\n'.join(bodies)
    with tempfile.TemporaryDirectory(prefix='openmx-mixing-rhok-workspace-') as directory:
        folder = Path(directory)
        (folder / 'mpi.h').write_text('#pragma once\ntypedef int MPI_Comm;\n')
        (folder / 'omp.h').write_text('#pragma once\nstatic int omp_get_thread_num(){return 0;}\nstatic int omp_get_max_threads(){return 2;}\n')
        (folder / 'mixing_rhok_workspace.inc').write_text(generated)
        executable = folder / 'check'
        command = compiler_command('c++') + [
            '-I' + str(folder), '-I' + str(ROOT / 'source'), '-I' + str(ROOT / 'core/include'),
            '-I' + str(ROOT / 'core/tests'), str(ROOT / 'tests/mixing_rhok_workspace_harness.cpp'),
            allocation_fault_source(folder), str(library), '-Wl,-rpath,' + str(library.parent),
            '-o', str(executable)]
        subprocess.run(command, check=True)
        environment = os.environ.copy()
        environment.update(allocation_fault_environment(folder))
        subprocess.run([str(executable)], env=environment, check=True,
                       timeout=120, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()
