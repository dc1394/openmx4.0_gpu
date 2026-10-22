#!/usr/bin/env python3
"""Test the shared mixing state's real metadata/cleanup with the actual core.

GPU/MPI calls and worker dispatch are modeled. Production constructors,
preflight tiers, callbacks and allocation/cleanup code are compiled unchanged.
The kernels execute on modeled host memory, not a physical GPU.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from native_resource_test_bridge import calculation_state_declaration
from native_state_test_support import bounded_stack, compiler_command, allocation_fault_source, allocation_fault_environment
from run_cluster_col_state_cpu import function, pod

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    text = (ROOT / 'source/Mixing_H.c').read_text()
    names = ['MixH_AbortWithMessage', 'MixH_Align512', 'MixH_ComputeLayout',
             'MixH_GramPart', 'MixH_OptResidualPart', 'MixH_Worker',
             'MixH_Distributed_Free', 'MixH_ShardCost', 'MixH_Distributed_Allocate',
             'MixH_HostBacked_Allocate', 'MixH_Upload', 'MixH_GramDevice',
             'MixH_OptResidualDevice', 'MixH_IncComputeRow', 'Inverse', 'Inverse0', 'MixH_IncFree',
             'MixH_IncBuild', 'Mixing_H_Release_GPU', 'MixH_GpuPreflight',
             'MixH_Reset', 'Mixing_H_Bind_Resources']
    bodies = [function(text, name) for name in names]
    prototypes = '\n'.join(body[:body.index('{')].rstrip() + ';' for body in bodies)
    generated = calculation_state_declaration(text, 'MixingHState') + '\n'
    generated += text[text.index('enum { MIXH_PROBE'):text.index('typedef struct {', text.index('enum { MIXH_PROBE'))]
    generated += pod(text, 'MixH_Work') + '\n' + prototypes + '\n' + '\n'.join(bodies)
    with tempfile.TemporaryDirectory(prefix='openmx-mixing-h-state-') as directory:
        folder = Path(directory)
        # The SDK-free model executes the production host loops serially.
        # Real OpenMP mixing is also checked by paired application runs.
        (folder / 'mpi.h').write_text('#pragma once\ntypedef int MPI_Comm;\n')
        (folder / 'omp.h').write_text('#pragma once\nstatic int omp_get_thread_num(){return 0;}\nstatic int omp_get_max_threads(){return 1;}\n')
        (folder / 'mixing_h_state.inc').write_text(generated)
        for backend in ['cuda', 'hip']:
            executable = folder / backend
            command = compiler_command('c++') + [
                '-Wno-unknown-pragmas', '-DOPENMX_GPU_HIP=' + str(int(backend == 'hip')),
                '-I' + str(folder), '-I' + str(ROOT / 'source'), '-I' + str(ROOT / 'core/include'),
                '-I' + str(ROOT / 'core/tests/fixtures/native_resources'), '-I' + str(ROOT / 'core/tests'),
                str(ROOT / 'tests/mixing_h_state_harness.cpp'), allocation_fault_source(folder),
                str(ROOT / 'source/openmx_env.c'), str(library),
                '-Wl,-rpath,' + str(library.parent), '-o', str(executable)]
            subprocess.run(command, check=True)
            environment = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
            environment.update(allocation_fault_environment(folder))
            print('Shared Hamiltonian mixing state:', backend, flush=True)
            subprocess.run([str(executable)], env=environment, check=True,
                           timeout=120, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()
