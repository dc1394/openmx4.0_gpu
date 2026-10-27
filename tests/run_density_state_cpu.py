#!/usr/bin/env python3
"""Exercise production density cache ownership against the actual C++ core.

SDK allocations/mappings are modeled; constructors, upload rollback, staging,
invalidation and teardown are extracted unchanged from both native backends.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from run_cluster_col_state_cpu import function, pod
from native_state_test_support import (compiler_command, bounded_stack,
                                      allocation_fault_source, allocation_fault_environment)

ROOT = Path(__file__).resolve().parents[1]


def declarations(source):
    start = source.index('typedef struct {')
    end = source.index('\n};', source.index('struct OpenMXDensityGpuState {')) + 3
    return source[start:end]


def production(backend):
    source = (ROOT / ('source/Set_Density_Grid_GPU.c' if backend == 'cuda' else
                     'source/backends/hip/src/Set_Density_Grid_GPU.c')).read_text()
    names = ['SDG_malloc', 'SDG_acc_delete_if_present', 'SDG_delete_device', 'SDG_free_host',
             'SDG_local_delete_device', 'SDG_local_free', 'SDG_upload_device', 'SDG_local_upload',
             'Set_Density_Grid_GPU_Invalidate', 'Set_Density_Grid_GPU_EndCycle',
             'SDG_Reset', 'Set_Density_Grid_GPU_Bind_Resources']
    if backend == 'cuda':
        names += ['SDG_map', 'SDG_copyin', 'SDG_create', 'SDG_staging_free', 'SDG_staging_reserve']
    else:
        names += ['acc_copyin', 'acc_create', 'acc_is_present', 'acc_delete',
                  'acc_update_device', 'Set_Density_Grid_GPU_Release_Device_Caches']
    bodies = [function(source, n) for n in names]
    prototypes = '\n'.join(b[:b.index('{')].strip() + ';' for b in bodies)
    header = (ROOT / 'source/openmx_common.h').read_text()
    return pod(header, 'SetHamiltonianMETables') + '\n' + declarations(source) + '\n' + prototypes + '\n' + '\n'.join(bodies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='openmx-density-state-') as directory:
        folder = Path(directory)
        for backend in ('cuda', 'hip'):
            (folder / 'density_state.inc').write_text(production(backend))
            exe = folder / backend
            subprocess.run(compiler_command('c++') + [
                '-DOPENMX_GPU_HIP=' + str(int(backend == 'hip')),
                '-I' + str(folder), '-I' + str(ROOT / 'source'), '-I' + str(ROOT / 'core/include'),
                '-I' + str(ROOT / 'core/tests'),
                str(ROOT / 'tests/density_state_harness.cpp'), allocation_fault_source(folder),
                str(library), '-Wl,-rpath,' + str(library.parent), '-o', str(exe)], check=True)
            environment = dict(os.environ, **allocation_fault_environment(folder))
            subprocess.run([str(exe)], env=environment, check=True, timeout=60, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()
