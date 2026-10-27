#!/usr/bin/env python3
"""Exercise retained Hamiltonian storage with the real calculation owner.

The production builders/releases run against independently tracked host/SDK
allocations. Numerical OTF layouts have a separate oracle in the plan tests;
this fixture checks calculation/device isolation, partial residency, failure
rollback, non-building probes and destruction without input arrays or MPI.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile
from run_cluster_col_state_cpu import function
from native_state_test_support import (compiler_command, bounded_stack,
    allocation_fault_source, allocation_fault_environment)

ROOT = Path(__file__).resolve().parents[1]


def pod(source, name):
    end = source.index('} ' + name + ';') + len('} ' + name + ';')
    return source[source.rfind('typedef struct', 0, end):end]


def declarations(source, backend):
    names = ['SetHamiltonianProfileCounters', 'SetHOTFBatch', 'SetHOTFPlan']
    names += (['SetHamiltonianMatrixElementsCache', 'SetHamiltonianOnTheFlyCache',
        'SetHOTFDensityPlan', 'SetHamiltonianGpuTurnPlan', 'SetHamiltonianDirectWriter',
        'SetHGpuMapping', 'SetHOTFPairCost'] if backend == 'cuda' else
        ['SetHamiltonianMETableCache', 'SetHamOTFState', 'SetHOTFStatic'])
    # Plans refer to batches, and the state follows all of its field types.
    text = '\n'.join(pod(source, name) for name in names)
    start = source.index('struct OpenMXHamiltonianGpuState {')
    end = source.index('\n};', start) + 3
    text += '\n' + source[start:end]
    if backend == 'cuda':
        match = re.search(r'enum \{\s*SETH_RES_ORBS1[\s\S]*?\};', source)
        text += '\n' + match[0]
    else:
        text += '\n' + re.search(r'^#define SETH_OTF_ALIGN[^\n]*', source, re.M)[0]
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--backend', choices=['cuda', 'hip', 'both'], default='both')
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    common = ['Set_Hamiltonian_MatrixElementsTables_Ready', 'Set_Hamiltonian_GetMatrixElementsTables',
        'Set_Hamiltonian_OTF_Release', 'Set_Hamiltonian_OTF_Ensure', 'SetH_Reset',
        'Set_Hamiltonian_GPU_Bind_Resources']
    cuda = ['Set_Hamiltonian_checked_add', 'Set_Hamiltonian_MatrixElements_CountTotals', 'Set_Hamiltonian_ME_CacheMatches', 'Set_Hamiltonian_ME_RepackOrbTables',
        'SetH_Unmap', 'SetH_UnmapBatch', 'SetH_MapBatch', 'Set_Hamiltonian_ResidentMaskLabel',
        'Set_Hamiltonian_ME_DiscardHostPages', 'Set_Hamiltonian_ME_EnterDeviceCache',
        'Set_Hamiltonian_ME_ReleaseDeviceCache', 'Set_Hamiltonian_Yield_Resident_DeviceCache',
        'Set_Hamiltonian_Free_OpenACC_MatrixElements_Cache',
        'Set_Hamiltonian_Invalidate_OpenACC_MatrixElements_Cache',
        'Set_Hamiltonian_Ensure_OpenACC_MatrixElements_Cache', 'Set_Hamiltonian_DW_Push',
        'Set_Hamiltonian_DW_Flush', 'SETH_OTF_StagePotential', 'SETH_OTF_FreePlan',
        'SETH_OTF_FreePlans', 'SETH_OTF_PairCost', 'SETH_OTF_CompactCost', 'SETH_OTF_GetPlan']
    hip = ['Set_Hamiltonian_ME_Associate', 'Set_Hamiltonian_ME_Disassociate',
        'Set_Hamiltonian_Invalidate_OpenMP_MatrixElements_Cache',
        'Set_Hamiltonian_OTF_ReleaseDevice', 'Set_Hamiltonian_OTF_BatchSizes',
        'SetH_OTF_HostAlloc', 'SetH_OTF_HostFree', 'SetH_OTF_FreePlan',
        'SetH_OTF_ReleasePlans', 'SetH_OTF_EnsureStatic', 'SetH_OTF_KindBatchBytes', 'SetH_OTF_GetPlan']
    for backend in ['cuda', 'hip'] if args.backend == 'both' else [args.backend]:
        path = ROOT / ('source/Set_Hamiltonian.c' if backend == 'cuda' else
            'source/backends/hip/src/Set_Hamiltonian.c')
        source = path.read_text()
        bodies = [function(source, name) for name in common + (cuda if backend == 'cuda' else hip)]
        prototypes = '\n'.join(body[:body.index('{')].rstrip() + ';' for body in bodies)
        generated = declarations(source, backend) + '\n' + prototypes + '\n' + '''
#if !OPENMX_GPU_HIP
#define acc_malloc model_acc_malloc
#endif
#define malloc host_malloc
#define calloc host_calloc
#define free host_free
#define abort() binding_abort()
''' + '\n'.join(bodies) + '''
#if !OPENMX_GPU_HIP
#undef acc_malloc
#endif
#undef malloc
#undef calloc
#undef free
#undef abort
'''
        with tempfile.TemporaryDirectory(prefix='openmx-hamiltonian-state-') as directory:
            folder = Path(directory)
            (folder / 'hamiltonian_state.inc').write_text(generated)
            (folder / 'hamiltonian_tables.inc').write_text(pod((ROOT / 'source/openmx_common.h').read_text(), 'SetHamiltonianMETables'))
            # HIP cleanup uses OpenMP's default-device binding; no offload is
            # performed by this SDK model.
            (folder / 'omp.h').write_text('')
            exe = folder / backend
            command = compiler_command('c++') + ['-Wno-unknown-pragmas',
                '-DOPENMX_GPU_HIP=' + str(int(backend == 'hip')),
                '-I' + str(folder), '-I' + str(ROOT / 'source'), '-I' + str(ROOT / 'core/include'),
                '-I' + str(ROOT / 'core/tests/fixtures/native_resources'),
                '-I' + str(ROOT / 'core/tests'), str(ROOT / 'tests/hamiltonian_gpu_state_harness.cpp'),
                allocation_fault_source(folder), str(ROOT / 'source/openmx_env.c'),
                str(library), '-Wl,-rpath,' + str(library.parent), '-o', str(exe)]
            subprocess.run(command, check=True)
            environment = {k: v for k, v in os.environ.items()
                if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
            environment.update(allocation_fault_environment(folder))
            subprocess.run([str(exe)], check=True, env=environment, timeout=120, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()
