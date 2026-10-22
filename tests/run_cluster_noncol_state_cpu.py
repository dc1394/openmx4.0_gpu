#!/usr/bin/env python3
"""Check real noncollinear metadata/cleanup with explicit core owners.

Only SDK/MPI/BLACS operations are modeled. Production declarations, binding,
refinement controls, density demotion and deep teardown are compiled unchanged.
OpenACC data transfers are not executed; physical tests cover numerical data.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from run_cluster_col_state_cpu import function, pod, bounded_stack
from native_resource_test_bridge import calculation_state_declaration

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    common = ['ClusterNonCol_Reset', 'Cluster_DFT_NonCol_Bind_Resources',
        'ClusterNonCol_ReleaseCaches', 'ClusterNonCol_ResetGs2Fallback',
        'ClusterNonCol_DMEntryCache_Reset', 'ClusterNonCol_RootDMWorkspace_Reset',
        'ClusterNonCol_ReleaseGpuSolverCachedEVec', 'Cluster_DFT_NonCol_Release_GPU_Solver',
        'Cluster_DFT_NonCol_DemoteGpuSolverCachedEVec', 'ClusterNonCol_CheckedMulCount',
        'ClusterNonCol_AbortWithMessage', 'ClusterNonCol_RefineConfigure',
        'ClusterNonCol_RefineIterations', 'Cluster_DFT_NonCol_RefineLastRefined',
        'Cluster_DFT_NonCol_RefineAfterMixing']
    with tempfile.TemporaryDirectory(prefix='openmx-cluster-noncol-state-') as temporary:
        folder = Path(temporary)
        for backend in ['cuda', 'hip']:
            source = ROOT / ('source/Cluster_DFT_NonCol.c' if backend == 'cuda' else
                             'source/backends/hip/src/Cluster_DFT_NonCol.c')
            header = ROOT / ('source/eigen_refine_gpu.h' if backend == 'cuda' else
                             'source/backends/hip/include/eigen_refine_gpu.h')
            refine = ROOT / ('source/eigen_refine_gpu.c' if backend == 'cuda' else
                             'source/backends/hip/src/eigen_refine_gpu.cu')
            text = source.read_text()
            names = common + (['ClusterNonCol_DenseArena_Release', 'ClusterNonCol_GpuEigenCtx_Release',
                'ClusterNonCol_RefineRelease', 'ClusterNonCol_RootHostBuffers_Release',
                'ClusterNonCol_ArenaUnmap'] if backend == 'cuda' else [
                'ClusterNonCol_DeviceSolveRelease', 'ClusterNonCol_DeviceSolveDrop',
                'ClusterNonCol_DeviceFree', 'ClusterNonCol_HostStagingRelease',
                'ClusterNonCol_HostFree', 'ClusterNonCol_HostAlloc', 'ClusterNonCol_RefineStageReset',
                'ClusterNonCol_DeviceSolveGiveBack', 'ClusterNonCol_DeviceSolveParkBasis',
                'ClusterNonCol_ProfileEnabled',
                'ClusterNonCol_ProfileLap', 'ClusterNonCol_MpiRankForLog'])
            # Resolve the exact native helper names before compiling; a missing
            # helper is a fixture error, never substituted with a fake owner.
            bodies = [function(text, n) for n in names]
            declarations = '\n'.join(b[:b.index('{')].strip() + ';' for b in bodies)
            sdk_types = '\n'.join(pod(header.read_text(), n) for n in ['EigenRefineDevice', 'EigenRefineState'])
            state_release = '\n'.join(function(refine.read_text(), n) for n in ['refine_blocks_release', 'openmx_eigen_refine_state_release'])
            generated = sdk_types + '\nstatic void openmx_eigen_refine_device_release(EigenRefineDevice *dev) { (void)dev; refine_device_drains++; }\n'
            generated += calculation_state_declaration(text, 'ClusterNonColState') + '\n' + declarations + '\n' + state_release + '\n' + '\n'.join(bodies)
            (folder / 'cluster_noncol_state.inc').write_text(generated)
            exe = folder / backend
            command = shlex.split(os.environ.get('CXX', 'c++')) + [
                '-std=c++17', '-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                '-fno-pie', '-no-pie', '-Wno-unknown-pragmas', '-DOPENMX_GPU_HIP=' + str(int(backend == 'hip')),
                '-I' + str(folder), '-I' + str(ROOT / 'source'), '-I' + str(ROOT / 'core/include'),
                '-I' + str(ROOT / 'core/tests/fixtures/native_resources'), '-I' + str(ROOT / 'core/tests'),
                str(ROOT / 'tests/cluster_noncol_state_harness.cpp'), str(ROOT / 'core/tests/allocation_faults.cpp'),
                str(ROOT / 'source/openmx_env.c'), str(ROOT / 'source/openmx_precision.c'), str(library),
                '-Wl,-rpath,' + str(library.parent), '-o', str(exe)]
            subprocess.run(command, check=True)
            environment = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
            print('Backend:', backend, flush=True)
            subprocess.run([str(exe)], env=environment, check=True, timeout=120, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()
