#!/usr/bin/env python3
"""Exercise both solvers' real state binding/cleanup with the real core.

Only SDK operations and one-rank MPI/BLACS are modeled. Metadata definitions,
initializers, allocation callbacks, refinement controls and cleanup functions
are extracted unchanged from each backend. This is not a GPU numerical test.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command, allocation_fault_source

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    source = source.replace('extern "C" ', '')
    hit = re.search(r'^[ \t]*(?:[A-Za-z_]\w*[ \t*]+)+\b' + name + r'\([^;]*?\)\s*\{', source, re.M)
    if hit is None:
        raise ValueError(f"Missing production function {name}")
    start = source.index('{', hit.start())
    depth, end = 1, start + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[hit.start():end]


def pod(source, name):
    hit = re.search(r'typedef\s+struct\s*\{[^}]*\}\s*' + name + ';', source)
    if hit is None:
        raise ValueError(f"Missing production type {name}")
    return hit[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    common = ["Cluster_DFT_Col_GpuSwitchNum", "ClusterCol_ResetGs2Fallback", "ClusterCol_UnionGrid_Free",
              "ClusterCol_RefineRelease", "ClusterCol_GpuSolver_Destroy",
              "Cluster_DFT_Col_Release_GPU_Solver", "ClusterCol_ReleaseCaches",
              "ClusterCol_Reset", "Cluster_DFT_Col_Bind_Resources",
              "ClusterCol_AbortWithMessage", "ClusterCol_CheckedMulCount",
              "ClusterCol_MallocArray", "ClusterCol_TryDeviceMalloc",
              "ClusterCol_RefineDeviceMalloc", "ClusterCol_RefineHostAllocate",
              "ClusterCol_RefineConfigure", "ClusterCol_RefineIterations",
              "ClusterCol_RefineBeginSolve", "ClusterCol_RefineEndSolve",
              "Cluster_DFT_Col_RefineLastRefined", "Cluster_DFT_Col_RefineForceFP64",
              "Cluster_DFT_Col_RefineAfterMixing"]
    with tempfile.TemporaryDirectory(prefix='openmx-cluster-col-state-') as directory:
        folder = Path(directory)
        for backend in ['cuda', 'hip']:
            solver = ROOT / ('source/Cluster_DFT_Col.c' if backend == 'cuda' else
                             'source/backends/hip/src/Cluster_DFT_Col.c')
            header = ROOT / ('source/eigen_refine_gpu.h' if backend == 'cuda' else
                             'source/backends/hip/include/eigen_refine_gpu.h')
            implementation = ROOT / ('source/eigen_refine_gpu.c' if backend == 'cuda' else
                                     'source/backends/hip/src/eigen_refine_gpu.cu')
            text, refine = solver.read_text(), implementation.read_text()
            start = text.index('typedef struct', text.index('/* Persistent metadata'))
            end = text.index('typedef OpenMXClusterColState ClusterColState;')
            end += len('typedef OpenMXClusterColState ClusterColState;')
            fields = pod(header.read_text(), 'EigenRefineDevice') + '\n' + pod(header.read_text(), 'EigenRefineState')
            allocation = '\n'.join(function(header.read_text(), n) for n in
                ['openmx_eigen_refine_malloc', 'openmx_eigen_refine_host_malloc'])
            extra = [] if backend == 'cuda' else ["ClusterCol_SerialOverlapReset",
                "ClusterCol_DMEntryCache_Reset", "ClusterCol_RootDMWorkspace_Reset",
                "ClusterCol_ReleaseGpuSolverCachedSpinEVec", "ClusterCol_ReleaseGpuSolverCachedEVec",
                "ClusterCol_RefineDropParkedBases", "ClusterCol_RefineResetStage", "ClusterCol_RefineHostMalloc"]
            bodies = [function(text, n) for n in common + extra]
            prototypes = '\n'.join(body[:body.index('{')].rstrip() + ';' for body in bodies)
            release = '\n'.join(function(refine, n) for n in ['refine_blocks_release', 'openmx_eigen_refine_state_release'])
            generated = fields + '\nstatic void openmx_eigen_refine_device_release(EigenRefineDevice *dev) { (void)dev; refine_device_drains++; }\n' + allocation + '\n' + text[start:end] + '\n' + prototypes + '\n' + release + '\n' + '\n'.join(bodies)
            (folder / 'cluster_col_state.inc').write_text(generated)
            exe = folder / backend
            command = compiler_command('c++') + [
                '-Wno-unknown-pragmas', '-DOPENMX_GPU_HIP=' + str(int(backend == 'hip')),
                '-I' + str(folder), '-I' + str(ROOT / 'source'), '-I' + str(ROOT / 'core/include'),
                '-I' + str(ROOT / 'core/tests/fixtures/native_resources'), '-I' + str(ROOT / 'core/tests'),
                str(ROOT / 'tests/cluster_col_state_harness.cpp'), allocation_fault_source(folder),
                str(ROOT / 'source/openmx_env.c'), str(ROOT / 'source/openmx_precision.c'), str(library),
                '-Wl,-rpath,' + str(library.parent), '-o', str(exe)]
            subprocess.run(command, check=True)
            environment = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
            print('Backend:', backend, flush=True)
            subprocess.run([str(exe)], env=environment, check=True, timeout=120, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()
