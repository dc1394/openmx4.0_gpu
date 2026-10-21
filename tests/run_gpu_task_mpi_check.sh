#!/usr/bin/env bash
# Build on a login node, then run the fixture with one rank per physical node.
# --local uses one rank/two GPUs; --single-gpu-per-node uses two nodes/one GPU each.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
build=${1:?provide an output directory}; shift
mkdir -p "$build"
build=$(cd "$build" && pwd)
cuda=${CUDA_HOME:-/usr/local/cuda}
math=${CUDA_MATH_ROOT:-$cuda}
mpicc=${MPICC:-mpicc}
flags=(-O2 -std=c11 -acc -mp "-gpu=cc${GPU_ARCH:-80}" -Mnovect
       -I"$repo/source" -I"$cuda/include" -I"$math/include")
for name in openmx_gpu_workers openmx_gpu_task openmx_gpu_collective openmx_gpu_error; do
    "$mpicc" "${flags[@]}" -c "$repo/source/$name.c" -o "$build/$name.o"
done
"$mpicc" "${flags[@]}" "$repo/tests/gpu_task_mpi_check.c" \
    "$build/openmx_gpu_workers.o" "$build/openmx_gpu_task.o" \
    "$build/openmx_gpu_collective.o" "$build/openmx_gpu_error.o" \
    -Wl,--wrap=cudaMalloc -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" \
    -lcudart -lm -o "$build/gpu_task_mpi_check"
if [[ ${1:-} != --build-only ]]; then
    "${MPIEXEC:-mpirun}" -np "${MPI_RANKS:-2}" --map-by ppr:1:node \
        "$build/gpu_task_mpi_check" "$@"
fi
