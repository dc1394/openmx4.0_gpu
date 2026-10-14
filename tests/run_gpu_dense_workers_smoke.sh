#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
build=${OPENMX_GPU_TEST_BUILD:-$(mktemp -d)}
mkdir -p "$build"
cuda=${CUDA_HOME:-/usr/local/cuda}
math=${CUDA_MATH_ROOT:-$cuda}
nvc=${NVHPC_CC:-nvc}
nvcxx=${NVHPC_CXX:-nvc++}
hostcxx=${CUDAHOSTCXX:-g++}
arch=${OPENMX_GPU_TEST_ARCH:-80}
includes=(-I"$repo/source" -I"$cuda/include" -I"$math/include")
if [[ -n ${MPI_INCLUDE:-} ]]; then
  includes+=(-I"$MPI_INCLUDE")
else
  for directory in $(mpicc --showme:incdirs); do includes+=(-I"$directory"); done
fi
for name in openmx_gpu_workers openmx_gpu_views; do
  "$nvc" -O2 -std=c11 -acc -mp "-gpu=cc$arch" "${includes[@]}" -c "$repo/source/$name.c" -o "$build/$name.o"
done
"$nvc" -O2 -std=c11 -acc -mp "-gpu=cc$arch" "${includes[@]}" -c "$repo/tests/gpu_workers_test_hooks.c" -o "$build/hooks.o"
for name in openmx_gpu_transform openmx_gpu_dm; do
  "$cuda/bin/nvcc" -O3 -std=c++17 "-arch=sm_$arch" -ccbin "$hostcxx" "${includes[@]}" -c "$repo/source/$name.cu" -o "$build/$name.o"
done
# The fixture is host C++ with CUDA API calls and a long-double CPU oracle.
"$nvcxx" -O2 -std=c++17 -acc -mp "-gpu=cc$arch" "${includes[@]}" -x c++ -c "$repo/tests/gpu_dense_workers_smoke.cu" -o "$build/fixture.o"
"$nvcxx" -acc -mp "-gpu=cc$arch" "$build/fixture.o" "$build/hooks.o" \
  "$build/openmx_gpu_workers.o" "$build/openmx_gpu_views.o" "$build/openmx_gpu_transform.o" "$build/openmx_gpu_dm.o" \
  -L"$cuda/lib64" "-Wl,-rpath,$cuda/lib64" -lcudart -lpthread -o "$build/gpu_dense_workers_smoke"
if [[ ${OPENMX_GPU_TEST_COMPILE_ONLY:-0} != 1 ]]; then
  "$build/gpu_dense_workers_smoke"
fi
