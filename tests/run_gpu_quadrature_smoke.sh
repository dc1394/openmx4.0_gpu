#!/usr/bin/env bash
# Shared builder for the production quadrature and its device workers.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
fixture=${1:?quadrature fixture basename required}
case "$fixture" in setham_point_chunks_gpu|set_hamiltonian_stream_smoke) ;; *) exit 2;; esac
build=${OPENMX_GPU_TEST_BUILD:-$(mktemp -d)}
if [[ -z ${OPENMX_GPU_TEST_BUILD:-} ]]; then trap 'rm -rf "$build"' EXIT; fi
mkdir -p "$build"
cuda=${CUDA_HOME:-/usr/local/cuda}
math=${CUDA_MATH_ROOT:-$cuda}
nvc=${NVHPC_CC:-${NVHPC_ROOT:+$NVHPC_ROOT/compilers/bin/}nvc}
nvcxx=${NVHPC_CXX:-${NVHPC_ROOT:+$NVHPC_ROOT/compilers/bin/}nvc++}
hostcxx=${NVCC_HOST:-${CUDAHOSTCXX:-g++}}
arch=${GPU_ARCH:-${OPENMX_GPU_TEST_ARCH:-80}}
includes=(-I"$repo/source" -I"$cuda/include" -I"$math/include")
linkdirs=(-L"$cuda/lib64" "-Wl,-rpath,$cuda/lib64")
if [[ -n ${HOST_LIBDIR:-} ]]; then
  linkdirs+=(-L"$HOST_LIBDIR" "-Wl,-rpath,$HOST_LIBDIR")
fi
if [[ -n ${MPI_INCLUDE:-} ]]; then
  includes+=(-I"$MPI_INCLUDE")
else
  for directory in $(mpicc --showme:incdirs); do includes+=(-I"$directory"); done
fi
for name in openmx_gpu_workers; do
  "$nvc" -O2 -std=c11 -acc -mp "-gpu=cc$arch" "${includes[@]}" -c "$repo/source/$name.c" -o "$build/$name.o"
done
"$hostcxx" -O2 -std=c++17 -fopenmp "${includes[@]}" -c "$repo/source/openmx_gpu_views.cpp" -o "$build/openmx_gpu_views.o"
"$nvc" -O2 -std=c11 -acc -mp "-gpu=cc$arch" "${includes[@]}" -c "$repo/tests/gpu_workers_test_hooks.c" -o "$build/hooks.o"
"$nvc" -O2 -std=c11 "${includes[@]}" -c "$repo/source/openmx_env.c" -o "$build/env.o"
"${NVCC:-$cuda/bin/nvcc}" -O3 -lineinfo -std=c++17 "-arch=sm_$arch" -ccbin "$hostcxx" \
  "${includes[@]}" -c "$repo/source/set_hamiltonian_gpu.cu" -o "$build/kernels.o"
"$nvcxx" -O2 -std=c++17 -acc -mp "-gpu=cc$arch" "${includes[@]}" -x c++ -c "$repo/tests/$fixture.cu" -o "$build/fixture.o"
"$nvcxx" -acc -mp "-gpu=cc$arch" "$build/fixture.o" "$build/hooks.o" "$build/env.o" \
  "$build/openmx_gpu_workers.o" "$build/openmx_gpu_views.o" "$build/kernels.o" \
  "${linkdirs[@]}" -lcudart -lpthread -o "$build/$fixture"
if [[ ${OPENMX_GPU_TEST_COMPILE_ONLY:-0} != 1 ]]; then
  if [[ ${SETPOINT_SANITIZE:-${SETHAMILTONIAN_SANITIZE:-0}} != 0 ]]; then
    sanitizer=${COMPUTE_SANITIZER:-$cuda/bin/compute-sanitizer}
    if [[ ! -x "$sanitizer" && -z ${COMPUTE_SANITIZER:-} ]]; then
      sanitizer=$cuda/compute-sanitizer/compute-sanitizer
    fi
    "$sanitizer" --tool memcheck --error-exitcode 99 "$build/$fixture"
  else
    "$build/$fixture"
  fi
fi
