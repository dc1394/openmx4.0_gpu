#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
prefix=${MAGMA_PREFIX:-$root/source/third_party/magma-install}
cuda_root=${CUDA_HOME:?set CUDA_HOME to the chosen CUDA toolkit}
math_root=${CUDA_MATH_ROOT:-$cuda_root}
host_cxx=${CXX:-c++}
out=${MAGMA_TEST_DIR:-$root/source/build/magma-tests}
mkdir -p "$out"
"$host_cxx" -std=c++14 -O2 -Wall -Wextra -DOPENMX_HAVE_MAGMA \
    -I"$root/source" -I"$prefix/include" -I"$cuda_root/include" -I"$math_root/include" \
    "$root/tests/magma_mgpu_smoke.cpp" "$root/source/openmx_magma_mgpu.cpp" \
    -L"$prefix/lib" -L"$cuda_root/lib64" -L"$math_root/lib64" ${HOST_LIBDIR:+-L"$HOST_LIBDIR"} \
    -lmagma -lcudart -llapack -lblas -lpthread \
    -Wl,--disable-new-dtags,-rpath,"$prefix/lib",-rpath,"$cuda_root/lib64",-rpath,"$math_root/lib64" \
    ${HOST_LIBDIR:+-Wl,-rpath,"$HOST_LIBDIR"} -o "$out/magma_mgpu_smoke"
if [ "${1:-}" = --build-only ]; then exit 0; fi
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MAGMA_NUM_THREADS=${MAGMA_NUM_THREADS:-1} \
    "$out/magma_mgpu_smoke" "$@"
