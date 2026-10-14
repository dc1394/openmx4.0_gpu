#!/bin/sh
set -eu
# No MAGMA dependency is needed: this exercises runtime/GEMM groups only.
# Example: sh tests/run_mgpu_band_group_probe.sh source/build/band-group --build-only
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
out=${1:?provide an output directory};shift
mkdir -p "$out"
out=$(CDPATH= cd -- "$out" && pwd)
cuda=${CUDA_HOME:-/usr/local/cuda}
math=${CUDA_MATH_ROOT:-$cuda}
nvcc=${NVCC:-$cuda/bin/nvcc}
host=${NVCC_HOST:-g++}
arch=${GPU_ARCH:-80}
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
"$nvcc" -ccbin "$host" -std=c++17 -O2 -arch="sm_$arch" \
    -I"$root/source" -I"$math/include" \
    "$root/tests/mgpu_band_group_probe.cu" "$root/source/openmx_mgpu_runtime.cu" \
    "$root/source/openmx_mgpu_gemm.cu" "$root/source/openmx_magma_mgpu.cpp" \
    -L"$math/lib64" -L"$cuda/lib64" -lcublas -lcusolver -ldl \
    -Xlinker -rpath -Xlinker "$math/lib64" -Xlinker -rpath -Xlinker "$cuda/lib64" \
    ${HOST_LIBDIR:+-L"$HOST_LIBDIR"} ${HOST_LIBDIR:+-Xlinker -rpath -Xlinker "$HOST_LIBDIR"} \
    -o "$out/mgpu_band_group_probe"
if [ "${1:-}" = --build-only ]; then exit 0; fi
"$out/mgpu_band_group_probe" "$@"
