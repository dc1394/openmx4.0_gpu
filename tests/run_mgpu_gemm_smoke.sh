#!/bin/sh
set -eu
# Build only on a login node, then run the binary inside an allocated GPU job:
# GPU_ARCH=80 sh tests/run_mgpu_gemm_smoke.sh source/build/mgpu-tests --build-only
test_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
bin_dir=${1:?provide an output directory};shift
mkdir -p "$bin_dir"
bin_dir=$(CDPATH= cd -- "$bin_dir" && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda}
math_root=${CUDA_MATH_ROOT:-$cuda_root}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
host_cxx=${NVCC_HOST:-g++}
host_cc=${CC:-cc}
gpu_arch=${GPU_ARCH:-80}
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
"$host_cc" -std=c11 -O2 -I"$test_root/source" \
    -c "$test_root/source/openmx_env.c" -o "$bin_dir/env.o"
"$nvcc" -ccbin "$host_cxx" -std=c++17 -O2 -arch="sm_$gpu_arch" -Xcompiler -pthread,-fopenmp -DOPENMX_MGPU_TEST_BRIDGE \
    -I"$test_root/source" -I"$math_root/include" \
    "$test_root/tests/mgpu_gemm_smoke.cu" "$test_root/source/openmx_mgpu_gemm.cu" "$bin_dir/env.o" \
    -L"$math_root/lib64" -L"$cuda_root/lib64" -lcublas \
    -Xlinker -rpath -Xlinker "$math_root/lib64" -Xlinker -rpath -Xlinker "$cuda_root/lib64" \
    ${HOST_LIBDIR:+-L"$HOST_LIBDIR"} ${HOST_LIBDIR:+-Xlinker -rpath -Xlinker "$HOST_LIBDIR"} \
    -o "$bin_dir/mgpu_gemm_smoke"
"$nvcc" -ccbin "$host_cxx" -std=c++17 -O2 -arch="sm_$gpu_arch" -Xcompiler -pthread,-fopenmp \
    -I"$test_root/source" -I"$math_root/include" \
    "$test_root/tests/mgpu_gemm_bench.cu" "$test_root/source/openmx_mgpu_gemm.cu" "$bin_dir/env.o" \
    -L"$math_root/lib64" -L"$cuda_root/lib64" -lcublas \
    -Xlinker -rpath -Xlinker "$math_root/lib64" -Xlinker -rpath -Xlinker "$cuda_root/lib64" \
    ${HOST_LIBDIR:+-L"$HOST_LIBDIR"} ${HOST_LIBDIR:+-Xlinker -rpath -Xlinker "$HOST_LIBDIR"} \
    -o "$bin_dir/mgpu_gemm_bench"
if [ "${1:-}" = --build-only ]; then exit 0; fi
"$bin_dir/mgpu_gemm_smoke" "$@"
