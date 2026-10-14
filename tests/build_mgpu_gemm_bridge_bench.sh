#!/bin/sh
set -eu
# Compare the retained single-GPU bridge (including A100's accurate Ozaki
# fixed-forward policy) with the multi-GPU bridge on the same operands.
test_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
bin_dir=${1:?provide an output directory}
mkdir -p "$bin_dir"
bin_dir=$(CDPATH= cd -- "$bin_dir" && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda}
math_root=${CUDA_MATH_ROOT:-$cuda_root}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
host_cxx=${NVCC_HOST:-g++}
gpu_arch=${GPU_ARCH:-80}
gemmul8_root=${GEMMUL8_DIR:-$test_root/source/third_party/GEMMul8}
gemmul8_object=${GEMMUL8_TEST_OBJECT:?reuse a GEMMul8 template object built with the same toolkit/architecture}
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
"$nvcc" -ccbin "$host_cxx" -std=c++17 -O2 -arch="sm_$gpu_arch" -DOPENMX_MGPU_BENCH_BRIDGE \
    -I"$test_root/source" -I"$gemmul8_root/include" -I"$math_root/include" \
    "$test_root/tests/mgpu_gemm_bench.cu" "$test_root/source/openmx_mgpu_gemm.cu" \
    "$test_root/source/gemmul8_bridge.cu" "$gemmul8_object" \
    -L"$math_root/lib64" -L"$cuda_root/lib64" -lcublas -lcublasLt \
    -Xlinker -rpath -Xlinker "$math_root/lib64" -Xlinker -rpath -Xlinker "$cuda_root/lib64" \
    ${HOST_LIBDIR:+-L"$HOST_LIBDIR"} ${HOST_LIBDIR:+-Xlinker -rpath -Xlinker "$HOST_LIBDIR"} \
    -o "$bin_dir/mgpu_gemm_bench_bridge"
