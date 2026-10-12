#!/bin/sh
set -eu

# Build and optionally run the GEMMul8/native-FP64 comparison.
# Usage: run_gemmul8_a100_bench.sh <output-directory> [--build-only | benchmark options]
# Set GEMMUL8_TEST_OBJECT to reuse the build's template object; toolchain
# overrides match run_gemmul8_smoke.sh. GPU_ARCH=80 works on GPU-less logins.
# --path fixed times H X and X^T/H(H X) together through the production
# fixed-operand API; --path unblocked times one full-workspace product.
# --types D (or real) bounds memory for --sizes 7332x7332x7332,13000x13000x13000.
test_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
bin_dir=${1:?provide an output directory}
shift
mkdir -p "$bin_dir"
bin_dir=$(CDPATH= cd -- "$bin_dir" && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda-13.4}
math_root=${CUDA_MATH_ROOT:-$cuda_root}
sdk_root=${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.9}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
host_cxx=${NVCC_HOST:-$sdk_root/compilers/bin/nvc++}
gemmul8_root=${GEMMUL8_DIR:-$test_root/source/third_party/GEMMul8}
gpu_arch=${GPU_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '/^[0-9]+[.][0-9]+$/ {gsub(/[.]/, ""); print; exit}')}
gemmul8_object=${GEMMUL8_TEST_OBJECT:-$bin_dir/gemmul8.o}
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
if [ -z "${GEMMUL8_TEST_OBJECT:-}" ]; then
    "$nvcc" -ccbin "$host_cxx" -std=c++20 -O3 -diag-suppress 177 \
        -DGPU_ARCH="$gpu_arch" -arch="sm_$gpu_arch" \
        -I"$gemmul8_root/include" -I"$gemmul8_root/src" -I"$math_root/include" \
        -c "$test_root/source/gemmul8_openmx.cu" -o "$gemmul8_object"
fi
"$nvcc" -ccbin "$host_cxx" -std=c++20 -O3 -arch="sm_$gpu_arch" \
    -I"$gemmul8_root/include" -I"$math_root/include" \
    "$test_root/tests/gemmul8_a100_bench.cu" "$test_root/source/gemmul8_bridge.cu" \
    "$gemmul8_object" -L"$math_root/lib64" -L"$math_root/lib" -L"$cuda_root/lib64" \
    -lcublas -lcublasLt \
    -Xlinker -rpath -Xlinker "$math_root/lib64" -Xlinker -rpath -Xlinker "$math_root/lib" \
    -Xlinker -rpath -Xlinker "$cuda_root/lib64" \
    ${HOST_LIBDIR:+-L"$HOST_LIBDIR"} ${HOST_LIBDIR:+-Xlinker -rpath -Xlinker "$HOST_LIBDIR"} \
    -o "$bin_dir/gemmul8_a100_bench"
if [ "${1:-}" = --build-only ]; then exit 0; fi
unset GEMMUL8_DISABLE OPENMX_GEMMUL8_DISABLE GEMMUL8_DISABLE_D OPENMX_GEMMUL8_DISABLE_D
unset GEMMUL8_DISABLE_Z OPENMX_GEMMUL8_DISABLE_Z
OPENMX_GEMMUL8_TIMING=0 OPENMX_GEMMUL8_NUM_MOD_D=15 OPENMX_GEMMUL8_NUM_MOD_Z=15 \
OPENMX_GEMMUL8_FASTMODE_D=0 OPENMX_GEMMUL8_FASTMODE_Z=0 \
OPENMX_GEMMUL8_MAX_WORKSPACE_MB=256 OPENMX_GEMMUL8_LOCAL_RANKS=1 \
    "$bin_dir/gemmul8_a100_bench" "$@"
