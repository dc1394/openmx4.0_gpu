#!/bin/sh
set -eu

# Build and run the checks of the bridge's forward-transform entry points
# (tests/gemmul8_forward_smoke.cu).  Toolkit selection and GEMMUL8_TEST_OBJECT
# as in tests/run_gemmul8_smoke.sh.
test_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda-13.4}
math_root=${CUDA_MATH_ROOT:-$cuda_root}
sdk_root=${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.9}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
host_cxx=${NVCC_HOST:-$sdk_root/compilers/bin/nvc++}
gemmul8_root=${GEMMUL8_DIR:-$test_root/source/third_party/GEMMul8}
gpu_arch=${GPU_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '/^[0-9]+[.][0-9]+$/ {gsub(/[.]/, ""); print; exit}')}
test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/openmx-gemmul8-test.XXXXXX")
trap 'rm -rf "$test_tmp"' EXIT HUP INT TERM
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS

gemmul8_object=${GEMMUL8_TEST_OBJECT:-$test_tmp/gemmul8.o}
if [ -z "${GEMMUL8_TEST_OBJECT:-}" ]; then
    "$nvcc" -ccbin "$host_cxx" -std=c++20 -O3 -diag-suppress 177 \
        -DGPU_ARCH="$gpu_arch" -arch="sm_$gpu_arch" \
        -I"$gemmul8_root/include" -I"$gemmul8_root/src" -I"$math_root/include" \
        -c "$test_root/source/gemmul8_openmx.cu" -o "$gemmul8_object"
fi
"$nvcc" -ccbin "$host_cxx" -std=c++20 -O2 -I"$gemmul8_root/include" -I"$math_root/include" \
    "$test_root/tests/gemmul8_forward_smoke.cu" "$test_root/source/gemmul8_bridge.cu" \
    "$gemmul8_object" -L"$math_root/lib64" -L"$cuda_root/lib64" -lcublas -lcublasLt \
    -Xlinker -rpath -Xlinker "$math_root/lib64" \
    -Xlinker -rpath -Xlinker "$cuda_root/lib64" -o "$test_tmp/check"

OPENMX_GEMMUL8_DISABLE=0 OPENMX_GEMMUL8_DISABLE_D=0 OPENMX_GEMMUL8_DISABLE_Z=0 \
OPENMX_GEMMUL8_NUM_MOD_D=15 OPENMX_GEMMUL8_NUM_MOD_Z=15 \
OPENMX_GEMMUL8_MAX_WORKSPACE_MB=256 OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT=0 \
OPENMX_GEMMUL8_LOCAL_RANKS=1 "$test_tmp/check"
