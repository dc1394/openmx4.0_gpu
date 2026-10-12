#!/bin/sh
set -eu
# Validate the production CUDA matrix/density consumers of compact orbital
# tiles. SETPOINT_SANITIZE=1 checks every full and partial tile with memcheck.
test_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda-13.4}
sdk_root=${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.9}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
host_cxx=${NVCC_HOST:-$sdk_root/compilers/bin/nvc++}
gpu_arch=${GPU_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '/^[0-9]+[.][0-9]+$/ {gsub(/[.]/, ""); print; exit}')}
test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/openmx-point-chunks.XXXXXX")
trap 'rm -rf "$test_tmp"' EXIT HUP INT TERM
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
"$nvcc" -ccbin "$host_cxx" -std=c++17 -O3 -lineinfo -arch="sm_$gpu_arch" \
    --cudart shared "$test_root/tests/setham_point_chunks_gpu.cu" \
    "$test_root/source/set_hamiltonian_gpu.cu" -L"$cuda_root/lib64" \
    -Xlinker -rpath -Xlinker "$cuda_root/lib64" -o "$test_tmp/check"
if [ "${SETPOINT_SANITIZE:-0}" != 0 ]; then
    "$cuda_root/bin/compute-sanitizer" --tool memcheck --error-exitcode 99 "$test_tmp/check"
else
    "$test_tmp/check"
fi
