#!/bin/bash
# Build bench_gemm against the campaign's libgemmul8.a with the same nvcc
# flags as source/Makefile (NVCC_GEMMUL8_FLAGS).
#   dev box (RTX 5080):  ./build.sh
#   Pegasus (H100):      NVROOT=<nvhpc 26.5 root> GPU_ARCH=90 \
#                        G8_DIR=<tree>/source/third_party/GEMMul8 ./build.sh
set -eu
cd "$(dirname "$0")"
NVROOT=${NVROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5}
G8_DIR=${G8_DIR:-$HOME/openmx4.0_thesis/source/third_party/GEMMul8}
GPU_ARCH=${GPU_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')}
[ -n "$GPU_ARCH" ] || { echo "GPU_ARCH not set and no GPU visible" >&2; exit 1; }
[ -f "$G8_DIR/lib/libgemmul8.a" ] || { echo "missing $G8_DIR/lib/libgemmul8.a" >&2; exit 1; }

MATHLIB="$NVROOT/math_libs/13.2/targets/x86_64-linux/lib"
"$NVROOT/cuda/13.2/bin/nvcc" -ccbin "$NVROOT/compilers/bin/nvc++" \
    -std=c++20 -O3 -diag-suppress 177 -DGPU_ARCH="$GPU_ARCH" \
    -I"$G8_DIR/include" -I"$G8_DIR/src" \
    -gencode arch=compute_"$GPU_ARCH",code=sm_"$GPU_ARCH" \
    bench_gemm.cu "$G8_DIR/lib/libgemmul8.a" \
    -L"$MATHLIB" -Xlinker -rpath -Xlinker "$MATHLIB" -lcublas -lcublasLt -o bench_gemm
echo "built bench_gemm (arch sm_$GPU_ARCH, lib $G8_DIR/lib/libgemmul8.a)"
