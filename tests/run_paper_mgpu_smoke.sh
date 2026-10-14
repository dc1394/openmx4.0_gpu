#!/bin/sh
set -eu
# CUDA_HOME=/path/to/cuda CUDA_MATH_ROOT=/path/to/math_libs \
# NVCC_HOST=g++ HOST_LIBDIR=/path/to/lib64 GPU_ARCH=80 \
# LAPACK_LINK_FLAGS='-llapack -lblas' \
# sh tests/run_paper_mgpu_smoke.sh source/build/paper-tests --build-only
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
out=${1:?provide an output directory};shift
mkdir -p "$out"
out=$(CDPATH= cd -- "$out" && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda}
math_root=${CUDA_MATH_ROOT:-$cuda_root}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
host_cxx=${NVCC_HOST:-g++}
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
"$nvcc" -std=c++17 -O2 -arch="sm_${GPU_ARCH:-80}" -ccbin "$host_cxx" -Xcompiler -pthread -DOPENMX_PAPER_TESTING \
    -I"$root/source" -I"$math_root/include" \
    -c "$root/source/openmx_paper_mgpu.cu" -o "$out/openmx_paper_mgpu.o"
for kind in real complex; do
    suffix=
    if [ "$kind" = complex ]; then suffix=_complex; fi
    "$nvcc" -std=c++17 -O2 -ccbin "$host_cxx" -Xcompiler -pthread \
    -I"$root/source" "$root/tests/test_paper_mgpu$suffix.cpp" "$out/openmx_paper_mgpu.o" \
    -L"$math_root/lib64" -L"$cuda_root/lib64" -lcusolver -lcublas -lnvJitLink \
    -Xlinker -rpath -Xlinker "$math_root/lib64" -Xlinker -rpath -Xlinker "$cuda_root/lib64" \
    ${HOST_LIBDIR:+-L"$HOST_LIBDIR"} ${HOST_LIBDIR:+-Xlinker -rpath -Xlinker "$HOST_LIBDIR"} \
    ${LAPACK_LINK_FLAGS:--llapack -lblas} -o "$out/test_paper_mgpu$suffix"
done
if [ "${1:-}" = --build-only ]; then exit 0; fi
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
    "$out/test_paper_mgpu" "$@"
