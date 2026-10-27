#!/bin/sh
set -eu
# Build the production compatibility layer, runtime and the MAGMA backend.
# Build on the login node with --build-only; run on allocated GPUs afterward.
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
out=${1:?provide output directory};shift
mkdir -p "$out"
out=$(CDPATH= cd -- "$out" && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda}
math_root=${CUDA_MATH_ROOT:-$cuda_root}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
host_cxx=${NVCC_HOST:-g++}
host_cc=${CC:-gcc}
magma_prefix=${MAGMA_PREFIX:-$root/source/third_party/magma-install}
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
"$host_cc" -std=c11 -O2 -I"$root/source" \
    -c "$root/source/openmx_env.c" -o "$out/env.o"
"$host_cc" -std=c11 -O2 -I"$root/source" -I"$cuda_root/include" -I"$math_root/include" \
    -c "$root/source/openmx_cusolver_compat.c" -o "$out/compat.o"
"$host_cc" -std=c11 -O2 -I"$root/source" \
    -c "$root/source/openmx_dense_cpu.c" -o "$out/dense_cpu.o"
magma_compile=
magma_link=
if [ -f "$magma_prefix/lib/libmagma.so" ]; then
    magma_compile="-DOPENMX_HAVE_MAGMA -I$magma_prefix/include"
    magma_link="-L$magma_prefix/lib -Xlinker -rpath -Xlinker $magma_prefix/lib -lmagma"
fi
"$nvcc" -std=c++17 -O2 -arch="sm_${GPU_ARCH:-80}" -ccbin "$host_cxx" -Xcompiler -pthread \
    -DOPENMX_TEST_WRAP_MGPU -Xlinker --wrap=openmx_mgpu_solve_device_try \
    -I"$root/source" -I"$math_root/include" $magma_compile \
    "$root/tests/mgpu_solver_smoke.cpp" "$root/source/openmx_mgpu_runtime.cu" \
    "$root/source/openmx_mgpu_gemm.cu" "$root/source/openmx_magma_mgpu.cpp" \
    "$out/compat.o" "$out/dense_cpu.o" "$out/env.o" \
    $magma_link -L"$math_root/lib64" -L"$cuda_root/lib64" -lcusolver -lcublas -lnvJitLink -ldl \
    -Xlinker -rpath -Xlinker "$math_root/lib64" -Xlinker -rpath -Xlinker "$cuda_root/lib64" \
    ${HOST_LIBDIR:+-L"$HOST_LIBDIR"} ${HOST_LIBDIR:+-Xlinker -rpath -Xlinker "$HOST_LIBDIR"} \
    ${LAPACK_LINK_FLAGS:--llapack -lblas} -o "$out/mgpu_solver_smoke"
# A separate hardware probe expects direct NVLink peers; run it explicitly
# only on such an allocation. It checks the automatic GEMM gate and scope.
"$nvcc" -std=c++17 -O2 -arch="sm_${GPU_ARCH:-80}" -ccbin "$host_cxx" -Xcompiler -pthread \
    -I"$root/source" -I"$math_root/include" $magma_compile \
    "$root/tests/mgpu_dispatch_probe.cu" "$root/source/openmx_mgpu_runtime.cu" \
    "$root/source/openmx_mgpu_gemm.cu" "$root/source/openmx_magma_mgpu.cpp" \
    "$out/dense_cpu.o" "$out/env.o" \
    $magma_link -L"$math_root/lib64" -L"$cuda_root/lib64" -lcusolver -lcublas -lnvJitLink -ldl \
    -Xlinker -rpath -Xlinker "$math_root/lib64" -Xlinker -rpath -Xlinker "$cuda_root/lib64" \
    ${HOST_LIBDIR:+-L"$HOST_LIBDIR"} ${HOST_LIBDIR:+-Xlinker -rpath -Xlinker "$HOST_LIBDIR"} \
    ${LAPACK_LINK_FLAGS:--llapack -lblas} \
    -o "$out/mgpu_dispatch_probe"
if [ "${1:-}" = --build-only ]; then exit 0; fi
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
    "$out/mgpu_solver_smoke" "$@"
