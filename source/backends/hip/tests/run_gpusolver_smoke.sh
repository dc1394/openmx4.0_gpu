#!/bin/sh
set -eu

# GPU checks of the dense eigensolver layer (one process, < 2 GiB, about a
# minute on an RX 9060 XT): tests/hipsolver_smoke.c (the hipsolver_compat.h
# shim) and tests/gpusolver_wrapper_smoke.c (source/gpusolver_Syevdx.c and the
# entry points of source/magma_openmx.cu), built here from the current
# sources against the bundled MAGMA and GEMMul8 libraries (built by the
# OpenMX build) and the system reference LAPACK; nothing in source/ is
# written.  The wrapper check runs under the solver-policy variables:
#   default                                   architecture policy, retry on failure
#   OPENMX_GPUSOLVER_RETRY_TEST=1             every solve solved again by the next backend
#   OPENMX_MAGMA_OZAKI2=0 + RETRY_TEST=1      stock MAGMA only
#   RETRY_TEST=1, 10^6 ranks per device       the retry copies in host memory
#   OPENMX_MAGMA_OZAKI2_RETRY=0, CACHE_MB=0   no copy and no retry, no cached scratch
#   OPENMX_GPUSOLVER_PARTIAL_EVD_MIN_N=1      partial requests by the full divide and conquer
#   OPENMX_GPU_EIGENSOLVER_MAGMA_MIN_N=100000 the MAGMA entry points on rocSOLVER
#   OPENMX_GPU_EIGENSOLVER=hipsolver|magma   explicit policy through direct dense entries
# Usage: tests/run_gpusolver_smoke.sh [n of the shim check, default 257]
# Environment: ROCM (/opt/rocm-7.2.0), MPI_HOME (for mpi.h of openmx_common.h),
# ARCH (gfx1200), LAPACK_LIBS (-llapack -lblas).

test_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
src=$test_root/source
rocm=${ROCM:-/opt/rocm-7.2.0}
mpi=${MPI_HOME:-/home/kawai/opt/openmpi-rocmclang}
arch=${ARCH:-gfx1200}
lapack=${LAPACK_LIBS:--llapack -lblas}
clang=$rocm/llvm/bin/clang
test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/openmx-gpusolver-test.XXXXXX")
trap 'rm -rf "$test_tmp"' EXIT HUP INT TERM

for lib in "$src/third_party/magma/lib/libmagma.a" "$src/third_party/GEMMul8/lib/libgemmul8.a"; do
    if [ ! -f "$lib" ]; then
        echo "missing $lib: build OpenMX (its MAGMA and GEMMul8) first" >&2
        exit 2
    fi
done

"$rocm/bin/hipcc" -std=c++20 -O2 -fPIC --offload-arch="$arch" -I"$src" -I"$src/third_party/GEMMul8/include" \
    -I"$src/third_party/magma/include" -I"$src/third_party/magma/control" -D__HIP_PLATFORM_AMD__ \
    -ffp-contract=off -w -c "$src/magma_openmx.cu" -o "$test_tmp/magma_openmx.o"
"$clang" -O2 -std=c11 -w -fcommon -fopenmp --offload-arch="$arch" -D__HIP_PLATFORM_AMD__ -I"$src" \
    -I"$mpi/include" -I"$rocm/include" -c "$src/gpusolver_Syevdx.c" -o "$test_tmp/gpusolver_Syevdx.o"
"$clang" -O2 -std=c11 -fopenmp --offload-arch="$arch" -D__HIP_PLATFORM_AMD__ -I"$src" -I"$rocm/include" \
    -c "$test_root/tests/gpusolver_wrapper_smoke.c" -o "$test_tmp/wrapper.o"
"$clang" -fopenmp --offload-arch="$arch" "$test_tmp/wrapper.o" "$test_tmp/gpusolver_Syevdx.o" \
    "$test_tmp/magma_openmx.o" "$src/third_party/magma/lib/libmagma.a" \
    "$src/third_party/GEMMul8/lib/libgemmul8.a" -L"$rocm/lib" -Wl,-rpath,"$rocm/lib" \
    -lhipsolver -lrocsolver -lhipblas -lhipblaslt -lrocblas -lhipsparse -lamdhip64 $lapack -lstdc++ -lm \
    -o "$test_tmp/wrappers"
"$clang" -O2 -std=c11 -D__HIP_PLATFORM_AMD__ -I"$src" -I"$rocm/include" "$test_root/tests/hipsolver_smoke.c" \
    -L"$rocm/lib" -Wl,-rpath,"$rocm/lib" -lhipsolver -lamdhip64 -lm -o "$test_tmp/shim"

"$test_tmp/shim" "${1:-257}"

run() {
    echo "== $*"
    env "$@" OMP_NUM_THREADS=2 "$test_tmp/wrappers" 2>"$test_tmp/stderr" || {
        cat "$test_tmp/stderr" >&2
        exit 1
    }
    cat "$test_tmp/stderr" >&2
}

run OPENMX_TEST_DEFAULT=1
run OPENMX_GPUSOLVER_RETRY_TEST=1
grep -q "retry test" "$test_tmp/stderr" || { echo "the retry test did not retry" >&2; exit 1; }
run OPENMX_MAGMA_OZAKI2=0 OPENMX_GPUSOLVER_RETRY_TEST=1
run OPENMX_GPUSOLVER_RETRY_TEST=1 OPENMX_TEST_RANKS_PER_DEVICE=1000000
run OPENMX_MAGMA_OZAKI2_RETRY=0 OPENMX_GPUSOLVER_CACHE_MB=0
run OPENMX_GPUSOLVER_PARTIAL_EVD_MIN_N=1 OPENMX_CUSOLVER_CACHE_MB=1
run OPENMX_GPU_EIGENSOLVER_MAGMA_MIN_N=100000 OPENMX_TEST_EXPECT_BACKEND=1
run OPENMX_GPU_EIGENSOLVER=hipsolver OPENMX_TEST_EXPECT_BACKEND=1 OPENMX_GPU_EIGENSOLVER_PROFILE=1
grep -q 'backend=rocsolver' "$test_tmp/stderr" || { echo "rocSOLVER override was not used" >&2; exit 1; }
run OPENMX_GPU_EIGENSOLVER=magma OPENMX_TEST_EXPECT_BACKEND=0 OPENMX_GPU_EIGENSOLVER_PROFILE=1
grep -q 'backend=magma' "$test_tmp/stderr" || { echo "MAGMA override was not used" >&2; exit 1; }
echo "run_gpusolver_smoke: all runs passed"
