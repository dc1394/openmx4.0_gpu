#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "$root/tests/hip_test_build.sh"
magma=${MAGMA_DIR:-$root/source/third_party/magma}
out=${MAGMA_TEST_DIR:-$root/source/build/magma-tests}
mkdir -p "$out"
"$hipcc" -std=c++17 -O2 -fopenmp --offload-arch="$gpu_arch" -DOPENMX_HAVE_MAGMA -DADD_ \
    -I"$root/source" -I"$magma/include" \
    "$root/tests/magma_mgpu_smoke.cpp" "$root/source/openmx_magma_mgpu.cpp" \
    -x none "$magma/lib/libmagma.a" "${GEMMUL8_TEST_LIB:-$root/source/third_party/GEMMul8/lib/libgemmul8.a}" -L"$rocm/lib" -lhipblas -lhipblaslt -lhipsparse -lpthread \
    -L"$aocl/lib" -lflame -lblis \
    -Wl,--disable-new-dtags,-rpath,"$rocm/lib",-rpath,"$aocl/lib",-rpath,"$aocc_lib" \
    -o "$out/magma_mgpu_smoke"
if [ "${1:-}" = --build-only ]; then exit 0; fi
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MAGMA_NUM_THREADS=${MAGMA_NUM_THREADS:-1} \
    "$out/magma_mgpu_smoke" "$@"
