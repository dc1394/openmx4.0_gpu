#!/bin/sh
set -eu
# Runtime/group routing fixture: MAGMA itself is disabled in this probe.
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "$root/tests/hip_test_build.sh"
out=${1:?provide an output directory};shift
mkdir -p "$out"
for name in openmx_host_memory openmx_dense_cpu; do
    "$cc" -std=c11 -O2 -I"$root/source" -c "$root/source/$name.c" -o "$out/$name.o"
done
"$hipcc" -std=c++17 -O2 --offload-arch="$gpu_arch" -I"$root/source" \
    "$root/tests/mgpu_band_group_probe.cu" "$root/source/openmx_mgpu_runtime.cu" \
    "$root/source/openmx_mgpu_gemm.cu" "$root/source/openmx_magma_mgpu.cpp" \
    -x none "$out/openmx_host_memory.o" "$out/openmx_dense_cpu.o" \
    -L"$rocm/lib" -lhipblas -lhipsolver -ldl -lpthread \
    -L"$aocl/lib" -lflame -lblis -Wl,--disable-new-dtags,-rpath,"$aocl/lib",-rpath,"$aocc_lib",-rpath,"$rocm/lib" \
    -o "$out/mgpu_band_group_probe"
if [ "${1:-}" = --build-only ]; then exit 0; fi
"$out/mgpu_band_group_probe" "$@"
