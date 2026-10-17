#!/bin/sh
set -eu
# Compare AMD's retained general/Unblocked GEMMul8 bridge with tiled multi-GPU GEMM.
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "$root/tests/hip_test_build.sh"
out=${1:?provide an output directory}
mkdir -p "$out"
gemmul8=${GEMMUL8_DIR:-$root/source/third_party/GEMMul8}
gemmul8_lib=${GEMMUL8_TEST_LIB:-$gemmul8/lib/libgemmul8.a}
# The library must have been built with this toolkit/architecture by make.
test -f "$gemmul8_lib"
"$hipcc" -std=c++20 -O2 --offload-arch="$gpu_arch" -DOPENMX_MGPU_BENCH_BRIDGE \
    -D__HIP_PLATFORM_AMD__ -ffp-contract=off -DGPU_ARCH="$gpu_arch" \
    -I"$root/source" -I"$gemmul8/include" -I"$gemmul8/src" \
    "$root/tests/mgpu_gemm_bench.cu" "$root/source/openmx_mgpu_gemm.cu" \
    "$root/source/gemmul8_bridge.cu" -x none "$gemmul8_lib" \
    -L"$rocm/lib" -lhipblas -lhipblaslt -lpthread -Wl,-rpath,"$rocm/lib" \
    -o "$out/mgpu_gemm_bench_bridge"
