#!/bin/sh
set -eu
# Build on the login node; execute only inside a scheduler GPU allocation.
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "$root/tests/hip_test_build.sh"
out=${1:?provide an output directory};shift
mkdir -p "$out"
for fixture in mgpu_gemm_smoke mgpu_gemm_bench; do
    "$hipcc" -std=c++17 -O2 --offload-arch="$gpu_arch" -DOPENMX_MGPU_TEST_BRIDGE \
        -I"$root/source" "$root/tests/$fixture.cu" "$root/source/openmx_mgpu_gemm.cu" \
        -L"$rocm/lib" -lhipblas -Wl,-rpath,"$rocm/lib" -lpthread -o "$out/$fixture"
done
if [ "${1:-}" = --build-only ]; then exit 0; fi
"$out/mgpu_gemm_smoke" "$@"
