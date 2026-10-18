#!/usr/bin/env bash
set -euo pipefail
# Reuse production objects to exercise the real GEMMul8 bridge and adapter.
test_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
native=${OPENMX_NATIVE_BUILD:-$test_root/build/cuda/native/source}
out=${MAGMA_TEST_DIR:-$test_root/build/magma-refine-tests}
cuda_root=${CUDA_HOME:?Set CUDA_HOME}
math_root=${CUDA_MATH_ROOT:-$cuda_root}
prefix=${MAGMA_PREFIX:-$native/third_party/magma-install}
mkdir -p "$out"
"${CXX:-nvc++}" -std=c++17 -O2 -mp -DOPENMX_HAVE_MAGMA \
    -I"$test_root/source" -I"$cuda_root/include" -I"$math_root/include" \
    "$test_root/tests/magma_mgpu_smoke.cpp" "$native/openmx_magma_mgpu.o" \
    "$native/openmx_mgpu_gemm.o" "$native/gemmul8_bridge.o" \
    "$native/third_party/GEMMul8/src/gemmul8_openmx.o" \
    -L"$prefix/lib" -L"$cuda_root/lib64" -L"$math_root/lib64" \
    -lmagma -lcudart -lcublas -lcublasLt -llapack -lblas -lpthread \
    -Wl,--disable-new-dtags,-rpath,"$prefix/lib",-rpath,"$cuda_root/lib64",-rpath,"$math_root/lib64" \
    -o "$out/magma-refine-smoke"
export OPENMX_MAGMA_FP32=1 OPENMX_MAGMA_VALIDATE_FULL=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MAGMA_NUM_THREADS=${MAGMA_NUM_THREADS:-8}
for spectrum in simple clustered; do
    if [ "$spectrum" = clustered ]; then export OPENMX_TEST_CLUSTERED=1; else unset OPENMX_TEST_CLUSTERED; fi
    "$out/magma-refine-smoke" 2 513 1,0 0 > "$out/$spectrum.log" 2>&1
    [ "$(rg -c 'FP32 refined.*valid=1' "$out/$spectrum.log")" = 8 ]
    if rg 'FP32 refinement declined|FAIL' "$out/$spectrum.log"; then exit 1; fi
    tail -1 "$out/$spectrum.log"
done
# The same analytic clusters exercise the GPU generalized Rayleigh-Ritz path.
OPENMX_TEST_CLUSTERED=1 OPENMX_TEST_CLUSTER_SIZE=129 OPENMX_MAGMA_RR_GPU_MIN=2 "$out/magma-refine-smoke" 2 257 1,0 0 > "$out/gpu-clusters.log" 2>&1
[ "$(rg -c 'FP32 refined.*valid=1' "$out/gpu-clusters.log")" = 8 ]
if rg 'FP32 refinement declined|FAIL' "$out/gpu-clusters.log"; then exit 1; fi
tail -1 "$out/gpu-clusters.log"
OPENMX_TEST_FP32_DISABLED=1 "$out/magma-refine-smoke" 2 257 > "$out/disabled.log" 2>&1
if rg 'FP32 basis|FP32 refined' "$out/disabled.log"; then exit 1; fi
tail -1 "$out/disabled.log"
# Planned precision selection is distinct from an unsuccessful refinement.
OPENMX_MAGMA_FP32_COMPLEX_MAX_N=128 "$out/magma-refine-smoke" 2 257 > "$out/complex-policy.log" 2>&1
[ "$(rg -c 'FP32 refined.*valid=1' "$out/complex-policy.log")" = 4 ]
[ "$(rg -c 'precision policy.*mode=FP64' "$out/complex-policy.log")" = 4 ]
if rg 'FP32 refinement declined|FAIL' "$out/complex-policy.log"; then exit 1; fi
tail -1 "$out/complex-policy.log"
# A bounded-product allocation refusal must retry the original FP64 matrix.
# The analytic oracle also checks padding, untouched output and device restore.
OPENMX_MAGMA_REFINE_RESIDENT_MB=0 OPENMX_MGPU_GEMM_WORKSPACE_MB=0 "$out/magma-refine-smoke" 2 257 > "$out/fallback.log" 2>&1
[ "$(rg -c 'FP32 refinement declined' "$out/fallback.log")" = 8 ]
if rg 'FP32 refined|FAIL' "$out/fallback.log"; then exit 1; fi
tail -1 "$out/fallback.log"
