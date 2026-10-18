#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 input.dat (outputs are written in the working directory)" >&2
    exit 2
fi
source "$repo/build/env.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export OMP_NUM_THREADS=${OPENMX_RUN_THREADS:-32}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MAGMA_NUM_THREADS=$OMP_NUM_THREADS
export OPENMX_MAGMA_FP32=1 OPENMX_MGPU_BAND_SHARED=1 OPENMX_MGPU_NUM_GPUS=2
export OPENMX_MAGMA_FP32_COMPLEX_MAX_N=${OPENMX_MAGMA_FP32_COMPLEX_MAX_N:-16384}
export OPENMX_MGPU_GEMM_WORKSPACE_MB=${OPENMX_MGPU_GEMM_WORKSPACE_MB:-24576}
export OPENMX_MGPU_GEMM_HOST_MB=$OPENMX_MGPU_GEMM_WORKSPACE_MB
export OPENMX_MGPU_GEMM_PAGEABLE=1
export OPENMX_MAGMA_REFINE_RESIDENT_MB=${OPENMX_MAGMA_REFINE_RESIDENT_MB:-28672}
export OPENMX_MAGMA_REFINE_RESERVE_MB=2048
export OPENMX_MAGMA_REFINE_STEPS=${OPENMX_MAGMA_REFINE_STEPS:-3}
# A 26000 x 26000 spinor basis needs more memory than the collinear cases.
spin=$(awk 'tolower($1)=="scf.spinpolarization" {print tolower($2); exit}' "$1")
if [ "$spin" = nc ]; then
    export OPENMX_GEMMUL8_MAX_WORKSPACE_MB=${OPENMX_GEMMUL8_MAX_WORKSPACE_MB:-2048}
else
    export OPENMX_GEMMUL8_MAX_WORKSPACE_MB=${OPENMX_GEMMUL8_MAX_WORKSPACE_MB:-6144}
fi
export OPENMX_GEMMUL8_TIMING=0
export OPENMX_MGPU_VERBOSE=1 OPENMX_BAND_GPU_MAX_CONCURRENT_K=1
# Host states avoid repeating diagonalization after finding the Fermi level.
# Existing available-RAM/per-rank caps still apply inside OpenMX.
export OPENMX_BAND_COL_KCACHE_MB=${OPENMX_BAND_COL_KCACHE_MB:-4096}
export OPENMX_BAND_NONCOL_KCACHE_MB=${OPENMX_BAND_NONCOL_KCACHE_MB:-24576}
exec mpirun --bind-to none -np "${OPENMX_RUN_RANKS:-1}" "$repo/build/cuda/openmx" "$1" -nt "$OMP_NUM_THREADS"
