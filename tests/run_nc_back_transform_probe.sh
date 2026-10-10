#!/bin/sh
set -eu
# Build and run tests/nc_back_transform_probe.cu: the back transform of the
# non-collinear cluster solver as one complex GEMM against diag(S, S) (the
# old form), as a transpose plus two real GEMMs (the solver's form) and as
# two complex GEMMs against a complex S, on random matrices, for cuBLAS FP64
# and the GEMMul8 settings L=15 accurate, L=15 fast and L=12 fast, each
# without and with the release of the bridge's workspaces before every call
# (the solver releases them once per SCF step).  The GEMMul8 runs report a
# fallback to native cuBLAS on stderr.  Toolkit selection and
# GEMMUL8_TEST_OBJECT as in tests/run_gemmul8_smoke.sh.
#
#   tests/run_nc_back_transform_probe.sh [n maxn] ...
#
# Each "n maxn" pair is one shape (n2 = 2n, maxn solved states); the default
# shapes are those of Mn12_148_F_NC (1088 1087, first SCF step 1088 2176)
# and sidia333_nc_cluster (2808 1987, first step 2808 5616).  PROBE_RANKS
# (default 18) sets OPENMX_GEMMUL8_LOCAL_RANKS, the number of ranks the
# bridge assumes to share the GPU when it budgets its workspace; REPS (5)
# the timed calls per form.
[ $# -gt 0 ] || set -- 1088 1087 1088 2176 2808 1987 2808 5616
if [ $(($# % 2)) -ne 0 ]; then
    echo "usage: $0 [n maxn] ..." >&2
    exit 2
fi
test_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda-13.4}
math_root=${CUDA_MATH_ROOT:-$cuda_root}
sdk_root=${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.9}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
host_cxx=${NVCC_HOST:-$sdk_root/compilers/bin/nvc++}
gemmul8_root=${GEMMUL8_DIR:-$test_root/source/third_party/GEMMul8}
gpu_arch=${GPU_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '/^[0-9]+[.][0-9]+$/ {gsub(/[.]/, ""); print; exit}')}
test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/openmx-gemmul8-test.XXXXXX")
trap 'rm -rf "$test_tmp"' EXIT HUP INT TERM
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
gemmul8_object=${GEMMUL8_TEST_OBJECT:-$test_tmp/gemmul8.o}
if [ -z "${GEMMUL8_TEST_OBJECT:-}" ]; then
    "$nvcc" -ccbin "$host_cxx" -std=c++20 -O3 -diag-suppress 177 \
        -DGPU_ARCH="$gpu_arch" -arch="sm_$gpu_arch" \
        -I"$gemmul8_root/include" -I"$gemmul8_root/src" -I"$math_root/include" \
        -c "$test_root/source/gemmul8_openmx.cu" -o "$gemmul8_object"
fi
"$nvcc" -ccbin "$host_cxx" -std=c++20 -O2 -arch="sm_$gpu_arch" -I"$gemmul8_root/include" -I"$math_root/include" \
    "$test_root/tests/nc_back_transform_probe.cu" "$test_root/source/gemmul8_bridge.cu" \
    "$gemmul8_object" -L"$math_root/lib64" -L"$cuda_root/lib64" -lcublas -lcublasLt \
    -Xlinker -rpath -Xlinker "$math_root/lib64" \
    -Xlinker -rpath -Xlinker "$cuda_root/lib64" -o "$test_tmp/probe"

# the bridge's defaults, whatever the calling environment holds
unset OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT OPENMX_GEMMUL8_MAX_WORKSPACE_MB OPENMX_GEMMUL8_MIN_FREE_AFTER_MB \
      GEMMUL8_MAX_WORKSPACE_PERCENT GEMMUL8_MAX_WORKSPACE_MB GEMMUL8_MIN_FREE_AFTER_MB \
      GEMMUL8_DISABLE GEMMUL8_DISABLE_D GEMMUL8_DISABLE_Z GEMMUL8_NUM_MOD_D GEMMUL8_NUM_MOD_Z \
      GEMMUL8_FASTMODE_D GEMMUL8_FASTMODE_Z
export OPENMX_GEMMUL8_DISABLE=0 OPENMX_GEMMUL8_DISABLE_D=0 OPENMX_GEMMUL8_DISABLE_Z=0 OPENMX_GEMMUL8_VERBOSE=1
export OPENMX_GEMMUL8_LOCAL_RANKS="${PROBE_RANKS:-18}"
reps=${REPS:-5}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
while [ $# -ge 2 ]; do
    n=$1 maxn=$2
    shift 2
    for release in "" --release; do
        "$test_tmp/probe" "$n" "$maxn" "$reps" --fp64 $release
        for setting in "15 0 L15a" "15 1 L15f" "12 1 L12f"; do
            set -- $setting "$@"
            OPENMX_GEMMUL8_NUM_MOD_D=$1 OPENMX_GEMMUL8_NUM_MOD_Z=$1 \
            OPENMX_GEMMUL8_FASTMODE_D=$2 OPENMX_GEMMUL8_FASTMODE_Z=$2 \
                "$test_tmp/probe" "$n" "$maxn" "$reps" $release --label "$3"
            shift 3
        done
    done
done
