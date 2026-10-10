#!/bin/sh
set -eu
# Build tests/eigen_precision_probe.cu and run it on matrices sampled from an
# OpenMX run (OPENMX_GEMM_SAMPLE_DIR / OPENMX_GEMM_SAMPLE_ITERS): the dense
# eigensolve of the cluster solvers in native FP64, FP64 emulation (dynamic,
# with a mantissa bit offset, or with fixed mantissa bits; cuSOLVER >= 12.2),
# FP32, and FP32 followed by Ogita-Aishima refinement (fp32oaK, fp32soaK, with
# the products through the GEMMul8 bridge), each compared with native FP64.
# Toolkit selection and GEMMUL8_TEST_OBJECT as in tests/run_gemmul8_smoke.sh.
#
#   tests/run_eigen_precision_probe.sh <kind> <n> <maxn first step> <maxn> <electrons> <occupancy> <kT> <X|-> <sample> ...
#
# kind is complex for Hs2_scf*.bin of the non-collinear cluster solver (X is
# then -) and real for H_g*_scf*_s*.bin of the collinear one (X: its
# X_g*.bin).  Samples whose name contains scf001 are solved for all states
# (the first SCF step), the others for maxn.  occupancy is 1 for
# non-collinear and spin-polarized blocks, 2 otherwise; kT in Hartree
# (3.166811563e-6 times the electronic temperature in K).  REPS (default 3)
# sets the timed calls per mode, MODES a comma-separated list of the probe's
# modes (default: fp64, the emulation modes and fp32), PROBE_OA_GEMM=fp64 the
# refinement products in plain cuBLAS FP64; the GEMMul8 settings of the
# refinement products come from the environment as in OpenMX
# (OPENMX_GEMMUL8_NUM_MOD_Z, ..._FASTMODE_Z, ..._MAX_WORKSPACE_MB), and a
# fallback to native cuBLAS is reported on stderr.  Example,
# sidia333_nc_cluster sampled into samples/:
#
#   tests/run_eigen_precision_probe.sh complex 5616 5616 1987 864 1 9.50043e-4 - samples/Hs2_scf*.bin
if [ $# -lt 9 ]; then
    echo "usage: $0 <kind> <n> <maxn first step> <maxn> <electrons> <occupancy> <kT> <X|-> <sample> ..." >&2
    exit 2
fi
kind=$1 n=$2 maxn_first=$3 maxn=$4 electrons=$5 occupancy=$6 kT=$7 x=$8
shift 8
test_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda-13.4}
math_root=${CUDA_MATH_ROOT:-$cuda_root}
sdk_root=${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.9}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
host_cxx=${NVCC_HOST:-$sdk_root/compilers/bin/nvc++}
gemmul8_root=${GEMMUL8_DIR:-$test_root/source/third_party/GEMMul8}
gpu_arch=${GPU_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '/^[0-9]+[.][0-9]+$/ {gsub(/[.]/, ""); print; exit}')}
test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/openmx-eigen-probe.XXXXXX")
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
    "$test_root/tests/eigen_precision_probe.cu" "$test_root/source/gemmul8_bridge.cu" "$gemmul8_object" \
    -L"$math_root/lib64" -L"$cuda_root/lib64" -lcusolver -lcublas -lcublasLt \
    -Xlinker -rpath -Xlinker "$math_root/lib64" \
    -Xlinker -rpath -Xlinker "$cuda_root/lib64" -o "$test_tmp/probe"

export OPENMX_GEMMUL8_VERBOSE="${OPENMX_GEMMUL8_VERBOSE:-1}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
status=0
for sample in "$@"; do
    case $sample in
        *scf001*) states=$maxn_first ;;
        *) states=$maxn ;;
    esac
    "$test_tmp/probe" "$kind" "$sample" "$x" "$n" "$states" "$electrons" "$occupancy" "$kT" "${REPS:-3}" ${MODES:-} ||
        status=$?
done
exit $status
