#!/bin/sh
set -eu
# Build and run tools/nc_back_transform_probe_hip.cu: the device stages of the
# non-collinear cluster solver's root dense path (the forward stage's scatter
# and Hs2 assembly, the back transform in its device and host-staged forms,
# the density matrix as per-entry sums and as GEMMul8 products) on random
# matrices of the solver's shapes, against host builds and long-double
# references; HIP port and extension of the NVIDIA version's
# tests/run_nc_back_transform_probe.sh.  Builds the probe with the tree's
# source/Cluster_DFT_NonCol_dm_hip.cu and source/gemmul8_bridge.cu and the
# built source/third_party/GEMMul8/lib/libgemmul8.a, in a temporary directory.
#
#   tools/run_nc_back_transform_probe_hip.sh [n maxn nk] ...
#
# Each "n maxn nk" triple is one shape (n2 = 2n, maxn solved states, nk
# occupied ones); the default is sidia333_nc_cluster's (2808 1987 888, about
# 1.3 GiB of device memory) and a small one (1092 1000 400).  PROBE_RANKS
# (default 16) sets OPENMX_GEMMUL8_LOCAL_RANKS, the ranks the bridge assumes
# to share the GPU when it budgets its workspace; REPS (5) the timed calls.
# Extra probe options (--edm, --old, --fp64, --only ...) go in PROBE_ARGS.
[ $# -gt 0 ] || set -- 2808 1987 888 1092 1000 400
if [ $(($# % 3)) -ne 0 ]; then
    echo "usage: $0 [n maxn nk] ..." >&2
    exit 2
fi
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
rocm=${ROCM_PATH:-/opt/rocm-7.2.0}
arch=${GPU_ARCH:-gfx1200}
gemmul8=$root/source/third_party/GEMMul8
work=$(mktemp -d "${TMPDIR:-/tmp}/openmx-nc-probe.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
flags="-std=c++20 -O3 --offload-arch=$arch -I$root/source -I$gemmul8/include -I$gemmul8/src -I$rocm/include \
-D__HIP_PLATFORM_AMD__ -ffp-contract=off -Wno-unused-result -Wno-unused-command-line-argument -Wno-unused-value \
-DGPU_ARCH=$arch"
"$rocm/bin/hipcc" $flags -c "$root/source/gemmul8_bridge.cu" -o "$work/gemmul8_bridge.o"
"$rocm/bin/hipcc" $flags -c "$root/source/Cluster_DFT_NonCol_dm_hip.cu" -o "$work/nc_dm_hip.o"
"$rocm/bin/hipcc" $flags -c "$root/tools/nc_back_transform_probe_hip.cu" -o "$work/probe.o"
"$rocm/bin/hipcc" --offload-arch="$arch" "$work/probe.o" "$work/nc_dm_hip.o" "$work/gemmul8_bridge.o" \
    "$gemmul8/lib/libgemmul8.a" -L"$rocm/lib" -lhipblas -lhipblaslt -lrocblas -lamdhip64 \
    -Wl,-rpath,"$rocm/lib" -o "$work/probe"

# the bridge's defaults, whatever the calling environment holds
unset OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT OPENMX_GEMMUL8_MAX_WORKSPACE_MB OPENMX_GEMMUL8_MIN_FREE_AFTER_MB \
      OPENMX_GEMMUL8_SOLVE_MIN_FREE_AFTER_MB GEMMUL8_MAX_WORKSPACE_PERCENT GEMMUL8_MAX_WORKSPACE_MB \
      GEMMUL8_MIN_FREE_AFTER_MB GEMMUL8_DISABLE GEMMUL8_DISABLE_D GEMMUL8_DISABLE_Z GEMMUL8_NUM_MOD_D \
      GEMMUL8_NUM_MOD_Z GEMMUL8_FASTMODE_D GEMMUL8_FASTMODE_Z OPENMX_GEMMUL8_NUM_MOD_D OPENMX_GEMMUL8_NUM_MOD_Z \
      OPENMX_GEMMUL8_FASTMODE_D OPENMX_GEMMUL8_FASTMODE_Z
export OPENMX_GEMMUL8_DISABLE=0 OPENMX_GEMMUL8_DISABLE_D=0 OPENMX_GEMMUL8_DISABLE_Z=0 OPENMX_GEMMUL8_VERBOSE=1
export OPENMX_GEMMUL8_LOCAL_RANKS="${PROBE_RANKS:-16}"
reps=${REPS:-5}
status=0
while [ $# -ge 3 ]; do
    n=$1 maxn=$2 nk=$3
    shift 3
    "$work/probe" "$n" "$maxn" "$nk" "$reps" ${PROBE_ARGS:-} || status=1
done
exit $status
