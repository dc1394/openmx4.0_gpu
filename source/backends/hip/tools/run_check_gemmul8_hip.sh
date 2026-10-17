#!/bin/bash
# Builds tools/check_gemmul8_hip.cu against the GEMMul8 bridge object and runs
# it twice, without and with OPENMX_GEMMUL8_TIMING=1: both runs must pass every
# check, produce the same bridge results bit for bit (the checksum line), and
# only the timed run may print the general-GEMM report of
# openmx_gemmul8ReportTiming().  Single process, about 1.5 GiB of device
# memory; run it alone on the GPU (the trim and fallback checks read the
# device's free memory).
#
#   tools/run_check_gemmul8_hip.sh            (from the repository root, after building the objects)
#   BRIDGE_OBJ=/path/gemmul8_bridge.o OUT=/tmp/dir tools/run_check_gemmul8_hip.sh
#
# BRIDGE_OBJ  the bridge object (default source/gemmul8_bridge.o)
# GEMMUL8_LIB the GEMMul8 library with gemmul8_openmx.o and gemmul8_config.o
#             (default source/third_party/GEMMul8/lib/libgemmul8.a)
# OUT         build and log directory (default work/check_gemmul8_hip)
# ROCM        ROCm installation (default /opt/rocm-7.2.0); ARCH (default gfx1200)
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
ROCM=${ROCM:-/opt/rocm-7.2.0}
ARCH=${ARCH:-gfx1200}
BRIDGE_OBJ=${BRIDGE_OBJ:-$root/source/gemmul8_bridge.o}
GEMMUL8_LIB=${GEMMUL8_LIB:-$root/source/third_party/GEMMul8/lib/libgemmul8.a}
OUT=${OUT:-$root/work/check_gemmul8_hip}

mkdir -p "$OUT"
"$ROCM/bin/hipcc" -std=c++20 -O2 --offload-arch="$ARCH" -ffp-contract=off \
    -I"$root/source/third_party/GEMMul8/include" \
    -c "$root/tools/check_gemmul8_hip.cu" -o "$OUT/check_gemmul8_hip.o" 2> "$OUT/build.log" ||
    { cat "$OUT/build.log"; exit 1; }
"$ROCM/bin/hipcc" --offload-arch="$ARCH" "$OUT/check_gemmul8_hip.o" "$BRIDGE_OBJ" "$GEMMUL8_LIB" \
    -L"$ROCM/lib" -lhipblas -lhipblaslt -Wl,-rpath,"$ROCM/lib" -o "$OUT/check_gemmul8_hip" 2>> "$OUT/build.log" ||
    { cat "$OUT/build.log"; exit 1; }

# the bridge defaults, whatever the caller's environment says
unset OPENMX_GEMMUL8_DISABLE OPENMX_GEMMUL8_DISABLE_D OPENMX_GEMMUL8_DISABLE_Z GEMMUL8_DISABLE GEMMUL8_DISABLE_D \
      GEMMUL8_DISABLE_Z OPENMX_GEMMUL8_SOLVE_MIN_FREE_AFTER_MB GEMMUL8_SOLVE_MIN_FREE_AFTER_MB OPENMX_GEMMUL8_TIMING \
      GEMMUL8_MAX_WORKSPACE_MB GEMMUL8_NUM_MOD_D GEMMUL8_NUM_MOD_Z GEMMUL8_FASTMODE_D GEMMUL8_FASTMODE_Z \
      OPENMX_GEMMUL8_MIN_FREE_AFTER_MB GEMMUL8_MIN_FREE_AFTER_MB OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT \
      GEMMUL8_MAX_WORKSPACE_PERCENT OMPI_COMM_WORLD_RANK PMI_RANK
export OPENMX_GEMMUL8_MAX_WORKSPACE_MB=256 OPENMX_GEMMUL8_NUM_MOD_D=15 OPENMX_GEMMUL8_NUM_MOD_Z=15 \
       OPENMX_GEMMUL8_FASTMODE_D=0 OPENMX_GEMMUL8_FASTMODE_Z=0 OPENMX_GEMMUL8_LOCAL_RANKS=1

status=0
"$OUT/check_gemmul8_hip" > "$OUT/plain.log" 2>&1 || { echo "FAIL: plain run"; tail -5 "$OUT/plain.log"; status=1; }
OPENMX_GEMMUL8_TIMING=1 "$OUT/check_gemmul8_hip" > "$OUT/timing.log" 2>&1 ||
    { echo "FAIL: timed run"; tail -5 "$OUT/timing.log"; status=1; }
[ $status -eq 0 ] || exit 1

plain_sum=$(grep '^bridge result checksum' "$OUT/plain.log")
timing_sum=$(grep '^bridge result checksum' "$OUT/timing.log")
if [ "$plain_sum" != "$timing_sum" ]; then
    echo "FAIL: OPENMX_GEMMUL8_TIMING changed the results ($plain_sum / $timing_sum)"
    exit 1
fi
if grep -q 'general GEMMs, rank' "$OUT/plain.log"; then
    echo "FAIL: the general-GEMM report without OPENMX_GEMMUL8_TIMING"
    exit 1
fi
if ! grep -q 'general GEMMs, rank -: real [1-9][0-9]* + [1-9][0-9]* calls' "$OUT/timing.log"; then
    echo "FAIL: no general-GEMM report (with GEMMul8 and hipBLAS calls) with OPENMX_GEMMUL8_TIMING=1"
    exit 1
fi
grep -E '^(PASS .*(unblocked|trim|native counter|without room)|All )' "$OUT/plain.log"
grep 'general GEMMs, rank' "$OUT/timing.log"
echo "$plain_sum in both runs"
echo "PASS: tools/check_gemmul8_hip.cu with and without OPENMX_GEMMUL8_TIMING"
