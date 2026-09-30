#!/bin/sh
set -eu

# The production executable is unchanged. LD_PRELOAD only injects a memory
# admission failure into this child job after the first SCF GPU solve.
test_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda-13.4}
sdk_root=${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.9}
test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/openmx-cluster-fallback.XXXXXX")
trap 'rm -rf "$test_tmp"' EXIT HUP INT TERM
unset LD_LIBRARY_PATH OPAL_PREFIX CPATH C_INCLUDE_PATH LIBRARY_PATH
export NVCOMPILER_COMM_LIBS_HOME=${NVCOMPILER_COMM_LIBS_HOME:-$sdk_root/comm_libs/13.3}
${CC:-cc} -O2 -std=gnu11 -shared -fPIC -I"$cuda_root/include" \
    "$test_root/tests/cluster_memory_pressure.c" -ldl -o "$test_tmp/pressure.so"
python3 "$test_root/tools/run_gpu_suites.py" \
    --binary "${1:-$test_root/source/openmx}" --output "$test_tmp/run" \
    --suite S --cases Benzene --ranks 2 --timeout 180 \
    --mpirun "$sdk_root/comm_libs/mpi/bin/mpirun" \
    --env "LD_PRELOAD=$test_tmp/pressure.so" \
    --env OPENMX_CUSOLVER_ALGORITHM=auto --env OPENMX_CUSOLVER_VERBOSE=1 \
    --env OPENMX_CLUSTER_GPU_SWITCH_NUM=1 --env OPENMX_CLUSTER_GPU_RETAIN_SCRATCH=0
python3 - "$test_tmp/run/Benzene/run.log" <<'PY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text()
for marker in ('TEST_MEMORY_PRESSURE:', 'Released ',
               'Falling back to the ELPA/ScaLAPACK diagonalization.'):
    if marker not in text:
        print(text, file=sys.stderr)
        raise SystemExit('Required path was not exercised: ' + marker)
print('GPU first SCF -> scratch release -> CPU fallback exercised.')
PY
python3 "$test_root/tools/compare_gpu_suites.py" "$test_tmp/run" --reference
