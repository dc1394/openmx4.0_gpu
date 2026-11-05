#!/usr/bin/env bash
# The baseline-01 and optimized-01 checkpoint runs are already complete.
# Each invocation creates a fresh run directory and refuses to overwrite it.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
results="$PWD/benchmarks/results/graviton3_sve256_20261105"
before="$results/baseline-install/bin/openmx"
after="$results/own-optimized-install/bin/openmx"
run() {
  local name=$1 binary=$2 mode=$3 blas=$4
  sync
  python3 tools/bench_sve256_local.py "$name" --binary "$binary" \
    --mode "$mode" --expect-blas-core "$blas" >"$results/$name-summary.log" 2>&1
}
if [[ ${1:-} != resume-after-first ]]; then
  run own-final-01 "$after" sve256 NEOVERSEV1
fi
python3 tools/analyze_sve256_local.py --baseline baseline-01 \
  --optimized own-final-01 --output own-first-comparison
python3 tools/analyze_sve256_local.py --baseline optimized-01 \
  --optimized own-final-01 --output own-first-ablation
run baseline-02 "$before" auto ARMV8
run own-final-02 "$after" sve256 NEOVERSEV1
python3 tools/analyze_sve256_local.py --baseline baseline-02 \
  --optimized own-final-01 own-final-02 --output own-second-comparison
run baseline-03 "$before" auto ARMV8
run own-final-03 "$after" sve256 NEOVERSEV1
# Retake the early baseline with the same complete BLAS audit as final runs.
run baseline-04 "$before" auto ARMV8
python3 tools/analyze_sve256_local.py --baseline baseline-02 baseline-03 baseline-04 \
  --optimized own-final-01 own-final-02 own-final-03
python3 tools/analyze_sve256_local.py --baseline optimized-01 \
  --optimized own-final-01 own-final-02 own-final-03 --output own-ablation
