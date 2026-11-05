#!/usr/bin/env bash
# Continue after the already completed baseline-01; never overwrite a run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
results="$PWD/benchmarks/results/graviton3_sve256_20261105"
before="$results/baseline-install/bin/openmx"
after="$results/optimized-install/bin/openmx"
run() {
  local name=$1 binary=$2 mode=$3 blas=$4
  sync
  python3 tools/bench_sve256_local.py "$name" --binary "$binary" \
    --mode "$mode" --expect-blas-core "$blas" >"$results/$name-summary.log" 2>&1
}
run optimized-01 "$after" sve256 NEOVERSEV1
python3 tools/analyze_sve256_local.py --baseline baseline-01 \
  --optimized optimized-01 --output first-comparison
run baseline-02 "$before" auto ARMV8
run optimized-02 "$after" sve256 NEOVERSEV1
python3 tools/analyze_sve256_local.py --baseline baseline-01 baseline-02 \
  --optimized optimized-01 optimized-02 --output second-comparison
run baseline-03 "$before" auto ARMV8
run optimized-03 "$after" sve256 NEOVERSEV1
python3 tools/analyze_sve256_local.py --baseline baseline-01 baseline-02 baseline-03 \
  --optimized optimized-01 optimized-02 optimized-03
