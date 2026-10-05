#!/bin/sh
# Matrix-level measurement of the two forward GEMMs, B = H X and C = X^T B,
# on synthetic matrices: builds tests/gemmul8_reuse_probe (two binaries),
# checks its reference arithmetic, and runs it for each matrix order, once
# for wall times and once for GEMMul8's phase times.
#
#   tools/measure_gemm_probe.sh <output dir> [order ...]      (default: 4000 8000)
#
# Toolkit selection through the environment as in
# tests/run_gemmul8_reuse_probe.sh; NVCC_HOST=g++ is recommended.  MODULI,
# FAST, REPS and PROBE_ARGS change the probe options.  Configurations that do
# not fit the device are skipped by the probe and listed in probe.log.
set -eu
[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 2; }
out=$1
shift
[ $# -ge 1 ] || set -- 4000 8000
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python=${PYTHON:-python3}
moduli=${MODULI:-8,10,12,14,15,16,20}
fast=${FAST:-0,1}
reps=${REPS:-3}

mkdir -p "$out"
out=$(CDPATH= cd -- "$out" && pwd)
bin=$out/bin
sh "$repo/tests/run_gemmul8_reuse_probe.sh" "$bin"

{
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "commit: $(git -C "$repo" log --oneline -1 2>/dev/null)"
  echo "modified: $(git -C "$repo" status --short -uno -- source tests tools 2>/dev/null | tr '\n' ';')"
  echo "GEMMul8: $(git -C "$repo/source/third_party/GEMMul8" describe --tags --always 2>/dev/null)"
  echo "host compiler: ${NVCC_HOST:-default}"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.used --format=csv 2>/dev/null
} > "$out/environment.txt"
cat "$out/environment.txt"

"$bin/gemmul8_reuse_probe" selftest | tee "$out/selftest.log"
grep -q '^PASS' "$out/selftest.log" || { echo "the reference arithmetic failed its self-test; try NVCC_HOST=g++" >&2; exit 1; }
"$bin/gemmul8_reuse_probe" capacity "$@" > "$out/capacity.txt"

for n in "$@"; do
  dir=$out/n$n
  mkdir -p "$dir"
  rm -f "$dir/probe.jsonl" "$dir/probe_prof.jsonl"
  echo
  echo "== n = $n: wall times"
  "$bin/gemmul8_reuse_probe" run --synthetic "$n" --steps 3 --label "synthetic_n$n" --moduli "$moduli" \
      --fast "$fast" --reps "$reps" ${PROBE_ARGS:-} --json "$dir/probe.jsonl" > "$dir/probe.log" 2>&1 ||
      echo "the probe failed for n = $n (see $dir/probe.log)"
  grep -E '^GPU|^skip|fp64 wall|time:' "$dir/probe.log" | cut -c1-200 || true
  echo "== n = $n: phase times"
  "$bin/gemmul8_reuse_probe_prof" run --synthetic "$n" --steps 3 --label "synthetic_n$n" --moduli "$moduli" \
      --fast "$fast" --reps "$reps" ${PROBE_ARGS:-} --json "$dir/probe_prof.jsonl" > "$dir/probe_prof.log" 2>&1 ||
      echo "the profiling probe failed for n = $n (see $dir/probe_prof.log)"
  "$python" "$repo/tools/summarize_gemm_probe.py" "$dir" > "$dir/RESULTS.md" || true
done
echo
echo "results: $out/n*/RESULTS.md"
