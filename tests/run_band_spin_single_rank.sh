#!/usr/bin/env bash
# One-rank spin-polarized band regression: refinement must retain the FP64
# GPU path because the rank processes both spins in sequence.
# Usage: bash tests/run_band_spin_single_rank.sh [binary] [new-output-dir]
# OPENMX_TEST_MPI_LAUNCHER overrides "mpirun --bind-to core".
# The output directory must be below this repository's work/ directory.
set -euo pipefail
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
binary=${1:-$repo/source/openmx}
out=${2:-$repo/work/band_spin_single_rank_regression}
launcher=${OPENMX_TEST_MPI_LAUNCHER:-mpirun --bind-to core}
binary=$(realpath "$binary")
out=$(realpath -m "$out")
if [[ -e "$out" ]]; then
  echo "Refusing to mix results with existing directory: $out" >&2
  exit 2
fi
python3 - "$repo" "$out" <<'PY'
from pathlib import Path
import re
import sys
repo, out = map(Path, sys.argv[1:])
out.relative_to(repo / 'work')
(out / 'input').mkdir(parents=True)
text = (repo / 'work/input_example/Crys-MnO.dat').read_text()
# A 1x1x1 mesh is converted to the cluster solver in Input_std.c.  Keep
# two k-points so this tests Band_DFT_Col and its sequential spin loop.
for key, value in [('scf.Kgrid', '2 1 1'), ('scf.maxIter', '3'),
                   ('DATA.PATH', str(repo / 'DFT_DATA19'))]:
    pattern = rf'(?im)^\s*{re.escape(key)}\s+.*$'
    if re.search(pattern, text):
        text = re.sub(pattern, key + ' ' + value, text)
    else:
        text += '\n' + key + ' ' + value + '\n'
(out / 'input/Crys-MnO.dat').write_text(text)
(out / 'variants.txt').write_text('R0 OPENMX_EIGEN_REFINE=0\nR2 OPENMX_EIGEN_REFINE=2\n')
PY
relative_input=${out#"$repo/work/"}/input
# DFT_GPU_DenseSwitchNum delegates to Band_DFT_Col_GpuSwitchNum for Solver=3,
# so this override also bypasses DFT's global small-matrix CPU cutoff.
env -u LD_LIBRARY_PATH -u OPAL_PREFIX OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  bash "$repo/tools/run_forward_variants.sh" -o "$out/results" -b "$binary" -n 1 \
  -m "$launcher" -v "$out/variants.txt" \
  -s 'OPENMX_BAND_GPU_SWITCH_NUM=1 OPENMX_BAND_GPU_DIAG=1 OPENMX_BAND_PROFILE=1 OPENMX_GRID_PRECISION=fp64 OPENMX_GRID_FP32=0' \
  "Crys-MnO:$relative_input"
python3 - "$repo" "$out/results/Crys-MnO" <<'PY'
from pathlib import Path
import re
import sys
repo, cases = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo / 'tools'))
from compare_gpu_suites import read_output
outputs = []
for label, enabled in [('R0', 0), ('R2', 2)]:
    std = (cases / label / 'Crys-MnO.std').read_text()
    assert 'The calculation was normally finished.' in std, label
    assert 'GPU initialization failed' not in std, label
    assert not re.search(r'global matrix dimension \d+ is below', std), label
    assert f'OPENMX_EIGEN_REFINE={enabled}: refined FP32 eigensolver' in std, label
    assert re.search(r'<Band_DFT_Col> GPU device \d+: 1 k-owner rank\(s\)', std), label
    assert re.findall(r'BANDPROF id=0 it=(\d+) ', std) == ['1', '2', '3'], label
    assert not re.search(r'<Band_DFT_Col>.*(?:refinement step|k-point slot|this rank.s k-point)', std), label
    assert not re.search(r'<eigen_refine_gpu> FP32 .*solve', std), label
    outputs.append(read_output(cases / label / 'Crys-MnO.out'))
a, b = outputs
assert a[1] == b[1] and a[3] == b[3]
de = abs(a[0] - b[0])
df = max(abs(x-y) for x, y in zip(a[2], b[2]))
assert de <= 1e-10 and df <= 1e-6, (de, df)
print(f'PASS single-rank spin-polarized GPU band: FP64 fallback, 3 SCF steps, dE={de:.3g}, max dF={df:.3g}')
PY
