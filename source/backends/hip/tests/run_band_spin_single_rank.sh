#!/usr/bin/env bash
# One-rank spin-polarized band regression (AMD port of the NVIDIA test of
# the same name): a single rank solves both spins of every k-point in
# sequence (Band_DFT_Col's "goto diagonalize1"), so the refined FP32
# eigensolver must keep the FP64 GPU path there.  Crys-MnO with a 2x1x1
# k-mesh (a 1x1x1 mesh turns into the cluster solver) and scf.maxIter 3 runs
# twice on one rank, OPENMX_EIGEN_REFINE=0 (R0) and =2 (R2), with the dense
# GPU path forced for its small matrices (OPENMX_BAND_GPU_SWITCH_NUM=1,
# OPENMX_EIGEN_REFINE_MIN_N=0); then R2 must show no refined solve, the
# dense GPU solves must have run (OPENMX_SOLVE_TIMING lines), and the two
# runs must agree: |dUtot| <= 1e-10 Ha, max |dF| <= 1e-6 Ha/bohr.
# Usage: bash tests/run_band_spin_single_rank.sh [binary] [new-output-dir]
# OPENMX_TEST_MPIRUN overrides the MPI launcher (default: the Open MPI of
# this machine's build, else mpirun).  The output directory must not exist.
set -euo pipefail
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
binary=$(realpath "${1:-$repo/source/openmx}")
out=$(realpath -m "${2:-$repo/work/band_spin_single_rank_regression}")
mpirun=${OPENMX_TEST_MPIRUN:-}
if [[ -z "$mpirun" ]]; then
  if [[ -x /home/kawai/opt/openmpi-rocmclang/bin/mpirun ]]; then
    mpirun=/home/kawai/opt/openmpi-rocmclang/bin/mpirun
  else
    mpirun=mpirun
  fi
fi
if [[ -e "$out" ]]; then
  echo "Refusing to mix results with existing directory: $out" >&2
  exit 2
fi
[[ -x "$binary" ]] || { echo "No OpenMX binary: $binary" >&2; exit 2; }

python3 - "$repo" "$out" <<'PY'
from pathlib import Path
import re
import sys
repo, out = map(Path, sys.argv[1:])
text = (repo / 'work/input_example/Crys-MnO.dat').read_text()
for key, value in [('scf.Kgrid', '2 1 1'), ('scf.maxIter', '3'),
                   ('DATA.PATH', str(repo / 'DFT_DATA19'))]:
    pattern = rf'(?im)^\s*{re.escape(key)}\s+.*$'
    if re.search(pattern, text):
        text = re.sub(pattern, key + ' ' + value, text)
    else:
        text += '\n' + key + ' ' + value + '\n'
for label in ('R0', 'R2'):
    (out / label).mkdir(parents=True)
    (out / label / 'Crys-MnO.dat').write_text(text)
PY

for variant in R0:0 R2:2; do
  label=${variant%%:*}
  refine=${variant##*:}
  ( cd "$out/$label" &&
    env OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        OPENMX_EIGEN_REFINE=$refine OPENMX_EIGEN_REFINE_MIN_N=0 \
        OPENMX_BAND_GPU_SWITCH_NUM=1 OPENMX_BAND_GPU_DIAG=1 OPENMX_SOLVE_TIMING=1 \
        "$mpirun" -np 1 "$binary" Crys-MnO.dat -nt 1 > Crys-MnO.std 2>&1 ) ||
    { echo "OpenMX failed for $label (see $out/$label/Crys-MnO.std)" >&2; exit 1; }
done

python3 - "$repo" "$out" <<'PY'
from pathlib import Path
import re
import sys
repo, out = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo / 'tools'))
from compare_runtest_l3 import read_output
outputs = []
for label, enabled in [('R0', 0), ('R2', 2)]:
    std = (out / label / 'Crys-MnO.std').read_text()
    assert 'The calculation was normally finished.' in std, label
    assert not re.search(r'global matrix dimension \d+ is below', std), label
    assert f'OPENMX_EIGEN_REFINE={enabled}: refined FP32 eigensolver' in std, label
    assert re.findall(r'MD= 1  SCF=\s*(\d+)', std) == ['1', '2', '3'], label
    # the dense GPU eigensolver ran (FP64 solves of both spins)
    assert 'SOLVE_TIMING' in std, label
    # and no refined solve on any path: resident halves, two-pass slots,
    # staged turns, nor the FP32 eigensolver itself
    assert not re.search(r'<Band_DFT_Col>.*(?:refinement step|k-point slot|this rank.s k-point|'
                         r'serialized GPU turns)', std), label
    assert not re.search(r'<eigen_refine_gpu> FP32 .*solve', std), label
    outputs.append(read_output(out / label / 'Crys-MnO.out'))
a, b = outputs
assert a[1] == b[1] and a[3] == b[3], 'atoms or grids differ'
de = abs(a[0] - b[0])
df = max(abs(x - y) for x, y in zip(a[2], b[2]))
assert de <= 1e-10 and df <= 1e-6, (de, df)
print(f'PASS single-rank spin-polarized GPU band: FP64 kept, 3 SCF steps, dE={de:.3g}, max dF={df:.3g}')
PY
