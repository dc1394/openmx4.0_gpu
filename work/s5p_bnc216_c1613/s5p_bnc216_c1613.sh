#!/bin/bash
# RTX 5080 local benchmark (rtx5080_procedure.md; plan v2.6 sec. 7.2/8.2/8.3): s5p_bnc216_c1613.
# 216 Si atoms from sidia.dat supercells, band solver, Kgrid 2x2x2 (8 computed k).
# Config: CPU elpa2, flat MPI 16 ranks, no GPU (reference per plan 8.2).
# Fixed 25 SCF, MD.Type Opt / MD.maxIter 1; wall time = Max_Time column
# (slowest rank).  All 5080 series: -np 16 -nt 1 on Core i9-10980XE,
# 1x RTX 5080 16 GB (sm_120), binary = v2.0_thesis tag build.
# A crash here is a result, not a harness failure.
set -u
ulimit -c 0

NVROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
NPROC=16
CASE=s5p_bnc216_c1613
STDOUT=${CASE}.std
STDERR=${CASE}.err
ENVLOG=${CASE}.env
SMILOG=${CASE}.smi
PROGLOG=${CASE}.progress

progress() { echo "[$(date '+%H:%M:%S')] $*" >> "$PROGLOG"; }

cd "$(dirname "$0")" || exit 1
: > "$PROGLOG"
: > "$SMILOG"
progress "script start on $(hostname)"

unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"

{
  echo "=== host      : $(hostname)"
  echo "=== date      : $(date)"
  echo "=== layout    : 1 node x ${NPROC} ranks, CPU elpa2 (no GPU)"
  echo "=== eigen library / gemmul8 requested by the input ==="
  grep -E "scf.eigen.lib|scf.gemmul8" "${CASE}.dat" || true
  echo "=== host memory before the run ==="
  free -g
} > "$ENVLOG" 2>&1

( for _ in $(seq 1 3000); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " $3 "/" $2}'
    } >> "$SMILOG" 2>&1
    sleep 5
  done ) &
SMIPID=$!

progress "launching mpirun: ${NPROC} ranks"
mpirun -np "${NPROC}" --timeout 14400 \
       ../openmx "${CASE}.dat" -nt 1 > "$STDOUT" 2> "$STDERR" &
MPIPID=$!
progress "mpirun launched (pid ${MPIPID})"

DONE_MARK="The calculation was normally finished."
FAIL_RE="exited on signal|prterun noticed|Out Of Memory|oom-kill"
verdict=""
while kill -0 "$MPIPID" 2>/dev/null; do
  if grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
    verdict="done"; grace=60
  elif grep -qEs "$FAIL_RE" "$STDERR" "$STDOUT"; then
    verdict="crash"; grace=120
  fi
  if [ -n "$verdict" ]; then
    progress "${verdict} detected; ${grace} s grace for teardown"
    sleep "$grace"
    kill -0 "$MPIPID" 2>/dev/null && { kill -TERM "$MPIPID" 2>/dev/null; sleep 20; }
    kill -0 "$MPIPID" 2>/dev/null && kill -KILL "$MPIPID" 2>/dev/null
    break
  fi
  sleep 5
done

wait "$MPIPID"
rc=$?
progress "mpirun reaped rc=${rc}"
pkill -KILL -P "$SMIPID" 2>/dev/null
kill -KILL "$SMIPID" 2>/dev/null

if [ "$rc" -ne 0 ] && grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
  rc=0
fi
echo "=== openmx exit status = ${rc} ($(date)) ===" >> "$STDOUT"

echo "=== TIMING (machine-readable; Max_Time column, seconds) ==="
awk -v c="${CASE}" '
  /Computational Time \(second\)/ {inblk=1}
  inblk && /^ *Total Computational Time *=/ {tot=$NF}
  inblk && /^ *DFT *=/                      {dft=$NF}
  inblk && /^ *Diagonalization *=/          {diag=$NF}
  inblk && /^ *Set_Hamiltonian *=/          {sh=$NF}
  END {
    if (tot=="") { printf "TIMING %s INCOMPLETE\n", c }
    else { printf "TIMING %s total=%s dft=%s diag=%s set_hamiltonian=%s\n", c, tot, dft, diag, sh }
  }' "$STDOUT"

echo "=== full timing block ==="
sed -n '/Computational Time (second)/,/^The calculation/p' "$STDOUT" 2>/dev/null | head -35
echo "=== Utot ==="
grep -E "Utot\." "${CASE}.out" 2>/dev/null | tail -2
echo "=== SCF iterations completed ==="
grep -cE "SCF=" "$STDOUT" 2>/dev/null
echo "=== peak host memory, this node (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$SMILOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== progress markers ==="
cat "$PROGLOG" 2>/dev/null

exit ${rc}
