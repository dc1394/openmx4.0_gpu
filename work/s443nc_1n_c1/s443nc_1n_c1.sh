#!/bin/bash
#------- qsub option -----------
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 1
#PBS --cpunum-lhost=48
# No --gpunum-lhost.  elpa2 gates every device path off before any CUDA call
# happens, so asking for H100s here would reserve GPUs that stay idle.
#PBS -l elapstim_req=01:30:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N s443_1c1
#PBS -j o
#PBS -o s443nc_1n_c1.joblog

# sidia443_nc_cluster campaign, CPU path (elpa2) -- one of 36 requests
# ({1,2,3,4} nodes x {gpusolver+gemmul8-on, gpusolver+gemmul8-off, elpa2} x 3
# reps, all -npernode 36).  Binary: round-3 (commit e227507).  384 Si,
# s2p2d1, non-collinear -> matrix dimension 9984, Cluster solver, 25 SCF.
# Scale reference: sidia444_nc (dim 13312) CPU 2n x 48 took ~1210 s.
# A crash here is a result, not a failure of the harness.

set -u
ulimit -c 0

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
NNODE=1
PPN=36
NPROC=$((NNODE * PPN))
CASE=s443nc_1n_c1
STDOUT=${CASE}.std
STDERR=${CASE}.err
ENVLOG=${CASE}.env
MEMLOG=${CASE}.mem
PROGLOG=${CASE}.progress

progress() { echo "[$(date '+%H:%M:%S')] $*" >> "$PROGLOG"; }

cd "${PBS_O_WORKDIR}" || exit 1
: > "$PROGLOG"
: > "$MEMLOG"
progress "script start on $(hostname)"

command -v module >/dev/null 2>&1 && module purge >/dev/null 2>&1
unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"

# No MPS.  Cleared so a stale value cannot point anything at another job's daemon.
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

{
  echo "=== host      : $(hostname)"
  echo "=== date      : $(date)"
  echo "=== jobid     : ${PBS_JOBID:-unset}"
  echo "=== layout    : ${NNODE} nodes x ${PPN} ranks = ${NPROC} ranks, CPU only (elpa2), no MPS"
  echo "=== nodes ==="
  mpirun ${NQSV_MPIOPTS:-} -np "${NNODE}" -npernode 1 hostname
  echo "=== eigen library requested by the input ==="
  grep -E "scf.eigen.lib|scf.EigenvalueSolver" "${CASE}.dat"
  echo "=== host memory before the run ==="
  free -g
} > "$ENVLOG" 2>&1

( trap - EXIT INT TERM
  for _ in $(seq 1 3000); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " $3 "/" $2}'
    } >> "$MEMLOG" 2>&1
    sleep 5
  done ) &
MEMPID=$!

progress "launching mpirun: ${NPROC} ranks on ${NNODE} nodes"
mpirun ${NQSV_MPIOPTS:-} -np "${NPROC}" -npernode "${PPN}" --timeout 4800 \
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
    if kill -0 "$MPIPID" 2>/dev/null; then
      progress "mpirun still up after grace -> SIGTERM"
      echo "watchdog: ${verdict} detected, mpirun still up after ${grace} s; terminating" >> "$STDERR"
      kill -TERM "$MPIPID" 2>/dev/null
      sleep 20
      kill -0 "$MPIPID" 2>/dev/null && { progress "still up -> SIGKILL"; kill -KILL "$MPIPID" 2>/dev/null; }
    else
      progress "mpirun exited on its own during grace"
    fi
    break
  fi
  sleep 5
done
progress "watchdog loop exited (verdict='${verdict:-none}'); reaping mpirun"

wait "$MPIPID"
rc=$?
progress "mpirun reaped rc=${rc}"

pkill -KILL -P "$MEMPID" 2>/dev/null
kill -KILL "$MEMPID" 2>/dev/null
progress "sampler killed"

if [ "$rc" -ne 0 ] && grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
  echo "=== mpirun returned ${rc}, but openmx printed the completion banner;" >> "$STDOUT"
  echo "=== treating as success (teardown was killed by the watchdog). ===" >> "$STDOUT"
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
echo "=== Utot (GPU probes gave -1577.61926591245x; elpa2 should agree to ~12 digits) ==="
grep -E "Utot\." "${CASE}.out" 2>/dev/null | tail -2
echo "=== SCF iterations completed (of 25 max) ==="
grep -cE "SCF=" "$STDOUT" 2>/dev/null
echo "=== peak host memory, this node (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$MEMLOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== teardown hang? (watchdog line present => yes) ==="
grep -c "watchdog:" "$STDERR" 2>/dev/null
echo "=== progress markers ==="
cat "$PROGLOG" 2>/dev/null

exit ${rc}
