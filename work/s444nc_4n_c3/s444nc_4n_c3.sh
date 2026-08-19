#!/bin/bash
#------- qsub option -----------
# General use runs in gen_S/M/L under the GPU2026 allocation; the gen_* split is
# by node count and gen_S covers 1-31 nodes.
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 4
#PBS --cpunum-lhost=48
# No --gpunum-lhost.  elpa2 gates every device path off before any CUDA call
# happens, so asking for H100s here would reserve GPUs that stay idle.
#PBS -l elapstim_req=00:50:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N s444nc_4c3
#PBS -j o
#PBS -o s444nc_4n_c3.joblog

# sidia444_nc_cluster node-scaling rerun, CPU PATH -- one of 24 requests
# (2/3/4/6 nodes x {gpusolver+MPS, elpa2} x 3 repetitions) so that Total and
# Diagonalization come with a mean instead of a single sample.  Rerun with the
# host-memory-reduced binary (commits dd40d46 + 28ae7f8).
#
# Input: sidia444_nc_cluster.dat with System.Name changed and
# "scf.eigen.lib gpusolver" -> "elpa2".  512 Si, s2p2d1, non-collinear ->
# matrix dimension 13312, complex Hermitian, Cluster solver at the gamma
# point, 25 SCF iterations to 1e-13, energycutoff 200 Ry.
#
# Single-sample references (old binary, Max_Time): 2n 1215.0 s / 3n 965.7 s /
# 4n 887.7 s; 6n has never been run.  2n needs ~21 min, hence the 50 min
# walltime and mpirun --timeout 2400.
# Utot reference: -2103.47785856316x (last digits move with the rank count).

set -u

# Nothing here is debugged from a core dump, and a multi-rank abort fills the
# directory with them.
ulimit -c 0

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
NNODE=4
PPN=48
NPROC=$((NNODE * PPN))
CASE=s444nc_4n_c3
STDOUT=${CASE}.std
STDERR=${CASE}.err
ENVLOG=${CASE}.env
MEMLOG=${CASE}.mem
PROGLOG=${CASE}.progress

# Stage markers, flushed immediately and readable from the login node while the
# job is still running.  The NQSV joblog only materialises at job end.
progress() { echo "[$(date '+%H:%M:%S')] $*" >> "$PROGLOG"; }

cd "${PBS_O_WORKDIR}" || exit 1
# Truncated, not appended: the sampler writes with >>, so without this a rerun in
# the same directory would report a "peak" taken from the previous run.
: > "$PROGLOG"
: > "$MEMLOG"
progress "script start on $(hostname)"

# openmx carries a DT_RPATH into the NVHPC 26.5 tree (HPC-X 2.50 / Open MPI 5),
# so no module is needed at run time.  Keep the environment clean.
command -v module >/dev/null 2>&1 && module purge >/dev/null 2>&1
unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"

# No MPS.  Cleared so a stale value cannot point anything at another job's daemon.
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

{
  echo "=== host      : $(hostname)"
  echo "=== date      : $(date)"
  echo "=== jobid     : ${PBS_JOBID:-unset}"
  echo "=== workdir   : $(pwd)"
  echo "=== layout    : ${NNODE} nodes x ${PPN} ranks = ${NPROC} ranks, CPU only (elpa2), no MPS"
  echo "=== mpirun    : $(command -v mpirun)"
  echo "=== NQSV_MPIOPTS = [${NQSV_MPIOPTS:-}]"
  echo "=== nodes ==="
  mpirun ${NQSV_MPIOPTS:-} -np "${NNODE}" -npernode 1 hostname
  echo "=== eigen library requested by the input ==="
  grep -E "scf.eigen.lib|scf.EigenvalueSolver" "${CASE}.dat"
  echo "=== host memory before the run ==="
  free -g
  echo "=== GPUs visible (expected: none assigned) ==="
  nvidia-smi -L 2>&1 || true
  echo "=== openmx runtime libraries ==="
  ldd ../openmx | grep -E "libcud|libcublas|libcusolver|libmpi\."
} > "$ENVLOG" 2>&1

# Host memory only, every 5 s -- there is no GPU to sample and the runs are
# short.  "trap -" so a kill actually kills it.
( trap - EXIT INT TERM
  for _ in $(seq 1 3000); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " $3 "/" $2}'
    } >> "$MEMLOG" 2>&1
    sleep 5
  done ) &
MEMPID=$!

#----------------------------------------------------------------------
# run
#----------------------------------------------------------------------
progress "launching mpirun: ${NPROC} ranks on ${NNODE} nodes"
mpirun ${NQSV_MPIOPTS:-} -np "${NPROC}" -npernode "${PPN}" --timeout 2400 \
       ../openmx "${CASE}.dat" -nt 1 > "$STDOUT" 2> "$STDERR" &
MPIPID=$!
progress "mpirun launched (pid ${MPIPID})"

# Foreground watchdog.  DONE_MARK: science complete, anything after is teardown.
# FAIL_RE: a dead rank means the run is over; do not wait out --timeout.
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
      if kill -0 "$MPIPID" 2>/dev/null; then
        progress "still up -> SIGKILL"
        kill -KILL "$MPIPID" 2>/dev/null
      fi
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

# No "wait" on the sampler: a wait here is itself a way to block forever.
pkill -KILL -P "$MEMPID" 2>/dev/null
kill -KILL "$MEMPID" 2>/dev/null
progress "sampler killed"

# A watchdog kill shows up as a signal exit code, but the run did finish -- the
# banner is the real verdict.
if [ "$rc" -ne 0 ] && grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
  echo "=== mpirun returned ${rc}, but openmx printed the completion banner;" >> "$STDOUT"
  echo "=== treating as success (teardown was killed by the watchdog). ===" >> "$STDOUT"
  rc=0
fi

echo "=== openmx exit status = ${rc} ($(date)) ===" >> "$STDOUT"

# One machine-readable line for the aggregator.  Max_Time ($NF) = slowest rank =
# what the phase actually cost in wall clock.
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
echo "=== Utot (reference: -2103.47785856316x) ==="
grep -E "Utot\." "${CASE}.out" 2>/dev/null | tail -2
echo "=== SCF iterations completed (of 25 max) ==="
grep -cE "SCF=" "$STDOUT" 2>/dev/null
echo "=== peak host memory, this node (GiB); old single-sample 2n peak 90 ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$MEMLOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== teardown hang? (watchdog line present => yes) ==="
grep -c "watchdog:" "$STDERR" 2>/dev/null
echo "=== progress markers ==="
cat "$PROGLOG" 2>/dev/null

exit ${rc}
