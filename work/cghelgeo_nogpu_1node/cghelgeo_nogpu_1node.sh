#!/bin/bash
#------- qsub option -----------
# General use runs in gen_S/M/L under the GPU2026 allocation; the gen_* split is
# by node count and gen_S covers 1-31, so one node belongs here.
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 1
#PBS --cpunum-lhost=48
# No --gpunum-lhost.  elpa2 gates every device path off before any CUDA call, so
# asking for an H100 here would reserve one that stays idle.  See the 2-node
# sibling ../cghelgeo_nogpu/cghelgeo_nogpu.sh for where that gating lives.
#PBS -l elapstim_req=03:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N cghel1n_cpu
#PBS -j o
#PBS -o cghelgeo_nogpu_1node.joblog

# WHAT THIS RUN IS FOR
# --------------------
# It answers one question that had until now only been extrapolated: does
# cghelgeo fit on a SINGLE node at 48 ranks on the CPU path?
#
# Every cghelgeo run so far -- cghelgeo, cghelgeo_nogpu, cghelgeo_ppn16/24/36/48
# -- was 2 nodes.  The claim "1 node is impossible" came from a fit, not a
# measurement, so this run and its GPU sibling ../cghelgeo_1node close that gap.
#
# Input: ../cghelgeo/cghelgeo.dat with System.Name changed and
# "scf.eigen.lib gpusolver" -> "elpa2".  650 atoms (190 C, 240 H, 80 N, 120 O,
# 20 P), matrix dimension 10430, collinear, Band solver, k-grid 1x1x3, 70 SCF
# iterations to 1e-13.  One Pegasus node, 48 ranks.  No MPS, no GPU.
#
# WHAT IS EXPECTED, AND WHY IT IS ONLY AN EXPECTATION
# ---------------------------------------------------
# There is exactly ONE CPU-path measurement of this case: 2 nodes x 48 ranks
# peaked at 100 GiB of the node's 124 GiB.  A single point cannot be split into
# its rank-proportional and its divisible parts, so it cannot by itself predict
# one node.  Bounding it instead: with per-node peak P(n) = a*ppn + D/n, one node
# fits in 124 GiB only if a*48 >= 76 of the measured 100 GiB, i.e. only if at
# least three quarters of it is replicated per rank.  The GPU-path ppn scan on
# this same geometry puts the rank-proportional share at ~0.92 GiB/rank = ~44 GiB
# at ppn 48, well under that 76.  So OOM is likely -- but "likely" is the whole
# reason to run it.
#
# WALLTIME
# --------
# 3 h.  The same input on 96 CPU ranks took 2769 s; halving the ranks should land
# near 5500 s, and if it OOMs instead it dies in minutes.  Generous but not the
# 10 h the 2-node run used, which was set before any CPU timing existed.

set -u

# An OOM kill of 48 ranks is exactly the situation that fills the directory with
# core files -- a previous aborted run left 21 of them, 347 MB.  Nothing here is
# debugged from a core dump; the memory trace and the exit status are the data.
ulimit -c 0

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
NNODE=1
PPN=48
NPROC=$((NNODE * PPN))
CASE=cghelgeo_nogpu_1node
STDOUT=${CASE}.std
STDERR=${CASE}.err
ENVLOG=${CASE}.env
MEMLOG=${CASE}.mem
PROGLOG=${CASE}.progress

# Stage markers, flushed immediately and readable from the login node while the
# job is still running.  The NQSV joblog only materialises at job end, so a
# request sitting in RUN after its calculation has stopped is otherwise opaque.
progress() { echo "[$(date '+%H:%M:%S')] $*" >> "$PROGLOG"; }

cd "${PBS_O_WORKDIR}" || exit 1
# Truncated, not appended: the sampler writes with >>, so without this a rerun in
# the same directory would report a "peak" taken from the previous run.
: > "$PROGLOG"
: > "$MEMLOG"
progress "script start on $(hostname)"

# openmx carries a DT_RPATH into the NVHPC 26.5 tree (HPC-X 2.50 / Open MPI 5),
# so no module is needed at run time.  Keep the environment clean so nothing
# shadows it; only mpirun has to come from that same Open MPI.
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
  echo "=== layout    : ${NNODE} node x ${PPN} ranks = ${NPROC} ranks, CPU only (elpa2), no MPS"
  echo "=== mpirun    : $(command -v mpirun)"
  echo "=== NQSV_MPIOPTS = [${NQSV_MPIOPTS:-}]"
  echo "=== eigen library requested by the input ==="
  grep -E "scf.eigen.lib|scf.EigenvalueSolver" "${CASE}.dat"
  echo "=== host memory before the run ==="
  free -g
  echo "=== core dump limit (expect 0) ==="
  ulimit -c
  echo "=== openmx runtime libraries ==="
  ldd ../openmx | grep -E "libcud|libcublas|libcusolver|libmpi\."
} > "$ENVLOG" 2>&1

# 5 s, not the usual 10.  The memory curve IS the result here: a host OOM kills
# the run at its peak, and a coarse interval can miss that last sample entirely
# -- an earlier 30 s trace showed 33 GiB and then a dead job inside one gap.
# "trap -" so a kill actually kills it: a background subshell inherits the
# parent's traps and would otherwise handle the signal and resume its loop.
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
progress "launching mpirun: ${NPROC} ranks on ${NNODE} node"
mpirun ${NQSV_MPIOPTS:-} -np "${NPROC}" -npernode "${PPN}" --timeout 10200 \
       ../openmx "${CASE}.dat" -nt 1 > "$STDOUT" 2> "$STDERR" &
MPIPID=$!
progress "mpirun launched (pid ${MPIPID})"

# Foreground watchdog, deliberately not a background subshell (an earlier
# background version never fired in a real job although it worked standalone).
#
# Two exit conditions, because this run is expected to fail and a failure is the
# case the earlier watchdogs handled worst:
#
#   DONE_MARK  openmx prints it immediately before MPI_Finalize, so the science
#              is complete and on disk; anything after is teardown.
#   FAIL_MARK  prterun's "exited on signal" notice.  Once a rank is gone the run
#              is over, but prterun has repeatedly failed to return afterwards
#              and left the request idling to --timeout.  Do not wait for that.
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
  sleep 10
done
progress "watchdog loop exited (verdict='${verdict:-none}'); reaping mpirun"

wait "$MPIPID"
rc=$?
progress "mpirun reaped rc=${rc}"

# No "wait" on the sampler: it is a pure monitoring loop with nothing to flush,
# and a wait here is itself a way for the script to block forever.
pkill -KILL -P "$MEMPID" 2>/dev/null
kill -KILL "$MEMPID" 2>/dev/null
progress "sampler killed"

if [ "$rc" -ne 0 ] && grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
  echo "=== mpirun returned ${rc}, but openmx printed the completion banner;" >> "$STDOUT"
  echo "=== treating as success (teardown was killed by the watchdog). ===" >> "$STDOUT"
  rc=0
fi

echo "=== openmx exit status = ${rc} ($(date)) ===" >> "$STDOUT"

echo "=== VERDICT ==="
if grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
  echo "COMPLETED -- 1 node x 48 ranks fits on the CPU path after all."
else
  echo "DID NOT COMPLETE.  Distinguishing host OOM from anything else:"
  echo "  signal 9 / Killed lines (host OOM kill looks like this):"
  grep -hE "exited on signal|Killed|Out Of Memory|oom" "$STDERR" "$STDOUT" 2>/dev/null | sort -u | head -10
  echo "  core files (none expected, ulimit -c 0):"
  ls -1 core.* 2>/dev/null | wc -l
fi
echo "=== last host memory samples before the end (GiB) ==="
tail -12 "$MEMLOG" 2>/dev/null
echo "=== peak host memory (GiB); 2-node CPU reference was 100/124 per node ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$MEMLOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== SCF iterations completed (of 70 max) ==="
grep -cE "SCF=" "$STDOUT" 2>/dev/null
echo "=== Utot (2-node reference: -4154.835898598240) ==="
grep -E "Utot\." "${CASE}.out" 2>/dev/null | tail -2
echo "=== timing (2-node CPU at 96 ranks: total 2769.2 s) ==="
grep -E "Set_ProExpn_VNA |Set_Hamiltonian |Diagonalization|^   DFT |Total Computational" "$STDOUT" 2>/dev/null | tail -5
echo "=== progress markers ==="
cat "$PROGLOG" 2>/dev/null
echo "=== tail of openmx stderr ==="
tail -25 "$STDERR" 2>/dev/null

exit ${rc}
