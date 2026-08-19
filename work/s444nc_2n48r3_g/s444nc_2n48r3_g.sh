#!/bin/bash
#------- qsub option -----------
# General use runs in gen_S/M/L under the GPU2026 allocation; the gen_* split is
# by node count and gen_S covers 1-31 nodes.
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 2
# Per *logical host*: 48 cores and 1 GPU.  The queue's GPU default is 0, so the
# H100 must be asked for explicitly.
#PBS --cpunum-lhost=48
#PBS --gpunum-lhost=1
#PBS -l elapstim_req=00:50:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N s444nc2n48r
#PBS -j o
#PBS -o s444nc_2n48r3_g.joblog

# sidia444_nc_cluster node-scaling rerun, GPU PATH -- one of 24 requests
# (2/3/4/6 nodes x {gpusolver+MPS, elpa2} x 3 repetitions) so that Total and
# Diagonalization come with a mean instead of a single sample.  Rerun with the
# host-memory-reduced binary (commits dd40d46 + 28ae7f8).
#
# Input: sidia444_nc_cluster.dat with only System.Name changed.  512 Si,
# s2p2d1, non-collinear -> matrix dimension 13312, complex Hermitian (2.84 GB
# per dense matrix), Cluster solver at the gamma point, 25 SCF iterations to
# 1e-13, energycutoff 200 Ry.
#
# Old-binary history, Max_Time: only 4n finished (266.9 s, host peak 97 GiB).
# 2n died at mpirun's old --timeout 600 (rc=143, peak 96 GiB), 3n hit the host
# OOM killer (rc=137, peak 119 GiB), 1n OOMed too.  This rerun uses
# --timeout 2400, so whether 2n/3n now complete IS one of the measurements --
# a crash here is a result, not a failure of the harness.
# Utot reference: -2103.477858563170 (4n GPU); ~12-digit agreement with the
# elpa2 value -2103.47785856316x is expected.

set -u

# Nothing here is debugged from a core dump, and a multi-rank abort fills the
# directory with them -- an earlier one left 21 files, 347 MB.
ulimit -c 0

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
NNODE=2
PPN=48
NPROC=$((NNODE * PPN))
CASE=s444nc_2n48r3_g
STDOUT=${CASE}.std
STDERR=${CASE}.err
ENVLOG=${CASE}.env
SMILOG=${CASE}.smi
PROGLOG=${CASE}.progress

# Stage markers, flushed immediately and readable from the login node while the
# job is still running.  The NQSV joblog only materialises at job end.
progress() { echo "[$(date '+%H:%M:%S')] $*" >> "$PROGLOG"; }

cd "${PBS_O_WORKDIR}" || exit 1
# Truncated, not appended: the sampler writes with >>, so without this a rerun in
# the same directory would report a "peak" taken from the previous run.
: > "$PROGLOG"
: > "$SMILOG"
progress "script start on $(hostname)"

# openmx carries a DT_RPATH into the NVHPC 26.5 tree (CUDA 13.2, HPC-X 2.50 /
# Open MPI 5), so no module is needed at run time.  Keep the environment clean.
command -v module >/dev/null 2>&1 && module purge >/dev/null 2>&1
unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"

#----------------------------------------------------------------------
# NVIDIA MPS -- on every node, not just this one
#----------------------------------------------------------------------
# The batch script runs only on the first node, but MPS is a per-node service.
# /tmp is node-local, so this one path names a different directory on each node.
JOBTAG=$(echo "${PBS_JOBID:-$$}" | tr -c 'A-Za-z0-9._-' '_')
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-${USER}-${JOBTAG}/pipe"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-${USER}-${JOBTAG}/log"

MPS_X="-x CUDA_MPS_PIPE_DIRECTORY -x CUDA_MPS_LOG_DIRECTORY"
# shellcheck disable=SC2086
mps_on_each_node() {
  mpirun ${NQSV_MPIOPTS:-} -np "${NNODE}" -npernode 1 ${MPS_X} ./mps_node.sh "$1"
}
mps_stop_all() { mps_on_each_node stop >/dev/null 2>&1; }
# Covers normal exit, qdel and the elapstim kill.  The sampler subshell below
# clears this trap for itself -- a background subshell inherits the parent's
# traps and would otherwise handle SIGTERM and then resume its loop.
trap mps_stop_all EXIT INT TERM

{
  echo "=== host      : $(hostname)"
  echo "=== date      : $(date)"
  echo "=== jobid     : ${PBS_JOBID:-unset}"
  echo "=== workdir   : $(pwd)"
  echo "=== layout    : ${NNODE} nodes x ${PPN} ranks = ${NPROC} ranks, 1 H100/node, MPS on"
  echo "=== mpirun    : $(command -v mpirun)"
  echo "=== NQSV_MPIOPTS = [${NQSV_MPIOPTS:-}]"
  echo "=== nodes ==="
  mpirun ${NQSV_MPIOPTS:-} -np "${NNODE}" -npernode 1 hostname
  echo "=== eigen library requested by the input ==="
  grep -E "scf.eigen.lib" "${CASE}.dat"
  echo "=== host memory before the run ==="
  free -g
  echo "=== GPU ==="
  nvidia-smi -L 2>&1 || true
  echo "=== openmx runtime libraries ==="
  ldd ../openmx | grep -E "libcud|libcublas|libcusolver|libmpi\."
} > "$ENVLOG" 2>&1

if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "FATAL: nvidia-cuda-mps-control not found on $(hostname); aborting." | tee -a "$ENVLOG"
  exit 1
fi

mps_on_each_node start >> "$ENVLOG" 2>&1

{ echo "=== MPS per node ==="
  echo "pipe dir = ${CUDA_MPS_PIPE_DIRECTORY}"
  mps_on_each_node check
} > mps_check.tmp 2>&1
cat mps_check.tmp >> "$ENVLOG"

# A node that silently fell back to the default time-sliced mode would mislabel
# the result -- and here it would quietly poison one sample of a three-run mean.
# Abort instead of averaging a run that was not the configuration under test.
if [ "$(grep -c 'control_pipe=yes' mps_check.tmp)" -ne "${NNODE}" ]; then
  echo "FATAL: expected ${NNODE} nodes with MPS, got:" | tee -a "$ENVLOG"
  cat mps_check.tmp | tee -a "$ENVLOG"
  rm -f mps_check.tmp
  exit 1
fi
rm -f mps_check.tmp
progress "MPS up on all ${NNODE} nodes"

# Sampler for THIS node only, every 5 s: these runs are short (~80-260 s) and a
# possible 1-node host OOM dies within the first two minutes, so 10 s sampling
# would miss the climb.  "trap -" so a kill actually kills it.
( trap - EXIT INT TERM
  for _ in $(seq 1 3000); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " $3 "/" $2}'
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    } >> "$SMILOG" 2>&1
    sleep 5
  done ) &
SMIPID=$!

#----------------------------------------------------------------------
# run
#----------------------------------------------------------------------
progress "launching mpirun: ${NPROC} ranks on ${NNODE} nodes"
mpirun ${NQSV_MPIOPTS:-} -np "${NPROC}" -npernode "${PPN}" --timeout 2400 \
       ${MPS_X} \
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
pkill -KILL -P "$SMIPID" 2>/dev/null
kill -KILL "$SMIPID" 2>/dev/null
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
# what the phase actually cost in wall clock.  Scoped to the block so the
# per-iteration "DFT in total =" lines cannot be picked up by mistake.
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
echo "=== Utot (reference: -2103.477858563170; elpa2 gives ...316x) ==="
grep -E "Utot\." "${CASE}.out" 2>/dev/null | tail -2
echo "=== SCF iterations completed (of 25 max) ==="
grep -cE "SCF=" "$STDOUT" 2>/dev/null
echo "=== peak host memory, this node (GiB); old peaks: 2n 96, 3n OOM at 119, 4n 97 ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$SMILOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== peak GPU, this node ==="
grep -oE "^[0-9]+ %, [0-9]+ MiB" "$SMILOG" 2>/dev/null | sort -t, -k2 -rn | head -3
echo "=== MPS server per node (post-run) ==="
mps_on_each_node report 2>&1
echo "=== teardown hang? (watchdog line present => yes) ==="
grep -c "watchdog:" "$STDERR" 2>/dev/null
echo "=== progress markers ==="
cat "$PROGLOG" 2>/dev/null

exit ${rc}
