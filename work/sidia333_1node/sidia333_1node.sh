#!/bin/bash
#------- qsub option -----------
# General use runs in gen_S/M/L under the GPU2026 allocation; the gpu queue's ACL
# only lists CP24I024, which has a 0.00 budget.  gen_S covers 1-31 nodes.
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 1
#PBS --cpunum-lhost=48
# The queue's GPU default is 0, so the H100 must be asked for explicitly.
#PBS --gpunum-lhost=1
#PBS -l elapstim_req=01:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N sidia333_1node
#PBS -j o
#PBS -o sidia333_1node.joblog

# OpenMX 4.0 GPU: 216 Si, s2p2d1 -> matrix dimension 2808, collinear, Band
# solver, 2x2x2 k-grid, 25 SCF iterations.  1 Pegasus node(s), 48 ranks
# each = 48 ranks.  gpusolver with NVIDIA MPS enabled on every node.
#
# The 1-node point of the collinear node-scaling series.  ../sidia333 already
# measured this configuration with an older script -- Utot = -887.422830856492,
# total 79.7 s, DFT 68.0 s, Diagonalization 40.4 s -- and it is redone here so
# every point of the series comes from one script and one binary, the way
# ../sidia333_nc_1node was.
# Utot must not move with node count -- only the timings may.

set -u

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
NNODE=1
PPN=48
NPROC=$((NNODE * PPN))
CASE=sidia333_1node
STDOUT=${CASE}.std
STDERR=${CASE}.err
ENVLOG=${CASE}.env
MEMLOG=${CASE}.smi
PROGLOG=${CASE}.progress

# Stage markers, flushed immediately and readable from the login node while the
# job is still running.  The NQSV joblog only materialises at job end, so when a
# request sits in RUN after its calculation has finished there is otherwise no
# way to see which line the script is stuck on.
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

#----------------------------------------------------------------------
# NVIDIA MPS -- on every node, not just this one
#----------------------------------------------------------------------
# The batch script runs only on the first node, but MPS is a per-node service:
# each node needs its own control daemon and its own node-local pipe directory.
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
# Covers normal exit, qdel and the elapstim kill, so no daemon is left behind on
# any node.  The sampler subshell below clears this trap for itself.
trap mps_stop_all EXIT INT TERM

{
  echo "=== host      : $(hostname)"
  echo "=== date      : $(date)"
  echo "=== jobid     : ${PBS_JOBID:-unset}"
  echo "=== workdir   : $(pwd)"
  echo "=== layout    : ${NNODE} node(s) x ${PPN} ranks = ${NPROC} ranks, gpusolver + MPS"
  echo "=== mpirun    : $(command -v mpirun)"
  echo "=== NQSV_MPIOPTS = [${NQSV_MPIOPTS:-}]"
  echo "=== nodes ==="
  mpirun ${NQSV_MPIOPTS:-} -np "${NNODE}" -npernode 1 hostname
  echo "=== eigen library requested by the input ==="
  grep -E "scf.eigen.lib" "${CASE}.dat"
  echo "=== host memory (this node) ==="
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
# the result, so require MPS on every node rather than most of them.
if [ "$(grep -c 'control_pipe=yes' mps_check.tmp)" -ne "${NNODE}" ]; then
  echo "FATAL: expected ${NNODE} node(s) with MPS, got:" | tee -a "$ENVLOG"
  cat mps_check.tmp | tee -a "$ENVLOG"
  rm -f mps_check.tmp
  exit 1
fi
rm -f mps_check.tmp
progress "MPS up on all ${NNODE} node(s)"

# Sampler for THIS node only.  "trap -" so a kill actually kills it: a background
# subshell inherits the parent's traps and would otherwise run the handler and
# resume its loop, which previously kept finished jobs in RUN for 24 minutes.
( trap - EXIT INT TERM
  for _ in $(seq 1 3000); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " $3 "/" $2}'
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
      nvidia-smi --query-compute-apps=process_name --format=csv,noheader | sort | uniq -c | tr '\n' ';'
      echo
    } >> "$MEMLOG" 2>&1
    sleep 10
  done ) &
MEMPID=$!

#----------------------------------------------------------------------
# run
#----------------------------------------------------------------------
progress "launching mpirun: ${NPROC} ranks"
mpirun ${NQSV_MPIOPTS:-} -np "${NPROC}" -npernode "${PPN}" --timeout 3000 \
       ${MPS_X} \
       ../openmx "${CASE}.dat" -nt 1 > "$STDOUT" 2> "$STDERR" &
MPIPID=$!
progress "mpirun launched (pid ${MPIPID})"

# Foreground watchdog, deliberately not a background subshell.  OpenMX prints
# this banner immediately before MPI_Finalize, so once it appears the science is
# complete and on disk; on earlier runs mpirun then failed to return for 13
# minutes, leaving the request in RUN.  Grace period, then tear it down.
DONE_MARK="The calculation was normally finished."
while kill -0 "$MPIPID" 2>/dev/null; do
  if grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
    progress "completion banner seen; 60 s grace"
    sleep 60
    if kill -0 "$MPIPID" 2>/dev/null; then
      progress "mpirun still up after grace -> SIGTERM"
      echo "watchdog: banner seen, mpirun still up after 60 s; terminating" >> "$STDERR"
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
progress "watchdog loop exited; reaping mpirun"

wait "$MPIPID"
rc=$?
progress "mpirun reaped rc=${rc}"

# No "wait" on the sampler: it is a pure monitoring loop with nothing to flush,
# and a wait here is itself a way for the script to block forever if the kill
# does not land.
pkill -KILL -P "$MEMPID" 2>/dev/null
kill -KILL "$MEMPID" 2>/dev/null
progress "sampler killed"

# A watchdog kill shows up as a signal exit code, but the run did finish -- treat
# the banner as the real verdict.
if [ "$rc" -ne 0 ] && grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
  echo "=== mpirun returned ${rc}, but openmx printed the completion banner;" >> "$STDOUT"
  echo "=== treating as success (teardown was killed by the watchdog). ===" >> "$STDOUT"
  rc=0
fi

echo "=== openmx exit status = ${rc} ($(date)) ===" >> "$STDOUT"

echo "=== teardown hang? (watchdog line present => yes) ==="
grep -c "watchdog:" "$STDERR" 2>/dev/null
echo "=== MPS server per node (post-run) ==="
mps_on_each_node report 2>&1
echo "=== peak host memory, this node (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$MEMLOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== peak GPU, this node ==="
grep -oE "^[0-9]+ %, [0-9]+ MiB" "$MEMLOG" 2>/dev/null | sort -t, -k2 -rn | head -3
echo "=== SCF iterations completed (of 25 max) ==="
grep -cE "SCF=" "$STDOUT" 2>/dev/null
echo "=== Utot (../sidia333 with the older script: -887.422830856492) ==="
grep -E "Utot\." "${CASE}.out" 2>/dev/null | tail -2
echo "=== timing (../sidia333 older script: total 79.7 / DFT 68.0 / Diag 40.4 s) ==="
grep -E "Set_Hamiltonian |Diagonalization|^   DFT |Total Computational" "$STDOUT" 2>/dev/null | tail -4
echo "=== progress markers ==="
cat "$PROGLOG" 2>/dev/null

exit ${rc}
