#!/bin/bash
#------- qsub option -----------
# General use runs in gen_S/M/L under the GPU2026 allocation; the gpu queue's ACL
# only lists CP24I024, which has a 0.00 budget.  gen_S is still the right queue
# at 2 nodes -- the gen_* split is by node count: gen_S 1-31, gen_M 32-63,
# gen_L 64-150.
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 2
# Per *logical host*: 48 cores and 1 GPU, so 2 nodes -> 96 ranks and 2 H100s.
# The GPU default is 0 ("GPU Number ... Std: 0"), so it must be asked for.
#PBS --cpunum-lhost=48
#PBS --gpunum-lhost=1
# 70 SCF iterations at dimension 10430 with 3 k-points, on 24 ranks and 2 GPUs.
# Scaling the sidia333_nc diagonalisation (5616, 162.9 s) by (10430/5616)^3 puts
# a comparable count of diagonalisations near an hour, and Set_Hamiltonian over
# 650 atoms adds to that.  8 h is margin, not an estimate.
#PBS -l elapstim_req=08:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N cghelgeo_ppn24
#PBS -j o
#PBS -o cghelgeo_ppn24.joblog

# OpenMX 4.0 GPU: cghelgeo_ppn24.dat -- 650 atoms over 5 species (190 C, 240 H, 80 N,
# 120 O, 20 P).  Orbitals per atom: H s3p2=9, C s2p2d1=13, N s3p3d2f1=29,
# O s3p3d2=22, P s4p3d3f2=42, so the matrix dimension is 10430.  Collinear
# (spin off), Band solver, k-grid 1x1x3, up to 70 SCF iterations to 1e-13.
# Two Pegasus nodes, 24 ranks per node = 48 ranks, MPS on both nodes.
#
# WHY 24 RANKS PER NODE
# ---------------------
# This is the wider half of a pair with ../cghelgeo, which runs the identical
# case at 12 ranks/node.  That run measured a steady 66 GiB of the node's
# 124 GiB, so there is roughly 58 GiB of headroom to spend.
#
# Doubling the ranks does not simply double that.  With only 3 k-points and 48
# ranks, OpenMX has far more ranks than k-points, so each k-point is split across
# a larger process group and the per-rank share of the 10430^2 matrices gets
# *smaller*, not larger.  Working against that, grid and atom data carry a
# per-rank overhead that does grow with rank count.  Which term wins is the point
# of running this: .smi samples host memory every 10 s, so the pair gives a
# measured memory-vs-ranks curve for this case instead of an extrapolation.

set -u

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
NNODE=2
PPN=24
NPROC=$((NNODE * PPN))
STDOUT=cghelgeo_ppn24.std
STDERR=cghelgeo_ppn24.err
ENVLOG=cghelgeo_ppn24.env
SMILOG=cghelgeo_ppn24.smi

PROGLOG="${STDOUT%.std}.progress"
# Stage markers, flushed immediately and readable from the login node while the
# job is still running.  The NQSV joblog only materialises at job end, so when a
# request sits in RUN after its calculation has finished there is otherwise no
# way to see which line the script is stuck on -- which is exactly the situation
# this had to be debugged from twice.
progress() { echo "[$(date '+%H:%M:%S')] $*" >> "$PROGLOG"; }

cd "${PBS_O_WORKDIR}" || exit 1
: > "$PROGLOG"
progress "script start on $(hostname)"

# openmx carries a DT_RPATH into the NVHPC 26.5 tree (CUDA 13.2, HPC-X 2.50 /
# Open MPI 5), so no module is needed at run time.  Keep the environment clean
# so nothing shadows it; only mpirun has to come from that same Open MPI.
command -v module >/dev/null 2>&1 && module purge >/dev/null 2>&1
unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"

#----------------------------------------------------------------------
# NVIDIA MPS -- on every node, not just this one
#----------------------------------------------------------------------
# /tmp is node-local, so this same path names a different directory on each
# node, which is exactly what MPS wants.  Strip the colon PBS_JOBID contains.
JOBTAG=$(echo "${PBS_JOBID:-$$}" | tr -c 'A-Za-z0-9._-' '_')
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-${USER}-${JOBTAG}/pipe"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-${USER}-${JOBTAG}/log"

MPS_X="-x CUDA_MPS_PIPE_DIRECTORY -x CUDA_MPS_LOG_DIRECTORY"
# shellcheck disable=SC2086
mps_on_each_node() {
  mpirun ${NQSV_MPIOPTS:-} -np "${NNODE}" -npernode 1 ${MPS_X} ./mps_node.sh "$1"
}

mps_stop_all() {
  mps_on_each_node stop >/dev/null 2>&1
}
# Covers normal exit, qdel and the elapstim kill, so no daemon is left on either
# node.  NOTE: the sampler subshell below clears this trap for itself -- a
# background subshell inherits the parent's traps, and would otherwise handle
# SIGTERM and then *resume its loop* instead of dying, which previously left
# finished jobs sitting in RUN for 24 minutes burning budget.
trap mps_stop_all EXIT INT TERM

{
  echo "=== host      : $(hostname)"
  echo "=== date      : $(date)"
  echo "=== jobid     : ${PBS_JOBID:-unset}"
  echo "=== workdir   : $(pwd)"
  echo "=== layout    : ${NNODE} nodes x ${PPN} ranks = ${NPROC} ranks, 1 H100/node"
  echo "=== mpirun    : $(command -v mpirun)"
  echo "=== NQSV_MPIOPTS = [${NQSV_MPIOPTS:-}]"
  echo "=== nodes ==="
  mpirun ${NQSV_MPIOPTS:-} -np "${NNODE}" -npernode 1 hostname
  echo "=== host memory (this node) ==="
  free -g
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

# The point of this run is MPS on both nodes, so a node that silently fell back
# to the default time-sliced mode would mislabel the result.  Abort instead.
if grep -q "control_pipe=no" mps_check.tmp; then
  echo "FATAL: MPS did not come up on every node:" | tee -a "$ENVLOG"
  cat mps_check.tmp | tee -a "$ENVLOG"
  rm -f mps_check.tmp
  exit 1
fi
if [ "$(grep -c 'control_pipe=yes' mps_check.tmp)" -ne "${NNODE}" ]; then
  echo "FATAL: expected ${NNODE} nodes with MPS, got:" | tee -a "$ENVLOG"
  cat mps_check.tmp | tee -a "$ENVLOG"
  rm -f mps_check.tmp
  exit 1
fi
rm -f mps_check.tmp

# Sampler for THIS node only (the batch script's node); the other node's GPU is
# covered by the "report" pass after the run.  trap - so kill actually kills it.
( trap - EXIT INT TERM
  for _ in $(seq 1 3000); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " $3 "/" $2}'
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
      nvidia-smi --query-compute-apps=process_name --format=csv,noheader | sort | uniq -c | tr '\n' ';'
      echo
    } >> "$SMILOG" 2>&1
    sleep 10
  done ) &
SMIPID=$!

#----------------------------------------------------------------------
# run
#----------------------------------------------------------------------
# --timeout guards against prterun failing to tear down after a rank dies, which
# has left requests sitting in RUN until the elapstim limit.
mpirun ${NQSV_MPIOPTS:-} -np "${NPROC}" -npernode "${PPN}" --timeout 27000 \
       ${MPS_X} \
       ../openmx cghelgeo_ppn24.dat -nt 1 > "$STDOUT" 2> "$STDERR" &
MPIPID=$!
progress "mpirun launched (pid ${MPIPID})"

# Foreground watchdog -- deliberately NOT a "( ... ) &" subshell.  The previous
# attempt ran exactly this logic in a background subshell and it never fired in a
# real job, although it fires correctly standalone.  Rather than keep debugging a
# process we cannot inspect on the compute node, the dependency is removed: this
# runs in the main shell, and every branch leaves a progress marker.
#
# OpenMX prints the banner immediately before MPI_Finalize, so once it appears
# the science is complete and on disk; anything after that is teardown.
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

# No "wait" on the sampler.  It is a pure monitoring loop with nothing to flush,
# and a wait here is itself a way for the script to block forever if the kill
# does not land.  SIGKILL, children first, and move on.
pkill -KILL -P "$SMIPID" 2>/dev/null
kill -KILL "$SMIPID" 2>/dev/null
progress "sampler killed"

# A watchdog kill shows up as a signal exit code, but the run did finish -- treat
# the banner as the real verdict so a torn-down teardown is not reported as failure.
if [ "$rc" -ne 0 ] && grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
  echo "=== mpirun returned ${rc}, but openmx printed the completion banner;" >> "$STDOUT"
  echo "=== treating as success (teardown was killed by the watchdog). ===" >> "$STDOUT"
  rc=0
fi

echo "=== openmx exit status = ${rc} ($(date)) ===" >> "$STDOUT"

echo "=== MPS server per node (post-run) ==="
mps_on_each_node report 2>&1
echo "=== peak host memory, this node (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$SMILOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== peak GPU, this node ==="
grep -oE "^[0-9]+ %, [0-9]+ MiB" "$SMILOG" 2>/dev/null | sort -t, -k2 -rn | head -3
echo "=== Utot (no reference value: first run of this case) ==="
grep -E "Utot\." cghelgeo_ppn24.out 2>/dev/null | tail -2
echo "=== SCF iterations completed (of 70 max) ==="
grep -cE "SCF=" "$STDOUT" 2>/dev/null
echo "=== progress markers ==="
cat "$PROGLOG" 2>/dev/null
echo "=== tail of openmx stdout ==="
tail -25 "$STDOUT"

exit ${rc}
