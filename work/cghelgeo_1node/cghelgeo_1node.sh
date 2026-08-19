#!/bin/bash
#------- qsub option -----------
# General use runs in gen_S/M/L under the GPU2026 allocation; the gen_* split is
# by node count and gen_S covers 1-31, so one node belongs here.
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 1
# Per *logical host*: 48 cores and 1 GPU.  The queue's GPU default is 0, so the
# H100 has to be asked for explicitly.
#PBS --cpunum-lhost=48
#PBS --gpunum-lhost=1
#PBS -l elapstim_req=01:30:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N cghel1n_gpu
#PBS -j o
#PBS -o cghelgeo_1node.joblog

# WHAT THIS RUN IS FOR
# --------------------
# The GPU sibling of ../cghelgeo_nogpu_1node.  Both exist to replace an
# extrapolation with a measurement: every cghelgeo run so far -- cghelgeo,
# cghelgeo_nogpu, cghelgeo_ppn16/24/36/48 -- was 2 nodes, and "1 node is
# impossible" was a claim from a fit that was never tested.
#
# Input: ../cghelgeo/cghelgeo.dat with only System.Name changed.  650 atoms
# (190 C, 240 H, 80 N, 120 O, 20 P), matrix dimension 10430, collinear, Band
# solver, k-grid 1x1x3, 70 SCF iterations to 1e-13.  One Pegasus node, 48 ranks,
# one H100, MPS on.
#
# WHAT THE FIT PREDICTS
# ---------------------
# Measured peaks at 2 nodes, MPS on, per node:
#
#   ppn/node   ranks   host peak
#     16         32      72 GiB   (killed early -- partial, not used in the fit)
#     24         48      83 GiB
#     36         72      92 GiB
#     48         96     105 GiB
#
# Fitting per-node peak = a*ppn + D/n over 24/36/48 gives a ~ 0.92 GiB/rank and
# D ~ 122 GiB of divisible data.  One node at ppn 48 is then 0.92*48 + 122 ~ 166
# GiB against a 124 GiB host, so a host OOM kill is the expectation.  The point
# is to see it rather than assert it, and to get the real number if the fit is
# wrong -- it has been wrong in both directions on this machine before.
#
# DS_VNA will take the CPU fallback here as it does at 2 nodes: its device
# estimate is 3.274 GiB/rank independent of rank count, so 48 ranks ask 157 GiB
# of an 80 GiB card and Set_ProExpn_VNA.c falls back.  That is the fixed path,
# not a new failure.
#
# WALLTIME
# --------
# 1.5 h.  550 s at 96 ranks on 2 GPUs; 48 ranks on one GPU should be under 1000 s
# if it survives at all, and an OOM dies in minutes.

set -u

# 48 ranks OOM-killed at once is exactly what fills a directory with core files
# -- a previous aborted run left 21 of them, 347 MB.  The memory trace and the
# exit status are the data here; nothing is debugged from a core dump.
ulimit -c 0

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
NNODE=1
PPN=48
NPROC=$((NNODE * PPN))
CASE=cghelgeo_1node
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
# NVIDIA MPS
#----------------------------------------------------------------------
# One node here, but the same per-node launch is kept so this file stays
# comparable with the 2-node scripts.  /tmp is node-local, so this path names a
# different directory on each node; strip the colon PBS_JOBID contains.
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
# traps and would otherwise handle SIGTERM and then resume its loop, which once
# left finished jobs sitting in RUN for 24 minutes burning budget.
trap mps_stop_all EXIT INT TERM

{
  echo "=== host      : $(hostname)"
  echo "=== date      : $(date)"
  echo "=== jobid     : ${PBS_JOBID:-unset}"
  echo "=== workdir   : $(pwd)"
  echo "=== layout    : ${NNODE} node x ${PPN} ranks = ${NPROC} ranks, 1 H100, MPS on"
  echo "=== mpirun    : $(command -v mpirun)"
  echo "=== NQSV_MPIOPTS = [${NQSV_MPIOPTS:-}]"
  echo "=== eigen library requested by the input ==="
  grep -E "scf.eigen.lib" "${CASE}.dat"
  echo "=== host memory before the run ==="
  free -g
  echo "=== core dump limit (expect 0) ==="
  ulimit -c
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
# the result, so require MPS rather than hope for it.
if [ "$(grep -c 'control_pipe=yes' mps_check.tmp)" -ne "${NNODE}" ]; then
  echo "FATAL: expected ${NNODE} node(s) with MPS, got:" | tee -a "$ENVLOG"
  cat mps_check.tmp | tee -a "$ENVLOG"
  rm -f mps_check.tmp
  exit 1
fi
rm -f mps_check.tmp
progress "MPS up on ${NNODE} node"

# 5 s, not the usual 10.  The memory curve IS the result here: a host OOM kills
# the run at its peak, and a coarse interval can miss that last sample entirely.
# "trap -" so a kill actually kills it.
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
progress "launching mpirun: ${NPROC} ranks on ${NNODE} node"
mpirun ${NQSV_MPIOPTS:-} -np "${NPROC}" -npernode "${PPN}" --timeout 5100 \
       ${MPS_X} \
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
pkill -KILL -P "$SMIPID" 2>/dev/null
kill -KILL "$SMIPID" 2>/dev/null
progress "sampler killed"

if [ "$rc" -ne 0 ] && grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
  echo "=== mpirun returned ${rc}, but openmx printed the completion banner;" >> "$STDOUT"
  echo "=== treating as success (teardown was killed by the watchdog). ===" >> "$STDOUT"
  rc=0
fi

echo "=== openmx exit status = ${rc} ($(date)) ===" >> "$STDOUT"

echo "=== VERDICT (fit predicted ~166 GiB against a 124 GiB host) ==="
if grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
  echo "COMPLETED -- 1 node x 48 ranks fits on the GPU path after all; the fit was wrong."
else
  echo "DID NOT COMPLETE.  Distinguishing host OOM from anything else:"
  echo "  signal 9 / Killed lines (host OOM kill looks like this):"
  grep -hE "exited on signal|Killed|Out Of Memory|oom" "$STDERR" "$STDOUT" 2>/dev/null | sort -u | head -10
  echo "  device-side complaints (GEMMul8 / cuSOLVER / OpenACC), which are NOT host OOM:"
  grep -hiE "out of memory|workspace|acc_|CUDA_ERROR|cusolver|Signal: Aborted" "$STDERR" 2>/dev/null | sort -u | head -10
  echo "  core files (none expected, ulimit -c 0):"
  ls -1 core.* 2>/dev/null | wc -l
fi
echo "=== DS_VNA GPU/CPU decision ==="
grep -m2 -hE "Set_ProExpn_VNA DS_VNA GPU" "$STDERR" 2>/dev/null || echo "(no fallback message)"
echo "=== last samples before the end ==="
tail -15 "$SMILOG" 2>/dev/null
echo "=== peak host memory (GiB); 2-node GPU reference was 105/124 per node ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$SMILOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== peak GPU (2-node reference: 65.1 GiB per card) ==="
grep -oE "^[0-9]+ %, [0-9]+ MiB" "$SMILOG" 2>/dev/null | sort -t, -k2 -rn | head -3
echo "=== MPS server (post-run) ==="
mps_on_each_node report 2>&1
echo "=== SCF iterations completed (of 70 max) ==="
grep -cE "SCF=" "$STDOUT" 2>/dev/null
echo "=== Utot (2-node reference: -4154.835898598240) ==="
grep -E "Utot\." "${CASE}.out" 2>/dev/null | tail -2
echo "=== timing (2-node GPU at 96 ranks: total 550.3 s, Diag 447.0 s) ==="
grep -E "Set_ProExpn_VNA |Set_Hamiltonian |Diagonalization|^   DFT |Total Computational" "$STDOUT" 2>/dev/null | tail -5
echo "=== progress markers ==="
cat "$PROGLOG" 2>/dev/null
echo "=== tail of openmx stderr ==="
tail -25 "$STDERR" 2>/dev/null

exit ${rc}
