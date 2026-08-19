#!/bin/bash
#------- qsub option -----------
# General use runs in gen_S/M/L under the GPU2026 allocation; the gpu queue's ACL
# only lists CP24I024, which has a 0.00 budget.  The gen_* split is by node
# count: gen_S 1-31, gen_M 32-63, gen_L 64-150.
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 1
# Per logical host: 48 cores and 1 GPU.  The GPU default is 0
# ("GPU Number ... Std: 0"), so it has to be asked for explicitly.
#PBS --cpunum-lhost=48
#PBS --gpunum-lhost=1
#PBS -l elapstim_req=01:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N sd333nc_1n_nomps
#PBS -j o
#PBS -o sidia333_nc_1node_nomps.joblog

# OpenMX 4.0 GPU: 216 Si, s2p2d1, non-collinear (dimension 5616), Band solver,
# 2x2x2 k-grid, 25 SCF iters.  One Pegasus node, 48 ranks, one H100,
# **NVIDIA MPS deliberately NOT enabled**.
#
# This is the no-MPS control for ../sidia333_nc (same case, same 48 ranks, MPS
# on), so the pair isolates what MPS is worth when 48 ranks share one GPU:
#
#   ../sidia333_nc  (MPS on) : DFT 238.5 s, Diagonalization 162.9 s,
#                              Set_Hamiltonian 24.1 s, Utot -887.422830856493
#
# Utot must not move; only the timings should.  It is also the control for a
# teardown hang seen with MPS on, where openmx printed its completion banner and
# mpirun then sat for 13 minutes -- if that recurs here, MPS was not the cause.

set -u

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
NNODE=1
PPN=48
NPROC=$((NNODE * PPN))
CASE=sidia333_nc_1node_nomps
STDOUT=${CASE}.std
STDERR=${CASE}.err
ENVLOG=${CASE}.env
SMILOG=${CASE}.smi

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

# No MPS here.  Nothing starts a control daemon, and CUDA_MPS_PIPE_DIRECTORY is
# explicitly cleared so a stale value from the environment cannot silently point
# the ranks at some other job's daemon.
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

{
  echo "=== host      : $(hostname)"
  echo "=== date      : $(date)"
  echo "=== jobid     : ${PBS_JOBID:-unset}"
  echo "=== workdir   : $(pwd)"
  echo "=== layout    : ${NNODE} node x ${PPN} ranks = ${NPROC} ranks, MPS OFF"
  echo "=== mpirun    : $(command -v mpirun)"
  echo "=== NQSV_MPIOPTS = [${NQSV_MPIOPTS:-}]"
  echo "=== MPS control daemons on this node (expect 0) ==="
  pgrep -fc nvidia-cuda-mps-control 2>/dev/null || echo 0
  echo "=== host memory ==="
  free -g
  echo "=== GPU ==="
  nvidia-smi
  echo "=== openmx runtime libraries ==="
  ldd ../openmx | grep -E "libcud|libcublas|libcusolver|libmpi\."
} > "$ENVLOG" 2>&1

# Sampler.  "trap -" so a kill actually kills it: a background subshell inherits
# the parent's traps and would otherwise run the handler and resume its loop,
# which previously kept finished jobs in RUN for 24 minutes.  Recording
# process_name here is what shows MPS is genuinely absent -- with MPS on, an
# nvidia-cuda-mps-server appears alongside the ranks.
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
mpirun ${NQSV_MPIOPTS:-} -np "${NPROC}" -npernode "${PPN}" --timeout 3000 \
       ../openmx "${CASE}.dat" -nt 1 > "$STDOUT" 2> "$STDERR" &
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
# the banner as the real verdict.
if [ "$rc" -ne 0 ] && grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
  echo "=== mpirun returned ${rc}, but openmx printed the completion banner;" >> "$STDOUT"
  echo "=== treating as success (teardown was killed by the watchdog). ===" >> "$STDOUT"
  rc=0
fi

echo "=== openmx exit status = ${rc} ($(date)) ===" >> "$STDOUT"

echo "=== did the teardown hang? (watchdog line present => yes) ==="
grep -c "watchdog:" "$STDERR" 2>/dev/null
echo "=== MPS servers seen during the run (expect 0) ==="
grep -c "nvidia-cuda-mps-server" "$SMILOG" 2>/dev/null
echo "=== peak host memory (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$SMILOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== peak GPU ==="
grep -oE "^[0-9]+ %, [0-9]+ MiB" "$SMILOG" 2>/dev/null | sort -t, -k2 -rn | head -3
echo "=== Utot (MPS reference: -887.422830856493) ==="
grep -E "Utot\." "${CASE}.out" 2>/dev/null | tail -2
echo "=== timing (MPS reference: DFT 238.5 / Diag 162.9 / SetH 24.1 s) ==="
grep -E "Diagonalization|Set_Hamiltonian |^   DFT " "$STDOUT" 2>/dev/null | tail -3

exit ${rc}
