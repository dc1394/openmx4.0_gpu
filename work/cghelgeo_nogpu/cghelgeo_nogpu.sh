#!/bin/bash
#------- qsub option -----------
# General use runs in gen_S/M/L under the GPU2026 allocation; the gpu queue's ACL
# only lists CP24I024, which has a 0.00 budget.  gen_S is still the right queue
# at 2 nodes -- the gen_* split is by node count: gen_S 1-31, gen_M 32-63,
# gen_L 64-150.
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 2
#PBS --cpunum-lhost=48
# No --gpunum-lhost.  This run never touches the GPU, and the queue's GPU
# default is 0, so leaving it out is what actually expresses that -- see below.
#PBS -l elapstim_req=10:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N cghelgeo_nogpu
#PBS -j o
#PBS -o cghelgeo_nogpu.joblog

# OpenMX 4.0 GPU build, run as a pure CPU job: cghelgeo_nogpu.dat is
# ../cghelgeo/cghelgeo.dat with two lines changed -- System.Name, and
# "scf.eigen.lib gpusolver" -> "scf.eigen.lib elpa2".  Everything physical is
# identical: 650 atoms (190 C, 240 H, 80 N, 120 O, 20 P), matrix dimension
# 10430, collinear, Band solver, k-grid 1x1x3, 70 SCF iterations to 1e-13.
# Two Pegasus nodes, 48 ranks per node = 96 ranks.  No MPS.
#
# NO GPU IS REQUESTED, AND NONE IS NEEDED
# ---------------------------------------
# Every device path in this build is gated on the eigen library actually being
# GPUSOLVER, so elpa2 turns them all off before any CUDA call happens:
#
#   DFT.c            DFT_GPU_DeviceInit() returns at
#                    "if (!(scf_eigen_lib_flag_input == GPUSOLVER)) return;"
#                    -- ahead of its cudaGetDeviceCount;
#   Band_DFT_Col.c   device selection sits behind use_gpusolver_dense, which is
#                    (scf_eigen_lib_flag == GPUSOLVER && ...);
#   Set_ProExpn_VNA.c both GpuBegin entry points open with
#                    "if (scf_eigen_lib_flag!=GPUSOLVER) return 0;".
#
# The binary still links libcudart/libcublas/libcusolver, but linking does not
# initialise a context.  Asking for a GPU here would reserve an H100 that stays
# idle for the whole run.
#
# WALLTIME
# --------
# 10 h as requested.  For scale, the same geometry on the same 96 ranks with
# gpusolver took 550 s wall, 447 s of it in the diagonalisation; that part now
# runs on ELPA2 across CPU cores instead of two H100s, so the slowdown lands
# almost entirely there and a large margin is the right call.

set -u

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
NNODE=2
PPN=48
NPROC=$((NNODE * PPN))
CASE=cghelgeo_nogpu
STDOUT=${CASE}.std
STDERR=${CASE}.err
ENVLOG=${CASE}.env
MEMLOG=${CASE}.mem
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

# No MPS.  Nothing starts a control daemon, and these are cleared so a stale
# value in the environment cannot point anything at another job's daemon.
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
  echo "=== host memory (this node) ==="
  free -g
  echo "=== GPUs visible (expected: none assigned) ==="
  nvidia-smi -L 2>&1 || true
  echo "=== MPS control daemons per node (expect 0) ==="
  mpirun ${NQSV_MPIOPTS:-} -np "${NNODE}" -npernode 1 \
    bash -c 'echo "$(hostname): $(pgrep -fc nvidia-cuda-mps-control 2>/dev/null || echo 0)"'
  echo "=== openmx runtime libraries ==="
  ldd ../openmx | grep -E "libcud|libcublas|libcusolver|libmpi\."
} > "$ENVLOG" 2>&1

# Host memory only -- there is no GPU to sample.  It still matters: the gpusolver
# run at this rank count peaked at 105 GiB of the node's 124 GiB, and ELPA2 keeps
# its own distributed workspace on the host instead.
# "trap -" so a kill actually kills it: a background subshell inherits the
# parent's traps and would otherwise handle the signal and resume its loop.
( trap - EXIT INT TERM
  for _ in $(seq 1 4000); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " $3 "/" $2}'
    } >> "$MEMLOG" 2>&1
    sleep 10
  done ) &
MEMPID=$!

#----------------------------------------------------------------------
# run
#----------------------------------------------------------------------
progress "launching mpirun: ${NPROC} ranks"
mpirun ${NQSV_MPIOPTS:-} -np "${NPROC}" -npernode "${PPN}" --timeout 34200 \
       ../openmx "${CASE}.dat" -nt 1 > "$STDOUT" 2> "$STDERR" &
MPIPID=$!
progress "mpirun launched (pid ${MPIPID})"

# Foreground watchdog, deliberately not a background subshell.  OpenMX prints
# this banner immediately before MPI_Finalize, so once it appears the science is
# complete and on disk; on earlier runs mpirun then failed to return for 13
# minutes, leaving the request in RUN.  Give the ranks a grace period to finalise
# on their own, then tear it down.
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
echo "=== peak host memory, this node (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$MEMLOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== SCF iterations completed (of 70 max) ==="
grep -cE "SCF=" "$STDOUT" 2>/dev/null
echo "=== Utot (gpusolver reference: -4154.835898598240 at 96 ranks) ==="
grep -E "Utot\." "${CASE}.out" 2>/dev/null | tail -2
echo "=== timing (gpusolver at 96 ranks: total 550.3 s, Diagonalization 447.0 s) ==="
grep -E "Set_ProExpn_VNA |Set_Hamiltonian |Diagonalization|^   DFT |Total Computational" "$STDOUT" 2>/dev/null | tail -5
echo "=== progress markers ==="
cat "$PROGLOG" 2>/dev/null

exit ${rc}
