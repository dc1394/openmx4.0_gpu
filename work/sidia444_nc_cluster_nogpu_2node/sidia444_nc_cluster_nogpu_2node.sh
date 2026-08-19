#!/bin/bash
#------- qsub option -----------
# General use runs in gen_S/M/L under the GPU2026 allocation; the gpu queue's ACL
# only lists CP24I024, which has a 0.00 budget.  gen_S covers 1-31 nodes.
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 2
#PBS --cpunum-lhost=48
#PBS -l elapstim_req=06:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N s333_2node_nogpu
#PBS -j o
#PBS -o sidia444_nc_cluster_nogpu_2node.joblog

# OpenMX 4.0 GPU: 512 Si, s2p2d1, non-collinear -- 2-component spinors put the
# Cluster matrices at 13312x13312 complex, 2.84 GB each, eight times the 0.35 GB
# real matrices of the collinear twin ../sidia444_cluster.  Gamma point only
# (scf.Kgrid 1 1 1), 25 SCF iterations to 1e-13, energycutoff 200 Ry.
#
# MEMORY IS THE OPEN QUESTION HERE
# --------------------------------
# The collinear twin was measured at 48 ranks/node over 2-4 nodes and fits
# host-per-node = (rank-proportional) + (divisible total)/nodes as:
#
#            rank-proportional   divisible total   1-node need   outcome
#   gpu             40 GiB           120 GiB          160 GiB    OOM
#   elpa2           38 GiB            68 GiB          106 GiB    ran, 110 measured
#
# The divisible term is grid data plus solver workspace; only the workspace part
# scales with the matrices, and here that part is 8x heavier.  Which is why even
# the CPU 1-node point is not expected to fit, and why the multi-node points are
# run rather than predicted: .smi samples host memory every 10 s, so whatever
# fails still yields the number that says by how much.
#
# First run of this case: no reference Utot exists yet, so the eight points of
# the pair check each other.
# Utot must not move with node count -- only the timings may.

set -u

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
NNODE=2
PPN=48
NPROC=$((NNODE * PPN))
CASE=sidia444_nc_cluster_nogpu_2node
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

# No MPS and no GPU.  Every device path in this build is gated on the eigen
# library actually being GPUSOLVER, and this input asks for elpa2, so no CUDA
# call happens at all: DFT_GPU_DeviceInit() returns before its
# cudaGetDeviceCount, Band_DFT_Col's device selection sits behind
# use_gpusolver_dense, and both Set_ProExpn_VNA GpuBegin entry points open with
# "if (scf_eigen_lib_flag!=GPUSOLVER) return 0;".  Hence no --gpunum-lhost:
# asking for an H100 here would only reserve one to sit idle.
# These are cleared so a stale value cannot point ranks at another job's daemon.
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

{
  echo "=== host      : $(hostname)"
  echo "=== date      : $(date)"
  echo "=== jobid     : ${PBS_JOBID:-unset}"
  echo "=== workdir   : $(pwd)"
  echo "=== layout    : ${NNODE} node(s) x ${PPN} ranks = ${NPROC} ranks, elpa2 CPU, no MPS"
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

# Host memory only -- there is no GPU in play.  "trap -" so a kill actually kills
# it: a background subshell inherits the parent's traps and would otherwise run
# the handler and resume its loop.
( trap - EXIT INT TERM
  for _ in $(seq 1 3000); do
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
mpirun ${NQSV_MPIOPTS:-} -np "${NPROC}" -npernode "${PPN}" --timeout 20400 \
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
echo "=== peak host memory, this node (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$MEMLOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== SCF iterations completed (of 25 max) ==="
grep -cE "SCF=" "$STDOUT" 2>/dev/null
echo "=== Utot (no reference yet; must agree across all eight runs) ==="
grep -E "Utot\." "${CASE}.out" 2>/dev/null | tail -2
echo "=== timing ==="
grep -E "Set_Hamiltonian |Diagonalization|^   DFT |Total Computational" "$STDOUT" 2>/dev/null | tail -4
echo "=== progress markers ==="
cat "$PROGLOG" 2>/dev/null

exit ${rc}
