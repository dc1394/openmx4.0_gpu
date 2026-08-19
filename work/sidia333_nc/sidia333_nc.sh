#!/bin/bash
#------- qsub option -----------
# General use runs in gen_S/M/L under the GPU2026 allocation; the gpu queue's ACL
# only lists CP24I024, which has a 0.00 budget.
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 1
# gen_S's GPU default is 0 ("GPU Number ... Std: 0"), so the H100 must be asked
# for explicitly -- without this the CUDA paths get no device at all.
#PBS --cpunum-lhost=48
#PBS --gpunum-lhost=1
# The collinear twin of this case ran in 105 s wall (Diagonalization 40 s) at
# dimension 2808.  Non-collinear doubles that to 5616, and the eigensolver is
# O(N^3), so budget roughly 8x the diagonalisation -- 2 h leaves ample margin
# even if SCF runs the full 25 iterations.
#PBS -l elapstim_req=02:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N sidia333_nc
#PBS -j o
#PBS -o sidia333_nc.joblog

# OpenMX 4.0 GPU: sidia333_nc.dat -- the non-collinear (scf.SpinPolarization NC)
# twin of ../sidia333.  216 Si, s2p2d1, Band solver, 2x2x2 k-grid, 25 SCF iters,
# on one Pegasus node with flat MPI 48 ranks sharing a single H100 PCIe *through
# NVIDIA MPS*.
#
# NC makes every matrix a 2-component spinor, so the dimension is 5616 rather
# than 2808 and the arithmetic is complex throughout.  The collinear run already
# peaked at 58 GiB of the H100's 80 GiB, so GPU memory is the thing to watch
# here -- .smi records it every 30 s.
#
# MPS matters here because 48 ranks time-slice one GPU by default: each rank's
# kernels get their own context and the driver serialises them, so small
# per-rank kernels leave the H100 mostly idle.  MPS funnels all 48 into one
# server context so their kernels actually run concurrently.
#
# Submit from this directory: everything (script, .std, .err, .env, .smi,
# .joblog, and openmx's own output) stays under work/sidia333_nc.

set -u

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
STDOUT=sidia333_nc.std
STDERR=sidia333_nc.err
ENVLOG=sidia333_nc.env
SMILOG=sidia333_nc.smi

# PBS_O_WORKDIR is work/sidia333_nc, so ../openmx is the binary and the input's
# "DATA.PATH ../../DFT_DATA19" resolves to the repo's DFT_DATA19.
cd "${PBS_O_WORKDIR}" || exit 1

# openmx carries a DT_RPATH into the NVHPC 26.5 tree (CUDA 13.2, HPC-X 2.50 /
# Open MPI 5), so no module is needed at run time.  Keep the environment clean
# so nothing shadows it; only mpirun has to come from that same Open MPI.
command -v module >/dev/null 2>&1 && module purge >/dev/null 2>&1
unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"

#----------------------------------------------------------------------
# NVIDIA MPS
#----------------------------------------------------------------------
# Node-local, job-private pipe/log dirs: the pipe directory must live on a local
# filesystem (never Lustre), and a job-specific path avoids colliding with or
# inheriting stale state from anything else that used the default /tmp/nvidia-mps.
# PBS_JOBID contains a colon, which is legal in a path but noisy -- strip it.
JOBTAG=$(echo "${PBS_JOBID:-$$}" | tr -c 'A-Za-z0-9._-' '_')
MPSROOT="/tmp/mps-${USER}-${JOBTAG}"
export CUDA_MPS_PIPE_DIRECTORY="${MPSROOT}/pipe"
export CUDA_MPS_LOG_DIRECTORY="${MPSROOT}/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" || exit 1

mps_stop() {
  if [ -e "${CUDA_MPS_PIPE_DIRECTORY}/control" ]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1
  fi
  rm -rf "$MPSROOT"
}
# Covers normal exit, qdel, and the elapstim kill, so no daemon is left behind.
trap mps_stop EXIT INT TERM

{
  echo "=== host      : $(hostname)"
  echo "=== date      : $(date)"
  echo "=== jobid     : ${PBS_JOBID:-unset}"
  echo "=== workdir   : $(pwd)"
  echo "=== mpirun    : $(command -v mpirun)"
  echo "=== NQSV_MPIOPTS = [${NQSV_MPIOPTS:-}]"
  echo "=== GPU ==="
  nvidia-smi
  echo "=== openmx runtime libraries ==="
  ldd ../openmx | grep -E "libcud|libcublas|libcusolver|libmpi\."
} > "$ENVLOG" 2>&1

# The point of this run is MPS, so a silent fall-through to the default
# time-sliced mode would mislabel the result.  Abort instead.
if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "FATAL: nvidia-cuda-mps-control not found on $(hostname); aborting." | tee -a "$ENVLOG"
  exit 1
fi

nvidia-cuda-mps-control -d
for _ in $(seq 1 20); do
  [ -e "${CUDA_MPS_PIPE_DIRECTORY}/control" ] && break
  sleep 0.5
done
if [ ! -e "${CUDA_MPS_PIPE_DIRECTORY}/control" ]; then
  echo "FATAL: MPS control pipe never appeared; aborting." | tee -a "$ENVLOG"
  exit 1
fi

{
  echo "=== MPS ==="
  echo "pipe dir = ${CUDA_MPS_PIPE_DIRECTORY}"
  echo -n "default_active_thread_percentage = "
  echo get_default_active_thread_percentage | nvidia-cuda-mps-control
} >> "$ENVLOG" 2>&1

# Sample the GPU during the run.  The presence of an nvidia-cuda-mps-server
# process alongside the ranks is what confirms MPS is actually in effect -- note
# that driver 580 still lists each MPS client separately, so the client count is
# not the signal.
( for _ in $(seq 1 720); do
    { date '+--- %H:%M:%S'
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    } >> "$SMILOG" 2>&1
    sleep 30
  done ) &
SMIPID=$!

#----------------------------------------------------------------------
# run
#----------------------------------------------------------------------
# -x propagates the MPS pipe path to every rank; without it the ranks would fall
# back to the default pipe directory and quietly bypass the daemon we started.
mpirun ${NQSV_MPIOPTS:-} -np 48 -npernode 48 \
       -x CUDA_MPS_PIPE_DIRECTORY -x CUDA_MPS_LOG_DIRECTORY \
       ../openmx sidia333_nc.dat -nt 1 > "$STDOUT" 2> "$STDERR"
rc=$?

kill "$SMIPID" 2>/dev/null

echo "=== openmx exit status = ${rc} ($(date)) ===" >> "$STDOUT"

echo "=== MPS server sightings in ${SMILOG} ==="
grep -c "nvidia-cuda-mps-server" "$SMILOG" 2>/dev/null
echo "=== highest GPU utilisation samples ==="
grep -oE "^[0-9]+ %, [0-9]+ MiB" "$SMILOG" 2>/dev/null | sort -rn | head -5
echo "=== tail of openmx stdout ==="
tail -25 "$STDOUT"

exit ${rc}
