#!/bin/bash
#------- qsub option -----------
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 1
#PBS --cpunum-lhost=48
#PBS --gpunum-lhost=1
#PBS -l elapstim_req=04:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N rtb_L_gnom
#PBS -j o
#PBS -o rtb_L_gpu_nomps.joblog

# README benchmark: OpenMX "-runtestL" (16 medium/large systems), GPU run
# WITHOUT CUDA MPS -- the 48 ranks time-slice the H100 with one CUDA
# context each.  This is the "H100 GPU, MPS off" column of the README
# table; everything else is identical to rtb_L_gpu (input-file defaults:
# scf.eigen.lib gpusolver, scf.gemmul8.enable on).  1 Pegasus gen_S node
# (Xeon Platinum 8468, 48 cores + 1 H100 PCIe), flat MPI 48 ranks, -nt 1.
# Same binary as every other column (work/openmx, v2.0_thesis tag build;
# identity recorded in the per-case *.manifest.json).  large_example/ and
# ../DFT_DATA19 are reached through symlinks so the benchmark configs run
# isolated.  Reference: with MPS the suite took 1481.62 s on this machine.

set -u
ulimit -c 0
NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
CASE=rtb_L_gpu_nomps
cd "${PBS_O_WORKDIR}" || exit 1

command -v module >/dev/null 2>&1 && module purge >/dev/null 2>&1
unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"

{
  echo "=== host   : $(hostname)"
  echo "=== date   : $(date)"
  echo "=== jobid  : ${PBS_JOBID:-unset}"
  echo "=== cpu    : $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //') x $(nproc) cores"
  echo "=== mem ==="
  free -g
  echo "=== gpu    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
  echo "=== config : gpusolver + GEMMul8 on (input defaults), 48 ranks, NO MPS (time-sliced)"
  echo "=== mps daemon on this node (must be 0): $(pgrep -fc nvidia-cuda-mps-control 2>/dev/null || echo 0)"
  echo "=== mpirun : $(command -v mpirun)"
  ldd ./openmx | grep -E "libcud|libcublas|libcusolver|libmpi\."
} > "${CASE}.env" 2>&1

( for _ in $(seq 1 2900); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " $3 "/" $2}'
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    } >> "${CASE}.smi" 2>&1
    sleep 5
  done ) &
SMIPID=$!

mpirun ${NQSV_MPIOPTS:-} -np 48 -npernode 48 --timeout 14040 \
       ./openmx -runtestL -nt 1 > "${CASE}.std" 2> "${CASE}.err"
rc=$?
echo "=== openmx exit status = ${rc} ($(date)) ===" >> "${CASE}.std"

pkill -KILL -P "$SMIPID" 2>/dev/null
kill -KILL "$SMIPID" 2>/dev/null

echo "=== runtestL.result ==="
cat runtestL.result 2>&1
echo "=== peak GPU (util %, MiB) ==="
grep -oE "^[0-9]+ %, [0-9]+ MiB" "${CASE}.smi" 2>/dev/null | sort -t, -k2 -rn | head -3
echo "=== peak host memory (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "${CASE}.smi" 2>/dev/null | sort -t: -k2 -rn | head -3
exit ${rc}
