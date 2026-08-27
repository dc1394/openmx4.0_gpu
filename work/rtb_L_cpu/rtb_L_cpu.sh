#!/bin/bash
#------- qsub option -----------
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 1
#PBS --cpunum-lhost=48
#PBS -l elapstim_req=04:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N rtb_L_cpu
#PBS -j o
#PBS -o rtb_L_cpu.joblog

# README benchmark: OpenMX "-runtestL" (16 medium/large systems), CPU
# reference run.  1 Pegasus gen_S node (Xeon Platinum 8468, 48 cores), flat
# MPI 48 ranks, -nt 1.  No GPU is requested (gen_S hands out 0 GPUs by
# default) and OPENMX_GPU=0 demotes the run to the CPU (ELPA2) paths
# explicitly -- the same binary as the GPU runs (work/openmx, v2.0_thesis
# tag build; identity recorded in the per-case *.manifest.json).
# large_example/ and ../DFT_DATA19 are reached through symlinks so the four
# benchmark configs run isolated from work/ and from each other.
# Reference: the same suite on 18 ranks of a Core i9-10980XE took 5857 s.

set -u
ulimit -c 0
NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
CASE=rtb_L_cpu
cd "${PBS_O_WORKDIR}" || exit 1

command -v module >/dev/null 2>&1 && module purge >/dev/null 2>&1
unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"
export OPENMX_GPU=0

{
  echo "=== host   : $(hostname)"
  echo "=== date   : $(date)"
  echo "=== jobid  : ${PBS_JOBID:-unset}"
  echo "=== cpu    : $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //') x $(nproc) cores"
  echo "=== mem ==="
  free -g
  echo "=== OPENMX_GPU=${OPENMX_GPU} (CPU/ELPA2 paths; no GPU allocated to this job)"
  echo "=== gpu    : $(nvidia-smi -L 2>&1 | head -2)"
  echo "=== mpirun : $(command -v mpirun)"
  ldd ./openmx | grep -E "libcud|libcublas|libcusolver|libmpi\."
} > "${CASE}.env" 2>&1

mpirun ${NQSV_MPIOPTS:-} -np 48 -npernode 48 --timeout 14040 \
       -x OPENMX_GPU \
       ./openmx -runtestL -nt 1 > "${CASE}.std" 2> "${CASE}.err"
rc=$?
echo "=== openmx exit status = ${rc} ($(date)) ===" >> "${CASE}.std"

echo "=== runtestL.result ==="
cat runtestL.result 2>&1
exit ${rc}
