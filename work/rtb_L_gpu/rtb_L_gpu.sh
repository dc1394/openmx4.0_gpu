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
#PBS -N rtb_L_gpu
#PBS -j o
#PBS -o rtb_L_gpu.joblog

# README benchmark: OpenMX "-runtestL" (16 medium/large systems), GPU run
# with the input-file defaults (scf.eigen.lib gpusolver, scf.gemmul8.enable
# on -- i.e. GEMMul8 INT8-based FP64 GEMM emulation active).  1 Pegasus
# gen_S node (Xeon Platinum 8468, 48 cores + 1 H100 PCIe), flat MPI 48
# ranks, -nt 1, all ranks sharing the GPU through CUDA MPS (the production
# configuration on this machine).  Same binary as the CPU runs
# (work/openmx, v2.0_thesis tag build; identity recorded in the per-case
# *.manifest.json).  large_example/ and ../DFT_DATA19 are reached through
# symlinks so the four benchmark configs run isolated from work/ and from
# each other.  Reference: the same suite took 5005 s on 18 ranks of a Core
# i9-10980XE + RTX 5080.

set -u
ulimit -c 0
NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
CASE=rtb_L_gpu
cd "${PBS_O_WORKDIR}" || exit 1

command -v module >/dev/null 2>&1 && module purge >/dev/null 2>&1
unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"

# NVIDIA MPS -- per-node daemon; /tmp is node-local.
JOBTAG=$(echo "${PBS_JOBID:-$$}" | tr -c 'A-Za-z0-9._-' '_')
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-${USER}-${JOBTAG}/pipe"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-${USER}-${JOBTAG}/log"
MPS_X="-x CUDA_MPS_PIPE_DIRECTORY -x CUDA_MPS_LOG_DIRECTORY"
# shellcheck disable=SC2086
mps_on_each_node() {
  mpirun ${NQSV_MPIOPTS:-} -np 1 -npernode 1 ${MPS_X} ./mps_node.sh "$1"
}
mps_stop_all() { mps_on_each_node stop >/dev/null 2>&1; }
trap mps_stop_all EXIT INT TERM

{
  echo "=== host   : $(hostname)"
  echo "=== date   : $(date)"
  echo "=== jobid  : ${PBS_JOBID:-unset}"
  echo "=== cpu    : $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //') x $(nproc) cores"
  echo "=== mem ==="
  free -g
  echo "=== gpu    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
  echo "=== config : gpusolver + GEMMul8 on (input defaults), 48 ranks via MPS"
  echo "=== mpirun : $(command -v mpirun)"
  ldd ./openmx | grep -E "libcud|libcublas|libcusolver|libmpi\."
} > "${CASE}.env" 2>&1

if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "FATAL: nvidia-cuda-mps-control not found on $(hostname); aborting." | tee -a "${CASE}.env"
  exit 1
fi
mps_on_each_node start >> "${CASE}.env" 2>&1
{ echo "=== MPS ==="
  mps_on_each_node check
} > mps_check.tmp 2>&1
cat mps_check.tmp >> "${CASE}.env"
if [ "$(grep -c 'control_pipe=yes' mps_check.tmp)" -ne 1 ]; then
  echo "FATAL: MPS did not come up:" | tee -a "${CASE}.env"
  cat mps_check.tmp | tee -a "${CASE}.env"
  rm -f mps_check.tmp
  exit 1
fi
rm -f mps_check.tmp

( for _ in $(seq 1 2900); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " $3 "/" $2}'
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    } >> "${CASE}.smi" 2>&1
    sleep 5
  done ) &
SMIPID=$!

# shellcheck disable=SC2086
mpirun ${NQSV_MPIOPTS:-} -np 48 -npernode 48 --timeout 14040 \
       ${MPS_X} \
       ./openmx -runtestL -nt 1 > "${CASE}.std" 2> "${CASE}.err"
rc=$?
echo "=== openmx exit status = ${rc} ($(date)) ===" >> "${CASE}.std"

pkill -KILL -P "$SMIPID" 2>/dev/null
kill -KILL "$SMIPID" 2>/dev/null

echo "=== runtestL.result ==="
cat runtestL.result 2>&1
echo "=== MPS server (post-run; mps_server>=1 proves the ranks attached) ==="
mps_on_each_node report 2>&1
echo "=== peak GPU (util %, MiB) ==="
grep -oE "^[0-9]+ %, [0-9]+ MiB" "${CASE}.smi" 2>/dev/null | sort -t, -k2 -rn | head -3
echo "=== peak host memory (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "${CASE}.smi" 2>/dev/null | sort -t: -k2 -rn | head -3
exit ${rc}
