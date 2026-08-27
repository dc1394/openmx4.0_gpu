#!/bin/bash
#------- qsub option -----------
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 1
#PBS --cpunum-lhost=48
#PBS --gpunum-lhost=1
#PBS -l elapstim_req=01:30:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N rtb_s_mps_ab
#PBS -j o
#PBS -o rtb_s_mps_ab.joblog

# README benchmark, MPS A/B: the "-runtest" suite three times on the SAME
# node -- (0) warm-up pass, discarded (PAO/VPS file cache, CUDA JIT);
# (1) without MPS (48 ranks time-slice the H100); (2) with MPS.  Same-node
# back-to-back runs so the ~+-18% gen_S node-to-node variance cancels; this
# pair is the evidence for the "run MPS when many ranks share a GPU"
# recommendation in README.md.  GPU defaults (gpusolver, GEMMul8 on), same
# binary as every other benchmark config (work/openmx, v2.0_thesis tag
# build; identity recorded in the per-case *.manifest.json).

set -u
ulimit -c 0
NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
CASE=rtb_s_mps_ab
cd "${PBS_O_WORKDIR}" || exit 1

command -v module >/dev/null 2>&1 && module purge >/dev/null 2>&1
unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"

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
  echo "=== gpu    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
  echo "=== plan   : warm-up (discard) -> no-MPS (timed) -> MPS (timed), same node"
  echo "=== mpirun : $(command -v mpirun)"
} > "${CASE}.env" 2>&1

run_suite() {  # $1 = label, $2 = extra mpirun args
  # shellcheck disable=SC2086
  mpirun ${NQSV_MPIOPTS:-} -np 48 -npernode 48 --timeout 1500 $2 \
         ./openmx -runtest -nt 1 > "${CASE}.$1.std" 2> "${CASE}.$1.err"
  rc=$?
  mv runtest.result "runtest.result.$1" 2>/dev/null
  echo "=== $1: exit=${rc}, total=$(grep 'Total elapsed' "runtest.result.$1" 2>/dev/null | awk '{print $NF}') s"
}

{
  run_suite warmup ""
  run_suite nomps  ""

  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "FATAL: nvidia-cuda-mps-control not found; aborting before the MPS pass."
    exit 1
  fi
  mps_on_each_node start
  mps_on_each_node check
  run_suite mps "${MPS_X}"
  echo "=== MPS server (post-run) ==="
  mps_on_each_node report
} >> "${CASE}.env" 2>&1

echo "=== A/B summary ==="
for l in warmup nomps mps; do
  echo "--- $l ---"
  tail -4 "runtest.result.$l" 2>/dev/null
done
exit 0
