#!/bin/bash
#------- qsub option -----------
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 1
#PBS --cpunum-lhost=48
#PBS --gpunum-lhost=1
#PBS -l elapstim_req=00:45:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N prof_bc300
#PBS -j o
#PBS -o prof_bcol300.joblog

# BANDPROF cross-check pair, H100 (mirrors prof/prof_bcol216_{o,g} on the
# 5080): bcol300 o then g, 5 SCF, np48, MPS on, OPENMX_BAND_PROFILE=1,
# tag binary ../../openmx (v2.0_thesis build, runtime-gated profiling --
# no rebuild).  Both legs run back to back on the SAME node.  BANDPROF
# lines go to stderr (one line per rank per SCF iteration).  Caveat carried
# from the 5080 doc: in the o leg the gemmF/gemmB buckets read ~0 because
# cublasZgemm is asynchronous (execution absorbed into syevdx's blocking
# copy); the g leg's buckets are real (gemmul8::gemm synchronizes).

set -u
ulimit -c 0
NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
cd "${PBS_O_WORKDIR}" || exit 1

command -v module >/dev/null 2>&1 && module purge >/dev/null 2>&1
unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"
export OPENMX_BAND_PROFILE=1

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
  echo "=== host: $(hostname), jobid ${PBS_JOBID:-unset}, $(date)"
  echo "=== binary: ../../openmx ($(md5sum ../../openmx | cut -d' ' -f1))"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > prof_bcol300.env 2>&1
mps_on_each_node start >> prof_bcol300.env 2>&1
mps_on_each_node check >> prof_bcol300.env 2>&1
grep -q 'control_pipe=yes' prof_bcol300.env || { echo "FATAL: no MPS" >> prof_bcol300.env; exit 1; }

for leg in o g; do
  # shellcheck disable=SC2086
  mpirun ${NQSV_MPIOPTS:-} -np 48 -npernode 48 --timeout 1200 \
         ${MPS_X} -x OPENMX_BAND_PROFILE \
         ../../openmx "prof_bcol300_${leg}.dat" -nt 1 \
         > "prof_bcol300_${leg}.stdout" 2> "prof_bcol300_${leg}.stderr"
  echo "=== leg ${leg}: exit $?, BANDPROF lines: $(grep -c '^BANDPROF' "prof_bcol300_${leg}.stderr")"
done

echo "=== MPS server (post-run) ==="
mps_on_each_node report 2>&1
exit 0
