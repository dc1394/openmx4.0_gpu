#!/bin/bash
#------- qsub option -----------
# General use ("一般利用") runs in gen_S/gen_M/gen_L, NOT the gpu queue -- the gpu
# queue's ACL lists CP24I024, which has a 0.00 budget, so it only ever answers
# "Budget exceeded".  The GPU2026 allocation (500.00 units) is the one with
# gen_* access.  Note that qstat -Q hides queues the current gid cannot reach, so
# gen_* is invisible under CP24I024; use `sg GPU2026 -c "qstat -Q"` to see it.
# gen_S is the 1-node class and takes up to 31 nodes per request.
#PBS -A GPU2026
#PBS -q gen_S
#PBS -b 1
# gen_S hands out 48 cores per logical host but its GPU default is 0
# ("GPU Number ... Std: 0" in qstat -Qf gen_S), so the H100 has to be asked for
# explicitly -- without this the CUDA paths get no device.
#PBS --cpunum-lhost=48
#PBS --gpunum-lhost=1
#PBS -l elapstim_req=02:00:00
#PBS -T openmpi
# The site JSV insists on an NQSV_MPI_VER tag.  NVHPC 26.5 has no modulefile, so
# name the nearest published HPC-X module; the job itself launches with the
# mpirun of the 26.5 SDK, which is the Open MPI openmx is actually linked to.
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N omx40_runtest48
#PBS -j o
#PBS -o runtest_np48.joblog

# OpenMX 4.0 GPU: "-runtest" over work/input_example (14 cases), flat MPI, 48 ranks
# on one Pegasus node (1 x Xeon Platinum 8468 / 48 cores + 1 x H100 PCIe).

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5

cd "${PBS_O_WORKDIR}" || exit 1

# openmx carries a DT_RPATH into the NVHPC 26.5 tree (CUDA 13.2, HPC-X 2.50 /
# Open MPI 5), so no module is needed at run time.  Keep the environment clean
# so nothing shadows it; only mpirun has to come from that same Open MPI.
command -v module >/dev/null 2>&1 && module purge >/dev/null 2>&1
unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"

{
  echo "=== host      : $(hostname)"
  echo "=== date      : $(date)"
  echo "=== mpirun    : $(command -v mpirun)"
  echo "=== NQSV_MPIOPTS = [${NQSV_MPIOPTS}]"
  echo "=== GPU ==="
  nvidia-smi
  echo "=== openmx runtime libraries ==="
  ldd ./openmx | grep -E "libcud|libcublas|libcusolver|libmpi\."
} > runtest_np48.env 2>&1

mpirun ${NQSV_MPIOPTS} -np 48 -npernode 48 \
       ./openmx -runtest -nt 1 > runtest_np48.std 2> runtest_np48.err
rc=$?

echo "=== openmx exit status = ${rc} ($(date)) ===" >> runtest_np48.std

# Runtest writes its verdict to work/runtest.result: one line per case with the
# largest deviation from the stored reference *.out.  Echo it into the joblog so
# the pass/fail picture is in one place.
echo "=== runtest.result ==="
cat runtest.result 2>&1

exit ${rc}
