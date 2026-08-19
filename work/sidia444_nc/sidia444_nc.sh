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
# sidia333_nc (dim 5616) took 238 s of DFT, 163 s of it diagonalisation, at 48
# ranks.  Here the dimension is 2.37x larger and the eigensolver is O(N^3), so
# the GPU-side diagonalisation should land near 2200 s; the CPU-side phases are
# much slower at 2 ranks, so budget ~2 h.  4 h rather than 6 h because a
# prterun hang after an OOM sits in RUN until the limit (see --timeout below).
#PBS -l elapstim_req=04:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda13/25.11
#PBS -N sidia444_nc
#PBS -j o
#PBS -o sidia444_nc.joblog

# OpenMX 4.0 GPU: sidia444_nc.dat -- 512 Si, s2p2d1, non-collinear, Band solver,
# 2x2x2 k-grid, 25 SCF iters, on one Pegasus node.  No MPS this time.
#
# WHY ONLY 2 RANKS
# ----------------
# NC makes every matrix a 2-component spinor: 512 atoms x 13 orbitals x 2 =
# dimension 13312, so one complex double matrix is 13312^2 x 16 B = 2.84 GB
# against 0.50 GB for sidia333_nc at dimension 5616.
#
# The rank count here is measured, not estimated.  Two runs died first:
#
#   899514, 8 ranks -> OOM killer (signal 9)
#   899533, 4 ranks -> OOM killer, host peak 119 of 124 GiB
#
# The compute node has 124 GiB of host memory, NOT the 250 GiB the login node
# reports.  The 4-rank trace, sampled every 10 s, is the useful one:
#
#   4 6 9 42 61 63 | 115 119 -> dead
#                 ^ grid setup plateau   ^ Band_DFT_NonCol allocation
#
# The ~63 GiB plateau is grid data for 512 atoms.  It is essentially
# rank-independent -- with fewer ranks each rank simply owns more atoms -- so
# cutting ranks does NOT shrink it.  Only the diagonalisation allocation scales
# with rank count, and it has just 124 - 63 = 61 GiB to live in.
#
# 4 ranks overran that, so the solver wants more than 61/4/2.84 = 5.4 full
# matrices per rank.  At 2 ranks the same 6-ish matrices cost ~34 GiB, landing
# near 97 GiB total.  That is the headroom this run is buying.
#
# The GPU was never the constraint: Set_Hamiltonian reported
# "peak=52.261 GiB, free=75.570 GiB" on the 80 GiB H100 even at 8 ranks.
#
# Dropping ranks costs less wall time than it looks: the diagonalisation runs on
# the GPU either way, and the total GPU work is fixed by the 8 k-points.  If the
# CPU-side phases turn out to dominate, the fix is threads rather than ranks
# (-nt N shares one address space), not a higher -np.

set -u

NVROOT=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5
NPROC=2
STDOUT=sidia444_nc.std
STDERR=sidia444_nc.err
ENVLOG=sidia444_nc.env
SMILOG=sidia444_nc.smi

# PBS_O_WORKDIR is work/sidia444_nc, so ../openmx is the binary and the input's
# "DATA.PATH ../../DFT_DATA19" resolves to the repo's DFT_DATA19.
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
  echo "=== jobid     : ${PBS_JOBID:-unset}"
  echo "=== workdir   : $(pwd)"
  echo "=== ranks     : ${NPROC}"
  echo "=== mpirun    : $(command -v mpirun)"
  echo "=== NQSV_MPIOPTS = [${NQSV_MPIOPTS:-}]"
  echo "=== host memory ==="
  free -g
  echo "=== GPU ==="
  nvidia-smi
  echo "=== openmx runtime libraries ==="
  ldd ../openmx | grep -E "libcud|libcublas|libcusolver|libmpi\."
} > "$ENVLOG" 2>&1

# Sample GPU *and* host memory every 30 s.  Host memory is the whole reason the
# rank count is 8, so if this run dies we want the trace that shows where it ran
# out rather than just an OOM kill with no context.
#
# "trap - ..." matters: a background subshell inherits the parent's traps, so if
# the parent ever sets a TERM handler the subshell would run it and then *resume
# the loop* instead of dying -- that is what left the previous two jobs sitting
# in RUN for 24 minutes after their calculation had finished, burning budget.
#
# 10 s, not 30 s: at 8 ranks the run went from 33 GiB to dead inside one 30 s
# gap, so the sample that mattered was never taken.
( trap - EXIT INT TERM
  for _ in $(seq 1 3000); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " $3 "/" $2}'
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    } >> "$SMILOG" 2>&1
    sleep 10
  done ) &
SMIPID=$!

#----------------------------------------------------------------------
# run
#----------------------------------------------------------------------
# --timeout is the guard against a repeat of 899514: when the OOM killer took a
# rank, prterun reported the death and then never returned, so the request sat
# in RUN burning budget until it was deleted by hand.  4 h is well past the
# expected run time and still well inside the 6 h elapstim.
mpirun ${NQSV_MPIOPTS:-} -np "${NPROC}" -npernode "${NPROC}" --timeout 10800 \
       ../openmx sidia444_nc.dat -nt 1 > "$STDOUT" 2> "$STDERR"
rc=$?

# Kill the sampler and the sleep it is parked in, then reap it, so the job ends
# when the calculation ends.
kill "$SMIPID" 2>/dev/null
pkill -P "$SMIPID" 2>/dev/null
wait "$SMIPID" 2>/dev/null

echo "=== openmx exit status = ${rc} ($(date)) ===" >> "$STDOUT"

echo "=== peak host memory (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$SMILOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== peak GPU ==="
grep -oE "^[0-9]+ %, [0-9]+ MiB" "$SMILOG" 2>/dev/null | sort -t, -k2 -rn | head -3
echo "=== tail of openmx stdout ==="
tail -25 "$STDOUT"

exit ${rc}
