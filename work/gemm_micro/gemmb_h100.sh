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
#PBS -N gemmb_h100
#PBS -j o
#PBS -o gemmb_h100.joblog

# M2 single-GEMM microbenchmark, H100 leg (mirrors gemm_micro_5080.csv):
# ./bench_gemm (built sm_90 against the campaign libgemmul8.a v3.2.0
# 833e5761 -- the exact library inside the v2.0_thesis openmx binary) over
# shapes_h100.txt = the full 5080 shape grid + the H100 production-matrix
# systems (bcol300, ccol360, cnc300).  Single process, idle dedicated GPU,
# no MPS, 7 reps interleaved ABAB, num_moduli 15, fastmode off -- identical
# method to the 5080 run.  CSV to gemm_micro_h100.csv, bench's own header
# line (GPU, config) to gemm_micro_h100.log, thermals to bench_env_h100.txt.

set -u
ulimit -c 0
cd "${PBS_O_WORKDIR}" || exit 1

command -v module >/dev/null 2>&1 && module purge >/dev/null 2>&1
unset LD_LIBRARY_PATH

{
  nvidia-smi --query-gpu=name,temperature.gpu,clocks.sm,memory.used --format=csv,noheader
  date
} > bench_env_h100.txt 2>&1

./bench_gemm shapes_h100.txt 7 15 > gemm_micro_h100.csv 2> gemm_micro_h100.log
rc=$?

{
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader
  echo "host $(hostname), jobid ${PBS_JOBID:-unset}, exit ${rc}, $(date)"
} >> bench_env_h100.txt 2>&1

echo "=== exit ${rc}; csv lines: $(wc -l < gemm_micro_h100.csv) ==="
tail -3 gemm_micro_h100.csv
exit ${rc}
