#!/bin/bash
#------- qsub option -----------
#PBS -A GPU2026
#PBS -q gen_S
#PBS -l elapstim_req=00:10:00
#PBS -v OMP_NUM_THREADS=48
#------- Program execution -----------
cd $PBS_O_WORKDIR
./a.ou
