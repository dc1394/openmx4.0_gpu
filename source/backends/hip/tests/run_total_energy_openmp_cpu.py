#!/usr/bin/env python3
"""Exercise production energy/stress loops with changing OpenMP teams.

GNU OpenMP's parallel launcher is wrapped to make the first team smaller than
later teams deterministically. MPI is stubbed for one rank; real OpenMP threads
check the energy and concurrent per-atom timing updates. Requires GCC/libgomp.
--sanitize enables ASan/UBSan. This does not exercise HIP or MPI execution.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


PREFIX = r'''
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PAIRS 32768
#define MPI_DOUBLE 0
#define MPI_SUM 0
static int mpi_comm_level1;
static void MPI_Comm_size(int comm, int *n) { *n = 1; }
static void MPI_Comm_rank(int comm, int *n) { *n = 0; }
static void MPI_Allreduce(const void *src, void *dst, int n, int type, int op, int comm)
{ memcpy(dst, src, (size_t)n * sizeof(double)); }
static void dtime(double *t)
{
    static _Thread_local unsigned long clock_value;
    *t = (double)clock_value++;
}

static int calls, small_first;
void __real_GOMP_parallel(void (*fn)(void *), void *data, unsigned n, unsigned flags);
void __wrap_GOMP_parallel(void (*fn)(void *), void *data, unsigned n, unsigned flags)
{
    /* A dynamic runtime is allowed to shrink a team without shrinking the
       following one. Force that sequence without relying on machine load. */
    __real_GOMP_parallel(fn, data, small_first && calls++ == 0 ? 1 : 4, flags);
}

static int SpinP_switch, Cnt_switch, Spe_Total_CNO[1] = {1};
static int WhatSpecies[2], M2G[2] = {0, 1}, neighbor[PAIRS];
static int *natn[2] = {NULL, neighbor};
static int OneD_Nloop = PAIRS, OneD2Mc_AN[PAIRS], OneD2h_AN[PAIRS];
static double time_per_atom[2];
static double hvalue[PAIRS], dmvalue[2][PAIRS];
static double *hrow[PAIRS][1], **hpair[PAIRS], ***hatom[2], ****hdir[1];
static double *****H0 = hdir, *****CntH0 = hdir;
static double *drow[2][PAIRS][1], **dpair[2][PAIRS], ***datom[2][2];
static double ****dspin[2], *****dstep[1], ******DM = dstep;
static int Matomnum = 1, FNAN[2] = {0, PAIRS - 1};
static int cells[PAIRS], *ncn[2] = {NULL, cells}, level_stdout, Host_ID;
static double Spe_Core_Charge[1] = {2.0}, atv[1][4] = {{0, 1.0, 0, 0}};
static double atom_xyz[44], distances[PAIRS];
static double *Gxyz[2] = {NULL, atom_xyz}, *Dis[2] = {NULL, distances};
static double ****DecEscc, Stress_Tensor[9];
'''

MAIN = r'''
int main(void)
{
    double hsum = 0.0;
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    hdir[0] = hatom;
    hatom[1] = hpair;
    dstep[0] = dspin;
    for (int s = 0; s < 2; s++) {
        dspin[s] = datom[s];
        datom[s][1] = dpair[s];
    }
    for (int i = 0; i < PAIRS; i++) {
        OneD2Mc_AN[i] = neighbor[i] = 1;
        OneD2h_AN[i] = i;
        distances[i] = 1.0;
        hvalue[i] = 1.0 + (i % 3);
        hsum += hvalue[i];
        hpair[i] = hrow[i];
        hrow[i][0] = &hvalue[i];
        for (int s = 0; s < 2; s++) {
            dmvalue[s][i] = s + 1.0;
            dpair[s][i] = drow[s][i];
            drow[s][i][0] = &dmvalue[s][i];
        }
    }
    for (small_first = 0; small_first <= 1; small_first++)
        for (Cnt_switch = 0; Cnt_switch <= 1; Cnt_switch++)
            for (int spin = 0; spin < 3; spin++)
                for (int repeat = 0; repeat < 8; repeat++) {
                    SpinP_switch = spin == 2 ? 3 : spin;
                    calls = 0;
                    time_per_atom[1] = 0.0;
                    double value = Calc_Ekin();
                    double energy = (spin == 0 ? 2.0 : 3.0) * hsum;
                    double timing = (spin == 1 ? 2.0 : 1.0) * PAIRS;
                    if (value != energy || time_per_atom[1] != timing) {
                        fprintf(stderr, "FAIL spin=%d contracted=%d first-small=%d "
                            "energy=%.17g expected=%.17g time=%.0f expected=%.0f\n",
                            SpinP_switch, Cnt_switch, small_first,
                            value, energy, time_per_atom[1], timing);
                        return 1;
                    }
                }
    for (small_first = 0; small_first <= 1; small_first++)
        for (int repeat = 0; repeat < 8; repeat++) {
            calls = 0;
            time_per_atom[1] = 0.0;
            memset(Stress_Tensor, 0, sizeof(Stress_Tensor));
            Stress_core();
            for (int k = 0; k < 9; k++) {
                double expected = k == 0 ? -2.0 * (PAIRS - 1) : 0.0;
                if (Stress_Tensor[k] != expected || time_per_atom[1] != 1.0) {
                    fprintf(stderr, "FAIL stress[%d]=%.17g expected=%.17g time=%.0f\n",
                            k, Stress_Tensor[k], expected, time_per_atom[1]);
                    return 1;
                }
            }
        }
    puts("PASS: 112 OpenMP energy/stress cases, dynamic team bounds and atom timing");
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root / "source/Total_Energy.c").read_text()
    start = source.index("\ndouble Calc_Ekin()\n")
    end = source.index("\ndouble Calc_Ena()\n", start)
    stress = (root / "source/Stress.c").read_text()
    stress_start = stress.index("\nvoid Stress_core()\n")
    stress_end = stress.index("\nvoid Stress_H0()\n", stress_start)
    with tempfile.TemporaryDirectory(prefix="openmx_energy_omp_") as directory:
        cfile = Path(directory) / "check.c"
        binary = Path(directory) / "check"
        cfile.write_text(PREFIX + source[start:end] + stress[stress_start:stress_end] + MAIN)
        flags = ["-std=c11", "-O1", "-g", "-fopenmp"]
        if args.sanitize:
            flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        command = shlex.split(os.environ.get("CC", "gcc")) + flags
        command += shlex.split(os.environ.get("CFLAGS", ""))
        subprocess.run(command + [str(cfile), "-Wl,--wrap=GOMP_parallel", "-lm",
                                  "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
