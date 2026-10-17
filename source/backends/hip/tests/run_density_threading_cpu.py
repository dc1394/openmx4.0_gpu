#!/usr/bin/env python3
"""Exercise the actual CPU density fallback with varying OpenMP teams.

Checks local/halo and contracted orbitals, spin 0/1/3, four-point and remainder
loops, empty overlaps and empty ranks against a direct density reference.
AMD already partitions whole atoms among threads and allocates scratch inside
each actual team. Exercise its scalar and available SIMD paths. CC/CFLAGS are
honored.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "source/Set_Density_Grid.c").read_text()
start = source.index("#pragma omp parallel shared(myid,G2ID,Orbs_Grid_FNAN", source.index("calculate Tmp_Den_Grid"))
end = source.index("  } /* !use_local_gpu */", start)
body = source[start:end]
helpers = source[source.index("#define  SDG_NC_GRID_BATCH_AVX2"):source.index("int Set_Density_Grid_GPU_Local_Prepare")]

PRELUDE = r'''
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define ATOMS 4
#define POINTS 9
#define ORBITALS 3
static int Matomnum = 2, SpinP_switch, Cnt_switch, Cnt_kind, myid;
static int List_YOUSO[64], M2G[ATOMS] = {0,1,2,3};
static int WhatSpecies[ATOMS] = {0,0,1,1};
static int Spe_Total_CNO[2] = {2,3};
static int GridN_Atom[ATOMS] = {0,9,7,8};
static int G2ID[ATOMS] = {0,0,0,1}, F_G2M[ATOMS] = {0,1,2,3};
static int FNAN[ATOMS] = {0,2,2,0};
static int natn[ATOMS][3] = {{0},{1,2,3},{2,1,3},{0}};
static int ncn[ATOMS][3], NumOLG[ATOMS][3];
static int GListTAtoms1[ATOMS][3][POINTS], GListTAtoms2[ATOMS][3][POINTS];
static float Orbs_Grid[ATOMS][POINTS][ORBITALS];
static float Orbs_Grid_FNAN[ATOMS][3][POINTS][ORBITALS];
static float COrbs_Grid[ATOMS][ORBITALS][POINTS];
static double CDM[4][ATOMS][3][ORBITALS][ORBITALS];
static double Tmp_Den_Grid[4][ATOMS][POINTS], reference[4][ATOMS][POINTS];
static double time_per_atom[ATOMS];
static int mpi_comm_level1;
static void MPI_Abort(int comm, int code) { (void)comm; exit(code); }
static int simd_case;
static _Thread_local int clock_tick;
static void dtime(double *t) { *t = (clock_tick++ % 2 == 0) ? 1.0 : 2.0; }
'''

PRELUDE += helpers + r'''
static void run_density(void)
{
    int Mc_AN,h_AN,Gc_AN,Cwan,NO0,Gh_AN,Mh_AN,Rnh,Hwan,NO1;
    int spin,i,j,Nog,Nc_0,Nc_1,Nc_2,Nc_3,Nh_0,Nh_1,Nh_2,Nh_3,Nc,Nh;
    double Stime_atom,Etime_atom;
    double ***tmp_CDM;
    double *orbs0_0,*orbs0_1,*orbs0_2,*orbs0_3;
    double *orbs1_0,*orbs1_1,*orbs1_2,*orbs1_3,*orbs0,*orbs1;
    double sum_0,sum_1,sum_2,sum_3,tmp0_0,tmp0_1,tmp0_2,tmp0_3,sum,tmp0;
    int nc_simd_mode = simd_case && SpinP_switch==3 ? SDG_nc_simd_mode() : 0;
    int col_simd_mode = simd_case && SpinP_switch!=3 ? SDG_col_simd_mode() : 0;
    int use_nc_simd = nc_simd_mode!=0, use_col_simd = col_simd_mode!=0;
    int nc_grid_batch = nc_simd_mode==SDG_NC_SIMD_AVX512 ? 8 : 4;
    int col_grid_batch = col_simd_mode==SDG_NC_SIMD_AVX512 ? 8 : 4;
'''

HARNESS = r'''
}
static void fixture(void)
{
    List_YOUSO[7] = ORBITALS;
    for (int atom = 1; atom < ATOMS; ++atom)
        for (int pt = 0; pt < POINTS; ++pt)
            for (int i = 0; i < ORBITALS; ++i) {
                Orbs_Grid[atom][pt][i] = (float)(atom + pt + i) / 16.0f;
                COrbs_Grid[atom][i][pt] = (float)(2*atom - pt + i) / 32.0f;
            }
    for (int atom = 1; atom <= 2; ++atom)
        for (int h = 0; h < 3; ++h) {
            int other = natn[atom][h];
            NumOLG[atom][h] = h == 0 ? GridN_Atom[atom] : (h == 1 ? 5 : (atom == 1 ? 3 : 0));
            for (int pt = 0; pt < NumOLG[atom][h]; ++pt) {
                GListTAtoms1[atom][h][pt] = (pt + h) % GridN_Atom[atom];
                GListTAtoms2[atom][h][pt] = (2*pt + h) % GridN_Atom[other];
                for (int i = 0; i < ORBITALS; ++i)
                    Orbs_Grid_FNAN[atom][h][pt][i] = (float)(atom + 2*pt - i) / 16.0f;
            }
            for (int spin = 0; spin < 4; ++spin)
                for (int i = 0; i < ORBITALS; ++i)
                    for (int j = 0; j < ORBITALS; ++j)
                        CDM[spin][atom][h][i][j] = 0.013 * (1+spin+atom+3*h+i-j);
        }
}

static void reference_density(void)
{
    memset(reference, 0, sizeof(reference));
    for (int atom = 1; atom <= Matomnum; ++atom)
        for (int h = 0; h <= FNAN[atom]; ++h) {
            int other = natn[atom][h];
            for (int pt = 0; pt < NumOLG[atom][h]; ++pt) {
                int nc = GListTAtoms1[atom][h][pt], nh = GListTAtoms2[atom][h][pt];
                for (int spin = 0; spin <= SpinP_switch; ++spin)
                    for (int i = 0; i < Spe_Total_CNO[WhatSpecies[atom]]; ++i)
                        for (int j = 0; j < Spe_Total_CNO[WhatSpecies[other]]; ++j) {
                            double a = Cnt_switch ? COrbs_Grid[atom][i][nc] : Orbs_Grid[atom][nc][i];
                            double b = Cnt_switch ? COrbs_Grid[other][j][nh] :
                                       G2ID[other] == myid ? Orbs_Grid[other][nh][j] :
                                       Orbs_Grid_FNAN[atom][h][pt][j];
                            reference[spin][atom][nc] += a * CDM[spin][atom][h][i][j] * b;
                        }
            }
        }
}

int main(void)
{
    fixture();
    omp_set_dynamic(0);
    for (simd_case=0; simd_case<2; ++simd_case)
    for (int empty = 0; empty < 2; ++empty) {
        Matomnum = empty ? 0 : 2;
        for (int threads = 1; threads <= 4; threads += 3) {
            omp_set_num_threads(threads);
            for (int spin_case = 0; spin_case < 3; ++spin_case) {
                SpinP_switch = spin_case == 2 ? 3 : spin_case;
                for (Cnt_switch = 0; Cnt_switch < 2; ++Cnt_switch) {
                    memset(Tmp_Den_Grid, 0, sizeof(Tmp_Den_Grid));
                    memset(time_per_atom, 0, sizeof(time_per_atom));
                    reference_density();
                    run_density();
                    for (int atom = 1; atom <= Matomnum; ++atom) {
                        assert(time_per_atom[atom] == 1.0);
                        for (int spin = 0; spin <= SpinP_switch; ++spin)
                            for (int pt = 0; pt < GridN_Atom[atom]; ++pt)
                                assert(fabs(Tmp_Den_Grid[spin][atom][pt] - reference[spin][atom][pt]) < 1e-13);
                    }
                }
            }
        }
    }
    puts("PASS: 48 CPU density cases with scalar/SIMD modes, varying teams, spins, contracted/halo orbitals and empty ranks");
    return 0;
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-density-threads-") as tmp:
    src, exe = Path(tmp) / "check.c", Path(tmp) / "check"
    src.write_text(PRELUDE + body + HARNESS)
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = shlex.split(os.environ.get("CFLAGS", "-O2 -g"))
    subprocess.run(compiler + flags + ["-std=c11", "-fopenmp", str(src), "-lm", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
