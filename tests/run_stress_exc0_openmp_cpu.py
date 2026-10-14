#!/usr/bin/env python3
"""Compare the production Exc0 stress OpenMP region with one/four threads.

Only radial/XC helpers are deterministic stubs. The original quadrature,
thread-local buffers and nine-component accumulation execute with libgomp.
"""

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
static double KumoF(int n, double x, double *a, double *b, double *c)
{ return exp(-0.1 * exp(2.0*x)); }
static double Dr_KumoF(int n, double x, double r, double *a, double *b, double *c)
{ return -0.2 * r * exp(-0.1*r*r); }
static double XC_Ceperly_Alder(double den, int mode)
{ return mode == 0 ? -0.5*den : -den; }

static void run(int threads, double result[10])
{
    int OMPID,Nthrds,Nprocs,ir,ia,h_AN,Gh_AN,Rn,Hwan,i;
    double r,w,sumt,sumtx,sumty,sumtz,x,x0,y0,z0,x1,y1,z1,dx,dy,dz,r1;
    double den,den0,gden0,dx1,dy1,dz1,exc0,vxc0,sumt1[9];
    const int CoarseGL_Mesh = 128, Num_Leb_Grid = 6;
    const int Gc_AN = 1, Cwan = 0, F_Vxc_flag = 1, PCC_switch = 0;
    double Dr = 2.0, Sr = 2.0;
    double Leb_Grid_XYZW[6][4] = {
        {1,0,0,1.0/6}, {-1,0,0,1.0/6}, {0,1,0,1.0/6},
        {0,-1,0,1.0/6}, {0,0,1,1.0/6}, {0,0,-1,1.0/6}};
    double CoarseGL_Abscissae[128], CoarseGL_Weight[128];
    int FNAN[10] = {0,7}, natn[10][8] = {{0},{1,2,3,4,5,6,7,8}};
    int ncn[10][8] = {{0}}, WhatSpecies[10] = {0};
    double Gxyz[10][4] = {{0}}, atv[1][4] = {{0}};
    double radial[1] = {0}, *Spe_PAO_XV[1] = {radial};
    double *Spe_PAO_RV[1] = {radial}, *Spe_Atomic_Den2[1] = {radial};
    int Spe_Num_Mesh_PAO[1] = {1};
    double My_sumr[4] = {0}, sx[4][8] = {{0}}, sy[4][8] = {{0}}, sz[4][8] = {{0}};
    double *My_sumrx[4], *My_sumry[4], *My_sumrz[4];
    double My_Stress[9] = {0}, st[4][9] = {{0}}, *My_Stress_threads[4];
    for (int t = 0; t < 4; t++) {
        My_sumrx[t] = sx[t]; My_sumry[t] = sy[t]; My_sumrz[t] = sz[t];
        My_Stress_threads[t] = st[t];
    }
    for (int k = 0; k < 128; k++) {
        CoarseGL_Abscissae[k] = -0.99 + 1.98 * k / 127.0;
        CoarseGL_Weight[k] = 2.0 / 128;
    }
    for (int a = 2; a < 9; a++) {
        Gxyz[a][1] = 0.13 * a;
        Gxyz[a][2] = -0.17 * a;
        Gxyz[a][3] = 0.11 * a;
    }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
'''

SUFFIX = r'''
    memset(result, 0, sizeof(double) * 10);
    for (int t = 0; t < threads; t++) {
        result[0] += My_sumr[t];
        for (int k = 0; k < 9; k++) result[k+1] += st[t][k];
    }
}
int main(void)
{
    double expected[10], actual[10];
    run(1, expected);
    for (int trial = 0; trial < 32; trial++) {
        run(4, actual);
        for (int k = 0; k < 10; k++) {
            if (!isfinite(actual[k]) || fabs(actual[k] - expected[k]) > 1e-12) {
                fprintf(stderr, "FAIL trial=%d component=%d value=%.17g expected=%.17g\n",
                        trial, k, actual[k], expected[k]);
                return 1;
            }
        }
    }
    puts("PASS: 32 Exc0 stress OpenMP comparisons; energy and nine stress components");
    return 0;
}
'''


def region(source):
    start = source.index("#pragma omp parallel shared(Leb_Grid_XYZW")
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / "source/Stress.c").read_text()
    with tempfile.TemporaryDirectory(prefix="openmx_stress_exc0_") as directory:
        cfile = Path(directory) / "check.c"
        binary = Path(directory) / "check"
        cfile.write_text(PREFIX + region(source) + SUFFIX)
        command = shlex.split(os.environ.get("CC", "gcc"))
        # Keep the loop-counter loads/stores visible: optimization can hide
        # the original race by keeping the shared counter in a register.
        command += ["-std=c11", "-O0", "-g", "-fopenmp"]
        command += shlex.split(os.environ.get("CFLAGS", ""))
        subprocess.run(command + [str(cfile), "-lm", "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
