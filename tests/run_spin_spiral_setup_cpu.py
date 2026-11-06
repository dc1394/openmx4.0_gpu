#!/usr/bin/env python3
"""Check the production spin-spiral mesh and eigenstate bounds under ASan/UBSan.

Extract the unchanged dimension and mesh setup, including its allocations.
Explicit quadrature nodes distinguish Gamma-centered and regular meshes;
small basis sets reproduce the formerly oversized eigensolver request.
No MPI, GPU, numerical library or PAO/VPS data is required.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

from native_state_test_support import bounded_stack, compiler_command
from run_cluster_col_state_cpu import function

ROOT = Path(__file__).resolve().parents[1]

PREFIX = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL: %s\n",#c); exit(2); } } while (0)
static void *allocations[8192];
static size_t allocation_count;
static void *tracked_malloc(size_t bytes) {
    CHECK(allocation_count < sizeof(allocations)/sizeof(*allocations));
    void *p = malloc(bytes);
    CHECK(p != NULL);
    allocations[allocation_count++] = p;
    return p;
}
static void release_allocations(void) {
    while (allocation_count) free(allocations[--allocation_count]);
}
#define malloc tracked_malloc
static int atomnum = 1, WhatSpecies[2] = {0,0}, Spe_Total_CNO[1];
static double Spe_Core_Charge[1], system_charge;
static int way_of_kpoint, num_non_eq_kpt = 3, level_stdout = 0, Host_ID = 0;
static double Shift_K_Point;
static double NE_KGrids1[] = {0.0,0.25,0.5};
static double NE_KGrids2[] = {0.0,0.125,0.25};
static double NE_KGrids3[] = {0.0,-0.125,-0.25};
static int NE_T_k_op[] = {2,3,5};
'''

SETUP = r'''
static void check_setup(int knum_i, int knum_j, int knum_k,
                        const double *x, const double *y, const double *z,
                        int expected_states) {
    int i,j,k,n,n2,wan,wanA,MaxN,ii,ij,ik,myid0=0;
    int T_knum=0,***k_op,*T_k_op=NULL,*T_k_ID=NULL;
    double TZ,lumos,k1,snum_i,snum_j,snum_k,sum_weights;
    double *KGrids1,*KGrids2,*KGrids3;
    double *T_KGrids1=NULL,*T_KGrids2=NULL,*T_KGrids3=NULL,**EIGEN=NULL;
'''

CHECKS = r'''
    CHECK(MaxN == expected_states);
    CHECK(MaxN > 0 && MaxN <= 2*n && MaxN+1 <= n2);
    if (way_of_kpoint == 2) {
        CHECK(T_knum == 3 && sum_weights == 10.0);
        for (int p=0;p<3;p++) {
            CHECK(T_KGrids1[p] == NE_KGrids1[p]);
            CHECK(T_KGrids2[p] == NE_KGrids2[p]);
            CHECK(T_KGrids3[p] == NE_KGrids3[p]);
            CHECK(T_k_op[p] == NE_T_k_op[p]);
        }
    } else {
        CHECK(T_knum == knum_i*knum_j*knum_k);
        CHECK(sum_weights == (double)T_knum);
        int p=0;
        for (int a=0;a<knum_i;a++) for (int b=0;b<knum_j;b++)
            for (int c=0;c<knum_k;c++,p++) {
                CHECK(fabs(T_KGrids1[p]-x[a]) < 1e-14);
                CHECK(fabs(T_KGrids2[p]-y[b]) < 1e-14);
                CHECK(fabs(T_KGrids3[p]-z[c]) < 1e-14);
                CHECK(T_k_op[p] == 1);
            }
    }
    for (int p=0;p<T_knum;p++)
        for (int state=0;state<n2;state++) CHECK(EIGEN[p][state] == 1e5);
    release_allocations();
}
int main(int argc,char **argv) {
    const int mode=argc>1 ? atoi(argv[1]) : 3;
    const double zero[]={0.0};
    const double gamma2[]={0.0,0.5}, regular2[]={-0.25,0.25};
    const double gamma3[]={0.0,1.0/3.0,2.0/3.0};
    const double regular3[]={-1.0/3.0,0.0,1.0/3.0};
    const double shifted_gx[]={0.125,0.625}, shifted_gy[]={-0.125,5.0/24.0,13.0/24.0};
    const double shifted_rx[]={-0.125,0.375}, shifted_ry[]={-11.0/24.0,-0.125,5.0/24.0};
    const double shifted_gz[]={0.25,7.0/12.0,11.0/12.0};
    const double shifted_rz[]={-1.0/12.0,0.25,7.0/12.0};
    CHECK(mode==1 || mode==2 || mode==3);
    way_of_kpoint=mode;
    Spe_Total_CNO[0]=1;Spe_Core_Charge[0]=1.0;
    check_setup(2,1,1,mode==3?gamma2:regular2,zero,zero,2);
    check_setup(1,3,1,zero,mode==3?gamma3:regular3,zero,2);
    check_setup(1,1,3,zero,zero,mode==3?gamma3:regular3,2);
    /* Fe4.0H-s3p3d3f2 has 41 spatial orbitals and 16 valence electrons. */
    Spe_Total_CNO[0]=41;Spe_Core_Charge[0]=16.0;
    Shift_K_Point=0.125;
    check_setup(2,3,1,mode==3?shifted_gx:shifted_rx,
                mode==3?shifted_gy:shifted_ry,zero,82);
    check_setup(1,1,3,zero,zero,mode==3?shifted_gz:shifted_rz,82);
    /* A large basis still requests the intended occupied + empty states. */
    Spe_Total_CNO[0]=300;Spe_Core_Charge[0]=300.0;
    Shift_K_Point=0.0;
    check_setup(3,3,3,mode==3?gamma3:regular3,
                mode==3?gamma3:regular3,mode==3?gamma3:regular3,360);
    puts("PASS: production spin-spiral mesh, weights and physical eigenstate bounds");
    return 0;
}
'''


def setup_blocks(text):
    dimension_start = text.index('  n = 0;')
    dimension_end = text.index('  /****************************************************\n   Allocation', dimension_start)
    allocation_start = text.index('  KGrids1 = (double*)malloc')
    allocation_end = text.index('  MP = (int*)malloc', allocation_start)
    mesh_start = text.index('  if (way_of_kpoint==')
    mesh_end = text.rfind('  /***********************************************', mesh_start,
                          text.index('allocate k-points into processors', mesh_start))
    assert mesh_end > mesh_start
    return text[dimension_start:dimension_end] + text[allocation_start:allocation_end] + text[mesh_start:mesh_end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    text = (ROOT / 'source/Band_DFT_NonCol_GB.c').read_text()
    inversion = function((ROOT / 'source/openmx_common.c').read_text(), 'k_inversion')
    blocks = setup_blocks(text)
    with tempfile.TemporaryDirectory(prefix='openmx-spin-spiral-') as directory:
        folder = Path(directory)
        source, executable = folder/'check.c', folder/'check'
        command = compiler_command('c') + ['-x','c',str(source),'-lm','-o',str(executable)]

        def build(body):
            source.write_text(PREFIX + inversion + SETUP + body + CHECKS)
            subprocess.run(command,check=True)

        build(blocks)
        for mode in ['3','1','2']:
            subprocess.run([str(executable),mode],check=True,preexec_fn=bounded_stack)
        if args.negative_control:
            mutations = [
                (' || way_of_kpoint==3','', '3','T_knum == knum_i*knum_j*knum_k'),
                ('if (2*n<MaxN) MaxN = 2*n;', 'if (n2<MaxN) MaxN = n2;', '1','MaxN == expected_states'),
            ]
            for before,after,mode,diagnostic in mutations:
                assert blocks.count(before)==1, before
                build(blocks.replace(before,after))
                result=subprocess.run([str(executable),mode],capture_output=True,text=True,
                                      preexec_fn=bounded_stack)
                assert result.returncode==2 and diagnostic in result.stderr, result
            print('PASS: both former bugs are rejected by independent negative controls')


if __name__ == '__main__':
    main()
