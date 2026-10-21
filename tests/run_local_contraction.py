#!/usr/bin/env python3
"""Check native contraction against dense C H C^T, including all NC channels.

Compile the actual shared production functions with ASan/UBSan. Unequal
species, reordered local atoms, nontrivial orbital maps and a halo neighbor
exercise the flat coefficient layout without initializing MPI or a GPU.
"""
import os
from pathlib import Path
import resource
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "openmx_core_local_basis.h"
static int Matomnum=2, SpinP_switch, List_YOUSO[60];
static int M2G[3]={0,2,1},F_G2M[4]={0,2,1,3},WhatSpecies[4]={0,0,1,0};
static int FNAN[4]={0,1,1,0},n1[2]={1,2},n2[2]={2,3},*natn[4]={NULL,n1,n2,NULL};
static int Spe_Total_CNO[2]={2,1},counts[2][2]={{2,1},{2,0}};
static int *Spe_Specified_Num[2]={counts[0],counts[1]};
static int maps[2][2][2]={{{0,2},{1,0}},{{1,0},{0,0}}};
static int *map_rows[2][2]={{maps[0][0],maps[0][1]},{maps[1][0],maps[1][1]}};
static int **Spe_Trans_Orbital[2]={map_rows[0],map_rows[1]};
/* Legacy operator adapters belong to the test. Production coefficients have
   only a flat view, so a hidden dependence on CntCoes fails to compile. */
typedef struct {
  double values[4][3][2][3][3];
  double *rows[4][3][2][3],**blocks[4][3][2],***atoms[4][3],****spins[4];
} Matrix;
static void initialize(Matrix *m,double offset) {
  for(int s=0;s<4;s++) {
    m->spins[s]=m->atoms[s];
    for(int a=0;a<3;a++) {
      m->atoms[s][a]=m->blocks[s][a];
      for(int h=0;h<2;h++) {
        m->blocks[s][a][h]=m->rows[s][a][h];
        for(int i=0;i<3;i++) {
          m->rows[s][a][h][i]=m->values[s][a][h][i];
          for(int j=0;j<3;j++) m->values[s][a][h][i][j]=offset+s*.7+a*.13-h*.21+i*.31-j*.11;
        }
      }
    }
  }
}
static void equal(double actual,long double expected) {
  if(!isfinite(actual) || fabsl(actual-expected)>2e-13L*(1+fabsl(expected))) {
    fprintf(stderr,"contraction: %.17g != %.21Lg\n",actual,expected);abort();
  }
}
'''

CHECK = r'''
int main(void) {
  Matrix h,overlap,result,contracted_overlap;
  double coefficients[4][2][2],dense[4][2][3]={0};
  OpenMXLocalBasisViews local={0};
  local.tensors[OPENMX_LOCAL_CONTRACTION]=(OpenMXLocalTensor3){4,2,2,&coefficients[0][0][0]};
  for(int a=1;a<=3;a++) {
    int species=a==1 ? 1 : 0;
    for(int i=0;i<2;i++) for(int p=0;p<2;p++) coefficients[a][i][p]=.4+a*.1-i*.3-p*.9;
    for(int i=0;i<Spe_Total_CNO[species];i++)
      for(int p=0;p<counts[species][i];p++) dense[a][i][maps[species][i][p]]=coefficients[a][i][p];
  }
  int comparisons=0;
  for(int mode=0;mode<4;mode++) {
    SpinP_switch=mode<3 ? (int[]){0,1,3}[mode] : 3;
    const int components=mode==3 ? 3 : SpinP_switch+1;
    initialize(&h,1.2);initialize(&overlap,-.7);
    initialize(&result,-123);initialize(&contracted_overlap,-456);
    List_YOUSO[5]=components;
    if(mode==3) Contract_iHNL(h.spins,result.spins,&local);
    else Contract_Hamiltonian(h.spins,result.spins,overlap.spins,contracted_overlap.spins,&local);
    for(int a=1;a<=2;a++) for(int n=0;n<2;n++) {
      int global=M2G[a],neighbor=natn[global][n],b=F_G2M[neighbor];
      int s0=WhatSpecies[global],s1=WhatSpecies[neighbor];
      for(int i=0;i<Spe_Total_CNO[s0];i++) for(int j=0;j<Spe_Total_CNO[s1];j++) {
        for(int spin=0;spin<components;spin++) {
          long double expected=0;
          for(int p=0;p<3;p++) for(int q=0;q<3;q++)
            expected+=(long double)dense[a][i][p]*h.values[spin][a][n][p][q]*dense[b][j][q];
          equal(result.values[spin][a][n][i][j],expected);++comparisons;
        }
        if(mode!=3) {
          long double expected=0;
          for(int p=0;p<3;p++) for(int q=0;q<3;q++)
            expected+=(long double)dense[a][i][p]*overlap.values[0][a][n][p][q]*dense[b][j][q];
          equal(contracted_overlap.values[0][a][n][i][j],expected);++comparisons;
        }
      }
    }
    /* Unused channels and padding may not be written. */
    for(int spin=0;spin<4;spin++) for(int a=0;a<3;a++) for(int n=0;n<2;n++)
      for(int i=0;i<3;i++) for(int j=0;j<3;j++) {
        int written=a && spin<components;
        if(written) written=i<Spe_Total_CNO[WhatSpecies[M2G[a]]] &&
          j<Spe_Total_CNO[WhatSpecies[natn[M2G[a]][n]]];
        if(!written) equal(result.values[spin][a][n][i][j],-123+spin*.7+a*.13-n*.21+i*.31-j*.11);
      }
  }
  printf("PASS: %d dense contraction values, spin 0/1/3, imaginary SOC, mixed species and halo coefficients\n",comparisons);
  return 0;
}
'''

with tempfile.TemporaryDirectory(prefix="openmx-local-contraction-") as directory:
    folder = Path(directory)
    parts = [HARNESS]
    for name in ("Contract_Hamiltonian.c", "Contract_iHNL.c"):
        parts.append((ROOT / "source" / name).read_text().replace('#include "openmx_common.h"', ''))
    (folder / "check.c").write_text("\n".join(parts) + CHECK)
    command = shlex.split(os.environ.get("CC", "cc")) + [
        "-std=c11", "-O1", "-g", "-fno-pie", "-no-pie", "-ffp-contract=off",
        "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
        "-I" + str(ROOT / "source"), str(folder / "check.c"), "-lm", "-o", str(folder / "check")]
    subprocess.run(command, check=True)

    def bounded_stack():
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))

    subprocess.run([str(folder / "check")], check=True, timeout=30, preexec_fn=bounded_stack)
