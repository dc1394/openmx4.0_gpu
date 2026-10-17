#!/usr/bin/env python3
"""Compare the density recovery adapter with the original pair/grid sum."""
import ast
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'tests/run_gpu_vector_cpu.py').read_text())
prefix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='prefix' for t in n.targets))
body=(root/'source/openmx_gpu_bilinear.c').read_text().replace('#include "hip_runtime_compat.h"','').replace('#include "set_hip_default_device_from_local_rank.h"','')
sdg=(root/'source/Set_Density_Grid_GPU.c').read_text()
sdg=sdg[sdg.index('typedef struct {int mc,h,point;} SDGRecoveryTerm;'):sdg.index('static int SDG_env_bool(')]
fixture=r'''
#ifndef ORB_DOUBLE
typedef float Type_Orbs_Grid;
#else
typedef double Type_Orbs_Grid;
#endif
#define GPUSOLVER 7
#define MPI_INT 0
#define MPI_MAX 0
static int scf_eigen_lib_flag=GPUSOLVER,SpinP_switch,Cnt_switch,Matomnum=3,mpi_comm_level1=0;
static int M2G[5]={0,2,1,3},F_G2M[5]={0,2,1,3,4},G2ID[5]={0,0,0,0,1};
static int WhatSpecies[5]={0,0,1,0,1},Spe_Total_CNO[2]={2,3},GridN_Atom[5]={0,6,7,5,6},FNAN[5]={0,1,1,1,1};
static int nn[5][2],*natn[5],ng[5][2],*NumOLG[5],g1[5][2][7],g2[5][2][7],*g1p[5][2],*g2p[5][2],**GListTAtoms1[5],**GListTAtoms2[5];
static Type_Orbs_Grid o[5][7][3],co[5][3][7],fo[5][2][7][3];
static Type_Orbs_Grid *op[5][7],**Orbs_Grid[5],*cp[5][3],**COrbs_Grid[5],*fp[5][2][7],**fpp[5][2],***Orbs_Grid_FNAN[5];
static double d[4][5][2][3][3],*dp[4][5][2][3],**dpp[4][5][2],***dppp[4][5],****dm[4];
static double den[4][5][9],*denp[4][5],**density[4],want[4][5][7];
static int MPI_Comm_rank(int comm,int *rank){(void)comm;*rank=0;return 0;}
static int MPI_Allreduce(const int *in,int *out,int n,int type,int op,int comm){(void)n;(void)type;(void)op;(void)comm;*out=*in;return 0;}
unsigned long long openmx_host_mem_available(void){return 1ULL<<36;}
int openmx_gpu_local_size_noncollective(void){return 2;}
'''
main=r'''
int main(void){
 for(int g=0;g<5;g++){nn[g][0]=g;nn[g][1]=g%4+1;}
 for(int mc=0;mc<5;mc++){
  natn[mc]=nn[mc];nn[mc][0]=mc;nn[mc][1]=mc%4+1;
  Orbs_Grid[mc]=op[mc];COrbs_Grid[mc]=cp[mc];Orbs_Grid_FNAN[mc]=fpp[mc];
  NumOLG[mc]=ng[mc];GListTAtoms1[mc]=g1p[mc];GListTAtoms2[mc]=g2p[mc];
  for(int g=0;g<7;g++){
   op[mc][g]=o[mc][g];for(int i=0;i<3;i++){o[mc][g][i]=.07*(mc+g+i);co[mc][i][g]=.03*(mc-g+2*i);}
  }
  for(int i=0;i<3;i++)cp[mc][i]=co[mc][i];
  for(int h=0;h<2;h++){
   fpp[mc][h]=fp[mc][h];g1p[mc][h]=g1[mc][h];g2p[mc][h]=g2[mc][h];
   ng[mc][h]=mc?4+h:0;
   int gc=mc<4?M2G[mc]:4,gh=nn[gc][h];
   for(int k=0;k<7;k++){
    g1[mc][h][k]=mc?(k*2+1)%GridN_Atom[gc]:0;
    g2[mc][h][k]=mc?(k*3+1)%GridN_Atom[gh]:0;fp[mc][h][k]=fo[mc][h][k];
    for(int j=0;j<3;j++)fo[mc][h][k][j]=.09*(mc-h+k-j);
   }
  }
 }
 for(int s=0;s<4;s++){
  dm[s]=dppp[s];density[s]=denp[s];
  for(int mc=0;mc<5;mc++){
   dppp[s][mc]=dpp[s][mc];denp[s][mc]=den[s][mc]+1;
   for(int h=0;h<2;h++){
    dpp[s][mc][h]=dp[s][mc][h];
    for(int i=0;i<3;i++){dp[s][mc][h][i]=d[s][mc][h][i];for(int j=0;j<3;j++)d[s][mc][h][i][j]=.11*(1+s+mc+h-i+j);}
   }
  }
 }
 int cases=0;
 for(int spins=1;spins<=4;spins*=2)for(Cnt_switch=0;Cnt_switch<=1;Cnt_switch++){
  SpinP_switch=spins-1;memset(want,0,sizeof(want));
  for(int mc=1;mc<=3;mc++){
   int gc=M2G[mc],no0=Spe_Total_CNO[WhatSpecies[gc]];
   for(int h=0;h<=FNAN[gc];h++){
    int gh=natn[gc][h],mh=F_G2M[gh],no1=Spe_Total_CNO[WhatSpecies[gh]];
    for(int k=0;k<NumOLG[mc][h];k++){
     int nc=GListTAtoms1[mc][h][k],nh=GListTAtoms2[mc][h][k];
     for(int s=0;s<spins;s++)for(int i=0;i<no0;i++)for(int j=0;j<no1;j++){
      double a=Cnt_switch?co[mc][i][nc]:o[mc][nc][i];
      double b=Cnt_switch?co[mh][j][nh]:G2ID[gh]?fo[mc][h][k][j]:o[mh][nh][j];
      want[s][mc][nc]+=a*d[s][mc][h][i][j]*b;
     }
    }
   }
  }
  for(workers=1;workers<=4;workers++)for(int tier=0;tier<4;tier++){
   local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;
   for(int s=0;s<4;s++)for(int mc=0;mc<5;mc++)for(int g=0;g<9;g++)den[s][mc][g]=987;
   int ok=Set_Density_Grid_GPU_Capacity(0,0,dm,density);assert(ok==(tier<2));
   for(int s=0;s<spins;s++)for(int mc=1;mc<=3;mc++){
    for(int g=0;g<GridN_Atom[M2G[mc]];g++)assert(fabs(density[s][mc][g]-(ok?want[s][mc][g]:987))<1e-12);
    assert(den[s][mc][0]==987 && den[s][mc][1+GridN_Atom[M2G[mc]]]==987);
   }
   assert(Set_Density_Grid_GPU_Capacity(0,1,dm,density)==1);cases++;
  }
 }
 printf("PASS: %d density adapter cases, orbital width=%zu\n",cases,sizeof(Type_Orbs_Grid));
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-density-') as folder:
 p=Path(folder);(p/'mpi.h').write_text('typedef int MPI_Comm;\n')
 (p/'check.c').write_text(prefix+body+fixture+sdg+main)
 cc=shlex.split(os.environ.get('CC','cc'));cxx=shlex.split(os.environ.get('CXX','c++'))
 flags=['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I'+str(p),'-I'+str(root/'source')]
 subprocess.run(cxx+['-std=c++17']+flags+['-c',str(root/'source/openmx_gpu_bilinear_pack.cpp'),'-o',str(p/'pack.o')],check=True)
 for mode in ([],['-DORB_DOUBLE']):
  subprocess.run(cc+['-std=c11','-Wno-unknown-pragmas']+flags+mode+['-c',str(p/'check.c'),'-o',str(p/'check.o')],check=True)
  subprocess.run(cxx+flags+[str(p/'check.o'),str(p/'pack.o'),'-lm','-o',str(p/'check')],check=True)
  subprocess.run([str(p/'check')],check=True,timeout=60)
