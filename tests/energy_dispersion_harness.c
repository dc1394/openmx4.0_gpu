/* Actual D3 routine and C++ owner; prescribed reference coefficients and
   single-rank MPI. Independent energy, coordinate and strain derivatives. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#define CHECK(c) do{if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(2);}}while(0)
static MPI_Comm mpi_comm_level1=1;
static int atomnum=4,Matomnum=4,M2G[5]={0,2,1,4,3},G2ID[5];
static int WhatSpecies[5]={0,0,1,0,2},Spe_WhatAtom[3]={1,8,0},maxcn_dftD[3]={2,2,0};
/* Only parsed by unrelated legacy residue adapters in the shared header. */
static int SpeciesNum=3,Spe_Total_CNO[3],FNAN[5],*natn[5];
static int n1_CN_DFT_D,n2_CN_DFT_D,n3_CN_DFT_D,n1_DFT_D,n2_DFT_D,n3_DFT_D;
static int DFTD3_damp_dftD,Energy_Decomposition_flag;
static double positions[5][61],*Gxyz[5],tv[4][4],Stress_Tensor[9],DecEvdw[2][5][1];
static double rcovab_dftD[3][3],r2r4ab_dftD[3][3],r0ab_dftD[3][3],C6ab_dftD[3][3][2][2][3];
static double rcut_dftD=24,cncut_dftD=20,k1_dftD=16,k3_dftD=4;
static double s6_dftD=1,s8_dftD=.5,sr6_dftD=1.2,sr8_dftD=1,alp6_dftD=14,alp8_dftD=16;
static double a1_dftD=.4,a2_dftD=3.7;
#include "openmx_energy_workspace.h"
int MPI_Abort(MPI_Comm c,int code){(void)c;(void)code;_Exit(77);}
int MPI_Comm_size(MPI_Comm c,int *n){(void)c;*n=1;return 0;}
int MPI_Comm_rank(MPI_Comm c,int *n){(void)c;*n=0;return 0;}
int MPI_Barrier(MPI_Comm c){(void)c;return 0;}
int MPI_Bcast(void *v,int n,int type,int root,MPI_Comm c)
{(void)v;(void)root;(void)c;CHECK((n==1||n==3)&&type==MPI_DOUBLE);return 0;}
int MPI_Allreduce(const void *s,void *r,int n,int type,int op,MPI_Comm c)
{(void)c;CHECK((n==1||n==9||n==atomnum+1)&&type==MPI_DOUBLE&&op==MPI_SUM);if(s!=MPI_IN_PLACE)memcpy(r,s,n*sizeof(double));return 0;}
#include "dispersion_native.inc"

/* Sum the model's ordered-pair energy with long-double scalar arithmetic.
   This oracle neither uses native gradients nor any scratch/view adapter. */
static long double energy(long double atoms[5][3],long double cell[3][3])
{
  long double cn[5]={0},total=0;
  const int radii[3]={n1_CN_DFT_D,n2_CN_DFT_D,n3_CN_DFT_D};
  for(int a=1;a<=atomnum;a++)if(Spe_WhatAtom[WhatSpecies[a]]>0)
    for(int b=1;b<=atomnum;b++)if(Spe_WhatAtom[WhatSpecies[b]]>0){
      const int periodic=(int)positions[b][60];
      for(int x=-periodic*radii[0];x<=periodic*radii[0];x++)
      for(int y=-periodic*radii[1];y<=periodic*radii[1];y++)
      for(int z=-periodic*radii[2];z<=periodic*radii[2];z++){
        long double r2=0;for(int d=0;d<3;d++){long double r=atoms[a][d]-atoms[b][d]-x*cell[0][d]-y*cell[1][d]-z*cell[2][d];r2+=r*r;}
        if(r2>.1L && r2<cncut_dftD*cncut_dftD)
          cn[a]+=1/(1+expl(-k1_dftD*(rcovab_dftD[WhatSpecies[a]][WhatSpecies[b]]/sqrtl(r2)-1)));
      }
    }
  for(int a=1;a<=atomnum;a++)if(Spe_WhatAtom[WhatSpecies[a]]>0)
    for(int b=1;b<=atomnum;b++)if(Spe_WhatAtom[WhatSpecies[b]]>0){
      const int wa=WhatSpecies[a],wb=WhatSpecies[b];long double numerator=0,denominator=0,shift=-1e300L;
      for(int i=0;i<2;i++)for(int j=0;j<2;j++){
        long double da=cn[a]-C6ab_dftD[wa][wb][i][j][1],db=cn[b]-C6ab_dftD[wa][wb][i][j][2];
        long double exponent=-k3_dftD*(da*da+db*db);if(exponent>shift)shift=exponent;
      }
      for(int i=0;i<2;i++)for(int j=0;j<2;j++){
        long double da=cn[a]-C6ab_dftD[wa][wb][i][j][1],db=cn[b]-C6ab_dftD[wa][wb][i][j][2];
        long double weight=expl(-k3_dftD*(da*da+db*db)-shift);
        numerator+=C6ab_dftD[wa][wb][i][j][0]*weight;denominator+=weight;
      }
      long double c6=numerator/denominator,c8=3*c6*r2r4ab_dftD[wa][wb];
      const int periodic=(int)positions[b][60],r[3]={n1_DFT_D,n2_DFT_D,n3_DFT_D};
      for(int x=-periodic*r[0];x<=periodic*r[0];x++)
      for(int y=-periodic*r[1];y<=periodic*r[1];y++)
      for(int z=-periodic*r[2];z<=periodic*r[2];z++){
        long double r2=0;for(int d=0;d<3;d++){long double v=atoms[a][d]-atoms[b][d]-x*cell[0][d]-y*cell[1][d]-z*cell[2][d];r2+=v*v;}
        if(r2>.1L && r2<rcut_dftD*rcut_dftD){
          long double distance=sqrtl(r2),r6=r2*r2*r2,r8=r6*r2,value;
          if(DFTD3_damp_dftD==1)
            value=s6_dftD*c6/(r6*(1+6*powl(distance/(sr6_dftD*r0ab_dftD[wb][wa]),-alp6_dftD)))+
                s8_dftD*c8/(r8*(1+6*powl(distance/(sr8_dftD*r0ab_dftD[wb][wa]),-alp8_dftD)));
          else {long double damping=a1_dftD*sqrtl(3*r2r4ab_dftD[wa][wb])+a2_dftD;
            value=s6_dftD*c6/(r6+powl(damping,6))+s8_dftD*c8/(r8+powl(damping,8));}
          total-=.5L*value;
        }
      }
    }
  return total;
}
static void near(double a,long double b,long double tolerance)
{CHECK(isfinite(a)&&isfinite((double)b)&&fabsl(a-b)<tolerance*(1+fabsl(b)));}
static int run(OpenMXSolverInvocation *owner,void *unused,double *result)
{
  (void)unused;
  for(int repeat=0;repeat<3;repeat++){
    long double atoms[5][3]={{0}},cell[3][3],h=1e-5L;
    /* Different shapes and a repeated larger shape exercise replacement and reset. */
    n1_CN_DFT_D=n1_DFT_D=repeat==1?0:1;
    n2_CN_DFT_D=n2_DFT_D=0;n3_CN_DFT_D=n3_DFT_D=repeat==1?0:2;
    for(int a=0;a<5;a++)for(int d=0;d<3;d++){atoms[a][d]=positions[a][d+1];positions[a][d+17]=.03*(a+1)*(d+1);}
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)cell[i][j]=tv[i+1][j+1];
    for(int d=0;d<9;d++)Stress_Tensor[d]=.02*(d+1);
    for(int s=0;s<2;s++)for(int a=0;a<5;a++)DecEvdw[s][a][0]=99;
    double actual=Calc_EdftD3(owner);long double expected=Matomnum?energy(atoms,cell):0;near(actual,expected,1e-14L);
    for(int a=1;a<=atomnum;a++)for(int d=0;d<3;d++){
      atoms[a][d]+=h;long double plus=energy(atoms,cell);atoms[a][d]-=2*h;long double minus=energy(atoms,cell);atoms[a][d]+=h;
      near(positions[a][d+17]-.03*(a+1)*(d+1),Matomnum?(plus-minus)/(2*h):0,2e-10L);
    }
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){
      long double ap[5][3],am[5][3],cp[3][3],cm[3][3];
      memcpy(ap,atoms,sizeof(ap));memcpy(am,atoms,sizeof(am));memcpy(cp,cell,sizeof(cp));memcpy(cm,cell,sizeof(cm));
      for(int a=1;a<=atomnum;a++){ap[a][j]+=h*atoms[a][i];am[a][j]-=h*atoms[a][i];}
      for(int v=0;v<3;v++){cp[v][j]+=h*cell[v][i];cm[v][j]-=h*cell[v][i];}
      near(Stress_Tensor[3*i+j]-.02*(3*i+j+1),Matomnum?(energy(ap,cp)-energy(am,cm))/(2*h):0,2e-9L);
    }
    double sum=0;for(int a=1;a<=Matomnum;a++)if(Spe_WhatAtom[WhatSpecies[M2G[a]]]>0){
      if(Energy_Decomposition_flag){near(DecEvdw[0][a][0],DecEvdw[1][a][0],0.1e-14L);sum+=DecEvdw[0][a][0];}
      else near(DecEvdw[0][a][0],99,1e-14L);
    }
    if(Energy_Decomposition_flag)near(sum,expected,1e-14L);
  }
  *result=1;return 1;
}
int main(void)
{
  for(int a=0;a<5;a++)Gxyz[a]=positions[a];
  const double xyz[5][3]={{0},{1.2,-.7,.3},{4.7,1.1,-.8},{-.6,4.8,1.7},{2.2,2.3,1.4}};
  for(int a=1;a<5;a++)for(int d=0;d<3;d++)positions[a][d+1]=xyz[a][d];
  tv[1][1]=12;tv[2][2]=14;tv[3][3]=16;tv[1][2]=.3;tv[2][3]=.2;
  for(int a=0;a<2;a++)for(int b=0;b<2;b++){
    rcovab_dftD[a][b]=3.4+.2*(a+b);r2r4ab_dftD[a][b]=1.2+.1*(a+b);r0ab_dftD[a][b]=3+.1*(a+b);
    for(int i=0;i<2;i++)for(int j=0;j<2;j++){
      C6ab_dftD[a][b][i][j][0]=10+4*(a+b)+3*(i+j)+i*j;
      C6ab_dftD[a][b][i][j][1]=.25+.75*i;C6ab_dftD[a][b][i][j][2]=.25+.75*j;
    }
  }
  int cases=0;
  for(int damping=1;damping<=2;damping++)for(int periodic=0;periodic<=1;periodic++)
  for(int ghost=0;ghost<=1;ghost++)for(int empty=0;empty<=1;empty++)for(int decomp=0;decomp<=1;decomp++){
    DFTD3_damp_dftD=damping;Energy_Decomposition_flag=decomp;atomnum=ghost?4:3;Matomnum=empty?0:atomnum;
    M2G[3]=ghost?4:3;M2G[4]=3;
    for(int a=1;a<=4;a++)positions[a][60]=periodic;
    double result;char error[256];CHECK(openmx_core_solver_run(1,run,NULL,&result,error,sizeof(error))&&result==1);++cases;
  }
  printf("PASS: %d production D3 cases, each with three shapes/reset evaluations\n",cases);return 0;
}
