/* Production stress helpers with actual C++ ownership. MPI and sequential
   logical OpenMP teams are modeled; the FFT supplies prescribed coefficients. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#define PI 3.1415926535897932384626433832795
#define CHECK(c) do{if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(2);}}while(0)
static MPI_Comm mpi_comm_level1=1;
static int atomnum=3,Matomnum=3,SpeciesNum=2,Host_ID,level_stdout;
static int M2G[4]={0,3,1,2},WhatSpecies[4]={0,0,1,0},Spe_Total_CNO[2]={1,1},FNAN[4]={0,2,2,2};
static int neighbors[4][3]={{0},{1,2,3},{2,1,3},{3,1,2}},*natn[4],images[4][3],*ncn[4];
static double coordinates[4][4]={{0},{.2,-.7,.4},{3.7,1.2,-.6},{-1.1,4.3,2.1}},*Gxyz[4];
static double distances[4][3],*Dis[4],atv[1][4],Spe_Core_Charge[2]={2,3},time_per_atom[4],Stress_Tensor[9];
static double DecEscc[2][4],rtv[4][4],Cell_Volume=40;
static int Ngrid1=4,Ngrid2=4,Ngrid3=2,My_Max_NumGridB,My_NumGridB_CB,F_dVHart_flag;
static int logical_team=1,logical_worker;
#include "openmx_stress_workspace.h"
static double ticks;
static void dtime(double *t){*t=++ticks;}
int MPI_Abort(MPI_Comm c,int code){(void)c;(void)code;_Exit(77);}
int MPI_Comm_size(MPI_Comm c,int *n){(void)c;*n=1;return 0;}
int MPI_Comm_rank(MPI_Comm c,int *n){(void)c;*n=0;return 0;}
int MPI_Barrier(MPI_Comm c){(void)c;return 0;}
int MPI_Allreduce(const void *s,void *r,int n,int type,int op,MPI_Comm c)
{(void)c;CHECK(n==9&&type==MPI_DOUBLE&&op==MPI_SUM);memcpy(r,s,9*sizeof(double));return 0;}
static double FFT_Density(int mode,double *re,double *im)
{
  CHECK(mode==0&&re&&im&&re!=im);
  for(int i=0;i<My_Max_NumGridB;i++){CHECK(re[i]==0&&im[i]==0);re[i]=sin(.1*(i+1));im[i]=cos(.3*i+.2);}
  return 0;
}
#include "stress_native.inc"

static void near(double a,long double b)
{CHECK(isfinite(a)&&isfinite((double)b)&&fabsl(a-b)<2e-9L*(1+fabsl(b)));}
static long double nuclear_energy(int axis,int other,long double strain)
{
  long double total=0;
  for(int a=1;a<=3;a++)for(int b=1;b<=3;b++)if(a!=b){
    long double r2=0;
    for(int d=0;d<3;d++){
      long double delta=coordinates[a][d+1]-coordinates[b][d+1];
      if(d==other)delta+=strain*(coordinates[a][axis+1]-coordinates[b][axis+1]);
      r2+=delta*delta;
    }
    total+=.5L*Spe_Core_Charge[WhatSpecies[a]]*Spe_Core_Charge[WhatSpecies[b]]/sqrtl(r2);
  }
  return total;
}
/* Reciprocal deformation at fixed FFT coefficients and fixed volume factor:
   this is the anisotropic H1 contribution, not the complete physical stress. */
static long double reciprocal_energy(int axis,int other,long double strain)
{
  long double total=0;
  for(int k=0;k<My_NumGridB_CB;k++)if(k){
    int x=k%Ngrid1,y=(k/Ngrid1)%Ngrid2,z=k/(Ngrid1*Ngrid2);
    int indices[3]={x<Ngrid1/2?x:x-Ngrid1,y<Ngrid2/2?y:y-Ngrid2,z<Ngrid3/2?z:z-Ngrid3};
    long double g[3]={0},r2=0;
    for(int d=0;d<3;d++)for(int a=0;a<3;a++)g[d]+=indices[a]*(long double)rtv[a+1][d+1];
    g[other]-=strain*g[axis];for(int d=0;d<3;d++)r2+=g[d]*g[d];
    long double re=sin(.1*(k+1)),im=cos(.3*k+.2);
    total+=2*(long double)PI*Cell_Volume*(re*re+im*im)/(r2*powl(Ngrid1*Ngrid2*Ngrid3,2));
  }
  return F_dVHart_flag*total;
}
static int run(OpenMXSolverInvocation *owner,void *unused,double *result)
{
  (void)unused;const long double h=1e-5L;const int teams[3]={1,4,2};
  double *previous=NULL;
  for(int repeat=0;repeat<3;repeat++){
    logical_team=teams[repeat];logical_worker=0;
    OpenMXSolverRealTensor check;
    OpenMX_Stress_ThreadsPrepare(owner,4,&check);
    CHECK(check.extent[0]==4&&check.extent[1]==32);
    if(previous)CHECK(previous==check.values);previous=check.values;
    for(int t=0;t<4;t++)for(int i=0;i<9;i++)openmx_solver_real_matrix_row(&check,t)[i]=91+t+i;
    for(int i=0;i<9;i++)Stress_Tensor[i]=.017*(i+1);
    Stress_core(owner);
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){
      long double derivative=Matomnum?(nuclear_energy(i,j,h)-nuclear_energy(i,j,-h))/(2*h):0;
      near(Stress_Tensor[3*i+j]-.017*(3*i+j+1),derivative);
    }
    double initial[9];memcpy(initial,Stress_Tensor,sizeof(initial));
    /* Grow, shrink to an empty rank, then rebind the larger FFT view. */
    My_Max_NumGridB=repeat==1?0:33;My_NumGridB_CB=repeat==1?0:32;
    Stress_H1(owner);
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){
      long double derivative=(reciprocal_energy(i,j,h)-reciprocal_energy(i,j,-h))/(2*h);
      near(Stress_Tensor[3*i+j]-initial[3*i+j],derivative);
    }
  }
  *result=1;return 1;
}
static int invalid(OpenMXSolverInvocation *owner,void *mode,double *result)
{
  OpenMXSolverRealTensor view;
  if(*(char*)mode=='w')OpenMX_Stress_ThreadsPrepare(owner,0,&view);
  else OpenMX_Stress_ReciprocalPrepare(owner,-1,&view);
  *result=0;return 1;
}
int main(int argc,char **argv)
{
  char error[256];double result;
  if(argc==2){CHECK(openmx_core_solver_run(4,invalid,argv[1],&result,error,sizeof(error)));return 3;}
  for(int a=0;a<4;a++){Gxyz[a]=coordinates[a];Dis[a]=distances[a];natn[a]=neighbors[a];ncn[a]=images[a];}
  for(int a=1;a<=3;a++)for(int h=1;h<=2;h++){
    int b=neighbors[a][h];double r2=0;for(int d=1;d<=3;d++){double x=coordinates[a][d]-coordinates[b][d];r2+=x*x;}distances[a][h]=sqrt(r2);
  }
  rtv[1][1]=1.2;rtv[1][2]=.1;rtv[2][2]=1.5;rtv[2][3]=.2;rtv[3][1]=.3;rtv[3][3]=1.7;
  int cases=0;
  for(int empty=0;empty<2;empty++)for(int ghost=0;ghost<2;ghost++)for(int enabled=0;enabled<2;enabled++){
    Matomnum=empty?0:3;Spe_Core_Charge[1]=ghost?0:3;F_dVHart_flag=enabled;
    CHECK(openmx_core_solver_run(4,run,NULL,&result,error,sizeof(error))&&result==1);cases++;
  }
  printf("PASS: %d stress cases, each with 1/4/2 logical teams and FFT growth/empty/rebind\n",cases);return 0;
}
