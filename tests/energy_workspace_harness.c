/* MPI, XC and accelerated grid kernels are modeled; scratch is owned by
   the actual C++ invocation. The runner checks the production output against
   an independent numerical oracle, including one-CWF allocations. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "openmx_core_cwf.h"
#define CHECK(c) do {if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(2);}} while(0)
#define GPUSOLVER 7
#define YOUSO10 1024
#define fp_bsize 1024
static MPI_Comm mpi_comm_level1=1;
/* Names used only in other inline legacy workspace adapters. */
static int atomnum,Matomnum,SpeciesNum,*M2G,*WhatSpecies,*Spe_Total_CNO,*FNAN,**natn;
#include "openmx_energy_workspace.h"
static int SpinP_switch,TNum_CWFs,XC_switch,Host_ID,scf_eigen_lib_flag;
static int Max_Grid_Index_D[4]={0,5,5,5},Min_Grid_Index_D[4]={0,0,0,0};
static int Num_Rcv_Grid_B2D[1]={4},*Index_Rcv_Grid_B2D[1],*Index_Snd_Grid_B2D[1];
static int My_NumGridB_AB=4,*CWF_Num_predefined,*G2ID,*F_G2M;
static double *Density_Grid_D[4],*Density_Grid_B[4],*Vxc_Grid_D[4],*Vxc_Grid_B[4];
static double *ADensity_Grid_B,*dVHart_Grid_B,*InitN_USpin,*InitN_DSpin,***DecEscc;
static double GridVol=.25;
static char filepath[YOUSO10]="",filename[YOUSO10]="energy";
static char *SpeName[2]={"A","B"};
static int grid_calls,xc_calls;
int MPI_Abort(MPI_Comm comm,int code){(void)comm;(void)code;_Exit(77);}
int MPI_Comm_size(MPI_Comm comm,int *n){(void)comm;*n=1;return 0;}
int MPI_Comm_rank(MPI_Comm comm,int *rank){(void)comm;*rank=0;return 0;}
int MPI_Barrier(MPI_Comm comm){(void)comm;return 0;}
int MPI_Allreduce(const void *send,void *receive,int n,int type,int op,MPI_Comm comm){
  (void)comm;CHECK(n>0 && type==MPI_DOUBLE && op==MPI_SUM);
  if(send!=MPI_IN_PLACE)memcpy(receive,send,(size_t)n*sizeof(double));
  return 0;
}
static void Set_XC_Grid(int mode,int derivative,int xc,double *d0,double *d1,double *d2,double *d3,
    double *v0,double *v1,double *v2,double *v3,void *a,void *b){
  CHECK(mode==2 && derivative==1 && xc==XC_switch && !a && !b);
  CHECK(d0==Density_Grid_D[0] && d1==Density_Grid_D[1] && d2==Density_Grid_D[2] && d3==Density_Grid_D[3]);
  CHECK(v0==Vxc_Grid_D[0] && v1==Vxc_Grid_D[1] && v2==Vxc_Grid_D[2] && v3==Vxc_Grid_D[3]);
  ++xc_calls; /* Prescribed XC potential, independent of the real XC library. */
}
static void grid_dc(int spinmax,double *hartree,double *xc){
  ++grid_calls;
  for(int s=0;s<=spinmax;s++)for(int b=0;b<4;b++){
    hartree[s]+=(Density_Grid_B[s][b]+ADensity_Grid_B[b])*dVHart_Grid_B[b];
    xc[s]+=Density_Grid_B[s][b]*Vxc_Grid_B[s][b];
  }
}
static void TotalEnergy_CWF_Dc_Grid_OpenMP(int s,double *h,double *x){grid_dc(s,h,x);}
static void TotalEnergy_CWF_Dc_Grid_OpenACC(int s,double *h,double *x){grid_dc(s,h,x);}
#include "energy_native.inc"

static int nested(OpenMXSolverInvocation *owner,void *outer,double *result){
  OpenMXEnergyWorkspace view;OpenMX_Energy_PairsPrepare(owner,5,&view);
  CHECK(view.owner==owner && view.pair_indices.values!=outer);
  *result=1;return 1;
}
static int layout(OpenMXSolverInvocation *owner,void *argument,double *result){
  const char *mode=argument;
  if(!strcmp(mode,"bad-count"))OpenMX_Energy_PairsPrepare(owner,(size_t)INT_MAX+1,NULL);
  if(!strcmp(mode,"bad-eh0-count"))OpenMX_Energy_EH0BatchPrepare(owner,(size_t)INT_MAX+1,NULL);
  if(!strcmp(mode,"bad-cwf")){
    OpenMXSolverRealTensor a,b;OpenMX_Energy_CWFScratch(owner,0,&a,&b);
  }
  const size_t counts[3]={0,5,2};
  for(int k=0;k<3;k++){
    OpenMXEnergyWorkspace a,b;const size_t n=counts[k];
    OpenMX_Energy_PairsPrepare(owner,n,&a);
    CHECK(a.owner==owner && a.pair_count==(int)n && a.pair_indices.extent[0]==2);
    CHECK(a.pair_indices.extent[1]==n+1);
    for(size_t row=0;row<2;row++)for(size_t i=0;i<=n;i++)
      openmx_solver_int_matrix_row(&a.pair_indices,row)[i]=(int)(100*row+i);
    OpenMX_Energy_PairsPrepare(owner,n,&b);CHECK(b.pair_indices.values==a.pair_indices.values);
    for(size_t row=0;row<2;row++)for(size_t i=0;i<=n;i++)
      CHECK(openmx_solver_int_matrix_row(&b.pair_indices,row)[i]==(int)(100*row+i));
    CHECK(OpenMX_Solver_Run(nested,a.pair_indices.values)==1);
  }
  for(int k=0;k<3;k++){
    OpenMXEnergyEH0Batch a,b;const size_t n=counts[k],capacity=n?n:1;
    OpenMX_Energy_EH0BatchPrepare(owner,n,&a);
    CHECK(a.count==(int)n && a.indices.extent[0]==3 && a.values.extent[0]==8);
    CHECK(a.indices.extent[1]==capacity && a.values.extent[1]==capacity);
    CHECK(a.indices.values && a.values.values);
    for(size_t row=0;row<3;row++)for(size_t i=0;i<capacity;i++)
      openmx_solver_int_matrix_row(&a.indices,row)[i]=(int)(100*row+i);
    for(size_t row=0;row<8;row++)for(size_t i=0;i<capacity;i++)
      openmx_solver_real_matrix_row(&a.values,row)[i]=100*row+i+.5;
    OpenMX_Energy_EH0BatchPrepare(owner,n,&b);
    CHECK(a.indices.values==b.indices.values && a.values.values==b.values.values);
    for(size_t row=0;row<3;row++)for(size_t i=0;i<capacity;i++)
      CHECK(openmx_solver_int_matrix_row(&b.indices,row)[i]==(int)(100*row+i));
    for(size_t row=0;row<8;row++)for(size_t i=0;i<capacity;i++)
      CHECK(openmx_solver_real_matrix_row(&b.values,row)[i]==100*row+i+.5);
  }
  OpenMXSolverRealTensor a,b;OpenMX_Energy_CWFScratch(owner,3,&a,&b);
  CHECK(a.extent[0]==2 && a.extent[1]==3 && a.values!=b.values);
  for(int i=0;i<6;i++){a.values[i]=i+.5;b.values[i]=-i-.5;}
  OpenMXSolverRealTensor c,d;OpenMX_Energy_CWFScratch(owner,3,&c,&d);
  CHECK(a.values==c.values && b.values==d.values);
  for(int i=0;i<6;i++){CHECK(c.values[i]==i+.5);CHECK(d.values[i]==-i-.5);}
  *result=1;return 1;
}
typedef struct {OpenMXCWFObservables values;double energy[8];} Arguments;
static int decompose(OpenMXSolverInvocation *owner,void *argument,double *result){
  Arguments *a=argument;
  /* Two calls in one invocation exercise reuse and mandatory zeroing. */
  Energy_Decomposition_CWF(owner,&a->values,a->energy);
  Energy_Decomposition_CWF(owner,&a->values,a->energy);
  CHECK(xc_calls==2 && grid_calls==(scf_eigen_lib_flag==GPUSOLVER?2:0));
  *result=1;return 1;
}
int main(int argc,char **argv){
  CHECK(argc>=2);
  if(argc==2){CHECK(OpenMX_Solver_Run(layout,argv[1])==1);return 0;}
  CHECK(argc==4);SpinP_switch=atoi(argv[1]);const int base=atoi(argv[2]);
  scf_eigen_lib_flag=atoi(argv[3]);CHECK(base==1 || base==2 || base==4);
  CHECK(SpinP_switch==0 || SpinP_switch==1 || SpinP_switch==3);
  atomnum=base==1?1:2;
  int species[3]={0,0,1},counts[2]={1,base-1},ranks[3]={0,0,0},local[3]={0,1,2};
  WhatSpecies=species;CWF_Num_predefined=counts;G2ID=ranks;F_G2M=local;
  TNum_CWFs=base*(SpinP_switch==3?2:1);
  double up[3]={0,.9,1.3},down[3]={0,.7,1.1};InitN_USpin=up;InitN_DSpin=down;
  int receive[4]={86,87,92,93},send[4]={0,1,2,3};
  Index_Rcv_Grid_B2D[0]=receive;Index_Snd_Grid_B2D[0]=send;
  double density_d[4][216]={{0}},potential_d[4][216]={{0}};
  double density_b[4][4],potential_b[4][4]={{0}},atomic[4],hartree[4];
  for(int s=0;s<4;s++){
    Density_Grid_D[s]=density_d[s];Vxc_Grid_D[s]=potential_d[s];
    Density_Grid_B[s]=density_b[s];Vxc_Grid_B[s]=potential_b[s];
    for(int b=0;b<4;b++){
      density_b[s][b]=.2+.1*s+.03*b;potential_d[s][receive[b]]=-.4+.05*s+.02*b;
    }
  }
  for(int b=0;b<4;b++){atomic[b]=.05+.01*b;hartree[b]=.3-.04*b;}
  ADensity_Grid_B=atomic;dVHart_Grid_B=hartree;
  double dec[3]={0,.4,.7},*dec_rows[3]={&dec[0],&dec[1],&dec[2]},**dec_spins[1]={dec_rows};
  DecEscc=dec_spins;
  /* Padding tests that observable row capacity, not active CWF count, is the stride. */
  const size_t capacity=(size_t)TNum_CWFs+3;
  double *charge=malloc(2*capacity*sizeof(double)),*energy=malloc(2*capacity*sizeof(double));
  CHECK(charge && energy);
  for(int s=0;s<2;s++)for(size_t i=0;i<capacity;i++){
    charge[s*capacity+i]=i<(size_t)TNum_CWFs?.2+.1*s+.02*i:NAN;
    energy[s*capacity+i]=i<(size_t)TNum_CWFs?-.3-.07*s+.04*i:NAN;
  }
  Arguments a={{2,capacity,charge,energy,0},{0,0,0,0,0,0,-.31,-.27}};
  CHECK(OpenMX_Solver_Run(decompose,&a)==1);
  for(int s=0;s<2;s++)for(size_t i=0;i<capacity;i++){
    if(i<(size_t)TNum_CWFs){CHECK(charge[s*capacity+i]==.2+.1*s+.02*i);CHECK(energy[s*capacity+i]==-.3-.07*s+.04*i);}
    else {CHECK(isnan(charge[s*capacity+i]) && isnan(energy[s*capacity+i]));}
  }
  free(charge);free(energy);return 0;
}
