/* Test production Coulomb assembly with the actual C++ scratch owner.
   MPI, one-worker OpenMP and radial/two-center kernels are modeled. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "openmx_core_atomic.h"
#include "openmx_neutral_atom.h"
#define CHECK(c) do{if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(2);}}while(0)
#define PI 3.1415926535897932384626433832795
#define GPUSOLVER 7
#define measure_time 0
static MPI_Comm mpi_comm_level1=1;
static int atomnum=3,Matomnum,SpeciesNum=2,M2G[4]={0,3,1,2};
static int WhatSpecies[4]={0,0,1,0},Spe_Total_CNO[2]={1,1},FNAN[4]={0,1,2,1},G2ID[4];
static int neighbors[4][3]={{0},{1,2},{2,1,3},{3,2}},*natn[4];
static int images[4][3]={{0}},*ncn[4],reverse_data[4][3][1],*reverse_rows[4][3],**RMI1[4];
static double positions[4][50],*Gxyz[4],distances[4][3],*Dis[4],atv[1][4]={{0}};
static double Spe_Core_Charge[2]={2,4},time_per_atom[4],EH0_scaling[2][2];
static int Host_ID,MYID_MPI_COMM_WORLD,level_stdout,SpinP_switch,Energy_Decomposition_flag;
static int List_YOUSO[100],Spe_MaxL_Basis[2],basis_data[2][1],*Spe_Num_Basis[2];
static int scf_eigen_lib_flag,Max_Nd,Max_TGN_EH0,CoarseGL_Mesh=2,TGN_EH0[2];
static double CoarseGL_Abscissae[2]={-.5,.5},CoarseGL_Weight[2]={1,1},dv_EH0[2];
static double grid[5][2][64],*GridX_EH0[2],*GridY_EH0[2],*GridZ_EH0[2],*Arho_EH0[2],*Wt_EH0[2];
static double dec[2][4],*dec_rows[2][4],**DecEscc[2];
#include "openmx_energy_workspace.h"
static double ticks;
static int batch_calls,grid_allocations;
static void near(double a,double b){CHECK(isfinite(a)&&isfinite(b)&&fabs(a-b)<2e-12*(1+fabs(b)));}
static void dtime(double *t){*t=++ticks;}
static int TotalEnergyUseOpenMP(void){return 0;}
static int TotalEnergyUseOpenACC(void){return 0;}
int MPI_Abort(MPI_Comm comm,int code){(void)comm;(void)code;_Exit(77);}
int MPI_Comm_size(MPI_Comm comm,int *n){(void)comm;*n=1;return 0;}
int MPI_Comm_rank(MPI_Comm comm,int *rank){(void)comm;*rank=0;return 0;}
int MPI_Bcast(void *p,int n,int type,int root,MPI_Comm comm){(void)p;(void)root;(void)comm;CHECK(n==3&&type==MPI_DOUBLE);return 0;}
int MPI_Allreduce(const void *s,void *r,int n,int type,int op,MPI_Comm comm){(void)comm;CHECK(n==1&&type==MPI_DOUBLE&&op==MPI_SUM);memcpy(r,s,sizeof(double));return 0;}
static void Allocate_Arrays(int mode){CHECK(mode==4&&Max_TGN_EH0<=64);++grid_allocations;}
double KumoF(int mesh,double x,double *a,double *b,double *c){CHECK(mesh==2&&a&&b&&c&&isfinite(x));return .2+.03*x;}
double VH_AtomF(double charge,int mesh,double x,double r,double *a,double *b,double *c){CHECK(mesh==2&&a&&b&&c&&isfinite(x)&&r>0);return 1+.1*charge;}
static void model_pair(int base,int other,int derivative,double distance,const double direction[3],double out[4]){
  out[0]=.3+.04*base+.07*other+.02*distance;
  for(int d=0;d<3;d++)out[d+1]=derivative?(.4+.09*base-.05*other)*direction[d]:0;
}
static void EH0_TwoCenter(const OpenMXAtomicViews *atomic,int atom,int h,double out[4]){
  int other=natn[atom][h],a=WhatSpecies[atom],b=WhatSpecies[other];double direction[3]={0};
  if(h)for(int d=0;d<3;d++)direction[d]=(Gxyz[atom][d+1]-Gxyz[other][d+1])/Dis[atom][h];
  model_pair(atomic->species.species_index[a],b,h!=0,Dis[atom][h],direction,out);
}
static void batch(const OpenMXAtomicViews *atomic,int count,int *base,int *other,int *derivative,
    double *distance,double *x,double *y,double *z,double *a,double *b,double *c,double *d){
  (void)atomic;++batch_calls;CHECK(count==(Matomnum?14:0));
  CHECK(base&&other&&derivative&&distance&&x&&y&&z&&a&&b&&c&&d);
  for(int i=0;i<count;i++){
    const double direction[3]={x[i],y[i],z[i]};double out[4];
    model_pair(base[i],other[i],derivative[i],distance[i],direction,out);
    a[i]=out[0];b[i]=out[1];c[i]=out[2];d[i]=out[3];
  }
}
static void TotalEnergy_EH0_TwoCenter_Batch_OpenACC(const OpenMXAtomicViews *a,int n,int *b,int *w,int *h,double *r,double *x,double *y,double *z,double *o0,double *o1,double *o2,double *o3){batch(a,n,b,w,h,r,x,y,z,o0,o1,o2,o3);}
static void TotalEnergy_EH0_TwoCenter_Batch_OpenMP(const OpenMXAtomicViews *a,int n,int *b,int *w,int *h,double *r,double *x,double *y,double *z,double *o0,double *o1,double *o2,double *o3){batch(a,n,b,w,h,r,x,y,z,o0,o1,o2,o3);}
static void EH0_TwoCenter_at_Cutoff(OpenMXSolverInvocation*,const OpenMXAtomicViews*,int,int,double[4]);
#include "coulomb_native.inc"

/* Independent assembly oracle: directly sum ordered physical neighbor pairs.
   No batch rows, reverse map, native helper or workspace is used here. */
static void expected(const OpenMXAtomicViews *atomic,double *core,double *neutral,double forces[4][3],double atom_energy[4]){
  *core=*neutral=0;
  for(int m=1;m<=Matomnum;m++){
    const int a=M2G[m],sa=WhatSpecies[a];double ec=0,eh=0;
    for(int j=0;j<=FNAN[a];j++){
      const int b=natn[a][j],sb=WhatSpecies[b];const double distance=Dis[a][j];
      if(j){
        const double v=Spe_Core_Charge[sa]*Spe_Core_Charge[sb]/distance;ec+=.5*v;
        for(int d=0;d<3;d++)forces[a][d]-=(positions[a][d+1]-positions[b][d+1])*v/(distance*distance);
      }
      const double first=j?EH0_scaling[sa][sb]:1,second=j?EH0_scaling[sb][sa]:1;
      const double forward=.3+.04*atomic->species.species_index[sa]+.07*sb+.02*distance;
      const double reverse=.3+.04*atomic->species.species_index[sb]+.07*sa+.02*distance;
      eh-=.25*(first*forward+second*reverse);
      if(j)for(int d=0;d<3;d++){
        const double direction=(positions[a][d+1]-positions[b][d+1])/distance;
        forces[a][d]-=.5*direction*(first*(.4+.09*atomic->species.species_index[sa]-.05*sb)+second*(.4+.09*atomic->species.species_index[sb]-.05*sa));
      }
    }
    atom_energy[m]=.5*(ec+eh);*core+=ec;*neutral+=eh;
  }
}
typedef struct {OpenMXAtomicViews atomic;int md;} Arguments;
static int exercise(OpenMXSolverInvocation *owner,void *opaque,double *result){
  Arguments *args=opaque;OpenMXEnergyWorkspace pairs;
  OpenMX_Energy_PairsPrepare(owner,Matomnum?7:0,&pairs);
  for(int repeat=0;repeat<2;repeat++){
    double forces[4][3]={{0}},atom_energy[4]={0},core,neutral;
    for(int a=1;a<=3;a++)for(int d=0;d<3;d++)forces[a][d]=positions[a][d+17]=.01*(d+1)*a;
    for(int s=0;s<2;s++)for(int a=0;a<4;a++)dec[s][a]=99;
    batch_calls=grid_allocations=0;
    const double actual_core=Calc_Ecore(owner),actual_neutral=Calc_EH0(&pairs,&args->atomic,args->md);
    CHECK(batch_calls==(scf_eigen_lib_flag==GPUSOLVER?1:0));CHECK(grid_allocations==(args->md==1?1:0));
    if(args->md==1)for(int s=0;s<2;s++){
      CHECK(args->atomic.species.species_index[s]==s&&TGN_EH0[s]>0&&TGN_EH0[s]<=Max_TGN_EH0);
      /* Constant prescribed potential makes the cutoff integral analytic
         on the supplied quadrature: sum(weight*density)*dv*potential. */
      double integral=0;
      for(int i=0;i<TGN_EH0[s];i++)integral+=Wt_EH0[s][i]*Arho_EH0[s][i];
      for(int t=0;t<2;t++)near(EH0_scaling[s][t],Spe_Core_Charge[s]*Spe_Core_Charge[t]/
          ((args->atomic.species.cutoff[s]+args->atomic.species.cutoff[t])*integral*dv_EH0[s]*(1+.1*Spe_Core_Charge[t])));
    }
    expected(&args->atomic,&core,&neutral,forces,atom_energy);near(actual_core,core);near(actual_neutral,neutral);
    for(int a=1;a<=3;a++)for(int d=0;d<3;d++)near(positions[a][d+17],forces[a][d]);
    for(int m=1;m<=3;m++)for(int s=0;s<2;s++)near(dec[s][m],Energy_Decomposition_flag&&m<=Matomnum?atom_energy[m]:99);
  }
  *result=1;return 1;
}
int main(void){
  for(int a=0;a<4;a++){
    Gxyz[a]=positions[a];Dis[a]=distances[a];natn[a]=neighbors[a];ncn[a]=images[a];RMI1[a]=reverse_rows[a];
    for(int h=0;h<3;h++)reverse_rows[a][h]=reverse_data[a][h];
    for(int s=0;s<2;s++)dec_rows[s][a]=&dec[s][a];
  }
  for(int s=0;s<2;s++){
    DecEscc[s]=dec_rows[s];Spe_Num_Basis[s]=basis_data[s];
    GridX_EH0[s]=grid[0][s];GridY_EH0[s]=grid[1][s];GridZ_EH0[s]=grid[2][s];Arho_EH0[s]=grid[3][s];Wt_EH0[s]=grid[4][s];
  }
  for(int a=1;a<=3;a++){positions[a][1]=.3*a;positions[a][2]=.07*a*a;positions[a][3]=-.11*a;}
  for(int a=1;a<=3;a++)for(int h=0;h<=FNAN[a];h++){
    const int b=natn[a][h];double r2=0;
    for(int d=1;d<=3;d++)r2+=pow(positions[a][d]-positions[b][d],2);
    distances[a][h]=sqrt(r2);
  }
  for(int m=1;m<=3;m++)for(int h=0;h<=FNAN[M2G[m]];h++){
    const int a=M2G[m],b=natn[a][h];int found=-1;
    for(int j=0;j<=FNAN[b];j++)if(natn[b][j]==a)found=j;
    CHECK(found>=0);reverse_data[m][h][0]=found;
  }
  double cutoff[2]={.11,.14},radial[2][2]={{0}};int index[2],mesh[2]={2,2};
  Arguments args={0};args.atomic.species.species=2;args.atomic.species.species_index=index;
  args.atomic.species.cutoff=cutoff;args.atomic.species.core_charge=Spe_Core_Charge;
  args.atomic.species.pao_mesh=args.atomic.species.vps_mesh=mesh;
  args.atomic.pao_log_radii=args.atomic.pao_radii=args.atomic.density=args.atomic.vps_log_radii=args.atomic.vps_radii=args.atomic.hartree_potential=(OpenMXRadialView2){&radial[0][0],{2,2}};
  for(int empty=0;empty<2;empty++)for(int decomp=0;decomp<2;decomp++)for(int md=1;md<=2;md++)for(int gpu=0;gpu<2;gpu++)for(int spin=0;spin<3;spin++){
    Matomnum=empty?0:3;Energy_Decomposition_flag=decomp;SpinP_switch=spin==2?3:spin;
    scf_eigen_lib_flag=gpu?GPUSOLVER:1;args.md=md;index[0]=1;index[1]=0;
    for(int s=0;s<2;s++)for(int t=0;t<2;t++)EH0_scaling[s][t]=.8+.2*s+.3*t;
    CHECK(OpenMX_Solver_Run(exercise,&args)==1);
  }
  puts("PASS: 48 Coulomb cases, each twice per invocation, including empty ranks and MD initialization");return 0;
}
