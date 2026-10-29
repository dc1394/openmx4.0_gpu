/* Real C++ ownership; radial/XC/GPU/MPI calls and OpenMP teams are
   modeled. Workers run sequentially so every host toolchain can check the
   exact production partition and scratch reuse; native concurrency is also
   exercised by the application's OpenMP regression inputs. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "openmx_core_atomic.h"
#include "openmx_core_quadrature.h"
#define CHECK(c) do{if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(2);}}while(0)
#define PI 3.1415926535897932384626433832795
#define GPUSOLVER 7
int energy_fixture_thread=0,energy_fixture_team=1;
static MPI_Comm mpi_comm_level1=1;
static int atomnum=3,Matomnum,SpeciesNum=2,M2G[4]={0,3,1,2},WhatSpecies[4]={0,0,1,0},Spe_Total_CNO[2]={1,1};
static int FNAN[4]={0,1,2,1},neighbor_data[4][3]={{0},{1,2},{2,1,3},{3,2}},*natn[4];
static int image_data[4][3]={{0}},*ncn[4];
static double Gxyz[4][50],atv[3][4]={{0}};
static int F_Vxc_flag,Exc0_correction_flag,Num_Leb_Grid,scf_eigen_lib_flag,PCC_switch,level_stdout;
static int region_calls,grid_calls,Host_ID;
static double Stress_Tensor[9];
static const int CoarseGL_Mesh=5;
static double CoarseGL_Abscissae[5]={-.8,-.35,.05,.4,.75},CoarseGL_Weight[5]={.3,.45,.5,.4,.35};
#include "openmx_core_angular_grid.h"
static int counted_lebedev_prepare(OpenMXSolverInvocation *owner,int points,
    OpenMXLebedevRuleData *out,char *error,size_t capacity){
  ++grid_calls;return openmx_core_solver_lebedev_rule(owner,points,out,error,capacity);
}
#define openmx_core_solver_lebedev_rule counted_lebedev_prepare
#include "openmx_stress_workspace.h"
#undef openmx_core_solver_lebedev_rule
int MPI_Abort(MPI_Comm comm,int code){(void)comm;(void)code;_Exit(77);}
int MPI_Allreduce(const void *s,void *r,int n,int type,int op,MPI_Comm comm){(void)comm;CHECK(n==9&&type==MPI_DOUBLE&&op==MPI_SUM);memcpy(r,s,9*sizeof(double));return 0;}
static void near(double a,double b,double tolerance){CHECK(isfinite(a)&&isfinite(b)&&fabs(a-b)<tolerance*(1+fabs(b)));}
static int next_team(void){const int teams[3]={1,4,2};return energy_fixture_team=teams[region_calls++%3];}

static double KumoF(int mesh,double x,const double *a,const double *b,const double *c){
  CHECK(mesh==2&&a&&b);return c[0]*exp(-c[1]*exp(2*x));
}
static double Dr_KumoF(int mesh,double x,double r,const double *a,const double *b,const double *c){
  return -2*c[1]*r*KumoF(mesh,x,a,b,c);
}
static double XC_Ceperly_Alder(double n,int mode){
  CHECK(mode==0||mode==3);return mode==0?-.2-.4*n+.05*n*n:-.4+.1*n;
}
/* The production code calls the xc package's entry; the harness keeps the model. */
static double openmx_xc_ceperley_alder(double density,int quantity){return XC_Ceperly_Alder(density,quantity);}
/* Independently integrate the prescribed density and polynomial XC energy.
   Coordinate differences of this scalar check the native gradient fields. */
static double oracle_energy(const OpenMXAtomicViews *atomic,int points,const double *rows){
  if(!F_Vxc_flag)return 0;
  double sum=0;
  for(int m=1;m<=Matomnum;m++){
    const int a=M2G[m],s=WhatSpecies[a];const double cutoff=atomic->species.cutoff[s];
    const double *self=openmx_radial_row2(&atomic->density_backup,(size_t)s);
    double radial=0;
    for(int ir=0;ir<5;ir++){
      const double radius=.5*cutoff*(CoarseGL_Abscissae[ir]+1);double angular=0;
      const double central=self[0]*exp(-self[1]*radius*radius);
      for(int i=0;i<points;i++){
        double n=0;
        for(int h=0;h<=FNAN[a];h++){
          const int b=natn[a][h],t=WhatSpecies[b];double r2=0;
          for(int d=0;d<3;d++){
            const double delta=Gxyz[b][d+1]+atv[ncn[a][h]][d+1]-(Gxyz[a][d+1]+radius*rows[4*i+d]);r2+=delta*delta;
          }
          const double *density=openmx_radial_row2(&atomic->density_backup,(size_t)t);
          n+=density[0]*exp(-density[1]*r2);
        }
        angular+=rows[4*i+3]*central*(-.2-.4*n+.05*n*n);
      }
      radial+=radius*radius*CoarseGL_Weight[ir]*angular;
    }
    sum+=2*PI*cutoff*radial;
  }
  return sum;
}
static double difference(const OpenMXAtomicViews *atomic,int points,const double *rows,int atom,int d){
  const double step=1e-6,saved=Gxyz[atom][d];
  Gxyz[atom][d]=saved+step;const double plus=oracle_energy(atomic,points,rows);
  Gxyz[atom][d]=saved-step;const double minus=oracle_energy(atomic,points,rows);
  Gxyz[atom][d]=saved;return (plus-minus)/(2*step);
}

/* Deform centers and image translations; the atomic radial/angular rule
   follows its center without a volume factor in this fine correction. */
static double strain(const OpenMXAtomicViews *atomic,int points,const double *rows,int i,int j){
  double saved[4][4]={{0}},translations[3][4];memcpy(translations,atv,sizeof(atv));
  for(int a=1;a<=3;a++)for(int d=1;d<=3;d++)saved[a][d]=Gxyz[a][d];
  const double h=1e-6;
  for(int a=1;a<=3;a++)Gxyz[a][i]=saved[a][i]+h*saved[a][j];
  for(int a=0;a<3;a++)atv[a][i]=translations[a][i]+h*translations[a][j];
  double plus=oracle_energy(atomic,points,rows);
  for(int a=1;a<=3;a++)Gxyz[a][i]=saved[a][i]-h*saved[a][j];
  for(int a=0;a<3;a++)atv[a][i]=translations[a][i]-h*translations[a][j];
  double minus=oracle_energy(atomic,points,rows);
  for(int a=1;a<=3;a++)for(int d=1;d<=3;d++)Gxyz[a][d]=saved[a][d];memcpy(atv,translations,sizeof(atv));
  return(plus-minus)/(2*h);
}
static void fine(OpenMXSolverInvocation *scratch,const OpenMXAtomicViews *atomic){
  const OpenMXGaussRuleData radial={CoarseGL_Mesh,CoarseGL_Abscissae,CoarseGL_Weight};
  int Nthrds0,Gc_AN,Mc_AN,Cwan,i,j,OMPID,Nthrds,Nprocs,ir,ia,h_AN,Gh_AN,Rn,Hwan,Nloop,myid=0;
  double Stress[9]={0},My_Stress[9]={0},sumt1[9],x,y,z,dx,dy,dz,r;
  OpenMXSolverRealTensor stress_threads;
#include "stress_xc_native.inc"
}
typedef struct {OpenMXAtomicViews atomic;} Arguments;
static int exercise(OpenMXSolverInvocation *owner,void *opaque,double *result){
  Arguments *args=opaque;const int counts[3]={590,1202,590};
  for(int repeat=0;repeat<3;repeat++){
    Num_Leb_Grid=counts[repeat];OpenMXLebedevRuleData angular;
    OpenMX_XC_LebedevPrepare(owner,Num_Leb_Grid,&angular);const double *rows=angular.xyzw;
    double gradients[4][3]={{0}},stress[9]={0};
    if(Exc0_correction_flag){
      for(int a=1;a<=3;a++)for(int d=1;d<=3;d++)gradients[a][d-1]=difference(&args->atomic,Num_Leb_Grid,rows,a,d);
      for(int i=1;i<=3;i++)for(int j=1;j<=3;j++)stress[3*(i-1)+j-1]=strain(&args->atomic,Num_Leb_Grid,rows,i,j);
    }
    for(int a=1;a<=3;a++)for(int d=1;d<=3;d++)Gxyz[a][d+40]=77;
    for(int i=0;i<9;i++)Stress_Tensor[i]=.01*(i+1);
    grid_calls=0;fine(owner,&args->atomic);CHECK(grid_calls==(Exc0_correction_flag?1:0));
    for(int i=0;i<9;i++)near(Stress_Tensor[i],.01*(i+1)+stress[i],3e-7);
    for(int a=1;a<=3;a++)for(int d=1;d<=3;d++)near(Gxyz[a][d+40],Exc0_correction_flag?gradients[a][d-1]:77,3e-7);
    CHECK(energy_fixture_thread==0);
  }
  *result=1;return 1;
}
static int invalid(OpenMXSolverInvocation *owner,void *opaque,double *result){
  Arguments args={0};Num_Leb_Grid=atoi(opaque);Exc0_correction_flag=1;fine(owner,&args.atomic);*result=0;return 1;
}
int main(int argc,char **argv){
  if(argc>1){OpenMX_Solver_Run(invalid,argv[1]);return 3;}
  for(int a=0;a<4;a++){natn[a]=neighbor_data[a];ncn[a]=image_data[a];}
  for(int a=1;a<=3;a++){Gxyz[a][1]=.13*a;Gxyz[a][2]=.11*a*a;Gxyz[a][3]=-.09*a;}
  double cutoff[2]={1.4,1.8},radial[2][2]={{.4,.7},{.6,.9}};int mesh[2]={2,2};
  Arguments args={0};args.atomic.species.cutoff=cutoff;args.atomic.species.pao_mesh=mesh;
  args.atomic.pao_log_radii=args.atomic.pao_radii=args.atomic.density_backup=(OpenMXRadialView2){&radial[0][0],{2,2}};
  int cases=0;
  for(int images=0;images<2;images++)for(int empty=0;empty<2;empty++)for(int enabled=0;enabled<2;enabled++)for(int xc=0;xc<2;xc++){
    Matomnum=empty?0:3;Exc0_correction_flag=enabled;F_Vxc_flag=xc;
    const double shift[3]={.17,.12,-.11};for(int d=1;d<=3;d++){atv[1][d]=images?shift[d-1]:0;atv[2][d]=-atv[1][d];}
    ncn[1][1]=1;ncn[2][1]=2;region_calls=0;CHECK(OpenMX_Solver_Run(exercise,&args)==1);cases++;
  }
  printf("PASS: %d fine-stress cases, each three times, logical teams and coordinate/strain derivatives\n",cases);return 0;
}
