#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "openmx_core_atomic.h"
#include "openmx_core_neutral_grid.h"
#include "openmx_core_neutral_grid_metadata.h"
#include "openmx_neutral_atom.h"
#define PI 3.1415926535897932384626433832795
#define measure_time 0
#define CHECK(c) do{if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(2);}}while(0)
int fixture_thread,fixture_team=1;
static int team_cycle;
static int fixture_next_team(void){const int teams[3]={1,4,2};return fixture_team=teams[team_cycle++%3];}
static MPI_Comm mpi_comm_level1=1;
static int atomnum=3,Matomnum,SpeciesNum=2,M2G[4]={0,3,1,2};
static int WhatSpecies[4]={0,0,1,0},Spe_Total_CNO[2]={1,1},FNAN[4]={99,1,2,1};
static int neighbors[4][3]={{0},{1,2},{2,1,3},{3,2}},*natn[4],images[4][3]={{0}},*ncn[4];
static int reverse_data[4][3][1],*reverse_rows[4][3],**RMI1[4];
static double positions[4][50],*Gxyz[4],distances[4][3],*Dis[4],atv[1][4]={{0}};
static double Spe_Core_Charge[2]={2,4},time_per_atom[4],EH0_scaling[2][2],Stress_Tensor[9];
static int Host_ID,level_stdout,Max_Nd,Max_TGN_EH0,CoarseGL_Mesh=2,*fixture_counts;
static double CoarseGL_Abscissae[3]={-.5,.5,0},CoarseGL_Weight[3]={1,1,0},*fixture_spacing;
static OpenMXNeutralGridTables fixture_tables;
#include "openmx_stress_workspace.h"
static void near(double a,double b){if(!isfinite(a)||fabs(a-b)>3e-10*(1+fabs(b))){fprintf(stderr,"values %.17g %.17g\n",a,b);CHECK(0);}}
static int allocations;
static double ticks;
static void dtime(double *t){*t=++ticks;}
int MPI_Abort(MPI_Comm comm,int code){(void)comm;(void)code;_Exit(77);}
int MPI_Comm_size(MPI_Comm comm,int *size){(void)comm;*size=1;return 0;}
int MPI_Comm_rank(MPI_Comm comm,int *rank){(void)comm;*rank=0;return 0;}
int MPI_Allreduce(const void *s,void *r,int n,int type,int op,MPI_Comm comm){(void)comm;CHECK(n==9&&type==MPI_DOUBLE&&op==MPI_SUM);memcpy(r,s,9*sizeof(double));return 0;}
static void Allocate_Arrays_with_context(int mode,OpenMXCoreContext *context){
  CHECK(mode==4&&Max_TGN_EH0<=64&&context);
  const OpenMXNeutralGridShape shape={SpeciesNum,Max_TGN_EH0};OpenMXNeutralGridTables tables;char error[256];
  CHECK(openmx_core_neutral_grid_prepare(context,&shape,&tables,error,sizeof(error)));
  fixture_tables=tables;
  ++allocations;
}
double KumoF(int mesh,double x,double *a,double *b,double *c){CHECK(mesh==2&&a&&b&&c&&isfinite(x));return .2+.03*x;}
double VH_AtomF(double charge,int mesh,double x,double r,double *a,double *b,double *c){CHECK(mesh==2&&a&&b&&c&&isfinite(x)&&r>0);return 1+.1*charge+.03*r*r;}
static double slope(int base,int other){return .4+.09*base-.05*other;}
static void EH0_TwoCenter(const OpenMXNeutralGridTables *tables,const OpenMXNeutralGridMetadataView *metadata,const OpenMXAtomicViews *atomic,int atom,int h,double out[4]){
  CHECK(tables->values==fixture_tables.values&&metadata->counts==fixture_counts&&metadata->spacing==fixture_spacing);
  const int other=natn[atom][h],a=WhatSpecies[atom],b=WhatSpecies[other];
  out[0]=.3+.04*a+.07*b+.02*Dis[atom][h];
  for(int d=0;d<3;d++)out[d+1]=h?slope(atomic->species.species_index[a],b)*(positions[atom][d+1]-positions[other][d+1])/Dis[atom][h]:0;
}
#include "stress_neutral_native.inc"
/* A radial-square potential reduces the cutoff integral to three scalar
   moments. No cutoff-distance loop or worker reduction is reused here. */
static void verify_grid(const OpenMXAtomicViews *atomic){
  for(int s=0;s<2;s++){
    const double cutoff=atomic->species.cutoff[s];const int nz=2*(int)(cutoff/(PI/sqrt(9600.)))+1;
    const double spacing=2*cutoff/(nz-1);near(fixture_spacing[s],spacing);CHECK(fixture_counts[s]==nz*CoarseGL_Mesh&&atomic->species.species_index[s]==s);
    double moment0=0,momentz=0,momentr2=0;
    for(int i=0;i<fixture_counts[s];i++){
      const int zindex=i/CoarseGL_Mesh,xindex=i%CoarseGL_Mesh;const double x=.5*cutoff*(1+CoarseGL_Abscissae[xindex]),z=-cutoff+spacing*zindex;
      const double r2=x*x+z*z,rho=.2+.015*log(r2),weight=PI*x*cutoff*CoarseGL_Weight[xindex];
      near(openmx_core_neutral_grid_row(&fixture_tables,0,s)[i],x);near(openmx_core_neutral_grid_row(&fixture_tables,1,s)[i],0);near(openmx_core_neutral_grid_row(&fixture_tables,2,s)[i],z);near(openmx_core_neutral_grid_row(&fixture_tables,3,s)[i],rho);near(openmx_core_neutral_grid_row(&fixture_tables,4,s)[i],weight);
      moment0+=weight*rho;momentz+=weight*rho*z;momentr2+=weight*rho*r2;
    }
    for(int t=0;t<2;t++){
      const double rc=cutoff+atomic->species.cutoff[t];
      const double integral=spacing*((1+.1*Spe_Core_Charge[t]+.03*rc*rc)*moment0+.03*momentr2-.06*rc*momentz);
      near(EH0_scaling[s][t],Spe_Core_Charge[s]*Spe_Core_Charge[t]/(rc*integral));
    }
  }
}
/* Ordered-pair potential with derivative equal to the prescribed native
   two-center kernel. Apply affine strain to coordinates, independently of
   the production virial assembly and its reverse-neighbor indexing. */
static long double strain_energy(int axis,int other,long double epsilon){
  long double value=0;
  for(int m=1;m<=Matomnum;m++){
    const int a=M2G[m],sa=WhatSpecies[a];
    for(int h=1;h<=FNAN[a];h++){
      const int b=neighbors[a][h],sb=WhatSpecies[b];long double r[3],r2=0;
      for(int d=0;d<3;d++)r[d]=(long double)positions[a][d+1]-(long double)positions[b][d+1];
      r[axis]+=epsilon*r[other];for(int d=0;d<3;d++)r2+=r[d]*r[d];
      value-=.25L*((long double)EH0_scaling[sa][sb]*slope(sa,sb)+(long double)EH0_scaling[sb][sa]*slope(sb,sa))*sqrtl(r2);
    }
  }return value;
}
typedef struct{OpenMXAtomicViews atomic;const char *invalid;OpenMXCoreContext *context;}Arguments;
static int exercise(OpenMXSolverInvocation *owner,void *opaque,double *result){
  Arguments *args=opaque;
  if(args->invalid){
    if(!strcmp(args->invalid,"axis"))(void)OpenMX_Stress_AxisBuffer(owner,0);
    else if(!strcmp(args->invalid,"workers")){OpenMXSolverRealTensor v;OpenMX_Stress_CutoffPrepare(owner,0,&v);}
    else{int *a,*b;OpenMX_Stress_OverlapCountsPrepare(owner,0,&a,&b);}CHECK(0);
  }
  for(int repeat=0;repeat<3;repeat++){
    args->atomic.species.cutoff[0]=repeat==1?.19:.11;args->atomic.species.cutoff[1]=repeat==1?.23:.14;
    for(int s=0;s<2;s++)args->atomic.species.species_index[s]=99;
    for(int k=0;k<9;k++)Stress_Tensor[k]=.003*(k+1);
    allocations=0;Stress_H0(owner,args->context,&args->atomic);CHECK(allocations==1&&FNAN[0]==99);verify_grid(&args->atomic);
    for(int a=0;a<3;a++)for(int b=0;b<3;b++){
      const long double delta=2e-6L;near(Stress_Tensor[3*a+b],.003*(3*a+b+1)+(strain_energy(a,b,delta)-strain_energy(a,b,-delta))/(2*delta));
    }
    int *send,*receive;OpenMX_Stress_OverlapCountsPrepare(owner,4,&send,&receive);CHECK(send!=receive);send[3]=17;receive[3]=29;CHECK(send[3]==17);
  }
  *result=1;return 1;
}
int main(int argc,char **argv){
  for(int a=0;a<4;a++){
    Gxyz[a]=positions[a];Dis[a]=distances[a];natn[a]=neighbors[a];ncn[a]=images[a];RMI1[a]=reverse_rows[a];for(int h=0;h<3;h++)reverse_rows[a][h]=reverse_data[a][h];
  }
  for(int a=1;a<=3;a++){positions[a][1]=.3*a;positions[a][2]=.07*a*a;positions[a][3]=-.11*a;}
  for(int a=1;a<=3;a++)for(int h=0;h<=FNAN[a];h++){
    const int b=neighbors[a][h];double r2=0;for(int d=1;d<=3;d++)r2+=pow(positions[a][d]-positions[b][d],2);distances[a][h]=sqrt(r2);
  }
  for(int m=1;m<=3;m++)for(int h=0;h<=FNAN[M2G[m]];h++){
    const int a=M2G[m],b=neighbors[a][h];int found=-1;for(int j=0;j<=FNAN[b];j++)if(neighbors[b][j]==a)found=j;CHECK(found>=0);reverse_data[m][h][0]=found;
  }
  double cutoff[2]={.11,.14},radial[2][2]={{0}};int index[2],mesh[2]={2,2};Arguments args={0};char context_error[256];args.context=openmx_core_create_host(context_error,sizeof(context_error));CHECK(args.context);
  {const OpenMXNeutralGridMetadataShape shape={2};OpenMXNeutralGridMetadataView metadata;char message[256];
   CHECK(openmx_core_neutral_grid_metadata_prepare(args.context,&shape,&metadata,message,sizeof(message)));
   fixture_counts=metadata.counts;fixture_spacing=metadata.spacing;}
args.invalid=argc>1?argv[1]:NULL;
  args.atomic.species.species=2;args.atomic.species.species_index=index;args.atomic.species.cutoff=cutoff;args.atomic.species.core_charge=Spe_Core_Charge;
  args.atomic.species.pao_mesh=args.atomic.species.vps_mesh=mesh;args.atomic.pao_log_radii=args.atomic.pao_radii=args.atomic.density=args.atomic.vps_log_radii=args.atomic.vps_radii=args.atomic.hartree_potential=(OpenMXRadialView2){&radial[0][0],{2,2}};
  for(int mesh_count=2;mesh_count<=3;mesh_count++){
    CoarseGL_Mesh=mesh_count;
    if(mesh_count==3){
      CoarseGL_Abscissae[0]=-sqrt(3.0/5.0);CoarseGL_Abscissae[1]=0;CoarseGL_Abscissae[2]=sqrt(3.0/5.0);
      CoarseGL_Weight[0]=CoarseGL_Weight[2]=5.0/9.0;CoarseGL_Weight[1]=8.0/9.0;
    }
    for(int empty=0;empty<2;empty++){Matomnum=empty?0:3;CHECK(OpenMX_Solver_Run(exercise,&args)==1);}
  }
  openmx_core_destroy(args.context);
  puts("PASS: native neutral grid/cutoff/stress, two/three Gauss points, three capacities each, empty/local atoms, logical teams1/4/2, independent coordinates/moments/nine strain components");return 0;
}
