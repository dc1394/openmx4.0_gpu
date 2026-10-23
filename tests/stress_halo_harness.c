/* Real C++ ownership. MPI/OpenMP are real in the four-rank variant; the
   serial variant models collectives and a single logical worker. The scalar
   hopping energy independently checks the production kinetic virial. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mpi.h"
#include "openmx_core_atomic.h"
#define measure_time 0
#ifndef STRESS_REAL_MPI
#define STRESS_REAL_MPI 0
#endif
#ifndef STRESS_REAL_OPENMP
#define STRESS_REAL_OPENMP 0
#endif
static void failure(int line,const char *what){
  fprintf(stderr,"FAIL line %d: %s\n",line,what);
#if STRESS_REAL_MPI
  MPI_Abort(MPI_COMM_WORLD,2);
#endif
  exit(2);
}
#define CHECK(c) do{if(!(c))failure(__LINE__,#c);}while(0)
static MPI_Comm mpi_comm_level1;
static int world_rank,world_size,atomnum=3,SpeciesNum=3,Matomnum,MatomnumF;
static int SpinP_switch,SO_switch,Hub_U_switch,F_U_flag,Constraint_NCS_switch,Zeeman_NCS_switch,Zeeman_NCO_switch;
static int F_Kin_flag,Cnt_switch,F_NL_flag,Hub_U_occupation,Host_ID,level_stdout;
static int M2G[4],F_M2G[4],F_G2M[4],WhatSpecies[4]={0,0,1,2},Spe_Total_CNO[3]={1,2,3},List_YOUSO[40]={[7]=3};
static int FNAN[4]={99,2,2,2},neighbors[4][3]={{0},{1,2,3},{2,3,1},{3,1,2}},*natn[4];
static int images[4][3],*ncn[4],rmi_data[4][3][3],*rmi_rows[4][3],**RMI1[4];
static int F_Snd_Num[4],F_Rcv_Num[4],F_TopMAN[4],snd_man[4][1],snd_gan[4][1],rcv_gan[4][1];
static int *Snd_MAN[4],*Snd_GAN[4],*Rcv_GAN[4];
static double xyz[4][4]={{0},{.2,-.7,.4},{1.3,.8,-.2},{-.5,1.7,1.2}},*Gxyz[4];
static double atv[2][4]={{0},{0,.15,-.12,.18}},time_per_atom[4],Stress_Tensor[9];
#include "openmx_stress_workspace.h"
#include "openmx_core_local_basis.h"
#if !STRESS_REAL_MPI
int MPI_Abort(MPI_Comm comm,int code){(void)comm;(void)code;_Exit(77);}
int MPI_Comm_size(MPI_Comm comm,int *out){(void)comm;*out=1;return 0;}
int MPI_Comm_rank(MPI_Comm comm,int *out){(void)comm;*out=0;return 0;}
int MPI_Barrier(MPI_Comm comm){(void)comm;return 0;}
int MPI_Finalize(void){return 0;}
int MPI_Allreduce(const void *s,void *r,int n,int type,int op,MPI_Comm comm){
  (void)comm;CHECK(type==MPI_DOUBLE&&op==MPI_SUM);memcpy(r,s,(size_t)n*sizeof(double));return 0;
}
int MPI_Isend(const void *b,int n,int t,int d,int tag,MPI_Comm c,MPI_Request *r){(void)b;(void)n;(void)t;(void)d;(void)tag;(void)c;(void)r;CHECK(0);return 1;}
int MPI_Recv(void *b,int n,int t,int d,int tag,MPI_Comm c,MPI_Status *r){(void)b;(void)n;(void)t;(void)d;(void)tag;(void)c;(void)r;CHECK(0);return 1;}
int MPI_Wait(MPI_Request *r,MPI_Status *s){(void)r;(void)s;CHECK(0);return 1;}
#endif
static void dtime(double *value){struct timespec now;timespec_get(&now,TIME_UTC);*value=(double)now.tv_sec+1e-9*now.tv_nsec;}
static dcomplex Complex(double r,double i){const dcomplex value={r,i};return value;}
/* Borrowed source rows use fixed fixture capacity; only C++ owns the output. */
typedef struct {
  double values[4][4][3][3][3];
  double *rows[4][4][3][3],**blocks[4][4][3],***atoms[4][4],****components[4];
} Matrix;
static Matrix density,imaginary,kinetic,contracted;
static double *****imaginary_groups[1];
static double density_value(int component,int center,int neighbor,int i,int j,int imag){
  const double value=.13+.07*component+.03*center-.02*neighbor+.011*i-.017*j;
  return imag?-.37*value:value;
}
static void displacement(int center,int h,double out[3]){
  const int neighbor=neighbors[center][h];
  for(int k=0;k<3;k++)out[k]=xyz[center][k+1]-xyz[neighbor][k+1]-atv[images[center][h]][k+1];
}
static double hopping(int center,int neighbor,int i,int j,const double r[3]){
  const double beta=.12+.02*center+.01*neighbor+.005*i+.003*j;
  const double coefficient=.4+.07*center+.03*neighbor+.013*i-.017*j;
  return coefficient*exp(-beta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
}
static void initialize(Matrix *matrix,int imaginary_flag,int derivative,int contract){
  for(int c=0;c<4;c++){
    matrix->components[c]=matrix->atoms[c];
    for(int a=0;a<4;a++){
      matrix->atoms[c][a]=matrix->blocks[c][a];
      for(int h=0;h<3;h++){
        matrix->blocks[c][a][h]=matrix->rows[c][a][h];
        for(int i=0;i<3;i++){
          matrix->rows[c][a][h][i]=matrix->values[c][a][h][i];
          for(int j=0;j<3;j++)matrix->values[c][a][h][i][j]=0;
        }
      }
    }
  }
  for(int a=1;a<=Matomnum;a++)for(int h=0;h<3;h++){
    const int g=M2G[a],b=neighbors[g][h];double r[3];displacement(g,h,r);
    for(int i=0;i<Spe_Total_CNO[WhatSpecies[g]];i++)for(int j=0;j<Spe_Total_CNO[WhatSpecies[b]];j++)for(int c=0;c<4;c++){
      double value=density_value(c,g,b,i,j,imaginary_flag);
      if(derivative){const double beta=.12+.02*g+.01*b+.005*i+.003*j;value=c?-2*beta*r[c-1]*hopping(g,b,i,j,r):0;}
      matrix->values[c][a][h][i][j]=value*(contract?1.17:1);
    }
  }
}
static double reference_energy(int axis,int other,double strain){
  if(!F_Kin_flag)return 0;
  double energy=0;
  for(int g=1;g<=3;g++)for(int h=1;h<3;h++){
    const int b=neighbors[g][h];double r[3];displacement(g,h,r);r[other]+=strain*r[axis];
    for(int i=0;i<Spe_Total_CNO[WhatSpecies[g]];i++)for(int j=0;j<Spe_Total_CNO[WhatSpecies[b]];j++){
      const double weight=SpinP_switch?density_value(0,g,b,i,j,0)+density_value(1,g,b,i,j,0):2*density_value(0,g,b,i,j,0);
      energy+=weight*hopping(g,b,i,j,r)*(Cnt_switch?1.17:1);
    }
  }
  return energy;
}
static void near(double a,double b){CHECK(isfinite(a)&&isfinite(b)&&fabs(a-b)<2e-8*(1+fabs(b)));}
static void verify_density(const OpenMXForceWorkspace *halo){
  CHECK(FNAN[0]==99&&halo->density.layout.shapes[0]==1&&halo->density.layout.shapes[1]==1);
  for(int c=0;c<=SpinP_switch;c++){
    CHECK(halo->density_rows[c][0][0][0][0]==0);
    for(int a=1;a<=Matomnum+MatomnumF;a++)for(int h=0;h<3;h++){
      const int g=F_M2G[a],b=neighbors[g][h];
      for(int i=0;i<Spe_Total_CNO[WhatSpecies[g]];i++)for(int j=0;j<Spe_Total_CNO[WhatSpecies[b]];j++)
        CHECK(halo->density_rows[c][a][h][i][j]==density_value(c,g,b,i,j,0));
    }
  }
  CHECK((halo->imaginary_rows!=NULL)==(SO_switch!=0));
  if(halo->imaginary_rows)for(int c=0;c<2;c++)for(int a=1;a<=Matomnum+MatomnumF;a++)for(int h=0;h<3;h++){
    const int g=F_M2G[a],b=neighbors[g][h];
    for(int i=0;i<Spe_Total_CNO[WhatSpecies[g]];i++)for(int j=0;j<Spe_Total_CNO[WhatSpecies[b]];j++)
      CHECK(halo->imaginary_rows[c][a][h][i][j]==density_value(c,g,b,i,j,1));
  }
}
typedef struct {const char *invalid;} State;
static int callback(const OpenMXForceWorkspace *halo,void *opaque,double *result){
  const State *state=opaque;OpenMXSolverInvocation *scratch=halo->scratch;
  OpenMXAtomicViews atom_views={0};atom_views.species.contracted_orbitals=Spe_Total_CNO;
  const OpenMXAtomicViews *atomic=&atom_views;
  OpenMXLocalBasisViews basis={0};const OpenMXLocalBasisViews *local_basis=&basis;
  double *****CDM=density.components,******iDM=imaginary_groups;
  double *****H0=kinetic.components,*****CntH0=contracted.components,*****OLP=NULL;
  double *****CDM0,*****iDM0;
  int *Snd_CDM0_Size,*Rcv_CDM0_Size,*Snd_iDM0_Size,*Rcv_iDM0_Size;
  double *tmp_array,*tmp_array2;
  int numprocs=world_size,myid=world_rank,tag=999,ID,IDS,IDR,n,num,size1,size2;
  int OMPID,Nthrds,Nthrds0,Nprocs,Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,Mh_AN;
  int i,j,k,l,so,spin,spinmax,s1,s2,ian,jan,start_q_AN,q_AN,Gq_AN,Mq_AN,Qwan,kl,Rn1,Rn2,tno1,tno2;
  double stime,etime,Stime_atom,Etime_atom,dEx,dEy,dEz,pref,lx1,ly1,lz1,lx2,ly2,lz2,lx,ly,lz;
  double Stress[9],My_Stress[9];OpenMXSolverRealTensor stress_threads;
  OpenMXSolverComplexTensor Hx,Hy,Hz;
  double ***HUx,***HUy,***HUz;dcomplex ****NC_HUx,****NC_HUy,****NC_HUz;
  MPI_Request request;MPI_Status stat;
  if(state->invalid){
    if(!strcmp(state->invalid,"count"))OpenMX_Stress_HaloBuffer(scratch,"invalid",-1);
    else OpenMX_Stress_KineticPrepare(scratch,"invalid",0,&Hx);
    CHECK(0);
  }
  CHECK(OpenMX_Stress_HaloBuffer(scratch,"empty-message",0)!=NULL);
  for(int repeat=0;repeat<3;repeat++){
    List_YOUSO[7]=repeat==1?4:3;
#include "stress_halo_native.inc"
    verify_density(halo);
    if(!SO_switch){
      memset(Stress_Tensor,0,sizeof(Stress_Tensor));
#include "stress_kinetic_native.inc"
      for(int axis=0;axis<3;axis++)for(int other=0;other<3;other++){
        const double delta=2e-6;
        near(Stress_Tensor[3*axis+other],(reference_energy(axis,other,delta)-reference_energy(axis,other,-delta))/(2*delta));
      }
    }
    MPI_Barrier(mpi_comm_level1);
  }
  *result=1;return 1;
}
static void setup(void){
  CHECK(world_size==1||world_size==4);
  for(int g=0;g<4;g++){
    natn[g]=neighbors[g];ncn[g]=images[g];Gxyz[g]=xyz[g];
    for(int h=0;h<3;h++)images[g][h]=h==2;
  }
  if(world_size==1){Matomnum=3;MatomnumF=0;M2G[1]=F_M2G[1]=3;M2G[2]=F_M2G[2]=1;M2G[3]=F_M2G[3]=2;}
  else{
    Matomnum=world_rank<3?1:0;MatomnumF=3-Matomnum;int next=Matomnum+1;
    if(Matomnum)M2G[1]=F_M2G[1]=world_rank+1;
    for(int peer=0;peer<4;peer++){
      Snd_MAN[peer]=snd_man[peer];Snd_GAN[peer]=snd_gan[peer];Rcv_GAN[peer]=rcv_gan[peer];
      if(peer==world_rank)continue;
      F_Snd_Num[peer]=Matomnum;snd_man[peer][0]=1;snd_gan[peer][0]=world_rank+1;
      if(peer<3){F_Rcv_Num[peer]=1;F_TopMAN[peer]=next;rcv_gan[peer][0]=peer+1;F_M2G[next++]=peer+1;}
    }
    CHECK(next==4);
  }
  for(int a=1;a<=3;a++)F_G2M[F_M2G[a]]=a;
  for(int a=1;a<=Matomnum;a++){
    RMI1[a]=rmi_rows[a];const int g=M2G[a];
    for(int h=0;h<3;h++){
      rmi_rows[a][h]=rmi_data[a][h];const int neighbor=neighbors[g][h];
      for(int q=0;q<3;q++){
        int index=0;while(neighbors[neighbor][index]!=neighbors[g][q])++index;
        rmi_data[a][h][q]=index;
      }
    }
  }
}
int main(int argc,char **argv){
#if STRESS_REAL_MPI
  int provided;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);CHECK(provided>=MPI_THREAD_FUNNELED);
  mpi_comm_level1=MPI_COMM_WORLD;MPI_Comm_rank(mpi_comm_level1,&world_rank);MPI_Comm_size(mpi_comm_level1,&world_size);
#if STRESS_REAL_OPENMP
  omp_set_num_threads(2);
#endif
#else
  mpi_comm_level1=1;world_size=1;
#endif
  setup();int cases=0;State state={argc==2?argv[1]:NULL};
  for(int spin=0;spin<3;spin++)for(int imag=0;imag<2;imag++){
    SpinP_switch=spin==2?3:spin;if(imag&&SpinP_switch!=3)continue;SO_switch=imag;
    for(int cnt=0;cnt<2;cnt++)for(int active=0;active<2;active++){
      Cnt_switch=cnt;F_Kin_flag=active;initialize(&density,0,0,0);initialize(&imaginary,1,0,0);
      initialize(&kinetic,0,1,0);initialize(&contracted,0,1,1);imaginary_groups[0]=imaginary.components;
      const OpenMXForceInput input={atomnum,Matomnum,MatomnumF,SpeciesNum,SpinP_switch,imag,
          F_M2G,WhatSpecies,Spe_Total_CNO,FNAN,natn};
      CHECK(OpenMX_Mechanical_Run(&input,callback,&state)==1&&FNAN[0]==99);++cases;
      MPI_Barrier(mpi_comm_level1);
    }
  }
  CHECK(cases==16);if(!world_rank)printf("PASS: %d halo/kinetic cases, three shapes each, %d ranks, %d workers; padding/imaginary/strain checks\n",cases,world_size,omp_get_max_threads());
#if STRESS_REAL_MPI
  MPI_Finalize();
#endif
  return 0;
}
