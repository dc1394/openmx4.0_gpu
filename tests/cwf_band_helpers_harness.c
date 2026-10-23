/* Production CWF helpers are inserted below the small host/MPI adapters.
   The analytic real-space Hamiltonians are independent numerical oracles. */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "openmx_complex.h"
#include "openmx_core_cwf.h"
#include "openmx_core_chemical.h"

#define CHECK(c) do { if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(2);} } while(0)
#define PI 3.141592653589793238462643383279502884
#define measure_time 0
#define YOUSO10 1024
#define fp_bsize 1024
#define GPUSOLVER 7
#define F77_NAME(lower,upper) lower
static MPI_Comm mpi_comm_level1=1;
static int atomnum,Matomnum,SpeciesNum,*M2G,*WhatSpecies,*Spe_Total_CNO,*FNAN,**natn,**ncn,**atv_ijk;
static int nblk,np_rows,np_cols,my_prow,my_pcol,na_rows,memoryusage_fileout;
#include "openmx_cwf_workspace.h"

static int Host_ID=0,level_stdout=0,SpinP_switch,scf_eigen_lib_flag,GPU_CPU_SWITCH_NUM=0;
static char filename[YOUSO10],filepath[YOUSO10];
static double rtv[4][4];
static int fake_rank,fake_size=1,broadcasts,component,solve_calls,native_spins,native_complex;
static const int counts[3]={3,0,6},local_atoms[3]={1,0,1},ga_order[3]={2,0,1};
static const double packed[2][9]={{2.75,.3,-.4,1.5,.2,.3,.2,-.5,-.4},
                                 {0,.2,-.15,0,.1,-.2,-.1,0,.15}};
static const double points_x[8]={0,.25,.5,.17,-.25,-1.0/12,1.0/12,.25};
static void near(double complex a,double complex b) {CHECK(cabs(a-b)<3e-13);}
static void dtime(double *time) {*time=0;}
static void PrintMemory(const char *name,double bytes,const char *label) {(void)name;(void)bytes;(void)label;}
static void fnjoint(const char *path,const char *name,char *suffix) {
  char copy[YOUSO10];strcpy(copy,suffix);CHECK(strlen(path)+strlen(name)+strlen(copy)<YOUSO10);
  strcpy(suffix,path);strcat(suffix,name);strcat(suffix,copy);
}
int MPI_Abort(MPI_Comm comm,int code) {(void)comm;(void)code;_Exit(77);}
int MPI_Comm_size(MPI_Comm comm,int *n) {(void)comm;*n=fake_size;return 0;}
int MPI_Comm_rank(MPI_Comm comm,int *rank) {(void)comm;*rank=fake_rank;return 0;}
int MPI_Barrier(MPI_Comm comm) {(void)comm;return 0;}
int MPI_Comm_split(MPI_Comm comm,int color,int key,MPI_Comm *out) {(void)comm;(void)color;(void)key;*out=2;return 0;}
int MPI_Comm_free(MPI_Comm *comm) {*comm=0;return 0;}
int MPI_Comm_c2f(MPI_Comm comm) {return comm;}
int MPI_Bcast(void *data,int n,int type,int root,MPI_Comm comm) {
  (void)comm;CHECK(fake_size==3 && type==MPI_INT && root==broadcasts%3);
  const int group=broadcasts++/3;int value=0;
  if(group==0){CHECK(n==1);value=counts[root];}
  else if(group==1){CHECK(n==1);value=local_atoms[root];}
  else {CHECK(group==2 && n==local_atoms[root]);value=ga_order[root];}
  if(n){if(root==fake_rank)CHECK(*(int*)data==value);*(int*)data=value;}
  return 0;
}
int MPI_Allreduce(const void *send,void *receive,int n,int type,int op,MPI_Comm comm) {
  (void)comm;CHECK(fake_size==3 && send==MPI_IN_PLACE && n==9 && type==MPI_DOUBLE && op==MPI_SUM);
  const int start=fake_rank==0?0:3,end=start+counts[fake_rank];double *values=receive;
  for(int i=0;i<9;i++)near(values[i],i>=start && i<end?packed[component][i]:0);
  memcpy(values,packed[component],9*sizeof(double));return 0;
}

static double complex entry(int spin,int row,int column,int cell,int complex_values) {
  const double complex a[4]={2,.25,.25,3};
  const double complex t[4]={.1,-.04,.07,.12};
  const double complex ac[4]={2,.25-.2*I,.25+.2*I,3};
  const double complex tc[4]={.1+.03*I,-.04+.02*I,.07-.05*I,.12-.04*I};
  const int offset=2*column+row;
  if(cell==0)return (complex_values?ac[offset]:a[offset])+(row==column?5*spin:0);
  if(cell==1)return (spin+1)*(complex_values?tc[offset]:t[offset]);
  return conj((spin+1)*(complex_values?tc[2*row+column]:t[2*row+column]));
}
static void diagonalize(int n,const dcomplex *h,int ld,double *eigen) {
  CHECK(n==2 && ld==2 && solve_calls<8*native_spins);
  const int spin=solve_calls%native_spins,point=solve_calls/native_spins;
  const double complex phase=cexp(2*PI*I*points_x[point]);double complex matrix[4];
  for(int j=0;j<2;j++)for(int i=0;i<2;i++){
    matrix[2*j+i]=h[2*j+i].r+I*h[2*j+i].i;
    CHECK(isfinite(creal(matrix[2*j+i])) && isfinite(cimag(matrix[2*j+i])));
    near(matrix[2*j+i],entry(spin,i,j,0,native_complex)+
         entry(spin,i,j,1,native_complex)*phase+entry(spin,i,j,-1,native_complex)*conj(phase));
  }
  near(matrix[1],conj(matrix[2]));near(cimag(matrix[0]),0);near(cimag(matrix[3]),0);
  const double middle=.5*creal(matrix[0]+matrix[3]);
  const double radius=hypot(.5*creal(matrix[0]-matrix[3]),cabs(matrix[1]));
  eigen[0]=middle-radius;eigen[1]=middle+radius;++solve_calls;
}
static void solve_evp_complex(int *n,int *nev,dcomplex *h,int *ld,double *e,dcomplex *z,
    int *ldz,int *block,int *rows,int *cols) {
  (void)z;(void)ldz;(void)block;(void)rows;(void)cols;CHECK(*nev==*n);diagonalize(*n,h,*ld,e);
}
static void elpa_solve_evp_complex_2stage_double_impl(int *n,int *nev,dcomplex *h,int *ld,
    double *e,dcomplex *z,int *ldz,int *block,int *columns,int *rows,int *cols,int *world) {
  (void)columns;(void)world;solve_evp_complex(n,nev,h,ld,e,z,ldz,block,rows,cols);
}
static void OpenMX_GpuSolver_DenseZheevx_0based(dcomplex *h,dcomplex *z,double *e,int n,int nev,const char *label) {
  (void)z;CHECK(n==nev && label && *label);diagonalize(n,h,n,e);
}

#include "cwf_native.inc"

static void layout(OpenMXSolverInvocation *owner) {
  int counts_by_path[4]={0,3,1,4};
  OpenMXCWFBlocks *values=OpenMX_CWF_PathValues(owner,3,counts_by_path,2,3);
  for(int path=0;path<4;path++)for(int spin=0;spin<2;spin++)for(int point=0;point<=counts_by_path[path];point++){
    double *row=OpenMX_CWF_PathRow(values,path,point,spin);CHECK(row);
    for(int e=0;e<5;e++)row[e]=10000*path+1000*spin+100*point+e;
  }
  CHECK(values==OpenMX_CWF_PathValues(owner,3,counts_by_path,2,3));
  for(int path=0;path<4;path++)for(int spin=0;spin<2;spin++)for(int point=0;point<=counts_by_path[path];point++){
    double *row=OpenMX_CWF_PathRow(values,path,point,spin);
    for(int e=0;e<5;e++)near(row[e],10000*path+1000*spin+100*point+e);
  }
  counts_by_path[1]=5;values=OpenMX_CWF_PathValues(owner,3,counts_by_path,2,3);
  CHECK(values[1].shape.cells[0]==6 && values[2].shape.cells[0]==2);
  const double expected[5]={.13,.33,.53,.73,.93};
  for(int i=1;i<=5;i++)near(OpenMX_CWF_PathCoordinate(.13,.93,i,5),expected[i-1]);
  near(OpenMX_CWF_PathCoordinate(.17,.83,1,1),.17);
  CHECK(OpenMX_CWF_SVDWorkSize(1)==1 && OpenMX_CWF_SVDWorkSize(INT_MAX)==INT_MAX);
  OpenMX_CWF_SVDStatus("successful fixture query",0);
}
static void matrix(OpenMXSolverInvocation *owner) {
  int species[3]={0,0,1},orbitals[2]={2,1},neighbors[3]={0,1,1},local_to_global[2];
  int n0[2]={0},n1[2]={1,2},n2[2]={2,1},*neighbor_table[3]={n0,n1,n2};
  int c0[2]={0},c1[2]={0,1},c2[2]={0,2},*cell_table[3]={c0,c1,c2};
  int r0[4]={0},r1[4]={0,1,-2,1},r2[4]={0,-1,2,-1},*cells[3]={r0,r1,r2};
  int mp[3];double *packed_values=OpenMX_Solver_OverwriteReal(owner,"fixture:packed",9);
  atomnum=2;SpeciesNum=2;WhatSpecies=species;Spe_Total_CNO=orbitals;FNAN=neighbors;
  natn=neighbor_table;ncn=cell_table;atv_ijk=cells;M2G=local_to_global;
  nblk=1;np_rows=3;np_cols=1;my_pcol=0;na_rows=1;fake_size=3;
  for(int trial=0;trial<5;trial++){
    const double k1=.17+.13*trial,k2=.11,k3=-.07;
    const double complex phase=cexp(2*PI*I*(k1-2*k2+k3));
    const double complex expected[9]={1.5,.2-.1*I,conj((.3-.2*I)*phase),
      .2+.1*I,-.5,conj((-.4+.15*I)*phase),(.3-.2*I)*phase,(-.4+.15*I)*phase,2.75};
    for(fake_rank=0;fake_rank<3;fake_rank++){
      double data[2][2][2]={{{0}}};double *rows[2][2];double **blocks[2];double ***atoms[2]={NULL};
      for(int h=0;h<2;h++){blocks[h]=rows[h];for(int i=0;i<2;i++)rows[h][i]=data[h][i];}
      atoms[1]=blocks;Matomnum=local_atoms[fake_rank];local_to_global[1]=ga_order[fake_rank];my_prow=fake_rank;
      dcomplex local[3]={{0}};
      for(component=0;component<2;component++){
        if(fake_rank==0){data[0][0][0]=component?0:2.75;data[1][0][0]=component?.2:.3;data[1][0][1]=component?-.15:-.4;}
        else if(fake_rank==2){
          data[0][0][0]=component?0:1.5;data[0][0][1]=component?.1:.2;
          data[0][1][0]=component?-.1:.2;data[0][1][1]=component?0:-.5;
          data[1][0][0]=component?-.2:.3;data[1][1][0]=component?.15:-.4;
        }
        broadcasts=0;Construct_Band_Ms(owner,component,atoms,packed_values,local,mp,k1,k2,k3);
        CHECK(broadcasts==9 && mp[1]==1 && mp[2]==3);
      }
      for(int j=0;j<3;j++)near(local[j].r+I*local[j].i,expected[3*j+fake_rank]);
    }
  }
  fake_rank=0;fake_size=1;
}
static void bands(OpenMXSolverInvocation *owner,int complex_values,int spinful) {
  const int lengths[4]={0,3,1,4};int *point_counts=(int*)lengths;
  double kp[4][3][4]={{{0}}},*kp_rows[4][3],**paths[4];
  char *labels[4][3],**names[4];
  for(int path=0;path<4;path++){
    paths[path]=kp_rows[path];names[path]=labels[path];
    for(int end=0;end<3;end++){kp_rows[path][end]=kp[path][end];labels[path][end]="K";}
  }
  kp[1][2][1]=.5;kp[2][1][1]=.17;kp[2][2][1]=.43;kp[3][1][1]=-.25;kp[3][2][1]=.25;
  OpenMXCWFGrids grids={0};OpenMXChemicalPotentials chemical={0};
  OpenMXCWFGrid *g=&grids.matrix[OPENMX_CWF_BAND_GRID];
  g->local_rows=g->local_columns=g->maximum_rows=g->maximum_columns=2;g->block_size=1;
  g->process_rows=g->process_columns=1;g->descriptor[8]=2;
  OpenMXCWFBlocks hopping;SpinP_switch=complex_values?3:spinful;native_spins=spinful?2:1;native_complex=complex_values;
  OpenMX_CWF_Blocks(owner,"fixture:Hop",native_spins,3,1,1,5,complex_values?sizeof(double complex):sizeof(double),&hopping);
  for(int spin=0;spin<native_spins;spin++)for(int cell=0;cell<3;cell++)for(int j=0;j<2;j++)for(int i=0;i<2;i++){
    void *row=openmx_cwf_block_cell(&hopping,spin,cell,0,0);
    if(complex_values)((double complex*)row)[2*j+i]=entry(spin,i,j,cell-1,1);
    else((double*)row)[2*j+i]=creal(entry(spin,i,j,cell-1,0));
  }
  solve_calls=0;
  if(complex_values)Band_Dispersion_NonCol_CWF(&grids,owner,&chemical,3,point_counts,paths,names,2,&hopping,3,1,1,1);
  else Band_Dispersion_Col_CWF(&grids,owner,&chemical,3,point_counts,paths,names,2,&hopping,3,1,1,1);
  CHECK(solve_calls==8*native_spins);
}
static int execute(OpenMXSolverInvocation *owner,void *opaque,double *result) {
  const char *mode=opaque;
  if(!strcmp(mode,"layout"))layout(owner);
  else if(!strcmp(mode,"matrix"))matrix(owner);
  else if(!strcmp(mode,"col"))bands(owner,0,1);
  else if(!strcmp(mode,"nc"))bands(owner,1,0);
  else if(!strcmp(mode,"bad-info"))OpenMX_CWF_SVDStatus("forced SVD query",-4);
  else if(!strcmp(mode,"bad-convergence"))OpenMX_CWF_SVDStatus("forced SVD decomposition",3);
  else if(!strcmp(mode,"bad-point"))OpenMX_CWF_PathCoordinate(0,1,0,3);
  else if(!strcmp(mode,"bad-path")){int count=-1;OpenMX_CWF_PathValues(owner,0,&count,1,2);}
  else if(!strcmp(mode,"bad-zero"))OpenMX_CWF_SVDWorkSize(0);
  else if(!strcmp(mode,"bad-nan"))OpenMX_CWF_SVDWorkSize(NAN);
  else if(!strcmp(mode,"bad-infinite"))OpenMX_CWF_SVDWorkSize(INFINITY);
  else if(!strcmp(mode,"bad-range"))OpenMX_CWF_SVDWorkSize((double)INT_MAX+1);
  else if(!strcmp(mode,"bad-fraction"))OpenMX_CWF_SVDWorkSize(1.5);
  else CHECK(0);
  *result=1;return 1;
}
int main(int argc,char **argv) {
  CHECK(argc==3);scf_eigen_lib_flag=atoi(argv[2]);snprintf(filename,sizeof(filename),"%s-%d",argv[1],scf_eigen_lib_flag);
  for(int i=1;i<=3;i++)rtv[i][i]=1;
  near(OpenMX_Solver_Run(execute,argv[1]),1);puts("PASS: production CWF band helpers");return 0;
}
