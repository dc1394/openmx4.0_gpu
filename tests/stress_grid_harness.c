/* Production Stress3, real C++ ownership, and independent strain oracles.
   MPI, one logical OpenMP worker, orbitals and XC producers are modeled. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "openmx_core_atomic.h"
#include "openmx_core_local_basis.h"
#define CHECK(c) do{if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(2);}}while(0)
static MPI_Comm mpi_comm_level1=1;
static int SpinP_switch,Max_GridN_Atom=3,List_YOUSO[40]={[7]=3},Matomnum=1;
static int atomnum=2,SpeciesNum=2,Spe_Total_CNO[2]={1,2};
static int Cnt_switch,ProExpn_VNA,E_Field_switch,XC_switch,Host_ID=0,level_stdout=0;
static int F_dVHart_flag=1,F_Vxc_flag=1,F_VNA_flag=1,F_VEF_flag=1;
static int My_NumGridD=216,My_NumGridB_AB=2,Max_Grid_Index_D[4]={0,5,5,5},Min_Grid_Index_D[4]={0};
static int Num_Rcv_Grid_B2D[1]={2},recv_indices[2]={86,129},send_indices[2]={1,0};
static int *Index_Rcv_Grid_B2D[1]={recv_indices},*Index_Snd_Grid_B2D[1]={send_indices};
static double GridVol=.63,Stress_Tensor[9],Gxyz[3][32],atv[2][4]={{0},{0,.11,-.17,.19}};
static double dVHart_Grid[3]={.3,-.2,.1},VNA_Grid[3]={.4,.1,-.3},VEF_Grid[3]={-.15,.21,.32};
static double xc_values[4][3]={{-.6,.2,.1},{.15,-.4,.21},{.32,-.11,.09},{-.21,.17,.4}},*Vxc_Grid[4];
static double density_data[4][216],*Density_Grid_D[4];
static int M2G[2]={0,2},WhatSpecies[3]={0,0,1},FNAN[3]={0,0,1},G2ID[3]={0,1,0},F_G2M[3]={0,0,1};
static int grid_counts[3]={0,0,3},*GridN_Atom=grid_counts;
static int grid_ids[5]={0,1,2,3,4},cell_ids[5]={1,0,1,0,1},mgrid_ids[5]={0,1,-1,0,1};
static int *GridListAtom[2]={0,grid_ids},*CellListAtom[2]={0,cell_ids},*MGridListAtom[2]={0,mgrid_ids};
static int neighbor_atoms[2]={2,1},neighbor_images[2]={0,1},*natn[3]={0,0,neighbor_atoms},*ncn[3]={0,0,neighbor_images};
static int overlap_counts[2]={3,3},*NumOLG[2]={0,overlap_counts};
static int source_points[2][5]={{2,0,1,4,3},{1,2,0,3,4}},neighbor_points[2][5]={{1,2,0,3,4},{2,0,1,4,3}};
static int *source_rows[2]={source_points[0],source_points[1]},*neighbor_rows[2]={neighbor_points[0],neighbor_points[1]};
static int **GListTAtoms1[2]={0,source_rows},**GListTAtoms2[2]={0,neighbor_rows};
static double local_values[5][3],*local_rows[5],**Orbs_Grid[2]={0,local_rows};
static double remote_values[5][3],*remote_rows[5],**remote_neighbors[2]={0,remote_rows},***Orbs_Grid_FNAN[2]={0,remote_neighbors};
static double dm_values[4][2][3][3],*dm_columns[4][2][3],**dm_neighbors[4][2],***dm_atoms[4][2],****dm_spins[4],*****dm_groups[1];
static int xc_calls,orbital_calls,cnt_calls,collectives;
#include "openmx_stress_workspace.h"
int MPI_Abort(MPI_Comm comm,int code){(void)comm;(void)code;_Exit(77);}
int MPI_Comm_size(MPI_Comm comm,int *n){(void)comm;*n=1;return 0;}
int MPI_Comm_rank(MPI_Comm comm,int *n){(void)comm;*n=0;return 0;}
int MPI_Allreduce(const void *s,void *r,int n,int type,int op,MPI_Comm comm){
  (void)comm;CHECK(n==9&&type==MPI_DOUBLE&&op==MPI_SUM);memcpy(r,s,9*sizeof(double));++collectives;return 0;
}
static void dtime(double *t){static double time;*t=time+=.01;}
static void Get_Grid_XYZ(int n,double out[4]){
  out[0]=0;out[1]=.13+.19*n;out[2]=-.24+.07*n;out[3]=.37-.11*n;
}
static double orbital(int i,const double r[3]){
  const double beta=.27+.11*i,linear=.2+.07*i+r[i%3];
  return linear*exp(-beta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]))*(Cnt_switch?1.13:1);
}
static void fill_orbitals(const OpenMXAtomicViews *a,int species,double x,double y,double z,double **out){
  const double r[3]={x,y,z};
  for(int i=0;i<a->species.contracted_orbitals[species];i++){
    const double beta=.27+.11*i,e=exp(-beta*(x*x+y*y+z*z))*(Cnt_switch?1.13:1);
    out[0][i]=orbital(i,r);
    for(int k=0;k<3;k++)out[k+1][i]=(i%3==k?e:0)-2*beta*r[k]*out[0][i];
  }
}
static void Get_dOrbitals_work(const OpenMXAtomicViews *a,int species,double x,double y,double z,double **out,OpenMXOrbitalBuffers work){
  CHECK(work.RF&&work.dRF&&work.AF&&work.dAFQ&&work.dAFP);++orbital_calls;fill_orbitals(a,species,x,y,z,out);
}
static void Get_Cnt_dOrbitals(const OpenMXAtomicViews *a,int atom,double x,double y,double z,double **out,const OpenMXLocalBasisViews *basis){
  CHECK(atom==1&&basis);++cnt_calls;fill_orbitals(a,WhatSpecies[M2G[atom]],x,y,z,out);
}
/* Gradient energy uses a symmetric, anisotropic, coupled quadratic form. */
static void gradient(int spin,int point,double out[3]){
  for(int k=0;k<3;k++)out[k]=.07*(k+1)+.03*spin+.001*(point+1)*(k==1?-1:1);
}
static void xc_derivative(const double g[3],double out[3]){
  const double k[3][3]={{.7,.11,-.08},{.11,.5,.06},{-.08,.06,.9}};
  for(int i=0;i<3;i++){out[i]=0;for(int j=0;j<3;j++)out[i]+=k[i][j]*g[j];}
}
static void Set_XC_Grid(int mode,int part,int kind,double *n0,double *n1,double *n2,double *n3,
    double *v0,double *v1,double *v2,double *v3,double ***axc,double ***den){
  CHECK(mode==2&&part==3&&kind==4&&n0&&n1&&n2&&n3&&!v0&&!v1&&!v2&&!v3);++xc_calls;
  for(int spin=0;spin<2;spin++)for(int n=0;n<My_NumGridD;n++){
    double g[3],a[3];gradient(spin,n,g);xc_derivative(g,a);
    for(int k=0;k<3;k++){den[spin][k][n]=g[k];axc[spin][k][n]=a[k];}
  }
}
#include "stress_grid_native.inc"

/* Independently deform central orbital displacements. Neighbor values and
   potential are held fixed, as in this single derivative contribution. */
static double potential(int spin,int n){
  if(n<0)return 0;
  const double scalar=F_dVHart_flag*dVHart_Grid[n]+(!ProExpn_VNA?F_VNA_flag*VNA_Grid[n]:0)+(E_Field_switch?F_VEF_flag*VEF_Grid[n]:0);
  if(SpinP_switch==0)return 4*(scalar+F_Vxc_flag*Vxc_Grid[0][n]);
  if(SpinP_switch==1)return 2*(scalar+F_Vxc_flag*Vxc_Grid[spin][n]);
  if(spin<2)return 2*(scalar+F_Vxc_flag*Vxc_Grid[spin][n]);
  return (spin==2?4:-4)*F_Vxc_flag*Vxc_Grid[spin][n];
}
static double grid_energy(const OpenMXAtomicViews *atomic,int row,int column,double strain){
  double total=0;
  if(!Matomnum)return 0;
  for(int h=0;h<2;h++)for(int overlap=0;overlap<overlap_counts[h];overlap++){
    const int nc=source_points[h][overlap],nh=neighbor_points[h][overlap],species=WhatSpecies[neighbor_atoms[h]];
    double xyz[4],r[3];Get_Grid_XYZ(grid_ids[nc],xyz);
    for(int k=0;k<3;k++)r[k]=xyz[k+1]+atv[cell_ids[nc]][k+1]-Gxyz[2][k+1];
    r[row]-=strain*r[column];
    for(int spin=0;spin<=SpinP_switch;spin++)for(int i=0;i<2;i++)for(int j=0;j<atomic->species.contracted_orbitals[species];j++){
      const double neighbor=h?remote_values[overlap][j]:local_values[nh][j];
      total+=GridVol*potential(spin,mgrid_ids[nc])*orbital(i,r)*neighbor*dm_values[spin][h][i][j];
    }
  }
  return total;
}
/* F^-T transforms the density gradient; this checks the full GGA matrix. */
static double gradient_energy(int row,int column,double strain){
  if(XC_switch!=4)return 0;
  double total=0;const int spins=SpinP_switch?2:1;
  for(int spin=0;spin<spins;spin++)for(int b=0;b<My_NumGridB_AB;b++){
    const int n=recv_indices[b?0:1];double g[3],a[3];gradient(spin,n,g);g[column]-=strain*g[row];xc_derivative(g,a);
    for(int k=0;k<3;k++)total+=.5*g[k]*a[k]*GridVol*(SpinP_switch?1:2);
  }
  return total;
}
static void near(double a,double b){
  if(!(isfinite(a)&&isfinite(b)&&fabs(a-b)<2e-8*(1+fabs(b))))
    fprintf(stderr,"strain mismatch native=%.17g expected=%.17g spin=%d gga=%d cnt=%d projected=%d field=%d atoms=%d\n",a,b,SpinP_switch,XC_switch,Cnt_switch,ProExpn_VNA,E_Field_switch,Matomnum);
  CHECK(isfinite(a)&&isfinite(b)&&fabs(a-b)<2e-8*(1+fabs(b)));
}
static int callback(OpenMXSolverInvocation *scope,void *opaque,double *result){
  OpenMXAtomicViews *atomic=opaque;OpenMXElectronicViews electronic={0};OpenMXLocalBasisViews basis={0};electronic.DM=dm_groups;
  for(int repeat=0;repeat<3;repeat++){
    Max_GridN_Atom=repeat==1?5:3;grid_counts[2]=Max_GridN_Atom;overlap_counts[0]=overlap_counts[1]=Max_GridN_Atom;
    const int active=Matomnum;My_NumGridD=active?216:0;My_NumGridB_AB=active?2:0;Num_Rcv_Grid_B2D[0]=active?2:0;
    memset(Stress_Tensor,0,sizeof(Stress_Tensor));const int before=collectives,old_xc=xc_calls;
    Stress3(scope,atomic,&electronic,&basis);
    CHECK(collectives-before==1+(XC_switch==4?(SpinP_switch?2:1):0));CHECK(xc_calls-old_xc==(XC_switch==4));
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){
      const double delta=2e-6;
      const double energy_plus=grid_energy(atomic,i,j,delta)+gradient_energy(i,j,delta);
      const double energy_minus=grid_energy(atomic,i,j,-delta)+gradient_energy(i,j,-delta);
      near(Stress_Tensor[3*i+j],(energy_plus-energy_minus)/(2*delta));
    }
  }
  *result=1;return 1;
}
static void setup(void){
  Gxyz[2][1]=-.19;Gxyz[2][2]=.14;Gxyz[2][3]=.09;
  for(int spin=0;spin<4;spin++){Vxc_Grid[spin]=xc_values[spin];Density_Grid_D[spin]=density_data[spin];
    for(int h=0;h<2;h++){for(int i=0;i<3;i++){dm_columns[spin][h][i]=dm_values[spin][h][i];for(int j=0;j<3;j++)dm_values[spin][h][i][j]=.05+.03*spin+.07*h+.11*i-.04*j;}
      dm_neighbors[spin][h]=dm_columns[spin][h];}
    dm_atoms[spin][1]=dm_neighbors[spin];dm_spins[spin]=dm_atoms[spin];}
  dm_groups[0]=dm_spins;
  for(int n=0;n<5;n++){local_rows[n]=local_values[n];remote_rows[n]=remote_values[n];for(int i=0;i<3;i++){local_values[n][i]=.2+.09*n+.13*i;remote_values[n][i]=-.1+.07*n-.11*i;}}
}
static int invalid(OpenMXSolverInvocation *scope,void *opaque,double *result){
  const char *mode=opaque;OpenMXStressGridScratch grid;OpenMXSolverRealTensor tensor;
  if(!strcmp(mode,"spin"))OpenMX_Stress_GridPrepare(scope,2,3,3,&grid);
  else if(!strcmp(mode,"points"))OpenMX_Stress_GridPrepare(scope,0,-1,3,&grid);
  else if(!strcmp(mode,"orbitals"))OpenMX_Stress_OrbitalPrepare(scope,0,&tensor);
  else if(!strcmp(mode,"gradient"))OpenMX_Stress_GradientPrepare(scope,"invalid",-1,&tensor);
  else CHECK(0);
  *result=0;return 1;
}
int main(int argc,char **argv){
  setup();char error[256];double result=0;
  if(argc==2){CHECK(openmx_core_solver_run(1,invalid,argv[1],&result,error,sizeof(error)));return 3;}
  int counts[2]={1,2};OpenMXAtomicViews atomic={0};atomic.species.contracted_orbitals=counts;atomic.orbitals.extent[1]=2;atomic.orbitals.extent[2]=2;
  int cases=0;
  for(int spin=0;spin<3;spin++)for(int gga=0;gga<2;gga++)for(int contracted=0;contracted<2;contracted++)for(int projected=0;projected<2;projected++)for(int field=0;field<2;field++)for(int empty=0;empty<2;empty++){
    SpinP_switch=spin==2?3:spin;XC_switch=gga?4:1;Cnt_switch=contracted;ProExpn_VNA=projected;E_Field_switch=field;Matomnum=empty?0:1;
    CHECK(openmx_core_solver_run(1,callback,&atomic,&result,error,sizeof(error))&&result==1);++cases;
  }
  CHECK(cases==96&&orbital_calls&&cnt_calls&&xc_calls);printf("PASS: %d stress-grid cases, three shapes each, coordinate-grid and GGA strain oracles\n",cases);
  return 0;
}
