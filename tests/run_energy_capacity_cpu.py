#!/usr/bin/env python3
"""Compare capacity kernels with the pre-change energy quadrature under ASan.

The reference is the committed implementation; OpenACC loops execute on
the CPU. This tests remapped atoms, unequal neighbor lists, output ownership,
and packet/table transfer independently of the CUDA numerical regression.
"""
import ast
import os
from pathlib import Path
from task_worker_test_bridge import compile_task_test
import shlex
import subprocess
import tempfile

root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/"tests/run_gpu_vector_cpu.py").read_text())
prefix=next(ast.literal_eval(n.value) for n in tree.body
            if isinstance(n,ast.Assign) and any(getattr(t,"id","")=="prefix" for t in n.targets))
reference=(root/"tests/fixtures/energy_before_capacity.c").read_text()
def body(name):
    s=(root/"source"/name).read_text()
    for h in ('"openmx_common.h"','"set_cuda_default_device_from_local_rank.h"','<cuda_runtime.h>'):
        s=s.replace("#include "+h,"")
    return s
globals_=r'''
#include "openmx_core_atomic.h"
#include "openmx_core_neutral_grid.h"
#include "openmx_core_neutral_grid_metadata.h"
#define PI 3.14159265358979323846264338327950288
int mpi_comm_level1=0;
static int MPI_Abort(int comm,int code){(void)comm;exit(code);}
enum {NG=19,NS=2,MESH=16};
int My_NumGridB_AB=NG,ProExpn_VNA,E_Field_switch,Exc0_correction_flag,Ngrid2=3,Ngrid3=4;
double gtv[4][4],Grid_Origin[4],grid[11][NG];
double *Density_Grid_B[2]={grid[0],grid[1]},*ADensity_Grid_B=grid[2];
double *PCCDensity_Grid_B[2]={grid[3],grid[4]},*dVHart_Grid_B=grid[5];
double *Vxc_Grid_B[2]={grid[6],grid[7]},*RefVxc_Grid_B=grid[8],*VNA_Grid_B=grid[9],*VEF_Grid_B=grid[10];
int SpeciesNum=NS,Max_TGN_EH0=5,List_YOUSO[100],TGN_EH0[NS]={5,4},Spe_Num_Mesh_VPS[NS]={12,12};
int Spe_Num_Mesh_PAO[NS]={12,12},Matomnum=3,CoarseGL_Mesh=3,F_Vxc_flag=1;
int M2G[4]={0,2,1,3},FNAN[4]={0,2,1,2},WhatSpecies[4]={0,0,1,0};
int natn[4][3]={{0},{1,2,3},{2,1},{3,1,2}},ncn[4][3]={{0}};
double Gxyz[4][50],atv[1][4]={{0}},Spe_Atom_Cut1[NS]={1.3,1.4};
double dv_EH0[NS]={.4,.5},Spe_Core_Charge[NS]={4,5};
double GridX_EH0[NS][5],GridY_EH0[NS][5],GridZ_EH0[NS][5],Arho_EH0[NS][5],Wt_EH0[NS][5];
double Spe_VPS_XV[NS][MESH],Spe_VPS_RV[NS][MESH],Spe_VH_Atom[NS][MESH+2];
double Spe_PAO_XV[NS][MESH],Spe_PAO_RV[NS][MESH],Spe_Atomic_Den2[NS][MESH+2];
double CoarseGL_Abscissae[3]={-.71,.03,.68},CoarseGL_Weight[3]={.5,.9,.6};
'''
main=r'''
static void close_to(double a,double b){
 if(!isfinite(a) || !isfinite(b) || fabs(a-b)>2e-11*(1+fabs(b))){fprintf(stderr,"mismatch %.17g vs %.17g\n",a,b);abort();}
}
int main(void){
 double flat_grid[5][NS][5];
 OpenMXNeutralGridTables grid_tables={&flat_grid[0][0][0],{5,NS,5}};
 int borrowed_counts[NS];double borrowed_spacing[NS];
 memcpy(borrowed_counts,TGN_EH0,sizeof(borrowed_counts));memcpy(borrowed_spacing,dv_EH0,sizeof(borrowed_spacing));
 OpenMXNeutralGridMetadataView grid_metadata={borrowed_counts,borrowed_spacing,NS};
 OpenMXAtomicViews atomic={0};
 atomic.species.species=NS;atomic.species.vps_mesh=Spe_Num_Mesh_VPS;atomic.species.pao_mesh=Spe_Num_Mesh_PAO;
 atomic.species.cutoff=Spe_Atom_Cut1;atomic.species.core_charge=Spe_Core_Charge;
 atomic.vps_log_radii=(OpenMXRadialView2){&Spe_VPS_XV[0][0],{NS,MESH}};
 atomic.vps_radii=(OpenMXRadialView2){&Spe_VPS_RV[0][0],{NS,MESH}};
 atomic.hartree_potential=(OpenMXRadialView2){&Spe_VH_Atom[0][0],{NS,MESH+2}};
 atomic.pao_log_radii=(OpenMXRadialView2){&Spe_PAO_XV[0][0],{NS,MESH}};
 atomic.pao_radii=(OpenMXRadialView2){&Spe_PAO_RV[0][0],{NS,MESH}};
 atomic.density_backup=(OpenMXRadialView2){&Spe_Atomic_Den2[0][0],{NS,MESH+2}};

 List_YOUSO[21]=List_YOUSO[22]=MESH;
 for(int i=0;i<11;i++)for(int j=0;j<NG;j++)grid[i][j]=.03*(i+1)+.02*cos(.3*j+i);
 for(int i=1;i<=3;i++){Grid_Origin[i]=.2*i;for(int j=1;j<=3;j++)gtv[i][j]=i==j?.3:.01*(i+j);}
 for(int s=0;s<NS;s++){
  for(int i=0;i<MESH;i++){
   double x=-4+i*.4,r=exp(x);
   Spe_PAO_XV[s][i]=Spe_VPS_XV[s][i]=x;Spe_PAO_RV[s][i]=Spe_VPS_RV[s][i]=r;
  }
  for(int i=0;i<MESH+2;i++){Spe_VH_Atom[s][i]=1/(1+exp(-4+i*.4));Spe_Atomic_Den2[s][i]=.3*exp(-exp(-4+i*.4));}
  for(int i=0;i<5;i++){GridX_EH0[s][i]=.1+i*.04;GridY_EH0[s][i]=.2-i*.02;GridZ_EH0[s][i]=.25+i*.03;Arho_EH0[s][i]=.2;Wt_EH0[s][i]=.3;}
 }
 for(int c=0;c<5;c++)for(int sp=0;sp<NS;sp++)for(int point=0;point<5;point++)
  flat_grid[c][sp][point]=(c==0?GridX_EH0:c==1?GridY_EH0:c==2?GridZ_EH0:c==3?Arho_EH0:Wt_EH0)[sp][point];
 for(int a=1;a<4;a++){Gxyz[a][1]=a*.3;Gxyz[a][2]=a*.17;Gxyz[a][3]=a*.11;}
 double leb_data[4][4]={{1,0,0,.25},{0,1,0,.25},{0,0,1,.25},{-.577,-.577,-.577,.25}},*leb[4];
 for(int i=0;i<4;i++)leb[i]=leb_data[i];
 int cases=0;
 for(workers=1;workers<=4;workers++)for(int tier=0;tier<4;tier++){
  local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;
  for(int spin=0;spin<2;spin++)for(ProExpn_VNA=0;ProExpn_VNA<2;ProExpn_VNA++)for(E_Field_switch=0;E_Field_switch<2;E_Field_switch++)for(Exc0_correction_flag=0;Exc0_correction_flag<2;Exc0_correction_flag++){
   double ref[6]={0},got[6];
   Reference_EXC_EH1_Grid_OpenACC_Range(0,NG,0,spin,ref,ref+1,ref+2,ref+3);
   TotalEnergy_Grid_Capacity(0,spin,0,got);for(int i=0;i<6;i++)close_to(got[i],ref[i]);cases++;
  }
  double ref[6]={0},got[6];
  Reference_Dipole_Grid_OpenACC_Range(0,NG,0,7,ref,ref+1,ref+2,ref+3,ref+4,ref+5);
  TotalEnergy_Grid_Capacity(1,0,7,got);for(int i=0;i<6;i++)close_to(got[i],ref[i]);cases++;
  for(int spin=0;spin<2;spin++){
   memset(ref,0,sizeof(ref));Reference_CWF_Dc_Grid_OpenACC_Range(0,NG,0,spin,ref,ref+2);
   TotalEnergy_Grid_Capacity(2,spin,0,got);for(int i=0;i<6;i++)close_to(got[i],ref[i]);cases++;
  }
  int ban[7],wan[7],deriv[7];double dis[7],dx[7],dy[7],dz[7],rout[4][7],gout[4][7];
  for(int i=0;i<7;i++){ban[i]=i%NS;wan[i]=(i+1)%NS;deriv[i]=i%2;dis[i]=.13*i;dx[i]=.3;dy[i]=-.2;dz[i]=.4;}
  Reference_EH0_TwoCenter(0,7,ban,wan,deriv,dis,dx,dy,dz,rout[0],rout[1],rout[2],rout[3]);
  /* Poison historical globals: production must use the supplied metadata. */
  for(int s=0;s<NS;s++){TGN_EH0[s]=0;dv_EH0[s]=91;}
  TotalEnergy_EH0_TwoCenter_Batch_OpenACC(&grid_tables,&grid_metadata,&atomic,7,ban,wan,deriv,dis,dx,dy,dz,gout[0],gout[1],gout[2],gout[3]);
  memcpy(TGN_EH0,borrowed_counts,sizeof(borrowed_counts));memcpy(dv_EH0,borrowed_spacing,sizeof(borrowed_spacing));
  for(int a=0;a<4;a++)for(int i=0;i<7;i++)close_to(gout[a][i],rout[a][i]);cases++;
  double pairs[15],rsum,gsum,forces[4][3]={{0}};
  Reference_Exc0(0,3,M2G,pairs,4,leb,&rsum);
  int k=0;
  for(int mc=1;mc<=3;mc++){int gc=M2G[mc];for(int h=1;h<=FNAN[gc];h++,k++)for(int d=0;d<3;d++){
   forces[natn[gc][h]][d]+=pairs[3*k+d];forces[gc][d]-=pairs[3*k+d];
  }}
  for(int a=0;a<4;a++)for(int d=0;d<3;d++)Gxyz[a][41+d]=0;
  TotalEnergy_Exc0_Batch_OpenACC(&atomic,4,leb,&gsum);close_to(gsum,rsum);
  for(int a=0;a<4;a++)for(int d=0;d<3;d++)close_to(Gxyz[a][41+d],forces[a][d]);cases++;
 }
 printf("PASS: %d energy/dipole/EH0/Exc0 comparisons across local, remote and CPU tiers\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-energy-capacity-") as folder:
    p=Path(folder);(p/"mpi.h").write_text("typedef int MPI_Comm;\n#define MPI_COMM_NULL (-1)\n")
    (p/"cuda_runtime.h").write_text("/* Runtime mock is provided by the fixture prefix. */\n")
    (p/"openmx_gpu_device.h").write_text("/* Rank availability mock is provided by the fixture prefix. */\n")
    (p/"check.c").write_text(prefix+globals_+body("openmx_gpu_task.c")+body("Total_Energy_GPU.c")+reference+main)
    compile_task_test(shlex.split(os.environ.get("CC","cc"))+[
        "-std=c11","-O1","-g","-Wno-unknown-pragmas","-fsanitize=address,undefined",
        "-fno-sanitize-recover=all","-I"+str(p),"-I"+str(root/"source"),str(p/"check.c"),"-lm","-o",str(p/"check")],check=True)
    subprocess.run([str(p/"check")],check=True,timeout=60)
