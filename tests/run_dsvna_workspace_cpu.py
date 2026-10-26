#!/usr/bin/env python3
"""Compare shared DS_VNA CPU assembly with the frozen pre-RAII implementation.

Exercise ragged s/p/d bases, asymmetric/empty projector spaces, zero wavevector,
polar/on-site pairs, 1/2/8 threads and both DS_VNA storage precisions. An s-wave
closed form additionally checks the integral and Cartesian derivatives.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CORE = Path(os.environ.get("OPENMX_CORE_LIBRARY", ROOT / "build/core-serial/libopenmx_core.so")).resolve()
if not CORE.is_file():
    raise SystemExit("Build the C++ core and set OPENMX_CORE_LIBRARY")
source = Path(os.environ.get("OPENMX_VNA_SOURCE", ROOT / "source/Set_ProExpn_VNA.c")).read_text()
declaration = "static inline void Spherical_Bessel2(" if "static inline void Spherical_Bessel2(" in source else "inline void Spherical_Bessel2("
a = source.rindex(declaration)
b = source.index("{", a)
end, depth = b + 1, 1
while depth:
    depth += (source[end] == "{") - (source[end] == "}")
    end += 1
bessel = "static void Spherical_Bessel2(double x,int lmax,double *sb,double *dsb)\n" + source[b:end]
reference = (ROOT / "tests/fixtures/dsvna_cpu_before_raii.c").read_text()

HARNESS = r'''
#define NS 2
#define NA 2
#define NH 9
#define NP 11
#define NO 18
#define NC 18
#define xmin 0.0
#define asize_lmax 30
typedef OPENMX_TEST_DS_TYPE Type_DS_VNA;
#include "openmx_host_simd.h"
/* The adapter's host SIMD policy (production: openmx_host_simd.c). The legacy
   loops are the path this harness was written for; main repeats every case
   with the widest host SIMD ISA of the linked Core. */
static int harness_host_simd=OPENMX_HOST_SIMD_LEGACY;
int OpenMX_HostSimdMode(void){return harness_host_simd;}
static int List_YOUSO[40],M2G[NA+1]={0,1,2},WhatSpecies[NA+1]={0,0,1};
static int natn[NA+1][NH],ncn[NA+1][NH],Spe_MaxL_Basis[NS],Spe_Num_Basis[NS][3],GL_Mesh,measure_time;
static double PAO_Nkmax=2.0,Radial_kmin=0,Spe_Atom_Cut1[NS]={4,4};
static double time_per_atom[NA+1],atv[NH][4],Gxyz[NA+1][4];
static double Spe_VNA_Bessel[NS][3][2][NP];
static dcomplex Comp2Real[3][5][5];
static Type_DS_VNA output[4][NA+1][NH][NO][NC],expected[4][NA+1][NH][NO][NC];
static Type_DS_VNA *rows[4][NA+1][NH][NO];
static Type_DS_VNA **neighbors[4][NA+1][NH];
static Type_DS_VNA ***atoms[4][NA+1];
static Type_DS_VNA ****channels[4];
static dcomplex Complex(double r,double i){return (dcomplex){r,i};}
static dcomplex Cadd(dcomplex a,dcomplex b){return Complex(a.r+b.r,a.i+b.i);}
static dcomplex Cmul(dcomplex a,dcomplex b){return Complex(a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r);}
static dcomplex Conjg(dcomplex a){return Complex(a.r,-a.i);}
static dcomplex Im_pow(int sign,int power){
  const int i=((sign*power)%4+4)%4;
  return i==0?Complex(1,0):i==1?Complex(0,1):i==2?Complex(-1,0):Complex(0,-1);
}
static void dtime(double *t){static _Thread_local double tick;*t=tick++;}
static void PrintMemory(const char *name,size_t bytes,const char *mode){(void)name;(void)bytes;(void)mode;}
static void SetPro_gpu_abort(const char *message){fprintf(stderr,"%s\n",message);abort();}
static void xyz2spherical(double x,double y,double z,double a,double b,double c,double *s){
  (void)a;(void)b;(void)c;s[0]=sqrt(x*x+y*y+z*z);s[1]=atan2(hypot(x,y),z);s[2]=atan2(y,x);
}
static double Gaunt(int a,int b,int c,int d,int e,int f){
  if(!(a||b||c||d||e||f))return 1.0/sqrt(4.0*acos(-1.0));
  return (1+a+2*b+3*c+5*d+7*e+11*f)*0.015625;
}
static void ComplexSH(int l,int m,double theta,double phi,double *v,double *dt,double *dp){
  if(l==0){assert(m==0);v[0]=1.0/sqrt(4.0*acos(-1.0));v[1]=dt[0]=dt[1]=dp[0]=dp[1]=0;return;}
  const double a=1+0.125*l+0.03125*m;
  v[0]=a*cos(m*phi)*cos(theta);v[1]=a*sin(m*phi)*cos(theta);
  dt[0]=-a*cos(m*phi)*sin(theta);dt[1]=-a*sin(m*phi)*sin(theta);
  dp[0]=-m*v[1];dp[1]=m*v[0];
}
'''

REFERENCE_HEAD = r'''
static void reference_dsvna(Type_DS_VNA *****DS_VNA,const OpenMXVnaPreparationBuffers *view,
    const double *GL_NormK,int OneD_Nloop,const int *OneD2Mc_AN,const int *OneD2h_AN)
{
  OpenMXVnaPreparationBuffers preparation=*view;
  const int Num_RVNA=view->projectors;
  const int *VNA_List=view->projector_angular;
  const double *Bessel_Pro00=view->weighted[0],*Bessel_Pro01=view->weighted[1];
  double time1=0,time2=0,time3=0;
'''

MAIN = r'''
static void reset_output(void){
  for(size_t i=0;i<sizeof(output)/sizeof(Type_DS_VNA);i++)((Type_DS_VNA*)output)[i]=-91;
  memset(time_per_atom,0,sizeof(time_per_atom));
}
int main(void)
{
  size_t atom_offsets[(NA+1)+1],block_offsets[(NA+1)*NH+1];
  int shapes[2*(NA+1)*NH];
  const OpenMXProjectorMatrix matrix={test_rectangular_matrix(NA+1,4,NH,NO,NC,atom_offsets,block_offsets,shapes),&output[0][0][0][0][0]};
  double wavevectors[NP],maximum_error=0;int cases=0;
  /* The highest derivative reads sb[lmax+1] even for exactly zero k. */
  for(int l=0;l<=28;l++){
    double sb[31],dsb[31];for(int i=0;i<31;i++)sb[i]=dsb[i]=NAN;
    Spherical_Bessel2(0,l,sb,dsb);assert(sb[0]==1);
    for(int i=1;i<=l+1;i++)assert(sb[i]==0);
    for(int i=0;i<=l;i++)assert(dsb[i]==(i==1?1.0/3.0:0.0));
  }
  List_YOUSO[24]=2;List_YOUSO[25]=2;List_YOUSO[30]=2;
  Gxyz[1][1]=0.2;Gxyz[1][2]=0.4;Gxyz[1][3]=0.6;
  Gxyz[2][1]=0.8;Gxyz[2][2]=0.1;Gxyz[2][3]=1.5;
  for(int h=1;h<NH;h++){
    atv[h][1]=h==1?0:0.13*h;atv[h][2]=h==1?0:-0.17*h;atv[h][3]=0.25*h;
  }
  for(int a=1;a<=NA;a++)for(int h=0;h<NH;h++){natn[a][h]=h<2?a:3-a;ncn[a][h]=h;}
  for(int l=0;l<3;l++)for(int m=0;m<2*l+1;m++)for(int n=0;n<2*l+1;n++)
    Comp2Real[l][m][n]=l?Complex(0.125*(l+1)+(m==n),0.03125*(m-n)):Complex(1,0);
  for(int s=0;s<NS;s++)for(int l=0;l<3;l++)for(int r=0;r<2;r++)for(int i=0;i<NP;i++)
    Spe_VNA_Bessel[s][l][r][i]=0.7+0.11*s+0.03*l+0.025*r+0.005*i;
  for(int d=0;d<4;d++){
    channels[d]=atoms[d];
    for(int a=0;a<=NA;a++){
      atoms[d][a]=neighbors[d][a];
      for(int h=0;h<NH;h++){
        neighbors[d][a][h]=rows[d][a][h];
        for(int i=0;i<NO;i++)rows[d][a][h][i]=output[d][a][h][i];
      }
    }
  }
  int mc[NA*NH],neighbor[NA*NH];
  for(int a=1;a<=NA;a++)for(int h=0;h<NH;h++){mc[(a-1)*NH+h]=a;neighbor[(a-1)*NH+h]=h;}
  const int meshes[]={1,3,NP},threads[]={1,2,8};
  const int basis_cases[4][NS][3]={{{2,0,0},{1,0,0}},{{2,0,1},{0,2,0}},{{0,0,0},{1,1,1}},{{0,0,0},{0,0,0}}};
  const int projector_cases[][2]={{0,1},{2,2},{0,0}};
  omp_set_dynamic(0);
  const int policies[]={OPENMX_HOST_SIMD_LEGACY,openmx_core_host_simd_mode(NULL)};
  for(int policy=0;policy<2;policy++)
  for(int mesh=0;mesh<3;mesh++)for(int split=0;split<3;split++)for(int basis=0;basis<4;basis++)for(int projector=0;projector<3;projector++){
    harness_host_simd=policies[policy];
    GL_Mesh=meshes[mesh];omp_set_num_threads(threads[split]);
    for(int s=0;s<NS;s++){Spe_MaxL_Basis[s]=basis?2:0;for(int l=0;l<3;l++)Spe_Num_Basis[s][l]=basis_cases[basis][s][l];}
    List_YOUSO[34]=projector_cases[projector][1];List_YOUSO[35]=projector_cases[projector][0];
    int *counts[]={Spe_Num_Basis[0],Spe_Num_Basis[1]};
    const OpenMXVnaPreparationShape shape={2,2,List_YOUSO[35],List_YOUSO[34],GL_Mesh,NS,Spe_MaxL_Basis,counts};
    OpenMXVnaPreparationBuffers view;char error[256];
    OpenMXVnaPreparation *owner=openmx_core_vna_preparation_create(&shape,&view,error,sizeof(error));assert(owner);
    for(int i=0;i<GL_Mesh;i++)wavevectors[i]=i*0.13;
    for(int s=0;s<NS;s++)for(int l=0;l<=Spe_MaxL_Basis[s];l++)for(int r=0;r<Spe_Num_Basis[s][l];r++)for(int i=0;i<GL_Mesh;i++){
      const double k=wavevectors[i];const size_t offset=openmx_core_vna_radial_offset(&view,s,l,r)+i;
      view.weighted[0][offset]=(0.3+0.1*s+0.05*l+0.025*r+0.01*i)*k*k/GL_Mesh;
      view.weighted[1][offset]=view.weighted[0][offset]*k;
    }
    reset_output();double timing[3]={0};
    /* Production assembly must use its owned metadata rather than these globals. */
    int saved[40];memcpy(saved,List_YOUSO,sizeof(saved));memset(List_YOUSO,0,sizeof(List_YOUSO));
    memset(Spe_Num_Basis,0,sizeof(Spe_Num_Basis));
    OpenMXAtomPairBuffers pairs;OpenMXAtomPairPlan *plan=test_vna_pairs(&pairs);
    const SetProVnaTables tables={{&Spe_VNA_Bessel[0][0][0][0],{NS,3,2,NP}},{0},{0},
        &Comp2Real[0][0][0],5,NULL,Radial_kmin,PAO_Nkmax,time_per_atom};
    SetPro_DSVNA_CPU(&matrix,owner,&view,wavevectors,&pairs,&tables,timing,timing+1,timing+2);
    openmx_core_atom_pairs_destroy(plan);
    memcpy(List_YOUSO,saved,sizeof(saved));memcpy(Spe_Num_Basis,basis_cases[basis],sizeof(Spe_Num_Basis));
    for(int a=1;a<=NA;a++)assert(time_per_atom[a]==NH);
    if(basis==0 && projector==0)for(int a=1;a<=NA;a++)for(int h=0;h<NH;h++)for(int radial=0;radial<Spe_Num_Basis[WhatSpecies[a]][0];radial++){
      const int other=natn[a][h],species=WhatSpecies[a],potential=WhatSpecies[other];
      double direction[3],r2=0;for(int d=0;d<3;d++){direction[d]=Gxyz[other][d+1]+atv[h][d+1]-Gxyz[a][d+1];r2+=direction[d]*direction[d];}
      const double distance=sqrt(r2);double value=0,derivative=0;
      for(int i=0;i<GL_Mesh;i++){
        const double k=wavevectors[i],x=k*distance;
        const double factor=view.weighted[0][openmx_core_vna_radial_offset(&view,species,0,radial)+i]*Spe_VNA_Bessel[potential][0][0][i];
        value+=factor*(x?sin(x)/x:1.0);
        if(x)derivative+=factor*k*(cos(x)/x-sin(x)/(x*x));
      }
      for(int d=0;d<4;d++){
        const double want=d==0?8.0*(Type_DS_VNA)(value/(4*acos(-1.0))):h==0?0:
            -8.0*(Type_DS_VNA)(derivative/(4*acos(-1.0))*direction[d-1]/distance);
        const double error=fabs(output[d][a][h][radial][0]-want);
        const double tolerance=sizeof(Type_DS_VNA)==sizeof(float)?1e-7:2e-13;
        if(error>maximum_error)maximum_error=error;
        assert(error<=tolerance*(1+fabs(want)));
      }
    }
    memcpy(expected,output,sizeof(output));reset_output();
    /* Keep the frozen reference serial: its timing update has the legacy race.
       Every production thread partition above is compared with this oracle. */
    omp_set_num_threads(1);
    reference_dsvna(channels,&view,wavevectors,NA*NH,mc,neighbor);
    for(size_t i=0;i<sizeof(output)/sizeof(Type_DS_VNA);i++){
      const double a=((Type_DS_VNA*)output)[i],b=((Type_DS_VNA*)expected)[i];
      if(!isfinite(a)||!isfinite(b)||a!=b){
        fprintf(stderr,"DS_VNA mismatch mesh=%d threads=%d basis=%d projector=%d index=%zu reference=%.17g actual=%.17g\n",GL_Mesh,threads[split],basis,projector,i,a,b);return 1;
      }
    }
    openmx_core_vna_preparation_destroy(owner);cases++;
  }
  assert(cases==2*108);
  printf("PASS: %d DS_VNA cases (%zu-byte storage; legacy and %s host policies), exact legacy comparison, analytic max error %.3g; ragged/empty bases, independent metadata, zero-k Bessel, 1/2/8 threads\n",
      cases,sizeof(Type_DS_VNA),openmx_core_host_simd_name(policies[1]),maximum_error);
}
'''

with tempfile.TemporaryDirectory(prefix="openmx-dsvna-workspace-") as folder:
    src, exe = Path(folder) / "check.c", Path(folder) / "check"
    src.write_text('#include <assert.h>\n#include <math.h>\n#include <stdio.h>\n#include <stdlib.h>\n'
                   '#include <string.h>\n#include <omp.h>\n#include "openmx_core_dsvna.h"\n'
                   + HARNESS + bessel + '\n#include "openmx_dsvna_cpu.h"\n'
                   + REFERENCE_HEAD + reference + '\n}\n' + '\n#include \"vna_pair_fixture.h\"\n' + '\n#include "block_matrix_fixture.h"\n' + MAIN)
    for precision in ("float", "double"):
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) + shlex.split(os.environ.get("CFLAGS", "-O2 -g")) +
                       ["-std=c11", "-fopenmp", "-DOPENMX_TEST_DS_TYPE=" + precision,
                        "-I" + str(ROOT / "source"), "-I" + str(ROOT / "tests"), str(src), str(CORE),
                        "-Wl,-rpath," + str(CORE.parent), "-lm", "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
