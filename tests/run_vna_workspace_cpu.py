#!/usr/bin/env python3
"""Exercise the production VNA2/3 CPU adapter against analytic s-wave pairs.

Different species, multiplicities, Gauss meshes, both directions, polar and
on-site pairs run with 1/2/8 OpenMP threads and the real C++ workspace owner.
The closed-form j0 integral and Cartesian derivative are independent of the
angular assembly. No MPI, GPU runtime or PAO data is required.
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


def function(source, declaration):
    start = source.rindex(declaration)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


source = Path(os.environ.get("OPENMX_VNA_SOURCE", ROOT / "source/Set_ProExpn_VNA.c")).read_text()
declaration = ("static inline void Spherical_Bessel2(" if "static inline void Spherical_Bessel2(" in source
               else "inline void Spherical_Bessel2(")
bessel = function(source, declaration)
bessel = "static void Spherical_Bessel2(double x,int lmax,double *sb,double *dsb)\n" + bessel[bessel.index("{"):]
verbose = function(source, "static int SetPro_verbose(void)")
end_context = function(source, "static void SetPro_VNA23_GpuEnd(")

HARNESS = r'''
#define NS 2
#define NA 2
#define NH 9
#define NP 11
static int List_YOUSO[40], M2G[NA+1]={0,1,2};
static int WhatSpecies[NA+1]={0,0,1}, natn[NA+1][NH], ncn[NA+1][NH];
static int Spe_MaxL_Basis[NS], Spe_Num_Basis[NS][1], GL_Mesh;
static double PAO_Nkmax=2.1, Radial_kmin=0.1;
static double time_per_atom[NA+1], GL_Weight[NP], atv[NH][4], Gxyz[NA+1][4];
static double Spe_ProductRF_Bessel[NS][1][2][1][2][1][NP];
static double Spe_CrudeVNA_Bessel[NS][NP];
static dcomplex Comp2Real[1][1][1];
static double output[4][NA+1][NH][2][2];
static double *rows[4][NA+1][NH][2];
static double **neighbors[4][NA+1][NH];
static double ***atoms[4][NA+1];
static double ****channels[4];
static dcomplex Complex(double r,double i){return (dcomplex){r,i};}
static dcomplex Cadd(dcomplex a,dcomplex b){return Complex(a.r+b.r,a.i+b.i);}
static dcomplex Cmul(dcomplex a,dcomplex b){return Complex(a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r);}
static dcomplex CRmul(dcomplex a,double b){return Complex(a.r*b,a.i*b);}
static dcomplex Conjg(dcomplex a){return Complex(a.r,-a.i);}
static void dtime(double *t){static _Thread_local double tick;*t=tick++;}
static void PrintMemory(const char *name,size_t bytes,const char *mode){(void)name;(void)bytes;(void)mode;}
static void SetPro_gpu_abort(const char *message){fprintf(stderr,"%s\n",message);exit(1);}
static void xyz2spherical(double x,double y,double z,double a,double b,double c,double *s){
  (void)a;(void)b;(void)c;s[0]=sqrt(x*x+y*y+z*z);s[1]=atan2(hypot(x,y),z);s[2]=atan2(y,x);
}
static double Gaunt(int a,int b,int c,int d,int e,int f){
  assert(!(a||b||c||d||e||f));return 1.0/sqrt(4.0*acos(-1.0));
}
static void ComplexSH(int l,int m,double theta,double phi,double *v,double *dt,double *dp){
  assert(l==0&&m==0);(void)theta;(void)phi;v[0]=1.0/sqrt(4.0*acos(-1.0));v[1]=0;
  dt[0]=dt[1]=dp[0]=dp[1]=0;
}
'''

MAIN = r'''
static void context_lifetime_test(void)
{
  SetProGpu3Context first={0},second={0};
  SetProGpu3Context *contexts[]={&first,&second};
  for(int i=0;i<2;i++){
    SetProGpu3Context *g=contexts[i];g->enabled=1;g->chunk=13+i;
    g->sp_lfi=malloc(sizeof(int));g->sp_ncmb=malloc(sizeof(int));g->sp_tno=malloc(sizeof(int));
    g->cmb_L0=malloc(sizeof(int));g->cmb_num0=malloc(sizeof(int));g->c2r_off=malloc(sizeof(int));
    g->crude=malloc(sizeof(double));g->prod=malloc(sizeof(double));
    g->gnt2=malloc(sizeof(double));g->c2r=malloc(sizeof(double));
    assert(g->sp_lfi&&g->sp_ncmb&&g->sp_tno&&g->cmb_L0&&g->cmb_num0&&g->c2r_off&&
           g->crude&&g->prod&&g->gnt2&&g->c2r);
    g->c2r[0]=37+i;g->prod[0]=41+i;
  }
  double *retained=second.c2r;
  SetPro_VNA23_GpuEnd(&first);
  for(size_t i=0;i<sizeof(first);i++)assert(((unsigned char*)&first)[i]==0);
  assert(second.enabled==1&&second.chunk==14&&second.c2r==retained&&
         second.c2r[0]==38&&second.prod[0]==42);
  SetPro_VNA23_GpuEnd(&first); // Repeated cleanup must not release another invocation.
  SetPro_VNA23_GpuEnd(&second);
  SetPro_VNA23_GpuEnd(&second);
}

int main(void)
{
  double wavevectors[NP], max_error=0;
  int cases=0;
  context_lifetime_test();
  List_YOUSO[24]=2;List_YOUSO[25]=0;Comp2Real[0][0][0]=Complex(1,0);
  Gxyz[1][1]=0.2;Gxyz[1][2]=0.4;Gxyz[1][3]=0.6;
  Gxyz[2][1]=0.8;Gxyz[2][2]=0.1;Gxyz[2][3]=1.5;
  for(int h=1;h<NH;h++){
    atv[h][1]=h==1?0:0.13*h;atv[h][2]=h==1?0:-0.17*h;atv[h][3]=0.25*h;
  }
  for(int a=1;a<=NA;a++)for(int h=0;h<NH;h++){
    natn[a][h]=h<2?a:3-a;ncn[a][h]=h;
  }
  for(int d=0;d<4;d++){
    channels[d]=atoms[d];
    for(int a=0;a<=NA;a++){
      atoms[d][a]=neighbors[d][a];
      for(int h=0;h<NH;h++){
        neighbors[d][a][h]=rows[d][a][h];
        for(int i=0;i<2;i++)rows[d][a][h][i]=output[d][a][h][i];
      }
    }
  }
  int mc[NA*NH], neighbor[NA*NH];
  for(int a=1;a<=NA;a++)for(int h=0;h<NH;h++){
    mc[(a-1)*NH+h]=a;neighbor[(a-1)*NH+h]=h;
  }
  setenv("OPENMX_SETPRO_VERBOSE","1",1);assert(SetPro_verbose()==1);
  setenv("OPENMX_SETPRO_VERBOSE","0",1);assert(SetPro_verbose()==0);
  setenv("OPENMX_SETPRO_VERBOSE","1",1);assert(SetPro_verbose()==1);
  omp_set_dynamic(0);
  const int meshes[]={1,3,NP}, threads[]={1,2,8}, multiplicities[][2]={{2,1},{1,2},{0,2},{2,0},{0,0}};
  for(int mesh=0;mesh<3;mesh++)for(int split=0;split<3;split++)for(int basis=0;basis<5;basis++)for(int mirror=0;mirror<2;mirror++){
    GL_Mesh=meshes[mesh];omp_set_num_threads(threads[split]);
    for(int s=0;s<NS;s++){
      Spe_Num_Basis[s][0]=multiplicities[basis][s];
      for(int q=0;q<GL_Mesh;q++){
        wavevectors[q]=0.15+0.13*q;GL_Weight[q]=1.0/GL_Mesh;
        Spe_CrudeVNA_Bessel[s][q]=0.7+0.11*s+0.03*q;
        for(int i=0;i<2;i++)for(int j=0;j<2;j++)
          Spe_ProductRF_Bessel[s][0][i][0][j][0][q]=0.3+0.1*s+0.05*(i+j)+0.07*q;
      }
    }
    for(size_t i=0;i<sizeof(output)/sizeof(double);i++)((double*)output)[i]=-91;
    memset(time_per_atom,0,sizeof(time_per_atom));
    int *counts[NS];for(int s=0;s<NS;++s)counts[s]=Spe_Num_Basis[s];
    const OpenMXVnaPreparationShape shape={0,2,0,0,GL_Mesh,NS,Spe_MaxL_Basis,counts};
    OpenMXVnaPreparationBuffers prepared_basis;char error[256];
    OpenMXVnaPreparation *basis_owner=openmx_core_vna_preparation_create(&shape,&prepared_basis,error,sizeof(error));assert(basis_owner);
    OpenMXAtomPairBuffers pairs;OpenMXAtomPairPlan *plan=test_vna_pairs(&pairs);
    double *p6[NS][1][2][1][2][1],**p5[NS][1][2][1][2],***p4[NS][1][2][1];
    double ****p3[NS][1][2],*****p2[NS][1],******p1[NS];
    double *crude[NS];
    for(int s=0;s<NS;++s){p1[s]=p2[s];p2[s][0]=p3[s][0];crude[s]=Spe_CrudeVNA_Bessel[s];
      for(int i=0;i<2;++i){p3[s][0][i]=p4[s][0][i];p4[s][0][i][0]=p5[s][0][i][0];
        for(int j=0;j<2;++j){p5[s][0][i][0][j]=p6[s][0][i][0][j];p6[s][0][i][0][j][0]=Spe_ProductRF_Bessel[s][0][i][0][j][0];}}}
    const SetProVnaTables tables={NULL,p1,crude,&Comp2Real[0][0][0],1,GL_Weight,Radial_kmin,PAO_Nkmax,time_per_atom};
    SetPro_VNA23_CPU(channels,mirror,wavevectors,&prepared_basis,&pairs,&tables);
    openmx_core_atom_pairs_destroy(plan);openmx_core_vna_preparation_destroy(basis_owner);
    for(int a=1;a<=NA;a++){
      assert(time_per_atom[a]==NH);
      for(int h=0;h<NH;h++){
        const int other=natn[a][h], product=WhatSpecies[mirror?other:a], potential=WhatSpecies[mirror?a:other];
        double direction[3], r2=0;
        for(int d=0;d<3;d++){
          direction[d]=Gxyz[other][d+1]+atv[h][d+1]-Gxyz[a][d+1];
          if(mirror)direction[d]=-direction[d];
          r2+=direction[d]*direction[d];
        }
        const double r=sqrt(r2);
        for(int i=0;i<2;i++)for(int j=0;j<2;j++){
          const int active=i<Spe_Num_Basis[product][0]&&j<Spe_Num_Basis[product][0];
          double value=0, derivative=0;
          if(active)for(int q=0;q<GL_Mesh;q++){
            const double k=wavevectors[q],x=k*r;
            const double w=0.5*(PAO_Nkmax-Radial_kmin)*GL_Weight[q]*k*k*
                Spe_CrudeVNA_Bessel[potential][q]*Spe_ProductRF_Bessel[product][0][i][0][j][0][q];
            value+=w*(r?sin(x)/x:1.0);
            if(r)derivative+=w*k*(cos(x)/x-sin(x)/(x*x));
          }
          for(int d=0;d<4;d++){
            const double expected=!active?-91:d==0?2.0/acos(-1.0)*value:
                h==0?0:-2.0/acos(-1.0)*derivative*direction[d-1]/r;
            const double error=fabs(output[d][a][h][i][j]-expected);
            if(error>max_error)max_error=error;
            if(!(error<2e-12)){
              fprintf(stderr,"mesh=%d threads=%d basis=%d mirror=%d a=%d h=%d i=%d j=%d d=%d actual=%.17g expected=%.17g\n",
                  GL_Mesh,threads[split],basis,mirror,a,h,i,j,d,output[d][a][h][i][j],expected);return 1;
            }
          }
        }
      }
    }
    cases++;
  }
  printf("PASS: %d analytic VNA2/3 cases, 1/2/8 threads, empty/unequal species, polar/on-site derivatives; max error %.3g\n",cases,max_error);
  return 0;
}
'''

with tempfile.TemporaryDirectory(prefix="openmx-vna-workspace-") as folder:
    src, exe = Path(folder) / "check.c", Path(folder) / "check"
    src.write_text('#define _POSIX_C_SOURCE 200809L\n#include <assert.h>\n#include <math.h>\n'
                   '#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <omp.h>\n'
                   '#include "openmx_core_vna.h"\n#include "openmx_vna_gpu_context.h"\n'
                   '#define xmin 0.0\n#define asize_lmax 30\n'
                   + HARNESS + bessel + '\n' + verbose + '\n' + end_context
                   + '\n#include "openmx_vna_cpu.h"\n#include "vna_pair_fixture.h"\n' + MAIN)
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) +
                   shlex.split(os.environ.get("CFLAGS", "-O2 -g")) +
                   ["-std=c11", "-fopenmp", "-I" + str(ROOT / "source"), "-I" + str(ROOT / "tests"), str(src), str(CORE),
                    "-Wl,-rpath," + str(CORE.parent), "-lm", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
