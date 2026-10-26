#!/usr/bin/env python3
"""Exercise owned VNA preparation and production DS_VNA GPU host tabulation.

The real C++ owner, shared sampling loop and selected backend's begin/end
functions run without a GPU. Check every active packed row, basis snapshots,
projector order, independent invocations, gating and cleanup with 1/2/8 threads.
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


def function(declaration):
    start = source.rindex(declaration)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


PREFIX = r'''
#include "openmx_core_reciprocal.h"
#define NS 3
#define NL 3
#define NM 3
#define NP 17
#define SETPRO_DSVNA_CHUNK 256
typedef int MPI_Comm;
enum {MPI_COMM_NULL=0,MPI_COMM_TYPE_SHARED,MPI_INFO_NULL,MPI_INT,MPI_MIN,GPUSOLVER};
#define MPI_IN_PLACE ((void*)-1)
static int SpeciesNum=NS, GL_Mesh, mpi_comm_level1=1, scf_eigen_lib_flag=GPUSOLVER, allowed=1;
static double Spe_VNA_Bessel[NS][NL][NM][NP];
static const OpenMXReciprocalViews radial_views={.neutral_transform={&Spe_VNA_Bessel[0][0][0][0],{NS,NL,NM,NP}}};
static const OpenMXReciprocalViews *reciprocal=&radial_views;
static dcomplex Comp2Real[NL][2*NL-1][2*NL-1];
static int samples[NS][NL][NM];
static void MPI_Allreduce(const void *s,void *d,int n,int type,int op,MPI_Comm comm){
  (void)s;(void)d;(void)n;(void)type;(void)op;(void)comm;assert(s==MPI_IN_PLACE);
}
static void MPI_Comm_split_type(MPI_Comm c,int type,int key,int info,MPI_Comm *out){
  (void)c;(void)type;(void)key;(void)info;*out=2;
}
static void MPI_Comm_size(MPI_Comm c,int *n){(void)c;*n=1;}
static void MPI_Comm_rank(MPI_Comm c,int *n){(void)c;*n=0;}
static void MPI_Comm_free(MPI_Comm *c){*c=MPI_COMM_NULL;}
static int SetPro_collective_env_flag(const char *name,int value){(void)name;(void)value;return allowed;}
static int SetPro_gpu_stage_enabled(int id){(void)id;return allowed;}
static int SetPro_GpuSharing(MPI_Comm comm){(void)comm;return 1;}
static size_t SetPro_GpuBatchBudget(int n){(void)n;return SIZE_MAX;}
static size_t SetPro_HostSimdBatchBudget(void){return SIZE_MAX;}
static size_t SetPro_DSVNA_DeviceBytes(const SetProGpu2Context *g){(void)g;return 0;}
/* The production pair count reads these globals; the host SIMD vote is the
   collective helper of openmx_host_simd.c (a device vote unless forced). */
static int Matomnum=2,M2G[3]={0,1,2},FNAN[3]={0,3,4},host_vote,votes;
static int OpenMX_HostSimdStageOnHost(const char *stage,size_t need,size_t budget){
  (void)need;assert(!strcmp(stage,"Set_ProExpn_VNA: DS_VNA") && budget==SIZE_MAX);votes++;return host_vote;
}
static void SetPro_note(const char *stage,const char *fmt,...){(void)stage;(void)fmt;}
static void SetPro_DSVNA_GpuEnd(SetProGpu2Context *g);
static void *SetPro_checked_malloc(size_t n){void *p=malloc(n?n:1);assert(p);return p;}
static double Gaunt(int a,int b,int c,int d,int e,int f){return (a+3*b+5*c+7*d+11*e+13*f)*0.03125;}
static double RF_BesselF(const OpenMXReciprocalViews *reciprocal,int species,int angular,int radial,double wavevector){
  (void)reciprocal;assert(species<NS&&angular<NL&&radial<NM&&isfinite(wavevector));
#pragma omp atomic update
  samples[species][angular][radial]++;
  return 1.0+species+angular*0.0625+radial*0.00390625;
}
'''

MAIN = r'''
static void check_batch(const SetProGpu2Context *g,const OpenMXVnaPreparationBuffers *v)
{
  assert(g->enabled&&g->chunk==SETPRO_DSVNA_CHUNK&&g->num_rvna==v->projectors&&g->num_proj==v->projector_components);
  int max_rows=0,max_elements=0;
  for(int s=0;s<NS;s++){
    int radial_count=0,orbitals=0;
    for(int l=0;l<=v->maximum_angular[s];l++)radial_count+=v->basis_counts[s*v->angular_channels+l];
    if(radial_count>max_rows)max_rows=radial_count;
    assert(g->sp_ncmb[s]==radial_count);
    int combo=0;
    for(int l=0;l<=v->maximum_angular[s];l++)for(int radial=0;radial<v->basis_counts[s*v->angular_channels+l];radial++){
      assert(g->cmb_L0[s*g->cmb_max+combo]==l&&g->cmb_num0[s*g->cmb_max+combo]==orbitals);
      const size_t offset=openmx_core_vna_radial_offset(v,s,l,radial);
      for(int i=0;i<GL_Mesh;i++){
        assert(g->pro00[((size_t)s*g->cmb_max+combo)*GL_Mesh+i]==v->weighted[0][offset+i]);
        assert(g->pro01[((size_t)s*g->cmb_max+combo)*GL_Mesh+i]==v->weighted[1][offset+i]);
      }
      orbitals+=2*l+1;combo++;
    }
    assert(orbitals==v->orbitals[s]);
    if(orbitals*v->projector_components>max_elements)max_elements=orbitals*v->projector_components;
    for(;combo<g->cmb_max;combo++)assert(g->cmb_L0[s*g->cmb_max+combo]==-1);
    for(int p=0;p<v->projectors;p++)for(int i=0;i<GL_Mesh;i++)
      assert(g->vnab[((size_t)s*v->projectors+p)*GL_Mesh+i]==
             Spe_VNA_Bessel[s][v->projector_angular[p]][v->projector_radial[p]][i]);
  }
  assert(g->cmb_max==max_rows&&g->mn_max==max_elements);
  int offset=0;
  for(int p=0;p<v->projectors;p++){
    assert(g->vna_L1[p]==v->projector_angular[p]&&g->vna_num1[p]==offset);
    offset+=2*v->projector_angular[p]+1;
  }
  assert(offset==v->projector_components);
  const int maxl=g->l0max_all>g->l1max?g->l0max_all:g->l1max;
  offset=0;
  for(int l=0;l<=maxl;l++){
    assert(g->c2r_off[l]==offset);
    for(int a=0;a<2*l+1;a++)for(int b=0;b<2*l+1;b++){
      const int p=offset+a*(2*l+1)+b;
      assert(g->c2r[2*p]==Comp2Real[l][a][b].r&&g->c2r[2*p+1]==Comp2Real[l][a][b].i);
    }
    offset+=(2*l+1)*(2*l+1);
  }
}

int main(void)
{
  const int meshes[]={1,3,NP},threads[]={1,2,8},multiplicities[]={0,1,3};
  double wavevectors[NP],weights[NP];int cases=0;
  for(int s=0;s<NS;s++)for(int l=0;l<NL;l++)for(int r=0;r<NM;r++)for(int i=0;i<NP;i++)
    Spe_VNA_Bessel[s][l][r][i]=s+l*0.125+r*0.03125+i*0.001953125;
  for(int l=0;l<NL;l++)for(int a=0;a<2*l+1;a++)for(int b=0;b<2*l+1;b++)
    Comp2Real[l][a][b]=(dcomplex){0.125*(l+a+b),0.0625*(l-a+b)};
  omp_set_dynamic(0);
  for(int mesh=0;mesh<3;mesh++)for(int split=0;split<3;split++)for(int pm=0;pm<3;pm++)for(int empty=0;empty<2;empty++){
    int angular[]={2,-1,1},first[]={2,0,1},last[]={0,2};int *rows[]={first,NULL,last};
    if(empty){angular[0]=-1;angular[2]=-1;rows[0]=rows[2]=NULL;}
    GL_Mesh=meshes[mesh];omp_set_num_threads(threads[split]);
    const OpenMXVnaPreparationShape shape={2,2,2,multiplicities[pm],GL_Mesh,NS,angular,rows};
    OpenMXVnaPreparationBuffers v;char error[256];
    OpenMXVnaPreparation *owner=openmx_core_vna_preparation_create(&shape,&v,error,sizeof(error));
    assert(owner!=NULL);
    /* The caller can change or release input metadata before sampling. */
    angular[0]=-1;first[0]=0;last[1]=0;
    for(int i=0;i<GL_Mesh;i++){wavevectors[i]=i*0.125;weights[i]=(1+i)*0.03125;}
    memset(samples,0,sizeof(samples));
    SetPro_PrepareRadials(reciprocal,&v,wavevectors,weights,1.75);
    for(int s=0;s<NS;s++)for(int l=0;l<NL;l++)for(int r=0;r<NM;r++){
      const int active=r<v.basis_counts[s*v.angular_channels+l];
      assert(samples[s][l][r]==(active?GL_Mesh:0));
      if(active){
        const size_t offset=openmx_core_vna_radial_offset(&v,s,l,r);
        for(int i=0;i<GL_Mesh;i++){
          const double k=wavevectors[i],radial=1.0+s+l*0.0625+r*0.00390625;
          const double expected=radial*(0.5*1.75*weights[i]*k*k);
          assert(v.weighted[0][offset+i]==expected&&v.weighted[1][offset+i]==expected*k);
        }
      }
    }
    SetProGpu2Context a={0},b={0};
    assert(SetPro_LocalPairs()==9);
    /* A host vote returns without a batch and releases every table. */
    host_vote=1;votes=0;assert(!SetPro_DSVNA_GpuBegin(reciprocal,&a,&v)&&votes==1);
    for(size_t i=0;i<sizeof(a);i++)assert(((unsigned char*)&a)[i]==0);
    host_vote=0;votes=0;
    assert(SetPro_DSVNA_GpuBegin(reciprocal,&a,&v)&&votes==1);check_batch(&a,&v);
    SpeciesNum=1; // The prepared species count is owned by the snapshot.
    assert(SetPro_DSVNA_GpuBegin(reciprocal,&b,&v));check_batch(&b,&v);
    SpeciesNum=NS;
    double *retained=a.pro00;
    SetPro_DSVNA_GpuEnd(&b);SetPro_DSVNA_GpuEnd(&b);
    assert(a.enabled&&a.pro00==retained);check_batch(&a,&v);
    SetPro_DSVNA_GpuEnd(&a);SetPro_DSVNA_GpuEnd(&a);
    for(size_t i=0;i<sizeof(a);i++)assert(((unsigned char*)&a)[i]==0);
    allowed=0;assert(!openmx_env_set("OPENMX_SETPRO_GPU","0"));assert(!SetPro_DSVNA_GpuBegin(reciprocal,&a,&v)&&!a.enabled);SetPro_DSVNA_GpuEnd(&a);allowed=1;assert(!openmx_env_unset("OPENMX_SETPRO_GPU"));
    scf_eigen_lib_flag=0;assert(!SetPro_DSVNA_GpuBegin(reciprocal,&a,&v)&&!a.enabled);scf_eigen_lib_flag=GPUSOLVER;
    openmx_core_vna_preparation_destroy(owner);cases++;
  }
  puts("PASS: 54 VNA preparation cases, exact sampling/packing, metadata snapshots, 1/2/8 threads, empty groups, independent DS_VNA batches and cleanup");
  assert(cases==54);return 0;
}
'''

with tempfile.TemporaryDirectory(prefix="openmx-vna-preparation-") as folder:
    src, exe = Path(folder) / "check.c", Path(folder) / "check"
    src.write_text('#include <assert.h>\n#include <math.h>\n#include <stdint.h>\n#include <stdio.h>\n'
                   '#include <stdlib.h>\n#include <string.h>\n#include <omp.h>\n#include "openmx_complex.h"\n'
                   '#include "openmx_core_vna_preparation.h"\n#include "openmx_vna_gpu_context.h"\n'
                   + PREFIX + function("static int SetPro_LocalPairs(") + '\n'
                   + function("static int SetPro_DSVNA_GpuBegin(") + '\n'
                   + function("static void SetPro_DSVNA_GpuEnd(")
                   + '\n#include "openmx_vna_preparation.h"\n' + MAIN)
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + shlex.split(os.environ.get("CFLAGS", "-O2 -g")) +
                   ["-std=c11", "-D_POSIX_C_SOURCE=200809L", "-fopenmp", "-I" + str(ROOT / "source"), "-include", str(ROOT / "source/openmx_env.h"), str(ROOT / "source/openmx_env.c"), str(src), str(CORE),
                    "-Wl,-rpath," + str(CORE.parent), "-lm", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
