#!/usr/bin/env python3
"""Exercise the production VNA archive retry and contraction without a GPU.

Check numerical results, one upload per archive, admission, late allocation /
copy failures and cleanup under ASan/UBSan. MPI is local here; the full silicon
case separately checks cooperation between ranks on an actual GPU.
OPENMX_TEST_GPU=1 with CC=nvc also runs the indexed kernel on the GPU using
managed test allocations; the default needs only a host C compiler.
"""
import os
from pathlib import Path
import resource
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "source/Set_ProExpn_VNA.c").read_text()


def function(declaration):
    start = source.index(declaration)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


prefix = r'''
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_gpu_task.h"
#include "openmx_vna_gpu_context.h"
#include "openmx_core_host_simd.h"
enum {cudaSuccess, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost};
enum {MPI_INT, MPI_MAX};
static int Matomnum=2,SpeciesNum=2,mpi_comm_level1=0,fp32;
static int M2G[]={0,1,2},F_G2M[]={0,1,2,3},WhatSpecies[]={0,0,1,1};
static int Spe_Total_NO[]={2,3},FNAN[]={0,2,1,2};
static int natn[4][3]={{0},{1,2,3},{2,3},{3,1,2}};
static int RMI1[3][3][3]={{{0}},{{0,1,1},{1,-1,1},{1,2,0}},{{0,1},{2,0}}};
static double Spe_Atom_Cut1[]={2.0,3.0},Dis[3][3]={{0},{0,1,2},{0,1.5}};
static int held,usable=1,deny,inject,fail_alloc,fail_copy,allocations,copies,live;
static int uploads,collectives,capacity_calls;
static size_t available=SIZE_MAX;
static const void *archive_a,*archive_b;
static void *tracked[8];
static int openmx_fp32_enabled(void){return fp32;}
/* OPENMX_HOST_SIMD policy: legacy keeps the capacity recovery loop this
   harness checks; any host SIMD mode returns the device verdict instead. */
static int host_simd=OPENMX_HOST_SIMD_LEGACY;
static int OpenMX_HostSimdMode(void){return host_simd;}
static int gpu_rank_device_usable(void){return usable;}
static int openmx_gpu_capacity_test_recovery(void){return inject;}
static int openmx_gpu_local_size_noncollective(void){return 1;}
static size_t openmx_host_mem_available(void){return 0;}
static void SetPro_gpu_abort(const char *s){fprintf(stderr,"%s\n",s);abort();}
static void *SetPro_checked_malloc(size_t n){void *p=malloc(n?n:1);assert(p);return p;}
static void SetPro_note(const char *stage,const char *fmt,...){(void)stage;(void)fmt;}
static void dtime(double *t){*t=0;}
static double dampingF(double cutoff,double distance){return 1-distance/(2*cutoff);}
static int cudaGetDevice(int *d){*d=0;return 0;}
static int cudaDeviceSynchronize(void){assert(held);return 0;}
static int cudaGetLastError(void){return 0;}
static void acc_wait_all(void){assert(held);}
static void acc_clear_freelists(void){assert(held);}
static int OpenMX_GpuMemGetInfo(size_t *f,size_t *t){assert(held);*f=*t=available;return 0;}
static void *openmx_mgpu_admit_devices(const int *d,int n){
 assert(!held && n==1 && d[0]==0);if(deny)return NULL;held=1;return &held;
}
static void openmx_mgpu_release_devices(void *p){assert(p==&held && held && !live);held=0;}
static int cudaMalloc(void **p,size_t n){
 assert(held);*p=NULL;if(++allocations==fail_alloc)return 1;
 *p=malloc(n);assert(*p);for(int i=0;i<8;i++)if(!tracked[i]){tracked[i]=*p;live++;return 0;}
 abort();
}
static int cudaFree(void *p){
 assert(held);for(int i=0;i<8;i++)if(tracked[i]==p){free(p);tracked[i]=NULL;live--;return 0;}
 abort();
}
static int cudaMemcpy(void *d,const void *s,size_t n,int kind){
 assert(held);if(++copies==fail_copy)return 1;
 if(kind==cudaMemcpyHostToDevice && (s==archive_a || s==archive_b))uploads++;
 memcpy(d,s,n);return 0;
}
static void MPI_Allreduce(const void *s,void *d,int n,int type,int op,int comm){
 (void)type;(void)op;(void)comm;assert(!held);memcpy(d,s,n*sizeof(int));collectives++;
}
/* A CPU task executor for the fallback only. Feed the production contraction
   normalized CSR inputs, as the common task allocator does. */
int openmx_gpu_task_execute(MPI_Comm comm,OpenMXGpuTask *t,int enabled,const char *label){
 (void)comm;(void)enabled;(void)label;assert(!held);capacity_calls++;
 void *v[7];
 for(int i=0;i<7;i++){
  const OpenMXGpuTaskView *w=&t->view[i];v[i]=w->data;
  if(w->offsets){
   size_t index=((t->count+1)*sizeof(size_t)+15)&~(size_t)15;
   size_t bytes=w->offsets[t->count]*w->element_bytes;
   v[i]=malloc(index+bytes);assert(v[i]);memcpy(v[i],w->offsets,(t->count+1)*sizeof(size_t));
   memcpy((char*)v[i]+index,w->data,bytes);
  }
 }
 t->kernel(0,t->count,t->parameters,v,0);
 for(int i=0;i<7;i++)if(t->view[i].offsets)free(v[i]);return 0;
}
'''

main = r'''
static void reset(void){
 assert(!held && !live);deny=inject=fail_alloc=fail_copy=0;usable=1;available=SIZE_MAX;
 allocations=copies=uploads=collectives=capacity_calls=0;
}
static void faults(int scenario,size_t device_bytes){
 reset();
 if(scenario==1)available=device_bytes+256*1024*1024-1;
 if(scenario==2)deny=1;
 if(scenario==3)usable=0;
 if(scenario==4)inject=1;
 if(scenario>=5 && scenario<=7)fail_alloc=scenario-4;
 if(scenario>=8)fail_copy=scenario-7;
}
int main(void){
 enum {NR=24,NP=96};
 int base[]={0,0,3},hbase[]={0,0},length[NR];
 size_t off[]={0,2*NP,4*NP,6*NP,9*NP},hoff[]={0,3*NP,6*NP};
 float local[12*NP],halo[9*NP];double energy[2*NR];
 for(int i=0;i<12*NP;i++)local[i]=(float)sin(.071*i);
 for(int i=0;i<9*NP;i++)halo[i]=(float)cos(.039*i);
 for(int r=0;r<NR;r++)length[r]=2*(r%4)+1;
 for(int r=0;r<2*NR;r++)energy[r]=.013*(r-9);
 double values[3][3][3][3],expected[3][3][3][3];
 double *rows[3][3][3],**pairs[3][3],***atoms[3];
 for(int a=0;a<3;a++){
  atoms[a]=pairs[a];
  for(int j=0;j<3;j++){pairs[a][j]=rows[a][j];for(int m=0;m<3;m++)rows[a][j][m]=values[a][j][m];}
 }
 SetProGpuContext g={0};g.enabled=g.host_archive=1;g.num_proj=NP;g.num_rvna=NR;
 g.flat_host=local;g.halo_host=halo;g.flat_count=12*NP;g.halo_count=9*NP;
 g.mck_base=base;g.mck_off=off;g.halo_base=hbase;g.halo_off=hoff;g.ene=energy;g.l2p1=length;
 g.device_bytes=sizeof(local)+sizeof(halo)+64*1024*1024;
 archive_a=local;archive_b=halo;
 int cases=0;
 for(fp32=0;fp32<=1;fp32++){
  /* Independent scalar reference over atom, neighbour and projector channels. */
  for(int a=0;a<3;a++)for(int j=0;j<3;j++)for(int m=0;m<3;m++)for(int n=0;n<3;n++)expected[a][j][m][n]=-999;
  for(int a=1;a<=2;a++)for(int j=0;j<=FNAN[a];j++){
   int b=natn[a][j],na=Spe_Total_NO[WhatSpecies[a]],nb=Spe_Total_NO[WhatSpecies[b]];
   for(int m=0;m<na;m++)for(int n=0;n<nb;n++){
    double total=0;
    for(int k=0;k<=FNAN[a];k++){
     int kl=RMI1[a][j][k];if(kl<0)continue;
     const float *x=local+off[base[a]+k]+m*NP;
     const float *y=b<=2?local+off[base[b]+kl]+n*NP:halo+hoff[kl]+n*NP;
     double sum=0;int l=0;
     for(int r=0;r<NR;r++){
      double term=0;for(int z=0;z<length[r];z++,l++)term+=fp32?(double)(x[l]*y[l]):(double)x[l]*(double)y[l];
      sum+=energy[WhatSpecies[natn[a][k]]*NR+r]*term;
     }
     total+=sum;
    }
    expected[a][j][m][n]=dampingF(Spe_Atom_Cut1[WhatSpecies[a]]+Spe_Atom_Cut1[WhatSpecies[b]],Dis[a][j])*total;
   }
  }
  for(int scenario=0;scenario<23;scenario++){
   memcpy(values,expected,sizeof(values));
   for(int a=1;a<=2;a++)for(int j=0;j<=FNAN[a];j++)for(int m=0;m<Spe_Total_NO[WhatSpecies[a]];m++)for(int n=0;n<Spe_Total_NO[WhatSpecies[natn[a][j]]];n++)values[a][j][m][n]=-999;
   double initial[3][3][3][3];memcpy(initial,values,sizeof(values));
   faults(scenario,g.device_bytes);
   int committed=SetPro_HVNA_GpuRun_Archived(&g,atoms);
   assert(committed==(scenario==0 || scenario==22));
   assert(!held && !live);
   if(!committed)assert(!memcmp(values,initial,sizeof(values)));
   memcpy(values,initial,sizeof(values));
   faults(scenario,g.device_bytes);
   SetPro_HVNA_GpuRun(&g,atoms);
   assert(!held && !live && collectives>=1);
   assert(g.flat_dev==NULL && g.halo_dev==NULL);
   assert(g.flat_host==local && g.halo_host==halo);
   if(scenario==0){assert(uploads==2 && allocations==3 && capacity_calls==0);}
   else if(scenario<22)assert(capacity_calls>0);
   else assert(uploads==2 && capacity_calls==0); /* failure index past the last copy */
   for(int a=0;a<3;a++)for(int j=0;j<3;j++)for(int m=0;m<3;m++)for(int n=0;n<3;n++)
    assert(fabs(values[a][j][m][n]-expected[a][j][m][n])<1e-12);
   /* Host SIMD policy: the same device attempt, no capacity recovery; a
      failure is reported to the dispatcher with the outputs untouched. */
   memcpy(values,initial,sizeof(values));
   faults(scenario,g.device_bytes);host_simd=OPENMX_GRID_SCALAR;
   const int done=SetPro_HVNA_GpuRun(&g,atoms);host_simd=OPENMX_HOST_SIMD_LEGACY;
   assert(done==(scenario==0 || scenario==22));
   assert(!held && !live && capacity_calls==0 && collectives==0);
   assert(g.flat_dev==NULL && g.halo_dev==NULL && g.flat_host==local && g.halo_host==halo);
   if(done){for(int a=0;a<3;a++)for(int j=0;j<3;j++)for(int m=0;m<3;m++)for(int n=0;n<3;n++)
    assert(fabs(values[a][j][m][n]-expected[a][j][m][n])<1e-12);}
   else assert(!memcmp(values,initial,sizeof(values)));
   cases++;
  }
 }
 reset();Matomnum=0;assert(SetPro_HVNA_GpuRun_Archived(&g,atoms));assert(!allocations);
 reset();Matomnum=1;FNAN[1]=0;g.halo_count=0;g.halo_host=NULL;
 assert(SetPro_HVNA_GpuRun_Archived(&g,atoms));assert(uploads==1 && !held && !live);
 printf("PASS: %d archive/contraction cases, empty rank, empty halo, cleanup and fallback\n",cases);
}
'''

functions = ""
functions += function("static int SetPro_HVNA_GpuRun_Part(") + "\n"
start = source.index("typedef struct {int num_proj,num_rvna,fp32;}")
end = source.index("/* ------------------------------------------------------------------ */", start)
functions += source[start:end]
with tempfile.TemporaryDirectory(prefix="openmx-vna-archive-") as directory:
    p = Path(directory)
    (p / "mpi.h").write_text("typedef int MPI_Comm;\n")
    (p / "check.c").write_text(prefix + functions + main)
    flags = (["-acc", "-gpu=ccnative,mem:managed,nofma", "-Mnofma"]
             if os.environ.get("OPENMX_TEST_GPU") == "1" else
             ["-ffp-contract=off", "-Wno-unknown-pragmas",
              "-fsanitize=address,undefined", "-fno-sanitize-recover=all"])
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
        "-std=c11", "-O1", "-g", *flags,
        "-I" + str(p), "-I" + str(ROOT / "source"), str(p / "check.c"),
        "-lm", "-o", str(p / "check")], check=True)
    # With an unlimited stack, Linux may place libraries inside ASan's shadow
    # range. Bound only the child test's stack; keep the caller's shell intact.
    def sanitizer_stack():
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))

    subprocess.run([str(p / "check")], check=True, timeout=45,
                   preexec_fn=sanitizer_stack)
