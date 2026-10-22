#!/usr/bin/env python3
"""Check Force4B archive admission, uploads and failure cleanup with sanitizers."""
import os
from pathlib import Path
import resource
import shlex
import subprocess
import tempfile

from run_force4b_stream_cpu import extract_function, extract_struct

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "source/Force.c").read_text()
prefix = r'''
#include <assert.h>
#include "openmx_env.h"
#include "openmx_core_native_resources.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum {cudaSuccess, cudaMemcpyHostToDevice};
static int held,denied,inject,low_memory,allocation,copies,fail_allocation,fail_copy,live;
static int MYID_MPI_COMM_WORLD;
static int weak_fp64,device_query_failure,usable=1;
static void *tracked[4];
static int gpu_rank_device_usable(void){return usable;}
static int openmx_gpu_capacity_test_recovery(void){return inject;}
static int openmx_gpu_local_size_noncollective(void){return 18;}
static int Force_env_flag(const char *s,int d){(void)s;return d;}
static int cudaGetDevice(int *d){assert(usable);*d=0;return device_query_failure;}
static int openmx_gpu_device_weak_fp64(int d){assert(d==0);return weak_fp64;}
static int cudaGetLastError(void){return 0;}
static int cudaDeviceSynchronize(void){assert(held);return 0;}
static void acc_wait_all(void){assert(held);}
static void acc_clear_freelists(void){assert(held);}
static int OpenMX_GpuMemGetInfo(size_t *f,size_t *t){assert(held);*f=*t=low_memory?0:1073741824;return 0;}
static void *openmx_mgpu_admit_devices(const int *d,int n){assert(!held && n==1 && *d==0);if(denied)return NULL;held=1;return &held;}
static void openmx_mgpu_release_devices(void *p){assert(p==&held && held && !live);held=0;}
static int cudaMalloc(void **p,size_t n){
 assert(held);*p=NULL;if(++allocation==fail_allocation)return 1;
 *p=malloc(n);assert(*p);for(int i=0;i<4;i++)if(!tracked[i]){tracked[i]=*p;live++;return 0;}abort();
}
static int cudaFree(void *p){
 assert(held);for(int i=0;i<4;i++)if(tracked[i]==p){free(p);tracked[i]=NULL;live--;return 0;}abort();
}
static int cudaMemcpy(void *d,const void *s,size_t n,int k){
 assert(held && k==cudaMemcpyHostToDevice);if(++copies==fail_copy)return 1;memcpy(d,s,n);return 0;
}
/* Workspace planning has separate production planner tests. Supply two
   distinct metadata costs so both archive lifetimes are exercised here. */
static void Force4B_GpuWorkspaceSizes(size_t *a,size_t *b){*a=4096;*b=8192;}
'''
body = extract_struct(source, "Force4BGpuContext") + extract_struct(source, "Force4BArchiveLease")
for name in ("Force_gpu_node_reserve", "Force4B_ArchiveEnd", "Force4B_ArchiveBegin", "Force4B_ArchiveGpuPreferred"):
    body += extract_function(source, name)
main = r'''
int main(void){
 const char *controls[]={NULL,"auto","0","1"};
 int policy_cases=0;
 for(int setting=0;setting<4;setting++)for(int resident=0;resident<2;resident++)for(usable=0;usable<2;usable++)
 for(weak_fp64=0;weak_fp64<2;weak_fp64++)for(inject=0;inject<2;inject++)
 for(device_query_failure=0;device_query_failure<2;device_query_failure++){
  if(controls[setting])setenv("OPENMX_FORCE4B_ARCHIVE_GPU",controls[setting],1);
  else unsetenv("OPENMX_FORCE4B_ARCHIVE_GPU");
  int expected=resident || inject || (setting==2?0:setting==3?1:!usable || device_query_failure || !weak_fp64);
  assert(Force4B_ArchiveGpuPreferred(resident)==expected);
  policy_cases++;
 }
 unsetenv("OPENMX_FORCE4B_ARCHIVE_GPU");device_query_failure=weak_fp64=inject=0;usable=1;
 float flat[4*37],halo[29],c2[4*43];
 for(int i=0;i<4*37;i++)flat[i]=i+.125f;
 for(int i=0;i<29;i++)halo[i]=i-.25f;
 for(int i=0;i<4*43;i++)c2[i]=.5f-i;
 int cases=0;
 for(int kind=1;kind<=2;kind++)for(int scenario=0;scenario<9;scenario++){
  if(kind==2 && (scenario==5 || scenario==7))continue;
  assert(!held && !live);denied=inject=low_memory=allocation=copies=fail_allocation=fail_copy=0;
  Force4BGpuContext g={0};g.flat_host=flat;g.halo_host=halo;g.c2_host=c2;
  g.flat_stride=37;g.halo_count=29;g.c2_stride=43;
  if(scenario==1)denied=1;
  if(scenario==2)low_memory=1;
  if(scenario==3)inject=1;
  if(scenario==4)fail_allocation=1;
  if(scenario==5)fail_allocation=2;
  if(scenario==6)fail_copy=1;
  if(scenario==7)fail_copy=2;
  if(scenario==8){g.flat_dev=flat;g.halo_dev=halo;g.c2_dev=c2;}
  Force4BArchiveLease lease={0};
  int ready=Force4B_ArchiveBegin(&g,kind,&lease);
  assert(ready==(scenario==0));
  if(ready){
   assert(held && live==(kind==1?2:1) && copies==(kind==1?2:1));
   if(kind==1){assert(!memcmp(g.flat_dev,flat,sizeof(flat)));assert(!memcmp(g.halo_dev,halo,sizeof(halo)));}
   else assert(!memcmp(g.c2_dev,c2,sizeof(c2)));
  }
  Force4B_ArchiveEnd(&g,&lease);
  Force4B_ArchiveEnd(&g,&lease); /* cleanup after decline / repeated cleanup */
  assert(!held && !live && g.flat_host==flat && g.halo_host==halo && g.c2_host==c2);
  if(scenario==8)assert(g.flat_dev==flat && g.halo_dev==halo && g.c2_dev==c2);
  else assert(!g.flat_dev && !g.halo_dev && !g.c2_dev);
  cases++;
 }
 printf("PASS: %d Force4B archive cases and %d hardware/override policy cases\n",cases,policy_cases);
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-force-archive-") as directory:
    p = Path(directory)
    (p / "check.c").write_text(prefix + body + main)
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
        "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
        "-I"+str(ROOT/"source"), str(ROOT/"source/openmx_env.c"), str(p / "check.c"), "-o", str(p / "check")], check=True)

    def sanitizer_stack():
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (8*1024*1024, hard))

    subprocess.run([str(p / "check")], check=True, timeout=30, preexec_fn=sanitizer_stack)
