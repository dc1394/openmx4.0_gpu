#!/usr/bin/env python3
"""Inject each host-packet allocation failure before Force writes its packet."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
from run_force4b_stream_cpu import extract_function, extract_struct

root = Path(__file__).resolve().parents[1]
source = (root/"source/Force.c").read_text()
prefix = r'''
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_gpu_task.h"
static int mpi_comm_level1,allocations,fail_at;
static void *packet_malloc(size_t n) { return ++allocations==fail_at?NULL:malloc(n); }
static void Force4B_gpu_abort(const char *s) { fprintf(stderr,"%s\n",s);exit(42); }
static int openmx_fp32_enabled(void) { return 0; }
int MPI_Allreduce(const void *in,void *out,int n,int type,int op,int comm) {
 (void)n;(void)type;(void)op;(void)comm;*(int*)out=*(const int*)in;return 0;
}
int openmx_gpu_task_execute(MPI_Comm c,OpenMXGpuTask *t,int e,const char *label) {
 (void)c;(void)t;(void)e;(void)label;abort(); /* Failure must precede dispatch. */
}
static void Force4B_capacity_kernel(size_t a,size_t b,const void *p,void *const v[],int g) {
 (void)a;(void)b;(void)p;(void)v;(void)g;
}
static void Force_HNL_capacity_kernel(size_t a,size_t b,const void *p,void *const v[],int g) {
 (void)a;(void)b;(void)p;(void)v;(void)g;
}
static void Force4B_capacity_copy(float *a,const float *b,const float *c,size_t d,size_t e) {
 (void)a;(void)b;(void)c;(void)d;(void)e;abort();
}
#define malloc packet_malloc
'''
body = "".join(extract_struct(source, n) for n in (
    "Force4BGpuContext", "Force4BGpuItem", "Force4BCapacityParameters",
    "ForceHNLGpuContext", "ForceHNLCapacityItem"))
body += "".join(extract_function(source, n) for n in (
    "Force_env_flag", "Force_gpu_stage_default", "Force4B_capacity", "Force_HNL_capacity"))
main = r'''
int main(int argc,char **argv) {
 assert(argc==3);int kind=atoi(argv[1]);fail_at=atoi(argv[2]);
 if(kind==0) {
  unsetenv("OPENMX_FORCE_GPU");assert(Force_gpu_stage_default()==1);
  setenv("OPENMX_FORCE_GPU","0",1);assert(Force_gpu_stage_default()==0);
  const char *stages[]={"OPENMX_FORCE3_GPU","OPENMX_FORCE4B_GPU","OPENMX_FORCEHNL_GPU"};
  for(int i=0;i<3;++i) {
   unsetenv(stages[i]);assert(!Force_env_flag(stages[i],Force_gpu_stage_default()));
   setenv(stages[i],"1",1);assert(Force_env_flag(stages[i],Force_gpu_stage_default()));
   setenv("OPENMX_FORCE_GPU","1",1);setenv(stages[i],"0",1);
   assert(!Force_env_flag(stages[i],Force_gpu_stage_default()));
   setenv("OPENMX_FORCE_GPU","0",1);
  }
  return 0;
 }
 Force4BGpuItem item={0};item.ian=item.jan=item.k_n=1;
 Force4BGpuContext vna={0};vna.num_proj=1;
 ForceHNLGpuContext hnl={0};int nlp=1;
 if(kind==1)Force4B_capacity(1,&vna,1,&item,NULL,NULL,NULL,NULL,NULL,0);
 else Force_HNL_capacity(1,&hnl,1,&item,NULL,NULL,NULL,NULL,&nlp,NULL,NULL,NULL,0);
 return 3;
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-force-packets-") as directory:
    tmp = Path(directory)
    (tmp/"mpi.h").write_text(
        "typedef int MPI_Comm;\n#define MPI_INT 0\n#define MPI_MAX 0\n"
        "int MPI_Allreduce(const void*,void*,int,int,int,int);\n")
    (tmp/"test.c").write_text(prefix+body+main)
    subprocess.run(shlex.split(os.environ.get("CC", "cc"))+
        ["-std=c11", "-O0", "-Wall", "-Wextra", "-Werror=implicit-function-declaration",
         "-I"+str(tmp), "-I"+str(root/"source"), str(tmp/"test.c"), "-o", str(tmp/"check")], check=True)
    subprocess.run([str(tmp/"check"), "0", "0"], check=True)
    for kind, count in ((1,4),(2,5)):
        for allocation in range(1,count+1):
            result = subprocess.run([str(tmp/"check"),str(kind),str(allocation)], capture_output=True,text=True)
            assert result.returncode==42 and "cannot allocate a bounded host recovery packet" in result.stderr, result
    print("PASS: 9 Force packet allocation failures and umbrella/stage override policy")
