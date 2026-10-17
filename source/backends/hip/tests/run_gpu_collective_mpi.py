#!/usr/bin/env python3
"""Real MPI tests of capacity shard transfer/rollback with simulated nodes."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
driver = r'''
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_stdout_level.h"
int level_stdout=1;
static int rank_id,case_id,commits,remote_calls;
static int node_split(MPI_Comm comm,int type,int key,MPI_Info info,MPI_Comm *out) {
 (void)type;(void)info;
 int r;MPI_Comm_rank(comm,&r);
 return MPI_Comm_split(comm,case_id==6?0:r/2,key,out);
}
static void *fault_malloc(size_t n) {
 if(case_id==7 && rank_id==2 && n>128)return NULL;
 return malloc(n);
}
#define MPI_Comm_split_type node_split
#define malloc fault_malloc
#include "openmx_gpu_collective.c"
#undef malloc
#undef MPI_Comm_split_type
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL case=%d rank=%d line=%d: %s\n",case_id,rank_id,__LINE__,#x);MPI_Abort(MPI_COMM_WORLD,1);}}while(0)
typedef struct {size_t first,last; int origin;} Packet;
typedef struct {int values[12];} Work;
static int pack(void *ctx,size_t first,size_t last,void **p,size_t *bytes) {
 (void)ctx;
 if(case_id==4)return 0;
 *bytes=sizeof(Packet)+(case_id==8 ? (size_t)17*1024*1024 : case_id==7 ? 1024 : 0);
 Packet *q=calloc(1,*bytes);CHECK(q);
 q->first=first;q->last=last;q->origin=rank_id;*p=q;return 1;
}
static int run(void *context,const void *p,size_t bytes,void **out,size_t *n) {
 (void)context;
 const Packet *q=p;CHECK(bytes>=sizeof(*q));remote_calls++;
 if((case_id==3 && rank_id==2) || case_id==9 || (case_id==11 && rank_id==0))return 0;
 *n=(q->last-q->first)*sizeof(int);int *v=malloc(*n);CHECK(v);
 for(size_t i=q->first;i<q->last;i++)v[i-q->first]=q->origin*100+(int)i*3;
 *out=v;return 1;
}
static void commit(void *ctx,size_t first,size_t last,const void *v,size_t bytes) {
 Work *w=ctx;CHECK(bytes==(last-first)*sizeof(int));
 memcpy(w->values+first,v,bytes);commits++;
}
int main(int argc,char **argv) {
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank_id);
 OpenMXGpuCollectiveOps ops={pack,run,commit};
 for(case_id=0;case_id<12;case_id++) {
  Work w;for(int i=0;i<12;i++)w.values[i]=-7;
  commits=remote_calls=0;
  int pending=case_id==1 || ((case_id==2)?(rank_id==0 || rank_id==4):(case_id>0 && rank_id==0));
  int help=case_id==10?rank_id==2:case_id!=2 || rank_id<4;
  int n=case_id==5?1:12;
  int rc=openmx_gpu_collective_retry(MPI_COMM_WORLD,&ops,&w,pending?n:0,!pending,help,"mock");
  int failed=pending && (case_id==4 || case_id==6 || case_id==9);
  CHECK(rc==!failed);
  if(pending && !failed) {
   CHECK(commits>0);
   for(int i=0;i<n;i++)CHECK(w.values[i]==rank_id*100+i*3);
  } else {
   CHECK(commits==0);
   for(int i=0;i<12;i++)CHECK(w.values[i]==-7);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank_id==0)printf("PASS: collective case %d\n",case_id);
 }
 MPI_Finalize();return 0;
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-gpu-collective-") as folder:
    folder = Path(folder)
    (folder / "check.c").write_text(driver)
    exe = folder / "check"
    subprocess.run(["mpicc", "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fno-sanitize-recover=all", "-I"+str(root/"source"),
                    str(folder/"check.c"), "-o", str(exe)], check=True)
    subprocess.run(["mpirun", "--oversubscribe", "-np", "6", str(exe)], check=True, timeout=45)
