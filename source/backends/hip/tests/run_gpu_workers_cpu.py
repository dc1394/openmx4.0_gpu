#!/usr/bin/env python3
"""Validate GPU ownership, phase rollback and thread/device restoration."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
HERE=Path(__file__).resolve().parents[1]/"source"
with tempfile.TemporaryDirectory(prefix='openmx-aux-workers-') as directory:
    tmp=Path(directory)
    (tmp/'openmx_mgpu_runtime.h').write_text('''
#include <stddef.h>
typedef int hipError_t;
#define hipSuccess 0
int hipGetDevice(int*);int hipSetDevice(int);int hipFree(void*);
int hipGetLastError(void);int hipDeviceSynchronize(void);
int openmx_mgpu_scope_enter(int);void openmx_mgpu_scope_leave(int);
int openmx_mgpu_pool_devices(int*,int);int openmx_mgpu_scope_devices(int*,int);
void *openmx_mgpu_admit_devices(const int*,int);void openmx_mgpu_release_devices(void*);
''')
    (tmp/'set_hip_default_device_from_local_rank.h').write_text('''
int gpu_rank_device_usable(void);
int openmx_gpu_map_rank_to_device(int,int,int);
int OpenMX_GpuMemGetInfo(size_t*,size_t*);void OpenMX_GpuPresizeKernelStack(void);
''')
    (tmp/'openmx_target.h').write_text('''
#include <stddef.h>
void *openmx_target_alloc(size_t);void openmx_target_free(void*);void openmx_target_wait(void);
''')
    test=r'''
#define _POSIX_C_SOURCE 200809L
#include "openmx_gpu_workers.h"
#include <assert.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static _Thread_local int device=0,omp_device=0,scope=0;
static int permitted=4,group_first=0,group_count=0,held=0;
static size_t memory[64]; static int queried[64];
static pthread_mutex_t admission=PTHREAD_MUTEX_INITIALIZER;
static long checks=0;
#define CHECK(x) do { if(!(x)){fprintf(stderr,"check failed line %d: %s\n",__LINE__,#x);exit(1);}checks++; }while(0)
int hipGetDevice(int *p){*p=device;return 0;}
int hipSetDevice(int v){device=v;return 0;}
int hipFree(void *p){assert(!p);return 0;}
int hipGetLastError(void){return 0;}
int gpu_rank_device_usable(void){return 1;}
int hipDeviceSynchronize(void){return 0;}
int omp_get_default_device(void){return omp_device;}
void omp_set_default_device(int v){omp_device=v;}
void *openmx_target_alloc(size_t n){return malloc(n);}
void openmx_target_free(void *p){free(p);}
void openmx_target_wait(void){}
void acc_clear_freelists(void){}
int OpenMX_GpuMemGetInfo(size_t *f,size_t *t){queried[device]++;*t=(size_t)4<<30;*f=memory[device]?memory[device]:*t;return 0;}
void OpenMX_GpuPresizeKernelStack(void){}
int openmx_mgpu_scope_enter(int v){int old=scope;scope=v?old+1:0;return old;}
void openmx_mgpu_scope_leave(int v){scope=v;}
int openmx_mgpu_pool_devices(int *p,int capacity){assert(permitted<=capacity);for(int i=0;i<permitted;i++)p[i]=i;return permitted;}
int openmx_mgpu_scope_devices(int *p,int capacity){assert(group_count<=capacity);for(int i=0;i<group_count;i++)p[i]=group_first+i;return group_count;}
void *openmx_mgpu_admit_devices(const int *p,int n){for(int i=0;i<n;i++)assert(p[i]>=0 && p[i]<64);pthread_mutex_lock(&admission);assert(!held);held=1;return &held;}
void openmx_mgpu_release_devices(void *p){assert(p==&held && held);held=0;pthread_mutex_unlock(&admission);}
int openmx_gpu_map_rank_to_device(int rank,int size,int count){return count>0?rank/((size+count-1)/count):-1;}
typedef struct { const OpenMXGpuWorkers *pool;int fail_worker,fail_phase;
 int prepared[64],executed[64],committed[64],released[64];size_t offset[65];int *staging[64];int output[173]; } Job;
static int callback(void *opaque,int worker,int phase){
 Job *j=opaque;
 assert(device==j->pool->devices[worker] && omp_device==device);
 assert(openmx_gpu_worker_device()==device && scope==0);
 if(phase==OPENMX_GPU_WORKER_PREPARE){
  j->staging[worker]=malloc(173*sizeof(int));assert(j->staging[worker]);j->prepared[worker]++;
 }
 if(phase==j->fail_phase && worker==j->fail_worker)return 0;
 if(phase==OPENMX_GPU_WORKER_EXECUTE){
  for(int i=0;i<j->pool->count;i++)assert(j->prepared[i]==1);
  for(size_t i=j->offset[worker];i<j->offset[worker+1];i++)j->staging[worker][i]=(int)i*7-4;
  j->executed[worker]++;
  assert(!openmx_gpu_workers_run(j->pool,callback,opaque,"recursive"));
 }
 if(phase==OPENMX_GPU_WORKER_COMMIT){
  for(int i=0;i<j->pool->count;i++)assert(j->executed[i]==1);
  for(size_t i=j->offset[worker];i<j->offset[worker+1];i++)j->output[i]=j->staging[worker][i];
  j->committed[worker]++;
 }
 if(phase==OPENMX_GPU_WORKER_RELEASE){free(j->staging[worker]);j->staging[worker]=NULL;j->released[worker]++;}
 return 1;
}
static size_t cost(void *opaque,int worker,int count){
 (void)worker;size_t n=*(size_t*)opaque;return n/(size_t)count+(n%(size_t)count!=0);
}
static size_t uneven_cost(void *context,int worker,int count){
 (void)context;return (size_t)(count==1?5:worker?3:2)<<30;
}
int main(void){
 /* Auto leaves other HIP devices untouched while the owner fits; when
    it does not, it uses the smallest admitted group, then reports CPU. */
 {OpenMXGpuWorkers p;size_t need=(size_t)1<<30;
  unsetenv("OPENMX_GPU_WORKERS");device=omp_device=0;permitted=4;
  CHECK(openmx_gpu_workers_capacity(&p,173,0,cost,&need) && p.count==1 && p.devices[0]==0);
  CHECK(queried[0]==1 && queried[1]==0 && queried[2]==0 && queried[3]==0);
  OpenMXGpuWorkerRetry retry={cost,&need,{0}};
  for(int count=1;count<=4;count++){
   CHECK(openmx_gpu_workers_capacity(&p,173,0,openmx_gpu_worker_retry_cost,&retry) && p.count==count);
   retry.rejected[count]=1;
  }
  CHECK(!openmx_gpu_workers_capacity(&p,173,0,openmx_gpu_worker_retry_cost,&retry));
  need=(size_t)7<<30;
  CHECK(openmx_gpu_workers_capacity(&p,173,0,cost,&need) && p.count==2);
  CHECK(device==0 && omp_device==0);
  need=(size_t)12<<30;
  CHECK(openmx_gpu_workers_capacity(&p,173,0,cost,&need) && p.count==4);
  permitted=3;
  CHECK(!openmx_gpu_workers_capacity(&p,173,0,cost,&need) && p.count==0);
  permitted=4;setenv("OPENMX_GPU_WORKERS","off",1);
  CHECK(!openmx_gpu_workers_capacity(&p,173,0,cost,&need) && p.count==0);
  unsetenv("OPENMX_GPU_WORKERS");group_first=2;group_count=2;need=(size_t)7<<30;
  CHECK(openmx_gpu_workers_capacity(&p,173,2,cost,&need) && p.count==2 && p.devices[0]==2 && p.devices[1]==3);
  group_count=0;
  memory[0]=1;need=(size_t)1<<30;
  CHECK(openmx_gpu_workers_capacity(&p,1,0,cost,&need) && p.count==1 && p.devices[0]!=0);
  permitted=3;memory[1]=((size_t)2<<30)+((size_t)300<<20);memory[2]=((size_t)3<<30)+((size_t)300<<20);
  CHECK(openmx_gpu_workers_capacity(&p,173,0,uneven_cost,NULL) && p.count==2 && p.devices[0]==1 && p.devices[1]==2);
  memset(memory,0,sizeof(memory));permitted=4;need=SIZE_MAX;
  CHECK(!openmx_gpu_workers_capacity(&p,173,0,cost,&need));
 }
 /* The distribution is experimental and off by default: every pool is the
    rank's own GPU unless OPENMX_GPU_WORKERS is 1 or on. */
 for(int setting=0;setting<4;setting++){
  const char *value[4]={NULL,"0","off","2"};
  if(value[setting]) setenv("OPENMX_GPU_WORKERS",value[setting],1); else unsetenv("OPENMX_GPU_WORKERS");
  permitted=4;device=omp_device=1;openmx_gpu_workers_configure(0,1,4);
  OpenMXGpuWorkers p;openmx_gpu_workers_local(&p,173);CHECK(p.count==1 && p.devices[0]==1);
  openmx_gpu_workers_owner(&p,173);CHECK(p.count==1 && p.devices[0]==1);
 }
 setenv("OPENMX_GPU_WORKERS","ON",1);
 {OpenMXGpuWorkers p;openmx_gpu_workers_owner(&p,173);CHECK(p.count==4);}
 setenv("OPENMX_GPU_WORKERS","1",1);
 /* Every mapped owner is retained and every otherwise idle permitted GPU
    has exactly one extra owner, even when the old block map has holes. */
 for(int visible=1;visible<=16;visible++)for(int ranks=1;ranks<=32;ranks++)for(int cap=1;cap<=visible;cap++){
  permitted=cap;int coverage[64]={0},original[64]={0};
  for(int rank=0;rank<ranks;rank++)original[openmx_gpu_map_rank_to_device(rank,ranks,visible)]++;
  for(int rank=0;rank<ranks;rank++){
   device=omp_device=openmx_gpu_map_rank_to_device(rank,ranks,visible);
   openmx_gpu_workers_configure(rank,ranks,visible);
   OpenMXGpuWorkers p;openmx_gpu_workers_local(&p,173);
   CHECK(p.count>=1 && p.devices[0]==device);
   int seen[64]={0};
   for(int i=0;i<p.count;i++){int d=p.devices[i];CHECK(d>=0 && d<visible && !seen[d]++);coverage[d]++;if(i)CHECK(d<cap && !original[d]);}
  }
  if(cap>1)for(int d=0;d<cap;d++)CHECK(coverage[d]==(original[d]?original[d]:1));
 }
 /* Cuts cover every item once, including empty tails and near-SIZE_MAX. */
 for(int workers=1;workers<=64;workers++)for(int n=0;n<=173;n++){
  size_t previous=0;
  for(int i=0;i<=workers;i++){
   size_t cut=openmx_gpu_worker_cut(n,i,workers);CHECK(cut>=(size_t)previous && cut<=(size_t)n);previous=cut;
  }CHECK(previous==(size_t)n);
 }
 for(int workers=1;workers<=64;workers++){
  size_t n=SIZE_MAX-49,previous=0;
  for(int i=0;i<=workers;i++){
   size_t cut=openmx_gpu_worker_cut(n,i,workers);CHECK(cut>=previous && cut<=n);previous=cut;
  }CHECK(previous==n);
 }
 permitted=4;device=omp_device=3;scope=11;
 openmx_gpu_workers_configure(0,1,4);
 for(int dynamic=0;dynamic<2;dynamic++)for(int workers=1;workers<=4;workers*=2)
 for(int failure=-1;failure<workers;failure++)for(int phase=0;phase<2;phase++){
  omp_set_dynamic(dynamic);group_first=4-workers;group_count=workers;
  OpenMXGpuWorkers p;openmx_gpu_workers_scope(&p,173);CHECK(p.count==workers);
  for(int i=0;i<workers;i++)CHECK(p.devices[i]==group_first+i);
  Job j={0};j.pool=&p;j.fail_worker=failure;j.fail_phase=phase;
  for(int i=0;i<173;i++)j.output[i]=999;
  for(int i=0;i<=workers;i++)j.offset[i]=openmx_gpu_worker_cut(173,i,workers);
  CHECK(openmx_gpu_workers_run(&p,callback,&j,"test")== (failure<0));
  CHECK(device==3 && omp_device==3 && scope==11 && openmx_gpu_worker_device()==-1);
  for(int i=0;i<workers;i++)CHECK(j.released[i]==1 && j.staging[i]==NULL);
  for(int i=0;i<173;i++)CHECK(j.output[i]==(failure<0?i*7-4:999));
 }
 puts("PASS: GPU worker ownership, limits, cuts, phase barriers, failure rollback, binding restoration and dynamic OpenMP teams");
 printf("checks=%ld\n",checks);return 0;
}
'''
    (tmp/'check.c').write_text(test)
    (tmp/'openmx_gpu_workers.c').write_text((HERE/'openmx_gpu_workers.c').read_text())
    exe=tmp/'check'
    subprocess.run(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O1','-g','-fopenmp','-pthread','-Wall','-Wextra','-I'+str(tmp),'-I'+str(HERE),str(tmp/'openmx_gpu_workers.c'),str(tmp/'check.c'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],env={**os.environ,'OMP_NUM_THREADS':'1','OMP_THREAD_LIMIT':'4'},check=True)
