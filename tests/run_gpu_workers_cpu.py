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
typedef int cudaError_t;
#define cudaSuccess 0
int cudaGetDevice(int*);int cudaSetDevice(int);int cudaFree(void*);
int cudaGetLastError(void);int cudaDeviceSynchronize(void);
int openmx_mgpu_scope_enter(int);void openmx_mgpu_scope_leave(int);
int openmx_mgpu_pool_devices(int*,int);int openmx_mgpu_scope_devices(int*,int);
void *openmx_mgpu_admit_devices(const int*,int);void openmx_mgpu_release_devices(void*);
''')
    (tmp/'set_cuda_default_device_from_local_rank.h').write_text('''
int openmx_gpu_map_rank_to_device(int,int,int);
int OpenMX_GpuMemGetInfo(size_t*,size_t*);void OpenMX_GpuPresizeKernelStack(void);
''')
    (tmp/'openacc.h').write_text('''
#include <stddef.h>
typedef int acc_device_t;
#define acc_device_nvidia 5
int acc_get_device_type(void);int acc_get_device_num(int);void acc_set_device_num(int,int);
void *acc_malloc(size_t);void acc_free(void*);void acc_wait_all(void);
''')
    test=r'''
#include "openmx_gpu_workers.h"
#include <assert.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static _Thread_local int device=0,acc_device=0,acc_type=5,scope=0;
static int permitted=4,group_first=0,group_count=0,held=0;
static pthread_mutex_t admission=PTHREAD_MUTEX_INITIALIZER;
static long checks=0;
#define CHECK(x) do { if(!(x)){fprintf(stderr,"check failed line %d: %s\n",__LINE__,#x);exit(1);}checks++; }while(0)
int cudaGetDevice(int *p){*p=device;return 0;}
int cudaSetDevice(int v){device=v;return 0;}
int cudaFree(void *p){assert(!p);return 0;}
int cudaGetLastError(void){return 0;}
int cudaDeviceSynchronize(void){return 0;}
int acc_get_device_type(void){return acc_type;}
int acc_get_device_num(int type){(void)type;return acc_device;}
void acc_set_device_num(int v,int type){acc_device=v;acc_type=type;}
void *acc_malloc(size_t n){return malloc(n);}
void acc_free(void *p){free(p);}
void acc_wait_all(void){}
void acc_clear_freelists(void){}
int OpenMX_GpuMemGetInfo(size_t *f,size_t *t){*f=*t=(size_t)4<<30;return 0;}
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
 assert(device==j->pool->devices[worker] && acc_device==device && acc_type==5);
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
int main(void){
 /* Every mapped owner is retained and every otherwise idle permitted GPU
    has exactly one extra owner, even when the old block map has holes. */
 for(int visible=1;visible<=16;visible++)for(int ranks=1;ranks<=32;ranks++)for(int cap=1;cap<=visible;cap++){
  permitted=cap;int coverage[64]={0},original[64]={0};
  for(int rank=0;rank<ranks;rank++)original[openmx_gpu_map_rank_to_device(rank,ranks,visible)]++;
  for(int rank=0;rank<ranks;rank++){
   device=acc_device=openmx_gpu_map_rank_to_device(rank,ranks,visible);
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
 permitted=4;device=acc_device=3;scope=11;
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
  CHECK(device==3 && acc_device==3 && scope==11 && openmx_gpu_worker_device()==-1);
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
