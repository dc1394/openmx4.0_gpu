#!/usr/bin/env python3
"""Inject each late allocation failure into Hamiltonian cache mapping."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
root = Path(__file__).resolve().parents[1]
source = (root / 'source/Set_Hamiltonian.c').read_text()
helper = source[source.index('typedef struct { void *host; size_t bytes; int created; }'):source.index('static void Set_Hamiltonian_ME_EnterDeviceCache(')]
prefix = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#define Host_ID 0
static void *hosts[32], *devices[32];
static size_t lengths[32];
static int fail_after=-1, live=0, allocations=0;
static void *acc_deviceptr(void *host) {for(int i=0;i<32;i++)if(hosts[i]==host)return devices[i];return NULL;}
static void acc_map_data(void *host,void *device,size_t n) {
 assert(!acc_deviceptr(host));for(int i=0;i<32;i++)if(!hosts[i]){hosts[i]=host;devices[i]=device;lengths[i]=n;return;}abort();
}
static void acc_unmap_data(void *host) {for(int i=0;i<32;i++)if(hosts[i]==host){hosts[i]=devices[i]=NULL;return;}abort();}
static int cudaMalloc(void **p,size_t n) {if(allocations++==fail_after)return 1;*p=malloc(n);assert(*p);live++;return 0;}
static int cudaFree(void *p) {free(p);live--;return 0;}
static int cudaGetLastError(void) {return 0;}
static int cudaMemcpy(void *to,const void *from,size_t n,int kind) {memcpy(to,from,n);return 0;}
static void Set_Hamiltonian_abort(const char *a,const char *b,int r) {abort();}
'''
main = r'''
int main(void) {
 int host[13][17], cases=0;
 for(int i=0;i<13;i++)for(int j=0;j<17;j++)host[i][j]=i*73+j;
 for(int retained=0;retained<13;retained++)for(int fail=0;fail<=12;fail++) {
  SetHGpuMapping old={host[retained],sizeof(host[0]),0};
  fail_after=-1;allocations=0;assert(SetH_MapBatch(&old,1,1));
  void *saved=acc_deviceptr(host[retained]);
  SetHGpuMapping maps[14];
  for(int i=0;i<13;i++)maps[i]=(SetHGpuMapping){host[i],sizeof(host[i]),0};
  maps[13]=(SetHGpuMapping){NULL,0,0};
  fail_after=fail;allocations=0;
  int ok=SetH_MapBatch(maps,14,1);
  assert(ok==(fail==12));
  assert(acc_deviceptr(host[retained])==saved);
  assert(live==(ok?13:1));
  if(ok)for(int i=0;i<13;i++)assert(!memcmp(acc_deviceptr(host[i]),host[i],sizeof(host[i])));
  SetH_UnmapBatch(maps,14);
  assert(live==1);assert(!memcmp(saved,host[retained],sizeof(host[0])));
  SetH_UnmapBatch(&old,1);assert(live==0);cases++;
 }
 fail_after=-1;SetHGpuMapping direct={host[0],sizeof(host[0]),0};
 assert(SetH_MapBatch(&direct,1,0));assert(live==1);
 SetH_UnmapBatch(&direct,1);assert(live==0);
 printf("PASS: %d Hamiltonian late-allocation rollback cases plus direct allocation\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-hamiltonian-map-') as folder:
    p = Path(folder)
    (p/'check.c').write_text(prefix + helper + main)
    subprocess.run(shlex.split(os.environ.get('CC','cc')) + ['-std=c11','-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',str(p/'check.c'),'-o',str(p/'check')],check=True)
    subprocess.run([str(p/'check')],check=True,timeout=30)
