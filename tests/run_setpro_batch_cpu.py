#!/usr/bin/env python3
"""Check physical GPU sharing and GeForce VNA batch budgets without a GPU."""
import os
from pathlib import Path
import resource
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "source/Set_ProExpn_VNA.c").read_text()


def function(name):
    start = source.index(name)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int MPI_Comm;
enum {cudaSuccess=0,MPI_CHAR=1};
static int ranks=1,usable=1,device_failure,pci_failure,memory_failure,weak=1;
static int exchanges,memory_queries;
static size_t available;
static char identity[64],peers[64][64];
static int gpu_rank_device_usable(void){return usable;}
static int cudaGetDevice(int *device){assert(usable);*device=0;return device_failure;}
static int cudaDeviceGetPCIBusId(char *out,int n,int device){assert(device==0 && n==64);memcpy(out,identity,64);return pci_failure;}
static int cudaGetLastError(void){assert(usable);return 0;}
static int openmx_gpu_device_weak_fp64(int device){assert(device==0);return weak;}
static int OpenMX_GpuMemGetInfo(size_t *f,size_t *t){*f=available;*t=SIZE_MAX;memory_queries++;return memory_failure;}
static void *SetPro_checked_malloc(size_t n){void *p=malloc(n);assert(p);return p;}
static void MPI_Comm_size(MPI_Comm comm,int *n){assert(comm==7);*n=ranks;}
static void MPI_Allgather(const void *s,int n,int type,void *d,int m,int dtype,MPI_Comm comm){
 assert(n==64 && m==n && type==MPI_CHAR && dtype==type && comm==7);
 if(usable && !device_failure && !pci_failure)assert(!memcmp(s,identity,64));
 else assert(!((const char*)s)[0]);
 memcpy(d,peers,(size_t)ranks*64);exchanges++;
}
'''
main = r'''
int main(void){
 int cases=0;
 /* Every process may see its own GPU as ordinal 0; PCI identities still
    distinguish physical devices and count an uneven MPI distribution. */
 for(ranks=1;ranks<=33;ranks++)for(int devices=1;devices<=8;devices++){
  memset(peers,0,sizeof(peers));
  for(int rank=0;rank<ranks;rank++)snprintf(peers[rank],64,"0000:%02x:00.0",rank%devices+64);
  for(int rank=0;rank<ranks;rank++){
   memcpy(identity,peers[rank],64);
   int expected=ranks/devices+(rank%devices<ranks%devices);
   int before=exchanges;assert(SetPro_GpuSharing(7)==expected && exchanges==before+1);cases++;
  }
  for(int failure=0;failure<3;failure++){
   usable=failure!=0;device_failure=failure==1;pci_failure=failure==2;
   assert(SetPro_GpuSharing(7)==ranks);cases++;
  }
  usable=1;device_failure=pci_failure=0;
 }
 ranks=18;
 const size_t reserve=(size_t)256*1024*1024;
 for(int sharing=0;sharing<=18;sharing++)for(int room=0;room<5;room++){
  available=room==0?0:room==1?reserve-1:room==2?reserve:room==3?reserve+1:reserve+18*1234567;
  size_t expected=available>reserve?(available-reserve)/(size_t)(sharing?sharing:1):0;
  assert(SetPro_GpuBatchBudget(sharing)==expected);cases++;
 }
 weak=0;memory_queries=0;assert(SetPro_GpuBatchBudget(18)==SIZE_MAX && !memory_queries);
 weak=1;usable=0;assert(SetPro_GpuBatchBudget(18)==SIZE_MAX && !memory_queries);
 usable=1;device_failure=1;assert(SetPro_GpuBatchBudget(18)==SIZE_MAX && !memory_queries);
 device_failure=0;memory_failure=1;assert(SetPro_GpuBatchBudget(18)==SIZE_MAX && memory_queries==1);
 printf("PASS: %d VNA sharing/budget cases, invalid devices, A100/H100 policy and query failures\n",cases);
}
'''
body = function("static int SetPro_GpuSharing(") + function("static size_t SetPro_GpuBatchBudget(")
with tempfile.TemporaryDirectory(prefix="openmx-vna-budget-") as directory:
    p = Path(directory)
    (p / "check.c").write_text(prefix + body + main)
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
        "-std=c11", "-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
        str(p / "check.c"), "-o", str(p / "check")], check=True)
    def sanitizer_stack():
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (8*1024*1024, hard))

    subprocess.run([str(p / "check")], check=True, timeout=30, preexec_fn=sanitizer_stack)
