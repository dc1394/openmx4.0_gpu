#!/usr/bin/env python3
"""Exercise production free-memory capping across concurrent allocations."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "source/set_cuda_default_device_from_local_rank.c").read_text()
start = source.index("cudaError_t OpenMX_GpuMemGetInfo(")
end = source.index("\n/* Many ranks", start)
prefix = r'''
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdarg.h>
#define MiB ((size_t)1024*1024)
typedef int cudaError_t;
typedef int nvmlDevice_t;
typedef struct { size_t free; } nvmlMemory_t;
enum { cudaSuccess=0, NVML_SUCCESS=0 };
static int level_stdout=2, mpi_comm_level1=0;
static int calls, notices, enabled=1, nvml_ok=1, first_ok=1, second_ok=1;
static size_t first_free=1000*MiB, second_free=600*MiB, nvml_free=700*MiB;
static int cudaMemGetInfo(size_t *f,size_t *t) {
  ++calls; if ((calls==1 && !first_ok)||(calls>1 && !second_ok)) return 7;
  *f=calls==1?first_free:second_free; *t=2000*MiB; return 0;
}
static int gpu_meminfo_nvml_enabled(void) {return enabled;}
static int cudaGetDevice(int *d) {*d=0;return 0;}
static int cudaGetLastError(void) {return 0;}
static int gpu_meminfo_nvml_device(int d,nvmlDevice_t *h) {*h=d;return nvml_ok;}
static int gpu_meminfo_nvmlDeviceGetMemoryInfo(int h,nvmlMemory_t *m) {(void)h;m->free=nvml_free;return 0;}
static void MPI_Initialized(int *i) {*i=1;}
static void MPI_Comm_rank(int c,int *r) {(void)c;*r=0;}
static int notice(const char *fmt,...) {(void)fmt;notices++;return 0;}
#define printf notice
'''
main = r'''
#undef printf
static void check(int ok,const char *msg) {if(!ok){fprintf(stderr,"FAIL: %s\n",msg);exit(1);}}
int main(int argc,char **argv) {
  int mode=argc>1?atoi(argv[1]):0,expected_calls=2,expected_notices=0;
  size_t free_bytes=123,total=456,want=600*MiB;
  if(mode==1){second_free=1000*MiB;want=700*MiB;expected_notices=1;}
  if(mode==2){second_free=700*MiB;want=700*MiB;}
  if(mode==3){second_ok=0;want=700*MiB;}
  if(mode==4){enabled=0;want=first_free;expected_calls=1;}
  if(mode==5){nvml_ok=0;want=first_free;expected_calls=1;}
  if(mode==6){nvml_free=1500*MiB;want=first_free;expected_calls=1;}
  if(mode==7){nvml_free=900*MiB;want=nvml_free;expected_calls=1;}
  if(mode==8){first_ok=0;want=123;expected_calls=1;}
  int status=OpenMX_GpuMemGetInfo(&free_bytes,&total);
  check(status==(mode==8?7:0),"CUDA status");
  check(free_bytes==want,"conservative free memory");
  check(total==(mode==8?456:2000*MiB),"total memory");
  check(calls==expected_calls,"query count");
  check(notices==expected_notices,"no unsigned-underflow warning");
  calls=0; first_ok=1;
  check(OpenMX_GpuMemGetInfo(NULL,&total)==0 && calls==1,"optional free output");
  return 0;
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-meminfo-") as tmp:
    cfile, binary = Path(tmp)/"check.c", Path(tmp)/"check"
    cfile.write_text(prefix + source[start:end] + main)
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) +
                   ["-std=c99", "-Wall", "-Wextra", "-Werror", str(cfile), "-o", str(binary)], check=True)
    for case in range(9):
        subprocess.run([str(binary), str(case)], check=True)
print("PASS: 9 GPU memory-query race/fallback cases")
