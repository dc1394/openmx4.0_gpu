#!/usr/bin/env python3
"""Check HIP VNA stage-status lifetime using the production driver and wrapper."""
import os
from pathlib import Path
import re
import resource
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "source/backends/hip/src/Set_ProExpn_VNA.c").read_text()
assert "SetPro_gpu_used" not in source

def function(name):
    match = re.search(r"^double " + name + r"\([^;]*?\)\s*\{", source, re.M)
    assert match, name
    end = source.index("{", match.start()) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]

PREFIX = r'''
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef float Type_DS_VNA;
#include "openmx_vna_gpu_context.h"
#include "openmx_vna_status.h"
#define measure_time 0
static const int GL_Mesh=4;
static const double PAO_Nkmax=3,Radial_kmin=.1;
static const double GL_Abscissae[]={-1,-1.0/3,1.0/3,1};
static double *****HVNA3;
static _Thread_local int mode,notes,ends;
static void SetPro_note(const char *stage,const char *format,...){(void)stage;(void)format;notes++;}
static double Set_ProExpn(double ****h,Type_DS_VNA *****d,double *k,int *status){
 (void)h;(void)d;assert(*status==0);assert(fabs(k[0]-.1)<1e-14 && fabs(k[3]-3)<1e-14);
 *status=!!(mode&1);return 1;
}
static int SetPro_VNA23_GpuBegin(SetProGpu3Context *g){g->enabled=!(mode&8);return g->enabled;}
static void SetPro_VNA23_GpuEnd(SetProGpu3Context *g){assert(g->enabled);memset(g,0,sizeof(*g));ends++;}
static double Set_VNA2(double ****h,double *****h2,double *k,SetProGpu3Context *g){
 (void)h;(void)h2;(void)k;if(g->enabled && (mode&2))g->used_gpu=1;return 2;
}
static double Set_VNA3(double *****h,double *k,SetProGpu3Context *g){
 (void)h;(void)k;if(g->enabled && (mode&4))g->used_gpu=1;return 3;
}
'''
MAIN = r'''
int main(void){
 int calls=0;
 for(int threads=1;threads<=8;threads*=2){
  omp_set_dynamic(0);omp_set_num_threads(threads);
#pragma omp parallel for reduction(+:calls)
  for(int scenario=0;scenario<16;++scenario){
   mode=scenario;notes=ends=0;
   int result=77;
   double time=Set_ProExpn_VNA_WithStatus(NULL,NULL,NULL,&result);
   const int expected=!!(mode&1) || (!(mode&8) && !!(mode&6));
   assert(time==6 && result==expected && notes==2 && ends==!(mode&8));calls++;
   /* Compatibility calls and a subsequent CPU call must not retain a
      previous caller's marker. End has already cleared its local batch. */
   notes=ends=0;assert(Set_ProExpn_VNA(NULL,NULL,NULL)==6);
   assert(notes==2 && ends==!(mode&8));calls++;
   mode=0;result=77;notes=ends=0;
   assert(Set_ProExpn_VNA_WithStatus(NULL,NULL,NULL,&result)==6 && result==0);
   assert(notes==2 && ends==1);calls++;
  }
 }
 printf("PASS: %d VNA driver invocations; independent status, end cleanup, CPU reset and legacy wrapper\n",calls);
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-vna-status-") as directory:
    folder = Path(directory)
    path = folder / "check.c"
    path.write_text(PREFIX + function("Set_ProExpn_VNA_WithStatus") + function("Set_ProExpn_VNA") + MAIN)
    executable = folder / "check"
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
        "-std=c11", "-O1", "-g", "-fopenmp", "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all", "-I" + str(ROOT / "source"),
        str(path), "-lm", "-o", str(executable)], check=True)
    def sanitizer_stack():
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))
    subprocess.run([str(executable)], check=True, timeout=45, preexec_fn=sanitizer_stack)
