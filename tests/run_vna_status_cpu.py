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
source = (ROOT / "source/openmx_vna_assembly.h").read_text()
CORE = Path(os.environ.get("OPENMX_CORE_LIBRARY", ROOT / "build/core-serial/libopenmx_core.so")).resolve()
for backend in (ROOT / "source/Set_ProExpn_VNA.c", ROOT / "source/backends/hip/src/Set_ProExpn_VNA.c"):
    assert '#include "openmx_vna_assembly.h"' in backend.read_text()
    assert "SetPro_gpu_used" not in backend.read_text()

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
#include "openmx_core_vna_invocation.h"
#include "openmx_vna_tables.h"
#define measure_time 0
static const int GL_Mesh=4;
static const double PAO_Nkmax=3,Radial_kmin=.1;
static const double GL_Abscissae[]={-1,-1.0/3,1.0/3,1};

static int List_YOUSO[40]={[24]=1},SpeciesNum=1,atomnum=0,Matomnum=0,TCpyCell=-1;
static int angular[1]={0},count[1]={1},*Spe_Num_Basis[1]={count},*Spe_MaxL_Basis=angular;
static int *M2G,*WhatSpecies,*FNAN,**natn,**ncn;
static double **Gxyz,**atv,****Spe_VNA_Bessel,*******Spe_ProductRF_Bessel,**Spe_CrudeVNA_Bessel;
static dcomplex Comp2Real[1][1][1];
static double GL_Weight[4],*time_per_atom;
static void SetPro_gpu_abort(const char *message){fprintf(stderr,"%s\n",message);abort();}
static _Thread_local int mode,notes,ends;
static void SetPro_note(const char *stage,const char *format,...){(void)stage;(void)format;notes++;}
static double SetPro_Separable(const OpenMXVnaMatrixViews *matrices,
    const OpenMXVnaInvocationBuffers *call,const SetProVnaTables *tables,int *status){
 (void)matrices;(void)tables;const double *k=call->wavevectors;
 assert(*status==0);assert(fabs(k[0]-.1)<1e-14 && fabs(k[3]-3)<1e-14);
 assert(call->pairs.count==0 && call->preparation.orbitals[0]==1);
 *status=!!(mode&1);return 1;
}
static int SetPro_VNA23_GpuBegin(SetProGpu3Context *g){g->enabled=!(mode&8);return g->enabled;}
static void SetPro_VNA23_GpuEnd(SetProGpu3Context *g){assert(g->enabled);memset(g,0,sizeof(*g));ends++;}
static double SetPro_OneCenter(const OpenMXVnaMatrixViews *matrices,int mirror,
    const OpenMXVnaInvocationBuffers *call,const SetProVnaTables *tables,SetProGpu3Context *g){
 (void)matrices;(void)call;(void)tables;
 if(g->enabled && (mode & (mirror?4:2)))g->used_gpu=1;return mirror?3:2;
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
   double time=Set_ProExpn_VNA_WithStatus(NULL,&result);
   const int expected=!!(mode&1) || (!(mode&8) && !!(mode&6));
   assert(time==6 && result==expected && notes==2 && ends==!(mode&8));calls++;
   /* Compatibility calls and a subsequent CPU call must not retain a
      previous caller's marker. End has already cleared its local batch. */
   notes=ends=0;assert(Set_ProExpn_VNA(NULL)==6);
   assert(notes==2 && ends==!(mode&8));calls++;
   mode=0;result=77;notes=ends=0;
   assert(Set_ProExpn_VNA_WithStatus(NULL,&result)==6 && result==0);
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
        str(path), str(CORE), "-Wl,-rpath," + str(CORE.parent), "-lm", "-o", str(executable)], check=True)
    def sanitizer_stack():
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))
    subprocess.run([str(executable)], check=True, timeout=45, preexec_fn=sanitizer_stack)
