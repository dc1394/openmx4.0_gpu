#!/usr/bin/env python3
"""Exercise both production nonlocal invocation wrappers without MPI/GPU.

The real C++ owner must release each call's worker tensors. A subsequent CPU
call must reset HIP's GPU-use marker instead of inheriting the previous case.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CORE = Path(os.environ.get("OPENMX_CORE_LIBRARY", ROOT / "build/core-serial/libopenmx_core.so")).resolve(strict=True)


def function(source, declaration):
    begin = source.index(declaration)
    end = source.index("{", begin) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


PREFIX = r'''
#include "openmx_core_solver_workspace.h"
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct Atomic OpenMXAtomicViews;
typedef struct Reciprocal OpenMXReciprocalViews;
/* The workspace header supplies the actual electronic-view type. */
static int requested_gpu,SetNonlocal_gpu_used;
static void dtime(double *out){static double tick;*out=tick++;}
static double OpenMX_Solver_Run(OpenMXSolverOperation op,void *args){
  double result;char error[256];
  assert(openmx_core_solver_run(omp_get_max_threads(),op,args,&result,error,sizeof(error)));
  return result;
}
static void Nonlocal0(const OpenMXAtomicViews *a,const OpenMXReciprocalViews *r,
    double *****h,double ******d,const OpenMXElectronicViews *e,OpenMXSolverInvocation *owner){
  assert(!a && !r && !h && !d && !e);
  if(HIP_CASE)assert(SetNonlocal_gpu_used==0);
#pragma omp parallel
  {
    const int worker=omp_get_thread_num();
    const size_t shape[4]={3,2,5,1};
    OpenMXSolverComplexTensor first,again;char error[256];
    assert(openmx_core_solver_complex(owner,worker,"nonlocal-hamiltonian",shape,&first,error,sizeof(error)));
    for(size_t i=0;i<30;++i)assert(first.values[i].r==0 && first.values[i].i==0);
    openmx_solver_complex_plane_row(&first,2,1)[4]=(dcomplex){17+worker,-3-worker};
    assert(openmx_core_solver_complex(owner,worker,"nonlocal-hamiltonian",shape,&again,error,sizeof(error)));
    assert(again.values==first.values && again.values[29].r==17+worker);
  }
  SetNonlocal_gpu_used=requested_gpu;
}
'''
MAIN = r'''
int main(void){
  const int sequence[]={1,0,0,1,1,0};int calls=0;
  omp_set_dynamic(0);
  for(int threads=1;threads<=8;threads*=2){
    omp_set_num_threads(threads);
    for(size_t i=0;i<sizeof(sequence)/sizeof(sequence[0]);++i){
      requested_gpu=sequence[i];
      assert(Set_Nonlocal(NULL,NULL,NULL,NULL,NULL)==1);
      assert(SetNonlocal_gpu_used==requested_gpu);++calls;
    }
  }
  printf("PASS: %d nonlocal invocations; per-call scratch, worker reuse and GPU-to-CPU status reset\n",calls);
}
'''

with tempfile.TemporaryDirectory(prefix="openmx-nonlocal-lifetime-") as directory:
    tmp = Path(directory)
    for hip, name in enumerate(("source/Set_Nonlocal.c", "source/backends/hip/src/Set_Nonlocal.c")):
        source = (ROOT / name).read_text()
        arguments = re.search(r"typedef struct \{\s*const OpenMXAtomicViews \*atomic;.*?\} SetNonlocalArguments;", source, re.S)
        assert arguments, name
        program = (PREFIX.replace("HIP_CASE", str(hip)) + arguments[0] + "\n" +
                   function(source, "static int SetNonlocal_Operation(") + "\n" +
                   function(source, "double Set_Nonlocal(") + MAIN)
        c = tmp / ("hip.c" if hip else "cuda.c")
        exe = c.with_suffix("")
        c.write_text(program)
        command = shlex.split(os.environ.get("CC", "cc")) + shlex.split(os.environ.get("CFLAGS", ""))
        subprocess.run(command + ["-std=c11", "-O2", "-fopenmp", "-I" + str(ROOT / "core/include"),
                       str(c), str(CORE), "-Wl,-rpath," + str(CORE.parent), "-lm", "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
