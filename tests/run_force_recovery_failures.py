#!/usr/bin/env python3
"""Inject every real C++ packet allocation failure before native dispatch."""
from pathlib import Path
import os
import resource
import shlex
import subprocess
import tempfile

from force_gpu_fixture import core_library_flags
from run_force4b_stream_cpu import extract_function, extract_struct

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "source/Force.c").read_text()
hnl = (ROOT / "source/openmx_force_hnl_gpu.h").read_text()
a = hnl.index("struct ForceHNLGpuContext {")
b = hnl.index("};", a) + 2
prefix = r'''
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "openmx_core_force_recovery.h"
#include "allocation_faults.hpp"
#include "openmx_gpu_task.h"
#include "openmx_env.h"
typedef struct OpenMXNativeResources OpenMXNativeResources;
static int mpi_comm_level1;
static void Force4B_gpu_abort(const char *message){throw std::runtime_error(message);}
static int openmx_fp32_enabled(){return 0;}
static int MPI_Allreduce(const void *in,void *out,int,int,int,int){*(int*)out=*(const int*)in;return 0;}
int openmx_gpu_task_execute(MPI_Comm,OpenMXGpuTask*,int,const char*){std::abort();}
static void Force4B_capacity_kernel(size_t,size_t,const void*,void *const[],int){}
static void Force_HNL_capacity_kernel(size_t,size_t,const void*,void *const[],int){}
static int OpenMXGpuMemcpy(void*,const void*,size_t,int){std::abort();}
enum {OPENMX_GPU_COPY_DEVICE_TO_HOST,OPENMX_GPU_SUCCESS};
typedef OpenMXForceHnlRecoveryItem ForceHNLCapacityItem;
'''
body = extract_struct(source, "Force4BGpuContext") + extract_struct(source, "Force4BGpuItem")
body += "typedef struct ForceHNLGpuContext ForceHNLGpuContext;\n" + hnl[a:b] + "\n"
body += extract_struct(source, "Force4BCapacityParameters")
body += extract_function(source, "Force_gpu_stage_default")
body += (ROOT / "source/openmx_force_recovery.h").read_text()
main = r'''
int main(){
  unsetenv("OPENMX_FORCE_GPU");assert(Force_gpu_stage_default()==1);
  setenv("OPENMX_FORCE_GPU","0",1);assert(Force_gpu_stage_default()==0);
  const char *stages[]={"OPENMX_FORCE3_GPU","OPENMX_FORCE4B_GPU","OPENMX_FORCEHNL_GPU"};
  for(const char *name:stages){
    unsetenv(name);assert(!openmx_env_flag(name,Force_gpu_stage_default()));
    setenv(name,"yes",1);assert(openmx_env_flag(name,Force_gpu_stage_default()));
    setenv("OPENMX_FORCE_GPU","1",1);setenv(name,"off",1);
    assert(!openmx_env_flag(name,Force_gpu_stage_default()));
    setenv("OPENMX_FORCE_GPU","0",1);
  }
  int cases=0;
  for(int kind=1;kind<=2;++kind)for(int fail=0;fail<(kind==1?4:5);++fail){
    Force4BGpuItem item={0,0,1,1,1};Force4BGpuContext vna={};vna.num_proj=1;
    ForceHNLGpuContext hnl={};int nlp=1;bool caught=false;
    const auto live=openmx::test::live_allocations();openmx::test::fail_allocation_after(fail);
    try {
      if(kind==1)Force4B_capacity(1,&vna,1,&item,nullptr,nullptr,nullptr,nullptr,nullptr,0);
      else Force_HNL_capacity(1,&hnl,1,&item,nullptr,nullptr,nullptr,nullptr,&nlp,nullptr,nullptr,nullptr,0);
    }catch(const std::exception& error){caught=std::strstr(error.what(),"bounded host recovery packet");}
    openmx::test::fail_allocation_after(-1);
    assert(caught && openmx::test::live_allocations()==live);++cases;
  }
  std::printf("PASS: %d real C++ Force packet allocation failures, no dispatch/leaks; umbrella overrides\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-force-packets-") as directory:
    folder = Path(directory)
    (folder / "mpi.h").write_text("typedef int MPI_Comm;\n#define MPI_INT 0\n#define MPI_MAX 0\n")
    (folder / "check.cpp").write_text(prefix + body + main)
    command = shlex.split(os.environ.get("CXX", "c++")) + [
        "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
        "-fno-pie", "-no-pie", "-I" + str(folder), "-I" + str(ROOT / "source"),
        "-I" + str(ROOT / "core/tests"), str(folder / "check.cpp"),
        str(ROOT / "source/openmx_env.c"), str(ROOT / "core/tests/allocation_faults.cpp"),
        *core_library_flags(), "-o", str(folder / "check")]
    subprocess.run(command, check=True)

    def bounded_stack():
        _, hard = resource.getrlimit(resource.RLIMIT_STACK)
        resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))

    subprocess.run([str(folder / "check")], check=True, timeout=60, preexec_fn=bounded_stack)
