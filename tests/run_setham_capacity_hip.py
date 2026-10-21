#!/usr/bin/env python3
"""Check the production HIP capacity Hamiltonian against a long-double oracle.

Uses the real ragged GPU task packer and verifies its CPU recovery path too.
Requires an AMD GPU, ROCm, and the application MPI headers.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    start = source.index("static void " + name + "(")
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


FIXTURE = r'''
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "openmx_target.h"
#include "openmx_gpu_task.c"
typedef struct {int n0,n1,points;} SetHCapacityPair;
typedef struct {int spins,max_no;double volume;} SetHCapacityParameters;
@SETH_CAPACITY@

static void verify(const double *actual,const long double *expected,size_t count) {
  for(size_t i=0;i<count;i++) {
    long double error=fabsl((long double)actual[i]-expected[i]);
    if(!isfinite(actual[i]) || error>5e-12L*fmaxl(1,fabsl(expected[i]))) {
      fprintf(stderr,"matrix[%zu] %.17g expected %.21Lg error %.4Lg\n",i,actual[i],expected[i],error);
      exit(1);
    }
  }
}
static int compare_double(const void *a,const void *b) {
  double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);
}
int main(void) {
  int devices=0,initial=1;
  assert(hipGetDeviceCount(&devices)==hipSuccess && devices>0);
  assert(hipSetDevice(0)==hipSuccess);omp_set_default_device(0);
#pragma omp target map(from:initial)
  initial=omp_is_initial_device();
  assert(!initial);
  const int lengths[]={0,1,31,257,16387};
  for(int spins=1;spins<=4;spins*=2)for(int shape=0;shape<5;shape++) {
    enum {N=3,MAX_NO=6};
    SetHCapacityPair pairs[N]={{2,3,lengths[shape]},{5,4,lengths[shape]},{1,6,lengths[shape]}};
    SetHCapacityParameters parameters={spins,MAX_NO,.125};
    size_t oa[N+1]={0},ob[N+1]={0},ow[N+1]={0};
    for(int p=0;p<N;p++) {oa[p+1]=oa[p]+(size_t)pairs[p].n0*pairs[p].points;ob[p+1]=ob[p]+(size_t)pairs[p].n1*pairs[p].points;ow[p+1]=ow[p]+(size_t)spins*pairs[p].points;}
    double *a=malloc((oa[N]+1)*sizeof(double)),*b=malloc((ob[N]+1)*sizeof(double)),*weights=malloc((ow[N]+1)*sizeof(double));
    for(size_t i=0;i<oa[N];i++)a[i]=sin(i*.17);
    for(size_t i=0;i<ob[N];i++)b[i]=cos(i*.07);
    for(size_t i=0;i<ow[N];i++)weights[i]=(i%3?-.2:1.0)+(i%19)*.01;
    const size_t stride=(size_t)spins*MAX_NO*MAX_NO,count=N*stride;
    double *output=malloc(count*sizeof(double));long double *expected=calloc(count,sizeof(long double));
    for(int p=0;p<N;p++)for(int spin=0;spin<spins;spin++)for(int i=0;i<pairs[p].n0;i++)for(int j=0;j<pairs[p].n1;j++)
      for(int k=0;k<pairs[p].points;k++)
        expected[p*stride+(spin*MAX_NO+i)*MAX_NO+j]+=(long double)parameters.volume*weights[ow[p]+(size_t)k*spins+spin]*a[oa[p]+(size_t)k*pairs[p].n0+i]*b[ob[p]+(size_t)k*pairs[p].n1+j];
    OpenMXGpuTask task={0};task.count=N;task.inputs=4;task.outputs=1;
    task.parameters=&parameters;task.parameter_bytes=sizeof(parameters);task.kernel=SetH_capacity_kernel;
    task.view[0]=(OpenMXGpuTaskView){pairs,0,sizeof(pairs[0])};
    task.view[1]=(OpenMXGpuTaskView){a,0,0,oa,sizeof(double)};
    task.view[2]=(OpenMXGpuTaskView){b,0,0,ob,sizeof(double)};
    task.view[3]=(OpenMXGpuTaskView){weights,0,0,ow,sizeof(double)};
    task.view[4]=(OpenMXGpuTaskView){output,0,stride*sizeof(double)};
    TaskWork work={0};work.pool.count=1;work.pool.items=N;work.task=&task;
    assert(task_worker(&work,0,OPENMX_GPU_WORKER_PREPARE));
    assert(task_worker(&work,0,OPENMX_GPU_WORKER_EXECUTE));
    assert(task_worker(&work,0,OPENMX_GPU_WORKER_COMMIT));
    verify((double*)work.result[0],expected,count);
    if(spins==4 && lengths[shape]==16387) {
      double times[11];
      for(int r=0;r<11;r++) {double start=omp_get_wtime();assert(task_worker(&work,0,OPENMX_GPU_WORKER_EXECUTE));openmx_target_wait();times[r]=1000*(omp_get_wtime()-start);}
      qsort(times,11,sizeof(double),compare_double);
      printf("BENCH capacity-Hamiltonian pairs=%d points=%d spins=%d median_ms=%.6f\n",N,lengths[shape],spins,times[5]);
    }
    assert(task_worker(&work,0,OPENMX_GPU_WORKER_RELEASE));free(work.result[0]);
    task_cpu(&task);verify(output,expected,count);
    free(expected);free(output);free(weights);free(b);free(a);
  }
  puts("PASS: actual HIP capacity Hamiltonian and CPU fallback, ragged/empty/padded/spin cases with a long-double oracle");
}
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", type=Path, default=Path(os.environ.get("ROCM", "/opt/rocm-7.2.0")))
    parser.add_argument("--arch", default=os.environ.get("GPU_ARCH", "gfx1200"))
    parser.add_argument("--cc", default=os.environ.get("CC"))
    parser.add_argument("--mpicc", default=os.environ.get("MPICC", "mpicc"))
    args = parser.parse_args()
    compiler = shlex.split(args.cc) if args.cc else [str(args.rocm / "llvm/bin/clang")]
    mpi_flags = shlex.split(subprocess.check_output(
        shlex.split(args.mpicc) + ["--showme:compile"], text=True))
    env = os.environ.copy()
    env["OMP_TARGET_OFFLOAD"] = "MANDATORY"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("HSA_OVERRIDE_CPU_AFFINITY_DEBUG", "0")
    env["LD_LIBRARY_PATH"] = str(args.rocm / "llvm/lib") + ":" + str(args.rocm / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    with tempfile.TemporaryDirectory(prefix="openmx-hip-capacity-h-") as directory:
        folder = Path(directory)
        source = folder / "check.c"
        source.write_text(FIXTURE.replace("@SETH_CAPACITY@", function(
            (ROOT / "source/backends/hip/src/Set_Hamiltonian.c").read_text(), "SetH_capacity_kernel")))
        command = compiler + ["-x", "c", "-O3", "-std=c11", "-ffp-contract=off", "-fopenmp",
            "--offload-arch=" + args.arch, "-D__HIP_PLATFORM_AMD__",
            "-ffunction-sections", "-fdata-sections", "-I" + str(ROOT / "source"),
            "-I" + str(ROOT / "source/backends/hip/include"),
            "-I" + str(args.rocm / "include"), *mpi_flags, str(source),
            "-L" + str(args.rocm / "lib"), "-lamdhip64", "-lm",
            "-Wl,--gc-sections,-z,nostart-stop-gc", "-o", str(folder / "check")]
        subprocess.run(command, env=env, check=True, timeout=120)
        subprocess.run([str(folder / "check")], env=env, check=True, timeout=120)


if __name__ == "__main__":
    main()
