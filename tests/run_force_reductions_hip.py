#!/usr/bin/env python3
"""Check the production HIP atom-force reduction on an actual AMD GPU.

Uses the real ragged task packer and long-double references, including empty
atoms, unequal row lengths, cancellation and CPU recovery. --source permits
paired performance probes against a frozen production source.
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
#include "gpu_task_native_probe.h"
@FORCE_REDUCTION@
static int compare_double(const void *a,const void *b) {
  double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);
}
static void verify(const double *actual,const long double *expected,size_t n) {
  for(size_t i=0;i<n;i++) {
    long double error=fabsl((long double)actual[i]-expected[i]);
    if(!isfinite(actual[i]) || error>5e-12L*fmaxl(1,fabsl(expected[i]))) {
      fprintf(stderr,"force[%zu] %.17g expected %.21Lg error %.4Lg\n",i,actual[i],expected[i],error);
      exit(1);
    }
  }
}
int main(void) {
  int devices=0,initial=1;
  assert(hipGetDeviceCount(&devices)==hipSuccess && devices>0);
  assert(hipSetDevice(0)==hipSuccess);omp_set_default_device(0);
#pragma omp target map(from:initial)
  initial=omp_is_initial_device();
  assert(!initial);
  const size_t lengths[]={0,1,33,257,8195,65539};
  for(size_t atoms=1;atoms<=17;atoms=(atoms==1?3:17)) {
    for(size_t shape=0;shape<sizeof(lengths)/sizeof(lengths[0]);shape++) {
      size_t *offsets=calloc(atoms+1,sizeof(size_t));
      for(size_t a=0;a<atoms;a++) {
        size_t n=lengths[shape];
        if(atoms>1 && a%3==1)n/=7;
        if(atoms>1 && a%3==2)n=0;
        offsets[a+1]=offsets[a]+n;
      }
      size_t total=offsets[atoms];double *input[4],*output[3],*ordered[3];long double *expected[3];
      for(int c=0;c<4;c++)input[c]=malloc((total?total:1)*sizeof(double));
      for(int c=0;c<3;c++){output[c]=malloc(atoms*sizeof(double));expected[c]=calloc(atoms,sizeof(long double));ordered[c]=calloc(atoms,sizeof(double));}
      for(size_t i=0;i<total;i++) {
        input[0][i]=(i%5?-.3:1.)+(i%17)*.01;
        for(int c=0;c<3;c++)input[c+1][i]=sin((i%101)*.071+c*.19)*(i%2?1:-1);
      }
      for(size_t a=0;a<atoms;a++)for(size_t i=offsets[a];i<offsets[a+1];i++)
        for(int c=0;c<3;c++){expected[c][a]+=(long double)input[0][i]*input[c+1][i];ordered[c][a]+=input[0][i]*input[c+1][i];}
      OpenMXGpuTask task={0};task.count=atoms;task.inputs=4;task.outputs=3;task.kernel=Force_Reduction_Kernel;
      for(int c=0;c<4;c++)task.view[c]=(OpenMXGpuTaskView){input[c],0,0,offsets,sizeof(double)};
      for(int c=0;c<3;c++)task.view[4+c]=(OpenMXGpuTaskView){output[c],0,sizeof(double)};
      void *work=openmx_test_task_create(&task);assert(work);
      assert(openmx_test_task_phase(work,OPENMX_GPU_WORKER_PREPARE));
      assert(openmx_test_task_phase(work,OPENMX_GPU_WORKER_EXECUTE));
      assert(openmx_test_task_phase(work,OPENMX_GPU_WORKER_COMMIT));
      task_commit(&task,0,atoms,openmx_test_task_result(work),task_bytes(&task,0,atoms,task.inputs,task.inputs+task.outputs));
      for(int c=0;c<3;c++)verify(output[c],expected[c],atoms);
      if(atoms==3 && lengths[shape]) {
        double times[11];
        for(int r=0;r<11;r++){double start=omp_get_wtime();assert(openmx_test_task_phase(work,OPENMX_GPU_WORKER_EXECUTE));openmx_target_wait();times[r]=1000*(omp_get_wtime()-start);}
        qsort(times,11,sizeof(double),compare_double);
        printf("BENCH atom-force-reduction atoms=%zu terms=%zu median_ms=%.6f\n",atoms,total,times[5]);
      }
      assert(openmx_test_task_phase(work,OPENMX_GPU_WORKER_RELEASE));openmx_test_task_destroy(work);
      for(int threads=1;threads<=4;threads*=2) {
        omp_set_num_threads(threads);openmx_gpu_task_cpu(&task);
        for(int c=0;c<3;c++) {
          verify(output[c],expected[c],atoms);
          for(size_t a=0;a<atoms;a++)assert(output[c][a]==ordered[c][a]);
        }
      }
      if(atoms==17 && lengths[shape]==65539) {
        double times[11];
        for(int r=0;r<11;r++){double start=omp_get_wtime();openmx_gpu_task_cpu(&task);times[r]=1000*(omp_get_wtime()-start);}
        qsort(times,11,sizeof(double),compare_double);
        printf("BENCH atom-force-cpu threads=4 atoms=%zu terms=%zu median_ms=%.6f\n",atoms,total,times[5]);
      }
      omp_set_num_threads(1);
      for(int c=0;c<3;c++){free(ordered[c]);free(expected[c]);free(output[c]);}
      for(int c=0;c<4;c++)free(input[c]);free(offsets);
    }
    if(atoms==17)break;
  }
  puts("PASS: actual HIP atom-force reduction and CPU fallback, empty/ragged/tail/cancellation cases with a long-double oracle");
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", type=Path, default=Path(os.environ.get("ROCM", "/opt/rocm-7.2.0")))
    parser.add_argument("--arch", default=os.environ.get("GPU_ARCH", "gfx1200"))
    parser.add_argument("--cc", default=os.environ.get("CC"))
    parser.add_argument("--mpicc", default=os.environ.get("MPICC", "mpicc"))
    parser.add_argument("--source", type=Path, default=ROOT / "source/backends/hip/src/Force.c")
    args = parser.parse_args()
    compiler = shlex.split(args.cc) if args.cc else [str(args.rocm / "llvm/bin/clang")]
    mpi_flags = shlex.split(subprocess.check_output(shlex.split(args.mpicc) + ["--showme:compile"], text=True))
    env = os.environ.copy()
    env["OMP_TARGET_OFFLOAD"] = "MANDATORY"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("HSA_OVERRIDE_CPU_AFFINITY_DEBUG", "0")
    env["LD_LIBRARY_PATH"] = str(args.rocm / "llvm/lib") + ":" + str(args.rocm / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    with tempfile.TemporaryDirectory(prefix="openmx-hip-force-reduction-") as directory:
        folder = Path(directory)
        source = folder / "check.c"
        source.write_text(FIXTURE.replace("@FORCE_REDUCTION@", function(args.source.read_text(), "Force_Reduction_Kernel")))
        probe=folder / "task_probe.o"
        subprocess.run([str(args.rocm / "llvm/bin/clang++"),"-std=c++17","-O3",
            "-D__HIP_PLATFORM_AMD__","-ffunction-sections","-fdata-sections",
            "-I"+str(ROOT / "source"),"-I"+str(ROOT / "source/backends/hip/include"),
            "-I"+str(args.rocm / "include"),*mpi_flags,
            "-c",str(ROOT / "tests/gpu_task_native_probe.cpp"),"-o",str(probe)],
            env=env,check=True,timeout=120)
        host_objects=[]
        for module in ['cpu','packet']:
            output=folder / ('task_'+module+'.o')
            subprocess.run([str(args.rocm / "llvm/bin/clang++"),"-std=c++17","-O3",
                "-I"+str(ROOT / "source"),*mpi_flags,
                "-c",str(ROOT / ('source/openmx_gpu_task_'+module+'.cpp')),"-o",str(output)],
                env=env,check=True,timeout=120)
            host_objects.append(str(output))
        command = compiler + ["-x", "c", "-O3", "-std=c11", "-ffp-contract=off", "-fopenmp",
            "--offload-arch=" + args.arch, "-D__HIP_PLATFORM_AMD__", "-ffunction-sections", "-fdata-sections",
            "-I" + str(ROOT / "source"), "-I" + str(ROOT / "tests"), "-I" + str(ROOT / "source/backends/hip/include"),
            "-I" + str(args.rocm / "include"), *mpi_flags, str(source), "-L" + str(args.rocm / "lib"),
            "-x", "none", str(probe), *host_objects, "-lstdc++", "-lamdhip64", "-lm", "-Wl,--gc-sections,-z,nostart-stop-gc", "-o", str(folder / "check")]
        subprocess.run(command, env=env, check=True, timeout=120)
        subprocess.run([str(folder / "check")], env=env, check=True, timeout=120)


if __name__ == "__main__":
    main()
