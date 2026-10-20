#!/usr/bin/env python3
"""Run the production HIP Gram reductions against a long-double oracle.

The resident-history and host-backed worker implementations are compiled
unchanged from their production sources. A mandatory offload probe prevents
host fallback from being mistaken for a successful device test. Requires an
AMD GPU, ROCm, and the MPI headers used by the application build.
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
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "openmx_target.h"
#include "openmx_gpu_vector.c"

typedef struct {
  unsigned char *base;
  size_t L, o_ring, o_w, o_pair;
} MixH_GpuShard;
/* The generator inserts the production MixH_GramPart definition here. */
@MIXH_GRAM@

static int order_double(const void *a,const void *b) {
  double x=*(const double*)a,y=*(const double*)b;
  return (x>y)-(x<y);
}
static void verify(const char *label,const double *actual,const long double *expected,int count) {
  for(int p=0;p<count;p++) {
    long double error=fabsl((long double)actual[p]-expected[p]);
    if(!isfinite(actual[p]) || error>5e-12L*fmaxl(1,fabsl(expected[p]))) {
      fprintf(stderr,"%s[%d]: %.17g expected %.21Lg, error %.4Lg\n",label,p,actual[p],expected[p],error);
      exit(1);
    }
  }
}
int main(void) {
  int devices=0,initial=1;
  assert(hipGetDeviceCount(&devices)==hipSuccess && devices>0);
  assert(hipSetDevice(0)==hipSuccess);
  omp_set_default_device(0);
#pragma omp target map(from:initial)
  initial=omp_is_initial_device();
  assert(!initial);
  const size_t sizes[]={0,1,33,257,8195,65539};
  for(int c=1;c<=8;c*=2) for(int shape=0;shape<6;shape++) {
    size_t n=sizes[shape],ld=n+7;
    int pairs=c*(c+1)/2;
    double *data=malloc((size_t)c*ld*sizeof(double));
    double *weight=malloc((n?n:1)*sizeof(double));
    double *output=malloc((n>(size_t)pairs?n:pairs)*sizeof(double));
    long double *expected=malloc((n>(size_t)pairs?n:pairs)*sizeof(long double));
    int left[36],right[36],pm[36],pn[36],slot[8];
    for(int a=0;a<c;a++) {
      slot[a]=(3*a+1)%c;
      for(size_t i=0;i<ld;i++) data[(size_t)a*ld+i]=sin((i%101)*.071+a*.19)*(i%2?1:-1);
    }
    for(size_t i=0;i<n;i++) weight[i]=(i%5==0?-.3:1.0)+(i%17)*.01;
    int p=0;
    for(int a=0;a<c;a++)for(int b=0;b<=a;b++) {
      pm[p]=a;pn[p]=b;left[p]=slot[a];right[p]=slot[b];expected[p]=0;
      for(size_t i=0;i<n;i++) expected[p]+=(long double)weight[i]*data[(size_t)left[p]*ld+i]*data[(size_t)right[p]*ld+i];
      ++p;
    }
    VectorWork w={0};w.workers.count=1;w.workers.items=n;
    w.job=(OpenMXGpuVector){.n=n,.ld=ld,.mode=0,.columns=c,.pairs=pairs,.data=data,.weight=weight,.left=left,.right=right,.output=output};
    assert(vector_worker(&w,0,OPENMX_GPU_WORKER_PREPARE));
    assert(vector_worker(&w,0,OPENMX_GPU_WORKER_EXECUTE));
    assert(vector_worker(&w,0,OPENMX_GPU_WORKER_COMMIT));
    verify("host-backed",w.partial[0],expected,pairs);
    MixH_GpuShard shard={.base=(unsigned char*)w.arena[0],.L=n,.o_ring=0,.o_w=(size_t)c*n*sizeof(double),.o_pair=((size_t)c+1)*n*sizeof(double)};
    MixH_GramPart(&shard,c,pairs,pm,pn,slot,output);
    verify("resident",output,expected,pairs);
    if(n==65539 && c==8) {
      double timings[11];
      for(int kind=0;kind<2;kind++) {
        for(int r=0;r<11;r++) {
          double start=omp_get_wtime();
          if(kind)MixH_GramPart(&shard,c,pairs,pm,pn,slot,output);
          else { assert(vector_worker(&w,0,OPENMX_GPU_WORKER_EXECUTE));openmx_target_wait(); }
          timings[r]=1000*(omp_get_wtime()-start);
        }
        qsort(timings,11,sizeof(double),order_double);
        printf("BENCH %s n=%zu pairs=%d median_ms=%.6f\n",kind?"resident":"host-backed-kernel",n,pairs,timings[5]);
      }
    }
    assert(vector_worker(&w,0,OPENMX_GPU_WORKER_RELEASE));free(w.partial[0]);
    free(expected);free(output);free(weight);free(data);
  }
  puts("PASS: actual HIP reductions, empty/tail/padded/permuted cases and long-double oracle");
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
    with tempfile.TemporaryDirectory(prefix="openmx-hip-gram-") as directory:
        folder = Path(directory)
        source = folder / "check.c"
        source.write_text(FIXTURE.replace("@MIXH_GRAM@", function(
            (ROOT / "source/Mixing_H.c").read_text(), "MixH_GramPart")))
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
