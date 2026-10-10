#!/usr/bin/env python3
"""Run production refinement validity reductions on an actual NVIDIA GPU.

Requires NVIDIA HPC SDK nvc with OpenACC and CUDA. NVHPC_ROOT, CUDA_HOME,
NVHPC_CUDA_HOME, NVC and GPU_ARCH select the installation/device architecture.
The default architecture is sm_120 (RTX 5080); no eigensolver is simulated.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def function(source, name):
    match = re.search(r"^static int " + re.escape(name) + r"\([^;]*?\)\s*\{",
                      source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production function {name}")
    brace = source.index("{", match.start())
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


PREFIX = r'''
#include <float.h>
#include <math.h>
#include <openacc.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#define CHECK(test) do { if (!(test)) { \
    fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#test); exit(2); } } while (0)
#define CUDA(call) do { cudaError_t status=(call); if(status!=cudaSuccess) { \
    fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(status)); exit(3); } } while(0)
'''

HARNESS = r'''
enum { N=4, KC=3, MAX_COUNT=2*N*KC };
static int checks;
static void maxima(int complex_matrix, double *g, double *s, double *dg, double *ds,
                   int expected, double expected_s, double expected_r) {
    double max_s=-1,max_r=-1;
    int count=(complex_matrix?2:1)*N*KC;
    acc_memcpy_to_device(dg,g,(size_t)count*sizeof(double));
    acc_memcpy_to_device(ds,s,(size_t)count*sizeof(double));
    int finite=refine_maxima(complex_matrix,dg,ds,N,KC,&max_s,&max_r);
    CUDA(cudaDeviceSynchronize());
    CHECK(finite==expected);
    if(expected) {
        CHECK(fabs(max_s-expected_s)<1e-14);
        CHECK(fabs(max_r-expected_r)<1e-14);
    }
    checks++;
}
static void result(int width,double *x,double *lam,double *dx,double *dl,int expected) {
    acc_memcpy_to_device(dx,x,(size_t)width*N*KC*sizeof(double));
    acc_memcpy_to_device(dl,lam,KC*sizeof(double));
    int finite=refine_result_finite(width,dx,dl,N,KC);
    CUDA(cudaDeviceSynchronize());
    CHECK(finite==expected); checks++;
}
int main(void) {
    int device_count=acc_get_num_devices(acc_device_nvidia),device;
    struct cudaDeviceProp properties;
    CHECK(device_count>0);
    acc_init(acc_device_nvidia);
    CHECK(acc_get_device_type()==acc_device_nvidia);
    CUDA(cudaGetDevice(&device)); CUDA(cudaGetDeviceProperties(&properties,device));
    printf("GPU: %s (sm_%d%d)\n",properties.name,properties.major,properties.minor);
    double *dg=acc_malloc(MAX_COUNT*sizeof(double));
    double *ds=acc_malloc(MAX_COUNT*sizeof(double));
    double *dl=acc_malloc(KC*sizeof(double));
    CHECK(dg && ds && dl);
    const double invalid[3]={NAN,INFINITY,-INFINITY};
    for(int complex_matrix=0;complex_matrix<=1;complex_matrix++) {
        int width=complex_matrix?2:1,count=width*N*KC;
        double g[MAX_COUNT]={0},s[MAX_COUNT]={0},lam[KC]={1,2,3};
        for(int j=0;j<KC;j++) { g[width*(j+j*N)]=1; s[width*(j+j*N)]=lam[j]; }
        maxima(complex_matrix,g,s,dg,ds,1,0,0);
        /* A finite nonzero reduction checks both its value and validity. */
        g[width]=0.125; s[width]=0.25;
        maxima(complex_matrix,g,s,dg,ds,1,0.25,0.125);
        g[width]=s[width]=0;
        for(int matrix=0;matrix<2;matrix++) for(int i=0;i<count;i++) {
            double *host=matrix?s:g,saved=host[i];
            for(int fault=0;fault<3;fault++) {
                host[i]=invalid[fault];
                maxima(complex_matrix,g,s,dg,ds,0,0,0);
            }
            host[i]=saved;
        }
        for(int j=0;j<KC;j++) {
            int at=width*(j+j*N);
            g[at]=0; maxima(complex_matrix,g,s,dg,ds,0,0,0);
            g[at]=-1; maxima(complex_matrix,g,s,dg,ds,0,0,0);
            g[at]=1;
        }
        if(complex_matrix) {
            /* Finite components whose complex magnitude overflows. */
            s[width]=s[width+1]=DBL_MAX;
            maxima(complex_matrix,g,s,dg,ds,0,0,0);
            s[width]=s[width+1]=0;
        }
        result(width,g,lam,dg,dl,1);
        for(int i=0;i<count;i++) {
            double saved=g[i];
            for(int fault=0;fault<3;fault++) {
                g[i]=invalid[fault]; result(width,g,lam,dg,dl,0);
            }
            g[i]=saved;
        }
        for(int j=0;j<KC;j++) {
            double saved=lam[j];
            for(int fault=0;fault<3;fault++) {
                lam[j]=invalid[fault]; result(width,g,lam,dg,dl,0);
            }
            lam[j]=saved;
        }
        maxima(complex_matrix,g,s,dg,ds,1,0,0);
        result(width,g,lam,dg,dl,1);
    }
    acc_free(dg); acc_free(ds); acc_free(dl);
    CUDA(cudaDeviceSynchronize());
    printf("PASS: %d actual OpenACC real/complex refinement reduction checks\n",checks);
    return 0;
}
'''


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / "source/eigen_refine_gpu.c").read_text()
    code = PREFIX + "\n".join(function(source, name) for name in
                             ("refine_maxima", "refine_result_finite")) + HARNESS
    env = os.environ.copy()
    sdk = Path(env.get("NVHPC_ROOT", "/opt/nvidia/hpc_sdk/Linux_x86_64/26.9"))
    cuda = Path(env.get("NVHPC_CUDA_HOME", env.get("CUDA_HOME", "/usr/local/cuda-13.4")))
    arch = env.get("GPU_ARCH", "120")
    if not arch.isdigit():
        raise ValueError("GPU_ARCH must be numeric, e.g. 120")
    for name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH"):
        env.pop(name, None)
    env["NVHPC_CUDA_HOME"] = str(cuda)
    nvc = shlex.split(env.get("NVC", str(sdk / "compilers/bin/nvc")))
    with tempfile.TemporaryDirectory(prefix="openmx-refine-guards-gpu-") as tmp:
        cfile, exe = Path(tmp) / "check.c", Path(tmp) / "check"
        cfile.write_text(code)
        subprocess.run(nvc + ["-std=c11", "-O3", "-acc", f"-gpu=cc{arch}",
            "-I" + str(cuda / "include"), str(cfile), "-L" + str(cuda / "lib64"),
            "-Wl,-rpath," + str(cuda / "lib64"), "-lcudart", "-lm", "-o", str(exe)],
            env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()
