#!/usr/bin/env python3
"""Check NC MI300A policy, cleanup, and GEMM diagnostics without a GPU.

Extracts production functions and uses device/BLAS stubs. Separate processes
exercise cached defaults and environment overrides for MI300A and other GPUs.
The actual drop/release routines verify retained buffers still go back under
pressure and at solver teardown. Native-call counters exercise successful
native GEMMs (including workspace fallbacks) and GEMMul8 diagnostics.
This test checks dispatch/lifetime logic, not GPU numerics or performance.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    match = re.search(r"^static [^;{}]*\b" + re.escape(name) + r"\([^;]*?\)\s*\{", source, re.M)
    if match is None:
        raise ValueError(f"Function not found: {name}")
    pos, depth = source.index("{", match.start()) + 1, 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[match.start():pos]


def structure(source, name):
    end = source.index("} " + name + ";") + len("} " + name + ";")
    return source[source.rfind("typedef struct", 0, end):end]


PREFIX = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(2); } } while (0)
typedef void *hipblasHandle_t;
typedef int hipblasStatus_t;
typedef int hipblasOperation_t;
typedef struct { double r, i; } dcomplex;
#define HIPBLAS_STATUS_SUCCESS 0
#define wait_hipfunc(x) CHECK((x)==0)
#define NCPROF_RELEASE 6
static int mi300a, z_enabled, frees, parks, trims, releases, destroys, refine_releases;
static int native_mode, bridge_failure, bridge_calls, native_calls;
static long long native_count;
static int ClusterNonCol_refine_state;
static int openmx_gpu_is_mi300a(void) { return mi300a; }
static int openmx_gemmul8ZgemmEnabled(void) { return z_enabled; }
static long long openmx_gemmul8NativeCalls(int complex) { (void)complex; return native_count; }
static int hipFree(void *p) { CHECK(p != NULL); frees++; return 0; }
static int hipblasDestroy(void *p) { CHECK(p != NULL); destroys++; return 0; }
static void openmx_eigen_refine_state_release(int *p) { (void)p; refine_releases++; }
static int ClusterNonCol_ProfileEnabled(void) { return 0; }
static void ClusterNonCol_ProfileLap(int phase,double *t0) { (void)phase; (void)t0; CHECK(0); }
static void ClusterNonCol_DeviceSolveParkBasis(void) { parks++; }
static int ClusterNonCol_GemmWorkspaceTurnRelease(void) { return 0; }
static void openmx_gemmul8ReleaseWorkspaces(void) { releases++; }
static void openmx_gemmul8TrimWorkspaces(void) { trims++; }
static int ClusterNonCol_MpiRankForLog(void) { return 0; }
static int ClusterNonCol_GpuVerbose(void) { return 1; }
static const char *ClusterNonCol_HipblasStatusName(int s) { (void)s; return "stub error"; }
#define GEMM_ARGS hipblasHandle_t h,hipblasOperation_t ta,hipblasOperation_t tb,int m,int n,int k,const double *a,const double *A,int lda,const double *B,int ldb,const double *b,double *C,int ldc
static int openmx_gemmul8Dgemm(GEMM_ARGS) {
    (void)h;(void)ta;(void)tb;(void)m;(void)n;(void)k;(void)a;(void)A;(void)lda;(void)B;(void)ldb;(void)b;(void)C;(void)ldc;
    bridge_calls++; native_count += native_mode; return bridge_failure;
}
#define openmx_gemmul8DgemmSolve openmx_gemmul8Dgemm
static int hipblasDgemm(GEMM_ARGS) {
    (void)h;(void)ta;(void)tb;(void)m;(void)n;(void)k;(void)a;(void)A;(void)lda;(void)B;(void)ldb;(void)b;(void)C;(void)ldc;
    native_calls++; return 0;
}
'''


MAIN = r'''
int main(int argc,char **argv) {
    CHECK(argc==7);
    mi300a=atoi(argv[1]); z_enabled=atoi(argv[2]);
    int keep=atoi(argv[3]), dm=atoi(argv[4]); native_mode=atoi(argv[5]); bridge_failure=atoi(argv[6]);
    CHECK(ClusterNonCol_DeviceSolveKeep()==keep);
    CHECK(ClusterNonCol_DmByGemm()==dm);
    ClusterNonCol_dsolve.d_A=ClusterNonCol_dsolve.d_S=ClusterNonCol_dsolve.d_arena=(void*)1;
    ClusterNonCol_dsolve.d_x=ClusterNonCol_dsolve.d_work=(void*)1;
    ClusterNonCol_dsolve.d_w=(double*)1; ClusterNonCol_dsolve.d_info=(int32_t*)1;
    ClusterNonCol_dsolve.hipblas=(void*)1; ClusterNonCol_dsolve.n=2; ClusterNonCol_dsolve.n2=4;
    ClusterNonCol_dsolve.S_valid=ClusterNonCol_dsolve.x_on_device=1;
    ClusterNonCol_DeviceSolveEndStep();
    CHECK(frees==(keep ? 0 : 5) && parks==(keep ? 0 : 1) && trims==1 && releases==0);
    ClusterNonCol_DeviceSolveGiveBack();
    CHECK(frees==5 && parks==1);
    CHECK(ClusterNonCol_dsolve.d_A==NULL && ClusterNonCol_dsolve.d_S==NULL &&
          ClusterNonCol_dsolve.d_arena==NULL && ClusterNonCol_dsolve.d_x==NULL &&
          ClusterNonCol_dsolve.d_work==NULL && ClusterNonCol_dsolve.S_valid==0 &&
          ClusterNonCol_dsolve.x_on_device==0);
    ClusterNonCol_DeviceSolveRelease();
    CHECK(frees==7 && destroys==1 && refine_releases==1);
    CHECK(ClusterNonCol_dsolve.n==0 && ClusterNonCol_dsolve.n2==0 && ClusterNonCol_dsolve.hipblas==NULL);
    CHECK(ClusterNonCol_TryDeviceDgemm(NULL,0,0,2,2,2,NULL,NULL,NULL,"test")==0);
    CHECK(ClusterNonCol_TryDeviceDgemmSolve(NULL,0,0,2,2,2,NULL,NULL,NULL,"test solve")==0);
    CHECK(bridge_calls==2 && native_calls==(bridge_failure ? 2 : 0));
    puts("PASS"); return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    source = (ROOT / "source/Cluster_DFT_NonCol.c").read_text()
    names = ("ClusterNonCol_DmByGemm", "ClusterNonCol_DeviceSolveKeep", "ClusterNonCol_DeviceFree",
             "ClusterNonCol_DeviceSolveDrop", "ClusterNonCol_DeviceSolveRelease", "ClusterNonCol_DeviceSolveEndStep",
             "ClusterNonCol_DeviceSolveGiveBack", "ClusterNonCol_LogGemmRetry", "ClusterNonCol_LogGemmBackendOnce",
             "ClusterNonCol_TryDeviceDgemm", "ClusterNonCol_TryDeviceDgemmSolve")
    harness = "\n".join([PREFIX, structure(source, "ClusterNonColDeviceSolve"),
                         "static ClusterNonColDeviceSolve ClusterNonCol_dsolve;",
                         *[function(source, name) for name in names], MAIN])
    with tempfile.TemporaryDirectory(prefix="openmx-nc-mi300a-policy-") as tmp:
        src, exe = Path(tmp)/"check.c", Path(tmp)/"check"
        src.write_text(harness)
        flags = ["-std=c11", "-O2", "-Wall", "-Wextra"]
        if args.sanitize:
            flags += ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
        subprocess.run([*shlex.split(args.cc), *flags, str(src), "-o", str(exe)], check=True)
        checks = 0
        for mi300a in (0, 1):
            for z_enabled in (0, 1):
                for override in (None, "0", "1"):
                    env = dict(os.environ)
                    for key in ("OPENMX_CLUSTER_NC_DEVICE_KEEP", "OPENMX_CLUSTER_GPU_DM_GEMM"):
                        env.pop(key, None)
                        if override is not None:
                            env[key] = override
                    keep = mi300a if override is None else int(override)
                    dm = int(mi300a or z_enabled) if override is None else int(override)
                    for native_mode, bridge_failure in ((0,0), (1,0), (0,1), (1,1)):
                        run = subprocess.run([str(exe), *map(str,(mi300a,z_enabled,keep,dm,native_mode,bridge_failure))],
                                             env=env, text=True, capture_output=True, check=True)
                        backend = "native hipBLAS" if native_mode or bridge_failure else "GEMMul8"
                        assert f"using {backend} for DGEMM" in run.stderr, run.stderr
                        assert ("GPU GEMM bridge failed" in run.stderr) == bool(bridge_failure), run.stderr
                        checks += 1
    print(f"PASS: {checks} NC policy/lifetime/GEMM diagnostic cases; non-MI300A defaults and explicit overrides preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
