#!/usr/bin/env python3
"""Exercise NC band MI300A workspace policy without a GPU.

Production functions are extracted and compiled against tracking HIP/solver
stubs. Check native full-spectrum byte queries and their device/dimension
cache, failed-query retries, independent workspace release overrides, legacy
Radeon/APU behavior, unchanged dense entry routing and refined-step cleanup.
This verifies dispatch/accounting, not GPU numerical accuracy or performance.
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
    match = re.search(r"^(?:static )?(?:int|void|size_t)\s+" + re.escape(name) + r"\([^;{}]*\)\s*\{", source, re.M)
    if match is None:
        raise ValueError(f"Missing production function {name}")
    pos, depth = source.index("{", match.start()) + 1, 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[match.start():pos]


PREFIX = r'''
#define _POSIX_C_SOURCE 200809L
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(2); } } while (0)
#define wait_hipfunc(x) CHECK((x)==0)
typedef struct { double r,i; } dcomplex;
typedef dcomplex hipDoubleComplex;
typedef void *hipsolverDnHandle_t;
typedef void *hipStream_t;
typedef int hipError_t,hipsolverStatus_t,hipsolverEigMode_t,hipblasFillMode_t,hipsolverEigRange_t;
enum { hipSuccess=0,HIPSOLVER_STATUS_SUCCESS=0,hipStreamNonBlocking=1,
       HIPSOLVER_EIG_MODE_VECTOR=1,HIPBLAS_FILL_MODE_LOWER=2,
       HIPSOLVER_EIG_RANGE_ALL=3,HIPSOLVER_EIG_RANGE_I=4,HIP_C_64F=5,HIP_R_64F=6 };
static int gpu_kind,device,fail_at,native_queries,legacy_queries,handles,streams,policy_calls;
static int native_releases,magma_releases,policy_entries,foreign_entries,dense_destroys,backtransforms;
static int native_lwork=137;
static const int mpi_comm_level1=1;
static void BandNonCol_AbortWithMessage(const char *msg) { fprintf(stderr,"%s\n",msg); exit(77); }
static int openmx_gpu_is_mi300a(void) { return gpu_kind==1; }
static int openmx_gpu_is_apu(void) { return gpu_kind!=0; }
static int openmx_gpu_eigensolver_use_hipsolver(void) {
    const char *env=getenv("OPENMX_GPU_EIGENSOLVER");
    if (env && !strcmp(env,"hipsolver")) return 1;
    if (env && !strcmp(env,"magma")) return 0;
    return gpu_kind==1;
}
static int openmx_gpu_eigen_backend(int n,int maxn,int complex) {
    CHECK(n>0 && maxn>0 && maxn<=n && complex==1); policy_calls++;
    return openmx_gpu_eigensolver_use_hipsolver();
}
static int openmx_hipsolver_release_workspace(void) { native_releases++; return 0; }
static int openmx_magma_release_z_workspace(void) { magma_releases++; return 0; }
static int hipGetDevice(int *value) { *value=device; return fail_at==1; }
static int hipStreamCreateWithFlags(void **s,int flags) {
    CHECK(flags==hipStreamNonBlocking); if(fail_at==2)return 1;
    *s=(void*)1; streams++; return 0;
}
static int hipStreamDestroy(void *s) { CHECK(s!=NULL); streams--; return 0; }
static int hipsolverDnCreate(void **h) {
    if(fail_at==3)return 1;
    *h=(void*)1; handles++; return 0;
}
static int hipsolverDnSetStream(void *h,void *s) { CHECK(h && s); return fail_at==4; }
static int hipsolverDnDestroy(void *h) { CHECK(h); handles--; return 0; }
static int hipsolverDnZheevd_bufferSize(void *h,int job,int uplo,int n,void *a,int lda,double *w,int *lwork) {
    CHECK(h && job==HIPSOLVER_EIG_MODE_VECTOR && uplo==HIPBLAS_FILL_MODE_LOWER);
    CHECK(n>0 && lda==n && !a && !w); native_queries++;
    *lwork=fail_at==6 ? 0 : fail_at==7 ? -1 : native_lwork;
    return fail_at==5;
}
static int hipsolverDnXsyevdx_bufferSize(void *h,void *params,int job,int range,int uplo,int n,
    int atype,void *a,int lda,double *vl,double *vu,int64_t il,int64_t iu,int64_t *meig,
    int wtype,double *w,int ctype,size_t *db,size_t *hb) {
    CHECK(h && !params && job==HIPSOLVER_EIG_MODE_VECTOR && uplo==HIPBLAS_FILL_MODE_LOWER);
    CHECK(range==(n==iu ? HIPSOLVER_EIG_RANGE_ALL : HIPSOLVER_EIG_RANGE_I));
    CHECK(atype==HIP_C_64F && ctype==HIP_C_64F && wtype==HIP_R_64F);
    CHECK(!a && !w && lda==n && vl && vu && il==1 && iu>0 && meig);
    *db=333; *hb=0; legacy_queries++; return fail_at==5;
}
static int openmx_magma_zheevdx_gpu(int n,int maxn,void *a,double *w,int *mout) {
    CHECK(n>=maxn && a); policy_entries++; *mout=maxn;
    for(int i=0;i<maxn;i++)w[i]=i+0.5;
    return 0;
}
static int openmx_hipsolver_zheevd_gpu(int n,int maxn,void *a,double *w,int *mout) {
    CHECK(n>=maxn && a); foreign_entries++; *mout=maxn;
    for(int i=0;i<maxn;i++)w[i]=i+0.5;
    return 0;
}
static int BandNonCol_SolveTimingEnabled(void) { return 0; }
static int hipDeviceSynchronize(void) { return 0; }
static void dtime(double *v) { *v=0; }
static void MPI_Abort(int comm,int code) { (void)comm; (void)code; exit(78); }
typedef struct { dcomplex *d_H2; } BandNonColDenseGpuWorkspace;
typedef struct { int unused; } BandNonColRootDenseWorkspace;
static BandNonColDenseGpuWorkspace BandNonCol_dense_gpu_workspace;
static int BandNonCol_DenseBufferCacheEnabled(void) { return 0; }
static void BandNonCol_DenseGpu_Destroy(void) { dense_destroys++; }
static void BandNonCol_RootDenseBacktransform_HIP(int n,int n2,int maxn,int copy,
    BandNonColRootDenseWorkspace *w,dcomplex *panel) {
    (void)n;(void)n2;(void)maxn;(void)copy;(void)w;(void)panel;backtransforms++;
}
'''

MAIN = r'''
static int release_value(const char *name) {
    const char *value=getenv(name); return value ? atoi(value)!=0 : gpu_kind!=0;
}
static void policy_test(void) {
    int native=gpu_kind==1 && openmx_gpu_eigensolver_use_hipsolver();
    const char *env=getenv("OPENMX_NC_HAMILTONIAN_EIGENSOLVER");
    int route=openmx_gpu_eigensolver_use_hipsolver() && env && !strcasecmp(env,"hipsolver");
    int release_h=release_value("OPENMX_NC_HIPSOLVER_TURN_RELEASE");
    int release_m=release_value("OPENMX_NC_MAGMA_TURN_RELEASE");
    CHECK(BandNonCol_HamiltonianUseHipSolver()==route);
    CHECK(BandNonCol_MI300ANativePolicy(912,432)==native);
    if(gpu_kind!=1)CHECK(policy_calls==0);
    BandNonCol_ReleaseSolverTurnWorkspaces(912,432,route);
    CHECK(native_releases==(native || route ? release_h : 0));
    CHECK(magma_releases==(native || !route ? release_m : 0));
    native_releases=magma_releases=0;
    BandNonCol_ReleaseOverlapWorkspace(456);
    CHECK(native_releases==(openmx_gpu_eigensolver_use_hipsolver() ? release_h : 0));
    CHECK(magma_releases==(native ? release_m : 0));
    native_releases=magma_releases=0;
    dcomplex matrix[4]; double ko[3]={0},ev[3]={0}; double *kpoints[1]={ev}; double **spin[1]={kpoints};
    BandNonCol_dense_gpu_workspace.d_H2=matrix;
    BandNonCol_RootDenseCompleteOneK(1,2,2,0,0,ko,spin,0,0,0,NULL,NULL);
    CHECK(policy_entries==!route && foreign_entries==route && dense_destroys==1 && !backtransforms);
    CHECK(ev[1]==0.5 && ev[2]==1.5);
    CHECK(native_releases==(native || route ? release_h : 0));
    CHECK(magma_releases==(native || !route ? release_m : 0));
    native_releases=magma_releases=0;
    BandNonCol_RootDenseCompleteOneK(1,2,2,0,1,ko,spin,0,0,0,NULL,NULL);
    CHECK(!native_releases && !magma_releases && policy_entries+foreign_entries==1);
    CHECK(BandNonCol_QueryGpuSolverWorkBytes(912,432)==(native ? 137*sizeof(hipDoubleComplex) : 333));
    CHECK(native_queries==native && legacy_queries==!native && handles==0 && streams==0);
}
static void query_test(void) {
    gpu_kind=1; unsetenv("OPENMX_GPU_EIGENSOLVER");
    CHECK(BandNonCol_QueryGpuSolverWorkBytes(912,432)==137*sizeof(hipDoubleComplex));
    CHECK(BandNonCol_QueryGpuSolverWorkBytes(912,912)==137*sizeof(hipDoubleComplex));
    CHECK(native_queries==1); /* full/partial use identical full D&C scratch */
    CHECK(BandNonCol_QueryGpuSolverWorkBytes(456,456)==137*sizeof(hipDoubleComplex));
    CHECK(BandNonCol_QueryGpuSolverWorkBytes(912,100)==137*sizeof(hipDoubleComplex));
    CHECK(native_queries==2); /* retain overlap and H queries */
    setenv("OPENMX_GPU_EIGENSOLVER","magma",1);
    CHECK(BandNonCol_QueryGpuSolverWorkBytes(912,432)==333 && legacy_queries==1);
    unsetenv("OPENMX_GPU_EIGENSOLVER");
    device=1;
    CHECK(BandNonCol_QueryGpuSolverWorkBytes(912,432)==137*sizeof(hipDoubleComplex));
    CHECK(native_queries==3); /* a different device cannot reuse the query */
    for(int failure=1;failure<=7;failure++) {
        int n=1100+failure;
        fail_at=failure;
        CHECK(BandNonCol_QueryGpuSolverWorkBytes(n,n/2)==4*(size_t)n*n*sizeof(dcomplex));
        CHECK(handles==0 && streams==0);
        fail_at=0; native_lwork=200+failure;
        CHECK(BandNonCol_QueryGpuSolverWorkBytes(n,n/2)==(size_t)native_lwork*sizeof(hipDoubleComplex));
        CHECK(handles==0 && streams==0); /* failures are not cached */
    }
    native_lwork=INT_MAX;
    CHECK(BandNonCol_QueryGpuSolverWorkBytes(2048,100)==(size_t)INT_MAX*sizeof(hipDoubleComplex));
    gpu_kind=0;
    CHECK(BandNonCol_QueryGpuSolverWorkBytes(2048,100)==333);
    CHECK(BandNonCol_QueryGpuSolverWorkBytes(2048,2048)==333 && legacy_queries==3);
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    if(!strcmp(argv[1],"query"))query_test();
    else if(!strcmp(argv[1],"overflow")) {
        (void)BandNonCol_ArrayBytes(SIZE_MAX,2,"native query conversion test"); CHECK(0);
    }
    else { gpu_kind=atoi(argv[1]); policy_test(); }
    puts("PASS"); return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    source = (ROOT / 'source/Band_DFT_NonCol.c').read_text()
    names = ('BandNonCol_HipSolverWorkspaceTurnRelease', 'BandNonCol_MagmaWorkspaceTurnRelease',
             'BandNonCol_MI300ANativePolicy', 'BandNonCol_ReleaseSolverTurnWorkspaces',
             'BandNonCol_ReleaseOverlapWorkspace', 'BandNonCol_HamiltonianUseHipSolver',
             'BandNonCol_CheckedMul', 'BandNonCol_ArrayBytes', 'BandNonCol_GpuSolverWorkFallbackBytes',
             'BandNonCol_QueryGpuSolverWorkBytes', 'BandNonCol_DenseGpuEigen',
             'BandNonCol_RootDenseCompleteOneK')
    harness = '\n'.join([PREFIX, *(function(source, name) for name in names), MAIN])
    with tempfile.TemporaryDirectory(prefix='openmx-band-nc-workspace-') as tmp:
        src, exe = Path(tmp) / 'check.c', Path(tmp) / 'check'
        src.write_text(harness)
        # Check the query against OpenMX's real plain-C compatibility header
        # as well: a tracking stub must not hide missing enums/prototypes.
        header_probe = Path(tmp) / 'query_header.c'
        header_probe.write_text(
            '#include "hipsolver_compat.h"\n'
            'int BandNonCol_MI300ANativePolicy(int,int);\n'
            'size_t BandNonCol_GpuSolverWorkFallbackBytes(int);\n'
            'size_t BandNonCol_ArrayBytes(size_t,size_t,const char*);\n'
            + function(source, 'BandNonCol_QueryGpuSolverWorkBytes') + '\n')
        subprocess.run([*shlex.split(args.cc), '-std=c11', '-fsyntax-only',
                        '-I' + str(ROOT / 'source'), str(header_probe)], check=True)
        flags = ['-std=c11', '-O2', '-Wall', '-Wextra']
        if args.sanitize:
            flags += ['-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all']
        subprocess.run([*shlex.split(args.cc), *flags, str(src), '-o', str(exe)], check=True)
        keys = ('OPENMX_GPU_EIGENSOLVER', 'OPENMX_NC_HAMILTONIAN_EIGENSOLVER',
                'OPENMX_NC_HIPSOLVER_TURN_RELEASE', 'OPENMX_NC_MAGMA_TURN_RELEASE')
        clean_env = {k: v for k, v in os.environ.items() if k not in keys}
        checks = 0
        for gpu in (0, 1, 2):  # Radeon, MI300A, other APU
            for backend in (None, 'magma', 'hipsolver'):
                for route in (None, 'magma', 'hipsolver'):
                    for native_release in (None, '0', '1'):
                        for magma_release in (None, '0', '1'):
                            env = {**clean_env, **{k: v for k, v in zip(keys, (backend, route, native_release, magma_release)) if v is not None}}
                            subprocess.run([str(exe), str(gpu)], env=env, capture_output=True, text=True, check=True)
                            checks += 1
        subprocess.run([str(exe), 'query'], env=clean_env, capture_output=True, text=True, check=True)
        overflow = subprocess.run([str(exe), 'overflow'], env=clean_env, capture_output=True, text=True)
        assert overflow.returncode == 77 and 'size overflow' in overflow.stderr, overflow.stderr
    print(f'PASS: {checks} NC band policy/release/entry cases; query cache, seven failure/recovery cases and checked overflow')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
