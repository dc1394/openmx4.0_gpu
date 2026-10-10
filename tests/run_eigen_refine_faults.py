#!/usr/bin/env python3
"""CPU fault injection for production refinement guards and cuSOLVER retry.

The production functions are extracted verbatim; CUDA/cuSOLVER are host stubs.
This verifies fallback input preservation and error propagation, not GPU timing
or the numerical convergence of cuSOLVER. --sanitize adds ASan/UBSan.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def function(source, name):
    match = re.search(r"^(?:static )?(?:int|void|cusolverStatus_t) \b" + re.escape(name) +
                      r"\([^;]*?\)\s*\{", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production function {name}")
    brace = source.index("{", match.start())
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


PREFIX = r'''
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(2); } } while (0)
typedef int cudaError_t;
typedef int cusolverStatus_t;
typedef int cudaDataType;
typedef int cudaStream_t;
typedef int cusolverDnHandle_t;
typedef int cusolverMathMode_t;
typedef struct { double x,y; } cuDoubleComplex;
enum { cudaSuccess=0, CUSOLVER_STATUS_SUCCESS=0, CUSOLVER_STATUS_INVALID_VALUE=3,
       CUSOLVER_STATUS_INTERNAL_ERROR=7, CUSOLVER_STATUS_EXECUTION_FAILED=8,
       CUDA_R_64F=10, CUDA_C_64F=11, cudaMemcpyHostToDevice=12,
       cudaMemcpyDeviceToHost=13, cudaMemcpyDeviceToDevice=14,
       CUSOLVER_DEFAULT_MATH=15, CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH=16 };
#define CUSOLVER_VERSION 12304
'''

RETRY_PREFIX = PREFIX + r'''
typedef cusolverStatus_t (*openmx_solver_once_fn)(void *);
static int mode, malloc_fail, getstream_fail, sync_fail, sync_calls, copy_fail,
           native_mode_fail, restore_mode_fail, copy_calls, allocations,
           forced, solve_calls, emulated_info, native_info, native_fail,
           query_fail, matrix_reports, checks;
static void *pending_dst;
static const void *pending_src;
static size_t pending_bytes;
static int cusolverDnGetMathMode(int h,int *m) { *m=mode; return 0; }
static int cusolverDnGetStream(int h,int *s) { *s=23; return getstream_fail ? 7 : 0; }
static int cusolverDnSetMathMode(int h,int m) {
    if (m==CUSOLVER_DEFAULT_MATH && native_mode_fail) return 7;
    if (m==CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH && restore_mode_fail) return 7;
    mode=m; return 0;
}
static int cudaMalloc(void **p,size_t n) {
    if (malloc_fail) { *p=NULL; return 2; }
    *p=malloc(n); CHECK(*p); allocations++; return 0;
}
static int cudaFree(void *p) { free(p); allocations--; return 0; }
static int cudaGetLastError(void) { return 0; }
static int cudaStreamSynchronize(int s) {
    if (pending_dst) { memcpy(pending_dst,pending_src,pending_bytes); pending_dst=NULL; }
    sync_calls++; return sync_calls==sync_fail ? 1 : 0;
}
static int cudaMemcpy(void *d,const void *s,size_t n,int kind) {
    copy_calls++; if (copy_calls==copy_fail) return 1;
    memcpy(d,s,n); return 0;
}
static int cudaMemcpyAsync(void *d,const void *s,size_t n,int kind,int stream) {
    copy_calls++; if (copy_calls==copy_fail) return 1;
    if (kind==cudaMemcpyHostToDevice) {
        CHECK(!pending_dst); pending_dst=d; pending_src=s; pending_bytes=n;
    } else memcpy(d,s,n);
    return 0;
}
static int openmx_solver_retry_test(void) { return forced; }
static void openmx_solver_report_matrix(const void *p,int host,int type,int64_t lda,int64_t n,int info) {
    matrix_reports++;
}
typedef struct { double *a; int *info; size_t *db,*hb; int width; } TestContext;
static int solve(void *v) {
    TestContext *c=v; solve_calls++;
    if (pending_dst) { /* a CUDA launch is ordered after its stream's restore */
        memcpy(pending_dst,pending_src,pending_bytes); pending_dst=NULL;
    }
    for (int i=0;i<4*c->width;i++) CHECK(c->a[i]==i+0.25);
    for (int i=0;i<4*c->width;i++) c->a[i]=-123.0;
    *c->info=mode==CUSOLVER_DEFAULT_MATH ? native_info : emulated_info;
    return mode==CUSOLVER_DEFAULT_MATH && native_fail ? 7 : 0;
}
static int query(void *v) {
    TestContext *c=v; CHECK(mode==CUSOLVER_DEFAULT_MATH);
    *c->db=128; *c->hb=64; return query_fail ? 7 : 0;
}
static void reset(void) {
    CHECK(!allocations && !pending_dst);
    mode=CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH;
    malloc_fail=getstream_fail=sync_fail=sync_calls=copy_fail=copy_calls=0;
    native_mode_fail=restore_mode_fail=forced=solve_calls=emulated_info=native_info=native_fail=0;
    query_fail=matrix_reports=0;
}
'''

RETRY_HARNESS = r'''
static void one(int width,int host,int fault) {
    double a[8]; int info=-99, expect=0, expected_calls=1;
    TestContext c={a,&info,NULL,NULL,width};
    reset(); malloc_fail=host;
    for (int i=0;i<4*width;i++) a[i]=i+0.25;
    switch(fault) {
    case 0: break;
    case 1: emulated_info=122; expected_calls=2; break;
    case 2: emulated_info=-7; break; /* invalid arguments are not nonconvergence */
    case 3: forced=1; expected_calls=2; break;
    case 4: getstream_fail=1; expect=7; expected_calls=0; break;
    case 5: copy_fail=host?2:2; expect=8; break; /* INFO download */
    case 6: sync_fail=host?2:1; expect=8; break;
    case 7: emulated_info=122; copy_fail=3; expect=8; break; /* restore */
    case 8: emulated_info=122; native_mode_fail=1; expect=7; break;
    case 9: emulated_info=122; restore_mode_fail=1; expect=7; expected_calls=2; break;
    case 10: emulated_info=122; native_fail=1; expect=7; expected_calls=2; break;
    case 11: emulated_info=122; native_info=9; expected_calls=2; break;
    case 12: emulated_info=122; copy_fail=4; expect=8; expected_calls=2; break;
    case 13: copy_fail=1; expect=8; expected_calls=0; break;
    case 14: forced=1; emulated_info=-3; break;
    }
    CHECK(openmx_solver_retry(1,width==2?CUDA_C_64F:CUDA_R_64F,a,2,2,&info,solve,&c)==expect);
    CHECK(solve_calls==expected_calls && !allocations && !pending_dst);
    if (!restore_mode_fail) CHECK(mode==CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH);
    if (fault==11) CHECK(matrix_reports==1);
    if (fault==2 || fault==14) CHECK(info<0);
    checks++;
}
int main(void) {
    for (int w=1;w<=2;w++) for (int h=0;h<=1;h++) for(int f=0;f<=14;f++) one(w,h,f);
    for (int f=0;f<4;f++) {
        size_t db=64,hb=128;
        TestContext c={NULL,NULL,&db,&hb,1};
        reset(); native_mode_fail=f==1; restore_mode_fail=f==2; query_fail=f==3;
        CHECK(openmx_solver_both_modes(1,CUDA_R_64F,query,&c,&db,&hb)==(f?7:0));
        if (!f) CHECK(db==128 && hb==128);
        if (f==3) CHECK(db==64 && hb==128);
        if (f!=2) CHECK(mode==CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH);
        checks++;
    }
    reset();
    CHECK(openmx_solver_retry(1,CUDA_R_64F,NULL,INT64_MAX,INT64_MAX,&emulated_info,solve,NULL)==3);
    CHECK(!solve_calls); checks++;
    printf("PASS cuSOLVER retry: %d fault/control checks\n",checks);
    return 0;
}
'''

REFINE_PREFIX = PREFIX + r'''
typedef struct {
    double *b1,*b2,*b3,*lam,*occ; void *own; int products_ready,basis_valid,basis_n;
    double delta_cold;
} EigenRefineState;
typedef struct { int unused; void *(*host_malloc)(size_t,size_t,const char *); } EigenRefineDevice;
typedef struct {
    int cplx,n,maxn,iterations,transient,warm,defaulted;
    void *a,*x,*fp32; double *w; size_t region;
    double (*occupation)(double,int,void *); void *occupation_ctx;
} EigenRefineProblem;
typedef struct { int persistent,columns,rr_clusters,rr_largest; double max_r,max_s,delta,chain; } EigenRefineReport;
enum { CUBLAS_OP_N,CUBLAS_OP_C };
static double refine_anorm=4.0;
static int gemm_calls, fault_call, fault_component, rr_fault, checks;
static double fault_value;
static int env_flag(const char *s,int def) { return def; }
static void refine_check(int s,const char *what) { CHECK(!s); }
static void refine_check_blas(int s,const char *what) { CHECK(!s); }
static int cudaFree(void *p) { free(p); return 0; }
static int cudaMemcpy(void *d,const void *s,size_t n,int k) { memcpy(d,s,n); return 0; }
static int cudaDeviceSynchronize(void) { return 0; }
static long long openmx_gemmul8NativeCalls(int c) { return 0; }
static int refine_blocks_ensure(EigenRefineState *s,EigenRefineDevice *d,const EigenRefineProblem *p,EigenRefineReport *r) {
    CHECK(0); return 0;
}
static void *hmalloc(size_t n,size_t w,const char *label) { void *p=malloc(n*w); CHECK(p); return p; }
static int refine_gemm(const EigenRefineDevice *dev,int cplx,int ta,int tb,int m,int n,int k,
    const void *av,int lda,const void *bv,int ldb,double beta,void *cv,int ldc) {
    const double *a=av,*b=bv; double *c=cv; int w=cplx?2:1;
    for(int j=0;j<n;j++) for(int i=0;i<m;i++) {
        double re=0,im=0;
        for(int q=0;q<k;q++) {
            size_t ia=w*(ta==CUBLAS_OP_N ? i+(size_t)q*lda : q+(size_t)i*lda);
            size_t ib=w*(tb==CUBLAS_OP_N ? q+(size_t)j*ldb : j+(size_t)q*ldb);
            double ar=a[ia],ai=cplx?a[ia+1]*(ta==CUBLAS_OP_N?1:-1):0;
            double br=b[ib],bi=cplx?b[ib+1]*(tb==CUBLAS_OP_N?1:-1):0;
            re+=ar*br-ai*bi; im+=ar*bi+ai*br;
        }
        size_t ix=w*(i+(size_t)j*ldc);
        c[ix]=re+(beta?beta*c[ix]:0); if(cplx)c[ix+1]=im+(beta?beta*c[ix+1]:0);
    }
    if (++gemm_calls==fault_call) c[fault_component]=fault_value;
    return 0;
}
static int native_gemm(const EigenRefineDevice *d,int c,int ta,int tb,int m,int n,int k,
    const void *a,int lda,const void *b,int ldb,void *out,int ldc) {
    return refine_gemm(d,c,ta,tb,m,n,k,a,lda,b,ldb,0,out,ldc);
}
static int refine_rayleigh_ritz(EigenRefineDevice *d,int c,const void *a,double *x,int n,int m,
    double *z,double *small,double *wd,double *w) {
    CHECK(rr_fault); x[0]=NAN; return 1;
}
static double occupation(double e,int i,void *ctx) { return i==1?1.0:0.0; }
'''

REFINE_HARNESS = r'''
static void one(int cplx,int fault) {
    int n=4,width=cplx?2:1;
    double a[32]={0},saved[32],x[32]={0},lam[4]={1,2,3,4},occ[4],f[4]={1,0,0,0},out[4];
    EigenRefineState st={0}; EigenRefineReport rep={0};
    EigenRefineDevice dev={0,hmalloc};
    EigenRefineProblem pb={cplx,n,n,2,0,0,0,a,x,NULL,out,0,occupation,NULL};
    gemm_calls=0; fault_call=0; rr_fault=0; fault_component=0; fault_value=NAN;
    for(int i=0;i<n;i++) { a[width*(i+i*n)]=lam[i]; x[width*(i+i*n)]=1; out[i]=-999; }
    if(fault==1) { fault_call=2; fault_component=width; }
    if(fault==2) { fault_call=3; fault_component=cplx?1:0; fault_value=INFINITY; }
    if(fault==3) { fault_call=8; fault_component=cplx?1:0; }
    if(fault==4) f[2]=NAN;
    if(fault==5) { fault_call=3; fault_value=0; }
    if(fault==6) { fault_call=8; fault_value=INFINITY; }
    if(fault==7) {
        a[width*5]=lam[1]=1; x[0]=1.00000001; rr_fault=1;
    }
    memcpy(saved,a,sizeof(a)); st.lam=lam; st.occ=occ; st.basis_valid=1;
    st.own=calloc(64,sizeof(double)); CHECK(st.own); st.b1=st.own; st.b2=st.b1+32;
    int ok=openmx_eigen_refine_finish(&st,&dev,&pb,f,&rep);
    CHECK(ok==(fault==0)); CHECK(!st.own && !st.b1 && !st.b2);
    if(fault) {
        CHECK(!st.basis_valid && rep.persistent==1);
        CHECK(!memcmp(a,saved,sizeof(a)));
        for(int i=0;i<n;i++) CHECK(out[i]==-999);
    } else {
        CHECK(st.basis_valid);
        for(int i=0;i<n;i++) CHECK(fabs(out[i]-(i+1))<1e-14);
    }
    checks++;
}
int main(void) {
    for(int c=0;c<=1;c++) for(int f=0;f<=7;f++) one(c,f);
    printf("PASS refinement: %d real/complex success and fallback-preservation checks\n",checks);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    retry = (root / "source/openmx_cusolver_compat.c").read_text()
    refine = (root / "source/eigen_refine_gpu.c").read_text()
    bodies = {
        "retry": RETRY_PREFIX + "\n".join(function(retry, name) for name in
             ("openmx_solver_emulated", "openmx_solver_retry", "openmx_solver_both_modes")) + RETRY_HARNESS,
        "refine": REFINE_PREFIX + "\n".join(function(refine, name) for name in
             ("refine_blocks_release", "refine_maxima", "refine_nonfinite", "refine_result_finite",
              "openmx_eigen_refine_finish")) + REFINE_HARNESS,
    }
    cc = shlex.split(os.environ.get("CC", "cc"))
    flags = shlex.split(os.environ.get("CFLAGS", "-O2"))
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    with tempfile.TemporaryDirectory(prefix="openmx-refine-faults-") as tmp:
        for name, body in bodies.items():
            source = Path(tmp) / f"{name}.c"
            exe = Path(tmp) / name
            source.write_text(body)
            subprocess.run(cc + flags + ["-std=c11", "-Wall", "-Wextra", "-Wno-unused-parameter",
                "-Wno-unknown-pragmas", str(source), "-lm", "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
