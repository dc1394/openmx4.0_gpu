#!/usr/bin/env python3
"""Exercise every production bridge entry point with mocked CUDA/cuBLAS.

The mock tests dispatch, error propagation, and workspace avoidance. Numerical
accuracy and performance require run_gemmul8_smoke.sh / run_gemmul8_a100_bench.sh.
"""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
headers = {
    "cuda_runtime.h": r'''
#pragma once
#include <cstdlib>
#include <cstddef>
using cudaStream_t = void *;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaDevAttrSingleToDoublePrecisionPerfRatio = 1,
    cudaDevAttrComputeCapabilityMajor=2, cudaDevAttrComputeCapabilityMinor=3, cudaDevAttrMultiProcessorCount=4;
inline int mock_device=0, mock_ratio[8]={2,64,0,2,2,2,2,2}, mock_attr_calls=0;
inline int mock_major[8]={8,8,0,8,9,8,8,8},mock_minor[8]={0,9,0,0,0,0,0,0};
inline int mock_sms[8]={108,128,0,108,132,56,14,108};
inline int mock_device_error=0, mock_attr_error=0, mock_allocs=0, mock_memqueries=0;
inline int mock_memerror=0,mock_memfail_at=0,mock_allocerror=0;
inline size_t mock_free=size_t(40)<<30,mock_total=size_t(40)<<30;
inline int cudaGetDevice(int *d) {*d=mock_device;return mock_device_error;}
inline int cudaDeviceGetAttribute(int *v,int attr,int d) {
    ++mock_attr_calls;*v=attr==1?mock_ratio[d]:attr==2?mock_major[d]:attr==3?mock_minor[d]:mock_sms[d];
    return mock_attr_error;
}
inline int cudaGetDeviceCount(int *n) {*n=4;return 0;}
inline int cudaGetLastError() {return 0;}
inline const char *cudaGetErrorString(int) {return "mock";}
inline int cudaMemGetInfo(size_t *f,size_t *t) {
    ++mock_memqueries;*f=mock_free;*t=mock_total;return mock_memerror || mock_memfail_at==mock_memqueries;
}
inline int cudaMalloc(void **p,size_t) {
    ++mock_allocs;*p=mock_allocerror?nullptr:std::malloc(64);return *p?0:1;
}
inline int cudaFree(void *p) {std::free(p);return 0;}
inline int cudaStreamSynchronize(void *) {return 0;}
''',
    "cuComplex.h": r'''
#pragma once
struct cuDoubleComplex {double x,y;};
inline cuDoubleComplex make_cuDoubleComplex(double x,double y) {return {x,y};}
''',
    "cublas_v2.h": r'''
#pragma once
#include "cuda_runtime.h"
#include "cuComplex.h"
using cublasHandle_t = void *;
enum cublasStatus_t {CUBLAS_STATUS_SUCCESS=0,CUBLAS_STATUS_ALLOC_FAILED=3,CUBLAS_STATUS_INTERNAL_ERROR=14};
enum cublasOperation_t {CUBLAS_OP_N=0,CUBLAS_OP_T=1,CUBLAS_OP_C=2};
inline int mock_blas[2]={0,0};
inline cublasStatus_t mock_blas_status=CUBLAS_STATUS_SUCCESS;
struct Call {int ta,tb,m,n,k,lda,ldb,ldc;const void *a,*b,*alpha,*beta;void *c;};
inline Call mock_call;
inline cublasStatus_t cublasGetStream(cublasHandle_t,void **s) {*s=nullptr;return CUBLAS_STATUS_SUCCESS;}
template<class T> cublasStatus_t mock_gemm(int type,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
    const T *alpha,const T *a,int lda,const T *b,int ldb,const T *beta,T *c,int ldc) {
    ++mock_blas[type];mock_call={ta,tb,m,n,k,lda,ldb,ldc,a,b,alpha,beta,c};return mock_blas_status;
}
inline cublasStatus_t cublasDgemm(cublasHandle_t,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
    const double *alpha,const double *a,int lda,const double *b,int ldb,const double *beta,double *c,int ldc) {
    return mock_gemm(0,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
inline cublasStatus_t cublasZgemm(cublasHandle_t,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
    const cuDoubleComplex *alpha,const cuDoubleComplex *a,int lda,const cuDoubleComplex *b,int ldb,
    const cuDoubleComplex *beta,cuDoubleComplex *c,int ldc) {
    return mock_gemm(1,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
''',
    "gemmul8.hpp": r'''
#pragma once
#include <vector>
inline int mock_ozaki=0, mock_workqueries=0, mock_workconfig=0,mock_capped=0;
inline size_t mock_worksize=64;
namespace gemmul8 {
enum class Backend {INT8};
inline void set_memory_saving(void *,bool capped) {++mock_workconfig;if(capped)++mock_capped;}
inline void set_max_worksize(void *,size_t) {++mock_workconfig;}
template<bool,Backend> size_t workSize(size_t,size_t,size_t,int,bool=false,bool=false,
    size_t *a=nullptr,size_t *b=nullptr,bool=false) {
    ++mock_workqueries;if(a)*a=16;if(b)*b=16;return mock_worksize;
}
template<class T,Backend B,class... Args> std::vector<double> gemm(Args...) {++mock_ozaki;return {};}
}
''',
}
main = r'''
#include "gemmul8_bridge.cu"
extern "C" {int level_stdout=0;}
static void require(bool ok,const char *message) {
    if(!ok) {std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1);}
}
static void clear_env() {
    for(const char *name:{"GEMMUL8_DISABLE","OPENMX_GEMMUL8_DISABLE","GEMMUL8_DISABLE_D",
        "OPENMX_GEMMUL8_DISABLE_D","GEMMUL8_DISABLE_Z","OPENMX_GEMMUL8_DISABLE_Z"}) unsetenv(name);
}
static void enabled(int d,int z,const char *label) {
    require(openmx_gemmul8DgemmEnabled()==d && openmx_gemmul8ZgemmEnabled()==z,label);
    require(openmx_gemmul8DWorkspaceSize(7,5,3)==(d?64:0),"real workspace agrees with dispatch");
    require(openmx_gemmul8ZWorkspaceSize(7,5,3)==(z?64:0),"complex workspace agrees with dispatch");
}
static void all_entries(bool native=true) {
    double a[24]={},b[28]={},c[48]={},one=1,zero=0;
    cuDoubleComplex za[24]={},zb[28]={},zc[48]={},zone={1,0},zzero={0,0};
    require(openmx_gemmul8Dgemm(nullptr,CUBLAS_OP_C,CUBLAS_OP_N,7,5,3,&one,a,4,b,4,&zero,c,9)==0,"D dispatch");
    if(native) require(mock_call.ta==CUBLAS_OP_T && mock_call.tb==CUBLAS_OP_N && mock_call.m==7 && mock_call.n==5 &&
        mock_call.k==3 && mock_call.lda==4 && mock_call.ldb==4 && mock_call.ldc==9 && mock_call.a==a &&
        mock_call.b==b && mock_call.c==c && mock_call.alpha==&one && mock_call.beta==&zero,"D arguments unchanged");
    require(openmx_gemmul8Zgemm(nullptr,CUBLAS_OP_C,CUBLAS_OP_N,7,5,3,&zone,za,4,zb,4,&zzero,zc,9)==0,"Z dispatch");
    if(native) require(mock_call.ta==CUBLAS_OP_C && mock_call.a==za && mock_call.c==zc,"complex conjugation preserved");
    require(openmx_gemmul8DgemmUnblocked(nullptr,CUBLAS_OP_N,CUBLAS_OP_N,7,5,3,&one,a,8,b,4,&zero,c,9)==0,"D unblocked");
    require(openmx_gemmul8ZgemmUnblocked(nullptr,CUBLAS_OP_N,CUBLAS_OP_C,7,5,3,&zone,za,8,zb,6,&zzero,zc,9)==0,"Z unblocked");
    require(openmx_gemmul8DgemmFixed(nullptr,0,CUBLAS_OP_N,7,5,3,a,4,b,8,c,9,1,1)==0,"D fixed");
    require(openmx_gemmul8ZgemmFixed(nullptr,1,CUBLAS_OP_C,7,5,3,za,4,zb,4,zc,9,1,1)==0,"Z fixed");
}
static void fixed(bool ozaki,const char *label,int m=6144,int n=6144,int k=6144,bool complex=false) {
    double a=1,b=2,c=0;cuDoubleComplex za={1,0},zb={2,0},zc={0,0};
    const int before=mock_ozaki,native=mock_blas[0]+mock_blas[1];
    const auto status=complex
        ?openmx_gemmul8ZgemmFixed(nullptr,1,CUBLAS_OP_C,m,n,k,&za,k,&zb,k,&zc,m,0,1)
        :openmx_gemmul8DgemmFixed(nullptr,1,CUBLAS_OP_T,m,n,k,&a,k,&b,k,&c,m,0,1);
    require(status==0 && mock_ozaki-before==(ozaki?1:0) &&
            mock_blas[0]+mock_blas[1]-native==(ozaki?0:1),label);
}
static void adaptive_forward() {
    clear_env();openmx_gemmul8SetEnabled(-1);openmx_gemmul8SetForwardStage(0,15,0,0,1);mock_device=0;
    fixed(false,"A100 below threshold",6143,6143,6143);
    fixed(true,"A100 threshold is inclusive");
    fixed(true,"A100 above threshold",8000,8000,8000);
    fixed(false,"rectangular fixed stays native",6144,6143,6144);
    fixed(false,"complex fixed stays native",8000,8000,8000,true);
    setenv("OPENMX_GEMMUL8_A100_FORWARD_MIN_N","9000",1);
    fixed(false,"threshold can be raised",8000,8000,8000);
    fixed(true,"raised threshold inclusive",9000,9000,9000);
    setenv("OPENMX_GEMMUL8_A100_FORWARD_MIN_N","0",1);fixed(false,"zero disables automatic exception");
    setenv("OPENMX_GEMMUL8_A100_FORWARD_MIN_N","invalid",1);fixed(true,"invalid threshold uses default");
    unsetenv("OPENMX_GEMMUL8_A100_FORWARD_MIN_N");
    for(int device:{4,5,6}) {mock_device=device;fixed(false,"Hopper, A30 and small MIG do not inherit crossover");}
    mock_device=1;fixed(true,"large GeForce fixed stays Ozaki");fixed(true,"small GeForce fixed stays Ozaki",32,32,32);
    mock_device=7;enabled(0,0,"new full A100 general policy");mock_attr_error=1;
    fixed(false,"failed A100 capability query selects native");mock_attr_error=0;
    fixed(true,"failed A100 capability query is retried");mock_device=0;

    for(const char *name:{"GEMMUL8_DISABLE","OPENMX_GEMMUL8_DISABLE","GEMMUL8_DISABLE_D","OPENMX_GEMMUL8_DISABLE_D"}) {
        setenv(name,"1",1);fixed(false,"explicit global/type disable beats exception");
        setenv(name,"0",1);fixed(true,"explicit global/type enable remains Ozaki",32,32,32);unsetenv(name);
    }
    setenv("OPENMX_GEMMUL8_DISABLE","1",1);setenv("GEMMUL8_DISABLE_D","0",1);
    fixed(true,"type enable beats global disable",32,32,32);
    setenv("OPENMX_GEMMUL8_DISABLE_D","1",1);fixed(false,"OpenMX type disable beats legacy type enable");
    clear_env();setenv("OPENMX_GEMMUL8_DISABLE_Z","1",1);fixed(true,"complex override does not suppress real exception");
    clear_env();openmx_gemmul8SetEnabled(0);fixed(false,"input off beats automatic exception");
    openmx_gemmul8SetEnabled(1);fixed(true,"input on keeps small fixed Ozaki",32,32,32);
    openmx_gemmul8SetEnabled(-1);openmx_gemmul8SetForwardStage(1,15,0,0,1);
    fixed(false,"explicit fp64 stage beats automatic exception");
    openmx_gemmul8SetForwardStage(2,15,0,0,1);fixed(true,"explicit Ozaki stage keeps small fixed Ozaki",32,32,32);
    openmx_gemmul8SetEnabled(0);fixed(false,"input off also beats explicit Ozaki stage");
    openmx_gemmul8SetEnabled(-1);openmx_gemmul8SetForwardStage(0,15,0,0,0);
    fixed(false,"explicitly disabled unblocked path suppresses automatic exception");
    openmx_gemmul8SetForwardStage(0,15,0,0,1);
    double a=1,b=2,c=0,one=1,zero=0;const int before=mock_ozaki;
    require(openmx_gemmul8Dgemm(nullptr,CUBLAS_OP_N,CUBLAS_OP_N,8000,8000,8000,&one,&a,8000,&b,8000,&zero,&c,8000)==0 &&
        openmx_gemmul8DgemmUnblocked(nullptr,CUBLAS_OP_N,CUBLAS_OP_N,8000,8000,8000,&one,&a,8000,&b,8000,&zero,&c,8000)==0 &&
        before==mock_ozaki,"large general and unfixed unblocked APIs stay native");

    openmx_gemmul8ReleaseWorkspaces();mock_worksize=size_t(3)<<30;
    fixed(true,"automatic forward uses full workspace");const int allocs=mock_allocs;
    fixed(true,"second product reuses full scratch");require(mock_allocs==allocs,"scratch reused across pair");
    openmx_gemmul8TrimWorkspaces();fixed(true,"next trimmed pair reallocates scratch");
    require(mock_allocs==allocs+1,"production trim releases oversized scratch");
    openmx_gemmul8ReleaseWorkspaces();const int capped=mock_capped;
    mock_free=size_t(4)<<30;fixed(false,"insufficient reserve selects native");mock_free=size_t(40)<<30;
    mock_total=0;fixed(false,"unknown total memory selects native");mock_total=size_t(40)<<30;
    mock_memerror=1;fixed(false,"memory query failure selects native");mock_memerror=0;
    mock_memfail_at=mock_memqueries+2;fixed(false,"second memory query failure selects native");mock_memfail_at=0;
    mock_allocerror=1;fixed(false,"allocation race/failure selects native");mock_allocerror=0;
    require(mock_capped==capped,"automatic full-workspace failure never runs capped Ozaki");
    setenv("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT","1",1);fixed(false,"explicit workspace percentage respected");
    unsetenv("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT");
    mock_worksize=64;openmx_gemmul8ReleaseWorkspaces();
}
static void owner_budget() {
    clear_env();openmx_gemmul8SetEnabled(-1);openmx_gemmul8SetForwardStage(0,15,0,0,1);
    setenv("OPENMX_GEMMUL8_LOCAL_RANKS","8",1); // four visible GPUs => two ranks/GPU
    mock_worksize=size_t(9)<<30;mock_free=mock_total=size_t(16)<<30;
    fixed(true,"default scratch budgets dense owners, not every local rank");openmx_gemmul8ReleaseWorkspaces();
    setenv("GEMMUL8_MAX_WORKSPACE_PERCENT","100",1);
    fixed(false,"explicit fraction honors existing per-rank budget");
    setenv("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT","0",1);
    fixed(true,"OpenMX zero disables legacy fraction cap");openmx_gemmul8ReleaseWorkspaces();
    setenv("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT","30",1);fixed(false,"explicit small fraction cap");
    openmx_gemmul8SetEnabled(1);fixed(true,"explicit existing forward preserves free/reserve budget");
    openmx_gemmul8ReleaseWorkspaces();
    mock_device=1;openmx_gemmul8SetEnabled(-1);fixed(true,"existing GeForce full-scratch budget unchanged");
    openmx_gemmul8ReleaseWorkspaces();std::puts("PASS: dense-owner scratch budget and explicit rank/fraction overrides");
}
int main(int argc,char **argv) {
    if(argc>1) {
        if(std::strcmp(argv[1],"owner-budget")==0) {owner_budget();return 0;}
        // Fresh processes exercise the environment stage before any setter caches it.
        openmx_gemmul8SetEnabled(-1);
        if(std::strcmp(argv[1],"forward-fp64")==0) {
            setenv("OPENMX_GEMMUL8_FORWARD","fp64",1);fixed(false,"environment fp64 stage wins");
        } else if(std::strcmp(argv[1],"forward-gemmul8")==0) {
            setenv("OPENMX_GEMMUL8_FORWARD","gemmul8",1);
            fixed(true,"environment Ozaki stage wins for small products",32,32,32);
        } else if(std::strcmp(argv[1],"forward-blocked")==0) {
            setenv("OPENMX_GEMMUL8_FORWARD_UNBLOCKED","0",1);
            fixed(false,"environment disabled unblocked path wins");
        } else require(false,"unknown test case");
        openmx_gemmul8ReleaseWorkspaces();std::puts("PASS: forward environment override");return 0;
    }
    clear_env();openmx_gemmul8SetForwardStage(0,15,0,0,1);
    enabled(0,0,"A100 defaults to native FP64");
    all_entries();
    require(mock_attr_calls==1,"device attributes cached across all entry points");
    require(mock_blas[0]==3 && mock_blas[1]==3,"all six entry points use native cuBLAS");
    require(openmx_gemmul8NativeCalls(0)==3 && openmx_gemmul8NativeCalls(1)==3,"native counters");
    require(mock_ozaki==0 && mock_allocs==0 && mock_workqueries==0 && mock_workconfig==0 && mock_memqueries==0,
            "native A100 path bypasses Ozaki setup and allocation entirely");

    openmx_gemmul8SetEnabled(1);enabled(1,1,"explicit input on overrides device default");
    setenv("OPENMX_GEMMUL8_DISABLE","1",1);enabled(0,0,"environment can still disable explicit input on");
    openmx_gemmul8SetEnabled(-1);clear_env();
    setenv("GEMMUL8_DISABLE","0",1);enabled(1,1,"explicit legacy enable on A100");
    setenv("OPENMX_GEMMUL8_DISABLE","1",1);enabled(0,0,"OpenMX override beats legacy global");
    setenv("GEMMUL8_DISABLE_D","0",1);enabled(1,0,"type override beats global");
    setenv("OPENMX_GEMMUL8_DISABLE_D","1",1);enabled(0,0,"OpenMX type override beats legacy type");
    setenv("OPENMX_GEMMUL8_DISABLE_Z","0",1);enabled(0,1,"complex independently enabled");
    openmx_gemmul8SetEnabled(0);enabled(0,0,"input off is absolute");
    openmx_gemmul8SetForwardStage(2,15,0,1,1);all_entries();
    require(mock_ozaki==0,"input off applies even to explicit forward stages");

    openmx_gemmul8SetEnabled(-1);clear_env();
    setenv("OPENMX_GEMMUL8_DISABLE","",1);enabled(0,0,"empty override preserves hardware policy");
    mock_device=1;enabled(1,1,"GeForce preserves GEMMul8 default");
    openmx_gemmul8SetForwardStage(0,15,0,0,1);
    all_entries(false);
    require(mock_ozaki==6,"all six GeForce entry points preserve Ozaki execution");
    openmx_gemmul8ReleaseWorkspaces();
    mock_device=0;enabled(0,0,"switch back to A100 refreshes policy");
    mock_device=2;enabled(1,1,"unknown ratio preserves existing default");
    mock_device_error=1;enabled(1,1,"device detection failure preserves existing default");mock_device_error=0;
    mock_device=3;mock_attr_error=1;enabled(1,1,"attribute failure preserves existing default");
    mock_attr_error=0;enabled(0,0,"failed detection is retried");

    double a[1]={1},b[1]={2},c[1]={},one=1,zero=0;
    openmx_gemmul8SetForwardStage(2,15,0,1,1);
    require(openmx_gemmul8DgemmFixed(nullptr,1,CUBLAS_OP_N,1,1,1,a,1,b,1,c,1,0,1)==0,"explicit forward GEMMul8");
    require(mock_ozaki==7 && mock_allocs>0,"explicit forward stage opts into Ozaki on A100");
    openmx_gemmul8ReleasePrepared();openmx_gemmul8ReleaseWorkspaces();
    mock_device=1;openmx_gemmul8SetForwardStage(1,15,0,0,1);
    const int previous=mock_ozaki;
    require(openmx_gemmul8DgemmFixed(nullptr,1,CUBLAS_OP_N,1,1,1,a,1,b,1,c,1,0,1)==0 && mock_ozaki==previous,
            "explicit forward FP64 honored on GeForce");
    mock_device=0;mock_blas_status=CUBLAS_STATUS_INTERNAL_ERROR;
    require(openmx_gemmul8Dgemm(nullptr,CUBLAS_OP_N,CUBLAS_OP_N,1,1,1,&one,a,1,b,1,&zero,c,1)==mock_blas_status,
            "native error propagated");
    require(openmx_gemmul8Dgemm(nullptr,CUBLAS_OP_N,CUBLAS_OP_N,0,1,1,&one,a,1,b,1,&zero,c,1)==0,
            "empty products preserve quick return");
    mock_blas_status=CUBLAS_STATUS_SUCCESS;adaptive_forward();
    std::puts("PASS: full bridge A100/GeForce dispatch, automatic large forward boundaries, overrides, resource failures");
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-gemmul8-policy-") as tmp:
    directory = Path(tmp)
    for name, content in headers.items():
        (directory / name).write_text(content)
    source, binary = directory / "check.cpp", directory / "check"
    source.write_text(main)
    subprocess.run(shlex.split(os.environ.get("CXX", "c++")) +
                   ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
                    "-I" + str(directory), "-I" + str(root / "source"), str(source), "-o", str(binary)], check=True)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("OPENMX_", "GEMMUL8_"))}
    subprocess.run([str(binary)], env=environment, check=True)
    for case in ("owner-budget", "forward-fp64", "forward-gemmul8", "forward-blocked"):
        subprocess.run([str(binary), case], env=environment, check=True)
