#!/usr/bin/env python3
"""Run the production tiled GEMM and numerical fixture on fake CUDA devices.

Checks indexing, scalar/transpose arithmetic, budgets, device restoration and
transactional retries without a GPU. Real transfer/stream behavior additionally
requires run_mgpu_gemm_smoke.sh on at least two physical GPUs.
"""
from pathlib import Path
import os
import re
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
cuda = r'''
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unordered_map>
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaStreamNonBlocking=1,cudaEventDisableTiming=2,cudaHostAllocPortable=4;
constexpr int cudaErrorPeerAccessAlreadyEnabled=5;
enum cudaMemoryType {cudaMemoryTypeHost=1,cudaMemoryTypeDevice=2};
enum cudaMemcpyKind {cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice};
enum cudaStreamCaptureStatus {cudaStreamCaptureStatusNone};
struct MockStream {int device;};using cudaStream_t=MockStream *;
struct MockEvent {int device;};using cudaEvent_t=MockEvent *;
inline int mock_device=0;
struct Allocation {size_t bytes;int device;};
inline std::unordered_map<void *,Allocation> allocations;
inline std::unordered_map<void *,size_t> host_allocations;
inline int host_allocation_calls=0;
struct cudaPointerAttributes {cudaMemoryType type;int device;};
struct cudaPitchedPtr {void *ptr;size_t pitch,xsize,ysize;};
struct cudaExtent {size_t width,height,depth;};
struct cudaMemcpy3DPeerParms {cudaPitchedPtr srcPtr,dstPtr;int srcDevice,dstDevice;cudaExtent extent;};
inline cudaPitchedPtr make_cudaPitchedPtr(void *p,size_t pitch,size_t w,size_t h){return {p,pitch,w,h};}
inline cudaExtent make_cudaExtent(size_t w,size_t h,size_t d){return {w,h,d};}
inline int cudaGetDevice(int *v){*v=mock_device;return 0;}
inline int cudaSetDevice(int v){if(v<0||v>=4)return 1;mock_device=v;return 0;}
inline int cudaGetDeviceCount(int *v){*v=4;return 0;}
inline int cudaDeviceGetPCIBusId(char *s,int n,int d){std::snprintf(s,n,"0000:%02x:00.0",d);return 0;}
inline int cudaDeviceCanAccessPeer(int *can,int,int){*can=1;return 0;}
inline int cudaDeviceEnablePeerAccess(int,int){return 0;}
inline int cudaGetLastError(){return 0;}
inline const char *cudaGetErrorString(int){return "mock CUDA error";}
inline int cudaMemGetInfo(size_t *f,size_t *t){*f=*t=size_t(16)<<30;return 0;}
template<class T> inline int cudaMalloc(T **p,size_t n){*p=static_cast<T *>(std::malloc(n));if(!*p)return 1;allocations[*p]={n,mock_device};return 0;}
inline int cudaFree(void *p){if(!p)return 0;auto it=allocations.find(p);if(it==allocations.end()||it->second.device!=mock_device)return 1;allocations.erase(it);std::free(p);return 0;}
inline int cudaHostAlloc(void **p,size_t n,int){*p=std::malloc(n);if(*p){host_allocations[*p]=n;++host_allocation_calls;}return *p?0:1;}
inline int cudaFreeHost(void *p){if(!host_allocations.erase(p))return 1;std::free(p);return 0;}
inline int cudaPointerGetAttributes(cudaPointerAttributes *a,const void *p){
    for(auto &x:allocations){const auto base=reinterpret_cast<uintptr_t>(x.first),q=reinterpret_cast<uintptr_t>(p);
        if(q>=base&&q-base<x.second.bytes){a->type=cudaMemoryTypeDevice;a->device=x.second.device;return 0;}}
    return 1;
}
inline int cudaStreamCreateWithFlags(cudaStream_t *s,int){*s=new MockStream{mock_device};return 0;}
inline int cudaStreamSynchronize(cudaStream_t s){return s&&s->device!=mock_device?1:0;}
inline int cudaStreamDestroy(cudaStream_t s){if(s&&s->device!=mock_device)return 1;delete s;return 0;}
inline int cudaStreamIsCapturing(cudaStream_t,cudaStreamCaptureStatus *v){*v=cudaStreamCaptureStatusNone;return 0;}
inline int cudaEventCreateWithFlags(cudaEvent_t *e,int){*e=new MockEvent{mock_device};return 0;}
inline int cudaEventRecord(cudaEvent_t e,cudaStream_t s){return e->device!=mock_device||(s&&s->device!=mock_device);}
inline int cudaEventDestroy(cudaEvent_t e){if(e->device!=mock_device)return 1;delete e;return 0;}
inline int cudaStreamWaitEvent(cudaStream_t s,cudaEvent_t,int){return s&&s->device!=mock_device;}
inline int cudaMemcpy(void *d,const void *s,size_t n,cudaMemcpyKind){std::memcpy(d,s,n);return 0;}
inline int cudaMemcpyAsync(void *d,const void *s,size_t n,cudaMemcpyKind k,cudaStream_t st){if(st&&st->device!=mock_device)return 1;return cudaMemcpy(d,s,n,k);}
inline int cudaMemcpy2DAsync(void *d,size_t dp,const void *s,size_t sp,size_t w,size_t h,cudaMemcpyKind,cudaStream_t st){
    if(st&&st->device!=mock_device)return 1;if(w>dp||w>sp)return 1;
    for(size_t j=0;j<h;++j)std::memcpy(static_cast<char *>(d)+j*dp,static_cast<const char *>(s)+j*sp,w);return 0;
}
inline int cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms *p,cudaStream_t st){
    return cudaMemcpy2DAsync(p->dstPtr.ptr,p->dstPtr.pitch,p->srcPtr.ptr,p->srcPtr.pitch,p->extent.width,p->extent.height,cudaMemcpyDeviceToDevice,st);
}
'''
complex_header = r'''
#pragma once
struct cuDoubleComplex {double x,y;};
inline cuDoubleComplex make_cuDoubleComplex(double x,double y){return {x,y};}
'''
blas = r'''
#pragma once
#include "cuda_runtime.h"
#include "cuComplex.h"
#include <complex>
#include <type_traits>
enum cublasStatus_t {CUBLAS_STATUS_SUCCESS=0,CUBLAS_STATUS_ALLOC_FAILED=3,CUBLAS_STATUS_INVALID_VALUE=7,CUBLAS_STATUS_EXECUTION_FAILED=13,CUBLAS_STATUS_INTERNAL_ERROR=14};
enum cublasOperation_t {CUBLAS_OP_N,CUBLAS_OP_T,CUBLAS_OP_C};
enum cublasPointerMode_t {CUBLAS_POINTER_MODE_HOST,CUBLAS_POINTER_MODE_DEVICE};
enum cublasMath_t {CUBLAS_DEFAULT_MATH};
enum cublasAtomicsMode_t {CUBLAS_ATOMICS_NOT_ALLOWED};
struct MockBlas {int device;cudaStream_t stream=nullptr;cublasPointerMode_t pointer=CUBLAS_POINTER_MODE_HOST;
    cublasMath_t math=CUBLAS_DEFAULT_MATH;cublasAtomicsMode_t atomics=CUBLAS_ATOMICS_NOT_ALLOWED;};
using cublasHandle_t=MockBlas *;
inline cublasStatus_t cublasCreate(cublasHandle_t *h){*h=new MockBlas{mock_device};return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasDestroy(cublasHandle_t h){delete h;return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasSetStream(cublasHandle_t h,cudaStream_t s){h->stream=s;return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasGetStream(cublasHandle_t h,cudaStream_t *s){*s=h->stream;return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasGetPointerMode(cublasHandle_t h,cublasPointerMode_t *p){*p=h->pointer;return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasSetPointerMode(cublasHandle_t h,cublasPointerMode_t p){h->pointer=p;return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasGetMathMode(cublasHandle_t h,cublasMath_t *p){*p=h->math;return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasSetMathMode(cublasHandle_t h,cublasMath_t p){h->math=p;return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasGetAtomicsMode(cublasHandle_t h,cublasAtomicsMode_t *p){*p=h->atomics;return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasSetAtomicsMode(cublasHandle_t h,cublasAtomicsMode_t p){h->atomics=p;return CUBLAS_STATUS_SUCCESS;}
template<class T> std::complex<double> number(T x){if constexpr(std::is_same_v<T,double>)return {x,0};else return {x.x,x.y};}
template<class T> cublasStatus_t fakegemm(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
    const T *alpha,const T *a,int lda,const T *b,int ldb,const T *beta,T *c,int ldc){
    if(h->device!=mock_device||h->pointer!=CUBLAS_POINTER_MODE_HOST)return CUBLAS_STATUS_INVALID_VALUE;
    for(int j=0;j<n;++j)for(int i=0;i<m;++i){std::complex<double> sum=0;
        for(int p=0;p<k;++p){auto av=number(a[ta==CUBLAS_OP_N?i+size_t(p)*lda:p+size_t(i)*lda]);
            auto bv=number(b[tb==CUBLAS_OP_N?p+size_t(j)*ldb:j+size_t(p)*ldb]);
            if(ta==CUBLAS_OP_C)av=std::conj(av);if(tb==CUBLAS_OP_C)bv=std::conj(bv);sum+=av*bv;}
        auto out=number(*alpha)*sum;if(number(*beta)!=std::complex<double>(0))out+=number(*beta)*number(c[i+size_t(j)*ldc]);
        if constexpr(std::is_same_v<T,double>)c[i+size_t(j)*ldc]=out.real();else c[i+size_t(j)*ldc]={out.real(),out.imag()};}
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDgemm(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
const double *al,const double *a,int lda,const double *b,int ldb,const double *be,double *c,int ldc){return fakegemm(h,ta,tb,m,n,k,al,a,lda,b,ldb,be,c,ldc);}
inline cublasStatus_t cublasZgemm(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
const cuDoubleComplex *al,const cuDoubleComplex *a,int lda,const cuDoubleComplex *b,int ldb,const cuDoubleComplex *be,cuDoubleComplex *c,int ldc){return fakegemm(h,ta,tb,m,n,k,al,a,lda,b,ldb,be,c,ldc);}
inline cublasStatus_t cublasDaxpy(cublasHandle_t h,int n,const double *a,const double *x,int ix,double *y,int iy){
    if(h->device!=mock_device)return CUBLAS_STATUS_INVALID_VALUE;
    for(int i=0;i<n;++i)y[i*iy]+=*a*x[i*ix];return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasZaxpy(cublasHandle_t h,int n,const cuDoubleComplex *a,const cuDoubleComplex *x,int ix,cuDoubleComplex *y,int iy){
    if(h->device!=mock_device)return CUBLAS_STATUS_INVALID_VALUE;
    for(int i=0;i<n;++i){auto v=number(y[i*iy])+number(*a)*number(x[i*ix]);y[i*iy]={v.real(),v.imag()};}
    return CUBLAS_STATUS_SUCCESS;
}

'''
fixture = (root / "tests/mgpu_gemm_smoke.cu").read_text()
fixture = re.sub(r"__global__ void delay_input\([^)]*\) \{.*?\n\}", "", fixture, flags=re.S)
fixture = re.sub(r"^\s*delay_input<<<.*?;\n", "\n", fixture, flags=re.M)
fixture = fixture.replace('    openmx_mgpuGemmStagingEnd();\n    setenv("OPENMX_MGPU_GEMM_PAGEABLE","1",1);',
    '''    require(!host_allocations.empty(),"outer scope retains staging");
    const int registered=host_allocation_calls;
    run<double>(h,devices,CUBLAS_OP_T,CUBLAS_OP_N,337,223,311);
    // run() itself allocates one test input, but workers reuse their buffers.
    require(host_allocation_calls==registered+1,"worker registrations reused");
    for(auto &entry:host_allocations) require(entry.second<=1048576,"cached host budget");
    openmx_mgpuGemmStagingEnd();
    require(host_allocations.empty(),"outer scope releases all staging");
    setenv("OPENMX_MGPU_GEMM_PAGEABLE","1",1);''')
with tempfile.TemporaryDirectory(prefix="openmx-mgpu-cpu-") as tmp:
    directory = Path(tmp)
    for name, contents in {"cuda_runtime.h": cuda, "cuComplex.h": complex_header, "cublas_v2.h": blas}.items():
        (directory / name).write_text(contents)
    (directory / "check.cpp").write_text(fixture)
    binary = directory / "check"
    subprocess.run(shlex.split(os.environ.get("CXX", "c++")) +
                   ["-std=c++17", "-O2", "-DOPENMX_MGPU_TEST_BRIDGE", "-I" + str(directory),
                    "-I" + str(root / "source"), "-x", "c++", str(directory / "check.cpp"),
                    str(root / "source/openmx_mgpu_gemm.cu"), "-o", str(binary)], check=True)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("OPENMX_MGPU_")}
    for ngpu in (2, 4):
        subprocess.run([str(binary), "--ngpu", str(ngpu)], env=environment, check=True)
