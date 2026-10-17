#!/usr/bin/env python3
"""Run the production tiled GEMM and numerical fixture on fake HIP devices.

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
from mock_hip_sources import stage

root = Path(__file__).resolve().parents[1]
hip = r'''
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unordered_map>
using hipError_t=int;
constexpr int hipSuccess=0,hipStreamNonBlocking=1,hipEventDisableTiming=2,hipHostAllocPortable=4;
constexpr int hipErrorPeerAccessAlreadyEnabled=5;
enum hipMemoryType {hipMemoryTypeHost=1,hipMemoryTypeDevice=2};
enum hipMemcpyKind {hipMemcpyHostToDevice,hipMemcpyDeviceToHost,hipMemcpyDeviceToDevice};
enum hipStreamCaptureStatus {hipStreamCaptureStatusNone};
struct MockStream {int device;};using hipStream_t=MockStream *;
struct MockEvent {int device;};using hipEvent_t=MockEvent *;
inline int mock_device=0;
struct Allocation {size_t bytes;int device;};
inline std::unordered_map<void *,Allocation> allocations;
struct hipPointerAttribute_t {hipMemoryType type;int device;};
struct hipPitchedPtr {void *ptr;size_t pitch,xsize,ysize;};
struct hipExtent {size_t width,height,depth;};
struct hipMemcpy3DPeerParms {hipPitchedPtr srcPtr,dstPtr;int srcDevice,dstDevice;hipExtent extent;};
inline hipPitchedPtr make_hipPitchedPtr(void *p,size_t pitch,size_t w,size_t h){return {p,pitch,w,h};}
inline hipExtent make_hipExtent(size_t w,size_t h,size_t d){return {w,h,d};}
inline int hipGetDevice(int *v){*v=mock_device;return 0;}
inline int hipSetDevice(int v){if(v<0||v>=4)return 1;mock_device=v;return 0;}
inline int hipGetDeviceCount(int *v){*v=4;return 0;}
inline int hipDeviceGetPCIBusId(char *s,int n,int d){std::snprintf(s,n,"0000:%02x:00.0",d);return 0;}
inline int hipDeviceCanAccessPeer(int *can,int,int){*can=1;return 0;}
inline int hipDeviceEnablePeerAccess(int,int){return 0;}
inline int hipGetLastError(){return 0;}
inline const char *hipGetErrorString(int){return "mock HIP error";}
inline int hipMemGetInfo(size_t *f,size_t *t){*f=*t=size_t(16)<<30;return 0;}
template<class T> inline int hipMalloc(T **p,size_t n){*p=static_cast<T *>(std::malloc(n));if(!*p)return 1;allocations[*p]={n,mock_device};return 0;}
inline int hipFree(void *p){if(!p)return 0;auto it=allocations.find(p);if(it==allocations.end()||it->second.device!=mock_device)return 1;allocations.erase(it);std::free(p);return 0;}
inline int hipHostAlloc(void **p,size_t n,int){*p=std::malloc(n);return *p?0:1;}
inline int hipFreeHost(void *p){std::free(p);return 0;}
inline int hipPointerGetAttributes(hipPointerAttribute_t *a,const void *p){
    for(auto &x:allocations){const auto base=reinterpret_cast<uintptr_t>(x.first),q=reinterpret_cast<uintptr_t>(p);
        if(q>=base&&q-base<x.second.bytes){a->type=hipMemoryTypeDevice;a->device=x.second.device;return 0;}}
    return 1;
}
inline int hipStreamCreateWithFlags(hipStream_t *s,int){*s=new MockStream{mock_device};return 0;}
inline int hipStreamSynchronize(hipStream_t s){return s&&s->device!=mock_device?1:0;}
inline int hipStreamDestroy(hipStream_t s){if(s&&s->device!=mock_device)return 1;delete s;return 0;}
inline int hipStreamIsCapturing(hipStream_t,hipStreamCaptureStatus *v){*v=hipStreamCaptureStatusNone;return 0;}
inline int hipEventCreateWithFlags(hipEvent_t *e,int){*e=new MockEvent{mock_device};return 0;}
inline int hipEventRecord(hipEvent_t e,hipStream_t s){return e->device!=mock_device||(s&&s->device!=mock_device);}
inline int hipEventDestroy(hipEvent_t e){if(e->device!=mock_device)return 1;delete e;return 0;}
inline int hipStreamWaitEvent(hipStream_t s,hipEvent_t,int){return s&&s->device!=mock_device;}
inline int hipMemcpy(void *d,const void *s,size_t n,hipMemcpyKind){std::memcpy(d,s,n);return 0;}
inline int hipMemcpyAsync(void *d,const void *s,size_t n,hipMemcpyKind k,hipStream_t st){if(st&&st->device!=mock_device)return 1;return hipMemcpy(d,s,n,k);}
inline int hipMemcpy2DAsync(void *d,size_t dp,const void *s,size_t sp,size_t w,size_t h,hipMemcpyKind,hipStream_t st){
    if(st&&st->device!=mock_device)return 1;if(w>dp||w>sp)return 1;
    for(size_t j=0;j<h;++j)std::memcpy(static_cast<char *>(d)+j*dp,static_cast<const char *>(s)+j*sp,w);return 0;
}
inline int hipMemcpy3DPeerAsync(const hipMemcpy3DPeerParms *p,hipStream_t st){
    return hipMemcpy2DAsync(p->dstPtr.ptr,p->dstPtr.pitch,p->srcPtr.ptr,p->srcPtr.pitch,p->extent.width,p->extent.height,hipMemcpyDeviceToDevice,st);
}
'''
complex_header = r'''
#pragma once
struct hipDoubleComplex {double x,y;};
inline hipDoubleComplex make_hipDoubleComplex(double x,double y){return {x,y};}
'''
blas = r'''
#pragma once
#include "hip_runtime_compat.h"
#include "hip_complex_compat.h"
#include <complex>
#include <type_traits>
enum hipblasStatus_t {HIPBLAS_STATUS_SUCCESS=0,HIPBLAS_STATUS_ALLOC_FAILED=3,HIPBLAS_STATUS_INVALID_VALUE=7,HIPBLAS_STATUS_EXECUTION_FAILED=13,HIPBLAS_STATUS_INTERNAL_ERROR=14};
enum hipblasOperation_t {HIPBLAS_OP_N,HIPBLAS_OP_T,HIPBLAS_OP_C};
enum hipblasPointerMode_t {HIPBLAS_POINTER_MODE_HOST,HIPBLAS_POINTER_MODE_DEVICE};
enum hipblasMath_t {HIPBLAS_DEFAULT_MATH};
enum hipblasAtomicsMode_t {HIPBLAS_ATOMICS_NOT_ALLOWED};
struct MockBlas {int device;hipStream_t stream=nullptr;hipblasPointerMode_t pointer=HIPBLAS_POINTER_MODE_HOST;
    hipblasMath_t math=HIPBLAS_DEFAULT_MATH;hipblasAtomicsMode_t atomics=HIPBLAS_ATOMICS_NOT_ALLOWED;};
using hipblasHandle_t=MockBlas *;
inline hipblasStatus_t hipblasCreate(hipblasHandle_t *h){*h=new MockBlas{mock_device};return HIPBLAS_STATUS_SUCCESS;}
inline hipblasStatus_t hipblasDestroy(hipblasHandle_t h){delete h;return HIPBLAS_STATUS_SUCCESS;}
inline hipblasStatus_t hipblasSetStream(hipblasHandle_t h,hipStream_t s){h->stream=s;return HIPBLAS_STATUS_SUCCESS;}
inline hipblasStatus_t hipblasGetStream(hipblasHandle_t h,hipStream_t *s){*s=h->stream;return HIPBLAS_STATUS_SUCCESS;}
inline hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t h,hipblasPointerMode_t *p){*p=h->pointer;return HIPBLAS_STATUS_SUCCESS;}
inline hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t h,hipblasPointerMode_t p){h->pointer=p;return HIPBLAS_STATUS_SUCCESS;}
inline hipblasStatus_t hipblasGetMathMode(hipblasHandle_t h,hipblasMath_t *p){*p=h->math;return HIPBLAS_STATUS_SUCCESS;}
inline hipblasStatus_t hipblasSetMathMode(hipblasHandle_t h,hipblasMath_t p){h->math=p;return HIPBLAS_STATUS_SUCCESS;}
inline hipblasStatus_t hipblasGetAtomicsMode(hipblasHandle_t h,hipblasAtomicsMode_t *p){*p=h->atomics;return HIPBLAS_STATUS_SUCCESS;}
inline hipblasStatus_t hipblasSetAtomicsMode(hipblasHandle_t h,hipblasAtomicsMode_t p){h->atomics=p;return HIPBLAS_STATUS_SUCCESS;}
template<class T> std::complex<double> number(T x){if constexpr(std::is_same_v<T,double>)return {x,0};else return {x.x,x.y};}
template<class T> hipblasStatus_t fakegemm(hipblasHandle_t h,hipblasOperation_t ta,hipblasOperation_t tb,int m,int n,int k,
    const T *alpha,const T *a,int lda,const T *b,int ldb,const T *beta,T *c,int ldc){
    if(h->device!=mock_device||h->pointer!=HIPBLAS_POINTER_MODE_HOST)return HIPBLAS_STATUS_INVALID_VALUE;
    for(int j=0;j<n;++j)for(int i=0;i<m;++i){std::complex<double> sum=0;
        for(int p=0;p<k;++p){auto av=number(a[ta==HIPBLAS_OP_N?i+size_t(p)*lda:p+size_t(i)*lda]);
            auto bv=number(b[tb==HIPBLAS_OP_N?p+size_t(j)*ldb:j+size_t(p)*ldb]);
            if(ta==HIPBLAS_OP_C)av=std::conj(av);if(tb==HIPBLAS_OP_C)bv=std::conj(bv);sum+=av*bv;}
        auto out=number(*alpha)*sum;if(number(*beta)!=std::complex<double>(0))out+=number(*beta)*number(c[i+size_t(j)*ldc]);
        if constexpr(std::is_same_v<T,double>)c[i+size_t(j)*ldc]=out.real();else c[i+size_t(j)*ldc]={out.real(),out.imag()};}
    return HIPBLAS_STATUS_SUCCESS;
}
inline hipblasStatus_t hipblasDgemm(hipblasHandle_t h,hipblasOperation_t ta,hipblasOperation_t tb,int m,int n,int k,
const double *al,const double *a,int lda,const double *b,int ldb,const double *be,double *c,int ldc){return fakegemm(h,ta,tb,m,n,k,al,a,lda,b,ldb,be,c,ldc);}
inline hipblasStatus_t hipblasZgemm(hipblasHandle_t h,hipblasOperation_t ta,hipblasOperation_t tb,int m,int n,int k,
const hipDoubleComplex *al,const hipDoubleComplex *a,int lda,const hipDoubleComplex *b,int ldb,const hipDoubleComplex *be,hipDoubleComplex *c,int ldc){return fakegemm(h,ta,tb,m,n,k,al,a,lda,b,ldb,be,c,ldc);}
'''
fixture = (root / "tests/mgpu_gemm_smoke.cu").read_text()
fixture = re.sub(r"__global__ void delay_input\([^)]*\) \{.*?\n\}", "", fixture, flags=re.S)
fixture = re.sub(r"^\s*delay_input<<<.*?;\n", "\n", fixture, flags=re.M)
with tempfile.TemporaryDirectory(prefix="openmx-mgpu-cpu-") as tmp:
    directory = Path(tmp)
    for name, contents in {"hip_runtime_compat.h": hip, "hip_complex_compat.h": complex_header, "hipblas_compat.h": blas}.items():
        (directory / name).write_text(contents)
    (directory / "check.cpp").write_text(fixture)
    stage(root,directory,"openmx_mgpu_gemm.cu")
    binary = directory / "check"
    subprocess.run(shlex.split(os.environ.get("CXX", "c++")) +
                   ["-std=c++17", "-O2", "-DOPENMX_MGPU_TEST_BRIDGE", "-I" + str(directory),
                    "-I" + str(root / "source"), "-x", "c++", str(directory / "check.cpp"),
                    str(directory / "openmx_mgpu_gemm.cu"), "-o", str(binary)], check=True)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("OPENMX_MGPU_")}
    for ngpu in (2, 4):
        subprocess.run([str(binary), "--ngpu", str(ngpu)], env=environment, check=True)
