"""Host SDK stubs shared by GEMMul8 policy and cache tests.

These stubs record dispatch; they do not implement numerical GEMMs.
"""
HEADERS = {
    "cuda_runtime_api.h": '#include "cuda_runtime.h"\n',
    "cusolverDn.h": r'''
#pragma once
using cusolverDnHandle_t = void *;
using cusolverStatus_t = int;
using cusolverEigMode_t = int;
using cudaDataType = int;
using cublasFillMode_t = int;
''',
    "cuda_runtime.h": r'''
#pragma once
#include <cstdlib>
#include <cstddef>
using cudaStream_t = void *;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaDevAttrSingleToDoublePrecisionPerfRatio = 1,
    cudaDevAttrComputeCapabilityMajor=2, cudaDevAttrComputeCapabilityMinor=3, cudaDevAttrMultiProcessorCount=4;
constexpr int cudaErrorInvalidValue=1,cudaErrorMemoryAllocation=2;
inline int mock_device=0, mock_ratio[8]={2,64,0,2,2,2,2,2}, mock_attr_calls=0;
inline int mock_major[8]={8,8,0,8,9,8,8,8},mock_minor[8]={0,9,0,0,0,0,0,0};
inline int mock_sms[8]={108,128,0,108,132,56,14,108};
inline int mock_device_error=0, mock_attr_error=0, mock_allocs=0, mock_memqueries=0;
inline int mock_memerror=0,mock_memfail_at=0,mock_allocerror=0;
inline size_t mock_free=size_t(40)<<30,mock_total=size_t(40)<<30;
inline int cudaGetDevice(int *d) {*d=mock_device;return mock_device_error;}
inline int cudaSetDevice(int d) {mock_device=d;return 0;}
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
