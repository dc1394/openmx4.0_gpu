/* Independent implementation of the reordered, column-block pipeline in
 * Wang et al., arXiv:2511.16174v1 (2025), sections 3 and 4.
 *
 * The single-node executor uses host threads and CUDA streams instead of
 * MPI/NVSHMEM. SBR uses compact WY/ZY, BC runs on GPUs in compact band slabs,
 * and both backtransforms act on identity before the final Qsb*Qd product.
 * This implementation uses b=32 panel blocking; the recursive double-block
 * SBR and the paper's register-tiled BC-back are not implemented here.
 * No code from the authors' unlicensed research repository is incorporated.
 */
#include "openmx_paper_mgpu.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" void dstedc_(const char *, const int *, double *, double *, double *,
                        const int *, double *, const int *, int *, const int *, int *);


namespace {
thread_local std::string paper_last_error;
struct alignas(16) PaperComplex {
    double x,y;
    PaperComplex()=default;
    __host__ __device__ PaperComplex(double r,double i=0):x(r),y(i){}
    __host__ __device__ PaperComplex operator-()const{return {-x,-y};}
    __host__ __device__ PaperComplex &operator+=(PaperComplex b){x+=b.x;y+=b.y;return *this;}
    __host__ __device__ PaperComplex &operator*=(PaperComplex b){double r=x*b.x-y*b.y;y=x*b.y+y*b.x;x=r;return *this;}
};
static_assert(sizeof(PaperComplex)==sizeof(cuDoubleComplex),"interleaved complex ABI");
static_assert(alignof(PaperComplex)==alignof(cuDoubleComplex),"CUDA complex alignment");
__host__ __device__ PaperComplex operator-(PaperComplex a,PaperComplex b){return {a.x-b.x,a.y-b.y};}
__host__ __device__ PaperComplex operator*(PaperComplex a,PaperComplex b){return {a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x};}
__host__ __device__ PaperComplex operator/(PaperComplex a,double b){return {a.x/b,a.y/b};}
__host__ __device__ PaperComplex operator/(PaperComplex a,PaperComplex b){
    if(fabs(b.x)>=fabs(b.y)){double t=b.y/b.x,d=b.x+b.y*t;return {(a.x+a.y*t)/d,(a.y-a.x*t)/d};}
    double t=b.x/b.y,d=b.y+b.x*t;return {(a.x*t+a.y)/d,(a.y*t-a.x)/d};
}
__host__ __device__ double real_value(double x){return x;}
__host__ __device__ double real_value(PaperComplex x){return x.x;}
__host__ __device__ double conj_value(double x){return x;}
__host__ __device__ PaperComplex conj_value(PaperComplex x){return {x.x,-x.y};}
__host__ __device__ double abs_value(double x){return fabs(x);}
__host__ __device__ double abs_value(PaperComplex x){return hypot(x.x,x.y);}
__host__ __device__ double abs_squared(double x){return x*x;}
__host__ __device__ double abs_squared(PaperComplex x){return x.x*x.x+x.y*x.y;}
bool finite_value(double x){return std::isfinite(x);}
bool finite_value(PaperComplex x){return std::isfinite(x.x)&&std::isfinite(x.y);}
__device__ double shuffle_xor(double x,int d){return __shfl_xor_sync(0xffffffff,x,d);}
__device__ PaperComplex shuffle_xor(PaperComplex x,int d){return {__shfl_xor_sync(0xffffffff,x.x,d),__shfl_xor_sync(0xffffffff,x.y,d)};}
__device__ double shuffle_lane(double x,int lane){return __shfl_sync(0xffffffff,x,lane);}
__device__ PaperComplex shuffle_lane(PaperComplex x,int lane){return {__shfl_sync(0xffffffff,x.x,lane),__shfl_sync(0xffffffff,x.y,lane)};}
}

#define PAPER_COMPLEX 0
#include "openmx_paper_mgpu_impl.cuh"
#undef PAPER_COMPLEX
#define PAPER_COMPLEX 1
#include "openmx_paper_mgpu_impl.cuh"
#undef PAPER_COMPLEX

extern "C" int openmx_paper_mgpu_available(void) {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess){cudaGetLastError();return 0;}return count;
}
extern "C" const char *openmx_paper_mgpu_last_error(void){return paper_last_error.c_str();}
extern "C" int openmx_paper_mgpu_dsyevdx_host(int ngpu,int n,int nev,double *a,int lda,double *w){
    return paper_real::solve(ngpu,n,nev,a,lda,w);
}
extern "C" int openmx_paper_mgpu_zheevdx_host(int ngpu,int n,int nev,void *a,int lda,double *w){
    return paper_complex::solve(ngpu,n,nev,a,lda,w);
}
