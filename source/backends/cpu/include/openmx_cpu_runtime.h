#ifndef OPENMX_CPU_RUNTIME_H
#define OPENMX_CPU_RUNTIME_H

/* SDK-free compatibility surface for unreachable legacy device branches.
   This header is PRIVATE to OPENMX_GPU_BACKEND=HOST. Device allocation and
   numerical APIs always decline; the production CPU algorithms do the work.
   It must never shadow a real vendor SDK in a CUDA/HIP build. */
#if !defined(OPENMX_GPU_BACKEND) || OPENMX_GPU_BACKEND != 0
#error "CPU compatibility headers require OPENMX_GPU_BACKEND=0"
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int cudaError_t;
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
typedef int cudaDataType, cudaMemcpyKind, cudaStreamCaptureStatus;
enum { cudaSuccess=0, cudaErrorNoDevice=100, cudaErrorInvalidValue=1,
  cudaErrorMemoryAllocation=2, cudaErrorPeerAccessUnsupported=217,
  cudaErrorPeerAccessAlreadyEnabled=704, cudaMemoryTypeDevice=2,
  cudaMemcpyHostToHost=0, cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2,
  cudaMemcpyDeviceToDevice=3, cudaStreamNonBlocking=1, cudaEventDisableTiming=2,
  cudaHostAllocPortable=1, cudaStreamCaptureStatusNone=0,
  CUDA_R_64F=0, CUDA_C_64F=1, CUDA_R_32F=2, CUDA_C_32F=3,
  cudaDevAttrSingleToDoublePrecisionPerfRatio=0, cudaDevAttrMultiProcessorCount=1,
  cudaLimitStackSize=0, cudaFuncAttributeMaxDynamicSharedMemorySize=0 };
typedef struct { char bytes[16]; } cudaUUID_t;
struct cudaDeviceProp {
  char name[256]; size_t totalGlobalMem, sharedMemPerBlock;
  int major,minor,multiProcessorCount,singleToDoublePrecisionPerfRatio;
  int maxThreadsPerBlock, maxThreadsPerMultiProcessor, warpSize;
  cudaUUID_t uuid;
};
struct cudaPointerAttributes { int type,device; void *devicePointer,*hostPointer; };
struct cudaPitchedPtr { void *ptr; size_t pitch,xsize,ysize; };
struct cudaExtent { size_t width,height,depth; };
typedef struct cudaMemcpy3DPeerParms {
  struct cudaPitchedPtr srcPtr,dstPtr;
  int srcDevice,dstDevice;
  struct cudaExtent extent;
} cudaMemcpy3DPeerParms;
static inline struct cudaPitchedPtr make_cudaPitchedPtr(void *p,size_t pitch,size_t x,size_t y) {
  struct cudaPitchedPtr value={p,pitch,x,y};return value;
}
static inline struct cudaExtent make_cudaExtent(size_t x,size_t y,size_t z) {
  struct cudaExtent value={x,y,z};return value;
}
static inline cudaError_t cudaGetDeviceCount(int *n) { if(n)*n=0;return cudaSuccess; }
static inline cudaError_t cudaGetDevice(int *n) { if(n)*n=-1;return cudaErrorNoDevice; }
static inline cudaError_t cudaMemGetInfo(size_t *f,size_t *t) {
  if(f)*f=0;if(t)*t=0;return cudaErrorNoDevice;
}
static inline cudaError_t cudaGetLastError(void) { return cudaSuccess; }
static inline cudaError_t cudaPeekAtLastError(void) { return cudaSuccess; }
static inline const char *cudaGetErrorString(cudaError_t e) {
  return e==cudaSuccess?"success":"CPU-only build: GPU unavailable";
}
static inline cudaError_t cudaMalloc(void **p,size_t n) { (void)n;if(p)*p=NULL;return cudaErrorNoDevice; }
static inline cudaError_t cudaMallocHost(void **p,size_t n) { return cudaMalloc(p,n); }
static inline cudaError_t cudaHostAlloc(void **p,size_t n,unsigned f) { (void)f;return cudaMalloc(p,n); }
static inline cudaError_t cudaFree(void *p) { return p?cudaErrorNoDevice:cudaSuccess; }
static inline cudaError_t cudaFreeHost(void *p) { return p?cudaErrorNoDevice:cudaSuccess; }
#define cudaSetDevice(...) cudaErrorNoDevice
#define cudaDeviceReset(...) cudaSuccess
#define cudaDeviceSynchronize(...) cudaSuccess
#define cudaGetDeviceProperties(...) cudaErrorNoDevice
#define cudaDeviceGetAttribute(...) cudaErrorNoDevice
#define cudaDeviceGetPCIBusId(...) cudaErrorNoDevice
#define cudaDeviceCanAccessPeer(...) cudaErrorNoDevice
#define cudaDeviceEnablePeerAccess(...) cudaErrorNoDevice
#define cudaDeviceGetLimit(...) cudaErrorNoDevice
#define cudaDeviceSetLimit(...) cudaErrorNoDevice
#define cudaFuncSetAttribute(...) cudaErrorNoDevice
#define cudaMemcpy(...) cudaErrorNoDevice
#define cudaMemcpyAsync(...) cudaErrorNoDevice
#define cudaMemcpyPeer(...) cudaErrorNoDevice
#define cudaMemcpy2D(...) cudaErrorNoDevice
#define cudaMemcpy2DAsync(...) cudaErrorNoDevice
#define cudaMemcpy3DPeerAsync(...) cudaErrorNoDevice
#define cudaMemset(...) cudaErrorNoDevice
#define cudaMemsetAsync(...) cudaErrorNoDevice
#define cudaPointerGetAttributes(...) cudaErrorNoDevice
#define cudaStreamCreateWithFlags(...) cudaErrorNoDevice
#define cudaStreamDestroy(...) cudaSuccess
#define cudaStreamSynchronize(...) cudaSuccess
#define cudaStreamWaitEvent(...) cudaErrorNoDevice
#define cudaStreamIsCapturing(...) cudaErrorNoDevice
#define cudaEventCreateWithFlags(...) cudaErrorNoDevice
#define cudaEventRecord(...) cudaErrorNoDevice
#define cudaEventDestroy(...) cudaSuccess

typedef struct { double x,y; } cuDoubleComplex;
typedef struct { float x,y; } cuComplex;
static inline cuDoubleComplex make_cuDoubleComplex(double x,double y) { cuDoubleComplex z={x,y};return z; }
static inline double cuCreal(cuDoubleComplex a) { return a.x; }
static inline double cuCimag(cuDoubleComplex a) { return a.y; }
static inline cuDoubleComplex cuConj(cuDoubleComplex a) { return make_cuDoubleComplex(a.x,-a.y); }
static inline cuDoubleComplex cuCadd(cuDoubleComplex a,cuDoubleComplex b) { return make_cuDoubleComplex(a.x+b.x,a.y+b.y); }
static inline cuDoubleComplex cuCsub(cuDoubleComplex a,cuDoubleComplex b) { return make_cuDoubleComplex(a.x-b.x,a.y-b.y); }
static inline cuDoubleComplex cuCmul(cuDoubleComplex a,cuDoubleComplex b) { return make_cuDoubleComplex(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }

typedef void *cublasHandle_t;
typedef int cublasStatus_t,cublasOperation_t,cublasFillMode_t,cublasSideMode_t;
typedef int cublasAtomicsMode_t,cublasMath_t,cublasPointerMode_t;
enum { CUBLAS_STATUS_SUCCESS=0,CUBLAS_STATUS_NOT_SUPPORTED=15,
  CUBLAS_STATUS_NOT_INITIALIZED=1,CUBLAS_STATUS_ARCH_MISMATCH=8,
  CUBLAS_STATUS_MAPPING_ERROR=11,CUBLAS_STATUS_INTERNAL_ERROR=14,CUBLAS_STATUS_LICENSE_ERROR=16,
  CUBLAS_STATUS_ALLOC_FAILED=3,CUBLAS_STATUS_EXECUTION_FAILED=13,CUBLAS_STATUS_INVALID_VALUE=7,
  CUBLAS_OP_N=0,CUBLAS_OP_T=1,CUBLAS_OP_C=2,CUBLAS_FILL_MODE_LOWER=0,CUBLAS_FILL_MODE_UPPER=1,
  CUBLAS_SIDE_LEFT=0,CUBLAS_SIDE_RIGHT=1,CUBLAS_ATOMICS_NOT_ALLOWED=0,
  CUBLAS_DEFAULT_MATH=0,CUBLAS_PEDANTIC_MATH=2,CUBLAS_POINTER_MODE_HOST=0,CUBLAS_POINTER_MODE_DEVICE=1 };
#define cublasCreate(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasDestroy(...) CUBLAS_STATUS_SUCCESS
#define cublasDgemm(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasZgemm(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasDaxpy(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasZaxpy(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasDdgmm(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasZdgmm(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasDgeam(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasZgeam(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasZgetrfBatched(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasZgetriBatched(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasGetStream(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasSetStream(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasGetMathMode(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasSetMathMode(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasGetAtomicsMode(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasSetAtomicsMode(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasGetPointerMode(...) CUBLAS_STATUS_NOT_SUPPORTED
#define cublasSetPointerMode(...) CUBLAS_STATUS_NOT_SUPPORTED

typedef void *cusolverDnHandle_t,*cusolverDnParams_t;
typedef int cusolverStatus_t,cusolverEigMode_t,cusolverEigRange_t,cusolverAlgMode_t;
enum { CUSOLVER_STATUS_SUCCESS=0,CUSOLVER_STATUS_NOT_SUPPORTED=9,
  CUSOLVER_STATUS_NOT_INITIALIZED=1,CUSOLVER_STATUS_ARCH_MISMATCH=4,
  CUSOLVER_STATUS_MAPPING_ERROR=5,CUSOLVER_STATUS_INTERNAL_ERROR=7,
  CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8,CUSOLVER_STATUS_ZERO_PIVOT=10,
  CUSOLVER_STATUS_INVALID_LICENSE=11,CUSOLVER_STATUS_INVALID_WORKSPACE=12,
  CUSOLVER_STATUS_ALLOC_FAILED=2,CUSOLVER_STATUS_INVALID_VALUE=3,CUSOLVER_STATUS_EXECUTION_FAILED=6,
  CUSOLVER_EIG_MODE_VECTOR=1,CUSOLVER_EIG_MODE_NOVECTOR=0,CUSOLVER_EIG_RANGE_I=1,CUSOLVER_EIG_RANGE_ALL=0,
  CUSOLVER_ALG_0=0,CUSOLVER_ALG_1=1,CUSOLVERDN_SYEVD=0,
  MAJOR_VERSION=0,MINOR_VERSION=1,PATCH_LEVEL=2 };
#define CUSOLVER_VERSION 0
static inline cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *p) {
  if(p)*p=NULL;return CUSOLVER_STATUS_NOT_SUPPORTED;
}
static inline cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t h,...) { (void)h;return CUSOLVER_STATUS_NOT_SUPPORTED; }
static inline cusolverStatus_t cusolverDnXsyevd_bufferSize(cusolverDnHandle_t h,...) { (void)h;return CUSOLVER_STATUS_NOT_SUPPORTED; }
static inline cusolverStatus_t cusolverDnXsyevdx(cusolverDnHandle_t h,...) { (void)h;return CUSOLVER_STATUS_NOT_SUPPORTED; }
static inline cusolverStatus_t cusolverDnXsyevdx_bufferSize(cusolverDnHandle_t h,...) { (void)h;return CUSOLVER_STATUS_NOT_SUPPORTED; }
#define cusolverDnCreateParams(...) CUSOLVER_STATUS_NOT_SUPPORTED
#define cusolverDnDestroy(...) CUSOLVER_STATUS_SUCCESS
#define cusolverDnDestroyParams(...) CUSOLVER_STATUS_SUCCESS
#define cusolverDnSetAdvOptions(...) CUSOLVER_STATUS_NOT_SUPPORTED
#define cusolverDnGetStream(...) CUSOLVER_STATUS_NOT_SUPPORTED
#define cusolverDnSetStream(...) CUSOLVER_STATUS_NOT_SUPPORTED
#define cusolverDnDsygvd(...) CUSOLVER_STATUS_NOT_SUPPORTED
#define cusolverDnDsygvd_bufferSize(...) CUSOLVER_STATUS_NOT_SUPPORTED
#define cusolverDnZhegvd(...) CUSOLVER_STATUS_NOT_SUPPORTED
#define cusolverDnZhegvd_bufferSize(...) CUSOLVER_STATUS_NOT_SUPPORTED
#define cusolverGetProperty(...) CUSOLVER_STATUS_NOT_SUPPORTED

typedef int acc_device_t;
enum { acc_device_none=0,acc_device_default=1,acc_device_host=2,acc_device_nvidia=5 };
static inline int acc_get_num_devices(acc_device_t t) { return t==acc_device_host?1:0; }
static inline acc_device_t acc_get_device_type(void) { return acc_device_host; }
static inline int acc_get_device_num(acc_device_t t) { (void)t;return 0; }
static inline void *acc_deviceptr(void *p) { (void)p;return NULL; }
static inline void *acc_malloc(size_t n) { (void)n;return NULL; }
static inline void acc_free(void *p) { (void)p; }
#define acc_is_present(...) 0
#define acc_set_device_num(...) ((void)0)
#define acc_set_device_type(...) ((void)0)
#define acc_init(...) ((void)0)
#define acc_wait_all(...) ((void)0)
static inline void acc_clear_freelists(void) {}
#define acc_map_data(...) ((void)0)
#define acc_unmap_data(...) ((void)0)
#define acc_update_device(...) ((void)0)
#define acc_update_self(...) ((void)0)
#define acc_memcpy_from_device(...) ((void)0)
#define acc_memcpy_to_device(...) ((void)0)

#ifdef __cplusplus
template<class T> inline cudaError_t cudaMalloc(T **p,size_t n) { return cudaMalloc(reinterpret_cast<void**>(p),n); }
template<class T> inline cudaError_t cudaMallocHost(T **p,size_t n) { return cudaMalloc(reinterpret_cast<void**>(p),n); }
#endif
#endif
