#ifndef OPENMX_AMDGPU_CUDA_RUNTIME_COMPAT_H
#define OPENMX_AMDGPU_CUDA_RUNTIME_COMPAT_H

#ifdef __cplusplus
#include <hip/hip_runtime.h>
typedef hipError_t cudaError_t;
typedef hipStream_t cudaStream_t;
#else
#include <stddef.h>
typedef int cudaError_t;
typedef struct ihipStream_t *cudaStream_t;
enum {
    hipSuccess = 0,
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4,
    hipStreamNonBlocking = 1
};
const char *hipGetErrorString(cudaError_t error);
cudaError_t hipGetDeviceCount(int *count);
cudaError_t hipGetDevice(int *device);
cudaError_t hipSetDevice(int device);
cudaError_t hipDeviceReset(void);
cudaError_t hipDeviceSynchronize(void);
cudaError_t hipStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags);
cudaError_t hipStreamDestroy(cudaStream_t stream);
cudaError_t hipStreamSynchronize(cudaStream_t stream);
cudaError_t hipMalloc(void **ptr, size_t size);
cudaError_t hipMallocAsync(void **ptr, size_t size, cudaStream_t stream);
cudaError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags);
cudaError_t hipFree(void *ptr);
cudaError_t hipFreeAsync(void *ptr, cudaStream_t stream);
cudaError_t hipHostFree(void *ptr);
cudaError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, int kind);
cudaError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes, int kind, cudaStream_t stream);
cudaError_t hipMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                             size_t width, size_t height, int kind, cudaStream_t stream);
cudaError_t hipMemGetInfo(size_t *free, size_t *total);
#endif

#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaMalloc hipMalloc
#define cudaMallocAsync hipMallocAsync
#define cudaMallocHost(ptr, size) hipHostMalloc((ptr), (size), 0)
#define cudaFree hipFree
#define cudaFreeAsync hipFreeAsync
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemGetInfo hipMemGetInfo

#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyDefault hipMemcpyDefault

typedef enum cudaDataType {
    CUDA_R_64F = 0,
    CUDA_C_64F = 1
} cudaDataType;

#endif
