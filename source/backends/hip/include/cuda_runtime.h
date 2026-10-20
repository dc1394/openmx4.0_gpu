#ifndef OPENMX_AMDGPU_CUDA_RUNTIME_COMPAT_H
#define OPENMX_AMDGPU_CUDA_RUNTIME_COMPAT_H

#include "hip_runtime_compat.h"
typedef hipError_t cudaError_t;
typedef hipStream_t cudaStream_t;

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

typedef hipDataType cudaDataType;
#define CUDA_R_64F HIP_R_64F
#define CUDA_C_64F HIP_C_64F

#endif
