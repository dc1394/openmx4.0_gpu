#ifndef OPENMX_TEST_EXX_DEVICE_SDK_CUDA_RUNTIME_H
#define OPENMX_TEST_EXX_DEVICE_SDK_CUDA_RUNTIME_H

/* Model of the part of the CUDA runtime that source/Exact_Exchange_Device.c
   uses. The functions are defined in tests/exx_device_product_harness.c:
   device memory is host memory that the harness keeps apart and counts. */

#include <stddef.h>

typedef int cudaError_t;
/* a stream of the model: the queue of OpenACC has one (openacc.h) */
typedef struct OpenMXTestStream *cudaStream_t;
enum {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorNoDevice = 100,
  cudaErrorUnknown = 999
};
enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
};
enum { cudaStreamNonBlocking = 1 };

cudaError_t cudaGetDevice(int *device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaMalloc(void **pointer, size_t bytes);
cudaError_t cudaFree(void *pointer);
cudaError_t cudaMemcpyAsync(void *destination, const void *source, size_t bytes, enum cudaMemcpyKind kind,
                            cudaStream_t stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaGetLastError(void);

#endif
