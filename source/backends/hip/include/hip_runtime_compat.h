#ifndef OPENMX_AMDGPU_HIP_RUNTIME_COMPAT_H
#define OPENMX_AMDGPU_HIP_RUNTIME_COMPAT_H

/*
 * HIP runtime compatibility header for OpenMX's own sources.
 *
 * C++ translation units pull in the real ROCm HIP runtime.  Plain C
 * translation units cannot include <hip/hip_runtime.h> (it is C++), so this
 * header declares the small set of HIP runtime entry points OpenMX calls,
 * with C linkage-compatible prototypes.
 *
 * NOTE: the CUDA-named shim headers (cuda_runtime.h, cublas_v2.h, ...) are
 * intentionally kept in the tree because the bundled third-party libraries
 * (GEMMul8, MAGMA) still include them by their CUDA names.  OpenMX's own code
 * uses the hip* names directly through this header instead.
 */

#ifdef __cplusplus
#include <hip/hip_runtime.h>
#else
#include <stddef.h>
typedef int hipError_t;
typedef struct ihipStream_t *hipStream_t;
enum {
    hipSuccess = 0,
    hipErrorNoDevice = 100,
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4,
    hipStreamNonBlocking = 1
};
const char *hipGetErrorString(hipError_t error);
hipError_t hipGetDeviceCount(int *count);
hipError_t hipGetDevice(int *device);
hipError_t hipDeviceGetPCIBusId(char *pciBusId, int len, int device);
hipError_t hipSetDevice(int device);
hipError_t hipDeviceReset(void);
hipError_t hipDeviceSynchronize(void);
hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags);
hipError_t hipStreamDestroy(hipStream_t stream);
hipError_t hipStreamSynchronize(hipStream_t stream);
hipError_t hipMalloc(void **ptr, size_t size);
hipError_t hipMallocAsync(void **ptr, size_t size, hipStream_t stream);
hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags);
hipError_t hipFree(void *ptr);
hipError_t hipFreeAsync(void *ptr, hipStream_t stream);
hipError_t hipHostFree(void *ptr);
hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, int kind);
hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes, int kind, hipStream_t stream);
hipError_t hipMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                            size_t width, size_t height, int kind, hipStream_t stream);
hipError_t hipMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                       size_t width, size_t height, int kind);
hipError_t hipGetLastError(void);
hipError_t hipMemset(void *dst, int value, size_t sizeBytes);
hipError_t hipMemsetAsync(void *dst, int value, size_t sizeBytes, hipStream_t stream);
hipError_t hipMemGetInfo(size_t *free, size_t *total);

/* The values of ROCm's hip/library_types.h: a C translation unit may hand
   these to C++ code compiled against the real header. */
typedef enum hipDataType {
    HIP_R_32F = 0,
    HIP_R_64F = 1,
    HIP_C_32F = 4,
    HIP_C_64F = 5
} hipDataType;
#endif

#endif
