#ifndef OPENMX_GPU_COMPLEX_H
#define OPENMX_GPU_COMPLEX_H

#include "openmx_gpu_backend.h"

/* Compile-time aliases preserve the native ABI, status codes and ownership.
   Backend policies (memory accounting, solver retries and offload) stay
   in their implementations. No device is initialized by this header. */
#if OPENMX_GPU_HIP
#include "hip_complex_compat.h"
#define OpenMXGpuComplex hipDoubleComplex
#define OpenMXGpuMakeComplex make_hipDoubleComplex
#else
#include <cuComplex.h>
#define OpenMXGpuComplex cuDoubleComplex
#define OpenMXGpuMakeComplex make_cuDoubleComplex
#endif

#endif /* OPENMX_GPU_COMPLEX_H */
