#ifndef OPENMX_GPU_ERROR_H
#define OPENMX_GPU_ERROR_H

#include "openmx_gpu_backend.h"
/* openmx_common.h owns the backend's existing retry/error policy. */
#if OPENMX_GPU_HIP
#define OpenMXGpuWait(call) wait_hipfunc(call)
#else
#define OpenMXGpuWait(call) wait_cudafunc(call)
#endif

#endif /* OPENMX_GPU_ERROR_H */
