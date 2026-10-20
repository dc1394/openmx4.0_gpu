#ifndef OPENMX_AMDGPU_HIP_RUNTIME_COMPAT_H
#define OPENMX_AMDGPU_HIP_RUNTIME_COMPAT_H

/* Use the SDK declarations in C and C++; do not duplicate the vendor ABI. */
#ifdef __cplusplus
#include <hip/hip_runtime.h>
#else
#include <hip/hip_runtime_api.h>
#endif

#endif
