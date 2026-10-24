#ifndef OPENMX_GPU_DEVICE_H
#define OPENMX_GPU_DEVICE_H

#include "openmx_gpu_backend.h"

/* Compile-time aliases preserve the native ABI, status codes and ownership.
   Backend policies (memory accounting, solver retries and offload) stay
   in their implementations. No device is initialized by this header. */
#if OPENMX_GPU_HIP
#include "set_hip_default_device_from_local_rank.h"
#include "set_openmp_device_from_local_rank.h"
#define OpenMXGpuBindRank set_hip_default_device_from_local_rank
#define OpenMXGpuBindRankLocal set_hip_default_device_from_local_rank_noncollective
#define OpenMXOffloadBindRank set_openmp_nvidia_device_from_local_rank
#elif OPENMX_GPU_CUDA || OPENMX_GPU_HOST
#include "set_cuda_default_device_from_local_rank.h"
#include "set_openacc_device_from_local_rank.h"
#define OpenMXGpuBindRank set_cuda_default_device_from_local_rank
#define OpenMXGpuBindRankLocal set_cuda_default_device_from_local_rank_noncollective
#define OpenMXOffloadBindRank set_openacc_nvidia_device_from_local_rank
#else
#error "This native adapter supports CUDA/HIP only; use the portable core for other backends"
#endif

#endif /* OPENMX_GPU_DEVICE_H */
