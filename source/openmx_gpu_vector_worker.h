#ifndef OPENMX_GPU_VECTOR_WORKER_H
#define OPENMX_GPU_VECTOR_WORKER_H
#include "openmx_gpu_vector.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Internal bridge. Inputs remain borrowed for this call; the worker owns
   every device arena and unpublished host result. */
int openmx_gpu_vector_local(OpenMXGpuVector *job);
#ifdef __cplusplus
}
#endif
#endif
