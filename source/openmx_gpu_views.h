#ifndef OPENMX_GPU_VIEWS_H
#define OPENMX_GPU_VIEWS_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Read-only peer access when available; otherwise an owned device copy.
   CUDA contexts and inputs stay owned by their original caller. */
typedef struct { void *pointer; int allocated; } OpenMXGpuReadView;
int openmx_gpu_read_view(OpenMXGpuReadView *view,const void *source,size_t bytes);
void openmx_gpu_read_view_release(OpenMXGpuReadView *view);
/* Explicit cross-device transfer, with bounded host staging when necessary.
   Call only after producer kernels complete. Neither CUDA nor OpenACC
   binding changes on return, including a failed copy. */
int openmx_gpu_copy_device(void *destination,const void *source,size_t bytes);
#ifdef __cplusplus
}
#endif
#endif
