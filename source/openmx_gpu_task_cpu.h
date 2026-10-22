#ifndef OPENMX_GPU_TASK_CPU_H
#define OPENMX_GPU_TASK_CPU_H
#include "openmx_gpu_task.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Internal host recovery. Borrows the task and views for this call; each
   chunk owns its CSR/reduction/scratch arena. The kernel overwrites outputs
   and reductions are accumulated in chunk order, as in the C controller. */
void openmx_gpu_task_cpu(OpenMXGpuTask *task);
#ifdef __cplusplus
}
#endif
#endif
