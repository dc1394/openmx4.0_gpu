#ifndef OPENMX_GPU_TASK_NATIVE_PROBE_H
#define OPENMX_GPU_TASK_NATIVE_PROBE_H
#include "openmx_gpu_task.h"
#include "openmx_gpu_workers.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Test-only access to a single actual worker, keeping C GPU kernels in C. */
void *openmx_test_task_create(OpenMXGpuTask*);
int openmx_test_task_phase(void*,int);
void *openmx_test_task_result(void*);
void openmx_test_task_destroy(void*);
#ifdef __cplusplus
}
#endif
#endif
