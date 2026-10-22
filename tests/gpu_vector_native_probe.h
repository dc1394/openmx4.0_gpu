#ifndef OPENMX_GPU_VECTOR_NATIVE_PROBE_H
#define OPENMX_GPU_VECTOR_NATIVE_PROBE_H
#include "openmx_gpu_vector.h"
#include "openmx_gpu_workers.h"
#ifdef __cplusplus
extern "C" {
#endif
void *openmx_test_vector_create(OpenMXGpuVector*);
int openmx_test_vector_phase(void*,int);
void *openmx_test_vector_result(void*);
void *openmx_test_vector_arena(void*);
void openmx_test_vector_destroy(void*);
#ifdef __cplusplus
}
#endif
#endif
