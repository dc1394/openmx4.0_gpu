#ifndef _LU_ZINVERSE_GPU_H_
#define _LU_ZINVERSE_GPU_H_

#include "openmx_gpu_backend.h"
#include "openmx_gpu_blas.h"
#include "openmx_gpu_complex.h"

#if OPENMX_GPU_HIP
#pragma omp declare target
#else
#pragma acc routine
#endif
int LU_Zinverse_GPU(int n, OpenMXGpuComplex *A, OpenMXBlasHandle handle);
#if OPENMX_GPU_HIP
#pragma omp end declare target
#endif

#endif
