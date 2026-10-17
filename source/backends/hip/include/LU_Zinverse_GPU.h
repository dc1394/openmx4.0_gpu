#ifndef _LU_ZINVERSE_GPU_H_
#define _LU_ZINVERSE_GPU_H_

#include "hipblas_compat.h"

#pragma omp declare target
int LU_Zinverse_GPU(int n, hipDoubleComplex *A, hipblasHandle_t handle);
#pragma omp end declare target

#endif
