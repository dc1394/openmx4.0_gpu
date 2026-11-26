#ifndef OPENMX_TEST_EXX_DEVICE_SDK_CUBLAS_V2_H
#define OPENMX_TEST_EXX_DEVICE_SDK_CUBLAS_V2_H

/* Model of the part of cuBLAS that source/Exact_Exchange_Device.c uses.
   See cuda_runtime.h in this directory. */

#include "cuda_runtime.h"

typedef struct OpenMXTestBlasContext *cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;
enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_EXECUTION_FAILED = 13
};
enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 };

cublasStatus_t cublasCreate(cublasHandle_t *handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *stream);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream);
cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const double *alpha, const double *a, int lda, const double *b, int ldb,
                           const double *beta, double *c, int ldc);

#endif
