#ifndef OPENMX_AMDGPU_CUBLAS_V2_COMPAT_H
#define OPENMX_AMDGPU_CUBLAS_V2_COMPAT_H

#include "cuComplex.h"

#ifdef __cplusplus
#include <hipblas/hipblas.h>
typedef hipblasHandle_t cublasHandle_t;
typedef hipblasStatus_t cublasStatus_t;
typedef hipblasOperation_t cublasOperation_t;
typedef hipblasFillMode_t cublasFillMode_t;
typedef hipblasSideMode_t cublasSideMode_t;
#else
typedef struct hipblasContext *cublasHandle_t;
typedef int cublasStatus_t;
typedef enum {
    HIPBLAS_OP_N = 111,
    HIPBLAS_OP_T = 112,
    HIPBLAS_OP_C = 113
} cublasOperation_t;
typedef enum {
    HIPBLAS_FILL_MODE_UPPER = 121,
    HIPBLAS_FILL_MODE_LOWER = 122,
    HIPBLAS_FILL_MODE_FULL = 123
} cublasFillMode_t;
typedef enum {
    HIPBLAS_SIDE_LEFT = 141,
    HIPBLAS_SIDE_RIGHT = 142,
    HIPBLAS_SIDE_BOTH = 143
} cublasSideMode_t;
enum {
    HIPBLAS_STATUS_SUCCESS = 0,
    HIPBLAS_STATUS_NOT_INITIALIZED = 1,
    HIPBLAS_STATUS_ALLOC_FAILED = 2,
    HIPBLAS_STATUS_INVALID_VALUE = 3,
    HIPBLAS_STATUS_ARCH_MISMATCH = 4,
    HIPBLAS_STATUS_MAPPING_ERROR = 5,
    HIPBLAS_STATUS_EXECUTION_FAILED = 6,
    HIPBLAS_STATUS_INTERNAL_ERROR = 7,
    HIPBLAS_STATUS_NOT_SUPPORTED = 8
};
struct ihipStream_t;
typedef struct ihipStream_t *cudaStream_t;
cublasStatus_t hipblasCreate(cublasHandle_t *handle);
cublasStatus_t hipblasDestroy(cublasHandle_t handle);
cublasStatus_t hipblasSetStream(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t hipblasGetStream(cublasHandle_t handle, cudaStream_t *streamId);
cublasStatus_t hipblasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k, const double *alpha, const double *A, int lda,
                            const double *B, int ldb, const double *beta, double *C, int ldc);
cublasStatus_t hipblasDsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                            int m, int n, const double *alpha, const double *A, int lda,
                            const double *B, int ldb, const double *beta, double *C, int ldc);
cublasStatus_t hipblasZgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
cublasStatus_t hipblasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                            const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,
                            cuDoubleComplex *C, int ldc);
cublasStatus_t hipblasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                            const double *A, int lda, const double *x, int incx,
                            double *C, int ldc);
cublasStatus_t hipblasZgetrfBatched(cublasHandle_t handle, const int n, cuDoubleComplex *const A[],
                                    const int lda, int *P, int *INFO, const int batchCount);
cublasStatus_t hipblasZgetriBatched(cublasHandle_t handle, const int n, cuDoubleComplex *const A[],
                                    const int lda, const int *P, cuDoubleComplex *const C[], const int ldc,
                                    int *INFO, const int batchCount);
#endif

#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_INTERNAL_ERROR

#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_OP_C HIPBLAS_OP_C

#define CUBLAS_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER

#define CUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define CUBLAS_SIDE_RIGHT HIPBLAS_SIDE_RIGHT

#define cublasCreate hipblasCreate
#define cublasCreate_v2 hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasDestroy_v2 hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasGetStream hipblasGetStream
#define cublasDgemm hipblasDgemm
#define cublasDgemm_v2 hipblasDgemm
#define cublasDsymm hipblasDsymm
#define cublasDsymm_v2 hipblasDsymm
#define cublasZgemm hipblasZgemm
#define cublasZgemm_v2 hipblasZgemm
#define cublasZdgmm hipblasZdgmm
#define cublasDdgmm hipblasDdgmm
#define cublasZgetrfBatched hipblasZgetrfBatched
#define cublasZgetriBatched hipblasZgetriBatched

#endif
