#ifndef OPENMX_AMDGPU_HIPBLAS_COMPAT_H
#define OPENMX_AMDGPU_HIPBLAS_COMPAT_H

/*
 * hipBLAS compatibility header for OpenMX's own sources.
 * C++ TUs use the real ROCm hipBLAS header; plain C TUs get the small set of
 * hipBLAS prototypes/enums OpenMX calls.  See hip_runtime_compat.h for the
 * rationale behind keeping the CUDA-named shim headers for third-party code.
 */

#include "hip_complex_compat.h"

#ifdef __cplusplus
#include <hipblas/hipblas.h>
#else
typedef struct hipblasContext *hipblasHandle_t;
typedef int hipblasStatus_t;
typedef enum {
    HIPBLAS_OP_N = 111,
    HIPBLAS_OP_T = 112,
    HIPBLAS_OP_C = 113
} hipblasOperation_t;
typedef enum {
    HIPBLAS_FILL_MODE_UPPER = 121,
    HIPBLAS_FILL_MODE_LOWER = 122,
    HIPBLAS_FILL_MODE_FULL = 123
} hipblasFillMode_t;
typedef enum {
    HIPBLAS_SIDE_LEFT = 141,
    HIPBLAS_SIDE_RIGHT = 142,
    HIPBLAS_SIDE_BOTH = 143
} hipblasSideMode_t;
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
typedef struct ihipStream_t *hipStream_t;
hipblasStatus_t hipblasCreate(hipblasHandle_t *handle);
hipblasStatus_t hipblasDestroy(hipblasHandle_t handle);
hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId);
hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t *streamId);
hipblasStatus_t hipblasDgemm(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
                             int m, int n, int k, const double *alpha, const double *A, int lda,
                             const double *B, int ldb, const double *beta, double *C, int ldc);
hipblasStatus_t hipblasDsymm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                             int m, int n, const double *alpha, const double *A, int lda,
                             const double *B, int ldb, const double *beta, double *C, int ldc);
hipblasStatus_t hipblasZgemm(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
                             int m, int n, int k, const hipDoubleComplex *alpha, const hipDoubleComplex *A, int lda,
                             const hipDoubleComplex *B, int ldb, const hipDoubleComplex *beta, hipDoubleComplex *C, int ldc);
hipblasStatus_t hipblasZdgmm(hipblasHandle_t handle, hipblasSideMode_t mode, int m, int n,
                             const hipDoubleComplex *A, int lda, const hipDoubleComplex *x, int incx,
                             hipDoubleComplex *C, int ldc);
hipblasStatus_t hipblasDdgmm(hipblasHandle_t handle, hipblasSideMode_t mode, int m, int n,
                             const double *A, int lda, const double *x, int incx,
                             double *C, int ldc);
hipblasStatus_t hipblasZgetrfBatched(hipblasHandle_t handle, const int n, hipDoubleComplex *const A[],
                                     const int lda, int *P, int *INFO, const int batchCount);
hipblasStatus_t hipblasZgetriBatched(hipblasHandle_t handle, const int n, hipDoubleComplex *const A[],
                                     const int lda, const int *P, hipDoubleComplex *const C[], const int ldc,
                                     int *INFO, const int batchCount);
#endif

#endif
