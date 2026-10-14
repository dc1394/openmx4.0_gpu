#ifndef OPENMX_MGPU_RUNTIME_H
#define OPENMX_MGPU_RUNTIME_H

#include <stdint.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The only production scopes are Cluster_DFT_Col and Cluster_DFT_NonCol.
   Band, DC, LNO and the distributed ELPA/COSMA solver remain unaffected.
   Configure collectively obtained topology once per input; this function
   itself never calls MPI and never creates a CUDA context. */
void openmx_mgpu_configure(int enabled, int gpu_limit, int single_node);
int openmx_mgpu_scope_enter(int allowed);
void openmx_mgpu_scope_leave(int previous_scope);
int openmx_mgpu_scope_active(void);

/* Optional hooks keep standalone tests of the existing bridges linkable
   without the new runtime. The complete application links it explicitly. */
#if defined(__GNUC__)
#define OPENMX_MGPU_OPTIONAL __attribute__((weak))
#else
#define OPENMX_MGPU_OPTIONAL
#endif

typedef cusolverStatus_t (*openmx_mgpu_native_solve)(void *context);

/* 0 declines, preserving A for the original solve; 1 has produced status.
   Native calibration uses the caller's existing solver/workspace callback,
   so timings include exactly the same single-GPU implementation as fallback.
   Auto scores the minimum of two successful measurements per candidate for
   repeated SCF work; OPENMX_MGPU_TUNE_TRIALS=1 uses one (maximum accepted 3).
   MGPU scores include host preparation, transfers and result validation.
   Supports FP64 vectors, full spectra and leading-index subsets. */
int openmx_mgpu_solve_device_try(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
    cudaDataType type, void *A, int64_t lda, int64_t nev, double *W,
    int64_t *meig, int *info, openmx_mgpu_native_solve native_solve,
    void *native_context, cusolverStatus_t *status) OPENMX_MGPU_OPTIONAL;

int openmx_mgpu_dispatch_dgemm(cublasHandle_t handle,
    cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
    const double *alpha, const double *A, int lda, const double *B, int ldb,
    const double *beta, double *C, int ldc, cublasStatus_t *status) OPENMX_MGPU_OPTIONAL;
int openmx_mgpu_dispatch_zgemm(cublasHandle_t handle,
    cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
    const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc, cublasStatus_t *status) OPENMX_MGPU_OPTIONAL;

#undef OPENMX_MGPU_OPTIONAL
#ifdef __cplusplus
}
#endif
#endif
