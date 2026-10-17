#ifndef OPENMX_MGPU_GEMM_H
#define OPENMX_MGPU_GEMM_H

#include "hipblas_compat.h"
#include "hip_complex_compat.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The caller owns node/rank admission and any interprocess GPU lock. Device
 * IDs are HIP-visible ordinals, must name distinct physical GPUs, and are
 * never inferred from MPI. Inputs and C reside on the caller's current GPU.
 *
 * Return 0: declined BEFORE changing C; call the original single-GPU path.
 * Return 1: handled; inspect *status (never retry a failed commit with beta).
 * Success is synchronous, orders after handle's stream, and restores the
 * current device. The caller's handle and its settings remain unchanged.
 * All op(A)/op(B) N/T/C combinations and padded column-major strides work.
 * Output tiles retain the complete K dimension; there is no split-K sum.
 */
int openmx_mgpuDgemmTry(hipblasHandle_t handle, int ngpu, const int *devices,
                       hipblasOperation_t transa, hipblasOperation_t transb,
                       int m, int n, int k, const double *alpha,
                       const double *a, int lda, const double *b, int ldb,
                       const double *beta, double *c, int ldc, hipblasStatus_t *status);
int openmx_mgpuZgemmTry(hipblasHandle_t handle, int ngpu, const int *devices,
                       hipblasOperation_t transa, hipblasOperation_t transb,
                       int m, int n, int k, const hipDoubleComplex *alpha,
                       const hipDoubleComplex *a, int lda, const hipDoubleComplex *b, int ldb,
                       const hipDoubleComplex *beta, hipDoubleComplex *c, int ldc, hipblasStatus_t *status);

/* Capacity path: A/B/C and alpha/beta are host pointers. Only bounded
 * operand/output tiles reside on each worker GPU; the transaction is in
 * host memory. The same untouched-C/handled-error contract applies. */
int openmx_mgpuDgemmHostTry(int ngpu, const int *devices,
    hipblasOperation_t ta, hipblasOperation_t tb, int m, int n, int k,
    const double *alpha, const double *a, int lda, const double *b, int ldb,
    const double *beta, double *c, int ldc, hipblasStatus_t *status);
int openmx_mgpuZgemmHostTry(int ngpu, const int *devices,
    hipblasOperation_t ta, hipblasOperation_t tb, int m, int n, int k,
    const hipDoubleComplex *alpha, const hipDoubleComplex *a, int lda,
    const hipDoubleComplex *b, int ldb, const hipDoubleComplex *beta,
    hipDoubleComplex *c, int ldc, hipblasStatus_t *status);

/* Existing bridge entry points must skip multi-GPU dispatch while this is
 * true. Workers may call that bridge to preserve GEMMul8 precision controls.
 * Weak bridge symbols permit native-hipBLAS-only standalone linking too. */
int openmx_mgpuGemmInWorker(void);

typedef struct {
    int requested_gpus, active_gpus, tiles, bridge_calls;
    unsigned long long peer_transfers, host_transfers;
    size_t transaction_bytes, worker_bytes;
    double seconds;
    const char *reason; /* static string, valid until process exit */
} openmx_mgpu_gemm_report;

/* Per host thread; copied out so the caller can profile without parsing logs. */
void openmx_mgpuGemmLastReport(openmx_mgpu_gemm_report *report);
/* Release this host thread's cached streams/handles, outside an operation.
 * Operand/output scratch is always freed before Try returns. */
void openmx_mgpuGemmRelease(void);

#ifdef __cplusplus
}
#endif
#endif
