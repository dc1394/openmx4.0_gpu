#ifndef OPENMX_MGPU_GEMM_H
#define OPENMX_MGPU_GEMM_H

#include "openmx_gpu_backend.h"
#include "openmx_gpu_blas.h"
#include "openmx_gpu_complex.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The caller owns node/rank admission and any interprocess GPU lock. Device
 * IDs are CUDA-visible ordinals, must name distinct physical GPUs, and are
 * never inferred from MPI. Inputs and C reside on the caller's current GPU.
 *
 * Return 0: declined BEFORE changing C; call the original single-GPU path.
 * Return 1: handled; inspect *status (never retry a failed commit with beta).
 * Success is synchronous, orders after handle's stream, and restores the
 * current device. The caller's handle and its settings remain unchanged.
 * All op(A)/op(B) N/T/C combinations and padded column-major strides work.
 * Output tiles retain the complete K dimension; there is no split-K sum.
 */
int openmx_mgpuDgemmTry(OpenMXBlasHandle handle, int ngpu, const int *devices,
                       OpenMXBlasOperation transa, OpenMXBlasOperation transb,
                       int m, int n, int k, const double *alpha,
                       const double *a, int lda, const double *b, int ldb,
                       const double *beta, double *c, int ldc, OpenMXBlasStatus *status);
int openmx_mgpuZgemmTry(OpenMXBlasHandle handle, int ngpu, const int *devices,
                       OpenMXBlasOperation transa, OpenMXBlasOperation transb,
                       int m, int n, int k, const OpenMXGpuComplex *alpha,
                       const OpenMXGpuComplex *a, int lda, const OpenMXGpuComplex *b, int ldb,
                       const OpenMXGpuComplex *beta, OpenMXGpuComplex *c, int ldc, OpenMXBlasStatus *status);

/* Capacity path: A/B/C and alpha/beta are host pointers. Only bounded
 * operand/output tiles reside on each worker GPU; the transaction is in
 * host memory. The same untouched-C/handled-error contract applies. */
int openmx_mgpuDgemmHostTry(int ngpu, const int *devices,
    OpenMXBlasOperation ta, OpenMXBlasOperation tb, int m, int n, int k,
    const double *alpha, const double *a, int lda, const double *b, int ldb,
    const double *beta, double *c, int ldc, OpenMXBlasStatus *status);
int openmx_mgpuZgemmHostTry(int ngpu, const int *devices,
    OpenMXBlasOperation ta, OpenMXBlasOperation tb, int m, int n, int k,
    const OpenMXGpuComplex *alpha, const OpenMXGpuComplex *a, int lda,
    const OpenMXGpuComplex *b, int ldb, const OpenMXGpuComplex *beta,
    OpenMXGpuComplex *c, int ldc, OpenMXBlasStatus *status);

/* Existing bridge entry points must skip multi-GPU dispatch while this is
 * true. Workers may call that bridge to preserve GEMMul8 precision controls.
 * Weak bridge symbols permit native-cuBLAS-only standalone linking too. */
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
 * Device operand/output scratch is always freed before Try returns. */
void openmx_mgpuGemmRelease(void);
/* Optional per-thread, nestable host-GEMM sequence. Reuse pinned staging
 * within the existing workspace/host budgets until the outer End. Call only
 * between synchronous Try operations. End also applies on failed sequences. */
#if !OPENMX_GPU_HIP
void openmx_mgpuGemmStagingBegin(void);
void openmx_mgpuGemmStagingEnd(void);

/* Private eigensolver workspace. Packed host matrices belong to the caller's
 * unpublished refinement transaction. Each worker retains two n*n matrices
 * and one column panel, bounded by OPENMX_MAGMA_REFINE_RESIDENT_MB (default 0).
 * Project forms H=X^H A X and G=X^H X for the first end columns; Update applies
 * X += X E using the basis retained by a full Project. Methods synchronize
 * and restore the calling device. A nonzero method result invalidates the
 * private host outputs; release and discard the refinement transaction.
 * Create returns NULL before any host output change when capacity is absent.
 * Release returns nonzero on CUDA cleanup failure. Admission is caller-owned. */
void *openmx_mgpuRefineCreate(int ngpu,const int *devices,int n,int is_complex);
int openmx_mgpuRefineProject(void *workspace,const void *a,const void *x,
                            int end,void *h,void *g);
int openmx_mgpuRefineUpdate(void *workspace,const void *e,void *x);
int openmx_mgpuRefineRelease(void *workspace);

#endif
#ifdef __cplusplus
}
#endif
#endif
