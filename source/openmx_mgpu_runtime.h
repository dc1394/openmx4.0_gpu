#ifndef OPENMX_MGPU_RUNTIME_H
#define OPENMX_MGPU_RUNTIME_H

#include "openmx_gpu_backend.h"

#include <stdint.h>
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_blas.h"
#include "openmx_gpu_solver.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Production scopes cover cluster and ordinary collinear/noncollinear Band.
   DC, LNO and the distributed ELPA/COSMA solver remain unaffected.
   Configure collectively obtained topology once per input; this function
   itself never calls MPI and never creates a CUDA context. */
void openmx_mgpu_configure(int enabled, int gpu_limit, int single_node);
#if OPENMX_GPU_HIP
/* C-safe physical identity: no dependency on HIP's C++ device-property ABI. */
int openmx_gpu_physical_uuid(int device, unsigned char uuid[16]);
#endif
/* 1 when the configured run spans several nodes (single_node was 0). */
int openmx_mgpu_multi_node(void);
/* scf.Gpu.Num as configured (a per-process device cap; 0 or less: none). */
int openmx_mgpu_gpu_limit(void);
void openmx_mgpu_force_magma(int force);
int openmx_mgpu_host_required(void);
int openmx_mgpu_scope_enter(int allowed);
void openmx_mgpu_scope_leave(int previous_scope);
int openmx_mgpu_scope_active(void);

/* Noncollective support for non-diagonalization GPU workers and the
   cooperative products of a cluster owner. The pool is bounded by the same
   input and visibility limits and is the GPU set of the caller's node, in a
   multi-node run too. A scoped list uses the exact current Band group
   (single-node only). Admission includes the caller's device and shares the
   UUID locks used by dense solvers and matrix products. */
void openmx_mgpu_configure_auxiliary(int enabled);
int openmx_mgpu_pool_devices(int *devices, int capacity);
int openmx_mgpu_scope_devices(int *devices, int capacity);
void *openmx_mgpu_admit_devices(const int *devices, int count);
void openmx_mgpu_release_devices(void *admission);

/* Band groups use the actual computed k-point count, not the unreduced grid.
   G is the configured visible GPU pool, q=floor(G/K). Groups are contiguous
   [k*q,(k+1)*q), and only q*K GPUs are used. q<2 or q>8 declines. Planning
   does not require an active scope, create handles, or change devices.
   group() returns q (0 means unavailable); a null list queries only the size.
   used_gpus() lets input setup trim the pool before rank device binding. */
int openmx_mgpu_band_group(int total_kpoints, int kpoint_index,
    int *devices, int capacity);
int openmx_mgpu_band_used_gpus(int total_kpoints);
typedef struct {
    int active, band_kpoints, band_kpoint;
} openmx_mgpu_scope_state;
openmx_mgpu_scope_state openmx_mgpu_band_scope_enter(int allowed,
    int total_kpoints, int kpoint_index);
void openmx_mgpu_band_scope_leave(openmx_mgpu_scope_state previous);

/* One C caller owns at most one active k-point scope. Selecting another
   k-point closes the previous scope first, including its staging lifetime. */
typedef struct {
    openmx_mgpu_scope_state previous;
    int engaged;
} openmx_mgpu_band_scope;
static inline void openmx_mgpu_band_scope_close(openmx_mgpu_band_scope *scope)
{
    if (scope->engaged) {
        openmx_mgpu_band_scope_leave(scope->previous);
        scope->engaged = 0;
    }
}
static inline void openmx_mgpu_band_scope_select(openmx_mgpu_band_scope *scope,
    int allowed, int total_kpoints, int kpoint_index)
{
    openmx_mgpu_band_scope_close(scope);
    scope->previous = openmx_mgpu_band_scope_enter(allowed, total_kpoints, kpoint_index);
    scope->engaged = 1;
}

/* Host-backed capacity path. Planning is allowed outside an operation
 * scope; total_kpoints=0 means Cluster. Every selected group must admit
 * the MAGMA scratch estimate. Input/visibility limits are never expanded.
 * A Cluster plan uses the GPUs of the planning rank's node also in a
 * multi-node run (Band groups remain single-node).
 * A nonzero return from a host operation means success, otherwise callers
 * must stop. Resource/unavailable failures may use the explicitly logged
 * CPU LAPACK/BLAS capacity path. Numerical/execution/publication failures
 * are not retried on the CPU; an oversized single-GPU solve is never used. */
int openmx_mgpu_host_plan(int total_kpoints, int n, int complex_matrix);
/* The checks of openmx_mgpu_host_plan that need no GPU (group size within
 * the pool, LP64 order): the group size, or 0 with openmx_mgpu_host_error()
 * set.  Opens no CUDA context. */
int openmx_mgpu_host_precheck(int total_kpoints, int n, int complex_matrix);
const char *openmx_mgpu_host_error(void);
int openmx_mgpu_host_solve(int n, int nev, int complex_matrix, void *a, int lda, double *w);
int openmx_mgpu_host_dgemm(OpenMXBlasOperation ta, OpenMXBlasOperation tb,
    int m, int n, int k, const double *a, int lda, const double *b, int ldb, double *c, int ldc);
int openmx_mgpu_host_zgemm(OpenMXBlasOperation ta, OpenMXBlasOperation tb,
    int m, int n, int k, const void *a, int lda, const void *b, int ldb, void *c, int ldc);

/* Optional hooks keep standalone tests of the existing bridges linkable
   without the new runtime. The complete application links it explicitly. */
#if defined(__GNUC__) && !defined(OPENMX_MGPU_RUNTIME_IMPLEMENTATION)
#define OPENMX_MGPU_OPTIONAL __attribute__((weak))
#else
#define OPENMX_MGPU_OPTIONAL
#endif

typedef OpenMXSolverStatus (*openmx_mgpu_native_solve)(void *context);

#if OPENMX_GPU_HIP
int openmx_mgpu_solve_host_result_try(int complex_matrix,int n,int nev,void *a,
    double *w,int *mout,int (*native_solve)(void *,double *),void *native_context,int *result)
    OPENMX_MGPU_OPTIONAL;


#endif
/* 0 declines, preserving A for the original solve; 1 has produced status.
   Native calibration uses the caller's existing solver/workspace callback,
   so timings include exactly the same single-GPU implementation as fallback.
   Auto scores the minimum of two successful measurements per candidate for
   repeated SCF work; OPENMX_MGPU_TUNE_TRIALS=1 uses one (maximum accepted 3).
   MGPU scores include host preparation, transfers and result validation.
   Supports FP64 vectors, full spectra and leading-index subsets. */
int openmx_mgpu_solve_device_try(OpenMXSolverHandle handle,
    OpenMXSolverEigMode jobz, OpenMXBlasFillMode uplo, int64_t n,
    OpenMXGpuDataType type, void *A, int64_t lda, int64_t nev, double *W,
    int64_t *meig, int *info, openmx_mgpu_native_solve native_solve,
    void *native_context, OpenMXSolverStatus *status) OPENMX_MGPU_OPTIONAL;

int openmx_mgpu_dispatch_dgemm(OpenMXBlasHandle handle,
    OpenMXBlasOperation ta, OpenMXBlasOperation tb, int m, int n, int k,
    const double *alpha, const double *A, int lda, const double *B, int ldb,
    const double *beta, double *C, int ldc, OpenMXBlasStatus *status) OPENMX_MGPU_OPTIONAL;
int openmx_mgpu_dispatch_zgemm(OpenMXBlasHandle handle,
    OpenMXBlasOperation ta, OpenMXBlasOperation tb, int m, int n, int k,
    const OpenMXGpuComplex *alpha, const OpenMXGpuComplex *A, int lda,
    const OpenMXGpuComplex *B, int ldb, const OpenMXGpuComplex *beta,
    OpenMXGpuComplex *C, int ldc, OpenMXBlasStatus *status) OPENMX_MGPU_OPTIONAL;

#undef OPENMX_MGPU_OPTIONAL
#ifdef __cplusplus
}
#endif
#endif
