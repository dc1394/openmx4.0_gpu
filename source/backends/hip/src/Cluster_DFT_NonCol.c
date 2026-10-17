/**********************************************************************
  Cluster_DFT_NonCol.c:

     Cluster_DFT_NonCol.c is a subroutine to perform non-collinear
     cluster calculations.

  Log of Cluster_DFT_NonCol.c:

     21/Feb./2019  Released by T. Ozaki

***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include "openmx_mgpu_runtime.h"
#include "openmx_cluster_capacity.h"
#include "elpa_cosma_bridge.h"
#include "openmx_gpu_dm_collective.h"
#include "openmx_gpu_workers.h"
static int ClusterNonCol_mgpu_host_dense = 0;
static int ClusterNonCol_gs2_fallback_n2 = 0;
static int ClusterNonCol_cpu_fallback_n2 = 0;
static int ClusterNonCol_UseGs2(int n2) {
    return gpusolver2_flag || (n2>0 && ClusterNonCol_gs2_fallback_n2==n2);
}
static void ClusterNonCol_ResetGs2Fallback(void) {
    if (ClusterNonCol_gs2_fallback_n2) openmx_gs2_set_fallback_report(0);
    ClusterNonCol_gs2_fallback_n2=0;
}

#include "lapack_prototypes.h"
#include "set_hip_default_device_from_local_rank.h"
#include "elpa_cosma_bridge.h"
#include "eigen_refine_gpu.h"
#include "openmx_precision.h"
#include "eigen_refine_hooks.h"
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define  measure_time   0

extern int ClusterNonCol_CalcDMRootDense_HIP(int entry_count, int size_H1, int n, int n2, int nk_occ,
                                             int calc_edm, const int *basis0, const int *basis1,
                                             const double *occ, const double *occ_e,
                                             const dcomplex *dense_evec, int ld, double *dm_buffer);
extern int ClusterNonCol_CalcDMRootDenseDevice_HIP(int entry_count, int size_H1, int n, int n2, int nk_occ,
                                                   int calc_edm, const int *basis0, const int *basis1,
                                                   const double *occ, const double *occ_e,
                                                   const dcomplex *d_dense_evec, int ld, void *d_scratch,
                                                   size_t scratch_bytes, double *dm_buffer);
extern int ClusterNonCol_CalcDMRootDenseGemm_HIP(hipblasHandle_t handle, int entry_count, int size_H1, int n, int n2,
                                                 int nk_occ, int calc_edm, const int *basis0, const int *basis1,
                                                 const double *occ, const double *occ_e, const dcomplex *d_V, int ld,
                                                 void *d_scratch, size_t scratch_bytes, double *dm_buffer);
extern int ClusterNonCol_BackTransformDevice_HIP(hipblasHandle_t handle, int n, int n2, int maxn, void *d_A,
                                                 void *d_vt, const double *d_S);
extern int ClusterNonCol_BackTransformHost_HIP(hipblasHandle_t handle, int n, int n2, int maxn, const dcomplex *V,
                                               const double *S, dcomplex *evec, int ld);
extern int ClusterNonCol_ScatterDense_HIP(hipblasHandle_t handle, int tnum, int n, const double *values,
                                          const int *index, double *d_values, int *d_index, double *d_H);
extern int ClusterNonCol_AddDense_HIP(hipblasHandle_t handle, size_t count, double *d_y, const double *d_x);
extern int ClusterNonCol_AssembleHs2_HIP(hipblasHandle_t handle, int n, const double *r11, const double *i11,
                                         const double *r12, const double *i12, const double *r22, const double *i22,
                                         void *d_A);
extern int openmx_magma_dsyevdx_gpu(int n, int maxn, double *d_A, double *w, int *mout);
extern int openmx_magma_zheevdx_gpu(int n, int maxn, void *d_A, double *w, int *mout);
extern hipblasStatus_t openmx_gemmul8DgemmSolve(hipblasHandle_t handle, hipblasOperation_t transa,
                                                hipblasOperation_t transb, int m, int n, int k, const double *alpha,
                                                const double *A, int lda, const double *B, int ldb,
                                                const double *beta, double *C, int ldc);
extern void openmx_gemmul8TrimWorkspaces(void);
extern int openmx_gemmul8ZgemmEnabled(void);
extern long long openmx_gemmul8NativeCalls(int is_complex);
extern int openmx_gpu_is_apu(void);
extern int openmx_gpu_is_mi300a(void);
extern size_t Set_Hamiltonian_DemoteResidency(void);

static void ClusterNonCol_DeviceSolveRelease(void);
static size_t ClusterNonCol_DeviceSolveHeldBytes(void);
static void ClusterNonCol_DeviceSolveEndStep(void);
static void ClusterNonCol_DeviceSolveGiveBack(void);
static const void *ClusterNonCol_DeviceDenseEVec(int n2, int rows);
static hipblasHandle_t ClusterNonCol_DeviceSolveHandle(void);
static void *ClusterNonCol_DeviceSolveScratch(size_t *bytes);
static void ClusterNonCol_HostStagingRelease(void);
static void ClusterNonCol_RefineStageReset(void);

/* The ELPA path builds its transformed overlap (Ss, Ss2) at the first SCF
   step only, and the root dense path keeps its own: when the two alternate
   within an SCF cycle (the GPU fit test can change its mind), the path taken
   builds what it lacks.  Reset at the run boundary. */
static int ClusterNonCol_elpa_ss_ready = 0;   /* Ss/Ss2 of the current geometry */
static int ClusterNonCol_root_s_stale = 0;    /* the root path's overlap cache may be of another geometry */

static void ClusterNonCol_AbortWithMessage(const char *message)
{
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1, 1);
}

static int ClusterNonCol_GpuVerbose(void)
{
    const char *value = getenv("OPENMX_GPU_VERBOSE");

    return (value != NULL && value[0] == '1');
}

/* GPU/CPU crossover of the noncollinear cluster dense eigensolver path,
   compared against n2 = 2n.  As in the collinear path the root-dense GPU
   solve beats the multi-rank ELPA fallback at every measured size (on one
   RTX 5080 shared by 8 ranks: total 1.49x at n2=960 down to 1.13x at
   n2=480, the complex n2 x n2 problem favors the GPU even more than Col),
   so the default sits far below the global GPU_CPU_SWITCH_NUM.
   OPENMX_CLUSTER_NONCOL_GPU_SWITCH_NUM=<dim> overrides it.  Also consulted
   by DFT.c so the startup fallback notice reports the effective value. */
#define CLUSTER_NONCOL_GPU_CPU_SWITCH_NUM 400

int Cluster_DFT_NonCol_GpuSwitchNum(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("OPENMX_CLUSTER_NONCOL_GPU_SWITCH_NUM");
        int parsed;

        cached = CLUSTER_NONCOL_GPU_CPU_SWITCH_NUM;
        if (value != NULL && 0 < (parsed = atoi(value))) {
            cached = parsed;
        }
    }
    return cached;
}

static size_t ClusterNonCol_CheckedMulCount(size_t a, size_t b, const char *label)
{
    if (a != 0 && b > SIZE_MAX / a) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Dimension overflow in Cluster_DFT_NonCol.c: %s", label);
        ClusterNonCol_AbortWithMessage(msg);
    }
    return a * b;
}

static size_t ClusterNonCol_CheckedAddCount(size_t a, size_t b, const char *label)
{
    if (b > SIZE_MAX - a) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Count overflow in Cluster_DFT_NonCol.c: %s", label);
        ClusterNonCol_AbortWithMessage(msg);
    }
    return a + b;
}

static void *ClusterNonCol_MallocArray(size_t count, size_t elem_size, const char *label)
{
    size_t bytes = ClusterNonCol_CheckedMulCount(count, elem_size, label);
    void *ptr = malloc((bytes == 0) ? 1 : bytes);

    if (ptr == NULL) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Out of memory in Cluster_DFT_NonCol.c: %s (%zu bytes)", label, bytes);
        ClusterNonCol_AbortWithMessage(msg);
    }
    return ptr;
}

static const char *ClusterNonCol_HipblasStatusName(hipblasStatus_t status)
{
    switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
        return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:
        return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:
        return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:
        return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_ARCH_MISMATCH:
        return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_MAPPING_ERROR:
        return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:
        return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:
        return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:
        return "HIPBLAS_STATUS_NOT_SUPPORTED";
    default:
        return "HIPBLAS_STATUS_UNKNOWN";
    }
}

static int ClusterNonCol_MpiRankForLog(void)
{
    int initialized = 0;
    int rank = -1;

    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    return rank;
}

static void ClusterNonCol_RequireOpenMPTargetDevice(const char *where)
{
    int devices = omp_get_num_devices();

    if (devices <= 0) {
        fprintf(stderr,
                "<Cluster> rank %d: %s requires an OpenMP GPU target, but omp_get_num_devices() returned %d.\n",
                ClusterNonCol_MpiRankForLog(), where, devices);
        fflush(stderr);
        ClusterNonCol_AbortWithMessage("Cluster_DFT_NonCol GPU path cannot continue without an OpenMP target device.");
    }
}

static char ClusterNonCol_DgemmOpChar(hipblasOperation_t op)
{
    return (op == HIPBLAS_OP_N) ? 'N' : 'T';
}

static void ClusterNonCol_DgemmCPU(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k,
                                   const double *A, const double *B, double *C)
{
    char ta = ClusterNonCol_DgemmOpChar(transa);
    char tb = ClusterNonCol_DgemmOpChar(transb);
    INTEGER mi = m, ni = n, ki = k, lda = m, ldb = k, ldc = m;
    double alpha = 1.0, beta = 0.0;

    F77_NAME(dgemm, DGEMM)(&ta, &tb, &mi, &ni, &ki, &alpha, (double *)A, &lda, (double *)B, &ldb, &beta, C, &ldc);
}

static void ClusterNonCol_LogGemmRetry(const char *where, const char *kind, hipblasStatus_t status, int m, int n,
                                          int k)
{
    fprintf(stderr,
            "<GEMM> rank %d: GPU GEMM bridge failed in %s for %sGEMM(m=%d,n=%d,k=%d): %s (%d). "
            "Retrying with native hipBLAS.\n",
            ClusterNonCol_MpiRankForLog(), where, kind, m, n, k, ClusterNonCol_HipblasStatusName(status),
            (int)status);
    fflush(stderr);
}

static void ClusterNonCol_LogGemmCpuFallback(const char *where, const char *kind, const char *backend,
                                             hipblasStatus_t status, int m, int n, int k)
{
    fprintf(stderr,
            "<GEMM> rank %d: %s failed in %s for %sGEMM(m=%d,n=%d,k=%d): %s (%d). "
            "Falling back to CPU BLAS for this GEMM.\n",
            ClusterNonCol_MpiRankForLog(), backend, where, kind, m, n, k, ClusterNonCol_HipblasStatusName(status),
            (int)status);
    fflush(stderr);
}

static void ClusterNonCol_LogHipCpuFallback(const char *where, const char *kind, hipError_t status, int m, int n,
                                             int k)
{
    fprintf(stderr,
            "<GEMM> rank %d: GPU setup/transfer failed in %s for %sGEMM(m=%d,n=%d,k=%d): %s (%d). "
            "Falling back to CPU BLAS for this GEMM.\n",
            ClusterNonCol_MpiRankForLog(), where, kind, m, n, k, hipGetErrorString(status), (int)status);
    fflush(stderr);
}

static void ClusterNonCol_LogGemmBackendOnce(const char *kind, const char *backend, int m, int n, int k)
{
    static int logged_d_gemmul8 = 0;
    static int logged_d_native = 0;
    static int logged_z_gemmul8 = 0;
    static int logged_z_native = 0;
    int rank = ClusterNonCol_MpiRankForLog();
    int *logged;

    if (!ClusterNonCol_GpuVerbose()) return;

    if (rank != 0) return;

    if (kind[0] == 'D') {
        logged = (backend[0] == 'G') ? &logged_d_gemmul8 : &logged_d_native;
    }
    else {
        logged = (backend[0] == 'G') ? &logged_z_gemmul8 : &logged_z_native;
    }

    if (*logged) return;

    fprintf(stderr,
            "<GEMM> rank %d: using %s for %sGEMM(m=%d,n=%d,k=%d).\n",
            rank, backend, kind, m, n, k);
    fflush(stderr);
    *logged = 1;
}

static hipblasStatus_t ClusterNonCol_TryDeviceDgemm(hipblasHandle_t handle, hipblasOperation_t transa,
                                                   hipblasOperation_t transb, int m, int n, int k, const double *A,
                                                   const double *B, double *C, const char *where)
{
    const double alpha = 1.0;
    const double beta = 0.0;
    hipblasStatus_t status;
    const long long native_before = openmx_gemmul8NativeCalls(0);

    status = openmx_gemmul8Dgemm(handle, transa, transb, m, n, k, &alpha, A, m, B, k, &beta, C, m);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ClusterNonCol_LogGemmBackendOnce("D", openmx_gemmul8NativeCalls(0) == native_before ?
                                       "GEMMul8" : "native hipBLAS", m, n, k);
        return status;
    }

    ClusterNonCol_LogGemmRetry(where, "D", status, m, n, k);
    status = hipblasDgemm(handle, transa, transb, m, n, k, &alpha, A, m, B, k, &beta, C, m);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ClusterNonCol_LogGemmBackendOnce("D", "native hipBLAS", m, n, k);
    }
    return status;
}

/* The forward transform S^T M S of the host-staged solve: every matrix goes
   to the device and back.  shared is a hipBLAS handle to borrow (the device
   solve's), NULL for one of the call's own. */
static void ClusterNonCol_TransformRealDenseMatricesGPU(int n, const double *S, double **matrices, int matrix_count,
                                                        double *work, hipblasHandle_t shared)
{
    hipblasHandle_t handle = shared;
    double *d_S = NULL;
    double *d_A = NULL;
    double *d_C = NULL;
    size_t bytes = ClusterNonCol_CheckedMulCount((size_t)n * (size_t)n, sizeof(double), "real GEMM bytes");
    hipError_t hip_status;
    hipblasStatus_t blas_status = HIPBLAS_STATUS_SUCCESS;

    if (handle == NULL) blas_status = hipblasCreate(&handle);
    if (blas_status != HIPBLAS_STATUS_SUCCESS) {
        for (int i = 0; i < matrix_count; i++) {
            ClusterNonCol_LogGemmCpuFallback("ClusterNonCol_TransformRealDenseMatricesGPU:create", "D",
                                             "native hipBLAS", blas_status, n, n, n);
            ClusterNonCol_DgemmCPU(HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, matrices[i], S, work);
            ClusterNonCol_DgemmCPU(HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, n, S, work, matrices[i]);
        }
        return;
    }

    hip_status = hipMalloc((void **)&d_S, bytes);
    if (hip_status != hipSuccess) goto setup_failed;
    hip_status = hipMalloc((void **)&d_A, bytes);
    if (hip_status != hipSuccess) goto setup_failed;
    hip_status = hipMalloc((void **)&d_C, bytes);
    if (hip_status != hipSuccess) goto setup_failed;
    hip_status = hipMemcpy(d_S, S, bytes, hipMemcpyHostToDevice);
    if (hip_status != hipSuccess) goto setup_failed;

    for (int i = 0; i < matrix_count; i++) {
        int use_cpu = 0;

        hip_status = hipMemcpy(d_A, matrices[i], bytes, hipMemcpyHostToDevice);
        if (hip_status != hipSuccess) {
            ClusterNonCol_LogHipCpuFallback("ClusterNonCol_TransformRealDenseMatricesGPU:copyin", "D",
                                             hip_status, n, n, n);
            use_cpu = 1;
        }

        if (!use_cpu) {
            blas_status = ClusterNonCol_TryDeviceDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n,
                                                       d_A, d_S, d_C,
                                                       "ClusterNonCol_TransformRealDenseMatricesGPU:H*S");
            if (blas_status != HIPBLAS_STATUS_SUCCESS) {
                ClusterNonCol_LogGemmCpuFallback("ClusterNonCol_TransformRealDenseMatricesGPU:H*S", "D",
                                                 "native hipBLAS", blas_status, n, n, n);
                use_cpu = 1;
            }
        }

        if (!use_cpu) {
            blas_status = ClusterNonCol_TryDeviceDgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, n,
                                                       d_S, d_C, d_A,
                                                       "ClusterNonCol_TransformRealDenseMatricesGPU:S^T*(H*S)");
            if (blas_status != HIPBLAS_STATUS_SUCCESS) {
                ClusterNonCol_LogGemmCpuFallback("ClusterNonCol_TransformRealDenseMatricesGPU:S^T*(H*S)", "D",
                                                 "native hipBLAS", blas_status, n, n, n);
                use_cpu = 1;
            }
        }

        if (!use_cpu) {
            hip_status = hipMemcpy(matrices[i], d_A, bytes, hipMemcpyDeviceToHost);
            if (hip_status != hipSuccess) {
                ClusterNonCol_LogHipCpuFallback("ClusterNonCol_TransformRealDenseMatricesGPU:copyout", "D",
                                                 hip_status, n, n, n);
                use_cpu = 1;
            }
        }

        if (use_cpu) {
            ClusterNonCol_DgemmCPU(HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, matrices[i], S, work);
            ClusterNonCol_DgemmCPU(HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, n, S, work, matrices[i]);
        }
    }

    if (d_C != NULL) wait_hipfunc(hipFree(d_C));
    if (d_A != NULL) wait_hipfunc(hipFree(d_A));
    if (d_S != NULL) wait_hipfunc(hipFree(d_S));
    if (handle != shared) wait_hipfunc(hipblasDestroy(handle));
    return;

setup_failed:
    ClusterNonCol_LogHipCpuFallback("ClusterNonCol_TransformRealDenseMatricesGPU:setup", "D", hip_status, n, n, n);
    if (d_C != NULL) hipFree(d_C);
    if (d_A != NULL) hipFree(d_A);
    if (d_S != NULL) hipFree(d_S);
    if (handle != NULL && handle != shared) hipblasDestroy(handle);
    for (int i = 0; i < matrix_count; i++) {
        ClusterNonCol_DgemmCPU(HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, matrices[i], S, work);
        ClusterNonCol_DgemmCPU(HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, n, S, work, matrices[i]);
    }
}

/* The back transform of the host-staged solve (no device room for the
   device-resident one, or OPENMX_CLUSTER_NC_DEVICE_SOLVE=0): V (host,
   n2 x n2, the eigenvectors of the transformed problem in its first maxn
   columns) becomes the state-major dense_evec (host, state l of basis i at
   l + i ld, l < maxn),

       dense_evec[l + i ld] = sum_j S[i' + j' n] V[j + l n2]

   over the spin block of i (i' = i mod n, j' = j mod n): the transpose
   vt = V(:, 0:maxn)^T and, per spin half, one real GEMM of the real view of
   vt (2 maxn x n) with S^T, as the device-resident back transform does.  The
   former form, one complex GEMM against an n2 x n2 copy of diag(S, S) (four
   times the work, and that copy kept on the host: 505 MB at n2 = 5616), is
   gone.  On the device when it has room for the three panels (handle may
   be NULL), else with CPU BLAS. */
static void ClusterNonCol_BackTransformHostStaged(int n, int n2, int maxn, const dcomplex *V, const double *S,
                                                  dcomplex *dense_evec, int ld, hipblasHandle_t handle)
{
    dcomplex *vt;
    INTEGER m2 = 2 * maxn, ni = n, ldv = 2 * maxn, lds = n, ldc = 2 * ld;
    double alpha = 1.0, beta = 0.0;
    char ta = 'N', tb = 'T';

    if (ClusterNonCol_BackTransformHost_HIP(handle, n, n2, maxn, V, S, dense_evec, ld) == 0) return;

    if (2 <= level_stdout) {
        fprintf(stderr, "<GEMM> rank %d: the host-staged back transform (maxn=%d, n2=%d) has no room on the device; "
                "CPU BLAS.\n", ClusterNonCol_MpiRankForLog(), maxn, n2);
        fflush(stderr);
    }
    vt = (dcomplex *)ClusterNonCol_MallocArray(ClusterNonCol_CheckedMulCount((size_t)maxn, (size_t)n2,
                                                                             "back transform panel"),
                                               sizeof(dcomplex), "back transform panel");
    for (int col = 0; col < n2; col++) {
        for (int l = 0; l < maxn; l++) {
            vt[(size_t)l + (size_t)col * (size_t)maxn] = V[(size_t)col + (size_t)l * (size_t)n2];
        }
    }
    for (int half = 0; half < 2; half++) {
        size_t const col = (size_t)half * (size_t)n;

        F77_NAME(dgemm, DGEMM)(&ta, &tb, &m2, &ni, &ni, &alpha, (double *)vt + (size_t)2 * (size_t)maxn * col, &ldv,
                               (double *)S, &lds, &beta, (double *)dense_evec + (size_t)2 * (size_t)ld * col, &ldc);
    }
    free(vt);
}

typedef struct
{
    int                valid;
    int                n;
    int                entry_count;
    unsigned long long fingerprint;
    int               *basis0;
    int               *basis1;
} ClusterNonColDMEntryCache;

typedef struct
{
    int      max_state_capacity;
    size_t   dm_buffer_elems;
    double  *occ;
    double  *occ_e;
    double  *dm_buffer;
    int      dm_pinned;    /* dm_buffer is page-locked (the root's: the device DM comes back into it) */
} ClusterNonColRootDMWorkspace;

static ClusterNonColDMEntryCache    ClusterNonCol_dm_entry_cache = {0};
static ClusterNonColRootDMWorkspace ClusterNonCol_root_dm_workspace = {0};

static void ClusterNonCol_DMEntryCache_Reset(void)
{
    free(ClusterNonCol_dm_entry_cache.basis0);
    free(ClusterNonCol_dm_entry_cache.basis1);
    memset(&ClusterNonCol_dm_entry_cache, 0, sizeof(ClusterNonCol_dm_entry_cache));
}

static unsigned long long ClusterNonCol_DMEntryFingerprint(int *MP, int n)
{
    unsigned long long h = 1469598103934665603ULL;

#define CLUSTERNONCOL_HASH_INT(v) \
    do { \
        h ^= (unsigned long long)(unsigned int)(v); \
        h *= 1099511628211ULL; \
    } while (0)

    CLUSTERNONCOL_HASH_INT(n);
    CLUSTERNONCOL_HASH_INT(atomnum);

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int wanA = WhatSpecies[GA_AN];

        CLUSTERNONCOL_HASH_INT(GA_AN);
        CLUSTERNONCOL_HASH_INT(MP[GA_AN]);
        CLUSTERNONCOL_HASH_INT(wanA);
        CLUSTERNONCOL_HASH_INT(Spe_Total_CNO[wanA]);
        CLUSTERNONCOL_HASH_INT(FNAN[GA_AN]);

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int wanB  = WhatSpecies[GB_AN];

            CLUSTERNONCOL_HASH_INT(GB_AN);
            CLUSTERNONCOL_HASH_INT(wanB);
            CLUSTERNONCOL_HASH_INT(Spe_Total_CNO[wanB]);
            CLUSTERNONCOL_HASH_INT(MP[GB_AN]);
        }
    }

#undef CLUSTERNONCOL_HASH_INT

    return h;
}

static void ClusterNonCol_DMEntryCache_Ensure(int *MP, int n)
{
    ClusterNonColDMEntryCache *cache = &ClusterNonCol_dm_entry_cache;
    unsigned long long fingerprint = ClusterNonCol_DMEntryFingerprint(MP, n);
    int entry_count = 0;

    if (cache->valid && cache->n == n && cache->fingerprint == fingerprint) {
        return;
    }

    ClusterNonCol_DMEntryCache_Reset();

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]];

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int tnoB  = Spe_Total_CNO[WhatSpecies[GB_AN]];
            entry_count += tnoA * tnoB;
        }
    }

    cache->basis0 = (int *)ClusterNonCol_MallocArray((size_t)(entry_count + 1), sizeof(int), "DM basis0 cache");
    cache->basis1 = (int *)ClusterNonCol_MallocArray((size_t)(entry_count + 1), sizeof(int), "DM basis1 cache");

    entry_count = 0;
    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]];
        int Anum = MP[GA_AN];

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int tnoB  = Spe_Total_CNO[WhatSpecies[GB_AN]];
            int Bnum  = MP[GB_AN];

            for (int i = 0; i < tnoA; i++) {
                int ibasis = Anum + i - 1;

                for (int j = 0; j < tnoB; j++) {
                    cache->basis0[entry_count] = ibasis;
                    cache->basis1[entry_count] = Bnum + j - 1;
                    entry_count++;
                }
            }
        }
    }

    cache->valid       = 1;
    cache->n           = n;
    cache->fingerprint = fingerprint;
    cache->entry_count = entry_count;
}

/* Page-locked host memory for the transfers of the root dense solve (the
   copy engines take it directly instead of through a staging buffer), or
   plain memory when the pinning fails; *pinned says which.  NULL only when
   neither is to be had. */
static void *ClusterNonCol_HostAlloc(size_t bytes, int *pinned)
{
    void *ptr = NULL;

    *pinned = 0;
    if (hipHostMalloc(&ptr, (bytes == 0) ? 1 : bytes, 0) == hipSuccess && ptr != NULL) {
        *pinned = 1;
        return ptr;
    }
    (void)hipGetLastError();
    return malloc((bytes == 0) ? 1 : bytes);
}

static void ClusterNonCol_HostFree(void *ptr, int pinned)
{
    if (ptr == NULL) return;
    if (pinned) (void)hipHostFree(ptr);
    else free(ptr);
}

static void ClusterNonCol_RootDMWorkspace_Reset(void)
{
    free(ClusterNonCol_root_dm_workspace.occ);
    free(ClusterNonCol_root_dm_workspace.occ_e);
    ClusterNonCol_HostFree(ClusterNonCol_root_dm_workspace.dm_buffer, ClusterNonCol_root_dm_workspace.dm_pinned);
    memset(&ClusterNonCol_root_dm_workspace, 0, sizeof(ClusterNonCol_root_dm_workspace));
}

/* pin: the root's buffer, into which the device DM comes back */
static void ClusterNonCol_RootDMWorkspace_Ensure(int max_state, size_t dm_buffer_elems, int pin)
{
    ClusterNonColRootDMWorkspace *ws = &ClusterNonCol_root_dm_workspace;

    if (max_state <= ws->max_state_capacity && dm_buffer_elems <= ws->dm_buffer_elems) {
        return;
    }

    ClusterNonCol_RootDMWorkspace_Reset();

    ws->occ       = (double *)ClusterNonCol_MallocArray((size_t)max_state, sizeof(double), "root DM occupations");
    ws->occ_e     = (double *)ClusterNonCol_MallocArray((size_t)max_state, sizeof(double), "root DM energy occupations");
    if (pin) {
        ws->dm_buffer = (double *)ClusterNonCol_HostAlloc(
            ClusterNonCol_CheckedMulCount(dm_buffer_elems, sizeof(double), "root DM buffer"), &ws->dm_pinned);
        if (ws->dm_buffer == NULL) ClusterNonCol_AbortWithMessage("Out of memory in Cluster_DFT_NonCol.c: root DM buffer");
    }
    else {
        ws->dm_buffer = (double *)ClusterNonCol_MallocArray(dm_buffer_elems, sizeof(double), "root DM buffer");
        ws->dm_pinned = 0;
    }

    ws->max_state_capacity = max_state;
    ws->dm_buffer_elems    = dm_buffer_elems;
}

static int ClusterNonCol_PrecomputeOccupations(int kmin, int kmax, const double *ko, double *occ, double *occ_e)
{
    int nk = 0;
    double max_x = 60.0;

    for (int local_k = 0, global_k = kmin; global_k <= kmax; local_k++, global_k++) {
        double x = (ko[global_k] - ChemP) * Beta;
        double weight;

        if (x <= -max_x) x = -max_x;
        if (max_x <= x)  x =  max_x;

        weight = FermiFunc_NC(x, global_k);
        occ[local_k]   = weight;
        occ_e[local_k] = weight * ko[global_k];

        if (1.0e-13 < weight) {
            nk = local_k + 1;
        }
    }

    return nk;
}

static void ClusterNonCol_StoreDMBuffer(int myid, int size_H1, int *MP, double *****CDM, double *****iDM0,
                                        const double *dm_buffer)
{
    const double *rDM11 = dm_buffer;
    const double *rDM22 = dm_buffer + (size_t)size_H1;
    const double *rDM12 = dm_buffer + (size_t)size_H1 * 2;
    const double *iDM12 = dm_buffer + (size_t)size_H1 * 3;
    const double *iDM11 = dm_buffer + (size_t)size_H1 * 4;
    const double *iDM22 = dm_buffer + (size_t)size_H1 * 5;
    size_t p = 0;

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int MA_AN = F_G2M[GA_AN];
        int tnoA  = Spe_Total_CNO[WhatSpecies[GA_AN]];
        int ID    = G2ID[GA_AN];

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int tnoB  = Spe_Total_CNO[WhatSpecies[GB_AN]];

            if (myid == ID) {
                for (int i = 0; i < tnoA; i++) {
                    for (int j = 0; j < tnoB; j++, p++) {
                        CDM[0][MA_AN][LB_AN][i][j]  = rDM11[p];
                        CDM[1][MA_AN][LB_AN][i][j]  = rDM22[p];
                        CDM[2][MA_AN][LB_AN][i][j]  = rDM12[p];
                        CDM[3][MA_AN][LB_AN][i][j]  = iDM12[p];
                        iDM0[0][MA_AN][LB_AN][i][j] = iDM11[p];
                        iDM0[1][MA_AN][LB_AN][i][j] = iDM22[p];
                    }
                }
            }
            else {
                p += (size_t)tnoA * (size_t)tnoB;
            }
        }
    }
}

static void ClusterNonCol_StoreEDMBuffer(int myid, int size_H1, int *MP, double *****EDM, const double *edm_buffer)
{
    const double *rEDM11 = edm_buffer;
    const double *rEDM22 = edm_buffer + (size_t)size_H1;
    const double *rEDM12 = edm_buffer + (size_t)size_H1 * 2;
    const double *iEDM12 = edm_buffer + (size_t)size_H1 * 3;
    size_t p = 0;

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int MA_AN = F_G2M[GA_AN];
        int tnoA  = Spe_Total_CNO[WhatSpecies[GA_AN]];
        int ID    = G2ID[GA_AN];

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int tnoB  = Spe_Total_CNO[WhatSpecies[GB_AN]];

            if (myid == ID) {
                for (int i = 0; i < tnoA; i++) {
                    for (int j = 0; j < tnoB; j++, p++) {
                        EDM[0][MA_AN][LB_AN][i][j] = rEDM11[p];
                        EDM[1][MA_AN][LB_AN][i][j] = rEDM22[p];
                        EDM[2][MA_AN][LB_AN][i][j] = rEDM12[p];
                        EDM[3][MA_AN][LB_AN][i][j] = iEDM12[p];
                    }
                }
            }
            else {
                p += (size_t)tnoA * (size_t)tnoB;
            }
        }
    }
}

/* OPENMX_CLUSTER_GPU_DM_GEMM (default 1 on MI300A or when the complex
   GEMMul8 products are on): the density matrix of the root dense path as products,
   P11 = W1^H W1, P12 = W1^H W2 and P22 = W2^H W2 with W = diag(sqrt f) V
   over the occupied states (W1, W2 its spin halves: three n x n x nk complex
   GEMMs through GEMMul8), each followed by a gather of the sparse entries
   into the device copy of the DM buffer, instead of the per-entry sums over
   the states (ClusterNonColDMKernel); the energy density matrix of the
   orbital optimization the same way with W = diag(f e) V against V.  On an
   RX 9060 XT at the shape of sidia333 (n2 = 5616, 888 occupied states, 2.9
   million entries; tools/nc_back_transform_probe_hip.cu): 55 ms instead of
   404 (108 instead of 488 with the energy density matrix), 2e-15 of the
   largest entry off a long-double reference (the sums 3e-16).  Its buffers
   are carved from the solve's arena as far as it has room (all of them with
   the refined eigensolver's arena); a buffer that cannot be had or a failing
   product leaves the step to the sums.  On that Radeon, plain FP64 products
   cost what the sums do (412 against 403 ms), hence its default only with GEMMul8.
   MI300A also uses the products with native FP64 GEMM.  0 keeps the sums.
   The NVIDIA version forms all of P = W^H W, an n2 x n2
   matrix of its own (2 GiB of GEMMul8 workspace at that shape). */
static int ClusterNonCol_DmByGemm(void)
{
    static int cached = -1;   /* 0 or 1 from the environment, 2 the default */

    if (cached < 0) {
        const char *value = getenv("OPENMX_CLUSTER_GPU_DM_GEMM");

        cached = (value != NULL && value[0] != '\0') ? (atoi(value) != 0) : 2;
    }
    return (cached == 2) ? (openmx_gpu_is_mi300a() || openmx_gemmul8ZgemmEnabled()) : cached;
}

/* dense_evec: the vectors of the host-staged solve (state k of basis i at
   k + i dense_ld) when the device-resident solve left none in d_A */
static double ClusterNonCol_CalcDMRootDense_OpenMP(int myid, int size_H1, int *MP, int n, int n2, int max_state,
                                                    double *****CDM, double *****iDM0, double *****EDM, double *ko,
                                                    const dcomplex *dense_evec, int dense_ld, int calc_edm)
{
    ClusterNonColRootDMWorkspace *ws = &ClusterNonCol_root_dm_workspace;
    const int dm_components = calc_edm ? 10 : 6;
    size_t dm_elems = (size_t)size_H1 * (size_t)dm_components;
    int nk_occ = 0, recover = 0;
    double stime, etime;

    if ((size_t)INT_MAX < dm_elems) {
        ClusterNonCol_AbortWithMessage("Root DM buffer exceeds MPI count limit in Cluster_DFT_NonCol.c.");
    }

    ClusterNonCol_RootDMWorkspace_Ensure((myid == Host_ID) ? max_state : 0, dm_elems, myid == Host_ID);

    dtime(&stime);

    if (myid == Host_ID) {
        ClusterNonColDMEntryCache *cache;

        ClusterNonCol_DMEntryCache_Ensure(MP, n);
        cache = &ClusterNonCol_dm_entry_cache;
        if (cache->entry_count != size_H1) {
            ClusterNonCol_AbortWithMessage("DM entry cache size mismatch in Cluster_DFT_NonCol.c.");
        }

        nk_occ = ClusterNonCol_PrecomputeOccupations(1, max_state, ko, ws->occ, ws->occ_e);

        recover = nk_occ>0 && (ClusterNonCol_mgpu_host_dense || openmx_gpu_capacity_test_recovery());
        if (0 < nk_occ && !recover) {
            static int logged_dm_gpu = 0;
            const void *d_evec = ClusterNonCol_DeviceDenseEVec(n2, nk_occ);
            size_t scratch_bytes = 0;
            void *scratch = (d_evec != NULL) ? ClusterNonCol_DeviceSolveScratch(&scratch_bytes) : NULL;
            int dm_rc = 1, by_gemm = 0;
            long long native_products = 0;

            if (d_evec != NULL && ClusterNonCol_DmByGemm()) {
                const long long native_before = openmx_gemmul8NativeCalls(1);
                dm_rc = ClusterNonCol_CalcDMRootDenseGemm_HIP(ClusterNonCol_DeviceSolveHandle(), cache->entry_count,
                                                              size_H1, n, n2, nk_occ, calc_edm, cache->basis0,
                                                              cache->basis1, ws->occ, ws->occ_e,
                                                              (const dcomplex *)d_evec, n2, scratch, scratch_bytes,
                                                              ws->dm_buffer);
                native_products = openmx_gemmul8NativeCalls(1) - native_before;
                /* a product's full-size scratch does not outlive the step's
                   solve (the capped one goes at the end of the turn) */
                openmx_gemmul8TrimWorkspaces();
                if (dm_rc == 0) {
                    by_gemm = 1;
                }
                else {
                    static int warned = 0;

                    if (!warned) {
                        printf("<Cluster_DFT_NonCol> the density matrix by GEMM had no room or failed; per-entry "
                               "sums (OPENMX_CLUSTER_GPU_DM_GEMM)\n");
                        fflush(stdout);
                        warned = 1;
                    }
                }
            }

            if (ClusterNonCol_GpuVerbose() && !logged_dm_gpu) {
                fprintf(stderr,
                        "<Cluster> rank %d: density matrix generation uses %s on the selected rank only "
                        "(entries=%d, occupied states=%d%s).\n",
                        Host_ID, !by_gemm ? "a HIP GPU kernel" :
                        (native_products == 0 ? "GEMMul8 products and a HIP gather" :
                         (native_products == (calc_edm ? 6 : 3) ? "native hipBLAS products and a HIP gather" :
                          "GEMMul8/native hipBLAS products and a HIP gather")),
                        cache->entry_count, nk_occ, calc_edm ? ", with EDM" : "");
                fflush(stderr);
                logged_dm_gpu = 1;
            }

            if (!by_gemm) {
                if (d_evec != NULL)
                    dm_rc = ClusterNonCol_CalcDMRootDenseDevice_HIP(cache->entry_count, size_H1, n, n2, nk_occ,
                                                                    calc_edm, cache->basis0, cache->basis1, ws->occ,
                                                                    ws->occ_e, (const dcomplex *)d_evec, n2, scratch,
                                                                    scratch_bytes, ws->dm_buffer);
                else
                    dm_rc = ClusterNonCol_CalcDMRootDense_HIP(cache->entry_count, size_H1, n, n2, nk_occ, calc_edm,
                                                              cache->basis0, cache->basis1, ws->occ, ws->occ_e,
                                                              dense_evec, dense_ld, ws->dm_buffer);
            }
            if (dm_rc != 0) {
                recover=1;
            }
        }
        else {
            /* the device paths write every entry; without occupied states
               the buffer is zero */
            memset(ws->dm_buffer, 0, sizeof(double) * dm_elems);
        }
    }

    ClusterNonColDMEntryCache *cache=&ClusterNonCol_dm_entry_cache;
    double *outputs[10]={0};
    if (recover) for (int c=0;c<dm_components;c++) outputs[c]=ws->dm_buffer+(size_t)c*size_H1;
    openmx_gpu_dm_collective(mpi_comm_level1,OPENMX_GPU_DM_NC,n,nk_occ,dense_ld,
        recover?size_H1:0,dense_evec,1,cache->basis0,cache->basis1,NULL,0,NULL,NULL,
        ws->occ,ws->occ_e,NULL,outputs,0);

    /* only the root wrote entries: the other ranks receive its buffer */
    MPI_Bcast(ws->dm_buffer, (int)dm_elems, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    ClusterNonCol_StoreDMBuffer(myid, size_H1, MP, CDM, iDM0, ws->dm_buffer);

    if (calc_edm) {
        ClusterNonCol_StoreEDMBuffer(myid, size_H1, MP, EDM, ws->dm_buffer + (size_t)size_H1 * 6);
    }

    dtime(&etime);
    return etime - stime;
}

/* dense_evec: state k of basis i at k + i ld on the root */
static double ClusterNonCol_ScatterDenseEVecToLocal(int myid, int numprocs, int n2, const int *is2, const int *ie2,
                                                    const dcomplex *dense_evec, int ld, dcomplex *EVec1)
{
    const int tag = 1901;
    double stime, etime;
    int max_local_states = 0;
    dcomplex *sendbuf = NULL;

    dtime(&stime);

    for (int ID = 0; ID < numprocs; ID++) {
        int states = ie2[ID] - is2[ID] + 1;
        if (states < 0) states = 0;
        if (max_local_states < states) max_local_states = states;
    }

    if (myid == Host_ID) {
        size_t max_send_count = (size_t)max_local_states * (size_t)n2;

        if (0 < max_send_count) {
            if (max_send_count > (size_t)INT_MAX / 2u) {
                ClusterNonCol_AbortWithMessage("GPU solver dense eigenvector send buffer exceeds MPI count limit.");
            }
            sendbuf = (dcomplex *)ClusterNonCol_MallocArray(max_send_count, sizeof(dcomplex),
                                                            "GPU solver dense eigenvector send buffer");
        }

        for (int ID = 0; ID < numprocs; ID++) {
            int states = ie2[ID] - is2[ID] + 1;
            dcomplex *target = (ID == Host_ID) ? EVec1 : sendbuf;

            if (states <= 0) continue;

            for (int k = 0; k < states; k++) {
                int state = is2[ID] + k - 1;

                for (int basis = 0; basis < n2; basis++) {
                    /* Repack the dense column-major gpuSOLVER vectors into the
                       state-major EVec1 blocks used by the ELPA/ScaLAPACK DM path. */
                    target[(size_t)k * (size_t)n2 + (size_t)basis] =
                        dense_evec[(size_t)state + (size_t)basis * (size_t)ld];
                }
            }

            if (ID != Host_ID) {
                size_t send_count = (size_t)states * (size_t)n2;
                MPI_Send(sendbuf, (int)(send_count * 2u), MPI_DOUBLE, ID, tag, mpi_comm_level1);
            }
        }
    }
    else {
        int states = ie2[myid] - is2[myid] + 1;

        if (0 < states) {
            size_t recv_count = (size_t)states * (size_t)n2;
            MPI_Status stat;

            if (recv_count > (size_t)INT_MAX / 2u) {
                ClusterNonCol_AbortWithMessage("GPU solver dense eigenvector receive buffer exceeds MPI count limit.");
            }
            MPI_Recv(EVec1, (int)(recv_count * 2u), MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &stat);
        }
    }

    free(sendbuf);
    dtime(&etime);
    return etime - stime;
}

/* The vectors of the latest root dense solve, kept on the root for the
   energy density matrix and the forces after the SCF: the first rows states
   of every basis function (state k of basis i at k + i ld), in the host
   staging of the solve (ClusterNonCol_host_stage.evec), which the stash
   borrows and does not free. */
static dcomplex *ClusterNonCol_CachedGpuSolverDenseEVec = NULL;
static int ClusterNonCol_CachedGpuSolverDenseN2 = 0;
static int ClusterNonCol_CachedGpuSolverDenseLd = 0;

static void ClusterNonCol_ReleaseGpuSolverCachedEVec(int myid)
{
    (void)myid;
    ClusterNonCol_CachedGpuSolverDenseEVec = NULL;
    ClusterNonCol_CachedGpuSolverDenseN2 = 0;
    ClusterNonCol_CachedGpuSolverDenseLd = 0;
}

/* Set_Density_Grid_GPU.c when the device runs short of memory: the stashed
   vectors are a host copy already and stay for the post-SCF scatter (they
   used to be dropped here, which left the energy density matrix after the
   SCF without vectors); what the device still holds of the root solve (only
   with OPENMX_CLUSTER_NC_DEVICE_KEEP=1) goes back. */
void Cluster_DFT_NonCol_DemoteGpuSolverCachedEVec(void)
{
    int myid;

    MPI_Comm_rank(mpi_comm_level1, &myid);
    if (myid == Host_ID) ClusterNonCol_DeviceSolveGiveBack();
}

/* before the forces: the post-SCF scatter has taken the vectors */
void Cluster_DFT_NonCol_Release_GPU_Solver(void)
{
    int myid;

    MPI_Comm_rank(mpi_comm_level1, &myid);
    ClusterNonCol_ReleaseGpuSolverCachedEVec(myid);
    ClusterNonCol_DeviceSolveRelease();
}

/* Run-boundary reset (between -runtest/-runtestL inputs): the cached dense
   eigenvectors and the geometry-derived DM tables must not survive into a
   different system, nor the stage of the refined eigensolver (a native
   fallback that ended the refinement for the rest of a run would otherwise
   end it for every later input).  Idempotent; only the host rank owns the
   cached eigenvectors and the other ranks hold NULL. */
void Cluster_DFT_NonCol_Release_GPU_Caches(void)
{
    ClusterNonCol_mgpu_host_dense=0;
    ClusterNonCol_cpu_fallback_n2=0;
    ClusterNonCol_ResetGs2Fallback();
    ClusterNonCol_elpa_ss_ready = 0;
    ClusterNonCol_root_s_stale = 1;
    ClusterNonCol_DMEntryCache_Reset();
    ClusterNonCol_RootDMWorkspace_Reset();
    ClusterNonCol_ReleaseGpuSolverCachedEVec(Host_ID);
    ClusterNonCol_DeviceSolveRelease();
    ClusterNonCol_HostStagingRelease();
    ClusterNonCol_RefineStageReset();
}

static int ClusterNonCol_NativeGpuDiagFits(int n2, int myid)
{
    const char *env = getenv("OPENMX_CLUSTER_GPU_DIAG");
    const char *reserve_env = getenv("OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB");
    size_t free_bytes = 0, total_bytes = 0;
    /* 256 MB headroom by default: the dense buffers are transient (freed
       inside every solve) and the SCF-loop library workspaces are released
       before the force stage, so the old 1-GiB reserve only pushed VRAM-
       tight shared GPUs into the ELPA fallback.  Override with
       OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB. */
    size_t reserve = (size_t)256 * 1024 * 1024;
    size_t required = 0;
    int local_fit = 1, fit = 1;

    if (env != NULL && atoi(env) == 0) local_fit = 0;
    if (reserve_env != NULL) {
        long mib = atol(reserve_env);
        if (0 <= mib) reserve = (size_t)mib * 1024U * 1024U;
    }
    for (int attempt = 0; attempt < 2; attempt++) {
        int memory_fit = 1, all_memory_fit = 1;

        if (local_fit && myid == Host_ID) {
            /* H, eigenvectors and two transformation work panels, all complex. */
            if ((size_t)n2 > SIZE_MAX / (size_t)n2 / sizeof(dcomplex) / 4U) local_fit = 0;
            else {
                required = (size_t)4 * (size_t)n2 * (size_t)n2 * sizeof(dcomplex);
                if (required > SIZE_MAX - reserve ||
                    OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != 0 ||
                    free_bytes + ClusterNonCol_DeviceSolveHeldBytes() < required + reserve) memory_fit = 0;
            }
        }
        MPI_Allreduce(&memory_fit, &all_memory_fit, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
        if (all_memory_fit || attempt == 1) {
            if (!all_memory_fit) local_fit = 0;
            break;
        }
        /* the Hamiltonian's resident orbital tables leave the device for the
           rest of the run (every rank's copies share it), then once more */
        (void)Set_Hamiltonian_DemoteResidency();
        MPI_Barrier(mpi_comm_level1);
    }
    MPI_Allreduce(&local_fit, &fit, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
    if (!fit && myid == Host_ID) {
        printf("<Cluster_DFT_NonCol> GPU dense buffers do not fit (%.1f MiB free, %.1f MiB held by the solve, "
               "%.1f MiB needed with the reserve); checking capacity recovery.\n", (double)free_bytes / 1048576.0,
               (double)ClusterNonCol_DeviceSolveHeldBytes() / 1048576.0, (double)(required + reserve) / 1048576.0);
        fflush(stdout);
    }
    return fit;
}

static int ClusterNonCol_GpuDiagFits(int n2,int myid)
{
    int fit;
    ClusterNonCol_mgpu_host_dense=0;
    ClusterNonCol_ResetGs2Fallback();
    ClusterNonCol_cpu_fallback_n2=0;
    const char *enabled=getenv("OPENMX_CLUSTER_GPU_DIAG");
    if (enabled && atoi(enabled)==0 && !openmx_mgpu_host_required()) {
        ClusterNonCol_cpu_fallback_n2=n2;
        return 0;
    }
    fit=openmx_mgpu_host_required()?0:ClusterNonCol_NativeGpuDiagFits(n2,myid);
    if (!fit) {
        ClusterNonCol_DeviceSolveRelease();
        openmx_gemmul8ReleaseWorkspaces();
        MPI_Barrier(mpi_comm_level1);
        int capacity=openmx_cluster_capacity_select(mpi_comm_level1,n2,1,"Cluster NC");
        if (capacity==OPENMX_CLUSTER_CAPACITY_MAGMA) {
            ClusterNonCol_mgpu_host_dense=1;
            fit=1;
        }
        else if (capacity==OPENMX_CLUSTER_CAPACITY_GS2) {
            if (openmx_cluster_gs2_admit(mpi_comm_level1,ictxt1_2,n2,nblk2,1,"Cluster NC"))
                ClusterNonCol_gs2_fallback_n2=n2;
        }
    }
    if (!fit && !ClusterNonCol_UseGs2(n2)) ClusterNonCol_cpu_fallback_n2=n2;
    return fit;
}


/* the root's vectors of the step (ld > 0: their rows) become the stash */
static void ClusterNonCol_StashGpuSolverDenseEVec(int myid, int n2, dcomplex **dense_evec, int *ld)
{
    ClusterNonCol_ReleaseGpuSolverCachedEVec(myid);

    if (myid == Host_ID && *dense_evec != NULL && 0 < *ld) {
        ClusterNonCol_CachedGpuSolverDenseEVec = *dense_evec;
        ClusterNonCol_CachedGpuSolverDenseN2 = n2;
        ClusterNonCol_CachedGpuSolverDenseLd = *ld;
    }

    *dense_evec = NULL;
    *ld = 0;
}

double Cluster_DFT_NonCol_ScatterGpuSolverCachedEVec(int n2, int *is2, int *ie2, dcomplex *EVec1)
{
    int myid, numprocs;
    int valid = 0;
    double elapsed = 0.0;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    if (myid == Host_ID) {
        valid = (ClusterNonCol_CachedGpuSolverDenseEVec != NULL &&
                 ClusterNonCol_CachedGpuSolverDenseN2 == n2);
        if (valid) {
            /* the states of the scatter are those of the solve that stashed */
            for (int ID = 0; ID < numprocs; ID++) {
                if (is2[ID] <= ie2[ID] && ClusterNonCol_CachedGpuSolverDenseLd < ie2[ID]) {
                    ClusterNonCol_AbortWithMessage("The stashed GPU solver eigenvectors do not cover the states of "
                                                   "the post-SCF scatter in Cluster_DFT_NonCol.c.");
                }
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, Host_ID, mpi_comm_level1);

    if (valid) {
        elapsed = ClusterNonCol_ScatterDenseEVecToLocal(myid, numprocs, n2, is2, ie2,
                                                        ClusterNonCol_CachedGpuSolverDenseEVec,
                                                        ClusterNonCol_CachedGpuSolverDenseLd, EVec1);
    }

    ClusterNonCol_ReleaseGpuSolverCachedEVec(myid);

    return elapsed;
}

/* GPU GEMM wrappers (added by H.Kawai, ported from 3.9.9 GPU) */
static void ClusterNonCol_GEMMul8Dgemm_OpenMP(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n,
                                               int k, double const * A, double const * B, double * C)
{
    my_hipblasDgemm_openmp(transa, transb, m, n, k, A, B, C);
}

static void ClusterNonCol_GEMMul8Zgemm_OpenMP(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n,
                                               int k, dcomplex const * A, dcomplex const * B, dcomplex * C)
{
    my_hipblasZgemm_openmp(transa, transb, m, n, k, A, B, C);
}

static void ClusterNonCol_GpuSolver_DenseDsyevx(double *A, double *Z, double *ko, int n, int maxn,
                                               const char *where)
{
    int info,l,copy_cols,mout;
    double *d_A = NULL;
    size_t bytes = ClusterNonCol_CheckedMulCount((size_t)n * (size_t)n, sizeof(double), "MAGMA real eigensolver matrix");

    wait_hipfunc(hipMalloc((void **)&d_A, bytes));
    wait_hipfunc(hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice));
    info = openmx_magma_dsyevdx_gpu(n, maxn, d_A, ko, &mout);
    wait_hipfunc(hipMemcpy(A, d_A, bytes, hipMemcpyDeviceToHost));
    wait_hipfunc(hipFree(d_A));

    if (info!=0){
        fprintf(stderr,"%s: magma_dsyevdx_gpu failed, info=%d\n",where,info);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1,1);
    }
    if (mout!=maxn){
        fprintf(stderr,"%s: magma_dsyevdx_gpu returned %d eigenpairs, expected %d\n",where,mout,maxn);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1,1);
    }

    for (l=maxn; 1<=l; l--) ko[l] = ko[l-1];

    if (Z!=NULL){
        copy_cols = maxn;
        if (n<copy_cols) copy_cols = n;
        memcpy(Z,A,sizeof(double)*(size_t)n*(size_t)copy_cols);
    }
}

static void ClusterNonCol_GpuSolver_DenseZheevx(dcomplex *A, dcomplex *Z, double *ko, int n, int maxn,
                                               const char *where)
{
    int info,l,copy_cols,mout;
    void *d_A = NULL;
    size_t bytes = ClusterNonCol_CheckedMulCount((size_t)n * (size_t)n, sizeof(dcomplex),
                                                 "MAGMA complex eigensolver matrix");

    wait_hipfunc(hipMalloc(&d_A, bytes));
    wait_hipfunc(hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice));
    info = openmx_magma_zheevdx_gpu(n, maxn, d_A, ko, &mout);
    wait_hipfunc(hipMemcpy(A, d_A, bytes, hipMemcpyDeviceToHost));
    wait_hipfunc(hipFree(d_A));

    if (info!=0){
        fprintf(stderr,"%s: magma_zheevdx_gpu failed, info=%d\n",where,info);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1,1);
    }
    if (mout!=maxn){
        fprintf(stderr,"%s: magma_zheevdx_gpu returned %d eigenpairs, expected %d\n",where,mout,maxn);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1,1);
    }

    for (l=maxn; 1<=l; l--) ko[l] = ko[l-1];

    if (Z!=NULL){
        copy_cols = maxn;
        if (n<copy_cols) copy_cols = n;
        memcpy(Z,A,sizeof(dcomplex)*(size_t)n*(size_t)copy_cols);
    }
}

static int ClusterNonCol_ComputeLocalNZeroCount(void)
{
    int MA_AN, GA_AN, LB_AN, GB_AN;
    int wanA, wanB, tnoA, tnoB;
    size_t my_nzeros = 0;

    for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        size_t orbitals_in_neighbors = 0;

        GA_AN = M2G[MA_AN];
        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];

        for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            GB_AN = natn[GA_AN][LB_AN];
            wanB = WhatSpecies[GB_AN];
            tnoB = Spe_Total_CNO[wanB];
            orbitals_in_neighbors =
                ClusterNonCol_CheckedAddCount(orbitals_in_neighbors, (size_t)tnoB, "neighbor orbital count");
        }

        my_nzeros = ClusterNonCol_CheckedAddCount(
            my_nzeros, ClusterNonCol_CheckedMulCount((size_t)tnoA, orbitals_in_neighbors, "local H1 segment"),
            "local H1 segment");
    }

    if (my_nzeros > (size_t)INT_MAX) {
        ClusterNonCol_AbortWithMessage("Local H1 segment exceeds INT_MAX in Cluster_DFT_NonCol.c.");
    }

    return (int)my_nzeros;
}

static int ClusterNonCol_SetMPAndReturnNum(int *MP)
{
    int i, wanA;
    size_t orbital_offset = 1;

    for (i = 1; i <= atomnum; i++) {
        if (orbital_offset > (size_t)INT_MAX) {
            ClusterNonCol_AbortWithMessage("Orbital offsets exceed INT_MAX in Cluster_DFT_NonCol.c.");
        }

        MP[i] = (int)orbital_offset;
        wanA = WhatSpecies[i];
        orbital_offset =
            ClusterNonCol_CheckedAddCount(orbital_offset, (size_t)Spe_Total_CNO[wanA], "orbital offsets");
    }

    if (orbital_offset == 0 || orbital_offset - 1u > (size_t)INT_MAX) {
        ClusterNonCol_AbortWithMessage("Orbital count exceeds INT_MAX in Cluster_DFT_NonCol.c.");
    }

    return (int)(orbital_offset - 1u);
}

static void ClusterNonCol_PackLocalSegment(double ****RH, double *local_H1, int local_nzeros)
{
    int MA_AN, GA_AN, LB_AN, GB_AN;
    int wanA, wanB, tnoA, tnoB;
    int i, j, k;

    k = 0;
    for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        GA_AN = M2G[MA_AN];
        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];

        for (i = 0; i < tnoA; i++) {
            for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                GB_AN = natn[GA_AN][LB_AN];
                wanB = WhatSpecies[GB_AN];
                tnoB = Spe_Total_CNO[wanB];

                for (j = 0; j < tnoB; j++) {
                    local_H1[k] = RH[MA_AN][LB_AN][i][j];
                    k++;
                }
            }
        }
    }

    if (k != local_nzeros) {
        ClusterNonCol_AbortWithMessage("Packed H1 length mismatch in Cluster_DFT_NonCol.c.");
    }
}

static void ClusterNonCol_BuildDenseFromGathered(const double *H1, const int *order_GA, int *MP, double *H,
                                                 int n, int tnum)
{
    int AN, GA_AN, LB_AN, GB_AN;
    int wanA, wanB, tnoA, tnoB;
    size_t *atom_offsets;
    size_t dense_count;
    size_t offset;

    atom_offsets = (size_t *)ClusterNonCol_MallocArray((size_t)atomnum + 2u, sizeof(size_t), "atom offsets");

    offset = 0;
    for (AN = 1; AN <= atomnum; AN++) {
        size_t atom_block_count = 0;

        atom_offsets[AN] = offset;
        GA_AN = order_GA[AN];
        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];

        for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            GB_AN = natn[GA_AN][LB_AN];
            wanB = WhatSpecies[GB_AN];
            tnoB = Spe_Total_CNO[wanB];
            atom_block_count = ClusterNonCol_CheckedAddCount(
                atom_block_count, ClusterNonCol_CheckedMulCount((size_t)tnoA, (size_t)tnoB, "atom block span"),
                "atom block span");
        }

        offset = ClusterNonCol_CheckedAddCount(offset, atom_block_count, "dense assembly offsets");
    }
    atom_offsets[atomnum + 1] = offset;

    if (offset != (size_t)tnum) {
        ClusterNonCol_AbortWithMessage("Gathered H1 size mismatch in Cluster_DFT_NonCol.c.");
    }

    dense_count = ClusterNonCol_CheckedMulCount((size_t)n, (size_t)n, "dense H size");
    memset(H, 0, ClusterNonCol_CheckedMulCount(dense_count, sizeof(double), "dense H bytes"));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (AN = 1; AN <= atomnum; AN++) {
        int GA_AN_local;
        int wanA_local;
        int tnoA_local;
        int Anum_local;
        size_t k_local;

        GA_AN_local = order_GA[AN];
        wanA_local = WhatSpecies[GA_AN_local];
        tnoA_local = Spe_Total_CNO[wanA_local];
        Anum_local = MP[GA_AN_local];
        k_local = atom_offsets[AN];

        for (int i = 0; i < tnoA_local; i++) {
            int ig = Anum_local + i;

            for (int LB_AN_local = 0; LB_AN_local <= FNAN[GA_AN_local]; LB_AN_local++) {
                int GB_AN_local;
                int wanB_local;
                int tnoB_local;
                int Bnum_local;

                GB_AN_local = natn[GA_AN_local][LB_AN_local];
                wanB_local = WhatSpecies[GB_AN_local];
                tnoB_local = Spe_Total_CNO[wanB_local];
                Bnum_local = MP[GB_AN_local];

                for (int j = 0; j < tnoB_local; j++) {
                    int jg = Bnum_local + j;

                    if (ig < 1 || n < ig || jg < 1 || n < jg) {
                        ClusterNonCol_AbortWithMessage("Dense matrix index is out of range in Cluster_DFT_NonCol.c.");
                    }

                    H[(size_t)(jg - 1) * (size_t)n + (size_t)(ig - 1)] += H1[k_local];
                    k_local++;
                }
            }
        }

        if (k_local != atom_offsets[AN + 1]) {
            ClusterNonCol_AbortWithMessage("Atom-local H1 offsets are inconsistent in Cluster_DFT_NonCol.c.");
        }
    }

    free(atom_offsets);
}

static void Patch2Full_Cluster_NonCol_Owner(double ****RH, double *H, int *MP, int owns_dense, int n)
{
    int myid, numprocs;
    int dense_root = Host_ID;
    int NUM;
    int local_nzeros;
    int local_matomnum;
    int dummy_int = 0;
    int *recv_nzeros = NULL;
    int *recv_matomnum = NULL;
    int *h1_displs = NULL;
    int *atom_displs = NULL;
    int *order_GA = NULL;
    double *local_H1;
    double *gathered_H1 = NULL;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    NUM = ClusterNonCol_SetMPAndReturnNum(MP);
    if (NUM != n) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Orbital count mismatch in Cluster_DFT_NonCol.c: n=%d, NUM=%d", n, NUM);
        ClusterNonCol_AbortWithMessage(msg);
    }

    local_nzeros = ClusterNonCol_ComputeLocalNZeroCount();
    local_matomnum = Matomnum;
    local_H1 = (double *)ClusterNonCol_MallocArray((size_t)local_nzeros, sizeof(double), "local H1 segment");
    ClusterNonCol_PackLocalSegment(RH, local_H1, local_nzeros);

    if (myid == dense_root) {
        recv_nzeros = (int *)ClusterNonCol_MallocArray((size_t)numprocs, sizeof(int), "recv_nzeros");
        recv_matomnum = (int *)ClusterNonCol_MallocArray((size_t)numprocs, sizeof(int), "recv_matomnum");
    }

    MPI_Gather(&local_nzeros, 1, MPI_INT, recv_nzeros, 1, MPI_INT, dense_root, mpi_comm_level1);
    MPI_Gather(&local_matomnum, 1, MPI_INT, recv_matomnum, 1, MPI_INT, dense_root, mpi_comm_level1);

    if (myid == dense_root) {
        size_t total_nzeros = 0;
        size_t total_atoms = 0;

        h1_displs = (int *)ClusterNonCol_MallocArray((size_t)numprocs, sizeof(int), "h1_displs");
        atom_displs = (int *)ClusterNonCol_MallocArray((size_t)numprocs, sizeof(int), "atom_displs");

        for (int ID = 0; ID < numprocs; ID++) {
            if (recv_nzeros[ID] < 0 || recv_matomnum[ID] < 0) {
                ClusterNonCol_AbortWithMessage("Negative gather counts detected in Cluster_DFT_NonCol.c.");
            }

            if (total_nzeros > (size_t)INT_MAX || total_atoms > (size_t)INT_MAX) {
                ClusterNonCol_AbortWithMessage("Gather displacements exceed INT_MAX in Cluster_DFT_NonCol.c.");
            }

            h1_displs[ID] = (int)total_nzeros;
            atom_displs[ID] = (int)total_atoms;
            total_nzeros =
                ClusterNonCol_CheckedAddCount(total_nzeros, (size_t)recv_nzeros[ID], "gathered H1 size");
            total_atoms =
                ClusterNonCol_CheckedAddCount(total_atoms, (size_t)recv_matomnum[ID], "gathered atom order");
        }

        if (total_nzeros > (size_t)INT_MAX) {
            ClusterNonCol_AbortWithMessage("Gathered H1 size exceeds INT_MAX in Cluster_DFT_NonCol.c.");
        }

        if (total_atoms != (size_t)atomnum) {
            ClusterNonCol_AbortWithMessage("Gathered atom order length mismatch in Cluster_DFT_NonCol.c.");
        }

        gathered_H1 = (double *)ClusterNonCol_MallocArray(total_nzeros, sizeof(double), "gathered H1");
        order_GA = (int *)ClusterNonCol_MallocArray((size_t)atomnum + 1u, sizeof(int), "order_GA");
    }

    MPI_Gatherv(local_H1, local_nzeros, MPI_DOUBLE, gathered_H1, recv_nzeros, h1_displs, MPI_DOUBLE, dense_root,
                mpi_comm_level1);
    MPI_Gatherv((0 < Matomnum) ? &M2G[1] : &dummy_int, Matomnum, MPI_INT, (myid == dense_root) ? &order_GA[1] : NULL,
                recv_matomnum, atom_displs, MPI_INT, dense_root, mpi_comm_level1);

    if (myid == dense_root && owns_dense) {
        int gathered_nzeros = 0;

        if (0 < numprocs) {
            gathered_nzeros = h1_displs[numprocs - 1] + recv_nzeros[numprocs - 1];
        }

        ClusterNonCol_BuildDenseFromGathered(gathered_H1, order_GA, MP, H, n, gathered_nzeros);
    }

    free(gathered_H1);
    free(order_GA);
    free(atom_displs);
    free(h1_displs);
    free(recv_matomnum);
    free(recv_nzeros);
    free(local_H1);
}

static void ClusterNonCol_BuildDenseIndexFromGathered(const int *order_GA, int *MP, int n, int tnum,
                                                      int *dense_index)
{
    int k = 0;

    for (int AN = 1; AN <= atomnum; AN++) {
        int GA_AN = order_GA[AN];
        int wanA = WhatSpecies[GA_AN];
        int tnoA = Spe_Total_CNO[wanA];
        int Anum = MP[GA_AN];

        for (int i = 0; i < tnoA; i++) {
            int row = Anum + i - 1;

            for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                int GB_AN = natn[GA_AN][LB_AN];
                int wanB = WhatSpecies[GB_AN];
                int tnoB = Spe_Total_CNO[wanB];
                int Bnum = MP[GB_AN];

                for (int j = 0; j < tnoB; j++) {
                    int col = Bnum + j - 1;

                    if (row < 0 || n <= row || col < 0 || n <= col) {
                        ClusterNonCol_AbortWithMessage("Dense matrix index is out of range in Cluster_DFT_NonCol.c.");
                    }
                    if (tnum <= k) {
                        ClusterNonCol_AbortWithMessage("Dense index buffer overflow in Cluster_DFT_NonCol.c.");
                    }

                    dense_index[k] = row + col * n;
                    k++;
                }
            }
        }
    }

    if (k != tnum) {
        ClusterNonCol_AbortWithMessage("Dense index count mismatch in Cluster_DFT_NonCol.c.");
    }
}

static void ClusterNonCol_BuildDeviceDenseFromGathered(const double *H1, const int *dense_index,
                                                       int tnum, int n, double *H)
{
    size_t nn = ClusterNonCol_CheckedMulCount((size_t)n, (size_t)n, "device dense H");
    long long nn_ll;
    static int logged_dense_build = 0;

    if ((size_t)LLONG_MAX < nn) {
        ClusterNonCol_AbortWithMessage("Dense matrix size exceeds OpenMP loop limit in Cluster_DFT_NonCol.c.");
    }
    nn_ll = (long long)nn;

    ClusterNonCol_RequireOpenMPTargetDevice("dense matrix generation");
    if (ClusterNonCol_GpuVerbose() &&
        !logged_dense_build && ClusterNonCol_MpiRankForLog() == Host_ID) {
        fprintf(stderr,
                "<Cluster> rank %d: dense matrix generation uses OpenMP target GPU on the selected rank only "
                "(n=%d, packed entries=%d).\n",
                Host_ID, n, tnum);
        fflush(stderr);
        logged_dense_build = 1;
    }

#pragma omp target data map(to: H1[0 : tnum], dense_index[0 : tnum]) map(alloc: H[0 : nn])
    {
#pragma omp target teams distribute parallel for
    for (long long p = 0; p < nn_ll; p++) {
        H[p] = 0.0;
    }

#pragma omp target teams distribute parallel for
    for (int p = 0; p < tnum; p++) {
        int idx = dense_index[p];
#pragma omp atomic update
        H[idx] += H1[p];
    }

#pragma omp target update from(H[0 : nn])
    }
}

/* The staging of the gathered entries on their way into the device-resident
   root solve: two n x n slots of d_A that stay idle until Hs2 is assembled
   there.  The dense positions of the Set_Hamiltonian cache, the same for
   every matrix of a solve, are built and uploaded once per solve instead of
   once per matrix. */
typedef struct
{
    double *d_values;        /* the entries of one matrix, capacity doubles */
    int    *d_index;         /* their dense positions, capacity ints */
    size_t  capacity;        /* 0: no staging, the matrices are built on the host */
    int    *index_host;      /* the positions of the Set_Hamiltonian cache's entries */
    int     index_tnum;
    int     index_on_device; /* d_index holds index_host */
} ClusterNonColDenseStage;

static ClusterNonColDenseStage ClusterNonCol_dense_stage;

static void ClusterNonCol_DenseStageSet(double *d_values, int *d_index, size_t capacity)
{
    ClusterNonColDenseStage *st = &ClusterNonCol_dense_stage;

    free(st->index_host);
    memset(st, 0, sizeof(*st));
    st->d_values = d_values;
    st->d_index = d_index;
    st->capacity = capacity;
}

/* The root's dense n x n matrix of tnum gathered entries H1 at the dense
   positions dense_index (both on the host) into d_H on the device of the
   root solve: the values (and, with upload_index, the positions) through
   the staging and a scatter kernel -- the matrix no longer comes back to the
   host -- or, when they exceed the staging, built on the host and
   uploaded.  Either way the matrix is the OpenMP build's to the bit. */
static void ClusterNonCol_BuildDenseOnDevice(const double *H1, const int *dense_index, int upload_index, int tnum,
                                             int n, double *d_H)
{
    ClusterNonColDenseStage *st = &ClusterNonCol_dense_stage;
    size_t const nn = ClusterNonCol_CheckedMulCount((size_t)n, (size_t)n, "device dense H");

    if ((size_t)tnum <= st->capacity) {
        if (ClusterNonCol_ScatterDense_HIP(ClusterNonCol_DeviceSolveHandle(), tnum, n, H1,
                                           upload_index ? dense_index : NULL, st->d_values, st->d_index, d_H) != 0) {
            ClusterNonCol_AbortWithMessage("Cluster_DFT_NonCol: the device build of a dense matrix failed.");
        }
    }
    else {
        double *H = (double *)ClusterNonCol_MallocArray(nn, sizeof(double), "dense matrix upload");

        memset(H, 0, nn * sizeof(double));
        for (int p = 0; p < tnum; p++) H[dense_index[p]] += H1[p];
        wait_hipfunc(hipMemcpy(d_H, H, nn * sizeof(double), hipMemcpyHostToDevice));
        free(H);
    }
}

/* the dense matrix of a Set_Hamiltonian cache entry into H: the host
   (device 0, through the OpenMP build) or the device of the root solve */
static void ClusterNonCol_LoadDeviceDenseFromSetHamCache(const double *H1, int *MP, int n, double *H, int device)
{
    int tnum = Set_Hamiltonian_GpuSolver_Packed_Size();
    int *order_GA = Set_Hamiltonian_GpuSolver_Packed_OrderGA();
    int *dense_index;

    if (H1 == NULL || order_GA == NULL) {
        ClusterNonCol_AbortWithMessage("Set_Hamiltonian packed matrix cache is missing in Cluster_DFT_NonCol.c.");
    }

    if (device) {
        ClusterNonColDenseStage *st = &ClusterNonCol_dense_stage;

        if (st->index_host == NULL || st->index_tnum != tnum) {
            free(st->index_host);
            st->index_host = (int *)ClusterNonCol_MallocArray((size_t)tnum, sizeof(int), "Set_Hamiltonian dense_index");
            ClusterNonCol_BuildDenseIndexFromGathered(order_GA, MP, n, tnum, st->index_host);
            st->index_tnum = tnum;
            st->index_on_device = 0;
        }
        ClusterNonCol_BuildDenseOnDevice(H1, st->index_host, !st->index_on_device, tnum, n, H);
        st->index_on_device = 1;
        return;
    }

    dense_index = (int *)ClusterNonCol_MallocArray((size_t)tnum, sizeof(int), "Set_Hamiltonian dense_index");
    ClusterNonCol_BuildDenseIndexFromGathered(order_GA, MP, n, tnum, dense_index);
    ClusterNonCol_BuildDeviceDenseFromGathered(H1, dense_index, tnum, n, H);
    free(dense_index);
}

/* every rank: the gather of a sparse matrix to the dense root, which builds
   the dense matrix into H on the host or (device) on the device of the root
   solve */
static void Patch2Device_Cluster_NonCol_Owner(double ****RH, double *H, int *MP, int owns_dense, int n, int device)
{
    int myid, numprocs;
    int dense_root = Host_ID;
    int NUM;
    int local_nzeros;
    int local_matomnum;
    int dummy_int = 0;
    int *recv_nzeros = NULL;
    int *recv_matomnum = NULL;
    int *h1_displs = NULL;
    int *atom_displs = NULL;
    int *order_GA = NULL;
    int *dense_index = NULL;
    double *local_H1;
    double *gathered_H1 = NULL;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    NUM = ClusterNonCol_SetMPAndReturnNum(MP);
    if (NUM != n) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Orbital count mismatch in Cluster_DFT_NonCol.c: n=%d, NUM=%d", n, NUM);
        ClusterNonCol_AbortWithMessage(msg);
    }

    local_nzeros = ClusterNonCol_ComputeLocalNZeroCount();
    local_matomnum = Matomnum;
    local_H1 = (double *)ClusterNonCol_MallocArray((size_t)local_nzeros, sizeof(double), "local H1 segment");
    ClusterNonCol_PackLocalSegment(RH, local_H1, local_nzeros);

    if (myid == dense_root) {
        recv_nzeros = (int *)ClusterNonCol_MallocArray((size_t)numprocs, sizeof(int), "recv_nzeros");
        recv_matomnum = (int *)ClusterNonCol_MallocArray((size_t)numprocs, sizeof(int), "recv_matomnum");
    }

    MPI_Gather(&local_nzeros, 1, MPI_INT, recv_nzeros, 1, MPI_INT, dense_root, mpi_comm_level1);
    MPI_Gather(&local_matomnum, 1, MPI_INT, recv_matomnum, 1, MPI_INT, dense_root, mpi_comm_level1);

    if (myid == dense_root) {
        size_t total_nzeros = 0;
        size_t total_atoms = 0;

        h1_displs = (int *)ClusterNonCol_MallocArray((size_t)numprocs, sizeof(int), "h1_displs");
        atom_displs = (int *)ClusterNonCol_MallocArray((size_t)numprocs, sizeof(int), "atom_displs");

        for (int ID = 0; ID < numprocs; ID++) {
            if (recv_nzeros[ID] < 0 || recv_matomnum[ID] < 0) {
                ClusterNonCol_AbortWithMessage("Negative gather counts detected in Cluster_DFT_NonCol.c.");
            }

            if (total_nzeros > (size_t)INT_MAX || total_atoms > (size_t)INT_MAX) {
                ClusterNonCol_AbortWithMessage("Gather displacements exceed INT_MAX in Cluster_DFT_NonCol.c.");
            }

            h1_displs[ID] = (int)total_nzeros;
            atom_displs[ID] = (int)total_atoms;
            total_nzeros =
                ClusterNonCol_CheckedAddCount(total_nzeros, (size_t)recv_nzeros[ID], "gathered H1 size");
            total_atoms =
                ClusterNonCol_CheckedAddCount(total_atoms, (size_t)recv_matomnum[ID], "gathered atom order");
        }

        if (total_nzeros > (size_t)INT_MAX) {
            ClusterNonCol_AbortWithMessage("Gathered H1 size exceeds INT_MAX in Cluster_DFT_NonCol.c.");
        }

        if (total_atoms != (size_t)atomnum) {
            ClusterNonCol_AbortWithMessage("Gathered atom order length mismatch in Cluster_DFT_NonCol.c.");
        }

        gathered_H1 = (double *)ClusterNonCol_MallocArray(total_nzeros, sizeof(double), "gathered H1");
        order_GA = (int *)ClusterNonCol_MallocArray((size_t)atomnum + 1u, sizeof(int), "order_GA");
    }

    MPI_Gatherv(local_H1, local_nzeros, MPI_DOUBLE, gathered_H1, recv_nzeros, h1_displs, MPI_DOUBLE, dense_root,
                mpi_comm_level1);
    MPI_Gatherv((0 < Matomnum) ? &M2G[1] : &dummy_int, Matomnum, MPI_INT, (myid == dense_root) ? &order_GA[1] : NULL,
                recv_matomnum, atom_displs, MPI_INT, dense_root, mpi_comm_level1);

    if (myid == dense_root && owns_dense) {
        int gathered_nzeros = 0;

        if (0 < numprocs) {
            gathered_nzeros = h1_displs[numprocs - 1] + recv_nzeros[numprocs - 1];
        }

        dense_index = (int *)ClusterNonCol_MallocArray((size_t)gathered_nzeros, sizeof(int), "dense_index");
        ClusterNonCol_BuildDenseIndexFromGathered(order_GA, MP, n, gathered_nzeros, dense_index);
        if (device)
            ClusterNonCol_BuildDenseOnDevice(gathered_H1, dense_index, 1, gathered_nzeros, n, H);
        else
            ClusterNonCol_BuildDeviceDenseFromGathered(gathered_H1, dense_index, gathered_nzeros, n, H);
    }

    free(dense_index);
    free(gathered_H1);
    free(order_GA);
    free(atom_displs);
    free(h1_displs);
    free(recv_matomnum);
    free(recv_nzeros);
    free(local_H1);
}

/* ------------------------------------------------------------------
   The refined FP32 eigensolver (eigen_refine_gpu.h) in the root dense path.

   OPENMX_EIGEN_REFINE=K (0: off): every step of an SCF cycle solves the
   transformed Hamiltonian for all 2n pairs in FP32 (rocSOLVER) or starts
   from the previous step's vectors, and refines the MaxN wanted ones by K
   Ogita-Aishima steps with GEMMul8 products.  Unset, K = 2 on GPUs whose
   FP64 throughput is a small fraction of their FP32 one (FP32:FP64 of at
   least 8: the RDNA and consumer GCN parts; CDNA keeps FP64), as long as
   the complex GEMMul8 products are on; the root rank's device decides for
   all ranks.

   The vectors inside the fully occupied clusters stay FP32 mixtures, which
   the energy density matrix rebuilt after the SCF would see, so an SCF that
   ends on a refined step, by its stop condition or by scf.maxIter, has that
   step's Hamiltonian solved once more in FP64 (mode "fp64", one eigensolve;
   Cluster_DFT_NonCol_RefineLastRefined tells DFT.c), and the last step
   scf.maxIter allows is FP64 in any case.  OPENMX_EIGEN_REFINE_UNTIL=T
   (default 0, off) hands the rest of the cycle to the FP64 eigensolver once
   the printed NormRD falls below T instead.  A refinement that cannot take
   place (no room for its buffers, a Rayleigh-Ritz cluster beyond its size
   limit, a failing solve) solves the step in FP64 and leaves the rest of
   the SCF cycle to the FP64 eigensolver; a refinement on by default whose
   products fell back to plain hipBLAS FP64 ends for the rest of the run.
   Orbital optimization (Cnt_switch) keeps the FP64 eigensolver.  Ported
   from the NVIDIA version (openmx4.0_gpu, Cluster_DFT_NonCol.c). */
typedef struct
{
    int     iterations;    /* -1 until the environment is read */
    double  until;         /* printed NormRD below which the FP64 eigensolver takes over; 0 off */
    int     final_stage;   /* the FP64 eigensolver for the rest of the SCF cycle */
    int     last_refined;  /* the latest SCF eigensolve was a refined one */
    int     cycle_refined; /* some eigensolve of this SCF cycle was */
    int     defaulted;     /* on because of the device, not of OPENMX_EIGEN_REFINE */
    int     force_fp64;    /* mode "fp64" of Cluster_DFT_NonCol: this solve in FP64 whatever the stage */
} ClusterNonColRefineStage;

static ClusterNonColRefineStage ClusterNonCol_refine = {-1, 0.0, 0, 0, 0, 0, 0};
static EigenRefineState         ClusterNonCol_refine_state;

/* the run boundary (Cluster_DFT_NonCol_Release_GPU_Caches): the stage as at
   the start of a run, the environment read again at the next solve, so that
   a refinement ended for the rest of a run (a native fallback of its
   products) does not stay off for the later inputs of -runtest/-runtestL.
   Not at the release before the forces, where "the rest of the run" holds. */
static void ClusterNonCol_RefineStageReset(void)
{
    ClusterNonColRefineStage const initial = {-1, 0.0, 0, 0, 0, 0, 0};

    ClusterNonCol_refine = initial;
    ClusterNonCol_refine_state.basis_valid = 0;
}

/* every rank, once, before the first root dense solve: OPENMX_EIGEN_REFINE,
   or the default the root rank's device implies, for all ranks alike */
static void ClusterNonCol_RefineConfigure(void)
{
    ClusterNonColRefineStage *st = &ClusterNonCol_refine;
    int myid, config[2] = {0, 0};   /* iterations, FP32:FP64 ratio when defaulted on */

    /* The input veto must not become a cached policy for later inputs. */
    if (!openmx_fp32_enabled()) return;
    if (0 <= st->iterations) return;
    MPI_Comm_rank(mpi_comm_level1, &myid);
    if (myid == Host_ID) config[0] = openmx_eigen_refine_configure(1, 2 <= level_stdout, &config[1]);
    MPI_Bcast(config, 2, MPI_INT, Host_ID, mpi_comm_level1);
    st->iterations = config[0];
    st->defaulted = (0 < config[1]);
    st->until = openmx_eigen_refine_until();
}

/* the steps of the refinement, 0 when it is off or not configured yet */
static int ClusterNonCol_RefineIterations(void)
{
    return (openmx_fp32_enabled() && 0 < ClusterNonCol_refine.iterations) ? ClusterNonCol_refine.iterations : 0;
}

/* every rank, before each root dense solve: a new SCF cycle starts refined.
   The last step scf.maxIter allows is always FP64, as the energy density
   matrix after the SCF comes from it whether the SCF converged or not.
   Orbital optimization (Cnt_switch) restarts the step count within a cycle
   and has stop rules of its own, so it keeps the FP64 eigensolver. */
static void ClusterNonCol_RefineBeginSolve(int SCF_iter, int n2)
{
    ClusterNonColRefineStage *st = &ClusterNonCol_refine;

    ClusterNonCol_RefineConfigure();
    if (SCF_iter == 1) {
        st->final_stage = 0;
        st->cycle_refined = 0;
        ClusterNonCol_refine_state.basis_valid = 0;   /* a new cycle may come with a new geometry */
    }
    st->last_refined = (Solver == 2 && 0 < ClusterNonCol_RefineIterations() && !st->final_stage && Cnt_switch == 0 &&
                        SCF_iter < DFTSCF_loop && openmx_eigen_refine_size_ok(n2));
}

/* DFT.c: whether the latest eigensolve was a refined FP32 one (the same on
   every rank).  An SCF that ends on such a solve, by its stop condition or
   by scf.maxIter, has the same Hamiltonian solved once more in FP64 (mode
   "fp64" of Cluster_DFT_NonCol, one eigensolve without the density matrix)
   so that the vectors kept for the post-SCF energy density matrix are FP64
   ones; the density matrix of the step, refined to 1e-11, stands. */
int Cluster_DFT_NonCol_RefineLastRefined(void)
{
    return ClusterNonCol_refine.last_refined;
}

/* DFT.c, after the mixing of an SCF step that goes on: normrd is the printed
   NormRD */
void Cluster_DFT_NonCol_RefineAfterMixing(double normrd)
{
    ClusterNonColRefineStage *st = &ClusterNonCol_refine;
    int myid;

    if (!st->cycle_refined || st->final_stage || st->until <= 0.0 || st->until <= normrd) return;
    st->final_stage = 1;
    MPI_Comm_rank(mpi_comm_level1, &myid);
    if (myid == Host_ID && 2 <= level_stdout) {
        printf("<DFT>  eigensolver: NormRD %.3e < %.3e, FP64 eigensolver from the next SCF step\n", normrd,
               st->until);
        fflush(stdout);
    }
}

/* Fermi occupations of the solver: f(e) for a state with 1-based index
   (orbitals emptied by index) at the chemical potential mu */
typedef struct
{
    double mu;
} ClusterNonColOccupationCtx;

static double ClusterNonCol_RefineFermi(double e, int index, void *ctx)
{
    double x = (e - ((ClusterNonColOccupationCtx *)ctx)->mu) * Beta;

    if (x <= -60.0) x = -60.0;
    if (60.0 <= x) x = 60.0;
    return FermiFunc_NC(x, index);
}

/* Fermi occupations f[0..n) of the ascending eigenvalues e[0..n) for
   TZ - system_charge electrons over the lowest maxn states, as the solver's
   chemical-potential search sets them; *mu_out receives the potential */
static void ClusterNonCol_RefineOccupations(const double *e, int n, int maxn, double *f, double *mu_out)
{
    double TZ = 0.0, lo = e[0] - 1.0, hi = e[maxn - 1] + 1.0, mu = 0.0;
    ClusterNonColOccupationCtx ctx;

    for (int i = 1; i <= atomnum; i++) TZ += Spe_Core_Charge[WhatSpecies[i]];
    for (int it = 0; it < 200; it++) {
        double count = 0.0;

        mu = 0.5 * (lo + hi);
        ctx.mu = mu;
        for (int i = 0; i < maxn; i++) count += ClusterNonCol_RefineFermi(e[i], i + 1, &ctx);
        if (0.0 <= (TZ - count) - system_charge) lo = mu;
        else hi = mu;
    }
    ctx.mu = mu;
    for (int i = 0; i < n; i++) f[i] = ClusterNonCol_RefineFermi(e[i], i + 1, &ctx);
    *mu_out = mu;
}

/* every rank, at the first SCF step of the root dense path: the states it
   solves.  The FP64 solve takes all n2, as the occupied ones are not known
   yet; a refined one, whose FP32 solve yields all n2 eigenvalues anyway,
   refines the lowest TZ - system_charge + max(400, 0.2 n2) only -- the margin
   of the later steps -- which keeps the product blocks within the scratch
   of the later steps instead of three more n2 x n2 matrices on a GPU the
   other ranks share.  OPENMX_EIGEN_REFINE_FIRST_ALL=1 refines all n2. */
static int ClusterNonCol_RefineFirstStepStates(int n2, double TZ)
{
    const char *all = getenv("OPENMX_EIGEN_REFINE_FIRST_ALL");
    double lumos = 0.2 * (double)n2;
    double states;

    ClusterNonCol_RefineConfigure();
    if (Solver != 2 || ClusterNonCol_RefineIterations() <= 0 || Cnt_switch != 0 || DFTSCF_loop <= 1 ||
        !openmx_eigen_refine_size_ok(n2) || (all != NULL && atoi(all) != 0)) return n2;
    if (lumos < 400.0) lumos = 400.0;
    states = ceil(TZ - system_charge) + lumos;
    return (states < (double)n2) ? (int)states : n2;
}

/* OPENMX_CLUSTER_PROFILE=1 (as in Cluster_DFT_Col.c): the cumulative wall
   time of the phases of the root dense solve on its owner rank, one
   NCCLUSTERPROF line per solve with a density matrix, in the format of the
   NVIDIA version that tools/summarize_forward_variants.py reads: the
   transformed overlap (rebuilt at the first SCF step; its upload and the
   device buffers of the solve every step), the gather of the Hamiltonian
   blocks, the forward transform with the assembly of Hs2, the eigensolver,
   the back transform, the release (the vectors to the host, the basis
   parked on the host, the device buffers and the GEMMul8 workspaces given
   back) and the density matrices.  The device is synchronized around each
   phase, so the times are not those of an unprofiled run. */
enum { NCPROF_OVERLAP, NCPROF_GATHER, NCPROF_FORWARD, NCPROF_EIGEN, NCPROF_BACK, NCPROF_RELEASE, NCPROF_DM,
       NCPROF_COUNT };

static double ClusterNonCol_prof[NCPROF_COUNT];
static long ClusterNonCol_prof_solves = 0;

static int ClusterNonCol_ProfileEnabled(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("OPENMX_CLUSTER_PROFILE");
        cached = (value != NULL && atoi(value) != 0);
    }
    return cached;
}

/* charges the time since *t0 to phase (none if negative) after synchronizing
   the device, and restarts *t0 */
static void ClusterNonCol_ProfileLap(int phase, double *t0)
{
    double now;

    wait_hipfunc(hipDeviceSynchronize());
    now = MPI_Wtime();
    if (0 <= phase) ClusterNonCol_prof[phase] += now - *t0;
    *t0 = now;
}

static void ClusterNonCol_ProfileReport(int myid, int n2, int MaxN, double dm_seconds)
{
    if (myid != Host_ID || !ClusterNonCol_ProfileEnabled()) return;

    ClusterNonCol_prof[NCPROF_DM] += dm_seconds;
    ClusterNonCol_prof_solves++;
    printf("NCCLUSTERPROF rank=%d solves=%ld n2=%d maxn=%d overlap=%.6f gather=%.6f forward=%.6f eigen=%.6f "
           "back=%.6f release=%.6f dm=%.6f (s, cumulative)\n",
           myid, ClusterNonCol_prof_solves, n2, MaxN, ClusterNonCol_prof[NCPROF_OVERLAP],
           ClusterNonCol_prof[NCPROF_GATHER], ClusterNonCol_prof[NCPROF_FORWARD], ClusterNonCol_prof[NCPROF_EIGEN],
           ClusterNonCol_prof[NCPROF_BACK], ClusterNonCol_prof[NCPROF_RELEASE], ClusterNonCol_prof[NCPROF_DM]);
    fflush(stdout);
}

/* OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE (then OPENMX_BAND_GEMMUL8_TURN_RELEASE,
   as in the NVIDIA version): the GEMMul8 workspaces go back at the end of
   every turn of the root dense solve (its density matrix done), so that the
   capped workspace of the transforms and the density matrix does not hold
   the shared device through the grid stages of the step either; 0 keeps
   the capped workspace for the next solve and frees only the full-size
   scratch of the unblocked products.  Default: release, except on an APU,
   whose device memory is the host's (as BandCol_GemmWorkspaceTurnRelease). */
static int ClusterNonCol_GemmWorkspaceTurnRelease(void)
{
    const char *value = getenv("OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE");

    if (value == NULL) value = getenv("OPENMX_BAND_GEMMUL8_TURN_RELEASE");
    if (value == NULL) return openmx_gpu_is_apu() ? 0 : 1;
    return (atoi(value) != 0);
}

/* The device side of the root dense solve: the Hamiltonian blocks are
   gathered into device matrices, transformed, assembled into Hs2, solved,
   back-transformed and handed to the density matrix without leaving the
   device.  d_A (n2 x n2 complex) first holds six of the gathered n x n
   blocks and the staging of the gather in its eight n x n slots, then Hs2,
   its eigenvectors, and the state-major vectors of the density matrix; d_S
   the transformed overlap (n x n); d_arena the two remaining blocks during
   the forward stage, the FP32 scratch and the product blocks of a refined
   solve, the transposed vectors of the back transform, and the buffers of
   the density matrix; d_x the refined basis kept for the next step's warm
   start.  The buffers are given back after the density matrix of every
   step, since the GPU is shared with the other ranks and the grid stages,
   and the basis waits on the host (h_x, page-locked) for the next step;
   OPENMX_CLUSTER_NC_DEVICE_KEEP=1 keeps them on the device instead, until
   the forces, memory pressure, the end of a run or a change of the problem.
   This is the default on MI300A, where the device and CPU share ample HBM;
   other GPUs keep the per-step release default.  Explicit 0/1 overrides it.
   OPENMX_CLUSTER_NC_DEVICE_SOLVE=0 keeps the former host-staged solve. */
typedef struct
{
    int              n, n2;
    hipblasHandle_t  hipblas;
    void            *d_A;
    double          *d_S;
    int              S_valid;
    void            *d_arena;
    size_t           arena_bytes;
    void            *d_x;
    double          *d_w;
    int32_t         *d_info;
    void            *d_work;
    size_t           d_work_bytes;
    int              dense_rows;   /* d_A holds the state-major vectors of this many states, 0 none */
    int              x_on_device;  /* d_x holds the refined basis of the latest solve */
    dcomplex        *h_x;          /* the basis between the steps (the device buffers are given back) */
    size_t           h_x_elems;
    int              h_x_valid;
    int              h_x_pinned;
} ClusterNonColDeviceSolve;

static ClusterNonColDeviceSolve ClusterNonCol_dsolve;

/* Host buffers of the root dense solve that last between its calls,
   page-locked when the pinning succeeds: the transformed overlap of the
   current geometry (uploaded every step, as the device buffers are given
   back), and the vectors of the latest solve, the first MaxN states of every
   basis function (state k of basis i at k + i MaxN), which the stash lends
   to the energy density matrix after the SCF.  They go at the run boundary
   (with the refined basis's h_x) or with a change of the problem size. */
typedef struct
{
    int       n;
    double   *S;
    int       S_pinned;
    int       S_valid;       /* S holds the transformed overlap of the current geometry */
    dcomplex *evec;
    size_t    evec_elems;
    int       evec_pinned;
} ClusterNonColHostStage;

static ClusterNonColHostStage ClusterNonCol_host_stage;

static void ClusterNonCol_HostStagingRelease(void)
{
    ClusterNonColHostStage *hs = &ClusterNonCol_host_stage;
    ClusterNonColDeviceSolve *ds = &ClusterNonCol_dsolve;

    ClusterNonCol_HostFree(hs->S, hs->S_pinned);
    ClusterNonCol_HostFree(hs->evec, hs->evec_pinned);
    memset(hs, 0, sizeof(*hs));
    ClusterNonCol_HostFree(ds->h_x, ds->h_x_pinned);
    ds->h_x = NULL;
    ds->h_x_elems = 0;
    ds->h_x_valid = 0;
    ds->h_x_pinned = 0;
}

/* the transformed overlap's buffer for n (a new one, not valid, when n changed) */
static double *ClusterNonCol_HostStageS(int n)
{
    ClusterNonColHostStage *hs = &ClusterNonCol_host_stage;

    if (hs->S == NULL || hs->n != n) {
        ClusterNonCol_HostFree(hs->S, hs->S_pinned);
        hs->S = (double *)ClusterNonCol_HostAlloc(
            ClusterNonCol_CheckedMulCount(ClusterNonCol_CheckedMulCount((size_t)n, (size_t)n, "cached transformed overlap"),
                                          sizeof(double), "cached transformed overlap"),
            &hs->S_pinned);
        if (hs->S == NULL) ClusterNonCol_AbortWithMessage("Out of memory in Cluster_DFT_NonCol.c: cached transformed overlap");
        hs->n = n;
        hs->S_valid = 0;
    }
    return hs->S;
}

/* the vectors' buffer for elems complex (grown when short; the stash, which
   borrows it, is released before every solve) */
static dcomplex *ClusterNonCol_HostStageEVec(size_t elems)
{
    ClusterNonColHostStage *hs = &ClusterNonCol_host_stage;

    if (hs->evec == NULL || hs->evec_elems < elems) {
        ClusterNonCol_HostFree(hs->evec, hs->evec_pinned);
        hs->evec = (dcomplex *)ClusterNonCol_HostAlloc(
            ClusterNonCol_CheckedMulCount(elems, sizeof(dcomplex), "root dense eigenvectors"), &hs->evec_pinned);
        if (hs->evec == NULL) ClusterNonCol_AbortWithMessage("Out of memory in Cluster_DFT_NonCol.c: root dense eigenvectors");
        hs->evec_elems = elems;
    }
    return hs->evec;
}

/* the state-major vectors the latest device solve left in d_A (leading
   dimension n2) when they cover rows states, else NULL */
static const void *ClusterNonCol_DeviceDenseEVec(int n2, int rows)
{
    const ClusterNonColDeviceSolve *ds = &ClusterNonCol_dsolve;

    if (ds->d_A != NULL && ds->n2 == n2 && 0 < ds->dense_rows && rows <= ds->dense_rows) return ds->d_A;
    return NULL;
}

/* the hipBLAS handle of the device solve (NULL without one) */
static hipblasHandle_t ClusterNonCol_DeviceSolveHandle(void)
{
    return ClusterNonCol_dsolve.hipblas;
}

/* the arena of the device solve, idle once the vectors are back-transformed:
   the scratch of the density matrix (NULL without one) */
static void *ClusterNonCol_DeviceSolveScratch(size_t *bytes)
{
    *bytes = (ClusterNonCol_dsolve.d_arena != NULL) ? ClusterNonCol_dsolve.arena_bytes : 0;
    return ClusterNonCol_dsolve.d_arena;
}

static int ClusterNonCol_DeviceSolveEnabled(void)
{
    static int enabled = -1;

    if (enabled < 0) {
        const char *value = getenv("OPENMX_CLUSTER_NC_DEVICE_SOLVE");
        enabled = (value == NULL || value[0] == '\0' || atoi(value) != 0) ? 1 : 0;
    }
    return enabled;
}

static hipError_t ClusterNonCol_TryDeviceMalloc(void **ptr, size_t bytes)
{
    hipError_t status = hipMalloc(ptr, (bytes == 0) ? 1 : bytes);

    if (status != hipSuccess) {
        *ptr = NULL;
        (void)hipGetLastError();
    }
    return status;
}

static void *ClusterNonCol_RefineHostMalloc(size_t count, size_t elem, const char *label)
{
    void *ptr = ClusterNonCol_MallocArray(count, elem, label);

    memset(ptr, 0, (count * elem == 0) ? 1 : count * elem);
    return ptr;
}

static void ClusterNonCol_DeviceFree(void **ptr)
{
    if (*ptr != NULL) {
        wait_hipfunc(hipFree(*ptr));
        *ptr = NULL;
    }
}

/* the large device buffers of the solve go back (the handle and the small
   buffers stay) */
static void ClusterNonCol_DeviceSolveDrop(void)
{
    ClusterNonColDeviceSolve *ds = &ClusterNonCol_dsolve;

    ClusterNonCol_DeviceFree(&ds->d_A);
    ClusterNonCol_DeviceFree((void **)&ds->d_S);
    ClusterNonCol_DeviceFree(&ds->d_arena);
    ClusterNonCol_DeviceFree(&ds->d_x);
    ClusterNonCol_DeviceFree(&ds->d_work);
    ds->d_work_bytes = 0;
    ds->arena_bytes = 0;
    ds->S_valid = 0;
    ds->x_on_device = 0;
    ds->dense_rows = 0;
}

static void ClusterNonCol_DeviceSolveRelease(void)
{
    ClusterNonColDeviceSolve *ds = &ClusterNonCol_dsolve;

    openmx_eigen_refine_state_release(&ClusterNonCol_refine_state);
    ClusterNonCol_DeviceSolveDrop();
    ClusterNonCol_DeviceFree((void **)&ds->d_w);
    ClusterNonCol_DeviceFree((void **)&ds->d_info);
    /* the page-locked basis buffer stays for the next SCF cycle (pinning
       505 MB anew would cost every MD step); it goes at the run boundary */
    ds->h_x_valid = 0;
    ds->n = ds->n2 = 0;
    if (ds->hipblas != NULL) {
        (void)hipblasDestroy(ds->hipblas);
        ds->hipblas = NULL;
    }
}

static int ClusterNonCol_DeviceSolveKeep(void)
{
    static int keep = -1;

    if (keep < 0) {
        const char *value = getenv("OPENMX_CLUSTER_NC_DEVICE_KEEP");
        keep = (value != NULL) ? (atoi(value) != 0) : openmx_gpu_is_mi300a();
    }
    return keep;
}

/* device bytes the root solve holds between its calls (the fit test counts
   them as free: the next solve reuses them) */
static size_t ClusterNonCol_DeviceSolveHeldBytes(void)
{
    const ClusterNonColDeviceSolve *ds = &ClusterNonCol_dsolve;
    size_t const nn2 = (size_t)ds->n2 * (size_t)ds->n2;
    size_t bytes = ds->arena_bytes + ds->d_work_bytes;

    if (ds->d_A != NULL) bytes += nn2 * 16;
    if (ds->d_S != NULL) bytes += (size_t)ds->n * (size_t)ds->n * sizeof(double);
    if (ds->d_x != NULL) bytes += nn2 * 16;
    return bytes;
}

/* the refined basis of the latest solve to the host (h_x) for the next
   step's warm start */
static void ClusterNonCol_DeviceSolveParkBasis(void)
{
    ClusterNonColDeviceSolve *ds = &ClusterNonCol_dsolve;

    if (ds->d_x != NULL && ds->x_on_device && ClusterNonCol_refine_state.basis_valid &&
        ClusterNonCol_refine_state.basis_n == ds->n2) {
        size_t const elems = (size_t)ds->n2 * (size_t)ds->n2;

        if (ds->h_x == NULL || ds->h_x_elems < elems) {
            ClusterNonCol_HostFree(ds->h_x, ds->h_x_pinned);
            ds->h_x = (dcomplex *)ClusterNonCol_HostAlloc(
                ClusterNonCol_CheckedMulCount(elems, sizeof(dcomplex), "refined basis host copy"), &ds->h_x_pinned);
            if (ds->h_x == NULL) ClusterNonCol_AbortWithMessage("Out of memory in Cluster_DFT_NonCol.c: refined basis host copy");
            ds->h_x_elems = elems;
        }
        wait_hipfunc(hipMemcpy(ds->h_x, ds->d_x, elems * sizeof(dcomplex), hipMemcpyDeviceToHost));
        ds->h_x_valid = 1;
    }
}

/* the end of a root solve's step, its density matrix done: the large device
   buffers go back, the refined basis to the host for the next warm start,
   and the GEMMul8 workspaces of the turn (OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE;
   without it the full-size scratch of the unblocked products only) */
static void ClusterNonCol_DeviceSolveEndStep(void)
{
    int const profile = ClusterNonCol_ProfileEnabled();
    double t0 = 0.0;

    if (profile) ClusterNonCol_ProfileLap(-1, &t0);
    if (!ClusterNonCol_DeviceSolveKeep()) {
        ClusterNonCol_DeviceSolveParkBasis();
        ClusterNonCol_DeviceSolveDrop();
    }
    if (ClusterNonCol_GemmWorkspaceTurnRelease()) openmx_gemmul8ReleaseWorkspaces();
    else openmx_gemmul8TrimWorkspaces();
    if (profile) ClusterNonCol_ProfileLap(NCPROF_RELEASE, &t0);
}

/* the device buffers a kept solve still holds (OPENMX_CLUSTER_NC_DEVICE_KEEP=1)
   go back on demand, the basis parked on the host */
static void ClusterNonCol_DeviceSolveGiveBack(void)
{
    const ClusterNonColDeviceSolve *ds = &ClusterNonCol_dsolve;

    if (ds->d_A == NULL && ds->d_S == NULL && ds->d_arena == NULL && ds->d_x == NULL && ds->d_work == NULL) return;
    ClusterNonCol_DeviceSolveParkBasis();
    ClusterNonCol_DeviceSolveDrop();
}

/* the arena of a solve of maxn states: the FP32 scratch of a refined solve
   ((2 n2^2 + n2 + 1) floats), its three product blocks when they are fewer
   than all states (the first SCF step's blocks for all states take a
   buffer of their own), the transposed vectors of the back transform, and
   the two n x n blocks the forward stage keeps outside d_A */
static size_t ClusterNonCol_ArenaBytes(int n, int n2, int maxn, int refine)
{
    size_t const nn2 = (size_t)n2 * (size_t)n2;
    size_t const blocks = (size_t)2 * (size_t)n * (size_t)n * sizeof(double);
    size_t bytes = (size_t)16 * (size_t)maxn * (size_t)n2;

    if (refine) {
        size_t const fp32 = ((size_t)2 * nn2 + (size_t)n2 + 1) * sizeof(float);
        size_t const block = ((size_t)n2 * (size_t)maxn * 2 * sizeof(double) + 511u) / 512u * 512u;

        if (bytes < fp32) bytes = fp32;
        if (maxn < n2 && bytes < 3 * block) bytes = 3 * block;
    }
    if (bytes < blocks) bytes = blocks;
    return (bytes + 4095u) / 4096u * 4096u;
}

/* the buffers of a solve of maxn of n2 states, refine or not; 0 when the
   device has no room for them (the host-staged solve takes over, and what
   this call allocated goes back) */
static int ClusterNonCol_DeviceSolveEnsure(int n, int n2, int maxn, int refine)
{
    ClusterNonColDeviceSolve *ds = &ClusterNonCol_dsolve;
    size_t const nn2 = (size_t)n2 * (size_t)n2;
    size_t const arena_need = ClusterNonCol_ArenaBytes(n, n2, maxn, refine);
    size_t need = 0, free_bytes = 0, total_bytes = 0;
    size_t reserve = (size_t)256 * 1024 * 1024;
    const char *reserve_env = getenv("OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB");

    if (reserve_env != NULL && 0 <= atol(reserve_env)) reserve = (size_t)atol(reserve_env) * 1024U * 1024U;
    /* a refined solve also needs the GEMMul8 products' blocked workspace (256
       MiB) beside its buffers */
    if (refine) reserve += (size_t)256 * 1024 * 1024;
    if (ds->n2 != n2 || ds->n != n) ClusterNonCol_DeviceSolveRelease();
    if (ds->hipblas == NULL && hipblasCreate(&ds->hipblas) != HIPBLAS_STATUS_SUCCESS) {
        ds->hipblas = NULL;
        return 0;
    }
    if (ds->d_A == NULL) need += nn2 * 16;
    if (ds->d_S == NULL) need += (size_t)n * (size_t)n * sizeof(double);
    if (ds->arena_bytes < arena_need) need += arena_need;
    if (refine && ds->d_x == NULL) need += nn2 * 16;
    if (ds->d_w == NULL) need += (size_t)n2 * sizeof(double);
    /* the FP32 solver's workspace of a cold refined solve */
    if (refine && ds->d_work == NULL) need += openmx_eigen_refine_fp32_work_bytes(1, n2);
    if (0 < need) {
        if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != 0 || free_bytes < need + reserve) {
            static int warned = 0;

            if (!warned) {
                printf("<Cluster_DFT_NonCol> the device solve needs %.1f MiB more (%.1f MiB free, reserve %.1f MiB); "
                       "host-staged solve\n", (double)need / 1048576.0, (double)free_bytes / 1048576.0,
                       (double)reserve / 1048576.0);
                fflush(stdout);
                warned = 1;
            }
            ClusterNonCol_DeviceSolveDrop();
            return 0;
        }
    }
    ds->n = n;
    ds->n2 = n2;
    if (ds->arena_bytes < arena_need) {
        ClusterNonCol_DeviceFree(&ds->d_arena);
        ds->arena_bytes = 0;
        if (ClusterNonCol_TryDeviceMalloc(&ds->d_arena, arena_need) != hipSuccess) goto no_room;
        ds->arena_bytes = arena_need;
    }
    if (ds->d_A == NULL && ClusterNonCol_TryDeviceMalloc(&ds->d_A, nn2 * 16) != hipSuccess) goto no_room;
    if (ds->d_S == NULL) {
        if (ClusterNonCol_TryDeviceMalloc((void **)&ds->d_S, (size_t)n * (size_t)n * sizeof(double)) != hipSuccess)
            goto no_room;
        ds->S_valid = 0;
    }
    if (ds->d_w == NULL && ClusterNonCol_TryDeviceMalloc((void **)&ds->d_w, (size_t)n2 * sizeof(double)) != hipSuccess)
        goto no_room;
    if (ds->d_info == NULL && ClusterNonCol_TryDeviceMalloc((void **)&ds->d_info, sizeof(int32_t)) != hipSuccess)
        goto no_room;
    if (refine && ds->d_x == NULL) {
        if (ClusterNonCol_TryDeviceMalloc(&ds->d_x, nn2 * 16) != hipSuccess) goto no_room;
        ds->x_on_device = 0;
    }
    return 1;

no_room:
    ClusterNonCol_DeviceSolveDrop();
    return 0;
}

/* The refined solve of d_A (n2 x n2 complex, Hs2): on success the refined
   vectors are in the first maxn columns of d_A and ko[1..maxn] holds the
   refined eigenvalues.  Returns 1, 0 (d_A must be reloaded: the caller
   solves Hs2 in FP64, *persistent says for how long), or -1 when the warm
   start found the previous vectors too far from the matrix (nothing
   changed). */
static int ClusterNonCol_RefinedSolveDevice(int n2, int maxn, double *ko, int transient, int warm, int *persistent)
{
    ClusterNonColDeviceSolve *ds = &ClusterNonCol_dsolve;
    EigenRefineDevice dev;
    EigenRefineProblem pb;
    EigenRefineReport rep;
    ClusterNonColOccupationCtx occ = {0.0};
    double *e0 = (double *)ClusterNonCol_MallocArray((size_t)n2, sizeof(double), "refinement eigenvalues");
    double *f0 = (double *)ClusterNonCol_MallocArray((size_t)n2, sizeof(double), "refinement occupations");
    int rc;

    memset(&dev, 0, sizeof(dev));
    dev.hipblas = ds->hipblas;
    dev.d_work = &ds->d_work;
    dev.d_work_bytes = &ds->d_work_bytes;
    dev.d_info = ds->d_info;
    dev.try_malloc = ClusterNonCol_TryDeviceMalloc;
    dev.host_malloc = ClusterNonCol_RefineHostMalloc;

    memset(&pb, 0, sizeof(pb));
    pb.cplx = 1;
    pb.n = n2;
    pb.maxn = maxn;
    pb.iterations = ClusterNonCol_RefineIterations();
    pb.a = ds->d_A;
    pb.w = ds->d_w;
    pb.fp32 = ds->d_arena;
    pb.region = ds->arena_bytes;
    pb.x = ds->d_x;
    pb.transient = transient;
    pb.warm = warm;
    pb.defaulted = ClusterNonCol_refine.defaulted;
    pb.occupation = ClusterNonCol_RefineFermi;
    pb.occupation_ctx = &occ;

    memset(&rep, 0, sizeof(rep));
    *persistent = 0;
    rc = openmx_eigen_refine_prepare(&ClusterNonCol_refine_state, &dev, &pb, e0, &rep);
    if (rc == 1) {
        ClusterNonCol_RefineOccupations(e0, n2, maxn, f0, &occ.mu);
        rc = openmx_eigen_refine_finish(&ClusterNonCol_refine_state, &dev, &pb, f0, &rep);
    }
    free(e0);
    free(f0);
    *persistent = rep.persistent;
    if (rc == 1) {
        wait_hipfunc(hipMemcpy(ko + 1, ds->d_w, (size_t)maxn * sizeof(double), hipMemcpyDeviceToHost));
        if (3 <= level_stdout) {
            printf("<Cluster_DFT_NonCol> refined %d columns: max |r| %.1e, max |s| %.1e, delta %.1e; Rayleigh-Ritz in "
                   "%d partially occupied clusters (largest %d, chained within %.1e)\n",
                   rep.columns, rep.max_r, rep.max_s, rep.delta, rep.rr_clusters, rep.rr_largest, rep.chain);
            fflush(stdout);
        }
    }
    return rc;
}

/* the products of the device forward transform: GEMMul8 with the solve
   reserve of the bridge (the solve's own buffers are on the device by now,
   and the general reserve would send the products to plain FP64), then
   native hipBLAS */
static hipblasStatus_t ClusterNonCol_TryDeviceDgemmSolve(hipblasHandle_t handle, hipblasOperation_t transa,
                                                        hipblasOperation_t transb, int m, int n, int k,
                                                        const double *A, const double *B, double *C,
                                                        const char *where)
{
    const double alpha = 1.0;
    const double beta = 0.0;
    hipblasStatus_t status;
    const long long native_before = openmx_gemmul8NativeCalls(0);

    status = openmx_gemmul8DgemmSolve(handle, transa, transb, m, n, k, &alpha, A, m, B, k, &beta, C, m);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ClusterNonCol_LogGemmBackendOnce("D", openmx_gemmul8NativeCalls(0) == native_before ?
                                       "GEMMul8" : "native hipBLAS", m, n, k);
        return status;
    }

    ClusterNonCol_LogGemmRetry(where, "D", status, m, n, k);
    status = hipblasDgemm(handle, transa, transb, m, n, k, &alpha, A, m, B, k, &beta, C, m);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ClusterNonCol_LogGemmBackendOnce("D", "native hipBLAS", m, n, k);
    }
    return status;
}

/* The forward transform S^T M S of the real blocks d_mats[0..count) on the
   device, in place, with d_C (n x n) as the scratch: the products of the
   host-staged transform (ClusterNonCol_TransformRealDenseMatricesGPU) on
   blocks that stay on the device, in the same order and shapes, so every
   block is that transform's to the bit as long as both run GEMMul8.  A block
   whose product fails in GEMMul8 and in native hipBLAS (which then wrote
   nothing) is transformed on the CPU with the host copy S. */
static void ClusterNonCol_TransformDeviceMatrices(hipblasHandle_t handle, int n, const double *d_S, const double *S,
                                                  double **d_mats, int count, double *d_C)
{
    size_t const nn = ClusterNonCol_CheckedMulCount((size_t)n, (size_t)n, "device transform block");

    for (int i = 0; i < count; i++) {
        hipblasStatus_t status;
        int done = 0;   /* the products that succeeded */

        status = ClusterNonCol_TryDeviceDgemmSolve(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, d_mats[i], d_S, d_C,
                                                   "ClusterNonCol_TransformDeviceMatrices:H*S");
        if (status == HIPBLAS_STATUS_SUCCESS) {
            done = 1;
            status = ClusterNonCol_TryDeviceDgemmSolve(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, n, d_S, d_C,
                                                       d_mats[i], "ClusterNonCol_TransformDeviceMatrices:S^T*(H*S)");
        }
        if (status != HIPBLAS_STATUS_SUCCESS) {
            double *M = (double *)ClusterNonCol_MallocArray(nn, sizeof(double), "transform block on the CPU");
            double *work = (double *)ClusterNonCol_MallocArray(nn, sizeof(double), "transform work on the CPU");

            ClusterNonCol_LogGemmCpuFallback("ClusterNonCol_TransformDeviceMatrices", "D", "native hipBLAS", status,
                                             n, n, n);
            wait_hipfunc(hipDeviceSynchronize());
            if (done == 0) {
                wait_hipfunc(hipMemcpy(M, d_mats[i], nn * sizeof(double), hipMemcpyDeviceToHost));
                ClusterNonCol_DgemmCPU(HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, M, S, work);
            }
            else {
                wait_hipfunc(hipMemcpy(work, d_C, nn * sizeof(double), hipMemcpyDeviceToHost));
            }
            ClusterNonCol_DgemmCPU(HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, n, S, work, M);
            wait_hipfunc(hipMemcpy(d_mats[i], M, nn * sizeof(double), hipMemcpyHostToDevice));
            free(work);
            free(M);
        }
    }
}

/* The root dense solve of a step: the overlap (rebuilt with the geometry)
   and the Hamiltonian blocks gathered to the root, the forward transform,
   Hs2, the eigensolver (refined FP32 or FP64) and the back transform.  On
   the device when it has room (ClusterNonCol_DeviceSolveEnsure): the blocks
   go from the gather straight into device matrices -- six in the slots of
   d_A, the two of H11 in the arena, the gather staged through the last two
   slots -- are transformed there with the solve's hipBLAS handle and
   assembled into Hs2 in d_A by a kernel, which spares the transforms' own
   handle, buffers and uploads, the host assembly and the 505 MB upload of
   Hs2 (n2 = 5616) of the former path; only the first MaxN states of the
   vectors come back, to the page-locked staging the stash lends to the
   energy density matrix after the SCF.  Otherwise everything is staged
   through the host as before.  *dense_evec_out (root) receives those
   vectors, state k of basis i at k + i *dense_evec_ld_out. */
static void ClusterNonCol_MagmaRootDensePath(int SCF_iter, double *ko, double *****nh, double *****ImNL,
                                             double ****CntOLP, int *MP, int n, int n2, int MaxN,
                                             int myid, dcomplex **dense_evec_out,
                                             int *dense_evec_ld_out)
{
    const int owner=(myid==Host_ID);
    const size_t nn=(size_t)n*n, n2n2=(size_t)n2*n2;
    ClusterNonColHostStage *cache=&ClusterNonCol_host_stage;
    double *host_blocks=NULL;
    int rebuild=(SCF_iter==1 || !cache->S_valid || cache->n!=n);
    double *S=NULL, *blocks[6]={NULL,NULL,NULL,NULL,NULL,NULL};
    dcomplex *q=NULL, *out=NULL;
    double t0=0.0;
    const int profile=owner && ClusterNonCol_ProfileEnabled();
    if (profile) dtime(&t0);
    *dense_evec_out=NULL; *dense_evec_ld_out=0;
    ClusterNonCol_refine.last_refined=0;
    if (owner) {
        S=ClusterNonCol_HostStageS(n);
        host_blocks=ClusterNonCol_MallocArray(7*nn,sizeof(double),"host Hamiltonian blocks");
        for (int b=0;b<6;++b) blocks[b]=host_blocks+b*nn;
        q=ClusterNonCol_MallocArray(n2n2,sizeof(dcomplex),"host spinor Hamiltonian");
    }
    MPI_Bcast(&rebuild,1,MPI_INT,Host_ID,mpi_comm_level1);
    if (rebuild) {
        Patch2Device_Cluster_NonCol_Owner(CntOLP,S,MP,owner,n,0);
        if (owner) {
            if (!openmx_mgpu_host_solve(n,n,0,S,n,ko+1))
                ClusterNonCol_AbortWithMessage(openmx_mgpu_host_error());
#pragma omp parallel for
            for (int j=0;j<n;++j) {
                double scale=1.0/sqrt(fmax(ko[j+1],1.0e-10));
                for (int i=0;i<n;++i) S[(size_t)j*n+i]*=scale;
            }
            cache->S_valid=1;
        }
    }
    if (profile) ClusterNonCol_ProfileLap(NCPROF_OVERLAP,&t0);
    Patch2Device_Cluster_NonCol_Owner(nh[0],blocks[0],MP,owner,n,0);
    Patch2Device_Cluster_NonCol_Owner(nh[2],blocks[1],MP,owner,n,0);
    Patch2Device_Cluster_NonCol_Owner(nh[1],blocks[2],MP,owner,n,0);
    Patch2Device_Cluster_NonCol_Owner(ImNL[0],blocks[3],MP,owner,n,0);
    Patch2Device_Cluster_NonCol_Owner(nh[3],blocks[4],MP,owner,n,0);
    Patch2Device_Cluster_NonCol_Owner(ImNL[1],blocks[5],MP,owner,n,0);
    Patch2Device_Cluster_NonCol_Owner(ImNL[2],owner?(host_blocks+6*nn):NULL,MP,owner,n,0);
    if (profile) ClusterNonCol_ProfileLap(NCPROF_GATHER,&t0);
    if (owner) {
#pragma omp parallel for
        for (size_t i=0;i<nn;++i) blocks[4][i]+=(host_blocks+6*nn)[i];
        for (int b=0;b<6;++b) {
            if (!openmx_mgpu_host_dgemm(HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,blocks[b],n,S,n,(host_blocks+6*nn),n) ||
                !openmx_mgpu_host_dgemm(HIPBLAS_OP_T,HIPBLAS_OP_N,n,n,n,S,n,(host_blocks+6*nn),n,blocks[b],n))
                ClusterNonCol_AbortWithMessage(openmx_mgpu_host_error());
        }
#pragma omp parallel for collapse(2)
        for (int j=0;j<n;++j) for (int i=0;i<n;++i) {
            size_t ij=(size_t)j*n+i, full=(size_t)j*n2+i;
            q[full]=(dcomplex){blocks[0][ij],blocks[3][ij]};
            q[full+(size_t)n*n2]=(dcomplex){blocks[1][ij],blocks[4][ij]};
            q[(size_t)i*n2+j+n]=(dcomplex){blocks[1][ij],-blocks[4][ij]};
            q[full+(size_t)n*n2+n]=(dcomplex){blocks[2][ij],blocks[5][ij]};
        }
        if (profile) ClusterNonCol_ProfileLap(NCPROF_FORWARD,&t0);
        if (!openmx_mgpu_host_solve(n2,MaxN,1,q,n2,ko+1))
            ClusterNonCol_AbortWithMessage(openmx_mgpu_host_error());
        if (profile) ClusterNonCol_ProfileLap(NCPROF_EIGEN,&t0);
        dcomplex *vt=ClusterNonCol_MallocArray((size_t)MaxN*n2,sizeof(dcomplex),"host backtransform transpose");
        out=ClusterNonCol_HostStageEVec((size_t)MaxN*n2);
        memset(out,0,(size_t)MaxN*n2*sizeof(dcomplex));
#pragma omp parallel for collapse(2)
        for (int j=0;j<n2;++j) for (int i=0;i<MaxN;++i)
            vt[(size_t)j*MaxN+i]=q[(size_t)i*n2+j];
        for (int half=0;half<2;++half) {
            if (!openmx_mgpu_host_dgemm(HIPBLAS_OP_N,HIPBLAS_OP_T,2*MaxN,n,n,
                (double *)(vt+(size_t)half*n*MaxN),2*MaxN,S,n,
                (double *)(out+(size_t)half*n*MaxN),2*MaxN))
                ClusterNonCol_AbortWithMessage(openmx_mgpu_host_error());
        }
        free(vt);
        if (profile) ClusterNonCol_ProfileLap(NCPROF_BACK,&t0);
        *dense_evec_out=out;
        *dense_evec_ld_out=MaxN;
        free(host_blocks); free(q);
    }
    MPI_Bcast(ko,n2+1,MPI_DOUBLE,Host_ID,mpi_comm_level1);
}

static void ClusterNonCol_GpuSolverRootDensePath(int SCF_iter, double *ko, double *****nh, double *****ImNL,
                                                double ****CntOLP, int *MP, int n, int n2, int MaxN,
                                                int myid, dcomplex **dense_evec_out, int *dense_evec_ld_out)
{
    if (ClusterNonCol_mgpu_host_dense) {
        ClusterNonCol_MagmaRootDensePath(SCF_iter,ko,nh,ImNL,CntOLP,MP,n,n2,MaxN,myid,
            dense_evec_out,dense_evec_ld_out);
        return;
    }

    ClusterNonColHostStage *hs = &ClusterNonCol_host_stage;
    ClusterNonColDeviceSolve *ds = &ClusterNonCol_dsolve;
    const int owns_dense = (myid == Host_ID);
    const int profile = owns_dense && ClusterNonCol_ProfileEnabled();
    const int cache_ready = Set_Hamiltonian_GpuSolver_Packed_CacheReady();
    const int cache_order = Set_Hamiltonian_GpuSolver_Packed_OrderMode();
    const char *setham_cache_env = getenv("OPENMX_CLUSTER_NC_SETHAM_CACHE");
    const int use_setham_packed_cache = (cache_ready && cache_order == 1 &&
                                         (setham_cache_env==NULL || atoi(setham_cache_env)!=0));
    int rebuild_s = (SCF_iter == 1 || !hs->S_valid || hs->n != n || ClusterNonCol_root_s_stale);
    int device = 0;   /* the root's blocks, Hs2 and vectors stay on the device */
    double *S = NULL;
    /* rHs11, rHs12, rHs22, iHs11, iHs12, iHs22 and the gather of ImNL[2]: on
       the host or on the device */
    double *mats[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    double *host_blocks = NULL;
    double *d_scratch = NULL;
    dcomplex *Hs2 = NULL;
    dcomplex *dense_evec = NULL;
    size_t nn = 0;
    size_t n2n2 = 0;
    int solve_state[2] = {0, 0};   /* refined, persistent reason of a fallback */
    double t0 = 0.0;

    *dense_evec_out = NULL;
    *dense_evec_ld_out = 0;
    ds->dense_rows = 0;

    if (use_setham_packed_cache) {
        Set_Hamiltonian_GpuSolver_SetMP(MP);
    }

    if (ClusterNonCol_GpuVerbose() && myid == Host_ID) {
        static int logged_root_dense = 0;

        if (!logged_root_dense) {
            fprintf(stderr,
                    "<Cluster> rank %d: dense GPU path uses one selected rank for dense matrix generation, "
                    "GEMM, eigensolver, and density-matrix generation (n=%d, n2=%d).\n",
                    Host_ID, n, n2);
            fflush(stderr);
            logged_root_dense = 1;
        }
    }

    if (Set_Hamiltonian_OpenMP_Rank_Is_Selected()) {
        set_hip_default_device_from_local_rank_noncollective();
    }

    if (owns_dense) {
        nn = ClusterNonCol_CheckedMulCount((size_t)n, (size_t)n, "root dense real matrix");
        n2n2 = ClusterNonCol_CheckedMulCount((size_t)n2, (size_t)n2, "root dense complex matrix");
        S = ClusterNonCol_HostStageS(n);
        if (!hs->S_valid) rebuild_s = 1;
    }

    MPI_Bcast(&rebuild_s, 1, MPI_INT, Host_ID, mpi_comm_level1);
    ClusterNonCol_root_s_stale = 0;

    if (profile) ClusterNonCol_ProfileLap(-1, &t0);

    if (rebuild_s) {
        if (use_setham_packed_cache) {
            if (owns_dense) {
                if (!Set_Hamiltonian_GpuSolver_Packed_OwnsCache()) {
                    ClusterNonCol_AbortWithMessage("Set_Hamiltonian packed overlap cache is not owned by this rank.");
                }
                ClusterNonCol_LoadDeviceDenseFromSetHamCache(Set_Hamiltonian_GpuSolver_Packed_Overlap(), MP, n, S, 0);
            }
        }
        else {
            Patch2Device_Cluster_NonCol_Owner(CntOLP, S, MP, owns_dense, n, 0);
        }

        if (owns_dense) {
            ClusterNonCol_GpuSolver_DenseDsyevx(S, NULL, ko, n, n,
                                               "Cluster_DFT_NonCol root overlap");
            for (int l = 1; l <= n; l++) {
                if (ko[l] < 1.0e-10) ko[l] = 1.0e-10;
                ko[l] = 1.0 / sqrt(ko[l]);
            }

            for (int col = 0; col < n; col++) {
                double scale = ko[col + 1];
                for (int row = 0; row < n; row++) {
                    S[(size_t)row + (size_t)col * (size_t)n] *= scale;
                }
            }

            hs->S_valid = 1;
            ds->S_valid = 0;   /* the device copy is of the former overlap */
        }
    }

    if (owns_dense) {
        int const refine = (ClusterNonCol_refine.last_refined && !ClusterNonCol_refine.force_fp64);

        device = ClusterNonCol_DeviceSolveEnabled() && ClusterNonCol_DeviceSolveEnsure(n, n2, MaxN, refine);
        if (device) {
            double *slot = (double *)ds->d_A;   /* eight n x n slots */

            if (!ds->S_valid) {
                wait_hipfunc(hipMemcpy(ds->d_S, S, nn * sizeof(double), hipMemcpyHostToDevice));
                ds->S_valid = 1;
            }
            /* the blocks of the right half of Hs2 in the left half of d_A,
               which the assembly writes last, those of H11 outside d_A */
            mats[0] = (double *)ds->d_arena;
            mats[3] = (double *)ds->d_arena + nn;
            mats[1] = slot;
            mats[4] = slot + nn;
            mats[2] = slot + 2 * nn;
            mats[5] = slot + 3 * nn;
            mats[6] = slot + 4 * nn;
            d_scratch = slot + 5 * nn;
            ClusterNonCol_DenseStageSet(slot + 6 * nn, (int *)(slot + 7 * nn), nn);
        }
        else {
            host_blocks = (double *)ClusterNonCol_MallocArray(ClusterNonCol_CheckedMulCount(nn, 7, "root dense blocks"),
                                                              sizeof(double), "root dense blocks");
            for (int i = 0; i < 7; i++) mats[i] = host_blocks + (size_t)i * nn;
            Hs2 = (dcomplex *)ClusterNonCol_MallocArray(n2n2, sizeof(dcomplex), "root dense Hs2");
        }
    }

    if (profile) ClusterNonCol_ProfileLap(NCPROF_OVERLAP, &t0);

    if (use_setham_packed_cache) {
        if (owns_dense) {
            if (!Set_Hamiltonian_GpuSolver_Packed_OwnsCache()) {
                ClusterNonCol_AbortWithMessage("Set_Hamiltonian packed Hamiltonian cache is not owned by this rank.");
            }
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(Set_Hamiltonian_GpuSolver_Packed_H(0), MP, n, mats[0], device);
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(Set_Hamiltonian_GpuSolver_Packed_H(1), MP, n, mats[2], device);
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(Set_Hamiltonian_GpuSolver_Packed_H(2), MP, n, mats[1], device);
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(Set_Hamiltonian_GpuSolver_Packed_H(3), MP, n, mats[4], device);
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(Set_Hamiltonian_GpuSolver_Packed_ImNL(0), MP, n, mats[3],
                                                         device);
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(Set_Hamiltonian_GpuSolver_Packed_ImNL(1), MP, n, mats[5],
                                                         device);
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(Set_Hamiltonian_GpuSolver_Packed_ImNL(2), MP, n, mats[6],
                                                         device);
        }
    }
    else {
        Patch2Device_Cluster_NonCol_Owner(nh[0],   mats[0], MP, owns_dense, n, device);
        Patch2Device_Cluster_NonCol_Owner(nh[1],   mats[2], MP, owns_dense, n, device);
        Patch2Device_Cluster_NonCol_Owner(nh[2],   mats[1], MP, owns_dense, n, device);
        Patch2Device_Cluster_NonCol_Owner(nh[3],   mats[4], MP, owns_dense, n, device);
        Patch2Device_Cluster_NonCol_Owner(ImNL[0], mats[3], MP, owns_dense, n, device);
        Patch2Device_Cluster_NonCol_Owner(ImNL[1], mats[5], MP, owns_dense, n, device);
        Patch2Device_Cluster_NonCol_Owner(ImNL[2], mats[6], MP, owns_dense, n, device);
    }
    if (owns_dense) ClusterNonCol_DenseStageSet(NULL, NULL, 0);

    if (profile) ClusterNonCol_ProfileLap(NCPROF_GATHER, &t0);

    if (owns_dense && device) {
        int const refine = (ClusterNonCol_refine.last_refined && !ClusterNonCol_refine.force_fp64);
        int rc = 0;

        /* iHs12 += ImNL[2], S^T M S of the six blocks, Hs2 */
        if (ClusterNonCol_AddDense_HIP(ds->hipblas, nn, mats[4], mats[6]) != 0) {
            ClusterNonCol_AbortWithMessage("Cluster_DFT_NonCol: the device sum of the off-diagonal blocks failed.");
        }
        ClusterNonCol_TransformDeviceMatrices(ds->hipblas, n, ds->d_S, S, mats, 6, d_scratch);
        if (ClusterNonCol_AssembleHs2_HIP(ds->hipblas, n, mats[0], mats[3], mats[1], mats[4], mats[2], mats[5],
                                          ds->d_A) != 0) {
            ClusterNonCol_AbortWithMessage("Cluster_DFT_NonCol: the device assembly of Hs2 failed.");
        }

        if (profile) ClusterNonCol_ProfileLap(NCPROF_FORWARD, &t0);

        if (refine) {
            int warm = openmx_eigen_refine_warm_enabled() && ClusterNonCol_refine_state.basis_valid &&
                       ClusterNonCol_refine_state.basis_n == n2 && 1 < SCF_iter;

            if (warm && !ds->x_on_device) {
                if (ds->h_x_valid && ds->h_x != NULL) {
                    wait_hipfunc(hipMemcpy(ds->d_x, ds->h_x, n2n2 * sizeof(dcomplex), hipMemcpyHostToDevice));
                    ds->x_on_device = 1;
                }
                else warm = 0;
            }

            if (3 <= level_stdout) {
                printf("<Cluster_DFT_NonCol> SCF step %d: %s, %d refinement steps%s (OPENMX_EIGEN_REFINE)\n",
                       SCF_iter, warm ? "warm start from the previous vectors" : "FP32 eigensolver",
                       ClusterNonCol_RefineIterations(), MaxN == n2 ? " of all states" : "");
                fflush(stdout);
            }
            rc = ClusterNonCol_RefinedSolveDevice(n2, MaxN, ko, SCF_iter == 1, warm, &solve_state[1]);
            if (rc < 0) rc = ClusterNonCol_RefinedSolveDevice(n2, MaxN, ko, SCF_iter == 1, 0, &solve_state[1]);
            /* d_x now holds the new basis, or none */
            ds->x_on_device = (rc == 1);
            ds->h_x_valid = 0;
            /* a refinement that did not take place leaves in d_A the matrix
               made Hermitian from its lower triangle, the part (with the
               real diagonal) the FP64 eigensolver reads, which solves it as
               the NVIDIA version does */
        }
        if (rc == 1) {
            solve_state[0] = 1;
        }
        else {
            int mout = 0, l, info;

            /* the room of the refinement goes to the FP64 solve: its
               basis, the FP32 solver's workspace and the products'
               full-size scratch */
            if (refine) {
                wait_hipfunc(hipDeviceSynchronize());
                ClusterNonCol_DeviceFree(&ds->d_x);
                ds->x_on_device = 0;
                ClusterNonCol_DeviceFree(&ds->d_work);
                ds->d_work_bytes = 0;
                openmx_gemmul8TrimWorkspaces();
            }
            info = openmx_magma_zheevdx_gpu(n2, MaxN, ds->d_A, ko, &mout);

            if (info != 0 || mout != MaxN) {
                fprintf(stderr, "Cluster_DFT_NonCol root Hamiltonian: magma_zheevdx_gpu failed (info=%d, "
                        "%d of %d eigenpairs)\n", info, mout, MaxN);
                fflush(stderr);
                MPI_Abort(mpi_comm_level1, 1);
            }
            for (l = MaxN; 1 <= l; l--) ko[l] = ko[l - 1];
            ClusterNonCol_refine_state.basis_valid = 0;
        }

        if (profile) ClusterNonCol_ProfileLap(NCPROF_EIGEN, &t0);

        if (ClusterNonCol_BackTransformDevice_HIP(ds->hipblas, n, n2, MaxN, ds->d_A, ds->d_arena, ds->d_S) != 0) {
            ClusterNonCol_AbortWithMessage("Cluster_DFT_NonCol: the device back transform failed.");
        }
        ds->dense_rows = MaxN;

        if (profile) ClusterNonCol_ProfileLap(NCPROF_BACK, &t0);

        /* the post-SCF energy density matrix takes the vectors from the host
           copy (the first MaxN states of every basis function) */
        dense_evec = ClusterNonCol_HostStageEVec(ClusterNonCol_CheckedMulCount((size_t)MaxN, (size_t)n2,
                                                                               "root dense eigenvectors"));
        wait_hipfunc(hipMemcpy2D(dense_evec, (size_t)MaxN * sizeof(dcomplex), ds->d_A, (size_t)n2 * sizeof(dcomplex),
                                 (size_t)MaxN * sizeof(dcomplex), (size_t)n2, hipMemcpyDeviceToHost));
        *dense_evec_out = dense_evec;
        *dense_evec_ld_out = MaxN;

        if (profile) ClusterNonCol_ProfileLap(NCPROF_RELEASE, &t0);
    }
    else if (owns_dense) {
        double *rHs11 = mats[0], *rHs12 = mats[1], *rHs22 = mats[2];
        double *iHs11 = mats[3], *iHs12 = mats[4], *iHs22 = mats[5], *Cs = mats[6];
        double *transform_mats[6];

        for (size_t i = 0; i < nn; i++) {
            iHs12[i] += Cs[i];
        }

        for (int i = 0; i < 6; i++) transform_mats[i] = mats[i];
        ClusterNonCol_TransformRealDenseMatricesGPU(n, S, transform_mats, 6, Cs, ds->hipblas);

        for (int col = 0; col < n2; col++) {
            for (int row = 0; row < n2; row++) {
                if (row < n && col < n) {
                    Hs2[(size_t)row + (size_t)col * (size_t)n2].r = rHs11[(size_t)row + (size_t)col * (size_t)n];
                    Hs2[(size_t)row + (size_t)col * (size_t)n2].i = iHs11[(size_t)row + (size_t)col * (size_t)n];
                }
                else if (row < n && n <= col) {
                    int jj = col - n;
                    Hs2[(size_t)row + (size_t)col * (size_t)n2].r = rHs12[(size_t)row + (size_t)jj * (size_t)n];
                    Hs2[(size_t)row + (size_t)col * (size_t)n2].i = iHs12[(size_t)row + (size_t)jj * (size_t)n];
                }
                else if (n <= row && col < n) {
                    int ii = row - n;
                    Hs2[(size_t)row + (size_t)col * (size_t)n2].r = rHs12[(size_t)col + (size_t)ii * (size_t)n];
                    Hs2[(size_t)row + (size_t)col * (size_t)n2].i = -iHs12[(size_t)col + (size_t)ii * (size_t)n];
                }
                else {
                    int ii = row - n;
                    int jj = col - n;
                    Hs2[(size_t)row + (size_t)col * (size_t)n2].r = rHs22[(size_t)ii + (size_t)jj * (size_t)n];
                    Hs2[(size_t)row + (size_t)col * (size_t)n2].i = iHs22[(size_t)ii + (size_t)jj * (size_t)n];
                }
            }
        }

        if (profile) ClusterNonCol_ProfileLap(NCPROF_FORWARD, &t0);

        ClusterNonCol_refine_state.basis_valid = 0;
        ClusterNonCol_GpuSolver_DenseZheevx(Hs2, NULL, ko, n2, MaxN,
                                           "Cluster_DFT_NonCol root Hamiltonian");

        if (profile) ClusterNonCol_ProfileLap(NCPROF_EIGEN, &t0);

        dense_evec = ClusterNonCol_HostStageEVec(ClusterNonCol_CheckedMulCount((size_t)MaxN, (size_t)n2,
                                                                               "root dense eigenvectors"));
        ClusterNonCol_BackTransformHostStaged(n, n2, MaxN, Hs2, S, dense_evec, MaxN, ds->hipblas);
        *dense_evec_out = dense_evec;
        *dense_evec_ld_out = MaxN;

        if (profile) ClusterNonCol_ProfileLap(NCPROF_BACK, &t0);
    }

    MPI_Bcast(ko, n2 + 1, MPI_DOUBLE, Host_ID, mpi_comm_level1);

    /* every rank learns how the root solved, for the stop test of DFT.c: an
       FP64 fallback is no refined step, and a persistent reason for it ends
       the refined stage of this SCF cycle */
    MPI_Bcast(solve_state, 2, MPI_INT, Host_ID, mpi_comm_level1);
    if (0 < ClusterNonCol_RefineIterations()) {
        ClusterNonCol_refine.last_refined = solve_state[0];
        if (solve_state[0]) ClusterNonCol_refine.cycle_refined = 1;
        if (1 <= solve_state[1]) ClusterNonCol_refine.final_stage = 1;
        if (2 <= solve_state[1]) {
            ClusterNonCol_refine.iterations = 0;
            ClusterNonCol_refine.defaulted = 0;
        }
    }

    free(host_blocks);
    free(Hs2);
}

int Cluster_DFT_NonCol_MgpuCachedEDM(int n,int size_H1,int *MP,double *ko,double *****EDM,double *elapsed)
{
    int myid, valid=0, occupied=0;
    double start,end;
    const int n2=2*n;
    const size_t count=(size_t)size_H1*4;
    *elapsed=0;
    if (!ClusterNonCol_mgpu_host_dense) return 0;
    MPI_Comm_rank(mpi_comm_level1,&myid);
    if (myid==Host_ID) valid=ClusterNonCol_CachedGpuSolverDenseEVec && ClusterNonCol_CachedGpuSolverDenseN2==n2;
    MPI_Bcast(&valid,1,MPI_INT,Host_ID,mpi_comm_level1);
    if (!valid) return 0;
    if (count>INT_MAX) ClusterNonCol_AbortWithMessage("Post-SCF EDM exceeds MPI count limit");
    dtime(&start);
    ClusterNonCol_RootDMWorkspace_Ensure(n2,count,0);
    ClusterNonColRootDMWorkspace *ws=&ClusterNonCol_root_dm_workspace;
    ClusterNonColDMEntryCache *cache=&ClusterNonCol_dm_entry_cache;
    double *outputs[10]={0};
    if (myid==Host_ID) {
        memset(ws->dm_buffer,0,count*sizeof(double));
        ClusterNonCol_DMEntryCache_Ensure(MP,n);
        if (cache->entry_count!=size_H1) ClusterNonCol_AbortWithMessage("Post-SCF EDM entry count mismatch");
        occupied=ClusterNonCol_PrecomputeOccupations(1,ClusterNonCol_CachedGpuSolverDenseLd,ko,ws->occ,ws->occ_e);
        for (int c=0;c<4;c++) outputs[6+c]=ws->dm_buffer+(size_t)c*size_H1;
    }
    openmx_gpu_dm_collective(mpi_comm_level1,OPENMX_GPU_DM_NC,n,occupied,ClusterNonCol_CachedGpuSolverDenseLd,
        myid==Host_ID && occupied?size_H1:0,ClusterNonCol_CachedGpuSolverDenseEVec,1,
        cache->basis0,cache->basis1,NULL,0,NULL,NULL,ws->occ,ws->occ_e,NULL,outputs,0);
    MPI_Bcast(ws->dm_buffer,(int)count,MPI_DOUBLE,Host_ID,mpi_comm_level1);
    ClusterNonCol_StoreEDMBuffer(myid,size_H1,MP,EDM,ws->dm_buffer);
    dtime(&end); *elapsed=end-start;
    return 1;
}


void solve_evp_real_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, int *nblk,
                      int *mpi_comm_rows_int, int *mpi_comm_cols_int);

void elpa_solve_evp_real_2stage_double_impl_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2,
                                              int *nblk, int *na_cols1, int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld);

void solve_evp_complex_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, int *na_rows2_2,
                         int *nblk2, int *mpi_comm_rows_int, int *mpi_comm_cols_int );

void elpa_solve_evp_complex_2stage_double_impl_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2,
                                                 int *na_rows2_2, int *nblk2, int *na_cols2,
                                                 int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld );



static void Save_DOS_NonCol(int n, int n2, int MaxN, int *MP, double ****OLP0, dcomplex *EVec1, double *ko);
static void Save_LCAO_NonCol(int n, int n2, int MaxN, int *MP, double ****OLP0, dcomplex *EVec1, double *ko);
static dcomplex Lapack_LU_Zinverse(int n, dcomplex *A);

static double Calc_Oscillator_Strength( int n, int n2, int UMOmax, int Nocc[2], int *MP2,
                                        int *ind2ind,
                                        dcomplex *Z0, dcomplex *A, dcomplex *Z,
					dcomplex *Ax, dcomplex *Ay, dcomplex *Az,
					double Utot1, double Utot2,
					double XANES_Res[10] );


static void Calc_XANES_NonCol( int n, int n2, int MaxN, int *MP,
			       int *is2, int *ie2, dcomplex *EVec1, double *ko);

double Calc_DM_Cluster_non_collinear_ScaLAPACK(
    int calc_flag,
    int myid,
    int numprocs,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    double *****CDM,
    double *****iDM0,
    double *****EDM,
    double *ko,
    double *DM1,
    double *Work1,
    dcomplex *EVec1 );


double Calc_ParDM_Cluster_non_collinear_ScaLAPACK(
    int calc_flag,
    int myid,
    int numprocs,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    double *ko,
    double *DM1,
    double *Work1,
    dcomplex *EVec1 );




double Cluster_DFT_NonCol(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double *ko,
                   double *****nh,
                   double *****ImNL,
                   double ****CntOLP,
                   double *****CDM,
                   double *****EDM,
                   double Eele0[2], double Eele1[2],
                   int *MP,
		   int *is2,
		   int *ie2,
		   double *Ss,
		   double *Cs,
		   double *rHs11,
		   double *rHs12,
		   double *rHs22,
		   double *iHs11,
		   double *iHs12,
		   double *iHs22,
                   dcomplex *Ss2,
                   dcomplex *Hs2,
                   dcomplex *Cs2,
		   double *DM1,
		   int size_H1,
                   dcomplex *EVec1,
                   double *Work1)
{
  int mgpu_scope_previous=openmx_mgpu_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag);
  static int firsttime=1;
  int i,j,l,n,n2,n1,i1,i1s,j1,k1,l1;
  int ii1,jj1,jj2,ki,kj;
  int wan,HOMO0,HOMO1;
  int po,num0,num1;
  int mul,m,wan1,Gc_AN,bcast_flag;
  double time0,lumos;
  int ct_AN,k,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum,loopN;
  int MA_AN,LB_AN,GB_AN,Bnum,MaxN;
  double TZ,my_sum,sum,sumE,max_x=60.0;
  double My_Eele1[2];
  double Num_State,x,FermiF,Dnum,Dnum2;
  double FermiF2,x2,diffF;
  double dum,ChemP_MAX,ChemP_MIN;
  double TStime,TEtime;
  double FermiEps = 1.0e-13;
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  char file_EV[YOUSO10] = ".EV";
  FILE *fp_EV;
  char buf[fp_bsize];          /* setvbuf */
  double time1,time2,time3,time4,time5,time6,time7;
  double stime,etime;
  double av_num,tmp;
  int ig,jg;
  int numprocs,myid,ID;
  int use_gpusolver_direct_cluster_dm = 0;
  int gpusolver_dense_evec_ld = 0;
  int ke,ks,nblk_m,nblk_m2;
  int ID0,IDS,IDR,Max_Num_Snd_EV,Max_Num_Rcv_EV;
  int *Num_Snd_EV,*Num_Rcv_EV;
  int *index_Snd_i,*index_Snd_j,*index_Rcv_i,*index_Rcv_j;
  double *EVec_Snd,*EVec_Rcv;
  dcomplex *gpusolver_dense_evec = NULL;
  int ZERO=0,ONE=1,info;
  double Re_alpha = 1.0; double Re_beta = 0.0;
  dcomplex alpha = {1.0,0.0}; dcomplex beta = {0.0,0.0};

  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n  = n + Spe_Total_CNO[wanA];
  }
  n2 = 2*n;

  /* GPU dispatch (added by H.Kawai): assign HIP/OpenMP target device when GPUSOLVER is requested */
    if (scf_eigen_lib_flag == GPUSOLVER && n2 >= Cluster_DFT_NonCol_GpuSwitchNum() &&
        Set_Hamiltonian_OpenMP_Rank_Is_Selected()) {
      set_hip_default_device_from_local_rank_noncollective();
    }

  use_gpusolver_direct_cluster_dm =
    (scf_eigen_lib_flag == GPUSOLVER && gpusolver2_flag == 0 && (openmx_mgpu_host_required() || Cluster_DFT_NonCol_GpuSwitchNum() <= n2) &&
     ClusterNonCol_GpuDiagFits(n2,myid) &&
     (getenv("OPENMX_CLUSTER_GPU_ROOT_DENSE")==NULL ||
      atoi(getenv("OPENMX_CLUSTER_GPU_ROOT_DENSE"))!=0) &&
     (strcasecmp(mode,"scf") == 0 || strcasecmp(mode,"fp64") == 0) &&
     MO_fileout != 1 && xanes_calc != 1 && xanes_gs_fileout != 1 &&
     !cal_partial_charge && !Dos_fileout && !DosGauss_fileout);

  /****************************************************
                  allocation of arrays
  ****************************************************/

  Num_Snd_EV = (int*)malloc(sizeof(int)*numprocs);
  Num_Rcv_EV = (int*)malloc(sizeof(int)*numprocs);

  /* initialize variables of measuring elapsed time */

  if (measure_time){
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time4 = 0.0;
    time5 = 0.0;
    time6 = 0.0;
    time7 = 0.0;
  }

  /* ***************************************************
                  total core charge
  *************************************************** */

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ += Spe_Core_Charge[wan];
  }

  /* ***************************************************
         find the numbers of partions for MPI
  *************************************************** */

  /* find the maximum states in solved eigenvalues */

  if (SCF_iter==1){
    MaxN = n2;
    if (use_gpusolver_direct_cluster_dm && strcasecmp(mode,"scf")==0){
      MaxN = ClusterNonCol_RefineFirstStepStates(n2,TZ);
    }
  }
  else {

    if      ( strcasecmp(mode,"scf")==0 || strcasecmp(mode,"fp64")==0 )
      lumos = (double)n2*0.20;
    else if ( strcasecmp(mode,"dos")==0 )
      lumos = (double)n2*0.40;
    else if ( strcasecmp(mode,"lcaoout")==0 )
      lumos = (double)n2*0.40;
    else if ( strcasecmp(mode,"xanes")==0 )
      lumos = (double)n2*0.40;

    if (lumos<400.0) lumos = 400.0;
    MaxN = Cluster_HOMO[0] + (int)lumos;
    if (n2<MaxN) MaxN = n2;

    if (cal_partial_charge) MaxN = n2;
  }

  if ( numprocs<=MaxN ){

    av_num = (double)MaxN/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is2[ID] = (int)(av_num*(double)ID) + 1;
      ie2[ID] = (int)(av_num*(double)(ID+1));
    }

    is2[0] = 1;
    ie2[numprocs-1] = MaxN;
  }

  else{
    for (ID=0; ID<MaxN; ID++){
      is2[ID] = ID + 1;
      ie2[ID] = ID + 1;
    }
    for (ID=MaxN; ID<numprocs; ID++){
      is2[ID] = 1;
      ie2[ID] = 0;
    }
  }

  /* making data structure of MPI communicaition for eigenvectors */

  for (ID=0; ID<numprocs; ID++){
    Num_Snd_EV[ID] = 0;
    Num_Rcv_EV[ID] = 0;
  }

  for (i=0; i<na_rows2; i++){

    ig = np_rows2*nblk2*((i)/nblk2) + (i)%nblk2 + ((np_rows2+my_prow2)%np_rows2)*nblk2 + 1;

    po = 0;
    for (ID=0; ID<numprocs; ID++){
      if (is2[ID]<=ig && ig <=ie2[ID]){
        po = 1;
        ID0 = ID;
        break;
      }
    }

    if (po==1) Num_Snd_EV[ID0] += na_cols2;
  }

  for (ID=0; ID<numprocs; ID++){
    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;
    if (ID!=0){
      MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, mpi_comm_level1, &request);
      MPI_Recv(&Num_Rcv_EV[IDR], 1, MPI_INT, IDR, 999, mpi_comm_level1, &stat);
      MPI_Wait(&request,&stat);
    }
    else{
      Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
    }
  }

  Max_Num_Snd_EV = 0;
  Max_Num_Rcv_EV = 0;
  for (ID=0; ID<numprocs; ID++){
    if (Max_Num_Snd_EV<Num_Snd_EV[ID]) Max_Num_Snd_EV = Num_Snd_EV[ID];
    if (Max_Num_Rcv_EV<Num_Rcv_EV[ID]) Max_Num_Rcv_EV = Num_Rcv_EV[ID];
  }

  Max_Num_Snd_EV++;
  Max_Num_Rcv_EV++;

  index_Snd_i = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  index_Snd_j = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  EVec_Snd = (double*)malloc(sizeof(double)*Max_Num_Snd_EV*2);
  index_Rcv_i = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  index_Rcv_j = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  EVec_Rcv = (double*)malloc(sizeof(double)*Max_Num_Rcv_EV*2);

  /* print memory size */

  if (firsttime && memoryusage_fileout){
    PrintMemory("Cluster_DFT_NonCol: Num_Snd_EV",sizeof(int)*numprocs,NULL);
    PrintMemory("Cluster_DFT_NonCol: Num_Ecv_EV",sizeof(int)*numprocs,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Snd_i",sizeof(int)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Snd_j",sizeof(int)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Rcv_i",sizeof(int)*Max_Num_Rcv_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Rcv_j",sizeof(int)*Max_Num_Rcv_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: EVec_Snd",sizeof(double)*Max_Num_Snd_EV*2,NULL);
    PrintMemory("Cluster_DFT_NonCol: EVec_Rcv",sizeof(double)*Max_Num_Rcv_EV*2,NULL);
  }
  firsttime=0;

  if (use_gpusolver_direct_cluster_dm){

    if (measure_time) dtime(&stime);

    ClusterNonCol_ReleaseGpuSolverCachedEVec(myid);

    for (i1=0; i1<=n2; i1++){ ko[i1] = 1.0e+5; }

    if (SCF_iter==1) ClusterNonCol_elpa_ss_ready = 0;
    ClusterNonCol_RefineBeginSolve(SCF_iter,n2);
    ClusterNonCol_refine.force_fp64 = (strcasecmp(mode,"fp64")==0);
    ClusterNonCol_GpuSolverRootDensePath(SCF_iter,ko,nh,ImNL,CntOLP,MP,n,n2,MaxN,myid,
                                        &gpusolver_dense_evec,&gpusolver_dense_evec_ld);
    ClusterNonCol_refine.force_fp64 = 0;

    if (measure_time){
      dtime(&etime);
      time3 += etime - stime;
    }
  }
  else {

  /* the ELPA path solves in FP64: nothing of this cycle is refined */
  ClusterNonCol_refine.last_refined = 0;
  ClusterNonCol_refine.cycle_refined = 0;
  /* and the vectors of an earlier root dense solve are not this solve's:
     the post-SCF scatter must not take them over EVec1 */
  ClusterNonCol_ReleaseGpuSolverCachedEVec(myid);

  /* ***************************************************
            diagonalize the overlap matrix
  *************************************************** */

  if (SCF_iter==1 || !ClusterNonCol_elpa_ss_ready){

    if (measure_time) dtime(&stime);

    ClusterNonCol_elpa_ss_ready = 1;
    ClusterNonCol_root_s_stale = 1;

    Overlap_Cluster_Ss(CntOLP,Cs,MP,0);

    MPI_Comm_split(mpi_comm_level1,my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(mpi_comm_level1,my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

    /* diagonalize Cs */

    if (ClusterNonCol_UseGs2(n2)){

      /* distributed multi-GPU eigensolver (ELPA, AMD GPU kernels) */
      int gs2_info = openmx_gs2_eigen_real(n, n, Cs, descC, &ko[1], Ss, descS);
      if (gs2_info!=0){
        printf("Cluster_DFT_NonCol: the gpusolver2 overlap eigensolver failed (info=%d)\n",gs2_info);
        MPI_Abort(mpi_comm_level1,1);
      }
    }
    else if (scf_eigen_lib_flag==1){
      F77_NAME(solve_evp_real,SOLVE_EVP_REAL)( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk,
                                               &mpi_comm_rows_int, &mpi_comm_cols_int );
    }
    else if (scf_eigen_lib_flag==GPUSOLVER && Cluster_DFT_NonCol_GpuSwitchNum()<=n2 &&
            ClusterNonCol_cpu_fallback_n2!=n2 && !ClusterNonCol_mgpu_host_dense && na_rows==n && na_cols==n){
      ClusterNonCol_GpuSolver_DenseDsyevx(Cs,Ss,ko,n,n,"Cluster_DFT_NonCol overlap");
    }
    else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){

#ifndef kcomp

      int mpiworld;
      mpiworld = MPI_Comm_c2f(mpi_comm_level1);

      F77_NAME(elpa_solve_evp_real_2stage_double_impl,ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)
	( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols,
          &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#endif
    }

    MPI_Comm_free(&mpi_comm_rows);
    MPI_Comm_free(&mpi_comm_cols);

    /* print to the standard output */

    if (2<=level_stdout){
      for (l=1; l<=n; l++){
	printf("  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[l]);
      }
    }

    /* minus eigenvalues to 1.0e-10 */

    for (l=1; l<=n; l++){
      if (ko[l]<1.0e-10) ko[l] = 1.0e-10;
      ko[l] = 1.0/sqrt(ko[l]);
    }

    /* calculate S*1/sqrt(ko) */

    for(i=0; i<na_rows; i++){
      for(j=0; j<na_cols; j++){
	jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	Ss[j*na_rows+i] = Ss[j*na_rows+i]*ko[jg];
      }
    }

    /* make Ss2 */

    Overlap_Cluster_NC_Ss2( Ss, Ss2 );

    if (measure_time){
      dtime(&etime);
      time1 += etime - stime;
    }
  }

  /* ***************************************************
             transformation of H with Ss

    in case of SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0
               && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0

    H[i    ][j    ].r = RH[0];
    H[i    ][j    ].i = 0.0;
    H[i+NUM][j+NUM].r = RH[1];
    H[i+NUM][j+NUM].i = 0.0;
    H[i    ][j+NUM].r = RH[2];
    H[i    ][j+NUM].i = RH[3];

    in case of SO_switch==1 or Hub_U_switch==1 or 1<=Constraint_NCS_switch
               or Zeeman_NCS_switch==1 or Zeeman_NCO_switch==1

    H[i    ][j    ].r = RH[0];
    H[i    ][j    ].i = IH[0];
    H[i+NUM][j+NUM].r = RH[1];
    H[i+NUM][j+NUM].i = IH[1];
    H[i    ][j+NUM].r = RH[2];
    H[i    ][j+NUM].i = RH[3] + IH[2];
  *************************************************** */

  if (measure_time) dtime(&stime);

  /* set rHs and iHs */

  Hamiltonian_Cluster_Hs(nh[0],rHs11,MP,0,0);
  Hamiltonian_Cluster_Hs(nh[1],rHs22,MP,0,0);
  Hamiltonian_Cluster_Hs(nh[2],rHs12,MP,0,0);
  Hamiltonian_Cluster_Hs(nh[3],iHs12,MP,0,0);

  Hamiltonian_Cluster_Hs(ImNL[0],iHs11,MP,0,0);
  Hamiltonian_Cluster_Hs(ImNL[1],iHs22,MP,0,0);
  Hamiltonian_Cluster_Hs(ImNL[2],Cs,MP,0,0);

  for (i=0; i<na_rows*na_cols; i++) iHs12[i] += Cs[i];

  /* S^t x rHs11 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&Re_alpha,rHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,rHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs11[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pdgemm_("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs11,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs11,&ONE,&ONE,descH);

  /* S^t x rHs12 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&Re_alpha,rHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,rHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs12[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pdgemm_("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs12,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs12,&ONE,&ONE,descH);

  /* S^t x rHs22 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&Re_alpha,rHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,rHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs22[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pdgemm_("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs22,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs22,&ONE,&ONE,descH);

  /* S^t x iHs11 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&Re_alpha,iHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,iHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs11[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pdgemm_("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs11,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs11,&ONE,&ONE,descH);

  /* S^t x iHs12 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&Re_alpha,iHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,iHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs12[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pdgemm_("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs12,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs12,&ONE,&ONE,descH);

  /* S^t x iHs22 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&Re_alpha,iHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,iHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs22[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pdgemm_("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs22,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs22,&ONE,&ONE,descH);

  if (measure_time){
    dtime(&etime);
    time2 += etime - stime;
  }

  /* ***************************************************
             diagonalize the transformed H
  *************************************************** */

  if (measure_time) dtime(&stime);

  /* initialize ko */
  for (i1=0; i1<=n2; i1++){ ko[i1] = 1.0e+5; }

  Hamiltonian_Cluster_NC_Hs2( rHs11, rHs22, rHs12, iHs11, iHs22, iHs12, Hs2 );

  MPI_Comm_split(mpi_comm_level1,my_pcol2,my_prow2,&mpi_comm_rows);
  MPI_Comm_split(mpi_comm_level1,my_prow2,my_pcol2,&mpi_comm_cols);

  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

  if (ClusterNonCol_UseGs2(n2)){

    /* distributed multi-GPU eigensolver (ELPA, AMD GPU kernels); all n2
       eigenvalues are returned but only the lowest MaxN eigenvectors */
    int gs2_info = openmx_gs2_eigen_complex(n2, MaxN, Hs2, descH2, &ko[1], Cs2, descC2);
    if (gs2_info!=0){
      printf("Cluster_DFT_NonCol: the gpusolver2 Hamiltonian eigensolver failed (info=%d)\n",gs2_info);
      MPI_Abort(mpi_comm_level1,1);
    }
  }
  else if (scf_eigen_lib_flag==1){
    F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2,
                                                   &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
  }
  else if (scf_eigen_lib_flag==GPUSOLVER && Cluster_DFT_NonCol_GpuSwitchNum()<=n2 &&
           ClusterNonCol_cpu_fallback_n2!=n2 && !ClusterNonCol_mgpu_host_dense && na_rows2==n2 && na_cols2==n2){
    ClusterNonCol_GpuSolver_DenseZheevx(Hs2,Cs2,ko,n2,MaxN,"Cluster_DFT_NonCol Hamiltonian");
  }
  else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){

#ifndef kcomp

    int mpiworld;
    mpiworld = MPI_Comm_c2f(mpi_comm_level1);

    F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
      ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &na_cols2,
        &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );

#endif
  }

  MPI_Comm_free(&mpi_comm_rows);
  MPI_Comm_free(&mpi_comm_cols);

  if (2<=level_stdout){
    for (i1=1; i1<=MaxN; i1++){
      printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1,ko[i1]);
    }
  }

  if (measure_time){
    dtime(&etime);
    time3 += etime - stime;
  }

  /* ***************************************************
      Transformation to the original eigenvectors.
      JRCAT NOTE 244P  C = U * lambda^{-1/2} * D
  *************************************************** */

  if (measure_time) dtime(&stime);

  for(k=0; k<na_rows2*na_cols2; k++){
    Hs2[k].r = 0.0;
    Hs2[k].i = 0.0;
  }

  Cblacs_barrier(ictxt1_2,"A");
  if (ClusterNonCol_UseGs2(n2))
    openmx_gs2_pzgemm_("T","T", &n2,&n2,&n2,(const double*)&alpha,(const double*)Cs2,&ONE,&ONE,
                           descC2,(const double*)Ss2,&ONE,&ONE,descS2,(const double*)&beta,
                           (double*)Hs2,&ONE,&ONE,descH2);
  else
    F77_NAME(pzgemm,PZGEMM)("T","T", &n2,&n2,&n2,&alpha,Cs2,&ONE,&ONE,
			           descC2,Ss2,&ONE,&ONE,descS2,&beta,Hs2,
			           &ONE,&ONE,descH2);

  /* MPI communications of Hs2 */

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    k = 0;
    for(i=0; i<na_rows2; i++){
      ig = np_rows2*nblk2*((i)/nblk2) + (i)%nblk2 + ((np_rows2+my_prow2)%np_rows2)*nblk2 + 1;

      if (is2[IDS]<=ig && ig <=ie2[IDS]){

        for (j=0; j<na_cols2; j++){
          jg = np_cols2*nblk2*((j)/nblk2) + (j)%nblk2 + ((np_cols2+my_pcol2)%np_cols2)*nblk2 + 1;

          index_Snd_i[k] = ig;
          index_Snd_j[k] = jg;
          EVec_Snd[2*k  ] = Hs2[j*na_rows2+i].r;
          EVec_Snd[2*k+1] = Hs2[j*na_rows2+i].i;
          k++;
	}
      }
    }

    if (ID!=0){

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, mpi_comm_level1, &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, mpi_comm_level1, &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, mpi_comm_level1, &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, mpi_comm_level1, &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(EVec_Snd, Num_Snd_EV[IDS]*2, MPI_DOUBLE, IDS, 999, mpi_comm_level1, &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR]*2, MPI_DOUBLE, IDR, 999, mpi_comm_level1, &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }
    }
    else{
      for(k=0; k<Num_Snd_EV[IDS]; k++){
        index_Rcv_i[k] = index_Snd_i[k];
        index_Rcv_j[k] = index_Snd_j[k];
        EVec_Rcv[2*k  ] = EVec_Snd[2*k  ];
        EVec_Rcv[2*k+1] = EVec_Snd[2*k+1];
      }
    }

    for(k=0; k<Num_Rcv_EV[IDR]; k++){
      ig = index_Rcv_i[k];
      jg = index_Rcv_j[k];
      m = (ig-is2[myid])*n2 + jg - 1;
      EVec1[m].r = EVec_Rcv[2*k  ];
      EVec1[m].i = EVec_Rcv[2*k+1];
    }
  }

  if (measure_time){
    dtime(&etime);
    time4 += etime - stime;
  }

  }

  if ( strcasecmp(mode,"fp64")==0 ){

    /* the FP64 re-solve of an SCF that ended on a refined step: its vectors
       and eigenvalues replace the refined ones for the post-SCF energy
       density matrix; the density matrix of the step stands */

    if (use_gpusolver_direct_cluster_dm){
      ClusterNonCol_StashGpuSolverDenseEVec(myid,n2,&gpusolver_dense_evec,&gpusolver_dense_evec_ld);
      if (myid==Host_ID) ClusterNonCol_DeviceSolveEndStep();
    }

    free(Num_Snd_EV);
    free(Num_Rcv_EV);
    free(index_Snd_i);
    free(index_Snd_j);
    free(EVec_Snd);
    free(index_Rcv_i);
    free(index_Rcv_j);
    free(EVec_Rcv);

    MPI_Barrier(mpi_comm_level1);
    dtime(&TEtime);
    openmx_mgpu_scope_leave(mgpu_scope_previous);
    return TEtime - TStime;
  }

  if ( strcasecmp(mode,"scf")==0 || strcasecmp(mode,"ParDM")==0 ){

    /****************************************************
                  find chemical potential
    ****************************************************/

    double Beta_trial1;

    if (measure_time) dtime(&stime);

    /* first, find ChemP at 1200 K */

    Beta_trial1 = 1.0/kB/(1200.0/eV2Hartree);

    po = 0;
    loopN = 0;

    ChemP_MAX = 15.0;
    ChemP_MIN =-15.0;

    do {
      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
      Num_State = 0.0;

      for (i1=1; i1<=MaxN; i1++){
	x = (ko[i1] - ChemP)*Beta_trial1;
	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = FermiFunc_NC(x,i1);
	Num_State = Num_State + FermiF;
	if (0.5<FermiF) Cluster_HOMO[0] = i1;
      }

      Dnum = (TZ - Num_State) - system_charge;
      if (0.0<=Dnum) ChemP_MIN = ChemP;
      else           ChemP_MAX = ChemP;
      if (fabs(Dnum)<1.0e-12) po = 1;
      loopN++;
    }
    while (po==0 && loopN<1000);

    /* second, find ChemP at the temperatue, starting from the previously found ChemP. */

    po = 0;
    loopN = 0;

    ChemP_MAX = 15.0;
    ChemP_MIN =-15.0;

    do {

      if (loopN!=0){
	ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
      }

      Num_State = 0.0;

      for (i1=1; i1<=MaxN; i1++){
	x = (ko[i1] - ChemP)*Beta;
	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = FermiFunc_NC(x,i1);
	Num_State = Num_State + FermiF;
	if (0.5<FermiF) Cluster_HOMO[0] = i1;
      }

      Dnum = (TZ - Num_State) - system_charge;
      if (0.0<=Dnum) ChemP_MIN = ChemP;
      else           ChemP_MAX = ChemP;
      if (fabs(Dnum)<1.0e-12) po = 1;
      loopN++;
    }
    while (po==0 && loopN<1000);

    if (2<=level_stdout){
      printf("  ChemP=%15.12f\n",ChemP);
    }

    if (measure_time){
      dtime(&etime);
      time5 += etime - stime;
    }

    /****************************************************
            Energies by summing up eigenvalues
    ****************************************************/

    Eele0[0] = 0.0;
    Eele0[1] = 0.0;

    for (i1=1; i1<=MaxN; i1++){

      x = (ko[i1] - ChemP)*Beta;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = FermiFunc_NC(x,i1);
      Eele0[0] = Eele0[0] + ko[i1]*FermiF;
    }

    /****************************************************
      LCAO coefficients are stored for calculating
               values of MOs on grids
    ****************************************************/

    if ( (Cluster_HOMO[0]-num_HOMOs+1)<1 )  num_HOMOs = Cluster_HOMO[0];
    if ( (Cluster_HOMO[0]+num_LUMOs)>MaxN ) num_LUMOs = MaxN - Cluster_HOMO[0];

    if (myid==Host_ID && 2<=level_stdout){
      printf("  HOMO = %2d\n",Cluster_HOMO[0]);
    }

    if (MO_fileout==1){

      /* allocation of arrays */
      double *array0;
      int *is3,*ie3;

      array0 = (double*)malloc(sizeof(double)*(n2+2)*2);
      is3 = (int*)malloc(sizeof(int)*numprocs);
      ie3 = (int*)malloc(sizeof(int)*numprocs);

      /* set is3 and ie3 */

      if ( numprocs<=MaxN ){

	av_num = (double)MaxN/(double)numprocs;
	for (ID=0; ID<numprocs; ID++){
	  is3[ID] = (int)(av_num*(double)ID) + 1;
	  ie3[ID] = (int)(av_num*(double)(ID+1));
	}

	is3[0] = 1;
	ie3[numprocs-1] = MaxN;
      }
      else{
	for (ID=0; ID<MaxN; ID++){
	  is3[ID] = ID + 1;
	  ie3[ID] = ID + 1;
	}
	for (ID=MaxN; ID<numprocs; ID++){
	  is3[ID] =  1;
	  ie3[ID] =  0;
	}
      }

      /* HOMOs */

      for (j=0; j<num_HOMOs; j++){

	j1 = Cluster_HOMO[0] - j;

        /* store eigenvalue */
        HOMOs_Coef[0][0][j][0][0].r = ko[j1];
        HOMOs_Coef[0][1][j][0][0].r = ko[j1];

	/* store EVec1 */
	if (numprocs==1){
	  for (k=0; k<n2; k++){
	    m = (j1-1)*n2 + k;
	    array0[2*k  ] = EVec1[m].r;
	    array0[2*k+1] = EVec1[m].i;
	  }
	}
	else{

	  po = 0;
	  for (ID=0; ID<numprocs; ID++){
	    if (is3[ID]<=j1 && j1 <=ie3[ID]){
	      po = 1;
	      ID0 = ID;
	      break;
	    }
	  }

	  if (myid==ID0){
	    for (k=0; k<n2; k++){
	      m = (j1-is3[ID0])*n2 + k;
	      array0[2*k  ] = EVec1[m].r;
	      array0[2*k+1] = EVec1[m].i;
	    }
	  }

	  /* MPI communications */
	  MPI_Bcast(array0, n2*2, MPI_DOUBLE, ID0, mpi_comm_level1);
	}

	/* store eigenvector */
	for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){

	    HOMOs_Coef[0][0][j][GA_AN][i].r = array0[ (Anum+i-1)*2      ];
	    HOMOs_Coef[0][0][j][GA_AN][i].i = array0[ (Anum+i-1)*2+1    ];
	    HOMOs_Coef[0][1][j][GA_AN][i].r = array0[ (Anum+i-1)*2  +n2 ];
	    HOMOs_Coef[0][1][j][GA_AN][i].i = array0[ (Anum+i-1)*2+1+n2 ];
	  }
	}
      }

      /* LUMOs */

      for (j=0; j<num_LUMOs; j++){

	j1 = Cluster_HOMO[0] + 1 + j;

        /* store eigenvalue */
        LUMOs_Coef[0][0][j][0][0].r = ko[j1];
        LUMOs_Coef[0][1][j][0][0].r = ko[j1];

	/* store EVec1 */
	if (numprocs==1){
          for (k=0; k<n2; k++){
	    m = (j1-1)*n2 + k;
	    array0[2*k  ] = EVec1[m].r;
	    array0[2*k+1] = EVec1[m].i;
	  }
        }
	else{

	  po = 0;
	  for (ID=0; ID<numprocs; ID++){
	    if (is3[ID]<=j1 && j1 <=ie3[ID]){
	      po = 1;
	      ID0 = ID;
	      break;
	    }
	  }

	  if (myid==ID0){
	    for (k=0; k<n2; k++){
	      m = (j1-is3[ID0])*n2 + k;
	      array0[2*k  ] = EVec1[m].r;
	      array0[2*k+1] = EVec1[m].i;
	    }
	  }

	  /* MPI communications */
	  MPI_Bcast(array0, n2*2, MPI_DOUBLE, ID0, mpi_comm_level1);
	}

	/* store eigenvector */
	for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    LUMOs_Coef[0][0][j][GA_AN][i].r = array0[ (Anum+i-1)*2      ];
	    LUMOs_Coef[0][0][j][GA_AN][i].i = array0[ (Anum+i-1)*2+1    ];
	    LUMOs_Coef[0][1][j][GA_AN][i].r = array0[ (Anum+i-1)*2  +n2 ];
	    LUMOs_Coef[0][1][j][GA_AN][i].i = array0[ (Anum+i-1)*2+1+n2 ];
	  }
	}
      }

      free(array0);
      free(is3);
      free(ie3);

    } /* end of if (MO_fileout==1) */

    /****************************************************
        density matrix and energy density matrix

        CDM[0]  Re alpha alpha density matrix
        CDM[1]  Re beta  beta  density matrix
        CDM[2]  Re alpha beta  density matrix
        CDM[3]  Im alpha beta  density matrix
        iDM[0][0]  Im alpha alpha density matrix
        iDM[0][1]  Im beta  beta  density matrix

        EDM[0]  Re alpha alpha energy density matrix
        EDM[1]  Re beta  beta  energy density matrix
        EDM[2]  Re alpha beta  energy density matrix
        EDM[3]  Im alpha beta  energy density matrix
    ****************************************************/

    if (measure_time) dtime(&stime);

    if (use_gpusolver_direct_cluster_dm){

      if (myid==Host_ID && gpusolver_dense_evec_ld<=0){
        ClusterNonCol_AbortWithMessage("GPU solver device eigenvectors are not available in Cluster_DFT_NonCol.c.");
      }

      {
        double dm_time = ClusterNonCol_CalcDMRootDense_OpenMP(myid,size_H1,MP,n,n2,MaxN,CDM,iDM[0],EDM,ko,
                                                               gpusolver_dense_evec,gpusolver_dense_evec_ld,
                                                               (Cnt_switch==1));

        time6 += dm_time;

        /* Keep the latest direct gpuSOLVER eigenvectors for the post-SCF EDM/force
           rebuild.  Scatter to EVec1 only once after SCF convergence to avoid
           paying this MPI/data-transfer cost every SCF step. */
        ClusterNonCol_StashGpuSolverDenseEVec(myid,n2,&gpusolver_dense_evec,&gpusolver_dense_evec_ld);
        if (myid==Host_ID) ClusterNonCol_DeviceSolveEndStep();
        ClusterNonCol_ProfileReport(myid,n2,MaxN,dm_time);
      }
    }
    else {
      /* DM */

      time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 1, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                        CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

      time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 2, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                        CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

      time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 3, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                        CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

      time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 4, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                        CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

      time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 5, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                        CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

      time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 6, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                        CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

      /* EDM */

      if (Cnt_switch==1){

        time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 7, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                          CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

        time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 8, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                          CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

        time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 9, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                          CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

        time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 10, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                          CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);
      }

      /* calculation of ParDM */

      if ( strcasecmp(mode,"ParDM")==0 ){
        for (i=1; i<=6; i++){
          time6 += Calc_ParDM_Cluster_non_collinear_ScaLAPACK( i, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                               ko, DM1, Work1, EVec1);
        }
      }
    }

    /****************************************************
                      Bond Energies
    ****************************************************/

    My_Eele1[0] = 0.0;
    My_Eele1[1] = 0.0;

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (j=0; j<=FNAN[GA_AN]; j++){
	wanB = WhatSpecies[natn[GA_AN][j]];
	tnoB = Spe_Total_CNO[wanB];

	/* non-spin-orbit coupling and non-LDA+U */
	if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0
	    && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){
	  for (k=0; k<tnoA; k++){
	    for (l=0; l<tnoB; l++){
	      My_Eele1[0] +=
		+ CDM[0][MA_AN][j][k][l]*nh[0][MA_AN][j][k][l]
		+ CDM[1][MA_AN][j][k][l]*nh[1][MA_AN][j][k][l]
		+ 2.0*CDM[2][MA_AN][j][k][l]*nh[2][MA_AN][j][k][l]
		- 2.0*CDM[3][MA_AN][j][k][l]*nh[3][MA_AN][j][k][l];
	    }
	  }
	}

	/* spin-orbit coupling or LDA+U */
	else {
	  for (k=0; k<tnoA; k++){
	    for (l=0; l<tnoB; l++){
	      My_Eele1[0] +=
		+ CDM[0][MA_AN][j][k][l]*nh[0][MA_AN][j][k][l]
		- iDM[0][0][MA_AN][j][k][l]*ImNL[0][MA_AN][j][k][l]
		+ CDM[1][MA_AN][j][k][l]*nh[1][MA_AN][j][k][l]
		- iDM[0][1][MA_AN][j][k][l]*ImNL[1][MA_AN][j][k][l]
		+ 2.0*CDM[2][MA_AN][j][k][l]*nh[2][MA_AN][j][k][l]
		- 2.0*CDM[3][MA_AN][j][k][l]*(nh[3][MA_AN][j][k][l]
					      +ImNL[2][MA_AN][j][k][l]);
	    }
	  }
	}

      }
    }

    /* MPI, My_Eele1 */
    MPI_Allreduce(&My_Eele1[0], &Eele1[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&My_Eele1[1], &Eele1[1], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    if (2<=level_stdout && myid==Host_ID){
      printf("Eele0=%15.12f Eele1=%15.12f\n",Eele0[0],Eele1[0]);
    }

    if (measure_time){
      dtime(&etime);
      time6 += etime - stime;
    }

    /****************************************************
                           Output
    ****************************************************/

    if (measure_time) dtime(&stime);

    if (myid==Host_ID){

      sprintf(file_EV,"%s%s.EV",filepath,filename);

      if ((fp_EV = fopen(file_EV,"w")) != NULL){

	setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"      Eigenvalues (Hartree) for non-collinear KS-eq.       \n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n\n");

	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",ChemP);
	fprintf(fp_EV,"   Number of States             = %18.14f\n",Num_State);
	fprintf(fp_EV,"   HOMO = %2d\n",Cluster_HOMO[0]);

	fprintf(fp_EV,"   Eigenvalues\n");
	fprintf(fp_EV,"\n");
	for (i1=1; i1<=MaxN; i1++){
	  fprintf(fp_EV,"      %5d %18.14f\n",i1,ko[i1]);
	}

      }
      /* fclose of fp_EV */
      fclose(fp_EV);
    }

    if (2<=level_fileout){

      if (myid==Host_ID){

	sprintf(file_EV,"%s%s.EV",filepath,filename);

	if ((fp_EV = fopen(file_EV,"a")) != NULL){
	  setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */
	}
	else{
	  printf("Failure of saving the EV file.\n");
	}
      }

      if (myid==Host_ID){

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"  Eigenvalues (Hartree) and Eigenvectors for SCF KS-eq.  \n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");

	fprintf(fp_EV,"\n\n");
	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",ChemP);
	fprintf(fp_EV,"   HOMO = %2d\n",Cluster_HOMO[0]);

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"   Real (Re) and imaginary (Im) parts of LCAO coefficients\n\n");

      }

      /* allocation of arrays */
      double *array0;
      int *is3,*ie3;

      array0 = (double*)malloc(sizeof(double)*(n2+2)*4);
      is3 = (int*)malloc(sizeof(int)*numprocs);
      ie3 = (int*)malloc(sizeof(int)*numprocs);

      /* set is3 and ie3 */

      if ( numprocs<=MaxN ){

	av_num = (double)MaxN/(double)numprocs;
	for (ID=0; ID<numprocs; ID++){
	  is3[ID] = (int)(av_num*(double)ID) + 1;
	  ie3[ID] = (int)(av_num*(double)(ID+1));
	}

	is3[0] = 1;
	ie3[numprocs-1] = MaxN;
      }

      else{
	for (ID=0; ID<MaxN; ID++){
	  is3[ID] = ID + 1;
	  ie3[ID] = ID + 1;
	}
	for (ID=MaxN; ID<numprocs; ID++){
	  is3[ID] =  1;
	  ie3[ID] = -2;
	}
      }

      num0 = 2;
      num1 = MaxN/num0 + 1*(MaxN%num0!=0);

      for (i=1; i<=num1; i++){

	if (myid==Host_ID){

	  /* header */

	  fprintf(fp_EV,"\n");

	  for (i1=-2; i1<=0; i1++){

	    fprintf(fp_EV,"                     ");

	    for (j=1; j<=num0; j++){
	      j1 = num0*(i-1) + j;

	      if (j1<=MaxN){
		if (i1==-2){
		  fprintf(fp_EV," %4d",j1);
		  fprintf(fp_EV,"                                   ");
		}
		else if (i1==-1){
		  fprintf(fp_EV,"   %8.5f",ko[j1]);
		  fprintf(fp_EV,"                             ");
		}

		else if (i1==0){
		  fprintf(fp_EV,"     Re(U)");
		  fprintf(fp_EV,"     Im(U)");
		  fprintf(fp_EV,"     Re(D)");
		  fprintf(fp_EV,"     Im(D)");
		}

	      }
	    }
	    fprintf(fp_EV,"\n");
	    if (i1==-1)  fprintf(fp_EV,"\n");
	    if (i1==0)   fprintf(fp_EV,"\n");
	  }
	}

	/* MPI communication of EVec1 */

	if (numprocs==1){

	  for (j=1; j<=num0; j++){

	    j1 = num0*(i-1) + j;

	    if (j1<=MaxN){
	      for (k=0; k<n2; k++){
		m = (j1-1)*n2 + k;
		array0[(j-1)*n2*2+2*k  ] = EVec1[m].r;
		array0[(j-1)*n2*2+2*k+1] = EVec1[m].i;
	      }
	    }
	  }
	}
	else{

	  for (j=1; j<=num0; j++){

	    j1 = num0*(i-1) + j;

	    po = 0;
	    for (ID=0; ID<numprocs; ID++){
	      if (is3[ID]<=j1 && j1 <=ie3[ID]){
		po = 1;
		ID0 = ID;
		break;
	      }
	    }

	    if (j1<=MaxN){

	      if (myid==ID0){
		for (k=0; k<n2; k++){
		  m = (j1-is3[ID0])*n2 + k;
		  array0[(j-1)*n2*2+2*k  ] = EVec1[m].r;
		  array0[(j-1)*n2*2+2*k+1] = EVec1[m].i;
		}
	      }

	      /* MPI communications */
	      MPI_Bcast(&array0[(j-1)*n2*2], n2*2, MPI_DOUBLE, ID0, mpi_comm_level1);
	    }

	  } /* j */
	} /* else */

        /* LCAO coefficients */

	Name_Angular[0][0] = "s          ";
	Name_Angular[1][0] = "px         ";
	Name_Angular[1][1] = "py         ";
	Name_Angular[1][2] = "pz         ";
	Name_Angular[2][0] = "d3z^2-r^2  ";
	Name_Angular[2][1] = "dx^2-y^2   ";
	Name_Angular[2][2] = "dxy        ";
	Name_Angular[2][3] = "dxz        ";
	Name_Angular[2][4] = "dyz        ";
	Name_Angular[3][0] = "f5z^2-3r^2 ";
	Name_Angular[3][1] = "f5xz^2-xr^2";
	Name_Angular[3][2] = "f5yz^2-yr^2";
	Name_Angular[3][3] = "fzx^2-zy^2 ";
	Name_Angular[3][4] = "fxyz       ";
	Name_Angular[3][5] = "fx^3-3*xy^2";
	Name_Angular[3][6] = "f3yx^2-y^3 ";
	Name_Angular[4][0] = "g1         ";
	Name_Angular[4][1] = "g2         ";
	Name_Angular[4][2] = "g3         ";
	Name_Angular[4][3] = "g4         ";
	Name_Angular[4][4] = "g5         ";
	Name_Angular[4][5] = "g6         ";
	Name_Angular[4][6] = "g7         ";
	Name_Angular[4][7] = "g8         ";
	Name_Angular[4][8] = "g9         ";

	Name_Multiple[0] = "0";
	Name_Multiple[1] = "1";
	Name_Multiple[2] = "2";
	Name_Multiple[3] = "3";
	Name_Multiple[4] = "4";
	Name_Multiple[5] = "5";

	if (myid==Host_ID){

	  i1 = 1;

	  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	    wan1 = WhatSpecies[Gc_AN];

	    for (l=0; l<=Supported_MaxL; l++){
	      for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
		for (m=0; m<(2*l+1); m++){

		  if (l==0 && mul==0 && m==0)
		    fprintf(fp_EV,"%4d %3s %s %s",
			    Gc_AN,SpeName[wan1],Name_Multiple[mul],Name_Angular[l][m]);
		  else
		    fprintf(fp_EV,"         %s %s",
			    Name_Multiple[mul],Name_Angular[l][m]);

		  for (j=1; j<=num0; j++){

		    j1 = num0*(i-1) + j;

		    if (0<i1 && j1<=MaxN){
		      fprintf(fp_EV,"  %8.5f",array0[ (j-1)*n2*2+2*(i1-1)      ]);
		      fprintf(fp_EV,"  %8.5f",array0[ (j-1)*n2*2+2*(i1-1)+1    ]);
		      fprintf(fp_EV,"  %8.5f",array0[ (j-1)*n2*2+2*(i1-1)  +n2 ]);
		      fprintf(fp_EV,"  %8.5f",array0[ (j-1)*n2*2+2*(i1-1)+1+n2 ]);
		    }
		  }

		  fprintf(fp_EV,"\n");
		  if (i1==-1)  fprintf(fp_EV,"\n");
		  if (i1==0)   fprintf(fp_EV,"\n");

		  i1++;

		}
	      }
	    }
	  }
	}
      }

      /* freeing of arrays */

      free(array0);
      free(is3);
      free(ie3);

      /* fclose of fp_EV */

      if (myid==Host_ID){
	fclose(fp_EV);
      }

    } /* end of if (2<=level_fileout) */

    if (measure_time){
      dtime(&etime);
      time7 += etime - stime;
    }

  } /* if ( strcasecmp(mode,"scf")==0 ) */

  else if ( strcasecmp(mode,"dos")==0 ){
    Save_DOS_NonCol(n,n2,MaxN,MP,CntOLP,EVec1,ko);
  }
  else if ( strcasecmp(mode,"lcaoout")==0 ){
    Save_LCAO_NonCol(n,n2,MaxN,MP,CntOLP,EVec1,ko);
  }
  else if ( strcasecmp(mode,"xanes")==0 ){
    Calc_XANES_NonCol(n,n2,MaxN,MP,is2,ie2,EVec1,ko);
  }

  if (measure_time){
    printf("Cluster_DFT myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f\n",
            myid,time1,time2,time3,time4,time5,time6,time7);fflush(stdout);
  }

  /****************************************************
                          Free
  ****************************************************/

  free(Num_Snd_EV);
  free(Num_Rcv_EV);

  free(index_Snd_i);
  free(index_Snd_j);
  free(EVec_Snd);
  free(index_Rcv_i);
  free(index_Rcv_j);
  free(EVec_Rcv);

  if (use_gpusolver_direct_cluster_dm && myid==Host_ID) ClusterNonCol_DeviceSolveEndStep();

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  openmx_mgpu_scope_leave(mgpu_scope_previous);
  return time0;
}





double Calc_DM_Cluster_non_collinear_ScaLAPACK(
    int calc_flag,
    int myid,
    int numprocs,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    double *****CDM,
    double *****iDM0,
    double *****EDM,
    double *ko,
    double *DM1,
    double *Work1,
    dcomplex *EVec1 )
{
  int i,j,k,po,p,GA_AN,MA_AN,wanA,tnoA,Anum;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j1,ID;
  double max_x=60.0,dum;
  double FermiF,FermiF2,x,x2,diffF;
  double FermiEps = 1.0e-13;
  double stime,etime,time,lumos;
  MPI_Status stat;
  MPI_Request request;

  dtime(&stime);

  /* initialize DM1 */

  for (i=0; i<size_H1; i++){
    DM1[i] = 0.0;
  }

  /* calculation of DM1 */

  for (k=is2[myid]; k<=ie2[myid]; k++){

    x = (ko[k] - ChemP)*Beta;
    if (x<=-max_x) x = -max_x;
    if (max_x<=x)  x = max_x;
    FermiF = FermiFunc_NC(x,k);

    if (11<=calc_flag && cal_partial_charge){
      x2 = (ko[k] - (ChemP+ene_win_partial_charge))*Beta;
      if (x2<=-max_x) x2 = -max_x;
      if (max_x<=x2)  x2 = max_x;
      FermiF2 = FermiFunc_NC(x2,k);
      diffF = fabs(FermiF-FermiF2);
    }

    if ( FermiEps<FermiF ) {

      p = 0;
      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

              i1 = (k-is2[myid])*n2 + Anum + i - 1;
              j1 = (k-is2[myid])*n2 + Bnum + j - 1;

              switch (calc_flag){

		/* Re11 */
              case 1:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1].r + EVec1[i1].i*EVec1[j1].i);
		break;

		/* Re22 */
              case 2:
		DM1[p] += FermiF*(EVec1[i1+n].r*EVec1[j1+n].r + EVec1[i1+n].i*EVec1[j1+n].i);
		break;

		/* Re12 */
              case 3:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1+n].r + EVec1[i1].i*EVec1[j1+n].i);
		break;

		/* Im12 */
              case 4:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1+n].i - EVec1[i1].i*EVec1[j1+n].r);
		break;

		/* Im11 */
              case 5:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1].i - EVec1[i1].i*EVec1[j1].r);
		break;

		/* Im22 */
              case 6:
		DM1[p] += FermiF*(EVec1[i1+n].r*EVec1[j1+n].i - EVec1[i1+n].i*EVec1[j1+n].r);
		break;

		/* ReEDM11 */
              case 7:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1].r + EVec1[i1].i*EVec1[j1].i)*ko[k];
		break;

		/* ReEDM22 */
              case 8:
		DM1[p] += FermiF*(EVec1[i1+n].r*EVec1[j1+n].r + EVec1[i1+n].i*EVec1[j1+n].i)*ko[k];
		break;

		/* ReEDM12 */
              case 9:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1+n].r + EVec1[i1].i*EVec1[j1+n].i)*ko[k];
		break;

		/* ImEDM12 */
              case 10:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1+n].i - EVec1[i1].i*EVec1[j1+n].r)*ko[k];
		break;

		/* Partial_DM11 */
              case 11:
		DM1[p] += diffF*(EVec1[i1].r*EVec1[j1].r + EVec1[i1].i*EVec1[j1].i);
		break;

		/* Partial_DM22 */
              case 12:
		DM1[p] += diffF*(EVec1[i1+n].r*EVec1[j1+n].r + EVec1[i1+n].i*EVec1[j1+n].i);
		break;

	      }

	      /* increment of p */
	      p++;

	    }
	  }
	}
      } /* GA_AN */
    }
  }

  /* MPI_Allreduce */

  MPI_Allreduce(DM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  for (i=0; i<size_H1; i++) DM1[i] = Work1[i];

  /* store DM1 to a proper place */

  p = 0;
  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

    MA_AN = F_G2M[GA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];
    ID = G2ID[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      Bnum = MP[GB_AN];

      if (myid==ID){

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

            switch (calc_flag){

              /* Re11 */
              case 1:
		CDM[0][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Re22 */
              case 2:
		CDM[1][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Re12 */
              case 3:
		CDM[2][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Im12 */
              case 4:
		CDM[3][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Im11 */
              case 5:
		iDM0[0][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Im22 */
              case 6:
		iDM0[1][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* ReEDM11 */
              case 7:
		EDM[0][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* ReEDM22 */
              case 8:
		EDM[1][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* ReEDM12 */
              case 9:
		EDM[2][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* ImEDM12 */
              case 10:
		EDM[3][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Partial_DM11 */
              case 11:
		Partial_DM[0][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Partial_DM22 */
              case 12:
		Partial_DM[1][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;
	    }

	    /* increment of p */
	    p++;
	  }
	}
      }
      else{
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    /* increment of p */
	    p++;
	  }
	}
      }

    } /* LB_AN */
  } /* GA_AN */

  dtime(&etime);
  return (etime-stime);
}





double Calc_ParDM_Cluster_non_collinear_ScaLAPACK(
    int calc_flag,
    int myid,
    int numprocs,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    double *ko,
    double *DM1,
    double *Work1,
    dcomplex *EVec1 )
{
  int i,j,k,po,p,GA_AN,MA_AN,wanA,tnoA,Anum;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j1,ID;
  double windowF_Eps = 1.0e-13;
  double e,e0,e1,b0,b1,windowF;
  double stime,etime,time,lumos;
  MPI_Status stat;
  MPI_Request request;

  dtime(&stime);

  /* initialize DM1 */

  for (i=0; i<size_H1; i++){
    DM1[i] = 0.0;
  }

  /* calculation of DM1 */

  for (k=is2[myid]; k<=ie2[myid]; k++){

    if (CWF_WindowF_switch==0){ // Poly

      double in0,in1,out0,out1;
      out0 = CWF_disentangling_Erange[0];
      in0 = CWF_disentangling_Erange[1];
      in1 = CWF_disentangling_Erange[2];
      out1 = CWF_disentangling_Erange[3];
      e = ko[k];
      windowF = weight_CWF2(e,ChemP,in0,in1,out0,out1);
    }

    else if (CWF_WindowF_switch==1){ // Fermi
      e = ko[k];
      b0 = 1.0/CWF_disentangling_smearing_kBT0;
      b1 = 1.0/CWF_disentangling_smearing_kBT1;
      e0 = CWF_disentangling_Erange[0] + ChemP;
      e1 = CWF_disentangling_Erange[1] + ChemP;
      windowF = (1.0/(exp(b0*(e0-e))+1.0))*(1.0/(exp(b1*(e-e1))+1.0)) + CWF_disentangling_smearing_bound;
    }

    if ( windowF_Eps<windowF ) {

      p = 0;
      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

              i1 = (k-is2[myid])*n2 + Anum + i - 1;
              j1 = (k-is2[myid])*n2 + Bnum + j - 1;

              switch (calc_flag){

		/* Re11 */
              case 1:
		DM1[p] += windowF*(EVec1[i1].r*EVec1[j1].r + EVec1[i1].i*EVec1[j1].i);
		break;

		/* Re22 */
              case 2:
		DM1[p] += windowF*(EVec1[i1+n].r*EVec1[j1+n].r + EVec1[i1+n].i*EVec1[j1+n].i);
		break;

		/* Re12 */
              case 3:
		DM1[p] += windowF*(EVec1[i1].r*EVec1[j1+n].r + EVec1[i1].i*EVec1[j1+n].i);
		break;

		/* Im12 */
              case 4:
		DM1[p] += windowF*(EVec1[i1].r*EVec1[j1+n].i - EVec1[i1].i*EVec1[j1+n].r);
		break;

		/* Im11 */
              case 5:
		DM1[p] += windowF*(EVec1[i1].r*EVec1[j1].i - EVec1[i1].i*EVec1[j1].r);
		break;

		/* Im22 */
              case 6:
		DM1[p] += windowF*(EVec1[i1+n].r*EVec1[j1+n].i - EVec1[i1+n].i*EVec1[j1+n].r);
		break;

	      }

	      /* increment of p */
	      p++;

	    }
	  }
	}
      } /* GA_AN */
    }
  }

  /* MPI_Allreduce */

  MPI_Allreduce(DM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  for (i=0; i<size_H1; i++) DM1[i] = Work1[i];

  /* store DM1 to a proper place */

  p = 0;
  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

    MA_AN = F_G2M[GA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];
    ID = G2ID[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      Bnum = MP[GB_AN];

      if (myid==ID){

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

            switch (calc_flag){

              /* Re11 */
              case 1:
		ParDM[0][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Re22 */
              case 2:
		ParDM[1][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Re12 */
              case 3:
		ParDM[2][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Im12 */
              case 4:
		ParDM[3][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Im11 */
              case 5:
		iParDM[0][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Im22 */
              case 6:
		iParDM[1][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;
	    }

	    /* increment of p */
	    p++;
	  }
	}
      }
      else{
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    /* increment of p */
	    p++;
	  }
	}
      }

    } /* LB_AN */
  } /* GA_AN */

  dtime(&etime);
  return (etime-stime);
}







void Save_DOS_NonCol(int n, int n2, int MaxN, int *MP, double ****OLP0, dcomplex *EVec1, double *ko)
{
  int spin,ie,i,j,iemin,iemax,MA_AN,GA_AN,k,p,l;
  int Anum,Bnum,tnoA,tnoB,wanA,wanB;
  int LB_AN,GB_AN,MaxL,num,num0,num1,m,po,ID0;
  int numprocs,myid,ID,tag;
  int i_vec[10];
  double d0,d1,d2,d3,dum,av_num;
  double tmp,tmp1,tmp2,tmp3;
  double theta,phi,sit,cot,sip,cop;
  float *array0;
  float *SD,*Re11,*Re22,*Re12,*Im12;
  int *is3,*ie3;
  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */

  array0 = (float*)malloc(sizeof(float)*(n2+2)*2);
  is3 = (int*)malloc(sizeof(int)*numprocs);
  ie3 = (int*)malloc(sizeof(int)*numprocs);
  SD = (float*)malloc(sizeof(float)*2*(atomnum+1)*List_YOUSO[7]);
  Re11 = (float*)malloc(sizeof(float)*(atomnum+1)*List_YOUSO[7]);
  Re22 = (float*)malloc(sizeof(float)*(atomnum+1)*List_YOUSO[7]);
  Re12 = (float*)malloc(sizeof(float)*(atomnum+1)*List_YOUSO[7]);
  Im12 = (float*)malloc(sizeof(float)*(atomnum+1)*List_YOUSO[7]);

  /* open file pointers */

  if (myid==Host_ID){

    strcpy(file_eig,".Dos.val");
    fnjoint(filepath,filename,file_eig);
    if ( (fp_eig=fopen(file_eig,"w"))==NULL ) {
      printf("cannot open a file %s\n",file_eig);
    }

    strcpy(file_ev,".Dos.vec");
    fnjoint(filepath,filename,file_ev);
    if ( (fp_ev=fopen(file_ev,"w"))==NULL ) {
      printf("cannot open a file %s\n",file_ev);
    }
  }

  /* find iemin */

  iemin = n2;

  for (i=1; i<n2; i++) {
    if ( ((ChemP+Dos_Erange[0])<ko[i]) && (i-1)<iemin ) {
      iemin = i - 1;
      break;
    }
  }
  if (iemin<1) iemin = 1;

  /* find iemax */

  iemax = 1;
  for (i=1; i<=2*n; i++) {
    if ( ((ChemP+Dos_Erange[1])<ko[i])) {
      if (iemax<i) iemax = i;
      break;
    }
  }

  if (iemax==1)   iemax = MaxN;
  if (MaxN<iemax) iemax = MaxN;

  /* set is3 and ie3 */

  if ( numprocs<=MaxN ){

    av_num = (double)MaxN/(double)numprocs;
    for (ID=0; ID<numprocs; ID++){
      is3[ID] = (int)(av_num*(double)ID) + 1;
      ie3[ID] = (int)(av_num*(double)(ID+1));
    }

    is3[0] = 1;
    ie3[numprocs-1] = MaxN;
  }

  else{
    for (ID=0; ID<MaxN; ID++){
      is3[ID] = ID + 1;
      ie3[ID] = ID + 1;
    }
    for (ID=MaxN; ID<numprocs; ID++){
      is3[ID] =  1;
      ie3[ID] = -2;
    }
  }

  /****************************************************
                   save *.Dos.vec
  ****************************************************/

  for (p=iemin; p<=iemax; p++){

    /* store EVec1 */

    if (numprocs==1){
      for (k=0; k<n2; k++){
	m = (p-1)*n2 + k;
	array0[2*k  ] = (float)EVec1[m].r;
	array0[2*k+1] = (float)EVec1[m].i;
      }
    }

    else{

      po = 0;
      for (ID=0; ID<numprocs; ID++){
	if (is3[ID]<=p && p <=ie3[ID]){
	  po = 1;
	  ID0 = ID;
	  break;
	}
      }

      if (myid==ID0){
	for (k=0; k<n2; k++){
	  m = (p-is3[ID0])*n2 + k;
	  array0[2*k  ] = (float)EVec1[m].r;
	  array0[2*k+1] = (float)EVec1[m].i;
	}
      }

      /* MPI communications */
      MPI_Bcast(array0, n2*2, MPI_FLOAT, ID0, mpi_comm_level1);
    }

    /* initialize Re11, Re22, Re12, and Im12 */

    for (i=0; i<(atomnum+1)*List_YOUSO[7]; i++){
      Re11[i] = 0.0;
      Re22[i] = 0.0;
      Re12[i] = 0.0;
      Im12[i] = 0.0;
    }

    /* write a header */

    i_vec[0]=i_vec[1]=i_vec[2]=0;
    if (myid==Host_ID) fwrite(i_vec,sizeof(int),3,fp_ev);

    /* loop of MA_AN */

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

      for (i=0; i<tnoA; i++){
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];

	  for (j=0; j<tnoB; j++){

	    /* Re11 */

            Re11[Anum+i] += ( array0[(Anum+i-1)*2  ]*array0[(Bnum+j-1)*2  ]
                             +array0[(Anum+i-1)*2+1]*array0[(Bnum+j-1)*2+1] )*(float)OLP0[MA_AN][LB_AN][i][j];

	    /* Re22 */

            Re22[Anum+i] += ( array0[(Anum+i-1)*2  +n2]*array0[(Bnum+j-1)*2+ +n2]
                             +array0[(Anum+i-1)*2+1+n2]*array0[(Bnum+j-1)*2+1+n2] )*(float)OLP0[MA_AN][LB_AN][i][j];

	    /* Re12 */

            Re12[Anum+i] += ( array0[(Anum+i-1)*2  ]*array0[(Bnum+j-1)*2+ +n2]
                             +array0[(Anum+i-1)*2+1]*array0[(Bnum+j-1)*2+1+n2] )*(float)OLP0[MA_AN][LB_AN][i][j];

	    /* Im12
	       conjugate complex of Im12 due to difference in the definition
	       between density matrix and charge density
	    */

            Im12[Anum+i] +=-( array0[(Anum+i-1)*2  ]*array0[(Bnum+j-1)*2+1+n2]
                             -array0[(Anum+i-1)*2+1]*array0[(Bnum+j-1)*2  +n2] )*(float)OLP0[MA_AN][LB_AN][i][j];


	  } /* j */
	} /* LB_AN */
      } /* i */
    } /* MA_AN */

    /* MPI communication */

    MPI_Reduce(Re11, array0, n+1, MPI_FLOAT, MPI_SUM, Host_ID, mpi_comm_level1);
    for (i=0; i<=n; i++) Re11[i] = array0[i];

    MPI_Reduce(Re22, array0, n+1, MPI_FLOAT, MPI_SUM, Host_ID, mpi_comm_level1);
    for (i=0; i<=n; i++) Re22[i] = array0[i];

    MPI_Reduce(Re12, array0, n+1, MPI_FLOAT, MPI_SUM, Host_ID, mpi_comm_level1);
    for (i=0; i<=n; i++) Re12[i] = array0[i];

    MPI_Reduce(Im12, array0, n+1, MPI_FLOAT, MPI_SUM, Host_ID, mpi_comm_level1);
    for (i=0; i<=n; i++) Im12[i] = array0[i];

    /*  transform to up and down states */

    if (myid==Host_ID){

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (i=0; i<tnoA; i++){

	  d0 = Re11[Anum+i];
	  d1 = Re22[Anum+i];
	  d2 = Re12[Anum+i];
	  d3 = Im12[Anum+i];

	  theta = Angle0_Spin[GA_AN];
	  phi   = Angle1_Spin[GA_AN];

	  sit = sin(theta);
	  cot = cos(theta);
	  sip = sin(phi);
	  cop = cos(phi);

	  tmp1 = 0.5*(d0 + d1);
	  tmp2 = 0.5*cot*(d0 - d1);
	  tmp3 = (d2*cop - d3*sip)*sit;

	  SD[2*(Anum-1)+i     ] = (float)(tmp1 + tmp2 + tmp3); /* up   */
	  SD[2*(Anum-1)+tnoA+i] = (float)(tmp1 - tmp2 - tmp3); /* down */

	}
      }

      /* write *.Dos.vec */

      fwrite(&SD[0],sizeof(float),n2,fp_ev);
    }

  } /* p */

  /****************************************************
                   save *.Dos.val
  ****************************************************/

  if (myid==Host_ID){

    fprintf(fp_eig,"mode        1\n");
    fprintf(fp_eig,"NonCol      1\n");
    fprintf(fp_eig,"N           %d\n",n);
    fprintf(fp_eig,"Nspin       %d\n",1); /* switch to 1 */
    fprintf(fp_eig,"Erange      %lf %lf\n",Dos_Erange[0],Dos_Erange[1]);
    /*  fprintf(fp_eig,"irange      %d %d\n",iemin,iemax); */
    fprintf(fp_eig,"Kgrid       %d %d %d\n",1,1,1);
    fprintf(fp_eig,"atomnum     %d\n",atomnum);
    fprintf(fp_eig,"<WhatSpecies\n");
    for (i=1;i<=atomnum;i++) {
      fprintf(fp_eig,"%d ",WhatSpecies[i]);
    }
    fprintf(fp_eig,"\nWhatSpecies>\n");
    fprintf(fp_eig,"SpeciesNum     %d\n",SpeciesNum);
    fprintf(fp_eig,"<Spe_Total_CNO\n");
    for (i=0;i<SpeciesNum;i++) {
      fprintf(fp_eig,"%d ",Spe_Total_CNO[i]);
    }
    fprintf(fp_eig,"\nSpe_Total_CNO>\n");
    MaxL=Supported_MaxL;
    fprintf(fp_eig,"MaxL           %d\n",Supported_MaxL);
    fprintf(fp_eig,"<Spe_Num_CBasis\n");
    for (i=0;i<SpeciesNum;i++) {
      for (l=0;l<=MaxL;l++) {
	fprintf(fp_eig,"%d ",Spe_Num_CBasis[i][l]);
      }
      fprintf(fp_eig,"\n");
    }
    fprintf(fp_eig,"Spe_Num_CBasis>\n");
    fprintf(fp_eig,"ChemP       %lf\n",ChemP);

    fprintf(fp_eig,"<SpinAngle\n");
    for (i=1; i<=atomnum; i++) {
      fprintf(fp_eig,"%lf %lf\n",Angle0_Spin[i],Angle1_Spin[i]);
    }
    fprintf(fp_eig,"SpinAngle>\n");

    fprintf(fp_eig,"irange      %d %d\n",iemin,iemax);
    fprintf(fp_eig,"<Eigenvalues\n");

    for (spin=0; spin<=1; spin++) {
      fprintf(fp_eig,"%d %d %d ",0,0,0);
      for (ie=iemin; ie<=iemax; ie++) {
	fprintf(fp_eig,"%lf ",ko[ie]);
      }
      fprintf(fp_eig,"\n");
      /* printf("\n"); */
    }
    fprintf(fp_eig,"Eigenvalues>\n");

    printf("write eigenvalues\n");
    printf("write eigenvectors\n");
  }

  /* close file pointers */

  if (myid==Host_ID){
    if (fp_eig) fclose(fp_eig);
    if (fp_ev)  fclose(fp_ev);
  }

  /* freeing of array */

  free(array0);
  free(is3);
  free(ie3);
  free(SD);
  free(Re11);
  free(Re22);
  free(Re12);
  free(Im12);
}




void Save_LCAO_NonCol(int n, int n2, int MaxN, int *MP, double ****OLP0, dcomplex *EVec1, double *ko)
{
  int spin,ie,i,j,iemin,iemax;
  int po,ID0,m,p,k;
  int numprocs,myid,ID,tag;
  int i_vec[10],HOMO[2];
  double x,FermiF,av_num;
  double *array0;
  int *is3,*ie3;
  double max_x=60.0;
  char operate[YOUSO10];
  FILE *fp1;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */

  array0 = (double*)malloc(sizeof(double)*(n2+2)*2);
  is3 = (int*)malloc(sizeof(int)*numprocs);
  ie3 = (int*)malloc(sizeof(int)*numprocs);

  /* open file pointers */

  if (myid==Host_ID){

    sprintf(operate,"%s%s.lcao",filepath,filename);
    fp1 = fopen(operate,"ab");

    if (fp1!=NULL){
      remove(operate);
      fclose(fp1);
      fp1 = fopen(operate, "ab");
    }
  }

  /* find iemin */

  iemin = n2;

  for (i=1; i<n2; i++) {
    if ( ((ChemP+Dos_Erange[0])<ko[i]) && (i-1)<iemin ) {
      iemin = i - 1;
      break;
    }
  }
  if (iemin<1) iemin = 1;

  /* set iemin and iemax */

  iemin = 1;
  iemax = MaxN;

  /****************************************************
                  find Cluster_HOMO
  ****************************************************/

  for (i=1; i<=MaxN; i++){
    x = (ko[i] - ChemP)*Beta;
    if (x<=-max_x) x = -max_x;
    if (max_x<=x)  x = max_x;
    FermiF = 1.0/(1.0 + exp(x));
    if (0.5<FermiF) HOMO[0] = i;
  }
  HOMO[1] = HOMO[0];

  /****************************************************
                  save LCAO coefficients
  ****************************************************/

  /* Save parameters */

  if (myid==Host_ID){
    fwrite(&n, sizeof(int), 1, fp1);
    fwrite(&MaxN, sizeof(int), 1, fp1);
    fwrite(HOMO, sizeof(int), 2, fp1);
    fwrite(&SpinP_switch, sizeof(int), 1, fp1);
    fwrite(&Utot, sizeof(double), 1, fp1);
    fwrite(&Total_SpinSx, sizeof(double),1,fp1); // S. An
    fwrite(&Total_SpinSy, sizeof(double),1,fp1); // S. An
    fwrite(&Total_SpinSz, sizeof(double),1,fp1); // S. An
  }

  /* calculate N */






  /* set is3 and ie3 */

  if ( numprocs<=MaxN ){

    av_num = (double)MaxN/(double)numprocs;
    for (ID=0; ID<numprocs; ID++){
      is3[ID] = (int)(av_num*(double)ID) + 1;
      ie3[ID] = (int)(av_num*(double)(ID+1));
    }

    is3[0] = 1;
    ie3[numprocs-1] = MaxN;
  }

  else{
    for (ID=0; ID<MaxN; ID++){
      is3[ID] = ID + 1;
      ie3[ID] = ID + 1;
    }
    for (ID=MaxN; ID<numprocs; ID++){
      is3[ID] =  1;
      ie3[ID] = -2;
    }
  }

  /* loop for p */

  for (p=iemin; p<=iemax; p++){

    /* store EVec1 */

    if (numprocs==1){
      for (k=0; k<n2; k++){
	m = (p-1)*n2 + k;
	array0[2*k  ] = EVec1[m].r;
	array0[2*k+1] = EVec1[m].i;
      }
    }

    else{

      po = 0;
      for (ID=0; ID<numprocs; ID++){
	if (is3[ID]<=p && p <=ie3[ID]){
	  po = 1;
	  ID0 = ID;
	  break;
	}
      }

      if (myid==ID0){
	for (k=0; k<n2; k++){
	  m = (p-is3[ID0])*n2 + k;
	  array0[2*k  ] = EVec1[m].r;
	  array0[2*k+1] = EVec1[m].i;
	}
      }

      /* MPI communications */
      MPI_Bcast(array0, n2*2, MPI_DOUBLE, ID0, mpi_comm_level1);

    } /* else */

      /* Save array0 */

    if (myid==Host_ID){
      fwrite(array0, sizeof(double), n2*2, fp1);
    }

  } /* p */

  /* close file pointers */

  if (myid==Host_ID){
    if (fp1) fclose(fp1);
  }

  /* freeing of array */

  free(array0);
  free(is3);
  free(ie3);
}




void Calc_XANES_NonCol( int n, int n2, int MaxN, int *MP,
			int *is2, int *ie2, dcomplex *EVec1, double *ko)

#define C1_ref(i,j) C1[ (n2)*(i)+(j) ]
#define C2_ref(i,j) C2[ (n2)*(i)+(j) ]

{
  int ii,jj,ia,jb,TNO1,TNO2,Rn;
  int spin,i,j,iemin,iemax,GA_AN,k,l;
  int Anum,Bnum,tnoA,tnoB,wanA,wanB;
  int MA_AN,LB_AN,GB_AN,MaxL,num,m,p,p1,po;
  int numprocs,myid,ID,tag,ID0,u1,u2,o1;
  int Num_Excited_States,Num_Occupied_States;
  int tno0,tno1,Hwan,h_AN,Mc_AN,Gc_AN,BN;
  int direction,Cwan,Gh_AN,spin0,spin1,spinstate;
  int i_vec[10],UMOmax,OMOmin;
  double dum,tmp,av_num;
  char operate[YOUSO10];
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp1;
  MPI_Status stat;
  MPI_Request request;
  int *is3,*ie3,*MP2;
  int ID1;
  int SpinP_switch1,n1,MaxN1;
  double Utot1,Utot2,Utot3;
  double sumr,sumi,sumxr,sumxi;
  double sumyr,sumyi,sumzr,sumzi,tmpr,tmpi;
  double *array0;
  dcomplex *C1,*C2,*Z,*Ax,*Ay,*Az;
  int Cluster_HOMO1[2],Num_XANES_Out,numex;
  int *ind2ind;
  double *****OLPmo,**XANES_Out;
  double XANES_Res[10];
  char file1[YOUSO10];

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* standard output */

  if (myid==Host_ID){
    printf("\n*******************************************************\n");
    printf("               Calculation of XANES spectrum            \n");
    printf("*******************************************************\n\n");
  }

  /* allocation of arrays */

  array0 = (double*)malloc(sizeof(dcomplex)*(n2+2)*2);
  is3 = (int*)malloc(sizeof(int)*numprocs);
  ie3 = (int*)malloc(sizeof(int)*numprocs);
  C2 = (dcomplex*)malloc(sizeof(dcomplex)*n2*MaxN);

  /* set iemin and imax */

  iemin = 1;
  iemax = MaxN;

  /****************************************************
   MPI communicate LCAO coefficients for excited state
  ****************************************************/

  /* MPI communicate LCAO coefficients */

  /* set is3 and ie3 */

  if ( numprocs<=MaxN ){

    av_num = (double)MaxN/(double)numprocs;
    for (ID=0; ID<numprocs; ID++){
      is3[ID] = (int)(av_num*(double)ID) + 1;
      ie3[ID] = (int)(av_num*(double)(ID+1));
    }

    is3[0] = 1;
    ie3[numprocs-1] = MaxN;
  }

  else{
    for (ID=0; ID<MaxN; ID++){
      is3[ID] = ID + 1;
      ie3[ID] = ID + 1;
    }
    for (ID=MaxN; ID<numprocs; ID++){
      is3[ID] =  1;
      ie3[ID] = -2;
    }
  }

  /* loop for p */

  for (p=iemin; p<=iemax; p++){

    /* store EVec1 */
    if (numprocs==1){
      for (k=0; k<n2; k++){
	m = (p-1)*n2 + k;
	array0[2*k  ] = EVec1[m].r;
	array0[2*k+1] = EVec1[m].i;
      }
    }
    else{

      po = 0;
      for (ID=0; ID<numprocs; ID++){
	if (is3[ID]<=p && p <=ie3[ID]){
	  po = 1;
	  ID0 = ID;
	  break;
	}
      }

      if (myid==ID0){
	for (k=0; k<n2; k++){
	  m = (p-is3[ID0])*n2 + k;
	  array0[2*k  ] = EVec1[m].r;
	  array0[2*k+1] = EVec1[m].i;
	}
      }

      /* MPI communications */
      MPI_Bcast(array0, n2*2, MPI_DOUBLE, ID0, mpi_comm_level1);

    } /* else */

    /* store array0 to C2 */

    for (k=0; k<n2; k++){
      C2[(p-iemin)*n2+k].r = array0[2*k  ];
      C2[(p-iemin)*n2+k].i = array0[2*k+1];
    }

  } /* p */

  /*****************************************************
      read the LCAO coefficients for the ground state
   *****************************************************/

  /* open file pointer */

  sprintf(operate,"%s%s",filepath,xanes_gs_file);

  if ((fp1 = fopen(operate,"rb")) != NULL){

    /* read parameters */

    fread(&n1, sizeof(int), 1, fp1);
    fread(&MaxN1, sizeof(int), 1, fp1);
    fread(Cluster_HOMO1, sizeof(int), 2, fp1);
    fread(&SpinP_switch1, sizeof(int), 1, fp1);
    fread(&Utot1, sizeof(double), 1, fp1);

    /* allocation of C1 */

    C1 = (dcomplex*)malloc(sizeof(dcomplex)*n2*MaxN1);

    /* read LCAO coefficients */

    fread(C1, sizeof(dcomplex), n2*MaxN1, fp1);

    /* close file pointer */
    fclose(fp1);
  }

  else{
    if (myid==Host_ID) printf("Could not find the xanes.gs.file: %s\n",operate);
    MPI_Finalize();
    exit(0);
  }

  /****************************************************
          start of calculation XANES spectrum
  ****************************************************/

  /* OLPmo: matrix for momentum operator */

  OLPmo = (double*****)malloc(sizeof(double****)*3);
  for (direction=0; direction<3; direction++){

    OLPmo[direction] = (double****)malloc(sizeof(double***)*(Matomnum+1));
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];
      }

      OLPmo[direction][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	}

	OLPmo[direction][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);
	for (i=0; i<tno0; i++){
	  OLPmo[direction][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
	}
      }
    }
  }

  /* calculation of momentum matrix elements */

  Calc_OLPmo(OLPmo);

  /* set MP */

  MP2 = (int*)malloc(sizeof(int)*(atomnum+1));

  p = 0;
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    MP2[Gc_AN] = p;
    Cwan = WhatSpecies[Gc_AN];
    tno0 = Spe_Total_CNO[Cwan];
    p += tno0;
  }

  /****************************************************
     determine the KS occupied and unoccupied states
     to be included
  ****************************************************/

  p = HOMO_XANES[0];
  UMOmax = p;

  for (k=p+1; k<=MaxN; k++){
    if ( (ko[k]-ko[p])<xanes_energy_range ) UMOmax = k;
  }

  OMOmin = p - 1;
  for (k=p-1; 1<=k; k--){
    if ( (ko[p]-ko[k])<xanes_energy_range ) OMOmin = k;
  }

  /* allocation of arrays */

  p = HOMO_XANES[0];

  Num_Excited_States = UMOmax - p;
  Num_Occupied_States = p - OMOmin;

  Num_XANES_Out = Num_Excited_States;

  XANES_Out = (double**)malloc(sizeof(double*)*Num_XANES_Out);
  for (i=0; i<Num_XANES_Out; i++){
    XANES_Out[i] = (double*)malloc(sizeof(double)*10);
  }

  Z = (dcomplex*)malloc(sizeof(dcomplex)*UMOmax*UMOmax);
  Ax = (dcomplex*)malloc(sizeof(dcomplex)*UMOmax*UMOmax);
  Ay = (dcomplex*)malloc(sizeof(dcomplex)*UMOmax*UMOmax);
  Az = (dcomplex*)malloc(sizeof(dcomplex)*UMOmax*UMOmax);

  ind2ind = (int*)malloc(sizeof(int)*UMOmax);

  /**************************************************************
   calculations of the overlap, px, py, and pz matrices between
   Kohn-Sham orbitals
  **************************************************************/

  for (ii=0; ii<UMOmax; ii++){
    for (jj=0; jj<UMOmax; jj++){

      sumr  = 0.0; sumi  = 0.0;
      sumxr = 0.0; sumxi = 0.0;
      sumyr = 0.0; sumyi = 0.0;
      sumzr = 0.0; sumzi = 0.0;

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	TNO1 = Spe_Total_CNO[Cwan];
	Anum = MP2[Gc_AN];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  Rn = ncn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[Hwan];
	  Bnum = MP2[Gh_AN];

	  for (i=0; i<TNO1; i++){

	    ia = Anum + i;

	    for (j=0; j<TNO2; j++){

	      jb = Bnum + j;

	      tmpr = C1_ref(ii,ia  ).r*C2_ref(jj,jb  ).r + C1_ref(ii,ia  ).i*C2_ref(jj,jb  ).i
                   + C1_ref(ii,ia+n).r*C2_ref(jj,jb+n).r + C1_ref(ii,ia+n).i*C2_ref(jj,jb+n).i;

	      tmpi = C1_ref(ii,ia  ).r*C2_ref(jj,jb  ).i - C1_ref(ii,ia  ).i*C2_ref(jj,jb  ).r
                   + C1_ref(ii,ia+n).r*C2_ref(jj,jb+n).i - C1_ref(ii,ia+n).i*C2_ref(jj,jb+n).r;

	      sumr  += tmpr*OLP[0][Mc_AN][h_AN][i][j];
	      sumi  += tmpi*OLP[0][Mc_AN][h_AN][i][j];
	      sumxr += tmpr*OLPmo[0][Mc_AN][h_AN][i][j];
	      sumxi += tmpi*OLPmo[0][Mc_AN][h_AN][i][j];
	      sumyr += tmpr*OLPmo[1][Mc_AN][h_AN][i][j];
	      sumyi += tmpi*OLPmo[1][Mc_AN][h_AN][i][j];
	      sumzr += tmpr*OLPmo[2][Mc_AN][h_AN][i][j];
	      sumzi += tmpi*OLPmo[2][Mc_AN][h_AN][i][j];

	    }
	  }
	}
      }

      Z[ UMOmax*jj+ii].r = sumr;   Z[ UMOmax*jj+ii].i = sumi;
      Ax[UMOmax*jj+ii].r = sumxr;  Ax[UMOmax*jj+ii].i = sumxi;
      Ay[UMOmax*jj+ii].r = sumyr;  Ay[UMOmax*jj+ii].i = sumyi;
      Az[UMOmax*jj+ii].r = sumzr;  Az[UMOmax*jj+ii].i = sumzi;

    } /* jj */
  } /* ii */

  /* MPI communication */

  MPI_Allreduce(MPI_IN_PLACE, Z,  UMOmax*UMOmax*2, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(MPI_IN_PLACE, Ax, UMOmax*UMOmax*2, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(MPI_IN_PLACE, Ay, UMOmax*UMOmax*2, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(MPI_IN_PLACE, Az, UMOmax*UMOmax*2, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /****************************************************
           contribution of single excitations
  ****************************************************/

  /* set parameters */

  p = HOMO_XANES[0];
  Num_Excited_States = UMOmax - p;
  Utot2 = Utot;

  /* loop for k */

  for (k=myid; k<Num_Excited_States; k+=numprocs){

    if (myid==Host_ID){
      printf("Calc... myid=%2d %4d/%4d\n",myid,k+1,Num_Excited_States);fflush(stdout);
    }

    /* set parameters */

    for (i=0; i<(p-1); i++){
      ind2ind[i] = i;
    }
    ind2ind[p-1] = p - 1 + k;

    /*
    if (myid==Host_ID){
      printf("Utot1=%15.12f Utot2=%15.12f Utot3=%15.12f p+k=%2d ko[p+k]=%15.12f\n",
	   Utot1,Utot2,Utot3,p+k,ko[p+k]);
    }
    */

    Utot3 = Utot2 + ko[p+k] - ko[p];

    Calc_Oscillator_Strength( n, n2, UMOmax, HOMO_XANES, MP2, ind2ind, C1, C2, Z, Ax, Ay, Az, Utot1, Utot3, XANES_Res );

    XANES_Out[k][0] = XANES_Res[0];
    XANES_Out[k][1] = XANES_Res[1];
    XANES_Out[k][2] = XANES_Res[2];
    XANES_Out[k][3] = XANES_Res[3];
    XANES_Out[k][4] = XANES_Res[4];
  }

  /* MPI */

  for (k=0; k<Num_Excited_States; k++){
    ID = k % numprocs;
    MPI_Bcast(&XANES_Out[k][0], 5, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  /*******************************************************
                       Output result
  *******************************************************/

  if (myid==Host_ID){

    sprintf(file1,"%s%s.xanes",filepath,filename);

    if ((fp1 = fopen(file1,"w")) != NULL){

      setvbuf(fp1,buf,_IOFBF,fp_bsize);  /* setvbuf */

      fprintf(fp1,"#1: Excitation energy (eV) #2: Oscillator Strength (OS) #3: x-comp. of OS #4: y-comp. of OS #5: z-comp. of OS\n");
      fprintf(fp1,"#\n");

      for (i=0; i<Num_XANES_Out; i++){
        fprintf(fp1,"%15.12f %15.12f %15.12f %15.12f %15.12f\n",
                    XANES_Out[i][0],XANES_Out[i][1],XANES_Out[i][2],XANES_Out[i][3],XANES_Out[i][4]);
      }

      /* fclose of fp1 */
      fclose(fp1);
    }
  }

  /*******************************************************
                    freeing of array
  *******************************************************/

  for (i=0; i<Num_XANES_Out; i++){
    free(XANES_Out[i]);
  }
  free(XANES_Out);

  free(Z);
  free(Ax);
  free(Ay);
  free(Az);
  free(ind2ind);
  free(MP2);
  free(C2);
  free(C1);

  free(array0);
  free(is3);
  free(ie3);

  for (direction=0; direction<3; direction++){

    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	}

	for (i=0; i<tno0; i++){
	  free(OLPmo[direction][Mc_AN][h_AN][i]);
	}
	free(OLPmo[direction][Mc_AN][h_AN]);
      }
      free(OLPmo[direction][Mc_AN]);
    }
    free(OLPmo[direction]);
  }
  free(OLPmo);
  OLPmo=NULL;
}







double Calc_Oscillator_Strength( int n, int n2, int UMOmax, int Nocc[2], int *MP2,
                                 int *ind2ind,
                                 dcomplex *Z0, dcomplex *A, dcomplex *Z,
                                 dcomplex *Ax, dcomplex *Ay, dcomplex *Az,
                                 double Utot1, double Utot2,
                                 double XANES_Res[10] )
{
  int ii,jj,Gc_AN,spin,Cwan,Hwan,Mc_AN;
  int i,j,k,ia,ja,ib,jb,Anum,Bnum,h_AN,Gh_AN,Rn,TNO1,TNO2;
  double os,osx,osy,osz,p2,px2,py2,pz2;
  double fsumr,fsumi;
  double tmpr,tmpi;
  dcomplex tmp,det,sum,allsum;
  double stime,etime;
  int numprocs,myid;

  dtime(&stime);

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* set Z0 which is an overlap matrix of two Slater determinants */

  for (jj=0; jj<Nocc[0]; jj++){

    j = ind2ind[jj];

    for (ii=0; ii<Nocc[0]; ii++){
      Z0[Nocc[0]*jj+ii].r = Z[UMOmax*j+ii].r;
      Z0[Nocc[0]*jj+ii].i = Z[UMOmax*j+ii].i;
    }
  }

  /* calculation of the inverse Z */

  det = Lapack_LU_Zinverse(Nocc[0], Z0);

  /* calculation of Z0^{-1} * det */

  for (i=0; i<Nocc[0]; i++){
    for (j=0; j<Nocc[0]; j++){

      tmpr = Z0[Nocc[0]*j+i].r;
      tmpi = Z0[Nocc[0]*j+i].i;
      Z0[Nocc[0]*j+i].r = tmpr*det.r - tmpi*det.i;
      Z0[Nocc[0]*j+i].i = tmpr*det.i + tmpi*det.r;
    }
  }

  /* calculation of Ax*(Z0^{-1*det)}, Ay*(Z0^{-1}*det), and Az*(Z0^{-1}*det) */

  for (k=0; k<3; k++){

    /* set Ax, Ay, or Az */

    for (ii=0; ii<Nocc[0]; ii++){

      if (k==0){
	for (jj=0; jj<Nocc[0]; jj++){
	  j = ind2ind[jj];
	  A[Nocc[0]*ii+jj] = Ax[UMOmax*j+ii];
	}
      }

      else if (k==1){
	for (jj=0; jj<Nocc[0]; jj++){
	  j = ind2ind[jj];
	  A[Nocc[0]*ii+jj] = Ay[UMOmax*j+ii];
	}
      }

      else if (k==2){
	for (jj=0; jj<Nocc[0]; jj++){
	  j = ind2ind[jj];
	  A[Nocc[0]*ii+jj] = Az[UMOmax*j+ii];
	}
      }
    } /* ii */

    /* A*(Z0^{-1*det)} */

    fsumr = 0.0; fsumi = 0.0;

    for (i=0; i<Nocc[0]; i++){
      for (j=0; j<Nocc[0]; j++){
	fsumr += A[Nocc[0]*i+j].r*Z0[Nocc[0]*i+j].r - A[Nocc[0]*i+j].i*Z0[Nocc[0]*i+j].i;
	fsumi += A[Nocc[0]*i+j].r*Z0[Nocc[0]*i+j].i + A[Nocc[0]*i+j].i*Z0[Nocc[0]*i+j].r;
      }
    }

    if      (k==0) px2 = fsumr*fsumr + fsumi*fsumi;
    else if (k==1) py2 = fsumr*fsumr + fsumi*fsumi;
    else if (k==2) pz2 = fsumr*fsumr + fsumi*fsumi;

  } /* k */

  p2 = px2 + py2 + pz2;

  os  = p2/fabs(Utot1-Utot2);
  osx = px2/fabs(Utot1-Utot2);
  osy = py2/fabs(Utot1-Utot2);
  osz = pz2/fabs(Utot1-Utot2);

  XANES_Res[0] = fabs(Utot1-Utot2)*27.2113845;
  XANES_Res[1] = os;
  XANES_Res[2] = osx;
  XANES_Res[3] = osy;
  XANES_Res[4] = osz;

  dtime(&etime);
  return (etime-stime);
}




dcomplex Lapack_LU_Zinverse(int n, dcomplex *A)
{
    static char *thisprogram="Lapack_LU_inverse";
    int *ipiv;
    dcomplex *work,tmp,det;
    int lwork;
    int info,i,j;

    /* L*U factorization */

    ipiv = (int*) malloc(sizeof(int)*n);

    F77_NAME(zgetrf,ZGETRF)(&n,&n,A,&n,ipiv,&info);

    if ( info !=0 ) {
      printf("zgetrf failed, info=%i, %s\n",info,thisprogram);
    }

    /* calculation of determinant */

    det.r = 1.0; det.i = 0.0;
    for (i=0; i<n; i++){
      tmp = det;
      det.r = tmp.r*A[n*(i)+(i)].r - tmp.i*A[n*(i)+(i)].i;
      det.i = tmp.r*A[n*(i)+(i)].i + tmp.i*A[n*(i)+(i)].r;
    }

    for (i=0; i<n; i++){
      if (ipiv[i] != i+1) { det.r = -det.r; det.i = -det.i; }
    }

    /*
    printf("det %15.12f %15.12f\n",det.r,det.i);

    for (i=0; i<n; i++){
      printf("i=%2d ipiv=%2d\n",i,ipiv[i]);
    }
    */

    /* inverse L*U factorization */

    lwork = 4*n;
    work = (dcomplex*)malloc(sizeof(dcomplex)*lwork);

    F77_NAME(zgetri,ZGETRI)(&n, A, &n, ipiv, work, &lwork, &info);

    if ( info !=0 ) {
      printf("zgetrf failed, info=%i, %s\n",info,thisprogram);
    }

    free(work); free(ipiv);

    return det;
}
