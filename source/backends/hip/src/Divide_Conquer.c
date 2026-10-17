/**********************************************************************
  Divide_Conquer.c:

     Divide_Conquer.c is a subroutine to perform a divide and conquer
     method for eigenvalue problem

  Log of Divide_Conquer.c:

     10/Dec/2003  Released by T.Ozaki

***********************************************************************/

#include "openmx_common.h"
#include "set_hip_default_device_from_local_rank.h"
#include "set_openmp_device_from_local_rank.h"
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define measure_time 0
// #define _BENCHMARK   1

extern int openmx_magma_dsyevdx_gpu(int n, int maxn, double *d_A, double *w, int *mout);
/* complex Hermitian solves of magma_openmx.cu: device matrix (lower
   triangle, vectors in place), host eigenvalues; the hipSOLVER one guards
   rocSOLVER's zheevd against its NaN eigenvectors */
extern int openmx_magma_zheevdx_gpu(int n, int maxn, void *d_A, double *w, int *mout);
extern int openmx_hipsolver_zheevd_gpu(int n, int maxn, void *d_A, double *w, int *mout);
extern int openmx_gpu_eigensolver_use_hipsolver(void);
extern void openmx_magma_ozaki2_release(void);
extern int openmx_hipsolver_release_workspace(void);
/* geam is outside the C subset of hipblas_compat.h */
hipblasStatus_t hipblasDgeam(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m,
                             int n, const double *alpha, const double *A, int lda, const double *beta,
                             const double *B, int ldb, double *C, int ldc);
hipblasStatus_t hipblasZgeam(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m,
                             int n, const hipDoubleComplex *alpha, const hipDoubleComplex *A, int lda,
                             const hipDoubleComplex *beta, const hipDoubleComplex *B, int ldb, hipDoubleComplex *C,
                             int ldc);

/* MAGMA's allocation failures (magma_types.h) */
#define DC_MAGMA_ERR_HOST_ALLOC   (-112)
#define DC_MAGMA_ERR_DEVICE_ALLOC (-113)

static double DC_Col(char * mode, int SCF_iter, double ***** Hks, double **** OLP0, double ***** CDM, double ***** EDM,
                     double Eele0[2], double Eele1[2]);

static double DC_NonCol(char * mode, int SCF_iter, double ***** Hks, double ***** ImNL, double **** OLP0,
                        double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2]);

static void Save_DOS_Col(double ****** Residues, double **** OLP0, double *** EVal, int * Msize);
static void Save_DOS_NonCol(dcomplex ****** Residues, double **** OLP0, double ** EVal, int * Msize);

/* GPU/CPU crossover for the per-atom truncated-cluster eigensolves;
   collinear compares NUM, noncollinear compares 2*NUM (spinor dimension) */
#define DC_GPU_CPU_SWITCH_NUM         400
#define DC_NONCOL_GPU_CPU_SWITCH_NUM  300

static int DC_GPU_Threshold(void)
{
    static int initialized = 0;
    static int threshold   = DC_GPU_CPU_SWITCH_NUM;

    if (!initialized) {
        char *env = getenv("OPENMX_DC_GPU_THRESHOLD");
        if (env != NULL && env[0] != '\0') {
            int env_threshold = atoi(env);
            if (0 < env_threshold) {
                threshold = env_threshold;
            }
        }
        initialized = 1;
    }

    return threshold;
}

static int DC_NonCol_GPU_Threshold(void)
{
    static int initialized = 0;
    static int threshold   = DC_NONCOL_GPU_CPU_SWITCH_NUM;

    if (!initialized) {
        char *env = getenv("OPENMX_DC_NONCOL_GPU_THRESHOLD");
        if (env != NULL && env[0] != '\0') {
            int env_threshold = atoi(env);
            if (0 < env_threshold) {
                threshold = env_threshold;
            }
        }
        initialized = 1;
    }

    return threshold;
}

static inline double DC_FermiWeight(double eval, double chemP, double beta, double max_x)
{
    double x = (eval - chemP) * beta;
    if (x <= -max_x)
        x = -max_x;
    if (max_x <= x)
        x = max_x;
    return 1.0 / (1.0 + exp(x));
}

static inline void DC_DotDensityResidueReal(int n, const double * restrict residue,
                                            const double * restrict weight,
                                            const double * restrict eweight,
                                            double * restrict cdm,
                                            double * restrict edm)
{
    int    i;
    double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
    double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;

    for (i = 0; i + 3 < n; i += 4) {
        const double r0 = residue[i];
        const double r1 = residue[i + 1];
        const double r2 = residue[i + 2];
        const double r3 = residue[i + 3];

        c0 += weight[i] * r0;
        e0 += eweight[i] * r0;
        c1 += weight[i + 1] * r1;
        e1 += eweight[i + 1] * r1;
        c2 += weight[i + 2] * r2;
        e2 += eweight[i + 2] * r2;
        c3 += weight[i + 3] * r3;
        e3 += eweight[i + 3] * r3;
    }

    for (; i < n; i++) {
        const double r = residue[i];
        c0 += weight[i] * r;
        e0 += eweight[i] * r;
    }

    *cdm = (c0 + c1) + (c2 + c3);
    *edm = (e0 + e1) + (e2 + e3);
}

static inline void DC_DotDensityResidueNonCol(int n, const dcomplex * restrict res0,
                                              const dcomplex * restrict res1,
                                              const dcomplex * restrict res2,
                                              const double * restrict weight,
                                              const double * restrict eweight, int use_iDM,
                                              double * restrict cdm0, double * restrict edm0,
                                              double * restrict cdm1, double * restrict edm1,
                                              double * restrict cdm2, double * restrict edm2,
                                              double * restrict cdm3, double * restrict edm3,
                                              double * restrict idm00, double * restrict idm01)
{
    int    i;
    double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
    double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;
    double im00 = 0.0, im01 = 0.0;

    if (use_iDM) {
        for (i = 0; i < n; i++) {
            const double f  = weight[i];
            const double ef = eweight[i];
            const double r0 = res0[i].r;
            const double r1 = res1[i].r;
            const double r2 = res2[i].r;
            const double r3 = res2[i].i;

            c0 += f * r0;
            e0 += ef * r0;
            c1 += f * r1;
            e1 += ef * r1;
            c2 += f * r2;
            e2 += ef * r2;
            c3 += f * r3;
            e3 += ef * r3;
            im00 += f * res0[i].i;
            im01 += f * res1[i].i;
        }
    } else {
        for (i = 0; i < n; i++) {
            const double f  = weight[i];
            const double ef = eweight[i];
            const double r0 = res0[i].r;
            const double r1 = res1[i].r;
            const double r2 = res2[i].r;
            const double r3 = res2[i].i;

            c0 += f * r0;
            e0 += ef * r0;
            c1 += f * r1;
            e1 += ef * r1;
            c2 += f * r2;
            e2 += ef * r2;
            c3 += f * r3;
            e3 += ef * r3;
        }
    }

    *cdm0 = c0;
    *edm0 = e0;
    *cdm1 = c1;
    *edm1 = e1;
    *cdm2 = c2;
    *edm2 = e2;
    *cdm3 = c3;
    *edm3 = e3;
    *idm00 = im00;
    *idm01 = im01;
}

static void DC_AbortWithMessage(const char *message)
{
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1, 1);
}

static size_t DC_CheckedMulCount(size_t a, size_t b, const char *label)
{
    if (a != 0 && b > SIZE_MAX / a) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Dimension overflow in Divide_Conquer.c: %s", label);
        DC_AbortWithMessage(msg);
    }

    return a * b;
}

static size_t DC_CheckedArrayBytes(size_t count, size_t elem_size, const char *label)
{
    if (count != 0 && elem_size > SIZE_MAX / count) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Allocation size overflow in Divide_Conquer.c: %s", label);
        DC_AbortWithMessage(msg);
    }

    return count * elem_size;
}

static void *DC_MallocArray(size_t count, size_t elem_size, const char *label)
{
    size_t bytes = DC_CheckedArrayBytes(count, elem_size, label);
    void * ptr   = malloc(bytes == 0 ? 1 : bytes);

    if (ptr == NULL) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Out of memory in Divide_Conquer.c: %s (%zu bytes)", label, bytes);
        DC_AbortWithMessage(msg);
    }

    return ptr;
}

typedef struct {
    int                initialized;
    int                device_id;
    int                matrix_dim;
    int                loaded_s_dim;
    hipStream_t       stream;
    hipblasHandle_t     hipblas;
    double *           d_S;
    double *           d_H;
    double *           d_tmp;
    double *           d_A;
    double *           d_C;
    double *           h_matrix;
} DCGpuSolverCtx;

static DCGpuSolverCtx DC_gpusolver_ctx = {0};
static int           DC_gpusolver_gemm_disabled = 0;
/* the GEMMul8 products of DC are off by default on AMD GPUs; the run
   boundary resets the flag to this value, not to 0 */
#ifdef __HIP_PLATFORM_AMD__
#define DC_GPUSOLVER_GEMMUL8_DISABLED_DEFAULT 1
#else
#define DC_GPUSOLVER_GEMMUL8_DISABLED_DEFAULT 0
#endif
static int           DC_gpusolver_gemmul8_disabled = DC_GPUSOLVER_GEMMUL8_DISABLED_DEFAULT;
static int           DC_gpusolver_eigen_disabled = 0;
/* Once these monotonically growing workspaces have passed a memory
   preflight, equal or smaller per-atom solves need no further query. */
static int           DC_gpusolver_approved_gemm_n = 0;
static int           DC_gpusolver_approved_gemm_num1 = 0;
static int           DC_gpusolver_native_approved = 0;

typedef struct {
    int state;
    int matomnum;
    int *dim;
    int *valid;
    size_t *off;
    double *d_arena;
    size_t arena_elems;
} DCColSCacheCtx;

static DCColSCacheCtx DC_scache = {0};

/* A slot starts on a 512-byte boundary like a hipMalloc'd matrix, so a
   solve that reads it in place sees the operand alignment it had before. */
#define DC_SCACHE_SLOT_ALIGN 64

static int DC_GpuSolver_RankAllowed(void);
static int DC_GpuSolver_GemmDisabled(void);
static int DC_GpuSolver_EigenDisabled(void);

void Divide_Conquer_Release_GPU_SCache(void)
{
    if (DC_scache.d_arena != NULL) wait_hipfunc(hipFree(DC_scache.d_arena));
    free(DC_scache.dim);
    free(DC_scache.valid);
    free(DC_scache.off);
    memset(&DC_scache, 0, sizeof(DC_scache));
}

/* Ranks sharing this node's device(s); conservative on multi-GPU nodes.
   The split is collective, so the first call must be reached by every rank. */
static int DC_NodeRanks(void)
{
    static int ranks = 0;

    if (ranks == 0) {
        MPI_Comm node_comm = MPI_COMM_NULL;

        MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
        MPI_Comm_size(node_comm, &ranks);
        MPI_Comm_free(&node_comm);
        if (ranks < 1)
            ranks = 1;
    }

    return ranks;
}

static void DCCol_SCache_Prepare(int matomnum, const int *Msize, int scf_iter)
{
    const char *enabled_env = getenv("OPENMX_DC_GPU_S_CACHE");
    const char *reserve_env = getenv("OPENMX_DC_GPU_S_CACHE_RESERVE_MB");
    size_t free_bytes = 0, total_bytes = 0, reserve = (size_t)4096 * 1024 * 1024;
    size_t used = 0, budget;
    int threshold = DC_GPU_Threshold();
    /* collective on the first call, so it precedes every rank-local return */
    int node_ranks = DC_NodeRanks();
    int admitted = 0;

    if (enabled_env != NULL && atoi(enabled_env) == 0) return;
    if (reserve_env != NULL) {
        long mib = atol(reserve_env);
        if (0 <= mib) reserve = (size_t)mib * 1024U * 1024U;
    }
    /* a rank that never runs the GPU solve (rank limit, or the path is off
       for the rest of the run) holds no arena */
    if (!DC_GpuSolver_RankAllowed() || DC_GpuSolver_GemmDisabled() || DC_GpuSolver_EigenDisabled()) {
        Divide_Conquer_Release_GPU_SCache();
        return;
    }
    if (DC_scache.state == 1 && DC_scache.matomnum == matomnum) {
        int same = 1;
        for (int a=1; a<=matomnum; a++)
            if (DC_scache.dim[a] != 0 && DC_scache.dim[a] != Msize[a]) same = 0;
        if (same) {
            if (scf_iter <= 2) memset(DC_scache.valid, 0, sizeof(int)*(size_t)(matomnum+1));
            return;
        }
        Divide_Conquer_Release_GPU_SCache();
    }
    if (DC_scache.state == -1) return;

    DC_scache.dim = (int*)calloc((size_t)matomnum+1, sizeof(int));
    DC_scache.valid = (int*)calloc((size_t)matomnum+1, sizeof(int));
    DC_scache.off = (size_t*)calloc((size_t)matomnum+1, sizeof(size_t));
    if (DC_scache.dim == NULL || DC_scache.valid == NULL || DC_scache.off == NULL)
        DC_AbortWithMessage("Could not allocate DC GPU S-cache tables.");

    if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        (void)hipGetLastError();
        Divide_Conquer_Release_GPU_SCache();
        DC_scache.state = -1;
        return;
    }
    budget = free_bytes > reserve ? (free_bytes-reserve)/(size_t)node_ranks : 0;
    for (int a=1; a<=matomnum; a++) {
        size_t elems = DC_CheckedMulCount((size_t)Msize[a], (size_t)Msize[a], "DC S-cache slot");
        size_t bytes = DC_CheckedArrayBytes(elems, sizeof(double), "DC S-cache slot");
        if (Msize[a] < threshold || bytes > budget || used*sizeof(double) > budget-bytes) continue;
        DC_scache.dim[a] = Msize[a];
        DC_scache.off[a] = used;
        used += (elems + DC_SCACHE_SLOT_ALIGN - 1) / DC_SCACHE_SLOT_ALIGN * DC_SCACHE_SLOT_ALIGN;
        admitted++;
    }
    if (used != 0 && hipMalloc((void**)&DC_scache.d_arena, used*sizeof(double)) != hipSuccess) {
        /* the run goes on without the cache: no latched error may remain
           for the next hipGetLastError() caller */
        DC_scache.d_arena = NULL;
        (void)hipGetLastError();
        Divide_Conquer_Release_GPU_SCache();
        DC_scache.state = -1;
        return;
    }
    DC_scache.matomnum = matomnum;
    DC_scache.arena_elems = used;
    DC_scache.state = 1;

    if (0 < level_stdout && 0 < admitted) {
        int myid = -1;

        MPI_Comm_rank(mpi_comm_level1, &myid);
        if (myid == Host_ID && 2 <= level_stdout) {
            printf("<DC> GPU S-cache: rank %d keeps %d of %d cluster overlaps on the device (%.1f MiB)\n",
                   myid, admitted, matomnum, (double)(used * sizeof(double)) / (1024.0 * 1024.0));
            fflush(stdout);
        }
    }
}

static double *DCCol_SCache_Ptr(int Mc_AN, int n)
{
    if (DC_scache.state != 1 || Mc_AN < 1 || DC_scache.matomnum < Mc_AN ||
        DC_scache.dim[Mc_AN] != n) return NULL;
    return DC_scache.d_arena + DC_scache.off[Mc_AN];
}

/* the slot of Mc_AN when it holds this geometry's transformed overlap */
static const double *DCCol_SCache_ValidPtr(int Mc_AN, int n)
{
    double *slot = DCCol_SCache_Ptr(Mc_AN, n);

    return (slot != NULL && DC_scache.valid[Mc_AN]) ? slot : NULL;
}

static unsigned DC_EnvU32(const char *name, unsigned fallback)
{
    const char *value = getenv(name);
    char       *end   = NULL;
    unsigned long parsed;

    if (value == NULL || value[0] == '\0') {
        return fallback;
    }

    parsed = strtoul(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }

    return (unsigned)parsed;
}

static int DC_EnvI32Any(const char *name1, const char *name2, const char *name3, int fallback)
{
    const char *value = NULL;
    char       *end   = NULL;
    long        parsed;

    if (name1 != NULL)
        value = getenv(name1);
    if ((value == NULL || value[0] == '\0') && name2 != NULL)
        value = getenv(name2);
    if ((value == NULL || value[0] == '\0') && name3 != NULL)
        value = getenv(name3);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }

    parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }

    return (int)parsed;
}

static int DC_GpuSolver_RankAllowed(void)
{
    static int initialized = 0;
    static int allowed     = 1;

    if (!initialized) {
        int      local_rank;
        int      device_count = 1;
        unsigned max_ranks;

#ifdef __HIP_PLATFORM_AMD__
        max_ranks = DC_EnvU32("DC_GPU_MAX_RANKS_PER_DEVICE", 4u);
#else
        max_ranks = DC_EnvU32("DC_GPU_MAX_RANKS_PER_DEVICE", 1024u);
#endif
        max_ranks = DC_EnvU32("OPENMX_DC_GPU_MAX_RANKS_PER_DEVICE", max_ranks);

        local_rank = DC_EnvI32Any("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", -1);
        if (local_rank < 0) {
            MPI_Comm_rank(mpi_comm_level1, &local_rank);
        }

        if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) {
            device_count = 1;
        }

        allowed = (0 < max_ranks && (unsigned)(local_rank / device_count) < max_ranks);
        initialized = 1;
    }

    return allowed;
}

static const char *DC_GpuSolver_HipblasStatusName(hipblasStatus_t status)
{
    if (status == HIPBLAS_STATUS_SUCCESS)
        return "HIPBLAS_STATUS_SUCCESS";
    if (status == HIPBLAS_STATUS_NOT_INITIALIZED)
        return "HIPBLAS_STATUS_NOT_INITIALIZED";
    if (status == HIPBLAS_STATUS_ALLOC_FAILED)
        return "HIPBLAS_STATUS_ALLOC_FAILED";
    if (status == HIPBLAS_STATUS_INVALID_VALUE)
        return "HIPBLAS_STATUS_INVALID_VALUE";
    if (status == HIPBLAS_STATUS_ARCH_MISMATCH)
        return "HIPBLAS_STATUS_ARCH_MISMATCH";
    if (status == HIPBLAS_STATUS_MAPPING_ERROR)
        return "HIPBLAS_STATUS_MAPPING_ERROR";
    if (status == HIPBLAS_STATUS_EXECUTION_FAILED)
        return "HIPBLAS_STATUS_EXECUTION_FAILED";
    if (status == HIPBLAS_STATUS_INTERNAL_ERROR)
        return "HIPBLAS_STATUS_INTERNAL_ERROR";
    if (status == HIPBLAS_STATUS_NOT_SUPPORTED)
        return "HIPBLAS_STATUS_NOT_SUPPORTED";
    if (status == HIPBLAS_STATUS_INTERNAL_ERROR)
        return "HIPBLAS_STATUS_LICENSE_ERROR";
    return "HIPBLAS_STATUS_ERROR";
}

static void DC_GpuSolver_DisableGemmPathHip(const char *where, hipError_t status);

static void DC_Eigen_lapack_cpu(double **a, double *ko, int n, int EVmax)
{
    int saved_scf_eigen_lib_flag = scf_eigen_lib_flag;

    scf_eigen_lib_flag = ELPA2;
    Eigen_lapack(a, ko, n, EVmax);
    scf_eigen_lib_flag = saved_scf_eigen_lib_flag;
}

static int DC_GpuSolver_GemmDisabled(void)
{
    return DC_gpusolver_gemm_disabled;
}

static int DC_GpuSolver_EigenDisabled(void)
{
    return DC_gpusolver_eigen_disabled;
}

static size_t DC_GpuSolver_Gemmul8ReserveBytes(void)
{
    unsigned reserve_mib;

#ifdef __HIP_PLATFORM_AMD__
    reserve_mib = DC_EnvU32("GEMMUL8_MIN_FREE_AFTER_MB", 64u);
#else
    reserve_mib = DC_EnvU32("GEMMUL8_MIN_FREE_AFTER_MB", 1536u);
#endif
    reserve_mib = DC_EnvU32("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB", reserve_mib);

    return (size_t)reserve_mib * (size_t)1024 * (size_t)1024;
}

static size_t DC_GpuSolver_HipblasReserveBytes(void)
{
    unsigned reserve_mib;

#ifdef __HIP_PLATFORM_AMD__
    reserve_mib = DC_EnvU32("CUBLAS_MIN_FREE_AFTER_MB", 64u);
#else
    reserve_mib = DC_EnvU32("CUBLAS_MIN_FREE_AFTER_MB", 3072u);
#endif
    reserve_mib = DC_EnvU32("OPENMX_CUBLAS_MIN_FREE_AFTER_MB", reserve_mib);

    return (size_t)reserve_mib * (size_t)1024 * (size_t)1024;
}

static void DC_GpuSolver_ReleaseGemmScratch(void)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;

    if (ctx->d_H != NULL) {
        wait_hipfunc(hipFree(ctx->d_H));
        ctx->d_H = NULL;
    }
    if (ctx->d_tmp != NULL) {
        wait_hipfunc(hipFree(ctx->d_tmp));
        ctx->d_tmp = NULL;
    }
    if (ctx->d_A != NULL) {
        wait_hipfunc(hipFree(ctx->d_A));
        ctx->d_A = NULL;
    }
    if (ctx->d_C != NULL) {
        wait_hipfunc(hipFree(ctx->d_C));
        ctx->d_C = NULL;
    }
}

static void DC_GpuSolver_ReleaseTransformScratch(void)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;

    if (ctx->d_H != NULL) {
        wait_hipfunc(hipFree(ctx->d_H));
        ctx->d_H = NULL;
    }
    if (ctx->d_tmp != NULL) {
        wait_hipfunc(hipFree(ctx->d_tmp));
        ctx->d_tmp = NULL;
    }
}

static void DC_GpuSolver_ReleaseOverlapBuffer(void)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;

    if (ctx->d_S != NULL) {
        wait_hipfunc(hipFree(ctx->d_S));
        ctx->d_S = NULL;
    }
    ctx->loaded_s_dim = 0;
}

static void DC_GpuSolver_Destroy(void);
static void DCNonCol_GpuRelease(int keep_stage);

/* Run-boundary reset (between -runtest/-runtestL inputs): drop every scratch
   buffer and the overlap cache, all sized and indexed by the previous
   system, and return the stream, the hipBLAS handle and the pinned staging
   buffers too; the next GPU solve re-creates them on demand
   (DC_GpuSolver_Init).  The S-cache is not part of the context and is
   dropped on its own.  Idempotent. */
void Divide_Conquer_Release_GPU_Caches(void)
{
    DCNonCol_GpuRelease(0);
    DC_GpuSolver_Destroy();
    Divide_Conquer_Release_GPU_SCache();

    /* A memory admission failure belongs to the finished system: the next
       input starts with the GPU paths enabled again (and the GEMMul8
       products at their build default). */
    DC_gpusolver_gemm_disabled = 0;
    DC_gpusolver_gemmul8_disabled = DC_GPUSOLVER_GEMMUL8_DISABLED_DEFAULT;
    DC_gpusolver_eigen_disabled = 0;
    DC_gpusolver_approved_gemm_n = 0;
    DC_gpusolver_approved_gemm_num1 = 0;
    DC_gpusolver_native_approved = 0;
}

static void DC_GpuSolver_DisableEigenPathInfo(const char *where, int32_t info)
{
    int rank = -1;

    DC_gpusolver_eigen_disabled = 1;
    DC_gpusolver_gemm_disabled  = 1;
    openmx_gemmul8ReleaseWorkspaces();
    /* a failed MAGMA allocation can leave a latched HIP error behind */
    (void)hipGetLastError();

    MPI_Comm_rank(mpi_comm_level1, &rank);
    fprintf(stderr,
            "<DC> rank %d: MAGMA eigensolver failed in %s: magma_dsyevdx_gpu info=%d. "
            "Falling back to the CPU divide-conquer solve.\n",
            rank, where, (int)info);
    fflush(stderr);
}

static void DC_GpuSolver_DisableEigenPathEigenpairs(const char *where, int64_t h_meig, int maxn)
{
    int rank = -1;

    DC_gpusolver_eigen_disabled = 1;
    DC_gpusolver_gemm_disabled  = 1;
    openmx_gemmul8ReleaseWorkspaces();

    MPI_Comm_rank(mpi_comm_level1, &rank);
    fprintf(stderr,
            "<DC> rank %d: GPU solver returned %lld eigenpairs in %s, expected %d. "
            "Falling back to the CPU divide-conquer solve.\n",
            rank, (long long)h_meig, where, maxn);
    fflush(stderr);
}

static size_t DC_GpuSolver_EstimateGemmul8WorkspaceBytes(int m, int n, int k)
{
    size_t mn, mk, kn, max_elems;

    mn        = DC_CheckedMulCount((size_t)m, (size_t)n, "GEMMul8 workspace estimate mn");
    mk        = DC_CheckedMulCount((size_t)m, (size_t)k, "GEMMul8 workspace estimate mk");
    kn        = DC_CheckedMulCount((size_t)k, (size_t)n, "GEMMul8 workspace estimate kn");
    max_elems = mn;
    if (max_elems < mk)
        max_elems = mk;
    if (max_elems < kn)
        max_elems = kn;

    return DC_CheckedArrayBytes(max_elems, sizeof(double) * (size_t)20, "GEMMul8 workspace estimate");
}

static int DC_GpuSolver_HasHipblasMemoryForSolve(const char *where)
{
    size_t      reserve_bytes;
    size_t      free_bytes = 0, total_bytes = 0;
    hipError_t status;
    int         rank = -1;

    if (DC_GpuSolver_GemmDisabled()) {
        return 0;
    }

    if (DC_gpusolver_native_approved) {
        return 1;
    }

    reserve_bytes = DC_GpuSolver_HipblasReserveBytes();
    status        = OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes);
    if (status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipMemGetInfo(native hipBLAS preflight)", status);
        return 0;
    }

    if (free_bytes < reserve_bytes) {
        DC_gpusolver_gemm_disabled = 1;
        DC_gpusolver_eigen_disabled = 1;
        openmx_gemmul8ReleaseWorkspaces();

        MPI_Comm_rank(mpi_comm_level1, &rank);
        fprintf(stderr,
                "<DC> rank %d: native hipBLAS is also unsafe before %s; "
                "GPU free %.3f MiB / total %.3f MiB, reserve %.3f MiB. "
                "Falling back to the CPU divide-conquer Hamiltonian solve.\n",
                rank, where, (double)free_bytes / (1024.0 * 1024.0),
                (double)total_bytes / (1024.0 * 1024.0), (double)reserve_bytes / (1024.0 * 1024.0));
        fflush(stderr);
        return 0;
    }

    DC_gpusolver_native_approved = 1;
    return 1;
}

static int DC_GpuSolver_PrepareGemmBackendForSolve(int n, int num1)
{
    size_t      reserve_bytes, estimate_bytes, estimate2;
    size_t      free_bytes = 0, total_bytes = 0;
    hipError_t status;
    int         rank = -1;

    if (DC_GpuSolver_GemmDisabled()) {
        return 0;
    }

    if (DC_gpusolver_gemmul8_disabled) {
        return DC_GpuSolver_HasHipblasMemoryForSolve("DC Hamiltonian GEMM");
    }

    if (n <= DC_gpusolver_approved_gemm_n &&
        num1 <= DC_gpusolver_approved_gemm_num1) {
        return 1;
    }

    reserve_bytes  = DC_GpuSolver_Gemmul8ReserveBytes();
    estimate_bytes = DC_GpuSolver_EstimateGemmul8WorkspaceBytes(n, n, n);
    estimate2      = DC_GpuSolver_EstimateGemmul8WorkspaceBytes(n, num1, num1);
    if (estimate_bytes < estimate2) {
        estimate_bytes = estimate2;
    }

    status = OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes);
    if (status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipMemGetInfo(GEMM preflight)", status);
        return 0;
    }

    if (free_bytes < reserve_bytes || free_bytes - reserve_bytes < estimate_bytes) {
        DC_gpusolver_gemmul8_disabled = 1;
        openmx_gemmul8ReleaseWorkspaces();

        MPI_Comm_rank(mpi_comm_level1, &rank);
        fprintf(stderr,
                "<DC> rank %d: GEMMul8 is unsafe before DC Hamiltonian GEMM; "
                "estimated GEMMul8 workspace %.3f MiB, "
                "GPU free %.3f MiB / total %.3f MiB, reserve %.3f MiB. "
                "Trying native hipBLAS instead.\n",
                rank, (double)estimate_bytes / (1024.0 * 1024.0), (double)free_bytes / (1024.0 * 1024.0),
                (double)total_bytes / (1024.0 * 1024.0), (double)reserve_bytes / (1024.0 * 1024.0));
        fflush(stderr);
        return DC_GpuSolver_HasHipblasMemoryForSolve("DC Hamiltonian GEMM");
    }

    if (DC_gpusolver_approved_gemm_n < n)
        DC_gpusolver_approved_gemm_n = n;
    if (DC_gpusolver_approved_gemm_num1 < num1)
        DC_gpusolver_approved_gemm_num1 = num1;
    return 1;
}

static void DC_GpuSolver_DisableGemmPath(const char *where, const char *backend, hipblasStatus_t status, int m, int n,
                                        int k)
{
    int rank = -1;

    DC_gpusolver_gemm_disabled = 1;
    DC_gpusolver_eigen_disabled = 1;
    openmx_gemmul8ReleaseWorkspaces();

    /* pop any latched HIP error: this path recovers by falling back to the
       CPU, so the error must not linger for the next hipGetLastError() caller */
    (void)hipGetLastError();

    MPI_Comm_rank(mpi_comm_level1, &rank);
    fprintf(stderr,
            "<DC> rank %d: %s failed in %s for GEMM(m=%d,n=%d,k=%d): %s (%d). "
            "Falling back to the CPU divide-conquer Hamiltonian solve.\n",
            rank, backend, where, m, n, k, DC_GpuSolver_HipblasStatusName(status), (int)status);
    fflush(stderr);
}

static void DC_GpuSolver_DisableGemmPathHip(const char *where, hipError_t status)
{
    int rank = -1;

    DC_gpusolver_gemm_disabled = 1;
    DC_gpusolver_eigen_disabled = 1;
    openmx_gemmul8ReleaseWorkspaces();

    /* pop any latched HIP error: this path recovers by falling back to the
       CPU, so the error must not linger for the next hipGetLastError() caller */
    (void)hipGetLastError();

    MPI_Comm_rank(mpi_comm_level1, &rank);
    fprintf(stderr,
            "<DC> rank %d: %s failed in the GPU divide-conquer solve: %s (%d). "
            "Falling back to the CPU path.\n",
            rank, where, hipGetErrorString(status), (int)status);
    fflush(stderr);
}

static int DC_GpuSolver_TryGpuDgemm(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m,
                                   int n, int k, const double *alpha, const double *A, int lda, const double *B,
                                   int ldb, const double *beta, double *C, int ldc, const char *where)
{
    hipblasStatus_t status;
    int            rank = -1;

    if (DC_GpuSolver_GemmDisabled()) {
        return 0;
    }

    if (!DC_gpusolver_gemmul8_disabled) {
        status = openmx_gemmul8Dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        if (status == HIPBLAS_STATUS_SUCCESS) {
            return 1;
        }

        DC_gpusolver_gemmul8_disabled = 1;
        openmx_gemmul8ReleaseWorkspaces();

        MPI_Comm_rank(mpi_comm_level1, &rank);
        fprintf(stderr,
                "<DC> rank %d: GEMMul8 failed in %s for GEMM(m=%d,n=%d,k=%d): %s (%d). "
                "Retrying this DC Hamiltonian solve with native hipBLAS.\n",
                rank, where, m, n, k, DC_GpuSolver_HipblasStatusName(status), (int)status);
        fflush(stderr);

        if (!DC_GpuSolver_HasHipblasMemoryForSolve(where)) {
            return 0;
        }
    }

    status = hipblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        return 1;
    }

    DC_GpuSolver_DisableGemmPath(where, "native hipBLAS", status, m, n, k);
    return 0;
}

static void DC_GpuSolver_Destroy(void)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;

    if (ctx->d_S != NULL)
        wait_hipfunc(hipFree(ctx->d_S));
    if (ctx->d_H != NULL)
        wait_hipfunc(hipFree(ctx->d_H));
    if (ctx->d_tmp != NULL)
        wait_hipfunc(hipFree(ctx->d_tmp));
    if (ctx->d_A != NULL)
        wait_hipfunc(hipFree(ctx->d_A));
    if (ctx->d_C != NULL)
        wait_hipfunc(hipFree(ctx->d_C));
    if (ctx->h_matrix != NULL)
        wait_hipfunc(hipHostFree(ctx->h_matrix));
    if (ctx->hipblas != NULL)
        wait_hipfunc(hipblasDestroy(ctx->hipblas));
    if (ctx->stream != NULL)
        wait_hipfunc(hipStreamDestroy(ctx->stream));

    memset(ctx, 0, sizeof(*ctx));
    ctx->device_id    = -1;
    ctx->loaded_s_dim = 0;
}

/* Stream and hipBLAS handle, created on demand (also after the run-boundary
   release); a failure turns the GPU solve off, the caller falls back. */
static int DC_GpuSolver_Init(void)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;
    int            current_device;
    hipError_t     hip_status;
    hipblasStatus_t blas_status;

    hip_status = hipGetDevice(&current_device);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipGetDevice", hip_status);
        return 0;
    }

    if (ctx->initialized && ctx->device_id == current_device) {
        return 1;
    }

    if (ctx->initialized) {
        DC_GpuSolver_Destroy();
    }

    hip_status = hipStreamCreateWithFlags(&ctx->stream, hipStreamNonBlocking);
    if (hip_status != hipSuccess) {
        ctx->stream = NULL;
        DC_GpuSolver_DisableGemmPathHip("hipStreamCreateWithFlags", hip_status);
        DC_GpuSolver_Destroy();
        return 0;
    }
    blas_status = hipblasCreate(&ctx->hipblas);
    if (blas_status != HIPBLAS_STATUS_SUCCESS) {
        ctx->hipblas = NULL;
    } else {
        blas_status = hipblasSetStream(ctx->hipblas, ctx->stream);
    }
    if (blas_status != HIPBLAS_STATUS_SUCCESS) {
        DC_GpuSolver_DisableGemmPath("the solver setup", "hipblasCreate/hipblasSetStream", blas_status, 0, 0, 0);
        DC_GpuSolver_Destroy();
        return 0;
    }

    ctx->initialized  = 1;
    ctx->device_id    = current_device;
    ctx->loaded_s_dim = 0;
    return 1;
}

/* The pinned staging buffer and the capacity of the device matrices.
   Growing frees every device matrix, and DC_GpuSolver_AllocMatrix always
   allocates matrix_dim x matrix_dim, so a buffer that exists fits any
   atom up to matrix_dim (an overlap reloaded for a smaller atom must not
   be taken as large enough for the next one). */
static int DC_GpuSolver_EnsureMatrixCapacity(int num)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;
    size_t         matrix_bytes;
    hipError_t    hip_status;

    if (num <= 0) {
        DC_AbortWithMessage("Invalid matrix size in DC_GpuSolver_EnsureMatrixCapacity.");
    }

    if (!DC_GpuSolver_Init()) {
        return 0;
    }

    if (num <= ctx->matrix_dim && ctx->h_matrix != NULL) {
        return 1;
    }

    DC_GpuSolver_ReleaseOverlapBuffer();
    DC_GpuSolver_ReleaseGemmScratch();
    if (ctx->h_matrix != NULL) {
        wait_hipfunc(hipHostFree(ctx->h_matrix));
        ctx->h_matrix = NULL;
    }

    ctx->matrix_dim   = 0;
    ctx->loaded_s_dim = 0;

    matrix_bytes = DC_CheckedArrayBytes(DC_CheckedMulCount((size_t)num, (size_t)num, "GPU solver dense matrix"),
                                        sizeof(double), "GPU solver dense matrix");

    hip_status = hipHostMalloc((void **)&ctx->h_matrix, matrix_bytes, 0);
    if (hip_status != hipSuccess) {
        ctx->h_matrix = NULL;
        DC_GpuSolver_DisableGemmPathHip("hipHostMalloc(h_matrix)", hip_status);
        DC_GpuSolver_Destroy();
        return 0;
    }

    ctx->matrix_dim   = num;
    ctx->loaded_s_dim = 0;
    return 1;
}

/* *d_matrix becomes a matrix_dim x matrix_dim device matrix (kept if it
   exists); on failure the GPU path is off and the context is gone. */
static int DC_GpuSolver_AllocMatrix(double **d_matrix, const char *where)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;
    size_t         matrix_bytes;
    hipError_t    hip_status;

    if (*d_matrix != NULL) {
        return 1;
    }
    if (ctx->matrix_dim <= 0) {
        DC_AbortWithMessage("DC_GpuSolver_AllocMatrix called before DC_GpuSolver_EnsureMatrixCapacity.");
    }

    matrix_bytes = DC_CheckedArrayBytes(DC_CheckedMulCount((size_t)ctx->matrix_dim, (size_t)ctx->matrix_dim,
                                                           "GPU solver dense matrix"),
                                        sizeof(double), "GPU solver dense matrix");

    hip_status = hipMalloc((void **)d_matrix, matrix_bytes);
    if (hip_status != hipSuccess) {
        *d_matrix = NULL;
        DC_GpuSolver_DisableGemmPathHip(where, hip_status);
        DC_GpuSolver_Destroy();
        return 0;
    }
    return 1;
}

static int DC_GpuSolver_EnsureGemmCapacity(int num)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;

    return DC_GpuSolver_EnsureMatrixCapacity(num) && DC_GpuSolver_AllocMatrix(&ctx->d_H, "hipMalloc(d_H)") &&
           DC_GpuSolver_AllocMatrix(&ctx->d_tmp, "hipMalloc(d_tmp)") &&
           DC_GpuSolver_AllocMatrix(&ctx->d_A, "hipMalloc(d_A)");
}

static int DC_GpuSolver_Eigen(double *d_A, int m, int maxn, double *W, const char *where)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;
    int             info;
    int             mout = 0;
    hipError_t     hip_status;

    if (m <= 0 || maxn <= 0 || maxn > m || d_A == NULL || W == NULL) {
        DC_AbortWithMessage("Invalid eigensolver arguments in DC_GpuSolver_Eigen.");
    }

    if (DC_GpuSolver_EigenDisabled() || !DC_GpuSolver_Init()) {
        return 0;
    }

    hip_status = hipStreamSynchronize(ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipStreamSynchronize(MAGMA eigensolver input)", hip_status);
        return 0;
    }

    info = openmx_magma_dsyevdx_gpu(m, maxn, d_A, W, &mout);
    if (info != 0) {
        DC_GpuSolver_DisableEigenPathInfo(where, info);
        return 0;
    }
    if (mout != maxn) {
        DC_GpuSolver_DisableEigenPathEigenpairs(where, (int64_t)mout, maxn);
        return 0;
    }

    return 1;
}

/* The host matrices are row-contiguous, so staging them row by row into
   the pinned buffer is a plain streaming copy; the transpose that
   column-major hipBLAS and MAGMA need is done on the device instead
   (DC_GpuSolver_DeviceTranspose), which is bitwise-exact data movement. */
static void DCCol_GpuSolver_PackRows(int n, double **src, double *dst)
{
    int j;

    for (j = 1; j <= n; j++) {
        memcpy(dst + (size_t)(j - 1) * (size_t)n, &src[j][1], sizeof(double) * (size_t)n);
    }
}

static void DCCol_GpuSolver_UnpackRows(int n_rows, int n_cols, const double *src, double **dst)
{
    int i;

    for (i = 1; i <= n_rows; i++) {
        memcpy(&dst[i][1], src + (size_t)(i - 1) * (size_t)n_cols, sizeof(double) * (size_t)n_cols);
    }
}

/* d_out (rows_out x cols_out, column-major, ld_out) = transpose of d_in
   (cols_out x rows_out, column-major, ld_in).  geam in its in-place-B mode
   (B == C, transb == N) with beta == 0 never reads the old contents of
   d_out (checked on gfx1200 with d_out filled with NaN). */
static int DC_GpuSolver_DeviceTranspose(int rows_out, int cols_out, const double *d_in, int ld_in, double *d_out,
                                        int ld_out, const char *where)
{
    double          alpha = 1.0;
    double          beta  = 0.0;
    hipblasStatus_t status;

    status = hipblasDgeam(DC_gpusolver_ctx.hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, rows_out, cols_out, &alpha, d_in,
                          ld_in, &beta, d_out, ld_out, d_out, ld_out);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        DC_GpuSolver_DisableGemmPath(where, "hipblasDgeam", status, rows_out, cols_out, 0);
        return 0;
    }

    return 1;
}

/* rows of a host matrix -> column-major device matrix d_out, through the
   pinned buffer and the device scratch d_scratch */
static int DCCol_GpuSolver_UploadMatrix(int n, double **src, double *d_scratch, double *d_out, const char *where)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;
    hipError_t      hip_status;

    DCCol_GpuSolver_PackRows(n, src, ctx->h_matrix);
    hip_status = hipMemcpyAsync(d_scratch, ctx->h_matrix, sizeof(double) * (size_t)n * (size_t)n,
                                hipMemcpyHostToDevice, ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(where, hip_status);
        return 0;
    }

    return DC_GpuSolver_DeviceTranspose(n, n, d_scratch, n, d_out, n, where);
}

static int DCCol_GpuSolver_LoadTransformedOverlap(int n, double **S_DC)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;
    hipError_t      hip_status;

    if (!DC_GpuSolver_EnsureMatrixCapacity(n) || !DC_GpuSolver_AllocMatrix(&ctx->d_S, "hipMalloc(d_S)") ||
        !DC_GpuSolver_AllocMatrix(&ctx->d_tmp, "hipMalloc(d_tmp)")) {
        return 0;
    }

    if (!DCCol_GpuSolver_UploadMatrix(n, S_DC, ctx->d_tmp, ctx->d_S, "transformed overlap staging")) {
        return 0;
    }
    /* h_matrix is restaged by the Hamiltonian solve */
    hip_status = hipStreamSynchronize(ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipStreamSynchronize(transformed overlap)", hip_status);
        return 0;
    }

    ctx->loaded_s_dim = n;
    return 1;
}

/* Refresh Mc_AN's S-cache slot from the overlap just staged in ctx->d_S.
   *d_S_use is the slot for the solve to read in place, or NULL when the
   atom has no slot (the solve then uses, and reloads, ctx->d_S). */
static int DCCol_SCache_Store(int Mc_AN, int n, const double **d_S_use)
{
    DCGpuSolverCtx *ctx  = &DC_gpusolver_ctx;
    double *        slot = DCCol_SCache_Ptr(Mc_AN, n);
    hipError_t      hip_status;

    *d_S_use = NULL;
    if (slot == NULL) {
        return 1;
    }

    hip_status = hipMemcpyAsync(slot, ctx->d_S, sizeof(double) * (size_t)n * (size_t)n, hipMemcpyDeviceToDevice,
                                ctx->stream);
    if (hip_status != hipSuccess) {
        DC_scache.valid[Mc_AN] = 0;
        DC_GpuSolver_DisableGemmPathHip("hipMemcpyAsync(S-cache refresh)", hip_status);
        return 0;
    }

    DC_scache.valid[Mc_AN] = 1;
    *d_S_use = slot;
    return 1;
}

static int DCCol_GpuSolver_DiagonalizeOverlap(int n, double **S_DC, double *ko)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;
    hipError_t      hip_status;

    if (!DC_GpuSolver_EnsureMatrixCapacity(n) || !DC_GpuSolver_AllocMatrix(&ctx->d_S, "hipMalloc(d_S)") ||
        !DC_GpuSolver_AllocMatrix(&ctx->d_tmp, "hipMalloc(d_tmp)")) {
        return 0;
    }
    ctx->loaded_s_dim = 0;

    if (!DCCol_GpuSolver_UploadMatrix(n, S_DC, ctx->d_tmp, ctx->d_S, "overlap staging")) {
        return 0;
    }
    /* MAGMA gets the room of the staging scratch */
    hip_status = hipStreamSynchronize(ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipStreamSynchronize(overlap staging)", hip_status);
        return 0;
    }
    DC_GpuSolver_ReleaseTransformScratch();

    if (!DC_GpuSolver_Eigen(ctx->d_S, n, n, ko + 1, "overlap diagonalization")) {
        return 0;
    }

    if (!DC_GpuSolver_AllocMatrix(&ctx->d_tmp, "hipMalloc(d_tmp)") ||
        !DC_GpuSolver_DeviceTranspose(n, n, ctx->d_S, n, ctx->d_tmp, n, "overlap eigenvector readback")) {
        return 0;
    }
    hip_status = hipMemcpyAsync(ctx->h_matrix, ctx->d_tmp, sizeof(double) * (size_t)n * (size_t)n,
                                hipMemcpyDeviceToHost, ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipMemcpyAsync(overlap eigenvectors)", hip_status);
        return 0;
    }
    hip_status = hipStreamSynchronize(ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipStreamSynchronize(overlap diagonalization)", hip_status);
        return 0;
    }

    DCCol_GpuSolver_UnpackRows(n, n, ctx->h_matrix, S_DC);
    return 1;
}

static int DCCol_CPU_SolveHamiltonian(int n, int p_min, double **S_DC, double **H_DC_spin, double *ko, double **C)
{
    int    i1, j1, l, j1s;
    int    num1 = n - (p_min - 1);
    double tmp1, tmp2;
    double sum, sum1, sum2, sum3, sum4;

    for (i1 = 1; i1 <= n; i1++) {
        for (j1 = i1 + 1; j1 <= n; j1++) {
            tmp1         = S_DC[i1][j1];
            tmp2         = S_DC[j1][i1];
            S_DC[i1][j1] = tmp2;
            S_DC[j1][i1] = tmp1;
        }
    }

    for (j1 = 1; j1 <= n - 3; j1 = j1 + 4) {
        for (i1 = 1; i1 <= n; i1++) {
            sum1 = 0.0;
            sum2 = 0.0;
            sum3 = 0.0;
            sum4 = 0.0;
            for (l = 1; l <= n; l++) {
                sum1 += H_DC_spin[i1][l] * S_DC[j1][l];
                sum2 += H_DC_spin[i1][l] * S_DC[j1 + 1][l];
                sum3 += H_DC_spin[i1][l] * S_DC[j1 + 2][l];
                sum4 += H_DC_spin[i1][l] * S_DC[j1 + 3][l];
            }
            C[j1][i1]     = sum1;
            C[j1 + 1][i1] = sum2;
            C[j1 + 2][i1] = sum3;
            C[j1 + 3][i1] = sum4;
        }
    }

    j1s = n - n % 4 + 1;
    for (j1 = j1s; j1 <= n; j1++) {
        for (i1 = 1; i1 <= n; i1++) {
            sum = 0.0;
            for (l = 1; l <= n; l++) {
                sum += H_DC_spin[i1][l] * S_DC[j1][l];
            }
            C[j1][i1] = sum;
        }
    }

    for (j1 = 1; j1 <= n - 3; j1 = j1 + 4) {
        for (i1 = 1; i1 <= n; i1++) {
            sum1 = 0.0;
            sum2 = 0.0;
            sum3 = 0.0;
            sum4 = 0.0;
            for (l = 1; l <= n; l++) {
                sum1 += S_DC[i1][l] * C[j1][l];
                sum2 += S_DC[i1][l] * C[j1 + 1][l];
                sum3 += S_DC[i1][l] * C[j1 + 2][l];
                sum4 += S_DC[i1][l] * C[j1 + 3][l];
            }
            H_DC_spin[j1][i1]     = sum1;
            H_DC_spin[j1 + 1][i1] = sum2;
            H_DC_spin[j1 + 2][i1] = sum3;
            H_DC_spin[j1 + 3][i1] = sum4;
        }
    }

    j1s = n - n % 4 + 1;
    for (j1 = j1s; j1 <= n; j1++) {
        for (i1 = 1; i1 <= n; i1++) {
            sum1 = 0.0;
            for (l = 1; l <= n; l++) {
                sum1 += S_DC[i1][l] * C[j1][l];
            }
            H_DC_spin[j1][i1] = sum1;
        }
    }

    for (i1 = p_min; i1 <= n; i1++) {
        for (j1 = p_min; j1 <= n; j1++) {
            C[j1 - (p_min - 1)][i1 - (p_min - 1)] = H_DC_spin[i1][j1];
        }
    }

    DC_Eigen_lapack_cpu(C, ko, num1, num1);

    for (i1 = 1; i1 <= n; i1++) {
        for (j1 = 1; j1 <= num1; j1++) {
            H_DC_spin[j1][i1] = C[i1][j1];
        }
    }

    for (i1 = 1; i1 <= n; i1++) {
        for (j1 = i1 + 1; j1 <= n; j1++) {
            tmp1         = S_DC[i1][j1];
            tmp2         = S_DC[j1][i1];
            S_DC[i1][j1] = tmp2;
            S_DC[j1][i1] = tmp1;
        }
    }

    for (j1 = 1; j1 <= num1; j1++) {
        for (l = n; p_min <= l; l--) {
            H_DC_spin[j1][l] = H_DC_spin[j1][l - (p_min - 1)];
        }
    }

    for (j1 = 1; j1 <= n - 3; j1 = j1 + 4) {
        for (i1 = 1; i1 <= n; i1++) {
            sum1 = 0.0;
            sum2 = 0.0;
            sum3 = 0.0;
            sum4 = 0.0;
            for (l = p_min; l <= n; l++) {
                sum1 += S_DC[i1][l] * H_DC_spin[j1][l];
                sum2 += S_DC[i1][l] * H_DC_spin[j1 + 1][l];
                sum3 += S_DC[i1][l] * H_DC_spin[j1 + 2][l];
                sum4 += S_DC[i1][l] * H_DC_spin[j1 + 3][l];
            }
            C[i1][j1]     = sum1;
            C[i1][j1 + 1] = sum2;
            C[i1][j1 + 2] = sum3;
            C[i1][j1 + 3] = sum4;
        }
    }

    j1s = n - n % 4 + 1;
    for (j1 = j1s; j1 <= n; j1++) {
        for (i1 = 1; i1 <= n; i1++) {
            sum = 0.0;
            for (l = p_min; l <= n; l++) {
                sum += S_DC[i1][l] * H_DC_spin[j1][l];
            }
            C[i1][j1] = sum;
        }
    }

    return num1;
}

/* d_S_use == NULL means "use the overlap staged in ctx->d_S by
   LoadTransformedOverlap"; that copy makes room for MAGMA and is reloaded
   from the host S_DC afterwards.  Otherwise d_S_use is the resident copy in
   the S-cache arena, read in place, and S_DC is not used. */
static int DCCol_GpuSolver_SolveHamiltonian(int n, int p_min, const double *d_S_use, double **S_DC,
                                           double **H_DC_spin, double *ko, double **C)
{
    DCGpuSolverCtx *ctx = &DC_gpusolver_ctx;
    double         alpha = 1.0;
    double         beta  = 0.0;
    int            num1  = n - (p_min - 1);
    int            reload_s = (d_S_use == NULL);
    hipError_t    hip_status;

    if (reload_s && (ctx->loaded_s_dim != n || ctx->d_S == NULL)) {
        DC_AbortWithMessage("Transformed overlap is not loaded in DCCol_GpuSolver_SolveHamiltonian.");
    }
    if (num1 <= 0 || num1 > n) {
        DC_AbortWithMessage("Invalid active subspace size in DCCol_GpuSolver_SolveHamiltonian.");
    }

    if (!DC_GpuSolver_EnsureGemmCapacity(n)) {
        return 0;
    }
    if (reload_s) {
        d_S_use = ctx->d_S;
    }

    if (!DCCol_GpuSolver_UploadMatrix(n, H_DC_spin, ctx->d_tmp, ctx->d_H, "Hamiltonian staging")) {
        DC_GpuSolver_ReleaseGemmScratch();
        return 0;
    }

    if (!DC_GpuSolver_TryGpuDgemm(ctx->hipblas, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha, ctx->d_H, n, d_S_use,
                                 n, &beta, ctx->d_tmp, n, "H*S")) {
        DC_GpuSolver_ReleaseGemmScratch();
        return 0;
    }
    if (!DC_GpuSolver_TryGpuDgemm(ctx->hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, n, &alpha, d_S_use, n,
                                 ctx->d_tmp, n, &beta, ctx->d_H, n, "S^T*H*S")) {
        DC_GpuSolver_ReleaseGemmScratch();
        return 0;
    }

    hip_status = hipMemcpy2DAsync(ctx->d_A, sizeof(double) * (size_t)num1,
                                    ctx->d_H + (size_t)(p_min - 1) * (size_t)n + (size_t)(p_min - 1),
                                    sizeof(double) * (size_t)n, sizeof(double) * (size_t)num1, (size_t)num1,
                                    hipMemcpyDeviceToDevice, ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipMemcpy2DAsync(active Hamiltonian)", hip_status);
        DC_GpuSolver_ReleaseGemmScratch();
        return 0;
    }
    hip_status = hipStreamSynchronize(ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipStreamSynchronize(active Hamiltonian)", hip_status);
        DC_GpuSolver_ReleaseGemmScratch();
        return 0;
    }
    /* MAGMA gets the room of the transform scratch and of the staged
       overlap (a resident S-cache copy stays) */
    DC_GpuSolver_ReleaseTransformScratch();
    DC_GpuSolver_ReleaseOverlapBuffer();

    if (!DC_GpuSolver_Eigen(ctx->d_A, num1, num1, ko + 1, "Hamiltonian diagonalization")) {
        DC_GpuSolver_ReleaseGemmScratch();
        return 0;
    }

    if (!DC_GpuSolver_AllocMatrix(&ctx->d_C, "hipMalloc(d_C)")) {
        return 0;
    }
    if (reload_s) {
        /* d_C is the staging scratch until the back-transform writes it */
        if (!DC_GpuSolver_AllocMatrix(&ctx->d_S, "hipMalloc(d_S reload)")) {
            return 0;
        }
        if (!DCCol_GpuSolver_UploadMatrix(n, S_DC, ctx->d_C, ctx->d_S, "overlap reload")) {
            DC_GpuSolver_ReleaseGemmScratch();
            return 0;
        }
        ctx->loaded_s_dim = n;
        d_S_use           = ctx->d_S;
    }

    if (!DC_GpuSolver_TryGpuDgemm(ctx->hipblas, HIPBLAS_OP_N, HIPBLAS_OP_N, n, num1, num1, &alpha,
                                 d_S_use + (size_t)(p_min - 1) * (size_t)n, n, ctx->d_A, num1, &beta,
                                 ctx->d_C, n, "S*C")) {
        DC_GpuSolver_ReleaseGemmScratch();
        return 0;
    }

    /* row-major C (n x num1) for the host rows; d_A is free again */
    if (!DC_GpuSolver_DeviceTranspose(num1, n, ctx->d_C, n, ctx->d_A, num1, "eigenvector readback")) {
        DC_GpuSolver_ReleaseGemmScratch();
        return 0;
    }
    hip_status = hipMemcpyAsync(ctx->h_matrix, ctx->d_A, sizeof(double) * (size_t)n * (size_t)num1,
                                  hipMemcpyDeviceToHost, ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipMemcpyAsync(C)", hip_status);
        DC_GpuSolver_ReleaseGemmScratch();
        return 0;
    }

    hip_status = hipStreamSynchronize(ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipStreamSynchronize(C)", hip_status);
        DC_GpuSolver_ReleaseGemmScratch();
        return 0;
    }

    DCCol_GpuSolver_UnpackRows(n, num1, ctx->h_matrix, C);

    DC_GpuSolver_ReleaseGemmScratch();
    return 1;
}

/* Host copy of the transformed overlap, needed only by the CPU solve; the
   GPU path with an S-cache hit skips it until a fallback actually happens. */
static void DCCol_FillSDCFromS12(int Mc_AN, int n, double **S_DC)
{
    int i1, j1;

    for (i1 = 1; i1 <= n; i1++) {
        for (j1 = 1; j1 <= n; j1++) {
            S_DC[i1][j1] = S12[Mc_AN][i1][j1];
        }
    }
}

/****************************************************
   GPU solve of the non-collinear clusters (DC_NonCol)
****************************************************/

/* EigenBand_lapack on the CPU at every size: under scf.eigen.lib gpusolver
   it would send n >= GPU_CPU_SWITCH_NUM to the GPU solver wrappers, which
   the CPU path of DC_NonCol (also the fallback of its GPU solve) must not
   do. */
static void DC_EigenBand_lapack_cpu(dcomplex **a, double *ko, int n)
{
    int saved_scf_eigen_lib_flag = scf_eigen_lib_flag;

    scf_eigen_lib_flag = ELPA2;
    EigenBand_lapack(a, ko, n, n, 1);
    scf_eigen_lib_flag = saved_scf_eigen_lib_flag;
}

/* Device memory a GPU eigensolve leaves free (the NVIDIA version's cuSOLVER
   workspace preflight).  The collinear solves use MAGMA, which sizes and
   checks its own workspace; the non-collinear admission keeps this much
   of the shared device free. */
static size_t DC_GpuSolver_EigenReserveBytes(void)
{
    unsigned reserve_mib;

    reserve_mib = DC_EnvU32("GPUSOLVER_MIN_FREE_AFTER_MB", 1536u);
    reserve_mib = DC_EnvU32("OPENMX_GPUSOLVER_MIN_FREE_AFTER_MB", reserve_mib);

    return (size_t)reserve_mib * (size_t)1024 * (size_t)1024;
}

/* Buffers of the non-collinear solve.  The device matrices are sized once
   per DC_NonCol call for the largest admitted cluster and returned at its
   end; the stream and hipBLAS handle are those of DC_gpusolver_ctx, and the
   pinned staging buffer stays until the run boundary. */
typedef struct {
    int               num;     /* capacity, as the NUM of a cluster */
    int               off;     /* no device memory: the rest of this call runs on the CPU */
    int               reports; /* per-atom fallback messages printed in this run */
    hipDoubleComplex *d_A;     /* 2NUM x 2NUM */
    hipDoubleComplex *d_Y;     /* 2NUM x 2NUM */
    double *          d_T;     /* NUM x NUM */
    void *            h_stage; /* rows of T, then the 2NUM x 2NUM complex rows of H and C */
    size_t            h_stage_bytes;
} DCNonColGpuCtx;

static DCNonColGpuCtx DC_noncol_gpu = {0};

static void DCNonCol_GpuRelease(int keep_stage)
{
    DCNonColGpuCtx *nc = &DC_noncol_gpu;

    if (nc->d_A != NULL)
        wait_hipfunc(hipFree(nc->d_A));
    if (nc->d_Y != NULL)
        wait_hipfunc(hipFree(nc->d_Y));
    if (nc->d_T != NULL)
        wait_hipfunc(hipFree(nc->d_T));
    nc->d_A = NULL;
    nc->d_Y = NULL;
    nc->d_T = NULL;
    nc->num = 0;

    if (!keep_stage) {
        if (nc->h_stage != NULL)
            wait_hipfunc(hipHostFree(nc->h_stage));
        nc->h_stage       = NULL;
        nc->h_stage_bytes = 0;
        nc->reports       = 0;
    }
}

/* byte offset of the complex rows in the staging buffer */
static size_t DCNonCol_StageOffset(int num)
{
    size_t bytes = DC_CheckedArrayBytes(DC_CheckedMulCount((size_t)num, (size_t)num, "DC NonCol GPU overlap"),
                                        sizeof(double), "DC NonCol GPU overlap");

    return (bytes + 255) / 256 * 256;
}

static size_t DCNonCol_ComplexBytes(int num)
{
    return DC_CheckedArrayBytes(DC_CheckedMulCount(2 * (size_t)num, 2 * (size_t)num, "DC NonCol GPU matrix"),
                                sizeof(hipDoubleComplex), "DC NonCol GPU matrix");
}

/* Device bytes of one rank for clusters up to num: d_A, d_Y and d_T, the
   eigensolver workspace (MAGMA's is about one matrix, rocSOLVER's zheevd is
   counted as two), the hipBLAS (or GEMMul8) workspace and some slack. */
static size_t DCNonCol_GpuNeedBytes(int num)
{
    size_t zmat = DCNonCol_ComplexBytes(num);
    size_t gemm = DC_GpuSolver_HipblasReserveBytes();
    size_t need;

    if (!DC_gpusolver_gemmul8_disabled) {
        size_t gemmul8 = openmx_gemmul8DWorkspaceSize(4 * num, num, num);

        if (gemm < gemmul8)
            gemm = gemmul8;
    }

    need = DC_CheckedMulCount(zmat, openmx_gpu_eigensolver_use_hipsolver() ? 4u : 3u, "DC NonCol GPU memory");
    need += DCNonCol_StageOffset(num) + gemm + (size_t)64 * 1024 * 1024;
    return need;
}

/* Device-memory admission of the non-collinear GPU solve for one DC_NonCol
   call, agreed by the ranks sharing a GPU.  Collective over mpi_comm_level1:
   every rank calls it whenever scf.eigen.lib is gpusolver (that flag is the
   same on all ranks).  A rank needs DCNonCol_GpuNeedBytes() for its largest
   cluster with 2*NUM >= OPENMX_DC_NONCOL_GPU_THRESHOLD; the sum over the
   ranks of a GPU must fit into the smallest free memory they see minus
   GPUSOLVER_MIN_FREE_AFTER_MB.  Returns the admitted capacity (that NUM),
   or 0 when this call solves every cluster on the CPU. */
static int DCNonCol_GpuAdmit(int matomnum, const int *Msize)
{
    static int         warned = 0;
    MPI_Comm           node_comm = MPI_COMM_NULL, gpu_comm = MPI_COMM_NULL;
    unsigned long long my_need = 0, my_free, group_need = 0, group_free = 0;
    unsigned long long reserve = (unsigned long long)DC_GpuSolver_EigenReserveBytes();
    size_t             free_bytes = 0, total_bytes = 0;
    int                max_num = 0, device = 0, node_rank = 0, gpu_rank = 0, gpu_ranks = 1, Mc_AN, fits;

    DC_noncol_gpu.off = 0;

    if (!DC_GpuSolver_GemmDisabled() && !DC_GpuSolver_EigenDisabled()) {
        for (Mc_AN = 1; Mc_AN <= matomnum; Mc_AN++) {
            if (DC_NonCol_GPU_Threshold() <= 2 * Msize[Mc_AN] && max_num < Msize[Mc_AN]) {
                max_num = Msize[Mc_AN];
            }
        }
    }
    if (0 < max_num) {
        my_need = (unsigned long long)DCNonCol_GpuNeedBytes(max_num);
    }
    if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        (void)hipGetLastError();
        free_bytes = 0;
    }
    my_free = (unsigned long long)free_bytes;
    if (hipGetDevice(&device) != hipSuccess) {
        (void)hipGetLastError();
        device = 0;
    }

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_split(node_comm, device, node_rank, &gpu_comm);
    MPI_Comm_rank(gpu_comm, &gpu_rank);
    MPI_Comm_size(gpu_comm, &gpu_ranks);
    MPI_Allreduce(&my_need, &group_need, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, gpu_comm);
    MPI_Allreduce(&my_free, &group_free, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, gpu_comm);
    MPI_Comm_free(&gpu_comm);
    MPI_Comm_free(&node_comm);

    fits = (reserve <= group_free && group_need <= group_free - reserve);
    if (!fits && 0 < group_need && !warned && gpu_rank == 0 && 0 < level_stdout) {
        printf("<DC> GPU device %d: %d rank(s) would need %.3f GiB for the non-collinear solves "
               "(free %.3f GiB, reserve %.3f GiB); solving them on the CPU instead.\n",
               device, gpu_ranks, (double)group_need / (1024.0 * 1024.0 * 1024.0),
               (double)group_free / (1024.0 * 1024.0 * 1024.0), (double)reserve / (1024.0 * 1024.0 * 1024.0));
        fflush(stdout);
        warned = 1;
    }

    return fits ? max_num : 0;
}

/* Buffers for clusters up to num; a failure leaves the rest of this call
   on the CPU (the next call is admitted afresh). */
static int DCNonCol_GpuEnsure(int num)
{
    DCNonColGpuCtx *nc      = &DC_noncol_gpu;
    size_t          zbytes  = DCNonCol_ComplexBytes(num);
    size_t          dbytes  = DC_CheckedArrayBytes(DC_CheckedMulCount((size_t)num, (size_t)num, "DC NonCol GPU overlap"),
                                                   sizeof(double), "DC NonCol GPU overlap");
    size_t          sbytes  = DCNonCol_StageOffset(num) + zbytes;
    const char *    what    = NULL;
    hipError_t      status  = hipSuccess;
    int             rank    = -1;

    if (num <= nc->num && nc->d_A != NULL && nc->h_stage != NULL) {
        return 1;
    }

    if (!DC_GpuSolver_Init()) {
        return 0;
    }
    DCNonCol_GpuRelease(1);

    if ((status = hipMalloc((void **)&nc->d_A, zbytes)) != hipSuccess) {
        nc->d_A = NULL;
        what    = "hipMalloc(non-collinear H)";
    } else if ((status = hipMalloc((void **)&nc->d_Y, zbytes)) != hipSuccess) {
        nc->d_Y = NULL;
        what    = "hipMalloc(non-collinear work)";
    } else if ((status = hipMalloc((void **)&nc->d_T, dbytes)) != hipSuccess) {
        nc->d_T = NULL;
        what    = "hipMalloc(non-collinear overlap)";
    } else if (nc->h_stage_bytes < sbytes) {
        if (nc->h_stage != NULL)
            wait_hipfunc(hipHostFree(nc->h_stage));
        nc->h_stage       = NULL;
        nc->h_stage_bytes = 0;
        if ((status = hipHostMalloc(&nc->h_stage, sbytes, 0)) != hipSuccess) {
            nc->h_stage = NULL;
            what        = "hipHostMalloc(non-collinear staging)";
        } else {
            nc->h_stage_bytes = sbytes;
        }
    }

    if (what != NULL) {
        (void)hipGetLastError();
        DCNonCol_GpuRelease(1);
        nc->off = 1;
        if (nc->reports < 8) {
            nc->reports++;
            MPI_Comm_rank(mpi_comm_level1, &rank);
            fprintf(stderr,
                    "<DC> rank %d: %s failed for a %d x %d non-collinear cluster: %s (%d). "
                    "Solving the rest of this step on the CPU.\n",
                    rank, what, 2 * num, 2 * num, hipGetErrorString(status), (int)status);
            fflush(stderr);
        }
        return 0;
    }

    nc->num = num;
    return 1;
}

/* a failed solve of one cluster; the first few of a run are reported */
static void DCNonCol_GpuReport(const char *what, int code, int num)
{
    int rank = -1;

    if (8 <= DC_noncol_gpu.reports) {
        return;
    }
    DC_noncol_gpu.reports++;

    MPI_Comm_rank(mpi_comm_level1, &rank);
    fprintf(stderr,
            "<DC> rank %d: GPU solve of a %d x %d non-collinear cluster failed (%s: %d); "
            "solving this atom on the CPU.\n",
            rank, 2 * num, 2 * num, what, code);
    fflush(stderr);
}

/* d_out (dim x dim) = plain transpose of d_in, by the in-place-B mode of
   geam with beta == 0 (d_out's old contents are not read) */
static int DCNonCol_GpuTranspose(int dim, const hipDoubleComplex *d_in, hipDoubleComplex *d_out, const char *where)
{
    const hipDoubleComplex one  = make_hipDoubleComplex(1.0, 0.0);
    const hipDoubleComplex zero = make_hipDoubleComplex(0.0, 0.0);
    hipblasStatus_t        status;

    status = hipblasZgeam(DC_gpusolver_ctx.hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, dim, dim, &one, d_in, dim, &zero,
                          d_out, dim, d_out, dim);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        DC_GpuSolver_DisableGemmPath(where, "hipblasZgeam", status, dim, dim, 0);
        return 0;
    }

    return 1;
}

/* d_out = d_in * diag(R, R) for the complex dim x dim matrix d_in, column
   block by column block.  d_T holds the rows of T (column-major T^T), so
   op_t == HIPBLAS_OP_T gives R = T and HIPBLAS_OP_N gives R = T^T.  A
   complex column-major matrix times a real one on the right is a real GEMM
   on its interleaved (re, im) storage viewed as a 2*dim-row real matrix:
   half the flops of a complex GEMM against a zero-imaginary T. */
static int DCNonCol_GpuTimesT(int num, const hipDoubleComplex *d_in, hipblasOperation_t op_t,
                              hipDoubleComplex *d_out, const char *where)
{
    const double one  = 1.0;
    const double zero = 0.0;
    const int    dim  = 2 * num;
    int          k;

    for (k = 0; k < 2; k++) {
        if (!DC_GpuSolver_TryGpuDgemm(DC_gpusolver_ctx.hipblas, HIPBLAS_OP_N, op_t, 2 * dim, num, num, &one,
                                     (const double *)(d_in + (size_t)k * (size_t)num * (size_t)dim), 2 * dim,
                                     DC_noncol_gpu.d_T, num, &zero,
                                     (double *)(d_out + (size_t)k * (size_t)num * (size_t)dim), 2 * dim, where)) {
            return 0;
        }
    }

    return 1;
}

/* GPU solve of one non-collinear cluster with P_min == 1 (all NUM overlap
   eigenvectors kept): fills ko[1..2NUM] and C[1..2NUM][1..2NUM] like the
   CPU path of DC_NonCol and returns 1, or returns 0 for the CPU path.
   With T = S_DC (NUM x NUM, real, U * lambda^-1/2) and B = diag(T, T), the
   CPU path diagonalizes X^T H X with X = B P^T, P interleaving the two spin
   blocks, and back-transforms C = X V.  The same spectrum comes from
   H' = B^T H B, whose eigenvectors are W = P^T V, and C = B W:
     d_A <- rows of H        (column-major: H^T)
     d_Y <- H^T B
     d_A <- (H^T B)^T = B^T H
     d_Y <- B^T H B = H'     -> eigenvalues ko, eigenvectors W in place
     d_A <- W^T
     d_Y <- W^T B^T = C^T    (column-major: C row by row, as the host wants)
   S_DC and H_DC are only read, so the CPU path can redo the atom after any
   failure.  The context is shared, so DC_NonCol's atom loop stays serial. */
static int DCNonCol_GpuSolve(int num, int capacity, double **S_DC, dcomplex **H_DC, dcomplex **C, double *ko)
{
    DCGpuSolverCtx *  ctx    = &DC_gpusolver_ctx;
    DCNonColGpuCtx *  nc     = &DC_noncol_gpu;
    const int         dim    = 2 * num;
    const size_t      zcount = (size_t)dim * (size_t)dim;
    double *          h_T;
    hipDoubleComplex *h_Z;
    const double *    h_check;
    hipError_t        status;
    size_t            idx;
    int               i, info, mout = 0;

    if (omp_in_parallel() || nc->off || num <= 0) {
        return 0;
    }
    if (!DCNonCol_GpuEnsure(num < capacity ? capacity : num)) {
        return 0;
    }

    h_T = (double *)nc->h_stage;
    h_Z = (hipDoubleComplex *)((char *)nc->h_stage + DCNonCol_StageOffset(num));

    for (i = 0; i < num; i++) {
        memcpy(h_T + (size_t)i * (size_t)num, &S_DC[i + 1][1], sizeof(double) * (size_t)num);
    }
    status = hipMemcpyAsync(nc->d_T, h_T, sizeof(double) * (size_t)num * (size_t)num, hipMemcpyHostToDevice,
                            ctx->stream);
    if (status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipMemcpyAsync(non-collinear overlap)", status);
        return 0;
    }
    for (i = 0; i < dim; i++) {
        memcpy(h_Z + (size_t)i * (size_t)dim, &H_DC[i + 1][1], sizeof(dcomplex) * (size_t)dim);
    }
    status = hipMemcpyAsync(nc->d_A, h_Z, sizeof(hipDoubleComplex) * zcount, hipMemcpyHostToDevice, ctx->stream);
    if (status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipMemcpyAsync(non-collinear Hamiltonian)", status);
        return 0;
    }

    if (!DCNonCol_GpuTimesT(num, nc->d_A, HIPBLAS_OP_T, nc->d_Y, "non-collinear H*T") ||
        !DCNonCol_GpuTranspose(dim, nc->d_Y, nc->d_A, "non-collinear T^T*H") ||
        !DCNonCol_GpuTimesT(num, nc->d_A, HIPBLAS_OP_T, nc->d_Y, "non-collinear T^T*H*T")) {
        return 0;
    }

    /* MAGMA and rocSOLVER run outside ctx->stream */
    status = hipStreamSynchronize(ctx->stream);
    if (status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipStreamSynchronize(non-collinear eigensolver input)", status);
        return 0;
    }
    if (openmx_gpu_eigensolver_use_hipsolver()) {
        info = openmx_hipsolver_zheevd_gpu(dim, dim, nc->d_Y, ko + 1, &mout);
    } else {
        info = openmx_magma_zheevdx_gpu(dim, dim, nc->d_Y, ko + 1, &mout);
    }
    if (info != 0) {
        /* a failed solve can return before its closing device
           synchronization */
        (void)hipDeviceSynchronize();
        (void)hipGetLastError();
        if (info == DC_MAGMA_ERR_DEVICE_ALLOC || info == DC_MAGMA_ERR_HOST_ALLOC) {
            /* no memory for the eigensolver workspace: the rest of this
               call runs on the CPU */
            nc->off = 1;
            DCNonCol_GpuReport("eigensolver workspace allocation", info, num);
        } else {
            DCNonCol_GpuReport("eigensolver info", info, num);
        }
        return 0;
    }
    if (mout != dim) {
        DCNonCol_GpuReport("eigenpairs returned", mout, num);
        return 0;
    }
    for (i = 1; i <= dim; i++) {
        if (!isfinite(ko[i])) {
            DCNonCol_GpuReport("non-finite eigenvalue", i, num);
            return 0;
        }
    }

    if (!DCNonCol_GpuTranspose(dim, nc->d_Y, nc->d_A, "non-collinear eigenvector transpose") ||
        !DCNonCol_GpuTimesT(num, nc->d_A, HIPBLAS_OP_N, nc->d_Y, "non-collinear W^T*T^T")) {
        return 0;
    }
    status = hipMemcpyAsync(h_Z, nc->d_Y, sizeof(hipDoubleComplex) * zcount, hipMemcpyDeviceToHost, ctx->stream);
    if (status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipMemcpyAsync(non-collinear eigenvectors)", status);
        return 0;
    }
    status = hipStreamSynchronize(ctx->stream);
    if (status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip("hipStreamSynchronize(non-collinear eigenvectors)", status);
        return 0;
    }

    h_check = (const double *)h_Z;
    for (idx = 0; idx < 2 * zcount; idx++) {
        if (!isfinite(h_check[idx])) {
            DCNonCol_GpuReport("non-finite eigenvector element", (int)(idx / 2 / (size_t)dim) + 1, num);
            return 0;
        }
    }
    for (i = 0; i < dim; i++) {
        memcpy(&C[i + 1][1], h_Z + (size_t)i * (size_t)dim, sizeof(dcomplex) * (size_t)dim);
    }

    return 1;
}

double Divide_Conquer(char * mode, int SCF_iter, double ***** Hks, double ***** ImNL, double **** OLP0,
                      double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2])
{
    double time0;

    /****************************************************
           collinear without spin-orbit coupling
    ****************************************************/

    if ((SpinP_switch == 0 || SpinP_switch == 1) && SO_switch == 0) {
        time0 = DC_Col(mode, SCF_iter, Hks, OLP0, CDM, EDM, Eele0, Eele1);
    }

    /****************************************************
     non-collinear with and without spin-orbit coupling
    ****************************************************/

    else if (SpinP_switch == 3) {
        time0 = DC_NonCol(mode, SCF_iter, Hks, ImNL, OLP0, CDM, EDM, Eele0, Eele1);
    }

    return time0;
}

#pragma optimization_level 2
static double DC_Col(char * mode, int SCF_iter, double ***** Hks, double **** OLP0, double ***** CDM, double ***** EDM,
                     double Eele0[2], double Eele1[2])
{
    static int    firsttime = 1;
    int           Mc_AN, Gc_AN, i, Gi, wan, wanA, wanB, Anum;
    int           size1, size2, num, NUM, NUM1, n2, Cwan, Hwan;
    int           ih, ig, ian, j, kl, jg, jan, Bnum, m, n, spin;
    int           l, i1, j1, P_min, m_size;
    int           po, loopN, tno1, tno2, h_AN, Gh_AN;
    int           MA_AN, GA_AN, GB_AN, tnoA, tnoB, k;
    double        My_TZ, TZ, sum, FermiF, time0;
    double        tmp1, tmp2;
    double        My_Num_State, Num_State, x, Dnum;
    double        TStime, TEtime;
    double        My_Eele0[2], My_Eele1[2];
    double        max_x = 30.0;
    double        ChemP_MAX, ChemP_MIN, spin_degeneracy;
    double **     S_DC, ***H_DC, *ko, *M1;
    double **     C;
    double ***    EVal;
    double ****** Residues;
    double ***    PDOS_DC;
    int *         MP, *Msize;
    double *      tmp_array;
    double *      tmp_array2;
    int *         Snd_H_Size, *Rcv_H_Size;
    int *         Snd_S_Size, *Rcv_S_Size;
    int           numprocs, myid, ID, IDS, IDR, tag = 999;
    double        Stime_atom, Etime_atom;
    double        OLP_eigen_cut = Threshold_OLP_Eigen;
    double        stime, etime;
    double        time1, time2, time3, time4, time5, time6;

    double sum1, sum2, sum3, sum4;
    int    i1s, j1s;

    MPI_Status  stat;
    MPI_Request request;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    /* MPI */
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);
    int is_scf_mode = (strcasecmp(mode, "scf") == 0);

    // Set the device to be used by CUDA and OpenMP
    if (scf_eigen_lib_flag == GPUSOLVER) {
        // CUDA
        set_hip_default_device_from_local_rank();

        // OpenMP
        set_openmp_nvidia_device_from_local_rank();
    }

    dtime(&TStime);

    if (measure_time) {
        time1 = 0.0;
        time2 = 0.0;
        time3 = 0.0;
        time4 = 0.0;
        time5 = 0.0;
        time6 = 0.0;
    }

    /****************************************************
      allocation of arrays:

      int MP[List_YOUSO[2]];
      int Msize[Matomnum+1];
      double EVal[SpinP_switch+1][Matomnum+1][n2];
    ****************************************************/

    Snd_H_Size = (int *)malloc(sizeof(int) * numprocs);
    Rcv_H_Size = (int *)malloc(sizeof(int) * numprocs);
    Snd_S_Size = (int *)malloc(sizeof(int) * numprocs);
    Rcv_S_Size = (int *)malloc(sizeof(int) * numprocs);

    m_size = 0;
    Msize  = (int *)malloc(sizeof(int) * (Matomnum + 1));

    EVal = (double ***)malloc(sizeof(double **) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        EVal[spin] = (double **)malloc(sizeof(double *) * (Matomnum + 1));

        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN        = 0;
                FNAN[0]      = 1;
                SNAN[0]      = 0;
                n2           = 1;
                Msize[Mc_AN] = 1;
            } else {

                Gc_AN = M2G[Mc_AN];
                Anum  = 1;
                for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                    Gi   = natn[Gc_AN][i];
                    wanA = WhatSpecies[Gi];
                    Anum += Spe_Total_CNO[wanA];
                }
                NUM          = Anum - 1;
                Msize[Mc_AN] = NUM;
                n2           = NUM + 3;
            }

            m_size += n2;
            EVal[spin][Mc_AN] = (double *)malloc(sizeof(double) * n2);
        }
    }

    if (firsttime)
        PrintMemory("Divide_Conquer: EVal", sizeof(double) * m_size, NULL);

    /* needs the cluster sizes Msize[] of this call */
    if (scf_eigen_lib_flag == GPUSOLVER) {
        DCCol_SCache_Prepare(Matomnum, Msize, SCF_iter);
    }

    if (2 <= level_stdout) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            printf("<DC> myid=%i Mc_AN=%2d Gc_AN=%2d Msize=%3d\n", myid, Mc_AN, M2G[Mc_AN], Msize[Mc_AN]);
        }
    }

    /****************************************************
      allocation of arrays:

      double Residues[SpinP_switch+1]
                     [Matomnum+1]
                     [FNAN[Gc_AN]+1]
                     [Spe_Total_CNO[Gc_AN]]
                     [Spe_Total_CNO[Gh_AN]]
                     [NUM2]
       To reduce the memory size, the size of NUM2 is
       needed to be found in the loop.
    ****************************************************/

    m_size   = 0;
    Residues = (double ******)malloc(sizeof(double *****) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        Residues[spin] = (double *****)malloc(sizeof(double ****) * (Matomnum + 1));
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN   = 0;
                FNAN[0] = 1;
                tno1    = 1;
                n2      = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                wanA  = WhatSpecies[Gc_AN];
                tno1  = Spe_Total_CNO[wanA];
                n2    = Msize[Mc_AN] + 2;
            }

            Residues[spin][Mc_AN] = (double ****)malloc(sizeof(double ***) * (FNAN[Gc_AN] + 1));

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Mc_AN == 0) {
                    tno2 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                }

                Residues[spin][Mc_AN][h_AN] = (double ***)malloc(sizeof(double **) * tno1);
                for (i = 0; i < tno1; i++) {
                    Residues[spin][Mc_AN][h_AN][i] = (double **)malloc(sizeof(double *) * tno2);
                    for (j = 0; j < tno2; j++) {
                        Residues[spin][Mc_AN][h_AN][i][j] = (double *)malloc(sizeof(double) * n2);
                    }
                }

                m_size += tno1 * tno2 * n2;
            }
        }
    }

    if (firsttime)
        PrintMemory("Divide_Conquer: Residues", sizeof(double) * m_size, NULL);

    /****************************************************
      allocation of arrays:

      double PDOS[SpinP_switch+1]
                 [Matomnum+1]
                 [NUM]
    ****************************************************/

    m_size  = 0;
    PDOS_DC = (double ***)malloc(sizeof(double **) * (SpinP_switch + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        PDOS_DC[spin] = (double **)malloc(sizeof(double *) * (Matomnum + 1));
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0)
                n2 = 1;
            else
                n2 = Msize[Mc_AN] + 2;

            m_size += n2;
            PDOS_DC[spin][Mc_AN] = (double *)malloc(sizeof(double) * n2);
        }
    }

    if (firsttime)
        PrintMemory("Divide_Conquer: PDOS_DC", sizeof(double) * m_size, NULL);

    /****************************************************
     MPI

     Hks
    ****************************************************/

    /***********************************
               set data size
    ************************************/

    for (ID = 0; ID < numprocs; ID++) {

        IDS = (myid + ID) % numprocs;
        IDR = (myid - ID + numprocs) % numprocs;

        if (ID != 0) {
            tag = 999;

            /* find data size to send block data */
            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                size1 = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            size1 += tno1 * tno2;
                        }
                    }
                }

                Snd_H_Size[IDS] = size1;
                MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
            } else {
                Snd_H_Size[IDS] = 0;
            }

            /* receiving of size of data */

            if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                Rcv_H_Size[IDR] = size2;
            } else {
                Rcv_H_Size[IDR] = 0;
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
                MPI_Wait(&request, &stat);
        } else {
            Snd_H_Size[IDS] = 0;
            Rcv_H_Size[IDR] = 0;
        }
    }

    /***********************************
               data transfer
    ************************************/

    tag = 999;
    for (ID = 0; ID < numprocs; ID++) {

        IDS = (myid + ID) % numprocs;
        IDR = (myid - ID + numprocs) % numprocs;

        if (ID != 0) {

            /*****************************
                    sending of data
            *****************************/

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                size1 = Snd_H_Size[IDS];

                /* allocation of array */

                tmp_array = (double *)malloc(sizeof(double) * size1);

                /* multidimentional array to vector array */

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    tmp_array[num] = Hks[spin][Mc_AN][h_AN][i][j];
                                    num++;
                                }
                            }
                        }
                    }
                }

                MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
            }

            /*****************************
               receiving of block data
            *****************************/

            if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                size2 = Rcv_H_Size[IDR];

                /* allocation of array */
                tmp_array2 = (double *)malloc(sizeof(double) * size2);

                MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
                    for (n = 0; n < (F_Rcv_Num[IDR] + S_Rcv_Num[IDR]); n++) {
                        Mc_AN++;
                        Gc_AN = Rcv_GAN[IDR][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];

                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    Hks[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }
                }

                /* freeing of array */
                free(tmp_array2);
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                MPI_Wait(&request, &stat);
                free(tmp_array); /* freeing of array */
            }
        }
    }

    /****************************************************
     MPI

     OLP0
    ****************************************************/

    /***********************************
               set data size
    ************************************/

    if (SCF_iter <= 2) {

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {
                tag = 999;

                /* find data size to send block data */
                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    size1 = 0;
                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    size1++;
                                }
                            }
                        }
                    }

                    Snd_S_Size[IDS] = size1;
                    MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
                } else {
                    Snd_S_Size[IDS] = 0;
                }

                /* receiving of size of data */

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {
                    MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                    Rcv_S_Size[IDR] = size2;
                } else {
                    Rcv_S_Size[IDR] = 0;
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
                    MPI_Wait(&request, &stat);
            } else {
                Snd_S_Size[IDS] = 0;
                Rcv_S_Size[IDR] = 0;
            }
        }

        /***********************************
                   data transfer
        ************************************/

        tag = 999;
        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {

                /*****************************
                        sending of data
                *****************************/

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                    size1 = Snd_S_Size[IDS];

                    /* allocation of array */

                    tmp_array = (double *)malloc(sizeof(double) * size1);

                    /* multidimentional array to vector array */

                    num = 0;

                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    tmp_array[num] = OLP0[Mc_AN][h_AN][i][j];
                                    num++;
                                }
                            }
                        }
                    }

                    MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
                }

                /*****************************
                   receiving of block data
                *****************************/

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                    size2 = Rcv_S_Size[IDR];

                    /* allocation of array */
                    tmp_array2 = (double *)malloc(sizeof(double) * size2);

                    MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                    num   = 0;
                    Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
                    for (n = 0; n < (F_Rcv_Num[IDR] + S_Rcv_Num[IDR]); n++) {
                        Mc_AN++;
                        Gc_AN = Rcv_GAN[IDR][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];

                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    OLP0[Mc_AN][h_AN][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }

                    /* freeing of array */
                    free(tmp_array2);
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
                    free(tmp_array); /* freeing of array */
                }
            }
        }
    }

    /****************************************************
              find the total number of electrons
    ****************************************************/

    My_TZ = 0.0;
    for (i = 1; i <= Matomnum; i++) {
        Gc_AN = M2G[i];
        wan   = WhatSpecies[Gc_AN];
        My_TZ = My_TZ + Spe_Core_Charge[wan];
    }

    /* MPI, My_TZ */

    MPI_Allreduce(&My_TZ, &TZ, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    if (scf_eigen_lib_flag == GPUSOLVER) {
        // CUDA
        set_hip_default_device_from_local_rank();

        // OpenMP
        set_openmp_nvidia_device_from_local_rank();
    }

    /****************************************************
        Setting of Hamiltonian and overlap matrices

           MP indicates the starting position of
                atom i in arraies H and S
    ****************************************************/

    int mysize = Matomnum > 0 ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &mysize, 1, MPI_INT, MPI_SUM, mpi_comm_level1);

    // #pragma omp parallel shared(OLP_eigen_cut, List_YOUSO, Etime_atom, time_per_atom, time3, Residues, EVal, time2, time1, \
    //                                 S12, level_stdout, SpinP_switch, Hks, OLP0, SCF_iter, RMI1, S_G2M, Spe_Total_CNO,      \
    //                                 natn, FNAN, SNAN, WhatSpecies, M2G, Matomnum)                                          \
    //     private(OMPID, Nthrds, Nprocs, Mc_AN, Stime_atom, Gc_AN, wan, Anum, i, j, MP, Gi, wanA, NUM, NUM1, n2, spin, S_DC, \
    //                 H_DC, ko, M1, C, ig, ian, ih, kl, jg, jan, Bnum, m, n, stime, P_min, l, i1, j1, etime, tmp1, tmp2,     \
    //                 sum1, sum2, sum3, sum4, j1s, sum, tno1, h_AN, Gh_AN, wanB, tno2)
    {

        /* get info. on OpenMP */

        OMPID  = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        /* allocation of arrays */

        MP = (int *)malloc(sizeof(int) * List_YOUSO[2]);

        /* start of the Mc_AN loop which is parallelized by OpenMP */

        for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            wan   = WhatSpecies[Gc_AN];

            /***********************************************
             find the size of matrix for the atom Mc_AN
                       and set the MP vector

              Note:
               MP indicates the starting position of
                    atom i in arraies H and S
            ***********************************************/

            Anum = 1;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                MP[i] = Anum;
                Gi    = natn[Gc_AN][i];
                wanA  = WhatSpecies[Gi];
                Anum += Spe_Total_CNO[wanA];
            }
            NUM = Anum - 1;
            n2  = NUM + 3;

            /***********************************************
             allocation of arrays:

            double S_DC[n2][n2];
            double H_DC[SpinP_switch+1][n2][n2];
            double ko[n2];
            ***********************************************/

            double *S_DC_store =
                (double *)DC_MallocArray((size_t)n2 * (size_t)n2, sizeof(double), "DC overlap matrix");
            S_DC = (double **)DC_MallocArray((size_t)n2, sizeof(double *), "DC overlap row pointers");
            for (i = 0; i < n2; i++) {
                S_DC[i] = S_DC_store + (size_t)i * (size_t)n2;
            }

            double *H_DC_store =
                (double *)DC_MallocArray((size_t)(SpinP_switch + 1) * (size_t)n2 * (size_t)n2, sizeof(double),
                                         "DC Hamiltonian matrices");
            H_DC = (double ***)DC_MallocArray((size_t)(SpinP_switch + 1), sizeof(double **),
                                              "DC Hamiltonian spin pointers");
            for (spin = 0; spin <= SpinP_switch; spin++) {
                H_DC[spin] = (double **)DC_MallocArray((size_t)n2, sizeof(double *), "DC Hamiltonian row pointers");
                for (i = 0; i < n2; i++) {
                    H_DC[spin][i] = H_DC_store + ((size_t)spin * (size_t)n2 + (size_t)i) * (size_t)n2;
                }
            }

            ko = (double *)DC_MallocArray((size_t)n2, sizeof(double), "DC eigenvalues");
            M1 = (double *)DC_MallocArray((size_t)n2, sizeof(double), "DC overlap scales");

            double *C_store =
                (double *)DC_MallocArray((size_t)n2 * (size_t)n2, sizeof(double), "DC eigenvector matrix");
            C = (double **)DC_MallocArray((size_t)n2, sizeof(double *), "DC eigenvector row pointers");
            for (i = 0; i < n2; i++) {
                C[i] = C_store + (size_t)i * (size_t)n2;
            }

            int use_dc_gpu =
                (scf_eigen_lib_flag == GPUSOLVER && !DC_GpuSolver_GemmDisabled() && !DC_GpuSolver_EigenDisabled() &&
                 DC_GpuSolver_RankAllowed() && DC_GPU_Threshold() <= NUM);
            /* the overlap the GPU solve reads: a resident S-cache copy, or
               NULL for the one staged in the solver context */
            const double *d_S_use          = NULL;
            int           s_dc_host_filled = 1;
            if (SCF_iter <= 2) {
                memset(S_DC_store, 0, sizeof(double) * (size_t)n2 * (size_t)n2);
            }
            memset(H_DC_store, 0, sizeof(double) * (size_t)(SpinP_switch + 1) * (size_t)n2 * (size_t)n2);

            /***********************************************
             construct cluster full matrices of Hamiltonian
                   and overlap for the atom Mc_AN
            ***********************************************/

            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                ig   = natn[Gc_AN][i];
                ian  = Spe_Total_CNO[WhatSpecies[ig]];
                Anum = MP[i];
                ih   = S_G2M[ig];

                for (j = 0; j <= (FNAN[Gc_AN] + SNAN[Gc_AN]); j++) {

                    kl   = RMI1[Mc_AN][i][j];
                    jg   = natn[Gc_AN][j];
                    jan  = Spe_Total_CNO[WhatSpecies[jg]];
                    Bnum = MP[j];

                    if (0 <= kl) {

                        if (SCF_iter <= 2) {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    S_DC[Anum + m][Bnum + n] = OLP0[ih][kl][m][n];
                                }
                            }
                        }

                        for (spin = 0; spin <= SpinP_switch; spin++) {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    H_DC[spin][Anum + m][Bnum + n] = Hks[spin][ih][kl][m][n];
                                }
                            }
                        }
                    }
                }
            }

            /****************************************************
             Solve the generalized eigenvalue problem
             HC = SCE

             1) diagonalize S
             2) search negative eigenvalues of S
            ****************************************************/

            if (SCF_iter <= 2) {

                if (measure_time)
                    dtime(&stime);

                if (use_dc_gpu) {
                    if (!DCCol_GpuSolver_DiagonalizeOverlap(NUM, S_DC, ko)) {
                        use_dc_gpu = 0;
                        DC_Eigen_lapack_cpu(S_DC, ko, NUM, NUM);
                    }
                } else {
                    DC_Eigen_lapack_cpu(S_DC, ko, NUM, NUM);
                }

                /***********************************************
                      Searching of negative eigenvalues
                ************************************************/

                P_min = 1;
                for (l = 1; l <= NUM; l++) {

                    if (ko[l] < OLP_eigen_cut) {
                        P_min = l + 1;
                        if (3 <= level_stdout) {
                            printf("<DC>  Negative EV of OLP %2d %15.12f\n", l, ko[l]);
                        }
                    }
                }

                S12[Mc_AN][0][0] = P_min;

                for (l = 1; l < P_min; l++)
                    M1[l] = 0.0;
                for (l = P_min; l <= NUM; l++)
                    M1[l] = 1.0 / sqrt(ko[l]);

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = 1; j1 <= NUM; j1++) {
                        S_DC[i1][j1]       = S_DC[i1][j1] * M1[j1];
                        S12[Mc_AN][i1][j1] = S_DC[i1][j1];
                    }
                }

                if (measure_time) {
                    dtime(&etime);
                    time1 += etime - stime;
                }

            }

            else {

                P_min = (int)S12[Mc_AN][0][0];

                if (use_dc_gpu && (d_S_use = DCCol_SCache_ValidPtr(Mc_AN, NUM)) != NULL) {
                    s_dc_host_filled = 0; /* filled lazily, only if the GPU solve falls back */
                } else {
                    DCCol_FillSDCFromS12(Mc_AN, NUM, S_DC);
                }
            }

            /***********************************************
              transform Hamiltonian matrix
            ************************************************/

            // compiler's bug
            Anum = 1;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                MP[i] = Anum;
                Gi    = natn[Gc_AN][i];
                wanA  = WhatSpecies[Gi];
                Anum += Spe_Total_CNO[wanA];
            }
            NUM = Anum - 1;

            if (use_dc_gpu && d_S_use == NULL) {
                if (!DCCol_GpuSolver_LoadTransformedOverlap(NUM, S_DC) ||
                    !DCCol_SCache_Store(Mc_AN, NUM, &d_S_use)) {
                    use_dc_gpu = 0;
                }
            }

            int use_dc_openmp = 0;

            if (use_dc_openmp) {
                size_t free_memory, total_memory;
                size_t needmemsize = 0;
                needmemsize += sizeof(double) * (SpinP_switch + 1) * (NUM + 1) * (NUM + 1);
                needmemsize += sizeof(double) * (NUM + 1) * (NUM + 1);
                needmemsize += sizeof(double) * (NUM + 1);
                needmemsize += sizeof(double) * (NUM + 1) * (NUM + 1);

                while (true) {
                    OpenMX_GpuMemGetInfo(&free_memory, &total_memory);
                    if (free_memory > needmemsize) {
                        break;
                    }

                    double wait_time    = (double)rand() / (double)RAND_MAX * WAITTIME;
                    double start_time   = MPI_Wtime();
                    double current_time = start_time;
                    while ((current_time - start_time) < wait_time) {
                        current_time = MPI_Wtime();
                    }
                }
            }

            for (spin = 0; spin <= SpinP_switch; spin++) {

                if (measure_time)
                    dtime(&stime);

                if (use_dc_gpu) {
                    NUM1 = NUM - (P_min - 1);
                    if (!DC_GpuSolver_PrepareGemmBackendForSolve(NUM, NUM1) ||
                        !DCCol_GpuSolver_SolveHamiltonian(NUM, P_min, d_S_use, S_DC, H_DC[spin], ko, C)) {
                        use_dc_gpu = 0;
                        if (!s_dc_host_filled) {
                            DCCol_FillSDCFromS12(Mc_AN, NUM, S_DC);
                            s_dc_host_filled = 1;
                        }
                        NUM1 = DCCol_CPU_SolveHamiltonian(NUM, P_min, S_DC, H_DC[spin], ko, C);
                    }

                } else if (use_dc_openmp) {
                    // OpenMP

                    /* transpose S */
#pragma omp parallel for
                    for (i1 = 1; i1 <= NUM; i1++) {
                        for (j1 = i1 + 1; j1 <= NUM; j1++) {
                            tmp1         = S_DC[i1][j1];
                            tmp2         = S_DC[j1][i1];
                            S_DC[i1][j1] = tmp2;
                            S_DC[j1][i1] = tmp1;
                        }
                    }

                    /* H * U * M1 */

                    /***********************************************
                             find the size of matrix for the atom Mc_AN
                                    and set the MP vector

                            Note:
                            MP indicates the starting position of
                                    atom i in arraies H and S
                            ***********************************************/

#pragma omp parallel for collapse(2)
                    for (j1 = 1; j1 <= NUM - 3; j1 = j1 + 4) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            double sum1 = 0.0;
                            double sum2 = 0.0;
                            double sum3 = 0.0;
                            double sum4 = 0.0;
                            for (l = 1; l <= NUM; l++) {
                                sum1 += H_DC[spin][i1][l] * S_DC[j1][l];
                                sum2 += H_DC[spin][i1][l] * S_DC[j1 + 1][l];
                                sum3 += H_DC[spin][i1][l] * S_DC[j1 + 2][l];
                                sum4 += H_DC[spin][i1][l] * S_DC[j1 + 3][l];
                            }
                            C[j1][i1]     = sum1;
                            C[j1 + 1][i1] = sum2;
                            C[j1 + 2][i1] = sum3;
                            C[j1 + 3][i1] = sum4;
                        }
                    }

                    j1s = NUM - NUM % 4 + 1;
#pragma omp parallel for collapse(2)
                    for (j1 = j1s; j1 <= NUM; j1++) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum = 0.0;
                            for (l = 1; l <= NUM; l++) {
                                sum += H_DC[spin][i1][l] * S_DC[j1][l];
                            }
                            C[j1][i1] = sum;
                        }
                    }

                    /* M1 * U^+ H * U * M1 */

#pragma omp parallel for collapse(2)
                    for (j1 = 1; j1 <= NUM - 3; j1 = j1 + 4) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            double sum1 = 0.0;
                            double sum2 = 0.0;
                            double sum3 = 0.0;
                            double sum4 = 0.0;
                            for (l = 1; l <= NUM; l++) {
                                sum1 += S_DC[i1][l] * C[j1][l];
                                sum2 += S_DC[i1][l] * C[j1 + 1][l];
                                sum3 += S_DC[i1][l] * C[j1 + 2][l];
                                sum4 += S_DC[i1][l] * C[j1 + 3][l];
                            }
                            H_DC[spin][j1][i1]     = sum1;
                            H_DC[spin][j1 + 1][i1] = sum2;
                            H_DC[spin][j1 + 2][i1] = sum3;
                            H_DC[spin][j1 + 3][i1] = sum4;
                        }
                    }

                    j1s = NUM - NUM % 4 + 1;
#pragma omp parallel for collapse(2)
                    for (j1 = j1s; j1 <= NUM; j1++) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum1 = 0.0;
                            for (l = 1; l <= NUM; l++) {
                                sum1 += S_DC[i1][l] * C[j1][l];
                            }
                            H_DC[spin][j1][i1] = sum1;
                        }
                    }

                    /* H_DC to C (transposition) */

#pragma omp parallel for collapse(2)
                    for (i1 = P_min; i1 <= NUM; i1++) {
                        for (j1 = P_min; j1 <= NUM; j1++) {
                            C[j1 - (P_min - 1)][i1 - (P_min - 1)] = H_DC[spin][i1][j1];
                        }
                    }

                    /***********************************************
                             diagonalize the trasformed Hamiltonian matrix
                            ************************************************/

                    NUM1 = NUM - (P_min - 1);

                    Eigen_lapack_openmp(C, ko, NUM1, NUM1);

                    /* C to H (transposition) */

#pragma omp parallel for collapse(2)
                    for (i1 = 1; i1 <= NUM; i1++) {
                        for (j1 = 1; j1 <= NUM1; j1++) {
                            H_DC[spin][j1][i1] = C[i1][j1];
                        }
                    }

                    /***********************************************
                             transformation to the original eigen vectors.
                                            NOTE 244P
                            ***********************************************/

                    /* transpose */

#pragma omp parallel for
                    for (i1 = 1; i1 <= NUM; i1++) {
                        for (j1 = i1 + 1; j1 <= NUM; j1++) {
                            tmp1         = S_DC[i1][j1];
                            tmp2         = S_DC[j1][i1];
                            S_DC[i1][j1] = tmp2;
                            S_DC[j1][i1] = tmp1;
                        }
                    }

#pragma omp parallel for collapse(2)
                    for (j1 = 1; j1 <= NUM1; j1++) {
                        for (l = NUM; P_min <= l; l--) {
                            H_DC[spin][j1][l] = H_DC[spin][j1][l - (P_min - 1)];
                        }
                    }

#pragma omp parallel for collapse(2)
                    for (j1 = 1; j1 <= NUM - 3; j1 = j1 + 4) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            double sum1 = 0.0;
                            double sum2 = 0.0;
                            double sum3 = 0.0;
                            double sum4 = 0.0;
                            for (l = P_min; l <= NUM; l++) {
                                sum1 += S_DC[i1][l] * H_DC[spin][j1][l];
                                sum2 += S_DC[i1][l] * H_DC[spin][j1 + 1][l];
                                sum3 += S_DC[i1][l] * H_DC[spin][j1 + 2][l];
                                sum4 += S_DC[i1][l] * H_DC[spin][j1 + 3][l];
                            }
                            C[i1][j1]     = sum1;
                            C[i1][j1 + 1] = sum2;
                            C[i1][j1 + 2] = sum3;
                            C[i1][j1 + 3] = sum4;
                        }
                    }

                    j1s = NUM - NUM % 4 + 1;
#pragma omp parallel for collapse(2)
                    for (j1 = j1s; j1 <= NUM; j1++) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum = 0.0;
                            for (l = P_min; l <= NUM; l++) {
                                sum += S_DC[i1][l] * H_DC[spin][j1][l];
                            }
                            C[i1][j1] = sum;
                        }
                    }

                    // compiler's bug
                    Anum = 1;
                    for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                        MP[i] = Anum;
                        Gi    = natn[Gc_AN][i];
                        wanA  = WhatSpecies[Gi];
                        Anum += Spe_Total_CNO[wanA];
                    }
                    NUM = Anum - 1;

                } else {
                    if (!s_dc_host_filled) {
                        DCCol_FillSDCFromS12(Mc_AN, NUM, S_DC);
                        s_dc_host_filled = 1;
                    }
                    NUM1 = DCCol_CPU_SolveHamiltonian(NUM, P_min, S_DC, H_DC[spin], ko, C);
                }

                if (measure_time) {
                    dtime(&etime);
                    time2 += etime - stime;
                }

                /*
                  for (i=1; i<=NUM1; i++){
                  printf("ko %15.12f 1.0\n",ko[i]);
                  }

                  for (i=1; i<=NUM; i++){
                  for (j=1; j<=NUM; j++){
                  printf("%15.12f ",C[i][j]);
                  }
                  printf("\n");
                  }
                */

                /*
                  MPI_Finalize();
                  exit(0);
                */

                /***********************************************
                   store eigenvalues and residues of poles
                ***********************************************/

                if (measure_time)
                    dtime(&stime);

                for (i1 = 1; i1 <= NUM; i1++) {
                    EVal[spin][Mc_AN][i1 - 1] = 1000.0;
                }
                for (i1 = 1; i1 <= NUM1; i1++) {
                    EVal[spin][Mc_AN][i1 - 1] = ko[i1];
                }

                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];

                if (is_scf_mode) {
                    for (i1 = 0; i1 < Msize[Mc_AN]; i1++) {
                        PDOS_DC[spin][Mc_AN][i1] = 0.0;
                    }
                }

                for (i = 0; i < tno1; i++) {
                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB  = WhatSpecies[Gh_AN];
                        tno2  = Spe_Total_CNO[wanB];
                        Bnum  = MP[h_AN];
                        for (j = 0; j < tno2; j++) {
                            tmp1 = OLP0[Mc_AN][h_AN][i][j];
                            for (i1 = 1; i1 <= NUM1; i1++) {
                                tmp2 = C[1 + i][i1] * C[Bnum + j][i1];
                                Residues[spin][Mc_AN][h_AN][i][j][i1 - 1] = tmp2;
                                if (is_scf_mode) {
                                    PDOS_DC[spin][Mc_AN][i1 - 1] += tmp2 * tmp1;
                                }
                            }
                        }
                    }
                }

                if (measure_time) {
                    dtime(&etime);
                    time3 += etime - stime;
                }

            } /* end of spin */

            /****************************************************
                              free arrays
            ****************************************************/

            free(S_DC_store);
            free(S_DC);

            for (spin = 0; spin <= SpinP_switch; spin++) {
                free(H_DC[spin]);
            }
            free(H_DC_store);
            free(H_DC);

            free(ko);
            free(M1);

            free(C_store);
            free(C);

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
        } /* end of Mc_AN */

        /* freeing of arrays */

        free(MP);
    } /* //#pragma omp parallel */

    if (is_scf_mode) {

        /****************************************************
                  calculate projected DOS
        ****************************************************/

        if (measure_time)
            dtime(&stime);

        /* PDOS_DC was accumulated while storing Residues, avoiding a second pass here. */

        if (measure_time) {
            dtime(&etime);
            time4 += etime - stime;
        }

        /****************************************************
                       find chemical potential
        ****************************************************/

        if (measure_time)
            dtime(&stime);

        po    = 0;
        loopN = 0;

        ChemP_MAX = 30.0;
        ChemP_MIN = -30.0;
        if (SpinP_switch == 0)
            spin_degeneracy = 2.0;
        else if (SpinP_switch == 1)
            spin_degeneracy = 1.0;

        do {
            ChemP = 0.50 * (ChemP_MAX + ChemP_MIN);

            My_Num_State = 0.0;
            for (spin = 0; spin <= SpinP_switch; spin++) {
                for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                    dtime(&Stime_atom);

                    Gc_AN = M2G[Mc_AN];

                    for (i = 0; i < Msize[Mc_AN]; i++) {
                        FermiF = DC_FermiWeight(EVal[spin][Mc_AN][i], ChemP, Beta, max_x);
                        My_Num_State += spin_degeneracy * FermiF * PDOS_DC[spin][Mc_AN][i];
                    }

                    dtime(&Etime_atom);
                    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
                }
            }

            /* MPI, My_Num_State */

            MPI_Allreduce(&My_Num_State, &Num_State, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

            Dnum = (TZ - Num_State) - system_charge;
            if (0.0 <= Dnum)
                ChemP_MIN = ChemP;
            else
                ChemP_MAX = ChemP;
            if (fabs(Dnum) < 1.0e-13)
                po = 1;

            if (myid == Host_ID && 2 <= level_stdout) {
                printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n", ChemP, TZ, Num_State);
            }

            loopN++;
        } while (po == 0 && loopN < 1000);

        /****************************************************
            eigenenergy by summing up eigenvalues
        ****************************************************/

        double ***FermiW =
            (double ***)DC_MallocArray((size_t)(SpinP_switch + 1), sizeof(double **), "DC col Fermi weights");
        double ***EFermiW =
            (double ***)DC_MallocArray((size_t)(SpinP_switch + 1), sizeof(double **), "DC col energy Fermi weights");

        My_Eele0[0] = 0.0;
        My_Eele0[1] = 0.0;
        for (spin = 0; spin <= SpinP_switch; spin++) {
            FermiW[spin] =
                (double **)DC_MallocArray((size_t)(Matomnum + 1), sizeof(double *), "DC col Fermi weight atoms");
            EFermiW[spin] = (double **)DC_MallocArray((size_t)(Matomnum + 1), sizeof(double *),
                                                      "DC col energy Fermi weight atoms");
            FermiW[spin][0]  = NULL;
            EFermiW[spin][0] = NULL;

            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                dtime(&Stime_atom);

                Gc_AN = M2G[Mc_AN];
                {
                    const int     n       = Msize[Mc_AN];
                    const double *eval    = EVal[spin][Mc_AN];
                    double *      weights = (double *)DC_MallocArray((size_t)(2 * n), sizeof(double),
                                                                     "DC col Fermi weight buffer");
                    double *      fweight = weights;
                    double *      eweight = weights + n;

                    FermiW[spin][Mc_AN]  = fweight;
                    EFermiW[spin][Mc_AN] = eweight;

                    for (i = 0; i < n; i++) {
                        FermiF     = DC_FermiWeight(eval[i], ChemP, Beta, max_x);
                        fweight[i] = FermiF;
                        eweight[i] = FermiF * eval[i];
                        My_Eele0[spin] += eweight[i] * PDOS_DC[spin][Mc_AN][i];
                    }
                }

                dtime(&Etime_atom);
                time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
            }
        }

        /* MPI, My_Eele0 */
        for (spin = 0; spin <= SpinP_switch; spin++) {
            MPI_Allreduce(&My_Eele0[spin], &Eele0[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        }

        if (SpinP_switch == 0) {
            Eele0[1] = Eele0[0];
        }

        if (measure_time) {
            dtime(&etime);
            time5 += etime - stime;
        }

        /****************************************************
           calculate density and energy density matrices
        ****************************************************/

        if (measure_time)
            dtime(&stime);

        const int nspins_eele1 = SpinP_switch + 1;
        const int nthreads_eele1 = omp_get_max_threads();
        double *My_Eele1_threads = (double *)DC_MallocArray((size_t)(nthreads_eele1 * nspins_eele1),
                                                            sizeof(double), "DC col Eele1 threads");
        for (i = 0; i < nthreads_eele1 * nspins_eele1; i++) {
            My_Eele1_threads[i] = 0.0;
        }

        //#pragma omp parallel shared(FNAN, time_per_atom, EDM, CDM, Residues, natn, max_x, Beta, ChemP, EVal, Msize, Spe_Total_CNO, WhatSpecies, M2G, SpinP_switch, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, spin, Stime_atom, Gc_AN, wanA, tno1, i1, x, FermiF, h_AN, Gh_AN, wanB, tno2, i, j, tmp1, Etime_atom)
        {
            /* get info. on OpenMP */

            OMPID  = omp_get_thread_num();
            Nthrds = omp_get_num_threads();
            Nprocs = omp_get_num_procs();

            for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {
                for (spin = 0; spin <= SpinP_switch; spin++) {

                    dtime(&Stime_atom);

                    Gc_AN = M2G[Mc_AN];
                    wanA  = WhatSpecies[Gc_AN];
                    tno1  = Spe_Total_CNO[wanA];

                    const int     n       = Msize[Mc_AN];
                    const double *fweight = FermiW[spin][Mc_AN];
                    const double *eweight = EFermiW[spin][Mc_AN];

                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB  = WhatSpecies[Gh_AN];
                        tno2  = Spe_Total_CNO[wanB];
                        for (i = 0; i < tno1; i++) {
                            for (j = 0; j < tno2; j++) {
                                double cdm_sum = 0.0;
                                double edm_sum = 0.0;
                                double *res    = Residues[spin][Mc_AN][h_AN][i][j];

                                DC_DotDensityResidueReal(n, res, fweight, eweight, &cdm_sum, &edm_sum);
                                CDM[spin][Mc_AN][h_AN][i][j] = cdm_sum;
                                EDM[spin][Mc_AN][h_AN][i][j] = edm_sum;
                                My_Eele1_threads[OMPID * nspins_eele1 + spin] +=
                                    cdm_sum * Hks[spin][Mc_AN][h_AN][i][j];
                            }
                        }
                    }

                    dtime(&Etime_atom);
                    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
                }
            }

        } /* //#pragma omp parallel */

        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
                free(FermiW[spin][Mc_AN]);
            }
            free(EFermiW[spin]);
            free(FermiW[spin]);
        }
        free(EFermiW);
        free(FermiW);

        /****************************************************
                         bond energies
        ****************************************************/

        My_Eele1[0] = 0.0;
        My_Eele1[1] = 0.0;
        for (k = 0; k < nthreads_eele1; k++) {
            for (spin = 0; spin <= SpinP_switch; spin++) {
                My_Eele1[spin] += My_Eele1_threads[k * nspins_eele1 + spin];
            }
        }
        free(My_Eele1_threads);

        /* MPI, My_Eele1 */
        for (spin = 0; spin <= SpinP_switch; spin++) {
            MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        }

        if (SpinP_switch == 0) {
            Eele1[1] = Eele1[0];
        }

        if (3 <= level_stdout && myid == Host_ID) {
            printf("Eele00=%15.12f Eele01=%15.12f\n", Eele0[0], Eele0[1]);
            printf("Eele10=%15.12f Eele11=%15.12f\n", Eele1[0], Eele1[1]);
        }

        if (measure_time) {
            dtime(&etime);
            time6 += etime - stime;
        }

    } /* if ( strcasecmp(mode,"scf")==0 ) */

    else if (strcasecmp(mode, "dos") == 0) {
        Save_DOS_Col(Residues, OLP0, EVal, Msize);
    }

    if (measure_time) {
        printf("Divide_Conquer myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f\n",
               myid, time1, time2, time3, time4, time5, time6);
        fflush(stdout);
    }

    /****************************************************
      freeing of arrays:

    ****************************************************/

    free(Snd_H_Size);
    free(Rcv_H_Size);

    free(Snd_S_Size);
    free(Rcv_S_Size);

    free(Msize);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
            free(EVal[spin][Mc_AN]);
        }
        free(EVal[spin]);
    }
    free(EVal);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN   = 0;
                FNAN[0] = 1;
                tno1    = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                wanA  = WhatSpecies[Gc_AN];
                tno1  = Spe_Total_CNO[wanA];
            }

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                if (Mc_AN == 0) {
                    tno2 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                }

                for (i = 0; i < tno1; i++) {
                    for (j = 0; j < tno2; j++) {
                        free(Residues[spin][Mc_AN][h_AN][i][j]);
                    }
                    free(Residues[spin][Mc_AN][h_AN][i]);
                }
                free(Residues[spin][Mc_AN][h_AN]);
            }
            free(Residues[spin][Mc_AN]);
        }
        free(Residues[spin]);
    }
    free(Residues);

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
            free(PDOS_DC[spin][Mc_AN]);
        }
        free(PDOS_DC[spin]);
    }
    free(PDOS_DC);

    /* for time */
    dtime(&TEtime);
    time0 = TEtime - TStime;

    /* for PrintMemory */
    firsttime = 0;

    return time0;
}

#pragma optimization_level 2
static double DC_NonCol(char * mode, int SCF_iter, double ***** Hks, double ***** ImNL, double **** OLP0,
                        double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2])
{
    static int      firsttime = 1;
    int             Mc_AN, Gc_AN, i, Gi, wan, wanA, wanB, Anum;
    int             size1, size2, num, NUM, NUM1, n2, Cwan, Hwan;
    int             ih, ig, ian, j, kl, jg, jan, Bnum, m, n, spin, so;
    int             l, i1, j1, P_min, m_size;
    int             po, loopN, tno1, tno2, h_AN, Gh_AN;
    int             k, ii1, jj1, k1, l1, ompi;
    int             MA_AN, GA_AN, GB_AN, tnoA, tnoB;
    double          My_TZ, TZ, sum, FermiF, time0;
    double          My_Num_State, Num_State, x, Dnum;
    double          sum_r, sum_i, tmp1, tmp2;
    double          sum1_r, sum1_i, sum2_r, sum2_i;
    double          TStime, TEtime;
    double          My_Eele0[2], My_Eele1[2];
    double          max_x = 50.0;
    double          ChemP_MAX, ChemP_MIN, spin_degeneracy;
    double **       S_DC, *ko, *M1;
    dcomplex **     C, **H_DC;
    double **       EVal;
    dcomplex ****** Residues;
    double **       PDOS_DC;
    int *           MP, *Msize;
    double *        tmp_array;
    double *        tmp_array2;
    double *        omp_tmp;
    int *           Snd_H_Size, *Rcv_H_Size;
    int *           Snd_iHNL_Size, *Rcv_iHNL_Size;
    int *           Snd_S_Size, *Rcv_S_Size;
    int             numprocs, myid, ID, IDS, IDR, tag = 999;
    double          Stime_atom, Etime_atom;
    double          OLP_eigen_cut = Threshold_OLP_Eigen;

    MPI_Status  stat;
    MPI_Request request;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs, Nthrds0;

    /* MPI */
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);
    int is_scf_mode = (strcasecmp(mode, "scf") == 0);

    // Set the device to be used by CUDA and OpenMP
    if (scf_eigen_lib_flag == GPUSOLVER) {
        // CUDA
        set_hip_default_device_from_local_rank();

        // OpenMP
        set_openmp_nvidia_device_from_local_rank();
    }

    dtime(&TStime);

    /****************************************************
      allocation of arrays:

      int MP[List_YOUSO[2]];
      int Msize[Matomnum+1];
      double EVal[Matomnum+1][n2];
    ****************************************************/

    Snd_H_Size    = (int *)malloc(sizeof(int) * numprocs);
    Rcv_H_Size    = (int *)malloc(sizeof(int) * numprocs);
    Snd_iHNL_Size = (int *)malloc(sizeof(int) * numprocs);
    Rcv_iHNL_Size = (int *)malloc(sizeof(int) * numprocs);
    Snd_S_Size    = (int *)malloc(sizeof(int) * numprocs);
    Rcv_S_Size    = (int *)malloc(sizeof(int) * numprocs);

    m_size = 0;
    Msize  = (int *)malloc(sizeof(int) * (Matomnum + 1));

    EVal = (double **)malloc(sizeof(double *) * (Matomnum + 1));

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

        if (Mc_AN == 0) {
            Gc_AN        = 0;
            FNAN[0]      = 1;
            SNAN[0]      = 0;
            n2           = 1;
            Msize[Mc_AN] = 1;
        } else {
            Gc_AN = M2G[Mc_AN];
            Anum  = 1;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                Gi   = natn[Gc_AN][i];
                wanA = WhatSpecies[Gi];
                Anum = Anum + Spe_Total_CNO[wanA];
            }
            NUM          = Anum - 1;
            Msize[Mc_AN] = NUM;
            n2           = 2 * NUM + 3;
        }

        m_size += n2;

        EVal[Mc_AN] = (double *)malloc(sizeof(double) * n2);
    }

    if (firsttime)
        PrintMemory("Divide_Conquer: EVal", sizeof(double) * m_size, NULL);

    if (2 <= level_stdout) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            printf("<DC> myid=%i Mc_AN=%2d Gc_AN=%2d Msize=%3d\n", myid, Mc_AN, M2G[Mc_AN], Msize[Mc_AN]);
        }
    }

    /****************************************************
      allocation of arrays:

      dcomplex Residues[3]
                       [Matomnum+1]
                       [FNAN[Gc_AN]+1]
                       [Spe_Total_CNO[Gc_AN]]
                       [Spe_Total_CNO[Gh_AN]]
                       [n2]
       To reduce the memory size, the size of NUM2
       should be found in the loop.
    ****************************************************/

    m_size = 0;

    Residues = (dcomplex ******)malloc(sizeof(dcomplex *****) * 3);
    for (spin = 0; spin < 3; spin++) {
        Residues[spin] = (dcomplex *****)malloc(sizeof(dcomplex ****) * (Matomnum + 1));
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN   = 0;
                FNAN[0] = 1;
                tno1    = 1;
                n2      = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                wanA  = WhatSpecies[Gc_AN];
                tno1  = Spe_Total_CNO[wanA];
                n2    = 2 * Msize[Mc_AN] + 2;
            }

            Residues[spin][Mc_AN] = (dcomplex ****)malloc(sizeof(dcomplex ***) * (FNAN[Gc_AN] + 1));

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                if (Mc_AN == 0) {
                    tno2 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                }

                Residues[spin][Mc_AN][h_AN] = (dcomplex ***)malloc(sizeof(dcomplex **) * tno1);
                for (i = 0; i < tno1; i++) {
                    Residues[spin][Mc_AN][h_AN][i] = (dcomplex **)malloc(sizeof(dcomplex *) * tno2);
                    for (j = 0; j < tno2; j++) {
                        Residues[spin][Mc_AN][h_AN][i][j] = (dcomplex *)malloc(sizeof(dcomplex) * n2);
                    }
                }

                m_size += tno1 * tno2 * n2;
            }
        }
    }

    if (firsttime)
        PrintMemory("Divide_Conquer: Residues", sizeof(dcomplex) * m_size, NULL);

    /****************************************************
      allocation of arrays:

      double PDOS_DC[Matomnum+1]
                    [n2]
    ****************************************************/

    m_size = 0;

    PDOS_DC = (double **)malloc(sizeof(double *) * (Matomnum + 1));
    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

        if (Mc_AN == 0)
            n2 = 1;
        else
            n2 = 2 * Msize[Mc_AN] + 2;

        m_size += n2;
        PDOS_DC[Mc_AN] = (double *)malloc(sizeof(double) * n2);
    }

    if (firsttime)
        PrintMemory("Divide_Conquer: PDOS_DC", sizeof(double) * m_size, NULL);

    /* allocation of a temporal array for OpenMP */

    // #pragma omp parallel shared(Nthrds0)
    {
        Nthrds0 = omp_get_num_threads();
    }

    omp_tmp = (double *)malloc(sizeof(double) * Nthrds0);

    /****************************************************
     MPI

     Hks
    ****************************************************/

    /***********************************
              set data size
    ************************************/

    for (ID = 0; ID < numprocs; ID++) {

        IDS = (myid + ID) % numprocs;
        IDR = (myid - ID + numprocs) % numprocs;

        if (ID != 0) {
            tag = 999;

            /* find data size to send block data */
            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                size1 = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    size1++;
                                }
                            }
                        }
                    }
                }

                Snd_H_Size[IDS] = size1;
                MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
            } else {
                Snd_H_Size[IDS] = 0;
            }

            /* receiving of size of data */

            if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {
                MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                Rcv_H_Size[IDR] = size2;
            } else {
                Rcv_H_Size[IDR] = 0;
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
                MPI_Wait(&request, &stat);

        } else {
            Snd_H_Size[IDS] = 0;
            Rcv_H_Size[IDR] = 0;
        }
    }

    /***********************************
               data transfer
    ************************************/

    tag = 999;
    for (ID = 0; ID < numprocs; ID++) {

        IDS = (myid + ID) % numprocs;
        IDR = (myid - ID + numprocs) % numprocs;

        if (ID != 0) {

            /*****************************
                    sending of data
            *****************************/

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                size1 = Snd_H_Size[IDS];

                /* allocation of array */

                tmp_array = (double *)malloc(sizeof(double) * size1);

                /* multidimentional array to vector array */

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    tmp_array[num] = Hks[spin][Mc_AN][h_AN][i][j];
                                    num++;
                                }
                            }
                        }
                    }
                }

                MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
            }

            /*****************************
               receiving of block data
            *****************************/

            if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                size2 = Rcv_H_Size[IDR];

                /* allocation of array */
                tmp_array2 = (double *)malloc(sizeof(double) * size2);

                MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                num = 0;
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
                    for (n = 0; n < (F_Rcv_Num[IDR] + S_Rcv_Num[IDR]); n++) {
                        Mc_AN++;
                        Gc_AN = Rcv_GAN[IDR][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];

                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    Hks[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }
                }

                /* freeing of array */
                free(tmp_array2);
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                MPI_Wait(&request, &stat);
                free(tmp_array); /* freeing of array */
            }
        }
    }

    /****************************************************
     MPI

     ImNL
    ****************************************************/

    /***********************************
               set data size
    ************************************/

    /* spin-orbit coupling or LDA+U */

    if (SO_switch == 1 || Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 ||
        Zeeman_NCO_switch == 1) {

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {
                tag = 999;

                /* find data size to send block data */
                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                    size1 = 0;
                    for (so = 0; so < List_YOUSO[5]; so++) {
                        for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                            Mc_AN = Snd_MAN[IDS][n];
                            Gc_AN = Snd_GAN[IDS][n];
                            Cwan  = WhatSpecies[Gc_AN];
                            tno1  = Spe_Total_CNO[Cwan];
                            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                                Gh_AN = natn[Gc_AN][h_AN];
                                Hwan  = WhatSpecies[Gh_AN];
                                tno2  = Spe_Total_CNO[Hwan];
                                for (i = 0; i < tno1; i++) {
                                    for (j = 0; j < tno2; j++) {
                                        size1++;
                                    }
                                }
                            }
                        }
                    }

                    Snd_iHNL_Size[IDS] = size1;
                    MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
                } else {
                    Snd_iHNL_Size[IDS] = 0;
                }

                /* receiving of size of data */

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {
                    MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                    Rcv_iHNL_Size[IDR] = size2;
                } else {
                    Rcv_iHNL_Size[IDR] = 0;
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
                    MPI_Wait(&request, &stat);

            } else {
                Snd_iHNL_Size[IDS] = 0;
                Rcv_iHNL_Size[IDR] = 0;
            }
        }

        /***********************************
                   data transfer
        ************************************/

        tag = 999;
        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {

                /*****************************
                       sending of data
                *****************************/

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                    size1 = Snd_iHNL_Size[IDS];

                    /* allocation of array */

                    tmp_array = (double *)malloc(sizeof(double) * size1);

                    /* multidimentional array to vector array */

                    num = 0;
                    for (so = 0; so < List_YOUSO[5]; so++) {
                        for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                            Mc_AN = Snd_MAN[IDS][n];
                            Gc_AN = Snd_GAN[IDS][n];
                            Cwan  = WhatSpecies[Gc_AN];
                            tno1  = Spe_Total_CNO[Cwan];
                            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                                Gh_AN = natn[Gc_AN][h_AN];
                                Hwan  = WhatSpecies[Gh_AN];
                                tno2  = Spe_Total_CNO[Hwan];
                                for (i = 0; i < tno1; i++) {
                                    for (j = 0; j < tno2; j++) {
                                        tmp_array[num] = ImNL[so][Mc_AN][h_AN][i][j];
                                        num++;
                                    }
                                }
                            }
                        }
                    }

                    MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
                }

                /*****************************
                   receiving of block data
                *****************************/

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                    size2 = Rcv_iHNL_Size[IDR];

                    /* allocation of array */
                    tmp_array2 = (double *)malloc(sizeof(double) * size2);

                    MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                    num = 0;
                    for (so = 0; so < List_YOUSO[5]; so++) {
                        Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
                        for (n = 0; n < (F_Rcv_Num[IDR] + S_Rcv_Num[IDR]); n++) {
                            Mc_AN++;
                            Gc_AN = Rcv_GAN[IDR][n];
                            Cwan  = WhatSpecies[Gc_AN];
                            tno1  = Spe_Total_CNO[Cwan];

                            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                                Gh_AN = natn[Gc_AN][h_AN];
                                Hwan  = WhatSpecies[Gh_AN];
                                tno2  = Spe_Total_CNO[Hwan];
                                for (i = 0; i < tno1; i++) {
                                    for (j = 0; j < tno2; j++) {
                                        ImNL[so][Mc_AN][h_AN][i][j] = tmp_array2[num];
                                        num++;
                                    }
                                }
                            }
                        }
                    }

                    /* freeing of array */
                    free(tmp_array2);
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
                    free(tmp_array); /* freeing of array */
                }
            }
        }
    }

    /****************************************************
     MPI

     OLP0
    ****************************************************/

    /***********************************
               set data size
    ************************************/

    if (SCF_iter <= 2) {

        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {
                tag = 999;

                /* find data size to send block data */
                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    size1 = 0;
                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    size1++;
                                }
                            }
                        }
                    }

                    Snd_S_Size[IDS] = size1;
                    MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
                } else {
                    Snd_S_Size[IDS] = 0;
                }

                /* receiving of size of data */

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {
                    MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
                    Rcv_S_Size[IDR] = size2;
                } else {
                    Rcv_S_Size[IDR] = 0;
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
                    MPI_Wait(&request, &stat);
            } else {
                Snd_S_Size[IDS] = 0;
                Rcv_S_Size[IDR] = 0;
            }
        }

        /***********************************
                   data transfer
        ************************************/

        tag = 999;
        for (ID = 0; ID < numprocs; ID++) {

            IDS = (myid + ID) % numprocs;
            IDR = (myid - ID + numprocs) % numprocs;

            if (ID != 0) {

                /*****************************
                        sending of data
                *****************************/

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {

                    size1 = Snd_S_Size[IDS];

                    /* allocation of array */

                    tmp_array = (double *)malloc(sizeof(double) * size1);

                    /* multidimentional array to vector array */

                    num = 0;

                    for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++) {
                        Mc_AN = Snd_MAN[IDS][n];
                        Gc_AN = Snd_GAN[IDS][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];
                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    tmp_array[num] = OLP0[Mc_AN][h_AN][i][j];
                                    num++;
                                }
                            }
                        }
                    }

                    MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
                }

                /*****************************
                   receiving of block data
                *****************************/

                if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0) {

                    size2 = Rcv_S_Size[IDR];

                    /* allocation of array */
                    tmp_array2 = (double *)malloc(sizeof(double) * size2);

                    MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

                    num   = 0;
                    Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
                    for (n = 0; n < (F_Rcv_Num[IDR] + S_Rcv_Num[IDR]); n++) {
                        Mc_AN++;
                        Gc_AN = Rcv_GAN[IDR][n];
                        Cwan  = WhatSpecies[Gc_AN];
                        tno1  = Spe_Total_CNO[Cwan];

                        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                            Gh_AN = natn[Gc_AN][h_AN];
                            Hwan  = WhatSpecies[Gh_AN];
                            tno2  = Spe_Total_CNO[Hwan];
                            for (i = 0; i < tno1; i++) {
                                for (j = 0; j < tno2; j++) {
                                    OLP0[Mc_AN][h_AN][i][j] = tmp_array2[num];
                                    num++;
                                }
                            }
                        }
                    }

                    /* freeing of array */
                    free(tmp_array2);
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
                    free(tmp_array); /* freeing of array */
                }
            }
        }
    }

    /****************************************************
              find the total number of electrons
    ****************************************************/

    My_TZ = 0.0;
    for (i = 1; i <= Matomnum; i++) {
        Gc_AN = M2G[i];
        wan   = WhatSpecies[Gc_AN];
        My_TZ = My_TZ + Spe_Core_Charge[wan];
    }

    /* MPI, My_TZ */

    MPI_Allreduce(&My_TZ, &TZ, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    // static double timeA = 0.0, timeB = 0.0, timeC = 0.0, timeD = 0.0, timeE = 0.0, timeF = 0.0, timeG = 0.0, timeH = 0.0;

    /* the GPU solves of this call (the largest admitted NUM, 0 for none);
       collective, so every rank asks */
    int dc_gpu_capacity = 0;
    if (scf_eigen_lib_flag == GPUSOLVER) {
        dc_gpu_capacity = DCNonCol_GpuAdmit(Matomnum, Msize);
    }

    /****************************************************
        Setting of Hamiltonian and overlap matrices

           MP indicates the starting position of
                atom i in arraies H and S
    ****************************************************/

    // #pragma omp parallel shared(List_YOUSO, time_per_atom, Residues, EVal, S12, OLP_eigen_cut, ImNL, Hks, Zeeman_NCO_switch, Zeeman_NCS_switch, Constraint_NCS_switch, Hub_U_switch, SO_switch, OLP0, SCF_iter, RMI1, S_G2M, Spe_Total_CNO, natn, SNAN, FNAN, WhatSpecies, M2G, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, Etime_atom, Stime_atom, Gc_AN, wan, Anum, i, MP, Gi, wanA, NUM, n2, S_DC, H_DC, ko, M1, C, ig, ian, ih, j, kl, jg, jan, Bnum, m, n, P_min, l, i1, j1, tmp1, tmp2, k, jj1, k1, sum_r, sum_i, l1, sum1_r, sum1_i, sum2_r, sum2_i, ii1, NUM1, tno1, h_AN, Gh_AN, wanB, tno2)
    {

        /* get info. on OpenMP */

        OMPID  = 0;  // omp_get_thread_num();
        Nthrds = 1;  // omp_get_num_threads();
        Nprocs = 1;  // omp_get_num_procs();

        /* allocation of arrays */

        MP = (int *)malloc(sizeof(int) * List_YOUSO[2]);

        /* start of the Mc_AN loop which is parallelized by OpenMP */

        for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            wan   = WhatSpecies[Gc_AN];

            /***********************************************
            find the size of matrix for the atom Mc_AN
                      and set the MP vector

           Note:
               MP indicates the starting position of
                    atom i in arraies H and S
            ***********************************************/

            Anum = 1;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                MP[i] = Anum;
                Gi    = natn[Gc_AN][i];
                wanA  = WhatSpecies[Gi];
                Anum  = Anum + Spe_Total_CNO[wanA];
            }
            NUM = Anum - 1;

            n2 = 2 * NUM + 3;

            /***********************************************
             allocation of arrays:

             double   S_DC[NUM+2][NUM+2];
             dcomplex H_DC[n2][n2];
             double   ko[n2];
             double   M1[n2];
             dcomplex C[n2][n2];
            ***********************************************/

            /* contiguous stores: one allocation per matrix, and the GPU
               solve stages each one with a single transfer */
            double *S_DC_store = (double *)DC_MallocArray((size_t)(NUM + 2) * (size_t)(NUM + 2), sizeof(double),
                                                          "DC NonCol overlap matrix");
            S_DC = (double **)DC_MallocArray((size_t)(NUM + 2), sizeof(double *), "DC NonCol overlap rows");
            for (i = 0; i < (NUM + 2); i++) {
                S_DC[i] = S_DC_store + (size_t)i * (size_t)(NUM + 2);
            }

            dcomplex *H_DC_store = (dcomplex *)DC_MallocArray((size_t)n2 * (size_t)n2, sizeof(dcomplex),
                                                              "DC NonCol Hamiltonian matrix");
            H_DC = (dcomplex **)DC_MallocArray((size_t)n2, sizeof(dcomplex *), "DC NonCol Hamiltonian rows");
            for (i = 0; i < n2; i++) {
                H_DC[i] = H_DC_store + (size_t)i * (size_t)n2;
            }

            ko = (double *)DC_MallocArray((size_t)n2, sizeof(double), "DC NonCol eigenvalues");
            M1 = (double *)DC_MallocArray((size_t)n2, sizeof(double), "DC NonCol overlap scales");

            dcomplex *C_store = (dcomplex *)DC_MallocArray((size_t)n2 * (size_t)n2, sizeof(dcomplex),
                                                           "DC NonCol eigenvector matrix");
            C = (dcomplex **)DC_MallocArray((size_t)n2, sizeof(dcomplex *), "DC NonCol eigenvector rows");
            for (i = 0; i < n2; i++) {
                C[i] = C_store + (size_t)i * (size_t)n2;
            }

            if (SCF_iter <= 2) {
                memset(S_DC_store, 0, sizeof(double) * (size_t)(NUM + 2) * (size_t)(NUM + 2));
            }
            memset(H_DC_store, 0, sizeof(dcomplex) * (size_t)n2 * (size_t)n2);

            /***********************************************
             construct cluster full matrices of Hamiltonian
                     and overlap for the atom Mc_AN
            ***********************************************/

            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++) {
                ig   = natn[Gc_AN][i];
                ian  = Spe_Total_CNO[WhatSpecies[ig]];
                Anum = MP[i];
                ih   = S_G2M[ig];

                for (j = 0; j <= (FNAN[Gc_AN] + SNAN[Gc_AN]); j++) {

                    kl   = RMI1[Mc_AN][i][j];
                    jg   = natn[Gc_AN][j];
                    jan  = Spe_Total_CNO[WhatSpecies[jg]];
                    Bnum = MP[j];

                    if (0 <= kl) {

                        if (SCF_iter <= 2) {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    S_DC[Anum + m][Bnum + n] = OLP0[ih][kl][m][n];
                                }
                            }
                        }

                        /* non-spin-orbit coupling and non-LDA+U */
                        if (SO_switch == 0 && Hub_U_switch == 0 && Constraint_NCS_switch == 0 &&
                            Zeeman_NCS_switch == 0 && Zeeman_NCO_switch == 0) {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    H_DC[Anum + m][Bnum + n].r             = Hks[0][ih][kl][m][n];
                                    H_DC[Anum + m][Bnum + n].i             = 0.0;
                                    H_DC[Anum + m + NUM][Bnum + n + NUM].r = Hks[1][ih][kl][m][n];
                                    H_DC[Anum + m + NUM][Bnum + n + NUM].i = 0.0;
                                    H_DC[Anum + m][Bnum + n + NUM].r       = Hks[2][ih][kl][m][n];
                                    H_DC[Anum + m][Bnum + n + NUM].i       = Hks[3][ih][kl][m][n];
                                    H_DC[Bnum + n + NUM][Anum + m].r       = H_DC[Anum + m][Bnum + n + NUM].r;
                                    H_DC[Bnum + n + NUM][Anum + m].i       = -H_DC[Anum + m][Bnum + n + NUM].i;
                                }
                            }
                        }

                        /* spin-orbit coupling or LDA+U */
                        else {
                            for (m = 0; m < ian; m++) {
                                for (n = 0; n < jan; n++) {
                                    H_DC[Anum + m][Bnum + n].r             = Hks[0][ih][kl][m][n];
                                    H_DC[Anum + m][Bnum + n].i             = ImNL[0][ih][kl][m][n];
                                    H_DC[Anum + m + NUM][Bnum + n + NUM].r = Hks[1][ih][kl][m][n];
                                    H_DC[Anum + m + NUM][Bnum + n + NUM].i = ImNL[1][ih][kl][m][n];
                                    H_DC[Anum + m][Bnum + n + NUM].r       = Hks[2][ih][kl][m][n];
                                    H_DC[Anum + m][Bnum + n + NUM].i = Hks[3][ih][kl][m][n] + ImNL[2][ih][kl][m][n];
                                    H_DC[Bnum + n + NUM][Anum + m].r = H_DC[Anum + m][Bnum + n + NUM].r;
                                    H_DC[Bnum + n + NUM][Anum + m].i = -H_DC[Anum + m][Bnum + n + NUM].i;
                                }
                            }
                        }

                    }
                }
            }

            /*
          if (Gc_AN==1){

          printf("H_DC.r\n");
          for (i=1; i<=NUM*2; i++){
            for (j=1; j<=NUM*2; j++){
              printf("H_DC.r i=%2d j=%2d %7.4f\n",i,j,H_DC[i][j].r);
            }
          }

          printf("H_DC.i\n");
          for (i=1; i<=NUM*2; i++){
            for (j=1; j<=NUM*2; j++){
              printf("H_DC.i i=%2d j=%2d %7.4f\n",i,j,H_DC[i][j].i);
            }
          }
          }
            */

            /*
            EigenBand_lapack(H_DC, ko, NUM*2, NUM*2, 1);

            if (Gc_AN==1){
              for (l=1; l<=NUM*2; l++){
                printf("ABC0 myid=%2d l=%2d ko=%15.12f\n",myid,l,ko[l]);fflush(stdout);
              }
            }
            */

            /****************************************************
             Solve the generalized eigenvalue problem
             HC = SCE

             1) diagonalize S
             2) search negative eigenvalues of S
            ****************************************************/

            if (SCF_iter <= 2) {
                // dtime(&timeA);

                DC_Eigen_lapack_cpu(S_DC, ko, NUM, NUM);

                // dtime(&timeB);
                // printf("timeA = %.3f\n", timeB - timeA);

                /***********************************************
                     Searching of negative eigenvalues
                ************************************************/

                P_min = 1;
                for (l = 1; l <= NUM; l++) {
                    if (ko[l] < OLP_eigen_cut) {
                        P_min = l + 1;
                        if (3 <= level_stdout) {
                            printf("<DC>  Negative EV of OLP %2d %15.12f\n", l, ko[l]);
                        }
                    }
                }

                S12[Mc_AN][0][0] = P_min;

                for (l = 1; l < P_min; l++)
                    M1[l] = 0.0;
                for (l = P_min; l <= NUM; l++)
                    M1[l] = 1.0 / sqrt(ko[l]);

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = 1; j1 <= NUM; j1++) {
                        S_DC[i1][j1]       = S_DC[i1][j1] * M1[j1];
                        S12[Mc_AN][i1][j1] = S_DC[i1][j1];
                    }
                }
                // dtime(&timeC);
                // printf("timeB = %.3f\n", timeC - timeB);
            }

            else {

                P_min = (int)S12[Mc_AN][0][0];

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = 1; j1 <= NUM; j1++) {
                        S_DC[i1][j1] = S12[Mc_AN][i1][j1];
                    }
                }
                // dtime(&timeC);
            }

            /*
            printf("S_DC Gc_AN=%2d\n",Gc_AN);

          if (Gc_AN==1){

          printf("S_DC\n");


          for (i=1; i<=NUM; i++){
            for (j=1; j<=NUM; j++){
              printf("S_DC i=%2d j=%2d %7.4f\n",i,j,S_DC[i][j]);
            }
          }

          }
            */

            /***********************************************
                   transform Hamiltonian matrix
            ************************************************/

            /* GPU solve for the larger clusters.  It needs P_min == 1: for
               P_min > 1 the index arithmetic of the CPU path below solves
               NUM1 = 2*NUM-(P_min-1) states although its transform fills
               only 2*(NUM-P_min+1), so such clusters keep that path as it is. */
            int use_dc_gpu = (0 < dc_gpu_capacity && DC_NonCol_GPU_Threshold() <= 2 * NUM && P_min == 1 &&
                              !DC_GpuSolver_GemmDisabled() && !DC_GpuSolver_EigenDisabled());

            if (use_dc_gpu && DCNonCol_GpuSolve(NUM, dc_gpu_capacity, S_DC, H_DC, C, ko)) {
                NUM1 = 2 * NUM;
            } else {
                /* transpose S */

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = i1 + 1; j1 <= NUM; j1++) {
                        tmp1         = S_DC[i1][j1];
                        tmp2         = S_DC[j1][i1];
                        S_DC[i1][j1] = tmp2;
                        S_DC[j1][i1] = tmp1;
                    }
                }

                /* H * U * M1 */

                for (j1 = P_min; j1 <= NUM; j1++) {
                    for (k = 0; k <= 1; k++) {
                        jj1 = 2 * j1 - P_min + k;
                        k1  = k * NUM;

                        for (i1 = 1; i1 <= 2 * NUM; i1++) {

                            sum_r = 0.0;
                            sum_i = 0.0;

                            for (l = 1; l <= NUM; l++) {
                                l1 = k1 + l;
                                sum_r += H_DC[i1][l1].r * S_DC[j1][l];
                                sum_i += H_DC[i1][l1].i * S_DC[j1][l];
                            }

                            C[jj1][i1].r = sum_r;
                            C[jj1][i1].i = sum_i;
                        }
                    }
                }

                /*
              if (Gc_AN==1){
              printf("C.r\n");
              for (i=1; i<=NUM*2; i++){
                for (j=1; j<=NUM*2; j++){
                  printf("ABC1 C.r i=%2d j=%2d %7.4f\n",i,j,C[i][j].r);
                }
              }

              printf("C.i\n");
              for (i=1; i<=NUM*2; i++){
                for (j=1; j<=NUM*2; j++){
                  printf("ABC1 C.i i=%2d j=%2d %7.4f\n",i,j,C[i][j].i);
                }
              }
              }
                */

                /* M1 * U^+ H * U * M1 */

                for (j1 = 1; j1 <= 2 * NUM; j1++) {
                    for (i1 = P_min; i1 <= NUM; i1++) {

                        sum1_r = 0.0;
                        sum1_i = 0.0;
                        sum2_r = 0.0;
                        sum2_i = 0.0;

                        for (l = 1; l <= NUM; l++) {
                            sum1_r += S_DC[i1][l] * C[j1][l].r;
                            sum1_i += S_DC[i1][l] * C[j1][l].i;
                        }

                        for (l = NUM + 1; l <= 2 * NUM; l++) {
                            l1 = l - NUM;
                            sum2_r += S_DC[i1][l1] * C[j1][l].r;
                            sum2_i += S_DC[i1][l1] * C[j1][l].i;
                        }

                        ii1             = 2 * i1 - P_min;
                        H_DC[j1][ii1].r = sum1_r;
                        H_DC[j1][ii1].i = sum1_i;

                        ii1             = 2 * i1 - P_min + 1;
                        H_DC[j1][ii1].r = sum2_r;
                        H_DC[j1][ii1].i = sum2_i;
                    }
                }

                /* H to C (transposition) */

                for (i1 = P_min; i1 <= 2 * NUM; i1++) {
                    for (j1 = P_min; j1 <= 2 * NUM; j1++) {
                        C[j1 - (P_min - 1)][i1 - (P_min - 1)].r = H_DC[i1][j1].r;
                        C[j1 - (P_min - 1)][i1 - (P_min - 1)].i = H_DC[i1][j1].i;
                    }
                }

                if (Gc_AN == 1) {

                    /*
                  printf("C.r\n");
                  for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                      printf("ABC2 C.r i=%2d j=%2d %7.4f\n",i,j,C[i][j].r);
                    }
                  }

                  printf("C.i\n");
                  for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                      printf("ABC2 C.i i=%2d j=%2d %7.4f\n",i,j,C[i][j].i);
                    }
                  }
                    */

                    /*
                  printf("C.r\n");
                  for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                      printf("%7.4f",C[i][j].r);
                    }
                    printf("\n");
                  }

                  printf("C.i\n");
                  for (i=1; i<=NUM*2; i++){
                    for (j=1; j<=NUM*2; j++){
                      printf("%7.4f",C[i][j].i);
                    }
                    printf("\n");
                  }
                    */
                }

                /***********************************************
                 diagonalize the trasformed Hamiltonian matrix
                ************************************************/

                // dtime(&timeD);
                // printf("timeC = %.3f\n", timeD - timeC);

                NUM1 = 2 * NUM - (P_min - 1);
                DC_EigenBand_lapack_cpu(C, ko, NUM1);

                // dtime(&timeE);
                // printf("timeD = %.3f\n", timeE - timeD);

                /*
              if (Gc_AN==1){
                for (l=1; l<=NUM1; l++){
                  printf("ABC10 myid=%2d l=%2d ko=%15.12f\n",myid,l,ko[l]);fflush(stdout);
                }
              }

              MPI_Finalize();
              exit(0);
                */

                /* C to H (transposition) */

                for (i1 = 1; i1 <= NUM1; i1++) {
                    for (j1 = 1; j1 <= NUM1; j1++) {
                        H_DC[j1][i1].r = C[i1][j1].r;
                        H_DC[j1][i1].i = C[i1][j1].i;
                    }
                }

                /***********************************************
                transformation to the original eigen vectors.
                NOTE 244P    C = U * lambda^{-1/2} * D
                ***********************************************/

                /* transpose S */

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = i1 + 1; j1 <= NUM; j1++) {
                        tmp1         = S_DC[i1][j1];
                        tmp2         = S_DC[j1][i1];
                        S_DC[i1][j1] = tmp2;
                        S_DC[j1][i1] = tmp1;
                    }
                }

                for (j1 = 1; j1 <= NUM1; j1++) {
                    for (k = 0; k <= 1; k++) {
                        for (i1 = 1; i1 <= NUM; i1++) {
                            sum_r = 0.0;
                            sum_i = 0.0;
                            for (l = P_min; l <= NUM; l++) {
                                sum_r += S_DC[i1][l] * H_DC[j1][2 * (l - P_min) + 1 + k].r;
                                sum_i += S_DC[i1][l] * H_DC[j1][2 * (l - P_min) + 1 + k].i;
                            }
                            C[i1 + k * NUM][j1].r = sum_r;
                            C[i1 + k * NUM][j1].i = sum_i;
                        }
                    }
                }
            }

            // dtime(&timeF);
            // printf("timeE = %.3f\n", timeF - timeE);

            /***********************************************
              store eigenvalues and residues of poles
            ***********************************************/

            for (i1 = 1; i1 <= 2 * NUM; i1++) {
                EVal[Mc_AN][i1 - 1] = 1000.0;
            }
            for (i1 = 1; i1 <= NUM1; i1++) {
                EVal[Mc_AN][i1 - 1] = ko[i1];
            }

            wanA = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[wanA];

            if (is_scf_mode) {
                for (i1 = 0; i1 < 2 * Msize[Mc_AN]; i1++) {
                    PDOS_DC[Mc_AN][i1] = 0.0;
                }
            }

            for (i = 0; i < tno1; i++) {
                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                    Bnum  = MP[h_AN];
                    for (j = 0; j < tno2; j++) {
                        tmp1 = OLP0[Mc_AN][h_AN][i][j];
                        for (i1 = 1; i1 <= NUM1; i1++) {

                            /* Re11 */
                            const double re11 =
                                C[1 + i][i1].r * C[Bnum + j][i1].r + C[1 + i][i1].i * C[Bnum + j][i1].i;
                            Residues[0][Mc_AN][h_AN][i][j][i1 - 1].r = re11;

                            /* Re22 */
                            const double re22 = C[1 + i + NUM][i1].r * C[Bnum + j + NUM][i1].r +
                                                C[1 + i + NUM][i1].i * C[Bnum + j + NUM][i1].i;
                            Residues[1][Mc_AN][h_AN][i][j][i1 - 1].r = re22;

                            if (is_scf_mode) {
                                PDOS_DC[Mc_AN][i1 - 1] += (re11 + re22) * tmp1;
                            }

                            /* Re12 */
                            Residues[2][Mc_AN][h_AN][i][j][i1 - 1].r =
                                C[1 + i][i1].r * C[Bnum + j + NUM][i1].r + C[1 + i][i1].i * C[Bnum + j + NUM][i1].i;

                            /* Im12 */
                            Residues[2][Mc_AN][h_AN][i][j][i1 - 1].i =
                                C[1 + i][i1].r * C[Bnum + j + NUM][i1].i - C[1 + i][i1].i * C[Bnum + j + NUM][i1].r;

                            /* spin-orbit coupling or LDA+U */
                            if (SO_switch == 1 || Hub_U_switch == 1 || 1 <= Constraint_NCS_switch ||
                                Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1) {

                                /* Im11 */
                                Residues[0][Mc_AN][h_AN][i][j][i1 - 1].i =
                                    C[1 + i][i1].r * C[Bnum + j][i1].i - C[1 + i][i1].i * C[Bnum + j][i1].r;

                                /*
                                printf("VVV1 Mc_AN=%2d h_AN=%2d i=%2d j=%2d i1=%2d %15.10f\n",
                                        Mc_AN,h_AN,i,j,i1,Residues[0][Mc_AN][h_AN][i][j][i1-1].i);
                                */

                                /* Im22 */
                                Residues[1][Mc_AN][h_AN][i][j][i1 - 1].i =
                                    C[1 + i + NUM][i1].r * C[Bnum + j + NUM][i1].i -
                                    C[1 + i + NUM][i1].i * C[Bnum + j + NUM][i1].r;
                            }
                        }
                    }
                }
            }

            // dtime(&timeG);
            // printf("timeF = %.3f\n", timeG - timeF);

            /****************************************************
                              free arrays
            ****************************************************/

            free(S_DC_store);
            free(S_DC);

            free(H_DC_store);
            free(H_DC);

            free(ko);
            free(M1);

            free(C_store);
            free(C);

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

            /*
              for (i1=1; i1<=NUM1; i1++){
              printf("Mc_AN=%2d i1=%2d EVal=%15.12f\n",Mc_AN,i1,EVal[Mc_AN][i1-1]);
              }
            */

        } /* end of Mc_AN */

        /* freeing of arrays */

        free(MP);

    } /* //#pragma omp parallel */

    /* the device matrices of the non-collinear GPU solve go back to the
       shared GPU for the grid and force phases, and so do the device
       workspaces the eigensolvers keep (MAGMA's GEMMul8 cache, rocSOLVER's
       zheevd scratch); the next call re-admits */
    DCNonCol_GpuRelease(1);
    if (0 < dc_gpu_capacity) {
        openmx_magma_ozaki2_release();
        (void)openmx_hipsolver_release_workspace();
    }

    if (is_scf_mode) {

        /****************************************************
                  calculate projected DOS
        ****************************************************/

        /* PDOS_DC was accumulated while storing Residues, avoiding a second pass here. */

        /****************************************************
        find chemical potential
        ****************************************************/

        po    = 0;
        loopN = 0;

        ChemP_MAX = 15.0;
        ChemP_MIN = -15.0;

        do {
            ChemP = 0.50 * (ChemP_MAX + ChemP_MIN);

            // #pragma omp parallel shared(omp_tmp, max_x, ChemP, Beta, EVal, Msize, Matomnum, M2G) private(OMPID, Nthrds, Nprocs, Mc_AN, Gc_AN, i, x, FermiF)
            {

                /* get info. on OpenMP */

                OMPID  = omp_get_thread_num();
                Nthrds = omp_get_num_threads();
                Nprocs = omp_get_num_procs();

                omp_tmp[OMPID] = 0.0;

                for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {

                    Gc_AN = M2G[Mc_AN];

                    for (i = 0; i < 2 * Msize[Mc_AN]; i++) {
                        FermiF = DC_FermiWeight(EVal[Mc_AN][i], ChemP, Beta, max_x);
                        omp_tmp[OMPID] += FermiF * PDOS_DC[Mc_AN][i];
                    }
                }

            } /* //#pragma omp parallel */

            My_Num_State = 0.0;
            for (ompi = 0; ompi < Nthrds0; ompi++) {
                My_Num_State += omp_tmp[ompi];
            }

            /* MPI, My_Num_State */

            MPI_Allreduce(&My_Num_State, &Num_State, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

            Dnum = (TZ - Num_State) - system_charge;
            if (0.0 <= Dnum)
                ChemP_MIN = ChemP;
            else
                ChemP_MAX = ChemP;
            if (fabs(Dnum) < 1.0e-11)
                po = 1;

            if (myid == Host_ID && 2 <= level_stdout) {
                printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n", ChemP, TZ, Num_State);
            }

            loopN++;
        } while (po == 0 && loopN < 1000);

        /****************************************************
            eigenenergy by summing up eigenvalues
        ****************************************************/

        double **FermiW =
            (double **)DC_MallocArray((size_t)(Matomnum + 1), sizeof(double *), "DC noncol Fermi weights");
        double **EFermiW = (double **)DC_MallocArray((size_t)(Matomnum + 1), sizeof(double *),
                                                    "DC noncol energy Fermi weights");
        FermiW[0]  = NULL;
        EFermiW[0] = NULL;

        My_Eele0[0] = 0.0;
        My_Eele0[1] = 0.0;

        // #pragma omp parallel shared(PDOS_DC, Beta, ChemP, max_x, EVal, Msize, M2G, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, Gc_AN, i, x, FermiF)
        {

            /* get info. on OpenMP */

            OMPID  = omp_get_thread_num();
            Nthrds = omp_get_num_threads();
            Nprocs = omp_get_num_procs();

            omp_tmp[OMPID] = 0.0;

            for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {

                Gc_AN = M2G[Mc_AN];
                {
                    const int     n       = 2 * Msize[Mc_AN];
                    const double *eval    = EVal[Mc_AN];
                    double *      weights = (double *)DC_MallocArray((size_t)(2 * n), sizeof(double),
                                                                     "DC noncol Fermi weight buffer");
                    double *      fweight = weights;
                    double *      eweight = weights + n;

                    FermiW[Mc_AN]  = fweight;
                    EFermiW[Mc_AN] = eweight;

                    for (i = 0; i < n; i++) {
                        FermiF     = DC_FermiWeight(eval[i], ChemP, Beta, max_x);
                        fweight[i] = FermiF;
                        eweight[i] = FermiF * eval[i];
                        omp_tmp[OMPID] += eweight[i] * PDOS_DC[Mc_AN][i];
                    }
                }
            }

        } /* //#pragma omp parallel */

        for (ompi = 0; ompi < Nthrds0; ompi++) {
            My_Eele0[0] += omp_tmp[ompi];
        }

        /* MPI, My_Eele0 */

        MPI_Allreduce(&My_Eele0[0], &Eele0[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        MPI_Allreduce(&My_Eele0[1], &Eele0[1], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

        /****************************************************
          calculate density and energy density matrices

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

        for (ompi = 0; ompi < Nthrds0; ompi++) {
            omp_tmp[ompi] = 0.0;
        }

        // #pragma omp parallel shared(iDM, Zeeman_NCO_switch, Zeeman_NCS_switch, Constraint_NCS_switch, Hub_U_switch, SO_switch, time_per_atom, EDM, CDM, Residues, natn, max_x, Beta, ChemP, EVal, Msize, Spe_Total_CNO, WhatSpecies, M2G, Matomnum) private(OMPID, Nthrds, Nprocs, Mc_AN, Stime_atom, Gc_AN, wanA, tno1, i1, x, FermiF, h_AN, Gh_AN, wanB, tno2, i, j, tmp1, Etime_atom)
        {

            /* get info. on OpenMP */

            OMPID  = omp_get_thread_num();
            Nthrds = omp_get_num_threads();
            Nprocs = omp_get_num_procs();

            for (Mc_AN = 1 + OMPID; Mc_AN <= Matomnum; Mc_AN += Nthrds) {

                dtime(&Stime_atom);

                Gc_AN = M2G[Mc_AN];
                wanA  = WhatSpecies[Gc_AN];
                tno1  = Spe_Total_CNO[wanA];
                int use_iDM = (SO_switch == 1 || Hub_U_switch == 1 || 1 <= Constraint_NCS_switch ||
                               Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1);

                const int     n       = 2 * Msize[Mc_AN];
                const double *fweight = FermiW[Mc_AN];
                const double *eweight = EFermiW[Mc_AN];

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                    for (i = 0; i < tno1; i++) {
                        for (j = 0; j < tno2; j++) {
                            double cdm0 = 0.0, cdm1 = 0.0, cdm2 = 0.0, cdm3 = 0.0;
                            double edm0 = 0.0, edm1 = 0.0, edm2 = 0.0, edm3 = 0.0;
                            double idm00 = 0.0, idm01 = 0.0;
                            dcomplex *res0 = Residues[0][Mc_AN][h_AN][i][j];
                            dcomplex *res1 = Residues[1][Mc_AN][h_AN][i][j];
                            dcomplex *res2 = Residues[2][Mc_AN][h_AN][i][j];

                            DC_DotDensityResidueNonCol(n, res0, res1, res2, fweight, eweight, use_iDM,
                                                       &cdm0, &edm0, &cdm1, &edm1, &cdm2, &edm2, &cdm3, &edm3,
                                                       &idm00, &idm01);

                            CDM[0][Mc_AN][h_AN][i][j] = cdm0;
                            EDM[0][Mc_AN][h_AN][i][j] = edm0;
                            CDM[1][Mc_AN][h_AN][i][j] = cdm1;
                            EDM[1][Mc_AN][h_AN][i][j] = edm1;
                            CDM[2][Mc_AN][h_AN][i][j] = cdm2;
                            EDM[2][Mc_AN][h_AN][i][j] = edm2;
                            CDM[3][Mc_AN][h_AN][i][j] = cdm3;
                            EDM[3][Mc_AN][h_AN][i][j] = edm3;

                            if (use_iDM) {
                                iDM[0][0][Mc_AN][h_AN][i][j] = idm00;
                                iDM[0][1][Mc_AN][h_AN][i][j] = idm01;
                                omp_tmp[OMPID] +=
                                    +cdm0 * Hks[0][Mc_AN][h_AN][i][j] -
                                    idm00 * ImNL[0][Mc_AN][h_AN][i][j] +
                                    cdm1 * Hks[1][Mc_AN][h_AN][i][j] -
                                    idm01 * ImNL[1][Mc_AN][h_AN][i][j] +
                                    2.0 * cdm2 * Hks[2][Mc_AN][h_AN][i][j] -
                                    2.0 * cdm3 * (Hks[3][Mc_AN][h_AN][i][j] + ImNL[2][Mc_AN][h_AN][i][j]);
                            } else {
                                omp_tmp[OMPID] +=
                                    +cdm0 * Hks[0][Mc_AN][h_AN][i][j] +
                                    cdm1 * Hks[1][Mc_AN][h_AN][i][j] +
                                    2.0 * cdm2 * Hks[2][Mc_AN][h_AN][i][j] -
                                    2.0 * cdm3 * Hks[3][Mc_AN][h_AN][i][j];
                            }
                        }
                    }
                }

                dtime(&Etime_atom);
                time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
            }
        }

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            free(FermiW[Mc_AN]);
        }
        free(EFermiW);
        free(FermiW);

        /****************************************************
                         bond energies
        ****************************************************/

        My_Eele1[0] = 0.0;
        for (ompi = 0; ompi < Nthrds0; ompi++) {
            My_Eele1[0] += omp_tmp[ompi];
        }

        /* MPI, My_Eele1 */
        MPI_Allreduce(&My_Eele1[0], &Eele1[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

        if (3 <= level_stdout && myid == Host_ID) {
            printf("Eele0=%15.12f\n", Eele0[0]);
            printf("Eele1=%15.12f\n", Eele1[0]);
        }

    } /* if ( strcasecmp(mode,"scf")==0 ) */

    else if (strcasecmp(mode, "dos") == 0) {
        Save_DOS_NonCol(Residues, OLP0, EVal, Msize);
    }

    // dtime(&timeH);
    // printf("timeG = %.3f\n", timeH - timeG);
    /****************************************************
      freeing of arrays:

    ****************************************************/

    free(Snd_H_Size);
    free(Rcv_H_Size);

    free(Snd_iHNL_Size);
    free(Rcv_iHNL_Size);

    free(Snd_S_Size);
    free(Rcv_S_Size);

    free(Msize);

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
        free(EVal[Mc_AN]);
    }
    free(EVal);

    for (spin = 0; spin < 3; spin++) {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0) {
                Gc_AN   = 0;
                FNAN[0] = 1;
                tno1    = 1;
            } else {
                Gc_AN = M2G[Mc_AN];
                wanA  = WhatSpecies[Gc_AN];
                tno1  = Spe_Total_CNO[wanA];
            }

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                if (Mc_AN == 0) {
                    tno2 = 1;
                } else {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                }

                for (i = 0; i < tno1; i++) {
                    for (j = 0; j < tno2; j++) {
                        free(Residues[spin][Mc_AN][h_AN][i][j]);
                    }
                    free(Residues[spin][Mc_AN][h_AN][i]);
                }
                free(Residues[spin][Mc_AN][h_AN]);
            }
            free(Residues[spin][Mc_AN]);
        }
        free(Residues[spin]);
    }
    free(Residues);

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
        free(PDOS_DC[Mc_AN]);
    }
    free(PDOS_DC);

    free(omp_tmp);

    /* for time */
    dtime(&TEtime);
    time0 = TEtime - TStime;

    /* for PrintMemory */
    firsttime = 0;

    return time0;
}

void Save_DOS_Col(double ****** Residues, double **** OLP0, double *** EVal, int * Msize)
{
    int    spin, Mc_AN, wanA, Gc_AN, tno1;
    int    i1, i, j, MaxL, l, h_AN, Gh_AN, wanB, tno2;
    double Stime_atom, Etime_atom;
    double sum;
    int    i_vec[10];
    char   file_eig[YOUSO10], file_ev[YOUSO10];
    FILE * fp_eig, *fp_ev;
    int    numprocs, myid, ID, tag;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    /* MPI */

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    /* open file pointers */

    if (myid == Host_ID) {

        sprintf(file_eig, "%s%s.Dos.val", filepath, filename);
        if ((fp_eig = fopen(file_eig, "w")) == NULL) {
            printf("cannot open a file %s\n", file_eig);
        }
    }

    sprintf(file_ev, "%s%s.Dos.vec%i", filepath, filename, myid);
    if ((fp_ev = fopen(file_ev, "w")) == NULL) {
        printf("cannot open a file %s\n", file_ev);
    }

    /****************************************************
                     save *.Dos.vec
    ****************************************************/

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

            dtime(&Stime_atom);

            Gc_AN = M2G[Mc_AN];
            wanA  = WhatSpecies[Gc_AN];
            tno1  = Spe_Total_CNO[wanA];

            fprintf(fp_ev, "<AN%dAN%d\n", Gc_AN, spin);
            fprintf(fp_ev, "%d\n", Msize[Mc_AN]);

            for (i1 = 0; i1 < Msize[Mc_AN]; i1++) {

                fprintf(fp_ev, "%4d  %10.6f  ", i1, EVal[spin][Mc_AN][i1]);

                for (i = 0; i < tno1; i++) {

                    sum = 0.0;
                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB  = WhatSpecies[Gh_AN];
                        tno2  = Spe_Total_CNO[wanB];
                        for (j = 0; j < tno2; j++) {
                            sum += Residues[spin][Mc_AN][h_AN][i][j][i1] * OLP0[Mc_AN][h_AN][i][j];
                        }
                    }

                    fprintf(fp_ev, "%8.5f", sum);
                }

                fprintf(fp_ev, "\n");
            }

            fprintf(fp_ev, "AN%dAN%d>\n", Gc_AN, spin);

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
        }
    }

    /****************************************************
                     save *.Dos.val
    ****************************************************/

    if (myid == Host_ID) {

        fprintf(fp_eig, "mode        5\n");
        fprintf(fp_eig, "NonCol      0\n");
        /*      fprintf(fp_eig,"N           %d\n",n); */
        fprintf(fp_eig, "Nspin       %d\n", SpinP_switch);
        fprintf(fp_eig, "Erange      %lf %lf\n", Dos_Erange[0], Dos_Erange[1]);
        fprintf(fp_eig, "atomnum     %d\n", atomnum);

        fprintf(fp_eig, "<WhatSpecies\n");
        for (i = 1; i <= atomnum; i++) {
            fprintf(fp_eig, "%d ", WhatSpecies[i]);
        }
        fprintf(fp_eig, "\nWhatSpecies>\n");

        fprintf(fp_eig, "SpeciesNum     %d\n", SpeciesNum);
        fprintf(fp_eig, "<Spe_Total_CNO\n");
        for (i = 0; i < SpeciesNum; i++) {
            fprintf(fp_eig, "%d ", Spe_Total_CNO[i]);
        }
        fprintf(fp_eig, "\nSpe_Total_CNO>\n");

        MaxL = Supported_MaxL;
        fprintf(fp_eig, "MaxL           %d\n", Supported_MaxL);
        fprintf(fp_eig, "<Spe_Num_CBasis\n");
        for (i = 0; i < SpeciesNum; i++) {
            for (l = 0; l <= MaxL; l++) {
                fprintf(fp_eig, "%d ", Spe_Num_CBasis[i][l]);
            }
            fprintf(fp_eig, "\n");
        }
        fprintf(fp_eig, "Spe_Num_CBasis>\n");
        fprintf(fp_eig, "ChemP       %lf\n", ChemP);
    }

    /* close file pointers */

    if (myid == Host_ID) {
        if (fp_eig)
            fclose(fp_eig);
    }

    if (fp_ev)
        fclose(fp_ev);
}

void Save_DOS_NonCol(dcomplex ****** Residues, double **** OLP0, double ** EVal, int * Msize)
{
    int    spin, Mc_AN, wanA, Gc_AN, tno1;
    int    i1, i, j, MaxL, l, h_AN, Gh_AN, wanB, tno2;
    double Stime_atom, Etime_atom;
    double tmp1, tmp2, tmp3, sum, SDup, SDdn;
    double Re11, Re22, Re12, Im12;
    double theta, phi, sit, cot, sip, cop;
    int    i_vec[10];
    char   file_eig[YOUSO10], file_ev[YOUSO10];
    FILE * fp_eig, *fp_ev;
    int    numprocs, myid, ID, tag;

    /* for OpenMP */
    int OMPID, Nthrds, Nprocs;

    /* MPI */

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    /* open file pointers */

    if (myid == Host_ID) {

        sprintf(file_eig, "%s%s.Dos.val", filepath, filename);
        if ((fp_eig = fopen(file_eig, "w")) == NULL) {
            printf("cannot open a file %s\n", file_eig);
        }
    }

    sprintf(file_ev, "%s%s.Dos.vec%i", filepath, filename, myid);
    if ((fp_ev = fopen(file_ev, "w")) == NULL) {
        printf("cannot open a file %s\n", file_ev);
    }

    /****************************************************
                     save *.Dos.vec
    ****************************************************/

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

        dtime(&Stime_atom);

        Gc_AN = M2G[Mc_AN];
        wanA  = WhatSpecies[Gc_AN];
        tno1  = Spe_Total_CNO[wanA];

        theta = Angle0_Spin[Gc_AN];
        phi   = Angle1_Spin[Gc_AN];

        sit = sin(theta);
        cot = cos(theta);
        sip = sin(phi);
        cop = cos(phi);

        fprintf(fp_ev, "<AN%d\n", Gc_AN);
        fprintf(fp_ev, "%d %d\n", 2 * Msize[Mc_AN], 2 * Msize[Mc_AN]);

        for (i1 = 0; i1 < 2 * Msize[Mc_AN]; i1++) {

            fprintf(fp_ev, "%4d  %10.6f %10.6f ", i1, EVal[Mc_AN][i1], EVal[Mc_AN][i1]);

            for (i = 0; i < tno1; i++) {

                Re11 = 0.0;
                Re22 = 0.0;
                Re12 = 0.0;
                Im12 = 0.0;

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];

                    for (j = 0; j < tno2; j++) {

                        Re11 += Residues[0][Mc_AN][h_AN][i][j][i1].r * OLP0[Mc_AN][h_AN][i][j];

                        Re22 += Residues[1][Mc_AN][h_AN][i][j][i1].r * OLP0[Mc_AN][h_AN][i][j];

                        Re12 += Residues[2][Mc_AN][h_AN][i][j][i1].r * OLP0[Mc_AN][h_AN][i][j];

                        Im12 += Residues[2][Mc_AN][h_AN][i][j][i1].i * OLP0[Mc_AN][h_AN][i][j];
                    }
                }

                tmp1 = 0.5 * (Re11 + Re22);
                tmp2 = 0.5 * cot * (Re11 - Re22);
                tmp3 = (Re12 * cop - Im12 * sip) * sit;

                SDup = tmp1 + tmp2 + tmp3;
                SDdn = tmp1 - tmp2 - tmp3;

                fprintf(fp_ev, "%8.5f %8.5f ", SDup, SDdn);
            }
            fprintf(fp_ev, "\n");
        }

        fprintf(fp_ev, "AN%d>\n", Gc_AN);

        dtime(&Etime_atom);
        time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    /****************************************************
                     save *.Dos.val
    ****************************************************/

    if (myid == Host_ID) {

        fprintf(fp_eig, "mode        5\n");
        fprintf(fp_eig, "NonCol      1\n");
        /*      fprintf(fp_eig,"N           %d\n",n); */
        fprintf(fp_eig, "Nspin       %d\n", 1); /* switch to 1 */
        fprintf(fp_eig, "Erange      %lf %lf\n", Dos_Erange[0], Dos_Erange[1]);
        fprintf(fp_eig, "atomnum     %d\n", atomnum);

        fprintf(fp_eig, "<WhatSpecies\n");
        for (i = 1; i <= atomnum; i++) {
            fprintf(fp_eig, "%d ", WhatSpecies[i]);
        }
        fprintf(fp_eig, "\nWhatSpecies>\n");

        fprintf(fp_eig, "SpeciesNum     %d\n", SpeciesNum);
        fprintf(fp_eig, "<Spe_Total_CNO\n");
        for (i = 0; i < SpeciesNum; i++) {
            fprintf(fp_eig, "%d ", Spe_Total_CNO[i]);
        }
        fprintf(fp_eig, "\nSpe_Total_CNO>\n");

        MaxL = Supported_MaxL;
        fprintf(fp_eig, "MaxL           %d\n", Supported_MaxL);
        fprintf(fp_eig, "<Spe_Num_CBasis\n");
        for (i = 0; i < SpeciesNum; i++) {
            for (l = 0; l <= MaxL; l++) {
                fprintf(fp_eig, "%d ", Spe_Num_CBasis[i][l]);
            }
            fprintf(fp_eig, "\n");
        }
        fprintf(fp_eig, "Spe_Num_CBasis>\n");
        fprintf(fp_eig, "ChemP       %lf\n", ChemP);

        fprintf(fp_eig, "<SpinAngle\n");
        for (i = 1; i <= atomnum; i++) {
            fprintf(fp_eig, "%lf %lf\n", Angle0_Spin[i], Angle1_Spin[i]);
        }
        fprintf(fp_eig, "SpinAngle>\n");
    }

    /* close file pointers */

    if (myid == Host_ID) {
        if (fp_eig)
            fclose(fp_eig);
    }

    if (fp_ev)
        fclose(fp_ev);
}
