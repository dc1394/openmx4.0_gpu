/**********************************************************************
  Divide_Conquer.c:

     Divide_Conquer.c is a subroutine to perform a divide and conquer
     method for eigenvalue problem

  Log of Divide_Conquer.c:

     10/Dec/2003  Released by T.Ozaki

***********************************************************************/

#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_solver_workspace.h"
#include "openmx_native_resources.h"
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
/* MAGMA's allocation failures (magma_types.h) */
#define DC_MAGMA_ERR_HOST_ALLOC   (-112)
#define DC_MAGMA_ERR_DEVICE_ALLOC (-113)

/* Persistent metadata belongs to the explicit calculation owner. */
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

typedef struct {
    int state;
    int matomnum;
    int *dim;
    int *valid;
    size_t *off;
    double *d_arena;
    size_t arena_elems;
} DCColSCacheCtx;

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

struct OpenMXDCState {
  OpenMXNativeResources *resources; /* borrowed from this calculation */
  int initialized;
  DCNonColGpuCtx noncol_gpu;
  DCColSCacheCtx scache;
  int gpusolver_native_approved;
  int gpusolver_approved_gemm_num1;
  int gpusolver_approved_gemm_n;
  int gpusolver_eigen_disabled;
  int gpusolver_gemmul8_disabled;
  int gpusolver_gemm_disabled;
  DCGpuSolverCtx gpusolver_ctx;
  int NonCol_owned_firsttime;
  int Col_owned_firsttime;
  int DCNonCol_GpuAdmit_warned;
  int GpuSolver_RankAllowed_allowed;
  int GpuSolver_RankAllowed_initialized;
  int NodeRanks_ranks;
  int NonCol_GPU_Threshold_threshold;
  int NonCol_GPU_Threshold_initialized;
  int GPU_Threshold_threshold;
  int GPU_Threshold_initialized;
};
typedef OpenMXDCState DCState;

static double DC_Col(DCState *state, char * mode, int SCF_iter, double ***** Hks, double **** OLP0, double ***** CDM, double ***** EDM,
                     double Eele0[2], double Eele1[2], const OpenMXLocalBasisViews *local_basis);

static double DC_NonCol(DCState *state, char * mode, int SCF_iter, double ***** Hks, double ***** ImNL, double **** OLP0,
                        double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2], const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis);

static void Save_DOS_Col(OpenMXSolverResidueView Residues, double **** OLP0, OpenMXSolverRealFieldView EVal, int * Msize);
static void Save_DOS_NonCol(OpenMXSolverResidueView Residues, double **** OLP0, OpenMXSolverRealFieldView EVal, int * Msize);

/* GPU/CPU crossover for the per-atom truncated-cluster eigensolves;
   collinear compares NUM, noncollinear compares 2*NUM (spinor dimension) */
#define DC_GPU_CPU_SWITCH_NUM         400
#define DC_NONCOL_GPU_CPU_SWITCH_NUM  300

static int DC_GPU_Threshold(DCState *state)
{



    if (!state->GPU_Threshold_initialized) {
        char *env = openmx_env_get("OPENMX_DC_GPU_THRESHOLD");
        if (env != NULL && env[0] != '\0') {
            int env_threshold = atoi(env);
            if (0 < env_threshold) {
                state->GPU_Threshold_threshold = env_threshold;
            }
        }
        state->GPU_Threshold_initialized = 1;
    }

    return state->GPU_Threshold_threshold;
}

static int DC_NonCol_GPU_Threshold(DCState *state)
{



    if (!state->NonCol_GPU_Threshold_initialized) {
        char *env = openmx_env_get("OPENMX_DC_NONCOL_GPU_THRESHOLD");
        if (env != NULL && env[0] != '\0') {
            int env_threshold = atoi(env);
            if (0 < env_threshold) {
                state->NonCol_GPU_Threshold_threshold = env_threshold;
            }
        }
        state->NonCol_GPU_Threshold_initialized = 1;
    }

    return state->NonCol_GPU_Threshold_threshold;
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

static void *DC_MallocArray(DCState *state, size_t count, size_t elem_size, const char *label)
{
    size_t bytes = DC_CheckedArrayBytes(count, elem_size, label);
    void * ptr   = openmx_core_native_malloc(state->resources,1,bytes == 0 ? 1 : bytes,0);

    if (ptr == NULL) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Out of memory in Divide_Conquer.c: %s (%zu bytes)", label, bytes);
        DC_AbortWithMessage(msg);
    }

    return ptr;
}





/* the GEMMul8 products of DC are off by default on AMD GPUs; the run
   boundary resets the flag to this value, not to 0 */
#ifdef __HIP_PLATFORM_AMD__
#define DC_GPUSOLVER_GEMMUL8_DISABLED_DEFAULT 1
#else
#define DC_GPUSOLVER_GEMMUL8_DISABLED_DEFAULT 0
#endif


/* Once these monotonically growing workspaces have passed a memory
   preflight, equal or smaller per-atom solves need no further query. */








/* A slot starts on a 512-byte boundary like a hipMalloc'd matrix, so a
   solve that reads it in place sees the operand alignment it had before. */
#define DC_SCACHE_SLOT_ALIGN 64

static int DC_GpuSolver_RankAllowed(DCState *state);
static int DC_GpuSolver_GemmDisabled(DCState *state);
static int DC_GpuSolver_EigenDisabled(DCState *state);

void Divide_Conquer_Release_GPU_SCache(DCState *state)
{
    if (state->scache.d_arena != NULL) wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,state->scache.d_arena));
    openmx_core_native_free(state->resources,state->scache.dim);
    openmx_core_native_free(state->resources,state->scache.valid);
    openmx_core_native_free(state->resources,state->scache.off);
    memset(&state->scache, 0, sizeof(state->scache));
}

/* Ranks sharing this node's device(s); conservative on multi-GPU nodes.
   The split is collective, so the first call must be reached by every rank. */
static int DC_NodeRanks(DCState *state)
{


    if (state->NodeRanks_ranks == 0) {
        MPI_Comm node_comm = MPI_COMM_NULL;

        MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
        MPI_Comm_size(node_comm, &state->NodeRanks_ranks);
        MPI_Comm_free(&node_comm);
        if (state->NodeRanks_ranks < 1)
            state->NodeRanks_ranks = 1;
    }

    return state->NodeRanks_ranks;
}

static void DCCol_SCache_Prepare(DCState *state, int matomnum, const int *Msize, int scf_iter)
{
    const char *enabled_env = openmx_env_get("OPENMX_DC_GPU_S_CACHE");
    const char *reserve_env = openmx_env_get("OPENMX_DC_GPU_S_CACHE_RESERVE_MB");
    size_t free_bytes = 0, total_bytes = 0, reserve = (size_t)4096 * 1024 * 1024;
    size_t used = 0, budget;
    int threshold = DC_GPU_Threshold(state);
    /* collective on the first call, so it precedes every rank-local return */
    int node_ranks = DC_NodeRanks(state);
    int admitted = 0;

    if (enabled_env != NULL && atoi(enabled_env) == 0) return;
    if (reserve_env != NULL) {
        long mib = atol(reserve_env);
        if (0 <= mib) reserve = (size_t)mib * 1024U * 1024U;
    }
    /* a rank that never runs the GPU solve (rank limit, or the path is off
       for the rest of the run) holds no arena */
    if (!DC_GpuSolver_RankAllowed(state) || DC_GpuSolver_GemmDisabled(state) || DC_GpuSolver_EigenDisabled(state)) {
        Divide_Conquer_Release_GPU_SCache(state);
        return;
    }
    if (state->scache.state == 1 && state->scache.matomnum == matomnum) {
        int same = 1;
        for (int a=1; a<=matomnum; a++)
            if (state->scache.dim[a] != 0 && state->scache.dim[a] != Msize[a]) same = 0;
        if (same) {
            if (scf_iter <= 2) memset(state->scache.valid, 0, sizeof(int)*(size_t)(matomnum+1));
            return;
        }
        Divide_Conquer_Release_GPU_SCache(state);
    }
    if (state->scache.state == -1) return;

    state->scache.dim = (int*)openmx_core_native_malloc(state->resources,(size_t)matomnum+1,sizeof(int),1);
    state->scache.valid = (int*)openmx_core_native_malloc(state->resources,(size_t)matomnum+1,sizeof(int),1);
    state->scache.off = (size_t*)openmx_core_native_malloc(state->resources,(size_t)matomnum+1,sizeof(size_t),1);
    if (state->scache.dim == NULL || state->scache.valid == NULL || state->scache.off == NULL)
        DC_AbortWithMessage("Could not allocate DC GPU S-cache tables.");

    if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        (void)hipGetLastError();
        Divide_Conquer_Release_GPU_SCache(state);
        state->scache.state = -1;
        return;
    }
    budget = free_bytes > reserve ? (free_bytes-reserve)/(size_t)node_ranks : 0;
    for (int a=1; a<=matomnum; a++) {
        size_t elems = DC_CheckedMulCount((size_t)Msize[a], (size_t)Msize[a], "DC S-cache slot");
        size_t bytes = DC_CheckedArrayBytes(elems, sizeof(double), "DC S-cache slot");
        if (Msize[a] < threshold || bytes > budget || used*sizeof(double) > budget-bytes) continue;
        state->scache.dim[a] = Msize[a];
        state->scache.off[a] = used;
        used += (elems + DC_SCACHE_SLOT_ALIGN - 1) / DC_SCACHE_SLOT_ALIGN * DC_SCACHE_SLOT_ALIGN;
        admitted++;
    }
    if (used != 0 && OpenMX_Resource_GpuMalloc(state->resources,(void**)&state->scache.d_arena,used*sizeof(double)) != hipSuccess) {
        /* the run goes on without the cache: no latched error may remain
           for the next hipGetLastError() caller */
        state->scache.d_arena = NULL;
        (void)hipGetLastError();
        Divide_Conquer_Release_GPU_SCache(state);
        state->scache.state = -1;
        return;
    }
    state->scache.matomnum = matomnum;
    state->scache.arena_elems = used;
    state->scache.state = 1;

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

static double *DCCol_SCache_Ptr(DCState *state, int Mc_AN, int n)
{
    if (state->scache.state != 1 || Mc_AN < 1 || state->scache.matomnum < Mc_AN ||
        state->scache.dim[Mc_AN] != n) return NULL;
    return state->scache.d_arena + state->scache.off[Mc_AN];
}

/* the slot of Mc_AN when it holds this geometry's transformed overlap */
static const double *DCCol_SCache_ValidPtr(DCState *state, int Mc_AN, int n)
{
    double *slot = DCCol_SCache_Ptr(state, Mc_AN, n);

    return (slot != NULL && state->scache.valid[Mc_AN]) ? slot : NULL;
}

static unsigned DC_EnvU32(const char *name, unsigned fallback)
{
    const char *value = openmx_env_get(name);
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
        value = openmx_env_get(name1);
    if ((value == NULL || value[0] == '\0') && name2 != NULL)
        value = openmx_env_get(name2);
    if ((value == NULL || value[0] == '\0') && name3 != NULL)
        value = openmx_env_get(name3);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }

    parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }

    return (int)parsed;
}

static int DC_GpuSolver_RankAllowed(DCState *state)
{



    if (!state->GpuSolver_RankAllowed_initialized) {
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

        state->GpuSolver_RankAllowed_allowed = (0 < max_ranks && (unsigned)(local_rank / device_count) < max_ranks);
        state->GpuSolver_RankAllowed_initialized = 1;
    }

    return state->GpuSolver_RankAllowed_allowed;
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

static void DC_GpuSolver_DisableGemmPathHip(DCState *state, const char *where, hipError_t status);

static void DC_Eigen_lapack_cpu(double **a, double *ko, int n, int EVmax)
{
    int saved_scf_eigen_lib_flag = scf_eigen_lib_flag;

    scf_eigen_lib_flag = ELPA2;
    Eigen_lapack(a, ko, n, EVmax);
    scf_eigen_lib_flag = saved_scf_eigen_lib_flag;
}

static int DC_GpuSolver_GemmDisabled(DCState *state)
{
    return state->gpusolver_gemm_disabled;
}

static int DC_GpuSolver_EigenDisabled(DCState *state)
{
    return state->gpusolver_eigen_disabled;
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

static void DC_GpuSolver_ReleaseGemmScratch(DCState *state)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;

    if (ctx->d_H != NULL) {
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_H));
        ctx->d_H = NULL;
    }
    if (ctx->d_tmp != NULL) {
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_tmp));
        ctx->d_tmp = NULL;
    }
    if (ctx->d_A != NULL) {
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_A));
        ctx->d_A = NULL;
    }
    if (ctx->d_C != NULL) {
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_C));
        ctx->d_C = NULL;
    }
}

static void DC_GpuSolver_ReleaseTransformScratch(DCState *state)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;

    if (ctx->d_H != NULL) {
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_H));
        ctx->d_H = NULL;
    }
    if (ctx->d_tmp != NULL) {
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_tmp));
        ctx->d_tmp = NULL;
    }
}

static void DC_GpuSolver_ReleaseOverlapBuffer(DCState *state)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;

    if (ctx->d_S != NULL) {
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_S));
        ctx->d_S = NULL;
    }
    ctx->loaded_s_dim = 0;
}

static void DC_GpuSolver_Destroy(DCState *state);
static void DCNonCol_GpuRelease(DCState *state, int keep_stage);

/* Run-boundary reset (between -runtest/-runtestL inputs): drop every scratch
   buffer and the overlap cache, all sized and indexed by the previous
   system, and return the stream, the hipBLAS handle and the pinned staging
   buffers too; the next GPU solve re-creates them on demand
   (DC_GpuSolver_Init).  The S-cache is not part of the context and is
   dropped on its own.  Idempotent. */
static void Divide_Conquer_Release_GPU_Caches(DCState *state)
{
    DCNonCol_GpuRelease(state, 0);
    DC_GpuSolver_Destroy(state);
    Divide_Conquer_Release_GPU_SCache(state);

    /* A memory admission failure belongs to the finished system: the next
       input starts with the GPU paths enabled again (and the GEMMul8
       products at their build default). */
    state->gpusolver_gemm_disabled = 0;
    state->gpusolver_gemmul8_disabled = DC_GPUSOLVER_GEMMUL8_DISABLED_DEFAULT;
    state->gpusolver_eigen_disabled = 0;
    state->gpusolver_approved_gemm_n = 0;
    state->gpusolver_approved_gemm_num1 = 0;
    state->gpusolver_native_approved = 0;
}

static void DC_GpuSolver_DisableEigenPathInfo(DCState *state, const char *where, int32_t info)
{
    int rank = -1;

    state->gpusolver_eigen_disabled = 1;
    state->gpusolver_gemm_disabled  = 1;
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

static void DC_GpuSolver_DisableEigenPathEigenpairs(DCState *state, const char *where, int64_t h_meig, int maxn)
{
    int rank = -1;

    state->gpusolver_eigen_disabled = 1;
    state->gpusolver_gemm_disabled  = 1;
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

static int DC_GpuSolver_HasHipblasMemoryForSolve(DCState *state, const char *where)
{
    size_t      reserve_bytes;
    size_t      free_bytes = 0, total_bytes = 0;
    hipError_t status;
    int         rank = -1;

    if (DC_GpuSolver_GemmDisabled(state)) {
        return 0;
    }

    if (state->gpusolver_native_approved) {
        return 1;
    }

    reserve_bytes = DC_GpuSolver_HipblasReserveBytes();
    status        = OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes);
    if (status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipMemGetInfo(native hipBLAS preflight)", status);
        return 0;
    }

    if (free_bytes < reserve_bytes) {
        state->gpusolver_gemm_disabled = 1;
        state->gpusolver_eigen_disabled = 1;
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

    state->gpusolver_native_approved = 1;
    return 1;
}

static int DC_GpuSolver_PrepareGemmBackendForSolve(DCState *state, int n, int num1)
{
    size_t      reserve_bytes, estimate_bytes, estimate2;
    size_t      free_bytes = 0, total_bytes = 0;
    hipError_t status;
    int         rank = -1;

    if (DC_GpuSolver_GemmDisabled(state)) {
        return 0;
    }

    if (state->gpusolver_gemmul8_disabled) {
        return DC_GpuSolver_HasHipblasMemoryForSolve(state, "DC Hamiltonian GEMM");
    }

    if (n <= state->gpusolver_approved_gemm_n &&
        num1 <= state->gpusolver_approved_gemm_num1) {
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
        DC_GpuSolver_DisableGemmPathHip(state, "hipMemGetInfo(GEMM preflight)", status);
        return 0;
    }

    if (free_bytes < reserve_bytes || free_bytes - reserve_bytes < estimate_bytes) {
        state->gpusolver_gemmul8_disabled = 1;
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
        return DC_GpuSolver_HasHipblasMemoryForSolve(state, "DC Hamiltonian GEMM");
    }

    if (state->gpusolver_approved_gemm_n < n)
        state->gpusolver_approved_gemm_n = n;
    if (state->gpusolver_approved_gemm_num1 < num1)
        state->gpusolver_approved_gemm_num1 = num1;
    return 1;
}

static void DC_GpuSolver_DisableGemmPath(DCState *state, const char *where, const char *backend, hipblasStatus_t status, int m, int n,
                                        int k)
{
    int rank = -1;

    state->gpusolver_gemm_disabled = 1;
    state->gpusolver_eigen_disabled = 1;
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

static void DC_GpuSolver_DisableGemmPathHip(DCState *state, const char *where, hipError_t status)
{
    int rank = -1;

    state->gpusolver_gemm_disabled = 1;
    state->gpusolver_eigen_disabled = 1;
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

static int DC_GpuSolver_TryGpuDgemm(DCState *state, hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m,
                                   int n, int k, const double *alpha, const double *A, int lda, const double *B,
                                   int ldb, const double *beta, double *C, int ldc, const char *where)
{
    hipblasStatus_t status;
    int            rank = -1;

    if (DC_GpuSolver_GemmDisabled(state)) {
        return 0;
    }

    if (!state->gpusolver_gemmul8_disabled) {
        status = openmx_gemmul8Dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        if (status == HIPBLAS_STATUS_SUCCESS) {
            return 1;
        }

        state->gpusolver_gemmul8_disabled = 1;
        openmx_gemmul8ReleaseWorkspaces();

        MPI_Comm_rank(mpi_comm_level1, &rank);
        fprintf(stderr,
                "<DC> rank %d: GEMMul8 failed in %s for GEMM(m=%d,n=%d,k=%d): %s (%d). "
                "Retrying this DC Hamiltonian solve with native hipBLAS.\n",
                rank, where, m, n, k, DC_GpuSolver_HipblasStatusName(status), (int)status);
        fflush(stderr);

        if (!DC_GpuSolver_HasHipblasMemoryForSolve(state, where)) {
            return 0;
        }
    }

    status = hipblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        return 1;
    }

    DC_GpuSolver_DisableGemmPath(state, where, "native hipBLAS", status, m, n, k);
    return 0;
}

static void DC_GpuSolver_Destroy(DCState *state)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;

    if (ctx->d_S != NULL)
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_S));
    if (ctx->d_H != NULL)
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_H));
    if (ctx->d_tmp != NULL)
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_tmp));
    if (ctx->d_A != NULL)
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_A));
    if (ctx->d_C != NULL)
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_C));
    if (ctx->h_matrix != NULL)
        wait_hipfunc(OpenMX_Resource_HostFree(state->resources,ctx->h_matrix));
    if (ctx->hipblas != NULL)
        OpenMXBlasWait(OpenMX_Resource_BlasDestroy(state->resources,ctx->hipblas));
    if (ctx->stream != NULL)
        wait_hipfunc(OpenMX_Resource_StreamDestroy(state->resources,ctx->stream));

    memset(ctx, 0, sizeof(*ctx));
    ctx->device_id    = -1;
    ctx->loaded_s_dim = 0;
}

/* Stream and hipBLAS handle, created on demand (also after the run-boundary
   release); a failure turns the GPU solve off, the caller falls back. */
static int DC_GpuSolver_Init(DCState *state)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;
    int            current_device;
    hipError_t     hip_status;
    hipblasStatus_t blas_status;

    hip_status = hipGetDevice(&current_device);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipGetDevice", hip_status);
        return 0;
    }

    if (ctx->initialized && ctx->device_id == current_device) {
        return 1;
    }

    if (ctx->initialized) {
        DC_GpuSolver_Destroy(state);
    }

    hip_status = OpenMX_Resource_StreamCreate(state->resources,&ctx->stream,hipStreamNonBlocking);
    if (hip_status != hipSuccess) {
        ctx->stream = NULL;
        DC_GpuSolver_DisableGemmPathHip(state, "hipStreamCreateWithFlags", hip_status);
        DC_GpuSolver_Destroy(state);
        return 0;
    }
    blas_status = OpenMX_Resource_BlasCreate(state->resources,&ctx->hipblas);
    if (blas_status != HIPBLAS_STATUS_SUCCESS) {
        ctx->hipblas = NULL;
    } else {
        blas_status = hipblasSetStream(ctx->hipblas, ctx->stream);
    }
    if (blas_status != HIPBLAS_STATUS_SUCCESS) {
        DC_GpuSolver_DisableGemmPath(state, "the solver setup", "hipblasCreate/hipblasSetStream", blas_status, 0, 0, 0);
        DC_GpuSolver_Destroy(state);
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
static int DC_GpuSolver_EnsureMatrixCapacity(DCState *state, int num)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;
    size_t         matrix_bytes;
    hipError_t    hip_status;

    if (num <= 0) {
        DC_AbortWithMessage("Invalid matrix size in DC_GpuSolver_EnsureMatrixCapacity.");
    }

    if (!DC_GpuSolver_Init(state)) {
        return 0;
    }

    if (num <= ctx->matrix_dim && ctx->h_matrix != NULL) {
        return 1;
    }

    DC_GpuSolver_ReleaseOverlapBuffer(state);
    DC_GpuSolver_ReleaseGemmScratch(state);
    if (ctx->h_matrix != NULL) {
        wait_hipfunc(OpenMX_Resource_HostFree(state->resources,ctx->h_matrix));
        ctx->h_matrix = NULL;
    }

    ctx->matrix_dim   = 0;
    ctx->loaded_s_dim = 0;

    matrix_bytes = DC_CheckedArrayBytes(DC_CheckedMulCount((size_t)num, (size_t)num, "GPU solver dense matrix"),
                                        sizeof(double), "GPU solver dense matrix");

    hip_status = OpenMX_Resource_HostAlloc(state->resources,(void **)&ctx->h_matrix,matrix_bytes,0);
    if (hip_status != hipSuccess) {
        ctx->h_matrix = NULL;
        DC_GpuSolver_DisableGemmPathHip(state, "hipHostMalloc(h_matrix)", hip_status);
        DC_GpuSolver_Destroy(state);
        return 0;
    }

    ctx->matrix_dim   = num;
    ctx->loaded_s_dim = 0;
    return 1;
}

/* *d_matrix becomes a matrix_dim x matrix_dim device matrix (kept if it
   exists); on failure the GPU path is off and the context is gone. */
static int DC_GpuSolver_AllocMatrix(DCState *state, double **d_matrix, const char *where)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;
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

    hip_status = OpenMX_Resource_GpuMalloc(state->resources,(void **)d_matrix,matrix_bytes);
    if (hip_status != hipSuccess) {
        *d_matrix = NULL;
        DC_GpuSolver_DisableGemmPathHip(state, where, hip_status);
        DC_GpuSolver_Destroy(state);
        return 0;
    }
    return 1;
}

static int DC_GpuSolver_EnsureGemmCapacity(DCState *state, int num)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;

    return DC_GpuSolver_EnsureMatrixCapacity(state, num) && DC_GpuSolver_AllocMatrix(state, &ctx->d_H, "hipMalloc(d_H)") &&
           DC_GpuSolver_AllocMatrix(state, &ctx->d_tmp, "hipMalloc(d_tmp)") &&
           DC_GpuSolver_AllocMatrix(state, &ctx->d_A, "hipMalloc(d_A)");
}

static int DC_GpuSolver_Eigen(DCState *state, double *d_A, int m, int maxn, double *W, const char *where)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;
    int             info;
    int             mout = 0;
    hipError_t     hip_status;

    if (m <= 0 || maxn <= 0 || maxn > m || d_A == NULL || W == NULL) {
        DC_AbortWithMessage("Invalid eigensolver arguments in DC_GpuSolver_Eigen.");
    }

    if (DC_GpuSolver_EigenDisabled(state) || !DC_GpuSolver_Init(state)) {
        return 0;
    }

    hip_status = hipStreamSynchronize(ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipStreamSynchronize(MAGMA eigensolver input)", hip_status);
        return 0;
    }

    info = openmx_magma_dsyevdx_gpu(m, maxn, d_A, W, &mout);
    if (info != 0) {
        DC_GpuSolver_DisableEigenPathInfo(state, where, info);
        return 0;
    }
    if (mout != maxn) {
        DC_GpuSolver_DisableEigenPathEigenpairs(state, where, (int64_t)mout, maxn);
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
static int DC_GpuSolver_DeviceTranspose(DCState *state, int rows_out, int cols_out, const double *d_in, int ld_in, double *d_out,
                                        int ld_out, const char *where)
{
    double          alpha = 1.0;
    double          beta  = 0.0;
    hipblasStatus_t status;

    status = hipblasDgeam(state->gpusolver_ctx.hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, rows_out, cols_out, &alpha, d_in,
                          ld_in, &beta, d_out, ld_out, d_out, ld_out);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        DC_GpuSolver_DisableGemmPath(state, where, "hipblasDgeam", status, rows_out, cols_out, 0);
        return 0;
    }

    return 1;
}

/* rows of a host matrix -> column-major device matrix d_out, through the
   pinned buffer and the device scratch d_scratch */
static int DCCol_GpuSolver_UploadMatrix(DCState *state, int n, double **src, double *d_scratch, double *d_out, const char *where)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;
    hipError_t      hip_status;

    DCCol_GpuSolver_PackRows(n, src, ctx->h_matrix);
    hip_status = hipMemcpyAsync(d_scratch, ctx->h_matrix, sizeof(double) * (size_t)n * (size_t)n,
                                hipMemcpyHostToDevice, ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, where, hip_status);
        return 0;
    }

    return DC_GpuSolver_DeviceTranspose(state, n, n, d_scratch, n, d_out, n, where);
}

static int DCCol_GpuSolver_LoadTransformedOverlap(DCState *state, int n, double **S_DC)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;
    hipError_t      hip_status;

    if (!DC_GpuSolver_EnsureMatrixCapacity(state, n) || !DC_GpuSolver_AllocMatrix(state, &ctx->d_S, "hipMalloc(d_S)") ||
        !DC_GpuSolver_AllocMatrix(state, &ctx->d_tmp, "hipMalloc(d_tmp)")) {
        return 0;
    }

    if (!DCCol_GpuSolver_UploadMatrix(state, n, S_DC, ctx->d_tmp, ctx->d_S, "transformed overlap staging")) {
        return 0;
    }
    /* h_matrix is restaged by the Hamiltonian solve */
    hip_status = hipStreamSynchronize(ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipStreamSynchronize(transformed overlap)", hip_status);
        return 0;
    }

    ctx->loaded_s_dim = n;
    return 1;
}

/* Refresh Mc_AN's S-cache slot from the overlap just staged in ctx->d_S.
   *d_S_use is the slot for the solve to read in place, or NULL when the
   atom has no slot (the solve then uses, and reloads, ctx->d_S). */
static int DCCol_SCache_Store(DCState *state, int Mc_AN, int n, const double **d_S_use)
{
    DCGpuSolverCtx *ctx  = &state->gpusolver_ctx;
    double *        slot = DCCol_SCache_Ptr(state, Mc_AN, n);
    hipError_t      hip_status;

    *d_S_use = NULL;
    if (slot == NULL) {
        return 1;
    }

    hip_status = hipMemcpyAsync(slot, ctx->d_S, sizeof(double) * (size_t)n * (size_t)n, hipMemcpyDeviceToDevice,
                                ctx->stream);
    if (hip_status != hipSuccess) {
        state->scache.valid[Mc_AN] = 0;
        DC_GpuSolver_DisableGemmPathHip(state, "hipMemcpyAsync(S-cache refresh)", hip_status);
        return 0;
    }

    state->scache.valid[Mc_AN] = 1;
    *d_S_use = slot;
    return 1;
}

static int DCCol_GpuSolver_DiagonalizeOverlap(DCState *state, int n, double **S_DC, double *ko)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;
    hipError_t      hip_status;

    if (!DC_GpuSolver_EnsureMatrixCapacity(state, n) || !DC_GpuSolver_AllocMatrix(state, &ctx->d_S, "hipMalloc(d_S)") ||
        !DC_GpuSolver_AllocMatrix(state, &ctx->d_tmp, "hipMalloc(d_tmp)")) {
        return 0;
    }
    ctx->loaded_s_dim = 0;

    if (!DCCol_GpuSolver_UploadMatrix(state, n, S_DC, ctx->d_tmp, ctx->d_S, "overlap staging")) {
        return 0;
    }
    /* MAGMA gets the room of the staging scratch */
    hip_status = hipStreamSynchronize(ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipStreamSynchronize(overlap staging)", hip_status);
        return 0;
    }
    DC_GpuSolver_ReleaseTransformScratch(state);

    if (!DC_GpuSolver_Eigen(state, ctx->d_S, n, n, ko + 1, "overlap diagonalization")) {
        return 0;
    }

    if (!DC_GpuSolver_AllocMatrix(state, &ctx->d_tmp, "hipMalloc(d_tmp)") ||
        !DC_GpuSolver_DeviceTranspose(state, n, n, ctx->d_S, n, ctx->d_tmp, n, "overlap eigenvector readback")) {
        return 0;
    }
    hip_status = hipMemcpyAsync(ctx->h_matrix, ctx->d_tmp, sizeof(double) * (size_t)n * (size_t)n,
                                hipMemcpyDeviceToHost, ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipMemcpyAsync(overlap eigenvectors)", hip_status);
        return 0;
    }
    hip_status = hipStreamSynchronize(ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipStreamSynchronize(overlap diagonalization)", hip_status);
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
static int DCCol_GpuSolver_SolveHamiltonian(DCState *state, int n, int p_min, const double *d_S_use, double **S_DC,
                                           double **H_DC_spin, double *ko, double **C)
{
    DCGpuSolverCtx *ctx = &state->gpusolver_ctx;
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

    if (!DC_GpuSolver_EnsureGemmCapacity(state, n)) {
        return 0;
    }
    if (reload_s) {
        d_S_use = ctx->d_S;
    }

    if (!DCCol_GpuSolver_UploadMatrix(state, n, H_DC_spin, ctx->d_tmp, ctx->d_H, "Hamiltonian staging")) {
        DC_GpuSolver_ReleaseGemmScratch(state);
        return 0;
    }

    if (!DC_GpuSolver_TryGpuDgemm(state, ctx->hipblas, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha, ctx->d_H, n, d_S_use,
                                 n, &beta, ctx->d_tmp, n, "H*S")) {
        DC_GpuSolver_ReleaseGemmScratch(state);
        return 0;
    }
    if (!DC_GpuSolver_TryGpuDgemm(state, ctx->hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, n, &alpha, d_S_use, n,
                                 ctx->d_tmp, n, &beta, ctx->d_H, n, "S^T*H*S")) {
        DC_GpuSolver_ReleaseGemmScratch(state);
        return 0;
    }

    hip_status = hipMemcpy2DAsync(ctx->d_A, sizeof(double) * (size_t)num1,
                                    ctx->d_H + (size_t)(p_min - 1) * (size_t)n + (size_t)(p_min - 1),
                                    sizeof(double) * (size_t)n, sizeof(double) * (size_t)num1, (size_t)num1,
                                    hipMemcpyDeviceToDevice, ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipMemcpy2DAsync(active Hamiltonian)", hip_status);
        DC_GpuSolver_ReleaseGemmScratch(state);
        return 0;
    }
    hip_status = hipStreamSynchronize(ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipStreamSynchronize(active Hamiltonian)", hip_status);
        DC_GpuSolver_ReleaseGemmScratch(state);
        return 0;
    }
    /* MAGMA gets the room of the transform scratch and of the staged
       overlap (a resident S-cache copy stays) */
    DC_GpuSolver_ReleaseTransformScratch(state);
    DC_GpuSolver_ReleaseOverlapBuffer(state);

    if (!DC_GpuSolver_Eigen(state, ctx->d_A, num1, num1, ko + 1, "Hamiltonian diagonalization")) {
        DC_GpuSolver_ReleaseGemmScratch(state);
        return 0;
    }

    if (!DC_GpuSolver_AllocMatrix(state, &ctx->d_C, "hipMalloc(d_C)")) {
        return 0;
    }
    if (reload_s) {
        /* d_C is the staging scratch until the back-transform writes it */
        if (!DC_GpuSolver_AllocMatrix(state, &ctx->d_S, "hipMalloc(d_S reload)")) {
            return 0;
        }
        if (!DCCol_GpuSolver_UploadMatrix(state, n, S_DC, ctx->d_C, ctx->d_S, "overlap reload")) {
            DC_GpuSolver_ReleaseGemmScratch(state);
            return 0;
        }
        ctx->loaded_s_dim = n;
        d_S_use           = ctx->d_S;
    }

    if (!DC_GpuSolver_TryGpuDgemm(state, ctx->hipblas, HIPBLAS_OP_N, HIPBLAS_OP_N, n, num1, num1, &alpha,
                                 d_S_use + (size_t)(p_min - 1) * (size_t)n, n, ctx->d_A, num1, &beta,
                                 ctx->d_C, n, "S*C")) {
        DC_GpuSolver_ReleaseGemmScratch(state);
        return 0;
    }

    /* row-major C (n x num1) for the host rows; d_A is free again */
    if (!DC_GpuSolver_DeviceTranspose(state, num1, n, ctx->d_C, n, ctx->d_A, num1, "eigenvector readback")) {
        DC_GpuSolver_ReleaseGemmScratch(state);
        return 0;
    }
    hip_status = hipMemcpyAsync(ctx->h_matrix, ctx->d_A, sizeof(double) * (size_t)n * (size_t)num1,
                                  hipMemcpyDeviceToHost, ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipMemcpyAsync(C)", hip_status);
        DC_GpuSolver_ReleaseGemmScratch(state);
        return 0;
    }

    hip_status = hipStreamSynchronize(ctx->stream);
    if (hip_status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipStreamSynchronize(C)", hip_status);
        DC_GpuSolver_ReleaseGemmScratch(state);
        return 0;
    }

    DCCol_GpuSolver_UnpackRows(n, num1, ctx->h_matrix, C);

    DC_GpuSolver_ReleaseGemmScratch(state);
    return 1;
}

/* Host copy of the transformed overlap, needed only by the CPU solve; the
   GPU path with an S-cache hit skips it until a fallback actually happens. */
static void DCCol_FillSDCFromS12(int Mc_AN, int n, double **S_DC, const OpenMXLocalBasisViews *local_basis)
{

    int i1, j1;

    for (i1 = 1; i1 <= n; i1++) {
        for (j1 = 1; j1 <= n; j1++) {
            S_DC[i1][j1] = openmx_local_inverse_overlap_row(local_basis, Mc_AN, i1)[j1];
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




static void DCNonCol_GpuRelease(DCState *state, int keep_stage)
{
    DCNonColGpuCtx *nc = &state->noncol_gpu;

    if (nc->d_A != NULL)
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,nc->d_A));
    if (nc->d_Y != NULL)
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,nc->d_Y));
    if (nc->d_T != NULL)
        wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,nc->d_T));
    nc->d_A = NULL;
    nc->d_Y = NULL;
    nc->d_T = NULL;
    nc->num = 0;

    if (!keep_stage) {
        if (nc->h_stage != NULL)
            wait_hipfunc(OpenMX_Resource_HostFree(state->resources,nc->h_stage));
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
static size_t DCNonCol_GpuNeedBytes(DCState *state, int num)
{
    size_t zmat = DCNonCol_ComplexBytes(num);
    size_t gemm = DC_GpuSolver_HipblasReserveBytes();
    size_t need;

    if (!state->gpusolver_gemmul8_disabled) {
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
static int DCNonCol_GpuAdmit(DCState *state, int matomnum, const int *Msize)
{

    MPI_Comm           node_comm = MPI_COMM_NULL, gpu_comm = MPI_COMM_NULL;
    unsigned long long my_need = 0, my_free, group_need = 0, group_free = 0;
    unsigned long long reserve = (unsigned long long)DC_GpuSolver_EigenReserveBytes();
    size_t             free_bytes = 0, total_bytes = 0;
    int                max_num = 0, device = 0, node_rank = 0, gpu_rank = 0, gpu_ranks = 1, Mc_AN, fits;

    state->noncol_gpu.off = 0;

    if (!DC_GpuSolver_GemmDisabled(state) && !DC_GpuSolver_EigenDisabled(state)) {
        for (Mc_AN = 1; Mc_AN <= matomnum; Mc_AN++) {
            if (DC_NonCol_GPU_Threshold(state) <= 2 * Msize[Mc_AN] && max_num < Msize[Mc_AN]) {
                max_num = Msize[Mc_AN];
            }
        }
    }
    if (0 < max_num) {
        my_need = (unsigned long long)DCNonCol_GpuNeedBytes(state, max_num);
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
    if (!fits && 0 < group_need && !state->DCNonCol_GpuAdmit_warned && gpu_rank == 0 && 0 < level_stdout) {
        printf("<DC> GPU device %d: %d rank(s) would need %.3f GiB for the non-collinear solves "
               "(free %.3f GiB, reserve %.3f GiB); solving them on the CPU instead.\n",
               device, gpu_ranks, (double)group_need / (1024.0 * 1024.0 * 1024.0),
               (double)group_free / (1024.0 * 1024.0 * 1024.0), (double)reserve / (1024.0 * 1024.0 * 1024.0));
        fflush(stdout);
        state->DCNonCol_GpuAdmit_warned = 1;
    }

    return fits ? max_num : 0;
}

/* Buffers for clusters up to num; a failure leaves the rest of this call
   on the CPU (the next call is admitted afresh). */
static int DCNonCol_GpuEnsure(DCState *state, int num)
{
    DCNonColGpuCtx *nc      = &state->noncol_gpu;
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

    if (!DC_GpuSolver_Init(state)) {
        return 0;
    }
    DCNonCol_GpuRelease(state, 1);

    if ((status = OpenMX_Resource_GpuMalloc(state->resources,(void **)&nc->d_A,zbytes)) != hipSuccess) {
        nc->d_A = NULL;
        what    = "hipMalloc(non-collinear H)";
    } else if ((status = OpenMX_Resource_GpuMalloc(state->resources,(void **)&nc->d_Y,zbytes)) != hipSuccess) {
        nc->d_Y = NULL;
        what    = "hipMalloc(non-collinear work)";
    } else if ((status = OpenMX_Resource_GpuMalloc(state->resources,(void **)&nc->d_T,dbytes)) != hipSuccess) {
        nc->d_T = NULL;
        what    = "hipMalloc(non-collinear overlap)";
    } else if (nc->h_stage_bytes < sbytes) {
        if (nc->h_stage != NULL)
            wait_hipfunc(OpenMX_Resource_HostFree(state->resources,nc->h_stage));
        nc->h_stage       = NULL;
        nc->h_stage_bytes = 0;
        if ((status = OpenMX_Resource_HostAlloc(state->resources,&nc->h_stage,sbytes,0)) != hipSuccess) {
            nc->h_stage = NULL;
            what        = "hipHostMalloc(non-collinear staging)";
        } else {
            nc->h_stage_bytes = sbytes;
        }
    }

    if (what != NULL) {
        (void)hipGetLastError();
        DCNonCol_GpuRelease(state, 1);
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
static void DCNonCol_GpuReport(DCState *state, const char *what, int code, int num)
{
    int rank = -1;

    if (8 <= state->noncol_gpu.reports) {
        return;
    }
    state->noncol_gpu.reports++;

    MPI_Comm_rank(mpi_comm_level1, &rank);
    fprintf(stderr,
            "<DC> rank %d: GPU solve of a %d x %d non-collinear cluster failed (%s: %d); "
            "solving this atom on the CPU.\n",
            rank, 2 * num, 2 * num, what, code);
    fflush(stderr);
}

/* d_out (dim x dim) = plain transpose of d_in, by the in-place-B mode of
   geam with beta == 0 (d_out's old contents are not read) */
static int DCNonCol_GpuTranspose(DCState *state, int dim, const hipDoubleComplex *d_in, hipDoubleComplex *d_out, const char *where)
{
    const hipDoubleComplex one  = make_hipDoubleComplex(1.0, 0.0);
    const hipDoubleComplex zero = make_hipDoubleComplex(0.0, 0.0);
    hipblasStatus_t        status;

    status = hipblasZgeam(state->gpusolver_ctx.hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, dim, dim, &one, d_in, dim, &zero,
                          d_out, dim, d_out, dim);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        DC_GpuSolver_DisableGemmPath(state, where, "hipblasZgeam", status, dim, dim, 0);
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
static int DCNonCol_GpuTimesT(DCState *state, int num, const hipDoubleComplex *d_in, hipblasOperation_t op_t,
                              hipDoubleComplex *d_out, const char *where)
{
    const double one  = 1.0;
    const double zero = 0.0;
    const int    dim  = 2 * num;
    int          k;

    for (k = 0; k < 2; k++) {
        if (!DC_GpuSolver_TryGpuDgemm(state, state->gpusolver_ctx.hipblas, HIPBLAS_OP_N, op_t, 2 * dim, num, num, &one,
                                     (const double *)(d_in + (size_t)k * (size_t)num * (size_t)dim), 2 * dim,
                                     state->noncol_gpu.d_T, num, &zero,
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
static int DCNonCol_GpuSolve(DCState *state, int num, int capacity, double **S_DC, dcomplex **H_DC, dcomplex **C, double *ko)
{
    DCGpuSolverCtx *  ctx    = &state->gpusolver_ctx;
    DCNonColGpuCtx *  nc     = &state->noncol_gpu;
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
    if (!DCNonCol_GpuEnsure(state, num < capacity ? capacity : num)) {
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
        DC_GpuSolver_DisableGemmPathHip(state, "hipMemcpyAsync(non-collinear overlap)", status);
        return 0;
    }
    for (i = 0; i < dim; i++) {
        memcpy(h_Z + (size_t)i * (size_t)dim, &H_DC[i + 1][1], sizeof(dcomplex) * (size_t)dim);
    }
    status = hipMemcpyAsync(nc->d_A, h_Z, sizeof(hipDoubleComplex) * zcount, hipMemcpyHostToDevice, ctx->stream);
    if (status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipMemcpyAsync(non-collinear Hamiltonian)", status);
        return 0;
    }

    if (!DCNonCol_GpuTimesT(state, num, nc->d_A, HIPBLAS_OP_T, nc->d_Y, "non-collinear H*T") ||
        !DCNonCol_GpuTranspose(state, dim, nc->d_Y, nc->d_A, "non-collinear T^T*H") ||
        !DCNonCol_GpuTimesT(state, num, nc->d_A, HIPBLAS_OP_T, nc->d_Y, "non-collinear T^T*H*T")) {
        return 0;
    }

    /* MAGMA and rocSOLVER run outside ctx->stream */
    status = hipStreamSynchronize(ctx->stream);
    if (status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipStreamSynchronize(non-collinear eigensolver input)", status);
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
            DCNonCol_GpuReport(state, "eigensolver workspace allocation", info, num);
        } else {
            DCNonCol_GpuReport(state, "eigensolver info", info, num);
        }
        return 0;
    }
    if (mout != dim) {
        DCNonCol_GpuReport(state, "eigenpairs returned", mout, num);
        return 0;
    }
    for (i = 1; i <= dim; i++) {
        if (!isfinite(ko[i])) {
            DCNonCol_GpuReport(state, "non-finite eigenvalue", i, num);
            return 0;
        }
    }

    if (!DCNonCol_GpuTranspose(state, dim, nc->d_Y, nc->d_A, "non-collinear eigenvector transpose") ||
        !DCNonCol_GpuTimesT(state, num, nc->d_A, HIPBLAS_OP_N, nc->d_Y, "non-collinear W^T*T^T")) {
        return 0;
    }
    status = hipMemcpyAsync(h_Z, nc->d_Y, sizeof(hipDoubleComplex) * zcount, hipMemcpyDeviceToHost, ctx->stream);
    if (status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipMemcpyAsync(non-collinear eigenvectors)", status);
        return 0;
    }
    status = hipStreamSynchronize(ctx->stream);
    if (status != hipSuccess) {
        DC_GpuSolver_DisableGemmPathHip(state, "hipStreamSynchronize(non-collinear eigenvectors)", status);
        return 0;
    }

    h_check = (const double *)h_Z;
    for (idx = 0; idx < 2 * zcount; idx++) {
        if (!isfinite(h_check[idx])) {
            DCNonCol_GpuReport(state, "non-finite eigenvector element", (int)(idx / 2 / (size_t)dim) + 1, num);
            return 0;
        }
    }
    for (i = 0; i < dim; i++) {
        memcpy(&C[i + 1][1], h_Z + (size_t)i * (size_t)dim, sizeof(dcomplex) * (size_t)dim);
    }

    return 1;
}

double Divide_Conquer(DCState *state, char * mode, int SCF_iter, double ***** Hks, double ***** ImNL, double **** OLP0,
                      double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2], const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis)
{
    double time0;

    /****************************************************
           collinear without spin-orbit coupling
    ****************************************************/

    if ((SpinP_switch == 0 || SpinP_switch == 1) && SO_switch == 0) {
        time0 = DC_Col(state, mode, SCF_iter, Hks, OLP0, CDM, EDM, Eele0, Eele1, local_basis);
    }

    /****************************************************
     non-collinear with and without spin-orbit coupling
    ****************************************************/

    else if (SpinP_switch == 3) {
        time0 = DC_NonCol(state, mode, SCF_iter, Hks, ImNL, OLP0, CDM, EDM, Eele0, Eele1, electronic, local_basis);
    }

    return time0;
}

#pragma optimization_level 2
typedef struct {
  DCState *state;
  char * mode;
  int SCF_iter;
  double ***** Hks;
  double **** OLP0;
  double ***** CDM;
  double ***** EDM;
  double *Eele0;
  double *Eele1;
  const OpenMXLocalBasisViews *local_basis;
} DC_Col_arguments;
static double DC_Col_owned(DCState *state, char * mode, int SCF_iter, double ***** Hks, double **** OLP0, double ***** CDM, double ***** EDM,
                     double Eele0[2], double Eele1[2], const OpenMXLocalBasisViews *local_basis, OpenMXSolverInvocation *solver_owner);
static int DC_Col_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  DC_Col_arguments *a=(DC_Col_arguments*)arguments;
  *result=DC_Col_owned(a->state, a->mode,a->SCF_iter,a->Hks,a->OLP0,a->CDM,a->EDM,a->Eele0,a->Eele1,a->local_basis,owner);
  return 1;
}
static double DC_Col(DCState *state, char * mode, int SCF_iter, double ***** Hks, double **** OLP0, double ***** CDM, double ***** EDM,
                     double Eele0[2], double Eele1[2], const OpenMXLocalBasisViews *local_basis)
{
  DC_Col_arguments arguments={state,mode,SCF_iter,Hks,OLP0,CDM,EDM,Eele0,Eele1,local_basis};
  return OpenMX_Solver_Run(DC_Col_operation,&arguments);
}

#pragma optimization_level 2
static double DC_Col_owned(DCState *state, char * mode, int SCF_iter, double ***** Hks, double **** OLP0, double ***** CDM, double ***** EDM,
                     double Eele0[2], double Eele1[2], const OpenMXLocalBasisViews *local_basis, OpenMXSolverInvocation *solver_owner)
{



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
    OpenMXSolverRealFieldView EVal;
    OpenMXSolverResidueView Residues;
    OpenMXSolverRealFieldView PDOS_DC;
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

    Snd_H_Size=OpenMX_Solver_Int(solver_owner,"Snd_H_Size",((size_t)1) * numprocs);
    Rcv_H_Size=OpenMX_Solver_Int(solver_owner,"Rcv_H_Size",((size_t)1) * numprocs);
    Snd_S_Size=OpenMX_Solver_Int(solver_owner,"Snd_S_Size",((size_t)1) * numprocs);
    Rcv_S_Size=OpenMX_Solver_Int(solver_owner,"Rcv_S_Size",((size_t)1) * numprocs);

    m_size = 0;
    Msize=OpenMX_Solver_Int(solver_owner,"Msize",((size_t)1) * (Matomnum + 1));

    EVal=OpenMX_Solver_Field(solver_owner,((size_t)1) * (SpinP_switch + 1),((size_t)1) * (Matomnum + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {

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
            OpenMX_Solver_FieldResize(EVal,spin,Mc_AN,((size_t)1) * n2);
        }
    }

    if (state->Col_owned_firsttime)
        PrintMemory("Divide_Conquer: EVal", sizeof(double) * m_size, NULL);

    /* needs the cluster sizes Msize[] of this call */
    if (scf_eigen_lib_flag == GPUSOLVER) {
        DCCol_SCache_Prepare(state, Matomnum, Msize, SCF_iter);
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
    Residues=OpenMX_Solver_Residues(solver_owner,((size_t)1) * (SpinP_switch + 1),1,0);
    for (spin = 0; spin < (int)Residues.components; spin++) {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
            n2 = Mc_AN == 0 ? 1 : Msize[Mc_AN] + 2;
            OpenMX_Solver_ResidueResize(Residues,spin,Mc_AN,n2);
            m_size += n2 * (Residues.pair_offsets[Residues.atom_offsets[Mc_AN+1]] -
                             Residues.pair_offsets[Residues.atom_offsets[Mc_AN]]);
        }
    }

    if (state->Col_owned_firsttime)
        PrintMemory("Divide_Conquer: Residues", sizeof(double) * m_size, NULL);

    /****************************************************
      allocation of arrays:

      double PDOS[SpinP_switch+1]
                 [Matomnum+1]
                 [NUM]
    ****************************************************/

    m_size  = 0;
    PDOS_DC=OpenMX_Solver_Field(solver_owner,((size_t)1) * (SpinP_switch + 1),((size_t)1) * (Matomnum + 1));
    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

            if (Mc_AN == 0)
                n2 = 1;
            else
                n2 = Msize[Mc_AN] + 2;

            m_size += n2;
            OpenMX_Solver_FieldResize(PDOS_DC,spin,Mc_AN,((size_t)1) * n2);
        }
    }

    if (state->Col_owned_firsttime)
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

                tmp_array=OpenMX_Solver_Real(solver_owner,"tmp_array",((size_t)1) * size1);

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
                tmp_array2=OpenMX_Solver_Real(solver_owner,"tmp_array2",((size_t)1) * size2);

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
                ;
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                MPI_Wait(&request, &stat);
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

                    tmp_array=OpenMX_Solver_Real(solver_owner,"tmp_array",((size_t)1) * size1);

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
                    tmp_array2=OpenMX_Solver_Real(solver_owner,"tmp_array2",((size_t)1) * size2);

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
                    ;
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
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

        MP=OpenMX_Solver_Int(solver_owner,"MP",((size_t)1) * List_YOUSO[2]);

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

            double *S_DC_store=OpenMX_Solver_Real(solver_owner,"S_DC_store",(size_t)n2 * (size_t)n2);
            S_DC = OpenMX_Solver_RealAlias(solver_owner,"S_DC",S_DC_store,n2,n2);

            double *H_DC_store=OpenMX_Solver_Real(solver_owner,"H_DC_store",(size_t)(SpinP_switch + 1) * (size_t)n2 * (size_t)n2);
            H_DC = OpenMX_Solver_RealPlanes(solver_owner,"H_DC",H_DC_store,SpinP_switch+1,n2,n2);

            ko=OpenMX_Solver_Real(solver_owner,"ko",(size_t)n2);
            M1=OpenMX_Solver_Real(solver_owner,"M1",(size_t)n2);

            double *C_store=OpenMX_Solver_Real(solver_owner,"C_store",(size_t)n2 * (size_t)n2);
            C = OpenMX_Solver_RealAlias(solver_owner,"C",C_store,n2,n2);

            int use_dc_gpu =
                (scf_eigen_lib_flag == GPUSOLVER && !DC_GpuSolver_GemmDisabled(state) && !DC_GpuSolver_EigenDisabled(state) &&
                 DC_GpuSolver_RankAllowed(state) && DC_GPU_Threshold(state) <= NUM);
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
                    if (!DCCol_GpuSolver_DiagonalizeOverlap(state, NUM, S_DC, ko)) {
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

                openmx_local_inverse_overlap_row(local_basis, Mc_AN, 0)[0] = P_min;

                for (l = 1; l < P_min; l++)
                    M1[l] = 0.0;
                for (l = P_min; l <= NUM; l++)
                    M1[l] = 1.0 / sqrt(ko[l]);

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = 1; j1 <= NUM; j1++) {
                        S_DC[i1][j1]       = S_DC[i1][j1] * M1[j1];
                        openmx_local_inverse_overlap_row(local_basis, Mc_AN, i1)[j1] = S_DC[i1][j1];
                    }
                }

                if (measure_time) {
                    dtime(&etime);
                    time1 += etime - stime;
                }

            }

            else {

                P_min = (int)openmx_local_inverse_overlap_row(local_basis, Mc_AN, 0)[0];

                if (use_dc_gpu && (d_S_use = DCCol_SCache_ValidPtr(state, Mc_AN, NUM)) != NULL) {
                    s_dc_host_filled = 0; /* filled lazily, only if the GPU solve falls back */
                } else {
                    DCCol_FillSDCFromS12(Mc_AN, NUM, S_DC, local_basis);
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
                if (!DCCol_GpuSolver_LoadTransformedOverlap(state, NUM, S_DC) ||
                    !DCCol_SCache_Store(state, Mc_AN, NUM, &d_S_use)) {
                    use_dc_gpu = 0;
                }
            }




            for (spin = 0; spin <= SpinP_switch; spin++) {

                if (measure_time)
                    dtime(&stime);

                if (use_dc_gpu) {
                    NUM1 = NUM - (P_min - 1);
                    if (!DC_GpuSolver_PrepareGemmBackendForSolve(state, NUM, NUM1) ||
                        !DCCol_GpuSolver_SolveHamiltonian(state, NUM, P_min, d_S_use, S_DC, H_DC[spin], ko, C)) {
                        use_dc_gpu = 0;
                        if (!s_dc_host_filled) {
                            DCCol_FillSDCFromS12(Mc_AN, NUM, S_DC, local_basis);
                            s_dc_host_filled = 1;
                        }
                        NUM1 = DCCol_CPU_SolveHamiltonian(NUM, P_min, S_DC, H_DC[spin], ko, C);
                    }

                } else {
                    if (!s_dc_host_filled) {
                        DCCol_FillSDCFromS12(Mc_AN, NUM, S_DC, local_basis);
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

                {
                  double *eval_row_23=openmx_solver_field_row(&EVal,spin,Mc_AN);
                  for (i1 = 1; i1 <= NUM; i1++) {
                    eval_row_23[i1 - 1] = 1000.0;
                }
                }
                {
                  double *eval_row_22=openmx_solver_field_row(&EVal,spin,Mc_AN);
                  for (i1 = 1; i1 <= NUM1; i1++) {
                    eval_row_22[i1 - 1] = ko[i1];
                }
                }

                wanA = WhatSpecies[Gc_AN];
                tno1 = Spe_Total_CNO[wanA];

                if (is_scf_mode) {
                    {
                      double *pdos_dc_row_21=openmx_solver_field_row(&PDOS_DC,spin,Mc_AN);
                      for (i1 = 0; i1 < Msize[Mc_AN]; i1++) {
                        pdos_dc_row_21[i1] = 0.0;
                    }
                    }
                }

                {
                  double *pdos_dc_row_20=openmx_solver_field_row(&PDOS_DC,spin,Mc_AN);
                  for (i = 0; i < tno1; i++) {
                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB  = WhatSpecies[Gh_AN];
                        tno2  = Spe_Total_CNO[wanB];
                        Bnum  = MP[h_AN];
                        for (j = 0; j < tno2; j++) {
                            tmp1 = OLP0[Mc_AN][h_AN][i][j];
                            {
                              double *residues_row_16=openmx_solver_residue_row(&Residues,spin,Mc_AN,h_AN,i,j);
                              for (i1 = 1; i1 <= NUM1; i1++) {
                                tmp2 = C[1 + i][i1] * C[Bnum + j][i1];
                                residues_row_16[i1 - 1] = tmp2;
                                if (is_scf_mode) {
                                    pdos_dc_row_20[i1 - 1] += tmp2 * tmp1;
                                }
                            }
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

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
        } /* end of Mc_AN */

        /* freeing of arrays */

        ;
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

                    {
                      double *eval_row_14=openmx_solver_field_row(&EVal,spin,Mc_AN);
                      double *pdos_dc_row_15=openmx_solver_field_row(&PDOS_DC,spin,Mc_AN);
                      for (i = 0; i < Msize[Mc_AN]; i++) {
                        FermiF = DC_FermiWeight(eval_row_14[i], ChemP, Beta, max_x);
                        My_Num_State += spin_degeneracy * FermiF * pdos_dc_row_15[i];
                    }
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

        OpenMXSolverRealFieldView FermiW=OpenMX_Solver_Field(solver_owner,EVal.components,EVal.atoms);
        OpenMXSolverRealFieldView EFermiW=OpenMX_Solver_Field(solver_owner,EVal.components,EVal.atoms);

        My_Eele0[0] = 0.0;
        My_Eele0[1] = 0.0;
        for (spin = 0; spin <= SpinP_switch; spin++) {

            for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

                dtime(&Stime_atom);

                Gc_AN = M2G[Mc_AN];
                {
                    const int     n       = Msize[Mc_AN];
                    const double *eval    = openmx_solver_field_row(&EVal,spin,Mc_AN);
                    OpenMX_Solver_FieldResize(FermiW,spin,Mc_AN,(size_t)n);
                    OpenMX_Solver_FieldResize(EFermiW,spin,Mc_AN,(size_t)n);
                    double *      fweight=openmx_solver_field_row(&FermiW,spin,Mc_AN);
                    double *      eweight=openmx_solver_field_row(&EFermiW,spin,Mc_AN);

                    ;

                    {
                      double *pdos_dc_row_13=openmx_solver_field_row(&PDOS_DC,spin,Mc_AN);
                      for (i = 0; i < n; i++) {
                        FermiF     = DC_FermiWeight(eval[i], ChemP, Beta, max_x);
                        fweight[i] = FermiF;
                        eweight[i] = FermiF * eval[i];
                        My_Eele0[spin] += eweight[i] * pdos_dc_row_13[i];
                    }
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
        double *My_Eele1_threads=OpenMX_Solver_Real(solver_owner,"My_Eele1_threads",(size_t)(nthreads_eele1 * nspins_eele1));
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
                    const double *fweight = openmx_solver_field_row(&FermiW,spin,Mc_AN);
                    const double *eweight = openmx_solver_field_row(&EFermiW,spin,Mc_AN);

                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB  = WhatSpecies[Gh_AN];
                        tno2  = Spe_Total_CNO[wanB];
                        for (i = 0; i < tno1; i++) {
                            for (j = 0; j < tno2; j++) {
                                double cdm_sum = 0.0;
                                double edm_sum = 0.0;
                                double *res    = openmx_solver_residue_row(&Residues,spin,Mc_AN,h_AN,i,j);

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
            }
        }

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

    /* Storage is released by the enclosing C++ invocation. */
    FNAN[0] = 1;

    /* for time */
    dtime(&TEtime);
    time0 = TEtime - TStime;

    /* for PrintMemory */
    state->Col_owned_firsttime = 0;

    return time0;
}

#pragma optimization_level 2
typedef struct {
  DCState *state;
  char * mode;
  int SCF_iter;
  double ***** Hks;
  double ***** ImNL;
  double **** OLP0;
  double ***** CDM;
  double ***** EDM;
  double *Eele0;
  double *Eele1;
  const OpenMXElectronicViews *electronic;
  const OpenMXLocalBasisViews *local_basis;
} DC_NonCol_arguments;
static double DC_NonCol_owned(DCState *state, char * mode, int SCF_iter, double ***** Hks, double ***** ImNL, double **** OLP0,
                        double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2], const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis, OpenMXSolverInvocation *solver_owner);
static int DC_NonCol_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  DC_NonCol_arguments *a=(DC_NonCol_arguments*)arguments;
  *result=DC_NonCol_owned(a->state, a->mode,a->SCF_iter,a->Hks,a->ImNL,a->OLP0,a->CDM,a->EDM,a->Eele0,a->Eele1,a->electronic,a->local_basis,owner);
  return 1;
}
static double DC_NonCol(DCState *state, char * mode, int SCF_iter, double ***** Hks, double ***** ImNL, double **** OLP0,
                        double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2], const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis)
{
  /* A spinor solve returns the complete bond energy in channel zero.
     Define the unused channel before DFT sums both entries; callers may
     reuse uninitialized or poisoned output storage on different ranks. */
  if (strcasecmp(mode,"scf")==0) {
    Eele0[1] = 0.0;
    Eele1[1] = 0.0;
  }

  DC_NonCol_arguments arguments={state,mode,SCF_iter,Hks,ImNL,OLP0,CDM,EDM,Eele0,Eele1,electronic,local_basis};
  return OpenMX_Solver_Run(DC_NonCol_operation,&arguments);
}

#pragma optimization_level 2
static double DC_NonCol_owned(DCState *state, char * mode, int SCF_iter, double ***** Hks, double ***** ImNL, double **** OLP0,
                        double ***** CDM, double ***** EDM, double Eele0[2], double Eele1[2], const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis, OpenMXSolverInvocation *solver_owner)
{


  double ******iDM=electronic->iDM;


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
    OpenMXSolverRealFieldView EVal;
    OpenMXSolverResidueView Residues;
    OpenMXSolverRealFieldView PDOS_DC;
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

    Snd_H_Size=OpenMX_Solver_Int(solver_owner,"Snd_H_Size",((size_t)1) * numprocs);
    Rcv_H_Size=OpenMX_Solver_Int(solver_owner,"Rcv_H_Size",((size_t)1) * numprocs);
    Snd_iHNL_Size=OpenMX_Solver_Int(solver_owner,"Snd_iHNL_Size",((size_t)1) * numprocs);
    Rcv_iHNL_Size=OpenMX_Solver_Int(solver_owner,"Rcv_iHNL_Size",((size_t)1) * numprocs);
    Snd_S_Size=OpenMX_Solver_Int(solver_owner,"Snd_S_Size",((size_t)1) * numprocs);
    Rcv_S_Size=OpenMX_Solver_Int(solver_owner,"Rcv_S_Size",((size_t)1) * numprocs);

    m_size = 0;
    Msize=OpenMX_Solver_Int(solver_owner,"Msize",((size_t)1) * (Matomnum + 1));

    EVal=OpenMX_Solver_Field(solver_owner,1,((size_t)1) * (Matomnum + 1));

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

        OpenMX_Solver_FieldResize(EVal,0,Mc_AN,((size_t)1) * n2);
    }

    if (state->NonCol_owned_firsttime)
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

    Residues=OpenMX_Solver_Residues(solver_owner,((size_t)1) * 3,1,1);
    for (spin = 0; spin < (int)Residues.components; spin++) {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {
            n2 = Mc_AN == 0 ? 1 : 2 * Msize[Mc_AN] + 2;
            OpenMX_Solver_ResidueResize(Residues,spin,Mc_AN,n2);
            m_size += n2 * (Residues.pair_offsets[Residues.atom_offsets[Mc_AN+1]] -
                             Residues.pair_offsets[Residues.atom_offsets[Mc_AN]]);
        }
    }

    if (state->NonCol_owned_firsttime)
        PrintMemory("Divide_Conquer: Residues", sizeof(dcomplex) * m_size, NULL);

    /****************************************************
      allocation of arrays:

      double PDOS_DC[Matomnum+1]
                    [n2]
    ****************************************************/

    m_size = 0;

    PDOS_DC=OpenMX_Solver_Field(solver_owner,1,((size_t)1) * (Matomnum + 1));
    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++) {

        if (Mc_AN == 0)
            n2 = 1;
        else
            n2 = 2 * Msize[Mc_AN] + 2;

        m_size += n2;
        OpenMX_Solver_FieldResize(PDOS_DC,0,Mc_AN,((size_t)1) * n2);
    }

    if (state->NonCol_owned_firsttime)
        PrintMemory("Divide_Conquer: PDOS_DC", sizeof(double) * m_size, NULL);

    /* allocation of a temporal array for OpenMP */

    // #pragma omp parallel shared(Nthrds0)
    {
        Nthrds0 = omp_get_num_threads();
    }

    omp_tmp=OpenMX_Solver_Real(solver_owner,"omp_tmp",((size_t)1) * Nthrds0);

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

                tmp_array=OpenMX_Solver_Real(solver_owner,"tmp_array",((size_t)1) * size1);

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
                tmp_array2=OpenMX_Solver_Real(solver_owner,"tmp_array2",((size_t)1) * size2);

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
                ;
            }

            if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                MPI_Wait(&request, &stat);
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

                    tmp_array=OpenMX_Solver_Real(solver_owner,"tmp_array",((size_t)1) * size1);

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
                    tmp_array2=OpenMX_Solver_Real(solver_owner,"tmp_array2",((size_t)1) * size2);

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
                    ;
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
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

                    tmp_array=OpenMX_Solver_Real(solver_owner,"tmp_array",((size_t)1) * size1);

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
                    tmp_array2=OpenMX_Solver_Real(solver_owner,"tmp_array2",((size_t)1) * size2);

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
                    ;
                }

                if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0) {
                    MPI_Wait(&request, &stat);
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
        dc_gpu_capacity = DCNonCol_GpuAdmit(state, Matomnum, Msize);
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

        MP=OpenMX_Solver_Int(solver_owner,"MP",((size_t)1) * List_YOUSO[2]);

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
            double *S_DC_store=OpenMX_Solver_Real(solver_owner,"S_DC_store",(size_t)(NUM + 2) * (size_t)(NUM + 2));
            S_DC = OpenMX_Solver_RealAlias(solver_owner,"S_DC",S_DC_store,NUM+2,NUM+2);

            dcomplex *H_DC_store=OpenMX_Solver_Complex(solver_owner,"H_DC_store",(size_t)n2 * (size_t)n2);
            H_DC = OpenMX_Solver_ComplexAlias(solver_owner,"H_DC",H_DC_store,n2,n2);

            ko=OpenMX_Solver_Real(solver_owner,"ko",(size_t)n2);
            M1=OpenMX_Solver_Real(solver_owner,"M1",(size_t)n2);

            dcomplex *C_store=OpenMX_Solver_Complex(solver_owner,"C_store",(size_t)n2 * (size_t)n2);
            C = OpenMX_Solver_ComplexAlias(solver_owner,"C",C_store,n2,n2);

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

                openmx_local_inverse_overlap_row(local_basis, Mc_AN, 0)[0] = P_min;

                for (l = 1; l < P_min; l++)
                    M1[l] = 0.0;
                for (l = P_min; l <= NUM; l++)
                    M1[l] = 1.0 / sqrt(ko[l]);

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = 1; j1 <= NUM; j1++) {
                        S_DC[i1][j1]       = S_DC[i1][j1] * M1[j1];
                        openmx_local_inverse_overlap_row(local_basis, Mc_AN, i1)[j1] = S_DC[i1][j1];
                    }
                }
                // dtime(&timeC);
                // printf("timeB = %.3f\n", timeC - timeB);
            }

            else {

                P_min = (int)openmx_local_inverse_overlap_row(local_basis, Mc_AN, 0)[0];

                for (i1 = 1; i1 <= NUM; i1++) {
                    for (j1 = 1; j1 <= NUM; j1++) {
                        S_DC[i1][j1] = openmx_local_inverse_overlap_row(local_basis, Mc_AN, i1)[j1];
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
            int use_dc_gpu = (0 < dc_gpu_capacity && DC_NonCol_GPU_Threshold(state) <= 2 * NUM && P_min == 1 &&
                              !DC_GpuSolver_GemmDisabled(state) && !DC_GpuSolver_EigenDisabled(state));

            if (use_dc_gpu && DCNonCol_GpuSolve(state, NUM, dc_gpu_capacity, S_DC, H_DC, C, ko)) {
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

            {
              double *eval_row_12=openmx_solver_field_row(&EVal,0,Mc_AN);
              for (i1 = 1; i1 <= 2 * NUM; i1++) {
                eval_row_12[i1 - 1] = 1000.0;
            }
            }
            {
              double *eval_row_11=openmx_solver_field_row(&EVal,0,Mc_AN);
              for (i1 = 1; i1 <= NUM1; i1++) {
                eval_row_11[i1 - 1] = ko[i1];
            }
            }

            wanA = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_CNO[wanA];

            if (is_scf_mode) {
                {
                  double *pdos_dc_row_10=openmx_solver_field_row(&PDOS_DC,0,Mc_AN);
                  for (i1 = 0; i1 < 2 * Msize[Mc_AN]; i1++) {
                    pdos_dc_row_10[i1] = 0.0;
                }
                }
            }

            {
              double *pdos_dc_row_9=openmx_solver_field_row(&PDOS_DC,0,Mc_AN);
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
                            openmx_solver_complex_residue_row(&Residues,0,Mc_AN,h_AN,i,j)[i1 - 1].r = re11;

                            /* Re22 */
                            const double re22 = C[1 + i + NUM][i1].r * C[Bnum + j + NUM][i1].r +
                                                C[1 + i + NUM][i1].i * C[Bnum + j + NUM][i1].i;
                            openmx_solver_complex_residue_row(&Residues,1,Mc_AN,h_AN,i,j)[i1 - 1].r = re22;

                            if (is_scf_mode) {
                                pdos_dc_row_9[i1 - 1] += (re11 + re22) * tmp1;
                            }

                            /* Re12 */
                            openmx_solver_complex_residue_row(&Residues,2,Mc_AN,h_AN,i,j)[i1 - 1].r =
                                C[1 + i][i1].r * C[Bnum + j + NUM][i1].r + C[1 + i][i1].i * C[Bnum + j + NUM][i1].i;

                            /* Im12 */
                            openmx_solver_complex_residue_row(&Residues,2,Mc_AN,h_AN,i,j)[i1 - 1].i =
                                C[1 + i][i1].r * C[Bnum + j + NUM][i1].i - C[1 + i][i1].i * C[Bnum + j + NUM][i1].r;

                            /* spin-orbit coupling or LDA+U */
                            if (SO_switch == 1 || Hub_U_switch == 1 || 1 <= Constraint_NCS_switch ||
                                Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1) {

                                /* Im11 */
                                openmx_solver_complex_residue_row(&Residues,0,Mc_AN,h_AN,i,j)[i1 - 1].i =
                                    C[1 + i][i1].r * C[Bnum + j][i1].i - C[1 + i][i1].i * C[Bnum + j][i1].r;

                                /*
                                printf("VVV1 Mc_AN=%2d h_AN=%2d i=%2d j=%2d i1=%2d %15.10f\n",
                                        Mc_AN,h_AN,i,j,i1,Residues[0][Mc_AN][h_AN][i][j][i1-1].i);
                                */

                                /* Im22 */
                                openmx_solver_complex_residue_row(&Residues,1,Mc_AN,h_AN,i,j)[i1 - 1].i =
                                    C[1 + i + NUM][i1].r * C[Bnum + j + NUM][i1].i -
                                    C[1 + i + NUM][i1].i * C[Bnum + j + NUM][i1].r;
                            }
                        }
                    }
                }
            }
            }

            // dtime(&timeG);
            // printf("timeF = %.3f\n", timeG - timeF);

            dtime(&Etime_atom);
            time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

            /*
              for (i1=1; i1<=NUM1; i1++){
              printf("Mc_AN=%2d i1=%2d EVal=%15.12f\n",Mc_AN,i1,EVal[Mc_AN][i1-1]);
              }
            */

        } /* end of Mc_AN */

        /* freeing of arrays */

        ;

    } /* //#pragma omp parallel */

    /* the device matrices of the non-collinear GPU solve go back to the
       shared GPU for the grid and force phases, and so do the device
       workspaces the eigensolvers keep (MAGMA's GEMMul8 cache, rocSOLVER's
       zheevd scratch); the next call re-admits */
    DCNonCol_GpuRelease(state, 1);
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

                    {
                      double *eval_row_4=openmx_solver_field_row(&EVal,0,Mc_AN);
                      double *pdos_dc_row_5=openmx_solver_field_row(&PDOS_DC,0,Mc_AN);
                      for (i = 0; i < 2 * Msize[Mc_AN]; i++) {
                        FermiF = DC_FermiWeight(eval_row_4[i], ChemP, Beta, max_x);
                        omp_tmp[OMPID] += FermiF * pdos_dc_row_5[i];
                    }
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

        OpenMXSolverRealFieldView FermiW=OpenMX_Solver_Field(solver_owner,EVal.components,EVal.atoms);
        OpenMXSolverRealFieldView EFermiW=OpenMX_Solver_Field(solver_owner,EVal.components,EVal.atoms);

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
                    const double *eval    = openmx_solver_field_row(&EVal,0,Mc_AN);
                    OpenMX_Solver_FieldResize(FermiW,0,Mc_AN,(size_t)n);
                    OpenMX_Solver_FieldResize(EFermiW,0,Mc_AN,(size_t)n);
                    double *      fweight=openmx_solver_field_row(&FermiW,0,Mc_AN);
                    double *      eweight=openmx_solver_field_row(&EFermiW,0,Mc_AN);

                    ;

                    {
                      double *pdos_dc_row_3=openmx_solver_field_row(&PDOS_DC,0,Mc_AN);
                      for (i = 0; i < n; i++) {
                        FermiF     = DC_FermiWeight(eval[i], ChemP, Beta, max_x);
                        fweight[i] = FermiF;
                        eweight[i] = FermiF * eval[i];
                        omp_tmp[OMPID] += eweight[i] * pdos_dc_row_3[i];
                    }
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
                const double *fweight = openmx_solver_field_row(&FermiW,0,Mc_AN);
                const double *eweight = openmx_solver_field_row(&EFermiW,0,Mc_AN);

                for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                    Gh_AN = natn[Gc_AN][h_AN];
                    wanB  = WhatSpecies[Gh_AN];
                    tno2  = Spe_Total_CNO[wanB];
                    for (i = 0; i < tno1; i++) {
                        for (j = 0; j < tno2; j++) {
                            double cdm0 = 0.0, cdm1 = 0.0, cdm2 = 0.0, cdm3 = 0.0;
                            double edm0 = 0.0, edm1 = 0.0, edm2 = 0.0, edm3 = 0.0;
                            double idm00 = 0.0, idm01 = 0.0;
                            dcomplex *res0 = openmx_solver_complex_residue_row(&Residues,0,Mc_AN,h_AN,i,j);
                            dcomplex *res1 = openmx_solver_complex_residue_row(&Residues,1,Mc_AN,h_AN,i,j);
                            dcomplex *res2 = openmx_solver_complex_residue_row(&Residues,2,Mc_AN,h_AN,i,j);

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
        }

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
    /* Storage is released by the enclosing C++ invocation. */
    FNAN[0] = 1;

    /* for time */
    dtime(&TEtime);
    time0 = TEtime - TStime;

    /* for PrintMemory */
    state->NonCol_owned_firsttime = 0;

    return time0;
}

void Save_DOS_Col(OpenMXSolverResidueView Residues, double **** OLP0, OpenMXSolverRealFieldView EVal, int * Msize)
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

            {
              double *eval_row_2=openmx_solver_field_row(&EVal,spin,Mc_AN);
              for (i1 = 0; i1 < Msize[Mc_AN]; i1++) {

                fprintf(fp_ev, "%4d  %10.6f  ", i1, eval_row_2[i1]);

                for (i = 0; i < tno1; i++) {

                    sum = 0.0;
                    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                        Gh_AN = natn[Gc_AN][h_AN];
                        wanB  = WhatSpecies[Gh_AN];
                        tno2  = Spe_Total_CNO[wanB];
                        for (j = 0; j < tno2; j++) {
                            sum += openmx_solver_residue_row(&Residues,spin,Mc_AN,h_AN,i,j)[i1] * OLP0[Mc_AN][h_AN][i][j];
                        }
                    }

                    fprintf(fp_ev, "%8.5f", sum);
                }

                fprintf(fp_ev, "\n");
            }
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

void Save_DOS_NonCol(OpenMXSolverResidueView Residues, double **** OLP0, OpenMXSolverRealFieldView EVal, int * Msize)
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

        {
          double *eval_row_1=openmx_solver_field_row(&EVal,0,Mc_AN);
          for (i1 = 0; i1 < 2 * Msize[Mc_AN]; i1++) {

            fprintf(fp_ev, "%4d  %10.6f %10.6f ", i1, eval_row_1[i1], eval_row_1[i1]);

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

                        Re11 += openmx_solver_complex_residue_row(&Residues,0,Mc_AN,h_AN,i,j)[i1].r * OLP0[Mc_AN][h_AN][i][j];

                        Re22 += openmx_solver_complex_residue_row(&Residues,1,Mc_AN,h_AN,i,j)[i1].r * OLP0[Mc_AN][h_AN][i][j];

                        Re12 += openmx_solver_complex_residue_row(&Residues,2,Mc_AN,h_AN,i,j)[i1].r * OLP0[Mc_AN][h_AN][i][j];

                        Im12 += openmx_solver_complex_residue_row(&Residues,2,Mc_AN,h_AN,i,j)[i1].i * OLP0[Mc_AN][h_AN][i][j];
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

static void DC_Reset(OpenMXNativeResources *owner, void *metadata)
{
  DCState *state = (DCState *)metadata;
  (void)owner;
  if (state->initialized) Divide_Conquer_Release_GPU_Caches(state);
}

OpenMXDCState *Divide_Conquer_Bind_Resources(OpenMXCoreContext *core)
{
  OpenMXNativeResources *owner = NULL;
  void *metadata = NULL;
  char error[256];
  if (!openmx_core_native_resources_state(core, "Divide_Conquer", sizeof(DCState),
      DC_Reset, &owner, &metadata, error, sizeof(error))) {
    fprintf(stderr, "OpenMX Divide_Conquer state: %s\n", error);
    abort();
  }
  DCState *state = (DCState *)metadata;
  if (!state->initialized) {
    state->resources = owner;
    state->gpusolver_gemmul8_disabled = DC_GPUSOLVER_GEMMUL8_DISABLED_DEFAULT;
    state->NonCol_owned_firsttime = 1;
    state->Col_owned_firsttime = 1;
    state->GpuSolver_RankAllowed_allowed = 1;
    state->NonCol_GPU_Threshold_threshold = DC_NONCOL_GPU_CPU_SWITCH_NUM;
    state->GPU_Threshold_threshold = DC_GPU_CPU_SWITCH_NUM;
    state->initialized = 1;
  }
  return state;
}
