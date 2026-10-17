/**********************************************************************
  Band_DFT_Col.c:

     Band_DFT_Col.c is a subroutine to perform band calculations
     based on a collinear DFT

  Log of Band_DFT_Col.c:

     22/Nov/2001  Released by T. Ozaki

***********************************************************************/

#include "mpi.h"
#include "lapack_prototypes.h"
#include "openmx_common.h"
#include "openmx_mgpu_runtime.h"
#include "openmx_gpu_transform.h"
#include "openmx_mgpu_band_plan.h"
#include "openmx_gpu_dm_collective.h"
#include "openmx_gpu_workers.h"

#include "set_hip_default_device_from_local_rank.h"
#include "tran_variables.h"
#include "eigen_refine_gpu.h"
#include "openmx_precision.h"
#include "eigen_refine_hooks.h"
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define measure_time 0

void set_up_blacsgrid_f(int mpi_comm_parent, int np_rows, int np_cols, char layout, int * my_blacs_ctxt, int * my_prow,
                        int * my_pcol);
void set_up_blacs_descriptor_f(int na, int nblk, int my_prow, int my_pcol, int np_rows, int np_cols, int * na_rows,
                               int * na_cols, int sc_desc[9], int my_blacs_ctxt, int * info);

void solve_evp_complex_(int * n1, int * n2, dcomplex * Cs, int * na_rows1, double * ko, dcomplex * Ss, int * na_rows2,
                        int * nblk, int * mpi_comm_rows_int, int * mpi_comm_cols_int);

void elpa_solve_evp_complex_2stage_double_impl_(int * n, int * MaxN, dcomplex * Hs, int * na_rows1, double * ko,
                                                dcomplex * Cs, int * na_rows2, int * nblk, int * na_cols1,
                                                int * mpi_comm_rows_int, int * mpi_comm_cols_int, int * mpiworld);

extern int openmx_magma_zheevdx_gpu(int n, int maxn, void *d_A, double *w, int *mout);
extern int openmx_hipsolver_zheevd_gpu(int n, int maxn, void *d_A, double *w, int *mout);
extern int openmx_gpu_eigensolver_use_hipsolver(void);
extern int openmx_gpu_is_apu(void);
extern int openmx_magma_release_z_workspace(void);
extern void openmx_gemmul8TrimWorkspaces(void);
extern size_t Set_Hamiltonian_DemoteResidency(void);
extern int openmx_hipsolver_release_workspace(void);
extern int BandCol_ReleaseDMDeviceWorkspace_HIP(void);
extern void Set_Density_Grid_GPU_Release_Device_Caches(void);
extern hipblasStatus_t openmx_gemmul8ZgemmSolve(hipblasHandle_t handle, hipblasOperation_t transa,
                                                hipblasOperation_t transb, int m, int n, int k,
                                                const hipDoubleComplex *alpha, const hipDoubleComplex *A, int lda,
                                                const hipDoubleComplex *B, int ldb, const hipDoubleComplex *beta,
                                                hipDoubleComplex *C, int ldc);

static void Construct_Band_CsHs(int SCF_iter, int all_knum, int * order_GA, int * MP, double * S1, double * H1,
                                double k1, double k2, double k3, dcomplex * Cs, dcomplex * Hs, int n,
                                int use_gpusolver_dense, int owns_global_dense_rank);
static int  BandCol_LastConstructOnDevice(void);

static double get_max_value(double localValue);
static void   BandCol_GpuSolver_EnsureMatrixCapacity(int n);
static void   BandCol_ConstructCache_Reset(void);
static void   BandCol_RefineRelease(void);
static void   BandCol_RefineResetStage(void);
static void   BandCol_RefineDropBasis(void);

static int BandCol_mgpu_host_dense=0, BandCol_host_n=0, BandCol_host_stride=0;
static dcomplex *BandCol_host_s=NULL,*BandCol_host_h=NULL,*BandCol_host_tmp=NULL;
static void BandCol_HostEnsure(int n);
static void BandCol_GpuSolver_Destroy(void);
typedef struct
{
    int                initialized;
    int                device_id;
    int                matrix_dim;
    int                scale_dim;
    int                transformed_s_valid;
    int                transformed_s_dim;
    size_t             d_work_bytes;
    size_t             h_work_bytes;
    hipblasHandle_t     hipblas;
    hipsolverDnHandle_t gpusolver;
    dcomplex *         d_S;
    dcomplex *         d_H;
    dcomplex *         d_tmp;
    dcomplex *         d_evec;
    dcomplex *         d_scale;
    double *           d_W;
    int32_t *          d_info;
    void *             d_work;
    void *             h_work;
    dcomplex *         h_scale;
    dcomplex *         h_evec;
    int                h_evec_dim;
    int                h_evec_stride;
    size_t             h_evec_count;
    int                d_evec_dim;
    int                d_evec_stride;
    size_t             d_evec_count;
    dcomplex *         h_transformed_s;
    int                h_transformed_s_dim;
} BandColGpuSolverCtx;

static BandColGpuSolverCtx BandCol_gpusolver_ctx = {0};

typedef struct {
    dcomplex *matrix;
    double k1, k2, k3;
} BandColKOverlap;

static BandColKOverlap *BandCol_k_overlap = NULL;
static int BandCol_k_overlap_n = 0, BandCol_k_overlap_points = 0;
static size_t BandCol_k_overlap_bytes = 0, BandCol_k_overlap_used = 0;

static void BandCol_KOverlapReset(void)
{
    for (int k = 0; k < BandCol_k_overlap_points; ++k)
        free(BandCol_k_overlap[k].matrix);
    free(BandCol_k_overlap);
    BandCol_k_overlap = NULL;
    BandCol_k_overlap_n = BandCol_k_overlap_points = 0;
    BandCol_k_overlap_bytes = BandCol_k_overlap_used = 0;
}

/* Overlap and eigenvector caches each receive at most 1/64 of available
   node RAM, divided among local ranks. Allocation failure keeps the
   original GPU recomputation path available. */
static size_t BandCol_HostCacheLimit(const char *name, double default_mib)
{
    const char *env = getenv(name);
    double cap = env == NULL ? default_mib : atof(env);
    unsigned long long available_kib = 0;
    int ranks = openmx_gpu_local_size_noncollective();
    FILE *file;
    char line[256];
    size_t limit;

    if (!(cap > 0.0) || cap >= (double)SIZE_MAX / 1048576.0) return 0;
    limit = (size_t)(cap * 1048576.0);
    file = fopen("/proc/meminfo", "r");
    if (file == NULL) return 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (sscanf(line, "MemAvailable: %llu kB", &available_kib) == 1) break;
    }
    fclose(file);
    if (ranks < 1) ranks = 1;
    if (available_kib / (unsigned)ranks * (1024ULL / 64ULL) < limit)
        limit = (size_t)(available_kib / (unsigned)ranks * (1024ULL / 64ULL));
    return limit;
}

/* An entry holds the transformed overlap (n x n complex) and, behind it,
   the n overlap eigenvalues, which a hit restores to ko and koS as the
   overlap solve would leave them (the outputs and diagnostics read them). */
static void BandCol_KOverlapPrepare(int n, int points)
{
    size_t matrix_bytes, bytes;

    if (BandCol_k_overlap_n != n || BandCol_k_overlap_points != points)
        BandCol_KOverlapReset();
    if (BandCol_k_overlap != NULL || n <= 0 || points <= 0 ||
        (size_t)n > SIZE_MAX / (size_t)n / sizeof(dcomplex)) return;
    matrix_bytes = (size_t)n * (size_t)n * sizeof(dcomplex);
    if ((size_t)n > (SIZE_MAX - matrix_bytes) / sizeof(double)) return;
    bytes = matrix_bytes + (size_t)n * sizeof(double);
    if (bytes > BandCol_HostCacheLimit("OPENMX_BAND_COL_OVERLAP_CACHE_MB", 512.0)) return;
    BandCol_k_overlap = (BandColKOverlap*)calloc((size_t)points, sizeof(BandColKOverlap));
    if (BandCol_k_overlap == NULL) return;
    BandCol_k_overlap_n = n;
    BandCol_k_overlap_points = points;
    BandCol_k_overlap_bytes = bytes;
}

static int BandCol_KOverlapRestore(int point, double k1, double k2, double k3, double *ko, double *koS)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    BandColKOverlap *entry;
    size_t matrix_bytes, eigen_bytes;

    if (BandCol_k_overlap == NULL || point < 0 || point >= BandCol_k_overlap_points) return 0;
    entry = &BandCol_k_overlap[point];
    if (entry->matrix == NULL || entry->k1 != k1 || entry->k2 != k2 || entry->k3 != k3) return 0;
    matrix_bytes = (size_t)BandCol_k_overlap_n * (size_t)BandCol_k_overlap_n * sizeof(dcomplex);
    eigen_bytes = (size_t)BandCol_k_overlap_n * sizeof(double);
    if (BandCol_mgpu_host_dense) {
        BandCol_HostEnsure(BandCol_k_overlap_n);
        memcpy(BandCol_host_s,entry->matrix,matrix_bytes);
    } else {
        BandCol_GpuSolver_EnsureMatrixCapacity(BandCol_k_overlap_n);
        wait_hipfunc(hipMemcpy(ctx->d_S, entry->matrix, matrix_bytes, hipMemcpyHostToDevice));
    }
    memcpy(ko + 1, (const char*)entry->matrix + matrix_bytes, eigen_bytes);
    memcpy(koS + 1, (const char*)entry->matrix + matrix_bytes, eigen_bytes);
    ctx->transformed_s_valid = 1;
    ctx->transformed_s_dim = BandCol_k_overlap_n;
    return 1;
}

static void BandCol_KOverlapSave(int point, double k1, double k2, double k3, const double *koS)
{
    BandColKOverlap *entry;
    size_t matrix_bytes;

    if (BandCol_k_overlap == NULL || point < 0 || point >= BandCol_k_overlap_points) return;
    entry = &BandCol_k_overlap[point];
    if (entry->matrix == NULL) {
        size_t limit = BandCol_HostCacheLimit("OPENMX_BAND_COL_OVERLAP_CACHE_MB", 512.0);
        if (BandCol_k_overlap_used > limit || BandCol_k_overlap_bytes > limit - BandCol_k_overlap_used) return;
        entry->matrix = (dcomplex*)malloc(BandCol_k_overlap_bytes);
        if (entry->matrix == NULL) return;
        BandCol_k_overlap_used += BandCol_k_overlap_bytes;
    }
    matrix_bytes = (size_t)BandCol_k_overlap_n * (size_t)BandCol_k_overlap_n * sizeof(dcomplex);
    if (BandCol_mgpu_host_dense) memcpy(entry->matrix,BandCol_host_s,matrix_bytes);
    else wait_hipfunc(hipMemcpy(entry->matrix, BandCol_gpusolver_ctx.d_S, matrix_bytes, hipMemcpyDeviceToHost));
    memcpy((char*)entry->matrix + matrix_bytes, koS + 1, (size_t)BandCol_k_overlap_n * sizeof(double));
    entry->k1 = k1;
    entry->k2 = k2;
    entry->k3 = k3;
}

typedef struct
{
    int      max_tno;
    int      max_nk;
    double * buf_Re0;
    double * buf_Im0;
    double * buf_Re1;
    double * buf_Im1;
    double * TmpEIGEN;
    double * OccWeight;
    double **ReEVec0;
    double **ImEVec0;
    double **ReEVec1;
    double **ImEVec1;
} BandColDMWorkspace;

static BandColDMWorkspace BandCol_dm_workspace = {0};

typedef struct
{
    int                valid;
    int                n;
    unsigned long long fingerprint;
    int                entry_count;
    int                pair_count;
    int *              basis0;
    int *              basis1;
    int *              phase_index;
    int *              pair_l1;
    int *              pair_l2;
    int *              pair_l3;
    double *           phase_r;
    double *           phase_i;
} BandColDMEntryCache;

static BandColDMEntryCache BandCol_dm_entry_cache = {0};

typedef struct
{
    int h_index;
    int index0;
    int index1;
    int l1;
    int l2;
    int l3;
    int phase_index;
} BandColConstructEntry;

/* d_entries is DEVICE-resident (the construct cache keeps no host copy) */
extern int BandCol_BuildDenseCsHs_HIP(int need_s, int count, int h_count, int phase_count, int n,
                                      const BandColConstructEntry *d_entries, const double *phase_r,
                                      const double *phase_i, const double *H1, const double *S1,
                                      dcomplex *d_H, dcomplex *d_S);
extern int BandCol_AccumulateDenseTransposedDM_HIP(int entry_count, int pair_count, int nk, int evec_stride,
                                                   const int *basis0, const int *basis1, const int *phase_index,
                                                   const double *phase_r, const double *phase_i,
                                                   const double *eigen, const double *occ_weight,
                                                   const dcomplex *evec_device, double *CDM1, double *EDM1);

typedef struct
{
    int                    valid;
    int                    n;
    int                    na_rows;
    int                    na_cols;
    int                    nblk;
    int                    np_rows;
    int                    np_cols;
    int                    my_prow;
    int                    my_pcol;
    unsigned long long     fingerprint;
    int                    dense_count;
    int                    local_count;
    int                    h_count;
    int                    dense_phase_count;
    int                    has_dense;
    int                    has_local;
    int *                  dense_phase_l1;
    int *                  dense_phase_l2;
    int *                  dense_phase_l3;
    double *               dense_phase_r;
    double *               dense_phase_i;
    /* device-resident: the entries are only ever read by the device
       construct kernel, so no host copy is kept */
    BandColConstructEntry *dense_entries_dev;
    BandColConstructEntry *local_entries;
} BandColConstructCache;

static BandColConstructCache BandCol_construct_cache = {0};
static int                   BandCol_last_construct_on_device = 0;

static void BandCol_AbortWithMessage(const char * msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

/* Reusing S(k) and the occupation-pass eigenvectors lowers the collinear
   band crossover: GEL (n=456) improves with eight ranks on gfx1200.
   OPENMX_BAND_GPU_SWITCH_NUM overrides this path only; DFT.c also uses
   the value for its startup fallback notice. */
#define BAND_GPU_CPU_SWITCH_NUM 400

int Band_DFT_Col_GpuSwitchNum(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("OPENMX_BAND_GPU_SWITCH_NUM");
        int parsed;

        cached = BAND_GPU_CPU_SWITCH_NUM;
        if (value != NULL && 0 < (parsed = atoi(value))) {
            cached = parsed;
        }

        /* See Band_DFT_NonCol_GpuSwitchNum: a GPU shared by too many ranks
           serializes the per-rank dense solves and contends on the packed
           H/S gather, so the CPU (ELPA2) path wins and is stable.  The
           validated 8-rank GPU configuration stays below the default limit
           and is unaffected. */
        if (openmx_band_gpu_dense_oversubscribed() && cached < INT_MAX) {
            cached = INT_MAX;
        }
    }
    return cached;
}

static int BandCol_DefaultGpuTurnGroup(void)
{
    /* Upper bound on how many MPI ranks (k-point turns) use the GPU at the same
       time.  With a single GPU this lets at most 2 k-points be diagonalized on
       the GPU concurrently; the remaining ranks wait their turn.  Override with
       OPENMX_BAND_GPU_MAX_CONCURRENT_K, OPENMX_BAND_GPU_MAX_RANKS_PER_DEVICE, or
       OPENMX_BAND_GPU_TURN_GROUP. */
    return 2;
}

static int BandCol_SerializeGpuSolverGpuTurns(void)
{
    const char *value = getenv("OPENMX_GPUSOLVER_SERIAL_GPU_TURNS");

    if (value == NULL) {
        return 1;
    }
    return (atoi(value) != 0);
}

static int BandCol_GpuPersistentEnabled(void)
{
    const char *value = getenv("OPENMX_BAND_GPU_PERSISTENT");

    if (value == NULL) {
        /* On an APU (MI300A) the save/load staging between SCF iterations is a
           pure unified-memory round trip and memory is plentiful, so keep the
           device buffers alive by default there. */
        return openmx_gpu_is_apu();
    }
    return (atoi(value) != 0);
}

static int BandCol_GpuPersistentDecide(void)
{
    static int decided = -1;

    if (decided < 0) {
        if (!BandCol_GpuPersistentEnabled()) {
            decided = 0;
        } else {
            size_t      free_bytes = 0, total_bytes = 0;
            size_t      min_free = 2048ULL * 1024ULL * 1024ULL;
            const char *value = getenv("OPENMX_BAND_GPU_PERSISTENT_MIN_FREE_MB");

            if (value != NULL) {
                long mb = atol(value);
                if (0 <= mb) {
                    min_free = (size_t)mb * 1024ULL * 1024ULL;
                }
            }

            if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
                decided = 0;
            } else if (free_bytes < min_free) {
                if (2 <= level_stdout) {
                    fprintf(stderr,
                            "Band_DFT_Col: persistent GPU buffers disabled (%.0f MiB free < %.0f MiB threshold).\n",
                            (double)free_bytes / (1024.0 * 1024.0), (double)min_free / (1024.0 * 1024.0));
                }
                decided = 0;
            } else {
                decided = 1;
            }
        }
    }
    return decided;
}

static int BandCol_GemmWorkspaceTurnRelease(void)
{
    const char *value = getenv("OPENMX_BAND_GEMMUL8_TURN_RELEASE");

    if (value == NULL) {
        /* Keep the GEMMul8 workspace across turns on the memory-rich APU;
           release per turn on smaller discrete GPUs as before. */
        return openmx_gpu_is_apu() ? 0 : 1;
    }
    return (atoi(value) != 0);
}

/* Preserve the historical SCF-boundary behaviour unless retention was
   explicitly requested.  The APU default above is useful between staggered
   k-point turns, but carrying a roughly GiB-sized Z-GEMMul8 workspace per
   rank into the next SCF is too broad a default for other inputs. */
static int BandCol_GemmWorkspaceScfRelease(void)
{
    const char *value = getenv("OPENMX_BAND_GEMMUL8_TURN_RELEASE");

    return (value == NULL) ? 1 : (atoi(value) != 0);
}

static int BandCol_GpuTurnGroup(void)
{
    const char *value = getenv("OPENMX_BAND_GPU_TURN_GROUP");
    long        group;

    if (value == NULL) {
        value = getenv("OPENMX_BAND_GPU_MAX_RANKS_PER_DEVICE");
        if (value == NULL) {
            group = BandCol_DefaultGpuTurnGroup();
        } else {
            group = strtol(value, NULL, 10);
        }
    } else {
        group = strtol(value, NULL, 10);
    }

    if (group < 1L) {
        return 1;
    }
    if (1024L < group) {
        return 1024;
    }
    return (int)group;
}

/* Cached overlap matrices remove a relatively long eigensolver call which
   used to stagger the k-point ranks naturally.  On a single shared GPU that
   can make every Hamiltonian solve arrive at once (a "thundering herd").
   Allow the cached-SCF concurrency to be tuned independently without changing
   the first-SCF setting.  Keeping the variable unset preserves the historical
   behaviour. */
static int BandCol_GpuTurnGroupForScf(int SCF_iter)
{
    const char *value;
    long        group;

    if (SCF_iter <= 1) {
        return BandCol_GpuTurnGroup();
    }

    value = getenv("OPENMX_BAND_GPU_CACHED_TURN_GROUP");
    if (value == NULL || *value == '\0') {
        return BandCol_GpuTurnGroup();
    }

    group = strtol(value, NULL, 10);
    if (group < 1L) {
        return 1;
    }
    if (1024L < group) {
        return 1024;
    }
    return (int)group;
}

static int BandCol_CacheTraceEnabled(void)
{
    static int enabled = -1;

    if (enabled < 0) {
        const char *value = getenv("OPENMX_BAND_CACHE_TRACE");
        enabled = (value != NULL && atoi(value) != 0);
    }
    return enabled;
}

static int BandCol_AbortAfterEigenScf(void)
{
    const char *value = getenv("OPENMX_ABORT_AFTER_EIGEN_SCF");

    if (value != NULL && *value != '\0') {
        long scf = strtol(value, NULL, 10);
        if (0 < scf && scf <= INT_MAX) {
            return (int)scf;
        }
    }

    value = getenv("OPENMX_ABORT_AFTER_EIGEN");
    return (value != NULL && atoi(value) != 0) ? 1 : 0;
}

static int BandCol_MaxConcurrentKGpuTurns(int n)
{
    const char *value = getenv("OPENMX_BAND_GPU_MAX_CONCURRENT_K");
    long limit;

    if (value != NULL) {
        char *endp = NULL;
        limit = strtol(value, &endp, 10);
        if (endp == value || limit < 1L) {
            return 1;
        }
        if ((long)INT_MAX < limit) {
            return INT_MAX;
        }
        return (int)limit;
    }

    value = getenv("OPENMX_BAND_GPU_MAX_RANKS_PER_DEVICE");
    if (value != NULL) {
        limit = strtol(value, NULL, 10);
        if (limit < 1L) {
            return 1;
        }
        if ((long)INT_MAX < limit) {
            return INT_MAX;
        }
        return (int)limit;
    }

    /* With cached S/eigenvectors, eight small k-point solves overlap well
       on gfx1200 (GEL, n=456). Larger matrices keep the two-turn budget. */
    return n <= 512 ? 8 : BandCol_DefaultGpuTurnGroup();
}

/* Fixed per-rank device cost of the BLAS/eigensolver libraries beyond the
   dense matrices (handle workspaces, kernel code objects): 200-300 MiB per
   rank measured on ROCm 7.2, independent of n, and an allocation failing
   inside hipBLASLt terminates the process.  Charged per concurrently solving
   rank; OPENMX_BAND_GPU_RESERVE_MB overrides the default of 512 MiB. */
static size_t BandCol_GpuLibraryReserveBytes(void)
{
    static size_t reserve = 0;
    static int checked = 0;

    if (!checked){
        const char *env = getenv("OPENMX_BAND_GPU_RESERVE_MB");
        long mib = 512L;
        if (env!=NULL && env[0]!='\0'){
            long value = atol(env);
            if (0L<=value) mib = value;
        }
        reserve = (size_t)mib*1024U*1024U;
        checked = 1;
    }
    return reserve;
}

/* Ranks of mpi_comm_level1 that share this rank's GPU.  Collective on the
   first call; every rank reaches BandCol_GpuDenseFits together. */
static int BandCol_device_ranks_cached = 0;

static int BandCol_DeviceRanks(void)
{
    if (BandCol_device_ranks_cached==0){
        MPI_Comm node_comm, device_comm;
        int hip_device = 0;
        int device_ranks = 0;

        MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
        if (hipGetDevice(&hip_device)!=hipSuccess) hip_device = 0;
        MPI_Comm_split(node_comm,hip_device,0,&device_comm);
        MPI_Comm_size(device_comm,&device_ranks);
        MPI_Comm_free(&device_comm);
        MPI_Comm_free(&node_comm);
        BandCol_device_ranks_cached = (device_ranks<1) ? 1 : device_ranks;
    }
    return BandCol_device_ranks_cached;
}

/* non-collective: for the per-k-point paths, which not every rank reaches
   the same number of times (BandCol_GpuDenseFits fills the cache first) */
static int BandCol_DeviceRanksCached(void)
{
    return (BandCol_device_ranks_cached<1) ? 1 : BandCol_device_ranks_cached;
}

static int BandCol_NativeGpuDenseFits(int n, int owns_dense)
{
    const char *env = getenv("OPENMX_BAND_GPU_DIAG");
    const char *reserve_env = getenv("OPENMX_BAND_GPU_DIAG_RESERVE_MB");
    size_t free_bytes=0,total_bytes=0,reserve=(size_t)1024*1024*1024;
    size_t per_rank,required=0,fixed_all;
    int device_ranks;
    int turns=BandCol_MaxConcurrentKGpuTurns(n);
    int local_fit=1,fit=1,has_budget=0;

    if (reserve_env!=NULL){
        long mib=atol(reserve_env);
        if (0<=mib) reserve=(size_t)mib*1024U*1024U;
    }
    if (turns<1) turns=1;
    if ((size_t)n>SIZE_MAX/(size_t)n/sizeof(dcomplex)/5U){
        per_rank=SIZE_MAX;
    }
    else per_rank=(size_t)5*(size_t)n*(size_t)n*sizeof(dcomplex)+(size_t)n*sizeof(double);
    /* every rank sharing the device pays the library-side fixed cost,
       however many of them solve at a time; collective, so keep it before
       the owner-only check */
    device_ranks=BandCol_DeviceRanks();
    fixed_all=BandCol_GpuLibraryReserveBytes()*(size_t)device_ranks;
    if (per_rank!=SIZE_MAX && reserve<=SIZE_MAX-fixed_all &&
        (size_t)turns<=(SIZE_MAX-reserve-fixed_all)/per_rank){
        required=per_rank*(size_t)turns+reserve+fixed_all;
        OpenMX_GpuPhaseNeed_Register("band_col",required);
        has_budget=1;
    }
    /* A completed DM/density phase may still own hundreds of MiB. Reclaim
       its reproducible device buffers before accepting a CPU fallback;
       keep the original admission budget and transformed overlap intact. */
    for (int attempt=0; attempt<2; ++attempt) {
        /* -1 is an explicit/invalid-budget fallback, 0 is memory pressure,
           and 1 fits. Only memory pressure should evict reusable caches. */
        local_fit=(has_budget && (env==NULL || atoi(env)!=0)) ? 1 : -1;
        if (local_fit==1 && owns_dense &&
            (hipMemGetInfo(&free_bytes,&total_bytes)!=hipSuccess || free_bytes<required))
            local_fit=0;
        MPI_Allreduce(&local_fit,&fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);
        if (fit!=0 || attempt==1) break;
        (void)BandCol_ReleaseDMDeviceWorkspace_HIP();
        BandCol_ConstructCache_Reset();
        /* the basis the resident refined path keeps for its warm start */
        BandCol_RefineDropBasis();
        Set_Density_Grid_GPU_Release_Device_Caches();
        /* the Hamiltonian's resident orbital tables as well, for the rest of
           the run: re-uploaded per call they leave the eigensolver its room */
        (void)Set_Hamiltonian_DemoteResidency();
        openmx_gemmul8ReleaseWorkspaces();
        (void)openmx_magma_release_z_workspace();
        (void)openmx_hipsolver_release_workspace();
        MPI_Barrier(mpi_comm_level1);
    }
    if (fit==0 && local_fit==0 && owns_dense) {
        int myid;
        MPI_Comm_rank(mpi_comm_level1,&myid);
        if (myid==Host_ID) {
            const double MiB=1024.0*1024.0;
            fprintf(stderr,
                    "Band_DFT_Col: GPU device is shared by %d rank(s), has %.3f MiB free "
                    "(%.3f MiB total), but %.3f MiB is required for the dense GPU diagonalization "
                    "(%.3f MiB library reserve per rank); switching to CPU path.\n",
                    device_ranks,(double)free_bytes/MiB,(double)total_bytes/MiB,
                    (double)required/MiB,(double)BandCol_GpuLibraryReserveBytes()/MiB);
            fflush(stderr);
        }
    }
    return fit==1;
}

static int BandCol_GpuDenseFits(int n,int owns_dense)
{
    const char *enabled=getenv("OPENMX_BAND_GPU_DIAG");
    BandCol_mgpu_host_dense=0;
    if (enabled && atoi(enabled)==0 && !openmx_mgpu_host_required()) return 0;
    int fit=openmx_mgpu_host_required()?0:BandCol_NativeGpuDenseFits(n,owns_dense);
    if (!fit && !gpusolver2_flag) {
        BandCol_GpuSolver_Destroy();
        openmx_gemmul8ReleaseWorkspaces();
        MPI_Barrier(mpi_comm_level1);
        int rank;
        MPI_Comm_rank(mpi_comm_level1,&rank);
        if (rank==Host_ID) {
            int total=openmx_mgpu_band_kpoint_count(SpinP_switch,way_of_kpoint,
                Kspace_grid1,Kspace_grid2,Kspace_grid3,num_non_eq_kpt);
            fit=openmx_mgpu_host_plan(total,n,1)!=0;
            if (level_stdout>=1) {
                if (fit) printf("<OpenMX_MGPU> Band collinear n=%d: host-backed MAGMA capacity path.\n",n);
                else printf("<OpenMX_MGPU> Band collinear n=%d: %s; CPU solve.\n",n,openmx_mgpu_host_error());
                fflush(stdout);
            }
        }
        MPI_Bcast(&fit,1,MPI_INT,Host_ID,mpi_comm_level1);
        BandCol_mgpu_host_dense=fit;
    }
    return fit;
}


static void BandCol_ConstructCache_Reset(void)
{
    if (BandCol_construct_cache.dense_entries_dev != NULL) {
        wait_hipfunc(hipFree(BandCol_construct_cache.dense_entries_dev));
    }

    free(BandCol_construct_cache.local_entries);
    free(BandCol_construct_cache.dense_phase_l1);
    free(BandCol_construct_cache.dense_phase_l2);
    free(BandCol_construct_cache.dense_phase_l3);
    free(BandCol_construct_cache.dense_phase_r);
    free(BandCol_construct_cache.dense_phase_i);
    memset(&BandCol_construct_cache, 0, sizeof(BandCol_construct_cache));
}

static unsigned long long BandCol_ConstructFingerprint(int *order_GA, int *MP)
{
    unsigned long long h = 1469598103934665603ULL;

#define BANDCOL_HASH_INT(v)                                                                                             \
    do {                                                                                                                \
        h ^= (unsigned long long)(unsigned int)(v);                                                                      \
        h *= 1099511628211ULL;                                                                                          \
    } while (0)

    BANDCOL_HASH_INT(atomnum);
    for (int AN = 1; AN <= atomnum; AN++) {
        int GA_AN = order_GA[AN];
        BANDCOL_HASH_INT(GA_AN);
        BANDCOL_HASH_INT(MP[GA_AN]);
        BANDCOL_HASH_INT(WhatSpecies[GA_AN]);
        BANDCOL_HASH_INT(Spe_Total_CNO[WhatSpecies[GA_AN]]);
        BANDCOL_HASH_INT(FNAN[GA_AN]);
        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            BANDCOL_HASH_INT(natn[GA_AN][LB_AN]);
            BANDCOL_HASH_INT(ncn[GA_AN][LB_AN]);
        }
    }

#undef BANDCOL_HASH_INT

    return h;
}

static void BandCol_ConstructCache_Ensure(int *order_GA, int *MP, int n, int want_dense)
{
    BandColConstructCache *cache = &BandCol_construct_cache;
    unsigned long long     fingerprint = BandCol_ConstructFingerprint(order_GA, MP);
    int                    dense_count = 0;
    int                    local_count = 0;
    int                    dense_phase_count = 0;
    int                    prev_dense_l1 = 0x7fffffff, prev_dense_l2 = 0x7fffffff, prev_dense_l3 = 0x7fffffff;
    BandColConstructEntry *dense_host = NULL;

    /* Only the side this rank actually consumes is built: the dense side
       feeds the device construct kernel of the GPU owner ranks, the local
       side feeds the host ScaLAPACK/ELPA construction.  Building both cost
       every ELPA rank the full dense table (~28 B per matrix element pair)
       it never read. */
    if (cache->valid && cache->n == n && cache->na_rows == na_rows && cache->na_cols == na_cols &&
        cache->nblk == nblk && cache->np_rows == np_rows && cache->np_cols == np_cols && cache->my_prow == my_prow &&
        cache->my_pcol == my_pcol && cache->fingerprint == fingerprint &&
        ((want_dense && cache->has_dense) || (!want_dense && cache->has_local))) {
        return;
    }

    BandCol_ConstructCache_Reset();

    for (int AN = 1; AN <= atomnum; AN++) {
        int GA_AN = order_GA[AN];
        int wanA  = WhatSpecies[GA_AN];
        int tnoA  = Spe_Total_CNO[wanA];
        int Anum  = MP[GA_AN];

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int Rn    = ncn[GA_AN][LB_AN];
            int wanB  = WhatSpecies[GB_AN];
            int tnoB  = Spe_Total_CNO[wanB];
            int Bnum  = MP[GB_AN];
            int l1    = atv_ijk[Rn][1];
            int l2    = atv_ijk[Rn][2];
            int l3    = atv_ijk[Rn][3];
            int block_dense_count = 0;

            for (int i = 0; i < tnoA; i++) {
                int ig = Anum + i;
                int j_start = ig - Bnum;
                int brow = (ig - 1) / nblk;
                int prow = brow % np_rows;

                if (j_start < 0) {
                    j_start = 0;
                } else if (j_start > tnoB) {
                    j_start = tnoB;
                }

                dense_count += tnoB - j_start;
                block_dense_count += tnoB - j_start;

                for (int j = 0; j < tnoB; j++) {
                    int jg   = Bnum + j;
                    int bcol = (jg - 1) / nblk;
                    int pcol = bcol % np_cols;

                    if (my_prow == prow && my_pcol == pcol) {
                        local_count++;
                    }
                }
            }

            if (0 < block_dense_count &&
                (l1 != prev_dense_l1 || l2 != prev_dense_l2 || l3 != prev_dense_l3)) {
                dense_phase_count++;
                prev_dense_l1 = l1;
                prev_dense_l2 = l2;
                prev_dense_l3 = l3;
            }
        }
    }

    if (want_dense) {
        dense_host = (BandColConstructEntry *)malloc(sizeof(BandColConstructEntry) * (size_t)(dense_count + 1));
        cache->dense_phase_l1 = (int *)malloc(sizeof(int) * (size_t)(dense_phase_count + 1));
        cache->dense_phase_l2 = (int *)malloc(sizeof(int) * (size_t)(dense_phase_count + 1));
        cache->dense_phase_l3 = (int *)malloc(sizeof(int) * (size_t)(dense_phase_count + 1));
        cache->dense_phase_r  = (double *)malloc(sizeof(double) * (size_t)(dense_phase_count + 1));
        cache->dense_phase_i  = (double *)malloc(sizeof(double) * (size_t)(dense_phase_count + 1));

        if (dense_host == NULL || cache->dense_phase_l1 == NULL ||
            cache->dense_phase_l2 == NULL || cache->dense_phase_l3 == NULL || cache->dense_phase_r == NULL ||
            cache->dense_phase_i == NULL) {
            free(dense_host);
            BandCol_ConstructCache_Reset();
            BandCol_AbortWithMessage("Failed to allocate Construct_Band_CsHs cache in Band_DFT_Col.c.");
        }
    } else {
        cache->local_entries = (BandColConstructEntry *)malloc(sizeof(BandColConstructEntry) * (size_t)(local_count + 1));

        if (cache->local_entries == NULL) {
            BandCol_ConstructCache_Reset();
            BandCol_AbortWithMessage("Failed to allocate Construct_Band_CsHs cache in Band_DFT_Col.c.");
        }
    }

    dense_count = 0;
    local_count = 0;
    dense_phase_count = 0;
    prev_dense_l1 = 0x7fffffff;
    prev_dense_l2 = 0x7fffffff;
    prev_dense_l3 = 0x7fffffff;
    int h_index = 0;
    for (int AN = 1; AN <= atomnum; AN++) {
        int GA_AN = order_GA[AN];
        int wanA  = WhatSpecies[GA_AN];
        int tnoA  = Spe_Total_CNO[wanA];
        int Anum  = MP[GA_AN];

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int Rn    = ncn[GA_AN][LB_AN];
            int wanB  = WhatSpecies[GB_AN];
            int tnoB  = Spe_Total_CNO[wanB];
            int Bnum  = MP[GB_AN];
            int l1    = atv_ijk[Rn][1];
            int l2    = atv_ijk[Rn][2];
            int l3    = atv_ijk[Rn][3];
            int phase_index = -1;

            for (int i = 0; i < tnoA; i++) {
                int ig = Anum + i;
                int j_start = ig - Bnum;
                int brow = (ig - 1) / nblk;
                int prow = brow % np_rows;

                if (j_start < 0) {
                    j_start = 0;
                } else if (j_start > tnoB) {
                    j_start = tnoB;
                }

                for (int j = 0; j < tnoB; j++, h_index++) {
                    int jg = Bnum + j;

                    if (want_dense && j_start <= j) {
                        if (phase_index < 0) {
                            if (l1 != prev_dense_l1 || l2 != prev_dense_l2 || l3 != prev_dense_l3) {
                                phase_index = dense_phase_count++;
                                cache->dense_phase_l1[phase_index] = l1;
                                cache->dense_phase_l2[phase_index] = l2;
                                cache->dense_phase_l3[phase_index] = l3;
                                prev_dense_l1 = l1;
                                prev_dense_l2 = l2;
                                prev_dense_l3 = l3;
                            } else {
                                phase_index = dense_phase_count - 1;
                            }
                        }

                        BandColConstructEntry *entry = &dense_host[dense_count++];
                        entry->h_index = h_index;
                        entry->index0  = (jg - 1) * n + (ig - 1);
                        entry->index1  = (jg > ig) ? (ig - 1) * n + (jg - 1) : -1;
                        entry->l1      = l1;
                        entry->l2      = l2;
                        entry->l3      = l3;
                        entry->phase_index = phase_index;
                    }

                    int bcol = (jg - 1) / nblk;
                    int pcol = bcol % np_cols;

                    if (!want_dense && my_prow == prow && my_pcol == pcol) {
                        int il = (brow / np_rows + 1) * nblk + 1;
                        int jl = (bcol / np_cols + 1) * nblk + 1;

                        if (((my_prow + np_rows) % np_rows) >= (brow % np_rows)) {
                            if (my_prow == prow) {
                                il += (ig - 1) % nblk;
                            }
                            il -= nblk;
                        }

                        if (((my_pcol + np_cols) % np_cols) >= (bcol % np_cols)) {
                            if (my_pcol == pcol) {
                                jl += (jg - 1) % nblk;
                            }
                            jl -= nblk;
                        }

                        BandColConstructEntry *entry = &cache->local_entries[local_count++];
                        entry->h_index = h_index;
                        entry->index0  = (jl - 1) * na_rows + il - 1;
                        entry->index1  = -1;
                        entry->l1      = l1;
                        entry->l2      = l2;
                        entry->l3      = l3;
                        entry->phase_index = -1;
                    }
                }
            }
        }
    }

    if (want_dense && 0 < dense_count) {
        /* the dense entry table is read exclusively by the device construct
           kernel, so it lives in device memory only; the host copy above is
           a build scratch and is released right here */
        size_t dense_bytes = sizeof(BandColConstructEntry) * (size_t)dense_count;

        wait_hipfunc(hipMalloc((void **)&cache->dense_entries_dev, dense_bytes));
        wait_hipfunc(hipMemcpy(cache->dense_entries_dev, dense_host, dense_bytes, hipMemcpyHostToDevice));
    }
    free(dense_host);

    cache->valid       = 1;
    cache->n           = n;
    cache->na_rows     = na_rows;
    cache->na_cols     = na_cols;
    cache->nblk        = nblk;
    cache->np_rows     = np_rows;
    cache->np_cols     = np_cols;
    cache->my_prow     = my_prow;
    cache->my_pcol     = my_pcol;
    cache->fingerprint = fingerprint;
    cache->dense_count = dense_count;
    cache->local_count = local_count;
    cache->h_count     = h_index;
    cache->dense_phase_count = dense_phase_count;
    cache->has_dense   = want_dense;
    cache->has_local   = !want_dense;
}

static void BandCol_DMWorkspace_Reset(void)
{
    BandColDMWorkspace *ws = &BandCol_dm_workspace;

    free(ws->buf_Re0);
    free(ws->buf_Im0);
    free(ws->buf_Re1);
    free(ws->buf_Im1);
    free(ws->TmpEIGEN);
    free(ws->OccWeight);
    free(ws->ReEVec0);
    free(ws->ImEVec0);
    free(ws->ReEVec1);
    free(ws->ImEVec1);

    memset(ws, 0, sizeof(*ws));
}

static void BandCol_DMWorkspace_Ensure(int max_tno, int nk)
{
    BandColDMWorkspace *ws = &BandCol_dm_workspace;
    size_t              vec_bytes;

    if (max_tno <= 0 || nk <= 0) {
        BandCol_AbortWithMessage("Invalid DM workspace size in Band_DFT_Col.c.");
    }

    if (ws->max_tno >= max_tno && ws->max_nk >= nk) {
        return;
    }

    BandCol_DMWorkspace_Reset();

    vec_bytes = sizeof(double) * (size_t)max_tno * (size_t)nk;

    ws->buf_Re0  = (double *)malloc(vec_bytes);
    ws->buf_Im0  = (double *)malloc(vec_bytes);
    ws->buf_Re1  = (double *)malloc(vec_bytes);
    ws->buf_Im1  = (double *)malloc(vec_bytes);
    ws->TmpEIGEN = (double *)malloc(sizeof(double) * (size_t)nk);
    ws->OccWeight = (double *)malloc(sizeof(double) * (size_t)nk);
    ws->ReEVec0  = (double **)malloc(sizeof(double *) * (size_t)max_tno);
    ws->ImEVec0  = (double **)malloc(sizeof(double *) * (size_t)max_tno);
    ws->ReEVec1  = (double **)malloc(sizeof(double *) * (size_t)max_tno);
    ws->ImEVec1  = (double **)malloc(sizeof(double *) * (size_t)max_tno);

    if (ws->buf_Re0 == NULL || ws->buf_Im0 == NULL || ws->buf_Re1 == NULL || ws->buf_Im1 == NULL ||
        ws->TmpEIGEN == NULL || ws->OccWeight == NULL || ws->ReEVec0 == NULL || ws->ImEVec0 == NULL || ws->ReEVec1 == NULL ||
        ws->ImEVec1 == NULL) {
        BandCol_DMWorkspace_Reset();
        BandCol_AbortWithMessage("Failed to allocate DM workspace in Band_DFT_Col.c.");
    }

    ws->max_tno = max_tno;
    ws->max_nk  = nk;
}

static void BandCol_ZeroComplex(dcomplex *a, size_t count)
{
    memset(a, 0, sizeof(dcomplex) * count);
}

static void BandCol_DMWorkspace_SetPointers(BandColDMWorkspace *ws, int max_tno, int nk)
{
    for (int i = 0; i < max_tno; ++i) {
        ws->ReEVec0[i] = ws->buf_Re0 + (size_t)i * (size_t)nk;
        ws->ImEVec0[i] = ws->buf_Im0 + (size_t)i * (size_t)nk;
        ws->ReEVec1[i] = ws->buf_Re1 + (size_t)i * (size_t)nk;
        ws->ImEVec1[i] = ws->buf_Im1 + (size_t)i * (size_t)nk;
    }
}

static void BandCol_DMEntryCache_Reset(void)
{
    free(BandCol_dm_entry_cache.basis0);
    free(BandCol_dm_entry_cache.basis1);
    free(BandCol_dm_entry_cache.phase_index);
    free(BandCol_dm_entry_cache.pair_l1);
    free(BandCol_dm_entry_cache.pair_l2);
    free(BandCol_dm_entry_cache.pair_l3);
    free(BandCol_dm_entry_cache.phase_r);
    free(BandCol_dm_entry_cache.phase_i);
    memset(&BandCol_dm_entry_cache, 0, sizeof(BandCol_dm_entry_cache));
}

static void BandCol_DMEntryCache_Ensure(int *order_GA, int *MP, int n)
{
    BandColDMEntryCache *cache = &BandCol_dm_entry_cache;
    unsigned long long   fingerprint = BandCol_ConstructFingerprint(order_GA, MP);
    int                  entry_count = 0;
    int                  pair_count = 0;

    if (cache->valid && cache->n == n && cache->fingerprint == fingerprint) {
        return;
    }

    BandCol_DMEntryCache_Reset();

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int wanA  = WhatSpecies[GA_AN];
        int tnoA  = Spe_Total_CNO[wanA];

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int wanB  = WhatSpecies[GB_AN];
            int tnoB  = Spe_Total_CNO[wanB];

            pair_count++;
            entry_count += tnoA * tnoB;
        }
    }

    cache->basis0 = (int *)malloc(sizeof(int) * (size_t)(entry_count + 1));
    cache->basis1 = (int *)malloc(sizeof(int) * (size_t)(entry_count + 1));
    cache->phase_index = (int *)malloc(sizeof(int) * (size_t)(entry_count + 1));
    cache->pair_l1 = (int *)malloc(sizeof(int) * (size_t)(pair_count + 1));
    cache->pair_l2 = (int *)malloc(sizeof(int) * (size_t)(pair_count + 1));
    cache->pair_l3 = (int *)malloc(sizeof(int) * (size_t)(pair_count + 1));
    cache->phase_r = (double *)malloc(sizeof(double) * (size_t)(pair_count + 1));
    cache->phase_i = (double *)malloc(sizeof(double) * (size_t)(pair_count + 1));

    if (cache->basis0 == NULL || cache->basis1 == NULL || cache->phase_index == NULL || cache->pair_l1 == NULL ||
        cache->pair_l2 == NULL || cache->pair_l3 == NULL || cache->phase_r == NULL || cache->phase_i == NULL) {
        BandCol_DMEntryCache_Reset();
        BandCol_AbortWithMessage("Failed to allocate DM entry cache in Band_DFT_Col.c.");
    }

    entry_count = 0;
    pair_count  = 0;

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int wanA  = WhatSpecies[GA_AN];
        int tnoA  = Spe_Total_CNO[wanA];
        int Anum  = MP[GA_AN];

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int Rn    = ncn[GA_AN][LB_AN];
            int wanB  = WhatSpecies[GB_AN];
            int tnoB  = Spe_Total_CNO[wanB];
            int Bnum  = MP[GB_AN];
            int pair  = pair_count++;

            cache->pair_l1[pair] = atv_ijk[Rn][1];
            cache->pair_l2[pair] = atv_ijk[Rn][2];
            cache->pair_l3[pair] = atv_ijk[Rn][3];

            for (int i = 0; i < tnoA; i++) {
                int ibasis = Anum + i - 1;

                for (int j = 0; j < tnoB; j++) {
                    cache->basis0[entry_count]      = ibasis;
                    cache->basis1[entry_count]      = Bnum + j - 1;
                    cache->phase_index[entry_count] = pair;
                    entry_count++;
                }
            }
        }
    }

    cache->valid       = 1;
    cache->n           = n;
    cache->fingerprint = fingerprint;
    cache->entry_count = entry_count;
    cache->pair_count  = pair_count;
}

static void BandCol_UpdateDMEntryPhases(double k1, double k2, double k3)
{
    BandColDMEntryCache *cache = &BandCol_dm_entry_cache;

    for (int pair = 0; pair < cache->pair_count; pair++) {
        double kRn = k1 * (double)cache->pair_l1[pair] + k2 * (double)cache->pair_l2[pair] +
                     k3 * (double)cache->pair_l3[pair];
        cache->phase_i[pair] = sin(2.0 * PI * kRn);
        cache->phase_r[pair] = cos(2.0 * PI * kRn);
    }
}

static void BandCol_AccumulateDenseTransposedDM(int n, int nk, int max_tno, int spin, int kloop, double k1,
                                                double k2, double k3, const dcomplex *evec_t, int evec_stride,
                                                int *MP, double ***EIGEN, const double *occ_weight, double *CDM1,
                                                double *EDM1)
{
    BandColDMWorkspace *ws;
    double *            TmpEIGEN_local;
    double **           ReEVec0_local;
    double **           ImEVec0_local;
    double **           ReEVec1_local;
    double **           ImEVec1_local;
    size_t              p = 0;

    BandCol_DMWorkspace_Ensure(max_tno, nk);

    ws = &BandCol_dm_workspace;
    BandCol_DMWorkspace_SetPointers(ws, max_tno, nk);

    TmpEIGEN_local = ws->TmpEIGEN;
    ReEVec0_local  = ws->ReEVec0;
    ImEVec0_local  = ws->ImEVec0;
    ReEVec1_local  = ws->ReEVec1;
    ImEVec1_local  = ws->ImEVec1;

    for (int k = 0; k < nk; ++k) {
        TmpEIGEN_local[k] = EIGEN[spin][kloop][k + 1];
    }

    for (int GA_AN = 1; GA_AN <= atomnum; ++GA_AN) {
        const int wanA = WhatSpecies[GA_AN];
        const int tnoA = Spe_Total_CNO[wanA];
        const int Anum = MP[GA_AN];

        for (int i = 0; i < tnoA; ++i) {
            const dcomplex * restrict v  = &evec_t[(size_t)(Anum + i - 1) * (size_t)evec_stride];
            double * restrict          r  = ReEVec0_local[i];
            double * restrict          im = ImEVec0_local[i];

            for (int k = 0; k < nk; ++k) {
                const double w = occ_weight[k];
                r[k]           = v[k].r * w;
                im[k]          = v[k].i * w;
            }
        }

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; ++LB_AN) {
            const int GB_AN = natn[GA_AN][LB_AN];
            const int Rn    = ncn[GA_AN][LB_AN];
            const int wanB  = WhatSpecies[GB_AN];
            const int tnoB  = Spe_Total_CNO[wanB];
            const int Bnum  = MP[GB_AN];

            const int    l1  = atv_ijk[Rn][1];
            const int    l2  = atv_ijk[Rn][2];
            const int    l3  = atv_ijk[Rn][3];
            const double kRn = k1 * l1 + k2 * l2 + k3 * l3;
            const double si  = sin(2.0 * PI * kRn);
            const double co  = cos(2.0 * PI * kRn);

            for (int j = 0; j < tnoB; ++j) {
                const dcomplex * restrict v  = &evec_t[(size_t)(Bnum + j - 1) * (size_t)evec_stride];
                double * restrict          r  = ReEVec1_local[j];
                double * restrict          im = ImEVec1_local[j];

                for (int k = 0; k < nk; ++k) {
                    const double w = occ_weight[k];
                    r[k]           = v[k].r * w;
                    im[k]          = v[k].i * w;
                }
            }

            for (int i = 0; i < tnoA; ++i) {
                const double * restrict r0  = ReEVec0_local[i];
                const double * restrict im0 = ImEVec0_local[i];

                for (int j = 0; j < tnoB; ++j, ++p) {
                    const double * restrict r1  = ReEVec1_local[j];
                    const double * restrict im1 = ImEVec1_local[j];
                    double                  d1 = 0.0, d2 = 0.0, d3 = 0.0, d4 = 0.0;
                    int                     k = 0;

                    for (; k + 3 < nk; k += 4) {
#define KSTEP(idx)                                                                                                     \
    do {                                                                                                               \
        double reA = r0[idx] * r1[idx] + im0[idx] * im1[idx];                                                          \
        double imA = r0[idx] * im1[idx] - im0[idx] * r1[idx];                                                          \
        d1 += reA;                                                                                                     \
        d2 += imA;                                                                                                     \
        d3 += reA * TmpEIGEN_local[idx];                                                                               \
        d4 += imA * TmpEIGEN_local[idx];                                                                               \
    } while (0)
                        KSTEP(k);
                        KSTEP(k + 1);
                        KSTEP(k + 2);
                        KSTEP(k + 3);
#undef KSTEP
                    }

                    for (; k < nk; ++k) {
                        double reA = r0[k] * r1[k] + im0[k] * im1[k];
                        double imA = r0[k] * im1[k] - im0[k] * r1[k];
                        d1 += reA;
                        d2 += imA;
                        d3 += reA * TmpEIGEN_local[k];
                        d4 += imA * TmpEIGEN_local[k];
                    }

                    CDM1[p] += co * d1 - si * d2;
                    EDM1[p] += co * d3 - si * d4;
                }
            }
        }
    }
}

/* Returns 0 on success and -1 when the HIP path failed (the device buffers
   are released in that case, CDM1/EDM1 are untouched); the fallback loops
   then redo the k point on the CPU, the dense paths abort. */
static int BandCol_AccumulateDenseTransposedDM_Device(int n, int nk, int spin, int kloop, double k1, double k2,
                                                      double k3, const dcomplex *evec_device, int evec_stride, int *MP,
                                                      int *order_GA, double ***EIGEN, const double *occ_weight,
                                                      double *CDM1, double *EDM1, int size_H1, int state_first, MPI_Comm dm_comm, int host_vectors)
{
    const int active=nk>0 && evec_device!=NULL;
    BandColDMEntryCache *cache=&BandCol_dm_entry_cache;
    int done=0;
    if(active){
        BandCol_DMEntryCache_Ensure(order_GA,MP,n);BandCol_UpdateDMEntryPhases(k1,k2,k3);
        if(cache->entry_count!=size_H1)BandCol_AbortWithMessage("DM entry count mismatch");
        if(!host_vectors && !openmx_gpu_capacity_test_recovery())
            done=BandCol_AccumulateDenseTransposedDM_HIP(cache->entry_count,cache->pair_count,nk,evec_stride,
                cache->basis0,cache->basis1,cache->phase_index,cache->phase_r,cache->phase_i,
                EIGEN[spin][kloop]+state_first,occ_weight,evec_device,CDM1,EDM1)==0;
    }
    double *outputs[10]={CDM1,EDM1};
    openmx_gpu_dm_collective(dm_comm,OPENMX_GPU_DM_COL,n,active && !done?nk:0,evec_stride,
        active && !done?cache->entry_count:0,evec_device,host_vectors,cache->basis0,cache->basis1,
        active && !done?cache->phase_index:NULL,cache->pair_count,cache->phase_r,cache->phase_i,
        occ_weight,active?EIGEN[spin][kloop]+state_first:NULL,NULL,outputs,1);
    return 0;
}

/* GPU DM accumulation inside the ScaLAPACK/ELPA fallback of the collinear
   band path.  The preflight refuses the full dense solve when its three
   n x n matrices plus the gpusolver workspace do not fit, but the DM step
   only needs the eigenvector panel and the packed DM arrays, which are far
   smaller, so it can stay on the GPU (the collinear cluster and the
   noncollinear band already keep their fallback DM there).  Sizes below the
   band threshold keep the CPU loop, which wins at those sizes.  Opt out
   with OPENMX_BAND_GPU_DM=0. */
static int BandCol_UseGpuFallbackDM(int n)
{
    const char *value = getenv("OPENMX_BAND_GPU_DM");

    if (value != NULL && atoi(value) == 0) return 0;
    return (scf_eigen_lib_flag == GPUSOLVER && Band_DFT_Col_GpuSwitchNum() <= n);
}

/* Uploads the host eigenvector panel of one fallback k point; returns NULL
   (leaving no latched HIP error behind) when the device cannot take it, so
   the caller falls back to the CPU loop. */
static dcomplex *BandCol_FallbackDMUpload(const dcomplex *host, size_t count)
{
    size_t bytes = count * sizeof(dcomplex);
    size_t free_bytes = 0, total_bytes = 0;
    void  *device = NULL;

    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        (void)hipGetLastError();
        return NULL;
    }
    /* Every rank sharing the device uploads its own panel and launches its
       own kernel at about the same time, and the launch pays the library
       reserve like the eigensolvers do, so the check is device-wide; a
       kernel that fails to launch for lack of memory cannot be caught. */
    if (bytes > SIZE_MAX / (size_t)BandCol_DeviceRanksCached() ||
        free_bytes < (bytes + BandCol_GpuLibraryReserveBytes()) * (size_t)BandCol_DeviceRanksCached()) return NULL;
    if (hipMalloc(&device, bytes) != hipSuccess) {
        (void)hipGetLastError();
        return NULL;
    }
    if (hipMemcpy(device, host, bytes, hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipGetLastError();
        if (hipFree(device) != hipSuccess)
            (void)hipGetLastError();
        return NULL;
    }
    return (dcomplex *)device;
}

static void BandCol_GpuSolver_ReleaseDeviceMemory(void)
{
    BandColGpuSolverCtx * ctx = &BandCol_gpusolver_ctx;

    if (ctx->d_S != NULL)
        wait_hipfunc(hipFree(ctx->d_S));
    if (ctx->d_H != NULL)
        wait_hipfunc(hipFree(ctx->d_H));
    if (ctx->d_tmp != NULL)
        wait_hipfunc(hipFree(ctx->d_tmp));
    if (ctx->d_evec != NULL)
        wait_hipfunc(hipFree(ctx->d_evec));
    if (ctx->d_scale != NULL)
        wait_hipfunc(hipFree(ctx->d_scale));
    if (ctx->d_W != NULL)
        wait_hipfunc(hipFree(ctx->d_W));
    if (ctx->d_info != NULL)
        wait_hipfunc(hipFree(ctx->d_info));
    if (ctx->d_work != NULL)
        wait_hipfunc(hipFree(ctx->d_work));
    if (ctx->h_work != NULL)
        free(ctx->h_work);
    if (ctx->h_scale != NULL)
        free(ctx->h_scale);
    if (ctx->gpusolver != NULL)
        wait_hipfunc(hipsolverDnDestroy(ctx->gpusolver));
    if (ctx->hipblas != NULL)
        wait_hipfunc(hipblasDestroy(ctx->hipblas));

    ctx->initialized         = 0;
    ctx->device_id           = -1;
    ctx->matrix_dim          = 0;
    ctx->scale_dim           = 0;
    ctx->d_evec_dim          = 0;
    ctx->d_evec_stride       = 0;
    ctx->d_evec_count        = 0;
    ctx->transformed_s_valid = 0;
    ctx->transformed_s_dim   = 0;
    ctx->d_work_bytes        = 0;
    ctx->h_work_bytes        = 0;
    ctx->hipblas              = NULL;
    ctx->gpusolver            = NULL;
    ctx->d_S                 = NULL;
    ctx->d_H                 = NULL;
    ctx->d_tmp               = NULL;
    ctx->d_evec              = NULL;
    ctx->d_scale             = NULL;
    ctx->d_W                 = NULL;
    ctx->d_info              = NULL;
    ctx->d_work              = NULL;
    ctx->h_work              = NULL;
    ctx->h_scale             = NULL;
}

static void BandCol_GpuSolver_Destroy(void)
{
    BandColGpuSolverCtx * ctx = &BandCol_gpusolver_ctx;

    BandCol_RefineRelease();
    BandCol_GpuSolver_ReleaseDeviceMemory();
    free(ctx->h_evec);
    free(ctx->h_transformed_s);
    ctx->h_evec     = NULL;
    ctx->h_evec_dim = 0;
    ctx->h_evec_stride = 0;
    ctx->h_evec_count = 0;
    ctx->h_transformed_s     = NULL;
    ctx->h_transformed_s_dim = 0;
}

/* Hard reset for a run boundary (between -runtest/-runtestL inputs every
   host array of the previous system is freed and reallocated, so a device
   cache validated through a host address or a matrix size could accept
   stale data once malloc reuses an address).  Idempotent; every branch is
   a no-op when the corresponding path never ran. */
void Band_DFT_Col_Release_GPU_Caches(void)
{
    free(BandCol_host_s); free(BandCol_host_h); free(BandCol_host_tmp);
    BandCol_host_s=BandCol_host_h=BandCol_host_tmp=NULL;
    BandCol_host_n=BandCol_host_stride=BandCol_mgpu_host_dense=0;
    BandCol_KOverlapReset();
    (void)BandCol_ReleaseDMDeviceWorkspace_HIP();
    BandCol_ConstructCache_Reset();
    BandCol_DMWorkspace_Reset();
    BandCol_DMEntryCache_Reset();
    BandCol_GpuSolver_Destroy();
    /* the refinement's stage (iterations, a cycle or run turned to FP64)
       starts over with the next input, read again by every rank */
    BandCol_RefineResetStage();
}

static dcomplex *BandCol_GpuSolver_SaveDeviceEigenvectors(dcomplex *evec_device, int n, int stride)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    size_t              evec_count;

    if (evec_device == NULL || n <= 0 || stride <= 0) {
        BandCol_AbortWithMessage("MAGMA dense device eigenvectors are not available in Band_DFT_Col.c.");
    }

    evec_count = (size_t)n * (size_t)stride;

    if (ctx->h_evec_count < evec_count) {
        dcomplex *new_evec = (dcomplex *)realloc(ctx->h_evec, sizeof(dcomplex) * evec_count);
        if (new_evec == NULL) {
            BandCol_AbortWithMessage("Failed to allocate host eigenvector buffer in Band_DFT_Col.c.");
        }
        ctx->h_evec     = new_evec;
        ctx->h_evec_count = evec_count;
    }
    ctx->h_evec_dim = n;
    ctx->h_evec_stride = stride;

    if (BandCol_mgpu_host_dense) memcpy(ctx->h_evec,evec_device,sizeof(dcomplex)*evec_count);
    else wait_hipfunc(hipMemcpy(ctx->h_evec, evec_device, sizeof(dcomplex) * evec_count, hipMemcpyDeviceToHost));
    return ctx->h_evec;
}

static dcomplex *BandCol_GpuSolver_HostEigenvectors(void)
{
    return BandCol_gpusolver_ctx.h_evec;
}

static dcomplex *BandCol_GpuSolver_UploadHostEigenvectors(int n)
{
    if (BandCol_mgpu_host_dense) return BandCol_gpusolver_ctx.h_evec;
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    size_t              evec_count;
    int                 stride;

    if (n <= 0) {
        BandCol_AbortWithMessage("Invalid eigenvector dimension in Band_DFT_Col.c.");
    }
    if (ctx->h_evec == NULL || ctx->h_evec_dim != n || ctx->h_evec_stride <= 0) {
        BandCol_AbortWithMessage("MAGMA dense host eigenvectors are not available in Band_DFT_Col.c.");
    }

    stride = ctx->h_evec_stride;
    evec_count = (size_t)n * (size_t)stride;

    if (ctx->d_evec_count < evec_count) {
        if (ctx->d_evec != NULL) {
            wait_hipfunc(hipFree(ctx->d_evec));
        }
        wait_hipfunc(hipMalloc((void **)&ctx->d_evec, sizeof(dcomplex) * evec_count));
        ctx->d_evec_count = evec_count;
    }
    ctx->d_evec_dim = n;
    ctx->d_evec_stride = stride;

    wait_hipfunc(hipMemcpy(ctx->d_evec, ctx->h_evec, sizeof(dcomplex) * evec_count, hipMemcpyHostToDevice));
    return ctx->d_evec;
}

static int BandCol_GpuSolver_EigenvectorStride(void)
{
    if (BandCol_mgpu_host_dense) return BandCol_host_stride;
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;

    if (ctx->d_evec != NULL && 0 < ctx->d_evec_stride) {
        return ctx->d_evec_stride;
    }
    if (ctx->h_evec != NULL && 0 < ctx->h_evec_stride) {
        return ctx->h_evec_stride;
    }
    return ctx->matrix_dim;
}

static int BandCol_GpuSolver_HasHostTransformedS(int n)
{
    if (BandCol_mgpu_host_dense) return BandCol_host_n==n;
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;

    return (ctx->h_transformed_s != NULL && ctx->h_transformed_s_dim == n);
}

static int BandCol_GpuSolver_HasDeviceTransformedS(int n)
{
    if (BandCol_mgpu_host_dense) return BandCol_host_n==n;
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;

    return (ctx->initialized && ctx->transformed_s_valid && ctx->transformed_s_dim == n);
}

static void BandCol_GpuSolver_SaveTransformedS(int n)
{
    if (BandCol_mgpu_host_dense) return;
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    size_t              matrix_count;

    matrix_count = (size_t)n * (size_t)n;

    if (ctx->h_transformed_s_dim < n) {
        dcomplex *new_s = (dcomplex *)realloc(ctx->h_transformed_s, sizeof(dcomplex) * matrix_count);
        if (new_s == NULL) {
            BandCol_AbortWithMessage("Failed to allocate host transformed-overlap buffer in Band_DFT_Col.c.");
        }
        ctx->h_transformed_s = new_s;
    }

    wait_hipfunc(hipMemcpy(ctx->h_transformed_s, ctx->d_S, sizeof(dcomplex) * matrix_count, hipMemcpyDeviceToHost));
    ctx->h_transformed_s_dim = n;
}

static void BandCol_GpuSolver_LoadTransformedS(int n)
{
    if (BandCol_mgpu_host_dense) return;
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    size_t              matrix_count;

    if (!BandCol_GpuSolver_HasHostTransformedS(n)) {
        BandCol_AbortWithMessage("Cached transformed overlap is not available in Band_DFT_Col.c.");
    }

    matrix_count = (size_t)n * (size_t)n;
    BandCol_GpuSolver_EnsureMatrixCapacity(n);
    wait_hipfunc(hipMemcpy(ctx->d_S, ctx->h_transformed_s, sizeof(dcomplex) * matrix_count, hipMemcpyHostToDevice));
    ctx->transformed_s_valid = 1;
    ctx->transformed_s_dim   = n;
}

static void BandCol_GpuSolver_ClearHostEigenvectors(void)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;

    free(ctx->h_evec);
    ctx->h_evec     = NULL;
    ctx->h_evec_dim = 0;
    ctx->h_evec_stride = 0;
    ctx->h_evec_count = 0;
}

static void BandCol_GpuSolver_Init(void)
{
    BandColGpuSolverCtx * ctx = &BandCol_gpusolver_ctx;
    int                  current_device;

    wait_hipfunc(hipGetDevice(&current_device));

    if (ctx->initialized && ctx->device_id == current_device) {
        return;
    }

    if (ctx->initialized) {
        BandCol_GpuSolver_Destroy();
    }

    wait_hipfunc(hipblasCreate(&ctx->hipblas));
    ctx->gpusolver = NULL;

    ctx->initialized = 1;
    ctx->device_id   = current_device;
}

static void BandCol_GpuSolver_EnsureMatrixCapacity(int n)
{
    BandColGpuSolverCtx * ctx = &BandCol_gpusolver_ctx;
    size_t               matrix_bytes;
    size_t               vector_bytes;

    if (n <= 0) {
        BandCol_AbortWithMessage("Invalid matrix size in BandCol_GpuSolver_EnsureMatrixCapacity.");
    }

    BandCol_GpuSolver_Init();

    if (n <= ctx->matrix_dim) {
        return;
    }

    if (ctx->d_S != NULL)
        wait_hipfunc(hipFree(ctx->d_S));
    if (ctx->d_H != NULL)
        wait_hipfunc(hipFree(ctx->d_H));
    if (ctx->d_tmp != NULL)
        wait_hipfunc(hipFree(ctx->d_tmp));
    if (ctx->d_scale != NULL)
        wait_hipfunc(hipFree(ctx->d_scale));
    if (ctx->d_W != NULL)
        wait_hipfunc(hipFree(ctx->d_W));
    if (ctx->d_info != NULL)
        wait_hipfunc(hipFree(ctx->d_info));

    matrix_bytes = sizeof(dcomplex) * (size_t)n * (size_t)n;
    vector_bytes = sizeof(dcomplex) * (size_t)n;

    wait_hipfunc(hipMalloc((void **)&ctx->d_S, matrix_bytes));
    wait_hipfunc(hipMalloc((void **)&ctx->d_H, matrix_bytes));
    wait_hipfunc(hipMalloc((void **)&ctx->d_tmp, matrix_bytes));
    wait_hipfunc(hipMalloc((void **)&ctx->d_scale, vector_bytes));
    wait_hipfunc(hipMalloc((void **)&ctx->d_W, sizeof(double) * (size_t)n));
    wait_hipfunc(hipMalloc((void **)&ctx->d_info, sizeof(int32_t)));

    ctx->matrix_dim          = n;
    ctx->transformed_s_valid = 0;
    ctx->transformed_s_dim   = 0;
}

static void BandCol_GpuSolver_Eigen(dcomplex * d_A, int m, int maxn, double * W_host)
{
    int  info;
    int  mout = 0;
    char msg[256];

    if (m <= 0 || maxn <= 0 || maxn > m || d_A == NULL || W_host == NULL) {
        BandCol_AbortWithMessage("Invalid MAGMA eigensolver dimensions in Band_DFT_Col.c.");
    }

    BandCol_GpuSolver_EnsureMatrixCapacity(m);

    if (openmx_gpu_eigensolver_use_hipsolver()) {
        info = openmx_hipsolver_zheevd_gpu(m, maxn, d_A, W_host, &mout);
    } else {
        info = openmx_magma_zheevdx_gpu(m, maxn, d_A, W_host, &mout);
    }

    if (info != 0) {
        snprintf(msg, sizeof(msg), "GPU eigensolver (zheevd/zheevdx) failed in Band_DFT_Col.c: info=%d", info);
        BandCol_AbortWithMessage(msg);
    }
    if (mout != maxn) {
        snprintf(msg, sizeof(msg), "GPU eigensolver returned %d eigenpairs, expected %d in Band_DFT_Col.c.",
                 mout, maxn);
        BandCol_AbortWithMessage(msg);
    }
}

static void BandCol_HostEnsure(int n)
{
    if (BandCol_host_n==n) return;
    free(BandCol_host_s); free(BandCol_host_h); free(BandCol_host_tmp);
    BandCol_host_s=malloc((size_t)n*n*sizeof(dcomplex));
    BandCol_host_h=malloc((size_t)n*n*sizeof(dcomplex));
    BandCol_host_tmp=malloc((size_t)n*n*sizeof(dcomplex));
    if (!BandCol_host_s || !BandCol_host_h || !BandCol_host_tmp)
        BandCol_AbortWithMessage("Cannot allocate MAGMA host dense workspace");
    BandCol_host_n=n;
}

static void BandCol_GpuSolver_PrepareTransformedS(int build_from_overlap, int n, dcomplex * Ss, double * ko,
                                                 double * koS)
{
    if (BandCol_mgpu_host_dense) {
        BandCol_HostEnsure(n);
        if (build_from_overlap) {
            /* Construct_Band_CsHs populated the private full matrix. Ss
               may be only a local ScaLAPACK block on this MPI rank. */
            if (!openmx_mgpu_host_solve(n,n,1,BandCol_host_s,n,ko+1)) BandCol_AbortWithMessage(openmx_mgpu_host_error());
#pragma omp parallel for
            for (int j=0;j<n;++j) {
                if (ko[j+1]<0.0) ko[j+1]=1.0e-10;
                koS[j+1]=ko[j+1];
                const double scale=1.0/sqrt(ko[j+1]);
                for (int i=0;i<n;++i) {
                    BandCol_host_s[(size_t)j*n+i].r*=scale;
                    BandCol_host_s[(size_t)j*n+i].i*=scale;
                }
            }
        }
        BandCol_gpusolver_ctx.transformed_s_valid=1;
        BandCol_gpusolver_ctx.transformed_s_dim=n;
        return;
    }

    BandColGpuSolverCtx * ctx = &BandCol_gpusolver_ctx;
    size_t               matrix_bytes;
    int                  l;

    BandCol_GpuSolver_EnsureMatrixCapacity(n);

    if (!build_from_overlap && ctx->transformed_s_valid && ctx->transformed_s_dim == n) {
        return;
    }

    matrix_bytes = sizeof(dcomplex) * (size_t)n * (size_t)n;

    if (build_from_overlap) {
        if (Ss != NULL) {
            wait_hipfunc(hipMemcpy(ctx->d_S, Ss, matrix_bytes, hipMemcpyHostToDevice));
        }
        BandCol_GpuSolver_Eigen(ctx->d_S, n, n, ko + 1);

        if (n > ctx->scale_dim) {
            dcomplex *new_scale = (dcomplex *)realloc(ctx->h_scale, sizeof(dcomplex) * (size_t)n);
            if (new_scale == NULL) {
                BandCol_AbortWithMessage("Failed to allocate overlap scale buffer in Band_DFT_Col.c.");
            }
            ctx->h_scale  = new_scale;
            ctx->scale_dim = n;
        }

        for (l = 1; l <= n; l++) {
            if (ko[l] < 0.0) {
                ko[l] = 1.0e-10;
            }
            koS[l]              = ko[l];
            ctx->h_scale[l - 1].r = 1.0 / sqrt(ko[l]);
            ctx->h_scale[l - 1].i = 0.0;
        }

        wait_hipfunc(hipMemcpy(ctx->d_scale, ctx->h_scale, sizeof(dcomplex) * (size_t)n, hipMemcpyHostToDevice));

        if (!openmx_gpu_transform ||
            !openmx_gpu_transform(OPENMX_GPU_SCALE_COLUMNS,1,n,n,ctx->d_S,n,ctx->d_scale,0,ctx->d_tmp,n,1,0.0))
        wait_hipfunc(hipblasZdgmm(ctx->hipblas, HIPBLAS_SIDE_RIGHT, n, n, (hipDoubleComplex *)ctx->d_S, n,
                                  (hipDoubleComplex *)ctx->d_scale, 1, (hipDoubleComplex *)ctx->d_tmp, n));

        {
            dcomplex * swap_tmp = ctx->d_S;
            ctx->d_S            = ctx->d_tmp;
            ctx->d_tmp          = swap_tmp;
        }

    } else {
        if (Ss != NULL) {
            wait_hipfunc(hipMemcpy(ctx->d_S, Ss, matrix_bytes, hipMemcpyHostToDevice));
        } else if (!(ctx->transformed_s_valid && ctx->transformed_s_dim == n)) {
            BandCol_AbortWithMessage("Device overlap is not ready in BandCol_GpuSolver_PrepareTransformedS.");
        }
    }

    ctx->transformed_s_valid = 1;
    ctx->transformed_s_dim   = n;
}

static void BandCol_ZgemmLogCpuFallbackOnce(hipblasStatus_t status, int m, int n, int k)
{
    static int logged = 0;

    if (logged) {
        return;
    }
    fprintf(stderr,
            "<Band_DFT_Col> GPU ZGEMM (GEMMul8/MAGMA) failed for m=%d,n=%d,k=%d: status=%d. "
            "Falling back to CPU BLAS zgemm.\n",
            m, n, k, (int)status);
    fflush(stderr);
    logged = 1;
}

static void BandCol_ZgemmCpu(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k,
                             dcomplex const *A, dcomplex const *B, dcomplex *C)
{
    char transa_cpu = (transa == HIPBLAS_OP_C) ? 'C' : ((transa == HIPBLAS_OP_T) ? 'T' : 'N');
    char transb_cpu = (transb == HIPBLAS_OP_C) ? 'C' : ((transb == HIPBLAS_OP_T) ? 'T' : 'N');
    int mi = m;
    int ni = n;
    int ki = k;
    int lda = m;
    int ldb = k;
    int ldc = m;
    dcomplex alpha = {1.0, 0.0};
    dcomplex beta = {0.0, 0.0};

    zgemm_(&transa_cpu, &transb_cpu, &mi, &ni, &ki, &alpha, (dcomplex *)A, &lda, (dcomplex *)B, &ldb, &beta, C, &ldc);
}

/* ZGEMM on device pointers: GEMMul8 first (which itself falls back to MAGMA when the
   GPU workspace cannot be allocated); if both fail, copy the operands to the host
   and run CPU BLAS zgemm, then copy the result back to the device. */
static void BandCol_GpuSolver_ZgemmDevice(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k,
                                         dcomplex const *d_A, dcomplex const *d_B, dcomplex *d_C)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    hipDoubleComplex     alpha = make_hipDoubleComplex(1.0, 0.0);
    hipDoubleComplex     beta  = make_hipDoubleComplex(0.0, 0.0);
    hipblasStatus_t      status;
    size_t              a_bytes, b_bytes, c_bytes;
    dcomplex           *A, *B, *C;

    status = openmx_gemmul8Zgemm(ctx->hipblas, transa, transb, m, n, k, &alpha, (hipDoubleComplex const *)d_A, m,
                                 (hipDoubleComplex const *)d_B, k, &beta, (hipDoubleComplex *)d_C, m);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        return;
    }

    BandCol_ZgemmLogCpuFallbackOnce(status, m, n, k);

    a_bytes = sizeof(dcomplex) * (size_t)m * (size_t)k;
    b_bytes = sizeof(dcomplex) * (size_t)k * (size_t)n;
    c_bytes = sizeof(dcomplex) * (size_t)m * (size_t)n;

    A = (dcomplex *)malloc(a_bytes);
    B = (dcomplex *)malloc(b_bytes);
    C = (dcomplex *)malloc(c_bytes);
    if (A == NULL || B == NULL || C == NULL) {
        BandCol_AbortWithMessage("Out of host memory in CPU ZGEMM fallback in Band_DFT_Col.c.");
    }

    wait_hipfunc(hipMemcpy(A, d_A, a_bytes, hipMemcpyDeviceToHost));
    wait_hipfunc(hipMemcpy(B, d_B, b_bytes, hipMemcpyDeviceToHost));

    BandCol_ZgemmCpu(transa, transb, m, n, k, A, B, C);

    wait_hipfunc(hipMemcpy(d_C, C, c_bytes, hipMemcpyHostToDevice));

    free(C);
    free(B);
    free(A);
}

static void BandCol_GpuSolver_Zgemm_OpenMP(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k,
                                           dcomplex const * A, dcomplex const * B, dcomplex * C)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    hipDoubleComplex     alpha = make_hipDoubleComplex(1.0, 0.0);
    hipDoubleComplex     beta  = make_hipDoubleComplex(0.0, 0.0);
    hipblasStatus_t      status = HIPBLAS_STATUS_SUCCESS;

    BandCol_GpuSolver_Init();

#pragma omp target data map(present, alloc: A[0 : m * k], B[0 : k * n], C[0 : m * n])
    {
#pragma omp target data use_device_ptr(A, B, C)
        {
            status = openmx_gemmul8Zgemm(ctx->hipblas, transa, transb, m, n, k, &alpha, (hipDoubleComplex const *)A, m,
                                         (hipDoubleComplex const *)B, k, &beta, (hipDoubleComplex *)C, m);
        }

        if (status != HIPBLAS_STATUS_SUCCESS) {
            BandCol_ZgemmLogCpuFallbackOnce(status, m, n, k);
#pragma omp target update from(A[0 : m * k], B[0 : k * n])
            BandCol_ZgemmCpu(transa, transb, m, n, k, A, B, C);
#pragma omp target update to(C[0 : m * n])
        }
    }
}

static void BandCol_GpuSolver_EigenHost(dcomplex *A, double *ko, int n, int maxn)
{
    dcomplex *d_A = NULL;
    size_t bytes;
    int info;
    int mout = 0;
    char msg[256];

    if (n <= 0 || maxn <= 0 || maxn > n || A == NULL || ko == NULL) {
        BandCol_AbortWithMessage("Invalid MAGMA host eigensolver dimensions in Band_DFT_Col.c.");
    }

    bytes = sizeof(dcomplex) * (size_t)n * (size_t)n;
    wait_hipfunc(hipMalloc((void **)&d_A, bytes));
    wait_hipfunc(hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice));

    info = openmx_magma_zheevdx_gpu(n, maxn, d_A, ko, &mout);

    wait_hipfunc(hipMemcpy(A, d_A, bytes, hipMemcpyDeviceToHost));
    wait_hipfunc(hipFree(d_A));

    if (info != 0) {
        snprintf(msg, sizeof(msg), "magma_zheevdx_gpu failed in Band_DFT_Col.c host path: info=%d", info);
        BandCol_AbortWithMessage(msg);
    }
    if (mout != maxn) {
        snprintf(msg, sizeof(msg), "magma_zheevdx_gpu returned %d eigenpairs, expected %d in Band_DFT_Col.c host path.",
                 mout, maxn);
        BandCol_AbortWithMessage(msg);
    }

    for (int i = maxn; i >= 1; i--) {
        ko[i] = ko[i - 1];
    }
}

static dcomplex *BandCol_GpuSolver_EnsureEigenvectorCapacity(int n, int stride)
{
    if(BandCol_mgpu_host_dense){
        if(n<=0 || stride<=0 || stride>n)BandCol_AbortWithMessage("Invalid host eigenvector stride");
        BandCol_HostEnsure(n);BandCol_host_stride=stride;return BandCol_host_tmp;
    }
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    size_t              evec_count;

    if (n <= 0 || stride <= 0) {
        BandCol_AbortWithMessage("Invalid eigenvector buffer dimensions in Band_DFT_Col.c.");
    }

    evec_count = (size_t)n * (size_t)stride;
    if (ctx->d_evec_count < evec_count) {
        if (ctx->d_evec != NULL) {
            wait_hipfunc(hipFree(ctx->d_evec));
        }
        wait_hipfunc(hipMalloc((void **)&ctx->d_evec, sizeof(dcomplex) * evec_count));
        ctx->d_evec_count = evec_count;
    }

    ctx->d_evec_dim = n;
    ctx->d_evec_stride = stride;
    return ctx->d_evec;
}

static int BandCol_SolveTimingEnabled(void)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("OPENMX_SOLVE_TIMING");
        enabled = (value != NULL && atoi(value) != 0);
    }
    return enabled;
}

/* ------------------------------------------------------------------ */
/* The refined FP32 eigensolver (eigen_refine_gpu.cu) on the dense GPU
   paths of the SCF solves.  OPENMX_EIGEN_REFINE=K (0: off), unset: on by
   default on GPUs whose FP64 throughput is a small fraction of their FP32
   one when the complex GEMMul8 products are on; the global root's device
   decides for all ranks.  The refinement's occupations need the chemical
   potential of all k-points, which the solver only knows once the
   eigenvalues are gathered:
   - one k-point per rank with every owner's buffers resident at once (the
     GPU turns of all owners in one group): every owner solves in FP32 (or
     starts from its previous vectors), the estimates are gathered, a
     chemical potential found on them, and the owners finish their
     refinements (two halves of one solve);
   - one k-point per rank in serialized GPU turn groups (the default of a
     GPU shared by many ranks, "staged") and the k-dense path (several
     k-points per rank): pass 1 refines with the previous step's chemical
     potential and keeps the basis on the host, pass 2 starts from it with
     the step's chemical potential and builds the vectors; from the second
     SCF step on.
   An SCF that ends on a refined step is solved once more in FP64
   (Band_DFT_Col_RefineForceFP64 around a repeated call, which recomputes the
   density matrices), and the last step scf.maxIter allows is FP64 in any
   case.  Ported from the NVIDIA version (openmx4.0_gpu, Band_DFT_Col.c). */
typedef struct
{
    int     iterations;    /* -1 until the environment is read */
    double  until;         /* printed NormRD below which the FP64 eigensolver takes over; 0 off */
    int     final_stage;   /* the FP64 eigensolver for the rest of the SCF cycle */
    int     last_refined;  /* the latest SCF eigensolve was a refined one (any k-point) */
    int     cycle_refined; /* some eigensolve of this SCF cycle was */
    int     defaulted;     /* on because of the device, not of OPENMX_EIGEN_REFINE */
    int     force_fp64;    /* Band_DFT_Col_RefineForceFP64: this call in FP64 whatever the stage */
    int     transient;     /* the first step of a cycle: its failures do not condemn the cycle */
    int     pending;       /* this rank's k-point: prepared, waiting for the chemical potential */
    int     solved;        /* this rank's k-point was refined this step */
    int     persistent;    /* the strongest persistent failure of this step on this rank */
    int     kdense;        /* this call refines the k-dense path (several k-points per rank) */
    int     scf_iter;      /* the step of the call */
    int     staged;        /* this call refines the one-k-point-per-rank path in serialized GPU turns: pass 1
                              and pass 2 as on the k-dense path, the bases on the host between them */
} BandColRefineState;

/* The two-pass solves keep, per k-point and spin, the basis of the latest
   refined solve on the host (the device buffers go with every turn): pass 2
   starts its refinement from it, and the next SCF step's pass 1 as well when
   OPENMX_EIGEN_REFINE_WARM_BAND is on.  The budget is
   OPENMX_BAND_REFINE_BASIS_MB per rank (default 1024), at most 1/8 of the
   node's available memory over its ranks. */
typedef struct
{
    dcomplex *basis;       /* n x n, host */
    double   *lam;         /* n eigenvalue estimates */
    int       valid;       /* basis and lam hold the latest refined solve */
    int       refined;     /* pass 1 of this step refined the k-point */
} BandColKRefineSlot;

typedef struct
{
    double mu;
} BandColOccupationCtx;

static BandColKRefineSlot  *BandCol_kref = NULL;
static int                  BandCol_kref_slots = 0, BandCol_kref_n = 0;
static size_t               BandCol_kref_used = 0;
static EigenRefineState     BandCol_kref_state;        /* the module's device state of the two-pass solves */
static const BandColRefineState BandCol_refine_initial = {-1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static BandColRefineState   BandCol_refine = {-1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static EigenRefineState     BandCol_refine_state;      /* the two-half solves */
static EigenRefineDevice    BandCol_refine_dev;
static EigenRefineProblem   BandCol_refine_pb;         /* the two-half solve in progress */
static BandColOccupationCtx BandCol_refine_occ = {0.0};
static dcomplex            *BandCol_refine_x = NULL;   /* the device basis, n x n */
static int                  BandCol_refine_x_n = 0;
static double              *BandCol_refine_e0 = NULL;  /* the estimates of the first half, n */
static int                  BandCol_refine_e0_n = 0;

static void BandCol_RefineDropBasis(void)
{
    if (BandCol_refine_x != NULL) wait_hipfunc(hipFree(BandCol_refine_x));
    BandCol_refine_x = NULL;
    BandCol_refine_x_n = 0;
    BandCol_refine_state.basis_valid = 0;
    BandCol_kref_state.basis_valid = 0;
}

static void BandCol_KRefReset(void)
{
    if (BandCol_kref != NULL) {
        for (int slot = 0; slot < BandCol_kref_slots; slot++) {
            free(BandCol_kref[slot].basis);
            free(BandCol_kref[slot].lam);
        }
        free(BandCol_kref);
    }
    BandCol_kref = NULL;
    BandCol_kref_slots = 0;
    BandCol_kref_n = 0;
    BandCol_kref_used = 0;
}

/* the run boundary (BandCol_GpuSolver_Destroy) */
static void BandCol_RefineRelease(void)
{
    openmx_eigen_refine_state_release(&BandCol_refine_state);
    openmx_eigen_refine_state_release(&BandCol_kref_state);
    BandCol_KRefReset();
    openmx_eigen_refine_device_release(&BandCol_refine_dev);
    BandCol_RefineDropBasis();
    free(BandCol_refine_e0);
    BandCol_refine_e0 = NULL;
    BandCol_refine_e0_n = 0;
}

/* the run boundary only (Band_DFT_Col_Release_GPU_Caches, every rank; not
   BandCol_GpuSolver_Destroy, which a device change of one rank also calls):
   the stage of the next input starts over, so a refinement turned off for
   the rest of a run does not reach the next input of -runtest/-runtestL */
static void BandCol_RefineResetStage(void)
{
    BandCol_refine = BandCol_refine_initial;
}

/* every rank, once: OPENMX_EIGEN_REFINE, or the default the global root's
   device implies, and OPENMX_EIGEN_REFINE_UNTIL, the global root's for all
   ranks alike (a stage that differed between the ranks would take them
   into different collectives) */
static void BandCol_RefineConfigure(void)
{
    BandColRefineState *st = &BandCol_refine;
    double config[3] = {0.0, 0.0, 0.0};
    int    myid;

    /* The input veto must not become a cached policy for later inputs. */
    if (!openmx_fp32_enabled()) return;
    if (0 <= st->iterations) return;
    MPI_Comm_rank(mpi_comm_level1, &myid);
    if (myid == Host_ID) {
        int ratio = 0;

        config[0] = (double)openmx_eigen_refine_configure(1, 2 <= level_stdout, &ratio);
        config[1] = (double)ratio;
        config[2] = openmx_eigen_refine_until();
    }
    MPI_Bcast(config, 3, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    st->iterations = (int)config[0];
    st->defaulted = (0.0 < config[1]);
    st->until = config[2];
}

static int BandCol_RefineIterations(void)
{
    return (openmx_fp32_enabled() && 0 < BandCol_refine.iterations) ? BandCol_refine.iterations : 0;
}

/* every rank: whether a dense owner's device has room for what a refined
   solve adds to the FP64 one (the basis, the FP32 eigensolver's workspace,
   the product blocks when they exceed the n x n scratch) for the turns that
   solve at once; base is this rank's view of whether the call may refine at
   all (OPENMX_EIGEN_REFINE_MIN_N is read per rank), folded into the common
   verdict so that every rank takes the collective and the same path; a
   refinement that would not fit leaves the step in FP64 */
static int BandCol_RefineFits(int base, int n, int maxn, int turns, int owns_dense)
{
    if (BandCol_mgpu_host_dense) return 0;
    size_t const nn = (size_t)n * (size_t)n;
    int          local_fit = base ? 1 : 0, fit = 0;

    if (BandCol_RefineIterations() <= 0) return 0;
    if (turns < 1) turns = 1;
    if (local_fit && owns_dense) {
        /* only the owners ask (the query creates the FP32 solver's handle) */
        size_t extra = sizeof(dcomplex) * nn + openmx_eigen_refine_fp32_work_bytes(1, n);
        size_t free_bytes = 0, total_bytes = 0;

        if (3 * (size_t)maxn > (size_t)n) extra += 3 * sizeof(dcomplex) * (size_t)n * (size_t)maxn;
        /* the basis the resident one-k-point path keeps on the device for
           the next step's warm start (that path's owners alike) is the one
           their next solves use, and it is taken from the free memory
           already: not counted twice */
        if (BandCol_refine_x != NULL && BandCol_refine_x_n == n) extra -= sizeof(dcomplex) * nn;

        /* the room the FP64 path leaves: its own budget is checked by
           BandCol_GpuDenseFits, with the turns' FP64 buffers allocated
           later, so the refinement's share comes on top of the reserve */
        if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != 0 ||
            free_bytes < extra * (size_t)turns + (size_t)5 * sizeof(dcomplex) * nn * (size_t)turns +
                             (size_t)512 * 1024 * 1024)
            local_fit = 0;
    }
    MPI_Allreduce(&local_fit, &fit, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
    return fit;
}

/* every rank, before the k-point loops of a call: possible says whether the
   call takes the one-k-point-per-rank path with every owner resident,
   possible_staged the same path in serialized turn groups, possible_kdense
   the k-dense path */
static void BandCol_RefineBeginStep(int SCF_iter, int possible, int possible_kdense, int possible_staged)
{
    if (BandCol_mgpu_host_dense) possible=possible_kdense=possible_staged=0;
    BandColRefineState *st = &BandCol_refine;
    int common;

    BandCol_RefineConfigure();
    if (SCF_iter == 1) {
        st->final_stage = 0;
        st->cycle_refined = 0;
        BandCol_refine_state.basis_valid = 0;   /* a new cycle may come with a new geometry */
        BandCol_KRefReset();
    }
    st->transient = (SCF_iter == 1);
    st->scf_iter = SCF_iter;
    st->pending = 0;
    st->solved = 0;
    st->persistent = 0;
    /* the occupations of the refinement are the plain Fermi function of one
       chemical potential: XANES and emptied orbitals or states keep the
       FP64 eigensolver */
    /* Configure and reset a new cycle before consulting its stage flags. */
    common = (0 < BandCol_RefineIterations() && !st->final_stage && !st->force_fp64 &&
              Cnt_switch == 0 && SCF_iter < DFTSCF_loop && xanes_calc == 0 &&
              empty_occupation_flag == 0 && empty_states_flag == 0);
    st->last_refined = (possible && common);
    /* the two-pass solves gate their first pass with the previous step's
       chemical potential, so they start at the second step */
    st->kdense = (possible_kdense && 2 <= SCF_iter && common);
    st->staged = (possible_staged && 2 <= SCF_iter && common);
    if (BandCol_kref != NULL) {
        for (int slot = 0; slot < BandCol_kref_slots; slot++) BandCol_kref[slot].refined = 0;
    }
}

/* every rank, after the refined solves of a call: the owners' solves decide
   the stage for all ranks alike */
static void BandCol_RefineEndStep(void)
{
    BandColRefineState *st = &BandCol_refine;
    int mine[2], all[2];

    mine[0] = st->solved;
    mine[1] = st->persistent;
    MPI_Allreduce(mine, all, 2, MPI_INT, MPI_MAX, mpi_comm_level1);
    st->last_refined = all[0];
    if (all[0]) st->cycle_refined = 1;
    if (1 <= all[1]) st->final_stage = 1;
    if (2 <= all[1]) {
        st->iterations = 0;
        st->defaulted = 0;
    }
}

int Band_DFT_Col_RefineLastRefined(void)
{
    return BandCol_refine.last_refined;
}

void Band_DFT_Col_RefineForceFP64(int on)
{
    BandCol_refine.force_fp64 = on;
}

/* the end of an SCF cycle: the warm-start bases go (the device one and the
   host slots), so the forces and the next cycle's first step start with the
   memory free */
void Band_DFT_Col_RefineEndCycle(void)
{
    BandCol_RefineDropBasis();
    BandCol_KRefReset();
    BandCol_refine_state.basis_valid = 0;
    BandCol_kref_state.basis_valid = 0;
}

/* DFT.c, after the mixing of an SCF step that goes on: normrd is the printed
   NormRD */
void Band_DFT_Col_RefineAfterMixing(double normrd)
{
    BandColRefineState *st = &BandCol_refine;
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

/* Fermi occupation at the chemical potential mu, as the solver's search
   counts the states (without XANES and emptied orbitals, see above) */
static double BandCol_RefineFermi(double e, int index, void *ctx)
{
    double x = (e - ((BandColOccupationCtx *)ctx)->mu) * Beta;

    (void)index;
    if (x <= -60.0) x = -60.0;
    if (60.0 <= x) x = 60.0;
    return 1.0 / (1.0 + exp(x));
}

/* The chemical potential of TZ - system_charge electrons over the lowest
   maxn estimates of every k-point and spin in EIGEN (the solver's search on
   the same weights), to the accuracy the refinement's occupations need */
static double BandCol_RefineChemP(double ***EIGEN, int T_knum, int maxn, int SpinP_switch, const int *T_k_op,
                                  double sum_weights, double TZ)
{
    double lo = -20.0, hi = 20.0, mu = 0.0;
    BandColOccupationCtx ctx;

    for (int it = 0; it < 1000; it++) {
        double count = 0.0, dnum;

        mu = 0.5 * (lo + hi);
        ctx.mu = mu;
        for (int kloop = 0; kloop < T_knum; kloop++) {
            for (int spin = 0; spin <= SpinP_switch; spin++) {
                for (int l = 1; l <= maxn; l++) {
                    count += BandCol_RefineFermi(EIGEN[spin][kloop][l], l, &ctx) * (double)T_k_op[kloop];
                }
            }
        }
        count = ((SpinP_switch == 0) ? 2.0 * count : count) / sum_weights;
        dnum = TZ - count - system_charge;
        if (0.0 <= dnum) lo = mu;
        else hi = mu;
        if (fabs(dnum) < 1.0e-12) break;
    }
    return mu;
}

static hipError_t BandCol_TryDeviceMalloc(void **ptr, size_t bytes)
{
    hipError_t status = hipMalloc(ptr, (bytes == 0) ? 1 : bytes);

    if (status != hipSuccess) {
        *ptr = NULL;
        (void)hipGetLastError();
    }
    return status;
}

static void *BandCol_MallocArray(size_t count, size_t elem_size, const char *label)
{
    void *ptr = malloc((count * elem_size == 0) ? 1 : count * elem_size);

    if (ptr == NULL) {
        char msg[256];

        snprintf(msg, sizeof(msg), "Failed to allocate %s (%zu bytes) in Band_DFT_Col.c.", label, count * elem_size);
        BandCol_AbortWithMessage(msg);
    }
    return ptr;
}

/* The forward transform of a k-point's Hamiltonian into d_H: H (uploaded
   from H_in when given, else already on the device) to X^H H X */
static void BandCol_GpuSolver_ForwardTransformDevice(int n, const dcomplex *H_in)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;

    if (!(ctx->transformed_s_valid && ctx->transformed_s_dim == n)) {
        BandCol_AbortWithMessage("Transformed overlap is not ready in BandCol_GpuSolver_ForwardTransformDevice.");
    }
    if (H_in != NULL) {
        wait_hipfunc(hipMemcpy(ctx->d_H, H_in, sizeof(dcomplex) * (size_t)n * (size_t)n, hipMemcpyHostToDevice));
    }
    BandCol_GpuSolver_ZgemmDevice(HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, ctx->d_H, ctx->d_S, ctx->d_tmp);
    BandCol_GpuSolver_ZgemmDevice(HIPBLAS_OP_C, HIPBLAS_OP_N, n, n, n, ctx->d_S, ctx->d_tmp, ctx->d_H);
}

/* OPENMX_BAND_GEMMUL8_BACKTRANSFORM (default 1): the back transform of the
   eigenvectors through GEMMul8, as the NVIDIA version does, with the solve's
   workspace cap and reserve (openmx_gemmul8ZgemmSolve: the k-point turns
   share the device); 0 the native FP64 hipBLAS ZGEMM.  On gfx1200, whose
   FP64 rate is 1/32 of FP32, the maxn x n x n native product can dominate a
   k-point's solve once the eigensolver is refined. */
static int BandCol_GemmulBackTransformEnabled(void)
{
    static int enabled = -1;

    if (enabled < 0) {
        const char *value = getenv("OPENMX_BAND_GEMMUL8_BACKTRANSFORM");
        enabled = (value == NULL || atoi(value) != 0);
    }
    return enabled;
}

/* C^T = (S V)^T of the first maxn vectors V in d_H: state-major with
   stride maxn into evec_t */
static void BandCol_GpuSolver_BackTransformGemm(int n, int maxn, dcomplex *evec_t)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    hipDoubleComplex     alpha = make_hipDoubleComplex(1.0, 0.0);
    hipDoubleComplex     beta = make_hipDoubleComplex(0.0, 0.0);

    if (BandCol_GemmulBackTransformEnabled()) {
        wait_hipfunc(openmx_gemmul8ZgemmSolve(ctx->hipblas, HIPBLAS_OP_T, HIPBLAS_OP_T, maxn, n, n, &alpha,
                                              (const hipDoubleComplex *)ctx->d_H, n,
                                              (const hipDoubleComplex *)ctx->d_S, n, &beta,
                                              (hipDoubleComplex *)evec_t, maxn));
    }
    else {
        wait_hipfunc(hipblasZgemm(ctx->hipblas, HIPBLAS_OP_T, HIPBLAS_OP_T, maxn, n, n, &alpha,
                                  (hipDoubleComplex *)ctx->d_H, n, (hipDoubleComplex *)ctx->d_S, n, &beta,
                                  (hipDoubleComplex *)evec_t, maxn));
    }
}

/* The back transform of the first maxn vectors in d_H: C^T = (S V)^T, state-
   major with stride maxn, into the eigenvector buffer */
static dcomplex *BandCol_GpuSolver_BackTransformDevice(int n, int maxn)
{
    dcomplex *evec_t = BandCol_GpuSolver_EnsureEigenvectorCapacity(n, maxn);

    BandCol_GpuSolver_BackTransformGemm(n, maxn, evec_t);
    return evec_t;
}

/* the module's view of the solver context */
static void BandCol_RefineBindDevice(EigenRefineDevice *dev)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;

    memset(dev, 0, sizeof(*dev));
    dev->hipblas = ctx->hipblas;
    dev->d_work = &ctx->d_work;
    dev->d_work_bytes = &ctx->d_work_bytes;
    dev->d_info = ctx->d_info;
    dev->try_malloc = BandCol_TryDeviceMalloc;
    dev->host_malloc = BandCol_MallocArray;
    dev->shared = 1;   /* the k-point owners' turns share the device */
}

/* the device basis buffer of the refined solves (n x n complex); 0 when the
   device has no room */
static int BandCol_RefineEnsureDeviceBasis(int n)
{
    size_t const nn = (size_t)n * (size_t)n;

    if (BandCol_refine_x != NULL && BandCol_refine_x_n != n) BandCol_RefineDropBasis();
    if (BandCol_refine_x == NULL) {
        if (BandCol_TryDeviceMalloc((void **)&BandCol_refine_x, sizeof(dcomplex) * nn) != hipSuccess) {
            if (2 <= level_stdout) {
                printf("<Band_DFT_Col> no room for the %.1f MiB basis of the refined eigensolver; FP64 eigensolver\n",
                       (double)(sizeof(dcomplex) * nn) / (1024.0 * 1024.0));
                fflush(stdout);
            }
            return 0;
        }
        BandCol_refine_x_n = n;
    }
    return 1;
}

static void BandCol_KRefEnsure(int slots, int n)
{
    if (BandCol_kref != NULL && (BandCol_kref_slots != slots || BandCol_kref_n != n)) BandCol_KRefReset();
    if (BandCol_kref == NULL) {
        BandCol_kref = (BandColKRefineSlot *)BandCol_MallocArray((size_t)slots, sizeof(BandColKRefineSlot),
                                                                 "refined basis slots");
        memset(BandCol_kref, 0, sizeof(BandColKRefineSlot) * (size_t)slots);
        BandCol_kref_slots = slots;
        BandCol_kref_n = n;
    }
}

/* the budget of the slots of one rank */
static size_t BandCol_RefineBasisLimit(void)
{
    static size_t limit = 0;
    static int    checked = 0;

    if (!checked) {
        const char        *env = getenv("OPENMX_BAND_REFINE_BASIS_MB");
        double             cap = (env == NULL) ? 1024.0 : atof(env);
        unsigned long long available_kib = 0;
        int                ranks = openmx_gpu_local_size_noncollective();
        FILE              *file;
        char               line[256];

        checked = 1;
        if (!(cap > 0.0) || cap >= (double)SIZE_MAX / 1048576.0) return 0;
        limit = (size_t)(cap * 1048576.0);
        file = fopen("/proc/meminfo", "r");
        if (file != NULL) {
            while (fgets(line, sizeof(line), file) != NULL) {
                if (sscanf(line, "MemAvailable: %llu kB", &available_kib) == 1) break;
            }
            fclose(file);
            if (ranks < 1) ranks = 1;
            if (0 < available_kib && available_kib / (unsigned)ranks * (1024ULL / 8ULL) < limit)
                limit = (size_t)(available_kib / (unsigned)ranks * (1024ULL / 8ULL));
        }
    }
    return limit;
}

/* the slot's host buffers, within the budget; 0 when there is no room */
static int BandCol_KRefSlotAllocate(BandColKRefineSlot *slot, int n)
{
    size_t const bytes = sizeof(dcomplex) * (size_t)n * (size_t)n + sizeof(double) * (size_t)n;
    size_t const limit = BandCol_RefineBasisLimit();

    if (slot->basis != NULL) return 1;
    if (limit < BandCol_kref_used || bytes > limit - BandCol_kref_used) return 0;
    slot->basis = (dcomplex *)malloc(sizeof(dcomplex) * (size_t)n * (size_t)n);
    slot->lam = (double *)malloc(sizeof(double) * (size_t)n);
    if (slot->basis == NULL || slot->lam == NULL) {
        free(slot->basis);
        free(slot->lam);
        slot->basis = NULL;
        slot->lam = NULL;
        return 0;
    }
    BandCol_kref_used += bytes;
    return 1;
}

static void BandCol_KRefSlotFree(BandColKRefineSlot *slot, int n)
{
    if (slot->basis != NULL) {
        BandCol_kref_used -= sizeof(dcomplex) * (size_t)n * (size_t)n + sizeof(double) * (size_t)n;
    }
    free(slot->basis);
    free(slot->lam);
    slot->basis = NULL;
    slot->lam = NULL;
    slot->valid = 0;
}

/* A refined solve of a two-pass path on the transformed Hamiltonian in d_H
   (forward transform done by the caller): warm from the slot's basis when
   allowed, with the occupations at mu, then ko[1..maxn] and the slot updated
   (the basis kept on the host when keep is set and there is room).  Returns 1
   when d_H's first maxn columns hold the refined vectors, 0 when the FP64
   eigensolver has to take over (d_H still the Hermitian matrix). */
static int BandCol_KDenseRefinedSolve(int n, int maxn, double *ko, BandColKRefineSlot *slot, int warm_allowed,
                                      double mu, int keep)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    BandColRefineState  *st = &BandCol_refine;
    EigenRefineDevice   *dev = &BandCol_refine_dev;
    EigenRefineState    *ks = &BandCol_kref_state;
    EigenRefineProblem   pb;
    EigenRefineReport    rep;
    size_t const         nn = (size_t)n * (size_t)n;
    double              *e0, *f0;
    int                  rc;

    BandCol_RefineBindDevice(dev);
    memset(&pb, 0, sizeof(pb));
    memset(&rep, 0, sizeof(rep));
    pb.cplx = 1;
    pb.n = n;
    pb.maxn = maxn;
    pb.iterations = BandCol_RefineIterations();
    pb.a = ctx->d_H;
    pb.w = ctx->d_W;
    pb.fp32 = ctx->d_tmp;
    pb.region = sizeof(dcomplex) * nn;
    pb.x = BandCol_refine_x;
    pb.transient = 0;
    pb.warm = (warm_allowed && slot->valid && slot->basis != NULL && ks->lam != NULL && ks->capacity >= n);
    pb.defaulted = st->defaulted;
    pb.occupation = BandCol_RefineFermi;
    pb.occupation_ctx = &BandCol_refine_occ;
    if (pb.warm) {
        wait_hipfunc(hipMemcpy(BandCol_refine_x, slot->basis, sizeof(dcomplex) * nn, hipMemcpyHostToDevice));
        wait_hipfunc(hipMemcpy(ks->lam, slot->lam, sizeof(double) * (size_t)n, hipMemcpyHostToDevice));
        ks->basis_valid = 1;
        ks->basis_n = n;
    }
    else {
        ks->basis_valid = 0;
    }
    e0 = (double *)BandCol_MallocArray((size_t)n, sizeof(double), "refinement eigenvalues");
    f0 = (double *)BandCol_MallocArray((size_t)n, sizeof(double), "refinement occupations");
    rc = openmx_eigen_refine_prepare(ks, dev, &pb, e0, &rep);
    if (rc < 0) {
        pb.warm = 0;
        ks->basis_valid = 0;
        rc = openmx_eigen_refine_prepare(ks, dev, &pb, e0, &rep);
    }
    if (rc == 1) {
        BandCol_refine_occ.mu = mu;
        for (int i = 0; i < n; i++) f0[i] = BandCol_RefineFermi(e0[i], i + 1, &BandCol_refine_occ);
        rc = openmx_eigen_refine_finish(ks, dev, &pb, f0, &rep);
    }
    if (rc != 1 && st->persistent < rep.persistent) st->persistent = rep.persistent;
    free(e0);
    free(f0);
    if (rc != 1) {
        ks->basis_valid = 0;
        slot->valid = 0;
        return 0;
    }
    wait_hipfunc(hipMemcpy(ko + 1, ctx->d_W, sizeof(double) * (size_t)maxn, hipMemcpyDeviceToHost));
    if (keep && BandCol_KRefSlotAllocate(slot, n)) {
        wait_hipfunc(hipMemcpy(slot->basis, BandCol_refine_x, sizeof(dcomplex) * nn, hipMemcpyDeviceToHost));
        wait_hipfunc(hipMemcpy(slot->lam, ks->lam, sizeof(double) * (size_t)n, hipMemcpyDeviceToHost));
        slot->valid = 1;
    }
    else {
        BandCol_KRefSlotFree(slot, n);
    }
    if (3 <= level_stdout) {
        printf("<Band_DFT_Col> k-point slot %d: %s, %d refinement steps of %d columns: max |r| %.1e, max |s| %.1e, "
               "delta %.1e; Rayleigh-Ritz in %d partially occupied clusters (largest %d)\n",
               (int)(slot - BandCol_kref), pb.warm ? "warm start" : "FP32 eigensolver", pb.iterations, rep.columns,
               rep.max_r, rep.max_s, rep.delta, rep.rr_clusters, rep.rr_largest);
        fflush(stdout);
    }
    st->solved = 1;
    return 1;
}

/* Pass 1 of a two-pass path for one k-point, its transformed overlap ready:
   the forward transform, the refined solve gated with the previous step's
   chemical potential (the eigenvalues are second order in the vectors'
   error, so this gating is enough for them), ko[1..maxn] for EIGEN and the
   basis kept for pass 2.  The FP64 eigenvalues follow when the refinement
   fails.  Returns 1 when the k-point was handled (refined or not; no
   eigenvectors are built), 0 when the caller is to solve it as usual
   (nothing touched). */
static int BandCol_KDenseRefinedPass1(int n, int maxn, const dcomplex *H_in, double *ko, int slot_index)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    BandColKRefineSlot  *slot = &BandCol_kref[slot_index];

    if (!BandCol_RefineEnsureDeviceBasis(n)) return 0;
    BandCol_GpuSolver_ForwardTransformDevice(n, H_in);
    slot->refined = BandCol_KDenseRefinedSolve(n, maxn, ko, slot,
                                               openmx_eigen_refine_warm_band_step(BandCol_refine.scf_iter), ChemP, 1);
    if (!slot->refined) {
        BandCol_GpuSolver_Eigen(ctx->d_H, n, maxn, ko + 1);
    }
    return 1;
}

/* Pass 2 for a k-point refined in pass 1: the forward transform again, the
   refinement from pass 1's basis with the occupations at the step's chemical
   potential, the back transform.  Returns the device eigenvectors (state-
   major, stride maxn), of an FP64 solve when the refinement failed. */
static dcomplex *BandCol_KDenseRefinedPass2(int n, int maxn, const dcomplex *H_in, double *ko, int slot_index,
                                            double mu)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    BandColKRefineSlot  *slot = &BandCol_kref[slot_index];
    int                  ok = 0;

    BandCol_GpuSolver_ForwardTransformDevice(n, H_in);
    if (BandCol_RefineEnsureDeviceBasis(n)) {
        ok = BandCol_KDenseRefinedSolve(n, maxn, ko, slot, 1, mu, openmx_eigen_refine_warm_band_enabled());
    }
    if (!ok) {
        BandCol_GpuSolver_Eigen(ctx->d_H, n, maxn, ko + 1);
    }
    return BandCol_GpuSolver_BackTransformDevice(n, maxn);
}

/* The first half of a refined solve of this rank's k-point (every owner
   resident): the forward transform into d_H, then the FP32 solve (or the
   warm start) whose estimates go to ko[1..maxn].  Returns NULL with the
   solve pending (BandCol_RefinedSecondHalf finishes it once the chemical
   potential is known), or, when the refinement is not possible, the device
   eigenvectors of an FP64 solve done instead (stride maxn). */
static dcomplex *BandCol_RefinedFirstHalf(int n, int maxn, const dcomplex *H_in, double *ko)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    BandColRefineState  *st = &BandCol_refine;
    EigenRefineDevice   *dev = &BandCol_refine_dev;
    EigenRefineProblem  *pb = &BandCol_refine_pb;
    EigenRefineReport    rep;
    size_t const         nn = (size_t)n * (size_t)n;
    int                  rc = 1;

    BandCol_GpuSolver_ForwardTransformDevice(n, H_in);
    st->pending = 0;
    if (BandCol_refine_x != NULL && BandCol_refine_x_n != n) BandCol_RefineDropBasis();
    if (BandCol_refine_x == NULL) {
        if (BandCol_TryDeviceMalloc((void **)&BandCol_refine_x, sizeof(dcomplex) * nn) != hipSuccess) {
            if (2 <= level_stdout) {
                printf("<Band_DFT_Col> no room for the %.1f MiB basis of the refined eigensolver; FP64 eigensolver%s\n",
                       (double)(sizeof(dcomplex) * nn) / (1024.0 * 1024.0),
                       st->transient ? "" : " for the rest of the SCF cycle");
                fflush(stdout);
            }
            if (!st->transient && st->persistent < 1) st->persistent = 1;
            rc = 0;
        }
        else {
            BandCol_refine_x_n = n;
        }
    }
    if (BandCol_refine_e0_n != n) {
        free(BandCol_refine_e0);
        BandCol_refine_e0 = (double *)BandCol_MallocArray((size_t)n, sizeof(double), "refinement eigenvalues");
        BandCol_refine_e0_n = n;
    }
    if (rc) {
        BandCol_RefineBindDevice(dev);
        memset(pb, 0, sizeof(*pb));
        memset(&rep, 0, sizeof(rep));
        pb->cplx = 1;
        pb->n = n;
        pb->maxn = maxn;
        pb->iterations = BandCol_RefineIterations();
        pb->a = ctx->d_H;
        pb->w = ctx->d_W;
        pb->fp32 = ctx->d_tmp;
        pb->region = sizeof(dcomplex) * nn;
        pb->x = BandCol_refine_x;
        pb->transient = st->transient;
        pb->warm = (openmx_eigen_refine_warm_band_step(st->scf_iter) && BandCol_refine_state.basis_valid &&
                    BandCol_refine_state.basis_n == n && !st->transient);
        pb->defaulted = st->defaulted;
        pb->occupation = BandCol_RefineFermi;
        pb->occupation_ctx = &BandCol_refine_occ;
        if (3 <= level_stdout) {
            printf("<Band_DFT_Col> this rank's k-point: %s, %d refinement steps (OPENMX_EIGEN_REFINE)\n",
                   pb->warm ? "warm start from the previous vectors" : "FP32 eigensolver", pb->iterations);
            fflush(stdout);
        }
        rc = openmx_eigen_refine_prepare(&BandCol_refine_state, dev, pb, BandCol_refine_e0, &rep);
        if (rc < 0) {
            pb->warm = 0;
            rc = openmx_eigen_refine_prepare(&BandCol_refine_state, dev, pb, BandCol_refine_e0, &rep);
        }
        if (rc != 1 && st->persistent < rep.persistent) st->persistent = rep.persistent;
    }
    if (rc == 1) {
        for (int l = 1; l <= maxn; l++) ko[l] = BandCol_refine_e0[l - 1];
        st->pending = 1;
        return NULL;
    }

    /* FP64 after all: d_H still holds the Hermitian matrix */
    BandCol_refine_state.basis_valid = 0;
    BandCol_GpuSolver_Eigen(ctx->d_H, n, maxn, ko + 1);
    return BandCol_GpuSolver_BackTransformDevice(n, maxn);
}

/* The second half: the refinement with the occupations at mu, the refined
   eigenvalues to ko[1..maxn] and the back transform.  Returns the device
   eigenvectors (stride maxn), of an FP64 solve when the refinement failed. */
static dcomplex *BandCol_RefinedSecondHalf(int n, int maxn, double *ko, double mu)
{
    BandColGpuSolverCtx *ctx = &BandCol_gpusolver_ctx;
    BandColRefineState  *st = &BandCol_refine;
    EigenRefineProblem  *pb = &BandCol_refine_pb;
    EigenRefineReport    rep;
    double              *f0 = (double *)BandCol_MallocArray((size_t)n, sizeof(double), "refinement occupations");
    int                  rc;

    memset(&rep, 0, sizeof(rep));
    BandCol_refine_occ.mu = mu;
    for (int i = 0; i < n; i++) f0[i] = BandCol_RefineFermi(BandCol_refine_e0[i], i + 1, &BandCol_refine_occ);
    rc = openmx_eigen_refine_finish(&BandCol_refine_state, &BandCol_refine_dev, pb, f0, &rep);
    free(f0);
    st->pending = 0;
    if (rc != 1 && st->persistent < rep.persistent) st->persistent = rep.persistent;
    if (rc == 1) {
        wait_hipfunc(hipMemcpy(ko + 1, ctx->d_W, sizeof(double) * (size_t)maxn, hipMemcpyDeviceToHost));
        if (3 <= level_stdout) {
            printf("<Band_DFT_Col> refinement step %d of %d columns: max |r| %.1e, max |s| %.1e, delta %.1e; "
                   "Rayleigh-Ritz in %d partially occupied clusters (largest %d, chained within %.1e)\n",
                   pb->iterations, rep.columns, rep.max_r, rep.max_s, rep.delta, rep.rr_clusters, rep.rr_largest,
                   rep.chain);
            fflush(stdout);
        }
        st->solved = 1;
    }
    else {
        /* d_H still holds the Hermitian matrix */
        BandCol_refine_state.basis_valid = 0;
        BandCol_GpuSolver_Eigen(ctx->d_H, n, maxn, ko + 1);
    }
    return BandCol_GpuSolver_BackTransformDevice(n, maxn);
}

/* ------------------------------------------------------------------ */

/* After pass 1 of a staged refined solve: the device goes to the next turn
   (unless the buffers persist), the basis lives in the host slot */
static void BandCol_AfterStagedTurn(void)
{
    if (BandCol_GemmWorkspaceTurnRelease()) {
        openmx_gemmul8ReleaseWorkspaces();
    }
    else {
        openmx_gemmul8TrimWorkspaces();
    }
    if (!BandCol_GpuPersistentDecide()) {
        BandCol_ConstructCache_Reset();
        BandCol_GpuSolver_ReleaseDeviceMemory();
    }
    BandCol_RefineDropBasis();
}

/* After a dense solve of the one-k-point-per-rank path that built the
   eigenvectors: the GEMMul8 workspace goes, and the eigenvectors stay on the
   device or go to the host with the device buffers released, as after the
   FP64 solve */
static void BandCol_AfterDenseSolve(dcomplex *evec_device, int n)
{
    if (BandCol_mgpu_host_dense) return;
    if (BandCol_GemmWorkspaceTurnRelease()) {
        openmx_gemmul8ReleaseWorkspaces();
    }
    else {
        openmx_gemmul8TrimWorkspaces();
    }
    if (BandCol_GpuPersistentDecide()) {
        (void)evec_device;
    } else {
        BandCol_GpuSolver_SaveDeviceEigenvectors(evec_device, n, BandCol_GpuSolver_EigenvectorStride());
        BandCol_ConstructCache_Reset();
        BandCol_GpuSolver_ReleaseDeviceMemory();
    }
}

static dcomplex *BandCol_GpuSolver_SolveHamiltonianImpl(int n, int maxn, const dcomplex *H_in, double *ko, dcomplex *C_out,
                                                       int build_eigenvectors)
{
    if (BandCol_mgpu_host_dense) {
        BandCol_HostEnsure(n);
        if (!openmx_mgpu_host_zgemm(HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,BandCol_host_h,n,BandCol_host_s,n,BandCol_host_tmp,n) ||
            !openmx_mgpu_host_zgemm(HIPBLAS_OP_C,HIPBLAS_OP_N,n,n,n,BandCol_host_s,n,BandCol_host_tmp,n,BandCol_host_h,n) ||
            !openmx_mgpu_host_solve(n,maxn,1,BandCol_host_h,n,ko+1)) BandCol_AbortWithMessage(openmx_mgpu_host_error());
        if (!build_eigenvectors) return NULL;
        memset(BandCol_host_tmp,0,(size_t)n*n*sizeof(dcomplex));
        if (!openmx_mgpu_host_zgemm(HIPBLAS_OP_T,HIPBLAS_OP_T,maxn,n,n,BandCol_host_h,n,BandCol_host_s,n,BandCol_host_tmp,maxn))
            BandCol_AbortWithMessage(openmx_mgpu_host_error());
        if (C_out) memcpy(C_out,BandCol_host_tmp,(size_t)n*maxn*sizeof(dcomplex));
        BandCol_host_stride=maxn;
        return BandCol_host_tmp;
    }

    BandColGpuSolverCtx * ctx = &BandCol_gpusolver_ctx;
    size_t               matrix_bytes;
    dcomplex            *evec_t;
    const int             tdbg = BandCol_SolveTimingEnabled();
    double                tt0 = 0.0, tt1 = 0.0, tt2 = 0.0, tt3 = 0.0, tt4 = 0.0, tt5 = 0.0;

    if (!(ctx->transformed_s_valid && ctx->transformed_s_dim == n)) {
        BandCol_AbortWithMessage("Transformed overlap is not ready in BandCol_GpuSolver_SolveHamiltonian.");
    }

    matrix_bytes = sizeof(dcomplex) * (size_t)n * (size_t)n;

    if (tdbg) dtime(&tt0);

    if (H_in != NULL) {
        wait_hipfunc(hipMemcpy(ctx->d_H, H_in, matrix_bytes, hipMemcpyHostToDevice));
    }

    if (tdbg) { wait_hipfunc(hipDeviceSynchronize()); dtime(&tt1); }

    BandCol_GpuSolver_ZgemmDevice(HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, ctx->d_H, ctx->d_S, ctx->d_tmp);

    if (tdbg) { wait_hipfunc(hipDeviceSynchronize()); dtime(&tt2); }

    BandCol_GpuSolver_ZgemmDevice(HIPBLAS_OP_C, HIPBLAS_OP_N, n, n, n, ctx->d_S, ctx->d_tmp, ctx->d_H);

    if (tdbg) { wait_hipfunc(hipDeviceSynchronize()); dtime(&tt3); }

    BandCol_GpuSolver_Eigen(ctx->d_H, n, maxn, ko + 1);

    if (tdbg) { wait_hipfunc(hipDeviceSynchronize()); dtime(&tt4); }

    if (!build_eigenvectors) {
        if (tdbg) {
            printf("SOLVE_TIMING h2d=%.3f zgemm1=%.3f zgemm2=%.3f zheevd=%.3f (novec)\n",
                   tt1 - tt0, tt2 - tt1, tt3 - tt2, tt4 - tt3);
            fflush(stdout);
        }
        return NULL;
    }

    evec_t = BandCol_GpuSolver_EnsureEigenvectorCapacity(n, maxn);
    BandCol_GpuSolver_BackTransformGemm(n, maxn, evec_t);

    if (C_out != NULL) {
        wait_hipfunc(hipMemcpy(C_out, evec_t, sizeof(dcomplex) * (size_t)n * (size_t)maxn,
                                 hipMemcpyDeviceToHost));
    }

    if (tdbg) {
        wait_hipfunc(hipDeviceSynchronize());
        dtime(&tt5);
        printf("SOLVE_TIMING h2d=%.3f zgemm1=%.3f zgemm2=%.3f zheevd=%.3f back=%.3f\n",
               tt1 - tt0, tt2 - tt1, tt3 - tt2, tt4 - tt3, tt5 - tt4);
        fflush(stdout);
    }

    return evec_t;
}

static void BandCol_GpuSolver_SolveHamiltonian(int n, int maxn, const dcomplex *H_in, double *ko, dcomplex *C_out)
{
    BandCol_GpuSolver_SolveHamiltonianImpl(n, maxn, H_in, ko, C_out, C_out != NULL);
}

static dcomplex *BandCol_GpuSolver_SolveHamiltonianDeviceOnly(int n, int maxn, const dcomplex *H_in, double *ko)
{
    return BandCol_GpuSolver_SolveHamiltonianImpl(n, maxn, H_in, ko, NULL, 1);
}

static dcomplex *BandCol_GpuSolver_SolveHamiltonianDeviceInput(int n, int maxn, double *ko)
{
    return BandCol_GpuSolver_SolveHamiltonianImpl(n, maxn, NULL, ko, NULL, 1);
}

static dcomplex *BandCol_GpuSolver_DeviceEigenvectors(void)
{
    if (BandCol_mgpu_host_dense) return BandCol_host_tmp;
    return BandCol_gpusolver_ctx.d_evec;
}

static int BandCol_LastConstructOnDevice(void)
{
    return BandCol_last_construct_on_device;
}

static void BandCol_ConstructDenseCsHs_HIP(int need_s, int n, double k1, double k2, double k3, const double *S1,
                                           const double *H1)
{
    BandColConstructCache *cache = &BandCol_construct_cache;
    BandColGpuSolverCtx *   ctx = &BandCol_gpusolver_ctx;
    BandColConstructEntry *entries = cache->dense_entries_dev;
    double *phase_r = cache->dense_phase_r;
    double *phase_i = cache->dense_phase_i;
    int count = cache->dense_count;
    int phase_count = cache->dense_phase_count;
    int h_count = cache->h_count;
    long long matrix_count = (long long)n * (long long)n;
    dcomplex *d_H;
    dcomplex *d_S;

    if (entries == NULL && 0 < count) {
        BandCol_AbortWithMessage("The dense construct cache is not device-resident in Band_DFT_Col.c.");
    }

    BandCol_GpuSolver_EnsureMatrixCapacity(n);

    d_H = ctx->d_H;
    d_S = ctx->d_S;

    for (int p = 0; p < phase_count; ++p) {
        double kRn = k1 * (double)cache->dense_phase_l1[p] + k2 * (double)cache->dense_phase_l2[p] +
                     k3 * (double)cache->dense_phase_l3[p];
        phase_i[p] = sin(2.0 * PI * kRn);
        phase_r[p] = cos(2.0 * PI * kRn);
    }

    if ((size_t)LLONG_MAX < (size_t)matrix_count) {
        BandCol_AbortWithMessage("Dense matrix size exceeds HIP loop limit in Band_DFT_Col.c.");
    }

    if (BandCol_BuildDenseCsHs_HIP(need_s, count, h_count, phase_count, n, entries, phase_r, phase_i,
                                   H1, S1, d_H, d_S) != 0) {
        /* Likely out of GPU memory: drop this rank's cached GEMMul8 workspace and retry. */
        openmx_gemmul8ReleaseWorkspaces();
        if (BandCol_BuildDenseCsHs_HIP(need_s, count, h_count, phase_count, n, entries, phase_r, phase_i,
                                       H1, S1, d_H, d_S) != 0) {
            BandCol_AbortWithMessage("Band_DFT_Col HIP dense matrix generation failed.");
        }
    }

    BandCol_last_construct_on_device = 1;
}

double Band_DFT_Col(int SCF_iter, int knum_i, int knum_j, int knum_k, int SpinP_switch, int ParDM_flag, double ***** nh,
                    double ***** ImNL, double **** CntOLP, double ***** CDM, double ***** EDM, double Eele0[2],
                    double Eele1[2], int * MP, int * order_GA, double * ko, double * koS, double *** EIGEN, double * H1,
                    double * S1, double * CDM1, double * EDM1, dcomplex ** EVec1, dcomplex * Ss, dcomplex * Cs,
                    dcomplex * Hs, int *** k_op, int * T_k_op, int ** T_k_ID, double * T_KGrids1, double * T_KGrids2,
                    double * T_KGrids3, int myworld1, int * NPROCS_ID1, int * Comm_World1, int * NPROCS_WD1,
                    int * Comm_World_StartID1, MPI_Comm * MPI_CommWD1, int myworld2, int * NPROCS_ID2, int * NPROCS_WD2,
                    int * Comm_World2, int * Comm_World_StartID2, MPI_Comm * MPI_CommWD2)
{
    openmx_mgpu_scope_state input_scope=openmx_mgpu_band_scope_enter(0,0,0);
    static int firsttime = 1;
    int        i, j, k, l, m, n, p, wan, MaxN, i0, ks;
    int        i1, i1s, j1, ia, jb, lmax, kmin, kmax, po, po1, spin, s1, e1;
    int        num2, RnB, l1, l2, l3, loop_num, ns, ne;
    int        ct_AN, h_AN, wanA, tnoA, wanB, tnoB;
    int        MA_AN, GA_AN, Anum, num_kloop0;
    int        T_knum, S_knum, E_knum, kloop, kloop0;
    double     av_num, lumos;
    double     time0;
    int        LB_AN, GB_AN, Bnum;
    double     k1, k2, k3, Fkw;
    double     sum, sumi, sum_weights;
    double     Num_State;
    double     My_Num_State;
    double     FermiF, tmp1;
    double     tmp, eig, kw, EV_cut0;
    double     x, Dnum, Dnum2, AcP, ChemP_MAX, ChemP_MIN;
    int *      is1, *ie1;
    int *      is2, *ie2;
    int *      My_NZeros;
    int *      SP_NZeros;
    int *      SP_Atoms;

    int      all_knum;
    dcomplex Ctmp1, Ctmp2;
    int      ii, ij, ik;
    int      BM, BN, BK;
    double   u2, v2, uv, vu;
    double   d1, d2, d3, d4, ReA, ImA;
    double   My_Eele1[2];
    double   TZ, dum, sumE, kRn, si, co;
    double   Resum, ResumE, Redum, Redum2;
    double   Imsum, ImsumE, Imdum, Imdum2;
    double   TStime, TEtime, SiloopTime, EiloopTime;
    double   Stime = 0.0, Etime = 0.0, Stime0, Etime0;
    double   Stime1, Etime1;
    double   FermiEps = 1.0e-13;
    double   x_cut    = 60.0;
    double   My_Eele0[2];

    char   file_EV[YOUSO10];
    FILE * fp_EV;
    char   buf[fp_bsize]; /* setvbuf */
    int    AN, Rn, size_H1;
    int    parallel_mode;
    int    numprocs0, myid0;
    int    ID, ID0, ID1;
    int    numprocs1, myid1;
    int    numprocs2, myid2;
    int    Num_Comm_World1;
    int    Num_Comm_World2;

    int         tag = 999, IDS, IDR;
    MPI_Status  stat;
    MPI_Request request;

    double time1, time2 = 0.0, time3 = 0.0;
    double time4 = 0.0, time5 = 0.0, time6;
    double time7, time8, time9;
    double time10, time11, time12;
    double time81, time82, time83;
    double time84, time85;
    double time51, time11A = 0.0, time11B = 0.0;

    MPI_Comm mpi_comm_rows = MPI_COMM_NULL, mpi_comm_cols = MPI_COMM_NULL;
    int      mpi_comm_rows_int = 0, mpi_comm_cols_int = 0;
    int      mpi_eigen_comms_ready = 0;
    int      info, ig, jg, il, jl, prow, pcol, brow, bcol;
    int      ZERO = 0, ONE = 1;
    dcomplex alpha = {1.0, 0.0};
    dcomplex beta  = {0.0, 0.0};

    int    LOCr, LOCc, node, irow, icol;
    double mC_spin_i1, C_spin_i1;

    int     Max_Num_Snd_EV, Max_Num_Rcv_EV;
    int *   Num_Snd_EV, *Num_Rcv_EV;
    int *   index_Snd_i, *index_Snd_j, *index_Rcv_i, *index_Rcv_j;
    double *EVec_Snd, *EVec_Rcv;
    int     max_tno;
    int     owns_dense_k_rank;
    int     owns_global_dense_rank;
    int     transformed_s_ready;
    int     use_gpusolver_dense;
    int     use_setham_packed_cache = 0;
    int *   gpu_k_order = NULL;
    dcomplex **k_evec_cache = NULL;
    size_t k_cache_bytes = 0, k_cache_used = 0, k_cache_limit = 0;
    int k_cache_hits = 0, k_overlap_hits = 0;
    int *   setham_order_GA = NULL;
    double *setham_S1 = NULL;
    double *setham_H1 = NULL;

    /* for time */
    dtime(&TStime);

    if (scf_eigen_lib_flag == GPUSOLVER && SCF_iter == 1) {
        BandCol_KOverlapReset();
        BandCol_ConstructCache_Reset();
    }

    time1  = 0.0;
    time2  = 0.0;
    time3  = 0.0;
    time4  = 0.0;
    time5  = 0.0;
    time6  = 0.0;
    time7  = 0.0;
    time8  = 0.0;
    time9  = 0.0;
    time10 = 0.0;
    time11 = 0.0;
    time12 = 0.0;
    time81 = 0.0;
    time82 = 0.0;
    time83 = 0.0;
    time84 = 0.0;
    time85 = 0.0;
    time51 = 0.0;

    double        part1      = 0.0;
    double        part2_1    = 0.0;
    double        part2_2    = 0.0;
    double        part2_3    = 0.0;
    double        part2_4    = 0.0;
    double        part2_5    = 0.0;
    double        part3      = 0.0;
    double        part4      = 0.0;
    double        partmul    = 0.0;
    static double part1sum   = 0.0;
    static double part2_1sum = 0.0;
    static double part2_2sum = 0.0;
    static double part2_3sum = 0.0;
    static double part2_4sum = 0.0;
    static double part2_5sum = 0.0;
    static double part3sum   = 0.0;
    static double part4sum   = 0.0;
    static double partmulsum = 0.0;
    double        starttime, endtime;

    /* MPI */
    MPI_Comm_size(mpi_comm_level1, &numprocs0);
    MPI_Comm_rank(mpi_comm_level1, &myid0);

    if (SCF_iter == 1 && myid0 == 0) {
        extern void openmx_gpu_zheevd_selftest(const char *label);
        const char *selftest_env = getenv("OPENMX_ZHEEVD_SELFTEST");
        if (selftest_env != NULL && atoi(selftest_env) != 0) {
            openmx_gpu_zheevd_selftest("band_top");
        }
    }

    MPI_Barrier(mpi_comm_level1);

    Num_Comm_World1 = SpinP_switch + 1;

    /***********************************************
         for pallalel calculations in myworld1
    ***********************************************/

    MPI_Comm_size(MPI_CommWD1[myworld1], &numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1], &myid1);

    MPI_Comm_rank(MPI_CommWD2[myworld2], &myid2);

    /****************************************************
     find the number of basis functions, n
    ****************************************************/

    n       = 0;
    max_tno = 0;
    for (i = 1; i <= atomnum; i++) {
        wanA = WhatSpecies[i];
        tnoA = Spe_Total_CNO[wanA];
        n += tnoA;
        if (max_tno < tnoA)
            max_tno = tnoA;
    }
    BandCol_RefineConfigure();
    use_gpusolver_dense = (scf_eigen_lib_flag == GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_Col_GpuSwitchNum() <= n) &&
                           BandCol_GpuDenseFits(n,myid1==0));

    /****************************************************
     find TZ
    ****************************************************/

    TZ = 0.0;
    for (i = 1; i <= atomnum; i++) {
        wan = WhatSpecies[i];
        TZ += Spe_Core_Charge[wan];
    }

    /***********************************************
       find the number of states to be solved
    ***********************************************/

    lumos = 0.4 * ((double)TZ - (double)system_charge) / 2.0;
    if (lumos < 50.0)
        lumos = 100.0;
    MaxN = (TZ - system_charge) / 2 + (int)lumos;
    if (n < MaxN)
        MaxN = n;

    /***********************************************
       allocation of arrays
    ***********************************************/

    My_NZeros = (int *)malloc(sizeof(int) * numprocs0);
    SP_NZeros = (int *)malloc(sizeof(int) * numprocs0);
    SP_Atoms  = (int *)malloc(sizeof(int) * numprocs0);
    /***********************************************
                k-points by regular mesh
    ***********************************************/

    if (way_of_kpoint == 1) {

        /**************************************************************
         k_op[i][j][k]: weight of DOS
                     =0   no calc.
                     =1   G-point
                     =2   which has k<->-k point
            Now, only the relation, E(k)=E(-k), is used.

        Future release: k_op will be used for symmetry operation
        *************************************************************/

        for (i = 0; i < knum_i; i++) {
            for (j = 0; j < knum_j; j++) {
                for (k = 0; k < knum_k; k++) {
                    k_op[i][j][k] = -999;
                }
            }
        }

        for (i = 0; i < knum_i; i++) {
            for (j = 0; j < knum_j; j++) {
                for (k = 0; k < knum_k; k++) {
                    if (k_op[i][j][k] == -999) {
                        k_inversion(i, j, k, knum_i, knum_j, knum_k, &ii, &ij, &ik);
                        if (i == ii && j == ij && k == ik) {
                            k_op[i][j][k] = 1;
                        }

                        else {
                            k_op[i][j][k]    = 2;
                            k_op[ii][ij][ik] = 0;
                        }
                    }
                } /* k */
            } /* j */
        } /* i */

        /***********************************
           one-dimentionalize for MPI
        ************************************/

        T_knum = 0;
        for (i = 0; i < knum_i; i++) {
            for (j = 0; j < knum_j; j++) {
                for (k = 0; k < knum_k; k++) {
                    if (0 < k_op[i][j][k]) {
                        T_knum++;
                    }
                }
            }
        }

        /* set T_KGrids1,2,3 and T_k_op */

        T_knum = 0;
        for (i = 0; i < knum_i; i++) {

            if (knum_i == 1)
                k1 = 0.0;
            else
                k1 = -0.5 + (2.0 * (double)i + 1.0) / (2.0 * (double)knum_i) + Shift_K_Point;

            for (j = 0; j < knum_j; j++) {

                if (knum_j == 1)
                    k2 = 0.0;
                else
                    k2 = -0.5 + (2.0 * (double)j + 1.0) / (2.0 * (double)knum_j) - Shift_K_Point;

                for (k = 0; k < knum_k; k++) {

                    if (knum_k == 1)
                        k3 = 0.0;
                    else
                        k3 = -0.5 + (2.0 * (double)k + 1.0) / (2.0 * (double)knum_k) + 2.0 * Shift_K_Point;

                    if (0 < k_op[i][j][k]) {

                        T_KGrids1[T_knum] = k1;
                        T_KGrids2[T_knum] = k2;
                        T_KGrids3[T_knum] = k3;
                        T_k_op[T_knum]    = k_op[i][j][k];

                        T_knum++;
                    }
                }
            }
        }

        if (myid0 == Host_ID && 0 < level_stdout) {

            printf(" KGrids1: ");
            fflush(stdout);
            for (i = 0; i <= knum_i - 1; i++) {
                if (knum_i == 1)
                    k1 = 0.0;
                else
                    k1 = -0.5 + (2.0 * (double)i + 1.0) / (2.0 * (double)knum_i) + Shift_K_Point;
                printf("%9.5f ", k1);
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);

            printf(" KGrids2: ");
            fflush(stdout);

            for (i = 0; i <= knum_j - 1; i++) {
                if (knum_j == 1)
                    k2 = 0.0;
                else
                    k2 = -0.5 + (2.0 * (double)i + 1.0) / (2.0 * (double)knum_j) - Shift_K_Point;
                printf("%9.5f ", k2);
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);

            printf(" KGrids3: ");
            fflush(stdout);
            for (i = 0; i <= knum_k - 1; i++) {
                if (knum_k == 1)
                    k3 = 0.0;
                else
                    k3 = -0.5 + (2.0 * (double)i + 1.0) / (2.0 * (double)knum_k) + 2.0 * Shift_K_Point;
                printf("%9.5f ", k3);
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);
        }

    }

    /***********************************************
                  Monkhorst-Pack k-points
    ***********************************************/

    else if (way_of_kpoint == 2) {

        T_knum = num_non_eq_kpt;

        for (k = 0; k < num_non_eq_kpt; k++) {
            T_KGrids1[k] = NE_KGrids1[k];
            T_KGrids2[k] = NE_KGrids2[k];
            T_KGrids3[k] = NE_KGrids3[k];
            T_k_op[k]    = NE_T_k_op[k];
        }
    }

    /***********************************************
                 Gamma-centered k-points
    ***********************************************/

    else if (way_of_kpoint == 3) {

        for (i = 0; i < knum_i; i++) {
            for (j = 0; j < knum_j; j++) {
                for (k = 0; k < knum_k; k++) {
                    k_op[i][j][k] = -999;
                }
            }
        }

        for (i = 0; i < knum_i; i++) {
            for (j = 0; j < knum_j; j++) {
                for (k = 0; k < knum_k; k++) {
                    if (k_op[i][j][k] == -999) {

                        if (i == 0 || 2 * i == knum_i)
                            ii = i;
                        else
                            ii = knum_i - i;

                        if (j == 0 || 2 * j == knum_j)
                            ij = j;
                        else
                            ij = knum_j - j;

                        if (k == 0 || 2 * k == knum_k)
                            ik = k;
                        else
                            ik = knum_k - k;

                        if ((i == 0 || 2 * i == knum_i) && (j == 0 || 2 * j == knum_j)
                            && (k == 0 || 2 * k == knum_k)) {
                            k_op[i][j][k] = 1;
                        }
                        else {
                            k_op[i][j][k]     = 2;
                            k_op[ii][ij][ik] = 0;
                        }
                    }
                } /* k */
            }     /* j */
        }         /* i */

        /***********************************
           one-dimentionalize for MPI
        ************************************/

        /* set T_KGrids1,2,3 and T_k_op */

        T_knum = 0;
        for (i = 0; i < knum_i; i++) {
            if (knum_i == 1)
                k1 = 0.0;
            else
                k1 = ((double)i) / ((double)knum_i) + Shift_K_Point;

            for (j = 0; j < knum_j; j++) {

                if (knum_j == 1)
                    k2 = 0.0;
                else
                    k2 = ((double)j) / ((double)knum_j) - Shift_K_Point;

                for (k = 0; k < knum_k; k++) {
                    if (knum_k == 1)
                        k3 = 0.0;
                    else
                        k3 = ((double)k) / ((double)knum_k) + 2.0 * Shift_K_Point;

                    if (0 < k_op[i][j][k]) {
                        T_KGrids1[T_knum] = k1;
                        T_KGrids2[T_knum] = k2;
                        T_KGrids3[T_knum] = k3;
                        T_k_op[T_knum]    = k_op[i][j][k];
                        T_knum++;
                    }
                }
            }
        }

        if (myid0 == Host_ID && 0 < level_stdout) {

            printf(" KGrids1: ");
            fflush(stdout);
            for (i = 0; i <= knum_i - 1; i++) {
                if (knum_i == 1)
                    k1 = 0.0;
                else
                    k1 = ((double)i) / ((double)knum_i) + Shift_K_Point;
                printf("%9.5f ", k1);
                fflush(stdout);
            }

            printf("\n");
            fflush(stdout);
            printf(" KGrids2: ");
            fflush(stdout);
            for (i = 0; i <= knum_j - 1; i++) {
                if (knum_j == 1)
                    k2 = 0.0;
                else
                    k2 = ((double)i) / ((double)knum_j) - Shift_K_Point;
                printf("%9.5f ", k2);
                fflush(stdout);
            }

            printf("\n");
            fflush(stdout);
            printf(" KGrids3: ");
            fflush(stdout);

            for (i = 0; i <= knum_k - 1; i++) {
                if (knum_k == 1)
                    k3 = 0.0;
                else
                    k3 = ((double)i) / ((double)knum_k) + 2.0 * Shift_K_Point;
                printf("%9.5f ", k3);
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);
        }
    }

    /***********************************************
              calculate the sum of weights
    ***********************************************/

    sum_weights = 0.0;
    for (k = 0; k < T_knum; k++) {
        sum_weights += (double)T_k_op[k];
    }

    /***********************************************
             allocate k-points into processors
    ***********************************************/

    if (numprocs1 < T_knum) {

        /* set parallel_mode */
        parallel_mode = 0;

        /* allocation of kloop to ID */

        for (ID = 0; ID < numprocs1; ID++) {
            tmp    = (double)T_knum / (double)numprocs1;
            S_knum = (int)((double)ID * (tmp + 1.0e-12));
            E_knum = (int)((double)(ID + 1) * (tmp + 1.0e-12)) - 1;
            if (ID == (numprocs1 - 1))
                E_knum = T_knum - 1;
            if (E_knum < 0)
                E_knum = 0;

            for (k = S_knum; k <= E_knum; k++) {
                /* ID in the first level world */
                T_k_ID[myworld1][k] = ID;
            }
        }

        /* find own informations */

        tmp    = (double)T_knum / (double)numprocs1;
        S_knum = (int)((double)myid1 * (tmp + 1.0e-12));
        E_knum = (int)((double)(myid1 + 1) * (tmp + 1.0e-12)) - 1;
        if (myid1 == (numprocs1 - 1))
            E_knum = T_knum - 1;
        if (E_knum < 0)
            E_knum = 0;

        num_kloop0 = E_knum - S_knum + 1;

        MPI_Comm_size(MPI_CommWD2[myworld2], &numprocs2);
        MPI_Comm_rank(MPI_CommWD2[myworld2], &myid2);
    }

    else {

        /* set parallel_mode */
        parallel_mode = 1;
        num_kloop0    = 1;

        Num_Comm_World2 = T_knum;
        MPI_Comm_size(MPI_CommWD2[myworld2], &numprocs2);
        MPI_Comm_rank(MPI_CommWD2[myworld2], &myid2);

        S_knum = myworld2;

        /* allocate k-points into processors */

        for (k = 0; k < T_knum; k++) {
            /* ID in the first level world */
            T_k_ID[myworld1][k] = Comm_World_StartID2[k];
        }
    }

    /****************************************************
     find all_knum
     if (all_knum==1), all the calculation will be made
     by the first diagonalization loop, and the second
     diagonalization will be skipped.
    ****************************************************/

    MPI_Allreduce(&num_kloop0, &all_knum, 1, MPI_INT, MPI_PROD, mpi_comm_level1);

    if (SpinP_switch == 1 && numprocs0 == 1 && all_knum == 1) {
        all_knum = 0;
    }

    owns_dense_k_rank = (use_gpusolver_dense && Set_Hamiltonian_OpenMP_Rank_Is_Selected());
    owns_global_dense_rank = (all_knum == 1 && owns_dense_k_rank);
    if (use_gpusolver_dense && all_knum != 1) {
        const int turns = Num_Comm_World1 * T_knum;
        int interleave_k = 1;
        gpu_k_order = (int*)malloc(sizeof(int)*(size_t)turns);
        if (gpu_k_order == NULL)
            BandCol_AbortWithMessage("Cannot allocate GPU k-point turn order.");
        for (int point=0; point<turns; ++point) gpu_k_order[point] = point;
        /* the order makes the turn groups of both passes: the root's
           OPENMX_BAND_COL_INTERLEAVE_K (default 1) holds for every rank */
        if (myid0 == Host_ID) {
            const char *interleave = getenv("OPENMX_BAND_COL_INTERLEAVE_K");
            if (interleave != NULL) interleave_k = (atoi(interleave) != 0);
        }
        MPI_Bcast(&interleave_k, 1, MPI_INT, Host_ID, mpi_comm_level1);
        if (interleave_k) {
            int *owners = (int*)malloc(sizeof(int)*(size_t)turns);
            int *next = (int*)calloc((size_t)numprocs0, sizeof(int));
            int turn = 0;
            if (owners == NULL || next == NULL)
                BandCol_AbortWithMessage("Cannot allocate GPU k-point owner cursors.");
            for (int point=0; point<turns; ++point) owners[point] = INT_MAX;
            if (owns_dense_k_rank) {
                for (int channel=0; channel<Num_Comm_World1; ++channel) {
                    if (numprocs0 != 1 && channel != myworld1) continue;
                    for (int point=S_knum; point<S_knum+num_kloop0; ++point)
                        owners[channel*T_knum+point] = myid0;
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, owners, turns, MPI_INT, MPI_MIN, mpi_comm_level1);
            for (int point=0; point<turns; ++point) {
                if (owners[point] == INT_MAX)
                    BandCol_AbortWithMessage("Missing GPU k-point owner.");
            }
            /* Adjacent k points belong to the same rank. Interleave owners
               to use the concurrency admitted by the GPU memory guard. */
            while (turn < turns) {
                for (int owner=0; owner<numprocs0; ++owner) {
                    while (next[owner] < turns && owners[next[owner]] != owner) ++next[owner];
                    if (next[owner] < turns) gpu_k_order[turn++] = next[owner]++;
                }
            }
            free(next);
            free(owners);
        }
    }

    if (use_gpusolver_dense && all_knum != 1) {
        BandCol_KOverlapPrepare(n, T_knum);
        if (MaxN > 0 && (size_t)n <= SIZE_MAX / (size_t)MaxN / sizeof(dcomplex)) {
            k_cache_bytes = (size_t)n * (size_t)MaxN * sizeof(dcomplex);
            k_cache_limit = BandCol_HostCacheLimit("OPENMX_BAND_COL_KCACHE_MB", 1024.0);
            if (k_cache_bytes <= k_cache_limit)
                k_evec_cache = (dcomplex**)calloc((size_t)Num_Comm_World1 * T_knum, sizeof(dcomplex*));
        }
    }

    use_setham_packed_cache =
        (use_gpusolver_dense && all_knum == 1 && Set_Hamiltonian_GpuSolver_Packed_CacheReady() &&
         Set_Hamiltonian_GpuSolver_Packed_OrderMode() == 0);
    if (use_setham_packed_cache) {
        size_H1 = Set_Hamiltonian_GpuSolver_Packed_Size();
        Set_Hamiltonian_GpuSolver_SetMP(MP);
        if (Set_Hamiltonian_GpuSolver_Packed_OwnsCache()) {
            setham_order_GA = Set_Hamiltonian_GpuSolver_Packed_OrderGA();
            setham_S1 = Set_Hamiltonian_GpuSolver_Packed_Overlap();
        }
    }

    // Set the device to be used by HIP and OpenMP target offload.
    if (use_gpusolver_dense && Set_Hamiltonian_OpenMP_Rank_Is_Selected()) {
        set_hip_default_device_from_local_rank_noncollective();
    }

    /* the paths of the refined eigensolver in this call, alike on every rank:
       one k-point per rank with every owner resident (all turns in one
       group), the same in serialized turn groups (staged), or the k-dense
       path; a single rank of a spin-polarized run keeps FP64 */
    {
        const int refine_base = (!(SpinP_switch == 1 && numprocs0 == 1) && ParDM_flag == 0 &&
                                 openmx_eigen_refine_size_ok(n));
        const int serialize_turns = BandCol_SerializeGpuSolverGpuTurns();
        const int total_turns = Num_Comm_World1 * T_knum;
        const int turn_group = serialize_turns ? BandCol_GpuTurnGroupForScf(SCF_iter) : total_turns;
        int refine_fits = 0;

        if (use_gpusolver_dense) {
            const int turns = (all_knum == 1) ? ((total_turns <= turn_group) ? total_turns : turn_group)
                                              : BandCol_MaxConcurrentKGpuTurns(n);

            refine_fits = BandCol_RefineFits(refine_base, n, MaxN, turns,
                                             (all_knum == 1) ? owns_global_dense_rank : owns_dense_k_rank);
        }
        BandCol_RefineBeginStep(SCF_iter, refine_fits && all_knum == 1 && total_turns <= turn_group,
                                refine_fits && all_knum != 1, refine_fits && all_knum == 1 && turn_group < total_turns);
        if (BandCol_refine.kdense || BandCol_refine.staged) BandCol_KRefEnsure(total_turns, n);
        /* the basis the resident path keeps on the device between its steps
           serves only that path's warm start: a call on another path (or in
           FP64) gives its memory back at once instead of at the end of the
           cycle */
        if (!BandCol_refine.last_refined) BandCol_RefineDropBasis();
    }

    /***********************************************
                hipblasmp & hipsolverMp initialize
    ***********************************************/

    // Options opts = {
    //     .m = n,
    //     .n = n,
    //     .k = n,
    //     .mbA = nblk,
    //     .nbA = nblk,
    //     .mbB = nblk,
    //     .nbB = nblk,
    //     .mbC = nblk,
    //     .nbC = nblk,
    //     .ia = 1,
    //     .ja = 1,
    //     .ib = 1,
    //     .jb = 1,
    //     .ic = 1,
    //     .jc = 1,
    //     .grid_layout = 'R'
    // };

    // Options2 opts2;

    // Options3 opts3 = {
    //     .n = n,
    //     .mbA = nblk,
    //     .nbA = nblk,
    //     .mbQ = nblk,
    //     .nbQ = nblk,
    //     .ia = 1,
    //     .ja = 1,
    //     .iq = 1,
    //     .jq = 1,
    // };

    // Options4 opts4;
    // if (all_knum == 1) {
    //     init_hipblasmp(myworld2, MPI_CommWD2, &opts, &opts2);
    //     init_gpusolvermp(myworld2, MPI_CommWD2, &opts2, &opts3, &opts4);
    // }

    if (all_knum != 1 && use_gpusolver_dense) {
        if (n != na_cols || n != na_rows) {
            BandCol_AbortWithMessage("GPUSOLVER band path requires full dense matrices in Band_DFT_Col.c.");
        }
    } else if (!use_gpusolver_dense) {
        MPI_Comm_split(MPI_CommWD2[myworld2], my_pcol, my_prow, &mpi_comm_rows);
        MPI_Comm_split(MPI_CommWD2[myworld2], my_prow, my_pcol, &mpi_comm_cols);

        mpi_comm_rows_int     = MPI_Comm_c2f(mpi_comm_rows);
        mpi_comm_cols_int     = MPI_Comm_c2f(mpi_comm_cols);
        mpi_eigen_comms_ready = 1;
    }

    /****************************************************
      if (parallel_mode==1 && all_knum==1)
       make is1, ie1, is2, ie2
    ****************************************************/

    if (all_knum == 1) {

        /* allocation */

        is1 = (int *)malloc(sizeof(int) * numprocs2);
        ie1 = (int *)malloc(sizeof(int) * numprocs2);

        is2 = (int *)malloc(sizeof(int) * numprocs2);
        ie2 = (int *)malloc(sizeof(int) * numprocs2);

        Num_Snd_EV = (int *)malloc(sizeof(int) * numprocs2);
        Num_Rcv_EV = (int *)malloc(sizeof(int) * numprocs2);

        /* make is1 and ie1 */

        if (numprocs2 <= n) {

            av_num = (double)n / (double)numprocs2;

            for (ID = 0; ID < numprocs2; ID++) {
                is1[ID] = (int)(av_num * (double)ID) + 1;
                ie1[ID] = (int)(av_num * (double)(ID + 1));
            }

            is1[0]             = 1;
            ie1[numprocs2 - 1] = n;

        }

        else {

            for (ID = 0; ID < n; ID++) {
                is1[ID] = ID + 1;
                ie1[ID] = ID + 1;
            }
            for (ID = n; ID < numprocs2; ID++) {
                is1[ID] = 1;
                ie1[ID] = 0;
            }
        }

        /* make is2 and ie2 */

        if (numprocs2 <= MaxN) {

            av_num = (double)MaxN / (double)numprocs2;

            for (ID = 0; ID < numprocs2; ID++) {
                is2[ID] = (int)(av_num * (double)ID) + 1;
                ie2[ID] = (int)(av_num * (double)(ID + 1));
            }

            is2[0]             = 1;
            ie2[numprocs2 - 1] = MaxN;
        }

        else {
            for (ID = 0; ID < MaxN; ID++) {
                is2[ID] = ID + 1;
                ie2[ID] = ID + 1;
            }
            for (ID = MaxN; ID < numprocs2; ID++) {
                is2[ID] = 1;
                ie2[ID] = 0;
            }
        }

        /****************************************************************
           making data structure of MPI communicaition for eigenvectors
        ****************************************************************/

        for (ID = 0; ID < numprocs2; ID++) {
            Num_Snd_EV[ID] = 0;
            Num_Rcv_EV[ID] = 0;
        }

        for (i = 0; i < na_rows; i++) {

            ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + my_prow) % np_rows) * nblk + 1;

            po = 0;
            for (ID = 0; ID < numprocs2; ID++) {
                if (is2[ID] <= ig && ig <= ie2[ID]) {
                    po  = 1;
                    ID0 = ID;
                    break;
                }
            }

            if (po == 1)
                Num_Snd_EV[ID0] += na_cols;
        }

        for (ID = 0; ID < numprocs2; ID++) {
            IDS = (myid2 + ID) % numprocs2;
            IDR = (myid2 - ID + numprocs2) % numprocs2;
            if (ID != 0) {
                MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
                MPI_Recv(&Num_Rcv_EV[IDR], 1, MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
                MPI_Wait(&request, &stat);
            } else {
                Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
            }
        }

        Max_Num_Snd_EV = 0;
        Max_Num_Rcv_EV = 0;
        for (ID = 0; ID < numprocs2; ID++) {
            if (Max_Num_Snd_EV < Num_Snd_EV[ID])
                Max_Num_Snd_EV = Num_Snd_EV[ID];
            if (Max_Num_Rcv_EV < Num_Rcv_EV[ID])
                Max_Num_Rcv_EV = Num_Rcv_EV[ID];
        }

        Max_Num_Snd_EV++;
        Max_Num_Rcv_EV++;

        index_Snd_i = (int *)malloc(sizeof(int) * Max_Num_Snd_EV);
        index_Snd_j = (int *)malloc(sizeof(int) * Max_Num_Snd_EV);
        EVec_Snd    = (double *)malloc(sizeof(double) * Max_Num_Snd_EV * 2);
        index_Rcv_i = (int *)malloc(sizeof(int) * Max_Num_Rcv_EV);
        index_Rcv_j = (int *)malloc(sizeof(int) * Max_Num_Rcv_EV);
        EVec_Rcv    = (double *)malloc(sizeof(double) * Max_Num_Rcv_EV * 2);

    } /* if (all_knum==1) */

    /****************************************************
       PrintMemory
    ****************************************************/

    if (firsttime && memoryusage_fileout) {
        PrintMemory("Band_DFT_Col: My_NZeros", sizeof(int) * numprocs0, NULL);
        PrintMemory("Band_DFT_Col: SP_NZeros", sizeof(int) * numprocs0, NULL);
        PrintMemory("Band_DFT_Col: SP_Atoms", sizeof(int) * numprocs0, NULL);
        if (all_knum == 1) {
            PrintMemory("Band_DFT_Col: is1", sizeof(int) * numprocs2, NULL);
            PrintMemory("Band_DFT_Col: ie1", sizeof(int) * numprocs2, NULL);
            PrintMemory("Band_DFT_Col: is2", sizeof(int) * numprocs2, NULL);
            PrintMemory("Band_DFT_Col: ie2", sizeof(int) * numprocs2, NULL);
            PrintMemory("Band_DFT_Col: Num_Snd_EV", sizeof(int) * numprocs2, NULL);
            PrintMemory("Band_DFT_Col: Num_Rcv_EV", sizeof(int) * numprocs2, NULL);
            PrintMemory("Band_DFT_Col: index_Snd_i", sizeof(int) * Max_Num_Snd_EV, NULL);
            PrintMemory("Band_DFT_Col: index_Snd_j", sizeof(int) * Max_Num_Snd_EV, NULL);
            PrintMemory("Band_DFT_Col: EVec_Snd", sizeof(double) * Max_Num_Snd_EV * 2, NULL);
            PrintMemory("Band_DFT_Col: index_Rcv_i", sizeof(int) * Max_Num_Rcv_EV, NULL);
            PrintMemory("Band_DFT_Col: index_Rcv_j", sizeof(int) * Max_Num_Rcv_EV, NULL);
            PrintMemory("Band_DFT_Col: EVec_Rcv", sizeof(double) * Max_Num_Rcv_EV * 2, NULL);
        }
    }

    /****************************************************
       communicate T_k_ID
    ****************************************************/

    if (numprocs0 == 1 && SpinP_switch == 1) {
        for (k = 0; k < T_knum; k++) {
            T_k_ID[1][k] = T_k_ID[0][k];
        }
    } else {
        for (spin = 0; spin <= SpinP_switch; spin++) {
            ID = Comm_World_StartID1[spin];
            MPI_Bcast(&T_k_ID[spin][0], T_knum, MPI_INT, ID, mpi_comm_level1);
        }
    }

    /****************************************************
       store in each processor all the matrix elements
          for overlap and Hamiltonian matrices
    ****************************************************/

    if (measure_time)
        dtime(&Stime);

    if (measure_time)
        dtime(&starttime);

    /* spin=myworld1 */

    spin = myworld1;

    /* set S1 */

    if (use_setham_packed_cache) {
        if (Set_Hamiltonian_GpuSolver_Packed_OwnsCache()) {
            if (setham_order_GA == NULL || setham_S1 == NULL) {
                BandCol_AbortWithMessage("Set_Hamiltonian packed overlap cache is missing in Band_DFT_Col.c.");
            }
        }
    }
    else if (SCF_iter == 1 || all_knum != 1) {
        size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    }

    transformed_s_ready = 0;

diagonalize1:

    /* set H1 */

    if (use_setham_packed_cache) {
        int cache_spin = (SpinP_switch == 0) ? 0 : spin;

        if (Set_Hamiltonian_GpuSolver_Packed_OwnsCache()) {
            setham_H1 = Set_Hamiltonian_GpuSolver_Packed_H(cache_spin);
            if (setham_H1 == NULL) {
                BandCol_AbortWithMessage("Set_Hamiltonian packed Hamiltonian cache is missing in Band_DFT_Col.c.");
            }
        }
    }
    else if (SpinP_switch == 0) {
        size_H1 = Get_OneD_HS_Col(1, nh[0], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    } else if (1 < numprocs0) {

        size_H1 = Get_OneD_HS_Col(1, nh[0], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
        size_H1 = Get_OneD_HS_Col(1, nh[1], CDM1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

        if (myworld1) {
            for (i = 0; i < size_H1; i++) {
                H1[i] = CDM1[i];
            }
        }
    } else {
        size_H1 = Get_OneD_HS_Col(1, nh[spin], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    }

    if (measure_time) {
        dtime(&Etime);
        time1 += Etime - Stime;
    }

    /****************************************************
                         start kloop
    ****************************************************/

    dtime(&SiloopTime);

    if (SCF_iter == 1 && myid0 == 0 && use_gpusolver_dense) {
        extern void openmx_gpu_zheevd_selftest(const char *label);
        const char *selftest_env = getenv("OPENMX_ZHEEVD_SELFTEST");
        if (selftest_env != NULL && atoi(selftest_env) != 0) {
            openmx_gpu_zheevd_selftest("band_entry");
        }
    }

    if (all_knum != 1 && use_gpusolver_dense) {
        const int max_concurrent_gpu_turns = BandCol_MaxConcurrentKGpuTurns(n);
        const int total_gpu_turns = Num_Comm_World1 * T_knum;

        for (int group_first = 0; group_first < total_gpu_turns; group_first += max_concurrent_gpu_turns) {
            const int group_last =
                (group_first + max_concurrent_gpu_turns < total_gpu_turns) ?
                (group_first + max_concurrent_gpu_turns) : total_gpu_turns;

            MPI_Barrier(mpi_comm_level1);

            for (int gpu_turn = group_first; gpu_turn < group_last; gpu_turn++) {
                dcomplex *evec_device;
                int construct_on_device;

                if (!owns_dense_k_rank || gpu_k_order[gpu_turn] / T_knum != spin) {
                    continue;
                }

                kloop = gpu_k_order[gpu_turn] % T_knum;
                if (kloop < S_knum || S_knum + num_kloop0 <= kloop) {
                    continue;
                }

                (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);

                k1 = T_KGrids1[kloop];
                k2 = T_KGrids2[kloop];
                k3 = T_KGrids3[kloop];

                /* make S and H */

                Construct_Band_CsHs(SCF_iter, all_knum, use_setham_packed_cache ? setham_order_GA : order_GA, MP,
                                    use_setham_packed_cache ? setham_S1 : S1,
                                    use_setham_packed_cache ? setham_H1 : H1, k1, k2, k3, Ss, Hs, n,
                                    use_gpusolver_dense, owns_global_dense_rank);
                construct_on_device = BandCol_LastConstructOnDevice();

                /* for blas */

                /* diagonalize S */

                if (measure_time)
                    dtime(&Stime);

                if (BandCol_KOverlapRestore(kloop, k1, k2, k3, ko, koS)) {
                    ++k_overlap_hits;
                } else {
                    BandCol_GpuSolver_PrepareTransformedS(1, n, construct_on_device ? NULL : Ss, ko, koS);
                    BandCol_KOverlapSave(kloop, k1, k2, k3, koS);
                }

                if (measure_time) {
                    dtime(&Etime);
                    time2 += Etime - Stime;
                }

                /****************************************************
                 1.0/sqrt(ko[l]) * U^t * H * U * 1.0/sqrt(ko[l])
                ****************************************************/

                if (measure_time)
                    dtime(&Stime);

                /* a refined pass 1 builds no vectors (pass 2 does, from its
                   basis), so it has nothing for the eigenvector cache */
                {
                    const int refined_pass1 =
                        BandCol_refine.kdense &&
                        BandCol_KDenseRefinedPass1(n, MaxN, construct_on_device ? NULL : Hs, ko, spin * T_knum + kloop);

                    if (!refined_pass1) {
                        if (construct_on_device) {
                            evec_device = BandCol_GpuSolver_SolveHamiltonianDeviceInput(n, MaxN, ko);
                        } else {
                            evec_device = BandCol_GpuSolver_SolveHamiltonianDeviceOnly(n, MaxN, Hs, ko);
                        }
                        if (k_evec_cache != NULL && k_cache_bytes <= k_cache_limit - k_cache_used) {
                            size_t slot = (size_t)spin * T_knum + kloop;
                            k_evec_cache[slot] = (dcomplex*)malloc(k_cache_bytes);
                            if (k_evec_cache[slot] != NULL) {
                                if(BandCol_mgpu_host_dense)memcpy(k_evec_cache[slot],evec_device,k_cache_bytes);
                                else wait_hipfunc(hipMemcpy(k_evec_cache[slot],evec_device,k_cache_bytes,hipMemcpyDeviceToHost));
                                k_cache_used += k_cache_bytes;
                            }
                        }
                    }
                    BandCol_GpuSolver_ReleaseDeviceMemory();
                    if (refined_pass1) {
                        /* the basis lives in the host slot until pass 2; the
                           products' unblocked scratch goes with the turn */
                        BandCol_RefineDropBasis();
                        openmx_gemmul8TrimWorkspaces();
                    }
                }

                if (measure_time) {
                    dtime(&Etime);
                    time3 += Etime - Stime;
                }

                for (int l = 1; l <= MaxN; l++) {
                    EIGEN[spin][kloop][l] = ko[l];
                }

                if (3 <= level_stdout && 0 <= kloop) {
                    printf(" myid0=%2d spin=%2d kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n", myid0, spin, kloop,
                           T_KGrids1[kloop], T_KGrids2[kloop], T_KGrids3[kloop]);
                    for (i1 = 1; i1 <= n; i1++) {
                        if (SpinP_switch == 0)
                            printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n", i1, EIGEN[0][kloop][i1],
                                   EIGEN[0][kloop][i1]);
                        else
                            printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n", i1, EIGEN[0][kloop][i1],
                                   EIGEN[1][kloop][i1]);
                    }
                }

                if (measure_time)
                    dtime(&Stime);

                if (measure_time) {
                    dtime(&Etime);
                    time5 += Etime - Stime;
                }
            } /* gpu_turn */
        } /* group_first */
    } else {
        for (kloop0 = 0; kloop0 < num_kloop0; kloop0++) {
            if (use_gpusolver_dense) {
                int my_gpu_turn;
                kloop = S_knum + kloop0;

                (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);

                k1 = T_KGrids1[kloop];
                k2 = T_KGrids2[kloop];
                k3 = T_KGrids3[kloop];

                /* make S and H */
                double starttimesh = 0.0, endtimesh = 0.0;
                if (measure_time) {
                    dtime(&starttimesh);
                }

                my_gpu_turn = spin * T_knum + kloop;
                const int serialize_gpu_turns = BandCol_SerializeGpuSolverGpuTurns();
                const int gpu_turn_group = serialize_gpu_turns ? BandCol_GpuTurnGroupForScf(SCF_iter) : 1;
                const int first_gpu_turn = serialize_gpu_turns ? 0 : my_gpu_turn;
                const int last_gpu_turn = serialize_gpu_turns ? Num_Comm_World1 * T_knum : (my_gpu_turn + 1);
                for (int gpu_turn = first_gpu_turn; gpu_turn < last_gpu_turn; gpu_turn++) {
                    if (serialize_gpu_turns && gpu_turn % gpu_turn_group == 0) {
                        MPI_Barrier(mpi_comm_level1);
                    }

                    if (owns_global_dense_rank && my_gpu_turn == gpu_turn) {
                        dcomplex *evec_device;
                        const int persistent_s = BandCol_GpuPersistentDecide();
                        const int device_s_ready = persistent_s && BandCol_GpuSolver_HasDeviceTransformedS(n);
                        const int host_s_ready = BandCol_GpuSolver_HasHostTransformedS(n);
                        const int build_s = (SCF_iter == 1) || (!device_s_ready && !host_s_ready);
                        const int construct_scf_iter = build_s ? 1 : SCF_iter;

                        Construct_Band_CsHs(construct_scf_iter, all_knum, use_setham_packed_cache ? setham_order_GA : order_GA, MP,
                                            use_setham_packed_cache ? setham_S1 : S1,
                                            use_setham_packed_cache ? setham_H1 : H1, k1, k2, k3, Ss, Hs,
                                            n, use_gpusolver_dense, owns_global_dense_rank);
                        const int construct_on_device = BandCol_LastConstructOnDevice();

                        if (measure_time) {
                            dtime(&endtimesh);

                            if (myid0 == 0) {
                                printf("make S and H: %.7f (sec)\n", endtimesh - starttimesh);
                            }
                        }

                        if (measure_time) {
                            dtime(&endtime);
                            part1 += endtime - starttime;
                            if (SCF_iter != 1) {
                                part1sum += part1;
                            }
                        }

                        if (measure_time)
                            dtime(&Stime);

                        if (build_s) {
                            BandCol_GpuSolver_PrepareTransformedS(1, n, construct_on_device ? NULL : Ss, ko, koS);
                            /* A persistent device cache is already the object
                               consumed by both ZGEMMs.  Copying another n*n
                               matrix to pageable host memory wastes bandwidth
                               and, on MI300A/XNACK, creates avoidable mapped
                               pages.  Keep a host fallback only when device
                               buffers will be released between SCFs. */
                            if (!persistent_s) {
                                BandCol_GpuSolver_SaveTransformedS(n);
                            }
                        } else if (!device_s_ready) {
                            BandCol_GpuSolver_LoadTransformedS(n);
                        }
                        transformed_s_ready = 1;

                        if (BandCol_CacheTraceEnabled()) {
                            printf("<Band_DFT> transformed-S cache rank=%d k=%d SCF=%d: %s%s\n",
                                   myid0, kloop, SCF_iter,
                                   build_s ? "build" : "hit",
                                   build_s ? (persistent_s ? " (device-only)" : " (host fallback saved)")
                                           : (device_s_ready ? " (device)" : " (host reload)"));
                            fflush(stdout);
                        }

                        if (SCF_iter == 1 && measure_time) {
                            dtime(&Etime);
                            time2 += Etime - Stime;
                            part2_1 += Etime - Stime;
                            part2_2 += Etime - Stime;
                        }

                        if (measure_time)
                            dtime(&Stime);

                        {
                            /* refined: pass 1 of the staged solve (the basis to
                               the host slot, no vectors), or the first half of
                               a solve every owner keeps resident until the
                               chemical potential is known (pending: nothing is
                               released) */
                            const int staged_pass1 =
                                BandCol_refine.staged &&
                                BandCol_KDenseRefinedPass1(n, MaxN, construct_on_device ? NULL : Hs, ko, my_gpu_turn);

                            evec_device = NULL;
                            if (!staged_pass1) {
                                if (BandCol_refine.last_refined) {
                                    evec_device = BandCol_RefinedFirstHalf(n, MaxN, construct_on_device ? NULL : Hs, ko);
                                } else if (construct_on_device) {
                                    evec_device = BandCol_GpuSolver_SolveHamiltonianDeviceInput(n, MaxN, ko);
                                } else {
                                    evec_device = BandCol_GpuSolver_SolveHamiltonianDeviceOnly(n, MaxN, Hs, ko);
                                }
                            }

                            if (staged_pass1) {
                                BandCol_AfterStagedTurn();
                            } else if (evec_device != NULL) {
                                if (BandCol_GemmWorkspaceTurnRelease()) {
                                    openmx_gemmul8ReleaseWorkspaces();
                                }

                                if (BandCol_GpuPersistentDecide()) {
                                    (void)evec_device;
                                } else {
                                    BandCol_GpuSolver_SaveDeviceEigenvectors(evec_device, n,
                                                                             BandCol_GpuSolver_EigenvectorStride());
                                    BandCol_ConstructCache_Reset();
                                    BandCol_GpuSolver_ReleaseDeviceMemory();
                                }
                            }
                        }

                        if (measure_time) {
                            dtime(&Etime);
                            time3 += Etime - Stime;
                            part2_3 += Etime - Stime;
                        }

                        if (measure_time)
                            dtime(&Stime);

                        for (l = 1; l <= MaxN; l++) {
                            EIGEN[spin][kloop][l] = ko[l];
                        }

                        if (measure_time) {
                            dtime(&Etime);
                            time4 += Etime - Stime;
                            part2_4 += Etime - Stime;
                        }

                        if (measure_time) {
                            dtime(&starttime);
                            dtime(&endtime);
                            partmul = endtime - starttime;
                        }
                    }
                }
                if (!serialize_gpu_turns) {
                    MPI_Barrier(mpi_comm_level1);
                }

		            } else {
                kloop = S_knum + kloop0;

                (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);

                k1 = T_KGrids1[kloop];
                k2 = T_KGrids2[kloop];
                k3 = T_KGrids3[kloop];

                /* make S and H */

                Construct_Band_CsHs(SCF_iter, all_knum, use_setham_packed_cache ? setham_order_GA : order_GA, MP,
                                    use_setham_packed_cache ? setham_S1 : S1,
                                    use_setham_packed_cache ? setham_H1 : H1, k1, k2, k3, Cs, Hs, n,
                                    use_gpusolver_dense, owns_global_dense_rank);

                if (measure_time) {
                    dtime(&endtime);
                    part1 += endtime - starttime;
                    if (SCF_iter != 1) {
                        part1sum += part1;
                    }
                }


                /* diagonalize S */

                if (measure_time)
                    dtime(&Stime);

                if (parallel_mode == 0 || (SCF_iter == 1 || all_knum != 1)) {
                    if (scf_eigen_lib_flag == 1) {
                        F77_NAME(solve_evp_complex, SOLVE_EVP_COMPLEX)
                        (&n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
                    } else if (scf_eigen_lib_flag == 2 || scf_eigen_lib_flag == GPUSOLVER) {

#ifndef kcomp
                        int mpiworld;
                        mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
                        F77_NAME(elpa_solve_evp_complex_2stage_double_impl, ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
                        (&n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int,
                         &mpi_comm_cols_int, &mpiworld);
#endif
                    }
                }

                if (measure_time) {
                    dtime(&Etime);
                    time2 += Etime - Stime;
                    part2_1 += Etime - Stime;
                    if (SCF_iter != 1) {
                        part2_1sum += part2_1;
                    }
                }

                if (measure_time) {
                    dtime(&Stime);
                }

                if (SCF_iter == 1 || all_knum != 1) {

                    if (3 <= level_stdout) {
                        printf(" myid0=%2d spin=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n", myid0, spin, kloop,
                               T_KGrids1[kloop], T_KGrids2[kloop], T_KGrids3[kloop]);
                        for (i1 = 1; i1 <= n; i1++) {
                            printf("  Eigenvalues of OLP  %2d  %15.12f\n", i1, ko[i1]);
                        }
                    }

                    /* minus eigenvalues to 1.0e-10 */

                    for (l = 1; l <= n; l++) {
                        if (ko[l] < 0.0) {
                            ko[l] = 1.0e-10;
                        }
                        koS[l] = ko[l];
                    }

                    /* calculate S*1/sqrt(ko) */

                    for (l = 1; l <= n; l++) {
                        ko[l] = 1.0 / sqrt(ko[l]);
                    }

                    /* S * 1.0/sqrt(ko[l]) */

                    for (j = 0; j < na_cols; j++) {
                        const int jg = np_cols * nblk * (j / nblk) + (j % nblk) + ((np_cols + my_pcol) % np_cols) * nblk + 1;
                        const double scale = ko[jg];
                        dcomplex * restrict col = &Ss[j * na_rows];

                        for (i = 0; i < na_rows; i++) {
                            col[i].r *= scale;
                            col[i].i *= scale;
                        }
                    }
                }

                // printf("Ss[1][1] = %.7f Ss[1][2] = %.7f Ss[1][3] = %.7f\n", Ss[0].r, Ss[n].r, Ss[2 * n].r);
                // sleep(100);
                // exit(-1);

                /****************************************************
             1.0/sqrt(ko[l]) * U^t * H * U * 1.0/sqrt(ko[l])
            ****************************************************/

                if (measure_time) {
                    dtime(&Etime);
                    part2_2 += Etime - Stime;
                    if (SCF_iter != 1) {
                        part2_2sum += part2_2;
                    }
                }

                if (measure_time)
                    dtime(&Stime);

                /* pzgemm */

                /* H * U * 1.0/sqrt(ko[l]) */

                // hipblasmp_zgemm(HIPBLAS_OP_N, HIPBLAS_OP_N, Hs, Ss, Cs, &opts, &opts2);

                BandCol_ZeroComplex(Cs, (size_t)na_rows_max * (size_t)na_cols_max);

                Cblacs_barrier(ictxt2, "A");
                F77_NAME(pzgemm, PZGEMM)
                ("N", "N", &n, &n, &n, &alpha, Hs, &ONE, &ONE, descH, Ss, &ONE, &ONE, descS, &beta, Cs, &ONE, &ONE,
                 descC);

                // printf("BLAS_C[1][1] = %.7f BLAS_C[1][2] = %.7f BLAS_C[1][3] = %.7f\n", Cs[0].r, Cs[n].r, Cs[2 * n].r);

                /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

                // hipblasmp_zgemm(HIPBLAS_OP_C, HIPBLAS_OP_N, Ss, Cs, Hs, &opts, &opts2);

                BandCol_ZeroComplex(Hs, (size_t)na_rows * (size_t)na_cols);

                Cblacs_barrier(ictxt2, "C");
                F77_NAME(pzgemm, PZGEMM)
                ("C", "N", &n, &n, &n, &alpha, Ss, &ONE, &ONE, descS, Cs, &ONE, &ONE, descC, &beta, Hs, &ONE, &ONE,
                 descH);

                if (measure_time) {
                    dtime(&Etime);
                    time3 += Etime - Stime;
                    part2_3 += Etime - Stime;
                    if (SCF_iter != 1) {
                        part2_3sum += part2_3;
                    }
                }

                /* diagonalize H' */

                if (measure_time)
                    dtime(&Stime);

                if (scf_eigen_lib_flag == 1) {
                    F77_NAME(solve_evp_complex, SOLVE_EVP_COMPLEX)
                    (&n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
                } else if (scf_eigen_lib_flag == 2 || scf_eigen_lib_flag == GPUSOLVER) {

#ifndef kcomp
                    int mpiworld;
                    mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
                    F77_NAME(elpa_solve_evp_complex_2stage_double_impl, ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
                    (&n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int,
                     &mpi_comm_cols_int, &mpiworld);
#endif
                }

                if (measure_time) {
                    dtime(&Etime);
                    time4 += Etime - Stime;
                    part2_4 += Etime - Stime;
                    if (SCF_iter != 1) {
                        part2_4sum += part2_4;
                    }
                }

                for (l = 1; l <= MaxN; l++) {
                    EIGEN[spin][kloop][l] = ko[l];
                }

                if (3 <= level_stdout && 0 <= kloop) {
                    printf(" myid0=%2d spin=%2d kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n", myid0, spin, kloop,
                           T_KGrids1[kloop], T_KGrids2[kloop], T_KGrids3[kloop]);
                    for (i1 = 1; i1 <= n; i1++) {
                        if (SpinP_switch == 0)
                            printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n", i1, EIGEN[0][kloop][i1],
                                   EIGEN[0][kloop][i1]);
                        else
                            printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n", i1, EIGEN[0][kloop][i1],
                                   EIGEN[1][kloop][i1]);
                    }
                }

                /**************************************************
              if (all_knum==1), wave functions are calculated.
            **************************************************/

                if (measure_time)
                    dtime(&Stime);

                if (all_knum == 1) {
                    // hipblasmp_zgemm(HIPBLAS_OP_T, HIPBLAS_OP_T, Cs, Ss, Hs, &opts, &opts2);

                    BandCol_ZeroComplex(Hs, (size_t)na_rows * (size_t)na_cols);
                    F77_NAME(pzgemm, PZGEMM)
                    ("T", "T", &n, &n, &n, &alpha, Cs, &ONE, &ONE, descS, Ss, &ONE, &ONE, descC, &beta, Hs, &ONE, &ONE,
                     descH);
                    Cblacs_barrier(ictxt2, "A");
                }
            }

            if (all_knum == 1 && !use_gpusolver_dense) {
                /* MPI communications of Hs and store them to EVec1 */

                for (ID = 0; ID < numprocs2; ID++) {

                    IDS = (myid2 + ID) % numprocs2;
                    IDR = (myid2 - ID + numprocs2) % numprocs2;

                    k = 0;
                    for (i = 0; i < na_rows; i++) {

                        ig = np_rows * nblk * ((i) / nblk) + (i) % nblk + ((np_rows + my_prow) % np_rows) * nblk + 1;
                        if (is2[IDS] <= ig && ig <= ie2[IDS]) {

                            for (j = 0; j < na_cols; j++) {
                                jg = np_cols * nblk * ((j) / nblk) + (j) % nblk +
                                     ((np_cols + my_pcol) % np_cols) * nblk + 1;

                                index_Snd_i[k]      = ig;
                                index_Snd_j[k]      = jg;
                                EVec_Snd[2 * k]     = Hs[j * na_rows + i].r;
                                EVec_Snd[2 * k + 1] = Hs[j * na_rows + i].i;

                                k++;
                            }
                        }
                    }

                    if (ID != 0) {

                        if (Num_Snd_EV[IDS] != 0) {
                            MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
                        }
                        if (Num_Rcv_EV[IDR] != 0) {
                            MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
                        }
                        if (Num_Snd_EV[IDS] != 0) {
                            MPI_Wait(&request, &stat);
                        }

                        if (Num_Snd_EV[IDS] != 0) {
                            MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
                        }
                        if (Num_Rcv_EV[IDR] != 0) {
                            MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
                        }
                        if (Num_Snd_EV[IDS] != 0) {
                            MPI_Wait(&request, &stat);
                        }

                        if (Num_Snd_EV[IDS] != 0) {
                            MPI_Isend(EVec_Snd, Num_Snd_EV[IDS] * 2, MPI_DOUBLE, IDS, 999, MPI_CommWD2[myworld2],
                                      &request);
                        }
                        if (Num_Rcv_EV[IDR] != 0) {
                            MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR] * 2, MPI_DOUBLE, IDR, 999, MPI_CommWD2[myworld2], &stat);
                        }
                        if (Num_Snd_EV[IDS] != 0) {
                            MPI_Wait(&request, &stat);
                        }

                    } else {
                        for (k = 0; k < Num_Snd_EV[IDS]; k++) {
                            index_Rcv_i[k] = index_Snd_i[k];
                            index_Rcv_j[k] = index_Snd_j[k];

                            EVec_Rcv[2 * k]     = EVec_Snd[2 * k];
                            EVec_Rcv[2 * k + 1] = EVec_Snd[2 * k + 1];
                        }
                    }

                    for (k = 0; k < Num_Rcv_EV[IDR]; k++) {

                        ig = index_Rcv_i[k];
                        jg = index_Rcv_j[k];
                        m  = (jg - 1) * (ie2[myid2] - is2[myid2] + 1) + ig - is2[myid2];

                        EVec1[spin][m].r = EVec_Rcv[2 * k];
                        EVec1[spin][m].i = EVec_Rcv[2 * k + 1];
                    }

                } /* ID */
            }

            if (measure_time) {
                dtime(&Etime);
                time5 += Etime - Stime;
                if (scf_eigen_lib_flag == ELPA2 || (use_gpusolver_dense && owns_global_dense_rank)) {
                    part2_5 += Etime - Stime;
                    if (SCF_iter != 1) {
                        part2_5sum += part2_5;
                    }
                }
            }
        } /* kloop0 */

        // if (measure_time == 1) {
        //     MPI_Allreduce(MPI_IN_PLACE, &time2, 1, MPI_DOUBLE, MPI_MAX, MPI_CommWD2[myworld2]);
        //     MPI_Allreduce(MPI_IN_PLACE, &time3, 1, MPI_DOUBLE, MPI_MAX, MPI_CommWD2[myworld2]);
        //     MPI_Allreduce(MPI_IN_PLACE, &time4, 1, MPI_DOUBLE, MPI_MAX, MPI_CommWD2[myworld2]);
        //     MPI_Allreduce(MPI_IN_PLACE, &time5, 1, MPI_DOUBLE, MPI_MAX, MPI_CommWD2[myworld2]);
        // }
    }

    if (measure_time)
        dtime(&EiloopTime);

    if (SpinP_switch == 1 && numprocs0 == 1 && spin == 0) {
        spin++;
        goto diagonalize1;
    }

    /****************************************************
      the refined eigensolver of the one-k-point-per-rank
      path, on every rank: the estimates (or pass 1's
      refined eigenvalues) of all k-points give a chemical
      potential, with which the owners finish their solves
      (every owner resident) or solve pass 2 in the same
      serialized turn groups (staged)
    ****************************************************/

    if (all_knum == 1 && (BandCol_refine.staged || BandCol_refine.last_refined)) {
        const int serialize_gpu_turns = BandCol_SerializeGpuSolverGpuTurns();
        const int total_gpu_turns = Num_Comm_World1 * T_knum;
        const int gpu_turn_group = serialize_gpu_turns ? BandCol_GpuTurnGroupForScf(SCF_iter) : total_gpu_turns;
        const int my_spin = myworld1, my_kloop = S_knum;
        const int my_gpu_turn = my_spin * T_knum + my_kloop;
        double    refine_mu;

        for (int spin_b = 0; spin_b <= SpinP_switch; spin_b++) {
            for (int kloop_b = 0; kloop_b < T_knum; kloop_b++) {
                ID = Comm_World_StartID1[spin_b] + T_k_ID[spin_b][kloop_b];
                MPI_Bcast(&EIGEN[spin_b][kloop_b][0], MaxN + 1, MPI_DOUBLE, ID, mpi_comm_level1);
            }
        }
        refine_mu = BandCol_RefineChemP(EIGEN, T_knum, MaxN, SpinP_switch, T_k_op, sum_weights, TZ);

        if (BandCol_refine.staged) {
            static int announced = 0;

            if (!announced && myid0 == Host_ID && 2 <= level_stdout) {
                printf("<Band_DFT_Col> refined eigensolver in serialized GPU turns (%d owners in groups of %d): pass 1 "
                       "with the previous chemical potential, pass 2 from its basis\n", total_gpu_turns,
                       gpu_turn_group);
                fflush(stdout);
                announced = 1;
            }
            for (int gpu_turn = 0; gpu_turn < total_gpu_turns; gpu_turn++) {
                if (gpu_turn % gpu_turn_group == 0) {
                    MPI_Barrier(mpi_comm_level1);
                }
                if (owns_global_dense_rank && my_gpu_turn == gpu_turn) {
                    dcomplex *evec_device;
                    const int persistent_s = BandCol_GpuPersistentDecide();
                    const int device_s_ready = persistent_s && BandCol_GpuSolver_HasDeviceTransformedS(n);
                    const int host_s_ready = BandCol_GpuSolver_HasHostTransformedS(n);
                    const int build_s = (!device_s_ready && !host_s_ready);
                    int       construct_on_device;

                    (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,my_kloop);

                    k1 = T_KGrids1[my_kloop];
                    k2 = T_KGrids2[my_kloop];
                    k3 = T_KGrids3[my_kloop];
                    Construct_Band_CsHs(build_s ? 1 : SCF_iter, all_knum,
                                        use_setham_packed_cache ? setham_order_GA : order_GA, MP,
                                        use_setham_packed_cache ? setham_S1 : S1,
                                        use_setham_packed_cache ? setham_H1 : H1, k1, k2, k3, Ss, Hs, n,
                                        use_gpusolver_dense, owns_global_dense_rank);
                    construct_on_device = BandCol_LastConstructOnDevice();
                    if (build_s) {
                        BandCol_GpuSolver_PrepareTransformedS(1, n, construct_on_device ? NULL : Ss, ko, koS);
                        if (!persistent_s) BandCol_GpuSolver_SaveTransformedS(n);
                    } else if (!device_s_ready) {
                        BandCol_GpuSolver_LoadTransformedS(n);
                    }
                    if (BandCol_kref != NULL && BandCol_kref[my_gpu_turn].refined) {
                        evec_device = BandCol_KDenseRefinedPass2(n, MaxN, construct_on_device ? NULL : Hs, ko,
                                                                 my_gpu_turn, refine_mu);
                    } else if (construct_on_device) {
                        evec_device = BandCol_GpuSolver_SolveHamiltonianDeviceInput(n, MaxN, ko);
                    } else {
                        evec_device = BandCol_GpuSolver_SolveHamiltonianDeviceOnly(n, MaxN, Hs, ko);
                    }
                    BandCol_AfterDenseSolve(evec_device, n);
                    BandCol_RefineDropBasis();
                    for (l = 1; l <= MaxN; l++) {
                        EIGEN[my_spin][my_kloop][l] = ko[l];
                    }
                }
            }
            MPI_Barrier(mpi_comm_level1);
        }
        else {
            if (owns_global_dense_rank && BandCol_refine.pending) {
                dcomplex *evec_device = BandCol_RefinedSecondHalf(n, MaxN, ko, refine_mu);

                BandCol_AfterDenseSolve(evec_device, n);
                /* the basis stays only for a warm start of the next step */
                if (!openmx_eigen_refine_warm_band_enabled()) BandCol_RefineDropBasis();
                for (l = 1; l <= MaxN; l++) {
                    EIGEN[my_spin][my_kloop][l] = ko[l];
                }
            }
        }
        BandCol_RefineEndStep();
    }

    /****************************************************
       MPI:

       EIGEN
    ****************************************************/

    if (measure_time) {
        MPI_Barrier(mpi_comm_level1);
        dtime(&Stime);
    }

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (kloop = 0; kloop < T_knum; kloop++) {

            /* get ID in the zeroth world */
            ID = Comm_World_StartID1[spin] + T_k_ID[spin][kloop];
            MPI_Bcast(&EIGEN[spin][kloop][0], MaxN + 1, MPI_DOUBLE, ID, mpi_comm_level1);
        }
    }

    if (measure_time) {
        dtime(&Etime);
        time6 += Etime - Stime;
    }

    if (measure_time) {
        dtime(&Stime);
    }

    /**************************************
           find chemical potential
    **************************************/

    /* for XANES */
    if (xanes_calc == 1) {

        for (spin = 0; spin <= SpinP_switch; spin++) {

            po        = 0;
            loop_num  = 0;
            ChemP_MAX = 20.0;
            ChemP_MIN = -20.0;

            do {

                loop_num++;

                ChemP     = 0.50 * (ChemP_MAX + ChemP_MIN);
                Num_State = 0.0;

                for (kloop = 0; kloop < T_knum; kloop++) {

                    for (l = 1; l <= MaxN; l++) {

                        x = (EIGEN[spin][kloop][l] - ChemP) * Beta;

                        if (x <= -x_cut)
                            x = -x_cut;
                        if (x_cut <= x)
                            x = x_cut;
                        FermiF = FermiFunc(x, spin, l, &l, &x);
                        Num_State += FermiF * (double)T_k_op[kloop];
                    }
                }

                if (SpinP_switch == 0)
                    Num_State = 2.0 * Num_State / sum_weights;
                else
                    Num_State = Num_State / sum_weights;
                Dnum = HOMO_XANES[spin] - Num_State;

                if (0.0 <= Dnum)
                    ChemP_MIN = ChemP;
                else
                    ChemP_MAX = ChemP;
                if (fabs(Dnum) < 1.0e-12)
                    po = 1;

            } while (po == 0 && loop_num < 1000);

            ChemP_XANES[spin]  = ChemP;
            Cluster_HOMO[spin] = HOMO_XANES[spin];

        } /* spin loop */

        /* set ChemP */

        if (SpinP_switch == 0) {
            ChemP = ChemP_XANES[0];
        } else {
            ChemP = 0.5 * (ChemP_XANES[0] + ChemP_XANES[1]);
        }

    } /* end of if (xanes_calc==1) */

    /* start of else for if (xanes_calc==1) */

    else {

        if (measure_time)
            dtime(&Stime);

        double Beta_trial1;

        /* first, find ChemP at 1200 K */

        Beta_trial1 = 1.0 / kB / (1200.0 / eV2Hartree);

        po        = 0;
        loop_num  = 0;
        ChemP_MAX = 20.0;
        ChemP_MIN = -20.0;

        do {

            loop_num++;

            ChemP     = 0.50 * (ChemP_MAX + ChemP_MIN);
            Num_State = 0.0;

            for (kloop = 0; kloop < T_knum; kloop++) {
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (l = 1; l <= MaxN; l++) {

                        x = (EIGEN[spin][kloop][l] - ChemP) * Beta_trial1;

                        if (x <= -x_cut)
                            x = -x_cut;
                        if (x_cut <= x)
                            x = x_cut;
                        FermiF = FermiFunc(x, spin, l, &l, &x);

                        Num_State += FermiF * (double)T_k_op[kloop];
                    }
                }
            }

            if (SpinP_switch == 0)
                Num_State = 2.0 * Num_State / sum_weights;
            else
                Num_State = Num_State / sum_weights;

            Dnum = TZ - Num_State - system_charge;

            if (0.0 <= Dnum)
                ChemP_MIN = ChemP;
            else
                ChemP_MAX = ChemP;
            if (fabs(Dnum) < 1.0e-12)
                po = 1;

        } while (po == 0 && loop_num < 1000);

        /* second, find ChemP at the temperatue, starting from the previously found ChemP. */

        po        = 0;
        loop_num  = 0;
        ChemP_MAX = 20.0;
        ChemP_MIN = -20.0;

        do {

            loop_num++;

            if (loop_num != 1) {
                ChemP = 0.50 * (ChemP_MAX + ChemP_MIN);
            }

            Num_State = 0.0;

            for (kloop = 0; kloop < T_knum; kloop++) {
                for (spin = 0; spin <= SpinP_switch; spin++) {
                    for (l = 1; l <= MaxN; l++) {

                        x = (EIGEN[spin][kloop][l] - ChemP) * Beta;

                        if (x <= -x_cut)
                            x = -x_cut;
                        if (x_cut <= x)
                            x = x_cut;
                        FermiF = FermiFunc(x, spin, l, &l, &x);

                        Num_State += FermiF * (double)T_k_op[kloop];
                    }
                }
            }

            if (SpinP_switch == 0)
                Num_State = 2.0 * Num_State / sum_weights;
            else
                Num_State = Num_State / sum_weights;

            Dnum = TZ - Num_State - system_charge;

            if (0.0 <= Dnum)
                ChemP_MIN = ChemP;
            else
                ChemP_MAX = ChemP;
            if (fabs(Dnum) < 1.0e-12)
                po = 1;
        } while (po == 0 && loop_num < 1000);

    } /* end of else for if (xanes_calc==1) */

    /* for the NEGF calculation */
    if (Solver == 4 && TRAN_ChemP_Band == 0)
        ChemP = 0.5 * (ChemP_e[0] + ChemP_e[1]);

    /****************************************************
             band energy in a finite temperature
    ****************************************************/

    Eele0[0] = 0.0;
    Eele0[1] = 0.0;

    for (kloop = 0; kloop < T_knum; kloop++) {
        for (spin = 0; spin <= SpinP_switch; spin++) {
            for (l = 1; l <= MaxN; l++) {

                if (xanes_calc == 1)
                    x = (EIGEN[spin][kloop][l] - ChemP_XANES[spin]) * Beta;
                else
                    x = (EIGEN[spin][kloop][l] - ChemP) * Beta;

                if (x <= -x_cut)
                    x = -x_cut;
                if (x_cut <= x)
                    x = x_cut;
                FermiF = FermiFunc(x, spin, l, &l, &x);

                Eele0[spin] += FermiF * EIGEN[spin][kloop][l] * (double)T_k_op[kloop];
            }
        }
    }

    if (SpinP_switch == 0) {
        Eele0[0] = Eele0[0] / sum_weights;
        Eele0[1] = Eele0[0];
    } else {
        Eele0[0] = Eele0[0] / sum_weights;
        Eele0[1] = Eele0[1] / sum_weights;
    }

    Uele = Eele0[0] + Eele0[1];

    if (2 <= level_stdout) {
        printf("myid0=%2d ChemP=%lf, Eele0[0]=%lf, Eele0[1]=%lf\n", myid0, ChemP, Eele0[0], Eele0[1]);
    }

    if (measure_time) {
        dtime(&Etime);
        time7 += Etime - Stime;
        part3 += Etime - Stime;
        if (SCF_iter != 1) {
            part3sum += part3;
        }
    }

    /****************************************************
           if all_knum==1, calculate CDM and EDM
    ****************************************************/

    if (measure_time)
        dtime(&Stime);

    if (all_knum == 1) {

        if (measure_time)
            dtime(&Stime0);

        /* initialize CDM1 and EDM1 */

        for (i = 0; i < size_H1; i++) {
            CDM1[i] = 0.0;
            EDM1[i] = 0.0;
        }

        /* calculate CDM and EDM */

        spin  = myworld1;
        kloop = S_knum;

        (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);

        k1 = T_KGrids1[kloop];
        k2 = T_KGrids2[kloop];
        k3 = T_KGrids3[kloop];

        /* weight of k-point */

        kw = (double)T_k_op[kloop];

        if (use_gpusolver_dense) {
            if (owns_global_dense_rank) {
                double *occ_weight;

                BandCol_DMWorkspace_Ensure(max_tno, MaxN);
                occ_weight = BandCol_dm_workspace.OccWeight;

                for (k = 0; k < MaxN; k++) {
                    occ_weight[k] = 0.0;
                }

                for (ID = 0; ID < numprocs2; ID++) {
                    for (k = is2[ID]; k <= ie2[ID]; k++) {
                        eig = EIGEN[spin][kloop][k];

                        if (xanes_calc == 1)
                            x = (eig - ChemP_XANES[spin]) * Beta;
                        else
                            x = (eig - ChemP) * Beta;

                        if (x <= -x_cut)
                            x = -x_cut;
                        if (x_cut <= x)
                            x = x_cut;
                        FermiF = FermiFunc(x, spin, k, &k, &x);

                        occ_weight[k - 1] = sqrt(kw * FermiF);

                        if (FermiF < FermiEps) {
                            break;
                        }
                    }
                }
            }

            if (measure_time) {
                dtime(&Etime0);
                time81 += Etime0 - Stime0;
                dtime(&Stime0);
            }

            {
                int my_gpu_turn = spin * T_knum + kloop;
                const int serialize_gpu_turns = BandCol_SerializeGpuSolverGpuTurns();
                const int gpu_turn_group = serialize_gpu_turns ? BandCol_GpuTurnGroupForScf(SCF_iter) : 1;
                const int first_gpu_turn = 0;
                const int last_gpu_turn = Num_Comm_World1 * T_knum;

                for (int gpu_turn = first_gpu_turn; gpu_turn < last_gpu_turn; gpu_turn++) {
                    if (serialize_gpu_turns && gpu_turn % gpu_turn_group == 0) {
                        MPI_Barrier(mpi_comm_level1);
                    }

                    (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,gpu_turn%T_knum);
                    if (owns_global_dense_rank && my_gpu_turn == gpu_turn) {
                        dcomplex *evec_device;

                        if (BandCol_GpuPersistentDecide()) {
                            evec_device = BandCol_GpuSolver_DeviceEigenvectors();
                        } else {
                            evec_device = BandCol_GpuSolver_UploadHostEigenvectors(n);
                        }

                        if (BandCol_AccumulateDenseTransposedDM_Device(n, MaxN, spin, kloop, k1, k2, k3, evec_device,
                                                                       BandCol_GpuSolver_EigenvectorStride(),
                                                                       MP, use_setham_packed_cache ? setham_order_GA : order_GA, EIGEN,
                                                                       BandCol_dm_workspace.OccWeight, CDM1, EDM1,
                                                                       size_H1, 1,mpi_comm_level1,BandCol_mgpu_host_dense) != 0) {
                            BandCol_AbortWithMessage("Band_DFT_Col HIP density matrix generation failed.");
                        }
                        if (!BandCol_GpuPersistentDecide()) {
                            BandCol_GpuSolver_ClearHostEigenvectors();
                            BandCol_GpuSolver_ReleaseDeviceMemory();
                        }
                    } else { BandCol_AccumulateDenseTransposedDM_Device(n,0,spin,0,0,0,0,NULL,0,MP,order_GA,EIGEN,NULL,
                            CDM1,EDM1,size_H1,1,mpi_comm_level1,1); }
                }
                if (!serialize_gpu_turns) {
                    MPI_Barrier(mpi_comm_level1);
                }
            }

            if (measure_time) {
                dtime(&Etime0);
                time82 += Etime0 - Stime0;
                dtime(&Stime0);
            }
        } else {
            /* pre-calculation of the Fermi function */

            po   = 0;
            kmin = is2[myid2];
            kmax = ie2[myid2];

            for (k = is2[myid2]; k <= ie2[myid2]; k++) {

                eig = EIGEN[spin][kloop][k];

                if (xanes_calc == 1)
                    x = (eig - ChemP_XANES[spin]) * Beta;
                else
                    x = (eig - ChemP) * Beta;

                if (x <= -x_cut)
                    x = -x_cut;
                if (x_cut <= x)
                    x = x_cut;
                FermiF = FermiFunc(x, spin, k, &k, &x);

                tmp1 = sqrt(kw * FermiF);

                for (i1 = 1; i1 <= n; i1++) {
                    i = (i1 - 1) * (ie2[myid2] - is2[myid2] + 1) + k - is2[myid2];

                    EVec1[spin][i].r *= tmp1;
                    EVec1[spin][i].i *= tmp1;
                }

                /* find kmax */

                if (FermiF < FermiEps && po == 0) {
                    kmax = k;
                    po   = 1;
                }
            }

            if (measure_time) {
                dtime(&Etime0);
                time81 += Etime0 - Stime0;
                dtime(&Stime0);
            }

            int gpu_dm_done = 0;

            int dm_requested=BandCol_UseGpuFallbackDM(n),dm_collective=dm_requested;
            MPI_Allreduce(MPI_IN_PLACE,&dm_collective,1,MPI_INT,MPI_MAX,mpi_comm_level1);
            if(dm_collective){
                int nk=dm_requested && kmin<=kmax?kmax-kmin+1:0;
                if(nk){BandCol_DMWorkspace_Ensure(max_tno,nk);for(int k=0;k<nk;k++)BandCol_dm_workspace.OccWeight[k]=1;}
                BandCol_AccumulateDenseTransposedDM_Device(n,nk,spin,kloop,k1,k2,k3,EVec1[spin],
                    ie2[myid2]-is2[myid2]+1,MP,order_GA,EIGEN,BandCol_dm_workspace.OccWeight,
                    CDM1,EDM1,size_H1,kmin,mpi_comm_level1,1);
                gpu_dm_done=nk>0;
            }

            if (!gpu_dm_done && kmin <= kmax) {
                const int nk      = kmax - kmin + 1;
                double * TmpEIGEN_local;
                double **ReEVec0_local;
                double **ImEVec0_local;
                double **ReEVec1_local;
                double **ImEVec1_local;

                BandCol_DMWorkspace_Ensure(max_tno, nk);

                TmpEIGEN_local = BandCol_dm_workspace.TmpEIGEN;
                ReEVec0_local  = BandCol_dm_workspace.ReEVec0;
                ImEVec0_local  = BandCol_dm_workspace.ImEVec0;
                ReEVec1_local  = BandCol_dm_workspace.ReEVec1;
                ImEVec1_local  = BandCol_dm_workspace.ImEVec1;

                BandCol_DMWorkspace_SetPointers(&BandCol_dm_workspace, max_tno, nk);

                {
                    const double * src = &EIGEN[spin][kloop][kmin];
                    for (int k = 0; k < nk; ++k)
                        TmpEIGEN_local[k] = src[k];
                }

                const int stride = ie2[myid2] - is2[myid2] + 1;
                size_t    p      = 0;

                for (int GA_AN = 1; GA_AN <= atomnum; ++GA_AN) {
                    const int wanA = WhatSpecies[GA_AN];
                    const int tnoA = Spe_Total_CNO[wanA];
                    const int Anum = MP[GA_AN];

                    for (int i = 0; i < tnoA; ++i) {
                        const size_t     base = (size_t)(Anum + i - 1) * stride - is2[myid2] + kmin;
                        const dcomplex * v    = &EVec1[spin][base];
                        double * restrict r   = ReEVec0_local[i];
                        double * restrict im  = ImEVec0_local[i];
                        for (int k = 0; k < nk; ++k) {
                            r[k]  = v[k].r;
                            im[k] = v[k].i;
                        }
                    }

                    for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; ++LB_AN) {
                        const int GB_AN = natn[GA_AN][LB_AN];
                        const int Rn    = ncn[GA_AN][LB_AN];
                        const int wanB  = WhatSpecies[GB_AN];
                        const int tnoB  = Spe_Total_CNO[wanB];
                        const int Bnum  = MP[GB_AN];

                        const int    l1  = atv_ijk[Rn][1];
                        const int    l2  = atv_ijk[Rn][2];
                        const int    l3  = atv_ijk[Rn][3];
                        const double kRn = k1 * l1 + k2 * l2 + k3 * l3;
                        const double si  = sin(2.0 * PI * kRn);
                        const double co  = cos(2.0 * PI * kRn);

                        for (int j = 0; j < tnoB; ++j) {
                            const size_t     base = (size_t)(Bnum + j - 1) * stride - is2[myid2] + kmin;
                            const dcomplex * v    = &EVec1[spin][base];
                            double * restrict r   = ReEVec1_local[j];
                            double * restrict im  = ImEVec1_local[j];
                            for (int k = 0; k < nk; ++k) {
                                r[k]  = v[k].r;
                                im[k] = v[k].i;
                            }
                        }

                        for (int i = 0; i < tnoA; ++i) {
                            const double * restrict r0  = ReEVec0_local[i];
                            const double * restrict im0 = ImEVec0_local[i];

                            for (int j = 0; j < tnoB; ++j, ++p) {
                                const double * restrict r1  = ReEVec1_local[j];
                                const double * restrict im1 = ImEVec1_local[j];

                                double d1 = 0.0, d2 = 0.0, d3 = 0.0, d4 = 0.0;
                                int    k = 0;

                                for (; k + 3 < nk; k += 4) {
#define KSTEP(idx)                                                                                                     \
    do {                                                                                                               \
        double reA = r0[idx] * r1[idx] + im0[idx] * im1[idx];                                                          \
        double imA = r0[idx] * im1[idx] - im0[idx] * r1[idx];                                                          \
        d1 += reA;                                                                                                     \
        d2 += imA;                                                                                                     \
        d3 += reA * TmpEIGEN_local[idx];                                                                               \
        d4 += imA * TmpEIGEN_local[idx];                                                                               \
    } while (0)
                                    KSTEP(k);
                                    KSTEP(k + 1);
                                    KSTEP(k + 2);
                                    KSTEP(k + 3);
#undef KSTEP
                                }
                                for (; k < nk; ++k) {
                                    double reA = r0[k] * r1[k] + im0[k] * im1[k];
                                    double imA = r0[k] * im1[k] - im0[k] * r1[k];
                                    d1 += reA;
                                    d2 += imA;
                                    d3 += reA * TmpEIGEN_local[k];
                                    d4 += imA * TmpEIGEN_local[k];
                                }

                                CDM1[p] += co * d1 - si * d2;
                                EDM1[p] += co * d3 - si * d4;
                            }
                        }
                    } /* LB_AN */
                } /* GA_AN */
            }

            if (measure_time) {
                dtime(&Etime0);
                time82 += Etime0 - Stime0;
                dtime(&Stime0);
            }
        }

        /* sum of CDM1 and EDM1 by Allreduce in MPI */

        /* CDM1 and EDM1 are no longer needed in their rank-local form.  Reduce
           directly into them so the large matrix-element buffers are not
           copied through H1 after every SCF. */
        MPI_Allreduce(MPI_IN_PLACE, CDM1, size_H1, MPI_DOUBLE, MPI_SUM,
                      MPI_CommWD1[myworld1]);
        MPI_Allreduce(MPI_IN_PLACE, EDM1, size_H1, MPI_DOUBLE, MPI_SUM,
                      MPI_CommWD1[myworld1]);

        if (measure_time) {
            dtime(&Etime0);
            time83 += Etime0 - Stime0;
            dtime(&Stime0);
        }

        /* store DM1 to a proper place in CDM and EDM */

        p = 0;
        for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {

            MA_AN = F_G2M[GA_AN];
            wanA  = WhatSpecies[GA_AN];
            tnoA  = Spe_Total_CNO[wanA];
            Anum  = MP[GA_AN];
            ID    = G2ID[GA_AN];

            for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                GB_AN = natn[GA_AN][LB_AN];
                wanB  = WhatSpecies[GB_AN];
                tnoB  = Spe_Total_CNO[wanB];
                Bnum  = MP[GB_AN];

                if (myid0 == ID) {

                    for (i = 0; i < tnoA; i++) {
                        for (j = 0; j < tnoB; j++) {

                            CDM[spin][MA_AN][LB_AN][i][j] = CDM1[p];
                            EDM[spin][MA_AN][LB_AN][i][j] = EDM1[p];

                            /* increment of p */
                            p++;
                        }
                    }
                } else {
                    for (i = 0; i < tnoA; i++) {
                        for (j = 0; j < tnoB; j++) {
                            /* increment of p */
                            p++;
                        }
                    }
                }

            } /* LB_AN */
        } /* GA_AN */

        if (measure_time) {
            dtime(&Etime0);
            time84 += Etime0 - Stime0;
            dtime(&Stime0);
        }

        /* if necessary, MPI communication of CDM and EDM */

        if (1 < numprocs0 && SpinP_switch == 1) {

            /* set spin */

            if (myworld1 == 0) {
                spin = 1;
            } else {
                spin = 0;
            }

            /* communicate CDM1 and EDM1 */

            for (i = 0; i <= 1; i++) {

                IDS = Comm_World_StartID1[i % 2];
                IDR = Comm_World_StartID1[(i + 1) % 2];

                if (myid0 == IDS) {
                    MPI_Isend(&CDM1[0], size_H1, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
                }

                if (myid0 == IDR) {
                    MPI_Recv(&H1[0], size_H1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
                }

                if (myid0 == IDS) {
                    MPI_Wait(&request, &stat);
                }

                if (myid0 == IDS) {
                    MPI_Isend(&EDM1[0], size_H1, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
                }

                if (myid0 == IDR) {
                    MPI_Recv(&S1[0], size_H1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
                }

                if (myid0 == IDS) {
                    MPI_Wait(&request, &stat);
                }
            }

            MPI_Bcast(&H1[0], size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);
            MPI_Bcast(&S1[0], size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);

            /* put CDM1 and EDM1 into CDM and EDM */

            k = 0;
            for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {

                MA_AN = F_G2M[GA_AN];
                wanA  = WhatSpecies[GA_AN];
                tnoA  = Spe_Total_CNO[wanA];

                for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {

                    GB_AN = natn[GA_AN][LB_AN];
                    wanB  = WhatSpecies[GB_AN];
                    tnoB  = Spe_Total_CNO[wanB];

                    for (i = 0; i < tnoA; i++) {
                        for (j = 0; j < tnoB; j++) {

                            if (1 <= MA_AN && MA_AN <= Matomnum) {
                                CDM[spin][MA_AN][LB_AN][i][j] = H1[k];
                                EDM[spin][MA_AN][LB_AN][i][j] = S1[k];
                            }

                            k++;
                        }
                    }
                }
            }
        }

        if (measure_time) {
            dtime(&Etime0);
            time85 += Etime0 - Stime0;
        }

    } /* if (all_knum==1) */

    if (measure_time) {
        dtime(&Etime);
        time8 += Etime - Stime;
        part4 += Etime - Stime;
        if (SCF_iter != 1) {
            part4sum += part4;
        }
    }

    dtime(&EiloopTime);

    if (myid0 == Host_ID && 0 < level_stdout) {
        printf("<Band_DFT>  Eigen, time=%lf\n", EiloopTime - SiloopTime);
        if (SCF_iter == 1 && BandCol_CacheTraceEnabled()) {
            printf("<Band_DFT>  timing scope: Eigen includes occupation/ChemP%s.\n",
                   (all_knum == 1) ? " and density-matrix collectives" : "");
        }
        fflush(stdout);
    }

    /* Debug hook: exit cleanly after a requested SCF eigen phase so profilers
       flush without running the (irrelevant here) force calculation.  The old
       OPENMX_ABORT_AFTER_EIGEN=1 remains an alias for SCF 1. */
    {
        const int abort_after_scf = BandCol_AbortAfterEigenScf();
        if (0 < abort_after_scf && abort_after_scf <= SCF_iter) {
            fflush(stdout);
            fflush(stderr);
            MPI_Barrier(mpi_comm_level1);
            MPI_Finalize();
            exit(0);
        }
    }

    dtime(&SiloopTime);

    /****************************************************
     ****************************************************
       diagonalization for calculating density matrix
     ****************************************************
    ****************************************************/

    if (all_knum != 1) {
        /* spin=myworld1 */

        spin = myworld1;

    diagonalize2:

        /* set S1 */

        size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

        /* set H1 */

        if (SpinP_switch == 0) {
            size_H1 = Get_OneD_HS_Col(1, nh[0], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
        } else if (1 < numprocs0) {
            size_H1 = Get_OneD_HS_Col(1, nh[0], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
            size_H1 = Get_OneD_HS_Col(1, nh[1], CDM1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

            if (myworld1) {
                for (i = 0; i < size_H1; i++) {
                    H1[i] = CDM1[i];
                }
            }
        } else {
            size_H1 = Get_OneD_HS_Col(1, nh[spin], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
        }

        /* initialize CDM1 and EDM1 */

        for (i = 0; i < size_H1; i++) {
            CDM1[i] = 0.0;
            EDM1[i] = 0.0;
        }

        /* initialize CDM, EDM, and iDM */

        for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
            GA_AN = M2G[MA_AN];
            wanA  = WhatSpecies[GA_AN];
            tnoA  = Spe_Total_CNO[wanA];
            Anum  = MP[GA_AN];
            for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                GB_AN = natn[GA_AN][LB_AN];
                wanB  = WhatSpecies[GB_AN];
                tnoB  = Spe_Total_CNO[wanB];
                Bnum  = MP[GB_AN];

                for (i = 0; i < tnoA; i++) {
                    for (j = 0; j < tnoB; j++) {
                        CDM[spin][MA_AN][LB_AN][i][j] = 0.0;
                        EDM[spin][MA_AN][LB_AN][i][j] = 0.0;

                        iDM[0][0][MA_AN][LB_AN][i][j] = 0.0;
                        iDM[0][1][MA_AN][LB_AN][i][j] = 0.0;
                    }
                }
            }
        }

        /* for kloop */

        if (use_gpusolver_dense) {
            const int max_concurrent_gpu_turns = BandCol_MaxConcurrentKGpuTurns(n);
            const int total_gpu_turns = Num_Comm_World1 * T_knum;

            for (int group_first = 0; group_first < total_gpu_turns; group_first += max_concurrent_gpu_turns) {
                const int group_last =
                    (group_first + max_concurrent_gpu_turns < total_gpu_turns) ?
                    (group_first + max_concurrent_gpu_turns) : total_gpu_turns;

                MPI_Barrier(mpi_comm_level1);

                for (int gpu_turn = group_first; gpu_turn < group_last; gpu_turn++) {
                    int kloop;
                    dcomplex *evec_device;
                    int construct_on_device;
                    double k1;
                    double k2;
                    double k3;

                    kloop = gpu_k_order[gpu_turn] % T_knum;
                    (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);
                    if (!owns_dense_k_rank || gpu_k_order[gpu_turn] / T_knum != spin ||
                        kloop < S_knum || S_knum + num_kloop0 <= kloop) {
                        BandCol_AccumulateDenseTransposedDM_Device(n,0,spin,0,0,0,0,NULL,0,MP,order_GA,EIGEN,NULL,
                            CDM1,EDM1,size_H1,1,mpi_comm_level1,1);
                        continue;
                    }

                    (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);

                    k1 = T_KGrids1[kloop];
                    k2 = T_KGrids2[kloop];
                    k3 = T_KGrids3[kloop];

                    size_t cache_slot = (size_t)spin * T_knum + kloop;
                    if (k_evec_cache != NULL && k_evec_cache[cache_slot] != NULL) {
                        BandCol_GpuSolver_Init();
                        evec_device = BandCol_GpuSolver_EnsureEigenvectorCapacity(n, MaxN);
                        if(BandCol_mgpu_host_dense)memcpy(evec_device,k_evec_cache[cache_slot],k_cache_bytes);
                        else wait_hipfunc(hipMemcpy(evec_device,k_evec_cache[cache_slot],k_cache_bytes,hipMemcpyHostToDevice));
                        free(k_evec_cache[cache_slot]);
                        k_evec_cache[cache_slot] = NULL;
                        memcpy(ko + 1, EIGEN[spin][kloop] + 1, sizeof(double) * (size_t)MaxN);
                        ++k_cache_hits;
                    } else {
                        /* make S and H */

                        Construct_Band_CsHs(SCF_iter, all_knum, use_setham_packed_cache ? setham_order_GA : order_GA, MP,
                                            use_setham_packed_cache ? setham_S1 : S1,
                                            use_setham_packed_cache ? setham_H1 : H1, k1, k2, k3, Ss, Hs, n,
                                            use_gpusolver_dense, owns_global_dense_rank);
                        construct_on_device = BandCol_LastConstructOnDevice();

                        /* diagonalize S */

                        if (measure_time)
                            dtime(&Stime);

                        if (BandCol_KOverlapRestore(kloop, k1, k2, k3, ko, koS)) {
                            ++k_overlap_hits;
                        } else {
                            BandCol_GpuSolver_PrepareTransformedS(1, n, construct_on_device ? NULL : Ss, ko, koS);
                            BandCol_KOverlapSave(kloop, k1, k2, k3, koS);
                        }

                        if (measure_time) {
                            dtime(&Etime);
                            time9 += Etime - Stime;
                        }

                        /* a cache hit restored the overlap eigenvalues too */
                        if (3 <= level_stdout) {
                            printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n", myid0, kloop, T_KGrids1[kloop],
                                   T_KGrids2[kloop], T_KGrids3[kloop]);
                            for (i1 = 1; i1 <= n; i1++)
                                printf("  Eigenvalues of OLP  %2d  %15.12f\n", i1, ko[i1]);
                        }

                        /****************************************************
                              1/sqrt(ko) * U^t * H * U * 1/sqrt(ko)
                        ****************************************************/

                        if (n != na_rows_max || n != na_cols_max) {
                            BandCol_AbortWithMessage("GPUSOLVER DM path requires full dense matrices in Band_DFT_Col.c.");
                        }

                        if (measure_time)
                            dtime(&Stime);

                        if (BandCol_refine.kdense && BandCol_kref != NULL && BandCol_kref[cache_slot].refined) {
                            /* pass 2 of a k-point refined in pass 1: from its
                               basis, with the step's chemical potential (the
                               density matrix weights stay those of EIGEN, the
                               eigenvalues of pass 1) */
                            evec_device = BandCol_KDenseRefinedPass2(n, MaxN, construct_on_device ? NULL : Hs, ko,
                                                                     (int)cache_slot, ChemP);
                        } else if (construct_on_device) {
                            evec_device = BandCol_GpuSolver_SolveHamiltonianDeviceInput(n, MaxN, ko);
                        } else {
                            evec_device = BandCol_GpuSolver_SolveHamiltonianDeviceOnly(n, MaxN, Hs, ko);
                        }

                        if (measure_time) {
                            dtime(&Etime);
                            time10 += Etime - Stime;
                        }

                    }

                    if (3 <= level_stdout && 0 <= kloop) {
                        printf("  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n", kloop, T_KGrids1[kloop], T_KGrids2[kloop],
                               T_KGrids3[kloop]);
                        for (i1 = 1; i1 <= n; i1++) {
                            printf("  Eigenvalues of Kohn-Sham(DM) spin=%2d i1=%2d %15.12f\n", spin, i1, ko[i1]);
                        }
                    }

                    /****************************************************
                                   calculate DM and EDM
                    ****************************************************/

                    if (measure_time)
                        dtime(&Stime);

                    /* weight of k-point */

                    double kw = (double)T_k_op[kloop];

                    int po   = 0;
                    int kmax = MaxN;

                    BandCol_DMWorkspace_Ensure(max_tno, MaxN);
                    double *occ_weight = BandCol_dm_workspace.OccWeight;

                    if (measure_time)
                        dtime(&Stime1);

                    for (int k = 1; k <= MaxN; k++) {

                        double eig = EIGEN[spin][kloop][k];
                        double x;

                        if (xanes_calc == 1)
                            x = (eig - ChemP_XANES[spin]) * Beta;
                        else
                            x = (eig - ChemP) * Beta;

                        if (x <= -x_cut)
                            x = -x_cut;
                        if (x_cut <= x)
                            x = x_cut;
                        double FermiF = FermiFunc(x, spin, k, &k, &x);

                        occ_weight[k - 1] = sqrt(kw * FermiF);

                        /* find kmax */

                        if (FermiF < FermiEps && po == 0) {
                            kmax = k;
                            po   = 1;
                        }
                    }

                    if (measure_time) {
                        dtime(&Etime1);
                        time11A += Etime1 - Stime1;
                        dtime(&Stime1);
                    }

                    if (BandCol_AccumulateDenseTransposedDM_Device(n, MaxN, spin, kloop, k1, k2, k3, evec_device,
                                                                   BandCol_GpuSolver_EigenvectorStride(),
                                                                   MP, use_setham_packed_cache ? setham_order_GA : order_GA,
                                                                   EIGEN, occ_weight, CDM1, EDM1, size_H1, 1,mpi_comm_level1,BandCol_mgpu_host_dense) != 0) {
                        BandCol_AbortWithMessage("Band_DFT_Col HIP density matrix generation failed.");
                    }
                    BandCol_GpuSolver_ReleaseDeviceMemory();
                    if (BandCol_refine.kdense) {
                        /* the basis waits in the host slot for the next step */
                        BandCol_RefineDropBasis();
                        openmx_gemmul8TrimWorkspaces();
                    }

                    if (measure_time) {
                        dtime(&Etime1);
                        time11B += Etime1 - Stime1;

                        dtime(&Etime);
                        time11 += Etime - Stime;
                    }

                } /* gpu_turn */
            } /* group_first */

        } else {
            int collective_dm=BandCol_UseGpuFallbackDM(n),dm_kloops=num_kloop0;
            MPI_Allreduce(MPI_IN_PLACE,&collective_dm,1,MPI_INT,MPI_MAX,mpi_comm_level1);
            if(collective_dm)MPI_Allreduce(MPI_IN_PLACE,&dm_kloops,1,MPI_INT,MPI_MAX,mpi_comm_level1);
            for (kloop0 = 0; kloop0 < dm_kloops; kloop0++) {
                if(kloop0>=num_kloop0){ BandCol_AccumulateDenseTransposedDM_Device(n,0,spin,0,0,0,0,NULL,0,MP,order_GA,EIGEN,NULL,
                            CDM1,EDM1,size_H1,1,mpi_comm_level1,1); continue; }
                kloop = kloop0 + S_knum;

                (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);

                k1 = T_KGrids1[kloop];
                k2 = T_KGrids2[kloop];
                k3 = T_KGrids3[kloop];

                /* make S and H */

                Construct_Band_CsHs(SCF_iter, all_knum, use_setham_packed_cache ? setham_order_GA : order_GA, MP,
                                    use_setham_packed_cache ? setham_S1 : S1,
                                    use_setham_packed_cache ? setham_H1 : H1, k1, k2, k3, Cs, Hs, n,
                                    use_gpusolver_dense, owns_global_dense_rank);


                /* diagonalize S */

                if (measure_time)
                    dtime(&Stime);

                if (scf_eigen_lib_flag == 1) {
                    F77_NAME(solve_evp_complex, SOLVE_EVP_COMPLEX)
                    (&n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
                } else if (scf_eigen_lib_flag == 2 || scf_eigen_lib_flag == GPUSOLVER) {

#ifndef kcomp
                    int mpiworld;
                    mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
                    F77_NAME(elpa_solve_evp_complex_2stage_double_impl, ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
                    (&n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int,
                     &mpi_comm_cols_int, &mpiworld);
#endif
                }

                if (measure_time) {
                    dtime(&Etime);
                    time9 += Etime - Stime;
                }

                if (3 <= level_stdout) {
                    printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n", myid0, kloop, T_KGrids1[kloop],
                           T_KGrids2[kloop], T_KGrids3[kloop]);
                    for (i1 = 1; i1 <= n; i1++) {
                        printf("  Eigenvalues of OLP  %2d  %15.12f\n", i1, ko[i1]);
                    }
                }

                /* minus eigenvalues to 1.0e-14 */

                for (l = 1; l <= n; l++) {
                    if (ko[l] < 0.0) {
                        ko[l] = 1.0e-10;
                    }
                }

                /* calculate S*1/sqrt(ko) */

                for (l = 1; l <= n; l++) {
                    ko[l] = 1.0 / sqrt(ko[l]);
                }

                /* S * 1.0/sqrt(ko[l])  */

                for (j = 0; j < na_cols; j++) {
                    const int jg = np_cols * nblk * (j / nblk) + (j % nblk) + ((np_cols + my_pcol) % np_cols) * nblk + 1;
                    const double scale = ko[jg];
                    dcomplex * restrict col = &Ss[j * na_rows];

                    for (i = 0; i < na_rows; i++) {
                        col[i].r *= scale;
                        col[i].i *= scale;
                    }
                }

                /****************************************************
                      1/sqrt(ko) * U^t * H * U * 1/sqrt(ko)
                ****************************************************/

                /* pzgemm */

                /* H * U * 1/sqrt(ko) */

                // hipblasmp_zgemm(HIPBLAS_OP_N, HIPBLAS_OP_N, Hs, Ss, Cs, &opts, &opts2);

                BandCol_ZeroComplex(Cs, (size_t)na_rows_max * (size_t)na_cols_max);

                Cblacs_barrier(ictxt2, "A");
                F77_NAME(pzgemm, PZGEMM)
                ("N", "N", &n, &n, &n, &alpha, Hs, &ONE, &ONE, descH, Ss, &ONE, &ONE, descS, &beta, Cs, &ONE, &ONE,
                 descC);

                /* 1/sqrt(ko) * U^+ H * U * 1/sqrt(ko) */

                // hipblasmp_zgemm(HIPBLAS_OP_C, HIPBLAS_OP_N, Ss, Cs, Hs, &opts, &opts2);

                BandCol_ZeroComplex(Hs, (size_t)na_rows * (size_t)na_cols);

                Cblacs_barrier(ictxt2, "C");
                F77_NAME(pzgemm, PZGEMM)
                ("C", "N", &n, &n, &n, &alpha, Ss, &ONE, &ONE, descS, Cs, &ONE, &ONE, descC, &beta, Hs, &ONE, &ONE,
                 descH);

                /* diagonalize H' */

                if (measure_time)
                    dtime(&Stime);

                if (scf_eigen_lib_flag == 1) {
                    F77_NAME(solve_evp_complex, SOLVE_EVP_COMPLEX)
                    (&n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
                } else if (scf_eigen_lib_flag == 2 || scf_eigen_lib_flag == GPUSOLVER) {

#ifndef kcomp
                    int mpiworld;
                    mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
                    F77_NAME(elpa_solve_evp_complex_2stage_double_impl, ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
                    (&n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int,
                     &mpi_comm_cols_int, &mpiworld);
#endif
                }

                if (measure_time) {
                    dtime(&Etime);
                    time10 += Etime - Stime;
                }

                if (3 <= level_stdout && 0 <= kloop) {
                    printf("  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n", kloop, T_KGrids1[kloop], T_KGrids2[kloop],
                           T_KGrids3[kloop]);
                    for (i1 = 1; i1 <= n; i1++) {
                        printf("  Eigenvalues of Kohn-Sham(DM) spin=%2d i1=%2d %15.12f\n", spin, i1, ko[i1]);
                    }
                }

                /****************************************************
                  transformation to the original eigenvectors.
                   NOTE JRCAT-244p and JAIST-2122p
                ****************************************************/

                // hipblasmp_zgemm(HIPBLAS_OP_T, HIPBLAS_OP_T, Cs, Ss, Hs, &opts, &opts2);

                BandCol_ZeroComplex(Hs, (size_t)na_rows * (size_t)na_cols);
                F77_NAME(pzgemm, PZGEMM)
                ("T", "T", &MaxN, &n, &n, &alpha, Cs, &ONE, &ONE, descS, Ss, &ONE, &ONE, descC, &beta, Hs, &ONE,
                 &ONE, descH);
                Cblacs_barrier(ictxt2, "A");
                if (na_rows < MaxN || na_cols < n) {
                    BandCol_AbortWithMessage("CPU all_knum!=1 DM path requires local dense eigenvectors in Band_DFT_Col.c.");
                }

                /****************************************************
                               calculate DM and EDM
                ****************************************************/

                if (measure_time)
                    dtime(&Stime);

                /* weight of k-point */

                kw = (double)T_k_op[kloop];

                po   = 0;
                kmax = MaxN;

                BandCol_DMWorkspace_Ensure(max_tno, MaxN);
                double *occ_weight = BandCol_dm_workspace.OccWeight;

                if (measure_time)
                    dtime(&Stime1);

                for (k = 1; k <= MaxN; k++) {

                    eig = EIGEN[spin][kloop][k];

                    if (xanes_calc == 1)
                        x = (eig - ChemP_XANES[spin]) * Beta;
                    else
                        x = (eig - ChemP) * Beta;

                    if (x <= -x_cut)
                        x = -x_cut;
                    if (x_cut <= x)
                        x = x_cut;
                    FermiF = FermiFunc(x, spin, k, &k, &x);

                    occ_weight[k - 1] = sqrt(kw * FermiF);

                    /* find kmax */

                    if (FermiF < FermiEps && po == 0) {
                        kmax = k;
                        po   = 1;
                    }
                }

                if (measure_time) {
                    dtime(&Etime1);
                    time11A += Etime1 - Stime1;
                    dtime(&Stime1);
                }

                {
                    int gpu_dm_done = 0;

                    if(collective_dm){
                        int active=BandCol_UseGpuFallbackDM(n);
                        BandCol_AccumulateDenseTransposedDM_Device(n,active?MaxN:0,spin,kloop,k1,k2,k3,Hs,
                            na_rows,MP,order_GA,EIGEN,occ_weight,CDM1,EDM1,size_H1,1,mpi_comm_level1,1);
                        gpu_dm_done=active;
                    }
                    if (!gpu_dm_done) {
                        BandCol_AccumulateDenseTransposedDM(n, MaxN, max_tno, spin, kloop, k1, k2, k3, Hs, na_rows,
                                                            MP, EIGEN, occ_weight, CDM1, EDM1);
                    }
                }

                if (measure_time) {
                    dtime(&Etime1);
                    time11B += Etime1 - Stime1;

                    dtime(&Etime);
                    time11 += Etime - Stime;
                }
            } /* kloop0 */
        }

        /*******************************************************
             sum of CDM1 and EDM1 by Allreduce in MPI
        *******************************************************/

        if (measure_time)
            dtime(&Stime);

        MPI_Allreduce(&CDM1[0], &H1[0], size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
        MPI_Allreduce(&EDM1[0], &S1[0], size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);

        /* CDM and EDM */

        k = 0;
        for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {

            MA_AN = F_G2M[GA_AN];
            wanA  = WhatSpecies[GA_AN];
            tnoA  = Spe_Total_CNO[wanA];
            ID    = G2ID[GA_AN];

            for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {

                GB_AN = natn[GA_AN][LB_AN];
                wanB  = WhatSpecies[GB_AN];
                tnoB  = Spe_Total_CNO[wanB];

                if (myid0 == ID) {

                    for (i = 0; i < tnoA; i++) {
                        for (j = 0; j < tnoB; j++) {
                            CDM[spin][MA_AN][LB_AN][i][j] = H1[k];
                            EDM[spin][MA_AN][LB_AN][i][j] = S1[k];
                            k++;
                        }
                    }
                }

                else {
                    for (i = 0; i < tnoA; i++) {
                        for (j = 0; j < tnoB; j++) {
                            k++;
                        }
                    }
                }
            }
        }

        if (measure_time) {
            dtime(&Etime);
            time12 += Etime - Stime;
        }

        if (SpinP_switch == 1 && numprocs0 == 1 && spin == 0) {
            spin++;
            goto diagonalize2;
        }

        /* if necessary, MPI communication of CDM and EDM */

        if (1 < numprocs0 && SpinP_switch == 1) {

            /* set spin */

            if (myworld1 == 0) {
                spin = 1;
            } else {
                spin = 0;
            }

            /* communicate CDM1 and EDM1 */

            for (i = 0; i <= 1; i++) {

                IDS = Comm_World_StartID1[i % 2];
                IDR = Comm_World_StartID1[(i + 1) % 2];

                if (myid0 == IDS) {
                    MPI_Isend(&H1[0], size_H1, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
                }

                if (myid0 == IDR) {
                    MPI_Recv(&CDM1[0], size_H1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
                }

                if (myid0 == IDS) {
                    MPI_Wait(&request, &stat);
                }

                if (myid0 == IDS) {
                    MPI_Isend(&S1[0], size_H1, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
                }

                if (myid0 == IDR) {
                    MPI_Recv(&EDM1[0], size_H1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
                }

                if (myid0 == IDS) {
                    MPI_Wait(&request, &stat);
                }
            }

            MPI_Bcast(&CDM1[0], size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);
            MPI_Bcast(&EDM1[0], size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);

            /* put CDM1 and EDM1 into CDM and EDM */

            k = 0;
            for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {

                MA_AN = F_G2M[GA_AN];
                wanA  = WhatSpecies[GA_AN];
                tnoA  = Spe_Total_CNO[wanA];

                for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {

                    GB_AN = natn[GA_AN][LB_AN];
                    wanB  = WhatSpecies[GB_AN];
                    tnoB  = Spe_Total_CNO[wanB];

                    for (i = 0; i < tnoA; i++) {
                        for (j = 0; j < tnoB; j++) {

                            if (1 <= MA_AN && MA_AN <= Matomnum) {
                                CDM[spin][MA_AN][LB_AN][i][j] = CDM1[k];
                                EDM[spin][MA_AN][LB_AN][i][j] = EDM1[k];
                            }

                            k++;
                        }
                    }
                }
            }
        }

        /* every rank: the k-dense refined solves of this step decide the
           stage (spin of a single rank is excluded, so this runs once) */
        if (BandCol_refine.kdense) {
            BandCol_RefineEndStep();
        }
    } /* if (all_knum!=1) */

    /****************************************************
             normalization of CDM, EDM, and iDM
    ****************************************************/

    dtime(&EiloopTime);

    if (myid0 == Host_ID && 0 < level_stdout) {
        printf("<Band_DFT>  DM, time=%lf\n", EiloopTime - SiloopTime);
        fflush(stdout);
    }

    dum = 1.0 / sum_weights;

    for (spin = 0; spin <= SpinP_switch; spin++) {
        for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
            GA_AN = M2G[MA_AN];
            wanA  = WhatSpecies[GA_AN];
            tnoA  = Spe_Total_CNO[wanA];
            Anum  = MP[GA_AN];
            for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                GB_AN = natn[GA_AN][LB_AN];
                wanB  = WhatSpecies[GB_AN];
                tnoB  = Spe_Total_CNO[wanB];
                Bnum  = MP[GB_AN];

                for (i = 0; i < tnoA; i++) {
                    for (j = 0; j < tnoB; j++) {
                        CDM[spin][MA_AN][LB_AN][i][j]    = CDM[spin][MA_AN][LB_AN][i][j] * dum;
                        EDM[spin][MA_AN][LB_AN][i][j]    = EDM[spin][MA_AN][LB_AN][i][j] * dum;
                        iDM[0][spin][MA_AN][LB_AN][i][j] = iDM[0][spin][MA_AN][LB_AN][i][j] * dum;
                    }
                }
            }
        }
    }

    /****************************************************
                         bond-energies
    ****************************************************/

    My_Eele1[0] = 0.0;
    My_Eele1[1] = 0.0;
    for (MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        GA_AN = M2G[MA_AN];
        wanA  = WhatSpecies[GA_AN];
        tnoA  = Spe_Total_CNO[wanA];

        for (j = 0; j <= FNAN[GA_AN]; j++) {
            GB_AN = natn[GA_AN][j];
            wanB  = WhatSpecies[GB_AN];
            tnoB  = Spe_Total_CNO[wanB];

            for (k = 0; k < tnoA; k++) {
                for (l = 0; l < tnoB; l++) {
                    for (spin = 0; spin <= SpinP_switch; spin++) {
                        My_Eele1[spin] += CDM[spin][MA_AN][j][k][l] * nh[spin][MA_AN][j][k][l];
                    }
                }
            }
        }
    }

    /* MPI, My_Eele1 */
    MPI_Barrier(mpi_comm_level1);
    for (spin = 0; spin <= SpinP_switch; spin++) {
        MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    }

    if (SpinP_switch == 0) {
        Eele1[1] = Eele1[0];
    }

    if (3 <= level_stdout && myid0 == Host_ID) {
        printf("Eele00=%15.12f Eele01=%15.12f\n", Eele0[0], Eele0[1]);
        printf("Eele10=%15.12f Eele11=%15.12f\n", Eele1[0], Eele1[1]);
    }

    /****************************************************
                          output
    ****************************************************/

    if (myid0 == Host_ID) {

        strcpy(file_EV, ".EV");
        fnjoint(filepath, filename, file_EV);

        if ((fp_EV = fopen(file_EV, "w")) != NULL) {

            setvbuf(fp_EV, buf, _IOFBF, fp_bsize);

            fprintf(fp_EV, "\n");
            fprintf(fp_EV, "***********************************************************\n");
            fprintf(fp_EV, "***********************************************************\n");
            fprintf(fp_EV, "           Eigenvalues (Hartree) of SCF KS-eq.           \n");
            fprintf(fp_EV, "***********************************************************\n");
            fprintf(fp_EV, "***********************************************************\n\n");
            fprintf(fp_EV, "   Chemical Potential (Hartree) = %18.14f\n", ChemP);
            fprintf(fp_EV, "   Number of States             = %18.14f\n", Num_State);
            fprintf(fp_EV, "   Eigenvalues\n");
            fprintf(fp_EV, "              Up-spin           Down-spin\n");

            for (kloop = 0; kloop < T_knum; kloop++) {

                (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);

                k1 = T_KGrids1[kloop];
                k2 = T_KGrids2[kloop];
                k3 = T_KGrids3[kloop];

                if (0 < T_k_op[kloop]) {

                    fprintf(fp_EV, "\n");
                    fprintf(fp_EV, "   kloop=%i\n", kloop);
                    fprintf(fp_EV, "   k1=%10.5f k2=%10.5f k3=%10.5f\n\n", k1, k2, k3);
                    for (l = 1; l <= MaxN; l++) {
                        if (SpinP_switch == 0) {
                            fprintf(fp_EV, "%5d  %18.14f %18.14f\n", l, EIGEN[0][kloop][l], EIGEN[0][kloop][l]);
                        } else if (SpinP_switch == 1) {
                            fprintf(fp_EV, "%5d  %18.14f %18.14f\n", l, EIGEN[0][kloop][l], EIGEN[1][kloop][l]);
                        }
                    }
                }
            }
            fclose(fp_EV);
        } else {
            printf("Failure of saving the EV file.\n");
        }
    }

    // Destroy hipblasmp & hipsolverMp
    // if (all_knum == 1) {
    //     destroy_gpusolvermp(&opts4);
    //     destroy_hipblasmp(&opts2);
    // }

    /****************************************************
                         free arrays
    ****************************************************/

    if (all_knum == 1) {

        free(EVec_Rcv);
        free(index_Rcv_j);
        free(index_Rcv_i);
        free(EVec_Snd);
        free(index_Snd_j);
        free(index_Snd_i);

        free(Num_Rcv_EV);
        free(Num_Snd_EV);

        free(ie2);
        free(is2);
        free(ie1);
        free(is1);
    }

    free(SP_Atoms);
    free(gpu_k_order);
    if (k_evec_cache != NULL) {
        for (size_t point = 0; point < (size_t)Num_Comm_World1 * T_knum; ++point)
            free(k_evec_cache[point]);
        free(k_evec_cache);
    }
    if (use_gpusolver_dense && all_knum != 1 && BandCol_CacheTraceEnabled())
        printf("BAND_COL_CACHE rank=%d SCF=%d evec_hits=%d overlap_hits=%d evec=%.1f MiB overlap=%.1f MiB\n",
               myid0, SCF_iter, k_cache_hits, k_overlap_hits,
               k_cache_used / 1048576.0, BandCol_k_overlap_used / 1048576.0);
    free(SP_NZeros);
    free(My_NZeros);

    /* for PrintMemory and allocation */
    firsttime = 0;

    /* for elapsed time */

    if (measure_time) {
        printf("myid0=%2d time1 =%9.4f\n", myid0, time1);
        fflush(stdout);
        printf("myid0=%2d time2 =%9.4f\n", myid0, time2);
        fflush(stdout);
        printf("myid0=%2d time3 =%9.4f\n", myid0, time3);
        fflush(stdout);
        printf("myid0=%2d time4 =%9.4f\n", myid0, time4);
        fflush(stdout);
        printf("myid0=%2d time5 =%9.4f\n", myid0, time5);
        fflush(stdout);
        printf("myid0=%2d time6 =%9.4f\n", myid0, time6);
        fflush(stdout);
        printf("myid0=%2d time7 =%9.4f\n", myid0, time7);
        fflush(stdout);
        printf("myid0=%2d time8 =%9.4f\n", myid0, time8);
        fflush(stdout);
        printf("myid0=%2d time81=%9.4f\n", myid0, time81);
        fflush(stdout);
        printf("myid0=%2d time82=%9.4f\n", myid0, time82);
        fflush(stdout);
        printf("myid0=%2d time83=%9.4f\n", myid0, time83);
        fflush(stdout);
        printf("myid0=%2d time84=%9.4f\n", myid0, time84);
        fflush(stdout);
        printf("myid0=%2d time85=%9.4f\n", myid0, time85);
        fflush(stdout);
        printf("myid0=%2d time9 =%9.4f\n", myid0, time9);
        fflush(stdout);
        printf("myid0=%2d time10=%9.4f\n", myid0, time10);
        fflush(stdout);
        printf("myid0=%2d time11=%9.4f\n", myid0, time11);
        fflush(stdout);
        printf("myid0=%2d time11A=%9.4f\n", myid0, time11A);
        fflush(stdout);
        printf("myid0=%2d time11B=%9.4f\n", myid0, time11B);
        fflush(stdout);
        printf("myid0=%2d time12=%9.4f\n", myid0, time12);
        fflush(stdout);
    }

    // double part1avescf1   = get_max_value(part1);
    // double part21avescf1  = get_max_value(part2_1);
    // double part22avescf1  = get_max_value(part2_2);
    // double part23avescf1  = get_max_value(part2_3);
    // double part24avescf1  = get_max_value(part2_4);
    // double part25avescf1  = get_max_value(part2_5);
    // double part3avescf1   = get_max_value(part3);
    // double part4avescf1   = get_max_value(part4);
    // double partmulavescf1 = get_max_value(partmul);

    // double part1sumscftotalm1   = get_max_value(part1sum);
    // double part21sumscftotalm1  = get_max_value(part2_1sum);
    // double part22sumscftotalm1  = get_max_value(part2_2sum);
    // double part23sumscftotalm1  = get_max_value(part2_3sum);
    // double part24sumscftotalm1  = get_max_value(part2_4sum);
    // double part25sumscftotalm1  = get_max_value(part2_5sum);
    // double part3sumscftotalm1   = get_max_value(part3sum);
    // double part4sumscftotalm1   = get_max_value(part4sum);
    // double partmulsumscftotalm1 = get_max_value(partmulsum);

    // if (measure_time && SCF_iter == 1 && myid0 == 0) {
    //     printf("myid0=%2d part1 =%9.4f\n", myid0, part1avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part21 =%9.4f\n", myid0, part21avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part22 =%9.4f\n", myid0, part22avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part23 =%9.4f\n", myid0, part23avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part24 =%9.4f\n", myid0, part24avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part25 =%9.4f\n", myid0, part25avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part3 =%9.4f\n", myid0, part3avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d part4 =%9.4f\n", myid0, part4avescf1);
    //     fflush(stdout);
    //     printf("myid0=%2d partmul =%9.4f\n", myid0, partmulavescf1);
    //     fflush(stdout);
    //     double total = part1avescf1 + part21avescf1 + part22avescf1 + part23avescf1 + part24avescf1 + part25avescf1 +
    //                    part3avescf1 + part4avescf1;

    //     printf("myid0=%2d total =%9.4f\n", myid0, total);
    //     fflush(stdout);
    // }

    // if (measure_time && SCF_iter != 1 && myid0 == 0) {
    //     printf("myid0=%2d part1ave =%9.4f\n", myid0, part1sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part21ave =%9.4f\n", myid0, part21sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part22ave =%9.4f\n", myid0, part22sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part23ave =%9.4f\n", myid0, part23sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part24ave =%9.4f\n", myid0, part24sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part25ave =%9.4f\n", myid0, part25sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part3ave =%9.4f\n", myid0, part3sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d part4ave =%9.4f\n", myid0, part4sumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);
    //     printf("myid0=%2d partmul =%9.4f\n", myid0, partmulsumscftotalm1 / (double)(SCF_iter - 1));
    //     fflush(stdout);

    //     double total = part1sumscftotalm1 + part21sumscftotalm1 + part22sumscftotalm1 + part23sumscftotalm1 +
    //                    part24sumscftotalm1 + part25sumscftotalm1 + part3sumscftotalm1 + part4sumscftotalm1;

    //     printf("myid0=%2d total =%9.4f\n", myid0, total / (double)(SCF_iter - 1));
    //     fflush(stdout);
    // }

    if (mpi_eigen_comms_ready) {
        MPI_Comm_free(&mpi_comm_rows);
        MPI_Comm_free(&mpi_comm_cols);
    }

    /* Honour the same retention policy at the SCF boundary as at each turn.
       Unconditionally freeing here made OPENMX_BAND_GEMMUL8_TURN_RELEASE=0
       ineffective and forced all cached-SCF k-point ranks to allocate their
       ~1-GiB workspaces simultaneously on the next ZGEMM. */
    if (BandCol_GemmWorkspaceScfRelease()) {
        openmx_gemmul8ReleaseWorkspaces();
    }

    MPI_Barrier(mpi_comm_level1);
    dtime(&TEtime);
    time0 = TEtime - TStime;
    openmx_mgpu_band_scope_leave(input_scope);
    return time0;
}

void Construct_Band_CsHs(int SCF_iter, int all_knum, int * order_GA, int * MP, double * S1, double * H1, double k1,
                         double k2, double k3, dcomplex * Cs, dcomplex * Hs, int n, int use_gpusolver_dense,
                         int owns_global_dense_rank)
{
    const int need_s = (SCF_iter == 1 || all_knum != 1);
    /* Honor the caller's memory-based fallback, including host matrix construction. */
    const int dense_gpusolver_owner =
        (use_gpusolver_dense &&
         ((all_knum == 1 && owns_global_dense_rank) ||
          (all_knum != 1 && Set_Hamiltonian_OpenMP_Rank_Is_Selected())));

    BandCol_last_construct_on_device = 0;

    if (use_gpusolver_dense && !dense_gpusolver_owner) {
        return;
    }

    if (dense_gpusolver_owner && BandCol_mgpu_host_dense) {
        /* The cooperative owner needs the complete matrix even when its
           ScaLAPACK descriptor describes only a local MPI block. H1 and S1
           are already globally packed in AN/LB_AN/i/j order. */
        size_t h_index = 0;
        BandCol_HostEnsure(n);
        Cs=BandCol_host_s;
        Hs=BandCol_host_h;
        BandCol_ZeroComplex(Hs, (size_t)n * n);
        if (need_s) BandCol_ZeroComplex(Cs, (size_t)n * n);
        for (int AN = 1; AN <= atomnum; ++AN) {
            const int GA = order_GA[AN];
            const int tnoA = Spe_Total_CNO[WhatSpecies[GA]];
            for (int LB = 0; LB <= FNAN[GA]; ++LB) {
                const int GB = natn[GA][LB], Rn = ncn[GA][LB];
                const int tnoB = Spe_Total_CNO[WhatSpecies[GB]];
                const double phase = 2.0 * PI * (k1 * atv_ijk[Rn][1] +
                    k2 * atv_ijk[Rn][2] + k3 * atv_ijk[Rn][3]);
                const double co = cos(phase), si = sin(phase);
                for (int i = 0; i < tnoA; ++i) {
                    for (int j = 0; j < tnoB; ++j, ++h_index) {
                        const size_t index = (size_t)(MP[GB] + j - 1) * n + MP[GA] + i - 1;
                        Hs[index].r += H1[h_index] * co;
                        Hs[index].i += H1[h_index] * si;
                        if (need_s) {
                            Cs[index].r += S1[h_index] * co;
                            Cs[index].i += S1[h_index] * si;
                        }
                    }
                }
            }
        }
        return;
    }

    BandCol_ConstructCache_Ensure(order_GA, MP, n, dense_gpusolver_owner);

    if (dense_gpusolver_owner) {
        BandCol_ConstructDenseCsHs_HIP(need_s, n, k1, k2, k3, S1, H1);
        return;
    }

    {
        const BandColConstructEntry *entries = BandCol_construct_cache.local_entries;
        const int count = BandCol_construct_cache.local_count;
        int prev_l1 = 0x7fffffff, prev_l2 = 0x7fffffff, prev_l3 = 0x7fffffff;
        double co = 0.0, si = 0.0;

        BandCol_ZeroComplex(Hs, (size_t)na_rows * (size_t)na_cols);
        if (need_s) {
            BandCol_ZeroComplex(Cs, (size_t)na_rows * (size_t)na_cols);
        }

        for (int idx = 0; idx < count; ++idx) {
            const BandColConstructEntry *entry = &entries[idx];
            const int h_index = entry->h_index;

            if (entry->l1 != prev_l1 || entry->l2 != prev_l2 || entry->l3 != prev_l3) {
                double kRn = k1 * (double)entry->l1 + k2 * (double)entry->l2 + k3 * (double)entry->l3;
                si = sin(2.0 * PI * kRn);
                co = cos(2.0 * PI * kRn);
                prev_l1 = entry->l1;
                prev_l2 = entry->l2;
                prev_l3 = entry->l3;
            }

            if (need_s) {
                Cs[entry->index0].r += S1[h_index] * co;
                Cs[entry->index0].i += S1[h_index] * si;
            }

            Hs[entry->index0].r += H1[h_index] * co;
            Hs[entry->index0].i += H1[h_index] * si;
        }
    }
}

static double get_max_value(double local_value)
{
    double global_max;
    MPI_Allreduce(&local_value, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_max;
}
