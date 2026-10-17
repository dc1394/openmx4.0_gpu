/**********************************************************************************
  Cluster_DFT_Col.c:

     Cluster_DFT_Col.c is a subroutine to perform cluster collinear calculations.

  Log of Cluster_DFT_Col.c:

     21/Feb./2019  Released by T. Ozaki

**********************************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include "openmx_mgpu_runtime.h"
#include "openmx_gpu_transform.h"
#include "openmx_cluster_capacity.h"
#include "openmx_gpu_dm_collective.h"
#include "openmx_gpu_workers.h"
#include "lapack_prototypes.h"
#include "set_hip_default_device_from_local_rank.h"
#include "elpa_cosma_bridge.h"
#include "eigen_refine_gpu.h"
#include "eigen_refine_hooks.h"
#include "openmx_precision.h"
extern void openmx_gemmul8TrimWorkspaces(void);
extern size_t openmx_gemmul8WorkspaceBytes(void);
extern hipblasStatus_t openmx_gemmul8DgemmUnblocked(hipblasHandle_t handle, hipblasOperation_t transa,
                                                    hipblasOperation_t transb, int m, int n, int k,
                                                    const double *alpha, const double *A, int lda,
                                                    const double *B, int ldb, const double *beta, double *C,
                                                    int ldc);
extern int openmx_gpu_is_apu(void);
extern int openmx_gpu_is_mi300a(void);
/* eigen_refine_gpu.cu: device bytes of the product blocks of a refined solve
   of maxn wanted states of an n x n problem */
extern size_t openmx_eigen_refine_product_bytes(int cplx, int n, int maxn);
#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <time.h>

static int ClusterCol_mgpu_host_dense = 0;
/* matrix order for which the automatic capacity fallback of gpusolver
   chose the distributed gpusolver2 path (0: none); see
   openmx_cluster_capacity.h */
static int ClusterCol_gs2_fallback_n = 0;

/* the distributed ELPA/COSMA path: requested by the input, or the capacity
   fallback of gpusolver for this order */
static int ClusterCol_UseGs2(int n)
{
    return gpusolver2_flag || (0 < n && ClusterCol_gs2_fallback_n == n);
}

static void ClusterCol_ResetGs2Fallback(void)
{
    if (ClusterCol_gs2_fallback_n != 0) openmx_gs2_set_fallback_report(0);
    ClusterCol_gs2_fallback_n = 0;
}

/* Grid over every rank of mpi_comm_level1 for the capacity fallback of a
   spin-polarized run.  The spin worlds are contiguous halves of the ranks,
   i.e. one node each under the usual block placement, so solving each spin
   on its own world's grid would never spread one solve over the GPUs of both
   nodes.  In the fallback both spins are solved in turn on this grid, and
   each spin's eigenvectors go to its own world in the layout of the
   ScaLAPACK path (EVec1[spin], states is2..ie2 of the world's ranks). */
typedef struct {
    int n;                       /* order the grid was built for (0: none) */
    int bhandle, ictxt;
    int np_rows, np_cols, my_prow, my_pcol, nblk;
    int na_rows, na_cols, na_rows_max, na_cols_max;
    int desc[9];
    double *Ss, *Hs, *Cs;
    int overlap_n;               /* order whose X = U lambda^-1/2 is in Ss */
} ClusterColUnionGrid;
static ClusterColUnionGrid ClusterCol_ugrid;

/* collective over mpi_comm_level1 */
static void ClusterCol_UnionGrid_Free(void)
{
    ClusterColUnionGrid *g = &ClusterCol_ugrid;

    if (g->n == 0) return;
    openmx_gs2_grid_free(g->ictxt);
    Cfree_blacs_system_handle(g->bhandle);
    Cblacs_gridexit(g->ictxt);
    free(g->Ss);
    free(g->Hs);
    free(g->Cs);
    memset(g, 0, sizeof(*g));
}

/* collective over mpi_comm_level1: the grid of order n, built like the
   cluster grids of DFT.c ("Row" order, np_cols the largest divisor of the
   rank count not above its square root, nblk from 128 down) */
static void ClusterCol_UnionGrid_Ensure(int n)
{
    ClusterColUnionGrid *g = &ClusterCol_ugrid;
    int numprocs0, myid0, nblk_m, info, ZERO = 0;

    if (g->n == n) return;
    ClusterCol_UnionGrid_Free();

    MPI_Comm_size(mpi_comm_level1, &numprocs0);
    MPI_Comm_rank(mpi_comm_level1, &myid0);

    g->np_cols = (int)(sqrt((float)numprocs0));
    do {
        if ((numprocs0 % g->np_cols) == 0) break;
        g->np_cols--;
    } while (g->np_cols >= 2);
    if (g->np_cols < 1) g->np_cols = 1;
    g->np_rows = numprocs0 / g->np_cols;

    nblk_m = NBLK;
    while ((nblk_m * g->np_rows > n || nblk_m * g->np_cols > n) && (nblk_m > 1)) nblk_m /= 2;
    if (nblk_m < 1) nblk_m = 1;
    MPI_Allreduce(&nblk_m, &g->nblk, 1, MPI_INT, MPI_MIN, mpi_comm_level1);

    g->my_prow = myid0 / g->np_cols;
    g->my_pcol = myid0 % g->np_cols;
    g->na_rows = numroc_(&n, &g->nblk, &g->my_prow, &ZERO, &g->np_rows);
    g->na_cols = numroc_(&n, &g->nblk, &g->my_pcol, &ZERO, &g->np_cols);

    g->bhandle = Csys2blacs_handle(mpi_comm_level1);
    g->ictxt = g->bhandle;
    Cblacs_gridinit(&g->ictxt, "Row", g->np_rows, g->np_cols);

    MPI_Allreduce(&g->na_rows, &g->na_rows_max, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
    MPI_Allreduce(&g->na_cols, &g->na_cols_max, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

    g->Ss = (double *)malloc(sizeof(double) * (size_t)g->na_rows * (size_t)g->na_cols + sizeof(double));
    g->Hs = (double *)malloc(sizeof(double) * (size_t)g->na_rows * (size_t)g->na_cols + sizeof(double));
    g->Cs = (double *)malloc(sizeof(double) * (size_t)g->na_rows_max * (size_t)g->na_cols_max + sizeof(double));
    if (g->Ss == NULL || g->Hs == NULL || g->Cs == NULL) {
        fprintf(stderr, "Cluster_DFT_Col: out of host memory for the %d x %d grid of the"
                        " distributed capacity fallback (n=%d)\n", g->np_rows, g->np_cols, n);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1, 1);
    }

    descinit_(g->desc, &n, &n, &g->nblk, &g->nblk, &ZERO, &ZERO, &g->ictxt, &g->na_rows, &info);
    g->n = n;
    g->overlap_n = 0;
}

/* Overlap_Cluster_Ss and Hamiltonian_Cluster_Hs fill the local block of
   the grid described by these globals; point them at the union grid for
   the duration of a fill */
typedef struct {
    int np_rows, np_cols, my_prow, my_pcol, nblk, na_rows, na_cols;
} ClusterColGridGlobals;

static ClusterColGridGlobals ClusterCol_EnterUnionGridGlobals(void)
{
    const ClusterColUnionGrid *g = &ClusterCol_ugrid;
    ClusterColGridGlobals saved;

    saved.np_rows = np_rows; saved.np_cols = np_cols;
    saved.my_prow = my_prow; saved.my_pcol = my_pcol;
    saved.nblk = nblk; saved.na_rows = na_rows; saved.na_cols = na_cols;
    np_rows = g->np_rows; np_cols = g->np_cols;
    my_prow = g->my_prow; my_pcol = g->my_pcol;
    nblk = g->nblk; na_rows = g->na_rows; na_cols = g->na_cols;
    return saved;
}

static void ClusterCol_LeaveUnionGridGlobals(ClusterColGridGlobals saved)
{
    np_rows = saved.np_rows; np_cols = saved.np_cols;
    my_prow = saved.my_prow; my_pcol = saved.my_pcol;
    nblk = saved.nblk; na_rows = saved.na_rows; na_cols = saved.na_cols;
}

#define  measure_time   0

double *Hex0;
double *Hex;

extern int ClusterCol_CalcDMRootDense_HIP(int entry_count, int size_H1, int n, int maxn, int nk_occ, int calc_pdm,
                                          const int *basis0, const int *basis1,
                                          const double *occ, const double *occ_e, const double *pocc,
                                          const double *dense_evec, const double *device_evec, double *dm_buffer);
extern int ClusterCol_BuildDeviceDenseFromPacked_HIP(const double *H1, const int *dense_index,
                                                     int tnum, int n, double *d_H);
extern void ClusterCol_HIPDetailTimer_Reset(void);
extern void ClusterCol_HIPDetailTimer_Get(double *dense_build_kernel, double *dm_kernel);
extern int ClusterCol_SymmetrizeLower_HIP(int n, double *a, hipStream_t stream);
extern int openmx_magma_dsyevdx_gpu(int n, int maxn, double *d_A, double *w, int *mout);
extern int openmx_magma_release_z_workspace(void);
extern int openmx_magma_release_d_workspace(void);
extern int openmx_hipsolver_release_workspace(void);
extern void Set_Density_Grid_GPU_Release_Device_Caches(void);

typedef struct {
    int valid;
    int n;
    int entry_count;
    unsigned long long fingerprint;
    int *basis0;
    int *basis1;
} ClusterColDMEntryCache;

typedef struct {
    int max_state_capacity;
    size_t dm_buffer_elems;
    double *occ;
    double *occ_e;
    double *pocc;
    double *dm_buffer;
} ClusterColRootDMWorkspace;

static ClusterColDMEntryCache ClusterCol_dm_entry_cache = {0};
static ClusterColRootDMWorkspace ClusterCol_root_dm_workspace = {0};
static double *ClusterCol_CachedGpuSolverDenseEVec[2] = {NULL, NULL};
static int ClusterCol_CachedGpuSolverDenseValid[2] = {0, 0};
static int ClusterCol_CachedGpuSolverDenseN = 0;
static int ClusterCol_CachedGpuSolverDenseMaxN = 0;
/* d_tmp retains the most recently backtransformed spin until the next
   solve. The other spin of a single-rank run still uses its host copy. */
static int ClusterCol_DeviceEvecSpin = -1;
/* Serial spin roots cannot retain both overlaps on the device. Keep a
   bounded host copy instead, valid only within the current SCF sequence. */
static double *ClusterCol_SerialOverlap = NULL;
static int ClusterCol_SerialOverlapN = 0;
/* The root GPU path does not populate the distributed Ss matrix. Track
   its initialization separately so a later ELPA fallback can build it. */
static int ClusterCol_DistributedOverlapN = 0;

static void ClusterCol_SerialOverlapReset(void)
{
    free(ClusterCol_SerialOverlap);
    ClusterCol_SerialOverlap = NULL;
    ClusterCol_SerialOverlapN = 0;
}

typedef struct {
    int active;
    int scf_iter;
    int n;
    int maxn;
    int dense_build_count;
    double dense_build;
    double overlap_eig;
    double overlap_scale;
    double h_transform_gemm;
    double syevdx;
    double backtransform;
    double evec_copyout;
    double dm_total;
} ClusterColDetailTimers;

static ClusterColDetailTimers ClusterCol_detail_timers = {0};

/* Dense GPU context (added by H.Kawai, ported from 3.9.9 GPU) */
typedef struct {
    int                initialized;
    int                device_id;
    int                matrix_dim;
    int                transformed_s_valid;
    int                transformed_s_dim;
    hipStream_t       stream;
    hipblasHandle_t     hipblas;
    double *           d_S;
    double *           d_H;
    double *           d_tmp;
    double *           d_W;
    void *             d_work;        /* the refined eigensolver's FP32 solve and device Rayleigh-Ritz */
    size_t             d_work_bytes;
    int32_t *          d_info;
} ClusterColGpuSolverCtx;

static ClusterColGpuSolverCtx ClusterCol_gpusolver_ctx = {0};

static void Patch2Full_Cluster_Owner(double ****RH, double *H, int *MP, int owns_dense);
static void Patch2Device_Cluster_Owner(double ****RH, int *MP, int owns_dense, int n, double *d_H);
static void ClusterCol_BuildDenseIndex(const int *order_GA, int *MP, int n, int tnum, int *dense_index);
static void ClusterCol_BuildDeviceDenseFromPacked(const double *H1, const int *dense_index,
                                                  int tnum, int n, double *d_H);

static void ClusterCol_AbortWithMessage(const char *message)
{
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1, 1);
}

static int ClusterCol_GpuVerbose(void)
{
    const char *value = getenv("OPENMX_GPU_VERBOSE");

    return (value != NULL && value[0] == '1');
}

/* GPU/CPU crossover of the collinear cluster dense eigensolver path.  The
   cluster diagonalization runs once per SCF step and the root-dense GPU
   solve of one rank beats the multi-rank ELPA fallback at every measured
   size (on one RTX 5080 shared by 8 ranks: total 1.27x at n=780 down to
   1.06x at n=240), so the default sits far below the global
   GPU_CPU_SWITCH_NUM; below it the remaining gain is within noise of the
   grid-dominated runtime.  OPENMX_CLUSTER_GPU_SWITCH_NUM=<dim> overrides
   it.  Also consulted by DFT.c so the startup fallback notice reports the
   effective value. */
#define CLUSTER_GPU_CPU_SWITCH_NUM 400

int Cluster_DFT_Col_GpuSwitchNum(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("OPENMX_CLUSTER_GPU_SWITCH_NUM");
        int parsed;

        cached = CLUSTER_GPU_CPU_SWITCH_NUM;
        if (value != NULL && 0 < (parsed = atoi(value))) {
            cached = parsed;
        }
    }
    return cached;
}

static size_t ClusterCol_CheckedMulCount(size_t a, size_t b, const char *label)
{
    if (a != 0 && b > ((size_t)-1) / a) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Dimension overflow in Cluster_DFT_Col.c: %s", label);
        ClusterCol_AbortWithMessage(msg);
    }

    return a * b;
}

static void *ClusterCol_MallocArray(size_t count, size_t elem_size, const char *label)
{
    size_t bytes = ClusterCol_CheckedMulCount(count, elem_size, label);
    void *ptr = malloc((bytes == 0) ? 1 : bytes);

    if (ptr == NULL) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Out of memory in Cluster_DFT_Col.c: %s (%zu bytes)", label, bytes);
        ClusterCol_AbortWithMessage(msg);
    }

    return ptr;
}

static void ClusterCol_DetailTimer_Begin(int myid1, int SCF_iter, int n, int MaxN)
{
    if (myid1 != 0) {
        return;
    }

    memset(&ClusterCol_detail_timers, 0, sizeof(ClusterCol_detail_timers));
    const char *profile = getenv("OPENMX_CLUSTER_PROFILE");
    const char *measure = getenv("OPENMX_CLUSTER_MEASURE_TIME");
    ClusterCol_detail_timers.active = (profile != NULL && atoi(profile) != 0)
                                   || (measure != NULL && atoi(measure) != 0);
    ClusterCol_detail_timers.scf_iter = SCF_iter;
    ClusterCol_detail_timers.n = n;
    ClusterCol_detail_timers.maxn = MaxN;
    ClusterCol_HIPDetailTimer_Reset();
}

static void ClusterCol_DetailTimer_Add(double *slot, double stime)
{
    double etime;

    if (!ClusterCol_detail_timers.active) {
        return;
    }

    dtime(&etime);
    *slot += etime - stime;
}

static void ClusterCol_DetailTimer_Report(int myid0, int myid1)
{
    if (myid1 != 0 || !ClusterCol_detail_timers.active) {
        return;
    }

    const ClusterColDetailTimers *t = &ClusterCol_detail_timers;
    printf("CLUSTER_PROFILE rank=%d SCF=%d n=%d maxn=%d build=%.6f overlap=%.6f transform=%.6f eig=%.6f back=%.6f copy=%.6f dm=%.6f\n",
           myid0, t->scf_iter, t->n, t->maxn, t->dense_build,
           t->overlap_eig + t->overlap_scale, t->h_transform_gemm,
           t->syevdx, t->backtransform, t->evec_copyout, t->dm_total);
    fflush(stdout);
    /* Per-call stage breakdown of the owning rank's dense GPU solve:
       set OPENMX_CLUSTER_MEASURE_TIME=1.  Diagnostic, default off. */
    {
        static int enabled = -1;

        if (enabled < 0) {
            const char *v = getenv("OPENMX_CLUSTER_MEASURE_TIME");
            enabled = (v != NULL && atoi(v) != 0);
        }
        if (enabled) {
            double build_kernel = 0.0, dm_kernel = 0.0;

            ClusterCol_HIPDetailTimer_Get(&build_kernel, &dm_kernel);
            printf("ClusterCol detail myid0=%2d SCF=%2d n=%d maxn=%d "
                   "dense_build=%.3f (builds=%d kernel=%.3f) overlap_eig=%.3f "
                   "overlap_scale=%.3f h_transform=%.3f syevdx=%.3f "
                   "backtransform=%.3f evec_copyout=%.3f dm_total=%.3f (kernel=%.3f)\n",
                   myid0, ClusterCol_detail_timers.scf_iter,
                   ClusterCol_detail_timers.n, ClusterCol_detail_timers.maxn,
                   ClusterCol_detail_timers.dense_build,
                   ClusterCol_detail_timers.dense_build_count, build_kernel,
                   ClusterCol_detail_timers.overlap_eig,
                   ClusterCol_detail_timers.overlap_scale,
                   ClusterCol_detail_timers.h_transform_gemm,
                   ClusterCol_detail_timers.syevdx,
                   ClusterCol_detail_timers.backtransform,
                   ClusterCol_detail_timers.evec_copyout,
                   ClusterCol_detail_timers.dm_total, dm_kernel);
            fflush(stdout);
        }
    }

    ClusterCol_detail_timers.active = 0;
}

static unsigned long long ClusterCol_DMEntryFingerprint(int *MP, int n)
{
    unsigned long long fp = 1469598103934665603ULL;

#define CLUSTER_COL_DM_MIX(value) \
    do { \
        fp ^= (unsigned long long)(uint64_t)(value); \
        fp *= 1099511628211ULL; \
    } while (0)

    CLUSTER_COL_DM_MIX(n);
    CLUSTER_COL_DM_MIX(atomnum);

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int wanA = WhatSpecies[GA_AN];

        CLUSTER_COL_DM_MIX(GA_AN);
        CLUSTER_COL_DM_MIX(wanA);
        CLUSTER_COL_DM_MIX(Spe_Total_CNO[wanA]);
        CLUSTER_COL_DM_MIX(FNAN[GA_AN]);
        CLUSTER_COL_DM_MIX(MP[GA_AN]);

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];

            CLUSTER_COL_DM_MIX(GB_AN);
            CLUSTER_COL_DM_MIX(WhatSpecies[GB_AN]);
            CLUSTER_COL_DM_MIX(MP[GB_AN]);
        }
    }

#undef CLUSTER_COL_DM_MIX

    return fp;
}

static void ClusterCol_DMEntryCache_Reset(void)
{
    free(ClusterCol_dm_entry_cache.basis0);
    free(ClusterCol_dm_entry_cache.basis1);
    memset(&ClusterCol_dm_entry_cache, 0, sizeof(ClusterCol_dm_entry_cache));
}

static void ClusterCol_DMEntryCache_Ensure(int *MP, int n)
{
    ClusterColDMEntryCache *cache = &ClusterCol_dm_entry_cache;
    unsigned long long fingerprint = ClusterCol_DMEntryFingerprint(MP, n);
    int entry_count = 0;

    if (cache->valid && cache->n == n && cache->fingerprint == fingerprint) {
        return;
    }

    ClusterCol_DMEntryCache_Reset();

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]];

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int tnoB = Spe_Total_CNO[WhatSpecies[GB_AN]];

            entry_count += tnoA * tnoB;
        }
    }

    cache->basis0 = (int *)ClusterCol_MallocArray((size_t)entry_count, sizeof(int), "collinear DM basis0 cache");
    cache->basis1 = (int *)ClusterCol_MallocArray((size_t)entry_count, sizeof(int), "collinear DM basis1 cache");

    entry_count = 0;
    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]];
        int Anum = MP[GA_AN];

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int tnoB = Spe_Total_CNO[WhatSpecies[GB_AN]];
            int Bnum = MP[GB_AN];

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

    cache->valid = 1;
    cache->n = n;
    cache->entry_count = entry_count;
    cache->fingerprint = fingerprint;
}

static void ClusterCol_RootDMWorkspace_Reset(void)
{
    free(ClusterCol_root_dm_workspace.occ);
    free(ClusterCol_root_dm_workspace.occ_e);
    free(ClusterCol_root_dm_workspace.pocc);
    free(ClusterCol_root_dm_workspace.dm_buffer);
    memset(&ClusterCol_root_dm_workspace, 0, sizeof(ClusterCol_root_dm_workspace));
}

static void ClusterCol_RootDMWorkspace_Ensure(int max_state, size_t dm_buffer_elems)
{
    ClusterColRootDMWorkspace *ws = &ClusterCol_root_dm_workspace;

    if (max_state <= ws->max_state_capacity && dm_buffer_elems <= ws->dm_buffer_elems) {
        return;
    }

    ClusterCol_RootDMWorkspace_Reset();

    ws->occ = (double *)ClusterCol_MallocArray((size_t)max_state, sizeof(double), "collinear root DM occupations");
    ws->occ_e = (double *)ClusterCol_MallocArray((size_t)max_state, sizeof(double), "collinear root DM energy occupations");
    ws->pocc = (double *)ClusterCol_MallocArray((size_t)max_state, sizeof(double), "collinear root DM partial occupations");
    ws->dm_buffer = (double *)ClusterCol_MallocArray(dm_buffer_elems, sizeof(double), "collinear root DM buffer");

    ws->max_state_capacity = max_state;
    ws->dm_buffer_elems = dm_buffer_elems;
}

static int ClusterCol_PrecomputeOccupations(int spin, int kmin, int kmax, const double *ko_spin,
                                            double *occ, double *occ_e, double *pocc, int calc_pdm)
{
    int nk = 0;
    int po = 0;
    const double max_x = 60.0;

    for (int local_k = 0, global_k = kmin; global_k <= kmax; local_k++, global_k++) {
        double x;
        double x_for_fermi;
        double weight;
        double partial_weight = 0.0;

        if (xanes_calc == 1) {
            x = (ko_spin[global_k] - ChemP_XANES[spin]) * Beta;
        }
        else {
            x = (ko_spin[global_k] - ChemP) * Beta;
        }

        if (x <= -max_x) x = -max_x;
        if (max_x <= x)  x =  max_x;

        x_for_fermi = x;
        weight = FermiFunc(x, spin, global_k, &po, &x_for_fermi);

        occ[local_k] = weight;
        occ_e[local_k] = weight * ko_spin[global_k];

        if (calc_pdm) {
            double x2;

            if (xanes_calc == 1) {
                x2 = (ko_spin[global_k] - (ChemP_XANES[spin] + ene_win_partial_charge)) * Beta;
            }
            else {
                x2 = (ko_spin[global_k] - (ChemP + ene_win_partial_charge)) * Beta;
            }

            if (x2 <= -max_x) x2 = -max_x;
            if (max_x <= x2)  x2 =  max_x;

            x_for_fermi = x;
            partial_weight = fabs(weight - FermiFunc(x2, spin, global_k, &po, &x_for_fermi));
            pocc[local_k] = partial_weight;
        }
        else {
            pocc[local_k] = 0.0;
        }

        if (1.0e-13 < fabs(weight) || 1.0e-13 < fabs(partial_weight)) {
            nk = local_k + 1;
        }
    }

    return nk;
}

static void ClusterCol_StoreDMComponents(int myid0, int spin, int *MP, double *****CDM, double *****EDM,
                                         const double *DM1, const double *EDM1, const double *PDM1)
{
    int p = 0;

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int MA_AN = F_G2M[GA_AN];
        int tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]];
        int ID = G2ID[GA_AN];

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int tnoB = Spe_Total_CNO[WhatSpecies[GB_AN]];

            if (myid0 == ID) {
                for (int i = 0; i < tnoA; i++) {
                    for (int j = 0; j < tnoB; j++, p++) {
                        CDM[spin][MA_AN][LB_AN][i][j] = DM1[p];
                        EDM[spin][MA_AN][LB_AN][i][j] = EDM1[p];
                        if (cal_partial_charge) {
                            Partial_DM[spin][MA_AN][LB_AN][i][j] = PDM1[p];
                        }
                    }
                }
            }
            else {
                p += tnoA * tnoB;
            }
        }
    }
}

static void ClusterCol_ReleaseGpuSolverCachedSpinEVec(int myid1, int spin)
{
    if (spin < 0 || 1 < spin) {
        return;
    }

    if (myid1 == 0) {
        free(ClusterCol_CachedGpuSolverDenseEVec[spin]);
    }

    ClusterCol_CachedGpuSolverDenseEVec[spin] = NULL;
    ClusterCol_CachedGpuSolverDenseValid[spin] = 0;
}

static void ClusterCol_ReleaseGpuSolverCachedEVec(int myid1)
{
    ClusterCol_DeviceEvecSpin = -1;
    ClusterCol_ReleaseGpuSolverCachedSpinEVec(myid1, 0);
    ClusterCol_ReleaseGpuSolverCachedSpinEVec(myid1, 1);
    ClusterCol_CachedGpuSolverDenseN = 0;
    ClusterCol_CachedGpuSolverDenseMaxN = 0;
}

static void ClusterCol_StashGpuSolverDenseEVec(int myid1, int spin, int n, int maxn, double **dense_evec)
{
    ClusterCol_ReleaseGpuSolverCachedSpinEVec(myid1, spin);

    if (myid1 == 0) {
        ClusterCol_CachedGpuSolverDenseEVec[spin] = *dense_evec;
        ClusterCol_CachedGpuSolverDenseValid[spin] = (*dense_evec != NULL);
        ClusterCol_CachedGpuSolverDenseN = n;
        ClusterCol_CachedGpuSolverDenseMaxN = maxn;
        ClusterCol_DeviceEvecSpin = spin;
    }

    *dense_evec = NULL;
}

static void ClusterCol_RefineRelease(void);
static void ClusterCol_RefineResetStage(void);

static void ClusterCol_GpuSolver_Destroy(void)
{
    ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;
    ClusterCol_DeviceEvecSpin = -1;

    ClusterCol_RefineRelease();
    if (ctx->d_work != NULL)     wait_hipfunc(hipFree(ctx->d_work));
    if (ctx->d_info != NULL)     wait_hipfunc(hipFree(ctx->d_info));
    if (ctx->d_S != NULL)        wait_hipfunc(hipFree(ctx->d_S));
    if (ctx->d_H != NULL)        wait_hipfunc(hipFree(ctx->d_H));
    if (ctx->d_tmp != NULL)      wait_hipfunc(hipFree(ctx->d_tmp));
    if (ctx->d_W != NULL)        wait_hipfunc(hipFree(ctx->d_W));
    if (ctx->hipblas != NULL)     wait_hipfunc(hipblasDestroy(ctx->hipblas));
    if (ctx->stream != NULL)     wait_hipfunc(hipStreamDestroy(ctx->stream));

    memset(ctx, 0, sizeof(*ctx));
    ctx->device_id = -1;
}

void Cluster_DFT_Col_Release_GPU_Solver(void)
{
    ClusterCol_GpuSolver_Destroy();
}

/* Run-boundary reset (between -runtest/-runtestL inputs): additionally drop
   the geometry-derived DM entry cache, the root DM workspace and the cached
   dense eigenvectors, none of which may survive into a different system,
   and return the refined eigensolver's stage to its initial state (every
   rank calls this, so the next input configures it collectively again).
   Idempotent; only the root rank ever owns the cached eigenvectors, and on
   the other ranks the pointers are NULL. */
void Cluster_DFT_Col_Release_GPU_Caches(void)
{
    ClusterCol_mgpu_host_dense=0;
    ClusterCol_ResetGs2Fallback();
    ClusterCol_UnionGrid_Free();
    ClusterCol_DistributedOverlapN = 0;
    ClusterCol_SerialOverlapReset();
    ClusterCol_DMEntryCache_Reset();
    ClusterCol_RootDMWorkspace_Reset();
    ClusterCol_ReleaseGpuSolverCachedEVec(0);
    ClusterCol_GpuSolver_Destroy();
    ClusterCol_RefineResetStage();
}

/* Budget the three retained OpenMX panels, MAGMA's ~1.5-panel device
   work array (rounded up to two), and BLAS workspaces per simultaneous
   owner. Group roots by node and PCI identity, so separate devices and
   sequential spins do not get charged twice. Already resident OpenMX
   buffers count toward the budget on later SCF iterations. Return 2 when
   one spin at a time fits, so a shared GPU need not fall back to ELPA.
   When the concurrent mode fits the budget, the owners reserve their
   matrices right here (ClusterCol_TryEnsureMatrixCapacity), so that a
   failed allocation is part of the collective verdict instead of an abort
   on one rank; *allocated (owners) accumulates what that allocated.
   maxn: the states a refined solve of this step would refine (its product
   blocks are budgeted, see ClusterCol_RefineBytes).
   OPENMX_CLUSTER_GPU_SERIAL (or the NVIDIA name OPENMX_CLUSTER_GPU_DIAG_
   SERIAL) =0 disables the serialized tier, =2 skips the concurrent one. */
static size_t ClusterCol_RefineBytes(int n, int maxn);
static int ClusterCol_TryEnsureMatrixCapacity(int n, int *allocated);
static void ClusterCol_GpuSolver_FreeMatrices(void);

/* dense owners of the current solve on this rank's device (owners only;
   GpuDiagFitsOnce): with two spin worlds on one device both forward
   transforms run at the same time */
static int ClusterCol_owners_on_device = 1;

static int ClusterCol_GpuDiagFitsOnce(int n, int maxn, int myworld1, int myid1, int *retryable, int *allocated)
{
    const char *env = getenv("OPENMX_CLUSTER_GPU_DIAG");
    const char *reserve_env = getenv("OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB");
    const char *serial_env = getenv("OPENMX_CLUSTER_GPU_SERIAL");
    const double mib = 1024.0 * 1024.0;
    double reserve = 256.0 * mib;
    int local_fit = 1, fit = 1;
    int local_serial_fit = 1, serial_fit = 1;
    int local_retryable = 1;
    MPI_Comm node_comm, roots_comm;

    if (serial_env == NULL) serial_env = getenv("OPENMX_CLUSTER_GPU_DIAG_SERIAL");
    if (env != NULL && atoi(env) == 0)
        local_fit = local_serial_fit = local_retryable = 0;
    if (serial_env != NULL && atoi(serial_env) == 0) local_serial_fit = 0;
    if (reserve_env != NULL && atof(reserve_env) >= 0.0)
        reserve = atof(reserve_env) * mib;

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &node_comm);
    MPI_Comm_split(node_comm, myid1 == 0 ? 0 : MPI_UNDEFINED, 0, &roots_comm);
    if (myid1 == 0) {
        ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;
        char bus[32] = {0}, *buses;
        double info[3] = {0.0, 0.0, 0.0}, *all_info;
        size_t free_bytes = 0, total_bytes = 0;
        int device = -1, roots, r;
        double required = 0.0, single_required = 0.0, resident = 0.0, available;

        MPI_Comm_size(roots_comm, &roots);
        buses = (char *)calloc((size_t)roots, sizeof(bus));
        all_info = (double *)calloc((size_t)roots * 3U, sizeof(double));
        if (buses == NULL || all_info == NULL)
            ClusterCol_AbortWithMessage("Cannot allocate GPU memory preflight metadata.");
        if (hipGetDevice(&device) != hipSuccess ||
            hipDeviceGetPCIBusId(bus, sizeof(bus), device) != hipSuccess ||
            OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != hipSuccess)
            local_fit = local_serial_fit = local_retryable = 0;
        if (local_fit) {
            int capacity = n;
            const char *ozaki = getenv("OPENMX_MAGMA_OZAKI2");
            const char *gemm = getenv("OPENMX_CLUSTER_GPU_GEMM");
            if (ctx->initialized && ctx->device_id == device) {
                double panel = (double)ctx->matrix_dim * ctx->matrix_dim * sizeof(double);
                if (ctx->matrix_dim > capacity) capacity = ctx->matrix_dim;
                info[1] = panel * ((ctx->d_S != NULL) + (ctx->d_H != NULL) +
                                   (ctx->d_tmp != NULL));
                if (ctx->d_W != NULL) info[1] += (double)ctx->matrix_dim * sizeof(double);
            }
            /* the GEMMul8 workspace of the forward transform is the capped
               one: its full workspace (openmx_gemmul8DgemmUnblocked) is taken
               only when the device has room beyond the solve reserve, and
               returned right after the transform */
            info[0] = (3.0 * capacity * capacity + 2.0 * n * n + capacity) * sizeof(double)
                    + reserve + (double)ClusterCol_RefineBytes(n, maxn);
            if (n >= 2048 && (gemm == NULL || atoi(gemm) != 0))
                info[0] += (double)openmx_gemmul8DWorkspaceSize(n, n, n);
            if (ozaki == NULL || ozaki[0] == '\0' || atoi(ozaki) != 0) {
                /* Defaults match magma_openmx.cu; honor user caps too. */
                const char *cap_env = getenv("MAGMA_OZAKI2_MAX_WS_MB");
                const char *pct_env = getenv("MAGMA_OZAKI2_MAX_WS_PERCENT");
                const char *ranks_env = getenv("MAGMA_OZAKI2_LOCAL_RANKS");
                double cap = cap_env == NULL ? 512.0 * mib : atof(cap_env) * mib;
                double pct = pct_env == NULL ? -1.0 : atof(pct_env);
                int local_ranks = ranks_env == NULL ? 4 : atoi(ranks_env);
                double percent_cap;
                if (local_ranks < 1) {
                    const char *mpi_ranks = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
                    const char *slurm_ranks = getenv("SLURM_NTASKS_PER_NODE");
                    local_ranks = mpi_ranks == NULL ? 0 : atoi(mpi_ranks);
                    if (local_ranks < 1 && slurm_ranks != NULL) local_ranks = atoi(slurm_ranks);
                    if (local_ranks < 1) local_ranks = 1;
                }
                if (pct > 100.0) pct = 100.0;
                if (cap <= 0.0 && pct < 0.0) pct = 30.0;
                percent_cap = (double)total_bytes * pct / (100.0 * local_ranks);
                if (cap <= 0.0 || (pct >= 0.0 && percent_cap < cap)) cap = percent_cap;
                info[0] += cap;
            }
            info[2] = (double)free_bytes;
        }
        MPI_Allgather(bus, sizeof(bus), MPI_CHAR, buses, sizeof(bus), MPI_CHAR, roots_comm);
        MPI_Allgather(info, 3, MPI_DOUBLE, all_info, 3, MPI_DOUBLE, roots_comm);
        available = (double)free_bytes;
        ClusterCol_owners_on_device = 0;
        for (r = 0; r < roots; ++r) {
            if (strcmp(bus, buses + r * sizeof(bus)) == 0) {
                required += all_info[3*r];
                if (single_required < all_info[3*r]) single_required = all_info[3*r];
                resident += all_info[3*r+1];
                if (all_info[3*r+2] < available) available = all_info[3*r+2];
                ClusterCol_owners_on_device++;
            }
        }
        if (available + resident < required) local_fit = 0;
        if (available + resident < single_required) local_serial_fit = 0;
        if (!local_fit && ClusterCol_GpuVerbose())
            printf("<Cluster_DFT_Col> spin group %d GPU %s: budget %.0f MiB, free+retained %.0f MiB.\n",
                   myworld1, bus, required / mib, (available + resident) / mib);
        free(buses);
        free(all_info);
        MPI_Comm_free(&roots_comm);
    }
    MPI_Comm_free(&node_comm);
    /* =2: the serialized tier without trying the concurrent one (a testing
       aid of the NVIDIA version), within the collective verdict */
    if (serial_env != NULL && atoi(serial_env) == 2 && SpinP_switch == 1) local_fit = 0;
    MPI_Allreduce(&local_fit, &fit, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
    MPI_Allreduce(&local_serial_fit, &serial_fit, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
    MPI_Allreduce(&local_retryable, retryable, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
    if (fit) {
        /* the owners reserve their matrices; any failure is everyone's
           verdict (the serialized tier, or a retry, or ELPA) */
        int reserved = 1, all_reserved = 1, now = 0;

        if (myid1 == 0) {
            reserved = ClusterCol_TryEnsureMatrixCapacity(n, &now);
            *allocated |= now;
        }
        MPI_Allreduce(&reserved, &all_reserved, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
        if (!reserved) {
            size_t free_bytes = 0, total_bytes = 0;

            (void)OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes);
            printf("<Cluster_DFT_Col> The dense owner of spin group %d could not allocate its GPU matrices "
                   "(n=%d; %.0f MiB free of %.0f MiB).\n", myworld1, n, free_bytes / mib, total_bytes / mib);
            fflush(stdout);
        }
        fit = all_reserved;
    }
    if (!fit && serial_fit && SpinP_switch == 1) {
        if (myid1 == 0 && myworld1 == 0) {
            if (2 <= level_stdout) {
                printf("<Cluster_DFT_Col> Serializing spin roots to fit the shared GPU.\n");
                fflush(stdout);
            }
        }
        return 2;
    }
    return fit;
}

static void ClusterCol_RefineConfigure(void);

static int ClusterCol_NativeGpuDiagFits(int n, int maxn, int myworld1, int myid1)
{
    int retryable = 0;
    int allocated = 0;
    int fit;

    ClusterCol_RefineConfigure();
    fit = ClusterCol_GpuDiagFitsOnce(n, maxn, myworld1, myid1, &retryable, &allocated);

    if (fit == 0 && retryable) {
        /* Completed SCF stages may leave library scratch behind. Reclaim
           only reproducible device buffers, preserving the transformed
           overlap and the original admission budget on every rank. */
        if (myid1 == 0 && myworld1 == 0 && ClusterCol_GpuVerbose()) {
            printf("<Cluster_DFT_Col> Reclaiming GPU workspaces before retrying the dense memory budget.\n");
            fflush(stdout);
        }
        Set_Density_Grid_GPU_Release_Device_Caches();
        openmx_gemmul8ReleaseWorkspaces();
        (void)openmx_magma_release_z_workspace();
        (void)openmx_hipsolver_release_workspace();
        MPI_Barrier(mpi_comm_level1);
        fit = ClusterCol_GpuDiagFitsOnce(n, maxn, myworld1, myid1, &retryable, &allocated);
    }
    if (fit == 0 && myid1 == 0) {
        /* what the reservation allocated for this step goes again: a new
           context entirely (its overlap was never transformed), else the
           scratch it gave back to a trimmed one */
        if (allocated & 2) ClusterCol_GpuSolver_FreeMatrices();
        else if (allocated & 1) {
            ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;

            if (ctx->d_H != NULL) wait_hipfunc(hipFree(ctx->d_H));
            if (ctx->d_tmp != NULL) wait_hipfunc(hipFree(ctx->d_tmp));
            ctx->d_H = ctx->d_tmp = NULL;
            ClusterCol_DeviceEvecSpin = -1;
        }
    }
    if (fit == 0 && myid1 == 0 && myworld1 == 0) {
        printf("<Cluster_DFT_Col> GPU dense buffers do not fit; checking capacity recovery.\n");
        fflush(stdout);
    }
    return fit;
}

static int ClusterCol_GpuDiagFits(int n, int maxn, int myworld1, int myid1)
{
    int numprocs0, myid0, fit;
    MPI_Comm_size(mpi_comm_level1,&numprocs0);
    MPI_Comm_rank(mpi_comm_level1,&myid0);
    ClusterCol_mgpu_host_dense=0;
    ClusterCol_ResetGs2Fallback();
    const char *enabled=getenv("OPENMX_CLUSTER_GPU_DIAG");
    if (enabled && atoi(enabled)==0 && !openmx_mgpu_host_required()) return 0;
    fit=openmx_mgpu_host_required()?0:ClusterCol_NativeGpuDiagFits(n,maxn,myworld1,myid1);
    if (!fit) {
        ClusterCol_GpuSolver_Destroy();
        openmx_gemmul8ReleaseWorkspaces();
        MPI_Barrier(mpi_comm_level1);
        int capacity=openmx_cluster_capacity_select(mpi_comm_level1,n,0,"Cluster collinear");
        if (capacity==OPENMX_CLUSTER_CAPACITY_MAGMA) {
            ClusterCol_mgpu_host_dense=1;
            fit=1;
        }
        else if (capacity==OPENMX_CLUSTER_CAPACITY_GS2) {
            int admitted;
            if (SpinP_switch==1 && numprocs0>1) {
                ClusterCol_UnionGrid_Ensure(n);
                admitted=openmx_cluster_gs2_admit(mpi_comm_level1,ClusterCol_ugrid.ictxt,n,ClusterCol_ugrid.nblk,0,"Cluster collinear");
            }
            else admitted=openmx_cluster_gs2_admit(mpi_comm_level1,ictxt1,n,nblk,0,"Cluster collinear");
            if (admitted) ClusterCol_gs2_fallback_n=n;
        }
    }
    if (fit || ClusterCol_gs2_fallback_n!=n) ClusterCol_UnionGrid_Free();
    return fit;
}


static void ClusterCol_GpuSolver_Init(void)
{
    ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;
    int                    current_device;

    wait_hipfunc(hipGetDevice(&current_device));

    if (ctx->initialized && ctx->device_id == current_device) return;
    if (ctx->initialized) ClusterCol_GpuSolver_Destroy();

    wait_hipfunc(hipStreamCreateWithFlags(&ctx->stream, hipStreamNonBlocking));
    wait_hipfunc(hipblasCreate(&ctx->hipblas));
    wait_hipfunc(hipblasSetStream(ctx->hipblas, ctx->stream));

    ctx->initialized = 1;
    ctx->device_id   = current_device;
}

static hipError_t ClusterCol_TryDeviceMalloc(void **ptr, size_t bytes);

/* the dense solver's matrices go, the stream and handle stay */
static void ClusterCol_GpuSolver_FreeMatrices(void)
{
    ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;

    if (ctx->d_S != NULL)    wait_hipfunc(hipFree(ctx->d_S));
    if (ctx->d_H != NULL)    wait_hipfunc(hipFree(ctx->d_H));
    if (ctx->d_tmp != NULL)  wait_hipfunc(hipFree(ctx->d_tmp));
    if (ctx->d_W != NULL)    wait_hipfunc(hipFree(ctx->d_W));
    ctx->d_S = ctx->d_H = ctx->d_tmp = ctx->d_W = NULL;
    ctx->matrix_dim = 0;
    ctx->transformed_s_valid = 0;
    ctx->transformed_s_dim = 0;
    ClusterCol_DeviceEvecSpin = -1;   /* d_tmp held the latest eigenvectors */
}

/* The dense solver's matrices for dimension n: 1 when they are there, 0 when
   the device has no room for them, with nothing of a partial allocation kept
   (the caller falls back instead of waiting in wait_hipfunc, ported from the
   NVIDIA version).  A context whose scratch (d_H, d_tmp) went at an SCF
   boundary (ClusterCol_TrimScratch) gets it back without losing the
   transformed overlap.  *allocated (may be NULL) receives 2 when the whole
   context was allocated anew, 1 when only the scratch was, else 0. */
static int ClusterCol_TryEnsureMatrixCapacity(int n, int *allocated)
{
    ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;
    size_t dense_count;
    size_t matrix_bytes;
    size_t vector_bytes;

    if (allocated != NULL) *allocated = 0;
    if (n<=0){
        ClusterCol_AbortWithMessage("Invalid matrix size in Cluster_DFT_Col.c GPU solver workspace.");
    }

    ClusterCol_GpuSolver_Init();

    if (n<=ctx->matrix_dim){
        if (ctx->d_H!=NULL && ctx->d_tmp!=NULL) return 1;
        dense_count = ClusterCol_CheckedMulCount((size_t)ctx->matrix_dim,(size_t)ctx->matrix_dim,"dense GPU scratch");
        matrix_bytes = ClusterCol_CheckedMulCount(dense_count,sizeof(double),"dense GPU scratch bytes");
        if ((ctx->d_H==NULL && ClusterCol_TryDeviceMalloc((void**)&ctx->d_H,matrix_bytes)!=hipSuccess) ||
            (ctx->d_tmp==NULL && ClusterCol_TryDeviceMalloc((void**)&ctx->d_tmp,matrix_bytes)!=hipSuccess)){
            if (ctx->d_H!=NULL) wait_hipfunc(hipFree(ctx->d_H));
            if (ctx->d_tmp!=NULL) wait_hipfunc(hipFree(ctx->d_tmp));
            ctx->d_H = ctx->d_tmp = NULL;
            ClusterCol_DeviceEvecSpin = -1;
            return 0;
        }
        if (allocated != NULL) *allocated = 1;
        return 1;
    }

    ClusterCol_GpuSolver_FreeMatrices();

    dense_count = ClusterCol_CheckedMulCount((size_t)n,(size_t)n,"dense GPU matrix");
    matrix_bytes = ClusterCol_CheckedMulCount(dense_count,sizeof(double),"dense GPU matrix bytes");
    vector_bytes = ClusterCol_CheckedMulCount((size_t)n,sizeof(double),"dense GPU eigenvalue bytes");

    if (ClusterCol_TryDeviceMalloc((void**)&ctx->d_S,matrix_bytes)   != hipSuccess ||
        ClusterCol_TryDeviceMalloc((void**)&ctx->d_H,matrix_bytes)   != hipSuccess ||
        ClusterCol_TryDeviceMalloc((void**)&ctx->d_tmp,matrix_bytes) != hipSuccess ||
        ClusterCol_TryDeviceMalloc((void**)&ctx->d_W,vector_bytes)   != hipSuccess){
        /* the partial reservation goes, so that the fallback path and the
           other ranks see the memory again */
        ClusterCol_GpuSolver_FreeMatrices();
        return 0;
    }

    ctx->matrix_dim = n;
    if (allocated != NULL) *allocated = 2;
    return 1;
}

/* the same where no fallback is left (the admission of the step reserved
   the matrices already, or the serialized spin mode allocates them for its
   turn): a failure ends the run with the reason */
static void ClusterCol_GpuSolver_EnsureMatrixCapacity(int n)
{
    if (!ClusterCol_TryEnsureMatrixCapacity(n,NULL)){
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        char msg[512];

        (void)OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes);
        snprintf(msg,sizeof(msg),
                 "Cluster_DFT_Col.c: out of GPU memory for the dense solver matrices of n=%d "
                 "(%.1f MiB each; %.1f MiB free of %.1f MiB). Reduce the MPI ranks sharing the device "
                 "or set OPENMX_CLUSTER_GPU_DIAG=0 for the ELPA diagonalization.",
                 n,(double)n*(double)n*sizeof(double)/(1024.0*1024.0),
                 (double)free_bytes/(1024.0*1024.0),(double)total_bytes/(1024.0*1024.0));
        ClusterCol_AbortWithMessage(msg);
    }
}

static void ClusterCol_GpuSolver_EigenDevice(double *d_A, int n, int maxn, double *ko, double *detail_slot)
{
    int info = 0;
    int mout = 0;
    char msg[256];
    double stime = 0.0;

    if (n<=0 || maxn<=0 || n<maxn){
        ClusterCol_AbortWithMessage("Invalid MAGMA eigensolver dimensions in Cluster_DFT_Col.c.");
    }

    ClusterCol_GpuSolver_EnsureMatrixCapacity(n);

    if (detail_slot != NULL && ClusterCol_detail_timers.active) {
        dtime(&stime);
    }

    wait_hipfunc(hipStreamSynchronize(ClusterCol_gpusolver_ctx.stream));
    info = openmx_magma_dsyevdx_gpu(n,maxn,d_A,ko,&mout);

    if (detail_slot != NULL && ClusterCol_detail_timers.active) {
        ClusterCol_DetailTimer_Add(detail_slot, stime);
    }

    if (info!=0){
        snprintf(msg,sizeof(msg),"magma_dsyevdx_gpu failed in Cluster_DFT_Col.c: info=%d",(int)info);
        ClusterCol_AbortWithMessage(msg);
    }
    if (mout!=maxn){
        snprintf(msg,sizeof(msg),"magma_dsyevdx_gpu returned %d eigenpairs in Cluster_DFT_Col.c, expected %d.",mout,maxn);
        ClusterCol_AbortWithMessage(msg);
    }
}

static void ClusterCol_GpuSolver_PrepareTransformedSDevice(int rebuild, int n, double *ko0)
{
    ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;
    double *old_s;
    double *new_s;

    ClusterCol_GpuSolver_EnsureMatrixCapacity(n);

    if (rebuild || !(ctx->transformed_s_valid && ctx->transformed_s_dim==n)){
        double stime = 0.0;

        ClusterCol_GpuSolver_EigenDevice(ctx->d_S,n,n,ko0+1,&ClusterCol_detail_timers.overlap_eig);

        for (int l=1; l<=n; l++){
            if (ko0[l]<1.0e-10) ko0[l] = 1.0e-10;
            ko0[l] = 1.0/sqrt(ko0[l]);
        }

        if (ClusterCol_detail_timers.active) {
            dtime(&stime);
        }

        wait_hipfunc(hipMemcpyAsync(ctx->d_W,ko0+1,sizeof(double)*(size_t)n,hipMemcpyHostToDevice,ctx->stream));

        old_s = ctx->d_S;
        new_s = ctx->d_tmp;
        if (!openmx_gpu_transform ||
            !openmx_gpu_transform(OPENMX_GPU_SCALE_COLUMNS,0,n,n,old_s,n,ctx->d_W,0,new_s,n,0,0.0))
        wait_hipfunc(hipblasDdgmm(ctx->hipblas,HIPBLAS_SIDE_RIGHT,n,n,
                                  old_s,n,ctx->d_W,1,new_s,n));

        ctx->d_S = new_s;
        ctx->d_tmp = old_s;
        ctx->transformed_s_valid = 1;
        ctx->transformed_s_dim = n;
        wait_hipfunc(hipStreamSynchronize(ctx->stream));

        if (ClusterCol_detail_timers.active) {
            ClusterCol_DetailTimer_Add(&ClusterCol_detail_timers.overlap_scale, stime);
        }
    }
}

static void ClusterCol_GpuSolver_PrepareTransformedS(int rebuild, int n, double *S, double *ko0)
{
    ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;
    size_t dense_count = ClusterCol_CheckedMulCount((size_t)n,(size_t)n,"transformed overlap");
    size_t matrix_bytes = ClusterCol_CheckedMulCount(dense_count,sizeof(double),"transformed overlap bytes");

    ClusterCol_GpuSolver_EnsureMatrixCapacity(n);
    if (rebuild || !(ctx->transformed_s_valid && ctx->transformed_s_dim==n)){
        wait_hipfunc(hipMemcpyAsync(ctx->d_S,S,matrix_bytes,hipMemcpyHostToDevice,ctx->stream));
    }
    ClusterCol_GpuSolver_PrepareTransformedSDevice(rebuild,n,ko0);
}

/* ------------------------------------------------------------------
   The refined FP32 eigensolver (eigen_refine_gpu.h) in the root dense path.

   OPENMX_EIGEN_REFINE=K (0: off), unset K = 2 on GPUs whose FP32:FP64
   throughput ratio is at least 8 when the real GEMMul8 products are on: the
   transformed Hamiltonian of each spin is solved for all n pairs in FP32
   (or starts from the previous step's vectors) and the MaxN wanted ones are
   refined by K Ogita-Aishima steps.  With two spin worlds the two owners
   exchange their FP32 estimates for a common chemical potential (the
   occupations decide which pairs the refinement rotates); a single rank
   solving both spins and the serialized spin mode keep the FP64 solver, as
   do XANES and emptied orbitals or states (the refinement's occupations are
   the plain Fermi function).  An SCF that ends on a refined step has that
   step's Hamiltonian solved once more in FP64 (Cluster_DFT_Col_RefineForceFP64
   around a repeated "scf" call, which recomputes the density matrix as
   well), and the last step scf.maxIter allows is FP64 in any case.  Ported
   from the NVIDIA version (openmx4.0_gpu, Cluster_DFT_Col.c). */
typedef struct
{
    int     iterations;    /* -1 until the environment is read */
    double  until;         /* printed NormRD below which the FP64 eigensolver takes over; 0 off */
    int     final_stage;   /* the FP64 eigensolver for the rest of the SCF cycle */
    int     last_refined;  /* the latest SCF eigensolve was a refined one (any spin) */
    int     cycle_refined; /* some eigensolve of this SCF cycle was */
    int     defaulted;     /* on because of the device, not of OPENMX_EIGEN_REFINE */
    int     force_fp64;    /* Cluster_DFT_Col_RefineForceFP64: this solve in FP64 whatever the stage */
    int     scf_mode;      /* the solve in progress is an SCF one (mode "scf") */
    int     transient;     /* the first step of a cycle: its failures do not condemn the cycle */
    int     spin_mode;     /* 0: one spin, 1: two spin worlds exchange their estimates, 2: unsupported */
    int     peer;          /* the other spin's owner in mpi_comm_level1 (spin_mode 1) */
    int     solved[2];     /* this step's solves: refined (1) or not (0), per spin */
    int     persistent;    /* the strongest persistent failure of this step */
    int     first_all;     /* OPENMX_EIGEN_REFINE_FIRST_ALL of the global root: the first SCF step refines all n */
} ClusterColRefineStage;

static ClusterColRefineStage ClusterCol_refine = {-1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, -1, {0, 0}, 0, 0};
static EigenRefineState      ClusterCol_refine_state[2];
static double               *ClusterCol_refine_x[2] = {NULL, NULL};   /* the bases, n x n each */
static int                   ClusterCol_refine_x_n = 0;
/* the bases parked on the host while the scratch is given back between the
   SCF steps (ClusterCol_TrimScratch), n x n each; valid while the state's
   basis is and its device copy is gone */
static double               *ClusterCol_refine_h[2] = {NULL, NULL};
static int                   ClusterCol_refine_h_valid[2] = {0, 0};

static void ClusterCol_RefineDropParkedBases(void)
{
    for (int spin = 0; spin < 2; spin++) {
        free(ClusterCol_refine_h[spin]);
        ClusterCol_refine_h[spin] = NULL;
        ClusterCol_refine_h_valid[spin] = 0;
    }
}

static void ClusterCol_RefineRelease(void)
{
    EigenRefineDevice none;

    for (int spin = 0; spin < 2; spin++) {
        openmx_eigen_refine_state_release(&ClusterCol_refine_state[spin]);
        if (ClusterCol_refine_x[spin] != NULL) wait_hipfunc(hipFree(ClusterCol_refine_x[spin]));
        ClusterCol_refine_x[spin] = NULL;
    }
    ClusterCol_refine_x_n = 0;
    ClusterCol_RefineDropParkedBases();
    /* and the refinement module's hipSOLVER handle, as the band solvers and
       the NVIDIA version do (created again on demand) */
    memset(&none, 0, sizeof(none));
    openmx_eigen_refine_device_release(&none);
}

/* The run boundary (Cluster_DFT_Col_Release_GPU_Caches, every rank): the
   stage goes back to its initial state, so that the next -runtest/-runtestL
   input reads OPENMX_EIGEN_REFINE and the device default again instead of
   inheriting what a native fallback of the products (iterations 0 for the
   rest of the run) or a stage of this run left. */
static void ClusterCol_RefineResetStage(void)
{
    static const ClusterColRefineStage initial = {-1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, -1, {0, 0}, 0, 0};

    ClusterCol_refine = initial;
}

/* the warm-start bases leave the device with the scratch: those still
   valid wait on the host for the next refined solve of their spin */
static void ClusterCol_RefineParkBases(void)
{
    size_t const count = (size_t)ClusterCol_refine_x_n * (size_t)ClusterCol_refine_x_n;

    for (int spin = 0; spin < 2; spin++) {
        if (ClusterCol_refine_x[spin] == NULL) continue;
        if (ClusterCol_refine_state[spin].basis_valid && ClusterCol_refine_state[spin].basis_n == ClusterCol_refine_x_n) {
            if (ClusterCol_refine_h[spin] == NULL)
                ClusterCol_refine_h[spin] = (double *)ClusterCol_MallocArray(count, sizeof(double),
                                                                             "refined basis host copy");
            wait_hipfunc(hipMemcpy(ClusterCol_refine_h[spin], ClusterCol_refine_x[spin], count * sizeof(double),
                                   hipMemcpyDeviceToHost));
            ClusterCol_refine_h_valid[spin] = 1;
        }
        wait_hipfunc(hipFree(ClusterCol_refine_x[spin]));
        ClusterCol_refine_x[spin] = NULL;
    }
}

static hipError_t ClusterCol_TryDeviceMalloc(void **ptr, size_t bytes)
{
    hipError_t status = hipMalloc(ptr, (bytes == 0) ? 1 : bytes);

    if (status != hipSuccess) {
        *ptr = NULL;
        (void)hipGetLastError();
    }
    return status;
}

static void *ClusterCol_RefineHostMalloc(size_t count, size_t elem, const char *label)
{
    void *ptr = ClusterCol_MallocArray(count, elem, label);

    memset(ptr, 0, (count * elem == 0) ? 1 : count * elem);
    return ptr;
}

/* every rank, once: OPENMX_EIGEN_REFINE, or the default the global root's
   device implies, for all ranks alike */
static void ClusterCol_RefineConfigure(void)
{
    ClusterColRefineStage *st = &ClusterCol_refine;
    int myid, config[3] = {0, 0, 0};

    /* Input files in -runtest share a process; do not cache an input veto. */
    if (!openmx_fp32_enabled()) return;
    if (0 <= st->iterations) return;
    MPI_Comm_rank(mpi_comm_level1, &myid);
    if (myid == Host_ID) {
        const char *all = getenv("OPENMX_EIGEN_REFINE_FIRST_ALL");

        config[0] = openmx_eigen_refine_configure(0, 2 <= level_stdout, &config[1]);
        /* the states of the first SCF step: the same MaxN on every rank */
        config[2] = (all != NULL && atoi(all) != 0);
    }
    MPI_Bcast(config, 3, MPI_INT, Host_ID, mpi_comm_level1);
    st->iterations = config[0];
    st->defaulted = (0 < config[1]);
    st->first_all = config[2];
    st->until = openmx_eigen_refine_until();
}

static int ClusterCol_RefineIterations(void)
{
    return (openmx_fp32_enabled() && 0 < ClusterCol_refine.iterations) ? ClusterCol_refine.iterations : 0;
}

/* what the refined solve of maxn wanted states adds to the dense owner's
   device memory: the basis, the FP32 solver's workspace, and its product
   blocks where they exceed the room the budget gives the FP64 eigensolver's
   device work array (two panels, idle while the refined solve replaces the
   FP64 one; blocks that fit the FP32 scratch d_tmp take no room at all).
   Nothing when the refinement is off or the problem too small for it. */
static size_t ClusterCol_RefineBytes(int n, int maxn)
{
    size_t bytes, blocks, fp64_work;

    if (ClusterCol_RefineIterations() <= 0 || !openmx_eigen_refine_size_ok(n)) return 0;
    bytes = sizeof(double) * (size_t)n * (size_t)n + openmx_eigen_refine_fp32_work_bytes(0, n);
    blocks = openmx_eigen_refine_product_bytes(0, n, maxn);
    fp64_work = 2 * sizeof(double) * (size_t)n * (size_t)n;
    if (fp64_work < blocks) bytes += blocks - fp64_work;
    return bytes;
}

/* every rank, at the start of the root dense path of a solve */
static void ClusterCol_RefineBeginSolve(int SCF_iter, int spin_mode, int peer)
{
    ClusterColRefineStage *st = &ClusterCol_refine;

    ClusterCol_RefineConfigure();
    if (st->scf_mode && SCF_iter == 1) {
        st->final_stage = 0;
        st->cycle_refined = 0;
        ClusterCol_refine_state[0].basis_valid = 0;   /* a new cycle may come with a new geometry */
        ClusterCol_refine_state[1].basis_valid = 0;
    }
    st->transient = (SCF_iter == 1);
    st->spin_mode = spin_mode;
    st->peer = peer;
    st->solved[0] = st->solved[1] = 0;
    st->persistent = 0;
    /* the occupations of the refinement are the plain Fermi function of
       one chemical potential: XANES and emptied orbitals or states keep
       the FP64 eigensolver */
    /* only the SCF loop of the cluster solver itself: the post-SCF PAO
       diagonalization of the Cluster-LNO solver (Solver 12, DFT.c) is a
       final solve with no FP64 re-solve after it */
    st->last_refined = (st->scf_mode && Solver == 2 && 0 < ClusterCol_RefineIterations() && !st->final_stage &&
                        !st->force_fp64 && Cnt_switch == 0 && SCF_iter < DFTSCF_loop && spin_mode != 2 &&
                        xanes_calc == 0 && empty_occupation_flag == 0 && empty_states_flag == 0);
}

/* every rank, at the end of the root dense path: the solves of the owners
   decide the stage for all ranks alike */
static void ClusterCol_RefineEndSolve(void)
{
    ClusterColRefineStage *st = &ClusterCol_refine;
    int mine[2], all[2];

    mine[0] = (st->solved[0] || st->solved[1]) ? 1 : 0;
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

int Cluster_DFT_Col_RefineLastRefined(void)
{
    return ClusterCol_refine.last_refined;
}

void Cluster_DFT_Col_RefineForceFP64(int on)
{
    ClusterCol_refine.force_fp64 = on;
}

/* DFT.c, after the mixing of an SCF step that goes on: normrd is the printed
   NormRD */
void Cluster_DFT_Col_RefineAfterMixing(double normrd)
{
    ClusterColRefineStage *st = &ClusterCol_refine;
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

/* Fermi occupation of the collinear solver at the chemical potential mu */
typedef struct
{
    double mu;
} ClusterColOccupationCtx;

static double ClusterCol_RefineFermi(double e, int index, void *ctx)
{
    double x = (e - ((ClusterColOccupationCtx *)ctx)->mu) * Beta;

    (void)index;
    if (x <= -60.0) x = -60.0;
    if (60.0 <= x) x = 60.0;
    return 1.0 / (1.0 + exp(x));
}

/* The chemical potential for TZ - system_charge electrons over the lowest
   maxn states of each spin (e_other NULL: one spin of occupancy 2), as the
   solver's search sets it, and the occupations f[0..n) of the states of e */
static void ClusterCol_RefineOccupations(const double *e, const double *e_other, int n, int maxn, double *f,
                                         double *mu_out)
{
    double const occupancy = (e_other == NULL) ? 2.0 : 1.0;
    double TZ = 0.0, lo = e[0] - 1.0, hi = e[maxn - 1] + 1.0, mu = 0.0;
    ClusterColOccupationCtx ctx;

    for (int i = 1; i <= atomnum; i++) TZ += Spe_Core_Charge[WhatSpecies[i]];
    if (e_other != NULL) {
        lo = fmin(lo, e_other[0] - 1.0);
        hi = fmax(hi, e_other[maxn - 1] + 1.0);
    }
    for (int it = 0; it < 200; it++) {
        double count = 0.0;

        mu = 0.5 * (lo + hi);
        ctx.mu = mu;
        for (int i = 0; i < maxn; i++) count += occupancy * ClusterCol_RefineFermi(e[i], i + 1, &ctx);
        if (e_other != NULL)
            for (int i = 0; i < maxn; i++) count += ClusterCol_RefineFermi(e_other[i], i + 1, &ctx);
        if (0.0 <= (TZ - count) - system_charge) lo = mu;
        else hi = mu;
    }
    ctx.mu = mu;
    for (int i = 0; i < n; i++) f[i] = ClusterCol_RefineFermi(e[i], i + 1, &ctx);
    *mu_out = mu;
}

/* The refined solve of spin's transformed Hamiltonian d_H (n x n, symmetric
   in its lower triangle) on its owner.  Two spin worlds exchange whether
   their prepares succeeded and their FP32 estimates for the chemical
   potential; a failure on either side makes both solve in FP64 this step.
   Returns 1 when the refined vectors are in the first maxn columns of d_H
   and ko_spin[1..maxn] holds their eigenvalues, else 0 with d_H still the
   symmetric matrix. */
static int ClusterCol_RefinedEigenDevice(int spin, int n, int maxn, double *ko_spin)
{
    ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;
    ClusterColRefineStage *st = &ClusterCol_refine;
    EigenRefineDevice dev;
    EigenRefineProblem pb;
    EigenRefineReport rep;
    ClusterColOccupationCtx occ = {0.0};
    size_t const nn = (size_t)n * (size_t)n;
    double *e0, *f0, *e_peer = NULL;
    int status[2] = {0, 0}, rc;

    if (ClusterCol_refine_x_n != n) {
        for (int sp = 0; sp < 2; sp++) {
            if (ClusterCol_refine_x[sp] != NULL) wait_hipfunc(hipFree(ClusterCol_refine_x[sp]));
            ClusterCol_refine_x[sp] = NULL;
            ClusterCol_refine_state[sp].basis_valid = 0;
        }
        ClusterCol_RefineDropParkedBases();
        ClusterCol_refine_x_n = n;
    }
    rc = 1;
    if (ctx->d_info == NULL && ClusterCol_TryDeviceMalloc((void **)&ctx->d_info, sizeof(int32_t)) != hipSuccess) rc = 0;
    if (rc && ClusterCol_refine_x[spin] == NULL) {
        if (ClusterCol_TryDeviceMalloc((void **)&ClusterCol_refine_x[spin], nn * sizeof(double)) != hipSuccess) rc = 0;
        else if (ClusterCol_refine_h_valid[spin] && ClusterCol_refine_state[spin].basis_valid &&
                 ClusterCol_refine_state[spin].basis_n == n) {
            /* the basis the scratch trim parked on the host */
            wait_hipfunc(hipMemcpy(ClusterCol_refine_x[spin], ClusterCol_refine_h[spin], nn * sizeof(double),
                                   hipMemcpyHostToDevice));
            ClusterCol_refine_h_valid[spin] = 0;
        }
        else {
            ClusterCol_refine_state[spin].basis_valid = 0;
        }
    }
    if (!rc) {
        if (2 <= level_stdout) {
            printf("<Cluster_DFT_Col> no room for the %.1f MiB basis of the refined eigensolver; FP64 eigensolver%s\n",
                   (double)(nn * sizeof(double)) / (1024.0 * 1024.0), st->transient ? "" : " for the rest of the SCF cycle");
            fflush(stdout);
        }
        if (!st->transient && st->persistent < 1) st->persistent = 1;
    }

    memset(&dev, 0, sizeof(dev));
    dev.hipblas = ctx->hipblas;
    dev.d_work = &ctx->d_work;
    dev.d_work_bytes = &ctx->d_work_bytes;
    dev.d_info = ctx->d_info;
    dev.try_malloc = ClusterCol_TryDeviceMalloc;
    dev.host_malloc = ClusterCol_RefineHostMalloc;
    /* the two spin worlds' owners refine at the same time on a shared GPU:
       the products keep to the capped GEMMul8 workspace */
    dev.shared = (st->spin_mode == 1);

    memset(&pb, 0, sizeof(pb));
    memset(&rep, 0, sizeof(rep));
    pb.cplx = 0;
    pb.n = n;
    pb.maxn = maxn;
    pb.iterations = ClusterCol_RefineIterations();
    pb.a = ctx->d_H;
    pb.w = ctx->d_W;
    pb.fp32 = ctx->d_tmp;
    pb.region = nn * sizeof(double);
    pb.x = ClusterCol_refine_x[spin];
    pb.transient = st->transient;
    pb.warm = (rc && openmx_eigen_refine_warm_enabled() && ClusterCol_refine_state[spin].basis_valid &&
               ClusterCol_refine_state[spin].basis_n == n && !st->transient);
    pb.defaulted = st->defaulted;
    pb.occupation = ClusterCol_RefineFermi;
    pb.occupation_ctx = &occ;

    e0 = (double *)ClusterCol_MallocArray((size_t)n, sizeof(double), "refinement eigenvalues");
    f0 = (double *)ClusterCol_MallocArray((size_t)n, sizeof(double), "refinement occupations");
    if (rc) {
        if (3 <= level_stdout) {
            printf("<Cluster_DFT_Col> spin %d: %s, %d refinement steps (OPENMX_EIGEN_REFINE)\n", spin,
                   pb.warm ? "warm start from the previous vectors" : "FP32 eigensolver", pb.iterations);
            fflush(stdout);
        }
        rc = openmx_eigen_refine_prepare(&ClusterCol_refine_state[spin], &dev, &pb, e0, &rep);
        if (rc < 0) {
            pb.warm = 0;
            rc = openmx_eigen_refine_prepare(&ClusterCol_refine_state[spin], &dev, &pb, e0, &rep);
        }
        if (rc != 1 && st->persistent < rep.persistent) st->persistent = rep.persistent;
    }
    status[0] = rc;
    if (st->spin_mode == 1) {
        /* the other spin's estimates, and whether its solve is on */
        MPI_Status mpi_status;

        e_peer = (double *)ClusterCol_MallocArray((size_t)n, sizeof(double), "peer eigenvalues");
        MPI_Sendrecv(status, 1, MPI_INT, st->peer, 2101, &status[1], 1, MPI_INT, st->peer, 2101, mpi_comm_level1,
                     &mpi_status);
        /* A failed prepare may leave e0 uninitialized.  Both owners know
           both status values now, so they skip this exchange together. */
        if (status[0] == 1 && status[1] == 1) {
            MPI_Sendrecv(e0, n, MPI_DOUBLE, st->peer, 2102, e_peer, n, MPI_DOUBLE, st->peer, 2102,
                         mpi_comm_level1, &mpi_status);
        }
        if (status[0] == 1 && status[1] != 1) {
            if (2 <= level_stdout) {
                printf("<Cluster_DFT_Col> spin %d: the other spin's solve is not refined this step; FP64 eigensolver\n",
                       spin);
                fflush(stdout);
            }
            openmx_eigen_refine_abandon(&ClusterCol_refine_state[spin]);
            rc = 0;
        }
    }
    if (rc == 1) {
        ClusterCol_RefineOccupations(e0, st->spin_mode == 1 ? e_peer : NULL, n, maxn, f0, &occ.mu);
        rc = openmx_eigen_refine_finish(&ClusterCol_refine_state[spin], &dev, &pb, f0, &rep);
        if (rc != 1 && st->persistent < rep.persistent) st->persistent = rep.persistent;
    }
    free(e0);
    free(f0);
    free(e_peer);
    if (rc == 1) {
        wait_hipfunc(hipMemcpy(ko_spin + 1, ctx->d_W, sizeof(double) * (size_t)maxn, hipMemcpyDeviceToHost));
        if (3 <= level_stdout) {
            printf("<Cluster_DFT_Col> spin %d: refinement step %d of %d columns: max |r| %.1e, max |s| %.1e, delta "
                   "%.1e; Rayleigh-Ritz in %d partially occupied clusters (largest %d, chained within %.1e)\n",
                   spin, pb.iterations, rep.columns, rep.max_r, rep.max_s, rep.delta, rep.rr_clusters, rep.rr_largest,
                   rep.chain);
            fflush(stdout);
        }
        st->solved[spin] = 1;
        return 1;
    }
    ClusterCol_refine_state[spin].basis_valid = 0;
    return 0;
}

/* OPENMX_GEMMUL8_FORWARD_UNBLOCKED (the NVIDIA name and parsing: on unless
   the value starts with a character other than '1'): the two products of
   the forward transform take the full GEMMul8 workspace when the device has
   room for it; 0 restores the capped products of the blocked path. */
static int ClusterCol_ForwardUnblocked(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("OPENMX_GEMMUL8_FORWARD_UNBLOCKED");

        cached = (value == NULL || value[0] == '\0' || value[0] == '1') ? 1 : 0;
    }
    return cached;
}

/* OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE (unset: OPENMX_BAND_GEMMUL8_TURN_RELEASE;
   unset too: 0 on an APU, 1 otherwise, as the turns of Band_DFT_Col): at the
   end of each root dense solve the owner releases all of its GEMMul8
   workspaces (1), so that the grid stages of the SCF step find that memory
   on a GPU shared by many ranks, or only the scratch above the cap of the
   blocked path (0), keeping the capped workspace for the next solve. */
static int ClusterCol_GemmWorkspaceTurnRelease(void)
{
    const char *value = getenv("OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE");

    if (value == NULL) value = getenv("OPENMX_BAND_GEMMUL8_TURN_RELEASE");
    if (value == NULL) return openmx_gpu_is_apu() ? 0 : 1;
    return (atoi(value) != 0);
}

/* the owner, after the solves of a root dense path (their streams done) */
static void ClusterCol_GemmWorkspaceTurnEnd(void)
{
    const int    release = ClusterCol_GemmWorkspaceTurnRelease();
    const size_t held = openmx_gemmul8WorkspaceBytes();

    if (release) openmx_gemmul8ReleaseWorkspaces();
    else openmx_gemmul8TrimWorkspaces();
    if (ClusterCol_GpuVerbose()) {
        printf("<Cluster_DFT_Col> end of the dense solve: %s GEMMul8 workspaces, %.1f of %.1f MiB returned "
               "(OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE=%d)\n", release ? "released the" : "trimmed the",
               (double)(held - openmx_gemmul8WorkspaceBytes()) / (1024.0 * 1024.0),
               (double)held / (1024.0 * 1024.0), release);
        fflush(stdout);
    }
}

/* OPENMX_CLUSTER_GPU_RETAIN_SCRATCH, after an SCF solve (mode "scf"; its
   density matrix is done by then): unset or nonzero keeps the owner's
   scratch for the next step (the default here: the admission of every step
   counts it as resident, and Set_Density_Grid_GPU returns only library
   workspaces under pressure); 0 returns it after every SCF step; "auto"
   returns it when it is at least 512 MiB and less than half of the device is
   free (the default of the NVIDIA version).  The scratch is d_H, d_tmp
   (whose latest eigenvectors the density matrix has used), the refined
   solve's device workspace, and the warm-start bases, which wait on the
   host for the next step (ClusterCol_RefineParkBases); the transformed
   overlap d_S and d_W stay, and the admission of the next step allocates the
   scratch again (ClusterCol_TryEnsureMatrixCapacity). */
static void ClusterCol_TrimScratch(void)
{
    ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;
    const char *retain = getenv("OPENMX_CLUSTER_GPU_RETAIN_SCRATCH");
    const int automatic = (retain != NULL && strcasecmp(retain, "auto") == 0);
    size_t panel, bytes, free_bytes = 0, total_bytes = 0;
    int device = -1;

    if (retain == NULL || retain[0] == '\0' || (!automatic && atoi(retain) != 0)) return;
    if (!ctx->initialized || ctx->matrix_dim <= 0) return;
    if (hipGetDevice(&device) != hipSuccess || device != ctx->device_id) return;
    panel = sizeof(double) * (size_t)ctx->matrix_dim * (size_t)ctx->matrix_dim;
    bytes = (ctx->d_H != NULL ? panel : 0) + (ctx->d_tmp != NULL ? panel : 0) + ctx->d_work_bytes;
    for (int spin = 0; spin < 2; spin++) {
        if (ClusterCol_refine_x[spin] != NULL)
            bytes += sizeof(double) * (size_t)ClusterCol_refine_x_n * (size_t)ClusterCol_refine_x_n;
    }
    if (bytes == 0) return;
    if (automatic && (bytes < 512ULL * 1024ULL * 1024ULL || OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != 0 ||
                      total_bytes / 2 <= free_bytes)) return;

    wait_hipfunc(hipStreamSynchronize(ctx->stream));
    ClusterCol_RefineParkBases();
    if (ctx->d_H != NULL) wait_hipfunc(hipFree(ctx->d_H));
    if (ctx->d_tmp != NULL) wait_hipfunc(hipFree(ctx->d_tmp));
    if (ctx->d_work != NULL) wait_hipfunc(hipFree(ctx->d_work));
    ctx->d_H = ctx->d_tmp = NULL;
    ctx->d_work = NULL;
    ctx->d_work_bytes = 0;
    ClusterCol_DeviceEvecSpin = -1;   /* the density matrix used them already */
    if (ClusterCol_GpuVerbose()) {
        printf("<Cluster_DFT_Col> Released %.1f MiB of SCF scratch; retained the transformed overlap (n=%d).\n",
               (double)bytes / (1024.0 * 1024.0), ctx->matrix_dim);
        fflush(stdout);
    }
}

/* The eigenpairs of the transformed Hamiltonian in d_H and their back
   transform into C, on the owner.  shared: another dense owner solves on
   this device at the same time (two spin worlds on one GPU), so the forward
   transform keeps to the capped GEMMul8 workspace, as the refined solve's
   products do (EigenRefineDevice.shared). */
static void ClusterCol_GpuSolver_SolveHamiltonianDevice(int n, int maxn, double *ko_spin, double *C, int spin,
                                                        int shared)
{
    /* OPENMX_CLUSTER_PROFILE=1: besides the CLUSTER_PROFILE line of each SCF
       step, the cumulative wall time of the phases of this rank's solves,
       one CLUSTERPROF line per solve in the form of the NVIDIA version
       (forward transform, eigensolver, back transform, eigenvector download;
       its error indicator does not exist here and stays 0).  The stream is
       synchronized around each phase. */
    static double prof[4] = {0.0, 0.0, 0.0, 0.0};
    static long prof_solves = 0;
    const ClusterColDetailTimers before = ClusterCol_detail_timers;
    ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;
    size_t dense_count = ClusterCol_CheckedMulCount((size_t)n,(size_t)n,"Hamiltonian");
    size_t evec_count = ClusterCol_CheckedMulCount((size_t)n,(size_t)maxn,"eigenvectors");
    size_t evec_bytes = ClusterCol_CheckedMulCount(evec_count,sizeof(double),"eigenvector bytes");
    double alpha = 1.0;
    double beta = 0.0;
    double stime = 0.0;
    const char *gemm_env = getenv("OPENMX_CLUSTER_GPU_GEMM");
    /* Small products do not amortize the INT8 conversion/launch overhead.
       The bridge retains scf.gemmul8.enable and its memory-safe fallback. */
    int use_gemm = n >= 2048 && (gemm_env == NULL || atoi(gemm_env) != 0);
    /* The forward products take the full GEMMul8 workspace when the device
       has room for it beyond the solve reserve (openmx_gemmul8DgemmUnblocked,
       which otherwise takes the capped workspace at the same precision):
       under the 256 MiB cap every block of a blocked product rescales its
       operands (in the NVIDIA version the transform went from 810 to 421 ms
       at n = 13000 on an A100).  The results are those of the capped
       products bit for bit while the full workspace fits the cap.  The
       scratch goes right after the transform: the eigensolvers allocate
       their own workspaces (MAGMA inside its call, rocSOLVER in the refined
       solve), which the memory admission budgets without it. */
    int unblocked = use_gemm && !shared && ClusterCol_ForwardUnblocked();

    if (!(ctx->transformed_s_valid && ctx->transformed_s_dim==n)){
        ClusterCol_AbortWithMessage("Transformed overlap is not ready in Cluster_DFT_Col.c.");
    }

    if (ClusterCol_detail_timers.active) dtime(&stime);
    if (use_gemm) {
        wait_hipfunc(ClusterCol_SymmetrizeLower_HIP(n,ctx->d_H,ctx->stream));
        if (unblocked) {
            wait_hipfunc(openmx_gemmul8DgemmUnblocked(ctx->hipblas,HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,
                                                      &alpha,ctx->d_H,n,ctx->d_S,n,&beta,ctx->d_tmp,n));
            wait_hipfunc(openmx_gemmul8DgemmUnblocked(ctx->hipblas,HIPBLAS_OP_T,HIPBLAS_OP_N,n,n,n,
                                                      &alpha,ctx->d_S,n,ctx->d_tmp,n,&beta,ctx->d_H,n));
            wait_hipfunc(hipStreamSynchronize(ctx->stream));
            openmx_gemmul8TrimWorkspaces();
        }
        else {
            wait_hipfunc(openmx_gemmul8Dgemm(ctx->hipblas,HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,
                                             &alpha,ctx->d_H,n,ctx->d_S,n,&beta,ctx->d_tmp,n));
            wait_hipfunc(openmx_gemmul8Dgemm(ctx->hipblas,HIPBLAS_OP_T,HIPBLAS_OP_N,n,n,n,
                                             &alpha,ctx->d_S,n,ctx->d_tmp,n,&beta,ctx->d_H,n));
        }
    }
    else {
        wait_hipfunc(hipblasDsymm(ctx->hipblas,HIPBLAS_SIDE_LEFT,HIPBLAS_FILL_MODE_LOWER,n,n,
                                  &alpha,ctx->d_H,n,ctx->d_S,n,&beta,ctx->d_tmp,n));
        wait_hipfunc(hipblasDgemm(ctx->hipblas,HIPBLAS_OP_T,HIPBLAS_OP_N,n,n,n,
                                  &alpha,ctx->d_S,n,ctx->d_tmp,n,&beta,ctx->d_H,n));
    }
    if (ClusterCol_detail_timers.active) {
        wait_hipfunc(hipStreamSynchronize(ctx->stream));
        ClusterCol_DetailTimer_Add(&ClusterCol_detail_timers.h_transform_gemm, stime);
    }

    {
        int refined = 0;

        if (0 <= spin && ClusterCol_refine.last_refined) {
            double rstime = 0.0;

            if (ClusterCol_detail_timers.active) dtime(&rstime);
            refined = ClusterCol_RefinedEigenDevice(spin, n, maxn, ko_spin);
            if (ClusterCol_detail_timers.active) ClusterCol_DetailTimer_Add(&ClusterCol_detail_timers.syevdx, rstime);
            if (!refined) {
                /* the products' full-size scratch goes before the FP64 solve */
                wait_hipfunc(hipStreamSynchronize(ctx->stream));
                openmx_gemmul8TrimWorkspaces();
            }
        }
        if (!refined) ClusterCol_GpuSolver_EigenDevice(ctx->d_H,n,maxn,ko_spin+1,&ClusterCol_detail_timers.syevdx);
    }

    if (ClusterCol_detail_timers.active) dtime(&stime);
    if (use_gemm) {
        wait_hipfunc(openmx_gemmul8Dgemm(ctx->hipblas,HIPBLAS_OP_T,HIPBLAS_OP_T,maxn,n,n,
                                      &alpha,ctx->d_H,n,ctx->d_S,n,&beta,ctx->d_tmp,maxn));
    }
    else {
        wait_hipfunc(hipblasDgemm(ctx->hipblas,HIPBLAS_OP_T,HIPBLAS_OP_T,maxn,n,n,
                                  &alpha,ctx->d_H,n,ctx->d_S,n,&beta,ctx->d_tmp,maxn));
    }
    if (ClusterCol_detail_timers.active) {
        wait_hipfunc(hipStreamSynchronize(ctx->stream));
        ClusterCol_DetailTimer_Add(&ClusterCol_detail_timers.backtransform, stime);
        dtime(&stime);
    }
    wait_hipfunc(hipMemcpyAsync(C,ctx->d_tmp,evec_bytes,hipMemcpyDeviceToHost,ctx->stream));
    wait_hipfunc(hipStreamSynchronize(ctx->stream));
    if (ClusterCol_detail_timers.active) {
        const ClusterColDetailTimers *t = &ClusterCol_detail_timers;
        int rank;

        ClusterCol_DetailTimer_Add(&ClusterCol_detail_timers.evec_copyout, stime);
        prof[0] += t->h_transform_gemm - before.h_transform_gemm;
        prof[1] += t->syevdx - before.syevdx;
        prof[2] += t->backtransform - before.backtransform;
        prof[3] += t->evec_copyout - before.evec_copyout;
        prof_solves++;
        MPI_Comm_rank(mpi_comm_level1,&rank);
        printf("CLUSTERPROF rank=%d solves=%ld n=%d forward=%.6f eigen=%.6f back=%.6f evec_d2h=%.6f "
               "indicator=%.6f (s, cumulative)\n",
               rank,prof_solves,n,prof[0],prof[1],prof[2],prof[3],0.0);
        fflush(stdout);
    }
}

static void ClusterCol_GpuSolver_SolveHamiltonian(int n, int maxn, const double *H, double *ko_spin, double *C)
{
    ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;
    size_t dense_count = ClusterCol_CheckedMulCount((size_t)n,(size_t)n,"Hamiltonian");
    size_t matrix_bytes = ClusterCol_CheckedMulCount(dense_count,sizeof(double),"Hamiltonian bytes");

    ClusterCol_GpuSolver_EnsureMatrixCapacity(n);
    wait_hipfunc(hipMemcpyAsync(ctx->d_H,H,matrix_bytes,hipMemcpyHostToDevice,ctx->stream));
    ClusterCol_GpuSolver_SolveHamiltonianDevice(n,maxn,ko_spin,C,-1,0);
}

static void ClusterCol_GEMMul8Dgemm_OpenMP(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k,
                                            double const * A, double const * B, double * C)
{
    ClusterCol_GpuSolver_Init();
#pragma omp target data map(present, alloc: A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma omp target data use_device_ptr(A, B, C)
    {
        double const alpha = 1.0;
        double const beta  = 0.0;
        wait_hipfunc(openmx_gemmul8Dgemm(ClusterCol_gpusolver_ctx.hipblas, transa, transb, m, n, k, &alpha, A, m, B, k, &beta, C, m));
    }
}

static void ClusterCol_GpuSolver_CheckInfo(const char *where, int info)
{
    if (info != 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s failed in Cluster_DFT_Col.c: info=%d", where, info);
        ClusterCol_AbortWithMessage(msg);
    }
}

/* every rank, at the first SCF step of the root dense path: the states it
   solves.  The FP64 solve takes all n, as it always has; a refined one,
   whose FP32 solve yields all n eigenvalues anyway, refines the lowest
   (TZ - system_charge)/2 + max(400, 0.1 n) only -- the margin of the later
   steps -- which keeps its product blocks within the scratch d_tmp of the
   later steps instead of two n x n blocks of their own, and its products a
   fraction of n^3 (as the non-collinear solver does).
   OPENMX_EIGEN_REFINE_FIRST_ALL=1 (the global root's) refines all n.  The
   conditions are those of ClusterCol_RefineBeginSolve; serialized: the
   serialized spin mode, which keeps the FP64 eigensolver.  Collective on
   mpi_comm_level1: OPENMX_EIGEN_REFINE_MIN_N is read on each rank, so the
   ranks agree on the largest count, which sets MaxN and the partitions. */
static int ClusterCol_RefineFirstStepStates(const char *mode, int n, double TZ, int SpinP_switch, int numprocs0,
                                            int serialized)
{
    double lumos = 0.1 * (double)n;
    int states = n, all_states = n;

    ClusterCol_RefineConfigure();
    if (!(strcasecmp(mode, "scf") != 0 || Solver != 2 || ClusterCol_RefineIterations() <= 0 ||
          ClusterCol_refine.force_fp64 || Cnt_switch != 0 || DFTSCF_loop <= 1 || cal_partial_charge ||
          xanes_calc != 0 || empty_occupation_flag != 0 || empty_states_flag != 0 || serialized ||
          (SpinP_switch == 1 && numprocs0 == 1) || !openmx_eigen_refine_size_ok(n) ||
          ClusterCol_refine.first_all)) {
        if (lumos < 400.0) lumos = 400.0;
        states = (int)((TZ - system_charge) / 2.0) + (int)lumos;
        if (n < states) states = n;
    }
    MPI_Allreduce(&states, &all_states, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
    return all_states;
}

/* the states of a solve of the root dense path: first_states at the first
   SCF step (ClusterCol_RefineFirstStepStates), the occupied ones and a
   margin afterwards */
static int ClusterCol_GpuPathMaxN(int SCF_iter, const char *mode, double TZ, int n, int first_states)
{
    double lumos;
    int maxn;

    if (SCF_iter==1) return first_states;

    if      (strcasecmp(mode,"scf")==0)     lumos = (double)n*0.100;
    else if (strcasecmp(mode,"dos")==0)     lumos = (double)n*0.200;
    else if (strcasecmp(mode,"lcaoout")==0) lumos = (double)n*0.200;
    else if (strcasecmp(mode,"xanes")==0)   lumos = (double)n*0.200;
    else if (strcasecmp(mode,"diag")==0)    lumos = (double)n*0.200;
    else                                    lumos = (double)n*0.200;

    if (lumos<400.0) lumos = 400.0;
    maxn = (int)((TZ-system_charge)/2.0) + (int)lumos;
    if (n<maxn) maxn = n;
    if (cal_partial_charge) maxn = n;
    return maxn;
}

static void ClusterCol_SetMaxNAndPartitions(int SCF_iter, const char *mode, double TZ, int n, int numprocs1,
                                            int first_states, int *MaxN, int *is2, int *ie2)
{
    int ID;
    double av_num;

    *MaxN = ClusterCol_GpuPathMaxN(SCF_iter,mode,TZ,n,first_states);

    if (numprocs1<=*MaxN){
        av_num = (double)(*MaxN)/(double)numprocs1;
        for (ID=0; ID<numprocs1; ID++){
            is2[ID] = (int)(av_num*(double)ID) + 1;
            ie2[ID] = (int)(av_num*(double)(ID+1));
        }
        is2[0] = 1;
        ie2[numprocs1-1] = *MaxN;
    }
    else{
        for (ID=0; ID<*MaxN; ID++){
            is2[ID] = ID + 1;
            ie2[ID] = ID + 1;
        }
        for (ID=*MaxN; ID<numprocs1; ID++){
            is2[ID] = 1;
            ie2[ID] = 0;
        }
    }
}

static void ClusterCol_DistributeDenseEvec(int n, int maxn, int myid1, int numprocs1, int *is2, int *ie2,
                                           MPI_Comm comm, const double *dense_evec, double *EVec1)
{
    const int root = 0;
    const int tag = 996;
    int max_count = 0;

    if (myid1==root){
        double *send_buf = NULL;

        for (int id=0; id<numprocs1; id++){
            int states = ie2[id] - is2[id] + 1;
            int count;
            if (states<0) states = 0;
            count = n*states;
            if (max_count<count) max_count = count;
        }

        if (0<max_count){
            send_buf = (double*)ClusterCol_MallocArray((size_t)max_count,sizeof(double),
                                                       "dense eigenvector send buffer");
        }

        for (int id=0; id<numprocs1; id++){
            int states = ie2[id] - is2[id] + 1;
            int count;
            double *dst;

            if (states<0) states = 0;
            count = n*states;
            if (count==0) continue;

            dst = (id==root) ? EVec1 : send_buf;
            for (int basis=0; basis<n; basis++){
                const double *src_col = dense_evec + (size_t)basis*(size_t)maxn + (size_t)(is2[id]-1);
                double *dst_col = dst + (size_t)basis*(size_t)states;
                memcpy(dst_col,src_col,sizeof(double)*(size_t)states);
            }

            if (id!=root){
                MPI_Send(send_buf,count,MPI_DOUBLE,id,tag,comm);
            }
        }

        free(send_buf);
    }
    else{
        int states = ie2[myid1] - is2[myid1] + 1;
        int count;
        MPI_Status stat;

        if (states<0) states = 0;
        count = n*states;
        if (0<count){
            MPI_Recv(EVec1,count,MPI_DOUBLE,root,tag,comm,&stat);
        }
    }
}

static double ClusterCol_ScatterGpuSolverCachedEVec(int SpinP_switch, int numprocs0, int myworld1,
                                                   int myid1, int numprocs1, MPI_Comm *MPI_CommWD1,
                                                   int n, int maxn, int *is2, int *ie2, double **EVec1)
{
    int spin_start = 0;
    int spin_end = SpinP_switch;
    double stime, etime;

    dtime(&stime);

    if (SpinP_switch == 1 && numprocs0 != 1) {
        spin_start = myworld1;
        spin_end = myworld1;
    }

    for (int spin = spin_start; spin <= spin_end; spin++) {
        int valid = 0;

        if (myid1 == 0) {
            valid = (ClusterCol_CachedGpuSolverDenseValid[spin] &&
                     ClusterCol_CachedGpuSolverDenseEVec[spin] != NULL &&
                     ClusterCol_CachedGpuSolverDenseN == n &&
                     ClusterCol_CachedGpuSolverDenseMaxN == maxn);
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_CommWD1[myworld1]);

        if (valid) {
            ClusterCol_DistributeDenseEvec(n, maxn, myid1, numprocs1, is2, ie2,
                                           MPI_CommWD1[myworld1],
                                           (myid1 == 0) ? ClusterCol_CachedGpuSolverDenseEVec[spin] : NULL,
                                           EVec1[spin]);
        }
    }

    dtime(&etime);
    return etime - stime;
}

static void ClusterCol_CalcOneSpinDMRootDense(int myid0, int myid1, int spin, int size_H1,
                                              int *MP, int n, int MaxN, MPI_Comm comm,
                                              double *****CDM, double *****EDM, double **ko,
                                              double *DM1, double *EDM1, double *PDM1)
{
    ClusterColRootDMWorkspace *ws = &ClusterCol_root_dm_workspace;
    const int calc_pdm = cal_partial_charge;
    const int dm_components = calc_pdm ? 3 : 2;
    size_t dm_elems = (size_t)size_H1 * (size_t)dm_components;

    if ((size_t)INT_MAX < dm_elems) {
        ClusterCol_AbortWithMessage("Root collinear DM buffer exceeds MPI count limit in Cluster_DFT_Col.c.");
    }

    ClusterCol_RootDMWorkspace_Ensure(MaxN, dm_elems);
    memset(ws->dm_buffer, 0, sizeof(double) * dm_elems);

    if (myid1 == 0) {
        ClusterColDMEntryCache *cache;
        const double *dense_evec = ClusterCol_CachedGpuSolverDenseEVec[spin];
        int nk_occ;

        if (!(ClusterCol_CachedGpuSolverDenseValid[spin] &&
              dense_evec != NULL &&
              ClusterCol_CachedGpuSolverDenseN == n &&
              ClusterCol_CachedGpuSolverDenseMaxN == MaxN)) {
            ClusterCol_AbortWithMessage("hipSOLVER dense eigenvectors are not available for collinear HIP DM.");
        }

        ClusterCol_DMEntryCache_Ensure(MP, n);
        cache = &ClusterCol_dm_entry_cache;
        if (cache->entry_count != size_H1) {
            ClusterCol_AbortWithMessage("DM entry cache size mismatch in Cluster_DFT_Col.c.");
        }

        nk_occ = ClusterCol_PrecomputeOccupations(spin, 1, MaxN, ko[spin], ws->occ, ws->occ_e, ws->pocc, calc_pdm);

        if (0 < nk_occ) {
            static int logged_dm_gpu = 0;

            if (ClusterCol_GpuVerbose() && !logged_dm_gpu) {
                fprintf(stderr,
                        "<Cluster> rank %d: collinear density matrix generation uses a HIP GPU kernel on the selected rank only "
                        "(entries=%d, occupied states=%d%s).\n",
                        myid0, cache->entry_count, nk_occ, calc_pdm ? ", with partial DM" : "");
                fflush(stderr);
                logged_dm_gpu = 1;
            }

            if (ClusterCol_CalcDMRootDense_HIP(cache->entry_count, size_H1, n, MaxN, nk_occ, calc_pdm,
                                               cache->basis0, cache->basis1,
                                               ws->occ, ws->occ_e, ws->pocc,
                                               dense_evec,
                                               ClusterCol_DeviceEvecSpin == spin ? ClusterCol_gpusolver_ctx.d_tmp : NULL,
                                               ws->dm_buffer) != 0) {
                ClusterCol_AbortWithMessage("Cluster_DFT_Col HIP density-matrix generation failed.");
            }
        }
    }

    MPI_Bcast(ws->dm_buffer, (int)dm_elems, MPI_DOUBLE, 0, comm);

    memcpy(DM1, ws->dm_buffer, sizeof(double) * (size_t)size_H1);
    memcpy(EDM1, ws->dm_buffer + (size_t)size_H1, sizeof(double) * (size_t)size_H1);
    if (calc_pdm) {
        memcpy(PDM1, ws->dm_buffer + (size_t)size_H1 * 2u, sizeof(double) * (size_t)size_H1);
    }

    ClusterCol_StoreDMComponents(myid0, spin, MP, CDM, EDM, DM1, EDM1, PDM1);
}

static void ClusterCol_ExchangeAndStoreOtherSpinDM(int myid0, int myworld1, int size_H1, int *MP,
                                                   MPI_Comm *MPI_CommWD1, int *Comm_World_StartID1,
                                                   double *****CDM, double *****EDM,
                                                   double *DM1, double *EDM1, double *PDM1, double *Work1)
{
    int other_spin = (myworld1 == 0) ? 1 : 0;
    int root_this = Comm_World_StartID1[myworld1];
    int root_other = Comm_World_StartID1[other_spin];
    MPI_Status stat;

    if (myid0 == root_this) {
        MPI_Sendrecv(DM1, size_H1, MPI_DOUBLE, root_other, 10,
                     Work1, size_H1, MPI_DOUBLE, root_other, 10,
                     mpi_comm_level1, &stat);
    }
    MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);
    memcpy(DM1, Work1, sizeof(double) * (size_t)size_H1);

    if (myid0 == root_this) {
        MPI_Sendrecv(EDM1, size_H1, MPI_DOUBLE, root_other, 20,
                     Work1, size_H1, MPI_DOUBLE, root_other, 20,
                     mpi_comm_level1, &stat);
    }
    MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);
    memcpy(EDM1, Work1, sizeof(double) * (size_t)size_H1);

    if (cal_partial_charge) {
        if (myid0 == root_this) {
            MPI_Sendrecv(PDM1, size_H1, MPI_DOUBLE, root_other, 30,
                         Work1, size_H1, MPI_DOUBLE, root_other, 30,
                         mpi_comm_level1, &stat);
        }
        MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);
        memcpy(PDM1, Work1, sizeof(double) * (size_t)size_H1);
    }

    ClusterCol_StoreDMComponents(myid0, other_spin, MP, CDM, EDM, DM1, EDM1, PDM1);
}

static double ClusterCol_CalcDMRootDense_HIP_Path(int myid0, int numprocs0, int myid1, int numprocs1,
                                                  int myworld1, int SpinP_switch, int size_H1,
                                                  int *MP, int n, int MaxN, MPI_Comm *MPI_CommWD1,
                                                  int *Comm_World_StartID1,
                                                  double *****CDM, double *****EDM, double **ko,
                                                  double *DM1, double *EDM1, double *PDM1, double *Work1)
{
    double stime, etime;

    (void)numprocs1;
    dtime(&stime);

    if (SpinP_switch == 0) {
        ClusterCol_CalcOneSpinDMRootDense(myid0, myid1, 0, size_H1, MP, n, MaxN,
                                          MPI_CommWD1[myworld1], CDM, EDM, ko, DM1, EDM1, PDM1);
    }
    else if (numprocs0 == 1) {
        ClusterCol_CalcOneSpinDMRootDense(myid0, myid1, 0, size_H1, MP, n, MaxN,
                                          MPI_CommWD1[myworld1], CDM, EDM, ko, DM1, EDM1, PDM1);
        ClusterCol_CalcOneSpinDMRootDense(myid0, myid1, 1, size_H1, MP, n, MaxN,
                                          MPI_CommWD1[myworld1], CDM, EDM, ko, DM1, EDM1, PDM1);
    }
    else {
        ClusterCol_CalcOneSpinDMRootDense(myid0, myid1, myworld1, size_H1, MP, n, MaxN,
                                          MPI_CommWD1[myworld1], CDM, EDM, ko, DM1, EDM1, PDM1);
        ClusterCol_ExchangeAndStoreOtherSpinDM(myid0, myworld1, size_H1, MP, MPI_CommWD1,
                                               Comm_World_StartID1, CDM, EDM, DM1, EDM1, PDM1, Work1);
    }

    dtime(&etime);
    if (myid1 == 0 && ClusterCol_detail_timers.active) {
        ClusterCol_detail_timers.dm_total += etime - stime;
        ClusterCol_DetailTimer_Report(myid0, myid1);
    }
    return etime - stime;
}

/* OPENMX_GEMM_SAMPLE_DIR=<existing directory> writes the inputs of the two
   forward GEMMs, B = H X and X^T B, as raw column-major doubles: the
   transformed overlap X whenever the root dense path rebuilds it
   (X_g<geometry>.bin, spin world 0's owner) and the Hamiltonian, as built
   before the transform, of the SCF steps listed in OPENMX_GEMM_SAMPLE_ITERS
   (comma separated, default "1"; H_g<geometry>_scf<step>_s<spin>.bin).  The
   files feed offline precision studies of the forward transform (the
   NVIDIA version's tests/gemmul8_reuse_probe.cu reads them).  The
   serialized spin mode samples nothing.  Ported from the NVIDIA version. */
static int ClusterCol_sample_geometry = 0;

static void ClusterCol_SampleDeviceMatrix(const double *d_A, int n, const char *name)
{
    ClusterColGpuSolverCtx *ctx = &ClusterCol_gpusolver_ctx;
    size_t count = ClusterCol_CheckedMulCount((size_t)n,(size_t)n,"sampled matrix");
    double *A = (double*)ClusterCol_MallocArray(count,sizeof(double),"sampled matrix");
    char path[4096];
    FILE *fp;

    snprintf(path,sizeof(path),"%s/%s",getenv("OPENMX_GEMM_SAMPLE_DIR"),name);
    wait_hipfunc(hipMemcpyAsync(A,d_A,count*sizeof(double),hipMemcpyDeviceToHost,ctx->stream));
    wait_hipfunc(hipStreamSynchronize(ctx->stream));

    fp = fopen(path,"wb");
    if (fp==NULL || fwrite(A,sizeof(double),count,fp)!=count || fclose(fp)!=0){
        char msg[4200];
        snprintf(msg,sizeof(msg),"Cluster_DFT_Col.c: could not write the sampled matrix %s",path);
        ClusterCol_AbortWithMessage(msg);
    }
    printf("<Cluster_DFT_Col> sampled %s (n=%d)\n",path,n);
    fflush(stdout);
    free(A);
}

static void ClusterCol_SampleTransformedS(int n, int myworld1)
{
    const char *dir = getenv("OPENMX_GEMM_SAMPLE_DIR");
    char name[64];

    ClusterCol_sample_geometry++;
    /* both spin worlds hold the same X; one copy is enough */
    if (dir==NULL || dir[0]=='\0' || myworld1!=0) return;

    snprintf(name,sizeof(name),"X_g%d.bin",ClusterCol_sample_geometry);
    ClusterCol_SampleDeviceMatrix(ClusterCol_gpusolver_ctx.d_S,n,name);
}

static void ClusterCol_SampleHamiltonian(int SCF_iter, int spin, int n)
{
    const char *dir = getenv("OPENMX_GEMM_SAMPLE_DIR");
    const char *list = getenv("OPENMX_GEMM_SAMPLE_ITERS");
    char name[64];
    int wanted = 0;

    if (dir==NULL || dir[0]=='\0') return;
    if (list==NULL || list[0]=='\0') list = "1";

    while (*list!='\0' && !wanted){
        char *end;
        long step = strtol(list,&end,10);
        if (end==list) break;
        wanted = (step==SCF_iter);
        list = (*end==',') ? end+1 : end;
    }
    if (!wanted) return;

    snprintf(name,sizeof(name),"H_g%d_scf%03d_s%d.bin",ClusterCol_sample_geometry,SCF_iter,spin);
    ClusterCol_SampleDeviceMatrix(ClusterCol_gpusolver_ctx.d_H,n,name);
}

static void ClusterCol_PartitionStates(int MaxN, int nw, int *is2, int *ie2)
{
    int ID;

    if (nw<=MaxN){
        const double av_num = (double)MaxN/(double)nw;
        for (ID=0; ID<nw; ID++){
            is2[ID] = (int)(av_num*(double)ID) + 1;
            ie2[ID] = (int)(av_num*(double)(ID+1));
        }
        is2[0] = 1;
        ie2[nw-1] = MaxN;
    }
    else{
        for (ID=0; ID<MaxN; ID++){ is2[ID] = ID + 1; ie2[ID] = ID + 1; }
        for (ID=MaxN; ID<nw; ID++){ is2[ID] = 1; ie2[ID] = 0; }
    }
}

static void ClusterCol_UnionSendEvec(int spin, int n, int MaxN, int myworld1, int myid1,
                                     int *is2, int *ie2, int *NPROCS_WD1,
                                     int *Comm_World_StartID1, double **EVec1)
{
    const ClusterColUnionGrid *g = &ClusterCol_ugrid;
    const int nw = NPROCS_WD1[spin];
    const int start = Comm_World_StartID1[spin];
    int numprocs0, ID, i, j, k;
    int *is2s, *ie2s, *owner, *scnt, *sdsp, *rcnt, *rdsp, *iscnt, *isdsp, *ircnt, *irdsp;
    int *idx_snd, *idx_rcv;
    double *val_snd, *val_rcv;
    long long nsnd = 0, nrcv = 0;

    MPI_Comm_size(mpi_comm_level1, &numprocs0);

    is2s = (int *)malloc(sizeof(int) * (size_t)nw);
    ie2s = (int *)malloc(sizeof(int) * (size_t)nw);
    owner = (int *)malloc(sizeof(int) * (size_t)(MaxN + 1));
    scnt = (int *)calloc((size_t)numprocs0, sizeof(int));
    sdsp = (int *)malloc(sizeof(int) * (size_t)numprocs0);
    rcnt = (int *)malloc(sizeof(int) * (size_t)numprocs0);
    rdsp = (int *)malloc(sizeof(int) * (size_t)numprocs0);
    iscnt = (int *)malloc(sizeof(int) * (size_t)numprocs0);
    isdsp = (int *)malloc(sizeof(int) * (size_t)numprocs0);
    ircnt = (int *)malloc(sizeof(int) * (size_t)numprocs0);
    irdsp = (int *)malloc(sizeof(int) * (size_t)numprocs0);
    if (!is2s || !ie2s || !owner || !scnt || !sdsp || !rcnt || !rdsp || !iscnt || !isdsp || !ircnt || !irdsp){
        fprintf(stderr, "Cluster_DFT_Col: out of host memory in the eigenvector distribution\n");
        MPI_Abort(mpi_comm_level1, 1);
    }

    ClusterCol_PartitionStates(MaxN, nw, is2s, ie2s);
    for (ID=0; ID<nw; ID++){
        for (k=is2s[ID]; k<=ie2s[ID]; k++) owner[k] = start + ID;
    }

    /* counts per destination */
    for (i=0; i<g->na_rows; i++){
        const int ig = g->np_rows*g->nblk*(i/g->nblk) + i%g->nblk
                     + ((g->np_rows+g->my_prow)%g->np_rows)*g->nblk + 1;
        if (ig<=MaxN) scnt[owner[ig]] += g->na_cols;
    }
    MPI_Alltoall(scnt, 1, MPI_INT, rcnt, 1, MPI_INT, mpi_comm_level1);
    for (ID=0; ID<numprocs0; ID++){
        sdsp[ID] = (int)nsnd;
        rdsp[ID] = (int)nrcv;
        nsnd += scnt[ID];
        nrcv += rcnt[ID];
        if (2LL*nsnd > INT_MAX || 2LL*nrcv > INT_MAX){
            fprintf(stderr, "Cluster_DFT_Col: the eigenvector distribution of the distributed capacity"
                            " fallback exceeds the MPI count limit (n=%d, MaxN=%d); use more MPI ranks\n",
                    n, MaxN);
            MPI_Abort(mpi_comm_level1, 1);
        }
        iscnt[ID] = 2*scnt[ID]; isdsp[ID] = 2*sdsp[ID];
        ircnt[ID] = 2*rcnt[ID]; irdsp[ID] = 2*rdsp[ID];
    }

    idx_snd = (int *)malloc(sizeof(int) * (size_t)(2*nsnd + 1));
    val_snd = (double *)malloc(sizeof(double) * (size_t)(nsnd + 1));
    idx_rcv = (int *)malloc(sizeof(int) * (size_t)(2*nrcv + 1));
    val_rcv = (double *)malloc(sizeof(double) * (size_t)(nrcv + 1));
    if (!idx_snd || !val_snd || !idx_rcv || !val_rcv){
        fprintf(stderr, "Cluster_DFT_Col: out of host memory in the eigenvector distribution\n");
        MPI_Abort(mpi_comm_level1, 1);
    }

    /* pack in destination order: rows are visited per destination */
    {
        int *pos = (int *)malloc(sizeof(int) * (size_t)numprocs0);
        if (pos == NULL){
            fprintf(stderr, "Cluster_DFT_Col: out of host memory in the eigenvector distribution\n");
            MPI_Abort(mpi_comm_level1, 1);
        }
        for (ID=0; ID<numprocs0; ID++) pos[ID] = sdsp[ID];
        for (i=0; i<g->na_rows; i++){
            const int ig = g->np_rows*g->nblk*(i/g->nblk) + i%g->nblk
                         + ((g->np_rows+g->my_prow)%g->np_rows)*g->nblk + 1;
            if (ig>MaxN) continue;
            for (j=0; j<g->na_cols; j++){
                const int jg = g->np_cols*g->nblk*(j/g->nblk) + j%g->nblk
                             + ((g->np_cols+g->my_pcol)%g->np_cols)*g->nblk + 1;
                const int q = pos[owner[ig]]++;
                idx_snd[2*q] = ig;
                idx_snd[2*q+1] = jg;
                val_snd[q] = g->Hs[(size_t)j*(size_t)g->na_rows + (size_t)i];
            }
        }
        free(pos);
    }

    MPI_Alltoallv(idx_snd, iscnt, isdsp, MPI_INT, idx_rcv, ircnt, irdsp, MPI_INT, mpi_comm_level1);
    MPI_Alltoallv(val_snd, scnt, sdsp, MPI_DOUBLE, val_rcv, rcnt, rdsp, MPI_DOUBLE, mpi_comm_level1);

    if (myworld1==spin){
        const int states = ie2[myid1] - is2[myid1] + 1;
        for (k=0; k<(int)nrcv; k++){
            const int ig = idx_rcv[2*k];
            const int jg = idx_rcv[2*k+1];
            EVec1[spin][(size_t)(jg-1)*(size_t)states + (size_t)(ig-is2[myid1])] = val_rcv[k];
        }
    }

    free(idx_snd); free(val_snd); free(idx_rcv); free(val_rcv);
    free(is2s); free(ie2s); free(owner);
    free(scnt); free(sdsp); free(rcnt); free(rdsp);
    free(iscnt); free(isdsp); free(ircnt); free(irdsp);
}

static void ClusterCol_UnionGs2Path(int n, int MaxN, double **ko, double *****nh, double ****CntOLP,
                                    int *MP, int myworld1, int myid1, int *is2, int *ie2,
                                    int *NPROCS_WD1, int *Comm_World_StartID1, double **EVec1)
{
    ClusterColUnionGrid *g = &ClusterCol_ugrid;
    ClusterColGridGlobals saved;
    double alpha = 1.0, beta = 0.0;
    int ONE = 1, spin, i, j, l, info;

    if (g->overlap_n != n){
        double *w = (double *)malloc(sizeof(double) * (size_t)(n + 1));

        if (w == NULL){
            fprintf(stderr, "Cluster_DFT_Col: out of host memory (overlap eigenvalues)\n");
            MPI_Abort(mpi_comm_level1, 1);
        }
        saved = ClusterCol_EnterUnionGridGlobals();
        Overlap_Cluster_Ss(CntOLP, g->Cs, MP, myworld1);
        ClusterCol_LeaveUnionGridGlobals(saved);

        info = openmx_gs2_eigen_real(n, n, g->Cs, g->desc, &w[1], g->Ss, g->desc);
        if (info!=0){
            printf("Cluster_DFT_Col: the gpusolver2 overlap eigensolver failed (info=%d)\n", info);
            MPI_Abort(mpi_comm_level1, 1);
        }
        for (l=1; l<=n; l++){
            if (w[l]<0.0) w[l] = 1.0e-10;
            w[l] = 1.0/sqrt(w[l]);
        }
        for (i=0; i<g->na_rows; i++){
            for (j=0; j<g->na_cols; j++){
                const int jg = g->np_cols*g->nblk*(j/g->nblk) + j%g->nblk
                             + ((g->np_cols+g->my_pcol)%g->np_cols)*g->nblk + 1;
                g->Ss[(size_t)j*(size_t)g->na_rows + (size_t)i] *= w[jg];
            }
        }
        free(w);
        g->overlap_n = n;
    }

    for (spin=0; spin<=1; spin++){

        saved = ClusterCol_EnterUnionGridGlobals();
        Hamiltonian_Cluster_Hs(nh[spin], g->Hs, MP, spin, spin);
        ClusterCol_LeaveUnionGridGlobals(saved);

        /* H X, then X^T (H X) */
        for (i=0; i<g->na_rows_max*g->na_cols_max; i++) g->Cs[i] = 0.0;
        Cblacs_barrier(g->ictxt, "A");
        openmx_gs2_pdgemm_("N","N",&n,&n,&n,&alpha,g->Hs,&ONE,&ONE,g->desc,g->Ss,&ONE,&ONE,g->desc,
                           &beta,g->Cs,&ONE,&ONE,g->desc);
        for (i=0; i<g->na_rows*g->na_cols; i++) g->Hs[i] = 0.0;
        Cblacs_barrier(g->ictxt, "C");
        openmx_gs2_pdgemm_("T","N",&n,&n,&n,&alpha,g->Ss,&ONE,&ONE,g->desc,g->Cs,&ONE,&ONE,g->desc,
                           &beta,g->Hs,&ONE,&ONE,g->desc);

        for (i=0; i<=n; i++) ko[spin][i] = 10000.0;
        info = openmx_gs2_eigen_real(n, MaxN, g->Hs, g->desc, &ko[spin][1], g->Cs, g->desc);
        if (info!=0){
            printf("Cluster_DFT_Col: the gpusolver2 Hamiltonian eigensolver failed (info=%d)\n", info);
            MPI_Abort(mpi_comm_level1, 1);
        }

        /* (X C)^T: one state per row */
        for (i=0; i<g->na_rows*g->na_cols; i++) g->Hs[i] = 0.0;
        Cblacs_barrier(g->ictxt, "A");
        openmx_gs2_pdgemm_("T","T",&n,&n,&n,&alpha,g->Cs,&ONE,&ONE,g->desc,g->Ss,&ONE,&ONE,g->desc,
                           &beta,g->Hs,&ONE,&ONE,g->desc);

        ClusterCol_UnionSendEvec(spin, n, MaxN, myworld1, myid1, is2, ie2, NPROCS_WD1,
                                 Comm_World_StartID1, EVec1);
    }
}

static void Patch2Host_Cluster_Owner(double ****RH, int *MP, int owns_dense, int n, double *d_H)
{
  int i,j,k;
  int MA_AN,GA_AN,LB_AN,GB_AN,AN;
  int wanA,wanB,tnoA,tnoB,Anum,NUM;
  int num,tnum;
  int ID,myid,numprocs;
  int *My_NZeros;
  int *is1,*ie1,*is2;
  int *My_Matomnum,*order_GA;
  double *H1;
  const int *dense_index = NULL;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  My_NZeros = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Device My_NZeros");
  My_Matomnum = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Device My_Matomnum");
  is1 = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Device is1");
  ie1 = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Device ie1");
  is2 = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Device is2");
  order_GA = (int*)ClusterCol_MallocArray((size_t)(atomnum+2),sizeof(int),"Patch2Device order_GA");

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    num = 0;
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num;
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++) tnum += My_NZeros[ID];

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;
  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }

  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  is2[0] = 1;
  for (ID=1; ID<numprocs; ID++){
    is2[ID] = is2[ID-1] + My_Matomnum[ID-1];
  }

  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    order_GA[is2[myid]+MA_AN-1] = M2G[MA_AN];
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&order_GA[is2[ID]],My_Matomnum[ID],MPI_INT,ID,mpi_comm_level1);
  }

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum += Spe_Total_CNO[wanA];
  }
  NUM = Anum - 1;
  if (NUM!=n){
    ClusterCol_AbortWithMessage("Orbital count mismatch in Cluster_DFT_Col.c Patch2Device.");
  }

  H1 = (double*)ClusterCol_MallocArray((size_t)(tnum+1),sizeof(double),"Patch2Device sparse buffer");

  k = is1[myid];
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (i=0; i<tnoA; i++){
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
        for (j=0; j<tnoB; j++){
          H1[k] = RH[MA_AN][LB_AN][i][j];
          k++;
        }
      }
    }
  }

  for (ID=0; ID<numprocs; ID++){
    k = is1[ID];
    MPI_Bcast(&H1[k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
  }

  if (owns_dense){
    memset(d_H,0,(size_t)n*n*sizeof(double));
    k=0;
    for (AN=1;AN<=atomnum;++AN) {
      GA_AN=order_GA[AN]; tnoA=Spe_Total_CNO[WhatSpecies[GA_AN]];
      for (i=0;i<tnoA;++i) for (LB_AN=0;LB_AN<=FNAN[GA_AN];++LB_AN) {
        GB_AN=natn[GA_AN][LB_AN]; tnoB=Spe_Total_CNO[WhatSpecies[GB_AN]];
        for (j=0;j<tnoB;++j)
          d_H[(size_t)(MP[GB_AN]+j-1)*n+MP[GA_AN]+i-1]+=H1[k++];
      }
    }
  }

  free(My_NZeros);
  free(My_Matomnum);
  free(is1);
  free(ie1);
  free(is2);
  free(order_GA);
  free(H1);
}

static void ClusterCol_MagmaRootDensePath(int SpinP_switch,double **ko,double *****nh,double ****CntOLP,
    int numprocs0,int myid0,int myworld1,int numprocs1,int myid1,MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,int *MP,int *is2,int *ie2,int n,int MaxN,double **EVec1)
{
    const int root=(myid0==Host_ID);
    const size_t nn=(size_t)n*n, panel=(size_t)n*MaxN;
    double *s=NULL,*h=NULL,*tmp=NULL,*c=NULL;
    ClusterCol_DeviceEvecSpin = -1;
    ClusterCol_ReleaseGpuSolverCachedEVec(myid1);
    ClusterCol_refine.last_refined=0;
    if (root) {
        s=ClusterCol_MallocArray(nn,sizeof(double),"MAGMA overlap");
        h=ClusterCol_MallocArray(nn,sizeof(double),"MAGMA Hamiltonian");
        tmp=ClusterCol_MallocArray(nn,sizeof(double),"MAGMA transform");
    }
    if (root || myid1==0) c=ClusterCol_MallocArray(panel,sizeof(double),"MAGMA eigenvectors");
    Patch2Host_Cluster_Owner(CntOLP,MP,root,n,s);
    if (root) {
        if (!openmx_mgpu_host_solve(n,n,0,s,n,ko[0]+1)) ClusterCol_AbortWithMessage(openmx_mgpu_host_error());
#pragma omp parallel for
        for (int j=0;j<n;++j) {
            const double scale=1.0/sqrt(fmax(ko[0][j+1],1.0e-10));
            for (int i=0;i<n;++i) s[(size_t)j*n+i]*=scale;
        }
    }
    for (int spin=0;spin<=SpinP_switch;++spin) {
        Patch2Host_Cluster_Owner(nh[spin],MP,root,n,h);
        if (root) {
            if (!openmx_mgpu_host_dgemm(HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,h,n,s,n,tmp,n) ||
                !openmx_mgpu_host_dgemm(HIPBLAS_OP_T,HIPBLAS_OP_N,n,n,n,s,n,tmp,n,h,n) ||
                !openmx_mgpu_host_solve(n,MaxN,0,h,n,ko[spin]+1) ||
                !openmx_mgpu_host_dgemm(HIPBLAS_OP_T,HIPBLAS_OP_T,MaxN,n,n,h,n,s,n,c,MaxN))
                ClusterCol_AbortWithMessage(openmx_mgpu_host_error());
        }
        MPI_Bcast(ko[spin]+1,MaxN,MPI_DOUBLE,Host_ID,mpi_comm_level1);
        const int world=(SpinP_switch==1 && numprocs0!=1)?spin:0;
        const int dest=Comm_World_StartID1[world];
        if (dest!=Host_ID) {
            const size_t chunk=64ULL*1024*1024;
            MPI_Status stat;
            for (size_t off=0;off<panel;off+=chunk) {
                int count=(int)((panel-off<chunk)?panel-off:chunk);
                if (root) MPI_Send(c+off,count,MPI_DOUBLE,dest,998,mpi_comm_level1);
                if (myid0==dest) MPI_Recv(c+off,count,MPI_DOUBLE,Host_ID,998,mpi_comm_level1,&stat);
            }
        }
        if (myworld1==world)
            ClusterCol_DistributeDenseEvec(n,MaxN,myid1,numprocs1,is2,ie2,MPI_CommWD1[world],c,EVec1[spin]);
    }
    free(s); free(h); free(tmp); free(c);
}

static void ClusterCol_GpuSolverRootDensePath(int SCF_iter, int SpinP_switch, double **ko,
                                             double *****nh, double ****CntOLP,
                                             int numprocs0, int myid0, int myworld1,
                                             int numprocs1, int myid1, MPI_Comm *MPI_CommWD1,
                                             int *Comm_World_StartID1,
                                             int *MP, int *is2, int *ie2,
                                             int n, int MaxN, double **EVec1)
{
    if (ClusterCol_mgpu_host_dense) {
        ClusterCol_MagmaRootDensePath(SpinP_switch,ko,nh,CntOLP,numprocs0,myid0,myworld1,numprocs1,myid1,
            MPI_CommWD1,Comm_World_StartID1,MP,is2,ie2,n,MaxN,EVec1);
        return;
    }

    int owns_dense = (myid1==0);
    int spin_start = 0;
    int spin_end = SpinP_switch;
    int rebuild_s = (SCF_iter==1);
    int use_setham_packed_cache =
        (Set_Hamiltonian_GpuSolver_Packed_CacheReady() &&
         Set_Hamiltonian_GpuSolver_Packed_OrderMode() == 0);
    /* the two spin worlds' owners solve at the same time; on one device
       (the admission counted them) neither forward transform takes the
       full GEMMul8 workspace */
    int shared = (SpinP_switch==1 && numprocs0!=1 && 1<ClusterCol_owners_on_device);

    if (SpinP_switch==1 && numprocs0!=1){
        spin_start = myworld1;
        spin_end = myworld1;
    }

    /* the refined eigensolver: one spin, or two spin worlds whose owners
       exchange their estimates; a single rank solving both spins keeps
       FP64 */
    {
        int spin_mode = 0, peer = -1;

        if (SpinP_switch==1){
            if (numprocs0==1) spin_mode = 2;
            else {
                spin_mode = 1;
                peer = Comm_World_StartID1[1-myworld1];
            }
        }
        /* a problem too small for the refinement to pay off keeps FP64
           (OPENMX_EIGEN_REFINE_MIN_N), as a single rank of both spins does */
        if (!openmx_eigen_refine_size_ok(n)) spin_mode = 2;
        ClusterCol_RefineBeginSolve(SCF_iter,spin_mode,peer);
    }

    if (use_setham_packed_cache) {
        Set_Hamiltonian_GpuSolver_SetMP(MP);
    }

    ClusterCol_ReleaseGpuSolverCachedEVec(myid1);
    ClusterCol_DetailTimer_Begin(myid1, SCF_iter, n, MaxN);

    if (ClusterCol_GpuVerbose() && myid1 == 0) {
        static int logged_root_dense = 0;

        if (!logged_root_dense) {
            fprintf(stderr,
                    "<Cluster> rank %d: MAGMA collinear dense GPU path uses one selected rank for dense matrix generation, "
                    "GEMM, eigensolver, and density-matrix generation (n=%d, maxn=%d).\n",
                    myid0, n, MaxN);
            fflush(stderr);
            logged_root_dense = 1;
        }
    }

    if (owns_dense){
        ClusterCol_GpuSolver_EnsureMatrixCapacity(n);

        if (rebuild_s || !(ClusterCol_gpusolver_ctx.transformed_s_valid &&
                           ClusterCol_gpusolver_ctx.transformed_s_dim==n)){
            rebuild_s = 1;
        }
    }

    MPI_Bcast(&rebuild_s,1,MPI_INT,0,MPI_CommWD1[myworld1]);

    if (rebuild_s){
        if (use_setham_packed_cache) {
            if (owns_dense) {
                int tnum = Set_Hamiltonian_GpuSolver_Packed_Size();
                int *cache_order_GA = Set_Hamiltonian_GpuSolver_Packed_OrderGA();
                double *cache_S = Set_Hamiltonian_GpuSolver_Packed_Overlap();
                int *dense_index = NULL;

                if (!Set_Hamiltonian_GpuSolver_Packed_OwnsCache() || cache_order_GA == NULL || cache_S == NULL) {
                    ClusterCol_AbortWithMessage("Set_Hamiltonian packed overlap cache is missing in Cluster_DFT_Col.c.");
                }
                dense_index = (int*)ClusterCol_MallocArray((size_t)tnum,sizeof(int),"packed overlap dense_index");
                ClusterCol_BuildDenseIndex(cache_order_GA,MP,n,tnum,dense_index);
                ClusterCol_BuildDeviceDenseFromPacked(cache_S,dense_index,tnum,n,ClusterCol_gpusolver_ctx.d_S);
                free(dense_index);
            }
        }
        else {
            Patch2Device_Cluster_Owner(CntOLP,MP,owns_dense,n,
                                       owns_dense ? ClusterCol_gpusolver_ctx.d_S : NULL);
        }
        if (owns_dense){
            ClusterCol_GpuSolver_PrepareTransformedSDevice(1,n,ko[0]);
            ClusterCol_SampleTransformedS(n,myworld1);
        }
    }

    for (int spin=spin_start; spin<=spin_end; spin++){
        if (use_setham_packed_cache) {
            if (owns_dense){
                int tnum = Set_Hamiltonian_GpuSolver_Packed_Size();
                int *cache_order_GA = Set_Hamiltonian_GpuSolver_Packed_OrderGA();
                double *cache_H = Set_Hamiltonian_GpuSolver_Packed_H(spin);
                int *dense_index = NULL;
                size_t nmax = ClusterCol_CheckedMulCount((size_t)n,(size_t)MaxN,"root dense eigenvectors");
                double *C = NULL;

                if (!Set_Hamiltonian_GpuSolver_Packed_OwnsCache() || cache_order_GA == NULL || cache_H == NULL) {
                    ClusterCol_AbortWithMessage("Set_Hamiltonian packed Hamiltonian cache is missing in Cluster_DFT_Col.c.");
                }
                dense_index = (int*)ClusterCol_MallocArray((size_t)tnum,sizeof(int),"packed Hamiltonian dense_index");
                ClusterCol_BuildDenseIndex(cache_order_GA,MP,n,tnum,dense_index);
                ClusterCol_BuildDeviceDenseFromPacked(cache_H,dense_index,tnum,n,ClusterCol_gpusolver_ctx.d_H);
                free(dense_index);

                ClusterCol_SampleHamiltonian(SCF_iter,spin,n);
                C = (double*)ClusterCol_MallocArray(nmax,sizeof(double),"root dense eigenvectors");
                ClusterCol_GpuSolver_SolveHamiltonianDevice(n,MaxN,ko[spin],C,spin,shared);
                ClusterCol_StashGpuSolverDenseEVec(myid1, spin, n, MaxN, &C);
                free(C);
            }
        }
        else {
            Patch2Device_Cluster_Owner(nh[spin],MP,owns_dense,n,
                                       owns_dense ? ClusterCol_gpusolver_ctx.d_H : NULL);
            if (owns_dense){
                size_t nmax = ClusterCol_CheckedMulCount((size_t)n,(size_t)MaxN,"root dense eigenvectors");
                double *C = NULL;

                ClusterCol_SampleHamiltonian(SCF_iter,spin,n);
                C = (double*)ClusterCol_MallocArray(nmax,sizeof(double),"root dense eigenvectors");
                ClusterCol_GpuSolver_SolveHamiltonianDevice(n,MaxN,ko[spin],C,spin,shared);
                ClusterCol_StashGpuSolverDenseEVec(myid1, spin, n, MaxN, &C);
                free(C);
            }
        }
    }

    /* the GEMMul8 workspaces of the solves go before the grid stages of the
       step; the transformed overlap stays on the device */
    if (owns_dense) ClusterCol_GemmWorkspaceTurnEnd();

    ClusterCol_RefineEndSolve();

    (void)numprocs1;
    (void)is2;
    (void)ie2;
    (void)EVec1;
}

static void ClusterCol_SerialOverlapSave(int n)
{
    const char *env = getenv("OPENMX_CLUSTER_GPU_OVERLAP_CACHE_MB");
    double cap_mib = env == NULL ? 2048.0 : atof(env);
    size_t bytes = ClusterCol_CheckedMulCount((size_t)n, (size_t)n,
                                               "serial overlap elements");
    unsigned long long available_kib = 0;
    char line[256];
    FILE *file;

    bytes = ClusterCol_CheckedMulCount(bytes, sizeof(double), "serial overlap bytes");
    if (!(cap_mib > 0.0) || (double)bytes > cap_mib * 1048576.0) return;
    /* There are at most two spin roots. Limit each to 1/32 of currently
       available node RAM rather than relying on malloc's overcommit. */
    file = fopen("/proc/meminfo", "r");
    if (file == NULL) return;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (sscanf(line, "MemAvailable: %llu kB", &available_kib) == 1) break;
    }
    fclose(file);
    if ((unsigned long long)bytes > available_kib / 32ULL * 1024ULL) return;

    ClusterCol_SerialOverlapReset();
    ClusterCol_SerialOverlap = (double*)malloc(bytes);
    if (ClusterCol_SerialOverlap == NULL) return;
    wait_hipfunc(hipMemcpyAsync(ClusterCol_SerialOverlap, ClusterCol_gpusolver_ctx.d_S,
                               bytes, hipMemcpyDeviceToHost, ClusterCol_gpusolver_ctx.stream));
    wait_hipfunc(hipStreamSynchronize(ClusterCol_gpusolver_ctx.stream));
    ClusterCol_SerialOverlapN = n;
}

/* Serial spin turns must return the selected solver's scratch before the
   next owner is admitted.  On MI300A the MAGMA-named real entry selects
   rocSOLVER and may retry with real MAGMA, so either library can retain
   memory.  Keep the existing release behavior on other devices. */
static void ClusterCol_ReleaseSerialSolverWorkspaces(void)
{
    (void)openmx_magma_release_z_workspace();
    if (openmx_gpu_is_mi300a()) {
        (void)openmx_hipsolver_release_workspace();
        (void)openmx_magma_release_d_workspace();
    }
}

/* Memory-limited spin case: all ranks assemble the same spin in the same
   order, but only its root allocates dense device matrices. Preserve the
   compact host eigenvectors for DM construction and release device/library
   workspaces before admitting the next root. A bounded host overlap cache
   avoids repeating the unchanged S eigensolve on every SCF iteration. */
static void ClusterCol_GpuSolverRootDenseSerialPath(
    int SCF_iter, double **ko, double *****nh, double ****CntOLP,
    int numprocs0, int myworld1, int myid1, int *MP, int n, int MaxN)
{
    int packed = Set_Hamiltonian_GpuSolver_Packed_CacheReady() &&
                 Set_Hamiltonian_GpuSolver_Packed_OrderMode() == 0;

    if (packed) Set_Hamiltonian_GpuSolver_SetMP(MP);
    ClusterCol_RefineBeginSolve(SCF_iter, 2, -1);   /* serialized spins keep the FP64 eigensolver */
    ClusterCol_ReleaseGpuSolverCachedEVec(myid1);
    ClusterCol_DetailTimer_Begin(myid1, SCF_iter, n, MaxN);
    if (myid1 == 0) {
        ClusterCol_GpuSolver_Destroy();
        openmx_gemmul8ReleaseWorkspaces();
        ClusterCol_ReleaseSerialSolverWorkspaces();
    }
    MPI_Barrier(mpi_comm_level1);

    for (int spin = 0; spin <= SpinP_switch; ++spin) {
        int owns_dense = myid1 == 0 && (numprocs0 == 1 || myworld1 == spin);
        int local_cached = !owns_dense ||
            (ClusterCol_SerialOverlap != NULL && ClusterCol_SerialOverlapN == n);
        int cached = 0;
        MPI_Allreduce(&local_cached, &cached, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
        if (owns_dense) ClusterCol_GpuSolver_EnsureMatrixCapacity(n);
        for (int hamiltonian = 0; hamiltonian <= 1; ++hamiltonian) {
            double *device_matrix = owns_dense
                ? (hamiltonian ? ClusterCol_gpusolver_ctx.d_H : ClusterCol_gpusolver_ctx.d_S)
                : NULL;
            if (!hamiltonian && cached) {
                if (owns_dense) {
                    size_t bytes = (size_t)n * (size_t)n * sizeof(double);
                    wait_hipfunc(hipMemcpyAsync(device_matrix, ClusterCol_SerialOverlap,
                                               bytes, hipMemcpyHostToDevice,
                                               ClusterCol_gpusolver_ctx.stream));
                    wait_hipfunc(hipStreamSynchronize(ClusterCol_gpusolver_ctx.stream));
                    ClusterCol_gpusolver_ctx.transformed_s_valid = 1;
                    ClusterCol_gpusolver_ctx.transformed_s_dim = n;
                    if (ClusterCol_GpuVerbose())
                        printf("CLUSTER_OVERLAP_CACHE SCF=%d spin=%d n=%d reused %.1f MiB\n",
                               SCF_iter, spin, n, bytes / 1048576.0);
                }
                continue;
            }
            if (packed) {
                if (owns_dense) {
                    int tnum = Set_Hamiltonian_GpuSolver_Packed_Size();
                    int *order = Set_Hamiltonian_GpuSolver_Packed_OrderGA();
                    double *matrix = hamiltonian ? Set_Hamiltonian_GpuSolver_Packed_H(spin)
                                                 : Set_Hamiltonian_GpuSolver_Packed_Overlap();
                    int *index;
                    if (!Set_Hamiltonian_GpuSolver_Packed_OwnsCache() || order == NULL || matrix == NULL)
                        ClusterCol_AbortWithMessage("Missing packed matrix for serial GPU spin solve.");
                    index = (int*)ClusterCol_MallocArray((size_t)tnum, sizeof(int), "serial dense index");
                    ClusterCol_BuildDenseIndex(order, MP, n, tnum, index);
                    ClusterCol_BuildDeviceDenseFromPacked(matrix, index, tnum, n, device_matrix);
                    free(index);
                }
            }
            else {
                Patch2Device_Cluster_Owner(hamiltonian ? nh[spin] : CntOLP,
                                           MP, owns_dense, n, device_matrix);
            }
            if (owns_dense) {
                if (!hamiltonian) {
                    ClusterCol_GpuSolver_PrepareTransformedSDevice(1, n, ko[spin]);
                    ClusterCol_SerialOverlapSave(n);
                }
                else {
                    size_t count = ClusterCol_CheckedMulCount((size_t)n, (size_t)MaxN,
                                                               "serial root eigenvectors");
                    double *evec = (double*)ClusterCol_MallocArray(count, sizeof(double),
                                                                  "serial root eigenvectors");
                    ClusterCol_GpuSolver_SolveHamiltonianDevice(n, MaxN, ko[spin], evec, -1, 0);
                    ClusterCol_StashGpuSolverDenseEVec(myid1, spin, n, MaxN, &evec);
                }
            }
        }
        if (owns_dense) {
            ClusterCol_GpuSolver_Destroy();
            openmx_gemmul8ReleaseWorkspaces();
            ClusterCol_ReleaseSerialSolverWorkspaces();
        }
        MPI_Barrier(mpi_comm_level1);
    }
    ClusterCol_RefineEndSolve();
}

void solve_evp_real_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, int *nblk, 
                      int *mpi_comm_rows_int, int *mpi_comm_cols_int);

void elpa_solve_evp_real_2stage_double_impl_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, 
                                              int *nblk, int *na_cols1, int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld);


static double Lapack_LU_Dinverse(int n, double *A);
static double Calc_Oscillator_Strength( int n, int UMOmax, int Nocc[2], int *MP,
                                        int **ind2ind, 
					double **Z0, double **A, double **Z, 
					double **Ax, double **Ay, double **Az,
					double Utot1, double Utot2,
					double XANES_Res[10] );

static void Calc_XANES_Col( int n, int MaxN, int myid1, 
			    double **ko,
			    double *****nh, 
			    double ****CntOLP,
			    double *****CDM,
			    double *****EDM,
			    double Eele0[2], double Eele1[2],
			    int myworld1,
			    int *NPROCS_ID1,
			    int *Comm_World1,
			    int *NPROCS_WD1,
			    int *Comm_World_StartID1,
			    MPI_Comm *MPI_CommWD1,
			    int *MP,
			    int *is2,
			    int *ie2,
			    double *Ss,
			    double *Cs,
			    double *Hs,
			    double *CDM1,
			    double *EDM1,
			    double *PDM1,
			    int size_H1,
			    int *SP_NZeros,
			    int *SP_Atoms,
			    double **EVec1,
			    double *Work1);

static void Save_LCAO_Col( int n, int MaxN, int myid1, int *is2, int *ie2, 
			   int *MP, double ****OLP0, double **EVec1, 
			   double **ko, int *NPROCS_WD1, int *Comm_World_StartID1 );

static void Save_DOS_Col( int n, int MaxN, int myid1, int *is2, int *ie2, 
			  int *MP, double *****nh, double ****OLP0, double **EVec1, 
			  double **ko, int *NPROCS_WD1, int *Comm_World_StartID1 );



static double Calc_DM_Cluster_collinear(int myid0,
					int numprocs0,
					int myid1,
					int numprocs1,
					int myworld1,
					int size_H1,
					int *is2,
					int *ie2,
					int *MP,
					int n,
					MPI_Comm *MPI_CommWD1,
					int *Comm_World_StartID1,
					double *****CDM,
					double *****EDM,
					double **ko,
					double *DM1,
					double *EDM1,
					double *PDM1,
					double *Work1,
					double **EVec1, 
					int *SP_NZeros,
					int *SP_Atoms );

static double Calc_ParDM_Cluster_collinear(
					   int myid0,
					   int numprocs0,
					   int myid1,
					   int numprocs1,
					   int myworld1,
					   int size_H1,
					   int *is2,
					   int *ie2,
					   int *MP,
					   int n,
					   MPI_Comm *MPI_CommWD1,
					   int *Comm_World_StartID1,
					   double *****CDM,
					   double **ko,
					   double *ParDM1,
					   double *Work1,
					   double **EVec1, 
					   int *SP_NZeros,
					   int *SP_Atoms );


static double Calc_KS_orbitals_on_grid_single(
                                        int myid0,
					int numprocs0,
					int myid1,
					int numprocs1,
					int myworld1,
					int size_H1,
					int *is2,
					int *ie2,
					int *MP,
					int n,
					MPI_Comm *MPI_CommWD1,
					int *Comm_World_StartID1,
					double *****CDM,
					double *****EDM,
					double **ko,
					double *DM1,
					double *EDM1,
					double *PDM1,
					double *Work1,
					double **EVec1, 
					int *SP_NZeros,
					int *SP_Atoms );


static double Calc_HF_Vx_single(
    int myid0,
    int numprocs0,
    int myid1,
    int numprocs1,
    int myworld1,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,
    double *****CDM,
    double *****EDM,
    double **ko,
    double *DM1,
    double *EDM1,
    double *PDM1,
    double *Work1,
    double **EVec1, 
    int *SP_NZeros,
    int *SP_Atoms );



static double Exchange_Poisson( double *work1_on_grid, double *work2_on_grid );


double Cluster_DFT_Col(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double **ko,
                   double *****nh, 
                   double ****CntOLP,
                   double *****CDM,
                   double *****EDM,
                   double Eele0[2], double Eele1[2],
		   int myworld1,
		   int *NPROCS_ID1,
		   int *Comm_World1,
		   int *NPROCS_WD1,
		   int *Comm_World_StartID1,
		   MPI_Comm *MPI_CommWD1,
                   int *MP,
		   int *is2,
		   int *ie2,
		   double *Ss,
		   double *Cs,
		   double *Hs,
		   double *CDM1,
		   double *EDM1,
		   double *PDM1,
		   int size_H1,
                   int *SP_NZeros,
                   int *SP_Atoms,
                   double **EVec1,
                   double *Work1)
{
  int mgpu_scope_previous=openmx_mgpu_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag);
  static int firsttime=1;
  int i,j,l,n,n2,n1,i1,i1s,j1,k1,l1;
  int wan,HOMO0,HOMO1;
  int spin,po,num0,num1,ires;
  int ct_AN,k,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum,loopN,Gc_AN;
  int MA_AN,LB_AN,GB_AN,Bnum,MaxN;
  int wan1,mul,m,bcast_flag;
  int *is1,*ie1;
  double time0,lumos,av_num;
  double *OneD_Mat1;
  double TZ,my_sum,sum,sumE,max_x=60.0;
  double sum0,sum1,sum2,sum3;
  double My_Eele1[2],tmp1,tmp2;
  double Num_State,x,FermiF,Dnum,Dnum2;
  double FermiF2,x2,diffF;
  double dum,ChemP_MAX,ChemP_MIN,spin_degeneracy;
  double TStime,TEtime;
  double FermiEps = 1.0e-13;
  double EV_cut0;
  double res;
  int numprocs0,myid0;
  int numprocs1,myid1;
  int ID,p,world_Snd,world_Rcv; 
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  double OLP_eigen_cut=Threshold_OLP_Eigen;
  char file_EV[YOUSO10] = ".EV";
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp_EV;
  double stime, etime;
  double time1,time2,time3,time4,time5,time6,time7,time8;
  int use_gpusolver_direct_cluster_dm = 0;
  int gpu_diag_mode = 0;
  int rebuild_distributed_overlap = (SCF_iter == 1);
  int gpusolver_direct_evec_scattered = 0;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  int info,ig,jg,il,jl,prow,pcol,brow,bcol;
  int ZERO=0, ONE=1;
  double alpha = 1.0; double beta = 0.0;
  int LOCr, LOCc, node, irow, icol;
  double C_spin_i1,mC_spin_i1;
  int sp;

  int ID0,IDS,IDR,Max_Num_Snd_EV,Max_Num_Rcv_EV;
  int *Num_Snd_EV,*Num_Rcv_EV;
  int *index_Snd_i=NULL,*index_Snd_j=NULL,*index_Rcv_i=NULL,*index_Rcv_j=NULL;
  double *EVec_Snd=NULL,*EVec_Rcv=NULL;
  MPI_Status stat;
  MPI_Request request;

  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n = n + Spe_Total_CNO[wanA];
  }
  n2 = n + 2;

  /* GPU dispatch (added by H.Kawai): assign HIP/OpenMP target device when GPUSOLVER is requested */
  if (scf_eigen_lib_flag == GPUSOLVER && n >= Cluster_DFT_Col_GpuSwitchNum() &&
      Set_Hamiltonian_OpenMP_Rank_Is_Selected()) {
      set_hip_default_device_from_local_rank_noncollective();
  }

  /*
  if (firsttime){
    Hex0 = (double*)malloc(sizeof(double)*n*n);
    Hex = (double*)malloc(sizeof(double)*n*n);
    for (i=0; i<(n*n); i++){
      Hex0[i] = 0.0;
      Hex[i] = 0.0;
    }
  }
  */
  
  /****************************************************
                 Allocation of arrays
  ****************************************************/

  is1 = (int*)malloc(sizeof(int)*numprocs1);
  ie1 = (int*)malloc(sizeof(int)*numprocs1);

  Num_Snd_EV = (int*)malloc(sizeof(int)*numprocs1);
  Num_Rcv_EV = (int*)malloc(sizeof(int)*numprocs1);

  time1 = 0.0;
  time2 = 0.0;
  time3 = 0.0;
  time4 = 0.0;
  time5 = 0.0;
  time6 = 0.0;
  time7 = 0.0;
  time8 = 0.0;

  if      (SpinP_switch==0) spin_degeneracy = 2.0;
  else if (SpinP_switch==1) spin_degeneracy = 1.0;

  /****************************************************
                   total core charge
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ = TZ + Spe_Core_Charge[wan];
  }
 
  /****************************************************
         find the numbers of partions for MPI
  ****************************************************/

  if ( numprocs1<=n ){

    av_num = (double)n/(double)numprocs1;

    for (ID=0; ID<numprocs1; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs1-1] = n; 

  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=n; ID<numprocs1; ID++){
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  if (scf_eigen_lib_flag==GPUSOLVER && SCF_iter==1) {
    ClusterCol_SerialOverlapReset();
    ClusterCol_ugrid.overlap_n=0;
    if (!gpusolver2_flag) ClusterCol_DistributedOverlapN = 0;
  }
  /* a solve outside the root dense path is an FP64 one; only SCF solves
     are refined */
  ClusterCol_refine.last_refined = 0;
  ClusterCol_refine.scf_mode = (strcasecmp(mode,"scf")==0);
  if (scf_eigen_lib_flag==GPUSOLVER && gpusolver2_flag==0 && (openmx_mgpu_host_required() || Cluster_DFT_Col_GpuSwitchNum()<=n)){
    /* the states a refined solve of this step would refine: the memory
       admission counts its product blocks */
    int refine_maxn = ClusterCol_GpuPathMaxN(SCF_iter,mode,TZ,n,
                                             ClusterCol_RefineFirstStepStates(mode,n,TZ,SpinP_switch,numprocs0,0));
    gpu_diag_mode = ClusterCol_GpuDiagFits(n,refine_maxn,myworld1,myid1);
  }
  if (gpu_diag_mode){
    ClusterCol_SetMaxNAndPartitions(SCF_iter,mode,TZ,n,numprocs1,
                                    ClusterCol_RefineFirstStepStates(mode,n,TZ,SpinP_switch,numprocs0,
                                                                     gpu_diag_mode==2),
                                    &MaxN,is2,ie2);
    use_gpusolver_direct_cluster_dm = !ClusterCol_mgpu_host_dense;
    firsttime = 0;
    if (gpu_diag_mode == 2)
      ClusterCol_GpuSolverRootDenseSerialPath(SCF_iter,ko,nh,CntOLP,
                                             numprocs0,myworld1,myid1,MP,n,MaxN);
    else
      ClusterCol_GpuSolverRootDensePath(SCF_iter,SpinP_switch,ko,nh,CntOLP,
                                       numprocs0,myid0,myworld1,numprocs1,myid1,
                                       MPI_CommWD1,Comm_World_StartID1,MP,is2,ie2,n,MaxN,EVec1);
    goto diagonalize_finished;
  }

  if (!gpusolver2_flag && SpinP_switch==1 && numprocs0>1 &&
      ClusterCol_gs2_fallback_n==n && ClusterCol_ugrid.n==n) {
    /* The distributed FP64 solve needs all states at the first SCF step;
       the native refined path alone may request a reduced first panel. */
    ClusterCol_SetMaxNAndPartitions(SCF_iter,mode,TZ,n,numprocs1,n,&MaxN,is2,ie2);
    firsttime=0;
    ClusterCol_UnionGs2Path(n,MaxN,ko,nh,CntOLP,MP,myworld1,myid1,is2,ie2,
        NPROCS_WD1,Comm_World_StartID1,EVec1);
    goto diagonalize_finished;
  }

  /****************************************************
         1. diagonalize the overlap matrix     
         2. search negative eigenvalues
  ****************************************************/

  /* A previous root GPU solve has not initialized distributed Ss. Agree
     before entering the overlap collectives on the first ELPA fallback.
     CPU-selected solvers retain their original SCF_iter == 1 behavior. */
  if (scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag) {
    int local_ready = (ClusterCol_DistributedOverlapN == n);
    int all_ready = 0;
    MPI_Allreduce(&local_ready, &all_ready, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
    if (!all_ready) {
      rebuild_distributed_overlap = 1;
      if (SCF_iter > 1 && myid0 == Host_ID && ClusterCol_GpuVerbose()) {
        printf("<Cluster_DFT_Col> Rebuilding distributed overlap for ELPA fallback at SCF=%d (n=%d).\n",
               SCF_iter, n);
        fflush(stdout);
      }
    }
  }

  MPI_Barrier(mpi_comm_level1);

  if (rebuild_distributed_overlap){
    Overlap_Cluster_Ss(CntOLP,Cs,MP,myworld1);
  }

  if (SpinP_switch==1 && numprocs0==1){
    Hamiltonian_Cluster_Hs(nh[0],Hs,MP,0,0);
  }
  else{
    for (spin=0; spin<=SpinP_switch; spin++){
      Hamiltonian_Cluster_Hs(nh[spin],Hs,MP,spin,myworld1);
    } 
  }

  if (rebuild_distributed_overlap){

    if (measure_time) dtime(&stime);

    MPI_Comm_split(MPI_CommWD1[myworld1],my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(MPI_CommWD1[myworld1],my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

    if (ClusterCol_UseGs2(n)){

      /* distributed multi-GPU eigensolver (ELPA, AMD GPU kernels) */
      int gs2_info = openmx_gs2_eigen_real(n, n, Cs, descC, &ko[0][1], Ss, descS);
      if (gs2_info!=0){
        printf("Cluster_DFT_Col: the gpusolver2 overlap eigensolver failed (info=%d)\n",gs2_info);
        MPI_Abort(mpi_comm_level1,1);
      }
    }

    else if (scf_eigen_lib_flag==1){

      F77_NAME(solve_evp_real,SOLVE_EVP_REAL)(&n, &n, Cs, &na_rows, &ko[0][1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
    }

    else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){

#ifndef kcomp

      int mpiworld;
      mpiworld = MPI_Comm_c2f(MPI_CommWD1[myworld1]);

      F77_NAME(elpa_solve_evp_real_2stage_double_impl,ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(&n, &n, Cs, &na_rows, &ko[0][1], 
                                            Ss, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld);

#endif

    }

    MPI_Comm_free(&mpi_comm_rows);
    MPI_Comm_free(&mpi_comm_cols);

    /* print to the standard output */

    if (2<=level_stdout && myid0==Host_ID){
      for (l=1; l<=n; l++){
	printf("  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[0][l]);fflush(stdout);
      }
    }

    /* minus eigenvalues to 1.0e-10 */

    for (l=1; l<=n; l++){
      if (ko[0][l]<0.0) ko[0][l] = 1.0e-10;
    }

    /* calculate S*1/sqrt(ko) */

    for (l=1; l<=n; l++){
      ko[0][l] = 1.0/sqrt(ko[0][l]);
    }

    for(i=0; i<na_rows; i++){
      for(j=0; j<na_cols; j++){
	jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	Ss[j*na_rows+i] = Ss[j*na_rows+i]*ko[0][jg];
      }
    }

    if (scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag)
      ClusterCol_DistributedOverlapN = n;

    if (measure_time){
      dtime(&etime);
      time1 += etime - stime; 
    }
  }

  /****************************************************
    calculations of eigenvalues for up and down spins

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  /* find the maximum states in solved eigenvalues */

  if (SCF_iter==1){
    MaxN = n;
  }
  else{

    if      ( strcasecmp(mode,"scf")==0 ) 
      lumos = (double)n*0.100;
    else if ( strcasecmp(mode,"dos")==0 )
      lumos = (double)n*0.200;      
    else if ( strcasecmp(mode,"lcaoout")==0 )
      lumos = (double)n*0.200;      
    else if ( strcasecmp(mode,"xanes")==0 )
      lumos = (double)n*0.200;      
    else if ( strcasecmp(mode,"diag")==0 )
      lumos = (double)n*0.200;      
    else 
      lumos = (double)n*0.200;      

    if (lumos<400.0) lumos = 400.0;
    MaxN = (TZ-system_charge)/2 + (int)lumos;
    if (n<MaxN) MaxN = n;
    
    if (cal_partial_charge) MaxN = n; 
  }

  if ( numprocs1<=MaxN ){
    
    av_num = (double)MaxN/(double)numprocs1;
    for (ID=0; ID<numprocs1; ID++){
      is2[ID] = (int)(av_num*(double)ID) + 1; 
      ie2[ID] = (int)(av_num*(double)(ID+1)); 
    }
    
    is2[0] = 1;
    ie2[numprocs1-1] = MaxN; 
  }

  else{

    for (ID=0; ID<MaxN; ID++){
      is2[ID] = ID + 1; 
      ie2[ID] = ID + 1;
    }

    for (ID=MaxN; ID<numprocs1; ID++){
      is2[ID] =  1;
      ie2[ID] =  0;
    }
  }

  /* making data structure of MPI communicaition for eigenvectors */

  for (ID=0; ID<numprocs1; ID++){
    Num_Snd_EV[ID] = 0;
    Num_Rcv_EV[ID] = 0;
  }

  for (i=0; i<na_rows; i++){

    ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;

    po = 0;
    for (ID=0; ID<numprocs1; ID++){
      if (is2[ID]<=ig && ig <=ie2[ID]){
        po = 1;
        ID0 = ID;
        break;
      }
    }

    if (po==1) Num_Snd_EV[ID0] += na_cols;
  }

  for (ID=0; ID<numprocs1; ID++){
    IDS = (myid1 + ID) % numprocs1;
    IDR = (myid1 - ID + numprocs1) % numprocs1;
    if (ID!=0){
      MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
      MPI_Recv(&Num_Rcv_EV[IDR], 1, MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
      MPI_Wait(&request,&stat);
    }
    else{
      Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
    }
  }

  Max_Num_Snd_EV = 0;
  Max_Num_Rcv_EV = 0;
  for (ID=0; ID<numprocs1; ID++){
    if (Max_Num_Snd_EV<Num_Snd_EV[ID]) Max_Num_Snd_EV = Num_Snd_EV[ID];
    if (Max_Num_Rcv_EV<Num_Rcv_EV[ID]) Max_Num_Rcv_EV = Num_Rcv_EV[ID];
  }  

  Max_Num_Snd_EV++;
  Max_Num_Rcv_EV++;

  index_Snd_i = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  index_Snd_j = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  EVec_Snd = (double*)malloc(sizeof(double)*Max_Num_Snd_EV);
  index_Rcv_i = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  index_Rcv_j = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  EVec_Rcv = (double*)malloc(sizeof(double)*Max_Num_Rcv_EV);

  /* for PrintMemory */

  if (firsttime && memoryusage_fileout){
    PrintMemory("Cluster_DFT_Col: is1",sizeof(int)*numprocs1,NULL);
    PrintMemory("Cluster_DFT_Col: ie1",sizeof(int)*numprocs1,NULL);
    PrintMemory("Cluster_DFT_Col: Num_Snd_EV",sizeof(int)*numprocs1,NULL);
    PrintMemory("Cluster_DFT_Col: Num_Snd_EV",sizeof(int)*numprocs1,NULL);
    PrintMemory("Cluster_DFT_Col: index_Snd_i",sizeof(int)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_Col: index_Snd_j",sizeof(int)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_Col: index_Rcv_i",sizeof(int)*Max_Num_Rcv_EV,NULL);
    PrintMemory("Cluster_DFT_Col: index_Rcv_j",sizeof(int)*Max_Num_Rcv_EV,NULL);
    PrintMemory("Cluster_DFT_Col: EVec_Snd",sizeof(double)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_Col: EVec_Rcv",sizeof(double)*Max_Num_Rcv_EV,NULL);
  }
  firsttime=0;

  /* spin=myworld1 */

  spin = myworld1;

 diagonalize:

  if (measure_time) dtime(&stime);

  /* pdgemm */

  /* H * U * 1.0/sqrt(ko[l]) */

  for(i=0; i<na_rows_max*na_cols_max; i++){
    Cs[i] = 0.0;
  }

  Cblacs_barrier(ictxt1,"A");
  if (ClusterCol_UseGs2(n))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

  /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

  for(i=0; i<na_rows*na_cols; i++){
    Hs[i] = 0.0;
  }

  Cblacs_barrier(ictxt1,"C");
  if (ClusterCol_UseGs2(n))
    openmx_gs2_pdgemm_("T","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);

  if (measure_time){
    dtime(&etime);
    time2 += etime - stime;
  }

  /* initialize ko */
  for (i1=0; i1<=n; i1++){ ko[spin][i1] = 10000.0; }

  /* The output C matrix is distributed by column. */

  if (measure_time) dtime(&stime);

  MPI_Comm_split(MPI_CommWD1[myworld1],my_pcol,my_prow,&mpi_comm_rows);
  MPI_Comm_split(MPI_CommWD1[myworld1],my_prow,my_pcol,&mpi_comm_cols);

  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

  if (ClusterCol_UseGs2(n)){

    /* distributed multi-GPU eigensolver (ELPA, AMD GPU kernels); all n
       eigenvalues are returned but only the lowest MaxN eigenvectors */
    int gs2_info = openmx_gs2_eigen_real(n, MaxN, Hs, descH, &ko[spin][1], Cs, descC);
    if (gs2_info!=0){
      printf("Cluster_DFT_Col: the gpusolver2 Hamiltonian eigensolver failed (info=%d)\n",gs2_info);
      MPI_Abort(mpi_comm_level1,1);
    }
  }
  else if (scf_eigen_lib_flag==1){
    F77_NAME(solve_evp_real,SOLVE_EVP_REAL)(&n, &MaxN, Hs, &na_rows, &ko[spin][1], Cs,
                                            &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
  }
  else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){

#ifndef kcomp
    int mpiworld;
    mpiworld = MPI_Comm_c2f(MPI_CommWD1[myworld1]);

    F77_NAME(elpa_solve_evp_real_2stage_double_impl,ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(&n, &MaxN, Hs, &na_rows, &ko[spin][1], 
									 		    Cs, &na_rows, &nblk, &na_cols, 
                                                                                            &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld);
#endif

  }

  MPI_Comm_free(&mpi_comm_rows);
  MPI_Comm_free(&mpi_comm_cols);

  if (measure_time){
    dtime(&etime);
    time3 += etime - stime;
  }

  /****************************************************
      transformation to the original eigenvectors.
                       NOTE 244P
  ****************************************************/

  if (measure_time) dtime(&stime);

  for (i=0; i<na_rows*na_cols; i++) Hs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  if (ClusterCol_UseGs2(n))
    openmx_gs2_pdgemm_("T","T",&n,&n,&n,&alpha,Cs,&ONE,&ONE,descC,Ss,&ONE,&ONE,descS,&beta,Hs,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","T",&n,&n,&n,&alpha,Cs,&ONE,&ONE,descC,Ss,&ONE,&ONE,descS,&beta,Hs,&ONE,&ONE,descH);

  /* MPI communications of Hs */

  for (ID=0; ID<numprocs1; ID++){
    
    IDS = (myid1 + ID) % numprocs1;
    IDR = (myid1 - ID + numprocs1) % numprocs1;

    k = 0;
    for(i=0; i<na_rows; i++){
      ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
      if (is2[IDS]<=ig && ig <=ie2[IDS]){

        for (j=0; j<na_cols; j++){
          jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
 
          index_Snd_i[k] = ig;
          index_Snd_j[k] = jg;
          EVec_Snd[k] = Hs[j*na_rows+i];
          k++; 
	}
      }
    }

    if (ID!=0){

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(EVec_Snd, Num_Snd_EV[IDS], MPI_DOUBLE, IDS, 999, MPI_CommWD1[myworld1], &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR], MPI_DOUBLE, IDR, 999, MPI_CommWD1[myworld1], &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }
    }
    else{
      for(k=0; k<Num_Snd_EV[IDS]; k++){
        index_Rcv_i[k] = index_Snd_i[k];
        index_Rcv_j[k] = index_Snd_j[k];
        EVec_Rcv[k] = EVec_Snd[k];
      } 
    }

    for(k=0; k<Num_Rcv_EV[IDR]; k++){
      ig = index_Rcv_i[k];
      jg = index_Rcv_j[k];
      m = (jg-1)*(ie2[myid1]-is2[myid1]+1)+ig-is2[myid1]; 
      EVec1[spin][m] = EVec_Rcv[k];
    }
  }

  if (measure_time){
    dtime(&etime);
    time4 += etime - stime;
  }

  if (SpinP_switch==1 && numprocs0==1 && spin==0){
    spin++;
    Hamiltonian_Cluster_Hs(nh[spin],Hs,MP,spin,spin);
    goto diagonalize; 
  }

 diagonalize_finished:

  /*
  printf("EVec1\n");
  for (i=0; i<n; i++){
    for (j=0; j<8; j++){
      printf("%8.5f ",EVec1[0][i*MaxN+j]);
    } 
    printf("\n");
  } 
  */

  /*********************************************** 
    MPI: ko
  ***********************************************/

  if (measure_time) dtime(&stime);

  for (sp=0; sp<=SpinP_switch; sp++){
    MPI_Bcast(&ko[sp][1],MaxN,MPI_DOUBLE,Comm_World_StartID1[sp],mpi_comm_level1);
  }

  if ( strcasecmp(mode,"scf")==0 ){

    if (2<=level_stdout){
      for (i1=1; i1<=MaxN; i1++){
	if (SpinP_switch==0)
	  printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n",
		 i1,ko[0][i1],ko[0][i1]);
	else 
	  printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n",
		 i1,ko[0][i1],ko[1][i1]);
      }
    }
    
    /* for XANES */
    if (xanes_calc==1){

      /****************************************************
              searching of chemical potential
      ****************************************************/

      for (spin=0; spin<=SpinP_switch; spin++){

	po = 0;
	loopN = 0;

	ChemP_MAX = 30.0;  
	ChemP_MIN =-30.0;

	do {

	  ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
	  Num_State = 0.0;

	  for (i1=1; i1<=MaxN; i1++){
	    x = (ko[spin][i1] - ChemP)*Beta;
	    if (x<=-max_x) x = -max_x;
	    if (max_x<=x)  x = max_x;
	    FermiF = 1.0/(1.0 + exp(x));
	    Num_State += FermiF;
	  }

	  Dnum = HOMO_XANES[spin] - Num_State;
	  if (0.0<=Dnum) ChemP_MIN = ChemP;
	  else           ChemP_MAX = ChemP;
	  if (fabs(Dnum)<1.0e-12) po = 1;

	  if (myid1==Host_ID && 2<=level_stdout){
	    printf("spin=%2d ChemP=%15.12f HOMO_XANES=%2d Num_state=%15.12f\n",
                    spin,ChemP,HOMO_XANES[spin],Num_State); 
	  }

	  loopN++;

	} while (po==0 && loopN<1000); 

        ChemP_XANES[spin] = ChemP;
        Cluster_HOMO[spin] = HOMO_XANES[spin];

      } /* spin */

      /* set ChemP */

      ChemP = 0.5*(ChemP_XANES[0] + ChemP_XANES[1]); 

    } /* end of if (xanes_calc==1) */

    /* start of else for if (xanes_calc==1) */

    else{

      /****************************************************
              searching of chemical potential
      ****************************************************/

      double Beta_trial1;

      /* first, find ChemP at 1200 K */

      Beta_trial1 = 1.0/kB/(1200.0/eV2Hartree);

      po = 0;
      loopN = 0;

      ChemP_MAX = 30.0;  
      ChemP_MIN =-30.0;
  
      do {

	ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
	Num_State = 0.0;

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i1=1; i1<=MaxN; i1++){
	    x = (ko[spin][i1] - ChemP)*Beta_trial1;
	    if (x<=-max_x) x = -max_x;
	    if (max_x<=x)  x = max_x;
	    FermiF = 1.0/(1.0 + exp(x));
	    Num_State = Num_State + spin_degeneracy*FermiF;
	    if (0.5<FermiF) Cluster_HOMO[spin] = i1;
	  }
	}

	Dnum = (TZ - Num_State) - system_charge;
	if (0.0<=Dnum) ChemP_MIN = ChemP;
	else           ChemP_MAX = ChemP;
	if (fabs(Dnum)<1.0e-12) po = 1;

	if (myid1==Host_ID && 2<=level_stdout){
	  printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",ChemP,TZ,Num_State); 
	}

	loopN++;

      } while (po==0 && loopN<1000); 

      /* second, find ChemP at the temperatue, starting from the previously found ChemP. */

      po = 0;
      loopN = 0;

      ChemP_MAX = 30.0;  
      ChemP_MIN =-30.0;
  
      do {

	if (loopN!=0){
	  ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
	}

	Num_State = 0.0;

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i1=1; i1<=MaxN; i1++){
	    x = (ko[spin][i1] - ChemP)*Beta;
	    if (x<=-max_x) x = -max_x;
	    if (max_x<=x)  x = max_x;
	    FermiF = 1.0/(1.0 + exp(x));

	    Num_State = Num_State + spin_degeneracy*FermiF;
	    if (0.5<FermiF) Cluster_HOMO[spin] = i1;
	  }
	}

	Dnum = (TZ - Num_State) - system_charge;
	if (0.0<=Dnum) ChemP_MIN = ChemP;
	else           ChemP_MAX = ChemP;
	if (fabs(Dnum)<1.0e-12) po = 1;

	if (myid1==Host_ID && 2<=level_stdout){
	  printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",ChemP,TZ,Num_State); 
	}

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

    } /* end of else for if (xanes_calc==1) */

    /****************************************************
          Energies by summing up eigenvalues
    ****************************************************/

    Eele0[0] = 0.0;
    Eele0[1] = 0.0;

    for (spin=0; spin<=SpinP_switch; spin++){
      for (i1=1; i1<=MaxN; i1++){

	if (xanes_calc==1) 
          x = (ko[spin][i1] - ChemP_XANES[spin])*Beta;
        else 
          x = (ko[spin][i1] - ChemP)*Beta;

	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = FermiFunc(x,spin,i1,&po,&x);

	Eele0[spin] += ko[spin][i1]*FermiF;

      }
    }

    if (SpinP_switch==0){
      Eele0[1] = Eele0[0];
    }

    /****************************************************
        LCAO coefficients are stored for calculating
                  values of MOs on grids
    ****************************************************/

    if (SpinP_switch==0){
      if ( (Cluster_HOMO[0]-num_HOMOs+1)<1 )  num_HOMOs = Cluster_HOMO[0];
      if ( (Cluster_HOMO[0]+num_LUMOs)>MaxN ) num_LUMOs = MaxN - Cluster_HOMO[0];
    }
    else if (SpinP_switch==1){
      if ( (Cluster_HOMO[0]-num_HOMOs+1)<1 )  num_HOMOs = Cluster_HOMO[0];
      if ( (Cluster_HOMO[1]-num_HOMOs+1)<1 )  num_HOMOs = Cluster_HOMO[1];
      if ( (Cluster_HOMO[0]+num_LUMOs)>MaxN ) num_LUMOs = MaxN - Cluster_HOMO[0];
      if ( (Cluster_HOMO[1]+num_LUMOs)>MaxN ) num_LUMOs = MaxN - Cluster_HOMO[1];
    }

    if (myid0==Host_ID){
      if (SpinP_switch==0 && 2<=level_stdout){
	printf("  HOMO = %2d\n",Cluster_HOMO[0]);
      }
      else if (SpinP_switch==1 && 2<=level_stdout){
	printf("  HOMO for up-spin   = %2d\n",Cluster_HOMO[0]);
	printf("  HOMO for down-spin = %2d\n",Cluster_HOMO[1]);
      }
    }

    if (use_gpusolver_direct_cluster_dm && MO_fileout==1 && !gpusolver_direct_evec_scattered){
      time4 += ClusterCol_ScatterGpuSolverCachedEVec(SpinP_switch,numprocs0,myworld1,myid1,numprocs1,
                                                    MPI_CommWD1,n,MaxN,is2,ie2,EVec1);
      gpusolver_direct_evec_scattered = 1;
    }

    if (MO_fileout==1){  

      /* allocation of arrays */
      double *array0;    
      int *is3,*ie3;
      int numprocs3,ID1;

      array0 = (double*)malloc(sizeof(double)*(n+2));
      is3 = (int*)malloc(sizeof(int)*numprocs0);
      ie3 = (int*)malloc(sizeof(int)*numprocs0);

      /* HOMOs */

      for (spin=0; spin<=SpinP_switch; spin++){

	/* set is3 and ie3 */

	numprocs3 = NPROCS_WD1[spin];

	if ( numprocs3<=MaxN ){

	  av_num = (double)MaxN/(double)numprocs3;
	  for (ID=0; ID<numprocs3; ID++){
	    is3[ID] = (int)(av_num*(double)ID) + 1; 
	    ie3[ID] = (int)(av_num*(double)(ID+1)); 
	  }

	  is3[0] = 1;
	  ie3[numprocs3-1] = MaxN; 
	}

	else{
	  for (ID=0; ID<MaxN; ID++){
	    is3[ID] = ID + 1; 
	    ie3[ID] = ID + 1;
	  }
	  for (ID=MaxN; ID<numprocs3; ID++){
	    is3[ID] =  1;
	    ie3[ID] =  0;
	  }
	}

	/* loop for j */

	for (j=0; j<num_HOMOs; j++){

	  j1 = Cluster_HOMO[spin] - j;

	  /* store eigenvalue */
	  HOMOs_Coef[0][spin][j][0][0].r = ko[spin][j1] ;

	  /* store EVec1 */
	  if (numprocs0==1){
	    for (k=0; k<n; k++){

	      m = k*(ie2[myid1]-is2[myid1]+1)+j1-1;
	      array0[k] = EVec1[spin][m];
	    }
	  }
	  else{

	    po = 0;
	    for (ID=0; ID<numprocs3; ID++){
	      if (is3[ID]<=j1 && j1 <=ie3[ID]){
		po = 1;
		ID0 = ID;
		break;
	      }
	    }

	    ID1 = Comm_World_StartID1[spin] + ID0;

	    if (myid0==ID1){
	      for (k=0; k<n; k++){

		m = k*(ie2[myid1]-is2[myid1]+1)+j1-is3[ID0];
		array0[k] = EVec1[spin][m];
	      }
	    }

	    /* MPI communications */
	    MPI_Bcast(array0, n, MPI_DOUBLE, ID1, mpi_comm_level1);  
	  }

	  /* store eigenvector */
	  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	    wanA = WhatSpecies[GA_AN];
	    tnoA = Spe_Total_CNO[wanA];
	    Anum = MP[GA_AN];
	    for (i=0; i<tnoA; i++){
	      HOMOs_Coef[0][spin][j][GA_AN][i].r = array0[Anum+i-1];
	      HOMOs_Coef[0][spin][j][GA_AN][i].i = 0.0;
	    }
	  }
	}
      }

      /* LUMOs */
   
      for (spin=0; spin<=SpinP_switch; spin++){

	/* set is3 and ie3 */

	numprocs3 = NPROCS_WD1[spin];

	if ( numprocs3<=MaxN ){

	  av_num = (double)MaxN/(double)numprocs3;
	  for (ID=0; ID<numprocs3; ID++){
	    is3[ID] = (int)(av_num*(double)ID) + 1; 
	    ie3[ID] = (int)(av_num*(double)(ID+1)); 
	  }

	  is3[0] = 1;
	  ie3[numprocs3-1] = MaxN; 
	}

	else{
	  for (ID=0; ID<MaxN; ID++){
	    is3[ID] = ID + 1; 
	    ie3[ID] = ID + 1;
	  }
	  for (ID=MaxN; ID<numprocs3; ID++){
	    is3[ID] =  1;
	    ie3[ID] = -2;
	  }
	}

	/* loop for j */

	for (j=0; j<num_LUMOs; j++){

	  j1 = Cluster_HOMO[spin] + 1 + j;

	  /* store eigenvalue */
	  LUMOs_Coef[0][spin][j][0][0].r = ko[spin][j1];

	  /* store EVec1 */
	  if (numprocs0==1){
	    for (k=0; k<n; k++){

	      m = k*(ie2[myid1]-is2[myid1]+1)+j1-1;
	      array0[k] = EVec1[spin][m];
	    }
	  }
	  else{

	    po = 0;
	    for (ID=0; ID<numprocs3; ID++){
	      if (is3[ID]<=j1 && j1 <=ie3[ID]){
		po = 1;
		ID0 = ID;
		break;
	      }
	    }

	    ID1 = Comm_World_StartID1[spin] + ID0;

	    if (myid0==ID1){
	      for (k=0; k<n; k++){

		m = k*(ie2[myid1]-is2[myid1]+1) + j1-is3[ID0];
		array0[k] = EVec1[spin][m];
	      }
	    }

	    /* MPI communications */
	    MPI_Bcast(array0, n, MPI_DOUBLE, ID1, mpi_comm_level1);  
	  }

	  /* store eigenvector */
	  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	    wanA = WhatSpecies[GA_AN];
	    tnoA = Spe_Total_CNO[wanA];
	    Anum = MP[GA_AN];
	    for (i=0; i<tnoA; i++){
	      LUMOs_Coef[0][spin][j][GA_AN][i].r = array0[Anum+i-1];
	      LUMOs_Coef[0][spin][j][GA_AN][i].i = 0.0;
	    }
	  }
	}
      }

      /* freeing of array0 */

      free(ie3);
      free(is3);
      free(array0);
    }

    /****************************************************
          density matrix and energy density matrix
                  for up and down spins
    ****************************************************/

    if (use_gpusolver_direct_cluster_dm){
      time6 += ClusterCol_CalcDMRootDense_HIP_Path(myid0,numprocs0,myid1,numprocs1,myworld1,
                                                   SpinP_switch,size_H1,MP,n,MaxN,
                                                   MPI_CommWD1,Comm_World_StartID1,
                                                   CDM,EDM,ko,CDM1,EDM1,PDM1,Work1);
    }
    else{
      time6 += Calc_DM_Cluster_collinear( myid0,numprocs0,myid1,numprocs1,myworld1,
					  size_H1,is2,ie2,MP,n,MPI_CommWD1,Comm_World_StartID1,
					  CDM,EDM,ko,CDM1,EDM1,PDM1,Work1,EVec1,SP_NZeros,SP_Atoms);
    }

    /****************************************************
                   calculation of ParDM
    ****************************************************/

    if ( strcasecmp(mode,"ParDM")==0 ){

      if (use_gpusolver_direct_cluster_dm && !gpusolver_direct_evec_scattered){
        time4 += ClusterCol_ScatterGpuSolverCachedEVec(SpinP_switch,numprocs0,myworld1,myid1,numprocs1,
                                                      MPI_CommWD1,n,MaxN,is2,ie2,EVec1);
        gpusolver_direct_evec_scattered = 1;
      }

      time6 += Calc_ParDM_Cluster_collinear( myid0,numprocs0,myid1,numprocs1,myworld1,
  			  		     size_H1,is2,ie2,MP,n,MPI_CommWD1,Comm_World_StartID1,
					     ParDM,ko,CDM1,Work1,EVec1,SP_NZeros,SP_Atoms);
    }

    /****************************************************
                        Bond Energies
    ****************************************************/
  
    My_Eele1[0] = 0.0;
    My_Eele1[1] = 0.0;

    for (spin=0; spin<=SpinP_switch; spin++){
      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
	GA_AN = M2G[MA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	for (j=0; j<=FNAN[GA_AN]; j++){
	  wanB = WhatSpecies[natn[GA_AN][j]];
	  tnoB = Spe_Total_CNO[wanB];
	  for (k=0; k<tnoA; k++){
	    for (l=0; l<tnoB; l++){
	      My_Eele1[spin] += CDM[spin][MA_AN][j][k][l]*nh[spin][MA_AN][j][k][l];
	    }
	  }
	}
      }
    }
  
    /* MPI, My_Eele1 */
    for (spin=0; spin<=SpinP_switch; spin++){
      MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    }

    if (SpinP_switch==0) Eele1[1] = Eele1[0];

    if (2<=level_stdout && myid0==Host_ID){
      printf("Eele0[0]=%15.12f Eele0[1]=%15.12f\n",Eele0[0],Eele0[1]);
      printf("Eele1[0]=%15.12f Eele1[1]=%15.12f\n",Eele1[0],Eele1[1]);
    }

    if (measure_time){
      dtime(&etime);
      time6 += etime - stime;
    }

    /****************************************************
            calculation of KS orbitals on grid
    ****************************************************/

    /*
    time7 += Calc_KS_orbitals_on_grid_single( myid0,numprocs0,myid1,numprocs1,myworld1,
				              size_H1,is2,ie2,MP,n,MPI_CommWD1,Comm_World_StartID1,
    				              CDM,EDM,ko,CDM1,EDM1,PDM1,Work1,EVec1,SP_NZeros,SP_Atoms);
    */

    /*
    time7 += Calc_HF_Vx_single( myid0,numprocs0,myid1,numprocs1,myworld1,
 		                size_H1,is2,ie2,MP,n,MPI_CommWD1,Comm_World_StartID1,
		                CDM,EDM,ko,CDM1,EDM1,PDM1,Work1,EVec1,SP_NZeros,SP_Atoms);
    */
    
    /****************************************************
                           Output
    ****************************************************/

    if (measure_time) dtime(&stime);

    if (myid0==Host_ID){

      sprintf(file_EV,"%s%s.EV",filepath,filename);

      if ((fp_EV = fopen(file_EV,"w")) != NULL){

	setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"            Eigenvalues (Hartree) for SCF KS-eq.           \n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n\n");

	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",ChemP);
	fprintf(fp_EV,"   Number of States             = %18.14f\n",Num_State);
	if (SpinP_switch==0){
	  fprintf(fp_EV,"   HOMO = %2d\n",Cluster_HOMO[0]);
	}
	else if (SpinP_switch==1){
	  fprintf(fp_EV,"   HOMO for up-spin   = %2d\n",Cluster_HOMO[0]);
	  fprintf(fp_EV,"   HOMO for down-spin = %2d\n",Cluster_HOMO[1]);
	}

	fprintf(fp_EV,"   Eigenvalues\n");
	fprintf(fp_EV,"                Up-spin            Down-spin\n");
	for (i1=1; i1<=MaxN; i1++){
	  if (SpinP_switch==0)
	    fprintf(fp_EV,"      %5d %18.14f %18.14f\n",i1,ko[0][i1],ko[0][i1]);
	  else if (SpinP_switch==1)
	    fprintf(fp_EV,"      %5d %18.14f %18.14f\n",i1,ko[0][i1],ko[1][i1]);
	}
      }

      /* fclose of fp_EV */
      fclose(fp_EV);
    }

    if (use_gpusolver_direct_cluster_dm && 2<=level_fileout && !gpusolver_direct_evec_scattered){
      time4 += ClusterCol_ScatterGpuSolverCachedEVec(SpinP_switch,numprocs0,myworld1,myid1,numprocs1,
                                                    MPI_CommWD1,n,MaxN,is2,ie2,EVec1);
      gpusolver_direct_evec_scattered = 1;
    }

    if (2<=level_fileout){

      if (myid0==Host_ID){

        sprintf(file_EV,"%s%s.EV",filepath,filename);

        if ((fp_EV = fopen(file_EV,"a")) != NULL){
  	  setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */
        }
	else{
	  printf("Failure of saving the EV file.\n");
	}
      }

      if (myid0==Host_ID){

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"  Eigenvalues (Hartree) and Eigenvectors for SCF KS-eq.  \n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");

	fprintf(fp_EV,"\n\n");
	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",ChemP);
	if (SpinP_switch==0){
	  fprintf(fp_EV,"   HOMO = %2d\n",Cluster_HOMO[0]);
	}
	else if (SpinP_switch==1){
	  fprintf(fp_EV,"   HOMO for up-spin   = %2d\n",Cluster_HOMO[0]);
	  fprintf(fp_EV,"   HOMO for down-spin = %2d\n",Cluster_HOMO[1]);
	}

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"   LCAO coefficients for up (U) and down (D) spins\n\n");
      }

      /* allocation of arrays */
      double *array0;    
      int *is3,*ie3;
      int numprocs3,ID1;

      array0 = (double*)malloc(sizeof(double)*(n+2)*7);
      is3 = (int*)malloc(sizeof(int)*numprocs0);
      ie3 = (int*)malloc(sizeof(int)*numprocs0);

      /* start of loop spin and i */

      num0 = 6;
      num1 = MaxN/num0 + 1*(MaxN%num0!=0);

      for (spin=0; spin<=SpinP_switch; spin++){

	/* set is3 and ie3 */

	numprocs3 = NPROCS_WD1[spin];

	if ( numprocs3<=MaxN ){

	  av_num = (double)MaxN/(double)numprocs3;
	  for (ID=0; ID<numprocs3; ID++){
	    is3[ID] = (int)(av_num*(double)ID) + 1; 
	    ie3[ID] = (int)(av_num*(double)(ID+1)); 
	  }

	  is3[0] = 1;
	  ie3[numprocs3-1] = MaxN; 
	}

	else{
	  for (ID=0; ID<MaxN; ID++){
	    is3[ID] = ID + 1; 
	    ie3[ID] = ID + 1;
	  }
	  for (ID=MaxN; ID<numprocs3; ID++){
	    is3[ID] =  1;
	    ie3[ID] = -2;
	  }
	}

	for (i=1; i<=num1; i++){

	  if (myid0==Host_ID){ 

	    /* header */ 
	    fprintf(fp_EV,"\n");
	    for (i1=-1; i1<=0; i1++){

	      if      (i1==-1) fprintf(fp_EV,"                     ");
	      else if (i1==0)  fprintf(fp_EV,"                      ");

	      for (j=1; j<=num0; j++){
		j1 = num0*(i-1) + j;
		if (j1<=MaxN){ 
		  if (i1==-1){

  		    if (spin==0)      fprintf(fp_EV," %5d (U)",j1);
		    else if (spin==1) fprintf(fp_EV," %5d (D)",j1);
		  }
		  else if (i1==0){
  		    fprintf(fp_EV,"  %8.5f",ko[spin][j1]);
		  }
		}
	      }
	      fprintf(fp_EV,"\n");
	      if (i1==0)  fprintf(fp_EV,"\n");
	    }
	  }

	  /* MPI communication of EVec1 */

          if (numprocs0==1){

	    for (j=1; j<=num0; j++){

	      j1 = num0*(i-1) + j;

  	      if (j1<=MaxN){
		for (k=0; k<n; k++){

		  m = k*(ie2[myid1]-is2[myid1]+1) + j1-1;
		  array0[(j-1)*n+k] = EVec1[spin][m];
		}
	      }
	    }
	  }
          else{

	    for (j=1; j<=num0; j++){

	      j1 = num0*(i-1) + j;

	      po = 0;
	      for (ID=0; ID<numprocs3; ID++){
		if (is3[ID]<=j1 && j1 <=ie3[ID]){
		  po = 1;
		  ID0 = ID;
		  break;
		}
	      }

	      ID1 = Comm_World_StartID1[spin] + ID0;

  	      if (j1<=MaxN){

		if (myid0==ID1){
		  for (k=0; k<n; k++){

		    m = k*(ie2[myid1]-is2[myid1]+1) + j1-is3[ID0];
		    array0[(j-1)*n+k] = EVec1[spin][m];
		  }
		}

		/* MPI communications */
		MPI_Bcast(&array0[(j-1)*n], n, MPI_DOUBLE, ID1, mpi_comm_level1);
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

	  if (myid0==Host_ID){ 

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
			fprintf(fp_EV,"  %8.5f",array0[(j-1)*n+(i1-1)]);
		      }
		    }
		  
		    fprintf(fp_EV,"\n");

		    i1++;
		  }
		}
	      }
	    }
	  }

	} /* i */
      } /* spin */

      /* freeing of arrays */

      free(ie3);
      free(is3);
      free(array0);

      /* fclose of fp_EV */

      if (myid0==Host_ID){
        fclose(fp_EV);
      }

    } /* end of if (2<=level_fileout) */

    if (measure_time){
      dtime(&etime);
      time8 += etime - stime;
    }

  } /* if ( strcasecmp(mode,"scf")==0 ) */

  else if ( strcasecmp(mode,"dos")==0 ){
    if (use_gpusolver_direct_cluster_dm && !gpusolver_direct_evec_scattered){
      time4 += ClusterCol_ScatterGpuSolverCachedEVec(SpinP_switch,numprocs0,myworld1,myid1,numprocs1,
                                                    MPI_CommWD1,n,MaxN,is2,ie2,EVec1);
      gpusolver_direct_evec_scattered = 1;
    }
    Save_DOS_Col(n,MaxN,myid1,is2,ie2,MP,nh,CntOLP,EVec1,ko,NPROCS_WD1,Comm_World_StartID1);
  }

  else if ( strcasecmp(mode,"lcaoout")==0 ){
    if (use_gpusolver_direct_cluster_dm && !gpusolver_direct_evec_scattered){
      time4 += ClusterCol_ScatterGpuSolverCachedEVec(SpinP_switch,numprocs0,myworld1,myid1,numprocs1,
                                                    MPI_CommWD1,n,MaxN,is2,ie2,EVec1);
      gpusolver_direct_evec_scattered = 1;
    }
    Save_LCAO_Col(n,MaxN,myid1,is2,ie2,MP,CntOLP,EVec1,ko,NPROCS_WD1,Comm_World_StartID1);
  }

  else if ( strcasecmp(mode,"xanes")==0 ){
    if (use_gpusolver_direct_cluster_dm && !gpusolver_direct_evec_scattered){
      time4 += ClusterCol_ScatterGpuSolverCachedEVec(SpinP_switch,numprocs0,myworld1,myid1,numprocs1,
                                                    MPI_CommWD1,n,MaxN,is2,ie2,EVec1);
      gpusolver_direct_evec_scattered = 1;
    }
    Calc_XANES_Col( n,MaxN,myid1,ko,nh,CntOLP,CDM,EDM,Eele0,Eele1,myworld1,NPROCS_ID1,
                    Comm_World1,NPROCS_WD1,Comm_World_StartID1,MPI_CommWD1,
                    MP,is2,ie2,Ss,Cs,Hs,CDM1,EDM1,PDM1,size_H1,SP_NZeros,SP_Atoms,EVec1,Work1);
  }

  else if ( strcasecmp(mode,"diag")==0 ){
    /* nothing is done. */
  }

  if (measure_time){
    printf("Cluster_DFT myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f time8=%7.3f\n",
	   myid0,time1,time2,time3,time4,time5,time6,time7,time8);fflush(stdout); 
  }

  /****************************************************
                          Free
  ****************************************************/

  free(EVec_Rcv);
  free(index_Rcv_j);
  free(index_Rcv_i);
  free(EVec_Snd);
  free(index_Snd_j);
  free(index_Snd_i);

  free(Num_Rcv_EV);
  free(Num_Snd_EV);
  free(ie1);
  free(is1);

  /* the dense owner's scratch between the SCF steps (opt-in,
     OPENMX_CLUSTER_GPU_RETAIN_SCRATCH) */
  if (strcasecmp(mode,"scf")==0) ClusterCol_TrimScratch();

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  openmx_mgpu_scope_leave(mgpu_scope_previous);
  return time0;
}





/* The local eigenvector panel already stores each basis row contiguously.
   Read it directly instead of duplicating both atoms' rows in every OpenMP
   thread.  size_t row offsets also cover panels larger than INT_MAX entries. */
static void ClusterCol_AccumulateDM_CPU(int size_H1, int nk, int *MP,
                                        const double *occ, const double *eig, const double *docc,
                                        const double *evec, double *DM1, double *EDM1, double *PDM1)
{
    size_t *atom_p;
    size_t pcount = 0;
    const int with_pdm = (docc!=NULL && PDM1!=NULL);

    /* Empty state partitions contribute only the caller's zeroed arrays. */
    if (nk<=0) return;

    atom_p = (size_t*)ClusterCol_MallocArray((size_t)atomnum+1,sizeof(size_t),"CPU DM atom offsets");
    for (int GA_AN=1; GA_AN<=atomnum; GA_AN++){
        const int tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]];
        atom_p[GA_AN] = pcount;
        for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            const int GB_AN = natn[GA_AN][LB_AN];
            pcount += (size_t)tnoA*(size_t)Spe_Total_CNO[WhatSpecies[GB_AN]];
        }
    }
    if (pcount!=(size_t)size_H1){
        free(atom_p);
        ClusterCol_AbortWithMessage("CPU DM entry count mismatch in Cluster_DFT_Col.c.");
    }

    /* One atom per task also uses several threads for small heavy-atom systems. */
#pragma omp parallel for schedule(dynamic,1)
    for (int GA_AN=1; GA_AN<=atomnum; GA_AN++){
        const int tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]];
        const size_t arow = (size_t)(MP[GA_AN]-1);
        size_t p = atom_p[GA_AN];

        for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            const int GB_AN = natn[GA_AN][LB_AN];
            const int tnoB = Spe_Total_CNO[WhatSpecies[GB_AN]];
            const size_t brow = (size_t)(MP[GB_AN]-1);

            for (int i=0; i<tnoA; i++){
                const double *e0 = evec+(arow+(size_t)i)*(size_t)nk;
                for (int j=0; j<tnoB; j++){
                    const double *e1 = evec+(brow+(size_t)j)*(size_t)nk;
                    double sum1 = 0.0, sum2 = 0.0;

                    for (int k=0; k<nk; k++){
                        const double dum = occ[k]*e0[k]*e1[k];
                        sum1 += dum;
                        sum2 += dum*eig[k];
                    }
                    DM1[p] = sum1;
                    EDM1[p] = sum2;
                    if (with_pdm){
                        sum1 = 0.0;
                        for (int k=0; k<nk; k++) sum1 += docc[k]*e0[k]*e1[k];
                        PDM1[p] = sum1;
                    }
                    p++;
                }
            }
        }
    }
    free(atom_p);
}

double Calc_DM_Cluster_collinear(
    int myid0,
    int numprocs0,
    int myid1,
    int numprocs1,
    int myworld1,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,
    double *****CDM,
    double *****EDM,
    double **ko,
    double *DM1,
    double *EDM1,
    double *PDM1,
    double *Work1,
    double **EVec1, 
    int *SP_NZeros,
    int *SP_Atoms )
{
  int spin,i,j,i0,j0,k,kmin,kmax,po,p,GA_AN,MA_AN,wanA,tnoA,Anum;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j1,ID;
  double max_x=60.0,dum;
  double FermiF,FermiF2,x,x2,diffF,sum1,sum2;
  double FermiEps = 1.0e-13;
  double stime,etime,time,lumos;
  MPI_Status stat;
  MPI_Request request;
  double *FF,*dFF;

  dtime(&stime);

  kmin = is2[myid1];
  kmax = ie2[myid1];

  /* allocation of arrays */

  FF = (double*)malloc(sizeof(double)*(n+1));
  dFF = (double*)malloc(sizeof(double)*(n+1));

  /* spin=myworld1 */

  spin = myworld1;

 calc_dm_collinear:

  /* initialize DM1 */

  for (i=0; i<size_H1; i++){
    DM1[i] = 0.0;
    EDM1[i] = 0.0;
    if (cal_partial_charge){
      PDM1[i] = 0.0;
    }
  }

  /* pre-calculation of Fermi Function */ 

  po = 0;

  for (k=is2[myid1]; k<=ie2[myid1]; k++){

    if (xanes_calc==1) 
      x = (ko[spin][k] - ChemP_XANES[spin])*Beta;
    else 
      x = (ko[spin][k] - ChemP)*Beta;

    if (x<=-max_x) x = -max_x;
    if (max_x<=x)  x = max_x;
    FermiF = FermiFunc(x,spin,k,&po,&x);

    FF[k] = FermiF;

    if (cal_partial_charge){

      if (xanes_calc==1) 
        x2 = (ko[spin][k] - (ChemP_XANES[spin]+ene_win_partial_charge))*Beta;
      else 
        x2 = (ko[spin][k] - (ChemP+ene_win_partial_charge))*Beta;

      if (x2<=-max_x) x2 = -max_x;
      if (max_x<=x2)  x2 = max_x;
      FermiF2 = FermiFunc(x2,spin,k,&po,&x);
      diffF = fabs(FermiF-FermiF2);
      dFF[k] = diffF;
    }
  }

  /* All spin worlds rendezvous; empty state partitions lend their GPUs.
     Keep the optimized root HIP/DGEMM path above, and recover distributed
     host panels without requiring another full device eigenvector copy. */
  const char *dm_env=getenv("OPENMX_CLUSTER_GPU_DM");
  int dm_requested=scf_eigen_lib_flag==GPUSOLVER &&
      (Cluster_DFT_Col_GpuSwitchNum()<=n || openmx_mgpu_host_required()) &&
      !(dm_env && atoi(dm_env)==0);
  int dm_collective=dm_requested,gpu_dm_done=0;
  MPI_Allreduce(MPI_IN_PLACE,&dm_collective,1,MPI_INT,MPI_MAX,mpi_comm_level1);
  if(dm_collective){
    int nk=dm_requested && kmin<=kmax?kmax-kmin+1:0;
    if(nk)ClusterCol_DMEntryCache_Ensure(MP,n);
    ClusterColDMEntryCache *cache=&ClusterCol_dm_entry_cache;
    double *outputs[10]={DM1,EDM1,cal_partial_charge?PDM1:NULL};
    openmx_gpu_dm_collective(mpi_comm_level1,OPENMX_GPU_DM_REAL,n,nk,nk,
        nk?cache->entry_count:0,EVec1[spin],1,cache->basis0,cache->basis1,NULL,0,NULL,NULL,
        nk?FF+kmin:NULL,nk?ko[spin]+kmin:NULL,nk && cal_partial_charge?dFF+kmin:NULL,outputs,1);
    gpu_dm_done=nk>0;
  }
  if(!gpu_dm_done && kmin<=kmax)
    ClusterCol_AccumulateDM_CPU(size_H1,kmax-kmin+1,MP,FF+kmin,ko[spin]+kmin,
                               cal_partial_charge ? dFF+kmin : NULL,EVec1[spin],
                               DM1,EDM1,cal_partial_charge ? PDM1 : NULL);

  /* MPI_Allreduce */

  MPI_Allreduce(DM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
  for (i=0; i<size_H1; i++) DM1[i] = Work1[i];

  MPI_Allreduce(EDM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
  for (i=0; i<size_H1; i++) EDM1[i] = Work1[i];

  if (cal_partial_charge){
    MPI_Allreduce(PDM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
    for (i=0; i<size_H1; i++) PDM1[i] = Work1[i];
  }

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

      if (myid0==ID){
         
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

            CDM[spin][MA_AN][LB_AN][i][j] = DM1[p];
            EDM[spin][MA_AN][LB_AN][i][j] = EDM1[p];
            if (cal_partial_charge){
              Partial_DM[spin][MA_AN][LB_AN][i][j] = PDM1[p];
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

  if (SpinP_switch==1 && numprocs0==1 && spin==0){
    spin++;  
    goto calc_dm_collinear;
  }
 
  else if (SpinP_switch==1 && numprocs0!=1){

    /* MPI communication of DM1 */

    if (Comm_World_StartID1[0]==myid0){
      MPI_Isend(DM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],10,mpi_comm_level1,&request);
    }
    if (Comm_World_StartID1[1]==myid0){
      MPI_Isend(DM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],20,mpi_comm_level1,&request);
    }
    if (Comm_World_StartID1[1]==myid0){
      MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],10,mpi_comm_level1,&stat);
      MPI_Wait(&request,&stat);
    }
    if (Comm_World_StartID1[0]==myid0){
      MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],20,mpi_comm_level1,&stat);
      MPI_Wait(&request,&stat);
    }

    MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);  
    for (i=0; i<size_H1; i++) DM1[i] = Work1[i];

    /* MPI communication of EDM1 */

    if (Comm_World_StartID1[0]==myid0){
      MPI_Isend(EDM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],10,mpi_comm_level1,&request);
    }
    if (Comm_World_StartID1[1]==myid0){
      MPI_Isend(EDM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],20,mpi_comm_level1,&request);
    }
    if (Comm_World_StartID1[1]==myid0){
      MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],10,mpi_comm_level1,&stat);
      MPI_Wait(&request,&stat);
    }
    if (Comm_World_StartID1[0]==myid0){
      MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],20,mpi_comm_level1,&stat);
      MPI_Wait(&request,&stat);
    }

    MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);  
    for (i=0; i<size_H1; i++) EDM1[i] = Work1[i];

    /* MPI communication of PDM1 */

    if (cal_partial_charge){
      if (Comm_World_StartID1[0]==myid0){
	MPI_Isend(PDM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],10,mpi_comm_level1,&request);
      }
      if (Comm_World_StartID1[1]==myid0){
	MPI_Isend(PDM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],20,mpi_comm_level1,&request);
      }
      if (Comm_World_StartID1[1]==myid0){
	MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],10,mpi_comm_level1,&stat);
	MPI_Wait(&request,&stat);
      }
      if (Comm_World_StartID1[0]==myid0){
	MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],20,mpi_comm_level1,&stat);
	MPI_Wait(&request,&stat);
      }

      MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);  
      for (i=0; i<size_H1; i++) PDM1[i] = Work1[i];
    }

    /* store DM1 to a proper place */

    if      (myworld1==0) spin = 1;
    else if (myworld1==1) spin = 0;

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

	if (myid0==ID){
         
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

              CDM[spin][MA_AN][LB_AN][i][j] = DM1[p];
              EDM[spin][MA_AN][LB_AN][i][j] = EDM1[p];
	      
              if (cal_partial_charge){
                Partial_DM[spin][MA_AN][LB_AN][i][j] = PDM1[p];
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
  } /* end of else if (SpinP_switch==1) */

  /* freeing of arrays */

  free(dFF);
  free(FF);

  dtime(&etime);
  return (etime-stime);

}




double Calc_ParDM_Cluster_collinear(
    int myid0,
    int numprocs0,
    int myid1,
    int numprocs1,
    int myworld1,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,
    double *****ParDM,
    double **ko,
    double *ParDM1,
    double *Work1,
    double **EVec1, 
    int *SP_NZeros,
    int *SP_Atoms )
{
  int spin,i,j,i0,j0,k,kmin,kmax,po,p,GA_AN,MA_AN,wanA,tnoA,Anum;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j1,ID;
  double dum,e0,e1,b0,b1,e,sum1,sum2;
  double stime,etime,time,lumos;
  MPI_Status stat;
  MPI_Request request;
  double *windowF;
  double **TmpEVec0,**TmpEVec1;

  dtime(&stime);

  kmin = is2[myid1];
  kmax = ie2[myid1];

  /* allocation of arrays */

  windowF = (double*)malloc(sizeof(double)*(n+1));

  TmpEVec0 = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    TmpEVec0[i] = (double*)malloc(sizeof(double)*(kmax-kmin+1));
  }

  TmpEVec1 = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    TmpEVec1[i] = (double*)malloc(sizeof(double)*(kmax-kmin+1));
  }

  /* spin=myworld1 */

  spin = myworld1;

 calc_dm_collinear:

  /* initialize ParDM1 */

  for (i=0; i<size_H1; i++){
    ParDM1[i] = 0.0;
  }

  /* pre-calculation of the window function */ 

  po = 0;

  for (k=is2[myid1]; k<=ie2[myid1]; k++){

    if (CWF_WindowF_switch==0){ // Poly

      double in0,in1,out0,out1; 

      out0 = CWF_disentangling_Erange[0];
      in0 = CWF_disentangling_Erange[1];
      in1 = CWF_disentangling_Erange[2];
      out1 = CWF_disentangling_Erange[3];

      e = ko[spin][k];
      windowF[k] = weight_CWF2(e,ChemP,in0,in1,out0,out1); 
    }

    else if (CWF_WindowF_switch==1){ // Fermi

      e = ko[spin][k];
      b0 = 1.0/CWF_disentangling_smearing_kBT0;
      b1 = 1.0/CWF_disentangling_smearing_kBT1;
      e0 = CWF_disentangling_Erange[0] + ChemP; 
      e1 = CWF_disentangling_Erange[1] + ChemP; 
      windowF[k] = (1.0/(exp(b0*(e0-e))+1.0))*(1.0/(exp(b1*(e-e1))+1.0)) + CWF_disentangling_smearing_bound;
    }

  }

  /* calculation of ParDM1 */ 

  p = 0;
  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];

    /* store EVec1 to a temporal array */

    for (i=0; i<tnoA; i++){
      i0 = (Anum + i - 1)*(ie2[myid1]-is2[myid1]+1) - is2[myid1];
      for (k=kmin; k<=kmax; k++){
        TmpEVec0[i][k-kmin] = EVec1[spin][i0+k];
      }        
    }

    /* loop for LB_AN */

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      Bnum = MP[GB_AN];

      /* store EVec1 to a temporal array */

      for (j=0; j<tnoB; j++){
        j0 = (Bnum + j - 1)*(ie2[myid1]-is2[myid1]+1) - is2[myid1];
	for (k=kmin; k<=kmax; k++){
	  TmpEVec1[j][k-kmin] = EVec1[spin][j0+k];
	}        
      }

      /* loops for i and j */

      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){

          sum1 = 0.0;
          sum2 = 0.0;

	  for (k=kmin; k<=kmax; k++){
	    dum = windowF[k]*TmpEVec0[i][k-kmin]*TmpEVec1[j][k-kmin];
	    sum1 += dum;
	    sum2 += dum*ko[spin][k];
	  }

	  ParDM1[p]  = sum1;

	  /* increment of p */
	  p++;  

	}
      }
    }
  } /* GA_AN */

  /* MPI_Allreduce */

  MPI_Allreduce(ParDM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
  for (i=0; i<size_H1; i++) ParDM1[i] = Work1[i];

  /* store ParDM1 to a proper place */

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

      if (myid0==ID){
         
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

            ParDM[spin][MA_AN][LB_AN][i][j] = ParDM1[p];

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

  if (SpinP_switch==1 && numprocs0==1 && spin==0){
    spin++;  
    goto calc_dm_collinear;
  }
 
  else if (SpinP_switch==1 && numprocs0!=1){

    /* MPI communication of ParDM1 */

    if (Comm_World_StartID1[0]==myid0){
      MPI_Isend(ParDM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],10,mpi_comm_level1,&request);
    }
    if (Comm_World_StartID1[1]==myid0){
      MPI_Isend(ParDM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],20,mpi_comm_level1,&request);
    }
    if (Comm_World_StartID1[1]==myid0){
      MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],10,mpi_comm_level1,&stat);
      MPI_Wait(&request,&stat);
    }
    if (Comm_World_StartID1[0]==myid0){
      MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],20,mpi_comm_level1,&stat);
      MPI_Wait(&request,&stat);
    }

    MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);  
    for (i=0; i<size_H1; i++) ParDM1[i] = Work1[i];

    /* store ParDM1 to a proper place */

    if      (myworld1==0) spin = 1;
    else if (myworld1==1) spin = 0;

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

	if (myid0==ID){
         
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

              ParDM[spin][MA_AN][LB_AN][i][j] = ParDM1[p];
	      
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
  } /* end of else if (SpinP_switch==1) */

  /* freeing of arrays */

  free(windowF);

  for (i=0; i<List_YOUSO[7]; i++){
    free(TmpEVec0[i]);
  }
  free(TmpEVec0);

  for (i=0; i<List_YOUSO[7]; i++){
    free(TmpEVec1[i]);
  }
  free(TmpEVec1);

  dtime(&etime);
  return (etime-stime);

}


  




void Save_DOS_Col( int n, int MaxN, int myid1, int *is2, int *ie2, 
                   int *MP, double *****nh, double ****OLP0, double **EVec1, 
                   double **ko, int *NPROCS_WD1, int *Comm_World_StartID1 )
{
  int spin,i,j,iemin,iemax,GA_AN,k,l;
  int i3,l1,l2,l3,RnB;
  int Anum,Bnum,tnoA,tnoB,wanA,wanB;
  int MA_AN,LB_AN,GB_AN,MaxL,num,m,p,po;
  int numprocs,myid,ID,tag,ID0;
  int i_vec[10];  
  double dum,tmp,av_num;
  char file_eig[YOUSO10],file_ev[YOUSO10],file_cohp[YOUSO10];
  FILE *fp_eig, *fp_ev,*fp_cohp; 
  MPI_Status stat;
  MPI_Request request;
  float *array0;    
  float *SD;
  double ***COHP,***COOP;
  int *is3,*ie3;
  int numprocs3,ID1;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */

  array0 = (float*)malloc(sizeof(float)*(n+2));
  is3 = (int*)malloc(sizeof(int)*numprocs);
  ie3 = (int*)malloc(sizeof(int)*numprocs);
  SD = (float*)malloc(sizeof(float)*(atomnum+1)*List_YOUSO[7]);
  
  if (COHP_calc_flag==1){
    COHP = (double***)malloc(sizeof(double**)*COHP_num_pairs);
    for (i=0; i<COHP_num_pairs; i++){
      COHP[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
        COHP[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }    
    }

    COOP = (double***)malloc(sizeof(double**)*COHP_num_pairs);
    for (i=0; i<COHP_num_pairs; i++){
      COOP[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
        COOP[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }    
    }
  }

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

    strcpy(file_cohp,".cohp");
    fnjoint(filepath,filename,file_cohp);
    if ( (fp_cohp=fopen(file_cohp,"w"))==NULL ) {
      printf("cannot open a file %s\n",file_cohp);
    }

  }

  /* find iemin */

  iemin = n;
  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=1; i<n; i++) {
      if ( ((ChemP+Dos_Erange[0])<ko[spin][i]) && (i-1)<iemin ) {
        iemin = i - 1;
        break;
      }
    }
  }
  if (iemin<1) iemin = 1;

  /* find iemax */

  iemax = 1;
  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=1; i<=n; i++) {
      if ( ((ChemP+Dos_Erange[1])<ko[spin][i])) {
        if (iemax<i) iemax = i;
        break;
      }
    }
  }
  if (iemax==1)   iemax = MaxN;
  if (MaxN<iemax) iemax = MaxN;

  /* COHP */  

  if (myid==Host_ID && COHP_calc_flag==1){
    fwrite(&Solver,sizeof(int),1,fp_cohp);
    fwrite(&SpinP_switch,sizeof(int),1,fp_cohp);
    i = 1;
    fwrite(&i,sizeof(int),1,fp_cohp);
    fwrite(&i,sizeof(int),1,fp_cohp);
    fwrite(&i,sizeof(int),1,fp_cohp);
    i = iemax - iemin + 1;
    fwrite(&i,sizeof(int),1,fp_cohp);
    fwrite(&COHP_num_pairs,sizeof(int),1,fp_cohp);
    fwrite(&ChemP,sizeof(double),1,fp_cohp);
  }

  /****************************************************
                   save *.Dos.vec
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){

    /* set is3 and ie3 */

    numprocs3 = NPROCS_WD1[spin];

    if ( numprocs3<=MaxN ){

      av_num = (double)MaxN/(double)numprocs3;
      for (ID=0; ID<numprocs3; ID++){
	is3[ID] = (int)(av_num*(double)ID) + 1; 
	ie3[ID] = (int)(av_num*(double)(ID+1)); 
      }

      is3[0] = 1;
      ie3[numprocs3-1] = MaxN; 
    }

    else{
      for (ID=0; ID<MaxN; ID++){
	is3[ID] = ID + 1; 
	ie3[ID] = ID + 1;
      }
      for (ID=MaxN; ID<numprocs3; ID++){
	is3[ID] =  1;
	ie3[ID] = -2;
      }
    }

    /* loop for p */

    for (p=iemin; p<=iemax; p++){

      /* store EVec1 */
      if (numprocs==1){
	for (k=0; k<n; k++){

	  m = k*(ie2[myid1]-is2[myid1]+1) + p-1;
	  array0[k] = (float)EVec1[spin][m];
	}
      }
      else{

	po = 0;
	for (ID=0; ID<numprocs3; ID++){
	  if (is3[ID]<=p && p <=ie3[ID]){
	    po = 1;
	    ID0 = ID;
	    break;
	  }
	}

	ID1 = Comm_World_StartID1[spin] + ID0;

	if (myid==ID1){
	  for (k=0; k<n; k++){

	    m = k*(ie2[myid1]-is2[myid1]+1) + p-is3[ID0];
	    array0[k] = (float)EVec1[spin][m];
	  }
	}

	/* MPI communications */
	MPI_Bcast(array0, n+1, MPI_FLOAT, ID1, mpi_comm_level1);
      }

      /* initialize SD */

      for (i=0; i<(atomnum+1)*List_YOUSO[7]; i++) SD[i] = 0.0;

      /* initialize COHP and COOP */

      if (COHP_calc_flag==1){
	for (i=0; i<COHP_num_pairs; i++){
	  for (j=0; j<List_YOUSO[7]; j++){
	    for (k=0; k<List_YOUSO[7]; k++){
	      COHP[i][j][k] = 0.0;
	      COOP[i][j][k] = 0.0;
	    }
	  }    
	}
      }

      i_vec[0]=i_vec[1]=i_vec[2]=0;
      if (myid==Host_ID) fwrite(i_vec,sizeof(int),3,fp_ev);

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
	      SD[Anum+i] += array0[Anum-1+i]*array0[Bnum-1+j]*(float)OLP0[MA_AN][LB_AN][i][j];
	    }

            if (COHP_calc_flag==1){

              RnB = ncn[GA_AN][LB_AN];
	      l1 = atv_ijk[RnB][1];
	      l2 = atv_ijk[RnB][2];
	      l3 = atv_ijk[RnB][3];

              for ( i3=0; i3<COHP_num_pairs; i3++ ){

                if (  GA_AN==COHP_AtomA[i3] 
                      && GB_AN==COHP_AtomB[i3] 
                      && l1==COHP_CellB1[i3]
                      && l2==COHP_CellB2[i3]
		      && l3==COHP_CellB3[i3] ){

		  for (j=0; j<tnoB; j++){
		    COHP[i3][i][j] = array0[Anum-1+i]*array0[Bnum-1+j]*nh[spin][MA_AN][LB_AN][i][j];
		    COOP[i3][i][j] = array0[Anum-1+i]*array0[Bnum-1+j]*OLP0[MA_AN][LB_AN][i][j];
		  }

		}
	      }
	    }

	  } /* LB_AN */
	} /* i */
      } /* MA_AN */

      /* MPI communication */

      for (i=1; i<=n; i++) array0[i] = SD[i];
      MPI_Reduce(array0, SD, n+1, MPI_FLOAT, MPI_SUM, Host_ID, mpi_comm_level1);

      if (COHP_calc_flag==1){
	for (i=0; i<COHP_num_pairs; i++){
	  for (j=0; j<List_YOUSO[7]; j++){
            MPI_Allreduce(MPI_IN_PLACE, &COHP[i][j][0], List_YOUSO[7], MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
            MPI_Allreduce(MPI_IN_PLACE, &COOP[i][j][0], List_YOUSO[7], MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
	  }    
	}
      }

      /* write *.Dos.vec */

      if (myid==Host_ID){
        fwrite(&SD[1],sizeof(float),n,fp_ev);
      }

      /* write *.cohp */

      if (myid==Host_ID && COHP_calc_flag==1){

        fwrite(&spin,sizeof(int),1,fp_cohp);
	i_vec[0] = 0; i_vec[1] = 0; i_vec[2] = 0;
	fwrite(i_vec,sizeof(int),3,fp_cohp);
        fwrite(&ko[spin][p],sizeof(double),1,fp_cohp);

	for (i=0; i<COHP_num_pairs; i++){

          GA_AN = COHP_AtomA[i];
          GB_AN = COHP_AtomB[i];
          wanA = WhatSpecies[GA_AN];
          wanB = WhatSpecies[GB_AN];
  	  tnoA = Spe_Total_CNO[wanA];
          tnoB = Spe_Total_CNO[wanB];
          l1 = COHP_CellB1[i];
          l2 = COHP_CellB2[i];
          l3 = COHP_CellB3[i];

	  fwrite(&i,sizeof(int),1,fp_cohp);
	  fwrite(&GA_AN,sizeof(int),1,fp_cohp);
	  fwrite(&GB_AN,sizeof(int),1,fp_cohp);
	  fwrite(&tnoA,sizeof(int),1,fp_cohp);
	  fwrite(&tnoB,sizeof(int),1,fp_cohp);
	  fwrite(&l1,sizeof(int),1,fp_cohp);
	  fwrite(&l2,sizeof(int),1,fp_cohp);
	  fwrite(&l3,sizeof(int),1,fp_cohp);

	  for (j=0; j<tnoA; j++){
            fwrite(&COHP[i][j][0],sizeof(double),tnoB,fp_cohp);
            fwrite(&COOP[i][j][0],sizeof(double),tnoB,fp_cohp);
	  }    
	} /* i */
      }

    } /* p */
  } /* spin */

  /****************************************************
                   save *.Dos.val
  ****************************************************/

  if (myid==Host_ID){

    fprintf(fp_eig,"mode        1\n");
    fprintf(fp_eig,"NonCol      0\n");
    fprintf(fp_eig,"N           %d\n",n);
    fprintf(fp_eig,"Nspin       %d\n",SpinP_switch);
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

    fprintf(fp_eig,"irange      %d %d\n",iemin,iemax);
    fprintf(fp_eig,"<Eigenvalues\n");

    for (spin=0; spin<=SpinP_switch; spin++) {
      fprintf(fp_eig,"%d %d %d ",0,0,0);
      for (i=iemin;i<=iemax;i++) {
	fprintf(fp_eig,"%lf ",ko[spin][i]);
	/* printf("%lf ",ko[spin][ie]); */
      }
      fprintf(fp_eig,"\n");
      /* printf("\n"); */
    }
    fprintf(fp_eig,"Eigenvalues>\n");

    printf("write eigenvalues\n");
    printf("write eigenvectors\n");

  } /* if (myid==Host_ID) */

  /* close file pointers */

  if (myid==Host_ID){
    if (fp_eig)  fclose(fp_eig);
    if (fp_ev)   fclose(fp_ev);
    if (fp_cohp) fclose(fp_cohp);
  }

  /* freeing of array */

  if (COHP_calc_flag==1){
    for (i=0; i<COHP_num_pairs; i++){
      for (j=0; j<List_YOUSO[7]; j++){
        free(COHP[i][j]);
      }    
      free(COHP[i]);
    }
    free(COHP);

    for (i=0; i<COHP_num_pairs; i++){
      for (j=0; j<List_YOUSO[7]; j++){
        free(COOP[i][j]);
      }    
      free(COOP[i]);
    }
    free(COOP);
  }

  free(SD);
  free(array0);
  free(is3);
  free(ie3);
}





void Save_LCAO_Col( int n, int MaxN, int myid1, int *is2, int *ie2, 
                    int *MP, double ****OLP0, double **EVec1, 
                    double **ko, int *NPROCS_WD1, int *Comm_World_StartID1 )
{
  int spin,i,j,iemin,iemax,GA_AN,k,l;
  int Anum,Bnum,tnoA,tnoB,wanA,wanB;
  int MA_AN,LB_AN,GB_AN,MaxL,num,m,p,po;
  int numprocs,myid,ID,tag,ID0;
  int i_vec[10],HOMO[2];  
  double dum,tmp,av_num,FermiF,x;
  double max_x=60.0;
  char operate[YOUSO10];
  FILE *fp1;
  MPI_Status stat;
  MPI_Request request;
  double *array0;    
  int *is3,*ie3;
  int numprocs3,ID1;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */

  array0 = (double*)malloc(sizeof(double)*(n+2));
  is3 = (int*)malloc(sizeof(int)*numprocs);
  ie3 = (int*)malloc(sizeof(int)*numprocs);

  /* open file pointer */

  if (myid==Host_ID){

    sprintf(operate,"%s%s.lcao",filepath,filename);
    fp1 = fopen(operate,"ab");

    if (fp1!=NULL){
      remove(operate); 
      fclose(fp1); 
      fp1 = fopen(operate, "ab");
    }
  }
  
  /* set iemin and iemax */

  iemin = 1;
  iemax = MaxN;

  /****************************************************
                  find Cluster_HOMO
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=1; i<=MaxN; i++){
      x = (ko[spin][i] - ChemP)*Beta;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = 1.0/(1.0 + exp(x));
      if (0.5<FermiF) HOMO[spin] = i;
    }
  }

  /****************************************************
                  save LCAO coefficients
  ****************************************************/

  /* Save parameters */      

  if (myid==Host_ID){
    fwrite(&n, sizeof(int), 1, fp1);
    fwrite(&MaxN, sizeof(int), 1, fp1);
    if (SpinP_switch==0) HOMO[1] = HOMO[0];
    fwrite(HOMO, sizeof(int), 2, fp1);
    fwrite(&SpinP_switch, sizeof(int), 1, fp1);
    fwrite(&Utot, sizeof(double), 1, fp1);
  }

  /* Save array0 */

  for (spin=0; spin<=SpinP_switch; spin++){

    /* set is3 and ie3 */

    numprocs3 = NPROCS_WD1[spin];

    if ( numprocs3<=MaxN ){

      av_num = (double)MaxN/(double)numprocs3;
      for (ID=0; ID<numprocs3; ID++){
	is3[ID] = (int)(av_num*(double)ID) + 1; 
	ie3[ID] = (int)(av_num*(double)(ID+1)); 
      }

      is3[0] = 1;
      ie3[numprocs3-1] = MaxN; 
    }

    else{
      for (ID=0; ID<MaxN; ID++){
	is3[ID] = ID + 1; 
	ie3[ID] = ID + 1;
      }
      for (ID=MaxN; ID<numprocs3; ID++){
	is3[ID] =  1;
	ie3[ID] = -2;
      }
    }

    /* loop for p */

    for (p=iemin; p<=iemax; p++){

      /* store EVec1 */
      if (numprocs==1){
	for (k=0; k<n; k++){

	  m = k*(ie2[myid1]-is2[myid1]+1) + p-1;
	  array0[k] = EVec1[spin][m];
	}
      }
      else{

	po = 0;
	for (ID=0; ID<numprocs3; ID++){
	  if (is3[ID]<=p && p<=ie3[ID]){
	    po = 1;
	    ID0 = ID;
	    break;
	  }
	}

	ID1 = Comm_World_StartID1[spin] + ID0;

	if (myid==ID1){
	  for (k=0; k<n; k++){

	    m = k*(ie2[myid1]-is2[myid1]+1) + p-is3[ID0];
	    array0[k] = EVec1[spin][m];
	  }
	}

	/* MPI communications */
	MPI_Bcast(array0, n, MPI_DOUBLE, ID1, mpi_comm_level1);

      } /* else */

      /* Save array0 */      

      if (myid==Host_ID){
        fwrite(array0, sizeof(double), n, fp1);
      }

    } /* p */
  } /* spin */

  /* close file pointer */

  if (myid==Host_ID){
    if (fp1) fclose(fp1);
  }

  /* freeing of array */

  free(array0);
  free(is3);
  free(ie3);
}




void Calc_XANES_Col( int n, int MaxN, int myid1, 
		     double **ko,
		     double *****nh, 
		     double ****CntOLP,
		     double *****CDM,
		     double *****EDM,
		     double Eele0[2], double Eele1[2],
		     int myworld1,
		     int *NPROCS_ID1,
		     int *Comm_World1,
		     int *NPROCS_WD1,
		     int *Comm_World_StartID1,
		     MPI_Comm *MPI_CommWD1,
		     int *MP,
		     int *is2,
		     int *ie2,
		     double *Ss,
		     double *Cs,
		     double *Hs,
		     double *CDM1,
		     double *EDM1,
		     double *PDM1,
		     int size_H1,
		     int *SP_NZeros,
		     int *SP_Atoms,
		     double **EVec1,
		     double *Work1)
{
  int spin,i,j,iemin,iemax,GA_AN,k,l,ii,jj;
  int TNO1,TNO2,ia,jb;
  int Anum,Bnum,tnoA,tnoB,wanA,wanB;
  int MA_AN,LB_AN,GB_AN,MaxL,num,m,p,p1,po;
  int numprocs,myid,ID,tag,ID0,u1,u2,o1;
  int Num_Excited_States,Num_Occupied_States;
  int tno0,tno1,Hwan,h_AN,Mc_AN,Gc_AN,BN;
  int direction,Cwan,Gh_AN,spin0,spin1,spinstate;
  int i_vec[10],UMOmax,OMOmin;
  double sum,dum,tmp,av_num,sumx,sumy,sumz;
  char operate[YOUSO10];
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp1;
  MPI_Status stat;
  MPI_Request request;
  double *array0;    
  int *is3,*ie3,*MP2;
  int numprocs3,ID1;

  int SpinP_switch1,n1,MaxN1;
  double Utot1,Utot2,Utot3;
  double **C1,**C2,**Z,**Ax,**Ay,**Az;
  int Cluster_HOMO1[2],Num_XANES_Out;
  int **ind2ind;
  double *****OLPmo,**XANES_Out;
  double XANES_Res[10];
  char file1[YOUSO10];

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* standard output */

  if (myid==Host_ID){
    printf("\n*******************************************************\n"); 
    printf("             Calculation of XANES spectrum              \n");
    printf("*******************************************************\n\n"); 
  }

  /* allocation of arrays */

  array0 = (double*)malloc(sizeof(double)*(n+2));
  is3 = (int*)malloc(sizeof(int)*numprocs);
  ie3 = (int*)malloc(sizeof(int)*numprocs);

  C2 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    C2[spin] = (double*)malloc(sizeof(double)*n*MaxN);
  }

  /* set iemin and imax */

  iemin = 1;
  iemax = MaxN;

  /****************************************************
   MPI communicate LCAO coefficients for excited state
  ****************************************************/

  /* MPI communicate LCAO coefficients */

  for (spin=0; spin<=SpinP_switch; spin++){

    /* set is3 and ie3 */

    numprocs3 = NPROCS_WD1[spin];

    if ( numprocs3<=MaxN ){

      av_num = (double)MaxN/(double)numprocs3;
      for (ID=0; ID<numprocs3; ID++){
	is3[ID] = (int)(av_num*(double)ID) + 1; 
	ie3[ID] = (int)(av_num*(double)(ID+1)); 
      }

      is3[0] = 1;
      ie3[numprocs3-1] = MaxN; 
    }

    else{
      for (ID=0; ID<MaxN; ID++){
	is3[ID] = ID + 1; 
	ie3[ID] = ID + 1;
      }
      for (ID=MaxN; ID<numprocs3; ID++){
	is3[ID] =  1;
	ie3[ID] = -2;
      }
    }

    /* loop for p */

    for (p=iemin; p<=iemax; p++){

      /* store EVec1 */
      if (numprocs==1){
	for (k=0; k<n; k++){

	  m = k*(ie2[myid1]-is2[myid1]+1) + p-1;
	  array0[k] = EVec1[spin][m];
	}
      }
      else{

	po = 0;
	for (ID=0; ID<numprocs3; ID++){
	  if (is3[ID]<=p && p<=ie3[ID]){
	    po = 1;
	    ID0 = ID;
	    break;
	  }
	}

	ID1 = Comm_World_StartID1[spin] + ID0;

	if (myid==ID1){
	  for (k=0; k<n; k++){

	    m = k*(ie2[myid1]-is2[myid1]+1) + p-is3[ID0];
	    array0[k] = EVec1[spin][m];
	  }
	}

	/* MPI communications */
	MPI_Bcast(array0, n, MPI_DOUBLE, ID1, mpi_comm_level1);
      }

      /* store array0 to C2 */      

      for (k=0; k<n; k++){
        C2[spin][(p-iemin)*n+k] = array0[k];
      }

    } /* p */
  } /* spin */

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

    C1 = (double**)malloc(sizeof(double*)*(SpinP_switch1+1));
    for (spin=0; spin<=SpinP_switch1; spin++){
      C1[spin] = (double*)malloc(sizeof(double)*n1*MaxN1);
    }

    /* read LCAO coefficients */

    for (spin=0; spin<=SpinP_switch1; spin++){
      fread(C1[spin], sizeof(double), n1*MaxN1, fp1);
    }

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

  /* find spinstate and set parameters */

  if      ( strcmp(Core_Hole_Orbital,"s")==0 ){
    if (Core_Hole_J==1) spinstate = 0;
    else                spinstate = 1;
  }
  else if ( strcmp(Core_Hole_Orbital,"p")==0 ){
    if (Core_Hole_J<=3) spinstate = 0;
    else                spinstate = 1;
  }
  else if ( strcmp(Core_Hole_Orbital,"d")==0 ){
    if (Core_Hole_J<=5) spinstate = 0;
    else                spinstate = 1;
  }
  else if ( strcmp(Core_Hole_Orbital,"f")==0 ){
    if (Core_Hole_J<=7) spinstate = 0;
    else                spinstate = 1;
  }

  spin0 = spinstate;
  if (spin0==0) spin1 = 1;
  else          spin1 = 0;

  /****************************************************
     determine the KS occupied and unoccupied states 
     to be included
  ****************************************************/

  static int xanes_single_flag,xanes_double0_flag,xanes_double1_flag;

  xanes_single_flag = 1;
  xanes_double0_flag = 0;
  xanes_double1_flag = 0;

  p = HOMO_XANES[spin0]; 
  UMOmax = p;

  for (k=p+1; k<=MaxN; k++){
    if ( (ko[spin0][k]-ko[spin0][p])<xanes_energy_range ) UMOmax = k;
  }  

  OMOmin = p - 1; 
  for (k=p-1; 1<=k; k--){
    if ( (ko[spin0][p]-ko[spin0][k])<xanes_energy_range ) OMOmin = k;
  }  

  /* allocation of XANES_Out */

  p = HOMO_XANES[spin0]; 
  p1 = HOMO_XANES[spin1]; 

  Num_Excited_States = UMOmax - p;
  Num_Occupied_States = p - OMOmin;

  if (xanes_single_flag==1)  Num_XANES_Out = Num_Excited_States;
  if (xanes_double0_flag==1) Num_XANES_Out+= Num_Excited_States*(Num_Excited_States-1)*Num_Occupied_States;
  if (xanes_double1_flag==1) Num_XANES_Out+= Num_Excited_States*Num_Excited_States*Num_Occupied_States;

  /* allocation of arrays */

  XANES_Out = (double**)malloc(sizeof(double*)*Num_XANES_Out);
  for (i=0; i<Num_XANES_Out; i++){
    XANES_Out[i] = (double*)malloc(sizeof(double)*10);
  }

  Z = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    Z[spin] = (double*)malloc(sizeof(double)*UMOmax*UMOmax);
  }

  Ax = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    Ax[spin] = (double*)malloc(sizeof(double)*UMOmax*UMOmax);
  }

  Ay = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    Ay[spin] = (double*)malloc(sizeof(double)*UMOmax*UMOmax);
  }

  Az = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    Az[spin] = (double*)malloc(sizeof(double)*UMOmax*UMOmax);
  }

  ind2ind = (int**)malloc(sizeof(int*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    ind2ind[spin] = (int*)malloc(sizeof(int)*UMOmax);
  }

  /**************************************************************
   calculations of the overlap, px, py, and pz matrices between 
   Kohn-Sham orbitals
  **************************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (ii=0; ii<UMOmax; ii++){
      for (jj=0; jj<UMOmax; jj++){

	sum = 0.0; 
	sumx = 0.0;
	sumy = 0.0;
	sumz = 0.0;
 
        for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

  	  Gc_AN = M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          TNO1 = Spe_Total_CNO[Cwan];
          Anum = MP2[Gc_AN];

          for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            TNO2 = Spe_Total_CNO[Hwan];
            Bnum = MP2[Gh_AN];

            for (i=0; i<TNO1; i++){

              ia = Anum + i;

              for (j=0; j<TNO2; j++){

                jb = Bnum + j;

  	   	tmp = C1[spin][n*ii+ia]*C2[spin][n*jj+jb];

                sum  += tmp*OLP[0][Mc_AN][h_AN][i][j];
                sumx += tmp*OLPmo[0][Mc_AN][h_AN][i][j];
                sumy += tmp*OLPmo[1][Mc_AN][h_AN][i][j];
                sumz += tmp*OLPmo[2][Mc_AN][h_AN][i][j];

	      }
	    }
	  }
	} /* Mc_AN */
  
        Z[spin][UMOmax*jj+ii]  = sum;
        Ax[spin][UMOmax*jj+ii] = sumx;
        Ay[spin][UMOmax*jj+ii] = sumy;
        Az[spin][UMOmax*jj+ii] = sumz;

      } /* jj */
    } /* ii */

    /* MPI communication of sum and store to Z */

    MPI_Allreduce(MPI_IN_PLACE,  &Z[spin][0], UMOmax*UMOmax, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, &Ax[spin][0], UMOmax*UMOmax, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, &Ay[spin][0], UMOmax*UMOmax, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, &Az[spin][0], UMOmax*UMOmax, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  } /* spin */  

  /****************************************************
           contribution of single excitations
  ****************************************************/

  if (xanes_single_flag==1){

    /* set parameters */ 
 
    for (i=0; i<UMOmax; i++){
      ind2ind[spin1][i] = i;
    }

    p = HOMO_XANES[spin0]; 
    Num_Excited_States = UMOmax - p;
    Utot2 = Utot;

    /* loop for k */

    for (k=myid; k<Num_Excited_States; k+=numprocs){

      if (myid==Host_ID){
        printf("Calc... myid=%2d %4d/%4d\n",myid,k+1,Num_Excited_States);fflush(stdout); 
      }
 
       /* set parameters */ 
 
      for (i=0; i<(p-1); i++){
         ind2ind[spin0][i] = i;    
      }
      ind2ind[spin0][p-1] = p - 1 + k;

      /*
      if (myid==Host_ID){
        printf("Utot1=%15.12f Utot2=%15.12f Utot3=%15.12f spin0=%2d spin1=%2d p+k=%2d ko[spin0][p+k]=%15.12f\n",
                Utot1,Utot2,Utot3,spin0,spin1,p+k,ko[spin0][p+k]);
      }
      */

      Utot3 = Utot2 + ko[spin0][p+k] - ko[spin0][p];

      Calc_Oscillator_Strength( n, UMOmax, HOMO_XANES, MP2, ind2ind, C1, C2, Z, Ax, Ay, Az, Utot1, Utot3, XANES_Res );

      XANES_Out[k][0] = XANES_Res[0];
      XANES_Out[k][1] = XANES_Res[1];
      XANES_Out[k][2] = XANES_Res[2];
      XANES_Out[k][3] = XANES_Res[3];
    }
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

      fprintf(fp1,"# core.hole.state: atom=%3d  orbial=%2s index=%2d\n",Core_Hole_Atom,Core_Hole_Orbital,Core_Hole_J);
      fprintf(fp1,"# #1:serial number #2:Excitation energy (eV) #4:Px #5:Py #6:Pz\n");
      fprintf(fp1,"#\n");

      for (i=0; i<Num_XANES_Out; i++){
        fprintf(fp1,"%4d %15.12f %15.12f %15.12f %15.12f  # excited state=%4d (spin=%d)\n",
		i+1,XANES_Out[i][0],XANES_Out[i][1],XANES_Out[i][2],XANES_Out[i][3],
                HOMO_XANES[spin0]+i,spin0);
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

  for (spin=0; spin<=SpinP_switch; spin++){
    free(Z[spin]);
  }
  free(Z);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(Ax[spin]);
  }
  free(Ax);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(Ay[spin]);
  }
  free(Ay);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(Az[spin]);
  }
  free(Az);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(ind2ind[spin]);
  }
  free(ind2ind);

  free(MP2);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(C2[spin]);
  }
  free(C2);

  for (spin=0; spin<=SpinP_switch1; spin++){
    free(C1[spin]);
  }
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







double Calc_Oscillator_Strength( int n, int UMOmax, int Nocc[2], int *MP,
                                 int **ind2ind, 
                                 double **Z0, double **A, double **Z,
                                 double **Ax, double **Ay, double **Az,
                                 double Utot1, double Utot2,
                                 double XANES_Res[10] )
{
  int ii,jj,Gc_AN,spin,Cwan,Hwan,Mc_AN;
  int i,j,k,ia,ja,ib,jb,Anum,Bnum,h_AN,Gh_AN,Rn,TNO1,TNO2;
  double tmp,det[2],alldet,sum,allsum,allsumx,allsumy,allsumz;
  double os,osx,osy,osz,p2,px2,py2,pz2,px,py,pz;
  double fsum,fsumx,fsumy,fsumz,sumx,sumy,sumz;
  double stime,etime;
  int numprocs,myid;

  dtime(&stime);
  
  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* set Z0 which is an overlap matrix of two Slater determinants */

  for (spin=0; spin<=SpinP_switch; spin++){
    for (jj=0; jj<Nocc[spin]; jj++){

      j = ind2ind[spin][jj];        
      for (ii=0; ii<Nocc[spin]; ii++){
        Z0[spin][Nocc[spin]*jj+ii] = Z[spin][UMOmax*j+ii];
      }
    }
  }  

  /* calculation of the inverse Z0 */

  for (spin=0; spin<=SpinP_switch; spin++){
    det[spin] = Lapack_LU_Dinverse(Nocc[spin], Z0[spin]);
  }

  /* calculation of Z0^{-1} * det */

  alldet = det[0]*det[1];

  for (spin=0; spin<=SpinP_switch; spin++){

    for (i=0; i<Nocc[spin]; i++){
      for (j=0; j<Nocc[spin]; j++){

        tmp = Z0[spin][Nocc[spin]*j+i]; 
        Z0[spin][Nocc[spin]*j+i] = tmp*alldet;
      }  
    } 
  }

  /* calculation of Ax*(Z^{-1*det)}, Ay*(Z^{-1}*det), and Az*(Z^{-1}*det) */

  for (k=0; k<3; k++){

    /* set Ax, Ay, or Az */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (ii=0; ii<Nocc[spin]; ii++){

        if (k==0){
          for (jj=0; jj<Nocc[spin]; jj++){
   	    j = ind2ind[spin][jj];
	    A[spin][Nocc[spin]*ii+jj] = Ax[spin][UMOmax*j+ii];
	  }
	}

        else if (k==1){
          for (jj=0; jj<Nocc[spin]; jj++){
   	    j = ind2ind[spin][jj];
	    A[spin][Nocc[spin]*ii+jj] = Ay[spin][UMOmax*j+ii];
	  }
	}

        else if (k==2){
          for (jj=0; jj<Nocc[spin]; jj++){
   	    j = ind2ind[spin][jj];
	    A[spin][Nocc[spin]*ii+jj] = Az[spin][UMOmax*j+ii];
	  }
	}

      } /* ii */
    } /* spin */

    /* A*(Z0^{-1*det)} */

    fsum = 0.0;
    for (spin=0; spin<=SpinP_switch; spin++){
      for (i=0; i<Nocc[spin]; i++){
	for (j=0; j<Nocc[spin]; j++){
	  fsum += A[spin][Nocc[spin]*i+j]*Z0[spin][Nocc[spin]*i+j];
	}
      }
    }

    if      (k==0){ px = fsum; px2 = fsum*fsum; }
    else if (k==1){ py = fsum; py2 = fsum*fsum; }
    else if (k==2){ pz = fsum; pz2 = fsum*fsum; }
  
  } /* k */

  p2 = px2 + py2 + pz2;

  XANES_Res[0] = fabs(Utot1-Utot2)*eV2Hartree;
  XANES_Res[1] = px;
  XANES_Res[2] = py;
  XANES_Res[3] = pz;

  /*
  p2 = px2 + py2 + pz2;
  os = p2/fabs(Utot1-Utot2);
  osx = px2/fabs(Utot1-Utot2);
  osy = py2/fabs(Utot1-Utot2);
  osz = pz2/fabs(Utot1-Utot2);

  XANES_Res[0] = fabs(Utot1-Utot2)*27.2113845;
  XANES_Res[1] = os;
  XANES_Res[2] = osx;
  XANES_Res[3] = osy;
  XANES_Res[4] = osz;
  */

  /*
  if (myid==Host_ID){
    printf(" TE (eV): %10.5f  os: %8.4f  x y z: %8.4f %8.4f %8.4f\n",
             fabs(Utot1-Utot2)*27.2113845,os,osx,osy,osz);
  }
  */

  dtime(&etime);
  return (etime-stime);
}




double Lapack_LU_Dinverse(int n, double *A)
{
  static char *thisprogram="Lapack_LU_inverse";
  int *ipiv;
  double *work,tmp,det;
  int lwork;
  int info,i,j;

  /* L*U factorization */

  ipiv = (int*) malloc(sizeof(int)*n);

  F77_NAME(dgetrf,DGETRF)(&n,&n,A,&n,ipiv,&info);

  if ( info !=0 ) {
    printf("dgetrf failed, info=%i, %s\n",info,thisprogram);
  }

  /* calculation of determinant */

  det = 1.0; 
  for (i=0; i<n; i++){
    tmp = det;
    det = tmp*A[n*(i)+(i)];
  }

  for (i=0; i<n; i++){
    if (ipiv[i] != i+1) { det = -det; }
  }

  /* 
     printf("det %15.12f\n",det);   

     for (i=0; i<n; i++){
     printf("i=%2d ipiv=%2d\n",i,ipiv[i]);
     }
  */

  /* inverse L*U factorization */

  lwork = 4*n;
  work = (double*)malloc(sizeof(double)*lwork);

  F77_NAME(dgetri,DGETRI)(&n, A, &n, ipiv, work, &lwork, &info);

  if ( info !=0 ) {
    printf("dgetrf failed, info=%i, %s\n",info,thisprogram);
  }

  free(work); free(ipiv);

  return det;
}




double Calc_KS_orbitals_on_grid_single(
    int myid0,
    int numprocs0,
    int myid1,
    int numprocs1,
    int myworld1,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,
    double *****CDM,
    double *****EDM,
    double **ko,
    double *DM1,
    double *EDM1,
    double *PDM1,
    double *Work1,
    double **EVec1, 
    int *SP_NZeros,
    int *SP_Atoms )
{
  int kmin,kmax,i0,wanA,Anum;
  int Mc_AN,Gc_AN,Nc,p,GN,tnoA;
  int Nmax,i,j,orb,spin,orb1,orb2;
  double **phi_on_grid,**b_on_grid;
  double *work1_on_grid,*work2_on_grid;
  double tmp,sum,*B,*ev,*asum,*proj_chi;
  double stime,etime,stime1,etime1;

  dtime(&stime);

  kmin = is2[myid1];
  kmax = ie2[myid1];

  /* allocation of arrays */

  spin = 0;
  
  for (i=0; i<n; i++){ Work1[i] = 0.0; }

  Nmax = Cluster_HOMO[0];

  work1_on_grid = (double*)malloc(sizeof(double)*TNumGrid);
  work2_on_grid = (double*)malloc(sizeof(double)*TNumGrid);
  
  phi_on_grid = (double**)malloc(sizeof(double*)*Nmax);
  for (i=0; i<Nmax; i++){
    phi_on_grid[i] = (double*)malloc(sizeof(double)*TNumGrid);
    for (j=0; j<TNumGrid; j++){
      phi_on_grid[i][j] = 0.0;
    }
  }

  b_on_grid = (double**)malloc(sizeof(double*)*Nmax);
  for (i=0; i<Nmax; i++){
    b_on_grid[i] = (double*)malloc(sizeof(double)*TNumGrid);
    for (j=0; j<TNumGrid; j++){
      b_on_grid[i][j] = 0.0;
    }
  }

  B = (double*)malloc(sizeof(double)*Nmax*Nmax);
  ev = (double*)malloc(sizeof(double)*Nmax);

  asum = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  proj_chi = (double*)malloc(sizeof(double)*n);

  /* loop fof orb for the calculation of phi on grid */
  
  for (orb=0; orb<Nmax; orb++){

    /* set LCAO coefficients of phi_k */
      
    for (i=0; i<n; i++){
      // EVec1[][]: the second index starts from 0. 
      i0 = i*(kmax-kmin+1) + orb + 1 - kmin;  
      Work1[i] = EVec1[spin][i0];
    }

    /* calculate phi on grid */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];    
      wanA = WhatSpecies[Gc_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[Gc_AN] - 1;

      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

	GN = GridListAtom[Mc_AN][Nc];
	  
	for (i=0; i<tnoA; i++){
	  phi_on_grid[orb][GN] += Work1[Anum+i]*Orbs_Grid[Mc_AN][Nc][i];
	}
      }
    }
  } 

  /* calculation of b on grid */

  if (1){
  
    for (orb1=0; orb1<Nmax; orb1++){
      for (orb2=0; orb2<Nmax; orb2++){

	/* calculate the potential of phi_1 * phi_2 */
      
	for (GN=0; GN<TNumGrid; GN++){
	  work1_on_grid[GN] = phi_on_grid[orb1][GN]*phi_on_grid[orb2][GN];
	}

        if (measure_time) dtime(&stime1);
	
	Exchange_Poisson( work1_on_grid, work2_on_grid );

	if (measure_time){
	  dtime(&etime1);
	  printf("orb1=%2d orb2=%2d Exchange_Poisson time=%15.10f\n",orb1,orb2,etime1-stime1); 
	}
	
	for (GN=0; GN<TNumGrid; GN++){
	  b_on_grid[orb1][GN] -= work2_on_grid[GN]*phi_on_grid[orb2][GN];
	}
      }
    }
  }

  else if (0) {

    for (orb1=0; orb1<Nmax; orb1++){

      for (GN=0; GN<TNumGrid; GN++){ work1_on_grid[GN] = 0.0; }
      for (orb2=0; orb2<Nmax; orb2++){
	for (GN=0; GN<TNumGrid; GN++){
	  work1_on_grid[GN] += phi_on_grid[orb2][GN];                       
	}
      }

      for (GN=0; GN<TNumGrid; GN++){ work1_on_grid[GN] *= phi_on_grid[orb1][GN]; }
      
      Exchange_Poisson( work1_on_grid, work2_on_grid );

      for (GN=0; GN<TNumGrid; GN++){ work1_on_grid[GN] = 0.0; }
      for (orb2=0; orb2<Nmax; orb2++){
	for (GN=0; GN<TNumGrid; GN++){
	  work1_on_grid[GN] += phi_on_grid[orb2][GN];                       
	}
      }
      
      for (GN=0; GN<TNumGrid; GN++){
	b_on_grid[orb1][GN] = -work2_on_grid[GN]*work1_on_grid[GN];
      }
      
    }
  }

  else if (0){

    for (orb1=0; orb1<Nmax; orb1++){
    
      for (GN=0; GN<TNumGrid; GN++){ work1_on_grid[GN] = 0.0; }
      for (orb2=0; orb2<Nmax; orb2++){
	for (GN=0; GN<TNumGrid; GN++){
	  work1_on_grid[GN] += 2.0*phi_on_grid[orb2][GN]*phi_on_grid[orb2][GN];                       
	}
      }

      sum = 0.0;
      for (GN=0; GN<TNumGrid; GN++){
	sum += fabs(work1_on_grid[GN]);
      }
      printf("ZZZ1 sum=%15.12f\n",sum);
      
      Exchange_Poisson( work1_on_grid, work2_on_grid );

      sum = 0.0;
      for (GN=0; GN<TNumGrid; GN++){
	sum += fabs(work2_on_grid[GN]);
      }
      printf("ZZZ2 sum=%15.12f\n",sum);
      
      for (GN=0; GN<TNumGrid; GN++){
	b_on_grid[orb1][GN] -= work2_on_grid[GN]*phi_on_grid[orb1][GN];
      }
    }
  }
  
  /* calculation of the matrix B */

  for (orb1=0; orb1<Nmax; orb1++){
    for (orb2=0; orb2<Nmax; orb2++){

      sum = 0.0;
      for (GN=0; GN<TNumGrid; GN++){
        sum += b_on_grid[orb1][GN]*phi_on_grid[orb2][GN];  
      }

      B[orb1*Nmax+orb2] = sum*GridVol;

      //printf("%7.4f ",B[orb1*Nmax+orb2]); 
    }
    //printf("\n");
  }

  /* diagonalization of the matrix B */

  Eigen_lapack3( B, ev, Nmax, Nmax );

  for (orb=0; orb<Nmax; orb++){
    printf("VVV1 orb=%2d ev=%15.12f\n",orb,ev[orb]);
  }
  
  for (orb=0; orb<Nmax; orb++){ ev[orb] = 1.0/ev[orb]; }

  /* calculate |a_p> by \sum_{q} |b_q> (U)_{qp} */

  for (orb1=0; orb1<Nmax; orb1++){

    for (GN=0; GN<TNumGrid; GN++){ phi_on_grid[orb1][GN] = 0.0; }

    for (orb2=0; orb2<Nmax; orb2++){

      tmp = B[orb1*Nmax+orb2];
      
      for (GN=0; GN<TNumGrid; GN++){
        phi_on_grid[orb1][GN] += b_on_grid[orb2][GN]*tmp;
      }
    }
  }
  
  /* calculate <chi_i|Vx|chi_j> */

  /*
  for (i=0; i<(n*n); i++){ Hex[i] = 0.0; }
  */  

  for (orb=0; orb<Nmax; orb++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      
      Gc_AN = M2G[Mc_AN];    
      wanA = WhatSpecies[Gc_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[Gc_AN] - 1;

      for (i=0; i<tnoA; i++){ asum[i] = 0.0; }
      
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	GN = GridListAtom[Mc_AN][Nc];

	tmp = phi_on_grid[orb][GN];
	for (i=0; i<tnoA; i++){
	  asum[i] += tmp*Orbs_Grid[Mc_AN][Nc][i];
	}
      }

      for (i=0; i<tnoA; i++){
	proj_chi[Anum+i] = asum[i]*GridVol;
      }
    } // Mc_AN

    /*
    for (i=0; i<n; i++){
      for (j=0; j<n; j++){
        Hex[i*n+j] += proj_chi[i]*proj_chi[j]*ev[orb];
      }
    }
    */

  } // orb

  /*
  printf("WWW2\n");
  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      printf("%7.4f ", Hex[i*n+j]);
    }
    printf("\n");
  }

  MPI_Finalize();
  exit(0);
  */

  /*  
  for (i=0; i<(n*n); i++){
    Hex[i] = 0.90*Hex0[i] + 0.10*Hex[i];
    Hex0[i] = Hex[i];
  }
  */

  /*
  printf("WWW3\n");
  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      printf("%7.4f ", Hex[i*n+j]);
    }
    printf("\n");
  }
  */
  
  /*
  for (i=0; i<n; i++){
    printf("GGG i=%2d Hex=%10.5f\n",i,Hex[i*n+i]);
  }
  */

  /*
  MPI_Finalize();
  exit(0);
  */
  
  /* freeing of arrays */

  free(work1_on_grid);
  free(work2_on_grid);
  
  for (i=0; i<Nmax; i++){
    free(phi_on_grid[i]);
  }
  free(phi_on_grid);

  for (i=0; i<Nmax; i++){
    free(b_on_grid[i]);
  }
  free(b_on_grid);

  free(B);
  free(ev);

  free(asum);
  free(proj_chi);

  //free(Hex);
  
  dtime(&etime);
  return (etime-stime);

}









double Exchange_Poisson( double *work1_on_grid, double *work2_on_grid )
{
  int i,j,k,k1,k2,k3,GN,GN_in,GN_out;
  double sk1,sk2,sk3,fac_invG2,Gx,Gy,Gz,tmp0;
  double stime,etime;
  double *ReTmp1,*ImTmp1;
  double *ReTmp2,*ImTmp2;
  fftw_complex *in, *out;
  fftw_plan p;

  dtime(&stime);

  ReTmp1 = (double*)malloc(sizeof(double)*TNumGrid);
  ImTmp1 = (double*)malloc(sizeof(double)*TNumGrid);

  ReTmp2 = (double*)malloc(sizeof(double)*TNumGrid);
  ImTmp2 = (double*)malloc(sizeof(double)*TNumGrid);
  
  in  = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]); 
  out = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]); 

  /***************************
          FFT of charge
  ***************************/
  
  /* FFT along the c-axis */

  p = fftw_plan_dft_1d(Ngrid3,in,out,-1,FFTW_ESTIMATE);
  
  for (i=0; i<Ngrid1; i++){
    for (j=0; j<Ngrid2; j++){

      GN_in  = i*Ngrid2*Ngrid3 + j*Ngrid3;

      for (k=0; k<Ngrid3; k++){
        in[k][0] = work1_on_grid[GN_in+k];
        in[k][1] = 0.0;
      }

      fftw_execute(p);
      
      for (k=0; k<Ngrid3; k++){
	GN_out = k*Ngrid1*Ngrid2 + i*Ngrid2 + j;
        ReTmp1[GN_out] = out[k][0];
        ImTmp1[GN_out] = out[k][1];
      }
    }
  }

  fftw_destroy_plan(p);  

  /* FFT along the b-axis */

  p = fftw_plan_dft_1d(Ngrid2,in,out,-1,FFTW_ESTIMATE);

  for (k=0; k<Ngrid3; k++){
    for (i=0; i<Ngrid1; i++){

      GN_in = k*Ngrid1*Ngrid2 + i*Ngrid2;

      for (j=0; j<Ngrid2; j++){
        in[j][0] = ReTmp1[GN_in+j];
        in[j][1] = ImTmp1[GN_in+j];
      }

      fftw_execute(p);

      for (j=0; j<Ngrid2; j++){
	GN_out = j*Ngrid3*Ngrid1 + k*Ngrid1 + i;
        ReTmp2[GN_out] = out[j][0];
        ImTmp2[GN_out] = out[j][1];
      }
    }
  }

  fftw_destroy_plan(p);  

  /* FFT along the a-axis */

  p = fftw_plan_dft_1d(Ngrid1,in,out,-1,FFTW_ESTIMATE);
  
  for (j=0; j<Ngrid2; j++){
    for (k=0; k<Ngrid3; k++){

      GN_in = j*Ngrid3*Ngrid1 + k*Ngrid1;

      for (i=0; i<Ngrid1; i++){
        in[i][0] = ReTmp2[GN_in+i];
        in[i][1] = ImTmp2[GN_in+i];
      }

      fftw_execute(p);

      for (i=0; i<Ngrid1; i++){
	GN_out = i*Ngrid2*Ngrid3 + j*Ngrid3 + k;
        ReTmp1[GN_out] = out[i][0];
        ImTmp1[GN_out] = out[i][1];
      }
    }
  }

  fftw_destroy_plan(p);

  /*******************************
    calculate 4pi*rho(G)/|G|^2
  *******************************/

  tmp0 = 4.0*PI/(double)(Ngrid1*Ngrid2*Ngrid3);
  
  for (k1=0; k1<Ngrid1; k1++){
    for (k2=0; k2<Ngrid2; k2++){
      for (k3=0; k3<Ngrid3; k3++){

	GN_in = k1*Ngrid2*Ngrid3 + k2*Ngrid3 + k3;

	if (k1<Ngrid1/2) sk1 = (double)k1;
	else             sk1 = (double)(k1 - Ngrid1);

	if (k2<Ngrid2/2) sk2 = (double)k2;
	else             sk2 = (double)(k2 - Ngrid2);

	if (k3<Ngrid3/2) sk3 = (double)k3;
	else             sk3 = (double)(k3 - Ngrid3);

	Gx = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
	Gy = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
	Gz = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];
	
        fac_invG2 = tmp0/(Gx*Gx + Gy*Gy + Gz*Gz);

	if (k1==0 && k2==0 && k3==0){
	  ReTmp1[GN_in] = 0.0;
	  ImTmp1[GN_in] = 0.0;
	}
	else{
	  ReTmp1[GN_in] *= fac_invG2;
	  ImTmp1[GN_in] *= fac_invG2;
	}

      }
    }
  }

  /***********************************
    inverse FFT of 4pi*rho(G)/|G|^2
  ***********************************/

  /* FFT along the c-axis */

  p = fftw_plan_dft_1d(Ngrid3,in,out,1,FFTW_ESTIMATE);
  
  for (i=0; i<Ngrid1; i++){
    for (j=0; j<Ngrid2; j++){

      GN_in  = i*Ngrid2*Ngrid3 + j*Ngrid3;

      for (k=0; k<Ngrid3; k++){
        in[k][0] = ReTmp1[GN_in+k];
        in[k][1] = ImTmp1[GN_in+k];
      }

      fftw_execute(p);
      
      for (k=0; k<Ngrid3; k++){
	GN_out = k*Ngrid1*Ngrid2 + i*Ngrid2 + j;
        ReTmp2[GN_out] = out[k][0];
        ImTmp2[GN_out] = out[k][1];
      }
    }
  }

  fftw_destroy_plan(p);  

  /* FFT along the b-axis */

  p = fftw_plan_dft_1d(Ngrid2,in,out,1,FFTW_ESTIMATE);

  for (k=0; k<Ngrid3; k++){
    for (i=0; i<Ngrid1; i++){

      GN_in = k*Ngrid1*Ngrid2 + i*Ngrid2;

      for (j=0; j<Ngrid2; j++){
        in[j][0] = ReTmp2[GN_in+j];
        in[j][1] = ImTmp2[GN_in+j];
      }

      fftw_execute(p);

      for (j=0; j<Ngrid2; j++){
	GN_out = j*Ngrid3*Ngrid1 + k*Ngrid1 + i;
        ReTmp1[GN_out] = out[j][0];
        ImTmp1[GN_out] = out[j][1];
      }
    }
  }

  fftw_destroy_plan(p);  

  /* FFT along the a-axis */

  p = fftw_plan_dft_1d(Ngrid1,in,out,1,FFTW_ESTIMATE);
  
  for (j=0; j<Ngrid2; j++){
    for (k=0; k<Ngrid3; k++){

      GN_in = j*Ngrid3*Ngrid1 + k*Ngrid1;

      for (i=0; i<Ngrid1; i++){
        in[i][0] = ReTmp1[GN_in+i];
        in[i][1] = ImTmp1[GN_in+i];
      }

      fftw_execute(p);

      for (i=0; i<Ngrid1; i++){
	GN_out = i*Ngrid2*Ngrid3 + j*Ngrid3 + k;
        ReTmp2[GN_out] = out[i][0];
        ImTmp2[GN_out] = out[i][1];
      }
    }
  }

  fftw_destroy_plan(p);

  /***********************************
    inverse FFT of 4pi*rho(G)/|G|^2
  ***********************************/

  for (GN=0; GN<TNumGrid; GN++){
    work2_on_grid[GN] = ReTmp2[GN];
  }

  /***********************************
           freeing of arrays
  ***********************************/

  free(ReTmp1);
  free(ImTmp1);

  free(ReTmp2);
  free(ImTmp2);

  fftw_free(in);
  fftw_free(out);
  
  dtime(&etime);
  return (etime-stime);

}




double Calc_HF_Vx_single(
    int myid0,
    int numprocs0,
    int myid1,
    int numprocs1,
    int myworld1,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,
    double *****CDM,
    double *****EDM,
    double **ko,
    double *DM1,
    double *EDM1,
    double *PDM1,
    double *Work1,
    double **EVec1, 
    int *SP_NZeros,
    int *SP_Atoms )
{
  int kmin,kmax,i0,wanA,Anum;
  int Mc_AN,Gc_AN,Nc,p,GN,tnoA;
  int Nmax,i,j,orb,spin,orb1,orb2;
  double **phi_on_grid,**b_on_grid;
  double *work1_on_grid,*work2_on_grid;
  double tmp,sum,*B,*ev,*asum,*proj_chi;
  double stime,etime,stime1,etime1;

  dtime(&stime);

  kmin = is2[myid1];
  kmax = ie2[myid1];

  /* allocation of arrays */

  spin = 0;
  
  for (i=0; i<n; i++){ Work1[i] = 0.0; }

  Nmax = Cluster_HOMO[0];

  work1_on_grid = (double*)malloc(sizeof(double)*TNumGrid);
  work2_on_grid = (double*)malloc(sizeof(double)*TNumGrid);
  
  phi_on_grid = (double**)malloc(sizeof(double*)*Nmax);
  for (i=0; i<Nmax; i++){
    phi_on_grid[i] = (double*)malloc(sizeof(double)*TNumGrid);
    for (j=0; j<TNumGrid; j++){
      phi_on_grid[i][j] = 0.0;
    }
  }

  b_on_grid = (double**)malloc(sizeof(double*)*Nmax);
  for (i=0; i<Nmax; i++){
    b_on_grid[i] = (double*)malloc(sizeof(double)*TNumGrid);
    for (j=0; j<TNumGrid; j++){
      b_on_grid[i][j] = 0.0;
    }
  }

  B = (double*)malloc(sizeof(double)*Nmax*Nmax);
  ev = (double*)malloc(sizeof(double)*Nmax);

  asum = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  proj_chi = (double*)malloc(sizeof(double)*n);

  /* loop fof orb for the calculation of phi on grid */
  
  for (orb=0; orb<Nmax; orb++){

    /* set LCAO coefficients of phi_k */
      
    for (i=0; i<n; i++){
      // EVec1[][]: the second index starts from 0. 
      i0 = i*(kmax-kmin+1) + orb + 1 - kmin;  
      Work1[i] = EVec1[spin][i0];
    }

    /* calculate phi on grid */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];    
      wanA = WhatSpecies[Gc_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[Gc_AN] - 1;

      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

	GN = GridListAtom[Mc_AN][Nc];
	  
	for (i=0; i<tnoA; i++){
	  phi_on_grid[orb][GN] += Work1[Anum+i]*Orbs_Grid[Mc_AN][Nc][i];
	}
      }
    }
  } 

  /* calculation of b on grid */

  for (orb1=0; orb1<Nmax; orb1++){
    for (orb2=0; orb2<Nmax; orb2++){

      /* calculate the potential of phi_1 * phi_2 */
      
      for (GN=0; GN<TNumGrid; GN++){
	work1_on_grid[GN] = phi_on_grid[orb1][GN]*phi_on_grid[orb2][GN];
      }

      if (measure_time) dtime(&stime1);
	
      Exchange_Poisson( work1_on_grid, work2_on_grid );

      if (measure_time){
	dtime(&etime1);
	printf("orb1=%2d orb2=%2d Exchange_Poisson time=%15.10f\n",orb1,orb2,etime1-stime1); 
      }
	
      for (GN=0; GN<TNumGrid; GN++){
	b_on_grid[orb1][GN] -= work2_on_grid[GN]*phi_on_grid[orb2][GN];
      }
    }
  }
  
  /* calculation of the matrix B */

  for (orb1=0; orb1<Nmax; orb1++){
    for (orb2=0; orb2<Nmax; orb2++){

      sum = 0.0;
      for (GN=0; GN<TNumGrid; GN++){
        sum += b_on_grid[orb1][GN]*phi_on_grid[orb2][GN];  
      }

      B[orb1*Nmax+orb2] = sum*GridVol;

      //printf("%7.4f ",B[orb1*Nmax+orb2]); 
    }
    //printf("\n");
  }

  /* diagonalization of the matrix B */

  Eigen_lapack3( B, ev, Nmax, Nmax );

  for (orb=0; orb<Nmax; orb++){
    printf("VVV1 orb=%2d ev=%15.12f\n",orb,ev[orb]);
  }
  
  for (orb=0; orb<Nmax; orb++){ ev[orb] = 1.0/ev[orb]; }

  /* calculate |a_p> by \sum_{q} |b_q> (U)_{qp} */

  for (orb1=0; orb1<Nmax; orb1++){

    for (GN=0; GN<TNumGrid; GN++){ phi_on_grid[orb1][GN] = 0.0; }

    for (orb2=0; orb2<Nmax; orb2++){

      tmp = B[orb1*Nmax+orb2];
      
      for (GN=0; GN<TNumGrid; GN++){
        phi_on_grid[orb1][GN] += b_on_grid[orb2][GN]*tmp;
      }
    }
  }
  
  /* calculate <chi_i|Vx|chi_j> */

  for (i=0; i<(n*n); i++){ Hex[i] = 0.0; }
  
  for (orb=0; orb<Nmax; orb++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      
      Gc_AN = M2G[Mc_AN];    
      wanA = WhatSpecies[Gc_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[Gc_AN] - 1;

      for (i=0; i<tnoA; i++){ asum[i] = 0.0; }
      
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	GN = GridListAtom[Mc_AN][Nc];

	tmp = phi_on_grid[orb][GN];
	for (i=0; i<tnoA; i++){
	  asum[i] += tmp*Orbs_Grid[Mc_AN][Nc][i];
	}
      }

      for (i=0; i<tnoA; i++){
	proj_chi[Anum+i] = asum[i]*GridVol;
      }
    } // Mc_AN

    for (i=0; i<n; i++){
      for (j=0; j<n; j++){
        Hex[i*n+j] += proj_chi[i]*proj_chi[j]*ev[orb];
      }
    }
  } // orb

  /*
  printf("WWW2\n");
  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      printf("%7.4f ", Hex[i*n+j]);
    }
    printf("\n");
  }

  MPI_Finalize();
  exit(0);
  */

  
  for (i=0; i<(n*n); i++){
    Hex[i] = 0.98*Hex0[i] + 0.02*Hex[i];
    //Hex[i] = 0.0;
    Hex0[i] = Hex[i];
  }

  /*
  printf("WWW3\n");
  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      printf("%7.4f ", Hex[i*n+j]);
    }
    printf("\n");
  }
  */
  
  /*
  for (i=0; i<n; i++){
    printf("GGG i=%2d Hex=%10.5f\n",i,Hex[i*n+i]);
  }
  */

  /*
  MPI_Finalize();
  exit(0);
  */
  
  /* freeing of arrays */

  free(work1_on_grid);
  free(work2_on_grid);
  
  for (i=0; i<Nmax; i++){
    free(phi_on_grid[i]);
  }
  free(phi_on_grid);

  for (i=0; i<Nmax; i++){
    free(b_on_grid[i]);
  }
  free(b_on_grid);

  free(B);
  free(ev);

  free(asum);
  free(proj_chi);

  //free(Hex);
  
  dtime(&etime);
  return (etime-stime);

}






static void Patch2Full_Cluster_Owner(double ****RH, double *H, int *MP, int owns_dense)
{
  int i,j,k;
  int MA_AN,GA_AN,LB_AN,GB_AN,AN;
  int wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM;
  int num,tnum;
  int ID,myid,numprocs;
  int *My_NZeros;
  int *is1,*ie1,*is2;
  int *My_Matomnum,*order_GA;
  double *H1;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  My_NZeros = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Full My_NZeros");
  My_Matomnum = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Full My_Matomnum");
  is1 = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Full is1");
  ie1 = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Full ie1");
  is2 = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Full is2");
  order_GA = (int*)ClusterCol_MallocArray((size_t)(atomnum+2),sizeof(int),"Patch2Full order_GA");

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    num = 0;
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num;
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++) tnum += My_NZeros[ID];

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;
  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }

  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  is2[0] = 1;
  for (ID=1; ID<numprocs; ID++){
    is2[ID] = is2[ID-1] + My_Matomnum[ID-1];
  }

  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    order_GA[is2[myid]+MA_AN-1] = M2G[MA_AN];
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&order_GA[is2[ID]],My_Matomnum[ID],MPI_INT,ID,mpi_comm_level1);
  }

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum += Spe_Total_CNO[wanA];
  }
  NUM = Anum - 1;

  H1 = (double*)ClusterCol_MallocArray((size_t)(tnum+1),sizeof(double),"Patch2Full sparse buffer");

  k = is1[myid];
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
      for (i=0; i<tnoA; i++){
        for (j=0; j<tnoB; j++){
          H1[k] = RH[MA_AN][LB_AN][i][j];
          k++;
        }
      }
    }
  }

  for (ID=0; ID<numprocs; ID++){
    k = is1[ID];
    MPI_Bcast(&H1[k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
  }

  if (owns_dense){
    for (i=0; i<NUM*NUM; i++) H[i] = 0.0;

    k = 0;
    for (AN=1; AN<=atomnum; AN++){
      GA_AN = order_GA[AN];
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
            H[(size_t)NUM*(size_t)(Bnum+j-1) + (size_t)(Anum+i-1)] += H1[k];
            k++;
          }
        }
      }
    }
  }

  free(My_NZeros);
  free(My_Matomnum);
  free(is1);
  free(ie1);
  free(is2);
  free(order_GA);
  free(H1);
}

static void ClusterCol_BuildDenseIndex(const int *order_GA, int *MP, int n, int tnum, int *dense_index)
{
  int i,j,k;
  int AN,GA_AN,LB_AN,GB_AN;
  int wanA,wanB,tnoA,tnoB,Anum,Bnum;

  k = 0;
  for (AN=1; AN<=atomnum; AN++){
    GA_AN = order_GA[AN];
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
          int row = Bnum + j - 1;
          int col = Anum + i - 1;

          if (row<0 || n<=row || col<0 || n<=col){
            ClusterCol_AbortWithMessage("Dense matrix index is out of range in Cluster_DFT_Col.c.");
          }
          if (tnum<=k){
            ClusterCol_AbortWithMessage("Dense index buffer overflow in Cluster_DFT_Col.c.");
          }

          dense_index[k] = row*n + col;
          k++;
        }
      }
    }
  }

  if (k!=tnum){
    ClusterCol_AbortWithMessage("Dense index count mismatch in Cluster_DFT_Col.c.");
  }
}

static void ClusterCol_BuildDeviceDenseFromPacked(const double *H1, const int *dense_index,
                                                  int tnum, int n, double *d_H)
{
  double stime = 0.0;

  if (d_H==NULL){
    ClusterCol_AbortWithMessage("NULL device dense matrix in Cluster_DFT_Col.c.");
  }

  if (ClusterCol_detail_timers.active) {
    dtime(&stime);
  }

  if (ClusterCol_BuildDeviceDenseFromPacked_HIP(H1,dense_index,tnum,n,d_H) != 0){
    ClusterCol_AbortWithMessage("Cluster_DFT_Col HIP dense matrix generation failed.");
  }

  if (ClusterCol_detail_timers.active) {
    ClusterCol_DetailTimer_Add(&ClusterCol_detail_timers.dense_build, stime);
    ClusterCol_detail_timers.dense_build_count++;
  }
}

static void Patch2Device_Cluster_Owner(double ****RH, int *MP, int owns_dense, int n, double *d_H)
{
  int i,j,k;
  int MA_AN,GA_AN,LB_AN,GB_AN,AN;
  int wanA,wanB,tnoA,tnoB,Anum,NUM;
  int num,tnum;
  int ID,myid,numprocs;
  int *My_NZeros;
  int *is1,*ie1,*is2;
  int *My_Matomnum,*order_GA;
  double *H1;
  int *dense_index = NULL;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  My_NZeros = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Device My_NZeros");
  My_Matomnum = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Device My_Matomnum");
  is1 = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Device is1");
  ie1 = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Device ie1");
  is2 = (int*)ClusterCol_MallocArray((size_t)numprocs,sizeof(int),"Patch2Device is2");
  order_GA = (int*)ClusterCol_MallocArray((size_t)(atomnum+2),sizeof(int),"Patch2Device order_GA");

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    num = 0;
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num;
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++) tnum += My_NZeros[ID];

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;
  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }

  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  is2[0] = 1;
  for (ID=1; ID<numprocs; ID++){
    is2[ID] = is2[ID-1] + My_Matomnum[ID-1];
  }

  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    order_GA[is2[myid]+MA_AN-1] = M2G[MA_AN];
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&order_GA[is2[ID]],My_Matomnum[ID],MPI_INT,ID,mpi_comm_level1);
  }

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum += Spe_Total_CNO[wanA];
  }
  NUM = Anum - 1;
  if (NUM!=n){
    ClusterCol_AbortWithMessage("Orbital count mismatch in Cluster_DFT_Col.c Patch2Device.");
  }

  H1 = (double*)ClusterCol_MallocArray((size_t)(tnum+1),sizeof(double),"Patch2Device sparse buffer");

  k = is1[myid];
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (i=0; i<tnoA; i++){
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
        for (j=0; j<tnoB; j++){
          H1[k] = RH[MA_AN][LB_AN][i][j];
          k++;
        }
      }
    }
  }

  for (ID=0; ID<numprocs; ID++){
    k = is1[ID];
    MPI_Bcast(&H1[k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
  }

  if (owns_dense){
    dense_index = (int*)ClusterCol_MallocArray((size_t)tnum,sizeof(int),"Patch2Device dense_index");
    ClusterCol_BuildDenseIndex(order_GA,MP,n,tnum,dense_index);
    ClusterCol_BuildDeviceDenseFromPacked(H1,dense_index,tnum,n,d_H);
  }

  free(dense_index);
  free(My_NZeros);
  free(My_Matomnum);
  free(is1);
  free(ie1);
  free(is2);
  free(order_GA);
  free(H1);
}
