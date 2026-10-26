/**********************************************************************************
  Cluster_DFT_Col.c:

     Cluster_DFT_Col.c is a subroutine to perform cluster collinear calculations.

  Log of Cluster_DFT_Col.c:

     21/Feb./2019  Released by T. Ozaki

**********************************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_solver_workspace.h"
#include "openmx_native_resources.h"
#include "openmx_mgpu_runtime.h"
#include "openmx_gpu_transform.h"
#include "openmx_gpu_dm_collective.h"
#include "openmx_gpu_workers.h"
#include "openmx_host_simd.h"

#include "lapack_prototypes.h"
#include "set_cuda_default_device_from_local_rank.h"
#include "set_openacc_device_from_local_rank.h"
#include "elpa_cosma_bridge.h"
#include "openmx_cluster_capacity.h"
#include "eigen_refine_gpu.h"
#include "openmx_precision.h"
#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <time.h>

/* Persistent metadata has exactly the explicit calculation lifetime. Native
   resource owners release buffers/handles; fields below only borrow them. */
typedef struct {
    int n;                       /* order the grid was built for (0: none) */
    int bhandle, ictxt;
    int np_rows, np_cols, my_prow, my_pcol, nblk;
    int na_rows, na_cols, na_rows_max, na_cols_max;
    int desc[9];
    double *Ss, *Hs, *Cs;
    int overlap_n;               /* order whose X = U lambda^-1/2 is in Ss */
} ClusterColUnionGrid;

typedef struct {
    int                initialized;
    int                device_id;
    int                matrix_dim;
    int                scratch_released;
    int                transformed_s_valid;
    int                transformed_s_dim;
    /* serial number of the current d_S: the forward GEMMs may reuse the
       prepared form of X only while this number stands */
    unsigned long long transformed_s_version;
    size_t             d_work_bytes;
    size_t             h_work_bytes;
    cudaStream_t       stream;
    cublasHandle_t     cublas;
    cusolverDnHandle_t gpusolver;
    double *           d_S;
    double *           d_H;
    double *           d_tmp;
    double *           d_W;
    int32_t *          d_info;
    void *             d_work;
    void *             h_work;
    /* per-spin device copy of the latest dense eigenvector panel
       (state-major, leading dimension evec_stash_maxn), so the DM
       accumulation on the dense-owning rank can skip the host round trip */
    double *           d_evec_stash[2];
    size_t             evec_stash_count[2];
    int                evec_stash_valid[2];
    int                evec_stash_n;
    int                evec_stash_maxn;
} ClusterColGpuSolverCtx;

typedef struct {
    int valid;
    int n;
    int entry_count;
    unsigned long long fingerprint;
    int *basis0;
    int *basis1;
} ClusterColDMEntryCache;

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
    int     basis_blocked; /* the reservation probe left no room for the basis: FP64 eigensolver this cycle */
} ClusterColRefineState;

typedef struct {
    int valid;
    int n;
    int tnum;
    unsigned long long fingerprint;
    int *dense_index;
} ClusterColDenseIndexCache;

struct OpenMXClusterColState {
  OpenMXChemicalPotentials *chemical; /* borrowed from this calculation */
  OpenMXThermodynamicEnergies *thermo; /* borrowed from this calculation */
  OpenMXNativeResources *resources; /* borrowed from this calculation */
  int initialized;
  int mgpu_host_dense;
  int gs2_fallback_n;
  ClusterColUnionGrid ugrid;
  double *Hex0;
  double *Hex;
  ClusterColGpuSolverCtx gpusolver_ctx;
  ClusterColRefineState refine;
  EigenRefineState      refine_state[2];
  double               *refine_x[2];
  int                   refine_x_n;
  ClusterColDMEntryCache dm_entry_cache;
  int sample_geometry;
  EigenRefineDevice     refine_dev;
  unsigned long long transformed_s_serial;
  int cpu_overlap_n;
  ClusterColDenseIndexCache dense_index_cache;
  int firsttime;
  int diag_verdict;
  int diag_verdict_n;
  double profile_seconds[4];
  long profile_solves;
};
typedef OpenMXClusterColState ClusterColState;

static void ClusterCol_ReleaseCaches(ClusterColState *state);
static void ClusterCol_Reset(OpenMXNativeResources *owner, void *metadata)
{
  ClusterColState *state = (ClusterColState *)metadata;
  (void)owner;
  if (state->initialized) ClusterCol_ReleaseCaches(state);
  openmx_gemmul8ReleasePreparedOwner(state);
}

OpenMXClusterColState *Cluster_DFT_Col_Bind_Resources(OpenMXCoreContext *core)
{
  OpenMXNativeResources *owner = NULL;
  void *metadata = NULL;
  char error[256];
  if (!openmx_core_native_resources_state(core, "Cluster_DFT_Col",
      sizeof(ClusterColState), ClusterCol_Reset, &owner, &metadata, error, sizeof(error))) {
    fprintf(stderr, "OpenMX collinear cluster state: %s\n", error);
    abort();
  }
  ClusterColState *state = (ClusterColState *)metadata;
  if (!state->initialized) {
    state->resources = owner;
    const ClusterColRefineState initial = {-1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, -1, {0, 0}, 0, 0};
    state->refine = initial;
    state->firsttime = 1;
    state->diag_verdict = -1;
    if(!openmx_core_thermodynamic_energies(core,&state->thermo,error,sizeof(error))) {
      fprintf(stderr,"OpenMX thermodynamic energies: %s\n",error);
      abort();
    }
    if(!openmx_core_chemical_potentials(core,&state->chemical,error,sizeof(error))) {
      fprintf(stderr,"OpenMX chemical potentials: %s\n",error);
      abort();
    }
    state->initialized = 1;
  }
  return state;
}

/* matrix order for which the automatic capacity fallback of gpusolver
   chose the distributed gpusolver2 path (0: none); see
   openmx_cluster_capacity.h */

/* the distributed ELPA/COSMA path: requested by the input, or the capacity
   fallback of gpusolver for this order */
static int ClusterCol_UseGs2(ClusterColState *state, int n)
{
    return gpusolver2_flag || (0 < n && state->gs2_fallback_n == n);
}

static void ClusterCol_ResetGs2Fallback(ClusterColState *state)
{
    if (state->gs2_fallback_n != 0) openmx_gs2_set_fallback_report(0);
    state->gs2_fallback_n = 0;
}

/* Grid over every rank of mpi_comm_level1 for the capacity fallback of a
   spin-polarized run.  The spin worlds are contiguous halves of the ranks,
   i.e. one node each under the usual block placement, so solving each spin
   on its own world's grid would never spread one solve over the GPUs of both
   nodes.  In the fallback both spins are solved in turn on this grid, and
   each spin's eigenvectors go to its own world in the layout of the
   ScaLAPACK path (EVec1[spin], states is2..ie2 of the world's ranks). */

/* collective over mpi_comm_level1 */
static void ClusterCol_UnionGrid_Free(ClusterColState *state)
{
    ClusterColUnionGrid *g = &state->ugrid;

    if (g->n == 0) return;
    openmx_gs2_grid_free(g->ictxt);
    Cfree_blacs_system_handle(g->bhandle);
    Cblacs_gridexit(g->ictxt);
    openmx_core_native_free(state->resources,g->Ss);
    openmx_core_native_free(state->resources,g->Hs);
    openmx_core_native_free(state->resources,g->Cs);
    memset(g, 0, sizeof(*g));
}

/* collective over mpi_comm_level1: the grid of order n, built like the
   cluster grids of DFT.c ("Row" order, np_cols the largest divisor of the
   rank count not above its square root, nblk from 128 down) */
static void ClusterCol_UnionGrid_Ensure(ClusterColState *state, int n)
{
    ClusterColUnionGrid *g = &state->ugrid;
    int numprocs0, myid0, nblk_m, info, ZERO = 0;

    if (g->n == n) return;
    ClusterCol_UnionGrid_Free(state);

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

    g->Ss = (double *)openmx_core_native_malloc(state->resources,1,sizeof(double) * (size_t)g->na_rows * (size_t)g->na_cols + sizeof(double),0);
    g->Hs = (double *)openmx_core_native_malloc(state->resources,1,sizeof(double) * (size_t)g->na_rows * (size_t)g->na_cols + sizeof(double),0);
    g->Cs = (double *)openmx_core_native_malloc(state->resources,1,sizeof(double) * (size_t)g->na_rows_max * (size_t)g->na_cols_max + sizeof(double),0);
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

static ClusterColGridGlobals ClusterCol_EnterUnionGridGlobals(ClusterColState *state)
{
    const ClusterColUnionGrid *g = &state->ugrid;
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

/* GPU GpuSolver context (added by H.Kawai, ported from 3.9.9 GPU) */

/* The refined FP32 eigensolver (eigen_refine_gpu.c) in the root dense path
   of the SCF solves, per spin.  OPENMX_EIGEN_REFINE=K (0: off), unset: on
   by default on GPUs whose FP64 throughput is a small fraction of their
   FP32 one when the real GEMMul8 products are on (see
   openmx_eigen_refine_configure); the global root's device decides for all
   ranks.  The occupations that define the clusters need the chemical
   potential of both spins: with spin polarization, the two spin worlds'
   owners exchange the estimates of their FP32 solves (the serialized mode,
   where one owner solves both spins, and a single rank solving both keep
   the FP64 eigensolver).  An SCF that ends on a refined step has the step
   solved once more in FP64 (mode "scf" with Cluster_DFT_Col_RefineForceFP64
   on, which recomputes the density matrix as well), and the last step
   scf.maxIter allows is FP64 in any case. */

   /* the bases, n x n each */

static void ClusterCol_RefineRelease(ClusterColState *state)
{
    for (int spin = 0; spin < 2; spin++) {
        openmx_eigen_refine_state_release(&state->refine_state[spin]);
        if (state->refine_x[spin] != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,state->refine_x[spin]));
        state->refine_x[spin] = NULL;
    }
    state->refine_x_n = 0;
    openmx_eigen_refine_device_release(&state->refine_dev);
}

/* every rank, once: OPENMX_EIGEN_REFINE, or the default the global root's
   device implies, for all ranks alike */
static void ClusterCol_RefineConfigure(ClusterColState *state)
{
    ClusterColRefineState *st = &state->refine;
    int myid, config[2] = {0, 0};

    /* Input files in -runtest share a process; do not cache an input veto. */
    if (!openmx_fp32_enabled()) return;
    if (0 <= st->iterations) return;
    MPI_Comm_rank(mpi_comm_level1, &myid);
    if (myid == Host_ID) config[0] = openmx_eigen_refine_configure(0, 2 <= level_stdout, &config[1]);
    MPI_Bcast(config, 2, MPI_INT, Host_ID, mpi_comm_level1);
    st->iterations = config[0];
    st->defaulted = (0 < config[1]);
    st->until = openmx_eigen_refine_until();
}

static int ClusterCol_RefineIterations(ClusterColState *state)
{
    return (openmx_fp32_enabled() && 0 < state->refine.iterations) ? state->refine.iterations : 0;
}

/* what the refined solve adds to the dense owner's device memory: the basis */
static size_t ClusterCol_RefineBasisBytes(ClusterColState *state, int n)
{
    return (0 < ClusterCol_RefineIterations(state)) ? sizeof(double) * (size_t)n * (size_t)n : 0;
}

/* every rank, at the start of the root dense path of a solve */
static void ClusterCol_RefineBeginSolve(ClusterColState *state, int SCF_iter, int spin_mode, int peer)
{
    ClusterColRefineState *st = &state->refine;

    ClusterCol_RefineConfigure(state);
    if (st->scf_mode && SCF_iter == 1) {
        st->final_stage = 0;
        st->cycle_refined = 0;
        state->refine_state[0].basis_valid = 0;   /* a new cycle may come with a new geometry */
        state->refine_state[1].basis_valid = 0;
    }
    st->transient = (SCF_iter == 1);
    st->spin_mode = spin_mode;
    st->peer = peer;
    st->solved[0] = st->solved[1] = 0;
    st->persistent = 0;
    /* the occupations of the refinement are the plain Fermi function of
       one chemical potential: XANES and emptied orbitals or states keep
       the FP64 eigensolver */
    st->last_refined = (st->scf_mode && 0 < ClusterCol_RefineIterations(state) && !st->final_stage && !st->force_fp64 &&
                        Cnt_switch == 0 && SCF_iter < DFTSCF_loop && spin_mode != 2 && xanes_calc == 0 &&
                        empty_occupation_flag == 0 && empty_states_flag == 0);
}

/* every rank, at the end of the root dense path: the solves of the owners
   decide the stage for all ranks alike */
static void ClusterCol_RefineEndSolve(ClusterColState *state)
{
    ClusterColRefineState *st = &state->refine;
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

int Cluster_DFT_Col_RefineLastRefined(ClusterColState *state)
{
    return state->refine.last_refined;
}

void Cluster_DFT_Col_RefineForceFP64(ClusterColState *state, int on)
{
    state->refine.force_fp64 = on;
}

/* DFT.c, after the mixing of an SCF step that goes on: normrd is the printed
   NormRD */
void Cluster_DFT_Col_RefineAfterMixing(ClusterColState *state, double normrd)
{
    ClusterColRefineState *st = &state->refine;
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

static int ClusterCol_RefinedEigenDevice(ClusterColState *state, int spin, int n, int maxn, double *ko_spin);
/* The generation is monotonic within this calculation, including cache resets. */

/* The CPU overlap is not constructed when the first SCF solve uses the
   GPU. A later memory-admission fallback must build it before using Ss. */

static void Patch2Device_Cluster_Owner(ClusterColState *state, double ****RH, int *MP, int owns_dense, int n, double *d_H);
static void Patch2Host_Cluster_Owner(ClusterColState *state, double ****RH, int *MP, int owns_dense, int n, double *d_H);
static void ClusterCol_BuildDenseIndex(const int *order_GA, int *MP, int n, int tnum, int *dense_index);
static void ClusterCol_BuildDeviceDenseFromPacked(const double *H1, const int *dense_index,
                                                  int tnum, int n, double *d_H);

static void ClusterCol_AbortWithMessage(const char *message)
{
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1, 1);
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

static void *ClusterCol_MallocArray(ClusterColState *state, size_t count, size_t elem_size, const char *label)
{
    size_t bytes = ClusterCol_CheckedMulCount(count, elem_size, label);
    void *ptr = openmx_core_native_malloc(state->resources,1,(bytes == 0) ? 1 : bytes,0);

    if (ptr == NULL) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Out of memory in Cluster_DFT_Col.c: %s (%zu bytes)", label, bytes);
        ClusterCol_AbortWithMessage(msg);
    }

    return ptr;
}

/* Bounded device allocation for the cluster dense path.  wait_cudafunc()
   retries forever, which suits the transient pressure the band-path turn
   ladder resolves, but on a device that stays full (many ranks sharing one
   small GPU) it spins silently while every other rank waits in MPI.  Absorb
   short races with a few retries and report a persistent failure instead. */
static cudaError_t ClusterCol_TryDeviceMalloc(ClusterColState *state, void **ptr, size_t bytes)
{
    cudaError_t err = cudaErrorMemoryAllocation;

    for (int attempt=0; attempt<8; attempt++){
        err = OpenMX_Resource_GpuMalloc(state->resources,ptr,bytes);
        if (err == cudaSuccess) return cudaSuccess;
        (void)cudaGetLastError();

        double wait_time    = drand48() * WAITTIME;
        double start_time   = MPI_Wtime();
        double current_time = start_time;
        while ((current_time - start_time) < wait_time) {
            current_time = MPI_Wtime();
        }
    }

    *ptr = NULL;
    return err;
}

/* 512-byte aligned sub-allocation of a device arena */
static size_t ClusterCol_ArenaOffset(size_t *pos, size_t bytes)
{
    size_t off = *pos;

    *pos = (off + bytes + 511U) & ~(size_t)511U;
    return off;
}

static void ClusterCol_DeviceOutOfMemoryAbort(const char *what, size_t bytes)
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    char msg[512];

    if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes) != cudaSuccess){
        (void)cudaGetLastError();
    }
    snprintf(msg, sizeof(msg),
             "Cluster_DFT_Col.c: out of GPU memory while allocating %s "
             "(%.1f MiB requested, %.1f MiB free of %.1f MiB). "
             "Reduce the MPI ranks sharing the device or set "
             "OPENMX_CLUSTER_GPU_DIAG=0 to use the CPU diagonalization.",
             what, (double)bytes/(1024.0*1024.0),
             (double)free_bytes/(1024.0*1024.0),
             (double)total_bytes/(1024.0*1024.0));
    ClusterCol_AbortWithMessage(msg);
}

static void ClusterCol_GpuSolver_Destroy(ClusterColState *state)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    int current_device = -1;
    int restore_device = 0;

    /*
       The context can outlive a phase which changes the CUDA device selected
       by the calling rank.  CUDA streams and library handles must be destroyed
       while their owning device is current.  Restore the caller's selection so
       this routine is also safe when it is used from the device-switch path in
       ClusterCol_GpuSolver_Init().
    */
    if (ctx->initialized && 0 <= ctx->device_id) {
        wait_cudafunc(cudaGetDevice(&current_device));
        if (current_device != ctx->device_id) {
            wait_cudafunc(cudaSetDevice(ctx->device_id));
            restore_device = 1;
        }
    }

    if (ctx->stream != NULL)     wait_cudafunc(cudaStreamSynchronize(ctx->stream));
    ClusterCol_RefineRelease(state);

    for (int spin=0; spin<2; spin++){
        if (ctx->d_evec_stash[spin] != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_evec_stash[spin]));
    }
    if (ctx->d_S != NULL)        wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_S));
    if (ctx->d_H != NULL)        wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_H));
    if (ctx->d_tmp != NULL)      wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_tmp));
    if (ctx->d_W != NULL)        wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_W));
    if (ctx->d_info != NULL)     wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_info));
    if (ctx->d_work != NULL)     wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_work));
    if (ctx->h_work != NULL)     openmx_core_native_free(state->resources,ctx->h_work);
    if (ctx->gpusolver != NULL)   OpenMXSolverWait(OpenMX_Resource_SolverDestroy(state->resources,ctx->gpusolver));
    if (ctx->cublas != NULL)     OpenMXBlasWait(OpenMX_Resource_BlasDestroy(state->resources,ctx->cublas));
    if (ctx->stream != NULL)     wait_cudafunc(OpenMX_Resource_StreamDestroy(state->resources,ctx->stream));

    memset(ctx, 0, sizeof(*ctx));
    ctx->device_id = -1;

    if (restore_device)          wait_cudafunc(cudaSetDevice(current_device));
}

static void ClusterCol_GpuSolver_Init(ClusterColState *state)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    int                    current_device;

    wait_cudafunc(cudaGetDevice(&current_device));

    if (ctx->initialized && ctx->device_id == current_device) return;
    if (ctx->initialized) ClusterCol_GpuSolver_Destroy(state);

    wait_cudafunc(OpenMX_Resource_StreamCreate(state->resources,&ctx->stream,cudaStreamNonBlocking));
    OpenMXBlasWait(OpenMX_Resource_BlasCreate(state->resources,&ctx->cublas));
    OpenMXSolverWait(OpenMX_Resource_SolverCreate(state->resources,&ctx->gpusolver));
    OpenMXBlasWait(cublasSetStream(ctx->cublas, ctx->stream));
    OpenMXSolverWait(cusolverDnSetStream(ctx->gpusolver, ctx->stream));

    ctx->initialized = 1;
    ctx->device_id   = current_device;
}

static int ClusterCol_TryEnsureMatrixCapacity(ClusterColState *state, int n)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    size_t dense_count;
    size_t matrix_bytes;
    size_t vector_bytes;

    if (n<=0){
        ClusterCol_AbortWithMessage("Invalid matrix size in Cluster_DFT_Col.c GpuSolver workspace.");
    }

    ClusterCol_GpuSolver_Init(state);

    if (n<=ctx->matrix_dim){
        /* Under device pressure the SCF boundary retains only the
           transformed overlap. Recreate scratch without invalidating S. */
        matrix_bytes = ClusterCol_CheckedMulCount(
            ClusterCol_CheckedMulCount((size_t)ctx->matrix_dim,(size_t)ctx->matrix_dim,
                                       "retained overlap dimensions"),
            sizeof(double),"dense scratch bytes");
        if ((ctx->d_H==NULL && ClusterCol_TryDeviceMalloc(state, (void**)&ctx->d_H,matrix_bytes)!=cudaSuccess) ||
            (ctx->d_tmp==NULL && ClusterCol_TryDeviceMalloc(state, (void**)&ctx->d_tmp,matrix_bytes)!=cudaSuccess)){
            if (ctx->d_H!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_H));
            if (ctx->d_tmp!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_tmp));
            ctx->d_H = ctx->d_tmp = NULL;
            return 0;
        }
        ctx->scratch_released = 0;
        return 1;
    }

    if (ctx->d_S != NULL)    wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_S));
    if (ctx->d_H != NULL)    wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_H));
    if (ctx->d_tmp != NULL)  wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_tmp));
    if (ctx->d_W != NULL)    wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_W));
    if (ctx->d_info != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_info));

    dense_count = ClusterCol_CheckedMulCount((size_t)n,(size_t)n,"GpuSolver dense matrix");
    matrix_bytes = ClusterCol_CheckedMulCount(dense_count,sizeof(double),"GpuSolver dense matrix bytes");
    vector_bytes = ClusterCol_CheckedMulCount((size_t)n,sizeof(double),"GpuSolver eigenvalue bytes");

    ctx->d_S = NULL;
    ctx->d_H = NULL;
    ctx->d_tmp = NULL;
    ctx->d_W = NULL;
    ctx->d_info = NULL;
    ctx->matrix_dim = 0;

    if (ClusterCol_TryDeviceMalloc(state, (void**)&ctx->d_S,matrix_bytes)      != cudaSuccess ||
        ClusterCol_TryDeviceMalloc(state, (void**)&ctx->d_H,matrix_bytes)      != cudaSuccess ||
        ClusterCol_TryDeviceMalloc(state, (void**)&ctx->d_tmp,matrix_bytes)    != cudaSuccess ||
        ClusterCol_TryDeviceMalloc(state, (void**)&ctx->d_W,vector_bytes)      != cudaSuccess ||
        ClusterCol_TryDeviceMalloc(state, (void**)&ctx->d_info,sizeof(int32_t)) != cudaSuccess){
        /* release the partial reservation so a peer rank (or the CPU
           fallback that the caller may pick) sees the memory again */
        if (ctx->d_S != NULL)    wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_S));
        if (ctx->d_H != NULL)    wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_H));
        if (ctx->d_tmp != NULL)  wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_tmp));
        if (ctx->d_W != NULL)    wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_W));
        if (ctx->d_info != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_info));
        ctx->d_S = NULL;
        ctx->d_H = NULL;
        ctx->d_tmp = NULL;
        ctx->d_W = NULL;
        ctx->d_info = NULL;
        return 0;
    }

    ctx->matrix_dim = n;
    ctx->scratch_released = 0;
    ctx->transformed_s_valid = 0;
    ctx->transformed_s_dim = 0;
    return 1;
}

static void ClusterCol_GpuSolver_EnsureMatrixCapacity(ClusterColState *state, int n)
{
    if (!ClusterCol_TryEnsureMatrixCapacity(state, n)){
        size_t dense_count = ClusterCol_CheckedMulCount((size_t)n,(size_t)n,"GpuSolver dense matrix");
        size_t matrix_bytes = ClusterCol_CheckedMulCount(dense_count,sizeof(double),"GpuSolver dense matrix bytes");

        ClusterCol_DeviceOutOfMemoryAbort("the dense solver matrices",matrix_bytes);
    }
}

/* Reserve the eigensolver device workspace for a full-range solve of
   dimension n — the largest request an SCF cycle makes (SCF_iter 1
   diagonalizes the overlap and the first Hamiltonian with maxn == n).
   Later partial-range solves reuse the same block when they fit in it.
   Returns 0 when the device cannot hold the workspace, leaving the
   context consistent for ClusterCol_GpuSolver_Destroy(). */
static int ClusterCol_TryEnsureEigenWorkspace(ClusterColState *state, int n)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    double vl = 0.0;
    double vu = 0.0;
    int64_t h_meig = 0;
    size_t d_bytes = 0;
    size_t h_bytes = 0;

    OpenMXSolverWait(cusolverDnXsyevdx_bufferSize(ctx->gpusolver,NULL,jobz,CUSOLVER_EIG_RANGE_ALL,uplo,n,
                                               CUDA_R_64F,ctx->d_H,n,&vl,&vu,1L,n,&h_meig,
                                               CUDA_R_64F,ctx->d_W,CUDA_R_64F,&d_bytes,&h_bytes));

    if (ctx->d_work_bytes<d_bytes){
        if (ctx->d_work!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_work));
        ctx->d_work = NULL;
        ctx->d_work_bytes = 0;
        if (0<d_bytes && ClusterCol_TryDeviceMalloc(state, (void**)&ctx->d_work,d_bytes) != cudaSuccess){
            return 0;
        }
        ctx->d_work_bytes = d_bytes;
    }
    return 1;
}

static void ClusterCol_GpuSolver_EnsureWorkspace(ClusterColState *state, int n, int maxn, double *d_A)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range;
    double vl = 0.0;
    double vu = 0.0;
    int64_t h_meig = 0;
    size_t d_bytes = 0;
    size_t h_bytes = 0;

    if (n<=0 || maxn<=0 || n<maxn){
        ClusterCol_AbortWithMessage("Invalid eigensolver dimensions in Cluster_DFT_Col.c.");
    }

    ClusterCol_GpuSolver_EnsureMatrixCapacity(state, n);
    range = (n==maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;

    OpenMXSolverWait(cusolverDnXsyevdx_bufferSize(ctx->gpusolver,NULL,jobz,range,uplo,n,
                                               CUDA_R_64F,d_A,n,&vl,&vu,1L,maxn,&h_meig,
                                               CUDA_R_64F,ctx->d_W,CUDA_R_64F,&d_bytes,&h_bytes));

    if (ctx->d_work_bytes<d_bytes){
        if (ctx->d_work!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_work));
        ctx->d_work = NULL;
        ctx->d_work_bytes = 0;
        if (0<d_bytes && ClusterCol_TryDeviceMalloc(state, (void**)&ctx->d_work,d_bytes) != cudaSuccess){
            ClusterCol_DeviceOutOfMemoryAbort("the eigensolver workspace",d_bytes);
        }
        ctx->d_work_bytes = d_bytes;
    }

    if (h_bytes==0){
        if (ctx->h_work!=NULL) openmx_core_native_free(state->resources,ctx->h_work);
        ctx->h_work = NULL;
        ctx->h_work_bytes = 0;
    }
    else if (ctx->h_work_bytes<h_bytes){
        if (ctx->h_work!=NULL) openmx_core_native_free(state->resources,ctx->h_work);
        ctx->h_work = ClusterCol_MallocArray(state, h_bytes,1,"GpuSolver host workspace");
        ctx->h_work_bytes = h_bytes;
    }
}

static void ClusterCol_GpuSolver_EigenDevice(ClusterColState *state, double *d_A, int n, int maxn, double *ko)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range;
    double vl = 0.0;
    double vu = 0.0;
    int64_t h_meig = 0;
    int32_t info = 0;
    char msg[256];

    ClusterCol_GpuSolver_EnsureWorkspace(state, n,maxn,d_A);
    range = (n==maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;

    OpenMXSolverWait(cusolverDnXsyevdx(ctx->gpusolver,NULL,jobz,range,uplo,n,
                                    CUDA_R_64F,d_A,n,&vl,&vu,1L,maxn,&h_meig,
                                    CUDA_R_64F,ctx->d_W,CUDA_R_64F,
                                    ctx->d_work,ctx->d_work_bytes,
                                    ctx->h_work,ctx->h_work_bytes,ctx->d_info));
    wait_cudafunc(cudaMemcpyAsync(&info,ctx->d_info,sizeof(int32_t),cudaMemcpyDeviceToHost,ctx->stream));
    wait_cudafunc(cudaMemcpyAsync(ko,ctx->d_W,sizeof(double)*(size_t)maxn,cudaMemcpyDeviceToHost,ctx->stream));
    wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    if (info!=0){
        snprintf(msg,sizeof(msg),"cusolverDnXsyevdx failed in Cluster_DFT_Col.c: info=%d",(int)info);
        ClusterCol_AbortWithMessage(msg);
    }
}

static void ClusterCol_GpuSolver_PrepareTransformedSDevice(ClusterColState *state, int rebuild, int n, double *ko0)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    double *old_s;
    double *new_s;

    ClusterCol_GpuSolver_EnsureMatrixCapacity(state, n);

    if (rebuild || !(ctx->transformed_s_valid && ctx->transformed_s_dim==n)){
        ClusterCol_GpuSolver_EigenDevice(state, ctx->d_S,n,n,ko0+1);

        for (int l=1; l<=n; l++){
            if (ko0[l]<1.0e-10) ko0[l] = 1.0e-10;
            ko0[l] = 1.0/sqrt(ko0[l]);
        }

        wait_cudafunc(cudaMemcpyAsync(ctx->d_W,ko0+1,sizeof(double)*(size_t)n,cudaMemcpyHostToDevice,ctx->stream));

        old_s = ctx->d_S;
        new_s = ctx->d_tmp;
        if (!openmx_gpu_transform(OPENMX_GPU_SCALE_COLUMNS,0,n,n,old_s,n,ctx->d_W,0,new_s,n,0,0.0))
        OpenMXBlasWait(cublasDdgmm(ctx->cublas,CUBLAS_SIDE_RIGHT,n,n,
                                  old_s,n,ctx->d_W,1,new_s,n));

        ctx->d_S = new_s;
        ctx->d_tmp = old_s;
        ctx->transformed_s_valid = 1;
        ctx->transformed_s_dim = n;
        ctx->transformed_s_version = ++state->transformed_s_serial;
        wait_cudafunc(cudaStreamSynchronize(ctx->stream));
    }
}

static void ClusterCol_GpuSolver_PrepareTransformedS(ClusterColState *state, int rebuild, int n, double *S, double *ko0)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    size_t dense_count = ClusterCol_CheckedMulCount((size_t)n,(size_t)n,"transformed overlap");
    size_t matrix_bytes = ClusterCol_CheckedMulCount(dense_count,sizeof(double),"transformed overlap bytes");

    ClusterCol_GpuSolver_EnsureMatrixCapacity(state, n);
    if (rebuild || !(ctx->transformed_s_valid && ctx->transformed_s_dim==n)){
        wait_cudafunc(cudaMemcpyAsync(ctx->d_S,S,matrix_bytes,cudaMemcpyHostToDevice,ctx->stream));
    }
    ClusterCol_GpuSolver_PrepareTransformedSDevice(state, rebuild,n,ko0);
}

static void ClusterCol_GEMMul8Dgemm_Device(ClusterColState *state, cublasOperation_t transa, cublasOperation_t transb,
                                           int m, int n, int k,
                                           const double *A, int lda,
                                           const double *B, int ldb,
                                           double *C, int ldc)
{
    double const alpha = 1.0;
    double const beta  = 0.0;

    ClusterCol_GpuSolver_Init(state);
    OpenMXBlasWait(openmx_gemmul8Dgemm(state->gpusolver_ctx.cublas, transa, transb, m, n, k,
                                      &alpha, A, lda, B, ldb, &beta, C, ldc));
}

/* OPENMX_CLUSTER_PROFILE=1: cumulative wall time of the phases of the dense
   solve (forward transform, eigensolver, back transform, eigenvector
   download), one CLUSTERPROF line per solve.  The stream is synchronized
   around each phase, so the times are not those of an unprofiled run. */
/* OPENMX_BAND_PROFILE=1: one CLUSTERDMPROF line per rank and call with the
   time of the density-matrix accumulation (device, or the host threads) */
static int ClusterCol_DMProfileEnabled(void)
{
    return openmx_env_flag_at(OPENMX_ENV_BAND_PROFILE, 0);
}

static int ClusterCol_ProfileEnabled(void)
{
    return openmx_env_flag_at(OPENMX_ENV_CLUSTER_PROFILE, 0);
}

static double ClusterCol_ProfileLap(ClusterColState *state, double *t0)
{
    double now;

    wait_cudafunc(cudaStreamSynchronize(state->gpusolver_ctx.stream));
    now = MPI_Wtime();
    {
        double lap = now - *t0;
        *t0 = now;
        return lap;
    }
}

static int ClusterCol_GemmWorkspaceTurnRelease(void);

/* The refined solve of the transformed Hamiltonian in d_H (n x n, spin's)
   on the dense owner: the FP32 scratch is d_tmp, the basis a buffer of its
   own per spin.  With two spin worlds the owners exchange the estimates of
   their solves for the chemical potential; a failure on either side makes
   both solve in FP64 this step.  Returns 1 when the refined vectors are in
   the first maxn columns of d_H and ko_spin[1..maxn] holds their
   eigenvalues, else 0 with d_H still the symmetric matrix. */
static cudaError_t ClusterCol_RefineDeviceMalloc(void *context, void **ptr, size_t bytes)
{
    return ClusterCol_TryDeviceMalloc((ClusterColState *)context, ptr, bytes);
}
static void *ClusterCol_RefineHostAllocate(void *context, size_t count, size_t bytes, const char *label)
{
    return ClusterCol_MallocArray((ClusterColState *)context, count, bytes, label);
}

static int ClusterCol_RefinedEigenDevice(ClusterColState *state, int spin, int n, int maxn, double *ko_spin)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    ClusterColRefineState *st = &state->refine;
    EigenRefineDevice *dev = &state->refine_dev;
    EigenRefineProblem pb;
    EigenRefineReport rep;
    ClusterColOccupationCtx occ = {0.0};
    size_t const nn = (size_t)n * (size_t)n;
    double *e0, *f0, *e_peer = NULL;
    int status[2] = {0, 0}, rc;

    if (state->refine_x_n != n) {
        for (int sp = 0; sp < 2; sp++) {
            if (state->refine_x[sp] != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,state->refine_x[sp]));
            state->refine_x[sp] = NULL;
            state->refine_state[sp].basis_valid = 0;
        }
        state->refine_x_n = n;
    }
    /* the reservation probe may have found the device too full for the
       basis next to the dense buffers: then it is not even attempted (it
       would take the margin the other ranks' transients rely on) and the
       cycle is condemned to FP64 at once, first step or not */
    if (state->refine_x[spin] == NULL &&
        (st->basis_blocked ||
         ClusterCol_TryDeviceMalloc(state, (void **)&state->refine_x[spin], nn * sizeof(double)) != cudaSuccess)) {
        const int condemn = (st->basis_blocked || !st->transient);

        if (2 <= level_stdout) {
            printf("<Cluster_DFT_Col> no room for the %.1f MiB basis of the refined eigensolver; FP64 eigensolver%s\n",
                   (double)(nn * sizeof(double)) / (1024.0 * 1024.0), condemn ? " for the rest of the SCF cycle" : "");
            fflush(stdout);
        }
        if (condemn && st->persistent < 1) st->persistent = 1;
        rc = 0;
    }
    else {
        rc = 1;
    }

    dev->cublas = ctx->cublas;
    dev->cusolver = ctx->gpusolver;
    dev->stream = ctx->stream;
    dev->d_work = &ctx->d_work;
    dev->d_work_bytes = &ctx->d_work_bytes;
    dev->h_work = &ctx->h_work;
    dev->h_work_bytes = &ctx->h_work_bytes;
    dev->d_info = ctx->d_info;
    dev->allocation_context = state;
    dev->try_malloc_owned = ClusterCol_RefineDeviceMalloc;
    dev->resources = state->resources;
    dev->host_malloc_owned = ClusterCol_RefineHostAllocate;

    memset(&pb, 0, sizeof(pb));
    memset(&rep, 0, sizeof(rep));
    pb.cplx = 0;
    pb.n = n;
    pb.maxn = maxn;
    pb.iterations = ClusterCol_RefineIterations(state);
    pb.a = ctx->d_H;
    pb.w = ctx->d_W;
    pb.fp32 = ctx->d_tmp;
    pb.region = nn * sizeof(double);
    pb.x = state->refine_x[spin];
    pb.transient = st->transient;
    pb.warm = (rc && openmx_eigen_refine_warm_enabled() && state->refine_state[spin].basis_valid &&
               state->refine_state[spin].basis_n == n && !st->transient);
    pb.defaulted = st->defaulted;
    pb.occupation = ClusterCol_RefineFermi;
    pb.occupation_ctx = &occ;

    e0 = (double *)ClusterCol_MallocArray(state, (size_t)n, sizeof(double), "refinement eigenvalues");
    f0 = (double *)ClusterCol_MallocArray(state, (size_t)n, sizeof(double), "refinement occupations");
    if (rc) {
        if (3 <= level_stdout) {
            printf("<Cluster_DFT_Col> spin %d: %s, %d refinement steps (OPENMX_EIGEN_REFINE)\n", spin,
                   pb.warm ? "warm start from the previous vectors" : "FP32 eigensolver", pb.iterations);
            fflush(stdout);
        }
        rc = openmx_eigen_refine_prepare(&state->refine_state[spin], dev, &pb, e0, &rep);
        if (rc < 0) {
            pb.warm = 0;
            rc = openmx_eigen_refine_prepare(&state->refine_state[spin], dev, &pb, e0, &rep);
        }
        if (rc != 1 && st->persistent < rep.persistent) st->persistent = rep.persistent;
    }
    status[0] = rc;
    if (st->spin_mode == 1) {
        /* the other spin's estimates, and whether its solve is on */
        MPI_Status mpi_status;

        e_peer = (double *)ClusterCol_MallocArray(state, (size_t)n, sizeof(double), "peer eigenvalues");
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
            openmx_eigen_refine_abandon(&state->refine_state[spin]);
            rc = 0;
        }
    }
    if (rc == 1) {
        ClusterCol_RefineOccupations(e0, st->spin_mode == 1 ? e_peer : NULL, n, maxn, f0, &occ.mu);
        rc = openmx_eigen_refine_finish(&state->refine_state[spin], dev, &pb, f0, &rep);
        if (rc != 1 && st->persistent < rep.persistent) st->persistent = rep.persistent;
    }
    openmx_core_native_free(state->resources,e0);
    openmx_core_native_free(state->resources,f0);
    openmx_core_native_free(state->resources,e_peer);
    if (rc == 1) {
        wait_cudafunc(cudaMemcpy(ko_spin + 1, ctx->d_W, sizeof(double) * (size_t)maxn, cudaMemcpyDeviceToHost));
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
    state->refine_state[spin].basis_valid = 0;
    return 0;
}

static void ClusterCol_GpuSolver_SolveHamiltonianDevice(ClusterColState *state, int n, int maxn, double *ko_spin, double *C, int spin)
{

    const int profile = ClusterCol_ProfileEnabled();
    const int trim_scratch = ClusterCol_GemmWorkspaceTurnRelease();
    double t0 = 0.0;
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    size_t evec_count = ClusterCol_CheckedMulCount((size_t)n,(size_t)maxn,"eigenvectors");
    size_t evec_bytes = ClusterCol_CheckedMulCount(evec_count,sizeof(double),"eigenvector bytes");

    if (!(ctx->transformed_s_valid && ctx->transformed_s_dim==n)){
        ClusterCol_AbortWithMessage("Transformed overlap is not ready in Cluster_DFT_Col.c.");
    }

    if (profile) (void)ClusterCol_ProfileLap(state, &t0);

    /* forward transform X^T (H X): X = d_S stays fixed during the SCF.  The
       two products take a full GEMMul8 workspace when the device has room,
       several times the capped one of the other products.  It is returned
       before the eigensolver and the eigenvector stash allocate, so they
       find the memory they found before; when the GEMMul8 workspaces are
       kept across the SCF step (OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE=0) the
       eigensolver workspace is secured first instead. */
    if (!trim_scratch) ClusterCol_GpuSolver_EnsureWorkspace(state, n,maxn,ctx->d_H);
    OpenMXBlasWait(openmx_gemmul8DgemmFixedOwned(state, ctx->cublas,0,CUBLAS_OP_N,n,n,n,
                                           ctx->d_S,n,ctx->d_H,n,ctx->d_tmp,n,
                                           0,ctx->transformed_s_version));
    OpenMXBlasWait(openmx_gemmul8DgemmFixedOwned(state, ctx->cublas,1,CUBLAS_OP_T,n,n,n,
                                           ctx->d_S,n,ctx->d_tmp,n,ctx->d_H,n,
                                           0,ctx->transformed_s_version));
    if (trim_scratch){
        wait_cudafunc(cudaStreamSynchronize(ctx->stream));
        openmx_gemmul8TrimWorkspaces();
    }
    if (profile) state->profile_seconds[0] += ClusterCol_ProfileLap(state, &t0);

    if (!(0 <= spin && state->refine.last_refined && ClusterCol_RefinedEigenDevice(state, spin,n,maxn,ko_spin))){
        ClusterCol_GpuSolver_EigenDevice(state, ctx->d_H,n,maxn,ko_spin+1);
    }
    if (profile) state->profile_seconds[1] += ClusterCol_ProfileLap(state, &t0);

    ClusterCol_GEMMul8Dgemm_Device(state, CUBLAS_OP_T,CUBLAS_OP_T,maxn,n,n,
                                   ctx->d_H,n,ctx->d_S,n,ctx->d_tmp,maxn);
    if (profile) state->profile_seconds[2] += ClusterCol_ProfileLap(state, &t0);
    wait_cudafunc(cudaMemcpyAsync(C,ctx->d_tmp,evec_bytes,cudaMemcpyDeviceToHost,ctx->stream));
    wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    if (profile){
        int rank;

        state->profile_seconds[3] += ClusterCol_ProfileLap(state, &t0);
        state->profile_solves++;
        MPI_Comm_rank(mpi_comm_level1,&rank);
        printf("CLUSTERPROF rank=%d solves=%ld n=%d forward=%.6f eigen=%.6f back=%.6f evec_d2h=%.6f "
               "(s, cumulative)\n",
               rank,state->profile_solves,n,state->profile_seconds[0],state->profile_seconds[1],state->profile_seconds[2],state->profile_seconds[3]);
        fflush(stdout);
    }
}

static void ClusterCol_GpuSolver_SolveHamiltonian(ClusterColState *state, int n, int maxn, const double *H, double *ko_spin, double *C)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    size_t dense_count = ClusterCol_CheckedMulCount((size_t)n,(size_t)n,"Hamiltonian");
    size_t matrix_bytes = ClusterCol_CheckedMulCount(dense_count,sizeof(double),"Hamiltonian bytes");

    ClusterCol_GpuSolver_EnsureMatrixCapacity(state, n);
    wait_cudafunc(cudaMemcpyAsync(ctx->d_H,H,matrix_bytes,cudaMemcpyHostToDevice,ctx->stream));
    ClusterCol_GpuSolver_SolveHamiltonianDevice(state, n,maxn,ko_spin,C,-1);
}

static void ClusterCol_InvalidateDeviceEvecStash(ClusterColState *state)
{
    state->gpusolver_ctx.evec_stash_valid[0] = 0;
    state->gpusolver_ctx.evec_stash_valid[1] = 0;
}

/* Large dense scratch can displace the following density-grid/Hamiltonian
   phases from the GPU. Once the density matrix and output are complete,
   retain S (which is reused across SCF steps) but return disposable storage
   when less than half of the device remains free. Small contexts stay hot. */
static void ClusterCol_TrimScratch(ClusterColState *state)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    const int retain = openmx_env_flag_at(OPENMX_ENV_CLUSTER_GPU_RETAIN_SCRATCH, -1);
    const int release_always = retain==0;
    size_t free_bytes, total_bytes, bytes;
    int device;

    if (!ctx->initialized || ctx->matrix_dim<=0 || ctx->scratch_released ||
        retain==1) return;
    if (cudaGetDevice(&device)!=cudaSuccess || device!=ctx->device_id) return;
    bytes = sizeof(double)*(size_t)ctx->matrix_dim*(size_t)ctx->matrix_dim;
    bytes = (ctx->d_H!=NULL ? bytes : 0) + (ctx->d_tmp!=NULL ? bytes : 0)
          + ctx->d_work_bytes;
    for (int spin=0; spin<2; ++spin)
        bytes += sizeof(double)*ctx->evec_stash_count[spin];
    for (int spin=0; spin<2; ++spin)
        if (state->refine_x[spin]!=NULL)
            bytes += sizeof(double)*(size_t)state->refine_x_n*(size_t)state->refine_x_n;
    if (!release_always && bytes<512ULL*1024ULL*1024ULL) return;
    if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess ||
        (!release_always && free_bytes>=total_bytes/2)) return;

    wait_cudafunc(cudaStreamSynchronize(ctx->stream));
    if (ctx->d_H!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_H));
    if (ctx->d_tmp!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_tmp));
    if (ctx->d_work!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_work));
    ctx->d_H = ctx->d_tmp = NULL;
    ctx->d_work = NULL;
    ctx->d_work_bytes = 0;
    /* the bases of the refined eigensolver go with the scratch; the next
       refined solve starts from an FP32 solve */
    for (int spin=0; spin<2; ++spin){
        if (state->refine_x[spin]!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,state->refine_x[spin]));
        state->refine_x[spin] = NULL;
        state->refine_state[spin].basis_valid = 0;
    }
    state->refine_x_n = 0;
    for (int spin=0; spin<2; ++spin){
        if (ctx->d_evec_stash[spin]!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_evec_stash[spin]));
        ctx->d_evec_stash[spin] = NULL;
        ctx->evec_stash_count[spin] = 0;
        ctx->evec_stash_valid[spin] = 0;
    }
    ctx->scratch_released = 1;
    {
        if (openmx_env_flag_at(OPENMX_ENV_CUSOLVER_VERBOSE, 0))
            fprintf(stderr,"<Cluster_DFT_Col> Released %.3f GiB of SCF scratch; retained transformed overlap (n=%d).\n",
                    (double)bytes/1073741824.0,ctx->matrix_dim);
    }
}

/* Keep the freshly solved eigenvector panel (still in ctx->d_tmp) on the
   device so the DM accumulation of the same SCF step can consume it without
   the D2H/repack/H2D round trip through EVec1. */
static void ClusterCol_StashDeviceEvec(ClusterColState *state, int spin, int n, int maxn)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    size_t count = ClusterCol_CheckedMulCount((size_t)n,(size_t)maxn,"device eigenvector stash");
    size_t bytes = ClusterCol_CheckedMulCount(count,sizeof(double),"device eigenvector stash bytes");

    if (spin<0 || 2<=spin){
        ClusterCol_AbortWithMessage("Invalid spin for the device eigenvector stash in Cluster_DFT_Col.c.");
    }

    if (ctx->d_evec_stash[spin]==NULL || ctx->evec_stash_count[spin]<count){
        if (ctx->d_evec_stash[spin]!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_evec_stash[spin]));
        ctx->d_evec_stash[spin] = NULL;
        ctx->evec_stash_count[spin] = 0;
        /* the stash only short-circuits the host round trip of the DM
           accumulation; when the shared device has no room left the solve
           is already complete, so skip stashing instead of waiting */
        if (ClusterCol_TryDeviceMalloc(state, (void**)&ctx->d_evec_stash[spin],bytes) != cudaSuccess){
            ctx->d_evec_stash[spin] = NULL;
            ctx->evec_stash_valid[spin] = 0;
            return;
        }
        ctx->evec_stash_count[spin] = count;
    }

    wait_cudafunc(cudaMemcpyAsync(ctx->d_evec_stash[spin],ctx->d_tmp,bytes,
                                  cudaMemcpyDeviceToDevice,ctx->stream));
    wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    ctx->evec_stash_n = n;
    ctx->evec_stash_maxn = maxn;
    ctx->evec_stash_valid[spin] = 1;
}

/* Returns the stashed device panel when it covers this rank's state slice
   (only the dense-owning rank, whose slice starts at state 1), else NULL. */
static const double *ClusterCol_DeviceEvecStash(ClusterColState *state, int spin, int n, int kmin, int kmax, int *lds)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;

    if (spin<0 || 2<=spin) return NULL;
    if (!ctx->evec_stash_valid[spin]) return NULL;
    if (ctx->evec_stash_n!=n) return NULL;
    if (kmin!=1 || kmax<kmin || ctx->evec_stash_maxn<kmax) return NULL;

    *lds = ctx->evec_stash_maxn;
    return ctx->d_evec_stash[spin];
}

static void ClusterCol_GEMMul8Dgemm_OpenACC(ClusterColState *state, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                            double const * A, double const * B, double * C)
{
    const int lda = (transa==CUBLAS_OP_N) ? m : k;
    const int ldb = (transb==CUBLAS_OP_N) ? k : n;

    ClusterCol_GpuSolver_Init(state);
#pragma acc data      present(A[0 : lda * ((transa==CUBLAS_OP_N) ? k : m)], \
                              B[0 : ldb * ((transb==CUBLAS_OP_N) ? n : k)], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        ClusterCol_GEMMul8Dgemm_Device(state, transa,transb,m,n,k,A,lda,B,ldb,C,m);
    }
}

static int ClusterCol_GemmWorkspaceTurnRelease(void)
{
    /* Releasing the GEMMul8 workspace once the cluster diagonalization is done
       keeps it from occupying the shared device through the grid/force phases
       of the SCF step; opt out with OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE=0
       (falls back to the band-path knob for a single switch). */
    return openmx_env_flag_at(OPENMX_ENV_CLUSTER_GEMMUL8_TURN_RELEASE, openmx_env_flag_at(OPENMX_ENV_BAND_GEMMUL8_TURN_RELEASE, 1));
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
    long parsed = openmx_env_long_at(OPENMX_ENV_CLUSTER_GPU_SWITCH_NUM, CLUSTER_GPU_CPU_SWITCH_NUM);

    return 0 < parsed && parsed <= INT_MAX ? (int)parsed : CLUSTER_GPU_CPU_SWITCH_NUM;
}

static int ClusterCol_UseGpuAccel(int n)
{
    return (scf_eigen_lib_flag==GPUSOLVER && Cluster_DFT_Col_GpuSwitchNum()<=n &&
            Set_Hamiltonian_OpenACC_Rank_Is_Selected());
}

/* Device-or-host decision of the cluster DM accumulation
   (OpenMX_HostSimdStageOnHost, collective over mpi_comm_level1; called by
   every rank behind the Allreduce'd request). need = the working set of the
   direct device attempt (the capacity_cost of openmx_gpu_dm_try_entries):
   two indices and twelve doubles per entry, three weights per state and,
   for host eigenvectors, the rows of all n orbitals over the states; budget
   = free device memory minus the 256 MiB reserve of the worker capacity
   check. 1 = every rank accumulates with ClusterCol_AccumulateDM_CPU. */
static int ClusterCol_DMOnHost(int entries, int states, size_t panel_bytes)
{
  size_t need = 0;
  if (0<entries && 0<states)
    need = (size_t)entries*(2*sizeof(int)+12*sizeof(double)) + 3*(size_t)states*sizeof(double) + panel_bytes;
  return OpenMX_HostSimdStageOnHost("Cluster_DFT_Col density matrix", need,
                                    OpenMX_HostSimdDeviceBudget((size_t)256<<20));
}

static int ClusterCol_UseGpuDM(int n)
{
    /* OpenACC accumulation of the cluster density matrix; opt out with
       OPENMX_CLUSTER_GPU_DM=0 (falls back to the original CPU loop).
       The rank that owns the dense solve reads the eigenvector panel still
       on the device; every other rank stages its own slice of states (a
       bounded-retry allocation that halves the block, the CPU loop when
       even the smallest block does not fit), so the accumulation no longer
       leaves those ranks on one core each: C1000 at 6 ranks spent 37 s of
       3 SCF steps there on one thread (RTX 5080).  OPENMX_CLUSTER_GPU_DM_ALL=0
       keeps the device accumulation to the owner rank. */
    if (!openmx_env_flag_at(OPENMX_ENV_CLUSTER_GPU_DM, 1)){
        return 0;
    }
    if (ClusterCol_UseGpuAccel(n)) return 1;
    if (!openmx_env_flag_at(OPENMX_ENV_CLUSTER_GPU_DM_ALL, 1)) return 0;
    return (scf_eigen_lib_flag==GPUSOLVER && Cluster_DFT_Col_GpuSwitchNum()<=n);
}

static unsigned long long ClusterCol_LayoutFingerprint(const int *order_GA, int *MP, int n)
{
    unsigned long long h = 1469598103934665603ULL;

#define CLUSTERCOL_HASH_INT(v)                                                                                         \
    do {                                                                                                               \
        h ^= (unsigned long long)(unsigned int)(v);                                                                    \
        h *= 1099511628211ULL;                                                                                        \
    } while (0)

    CLUSTERCOL_HASH_INT(n);
    CLUSTERCOL_HASH_INT(atomnum);
    for (int AN=1; AN<=atomnum; AN++){
        int GA_AN = (order_GA!=NULL) ? order_GA[AN] : AN;
        CLUSTERCOL_HASH_INT(GA_AN);
        CLUSTERCOL_HASH_INT(MP[GA_AN]);
        CLUSTERCOL_HASH_INT(WhatSpecies[GA_AN]);
        CLUSTERCOL_HASH_INT(Spe_Total_CNO[WhatSpecies[GA_AN]]);
        CLUSTERCOL_HASH_INT(FNAN[GA_AN]);
        for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            CLUSTERCOL_HASH_INT(natn[GA_AN][LB_AN]);
        }
    }

#undef CLUSTERCOL_HASH_INT

    return h;
}

static const int *ClusterCol_DenseIndexCache_Get(ClusterColState *state, const int *order_GA, int *MP, int n, int tnum)
{
    ClusterColDenseIndexCache *cache = &state->dense_index_cache;
    unsigned long long fingerprint = ClusterCol_LayoutFingerprint(order_GA,MP,n);

    if (cache->valid && cache->n==n && cache->tnum==tnum && cache->fingerprint==fingerprint){
        return cache->dense_index;
    }

    if (cache->dense_index!=NULL){
        int *old_index = cache->dense_index;
        int old_tnum = cache->tnum;
#pragma acc exit data delete(old_index[0 : old_tnum])
        openmx_core_native_free(state->resources,old_index);
    }
    memset(cache,0,sizeof(*cache));

    cache->dense_index = (int*)ClusterCol_MallocArray(state, (size_t)tnum,sizeof(int),"dense index cache");
    ClusterCol_BuildDenseIndex(order_GA,MP,n,tnum,cache->dense_index);

    /* the scatter pattern is geometry-fixed, so keep it resident on the
       device instead of re-uploading it with every dense build */
    {
        int *new_index = cache->dense_index;
#pragma acc enter data copyin(new_index[0 : tnum])
    }

    cache->valid = 1;
    cache->n = n;
    cache->tnum = tnum;
    cache->fingerprint = fingerprint;
    return cache->dense_index;
}

/* Pre-Force release: the dense solver context (matrices, eigensolver
   workspace, eigenvector stash) and the device-resident scatter index keep
   several GiB of the shared device between SCF cycles.  The force batches
   size themselves from the free device memory, so return everything before
   Force() runs; the next SCF cycle rebuilds on demand. */
void Cluster_DFT_Col_Release_GPU_Solver(ClusterColState *state)
{
    ClusterColDenseIndexCache *cache = &state->dense_index_cache;

    if (cache->dense_index != NULL) {
        int *old_index = cache->dense_index;
        int old_tnum = cache->tnum;
#pragma acc exit data delete(old_index[0 : old_tnum])
        openmx_core_native_free(state->resources,old_index);
    }
    memset(cache, 0, sizeof(*cache));

    ClusterCol_GpuSolver_Destroy(state);
}

static void ClusterCol_DMEntryCache_Ensure(ClusterColState *state, int *MP, int n)
{
    ClusterColDMEntryCache *cache = &state->dm_entry_cache;
    unsigned long long fingerprint = ClusterCol_LayoutFingerprint(NULL,MP,n);
    int entry_count = 0;

    if (cache->valid && cache->n==n && cache->fingerprint==fingerprint) return;

    openmx_core_native_free(state->resources,cache->basis0);
    openmx_core_native_free(state->resources,cache->basis1);
    memset(cache,0,sizeof(*cache));

    for (int GA_AN=1; GA_AN<=atomnum; GA_AN++){
        int tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]];

        for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            int GB_AN = natn[GA_AN][LB_AN];
            entry_count += tnoA*Spe_Total_CNO[WhatSpecies[GB_AN]];
        }
    }

    cache->basis0 = (int*)ClusterCol_MallocArray(state, (size_t)(entry_count+1),sizeof(int),"DM entry basis0");
    cache->basis1 = (int*)ClusterCol_MallocArray(state, (size_t)(entry_count+1),sizeof(int),"DM entry basis1");

    entry_count = 0;
    for (int GA_AN=1; GA_AN<=atomnum; GA_AN++){
        int wanA = WhatSpecies[GA_AN];
        int tnoA = Spe_Total_CNO[wanA];
        int Anum = MP[GA_AN];

        for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            int GB_AN = natn[GA_AN][LB_AN];
            int wanB = WhatSpecies[GB_AN];
            int tnoB = Spe_Total_CNO[wanB];
            int Bnum = MP[GB_AN];

            for (int i=0; i<tnoA; i++){
                int ibasis = Anum + i - 1;

                for (int j=0; j<tnoB; j++){
                    cache->basis0[entry_count] = ibasis;
                    cache->basis1[entry_count] = Bnum + j - 1;
                    entry_count++;
                }
            }
        }
    }

    cache->valid = 1;
    cache->n = n;
    cache->fingerprint = fingerprint;
    cache->entry_count = entry_count;
}

static void ClusterCol_GpuSolver_CheckInfo(const char *where, int info)
{
    if (info != 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s failed in Cluster_DFT_Col.c: info=%d", where, info);
        ClusterCol_AbortWithMessage(msg);
    }
}

static size_t ClusterCol_GpuDiagReserveBytes(void)
{
    return openmx_env_mib_bytes_at(OPENMX_ENV_CLUSTER_GPU_DIAG_RESERVE_MB, 1024);
}

/* Collectively decide whether the root-dense GPU diagonalization fits on
   the device.  When many ranks share one small GPU the dense matrices may
   never fit, and committing to the GPU path used to spin forever inside
   wait_cudafunc(cudaMalloc(...)) while every other rank sat in MPI_Bcast.
   The dense-owning rank of each spin world RESERVES what the dense path
   will allocate — the three n x n matrices plus the eigensolver workspace
   at the size cusolver actually asks for — instead of estimating it: with
   spin polarization two dense owners share one device, and per-owner
   free-memory estimates let both commit when only one fits.  The verdict
   is agreed on over mpi_comm_level1 because the ScaLAPACK/ELPA fallback
   below uses level1 collectives, so both spin worlds must take the same
   branch; when any owner falls short every owner releases its reservation
   again.  It is cached until the next SCF restart or a change of the
   matrix size.

   Returns 0 for the ELPA/ScaLAPACK fallback, 1 for the concurrent GPU
   path (each spin world's owner reserves and solves its own spin), and 2
   for the serialized GPU path: when the device cannot hold two owner
   reservations but can hold one, spin world 0's owner keeps the only
   reservation and solves both spins back to back from the packed
   Set_Hamiltonian cache (every selected rank owns all spins of it),
   shipping the second spin's eigenpairs to spin world 1's owner.
   OPENMX_CLUSTER_GPU_DIAG=1 forces the concurrent GPU path, =0 forces the
   fallback.  OPENMX_CLUSTER_GPU_DIAG_SERIAL=0 disables the serialized
   tier, =2 skips the concurrent probe and forces the serialized tier. */
static int ClusterCol_OwnerReserveProbe(ClusterColState *state, int n, int myworld1, const char *when)
{
    int my_fit = (ClusterCol_TryEnsureMatrixCapacity(state, n) &&
                  ClusterCol_TryEnsureEigenWorkspace(state, n));

    if (my_fit){
        size_t free_bytes = 0;
        size_t total_bytes = 0;

        /* the solve still makes smaller incidental allocations (the
           cusolver internals, the optional GEMMul8 workspace and
           eigenvector stash degrade gracefully) — keep a margin free */
        if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess){
            (void)cudaGetLastError();
            my_fit = 0;
        }
        else{
            const size_t reserve = ClusterCol_GpuDiagReserveBytes();
            const size_t basis = ClusterCol_RefineBasisBytes(state, n);

            if (reserve+basis<=free_bytes){
                state->refine.basis_blocked = 0;
            }
            else if (reserve<=free_bytes){
                /* The dense solve fits, the basis of the refined eigensolver
                   on top of it does not (many ranks sharing a small device):
                   keep the GPU path and solve in FP64 for this cycle rather
                   than fall back to ELPA.  Allocating the basis anyway would
                   eat the margin the other ranks' transients rely on. */
                state->refine.basis_blocked = 1;
                printf("<Cluster_DFT_Col> The dense owner of spin world %d holds its GPU diagonalization"
                       " buffers%s but not the %.1f MiB basis of the refined eigensolver"
                       " (%.1f MiB of %.1f MiB free): FP64 eigensolver for this SCF cycle.\n",
                       myworld1, when,
                       (double)basis/(1024.0*1024.0),
                       (double)free_bytes/(1024.0*1024.0),
                       (double)total_bytes/(1024.0*1024.0));
                fflush(stdout);
            }
            else{
                my_fit = 0;
            }
        }
    }

    if (!my_fit){
        size_t free_bytes = 0;
        size_t total_bytes = 0;

        if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess){
            (void)cudaGetLastError();
        }
        printf("<Cluster_DFT_Col> The dense owner of spin world %d could not reserve its GPU"
               " diagonalization buffers%s (%.1f MiB of %.1f MiB free).\n",
               myworld1, when,
               (double)free_bytes/(1024.0*1024.0),
               (double)total_bytes/(1024.0*1024.0));
        fflush(stdout);
    }

    return my_fit;
}

static int ClusterCol_GpuDiagFits(ClusterColState *state, int SCF_iter, int n, int myworld1, int myid1, int numprocs0)
{

    int serial_knob = (int)openmx_env_long_at(OPENMX_ENV_CLUSTER_GPU_DIAG_SERIAL, 1);
    int announce = (myworld1==0 && myid1==0);
    int owner = (myid1==0);
    int solver_owner = (myworld1==0 && myid1==0);
    int serial_allowed;
    int my_fit = 1;
    int fit = 0;

    /* the reservation probe counts the refined eigensolver's basis */
    ClusterCol_RefineConfigure(state);

    if (SCF_iter==1 || state->diag_verdict_n!=n){
      state->mgpu_host_dense=0;
      ClusterCol_ResetGs2Fallback(state);
    }
    if (openmx_env_flag_at(OPENMX_ENV_CLUSTER_GPU_DIAG, -1)==0 && !openmx_mgpu_host_required()) return 0;
    if (openmx_mgpu_host_required()) goto host_capacity;

    if (state->diag_verdict!=-1 && state->diag_verdict_n==n && 1<SCF_iter){
        int any_released = 0;
        if (state->diag_verdict==0) return state->diag_verdict;
        MPI_Allreduce(&state->gpusolver_ctx.scratch_released,&any_released,
                      1,MPI_INT,MPI_MAX,mpi_comm_level1);
        if (!any_released) return state->diag_verdict;
        /* Memory admission must be repeated after a trim. Preserve the
           previous serialized spin layout rather than allocating both
           owners merely to discover again that only one can fit. */
        if (state->diag_verdict==2) serial_knob = 2;
    }

    /* the serialized tier only helps when two spin worlds share the device,
       and it rebuilds the second spin's dense Hamiltonian from the packed
       Set_Hamiltonian cache on spin world 0's owner */
    serial_allowed = (serial_knob!=0 && SpinP_switch==1 && numprocs0!=1 &&
                      Set_Hamiltonian_GpuSolver_Packed_CacheReady() &&
                      Set_Hamiltonian_GpuSolver_Packed_OrderMode()==0);

    if (serial_knob==2 && serial_allowed){
        /* testing aid: skip the concurrent probe and go straight to the
           serialized tier */
    }
    else{
        if (owner){
            my_fit = ClusterCol_OwnerReserveProbe(state, n,myworld1,
                                                  serial_allowed ? " concurrently" : "");
        }
        MPI_Allreduce(&my_fit,&fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);
    }

    if (fit==1){
        state->diag_verdict = 1;
    }
    else if (serial_allowed){
        int my_serial = 1;
        int serial_fit = 0;

        /* only spin world 0's owner keeps a reservation in the serialized
           mode: release the other owner's buffers first so the probe below
           sees that memory, and order the release before the probe */
        if (owner && !solver_owner) ClusterCol_GpuSolver_Destroy(state);
        MPI_Barrier(mpi_comm_level1);

        if (solver_owner){
            my_serial = ClusterCol_OwnerReserveProbe(state, n,myworld1," for the serialized mode");
        }
        MPI_Allreduce(&my_serial,&serial_fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);

        if (serial_fit==1){
            state->diag_verdict = 2;
            if (announce && 2<=level_stdout){
                printf("<Cluster_DFT_Col> The device cannot hold both spin owners' GPU buffers at once;"
                       " spin world 0's owner solves both spins serially on the GPU"
                       " (disable with OPENMX_CLUSTER_GPU_DIAG_SERIAL=0).\n");
                fflush(stdout);
            }
        }
        else{
            state->diag_verdict = 0;
        }
    }
    else{
        state->diag_verdict = 0;
    }

    if (state->diag_verdict==0){
host_capacity:
        ClusterCol_GpuSolver_Destroy(state);
        openmx_gemmul8ReleaseWorkspaces();
        acc_wait_all(); acc_clear_freelists();
        MPI_Barrier(mpi_comm_level1);
        /* rank 0 (the announce rank) plans node-local MAGMA and, in a
           multi-node run, falls back to the distributed gpusolver2 path;
           the choice is broadcast before any rank commits to a path */
        const int capacity = openmx_cluster_capacity_select(mpi_comm_level1,n,0,"Cluster collinear");
        state->mgpu_host_dense=0;
        state->gs2_fallback_n=0;
        state->diag_verdict=0;
        if (capacity==OPENMX_CLUSTER_CAPACITY_GS2){
            int admitted;
            /* verdict 0 sends all ranks into the ScaLAPACK layout below,
               whose gpusolver2 branches then run (ClusterCol_UseGs2).  A
               spin-polarized run solves both spins in turn on a grid over
               all ranks instead of each spin on its own world's grid (the
               worlds are contiguous halves of the ranks, one node each). */
            if (SpinP_switch==1 && numprocs0>1){
                ClusterCol_UnionGrid_Ensure(state, n);
                state->ugrid.overlap_n = 0;
                admitted = openmx_cluster_gs2_admit(mpi_comm_level1,state->ugrid.ictxt,n,state->ugrid.nblk,0,
                                         "Cluster collinear");
            }
            else{
                admitted = openmx_cluster_gs2_admit(mpi_comm_level1,ictxt1,n,nblk,0,"Cluster collinear");
            }
            if (admitted) state->gs2_fallback_n = n;
        }
        else if (capacity==OPENMX_CLUSTER_CAPACITY_MAGMA){
            if (announce) {
                printf("<OpenMX_MGPU> Cluster collinear n=%d: %s MAGMA, host-backed dense matrices; scf.gpu.num upper bound respected.\n",
                       n,openmx_mgpu_host_required()?"forced":"cuSOLVER memory limit ->");
                fflush(stdout);
            }
            state->mgpu_host_dense=1; state->diag_verdict=1;
        }
    }

    /* the union grid lives only while the fallback of a spin-polarized run
       needs it (every rank reaches this point with the same verdict) */
    if (!(state->diag_verdict==0 && state->gs2_fallback_n==n)) ClusterCol_UnionGrid_Free(state);

    state->diag_verdict_n = n;
    return state->diag_verdict;
}

static void ClusterCol_SetMaxNAndPartitions(int SCF_iter, const char *mode, double TZ, int n, int numprocs1,
                                            int *MaxN, int *is2, int *ie2)
{
    int ID;
    double av_num;
    double lumos;

    if (SCF_iter==1){
        *MaxN = n;
    }
    else{
        if      (strcasecmp(mode,"scf")==0)     lumos = (double)n*0.100;
        else if (strcasecmp(mode,"dos")==0)     lumos = (double)n*0.200;
        else if (strcasecmp(mode,"lcaoout")==0) lumos = (double)n*0.200;
        else if (strcasecmp(mode,"xanes")==0)   lumos = (double)n*0.200;
        else if (strcasecmp(mode,"diag")==0)    lumos = (double)n*0.200;
        else                                    lumos = (double)n*0.200;

        if (lumos<400.0) lumos = 400.0;
        *MaxN = (int)((TZ-system_charge)/2.0) + (int)lumos;
        if (n<*MaxN) *MaxN = n;
        if (cal_partial_charge) *MaxN = n;
    }

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

/* the state partition of a world of nw ranks (as in
   ClusterCol_SetMaxNAndPartitions) */
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

/* Collective over mpi_comm_level1: the rows 1..MaxN (states) of the
   back-transformed matrix g->Hs = (X C)^T on the union grid go to the ranks
   of world `spin`, each receiving its states is2..ie2 for all basis
   functions in the layout of the ScaLAPACK path. */
static void ClusterCol_UnionSendEvec(ClusterColState *state, int spin, int n, int MaxN, int myworld1, int myid1,
                                     int *is2, int *ie2, int *NPROCS_WD1,
                                     int *Comm_World_StartID1, double **EVec1)
{
    const ClusterColUnionGrid *g = &state->ugrid;
    const int nw = NPROCS_WD1[spin];
    const int start = Comm_World_StartID1[spin];
    int numprocs0, ID, i, j, k;
    int *is2s, *ie2s, *owner, *scnt, *sdsp, *rcnt, *rdsp, *iscnt, *isdsp, *ircnt, *irdsp;
    int *idx_snd, *idx_rcv;
    double *val_snd, *val_rcv;
    long long nsnd = 0, nrcv = 0;

    MPI_Comm_size(mpi_comm_level1, &numprocs0);

    is2s = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)nw,0);
    ie2s = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)nw,0);
    owner = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)(MaxN + 1),0);
    scnt = (int *)openmx_core_native_malloc(state->resources,(size_t)numprocs0,sizeof(int),1);
    sdsp = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)numprocs0,0);
    rcnt = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)numprocs0,0);
    rdsp = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)numprocs0,0);
    iscnt = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)numprocs0,0);
    isdsp = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)numprocs0,0);
    ircnt = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)numprocs0,0);
    irdsp = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)numprocs0,0);
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

    idx_snd = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)(2*nsnd + 1),0);
    val_snd = (double *)openmx_core_native_malloc(state->resources,1,sizeof(double) * (size_t)(nsnd + 1),0);
    idx_rcv = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)(2*nrcv + 1),0);
    val_rcv = (double *)openmx_core_native_malloc(state->resources,1,sizeof(double) * (size_t)(nrcv + 1),0);
    if (!idx_snd || !val_snd || !idx_rcv || !val_rcv){
        fprintf(stderr, "Cluster_DFT_Col: out of host memory in the eigenvector distribution\n");
        MPI_Abort(mpi_comm_level1, 1);
    }

    /* pack in destination order: rows are visited per destination */
    {
        int *pos = (int *)openmx_core_native_malloc(state->resources,1,sizeof(int) * (size_t)numprocs0,0);
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
        openmx_core_native_free(state->resources,pos);
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

    openmx_core_native_free(state->resources,idx_snd); openmx_core_native_free(state->resources,val_snd); openmx_core_native_free(state->resources,idx_rcv); openmx_core_native_free(state->resources,val_rcv);
    openmx_core_native_free(state->resources,is2s); openmx_core_native_free(state->resources,ie2s); openmx_core_native_free(state->resources,owner);
    openmx_core_native_free(state->resources,scnt); openmx_core_native_free(state->resources,sdsp); openmx_core_native_free(state->resources,rcnt); openmx_core_native_free(state->resources,rdsp);
    openmx_core_native_free(state->resources,iscnt); openmx_core_native_free(state->resources,isdsp); openmx_core_native_free(state->resources,ircnt); openmx_core_native_free(state->resources,irdsp);
}

/* The distributed capacity fallback of a spin-polarized run: the overlap
   transform once per cycle and both spins in turn on the union grid (ELPA
   GPU kernels + COSMA through the gpusolver2 bridge, i.e. every rank and
   GPU of the run), with the same transforms as the ScaLAPACK path.  Every
   rank ends with ko[0] and ko[1]; the ranks of world s with EVec1[s]. */
static void ClusterCol_UnionGs2Path(ClusterColState *state, int n, int MaxN, double **ko, double *****nh, double ****CntOLP,
                                    int *MP, int myworld1, int myid1, int *is2, int *ie2,
                                    int *NPROCS_WD1, int *Comm_World_StartID1, double **EVec1)
{
    ClusterColUnionGrid *g = &state->ugrid;
    ClusterColGridGlobals saved;
    double alpha = 1.0, beta = 0.0;
    int ONE = 1, spin, i, j, l, info;

    if (g->overlap_n != n){
        double *w = (double *)openmx_core_native_malloc(state->resources,1,sizeof(double) * (size_t)(n + 1),0);

        if (w == NULL){
            fprintf(stderr, "Cluster_DFT_Col: out of host memory (overlap eigenvalues)\n");
            MPI_Abort(mpi_comm_level1, 1);
        }
        saved = ClusterCol_EnterUnionGridGlobals(state);
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
        openmx_core_native_free(state->resources,w);
        g->overlap_n = n;
    }

    for (spin=0; spin<=1; spin++){

        saved = ClusterCol_EnterUnionGridGlobals(state);
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

        ClusterCol_UnionSendEvec(state, spin, n, MaxN, myworld1, myid1, is2, ie2, NPROCS_WD1,
                                 Comm_World_StartID1, EVec1);
    }
}

static void ClusterCol_DistributeDenseEvec(ClusterColState *state, int n, int maxn, int myid1, int numprocs1, int *is2, int *ie2,
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
            send_buf = (double*)ClusterCol_MallocArray(state, (size_t)max_count,sizeof(double),
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

        openmx_core_native_free(state->resources,send_buf);
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

/* OPENMX_GEMM_SAMPLE_DIR=<existing directory> writes the inputs of the two
   forward GEMMs, B = H X and X^T B, as raw column-major doubles: the
   transformed overlap X whenever it is rebuilt (X_g<geometry>.bin) and the
   Hamiltonian of the SCF steps listed in OPENMX_GEMM_SAMPLE_ITERS (comma
   separated, default "1"; H_g<geometry>_scf<step>_s<spin>.bin).  The files
   feed the offline precision and reuse study tests/gemmul8_reuse_probe.cu. */

static void ClusterCol_SampleDeviceMatrix(ClusterColState *state, const double *d_A, int n, const char *name)
{
    ClusterColGpuSolverCtx *ctx = &state->gpusolver_ctx;
    size_t count = ClusterCol_CheckedMulCount((size_t)n,(size_t)n,"sampled matrix");
    double *A = (double*)ClusterCol_MallocArray(state, count,sizeof(double),"sampled matrix");
    char path[4096];
    FILE *fp;

    snprintf(path,sizeof(path),"%s/%s",openmx_env_at(OPENMX_ENV_GEMM_SAMPLE_DIR),name);
    wait_cudafunc(cudaMemcpyAsync(A,d_A,count*sizeof(double),cudaMemcpyDeviceToHost,ctx->stream));
    wait_cudafunc(cudaStreamSynchronize(ctx->stream));

    fp = fopen(path,"wb");
    if (fp==NULL || fwrite(A,sizeof(double),count,fp)!=count || fclose(fp)!=0){
        char msg[4200];
        snprintf(msg,sizeof(msg),"Cluster_DFT_Col.c: could not write the sampled matrix %s",path);
        ClusterCol_AbortWithMessage(msg);
    }
    printf("<Cluster_DFT_Col> sampled %s (n=%d)\n",path,n);
    openmx_core_native_free(state->resources,A);
}

static void ClusterCol_SampleTransformedS(ClusterColState *state, int n, int myworld1)
{
    const char *dir = openmx_env_at(OPENMX_ENV_GEMM_SAMPLE_DIR);
    char name[64];

    state->sample_geometry++;
    /* both spin worlds hold the same X; one copy is enough */
    if (dir==NULL || dir[0]=='\0' || myworld1!=0) return;

    snprintf(name,sizeof(name),"X_g%d.bin",state->sample_geometry);
    ClusterCol_SampleDeviceMatrix(state, state->gpusolver_ctx.d_S,n,name);
}

static void ClusterCol_SampleHamiltonian(ClusterColState *state, int SCF_iter, int spin, int n)
{
    const char *dir = openmx_env_at(OPENMX_ENV_GEMM_SAMPLE_DIR);
    const char *list = openmx_env_at(OPENMX_ENV_GEMM_SAMPLE_ITERS);
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

    snprintf(name,sizeof(name),"H_g%d_scf%03d_s%d.bin",state->sample_geometry,SCF_iter,spin);
    ClusterCol_SampleDeviceMatrix(state, state->gpusolver_ctx.d_H,n,name);
}

static void ClusterCol_MagmaRootDensePath(ClusterColState *state, int SpinP_switch,double **ko,double *****nh,double ****CntOLP,
    int numprocs0,int myid0,int myworld1,int numprocs1,int myid1,MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,int *MP,int *is2,int *ie2,int n,int MaxN,double **EVec1)
{
    const int root=(myid0==Host_ID);
    const size_t nn=(size_t)n*n, panel=(size_t)n*MaxN;
    double *s=NULL,*h=NULL,*tmp=NULL,*c=NULL;
    ClusterCol_InvalidateDeviceEvecStash(state);
    state->refine.last_refined=0;
    if (root) {
        s=ClusterCol_MallocArray(state, nn,sizeof(double),"MAGMA overlap");
        h=ClusterCol_MallocArray(state, nn,sizeof(double),"MAGMA Hamiltonian");
        tmp=ClusterCol_MallocArray(state, nn,sizeof(double),"MAGMA transform");
    }
    if (root || myid1==0) c=ClusterCol_MallocArray(state, panel,sizeof(double),"MAGMA eigenvectors");
    Patch2Host_Cluster_Owner(state, CntOLP,MP,root,n,s);
    if (root) {
        if (!openmx_mgpu_host_solve(n,n,0,s,n,ko[0]+1)) ClusterCol_AbortWithMessage(openmx_mgpu_host_error());
#pragma omp parallel for
        for (int j=0;j<n;++j) {
            const double scale=1.0/sqrt(fmax(ko[0][j+1],1.0e-10));
            for (int i=0;i<n;++i) s[(size_t)j*n+i]*=scale;
        }
    }
    for (int spin=0;spin<=SpinP_switch;++spin) {
        Patch2Host_Cluster_Owner(state, nh[spin],MP,root,n,h);
        if (root) {
            if (!openmx_mgpu_host_dgemm(CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,h,n,s,n,tmp,n) ||
                !openmx_mgpu_host_dgemm(CUBLAS_OP_T,CUBLAS_OP_N,n,n,n,s,n,tmp,n,h,n) ||
                !openmx_mgpu_host_solve(n,MaxN,0,h,n,ko[spin]+1) ||
                !openmx_mgpu_host_dgemm(CUBLAS_OP_T,CUBLAS_OP_T,MaxN,n,n,h,n,s,n,c,MaxN))
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
            ClusterCol_DistributeDenseEvec(state, n,MaxN,myid1,numprocs1,is2,ie2,MPI_CommWD1[world],c,EVec1[spin]);
    }
    openmx_core_native_free(state->resources,s); openmx_core_native_free(state->resources,h); openmx_core_native_free(state->resources,tmp); openmx_core_native_free(state->resources,c);
}

static void ClusterCol_GpuSolverRootDensePath(ClusterColState *state, int SCF_iter, int SpinP_switch, double **ko,
                                             double *****nh, double ****CntOLP,
                                             int numprocs0, int myid0, int myworld1,
                                             int numprocs1, int myid1, MPI_Comm *MPI_CommWD1,
                                             int *Comm_World_StartID1, int gpu_mode,
                                             int *MP, int *is2, int *ie2,
                                             int n, int MaxN, double **EVec1)
{
    if (state->mgpu_host_dense) {
        ClusterCol_MagmaRootDensePath(state, SpinP_switch,ko,nh,CntOLP,numprocs0,myid0,myworld1,numprocs1,myid1,
            MPI_CommWD1,Comm_World_StartID1,MP,is2,ie2,n,MaxN,EVec1);
        return;
    }
    int owns_dense = (myid1==0);
    /* gpu_mode 2: spin world 0's owner holds the only device reservation
       and solves both spins back to back (see ClusterCol_GpuDiagFits) */
    int serialized = (gpu_mode==2 && SpinP_switch==1 && numprocs0!=1);
    int solves_dense = owns_dense && (!serialized || myworld1==0);
    int spin_start = 0;
    int spin_end = SpinP_switch;
    int rebuild_s = (SCF_iter==1);
    int use_setham_packed_cache =
        (Set_Hamiltonian_GpuSolver_Packed_CacheReady() &&
         Set_Hamiltonian_GpuSolver_Packed_OrderMode() == 0);
    double *C = NULL;

    /* the stash refers to the previous SCF step's eigenvectors until the
       solves below refresh it */
    ClusterCol_InvalidateDeviceEvecStash(state);

    /* the refined eigensolver: one spin, or two spin worlds whose owners
       exchange their estimates; a single rank solving both spins and the
       serialized mode keep FP64 */
    {
        int spin_mode = 0, peer = -1;

        if (SpinP_switch==1){
            if (serialized || numprocs0==1) spin_mode = 2;
            else {
                spin_mode = 1;
                peer = Comm_World_StartID1[1-myworld1];
            }
        }
        ClusterCol_RefineBeginSolve(state, SCF_iter,spin_mode,peer);
    }

    if (SpinP_switch==1 && numprocs0!=1){
        spin_start = myworld1;
        spin_end = myworld1;
    }

    if (serialized && !use_setham_packed_cache){
        ClusterCol_AbortWithMessage("Cluster_DFT_Col.c: the serialized GPU diagonalization requires the Set_Hamiltonian packed cache.");
    }

    if (use_setham_packed_cache) {
        Set_Hamiltonian_GpuSolver_SetMP(MP);
    }

    if (owns_dense){
        size_t nmax = ClusterCol_CheckedMulCount((size_t)n,(size_t)MaxN,"root dense eigenvectors");

        C = (double*)ClusterCol_MallocArray(state, nmax,sizeof(double),"root dense eigenvectors");

        if (solves_dense){
            ClusterCol_GpuSolver_EnsureMatrixCapacity(state, n);

            if (rebuild_s || !(state->gpusolver_ctx.transformed_s_valid &&
                               state->gpusolver_ctx.transformed_s_dim==n)){
                rebuild_s = 1;
            }
        }
        else{
            /* serialized mode: this spin world's owner only receives and
               distributes; it holds no solver context to rebuild */
            rebuild_s = 0;
        }
    }

    MPI_Bcast(&rebuild_s,1,MPI_INT,0,MPI_CommWD1[myworld1]);

    if (rebuild_s){
        if (use_setham_packed_cache) {
            if (owns_dense) {
                int tnum = Set_Hamiltonian_GpuSolver_Packed_Size();
                int *cache_order_GA = Set_Hamiltonian_GpuSolver_Packed_OrderGA();
                double *cache_S = Set_Hamiltonian_GpuSolver_Packed_Overlap();
                const int *dense_index;

                if (!Set_Hamiltonian_GpuSolver_Packed_OwnsCache() || cache_order_GA == NULL || cache_S == NULL) {
                    ClusterCol_AbortWithMessage("Set_Hamiltonian packed overlap cache is missing in Cluster_DFT_Col.c.");
                }
                dense_index = ClusterCol_DenseIndexCache_Get(state, cache_order_GA,MP,n,tnum);
                ClusterCol_BuildDeviceDenseFromPacked(cache_S,dense_index,tnum,n,state->gpusolver_ctx.d_S);
            }
        }
        else {
            Patch2Device_Cluster_Owner(state, CntOLP,MP,owns_dense,n,
                                       owns_dense ? state->gpusolver_ctx.d_S : NULL);
        }
        if (owns_dense){
            ClusterCol_GpuSolver_PrepareTransformedSDevice(state, 1,n,ko[0]);
            ClusterCol_SampleTransformedS(state, n,myworld1);
        }
    }

    if (!serialized){
        for (int spin=spin_start; spin<=spin_end; spin++){
            if (use_setham_packed_cache) {
                if (owns_dense) {
                    int tnum = Set_Hamiltonian_GpuSolver_Packed_Size();
                    int *cache_order_GA = Set_Hamiltonian_GpuSolver_Packed_OrderGA();
                    double *cache_H = Set_Hamiltonian_GpuSolver_Packed_H(spin);
                    const int *dense_index;

                    if (!Set_Hamiltonian_GpuSolver_Packed_OwnsCache() || cache_order_GA == NULL || cache_H == NULL) {
                        ClusterCol_AbortWithMessage("Set_Hamiltonian packed Hamiltonian cache is missing in Cluster_DFT_Col.c.");
                    }
                    dense_index = ClusterCol_DenseIndexCache_Get(state, cache_order_GA,MP,n,tnum);
                    ClusterCol_BuildDeviceDenseFromPacked(cache_H,dense_index,tnum,n,state->gpusolver_ctx.d_H);
                }
            }
            else {
                Patch2Device_Cluster_Owner(state, nh[spin],MP,owns_dense,n,
                                           owns_dense ? state->gpusolver_ctx.d_H : NULL);
            }
            if (owns_dense){
                ClusterCol_SampleHamiltonian(state, SCF_iter,spin,n);
                ClusterCol_GpuSolver_SolveHamiltonianDevice(state, n,MaxN,ko[spin],C,spin);
                ClusterCol_StashDeviceEvec(state, spin,n,MaxN);
            }
            {
                /* OPENMX_BAND_PROFILE=1: the time of the eigenvector distribution */
                double dist_t0 = 0.0, dist_t1 = 0.0;
                const int dist_prof = ClusterCol_DMProfileEnabled();

                if (dist_prof) dtime(&dist_t0);
                ClusterCol_DistributeDenseEvec(state, n,MaxN,myid1,numprocs1,is2,ie2,
                                               MPI_CommWD1[myworld1],
                                               owns_dense ? C : NULL,EVec1[spin]);
                if (dist_prof){
                    dtime(&dist_t1);
                    fprintf(stderr, "CLUSTERDISTPROF id=%d spin=%d owner=%d dist=%.3f\n", myid1, spin, owns_dense,
                            dist_t1-dist_t0);
                    fflush(stderr);
                }
            }
        }
    }
    else{
        /* Serialized mode: spin world 0's owner solves both spins with its
           single reservation, using its own packed Set_Hamiltonian cache
           (the selected ranks own every spin of it).  The remote spin is
           solved first so spin world 1 can distribute its eigenvectors
           while spin 0 is still solving.  The device eigenvector stash is
           only kept for spin 0 — spin 1's density matrix is accumulated in
           spin world 1, whose owner has no device copy. */
        const int remote_owner = Comm_World_StartID1[1];
        const int home_owner = Comm_World_StartID1[0];
        const int ko_tag = 997;
        const int evec_tag = 998;
        const size_t evec_count = ClusterCol_CheckedMulCount((size_t)n,(size_t)MaxN,
                                                             "serialized eigenvector panel");
        const size_t chunk_limit = 64ULL*1024ULL*1024ULL;   /* doubles per message */

        if (myworld1==0){
            if (owns_dense){
                for (int spin=1; 0<=spin; spin--){
                    int tnum = Set_Hamiltonian_GpuSolver_Packed_Size();
                    int *cache_order_GA = Set_Hamiltonian_GpuSolver_Packed_OrderGA();
                    double *cache_H = Set_Hamiltonian_GpuSolver_Packed_H(spin);
                    const int *dense_index;

                    if (!Set_Hamiltonian_GpuSolver_Packed_OwnsCache() || cache_order_GA == NULL || cache_H == NULL) {
                        ClusterCol_AbortWithMessage("Set_Hamiltonian packed Hamiltonian cache is missing in Cluster_DFT_Col.c.");
                    }
                    dense_index = ClusterCol_DenseIndexCache_Get(state, cache_order_GA,MP,n,tnum);
                    ClusterCol_BuildDeviceDenseFromPacked(cache_H,dense_index,tnum,n,state->gpusolver_ctx.d_H);
                    ClusterCol_SampleHamiltonian(state, SCF_iter,spin,n);
                    ClusterCol_GpuSolver_SolveHamiltonianDevice(state, n,MaxN,ko[spin],C,-1);

                    if (spin==1){
                        MPI_Send(&ko[1][1],MaxN,MPI_DOUBLE,remote_owner,ko_tag,mpi_comm_level1);
                        for (size_t off=0; off<evec_count; off+=chunk_limit){
                            size_t len = evec_count - off;
                            if (chunk_limit<len) len = chunk_limit;
                            MPI_Send(C+off,(int)len,MPI_DOUBLE,remote_owner,evec_tag,mpi_comm_level1);
                        }
                    }
                    else{
                        ClusterCol_StashDeviceEvec(state, 0,n,MaxN);
                    }
                }
            }
            ClusterCol_DistributeDenseEvec(state, n,MaxN,myid1,numprocs1,is2,ie2,
                                           MPI_CommWD1[myworld1],
                                           owns_dense ? C : NULL,EVec1[0]);
        }
        else{
            if (owns_dense){
                MPI_Status serial_stat;

                MPI_Recv(&ko[1][1],MaxN,MPI_DOUBLE,home_owner,ko_tag,mpi_comm_level1,&serial_stat);
                for (size_t off=0; off<evec_count; off+=chunk_limit){
                    size_t len = evec_count - off;
                    if (chunk_limit<len) len = chunk_limit;
                    MPI_Recv(C+off,(int)len,MPI_DOUBLE,home_owner,evec_tag,mpi_comm_level1,&serial_stat);
                }
            }
            ClusterCol_DistributeDenseEvec(state, n,MaxN,myid1,numprocs1,is2,ie2,
                                           MPI_CommWD1[myworld1],
                                           owns_dense ? C : NULL,EVec1[1]);
        }
    }

    /* free the GEMMul8 workspace so the grid/force phases of this SCF step
       see the memory; the transformed overlap stays cached on the device. */
    if (owns_dense && ClusterCol_GemmWorkspaceTurnRelease()){
        openmx_gemmul8ReleaseWorkspaces();
    }

    ClusterCol_RefineEndSolve(state);
    openmx_core_native_free(state->resources,C);

    (void)myid0;
}

void solve_evp_real_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, int *nblk,
                      int *mpi_comm_rows_int, int *mpi_comm_cols_int);

void elpa_solve_evp_real_2stage_double_impl_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2,
                                              int *nblk, int *na_cols1, int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld);

static double Lapack_LU_Dinverse(ClusterColState *state, int n, double *A);
static double Calc_Oscillator_Strength(ClusterColState *state,  int n, int UMOmax, int Nocc[2], int *MP,
                                        int **ind2ind, 
					double **Z0, double **A, double **Z, 
					double **Ax, double **Ay, double **Az,
					double Utot1, double Utot2,
					double XANES_Res[10] );

static void Calc_XANES_Col(ClusterColState *state, const OpenMXAtomicViews *atomic,  int n, int MaxN, int myid1,
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
			    double *Work1, const OpenMXElectronicViews *electronic);

static void Save_LCAO_Col(ClusterColState *state,  int n, int MaxN, int myid1, int *is2, int *ie2,
			   int *MP, double ****OLP0, double **EVec1, 
			   double **ko, int *NPROCS_WD1, int *Comm_World_StartID1 );

static void Save_DOS_Col(ClusterColState *state,  int n, int MaxN, int myid1, int *is2, int *ie2,
			  int *MP, double *****nh, double ****OLP0, double **EVec1, 
			  double **ko, int *NPROCS_WD1, int *Comm_World_StartID1 );

static double Calc_DM_Cluster_collinear(ClusterColState *state, int myid0,
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
					int *SP_Atoms , const OpenMXElectronicViews *electronic);

static double Calc_ParDM_Cluster_collinear(ClusterColState *state,
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

static double Calc_KS_orbitals_on_grid_single(ClusterColState *state,
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

static double Calc_HF_Vx_single(ClusterColState *state,
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

static double Exchange_Poisson(ClusterColState *state,  double *work1_on_grid, double *work2_on_grid );

typedef struct {
  ClusterColState *state;
  const OpenMXAtomicViews *atomic;
  char *mode;
  int SCF_iter;
  int SpinP_switch;
  double **ko;
  double *****nh;
  double ****CntOLP;
  double *****CDM;
  double *****EDM;
  double *Eele0;
  double *Eele1;
  int myworld1;
  int *NPROCS_ID1;
  int *Comm_World1;
  int *NPROCS_WD1;
  int *Comm_World_StartID1;
  MPI_Comm *MPI_CommWD1;
  int *MP;
  int *is2;
  int *ie2;
  double *Ss;
  double *Cs;
  double *Hs;
  double *CDM1;
  double *EDM1;
  double *PDM1;
  int size_H1;
  int *SP_NZeros;
  int *SP_Atoms;
  double **EVec1;
  double *Work1;
  const OpenMXElectronicViews *electronic;
} Cluster_DFT_Col_arguments;
static double Cluster_DFT_Col_owned(ClusterColState *state, const OpenMXAtomicViews *atomic,
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
                   double *Work1, const OpenMXElectronicViews *electronic, OpenMXSolverInvocation *solver_owner);
static int Cluster_DFT_Col_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Cluster_DFT_Col_arguments *a=(Cluster_DFT_Col_arguments*)arguments;
  *result=Cluster_DFT_Col_owned(a->state, a->atomic,a->mode,a->SCF_iter,a->SpinP_switch,a->ko,a->nh,a->CntOLP,a->CDM,a->EDM,a->Eele0,a->Eele1,a->myworld1,a->NPROCS_ID1,a->Comm_World1,a->NPROCS_WD1,a->Comm_World_StartID1,a->MPI_CommWD1,a->MP,a->is2,a->ie2,a->Ss,a->Cs,a->Hs,a->CDM1,a->EDM1,a->PDM1,a->size_H1,a->SP_NZeros,a->SP_Atoms,a->EVec1,a->Work1,a->electronic,owner);
  return 1;
}
double Cluster_DFT_Col(ClusterColState *state, const OpenMXAtomicViews *atomic,
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
                   double *Work1, const OpenMXElectronicViews *electronic)
{
  Cluster_DFT_Col_arguments arguments={state,atomic,mode,SCF_iter,SpinP_switch,ko,nh,CntOLP,CDM,EDM,Eele0,Eele1,myworld1,NPROCS_ID1,Comm_World1,NPROCS_WD1,Comm_World_StartID1,MPI_CommWD1,MP,is2,ie2,Ss,Cs,Hs,CDM1,EDM1,PDM1,size_H1,SP_NZeros,SP_Atoms,EVec1,Work1,electronic};
  return OpenMX_Solver_Run(Cluster_DFT_Col_operation,&arguments);
}

static double Cluster_DFT_Col_owned(ClusterColState *state, const OpenMXAtomicViews *atomic,
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
                   double *Work1, const OpenMXElectronicViews *electronic, OpenMXSolverInvocation *solver_owner)
{

  double *****ParDM=electronic->ParDM;

  int mgpu_scope_previous = openmx_mgpu_scope_enter(
      scf_eigen_lib_flag == GPUSOLVER && !gpusolver2_flag);

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

  if (SCF_iter==1) state->cpu_overlap_n = 0;

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n = n + atomic->species.contracted_orbitals[wanA];
  }
  n2 = n + 2;

  /* GPU dispatch (added by H.Kawai): assign CUDA/OpenACC device when GPUSOLVER is requested */
  if (scf_eigen_lib_flag == GPUSOLVER && n >= Cluster_DFT_Col_GpuSwitchNum() &&
      Set_Hamiltonian_OpenACC_Rank_Is_Selected()) {
      set_cuda_default_device_from_local_rank_noncollective();
      set_openacc_nvidia_device_from_local_rank_noncollective();
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

  is1=OpenMX_Solver_Int(solver_owner,"is1",((size_t)1)*numprocs1);
  ie1=OpenMX_Solver_Int(solver_owner,"ie1",((size_t)1)*numprocs1);

  Num_Snd_EV=OpenMX_Solver_Int(solver_owner,"Num_Snd_EV",((size_t)1)*numprocs1);
  Num_Rcv_EV=OpenMX_Solver_Int(solver_owner,"Num_Rcv_EV",((size_t)1)*numprocs1);

  if (measure_time){
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time4 = 0.0;
    time5 = 0.0;
    time6 = 0.0;
    time7 = 0.0;
    time8 = 0.0;
  }

  if      (SpinP_switch==0) spin_degeneracy = 2.0;
  else if (SpinP_switch==1) spin_degeneracy = 1.0;

  /****************************************************
                   total core charge
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ = TZ + atomic->species.core_charge[wan];
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

  {
    int gpu_diag_mode = 0;

    /* a solve outside the root dense path is an FP64 one; only SCF solves
       are refined */
    state->refine.last_refined = 0;
    state->refine.scf_mode = (strcasecmp(mode,"scf")==0);
    if (scf_eigen_lib_flag==GPUSOLVER && gpusolver2_flag==0 && (openmx_mgpu_host_required() || Cluster_DFT_Col_GpuSwitchNum()<=n)){
      gpu_diag_mode = ClusterCol_GpuDiagFits(state, SCF_iter,n,myworld1,myid1,numprocs0);
    }
    if (gpu_diag_mode!=0){
      ClusterCol_SetMaxNAndPartitions(SCF_iter,mode,TZ,n,numprocs1, &MaxN,is2,ie2);
      state->firsttime = 0;
      ClusterCol_GpuSolverRootDensePath(state, SCF_iter,SpinP_switch,ko,nh,CntOLP,
                                       numprocs0,myid0,myworld1,numprocs1,myid1,
                                       MPI_CommWD1,Comm_World_StartID1,gpu_diag_mode,
                                       MP,is2,ie2,n,MaxN,EVec1);
      goto diagonalize_finished;
    }
    /* the distributed capacity fallback of a spin-polarized run: both spins
       on the grid over all ranks (see ClusterCol_UnionGrid_Ensure) */
    if (!gpusolver2_flag && SpinP_switch==1 && numprocs0>1 &&
        state->gs2_fallback_n==n && state->ugrid.n==n){
      ClusterCol_SetMaxNAndPartitions(SCF_iter,mode,TZ,n,numprocs1, &MaxN,is2,ie2);
      state->firsttime = 0;
      ClusterCol_UnionGs2Path(state, n,MaxN,ko,nh,CntOLP,MP,myworld1,myid1,is2,ie2,
                              NPROCS_WD1,Comm_World_StartID1,EVec1);
      goto diagonalize_finished;
    }
  }

  /****************************************************
         1. diagonalize the overlap matrix     
         2. search negative eigenvalues
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  if (state->cpu_overlap_n!=n){
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

  if (state->cpu_overlap_n!=n){

    if (measure_time) dtime(&stime);

    MPI_Comm_split(MPI_CommWD1[myworld1],my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(MPI_CommWD1[myworld1],my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

    if (ClusterCol_UseGs2(state, n)){

      /* distributed multi-GPU eigensolver (ELPA, NVIDIA GPU kernels) */
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
    state->cpu_overlap_n = n;

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

  index_Snd_i=OpenMX_Solver_Int(solver_owner,"index_Snd_i",((size_t)1)*Max_Num_Snd_EV);
  index_Snd_j=OpenMX_Solver_Int(solver_owner,"index_Snd_j",((size_t)1)*Max_Num_Snd_EV);
  EVec_Snd=OpenMX_Solver_Real(solver_owner,"EVec_Snd",((size_t)1)*Max_Num_Snd_EV);
  index_Rcv_i=OpenMX_Solver_Int(solver_owner,"index_Rcv_i",((size_t)1)*Max_Num_Rcv_EV);
  index_Rcv_j=OpenMX_Solver_Int(solver_owner,"index_Rcv_j",((size_t)1)*Max_Num_Rcv_EV);
  EVec_Rcv=OpenMX_Solver_Real(solver_owner,"EVec_Rcv",((size_t)1)*Max_Num_Rcv_EV);

  /* for PrintMemory */

  if (state->firsttime && memoryusage_fileout){
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
  state->firsttime=0;

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
  if (ClusterCol_UseGs2(state, n))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

  /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

  for(i=0; i<na_rows*na_cols; i++){
    Hs[i] = 0.0;
  }

  Cblacs_barrier(ictxt1,"C");
  if (ClusterCol_UseGs2(state, n))
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

  if (ClusterCol_UseGs2(state, n)){

    /* distributed multi-GPU eigensolver (ELPA, NVIDIA GPU kernels); all n
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
  if (ClusterCol_UseGs2(state, n))
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

	  state->chemical->chemical_potential = 0.50*(ChemP_MAX + ChemP_MIN);
	  Num_State = 0.0;

	  for (i1=1; i1<=MaxN; i1++){
	    x = (ko[spin][i1] - state->chemical->chemical_potential)*Beta;
	    if (x<=-max_x) x = -max_x;
	    if (max_x<=x)  x = max_x;
	    FermiF = 1.0/(1.0 + exp(x));
	    Num_State += FermiF;
	  }

	  Dnum = HOMO_XANES[spin] - Num_State;
	  if (0.0<=Dnum) ChemP_MIN = state->chemical->chemical_potential;
	  else           ChemP_MAX = state->chemical->chemical_potential;
	  if (fabs(Dnum)<1.0e-12) po = 1;

	  if (myid1==Host_ID && 2<=level_stdout){
	    printf("spin=%2d ChemP=%15.12f HOMO_XANES=%2d Num_state=%15.12f\n",
                    spin,state->chemical->chemical_potential,HOMO_XANES[spin],Num_State);
	  }

	  loopN++;

	} while (po==0 && loopN<1000); 

        state->chemical->xanes[spin] = state->chemical->chemical_potential;
        Cluster_HOMO[spin] = HOMO_XANES[spin];

      } /* spin */

      /* set ChemP */

      state->chemical->chemical_potential = 0.5*(state->chemical->xanes[0] + state->chemical->xanes[1]);

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

	state->chemical->chemical_potential = 0.50*(ChemP_MAX + ChemP_MIN);
	Num_State = 0.0;

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i1=1; i1<=MaxN; i1++){
	    x = (ko[spin][i1] - state->chemical->chemical_potential)*Beta_trial1;
	    if (x<=-max_x) x = -max_x;
	    if (max_x<=x)  x = max_x;
	    FermiF = 1.0/(1.0 + exp(x));
	    Num_State = Num_State + spin_degeneracy*FermiF;
	    if (0.5<FermiF) Cluster_HOMO[spin] = i1;
	  }
	}

	Dnum = (TZ - Num_State) - system_charge;
	if (0.0<=Dnum) ChemP_MIN = state->chemical->chemical_potential;
	else           ChemP_MAX = state->chemical->chemical_potential;
	if (fabs(Dnum)<1.0e-12) po = 1;

	if (myid1==Host_ID && 2<=level_stdout){
	  printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",state->chemical->chemical_potential,TZ,Num_State);
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
	  state->chemical->chemical_potential = 0.50*(ChemP_MAX + ChemP_MIN);
	}

	Num_State = 0.0;

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i1=1; i1<=MaxN; i1++){
	    x = (ko[spin][i1] - state->chemical->chemical_potential)*Beta;
	    if (x<=-max_x) x = -max_x;
	    if (max_x<=x)  x = max_x;
	    FermiF = 1.0/(1.0 + exp(x));

	    Num_State = Num_State + spin_degeneracy*FermiF;
	    if (0.5<FermiF) Cluster_HOMO[spin] = i1;
	  }
	}

	Dnum = (TZ - Num_State) - system_charge;
	if (0.0<=Dnum) ChemP_MIN = state->chemical->chemical_potential;
	else           ChemP_MAX = state->chemical->chemical_potential;
	if (fabs(Dnum)<1.0e-12) po = 1;

	if (myid1==Host_ID && 2<=level_stdout){
	  printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",state->chemical->chemical_potential,TZ,Num_State);
	}

	loopN++;

      } 
      while (po==0 && loopN<1000); 

      if (2<=level_stdout){
	printf("  ChemP=%15.12f\n",state->chemical->chemical_potential);
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
          x = (ko[spin][i1] - state->chemical->xanes[spin])*Beta;
        else 
          x = (ko[spin][i1] - state->chemical->chemical_potential)*Beta;

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

    if (MO_fileout==1){  

      /* allocation of arrays */
      double *array0;    
      int *is3,*ie3;
      int numprocs3,ID1;

      array0=OpenMX_Solver_Real(solver_owner,"array0",((size_t)1)*(n+2));
      is3=OpenMX_Solver_Int(solver_owner,"is3",((size_t)1)*numprocs0);
      ie3=OpenMX_Solver_Int(solver_owner,"ie3",((size_t)1)*numprocs0);

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
	    tnoA = atomic->species.contracted_orbitals[wanA];
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
	    tnoA = atomic->species.contracted_orbitals[wanA];
	    Anum = MP[GA_AN];
	    for (i=0; i<tnoA; i++){
	      LUMOs_Coef[0][spin][j][GA_AN][i].r = array0[Anum+i-1];
	      LUMOs_Coef[0][spin][j][GA_AN][i].i = 0.0;
	    }
	  }
	}
      }

      /* freeing of array0 */

      ;
    }

    /****************************************************
          density matrix and energy density matrix
                  for up and down spins
    ****************************************************/

    time6 += Calc_DM_Cluster_collinear(state,  myid0,numprocs0,myid1,numprocs1,myworld1,
					size_H1,is2,ie2,MP,n,MPI_CommWD1,Comm_World_StartID1,
					CDM,EDM,ko,CDM1,EDM1,PDM1,Work1,EVec1,SP_NZeros,SP_Atoms, electronic);

    /****************************************************
                   calculation of ParDM
    ****************************************************/

    if ( strcasecmp(mode,"ParDM")==0 ){

      time6 += Calc_ParDM_Cluster_collinear(state,  myid0,numprocs0,myid1,numprocs1,myworld1,
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
	tnoA = atomic->species.contracted_orbitals[wanA];
	for (j=0; j<=FNAN[GA_AN]; j++){
	  wanB = WhatSpecies[natn[GA_AN][j]];
	  tnoB = atomic->species.contracted_orbitals[wanB];
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

	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",state->chemical->chemical_potential);
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
	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",state->chemical->chemical_potential);
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

      array0=OpenMX_Solver_Real(solver_owner,"array0",((size_t)1)*(n+2)*7);
      is3=OpenMX_Solver_Int(solver_owner,"is3",((size_t)1)*numprocs0);
      ie3=OpenMX_Solver_Int(solver_owner,"ie3",((size_t)1)*numprocs0);

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
		for (mul=0; mul<atomic->species.contracted_basis[(size_t)(wan1)*atomic->species.angular_channels+(l)]; mul++){
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

      ;

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
    Save_DOS_Col(state, n,MaxN,myid1,is2,ie2,MP,nh,CntOLP,EVec1,ko,NPROCS_WD1,Comm_World_StartID1);
  }

  else if ( strcasecmp(mode,"lcaoout")==0 ){
    Save_LCAO_Col(state, n,MaxN,myid1,is2,ie2,MP,CntOLP,EVec1,ko,NPROCS_WD1,Comm_World_StartID1);
  }

  else if ( strcasecmp(mode,"xanes")==0 ){
    Calc_XANES_Col(state, atomic,  n,MaxN,myid1,ko,nh,CntOLP,CDM,EDM,Eele0,Eele1,myworld1,NPROCS_ID1,
                    Comm_World1,NPROCS_WD1,Comm_World_StartID1,MPI_CommWD1,
                    MP,is2,ie2,Ss,Cs,Hs,CDM1,EDM1,PDM1,size_H1,SP_NZeros,SP_Atoms,EVec1,Work1, electronic);
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

  ;

  ;

  if (strcasecmp(mode,"scf")==0) ClusterCol_TrimScratch(state);

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
static void ClusterCol_AccumulateDM_CPU(ClusterColState *state, int size_H1, int nk, int *MP,
                                        const double *occ, const double *eig, const double *docc,
                                        const double *evec, double *DM1, double *EDM1, double *PDM1)
{
    size_t *atom_p;
    size_t pcount = 0;
    const int with_pdm = (docc!=NULL && PDM1!=NULL);

    /* Empty state partitions contribute only the caller's zeroed arrays. */
    if (nk<=0) return;

    atom_p = (size_t*)ClusterCol_MallocArray(state, (size_t)atomnum+1,sizeof(size_t),"CPU DM atom offsets");
    for (int GA_AN=1; GA_AN<=atomnum; GA_AN++){
        const int tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]];
        atom_p[GA_AN] = pcount;
        for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            const int GB_AN = natn[GA_AN][LB_AN];
            pcount += (size_t)tnoA*(size_t)Spe_Total_CNO[WhatSpecies[GB_AN]];
        }
    }
    if (pcount!=(size_t)size_H1){
        openmx_core_native_free(state->resources,atom_p);
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
    openmx_core_native_free(state->resources,atom_p);
}

double Calc_DM_Cluster_collinear(ClusterColState *state,
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
    int *SP_Atoms , const OpenMXElectronicViews *electronic)
{
  double *****Partial_DM=electronic->Partial_DM;

  int spin,i,j,i0,j0,k,kmin,kmax,po,p,GA_AN,MA_AN,wanA,tnoA,Anum;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j1,ID;
  double max_x=60.0,dum;
  double FermiF,FermiF2,x,x2,diffF,sum1,sum2;
  double FermiEps = 1.0e-13;
  double stime,etime,time,lumos,dm_compute_end;
  MPI_Status stat;
  MPI_Request request;
  double *FF,*dFF;

  dtime(&stime);

  kmin = is2[myid1];
  kmax = ie2[myid1];

  /* allocation of arrays */

  FF = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(n+1),0);
  dFF = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(n+1),0);

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
      x = (ko[spin][k] - state->chemical->xanes[spin])*Beta;
    else 
      x = (ko[spin][k] - state->chemical->chemical_potential)*Beta;

    if (x<=-max_x) x = -max_x;
    if (max_x<=x)  x = max_x;
    FermiF = FermiFunc(x,spin,k,&po,&x);

    FF[k] = FermiF;

    if (cal_partial_charge){

      if (xanes_calc==1) 
        x2 = (ko[spin][k] - (state->chemical->xanes[spin]+ene_win_partial_charge))*Beta;
      else 
        x2 = (ko[spin][k] - (state->chemical->chemical_potential+ene_win_partial_charge))*Beta;

      if (x2<=-max_x) x2 = -max_x;
      if (max_x<=x2)  x2 = max_x;
      FermiF2 = FermiFunc(x2,spin,k,&po,&x);
      diffF = fabs(FermiF-FermiF2);
      dFF[k] = diffF;
    }
  }

  /* calculation of DM1 */

  int gpu_dm_done = 0;

  int dm_requested=ClusterCol_UseGpuDM(n),dm_collective=dm_requested;
  /* Both spin worlds reach this phase together. A world confined to one
     node must still be able to borrow capacity from the other nodes. */
  MPI_Allreduce(MPI_IN_PLACE,&dm_collective,1,MPI_INT,MPI_MAX,mpi_comm_level1);
  if(dm_collective){
    int nk=dm_requested && kmin<=kmax?kmax-kmin+1:0;
    int stride=nk;const double *device=NULL;
    if(nk){ClusterCol_DMEntryCache_Ensure(state, MP,n);device=ClusterCol_DeviceEvecStash(state, spin,n,kmin,kmax,&stride);}
    ClusterColDMEntryCache *cache=&state->dm_entry_cache;
    double *outputs[10]={DM1,EDM1,cal_partial_charge?PDM1:NULL};
    /* The vote of every rank, then the direct device attempt; a rank whose
       attempt declines accumulates with ClusterCol_AccumulateDM_CPU below
       (its EVec1 is complete: the device stash only skips the round trip),
       without the collective capacity recovery. OPENMX_HOST_SIMD=legacy
       keeps that recovery. */
    if(!ClusterCol_DMOnHost(nk?cache->entry_count:0,nk,device?0:(size_t)n*(size_t)nk*sizeof(double))){
      if(OpenMX_HostSimdMode()!=OPENMX_HOST_SIMD_LEGACY)
        gpu_dm_done=nk>0 && !openmx_gpu_capacity_test_recovery() &&
          openmx_gpu_dm_try_entries(OPENMX_GPU_DM_REAL,n,nk,device?stride:nk,cache->entry_count,
            device?(const void*)device:(const void*)EVec1[spin],device==NULL,
            cache->basis0,cache->basis1,NULL,0,NULL,NULL,FF+kmin,ko[spin]+kmin,
            cal_partial_charge?dFF+kmin:NULL,outputs,1);
      else{
        openmx_gpu_dm_collective(mpi_comm_level1,OPENMX_GPU_DM_REAL,n,nk,device?stride:nk,
          nk?cache->entry_count:0,device?(const void*)device:(const void*)EVec1[spin],device==NULL,
          cache->basis0,cache->basis1,NULL,0,NULL,NULL,nk?FF+kmin:NULL,nk?ko[spin]+kmin:NULL,
          nk && cal_partial_charge?dFF+kmin:NULL,outputs,1);
        gpu_dm_done=nk>0;
      }
    }
  }

  if (!gpu_dm_done && kmin<=kmax){
    ClusterCol_AccumulateDM_CPU(state, size_H1,kmax-kmin+1,MP,FF+kmin,ko[spin]+kmin,
                               cal_partial_charge ? dFF+kmin : NULL,EVec1[spin],
                               DM1,EDM1,cal_partial_charge ? PDM1 : NULL);
  }

  dtime(&dm_compute_end);

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

  openmx_core_native_free(state->resources,dFF);
  openmx_core_native_free(state->resources,FF);

  dtime(&etime);
  if (ClusterCol_DMProfileEnabled()){
    fprintf(stderr, "CLUSTERDMPROF id=%d compute=%.3f total=%.3f gpu=%d threads=%d\n", myid0,
            dm_compute_end-stime, etime-stime, gpu_dm_done, omp_get_max_threads());
    fflush(stderr);
  }
  return (etime-stime);

}

double Calc_ParDM_Cluster_collinear(ClusterColState *state,
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

  windowF = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(n+1),0);

  TmpEVec0 = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*List_YOUSO[7],0);
  for (i=0; i<List_YOUSO[7]; i++){
    TmpEVec0[i] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(kmax-kmin+1),0);
  }

  TmpEVec1 = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*List_YOUSO[7],0);
  for (i=0; i<List_YOUSO[7]; i++){
    TmpEVec1[i] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(kmax-kmin+1),0);
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
      windowF[k] = weight_CWF2(e,state->chemical->chemical_potential,in0,in1,out0,out1);
    }

    else if (CWF_WindowF_switch==1){ // Fermi

      e = ko[spin][k];
      b0 = 1.0/CWF_disentangling_smearing_kBT0;
      b1 = 1.0/CWF_disentangling_smearing_kBT1;
      e0 = CWF_disentangling_Erange[0] + state->chemical->chemical_potential;
      e1 = CWF_disentangling_Erange[1] + state->chemical->chemical_potential;
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

  openmx_core_native_free(state->resources,windowF);

  for (i=0; i<List_YOUSO[7]; i++){
    openmx_core_native_free(state->resources,TmpEVec0[i]);
  }
  openmx_core_native_free(state->resources,TmpEVec0);

  for (i=0; i<List_YOUSO[7]; i++){
    openmx_core_native_free(state->resources,TmpEVec1[i]);
  }
  openmx_core_native_free(state->resources,TmpEVec1);

  dtime(&etime);
  return (etime-stime);

}

void Save_DOS_Col(ClusterColState *state,  int n, int MaxN, int myid1, int *is2, int *ie2,
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

  array0 = (float*)openmx_core_native_malloc(state->resources,1,sizeof(float)*(n+2),0);
  is3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  ie3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  SD = (float*)openmx_core_native_malloc(state->resources,1,sizeof(float)*(atomnum+1)*List_YOUSO[7],0);
  
  if (COHP_calc_flag==1){
    COHP = (double***)openmx_core_native_malloc(state->resources,1,sizeof(double**)*COHP_num_pairs,0);
    for (i=0; i<COHP_num_pairs; i++){
      COHP[i] = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*List_YOUSO[7],0);
      for (j=0; j<List_YOUSO[7]; j++){
        COHP[i][j] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*List_YOUSO[7],0);
      }    
    }

    COOP = (double***)openmx_core_native_malloc(state->resources,1,sizeof(double**)*COHP_num_pairs,0);
    for (i=0; i<COHP_num_pairs; i++){
      COOP[i] = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*List_YOUSO[7],0);
      for (j=0; j<List_YOUSO[7]; j++){
        COOP[i][j] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*List_YOUSO[7],0);
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
      if ( ((state->chemical->chemical_potential+Dos_Erange[0])<ko[spin][i]) && (i-1)<iemin ) {
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
      if ( ((state->chemical->chemical_potential+Dos_Erange[1])<ko[spin][i])) {
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
    fwrite(&state->chemical->chemical_potential,sizeof(double),1,fp_cohp);
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
    fprintf(fp_eig,"ChemP       %lf\n",state->chemical->chemical_potential);

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
        openmx_core_native_free(state->resources,COHP[i][j]);
      }    
      openmx_core_native_free(state->resources,COHP[i]);
    }
    openmx_core_native_free(state->resources,COHP);

    for (i=0; i<COHP_num_pairs; i++){
      for (j=0; j<List_YOUSO[7]; j++){
        openmx_core_native_free(state->resources,COOP[i][j]);
      }    
      openmx_core_native_free(state->resources,COOP[i]);
    }
    openmx_core_native_free(state->resources,COOP);
  }

  openmx_core_native_free(state->resources,SD);
  openmx_core_native_free(state->resources,array0);
  openmx_core_native_free(state->resources,is3);
  openmx_core_native_free(state->resources,ie3);
}

void Save_LCAO_Col(ClusterColState *state,  int n, int MaxN, int myid1, int *is2, int *ie2,
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

  array0 = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(n+2),0);
  is3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  ie3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);

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
      x = (ko[spin][i] - state->chemical->chemical_potential)*Beta;
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
    fwrite(&state->thermo->total, sizeof(double), 1, fp1);
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

  openmx_core_native_free(state->resources,array0);
  openmx_core_native_free(state->resources,is3);
  openmx_core_native_free(state->resources,ie3);
}

void Calc_XANES_Col(ClusterColState *state, const OpenMXAtomicViews *atomic,  int n, int MaxN, int myid1,
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
		     double *Work1, const OpenMXElectronicViews *electronic)
{
  double *****OLP=electronic->OLP;

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

  array0 = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(n+2),0);
  is3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  ie3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);

  C2 = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*(SpinP_switch+1),0);
  for (spin=0; spin<=SpinP_switch; spin++){
    C2[spin] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*n*MaxN,0);
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

    C1 = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*(SpinP_switch1+1),0);
    for (spin=0; spin<=SpinP_switch1; spin++){
      C1[spin] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*n1*MaxN1,0);
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

  OLPmo = (double*****)openmx_core_native_malloc(state->resources,1,sizeof(double****)*3,0);
  for (direction=0; direction<3; direction++){

    OLPmo[direction] = (double****)openmx_core_native_malloc(state->resources,1,sizeof(double***)*(Matomnum+1),0);
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = atomic->species.contracted_orbitals[Cwan];
      }

      OLPmo[direction][Mc_AN] = (double***)openmx_core_native_malloc(state->resources,1,sizeof(double**)*(FNAN[Gc_AN]+1),0);
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = atomic->species.contracted_orbitals[Hwan];
	}

	OLPmo[direction][Mc_AN][h_AN] = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*tno0,0);
	for (i=0; i<tno0; i++){
	  OLPmo[direction][Mc_AN][h_AN][i] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*tno1,0);
	}
      }
    }
  }

  /* calculation of momentum matrix elements */

  Calc_OLPmo(atomic, OLPmo);

  /* set MP */

  MP2 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(atomnum+1),0);

  p = 0; 
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    MP2[Gc_AN] = p; 
    Cwan = WhatSpecies[Gc_AN];
    tno0 = atomic->species.contracted_orbitals[Cwan];
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

  XANES_Out = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*Num_XANES_Out,0);
  for (i=0; i<Num_XANES_Out; i++){
    XANES_Out[i] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*10,0);
  }

  Z = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*(SpinP_switch+1),0);
  for (spin=0; spin<=SpinP_switch; spin++){
    Z[spin] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*UMOmax*UMOmax,0);
  }

  Ax = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*(SpinP_switch+1),0);
  for (spin=0; spin<=SpinP_switch; spin++){
    Ax[spin] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*UMOmax*UMOmax,0);
  }

  Ay = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*(SpinP_switch+1),0);
  for (spin=0; spin<=SpinP_switch; spin++){
    Ay[spin] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*UMOmax*UMOmax,0);
  }

  Az = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*(SpinP_switch+1),0);
  for (spin=0; spin<=SpinP_switch; spin++){
    Az[spin] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*UMOmax*UMOmax,0);
  }

  ind2ind = (int**)openmx_core_native_malloc(state->resources,1,sizeof(int*)*(SpinP_switch+1),0);
  for (spin=0; spin<=SpinP_switch; spin++){
    ind2ind[spin] = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*UMOmax,0);
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
          TNO1 = atomic->species.contracted_orbitals[Cwan];
          Anum = MP2[Gc_AN];

          for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            TNO2 = atomic->species.contracted_orbitals[Hwan];
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
    Utot2 = state->thermo->total;

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

      Calc_Oscillator_Strength(state,  n, UMOmax, HOMO_XANES, MP2, ind2ind, C1, C2, Z, Ax, Ay, Az, Utot1, Utot3, XANES_Res );

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
    openmx_core_native_free(state->resources,XANES_Out[i]);
  }
  openmx_core_native_free(state->resources,XANES_Out);

  for (spin=0; spin<=SpinP_switch; spin++){
    openmx_core_native_free(state->resources,Z[spin]);
  }
  openmx_core_native_free(state->resources,Z);

  for (spin=0; spin<=SpinP_switch; spin++){
    openmx_core_native_free(state->resources,Ax[spin]);
  }
  openmx_core_native_free(state->resources,Ax);

  for (spin=0; spin<=SpinP_switch; spin++){
    openmx_core_native_free(state->resources,Ay[spin]);
  }
  openmx_core_native_free(state->resources,Ay);

  for (spin=0; spin<=SpinP_switch; spin++){
    openmx_core_native_free(state->resources,Az[spin]);
  }
  openmx_core_native_free(state->resources,Az);

  for (spin=0; spin<=SpinP_switch; spin++){
    openmx_core_native_free(state->resources,ind2ind[spin]);
  }
  openmx_core_native_free(state->resources,ind2ind);

  openmx_core_native_free(state->resources,MP2);

  for (spin=0; spin<=SpinP_switch; spin++){
    openmx_core_native_free(state->resources,C2[spin]);
  }
  openmx_core_native_free(state->resources,C2);

  for (spin=0; spin<=SpinP_switch1; spin++){
    openmx_core_native_free(state->resources,C1[spin]);
  }
  openmx_core_native_free(state->resources,C1);

  openmx_core_native_free(state->resources,array0);
  openmx_core_native_free(state->resources,is3);
  openmx_core_native_free(state->resources,ie3);

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
	tno0 = atomic->species.contracted_orbitals[Cwan];
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = atomic->species.contracted_orbitals[Hwan];
	}

	for (i=0; i<tno0; i++){
	  openmx_core_native_free(state->resources,OLPmo[direction][Mc_AN][h_AN][i]);
	}
	openmx_core_native_free(state->resources,OLPmo[direction][Mc_AN][h_AN]);
      }
      openmx_core_native_free(state->resources,OLPmo[direction][Mc_AN]);
    }
    openmx_core_native_free(state->resources,OLPmo[direction]);
  }
  openmx_core_native_free(state->resources,OLPmo);
  OLPmo=NULL;
}

double Calc_Oscillator_Strength(ClusterColState *state,  int n, int UMOmax, int Nocc[2], int *MP,
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
    det[spin] = Lapack_LU_Dinverse(state, Nocc[spin], Z0[spin]);
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

double Lapack_LU_Dinverse(ClusterColState *state, int n, double *A)
{
  const char *thisprogram="Lapack_LU_inverse";
  int *ipiv;
  double *work,tmp,det;
  int lwork;
  int info,i,j;

  /* L*U factorization */

  ipiv = (int*) openmx_core_native_malloc(state->resources,1,sizeof(int)*n,0);

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
  work = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*lwork,0);

  F77_NAME(dgetri,DGETRI)(&n, A, &n, ipiv, work, &lwork, &info);

  if ( info !=0 ) {
    printf("dgetrf failed, info=%i, %s\n",info,thisprogram);
  }

  openmx_core_native_free(state->resources,work); openmx_core_native_free(state->resources,ipiv);

  return det;
}

double Calc_KS_orbitals_on_grid_single(ClusterColState *state,
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

  work1_on_grid = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*TNumGrid,0);
  work2_on_grid = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*TNumGrid,0);
  
  phi_on_grid = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*Nmax,0);
  for (i=0; i<Nmax; i++){
    phi_on_grid[i] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*TNumGrid,0);
    for (j=0; j<TNumGrid; j++){
      phi_on_grid[i][j] = 0.0;
    }
  }

  b_on_grid = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*Nmax,0);
  for (i=0; i<Nmax; i++){
    b_on_grid[i] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*TNumGrid,0);
    for (j=0; j<TNumGrid; j++){
      b_on_grid[i][j] = 0.0;
    }
  }

  B = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*Nmax*Nmax,0);
  ev = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*Nmax,0);

  asum = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*List_YOUSO[7],0);
  proj_chi = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*n,0);

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
	
	Exchange_Poisson(state,  work1_on_grid, work2_on_grid );

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
      
      Exchange_Poisson(state,  work1_on_grid, work2_on_grid );

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
      
      Exchange_Poisson(state,  work1_on_grid, work2_on_grid );

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

  openmx_core_native_free(state->resources,work1_on_grid);
  openmx_core_native_free(state->resources,work2_on_grid);
  
  for (i=0; i<Nmax; i++){
    openmx_core_native_free(state->resources,phi_on_grid[i]);
  }
  openmx_core_native_free(state->resources,phi_on_grid);

  for (i=0; i<Nmax; i++){
    openmx_core_native_free(state->resources,b_on_grid[i]);
  }
  openmx_core_native_free(state->resources,b_on_grid);

  openmx_core_native_free(state->resources,B);
  openmx_core_native_free(state->resources,ev);

  openmx_core_native_free(state->resources,asum);
  openmx_core_native_free(state->resources,proj_chi);

  //free(Hex);
  
  dtime(&etime);
  return (etime-stime);

}

double Exchange_Poisson(ClusterColState *state,  double *work1_on_grid, double *work2_on_grid )
{
  int i,j,k,k1,k2,k3,GN,GN_in,GN_out;
  double sk1,sk2,sk3,fac_invG2,Gx,Gy,Gz,tmp0;
  double stime,etime;
  double *ReTmp1,*ImTmp1;
  double *ReTmp2,*ImTmp2;
  fftw_complex *in, *out;
  fftw_plan p;

  dtime(&stime);

  ReTmp1 = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*TNumGrid,0);
  ImTmp1 = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*TNumGrid,0);

  ReTmp2 = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*TNumGrid,0);
  ImTmp2 = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*TNumGrid,0);
  
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

  openmx_core_native_free(state->resources,ReTmp1);
  openmx_core_native_free(state->resources,ImTmp1);

  openmx_core_native_free(state->resources,ReTmp2);
  openmx_core_native_free(state->resources,ImTmp2);

  fftw_free(in);
  fftw_free(out);
  
  dtime(&etime);
  return (etime-stime);

}

double Calc_HF_Vx_single(ClusterColState *state,
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

  work1_on_grid = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*TNumGrid,0);
  work2_on_grid = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*TNumGrid,0);
  
  phi_on_grid = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*Nmax,0);
  for (i=0; i<Nmax; i++){
    phi_on_grid[i] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*TNumGrid,0);
    for (j=0; j<TNumGrid; j++){
      phi_on_grid[i][j] = 0.0;
    }
  }

  b_on_grid = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*Nmax,0);
  for (i=0; i<Nmax; i++){
    b_on_grid[i] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*TNumGrid,0);
    for (j=0; j<TNumGrid; j++){
      b_on_grid[i][j] = 0.0;
    }
  }

  B = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*Nmax*Nmax,0);
  ev = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*Nmax,0);

  asum = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*List_YOUSO[7],0);
  proj_chi = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*n,0);

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
	
      Exchange_Poisson(state,  work1_on_grid, work2_on_grid );

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

  for (i=0; i<(n*n); i++){ state->Hex[i] = 0.0; }
  
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
        state->Hex[i*n+j] += proj_chi[i]*proj_chi[j]*ev[orb];
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
    state->Hex[i] = 0.98*state->Hex0[i] + 0.02*state->Hex[i];
    //Hex[i] = 0.0;
    state->Hex0[i] = state->Hex[i];
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

  openmx_core_native_free(state->resources,work1_on_grid);
  openmx_core_native_free(state->resources,work2_on_grid);
  
  for (i=0; i<Nmax; i++){
    openmx_core_native_free(state->resources,phi_on_grid[i]);
  }
  openmx_core_native_free(state->resources,phi_on_grid);

  for (i=0; i<Nmax; i++){
    openmx_core_native_free(state->resources,b_on_grid[i]);
  }
  openmx_core_native_free(state->resources,b_on_grid);

  openmx_core_native_free(state->resources,B);
  openmx_core_native_free(state->resources,ev);

  openmx_core_native_free(state->resources,asum);
  openmx_core_native_free(state->resources,proj_chi);

  //free(Hex);
  
  dtime(&etime);
  return (etime-stime);

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
  size_t nn = ClusterCol_CheckedMulCount((size_t)n,(size_t)n,"device dense matrix");

  if (d_H==NULL){
    ClusterCol_AbortWithMessage("NULL device dense matrix in Cluster_DFT_Col.c.");
  }

  /* dense_index is device-resident via ClusterCol_DenseIndexCache_Get;
     only the packed matrix values travel to the device here */
#pragma acc enter data copyin(H1[0 : tnum])
if (!openmx_gpu_scatter(n,tnum,0,acc_deviceptr((void*)(dense_index)),tnum,acc_deviceptr((void*)(H1)),0,NULL,NULL,0,d_H)) {
#pragma acc parallel loop deviceptr(d_H)
  for (size_t p=0; p<nn; p++){
    d_H[p] = 0.0;
  }

#pragma acc parallel loop deviceptr(d_H) present(H1[0 : tnum], dense_index[0 : tnum])
  for (int p=0; p<tnum; p++){
    int idx = dense_index[p];
#pragma acc atomic update
    d_H[idx] += H1[p];
  }
}
#pragma acc exit data delete(H1[0 : tnum])
}

static void Patch2Device_Cluster_Owner(ClusterColState *state, double ****RH, int *MP, int owns_dense, int n, double *d_H)
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

  My_NZeros = (int*)ClusterCol_MallocArray(state, (size_t)numprocs,sizeof(int),"Patch2Device My_NZeros");
  My_Matomnum = (int*)ClusterCol_MallocArray(state, (size_t)numprocs,sizeof(int),"Patch2Device My_Matomnum");
  is1 = (int*)ClusterCol_MallocArray(state, (size_t)numprocs,sizeof(int),"Patch2Device is1");
  ie1 = (int*)ClusterCol_MallocArray(state, (size_t)numprocs,sizeof(int),"Patch2Device ie1");
  is2 = (int*)ClusterCol_MallocArray(state, (size_t)numprocs,sizeof(int),"Patch2Device is2");
  order_GA = (int*)ClusterCol_MallocArray(state, (size_t)(atomnum+2),sizeof(int),"Patch2Device order_GA");

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

  H1 = (double*)ClusterCol_MallocArray(state, (size_t)(tnum+1),sizeof(double),"Patch2Device sparse buffer");

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
    dense_index = ClusterCol_DenseIndexCache_Get(state, order_GA,MP,n,tnum);
    ClusterCol_BuildDeviceDenseFromPacked(H1,dense_index,tnum,n,d_H);
  }

  openmx_core_native_free(state->resources,My_NZeros);
  openmx_core_native_free(state->resources,My_Matomnum);
  openmx_core_native_free(state->resources,is1);
  openmx_core_native_free(state->resources,ie1);
  openmx_core_native_free(state->resources,is2);
  openmx_core_native_free(state->resources,order_GA);
  openmx_core_native_free(state->resources,H1);
}

static void Patch2Host_Cluster_Owner(ClusterColState *state, double ****RH, int *MP, int owns_dense, int n, double *d_H)
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

  My_NZeros = (int*)ClusterCol_MallocArray(state, (size_t)numprocs,sizeof(int),"Patch2Device My_NZeros");
  My_Matomnum = (int*)ClusterCol_MallocArray(state, (size_t)numprocs,sizeof(int),"Patch2Device My_Matomnum");
  is1 = (int*)ClusterCol_MallocArray(state, (size_t)numprocs,sizeof(int),"Patch2Device is1");
  ie1 = (int*)ClusterCol_MallocArray(state, (size_t)numprocs,sizeof(int),"Patch2Device ie1");
  is2 = (int*)ClusterCol_MallocArray(state, (size_t)numprocs,sizeof(int),"Patch2Device is2");
  order_GA = (int*)ClusterCol_MallocArray(state, (size_t)(atomnum+2),sizeof(int),"Patch2Device order_GA");

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

  H1 = (double*)ClusterCol_MallocArray(state, (size_t)(tnum+1),sizeof(double),"Patch2Device sparse buffer");

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

  openmx_core_native_free(state->resources,My_NZeros);
  openmx_core_native_free(state->resources,My_Matomnum);
  openmx_core_native_free(state->resources,is1);
  openmx_core_native_free(state->resources,ie1);
  openmx_core_native_free(state->resources,is2);
  openmx_core_native_free(state->resources,order_GA);
  openmx_core_native_free(state->resources,H1);
}

/* Run-boundary reset: geometry and device state must not survive into the
   next -runtest input, whose host allocations can reuse the same addresses.
   Called on every rank before Free_Arrays(0) frees the current system. */
static void ClusterCol_ReleaseCaches(ClusterColState *state)
{
    state->mgpu_host_dense=0;
    ClusterCol_ResetGs2Fallback(state);
    ClusterCol_UnionGrid_Free(state);
    state->cpu_overlap_n = 0;
    openmx_core_native_free(state->resources,state->dm_entry_cache.basis0);
    openmx_core_native_free(state->resources,state->dm_entry_cache.basis1);
    memset(&state->dm_entry_cache, 0, sizeof(state->dm_entry_cache));
    Cluster_DFT_Col_Release_GPU_Solver(state);
    /* The next input owns a fresh refinement policy and no prior basis. */
    const ClusterColRefineState initial = {-1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, -1, {0, 0}, 0, 0};
    state->refine = initial;
}
