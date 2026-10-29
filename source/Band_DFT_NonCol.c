/**********************************************************************
  Band_DFT_NonCol.c:

     Band_DFT_NonCol.c is a subroutine to perform band calculations
     based on a non-collinear DFT. 

  Log of Band_DFT_NonCol.c:

     16/Feb./2019  Released by T. Ozaki

***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_solver_workspace.h"
#include "openmx_native_resources.h"
#include "eigen_refine_gpu.h"
#include "openmx_precision.h"
#include "openmx_mgpu_runtime.h"
#include "openmx_gpu_transform.h"
#include "openmx_gpu_dm_collective.h"
#include "openmx_gpu_workers.h"
#include "openmx_host_simd.h"
#include "openmx_nc_dm_cpu.h"
#include "openmx_mgpu_density.h"

#include "lapack_prototypes.h"
#include "tran_variables.h"
#include "set_cuda_default_device_from_local_rank.h"
#include "set_openacc_device_from_local_rank.h"
#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <accel.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

/* Persistent metadata has the explicit calculation lifetime. Fields borrow
   buffers and handles owned by its native resource group. */
typedef struct
{
    int            initialized;
    int            device_id;
    int            max_tno;
    int            max_cols;
    int            max_nk;
    cudaStream_t   stream;
    cublasHandle_t cublas;
    dcomplex *     d_vec_up;
    dcomplex *     d_vec_up_e;
    dcomplex *     d_vec_dn;
    dcomplex *     d_vec_dn_e;
    dcomplex *     d_cols_up;
    dcomplex *     d_cols_dn;
    dcomplex *     d_mat11;
    dcomplex *     d_mat22;
    dcomplex *     d_mat12;
    dcomplex *     d_mat11e;
    dcomplex *     d_mat22e;
} BandNonColDMGpuWorkspace;

typedef struct
{
    int m_index;
    int dense_index;
    int phase_index;
} BandNonColConstructEntry;

typedef struct
{
    int                       valid;
    int                       n;
    int                       dense_count;
    int                       h_count;
    int                       phase_count;
    int                       dense_device_valid;
    unsigned long long        fingerprint;
    int *                     phase_l1;
    int *                     phase_l2;
    int *                     phase_l3;
    double *                  phase_r;
    double *                  phase_i;
    BandNonColConstructEntry *dense_entries;
} BandNonColConstructCache;

typedef struct
{
    int                initialized;
    int                device_id;
    int                matrix_dim;
    size_t             d_work_bytes;
    size_t             h_work_bytes;
    cudaStream_t       stream;
    cusolverDnHandle_t gpusolver;
    dcomplex *         d_A;
    double *           d_W;
    int32_t *          d_info;
    void *             d_work;
    void *             h_work;
} BandNonColGpuSolverWorkspace;

typedef struct
{
    int     iterations;    /* -1 until the environment is read */
    double  until;         /* printed NormRD below which the FP64 eigensolver takes over; 0 off */
    int     final_stage;   /* the FP64 eigensolver for the rest of the SCF cycle */
    int     last_refined;  /* the latest SCF eigensolve was a refined one (any k-point) */
    int     cycle_refined; /* some eigensolve of this SCF cycle was */
    int     defaulted;     /* on because of the device, not of OPENMX_EIGEN_REFINE */
    int     force_fp64;    /* Band_DFT_NonCol_RefineForceFP64: this call in FP64 whatever the stage */
    int     transient;     /* the first step of a cycle: its failures do not condemn the cycle */
    int     pending;       /* this rank's k-point: prepared, waiting for the chemical potential */
    int     solved;        /* this rank's k-point was refined this step */
    int     persistent;    /* the strongest persistent failure of this step on this rank */
    int     kdense;        /* this call refines the k-dense path (several k-points per rank) */
    int     scf_iter;      /* the step of the call */
    int     staged;        /* this call refines the root-dense path in serialized GPU turns: pass 1 and pass 2 as
                              on the k-dense path, the bases on the host between them */
} BandNonColRefineState;

typedef struct
{
    dcomplex *basis;       /* n2 x n2, host */
    double   *lam;         /* n2 eigenvalue estimates */
    int       valid;
    int       refined;     /* pass 1 of this step refined the k-point */
} BandNonColKRefineSlot;

typedef struct
{
    double mu;
} BandNonColOccupationCtx;

typedef struct {
    dcomplex *matrix;
    double k1, k2, k3;
} BandNonColKOverlap;

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
} BandNonColDMEntryCache;

typedef struct
{
    double *occ;
    double *eig_occ;
    int     max_occ_nk;
} BandNonColDMOccWorkspace;

typedef struct
{
    int       valid;
    int       s_valid;
    /* the solved eigenvectors stay device-resident (the host cs2 pages are
       never written): the MPI scatter packs per-destination panels on the
       device and the DM kernel's copyin degrades to a present hit */
    int       cs2_on_device;
    int       n;
    int       n2;
    int       maxn;
    int       t_knum;
    dcomplex *s_all;
    dcomplex *h11;
    dcomplex *h22;
    dcomplex *h12;
    dcomplex *work;
    dcomplex *hs2;
    dcomplex *cs2;
} BandNonColRootDenseWorkspace;

struct OpenMXBandNonColState {
  OpenMXChemicalPotentials *chemical; /* borrowed from this calculation */
  OpenMXBandEnergies *energies; /* borrowed from this calculation */
  OpenMXNativeResources *resources; /* borrowed from this calculation */
  OpenMXHamiltonianCache *hamiltonian_cache; /* borrowed from this calculation */
  int initialized;
  BandNonColRootDenseWorkspace root_dense_workspace;
  BandNonColDMOccWorkspace dm_occ_workspace;
  BandNonColDMEntryCache dm_entry_cache;
  int gpu_dense_verdict;
  size_t k_overlap_bytes;
  size_t k_overlap_used;
  int k_overlap_n;
  int k_overlap_points;
  BandNonColKOverlap *k_overlap;
  int refine_e0_n;
  double *refine_e0;
  int refine_x_n;
  dcomplex *refine_x;
  BandNonColOccupationCtx refine_occ;
  EigenRefineProblem refine_pb;
  EigenRefineDevice refine_dev;
  EigenRefineState refine_state;
  BandNonColRefineState refine;
  int refine_force_warm;
  size_t kref_used;
  int kref_slots;
  int kref_n2;
  BandNonColKRefineSlot *kref;
  BandNonColGpuSolverWorkspace gpusolver_workspace;
  BandNonColConstructCache construct_cache;
  BandNonColDMGpuWorkspace dm_gpu_workspace;
  int mgpu_host_dense;
  int Construct_Band_Ms_firsttime;
  int Construct_Band_DenseMs_firsttime;
  int Band_DFT_NonCol_owned_announced;
  int Band_DFT_NonCol_owned_firsttime;
  int AutoGpuTurnLimit_last_device_ranks[3];
  int AutoGpuTurnLimit_last_limit[3];
  int GpuDiagFits_verdict_n2;
  int GpuDiagFits_verdict;
  int QueryGpuSolverWorkBytes_cached_next;
  size_t QueryGpuSolverWorkBytes_cached_bytes[2];
  int QueryGpuSolverWorkBytes_cached_maxn[2];
  int QueryGpuSolverWorkBytes_cached_n[2];
};
typedef OpenMXBandNonColState BandNonColState;

static void BandNonCol_ReleaseCaches(BandNonColState *state);
static void BandNonCol_Reset(OpenMXNativeResources *owner, void *metadata)
{
  BandNonColState *state = (BandNonColState *)metadata;
  (void)owner;
  if (state->initialized) BandNonCol_ReleaseCaches(state);
}

OpenMXBandNonColState *Band_DFT_NonCol_Bind_Resources(OpenMXCoreContext *core)
{
  OpenMXNativeResources *owner = NULL;
  void *metadata = NULL;
  char error[256];
  if (!openmx_core_native_resources_state(core, "Band_DFT_NonCol",
      sizeof(BandNonColState), BandNonCol_Reset, &owner, &metadata, error, sizeof(error))) {
    fprintf(stderr, "OpenMX noncollinear band state: %s\n", error);
    abort();
  }
  BandNonColState *state = (BandNonColState *)metadata;
  if (!state->initialized) {
    state->resources = owner;
    state->hamiltonian_cache = Set_Hamiltonian_Cache_Bind_Resources(core);
    state->gpu_dense_verdict = 1;
    const BandNonColRefineState initial_refine = {-1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    state->refine = initial_refine;
    state->Construct_Band_Ms_firsttime = 1;
    state->Construct_Band_DenseMs_firsttime = 1;
    state->Band_DFT_NonCol_owned_firsttime = 1;
    const int initial_AutoGpuTurnLimit_last_device_ranks[3] = {-1,-1,-1};
    memcpy(state->AutoGpuTurnLimit_last_device_ranks, initial_AutoGpuTurnLimit_last_device_ranks, sizeof(state->AutoGpuTurnLimit_last_device_ranks));
    const int initial_AutoGpuTurnLimit_last_limit[3] = {-1,-1,-1};
    memcpy(state->AutoGpuTurnLimit_last_limit, initial_AutoGpuTurnLimit_last_limit, sizeof(state->AutoGpuTurnLimit_last_limit));
    state->GpuDiagFits_verdict = -1;
    const int initial_QueryGpuSolverWorkBytes_cached_maxn[2] = {-1,-1};
    memcpy(state->QueryGpuSolverWorkBytes_cached_maxn, initial_QueryGpuSolverWorkBytes_cached_maxn, sizeof(state->QueryGpuSolverWorkBytes_cached_maxn));
    const int initial_QueryGpuSolverWorkBytes_cached_n[2] = {-1,-1};
    memcpy(state->QueryGpuSolverWorkBytes_cached_n, initial_QueryGpuSolverWorkBytes_cached_n, sizeof(state->QueryGpuSolverWorkBytes_cached_n));
    if(!openmx_core_band_energies(core,&state->energies,error,sizeof(error))) {
      fprintf(stderr,"OpenMX band energies: %s\n",error);
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

#define  measure_time  0

/* GPU workspace for band non-collinear DM (added by H.Kawai, ported from 3.9.9 GPU) */

static void BandNonCol_AbortWithMessage(const char *msg);

/* The refined FP32 eigensolver (eigen_refine_gpu.c) on the one-k-point-
   per-k-world path of the SCF solves (all_knum == 1 with every owner's
   buffers on the device at once).  OPENMX_EIGEN_REFINE=K (0: off), unset:
   on by default on GPUs whose FP64 throughput is a small fraction of their
   FP32 one when the complex GEMMul8 products are on; the global root's
   device decides for all ranks.  The refinement's occupations need the
   chemical potential of all k-points, which the solver only knows after
   the eigenvalues are gathered: every owner first solves in FP32 (or
   starts from its previous vectors), the estimates are gathered and a
   chemical potential found on them, then the owners finish their
   refinements and the solver goes on with the refined eigenvalues.  The
   k-dense path (several k-points per rank) keeps the FP64 eigensolver.
   An SCF that ends on a refined step is solved once more in FP64
   (Band_DFT_NonCol_RefineForceFP64 around a repeated call, which
   recomputes the density matrices), and the last step scf.maxIter allows
   is FP64 in any case. */

/* The k-dense path keeps, per k-point of this rank, the basis of the latest
   refined solve on the host (the device workspace goes with every turn
   group): pass 2 starts its refinement from it, and the next step's pass 1
   as well when OPENMX_EIGEN_REFINE_WARM_BAND is on.  The budget is
   OPENMX_BAND_REFINE_BASIS_MB per rank (default 1024). */

static void BandNonCol_KRefReset(BandNonColState *state);




static void BandNonCol_RefineDropBasis(BandNonColState *state)
{
    if (state->refine_x != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,state->refine_x));
    state->refine_x = NULL;
    state->refine_x_n = 0;
    state->refine_state.basis_valid = 0;
}

static void BandNonCol_RefineRelease(BandNonColState *state)
{
    openmx_eigen_refine_state_release(&state->refine_state);
    BandNonCol_KRefReset(state);
    openmx_eigen_refine_device_release(&state->refine_dev);
    BandNonCol_RefineDropBasis(state);
    openmx_core_native_free(state->resources,state->refine_e0);
    state->refine_e0 = NULL;
    state->refine_e0_n = 0;
}

/* every rank, once: OPENMX_EIGEN_REFINE, or the default the global root's
   device implies, for all ranks alike */
static void BandNonCol_RefineConfigure(BandNonColState *state)
{
    BandNonColRefineState *st = &state->refine;
    int myid, config[2] = {0, 0};

    /* Input files in -runtest share a process; do not cache an input veto. */
    if (!openmx_fp32_enabled()) return;
    if (0 <= st->iterations) return;
    MPI_Comm_rank(mpi_comm_level1, &myid);
    if (myid == Host_ID) config[0] = openmx_eigen_refine_configure(1, 2 <= level_stdout, &config[1]);
    MPI_Bcast(config, 2, MPI_INT, Host_ID, mpi_comm_level1);
    st->iterations = config[0];
    st->defaulted = (0 < config[1]);
    st->until = openmx_eigen_refine_until();
}

static int BandNonCol_RefineIterations(BandNonColState *state)
{
    return (openmx_fp32_enabled() && 0 < state->refine.iterations) ? state->refine.iterations : 0;
}

/* what the refined solve adds to a k-point owner's device memory: the basis */
static size_t BandNonCol_RefineBasisBytes(BandNonColState *state, int n2)
{
    return (0 < BandNonCol_RefineIterations(state)) ? sizeof(dcomplex) * (size_t)n2 * (size_t)n2 : 0;
}

/* every rank, before the solves of a call: possible says whether the call
   takes the one-k-point-per-world path with resident buffers */
static void BandNonCol_RefineBeginStep(BandNonColState *state, int SCF_iter, int possible, int possible_kdense, int possible_staged)
{
    BandNonColRefineState *st = &state->refine;

    if (state->mgpu_host_dense) possible=possible_kdense=possible_staged=0;
    BandNonCol_RefineConfigure(state);
    if (SCF_iter == 1) {
        st->final_stage = 0;
        st->cycle_refined = 0;
        state->refine_state.basis_valid = 0;   /* a new cycle may come with a new geometry */
        BandNonCol_KRefReset(state);
    }
    st->transient = (SCF_iter == 1);
    st->scf_iter = SCF_iter;
    st->pending = 0;
    st->solved = 0;
    st->persistent = 0;
    /* the occupations of the refinement are FermiFunc_NC at one chemical
       potential: XANES and emptied states keep the FP64 eigensolver */
    st->last_refined = (possible && 0 < BandNonCol_RefineIterations(state) && !st->final_stage && !st->force_fp64 &&
                        Cnt_switch == 0 && SCF_iter < DFTSCF_loop && xanes_calc == 0 && empty_states_flag == 0);
    /* the k-dense path gates its first pass with the previous step's
       chemical potential, so it starts at the second step */
    st->kdense = (possible_kdense && 2 <= SCF_iter && 0 < BandNonCol_RefineIterations(state) && !st->final_stage &&
                  !st->force_fp64 && Cnt_switch == 0 && SCF_iter < DFTSCF_loop && xanes_calc == 0 &&
                  empty_states_flag == 0);
    st->staged = (possible_staged && 2 <= SCF_iter && 0 < BandNonCol_RefineIterations(state) && !st->final_stage &&
                  !st->force_fp64 && Cnt_switch == 0 && SCF_iter < DFTSCF_loop && xanes_calc == 0 &&
                  empty_states_flag == 0);
    if (state->kref != NULL) {
        for (int slot = 0; slot < state->kref_slots; slot++) state->kref[slot].refined = 0;
    }
}

/* every rank, after the second halves: the owners' solves decide the stage
   for all ranks alike */
static void BandNonCol_RefineEndStep(BandNonColState *state)
{
    BandNonColRefineState *st = &state->refine;
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

int Band_DFT_NonCol_RefineLastRefined(BandNonColState *state)
{
    return state->refine.last_refined;
}

void Band_DFT_NonCol_RefineForceFP64(BandNonColState *state, int on)
{
    state->refine.force_fp64 = on;
}

/* the end of an SCF cycle: the warm-start bases go (the device one and the
   host slots), so the forces and the next cycle's first step start with the
   device memory free */
void Band_DFT_NonCol_RefineEndCycle(BandNonColState *state)
{
    BandNonCol_RefineDropBasis(state);
    BandNonCol_KRefReset(state);
    state->refine_state.basis_valid = 0;
}

/* DFT.c, after the mixing of an SCF step that goes on: normrd is the printed
   NormRD */
void Band_DFT_NonCol_RefineAfterMixing(BandNonColState *state, double normrd)
{
    BandNonColRefineState *st = &state->refine;
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

/* Fermi occupation of the non-collinear band solver at the chemical
   potential mu (FermiFunc_NC handles the emptied orbitals by index) */
static double BandNonCol_RefineFermi(double e, int index, void *ctx)
{
    double x = (e - ((BandNonColOccupationCtx *)ctx)->mu) * Beta;

    if (x <= -60.0) x = -60.0;
    if (60.0 <= x) x = 60.0;
    return FermiFunc_NC(x, index);
}

/* The chemical potential of TZ - system_charge electrons over the lowest
   maxn estimates of every k-point in EIGEN (the solver's search on the
   same weights), to the accuracy the refinement's occupations need */
static double BandNonCol_RefineChemP(double ***EIGEN, int T_knum, int maxn, const int *T_k_op, double sum_weights,
                                     double TZ)
{
    double lo = -20.0, hi = 20.0, mu = 0.0;
    BandNonColOccupationCtx ctx;

    for (int it = 0; it < 1000; it++) {
        double count = 0.0, dnum;

        mu = 0.5 * (lo + hi);
        ctx.mu = mu;
        for (int kloop = 0; kloop < T_knum; kloop++) {
            for (int l = 1; l <= maxn; l++) {
                count += BandNonCol_RefineFermi(EIGEN[0][kloop][l], l, &ctx) * (double)T_k_op[kloop];
            }
        }
        count /= sum_weights;
        dnum = TZ - count - system_charge;
        if (0.0 <= dnum) lo = mu;
        else hi = mu;
        if (fabs(dnum) < 1.0e-12) break;
    }
    return mu;
}

static void BandNonCol_DMGpu_Destroy(BandNonColState *state)
{
    BandNonColDMGpuWorkspace * w = &state->dm_gpu_workspace;
    if (w->d_vec_up   != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_vec_up));
    if (w->d_vec_up_e != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_vec_up_e));
    if (w->d_vec_dn   != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_vec_dn));
    if (w->d_vec_dn_e != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_vec_dn_e));
    if (w->d_cols_up  != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_cols_up));
    if (w->d_cols_dn  != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_cols_dn));
    if (w->d_mat11    != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_mat11));
    if (w->d_mat22    != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_mat22));
    if (w->d_mat12    != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_mat12));
    if (w->d_mat11e   != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_mat11e));
    if (w->d_mat22e   != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_mat22e));
    if (w->cublas     != NULL) OpenMXBlasWait(OpenMX_Resource_BlasDestroy(state->resources,w->cublas));
    if (w->stream     != NULL) wait_cudafunc(OpenMX_Resource_StreamDestroy(state->resources,w->stream));
    memset(w, 0, sizeof(*w));
    w->device_id = -1;
}

static void BandNonCol_GpuSolver_Destroy(BandNonColState *state)
{
    BandNonColGpuSolverWorkspace * w = &state->gpusolver_workspace;

    if (w->d_A      != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_A));
    if (w->d_W      != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_W));
    if (w->d_info   != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_info));
    if (w->d_work   != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_work));
    if (w->h_work   != NULL) openmx_core_native_free(state->resources,w->h_work);
    if (w->gpusolver != NULL) OpenMXSolverWait(OpenMX_Resource_SolverDestroy(state->resources,w->gpusolver));
    if (w->stream   != NULL) wait_cudafunc(OpenMX_Resource_StreamDestroy(state->resources,w->stream));

    memset(w, 0, sizeof(*w));
    w->device_id = -1;
}

static void BandNonCol_DMGpu_Init(BandNonColState *state)
{
    BandNonColDMGpuWorkspace * w = &state->dm_gpu_workspace;
    int current_device;
    wait_cudafunc(cudaGetDevice(&current_device));
    if (w->initialized && w->device_id == current_device) return;
    if (w->initialized) BandNonCol_DMGpu_Destroy(state);
    wait_cudafunc(OpenMX_Resource_StreamCreate(state->resources,&w->stream,cudaStreamNonBlocking));
    OpenMXBlasWait(OpenMX_Resource_BlasCreate(state->resources,&w->cublas));
    OpenMXBlasWait(cublasSetStream(w->cublas, w->stream));
    w->initialized = 1;
    w->device_id   = current_device;
}

static void BandNonCol_GpuSolver_Init(BandNonColState *state)
{
    BandNonColGpuSolverWorkspace * w = &state->gpusolver_workspace;
    int current_device;

    wait_cudafunc(cudaGetDevice(&current_device));
    if (w->initialized && w->device_id == current_device) return;
    if (w->initialized) BandNonCol_GpuSolver_Destroy(state);

    wait_cudafunc(OpenMX_Resource_StreamCreate(state->resources,&w->stream,cudaStreamNonBlocking));
    OpenMXSolverWait(OpenMX_Resource_SolverCreate(state->resources,&w->gpusolver));
    OpenMXSolverWait(cusolverDnSetStream(w->gpusolver, w->stream));

    w->initialized = 1;
    w->device_id   = current_device;
}

static void BandNonCol_GpuSolver_EnsureInfo(BandNonColState *state)
{
    BandNonColGpuSolverWorkspace * w = &state->gpusolver_workspace;

    BandNonCol_GpuSolver_Init(state);
    if (w->d_info==NULL) wait_cudafunc(OpenMX_Resource_GpuMalloc(state->resources,(void**)&w->d_info,sizeof(int32_t)));
}

static void BandNonCol_GpuSolver_ReleaseDeviceWorkspace(BandNonColState *state)
{
    BandNonColGpuSolverWorkspace * w = &state->gpusolver_workspace;

    if (w->d_work!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_work));
    w->d_work = NULL;
    w->d_work_bytes = 0;
}

static void BandNonCol_SetDenseGemmul8Defaults(void)
{
    if (openmx_env_at(OPENMX_ENV_GEMMUL8_MAX_WORKSPACE_PERCENT)==NULL &&
        openmx_env_at(OPENMX_ENV_EXT_GEMMUL8_MAX_WORKSPACE_PERCENT)==NULL){
        openmx_env_set_if_unset("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT","50");
    }
}

static int BandNonCol_GemmWorkspaceTurnRelease(void)
{
    return openmx_env_flag_at(OPENMX_ENV_BAND_GEMMUL8_TURN_RELEASE, 1);
}

static int BandNonCol_EigenvaluesOnlyNoVectors(void)
{
    /* The first (all_knum!=1) diagonalization pass only needs eigenvalues for
       the chemical potential; jobz=CUSOLVER_EIG_MODE_NOVECTOR skips the
       eigenvector build inside cusolverDnXsyevdx and returns identical
       eigenvalues.  Opt out with OPENMX_BAND_SYEVDX_NOVECTOR=0. */
    return openmx_env_flag_at(OPENMX_ENV_BAND_SYEVDX_NOVECTOR, 1);
}

/* AMD's multi-k cache avoids diagonalizing H(k) again after finding the
   chemical potential. Keep only n2*MaxN backtransformed states within this
   SCF call, capped by both the user limit and available node RAM. */
static size_t BandNonCol_HostCacheLimit(int id, double default_mib,
                                       unsigned fraction)
{
    double cap = openmx_env_double_at(id, default_mib);
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
    available_kib /= (unsigned)ranks * (unsigned long long)fraction;
    if (available_kib < limit / 1024U)
        limit = (size_t)available_kib * 1024U;
    return limit;
}

static size_t BandNonCol_KCacheLimit(void)
{
    return BandNonCol_HostCacheLimit(OPENMX_ENV_BAND_NONCOL_KCACHE_MB, 1024.0, 32);
}

/* K points belong to contiguous per-rank ranges. Visiting consecutive k
   indices would serialize most GPU groups on one owner. Interleave owners
   while preserving each owner's ascending k order and hence its DM sums. */
static void BandNonCol_InterleaveKOrder(BandNonColState *state, int points, int ranks,
                                       const int *owner, int *order)
{
    int *head = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(size_t)ranks,0);
    int *next = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(size_t)points,0);
    int position = 0;
    if (head == NULL || next == NULL)
        BandNonCol_AbortWithMessage("Failed to allocate noncollinear k-point schedule.");
    for (int rank=0; rank<ranks; ++rank) head[rank] = -1;
    for (int point=points-1; point>=0; --point){
        int rank = owner[point];
        if (rank < 0 || rank >= ranks)
            BandNonCol_AbortWithMessage("Invalid owner in noncollinear k-point schedule.");
        next[point] = head[rank];
        head[rank] = point;
    }
    while (position < points){
        for (int rank=0; rank<ranks; ++rank){
            int point = head[rank];
            if (point < 0) continue;
            order[position++] = point;
            head[rank] = next[point];
        }
    }
    openmx_core_native_free(state->resources,next);
    openmx_core_native_free(state->resources,head);
}

/* The transformed overlap depends on geometry and k, but not on the SCF
   Hamiltonian. Keep a bounded host copy across SCFs, independently of the
   device workspaces that are released at each GPU concurrency group. */

static void BandNonCol_KOverlapReset(BandNonColState *state)
{
    for (int k = 0; k < state->k_overlap_points; ++k)
        openmx_core_native_free(state->resources,state->k_overlap[k].matrix);
    openmx_core_native_free(state->resources,state->k_overlap);
    state->k_overlap = NULL;
    state->k_overlap_n = state->k_overlap_points = 0;
    state->k_overlap_bytes = state->k_overlap_used = 0;
}

static size_t BandNonCol_KOverlapLimit(void)
{
    return BandNonCol_HostCacheLimit(OPENMX_ENV_BAND_NONCOL_OVERLAP_CACHE_MB, 512.0, 64);
}

static void BandNonCol_KOverlapPrepare(BandNonColState *state, int n, int points)
{
    size_t bytes;
    if (state->k_overlap_n != n || state->k_overlap_points != points)
        BandNonCol_KOverlapReset(state);
    if (state->k_overlap != NULL || n <= 0 || points <= 0 ||
        (size_t)n > SIZE_MAX / (size_t)n / sizeof(dcomplex)) return;
    bytes = (size_t)n * (size_t)n * sizeof(dcomplex);
    if (bytes > BandNonCol_KOverlapLimit()) return;
    state->k_overlap = (BandNonColKOverlap*)openmx_core_native_malloc(state->resources,(size_t)points,sizeof(BandNonColKOverlap),1);
    if (state->k_overlap == NULL) return;
    state->k_overlap_n = n;
    state->k_overlap_points = points;
    state->k_overlap_bytes = bytes;
}

static int BandNonCol_KOverlapRestore(BandNonColState *state, int point, double k1, double k2, double k3,
                                      dcomplex *matrix)
{
    BandNonColKOverlap *entry;
    if (state->k_overlap == NULL || point < 0 || point >= state->k_overlap_points) return 0;
    entry = &state->k_overlap[point];
    if (entry->matrix == NULL || entry->k1 != k1 || entry->k2 != k2 || entry->k3 != k3) return 0;
    memcpy(matrix, entry->matrix, state->k_overlap_bytes);
    return 1;
}

static void BandNonCol_KOverlapSave(BandNonColState *state, int point, double k1, double k2, double k3,
                                   const dcomplex *matrix)
{
    BandNonColKOverlap *entry;
    if (state->k_overlap == NULL || point < 0 || point >= state->k_overlap_points) return;
    entry = &state->k_overlap[point];
    if (entry->matrix == NULL) {
        size_t limit = BandNonCol_KOverlapLimit();
        if (state->k_overlap_used > limit ||
            state->k_overlap_bytes > limit - state->k_overlap_used) return;
        entry->matrix = (dcomplex*)openmx_core_native_malloc(state->resources,1,state->k_overlap_bytes,0);
        if (entry->matrix == NULL) return;
        state->k_overlap_used += state->k_overlap_bytes;
    }
    memcpy(entry->matrix, matrix, state->k_overlap_bytes);
    entry->k1 = k1;
    entry->k2 = k2;
    entry->k3 = k3;
}

static void BandNonCol_ClearOpenAccFreelists(void)
{
#pragma acc wait
    acc_clear_freelists();
}

static int BandNonCol_SerializeGpuSolverGpuTurns(void)
{
    return openmx_env_flag_at(OPENMX_ENV_GPUSOLVER_SERIAL_GPU_TURNS, 1);
}

static int BandNonCol_AutoGpuTurnLimit(BandNonColState *state, int requested, int n, int n2, int MaxN, int size_H1,
                                       int turn_mode, int owns_dense_rank, int myid0,
                                       int total_turns);

static int BandNonCol_GpuRequestedMaxTurns(void)
{
    /* By default every k-owner rank that shares the device may solve its
       k-point in the first wave (the plan clamps to the device-group size
       and to the total turn count); the variables only impose an explicit
       upper cap on the concurrency.  Matches the collinear path. */
    long requested = openmx_env_long_at(OPENMX_ENV_BAND_GPU_MAX_CONCURRENT_K,
                     openmx_env_long_at(OPENMX_ENV_BAND_GPU_TURN_GROUP,
                     openmx_env_long_at(OPENMX_ENV_BAND_GPU_MAX_RANKS_PER_DEVICE, (long)INT_MAX)));

    if (requested<1L){
        requested = 1L;
    }
    if ((long)INT_MAX<requested){
        requested = (long)INT_MAX;
    }
    return (int)requested;
}

static int BandNonCol_GpuTurnGroup(BandNonColState *state, int n, int n2, int MaxN, int size_H1,
                                   int turn_mode, int owns_dense_rank, int myid0,
                                   int total_turns)
{
    return BandNonCol_AutoGpuTurnLimit(state, BandNonCol_GpuRequestedMaxTurns(),
                                       n,n2,MaxN,size_H1,turn_mode,
                                       owns_dense_rank,myid0,total_turns);
}

static int BandNonCol_MaxConcurrentKGpuTurns(BandNonColState *state, int n, int n2, int MaxN, int size_H1,
                                             int turn_mode, int owns_dense_rank, int myid0,
                                             int total_turns)
{
    return BandNonCol_AutoGpuTurnLimit(state, BandNonCol_GpuRequestedMaxTurns(),
                                       n,n2,MaxN,size_H1,turn_mode,
                                       owns_dense_rank,myid0,total_turns);
}

static void BandNonCol_GpuSolver_EnsureMatrixCapacity(BandNonColState *state, int n)
{
    BandNonColGpuSolverWorkspace * w = &state->gpusolver_workspace;
    size_t nn;

    if (n<=0) BandNonCol_AbortWithMessage("Invalid GpuSolver matrix size in Band_DFT_NonCol.c.");

    BandNonCol_GpuSolver_Init(state);
    if (n<=w->matrix_dim) return;

    if (w->d_A    != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_A));
    if (w->d_W    != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_W));
    if (w->d_info != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_info));

    nn = (size_t)n*(size_t)n;
    wait_cudafunc(OpenMX_Resource_GpuMalloc(state->resources,(void**)&w->d_A,sizeof(dcomplex)*nn));
    wait_cudafunc(OpenMX_Resource_GpuMalloc(state->resources,(void**)&w->d_W,sizeof(double)*(size_t)n));
    wait_cudafunc(OpenMX_Resource_GpuMalloc(state->resources,(void**)&w->d_info,sizeof(int32_t)));

    w->matrix_dim = n;
}

static void BandNonCol_GpuSolver_EnsureWorkspace(BandNonColState *state, int n, int maxn)
{
    BandNonColGpuSolverWorkspace * w = &state->gpusolver_workspace;
    cusolverEigMode_t  jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t   uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range;
    double vl = 0.0;
    double vu = 0.0;
    int64_t h_meig = 0;
    size_t d_bytes = 0;
    size_t h_bytes = 0;

    if (n<=0 || maxn<=0 || n<maxn){
        BandNonCol_AbortWithMessage("Invalid GpuSolver eigensolver dimensions in Band_DFT_NonCol.c.");
    }

    BandNonCol_GpuSolver_EnsureMatrixCapacity(state, n);
    range = (n==maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;

    OpenMXSolverWait(cusolverDnXsyevdx_bufferSize(w->gpusolver,NULL,jobz,range,uplo,n,
                                               CUDA_C_64F,(cuDoubleComplex*)w->d_A,n,&vl,&vu,1L,maxn,&h_meig,
                                               CUDA_R_64F,w->d_W,CUDA_C_64F,&d_bytes,&h_bytes));

    if (w->d_work_bytes<d_bytes){
        if (w->d_work!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_work));
        w->d_work = NULL;
        if (0<d_bytes) wait_cudafunc(OpenMX_Resource_GpuMalloc(state->resources,(void**)&w->d_work,d_bytes));
        w->d_work_bytes = d_bytes;
    }

    if (h_bytes==0){
        if (w->h_work!=NULL) openmx_core_native_free(state->resources,w->h_work);
        w->h_work = NULL;
        w->h_work_bytes = 0;
    }
    else if (w->h_work_bytes<h_bytes){
        if (w->h_work!=NULL) openmx_core_native_free(state->resources,w->h_work);
        w->h_work = openmx_core_native_malloc(state->resources,1,h_bytes,0);
        if (w->h_work==NULL) BandNonCol_AbortWithMessage("Failed to allocate GpuSolver host workspace in Band_DFT_NonCol.c.");
        w->h_work_bytes = h_bytes;
    }
}

static void BandNonCol_GpuSolver_DenseZheevx_Device(BandNonColState *state, dcomplex *A, double *ko, int n, int maxn,
                                                   const char *where, int want_vectors)
{
    BandNonColGpuSolverWorkspace * w = &state->gpusolver_workspace;
    cusolverEigMode_t  jobz = want_vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t   uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range;
    double vl = 0.0;
    double vu = 0.0;
    int64_t h_meig = 0;
    int32_t info = 0;
    size_t nn = (size_t)n*(size_t)n;
    size_t d_bytes = 0;
    size_t h_bytes = 0;

    if (n<=0 || maxn<=0 || n<maxn){
        BandNonCol_AbortWithMessage("Invalid GpuSolver eigensolver dimensions in Band_DFT_NonCol.c.");
    }

    BandNonCol_GpuSolver_EnsureInfo(state);
    range = (n==maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;

#pragma acc wait
#pragma acc data present(A[0 : nn], ko[0 : n + 1])
#pragma acc host_data use_device(A, ko)
    {
        OpenMXSolverWait(cusolverDnXsyevdx_bufferSize(w->gpusolver,NULL,jobz,range,uplo,n,
                                                   CUDA_C_64F,(cuDoubleComplex*)A,n,&vl,&vu,1L,maxn,&h_meig,
                                                   CUDA_R_64F,ko+1,CUDA_C_64F,&d_bytes,&h_bytes));

        if (w->d_work_bytes<d_bytes){
            if (w->d_work!=NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,w->d_work));
            w->d_work = NULL;
            if (0<d_bytes) wait_cudafunc(OpenMX_Resource_GpuMalloc(state->resources,(void**)&w->d_work,d_bytes));
            w->d_work_bytes = d_bytes;
        }

        if (h_bytes==0){
            if (w->h_work!=NULL) openmx_core_native_free(state->resources,w->h_work);
            w->h_work = NULL;
            w->h_work_bytes = 0;
        }
        else if (w->h_work_bytes<h_bytes){
            if (w->h_work!=NULL) openmx_core_native_free(state->resources,w->h_work);
            w->h_work = openmx_core_native_malloc(state->resources,1,h_bytes,0);
            if (w->h_work==NULL) BandNonCol_AbortWithMessage("Failed to allocate GpuSolver host workspace in Band_DFT_NonCol.c.");
            w->h_work_bytes = h_bytes;
        }

        OpenMXSolverWait(cusolverDnXsyevdx(w->gpusolver,NULL,jobz,range,uplo,n,
                                        CUDA_C_64F,(cuDoubleComplex*)A,n,&vl,&vu,1L,maxn,&h_meig,
                                        CUDA_R_64F,ko+1,CUDA_C_64F,
                                        w->d_work,w->d_work_bytes,w->h_work,w->h_work_bytes,w->d_info));
        wait_cudafunc(cudaMemcpyAsync(&info,w->d_info,sizeof(int32_t),cudaMemcpyDeviceToHost,w->stream));
        wait_cudafunc(cudaStreamSynchronize(w->stream));
    }

    BandNonCol_GpuSolver_ReleaseDeviceWorkspace(state);

    if (info!=0){
        fprintf(stderr,"%s: gpusolver_Syevdx_Complex failed, info=%d\n",where,info);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1,1);
    }
    if (h_meig!=(int64_t)maxn){
        fprintf(stderr,"%s: gpusolver_Syevdx_Complex returned %lld eigenpairs, expected %d\n",
                where,(long long)h_meig,maxn);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1,1);
    }
}

static void BandNonCol_GEMMul8Zgemm_OpenACC(BandNonColState *state, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                            dcomplex const * A, dcomplex const * B, dcomplex * C)
{
    const int lda = (transa==CUBLAS_OP_N) ? m : k;
    const int ldb = (transb==CUBLAS_OP_N) ? k : n;

    BandNonCol_DMGpu_Init(state);
#pragma acc wait
#pragma acc data      present(A[0 : lda * ((transa==CUBLAS_OP_N) ? k : m)], \
                              B[0 : ldb * ((transb==CUBLAS_OP_N) ? n : k)], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        cuDoubleComplex const alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex const beta  = make_cuDoubleComplex(0.0, 0.0);
        OpenMXBlasWait(openmx_gemmul8Zgemm(state->dm_gpu_workspace.cublas, transa, transb, m, n, k, &alpha,
                                          (cuDoubleComplex const *)A, lda,
                                          (cuDoubleComplex const *)B, ldb,
                                          &beta,
                                          (cuDoubleComplex *)C, m));
        wait_cudafunc(cudaStreamSynchronize(state->dm_gpu_workspace.stream));
    }
}

/* Shared verdict of BandNonCol_GpuDiagFits, consulted by every dense-GPU
   branch so all helpers take the same GPU-vs-CPU path as the flags chosen at
   the top of Band_DFT_NonCol. */

/* GPU/CPU crossover of the noncollinear band dense eigensolver path,
   compared against n2 = 2n.  Like the collinear path (see
   Band_DFT_Col_GpuSwitchNum) the CPU fallback solves every k point twice
   with a serial ELPA per k-point world: measured on one RTX 5080 shared by
   8 ranks the GPU path is 2.2x at n2=832 and 1.4x at n2=416, with the
   crossover near n2=250-400, so 800 mirrors the collinear band default.
   OPENMX_BAND_NONCOL_GPU_SWITCH_NUM=<dim> overrides it.  Also consulted by
   DFT.c so the startup fallback notice reports the effective value. */
#define BAND_NONCOL_GPU_CPU_SWITCH_NUM 800

int Band_DFT_NonCol_GpuSwitchNum(void)
{
    long parsed = openmx_env_long_at(OPENMX_ENV_BAND_NONCOL_GPU_SWITCH_NUM, BAND_NONCOL_GPU_CPU_SWITCH_NUM);

    return 0 < parsed && parsed <= INT_MAX ? (int)parsed : BAND_NONCOL_GPU_CPU_SWITCH_NUM;
}

static int BandNonCol_UseDenseGpuMatrix(BandNonColState *state, int n, int n2)
{
    return (scf_eigen_lib_flag == GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum() <= n2 &&
            na_rows == n && na_cols == n && na_rows2 == n2 && na_cols2 == n2 &&
            state->gpu_dense_verdict);
}

static void BandNonCol_DenseTripleTransform_OpenACC(BandNonColState *state, int n, dcomplex *A, dcomplex *S, dcomplex *Work)
{
    size_t nn = (size_t)n * (size_t)n;

#pragma acc data copy(A[0 : nn]) copyin(S[0 : nn]) create(Work[0 : nn])
    {
if (!openmx_gpu_transform(OPENMX_GPU_ZERO,1,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(Work)),n,0,0.0)) {
#pragma acc parallel loop
        for (size_t idx = 0; idx < nn; idx++) {
            Work[idx].r = 0.0;
            Work[idx].i = 0.0;
        }
}

        BandNonCol_GEMMul8Zgemm_OpenACC(state, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, A, S, Work);

if (!openmx_gpu_transform(OPENMX_GPU_ZERO,1,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(A)),n,0,0.0)) {
#pragma acc parallel loop
        for (size_t idx = 0; idx < nn; idx++) {
            A[idx].r = 0.0;
            A[idx].i = 0.0;
        }
}

        BandNonCol_GEMMul8Zgemm_OpenACC(state, CUBLAS_OP_C, CUBLAS_OP_N, n, n, n, S, Work, A);
    }
#pragma acc wait
}

static void BandNonCol_DenseTripleTransform_PresentOpenACC(BandNonColState *state, int n, dcomplex *A, dcomplex *S, dcomplex *Work)
{
    int nn = n*n;

if (!openmx_gpu_transform(OPENMX_GPU_ZERO,1,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(Work)),n,0,0.0)) {
#pragma acc parallel loop present(Work[0 : nn])
    for (int idx=0; idx<nn; idx++){
        Work[idx].r = 0.0;
        Work[idx].i = 0.0;
    }
}

    BandNonCol_GEMMul8Zgemm_OpenACC(state, CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,A,S,Work);

if (!openmx_gpu_transform(OPENMX_GPU_ZERO,1,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(A)),n,0,0.0)) {
#pragma acc parallel loop present(A[0 : nn])
    for (int idx=0; idx<nn; idx++){
        A[idx].r = 0.0;
        A[idx].i = 0.0;
    }
}

    BandNonCol_GEMMul8Zgemm_OpenACC(state, CUBLAS_OP_C,CUBLAS_OP_N,n,n,n,S,Work,A);

#pragma acc wait
}

static void BandNonCol_SymmetrizeDenseHermitian(int n, dcomplex *A)
{
    for (int i=0; i<n; i++){
        A[(size_t)i + (size_t)i*(size_t)n].i = 0.0;
        for (int j=0; j<i; j++){
            size_t lij = (size_t)i + (size_t)j*(size_t)n;
            size_t uji = (size_t)j + (size_t)i*(size_t)n;
            double ar = 0.5*(A[lij].r + A[uji].r);
            double ai = 0.5*(A[lij].i - A[uji].i);
            A[lij].r = ar;
            A[lij].i = ai;
            A[uji].r =  ar;
            A[uji].i = -ai;
        }
    }
}

static void BandNonCol_SymmetrizeDenseHermitian_OpenACC(int n, dcomplex *A)
{
    int nn = n*n;

    if (openmx_gpu_transform(OPENMX_GPU_HERMITIAN,1,n,n,acc_deviceptr((void*)(A)),n,NULL,0,acc_deviceptr((void*)(A)),n,0,0.0)) return;

#pragma acc parallel loop present(A[0 : nn])
    for (int i=0; i<n; i++){
        A[(size_t)i + (size_t)i*(size_t)n].i = 0.0;
    }

#pragma acc parallel loop present(A[0 : nn])
    for (int idx=0; idx<nn; idx++){
        int j = idx/n;
        int i = idx - j*n;

        if (j<i){
            size_t lij = (size_t)i + (size_t)j*(size_t)n;
            size_t uji = (size_t)j + (size_t)i*(size_t)n;
            double ar = 0.5*(A[lij].r + A[uji].r);
            double ai = 0.5*(A[lij].i - A[uji].i);
            A[lij].r = ar;
            A[lij].i = ai;
            A[uji].r =  ar;
            A[uji].i = -ai;
        }
    }
}

/* Wave functions of a dense solve, on present arrays: for the MaxN solved
   states l

       Hs2[l + i n2] = sum_j Cs2[j + l n2] Ss2[i + j n2],

   with the rows from MaxN on cleared.  Ss2 = diag(S, S) holds the
   transformed overlap S (n x n) on both spin blocks, so instead of one
   n2 x n2 x n2 GEMM against Ss2 each spin half of Hs2 is one MaxN x n x n
   GEMM against S: half the work for all n2 states, and MaxN / n2 of that
   for MaxN of them.  Only the first MaxN columns of Cs2 need to be present;
   Hs2 is created on the device unless it already is. */
static void BandNonCol_DenseWavefunctions_PresentOpenACC(BandNonColState *state, int n, int n2, int MaxN, dcomplex *Cs2, dcomplex *S,
                                                        dcomplex *Hs2)
{
    size_t const nn = (size_t)n*(size_t)n;
    size_t const n2n2 = (size_t)n2*(size_t)n2;
    size_t const c_count = (size_t)n2*(size_t)MaxN;

    if (!acc_is_present(Hs2,sizeof(dcomplex)*n2n2)){
#pragma acc enter data create(Hs2[0 : n2n2])
    }

    if (MaxN<n2){
if (!openmx_gpu_transform(OPENMX_GPU_ZERO,1,n2-MaxN,n2,NULL,0,NULL,0,(dcomplex*)acc_deviceptr((void*)(Hs2))+MaxN,n2,0,0.0)) {
#pragma acc parallel loop collapse(2) present(Hs2[0 : n2n2])
        for (int col=0; col<n2; col++){
            for (int row=MaxN; row<n2; row++){
                Hs2[(size_t)row + (size_t)col*(size_t)n2].r = 0.0;
                Hs2[(size_t)row + (size_t)col*(size_t)n2].i = 0.0;
            }
        }
}

    }

    BandNonCol_DMGpu_Init(state);
#pragma acc wait
#pragma acc data      present(Cs2[0 : c_count], S[0 : nn], Hs2[0 : n2n2])
#pragma acc host_data use_device(Cs2, S, Hs2)
    {
        cuDoubleComplex const alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex const beta  = make_cuDoubleComplex(0.0, 0.0);

        for (int half=0; half<2; half++){
            size_t const off = (size_t)half*(size_t)n;

            OpenMXBlasWait(openmx_gemmul8Zgemm(state->dm_gpu_workspace.cublas, CUBLAS_OP_T, CUBLAS_OP_T,
                                              MaxN, n, n, &alpha,
                                              (cuDoubleComplex const *)Cs2 + off, n2,
                                              (cuDoubleComplex const *)S, n,
                                              &beta,
                                              (cuDoubleComplex *)Hs2 + off*(size_t)n2, n2));
        }
        wait_cudafunc(cudaStreamSynchronize(state->dm_gpu_workspace.stream));
    }
}

static void BandNonCol_DenseWavefunctions_OpenACC(BandNonColState *state, int n, int n2, int MaxN, dcomplex *Cs2, dcomplex *S,
                                                 dcomplex *Hs2)
{
    size_t const nn = (size_t)n*(size_t)n;
    size_t const n2n2 = (size_t)n2*(size_t)n2;
    size_t const c_count = (size_t)n2*(size_t)MaxN;

#pragma acc data copyin(Cs2[0 : c_count], S[0 : nn]) copyout(Hs2[0 : n2n2])
    {
        BandNonCol_DenseWavefunctions_PresentOpenACC(state, n,n2,MaxN,Cs2,S,Hs2);
    }
#pragma acc wait
}

static void BandNonCol_AddDense_OpenACC(int n, dcomplex *dst, const dcomplex *src)
{
    int nn = n*n;

    if (openmx_gpu_transform(OPENMX_GPU_ADD,1,n,n,acc_deviceptr((void*)(dst)),n,acc_deviceptr((void*)(src)),n,acc_deviceptr((void*)(dst)),n,0,0.0)) return;

#pragma acc parallel loop present(dst[0 : nn], src[0 : nn])
    for (int idx=0; idx<nn; idx++){
        dst[idx].r += src[idx].r;
        dst[idx].i += src[idx].i;
    }
}

static void BandNonCol_GpuSolver_DenseZheevx(BandNonColState *state, dcomplex *A, dcomplex *Z, double *ko, int n, int maxn,
                                            const char *where)
{
    BandNonColGpuSolverWorkspace * w = &state->gpusolver_workspace;
    cusolverEigMode_t  jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t   uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range;
    double vl = 0.0;
    double vu = 0.0;
    int64_t h_meig = 0;
    int32_t info = 0;
    int copy_cols;
    size_t nn;

    BandNonCol_GpuSolver_EnsureWorkspace(state, n,maxn);
    range = (n==maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;
    nn = (size_t)n*(size_t)n;

    wait_cudafunc(cudaMemcpyAsync(w->d_A,A,sizeof(dcomplex)*nn,cudaMemcpyHostToDevice,w->stream));

    OpenMXSolverWait(cusolverDnXsyevdx(w->gpusolver,NULL,jobz,range,uplo,n,
                                    CUDA_C_64F,(cuDoubleComplex*)w->d_A,n,&vl,&vu,1L,maxn,&h_meig,
                                    CUDA_R_64F,w->d_W,CUDA_C_64F,
                                    w->d_work,w->d_work_bytes,w->h_work,w->h_work_bytes,w->d_info));

    wait_cudafunc(cudaMemcpyAsync(A,w->d_A,sizeof(dcomplex)*nn,cudaMemcpyDeviceToHost,w->stream));
    wait_cudafunc(cudaMemcpyAsync(&ko[1],w->d_W,sizeof(double)*(size_t)maxn,cudaMemcpyDeviceToHost,w->stream));
    wait_cudafunc(cudaMemcpyAsync(&info,w->d_info,sizeof(int32_t),cudaMemcpyDeviceToHost,w->stream));
    wait_cudafunc(cudaStreamSynchronize(w->stream));

    if (info!=0){
        fprintf(stderr,"%s: gpusolver_Syevdx_Complex failed, info=%d\n",where,info);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1,1);
    }
    if (h_meig!=(int64_t)maxn){
        fprintf(stderr,"%s: gpusolver_Syevdx_Complex returned %lld eigenpairs, expected %d\n",
                where,(long long)h_meig,maxn);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1,1);
    }

    if (Z!=NULL){
        copy_cols = maxn;
        if (n<copy_cols) copy_cols = n;
        memcpy(Z,A,sizeof(dcomplex)*(size_t)n*(size_t)copy_cols);
    }
}

static void BandNonCol_AbortWithMessage(const char *msg)
{
    fprintf(stderr,"%s\n",msg);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1,1);
}

static size_t BandNonCol_CheckedAdd(size_t a, size_t b, const char *label)
{
    if (b>((size_t)-1)-a){
        char msg[256];
        snprintf(msg,sizeof(msg),"size overflow while estimating %s in Band_DFT_NonCol.c.",label);
        BandNonCol_AbortWithMessage(msg);
    }

    return a + b;
}

static size_t BandNonCol_CheckedMul(size_t a, size_t b, const char *label)
{
    if (a!=0 && b>((size_t)-1)/a){
        char msg[256];
        snprintf(msg,sizeof(msg),"size overflow while estimating %s in Band_DFT_NonCol.c.",label);
        BandNonCol_AbortWithMessage(msg);
    }

    return a*b;
}

static void BandNonCol_AddBytes(size_t *total, size_t bytes, const char *label)
{
    *total = BandNonCol_CheckedAdd(*total,bytes,label);
}

static size_t BandNonCol_ArrayBytes(size_t count, size_t elem_size, const char *label)
{
    return BandNonCol_CheckedMul(count,elem_size,label);
}

static size_t BandNonCol_MaxBytes(size_t a, size_t b)
{
    return (a<b) ? b : a;
}

static size_t BandNonCol_GpuSolverWorkFallbackBytes(int n)
{
    size_t nn = BandNonCol_CheckedMul((size_t)n,(size_t)n,"GpuSolver fallback workspace matrix count");
    size_t matrix_bytes = BandNonCol_ArrayBytes(nn,sizeof(dcomplex),"GpuSolver fallback workspace matrix");

    return BandNonCol_CheckedMul(matrix_bytes,4U,"GpuSolver fallback workspace");
}

static size_t BandNonCol_QueryGpuSolverWorkBytes(BandNonColState *state, int n, int maxn)
{
    /* The query costs a cusolver handle + stream create/destroy each time and
       runs several times per SCF iteration; memoize the (n, n)/(n2, MaxN)
       shapes the band path asks about. */

    cusolverDnHandle_t handle = NULL;
    cudaStream_t stream = NULL;
    cudaError_t cuda_status;
    cusolverStatus_t solver_status;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range = (n==maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;
    double vl = 0.0;
    double vu = 0.0;
    int64_t h_meig = 0;
    size_t d_bytes = 0;
    size_t h_bytes = 0;
    size_t fallback = BandNonCol_GpuSolverWorkFallbackBytes(n);

    for (int i=0; i<2; i++){
        if (state->QueryGpuSolverWorkBytes_cached_n[i]==n && state->QueryGpuSolverWorkBytes_cached_maxn[i]==maxn && state->QueryGpuSolverWorkBytes_cached_bytes[i]!=0U){
            return state->QueryGpuSolverWorkBytes_cached_bytes[i];
        }
    }

    cuda_status = OpenMX_Resource_StreamCreate(state->resources,&stream,cudaStreamNonBlocking);
    if (cuda_status!=cudaSuccess) return fallback;

    solver_status = OpenMX_Resource_SolverCreate(state->resources,&handle);
    if (solver_status==CUSOLVER_STATUS_SUCCESS){
        solver_status = cusolverDnSetStream(handle,stream);
    }

    if (solver_status==CUSOLVER_STATUS_SUCCESS){
        solver_status = cusolverDnXsyevdx_bufferSize(handle,NULL,jobz,range,uplo,n,
                                                     CUDA_C_64F,NULL,n,&vl,&vu,1L,maxn,&h_meig,
                                                     CUDA_R_64F,NULL,CUDA_C_64F,&d_bytes,&h_bytes);
    }

    if (handle!=NULL) OpenMX_Resource_SolverDestroy(state->resources,handle);
    OpenMX_Resource_StreamDestroy(state->resources,stream);

    if (solver_status!=CUSOLVER_STATUS_SUCCESS || d_bytes==0) return fallback;

    state->QueryGpuSolverWorkBytes_cached_n[state->QueryGpuSolverWorkBytes_cached_next]     = n;
    state->QueryGpuSolverWorkBytes_cached_maxn[state->QueryGpuSolverWorkBytes_cached_next]  = maxn;
    state->QueryGpuSolverWorkBytes_cached_bytes[state->QueryGpuSolverWorkBytes_cached_next] = d_bytes;
    state->QueryGpuSolverWorkBytes_cached_next               = (state->QueryGpuSolverWorkBytes_cached_next + 1) % 2;
    return d_bytes;
}

static size_t BandNonCol_GpuRankOverheadBytes(void)
{
    /* CUDA context, cublas/cusolver handles, and OpenACC pool slack that sit
       outside the explicitly estimated buffers; matches the collinear path
       (OPENMX_BAND_GPU_RANK_OVERHEAD_MB, default 256 MiB per owner rank). */
    return openmx_env_mib_bytes_at(OPENMX_ENV_BAND_GPU_RANK_OVERHEAD_MB, 256);
}

static size_t BandNonCol_RootDenseDeviceBytes(BandNonColState *state, int n, int n2, int MaxN, int size_H1, int need_evec)
{
    size_t nn = BandNonCol_CheckedMul((size_t)n,(size_t)n,"root dense n*n");
    size_t n2n2 = BandNonCol_CheckedMul((size_t)n2,(size_t)n2,"root dense n2*n2");
    size_t matrix_bytes = BandNonCol_ArrayBytes(nn,sizeof(dcomplex),"root dense n*n matrix");
    size_t matrix2_bytes = BandNonCol_ArrayBytes(n2n2,sizeof(dcomplex),"root dense n2*n2 matrix");
    size_t packed_count = (0<size_H1) ? (size_t)size_H1 : 0U;
    size_t construct_bytes = 0;
    size_t overlap_peak = 0;
    size_t hamiltonian_peak = 0;
    size_t wavefunction_peak = 0;
    size_t work_n = BandNonCol_QueryGpuSolverWorkBytes(state, n,n);
    size_t work_n2 = BandNonCol_QueryGpuSolverWorkBytes(state, n2,MaxN);
    size_t ko_bytes = BandNonCol_ArrayBytes((size_t)n2+1U,sizeof(double),"root dense eigenvalue vector");
    /* The GEMMul8 workspace is a single per-rank buffer that persists across
       the turns of one concurrency group, sized by its largest GEMM: the
       triple transforms are n x n x n, the eigenvector back transform is
       MaxN x n x n per spin half. */
    size_t gemmul8_bytes = BandNonCol_MaxBytes(openmx_gemmul8ZWorkspaceSize(n,n,n),
                                               need_evec ? openmx_gemmul8ZWorkspaceSize(MaxN,n,n) : 0U);
    size_t peak;

    BandNonCol_AddBytes(&construct_bytes,
                        BandNonCol_ArrayBytes(packed_count,sizeof(BandNonColConstructEntry),
                                              "root dense construct entries"),
                        "root dense construct entries");
    BandNonCol_AddBytes(&construct_bytes,
                        BandNonCol_ArrayBytes(packed_count,2U*sizeof(double),
                                              "root dense construct phases"),
                        "root dense construct phases");
    BandNonCol_AddBytes(&construct_bytes,
                        BandNonCol_ArrayBytes(packed_count,sizeof(double),
                                              "root dense packed matrix copy"),
                        "root dense packed matrix copy");

    BandNonCol_AddBytes(&overlap_peak,construct_bytes,"root dense overlap peak construct cache");
    BandNonCol_AddBytes(&overlap_peak,ko_bytes,"root dense overlap peak eigenvalues");
    BandNonCol_AddBytes(&overlap_peak,matrix_bytes,"root dense overlap matrix");
    BandNonCol_AddBytes(&overlap_peak,work_n,"root dense overlap GpuSolver workspace");

    BandNonCol_AddBytes(&hamiltonian_peak,construct_bytes,"root dense Hamiltonian peak construct cache");
    BandNonCol_AddBytes(&hamiltonian_peak,ko_bytes,"root dense Hamiltonian peak eigenvalues");
    BandNonCol_AddBytes(&hamiltonian_peak,
                        BandNonCol_CheckedMul(matrix_bytes,5U,"root dense Hamiltonian n*n matrices"),
                        "root dense Hamiltonian n*n matrices");
    BandNonCol_AddBytes(&hamiltonian_peak,matrix2_bytes,"root dense Hamiltonian n2*n2 matrix");
    BandNonCol_AddBytes(&hamiltonian_peak,work_n2,"root dense Hamiltonian GpuSolver workspace");
    BandNonCol_AddBytes(&hamiltonian_peak,sizeof(int32_t),"root dense Hamiltonian GpuSolver info");

    if (need_evec){
        BandNonCol_AddBytes(&wavefunction_peak,construct_bytes,"root dense wavefunction peak construct cache");
        BandNonCol_AddBytes(&wavefunction_peak,ko_bytes,"root dense wavefunction peak eigenvalues");
        BandNonCol_AddBytes(&wavefunction_peak,matrix_bytes,"root dense wavefunction overlap matrix");
        BandNonCol_AddBytes(&wavefunction_peak,
                            BandNonCol_CheckedMul(matrix2_bytes,2U,"root dense wavefunction n2*n2 matrices"),
                            "root dense wavefunction n2*n2 matrices");
    }

    peak = BandNonCol_MaxBytes(overlap_peak,BandNonCol_MaxBytes(hamiltonian_peak,wavefunction_peak));
    peak = BandNonCol_CheckedAdd(peak,gemmul8_bytes,"root dense GEMMul8 workspace");
    peak = BandNonCol_CheckedAdd(peak,BandNonCol_RefineBasisBytes(state, n2),"refined eigensolver basis");
    peak = BandNonCol_CheckedAdd(peak,BandNonCol_GpuRankOverheadBytes(),"root dense rank overhead");
    return peak;
}

/* turn_mode values understood by BandNonCol_AutoGpuTurnLimit */
#define BANDNONCOL_GPU_TURN_EIGVALS 0
#define BANDNONCOL_GPU_TURN_EIGVECS 1
#define BANDNONCOL_GPU_TURN_DM      2

static size_t BandNonCol_RootDenseDMDeviceBytes(int n, int n2, int MaxN, int size_H1)
{
    /* The OpenACC DM accumulation holds the dense eigenvectors, the per-pair
       phases and occupations, and one chunk of index/output buffers on the
       device; no GEMM runs in this phase. */
    const size_t chunk = 131072U;
    size_t n2n2 = BandNonCol_CheckedMul((size_t)n2,(size_t)n2,"DM dense n2*n2");
    size_t packed_count = (0<size_H1) ? (size_t)size_H1 : 0U;
    size_t bytes = BandNonCol_ArrayBytes(n2n2,sizeof(dcomplex),"DM dense eigenvectors");

    (void)n;
    BandNonCol_AddBytes(&bytes,
                        BandNonCol_ArrayBytes(packed_count,2U*sizeof(double),"DM dense phases"),
                        "DM dense phases");
    BandNonCol_AddBytes(&bytes,
                        BandNonCol_ArrayBytes(chunk,3U*sizeof(int)+8U*sizeof(double),"DM dense chunk buffers"),
                        "DM dense chunk buffers");
    BandNonCol_AddBytes(&bytes,
                        BandNonCol_ArrayBytes((size_t)MaxN,2U*sizeof(double),"DM dense occupations"),
                        "DM dense occupations");
    BandNonCol_AddBytes(&bytes,BandNonCol_GpuRankOverheadBytes(),"DM dense rank overhead");
    return bytes;
}

static size_t BandNonCol_RootDenseReserveBytes(size_t total_bytes)
{
    /* The per-rank estimate now covers the GEMMul8 workspace and OpenACC
       transients explicitly, so the reserve only shields the CUDA context and
       other GPU subsystems; matches the collinear path
       (max(total/10, 1536 MiB), raisable via OPENMX_BAND_GPU_RESERVE_MB). */
    size_t reserve = total_bytes/10U;
    const size_t minimum = 1536ULL*1024ULL*1024ULL;
    const size_t requested = openmx_env_mib_bytes_at(OPENMX_ENV_BAND_GPU_RESERVE_MB, 0);

    if (reserve<minimum) reserve = minimum;
    if (reserve<requested) reserve = requested;
    return reserve;
}

static int BandNonCol_GpuTurnMemoryFits(size_t free_bytes, size_t total_bytes,
                                        size_t required_bytes, int concurrent_ranks,
                                        size_t *group_required, size_t *reserve_bytes)
{
    size_t reserve;
    size_t required;

    if (concurrent_ranks<1) concurrent_ranks = 1;

    if ((size_t)concurrent_ranks!=0U &&
        required_bytes>SIZE_MAX/(size_t)concurrent_ranks){
        if (group_required!=NULL) *group_required = SIZE_MAX;
        if (reserve_bytes!=NULL) *reserve_bytes = 0U;
        return 0;
    }

    required = required_bytes*(size_t)concurrent_ranks;
    reserve = BandNonCol_RootDenseReserveBytes(total_bytes);

    if (group_required!=NULL) *group_required = required;
    if (reserve_bytes!=NULL) *reserve_bytes = reserve;

    return (required<=free_bytes && reserve<=free_bytes-required);
}

/* Collectively decide whether even ONE k-point's dense GPU solve fits on the
   device.  The adaptive turn ladder can only reduce the concurrency to one
   k-owner rank; when a single rank's requirement (the n2 dense matrices plus
   the cusolver workspace at the size cusolverDnXsyevdx_bufferSize reports,
   the GEMMul8 workspace and the staging transients) does not fit, committing
   to the GPU path would exhaust device memory. Ordinary gpusolver instead
   selects host-backed MAGMA within the configured pool, or exits. The verdict is
   collective over mpi_comm_level1 (the branch controls barrier and
   communicator structure) and sticky until the next SCF restart or a change
   of the matrix size.
   OPENMX_BAND_GPU_DIAG=0 explicitly selects the legacy CPU branch;
   neither =1 nor forced MAGMA bypasses memory admission. */
static int BandNonCol_GpuDiagFits(BandNonColState *state, int SCF_iter, int n, int n2, int MaxN, int size_H1, int myid0, int total_kpoints)
{

    unsigned long long my_required = 0ULL, req_max = 0ULL;
    unsigned long long my_free = ULLONG_MAX, free_min = 0ULL;
    int my_fit = 1;
    int fit = 0;

    if (openmx_env_flag_at(OPENMX_ENV_BAND_GPU_DIAG, -1)==0 && !openmx_mgpu_host_required()) {
        state->gpu_dense_verdict=0; return 0;
    }
    if (SCF_iter==1 || state->GpuDiagFits_verdict_n2!=n2) state->mgpu_host_dense=0;
    if (openmx_mgpu_host_required()) goto host_capacity;

    if (state->GpuDiagFits_verdict!=-1 && state->GpuDiagFits_verdict_n2==n2 && 1<SCF_iter){
        state->gpu_dense_verdict = state->GpuDiagFits_verdict;
        return state->GpuDiagFits_verdict;
    }

    if (Set_Hamiltonian_OpenACC_Rank_Is_Selected()){
        int cuda_device = -1;

        if (cudaGetDevice(&cuda_device)==cudaSuccess){
            acc_wait_all();
            if (cudaDeviceSynchronize()==cudaSuccess){
                acc_clear_freelists();
            }
        }
    }
    MPI_Barrier(mpi_comm_level1);

    if (Set_Hamiltonian_OpenACC_Rank_Is_Selected()){
        int cuda_device = -1;

        if (cudaGetDevice(&cuda_device)==cudaSuccess){
            size_t required = BandNonCol_MaxBytes(
                BandNonCol_RootDenseDeviceBytes(state, n,n2,MaxN,size_H1,1),
                BandNonCol_RootDenseDMDeviceBytes(n,n2,MaxN,size_H1));
            size_t free_bytes = 0U, total_bytes = 0U;

            if (required==0U || required==SIZE_MAX ||
                OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess){
                (void)cudaGetLastError();
                my_fit = 0;
            }
            else {
                size_t reserve = BandNonCol_RootDenseReserveBytes(total_bytes);

                my_required = (unsigned long long)required;
                my_free = (unsigned long long)free_bytes;
                my_fit = (required<=free_bytes && reserve<=free_bytes-required);
            }
        }
        else {
            (void)cudaGetLastError();
            my_fit = 0;
        }
    }

    MPI_Allreduce(&my_fit,&fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);
    MPI_Allreduce(&my_required,&req_max,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,mpi_comm_level1);
    MPI_Allreduce(&my_free,&free_min,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,mpi_comm_level1);

    if (!fit && gpusolver2_flag) {
        if (myid0==Host_ID) printf("<BandNonCol> GPU dense memory admission declined; falling back to ELPA/ScaLAPACK.\n");
        state->GpuDiagFits_verdict=0; state->GpuDiagFits_verdict_n2=n2; state->gpu_dense_verdict=0;
        return 0;
    }
    if (!fit) {
host_capacity:
        BandNonCol_GpuSolver_Destroy(state);
        openmx_gemmul8ReleaseWorkspaces();
        acc_wait_all();
        acc_clear_freelists();
        MPI_Barrier(mpi_comm_level1);
        if (myid0==Host_ID) {
            fit = openmx_mgpu_host_plan(total_kpoints,n2,1)!=0;
            if (fit) printf("<OpenMX_MGPU> Band NC n=%d: %s MAGMA, host-backed dense matrices; scf.gpu.num upper bound respected.\n",
                            n2,openmx_mgpu_host_required()?"forced":"cuSOLVER memory limit ->");
            else printf("<OpenMX_MGPU> Band NC n=%d: %s; using the CPU eigensolver and matrix products.\n",
                        n2,openmx_mgpu_host_error());
            fflush(stdout);
        }
        MPI_Bcast(&fit,1,MPI_INT,Host_ID,mpi_comm_level1);
        state->mgpu_host_dense=fit;
    }

    state->GpuDiagFits_verdict = fit;
    state->GpuDiagFits_verdict_n2 = n2;
    state->gpu_dense_verdict = fit;
    return state->GpuDiagFits_verdict;
}

static int BandNonCol_AutoGpuTurnLimit(BandNonColState *state, int requested, int n, int n2, int MaxN, int size_H1,
                                       int turn_mode, int owns_dense_rank, int myid0,
                                       int total_turns)
{
    if (state->mgpu_host_dense) return 1;
    MPI_Comm node_comm;
    MPI_Comm device_comm = MPI_COMM_NULL;
    int local_limit = requested;
    int global_limit = requested;
    int cuda_ok = 0;
    int cuda_device = -1;
    size_t required_bytes = 0;

    if (requested<=1){
        return 1;
    }

    if (owns_dense_rank){
        cudaError_t cuda_status = cudaGetDevice(&cuda_device);

        if (cuda_status==cudaSuccess){
            required_bytes = (turn_mode==BANDNONCOL_GPU_TURN_DM)
                ? BandNonCol_RootDenseDMDeviceBytes(n,n2,MaxN,size_H1)
                : BandNonCol_RootDenseDeviceBytes(state, n,n2,MaxN,size_H1,
                                                  turn_mode==BANDNONCOL_GPU_TURN_EIGVECS);
        }

        if (cuda_status==cudaSuccess && 0<required_bytes && required_bytes!=SIZE_MAX){
            cuda_ok = 1;
        }
        else {
            local_limit = 1;
        }
    }

    MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
    MPI_Comm_split(node_comm,cuda_ok ? cuda_device : MPI_UNDEFINED,0,&device_comm);
    MPI_Comm_free(&node_comm);

    if (cuda_ok && device_comm!=MPI_COMM_NULL){
        int device_rank = 0;
        int device_ranks = 0;
        int memory_ok = 0, group_memory_ok = 0;
        unsigned long long local_free = 0ULL, group_free = 0ULL;
        unsigned long long local_total = 0ULL, group_total = 0ULL;
        cudaError_t cuda_status;

        MPI_Comm_rank(device_comm,&device_rank);
        MPI_Comm_size(device_comm,&device_ranks);

        /* Trim the OpenACC freelists so cudaMemGetInfo reports the memory a
           new concurrency group can actually claim, and let every owner rank
           finish trimming before anyone measures. */
        acc_wait_all();
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status==cudaSuccess){
            acc_clear_freelists();
        }
        MPI_Barrier(device_comm);

        if (cuda_status==cudaSuccess){
            size_t free_bytes = 0U, total_bytes = 0U;
            cuda_status = OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes);
            if (cuda_status==cudaSuccess){
                memory_ok = 1;
                local_free = (unsigned long long)free_bytes;
                local_total = (unsigned long long)total_bytes;
            }
        }

        MPI_Allreduce(&memory_ok,&group_memory_ok,1,MPI_INT,MPI_MIN,device_comm);
        MPI_Allreduce(&local_free,&group_free,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,device_comm);
        MPI_Allreduce(&local_total,&group_total,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,device_comm);

        if (group_memory_ok){
            /* Start from every k-owner rank that shares this GPU (or the
               explicit cap) and step down one rank at a time until the wave
               plus the reserve fits into the group's free memory. */
            const int cmax = (device_ranks<requested) ? device_ranks : requested;
            int selected_limit = 0;
            size_t selected_peak = 0U;
            size_t selected_reserve = 0U;

            {
                /* Publish the memory this device needs to run the full
                   k-owner concurrency, so long-lived GPU caches
                   (Set_Hamiltonian's resident tables) leave room for it. */
                size_t full_required = 0U;
                size_t full_reserve = 0U;
                const char *const need_name[3] =
                    {"band_noncol_ev","band_noncol_vec","band_noncol_dm"};
                int mode_slot = (0<=turn_mode && turn_mode<3) ? turn_mode : 1;

                (void)BandNonCol_GpuTurnMemoryFits(SIZE_MAX,(size_t)group_total,
                                                   required_bytes,cmax,
                                                   &full_required,&full_reserve);
                if (full_required!=SIZE_MAX && full_required<=SIZE_MAX-full_reserve){
                    OpenMX_GpuPhaseNeed_Register(need_name[mode_slot],
                                                 full_required+full_reserve);
                }
            }

            for (int c=cmax; 1<=c; c--){
                size_t group_required = 0U;
                size_t reserve_bytes = 0U;

                if (BandNonCol_GpuTurnMemoryFits((size_t)group_free,(size_t)group_total,
                                                 required_bytes,c,
                                                 &group_required,&reserve_bytes)){
                    selected_limit = c;
                    selected_peak = group_required;
                    selected_reserve = reserve_bytes;
                    break;
                }
            }

            if (selected_limit==0){
                selected_limit = 1;
                selected_peak = required_bytes;
                selected_reserve = BandNonCol_RootDenseReserveBytes((size_t)group_total);
                if (device_rank==0 && 2<=level_stdout){
                    fprintf(stderr,
                            "<Band_DFT_NonCol> GPU device %d cannot retain the full reserve; "
                            "forcing one GPU k-owner rank.\n",cuda_device);
                    fflush(stderr);
                }
            }

            /* When every rank sharing the device fits at once, the turn
               groups no longer bound the memory: run all turns as one group
               so no barrier interrupts the concurrent solves.  The effective
               concurrency stays min(ranks, turns) either way. */
            if (selected_limit==device_ranks && device_ranks<=requested &&
                selected_limit<total_turns){
                local_limit = total_turns;
            }
            else {
                local_limit = selected_limit;
            }

            if (device_rank==0 && 2<=level_stdout){

                const char *const mode_label[3] =
                    {"eigenvalues only","eigenvectors","density matrix"};
                int mode_slot = (0<=turn_mode && turn_mode<3) ? turn_mode : 1;

                if (state->AutoGpuTurnLimit_last_limit[mode_slot]!=selected_limit ||
                    state->AutoGpuTurnLimit_last_device_ranks[mode_slot]!=device_ranks){
                    printf("<Band_DFT_NonCol> GPU device %d: %d k-owner rank(s), "
                           "GPU concurrency=%d (%s), turn group=%d, per-rank=%.3f GiB, "
                           "peak=%.3f GiB, free=%.3f GiB, reserve=%.3f GiB\n",
                           cuda_device,device_ranks,selected_limit,
                           mode_label[mode_slot],local_limit,
                           (double)required_bytes/(1024.0*1024.0*1024.0),
                           (double)selected_peak/(1024.0*1024.0*1024.0),
                           (double)group_free/(1024.0*1024.0*1024.0),
                           (double)selected_reserve/(1024.0*1024.0*1024.0));
                    fflush(stdout);
                    state->AutoGpuTurnLimit_last_limit[mode_slot] = selected_limit;
                    state->AutoGpuTurnLimit_last_device_ranks[mode_slot] = device_ranks;
                }
            }
        }
        else {
            local_limit = 1;
            if (device_rank==0 && 2<=level_stdout){
                fprintf(stderr,
                        "<Band_DFT_NonCol> failed to query common GPU memory; "
                        "using one GPU k-owner rank at a time.\n");
                fflush(stderr);
            }
        }

        MPI_Comm_free(&device_comm);
    }

    MPI_Allreduce(&local_limit,&global_limit,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    return global_limit;
}

static int BandNonCol_RootDenseParallelKWorldsFit(BandNonColState *state, int n, int n2, int MaxN, int size_H1,
                                                  int myid0, int myworld2, int Num_Comm_World2,
                                                  int *Comm_World_StartID2)
{
    MPI_Comm node_comm;
    MPI_Comm device_comm = MPI_COMM_NULL;
    int potential_owner;
    int selected_owner;
    int cuda_ok = 0;
    int cuda_device = -1;
    int local_fit = 1;
    int global_fit = 1;
    size_t required_bytes = 0;
    size_t reserve_bytes = 0;
    size_t free_bytes = 0;
    size_t total_bytes = 0;

    if (Num_Comm_World2<=1) return 1;

    potential_owner =
      (0<=myworld2 && myworld2<Num_Comm_World2 &&
       myid0==Comm_World_StartID2[myworld2]);
    selected_owner = potential_owner && Set_Hamiltonian_OpenACC_Rank_Is_Selected();

    if (potential_owner && !selected_owner){
        if (myid0==Comm_World_StartID2[myworld2] && 2<=level_stdout){
            fprintf(stderr,
                    "<Band>  Rank %d is a non-collinear dense GpuSolver k-world owner, "
                    "but it is not selected for CUDA/OpenACC; using serialized k-worlds.\n",
                    myid0);
            fflush(stderr);
        }
        local_fit = 0;
    }

    if (selected_owner){
        cudaError_t cuda_status;

        cuda_status = cudaGetDevice(&cuda_device);
        if (cuda_status==cudaSuccess){
            required_bytes = BandNonCol_RootDenseDeviceBytes(state, n,n2,MaxN,size_H1,1);
            cuda_status = OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes);
        }

        if (cuda_status==cudaSuccess){
            reserve_bytes = BandNonCol_RootDenseReserveBytes(total_bytes);
            cuda_ok = 1;
        }
        else {
            if (2<=level_stdout) {
                fprintf(stderr,
                        "<Band>  Rank %d failed to query CUDA memory for non-collinear dense GpuSolver (%s); "
                        "using serialized k-worlds.\n",
                        myid0,cudaGetErrorString(cuda_status));
                fflush(stderr);
            }
            local_fit = 0;
        }
    }

    MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
    MPI_Comm_split(node_comm,cuda_ok ? cuda_device : MPI_UNDEFINED,0,&device_comm);
    MPI_Comm_free(&node_comm);

    if (cuda_ok){
        unsigned long long local_required = (unsigned long long)required_bytes;
        unsigned long long group_required = 0ULL;
        unsigned long long local_free = (unsigned long long)free_bytes;
        unsigned long long group_free = 0ULL;
        unsigned long long local_reserve = (unsigned long long)reserve_bytes;
        unsigned long long group_reserve = 0ULL;
        int device_rank = 0;
        int device_ranks = 0;

        MPI_Allreduce(&local_required,&group_required,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,device_comm);
        MPI_Allreduce(&local_free,&group_free,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,device_comm);
        MPI_Allreduce(&local_reserve,&group_reserve,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,device_comm);
        MPI_Comm_rank(device_comm,&device_rank);
        MPI_Comm_size(device_comm,&device_ranks);

        if (group_free<group_required || group_free-group_required<group_reserve){
            local_fit = 0;
            if (device_rank==0 && 2<=level_stdout){
                printf("<Band>  Serializing non-collinear dense GpuSolver k-worlds: GPU device %d is shared by %d owner rank(s), "
                       "free %.3f MiB, need %.3f MiB plus %.3f MiB reserve.\n",
                       cuda_device,device_ranks,
                       (double)group_free/(1024.0*1024.0),
                       (double)group_required/(1024.0*1024.0),
                       (double)group_reserve/(1024.0*1024.0));
                fflush(stdout);
            }
        }

        MPI_Comm_free(&device_comm);
    }

    MPI_Allreduce(&local_fit,&global_fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    return global_fit;
}

static unsigned long long BandNonCol_HashInt(unsigned long long h, int value)
{
    h ^= (unsigned long long)(unsigned int)value;
    h *= 1099511628211ULL;
    return h;
}

static void BandNonCol_ConstructCache_Reset(BandNonColState *state)
{
    BandNonColConstructCache *cache = &state->construct_cache;

    if (cache->dense_device_valid){
        BandNonColConstructEntry *entries = cache->dense_entries;
        double *phase_r = cache->phase_r;
        double *phase_i = cache->phase_i;
        int dense_count = cache->dense_count;
        int phase_count = cache->phase_count;

        if (entries!=NULL && 0<dense_count){
#pragma acc exit data delete(entries[0 : dense_count])
        }
        if (phase_r!=NULL && phase_i!=NULL && 0<phase_count){
#pragma acc exit data delete(phase_r[0 : phase_count], phase_i[0 : phase_count])
        }
        BandNonCol_ClearOpenAccFreelists();
    }

    openmx_core_native_free(state->resources,cache->phase_l1);
    openmx_core_native_free(state->resources,cache->phase_l2);
    openmx_core_native_free(state->resources,cache->phase_l3);
    openmx_core_native_free(state->resources,cache->phase_r);
    openmx_core_native_free(state->resources,cache->phase_i);
    openmx_core_native_free(state->resources,cache->dense_entries);
    memset(cache,0,sizeof(*cache));
}

static unsigned long long BandNonCol_ConstructFingerprint(int *order_GA, int *MP)
{
    unsigned long long h = 1469598103934665603ULL;

    h = BandNonCol_HashInt(h,atomnum);
    for (int AN=1; AN<=atomnum; AN++){
        int GA_AN = order_GA[AN];
        h = BandNonCol_HashInt(h,GA_AN);
        h = BandNonCol_HashInt(h,MP[GA_AN]);
        h = BandNonCol_HashInt(h,WhatSpecies[GA_AN]);
        h = BandNonCol_HashInt(h,Spe_Total_CNO[WhatSpecies[GA_AN]]);
        h = BandNonCol_HashInt(h,FNAN[GA_AN]);
        for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            h = BandNonCol_HashInt(h,natn[GA_AN][LB_AN]);
            h = BandNonCol_HashInt(h,ncn[GA_AN][LB_AN]);
        }
    }

    return h;
}

static void BandNonCol_ConstructCache_Ensure(BandNonColState *state, int *order_GA, int *MP, int n)
{
    BandNonColConstructCache *cache = &state->construct_cache;
    unsigned long long fingerprint = BandNonCol_ConstructFingerprint(order_GA,MP);
    int dense_count = 0;
    int phase_count = 0;

    if (cache->valid && cache->n==n && cache->fingerprint==fingerprint) return;

    BandNonCol_ConstructCache_Reset(state);

    for (int AN=1; AN<=atomnum; AN++){
        int GA_AN = order_GA[AN];
        int wanA = WhatSpecies[GA_AN];
        int tnoA = Spe_Total_CNO[wanA];

        for (int i=0; i<tnoA; i++){
            for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
                int GB_AN = natn[GA_AN][LB_AN];
                int wanB = WhatSpecies[GB_AN];
                int tnoB = Spe_Total_CNO[wanB];

                phase_count++;
                dense_count += tnoB;
            }
        }
    }

    cache->dense_entries = (BandNonColConstructEntry*)openmx_core_native_malloc(state->resources,1,sizeof(BandNonColConstructEntry)*(size_t)(dense_count+1),0);
    cache->phase_l1 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(size_t)(phase_count+1),0);
    cache->phase_l2 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(size_t)(phase_count+1),0);
    cache->phase_l3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(size_t)(phase_count+1),0);
    cache->phase_r = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(size_t)(phase_count+1),0);
    cache->phase_i = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(size_t)(phase_count+1),0);

    if (cache->dense_entries==NULL || cache->phase_l1==NULL || cache->phase_l2==NULL ||
        cache->phase_l3==NULL || cache->phase_r==NULL || cache->phase_i==NULL){
        BandNonCol_ConstructCache_Reset(state);
        BandNonCol_AbortWithMessage("Failed to allocate Construct_Band_DenseMs cache in Band_DFT_NonCol.c.");
    }

    dense_count = 0;
    phase_count = 0;
    int h_index = 0;

    for (int AN=1; AN<=atomnum; AN++){
        int GA_AN = order_GA[AN];
        int wanA = WhatSpecies[GA_AN];
        int tnoA = Spe_Total_CNO[wanA];
        int Anum = MP[GA_AN];

        for (int i=0; i<tnoA; i++){
            int ig = Anum + i;

            for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
                int GB_AN = natn[GA_AN][LB_AN];
                int Rn = ncn[GA_AN][LB_AN];
                int wanB = WhatSpecies[GB_AN];
                int tnoB = Spe_Total_CNO[wanB];
                int Bnum = MP[GB_AN];
                int phase_index = phase_count++;

                cache->phase_l1[phase_index] = atv_ijk[Rn][1];
                cache->phase_l2[phase_index] = atv_ijk[Rn][2];
                cache->phase_l3[phase_index] = atv_ijk[Rn][3];

                for (int j=0; j<tnoB; j++, h_index++){
                    int jg = Bnum + j;
                    BandNonColConstructEntry *entry = &cache->dense_entries[dense_count++];

                    entry->m_index = h_index;
                    entry->dense_index = (jg-1)*n + (ig-1);
                    entry->phase_index = phase_index;
                }
            }
        }
    }

    cache->valid = 1;
    cache->n = n;
    cache->fingerprint = fingerprint;
    cache->dense_count = dense_count;
    cache->h_count = h_index;
    cache->phase_count = phase_count;
}

/* Return the physical pages of a device-resident staging buffer to the
   kernel while keeping its virtual range (the present-table key).  Same
   pattern as Set_Hamiltonian_ME_DiscardHostPages. */
static void BandNonCol_DiscardHostPages(void *ptr, size_t bytes)
{
    const size_t page = 4096;
    uintptr_t lo = ((uintptr_t)ptr + page - 1) & ~(uintptr_t)(page - 1);
    uintptr_t hi = ((uintptr_t)ptr + bytes) & ~(uintptr_t)(page - 1);

    if (ptr == NULL || hi <= lo) return;
    (void)madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
}

static void BandNonCol_ConstructCache_EnsureDenseDevice(BandNonColState *state)
{
    BandNonColConstructCache *cache = &state->construct_cache;

    if (cache->dense_device_valid) return;

    if (0<cache->dense_count){
        BandNonColConstructEntry *entries = cache->dense_entries;
        int dense_count = cache->dense_count;
#pragma acc enter data copyin(entries[0 : dense_count])
        /* only the device kernels read the entry table from here on (the
           CPU construct fallback is unreachable once this path is live);
           the host mirror is ~100 MB per k-owner, so give the pages back */
        BandNonCol_DiscardHostPages(entries, sizeof(BandNonColConstructEntry)*(size_t)dense_count);
    }

    if (0<cache->phase_count){
        double *phase_r = cache->phase_r;
        double *phase_i = cache->phase_i;
        int phase_count = cache->phase_count;
#pragma acc enter data create(phase_r[0 : phase_count], phase_i[0 : phase_count])
    }

    cache->dense_device_valid = 1;
}

static void BandNonCol_ConstructDenseMs_OpenACC(BandNonColState *state, int cpx_flag, int n, double k1, double k2, double k3,
                                                const double *M2, dcomplex *Ms)
{
    BandNonColConstructCache *cache = &state->construct_cache;
    BandNonColConstructEntry *entries = cache->dense_entries;
    double *phase_r = cache->phase_r;
    double *phase_i = cache->phase_i;
    int dense_count = cache->dense_count;
    int phase_count = cache->phase_count;
    int h_count = cache->h_count;
    int matrix_count = n*n;

    BandNonCol_ConstructCache_EnsureDenseDevice(state);

    for (int p=0; p<phase_count; p++){
        double kRn = k1*(double)cache->phase_l1[p]
                   + k2*(double)cache->phase_l2[p]
                   + k3*(double)cache->phase_l3[p];
        phase_i[p] = sin(2.0*PI*kRn);
        phase_r[p] = cos(2.0*PI*kRn);
    }

    if (0<phase_count){
#pragma acc update device(phase_r[0 : phase_count], phase_i[0 : phase_count])
    }

    if (!acc_is_present(Ms,sizeof(dcomplex)*(size_t)matrix_count)){
#pragma acc enter data create(Ms[0 : matrix_count])
    }

#pragma acc data copyin(M2[0 : h_count]) present(Ms[0 : matrix_count], entries[0 : dense_count], phase_r[0 : phase_count], phase_i[0 : phase_count])
    {
        if (!openmx_gpu_scatter(n,dense_count,3,acc_deviceptr((void*)(entries)),h_count,acc_deviceptr((void*)(M2)),phase_count,acc_deviceptr((void*)(phase_r)),acc_deviceptr((void*)(phase_i)),cpx_flag,acc_deviceptr((void*)(Ms)))) {
if (!openmx_gpu_transform(OPENMX_GPU_ZERO,1,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(Ms)),n,0,0.0)) {
#pragma acc parallel loop
        for (int idx=0; idx<matrix_count; idx++){
            Ms[idx].r = 0.0;
            Ms[idx].i = 0.0;
        }
}

#pragma acc parallel loop
        for (int idx=0; idx<dense_count; idx++){
            int m_index = entries[idx].m_index;
            int dense_index = entries[idx].dense_index;
            int phase_index = entries[idx].phase_index;
            double pr = phase_r[phase_index];
            double pi = phase_i[phase_index];
            double val_r;
            double val_i;

            if (cpx_flag==0){
                val_r = M2[m_index]*pr;
                val_i = M2[m_index]*pi;
            }
            else {
                val_r = -M2[m_index]*pi;
                val_i =  M2[m_index]*pr;
            }

#pragma acc atomic update
            Ms[dense_index].r += val_r;
#pragma acc atomic update
            Ms[dense_index].i += val_i;
        }
        }
    }

#pragma acc wait
}

static unsigned long long BandNonCol_DMLayoutFingerprint(int *MP, int n)
{
    unsigned long long h = 1469598103934665603ULL;

    h = BandNonCol_HashInt(h,n);
    h = BandNonCol_HashInt(h,atomnum);

    for (int GA_AN=1; GA_AN<=atomnum; GA_AN++){
        int wanA = WhatSpecies[GA_AN];

        h = BandNonCol_HashInt(h,GA_AN);
        h = BandNonCol_HashInt(h,wanA);
        h = BandNonCol_HashInt(h,Spe_Total_CNO[wanA]);
        h = BandNonCol_HashInt(h,MP[GA_AN]);
        h = BandNonCol_HashInt(h,FNAN[GA_AN]);

        for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            int GB_AN = natn[GA_AN][LB_AN];
            int wanB  = WhatSpecies[GB_AN];

            h = BandNonCol_HashInt(h,GB_AN);
            h = BandNonCol_HashInt(h,ncn[GA_AN][LB_AN]);
            h = BandNonCol_HashInt(h,wanB);
            h = BandNonCol_HashInt(h,Spe_Total_CNO[wanB]);
            h = BandNonCol_HashInt(h,MP[GB_AN]);
        }
    }

    return h;
}

static void BandNonCol_DMEntryCache_Reset(BandNonColState *state)
{
    openmx_core_native_free(state->resources,state->dm_entry_cache.basis0);
    openmx_core_native_free(state->resources,state->dm_entry_cache.basis1);
    openmx_core_native_free(state->resources,state->dm_entry_cache.phase_index);
    openmx_core_native_free(state->resources,state->dm_entry_cache.pair_l1);
    openmx_core_native_free(state->resources,state->dm_entry_cache.pair_l2);
    openmx_core_native_free(state->resources,state->dm_entry_cache.pair_l3);
    openmx_core_native_free(state->resources,state->dm_entry_cache.phase_r);
    openmx_core_native_free(state->resources,state->dm_entry_cache.phase_i);
    memset(&state->dm_entry_cache,0,sizeof(state->dm_entry_cache));
}

static void BandNonCol_DMEntryCache_Ensure(BandNonColState *state, int *MP, int n)
{
    BandNonColDMEntryCache *cache = &state->dm_entry_cache;
    unsigned long long fingerprint = BandNonCol_DMLayoutFingerprint(MP,n);
    int entry_count = 0;
    int pair_count = 0;

    if (cache->valid && cache->n==n && cache->fingerprint==fingerprint) return;

    BandNonCol_DMEntryCache_Reset(state);

    for (int GA_AN=1; GA_AN<=atomnum; GA_AN++){
        int wanA = WhatSpecies[GA_AN];
        int tnoA = Spe_Total_CNO[wanA];

        for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            int GB_AN = natn[GA_AN][LB_AN];
            int wanB  = WhatSpecies[GB_AN];
            int tnoB  = Spe_Total_CNO[wanB];

            pair_count++;
            entry_count += tnoA*tnoB;
        }
    }

    cache->basis0 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(size_t)(entry_count+1),0);
    cache->basis1 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(size_t)(entry_count+1),0);
    cache->phase_index = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(size_t)(entry_count+1),0);
    cache->pair_l1 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(size_t)(pair_count+1),0);
    cache->pair_l2 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(size_t)(pair_count+1),0);
    cache->pair_l3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(size_t)(pair_count+1),0);
    cache->phase_r = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(size_t)(pair_count+1),0);
    cache->phase_i = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(size_t)(pair_count+1),0);

    if (cache->basis0==NULL || cache->basis1==NULL || cache->phase_index==NULL ||
        cache->pair_l1==NULL || cache->pair_l2==NULL || cache->pair_l3==NULL ||
        cache->phase_r==NULL || cache->phase_i==NULL){
        BandNonCol_DMEntryCache_Reset(state);
        BandNonCol_AbortWithMessage("Failed to allocate DM entry cache in Band_DFT_NonCol.c.");
    }

    entry_count = 0;
    pair_count = 0;

    for (int GA_AN=1; GA_AN<=atomnum; GA_AN++){
        int wanA = WhatSpecies[GA_AN];
        int tnoA = Spe_Total_CNO[wanA];
        int Anum = MP[GA_AN];

        for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            int GB_AN = natn[GA_AN][LB_AN];
            int Rn    = ncn[GA_AN][LB_AN];
            int wanB  = WhatSpecies[GB_AN];
            int tnoB  = Spe_Total_CNO[wanB];
            int Bnum  = MP[GB_AN];
            int pair  = pair_count++;

            cache->pair_l1[pair] = atv_ijk[Rn][1];
            cache->pair_l2[pair] = atv_ijk[Rn][2];
            cache->pair_l3[pair] = atv_ijk[Rn][3];

            for (int i=0; i<tnoA; i++){
                int ibasis = Anum + i - 1;

                for (int j=0; j<tnoB; j++){
                    cache->basis0[entry_count] = ibasis;
                    cache->basis1[entry_count] = Bnum + j - 1;
                    cache->phase_index[entry_count] = pair;
                    entry_count++;
                }
            }
        }
    }

    cache->valid = 1;
    cache->n = n;
    cache->fingerprint = fingerprint;
    cache->entry_count = entry_count;
    cache->pair_count = pair_count;
}

static void BandNonCol_UpdateDMEntryPhases(BandNonColState *state, double k1, double k2, double k3)
{
    BandNonColDMEntryCache *cache = &state->dm_entry_cache;

    for (int pair=0; pair<cache->pair_count; pair++){
        double kRn = k1*(double)cache->pair_l1[pair]
                   + k2*(double)cache->pair_l2[pair]
                   + k3*(double)cache->pair_l3[pair];
        cache->phase_i[pair] = sin(2.0*PI*kRn);
        cache->phase_r[pair] = cos(2.0*PI*kRn);
    }
}

static void BandNonCol_DMOccWorkspace_Ensure(BandNonColState *state, int nk)
{
    BandNonColDMOccWorkspace *ws = &state->dm_occ_workspace;

    if (nk<=0) BandNonCol_AbortWithMessage("Invalid DM occupation workspace size in Band_DFT_NonCol.c.");
    if (ws->max_occ_nk>=nk) return;

    openmx_core_native_free(state->resources,ws->occ);
    openmx_core_native_free(state->resources,ws->eig_occ);

    ws->occ = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(size_t)nk,0);
    ws->eig_occ = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(size_t)nk,0);

    if (ws->occ==NULL || ws->eig_occ==NULL){
        openmx_core_native_free(state->resources,ws->occ);
        openmx_core_native_free(state->resources,ws->eig_occ);
        ws->occ = NULL;
        ws->eig_occ = NULL;
        ws->max_occ_nk = 0;
        BandNonCol_AbortWithMessage("Failed to allocate DM occupation workspace in Band_DFT_NonCol.c.");
    }

    ws->max_occ_nk = nk;
}

static int BandNonCol_BuildOccupationWeightsDense(const OpenMXChemicalPotentials *chemical, int kmin, int kmax, const double *eigen, double *occ, double *eig_occ)
{
    const double max_x = 60.0;
    const double FermiEps = 1.0e-13;
    int nk = 0;

    if (kmax<kmin) return 0;

    for (int k=kmin; k<=kmax; k++){
        double x = (eigen[k] - chemical->chemical_potential)*Beta;
        double FermiF;
        int local_k = k - kmin;

        if (x<=-max_x) x = -max_x;
        if (max_x<=x)  x = max_x;

        FermiF = FermiFunc_NC(x,k);
        occ[local_k] = FermiF;
        eig_occ[local_k] = FermiF*eigen[k];
        nk = local_k + 1;

        if (FermiF<FermiEps) break;
    }

    return nk;
}

/* host_recovery: only the direct device attempt runs, without the
   collective capacity recovery of openmx_gpu_dm_collective; 0 is returned
   when it declines, with the outputs untouched, and the caller accumulates
   this k point on the host (Calc_DM_Band_non_collinear, store_flag 0: no
   collective). Otherwise the collective path runs and 1 is returned. */
static int BandNonCol_AccumulateDMKPoint_OpenACC(BandNonColState *state, int myid2, int *is2, int *ie2, int *MP,
                                                  int n, int n2, int size_H1,
                                                  double k1, double k2, double k3, const double *ko,
                                                  const dcomplex *EVec1,
                                                  double *rDM11, double *rDM22, double *rDM12,
                                                  double *iDM12, double *iDM11, double *iDM22,
                                                  double *rEDM11, double *rEDM22,int active,MPI_Comm dm_comm,
                                                  int host_recovery)
{
    BandNonColDMEntryCache *cache=&state->dm_entry_cache;
    BandNonColDMOccWorkspace *ws=&state->dm_occ_workspace;
    int stride=active?ie2[myid2]-is2[myid2]+1:0,nk=0;
    if(stride>0){
        BandNonCol_DMOccWorkspace_Ensure(state, stride);
        nk=BandNonCol_BuildOccupationWeightsDense(state->chemical, is2[myid2],ie2[myid2],ko,ws->occ,ws->eig_occ);
        BandNonCol_DMEntryCache_Ensure(state, MP,n);BandNonCol_UpdateDMEntryPhases(state, k1,k2,k3);
        if(cache->entry_count!=size_H1)BandNonCol_AbortWithMessage("DM entry count mismatch");
    }
    const void *vectors=NULL;int host=0;
    if(nk){vectors=acc_deviceptr((void*)EVec1);host=vectors==NULL;if(host)vectors=EVec1;}
    double *outputs[10]={rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,rEDM11,rEDM22};
    if(host_recovery)
        return !nk || (!openmx_gpu_capacity_test_recovery() &&
            openmx_gpu_dm_try_entries(OPENMX_GPU_DM_NC,n,nk,stride,cache->entry_count,
                vectors,host,cache->basis0,cache->basis1,cache->phase_index,cache->pair_count,
                cache->phase_r,cache->phase_i,ws->occ,ws->eig_occ,NULL,outputs,1));
    openmx_gpu_dm_collective(dm_comm,OPENMX_GPU_DM_NC,n,nk,stride,nk?cache->entry_count:0,
        vectors,host,cache->basis0,cache->basis1,nk?cache->phase_index:NULL,cache->pair_count,
        cache->phase_r,cache->phase_i,ws->occ,ws->eig_occ,NULL,outputs,1);
    return 1;
}

/* GPU per-k DM accumulation for the legacy multi-k second loop, i.e. the
   ScaLAPACK/ELPA fallback (the k-dense GPU flow already accumulates its DM
   on the device through AccumulateDMRootDenseK).  The eigenvector panel
   still fits on the device even when the preflight refused the full dense
   solve.  Sizes below the noncollinear band threshold keep the CPU loop.
   Opt out with OPENMX_BAND_NONCOL_GPU_DM=0. */
static int BandNonCol_UseGpuFallbackDM(int n2,size_t evec_bytes)
{
    (void)evec_bytes;
    return openmx_env_flag_at(OPENMX_ENV_BAND_NONCOL_GPU_DM, 1) && scf_eigen_lib_flag==GPUSOLVER &&
        (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2);
}

/* Device-or-host decision of that fallback DM (OpenMX_HostSimdStageOnHost,
   collective over mpi_comm_level1: every rank, every k turn, behind the
   rank-uniform BandNonCol_UseGpuFallbackDM). need = the direct device
   attempt with host eigenvectors (its capacity_cost): two indices and
   twelve doubles per entry, three weights per state, the two spinor rows of
   all n orbitals over the states; budget = free device memory minus the
   256 MiB reserve of the worker capacity check. */
static int BandNonCol_DMOnHost(int n2, int states, int entries)
{
    size_t need = 0;
    if (0 < entries && 0 < states)
        need = (size_t)entries * (2 * sizeof(int) + 12 * sizeof(double)) + 3 * (size_t)states * sizeof(double)
             + (size_t)n2 * (size_t)states * sizeof(dcomplex);
    return OpenMX_HostSimdStageOnHost("Band_DFT_NonCol density matrix", need,
                                      OpenMX_HostSimdDeviceBudget((size_t)256 << 20));
}

static void BandNonCol_AccumulateDMRootDenseK_OpenACC(BandNonColState *state, int size_H1, int *MP, int n, int n2, int MaxN,
                                                      double k1, double k2, double k3,
                                                      double *ko, dcomplex *dense_evec,
                                                      double *rDM11, double *rDM22, double *rDM12,
                                                      double *iDM12, double *iDM11, double *iDM22,
                                                      double *rEDM11, double *rEDM22,MPI_Comm dm_comm)
{
    int nk=0;BandNonColDMOccWorkspace *ws=&state->dm_occ_workspace;
    BandNonColDMEntryCache *cache=&state->dm_entry_cache;
    if(dense_evec && MaxN>0){
        BandNonCol_DMOccWorkspace_Ensure(state, MaxN);
        nk=BandNonCol_BuildOccupationWeightsDense(state->chemical, 1,MaxN,ko,ws->occ,ws->eig_occ);
        BandNonCol_DMEntryCache_Ensure(state, MP,n);BandNonCol_UpdateDMEntryPhases(state, k1,k2,k3);
        if(cache->entry_count!=size_H1)BandNonCol_AbortWithMessage("DM entry count mismatch");
    }
    double *outputs[10]={rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,rEDM11,rEDM22};
    const void *vectors=NULL;int host=0;
    if(nk){vectors=acc_deviceptr((void*)dense_evec);host=vectors==NULL;if(host)vectors=dense_evec;}
    openmx_gpu_dm_collective(dm_comm,OPENMX_GPU_DM_NC,n,nk,n2,nk?cache->entry_count:0,
        vectors,host,cache->basis0,cache->basis1,nk?cache->phase_index:NULL,cache->pair_count,
        cache->phase_r,cache->phase_i,ws->occ,ws->eig_occ,NULL,outputs,1);
}

static void BandNonCol_CalcDMAllK1_OpenACC(BandNonColState *state, int myid0, int myid2, int size_H1,
                                           int *is2, int *ie2, int *MP, int n, int n2,
                                           double k1, double k2, double k3,
                                           double *****CDM, double *****iDM0, double *****EDM,
                                           double *ko, dcomplex *EVec1,
                                           double *rDM11, double *rDM22, double *rDM12,
                                           double *iDM12, double *iDM11, double *iDM22,
                                           double *rEDM11, double *rEDM22)
{
    size_t p = 0;

    memset(rDM11,0,sizeof(double)*(size_t)size_H1);
    memset(rDM22,0,sizeof(double)*(size_t)size_H1);
    memset(rDM12,0,sizeof(double)*(size_t)size_H1);
    memset(iDM12,0,sizeof(double)*(size_t)size_H1);
    memset(iDM11,0,sizeof(double)*(size_t)size_H1);
    memset(iDM22,0,sizeof(double)*(size_t)size_H1);
    memset(rEDM11,0,sizeof(double)*(size_t)size_H1);
    memset(rEDM22,0,sizeof(double)*(size_t)size_H1);

    BandNonCol_AccumulateDMKPoint_OpenACC(state, myid2,is2,ie2,MP,n,n2,size_H1,k1,k2,k3,ko,EVec1,
                                          rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,rEDM11,rEDM22,1,mpi_comm_level1,0);

    MPI_Allreduce(MPI_IN_PLACE,rDM11, size_H1,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE,rDM22, size_H1,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE,rDM12, size_H1,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE,iDM11, size_H1,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE,iDM22, size_H1,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE,iDM12, size_H1,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE,rEDM11,size_H1,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE,rEDM22,size_H1,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

    for (int GA_AN=1; GA_AN<=atomnum; GA_AN++){
        int MA_AN = F_G2M[GA_AN];
        int wanA = WhatSpecies[GA_AN];
        int tnoA = Spe_Total_CNO[wanA];
        int ID = G2ID[GA_AN];

        for (int LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            int GB_AN = natn[GA_AN][LB_AN];
            int wanB = WhatSpecies[GB_AN];
            int tnoB = Spe_Total_CNO[wanB];

            if (myid0==ID){
                for (int i=0; i<tnoA; i++){
                    for (int j=0; j<tnoB; j++, p++){
                        CDM[0][MA_AN][LB_AN][i][j] = rDM11[p];
                        CDM[1][MA_AN][LB_AN][i][j] = rDM22[p];
                        CDM[2][MA_AN][LB_AN][i][j] = rDM12[p];
                        CDM[3][MA_AN][LB_AN][i][j] = iDM12[p];
                        iDM0[0][MA_AN][LB_AN][i][j] = iDM11[p];
                        iDM0[1][MA_AN][LB_AN][i][j] = iDM22[p];
                        EDM[0][MA_AN][LB_AN][i][j] = rEDM11[p];
                        EDM[1][MA_AN][LB_AN][i][j] = rEDM22[p];
                    }
                }
            }
            else {
                p += (size_t)tnoA*(size_t)tnoB;
            }
        }
    }
}

static void BandNonCol_RootDenseEigenvectorsRelease(BandNonColRootDenseWorkspace *ws)
{
    if (ws->cs2_on_device && ws->cs2 != NULL){
        dcomplex *cs2 = ws->cs2;
        size_t n2n2 = (size_t)ws->n2*(size_t)ws->n2;

#pragma acc exit data delete(cs2[0 : n2n2])
    }
    ws->cs2_on_device = 0;
}

static void BandNonCol_RootDenseWorkspace_Reset(BandNonColState *state)
{
    BandNonColRootDenseWorkspace *ws = &state->root_dense_workspace;

    BandNonCol_RootDenseEigenvectorsRelease(ws);
    openmx_core_native_free(state->resources,ws->s_all);
    openmx_core_native_free(state->resources,ws->h11);
    openmx_core_native_free(state->resources,ws->h22);
    openmx_core_native_free(state->resources,ws->h12);
    openmx_core_native_free(state->resources,ws->work);
    openmx_core_native_free(state->resources,ws->hs2);
    openmx_core_native_free(state->resources,ws->cs2);
    memset(ws,0,sizeof(*ws));
}

static BandNonColRootDenseWorkspace *BandNonCol_RootDenseWorkspace_Ensure(BandNonColState *state, int owns_root_dense,
                                                                           int n, int n2, int MaxN,
                                                                           int T_knum, int SCF_iter)
{
    BandNonColRootDenseWorkspace *ws = &state->root_dense_workspace;
    size_t nn;
    size_t n2n2;

    if (!owns_root_dense) return ws;

    if (!ws->valid || ws->n!=n || ws->n2!=n2 || ws->maxn!=MaxN || ws->t_knum!=T_knum){
        BandNonCol_RootDenseWorkspace_Reset(state);

        nn = (size_t)n*(size_t)n;
        n2n2 = (size_t)n2*(size_t)n2;

        ws->s_all = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*nn*(size_t)T_knum,0);
        ws->h11   = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*nn,0);
        ws->h22   = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*nn,0);
        ws->h12   = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*nn,0);
        ws->work  = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*nn,0);
        ws->hs2   = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*n2n2,0);
        ws->cs2   = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*n2n2,0);

        if (ws->s_all==NULL || ws->h11==NULL || ws->h22==NULL || ws->h12==NULL ||
            ws->work==NULL || ws->hs2==NULL || ws->cs2==NULL){
            BandNonCol_RootDenseWorkspace_Reset(state);
            BandNonCol_AbortWithMessage("Failed to allocate root dense GpuSolver workspace in Band_DFT_NonCol.c.");
        }

        ws->valid = 1;
        ws->s_valid = 0;
        ws->n = n;
        ws->n2 = n2;
        ws->maxn = MaxN;
        ws->t_knum = T_knum;
    }

    if (SCF_iter==1) ws->s_valid = 0;

    return ws;
}

static void BandNonCol_KCacheSave(dcomplex *panel, BandNonColRootDenseWorkspace *ws)
{
    const size_t pitch = sizeof(dcomplex)*(size_t)ws->n2;
    const size_t width = sizeof(dcomplex)*(size_t)ws->maxn;

    if (!ws->cs2_on_device) {
        for (int j=0;j<ws->n2;++j) memcpy(panel+(size_t)j*ws->maxn,ws->cs2+(size_t)j*ws->n2,width);
        return;
    }

#pragma acc wait
    wait_cudafunc(cudaMemcpy2D(panel,width,acc_deviceptr(ws->cs2),pitch,
                               width,(size_t)ws->n2,cudaMemcpyDeviceToHost));
    /* A rank can own several k points in one concurrency group. Release
       this mapping before the next solve creates another cs2 mapping. */
    BandNonCol_RootDenseEigenvectorsRelease(ws);
}

static void BandNonCol_KCacheRestore(BandNonColState *state, const dcomplex *panel, BandNonColRootDenseWorkspace *ws)
{
    dcomplex *cs2 = ws->cs2;
    const size_t n2n2 = (size_t)ws->n2*(size_t)ws->n2;
    const size_t pitch = sizeof(dcomplex)*(size_t)ws->n2;
    const size_t width = sizeof(dcomplex)*(size_t)ws->maxn;

    if (state->mgpu_host_dense) {
        memset(cs2,0,n2n2*sizeof(dcomplex));
        for (int j=0;j<ws->n2;++j) memcpy(cs2+(size_t)j*ws->n2,panel+(size_t)j*ws->maxn,width);
        ws->cs2_on_device=0;
        return;
    }
    BandNonCol_RootDenseEigenvectorsRelease(ws);
#pragma acc enter data create(cs2[0 : n2n2])
    ws->cs2_on_device = 1;
    wait_cudafunc(cudaMemcpy2D(acc_deviceptr(cs2),pitch,panel,width,
                               width,(size_t)ws->n2,cudaMemcpyHostToDevice));
}

static void BandNonCol_ConstructDenseMsFromPacked(BandNonColState *state,  int cpx_flag, const double *M1, dcomplex *Ms,
                                                   int *order_GA, int *MP, double k1, double k2, double k3,
                                                   int n, int owns_dense );
static void BandNonCol_BuildDenseHs2_OpenACC(int n, int n2, const dcomplex *H11,
                                             const dcomplex *H22, const dcomplex *H12,
                                             dcomplex *H2);

static cudaError_t BandNonCol_TryDeviceMalloc(BandNonColState *state, void **ptr, size_t bytes)
{
    cudaError_t status = OpenMX_Resource_GpuMalloc(state->resources,ptr,bytes);

    if (status != cudaSuccess) {
        *ptr = NULL;
        (void)cudaGetLastError();
    }
    return status;
}

static void *BandNonCol_MallocArray(BandNonColState *state, size_t count, size_t elem_size, const char *label)
{
    void *ptr = openmx_core_native_malloc(state->resources,1,count * elem_size,0);

    if (ptr == NULL) {
        char msg[256];

        snprintf(msg, sizeof(msg), "Failed to allocate %s (%zu bytes) in Band_DFT_NonCol.c.", label,
                 count * elem_size);
        BandNonCol_AbortWithMessage(msg);
    }
    return ptr;
}

/* The first half of a refined solve of this rank's k-point: hs2 (present,
   Hermitian) is solved in FP32, or the warm start taken, with the device
   copy of cs2 (created here; the back transform fills it later) as the
   scratch.  The estimates go to ko (host) and EIGEN.  Returns 1 with the
   solve pending (hs2, active_S and ko stay present for the second half),
   0 when the FP64 eigensolver has to take over (hs2 untouched). */
static cudaError_t BandNonCol_RefineDeviceMalloc(void *context, void **ptr, size_t bytes)
{
    return BandNonCol_TryDeviceMalloc((BandNonColState *)context, ptr, bytes);
}
static void *BandNonCol_RefineHostAllocate(void *context, size_t count, size_t bytes, const char *label)
{
    return BandNonCol_MallocArray((BandNonColState *)context, count, bytes, label);
}

static int BandNonCol_RefinedPrepare(BandNonColState *state, int n2, int MaxN, dcomplex *hs2, dcomplex *cs2, double *ko, int kloop,
                                     double ***EIGEN)
{
    BandNonColGpuSolverWorkspace *w = &state->gpusolver_workspace;
    BandNonColRefineState *st = &state->refine;
    EigenRefineDevice *dev = &state->refine_dev;
    EigenRefineProblem *pb = &state->refine_pb;
    EigenRefineReport rep;
    size_t const n2n2 = (size_t)n2 * (size_t)n2;
    int rc = 1;

    st->pending = 0;
    BandNonCol_DMGpu_Init(state);
    BandNonCol_GpuSolver_EnsureInfo(state);
    if (!acc_is_present(cs2, sizeof(dcomplex) * n2n2)) {
#pragma acc enter data create(cs2[0 : n2n2])
    }
    if (state->refine_x != NULL && state->refine_x_n != n2) BandNonCol_RefineDropBasis(state);
    if (state->refine_x == NULL) {
        if (BandNonCol_TryDeviceMalloc(state, (void **)&state->refine_x, sizeof(dcomplex) * n2n2) != cudaSuccess) {
            if (2 <= level_stdout) {
                printf("<Band_DFT_NonCol> no room for the %.1f MiB basis of the refined eigensolver; FP64 eigensolver%s\n",
                       (double)(sizeof(dcomplex) * n2n2) / (1024.0 * 1024.0),
                       st->transient ? "" : " for the rest of the SCF cycle");
                fflush(stdout);
            }
            if (!st->transient && st->persistent < 1) st->persistent = 1;
            rc = 0;
        }
        else {
            state->refine_x_n = n2;
        }
    }
    if (state->refine_e0_n != n2) {
        openmx_core_native_free(state->resources,state->refine_e0);
        state->refine_e0 = (double *)BandNonCol_MallocArray(state, (size_t)n2, sizeof(double), "refinement eigenvalues");
        state->refine_e0_n = n2;
    }
    if (rc) {
        dev->cublas = state->dm_gpu_workspace.cublas;
        dev->cusolver = w->gpusolver;
        dev->stream = w->stream;
        dev->d_work = &w->d_work;
        dev->d_work_bytes = &w->d_work_bytes;
        dev->h_work = &w->h_work;
        dev->h_work_bytes = &w->h_work_bytes;
        dev->d_info = w->d_info;
        dev->allocation_context = state;
    dev->try_malloc_owned = BandNonCol_RefineDeviceMalloc;
        dev->resources = state->resources;
        dev->host_malloc_owned = BandNonCol_RefineHostAllocate;

        memset(pb, 0, sizeof(*pb));
        memset(&rep, 0, sizeof(rep));
        pb->cplx = 1;
        pb->n = n2;
        pb->maxn = MaxN;
        pb->iterations = BandNonCol_RefineIterations(state);
        pb->a = acc_deviceptr(hs2);
        pb->w = (double *)acc_deviceptr(ko) + 1;
        pb->fp32 = acc_deviceptr(cs2);
        pb->region = sizeof(dcomplex) * n2n2;
        pb->x = state->refine_x;
        pb->transient = st->transient;
        pb->warm = ((openmx_eigen_refine_warm_band_step(st->scf_iter) || state->refine_force_warm) &&
                    state->refine_state.basis_valid && state->refine_state.basis_n == n2 && !st->transient);
        pb->defaulted = st->defaulted;
        pb->occupation = BandNonCol_RefineFermi;
        pb->occupation_ctx = &state->refine_occ;
        if (3 <= level_stdout) {
            printf("<Band_DFT_NonCol> k-point %d: %s, %d refinement steps (OPENMX_EIGEN_REFINE)\n", kloop,
                   pb->warm ? "warm start from the previous vectors" : "FP32 eigensolver", pb->iterations);
            fflush(stdout);
        }
        rc = openmx_eigen_refine_prepare(&state->refine_state, dev, pb, state->refine_e0, &rep);
        if (rc < 0) {
            pb->warm = 0;
            rc = openmx_eigen_refine_prepare(&state->refine_state, dev, pb, state->refine_e0, &rep);
        }
        if (rc != 1 && st->persistent < rep.persistent) st->persistent = rep.persistent;
    }
    if (rc == 1) {
        for (int l = 1; l <= MaxN; l++) {
            ko[l] = state->refine_e0[l - 1];
            EIGEN[0][kloop][l] = ko[l];
        }
        st->pending = 1;
        return 1;
    }
    state->refine_state.basis_valid = 0;
    return 0;
}

/* The second half: the refinement with the occupations at mu (the FP64
   eigensolver when it fails), the eigenvalues to ko and EIGEN, the back
   transform into cs2, and the release of the device copies as in the
   unrefined solve. */
static void BandNonCol_RootDenseSolveOneK_SecondHalf(BandNonColState *state, int n, int n2, int MaxN, int kloop, double *ko,
                                                     double ***EIGEN, double mu,
                                                     BandNonColRootDenseWorkspace *rdw, int vectors)
{
    int nn = n*n;
    int n2n2 = n2*n2;
    int l;
    dcomplex *active_S = rdw->s_all;
    dcomplex *hs2 = rdw->hs2;
    dcomplex *cs2 = rdw->cs2;
    BandNonColRefineState *st = &state->refine;
    EigenRefineProblem *pb = &state->refine_pb;
    EigenRefineReport rep;
    double *f0 = (double *)BandNonCol_MallocArray(state, (size_t)n2, sizeof(double), "refinement occupations");
    int rc;

    memset(&rep, 0, sizeof(rep));
    state->refine_occ.mu = mu;
    for (int i = 0; i < n2; i++) f0[i] = BandNonCol_RefineFermi(state->refine_e0[i], i + 1, &state->refine_occ);
    rc = openmx_eigen_refine_finish(&state->refine_state, &state->refine_dev, pb, f0, &rep);
    openmx_core_native_free(state->resources,f0);
    st->pending = 0;
    if (rc != 1 && st->persistent < rep.persistent) st->persistent = rep.persistent;
    if (rc == 1) {
        if (3 <= level_stdout) {
            printf("<Band_DFT_NonCol> k-point %d: refinement step %d of %d columns: max |r| %.1e, max |s| %.1e, "
                   "delta %.1e; Rayleigh-Ritz in %d partially occupied clusters (largest %d, chained within %.1e)\n",
                   kloop, pb->iterations, rep.columns, rep.max_r, rep.max_s, rep.delta, rep.rr_clusters,
                   rep.rr_largest, rep.chain);
            fflush(stdout);
        }
        st->solved = 1;
    }
    else {
        /* hs2 still holds the Hermitian matrix */
        state->refine_state.basis_valid = 0;
        BandNonCol_GpuSolver_DenseZheevx_Device(state, hs2,ko,n2,MaxN,
                                               "Band_DFT_NonCol root dense Hamiltonian",vectors);
    }
#pragma acc update self(ko[0 : MaxN + 1])
    for (l=1; l<=MaxN; l++){
        EIGEN[0][kloop][l] = ko[l];
    }
    if (vectors){
        BandNonCol_DenseWavefunctions_PresentOpenACC(state, n,n2,MaxN,hs2,active_S,cs2);
        rdw->cs2_on_device = 1;
    }
    else {
        /* the device copy of cs2 was only the scratch of the refinement */
        rdw->cs2_on_device = 1;
        BandNonCol_RootDenseEigenvectorsRelease(rdw);
    }
#pragma acc exit data delete(active_S[0 : nn])
#pragma acc exit data delete(hs2[0 : n2n2])
#pragma acc exit data delete(ko[0 : n2 + 1])
    BandNonCol_ClearOpenAccFreelists();
}

/* the device basis buffer of the refined solves (n2 x n2 complex); 0 when the
   device has no room */
static int BandNonCol_RefineEnsureDeviceBasis(BandNonColState *state, int n2)
{
    size_t const n2n2 = (size_t)n2 * (size_t)n2;

    if (state->refine_x != NULL && state->refine_x_n != n2) BandNonCol_RefineDropBasis(state);
    if (state->refine_x == NULL) {
        if (BandNonCol_TryDeviceMalloc(state, (void **)&state->refine_x, sizeof(dcomplex) * n2n2) != cudaSuccess) {
            return 0;
        }
        state->refine_x_n = n2;
    }
    return 1;
}

static void BandNonCol_KRefReset(BandNonColState *state)
{
    if (state->kref != NULL) {
        for (int slot = 0; slot < state->kref_slots; slot++) {
            openmx_core_native_free(state->resources,state->kref[slot].basis);
            openmx_core_native_free(state->resources,state->kref[slot].lam);
        }
        openmx_core_native_free(state->resources,state->kref);
    }
    state->kref = NULL;
    state->kref_slots = 0;
    state->kref_n2 = 0;
    state->kref_used = 0;
}

static void BandNonCol_KRefEnsure(BandNonColState *state, int slots, int n2)
{
    if (state->kref != NULL && (state->kref_slots != slots || state->kref_n2 != n2)) BandNonCol_KRefReset(state);
    if (state->kref == NULL) {
        state->kref = (BandNonColKRefineSlot *)BandNonCol_MallocArray(state, (size_t)slots, sizeof(BandNonColKRefineSlot),
                                                                          "refined basis slots");
        memset(state->kref, 0, sizeof(BandNonColKRefineSlot) * (size_t)slots);
        state->kref_slots = slots;
        state->kref_n2 = n2;
    }
}

static int BandNonCol_KRefSlotAllocate(BandNonColState *state, BandNonColKRefineSlot *slot, int n2)
{
    size_t const bytes = sizeof(dcomplex) * (size_t)n2 * (size_t)n2 + sizeof(double) * (size_t)n2;

    if (slot->basis != NULL) return 1;
    {
        size_t const limit = BandNonCol_HostCacheLimit(OPENMX_ENV_BAND_REFINE_BASIS_MB, 1024.0, 32);

        if (limit < state->kref_used || bytes > limit - state->kref_used) return 0;
    }
    slot->basis = (dcomplex *)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex) * (size_t)n2 * (size_t)n2,0);
    slot->lam = (double *)openmx_core_native_malloc(state->resources,1,sizeof(double) * (size_t)n2,0);
    if (slot->basis == NULL || slot->lam == NULL) {
        openmx_core_native_free(state->resources,slot->basis);
        openmx_core_native_free(state->resources,slot->lam);
        slot->basis = NULL;
        slot->lam = NULL;
        return 0;
    }
    state->kref_used += bytes;
    return 1;
}

static void BandNonCol_KRefSlotFree(BandNonColState *state, BandNonColKRefineSlot *slot, int n2)
{
    if (slot->basis != NULL) {
        state->kref_used -= sizeof(dcomplex) * (size_t)n2 * (size_t)n2 + sizeof(double) * (size_t)n2;
    }
    openmx_core_native_free(state->resources,slot->basis);
    openmx_core_native_free(state->resources,slot->lam);
    slot->basis = NULL;
    slot->lam = NULL;
    slot->valid = 0;
}

static void BandNonCol_HostHermitian(int n,dcomplex *a)
{
#pragma omp parallel for
    for (int j=0;j<n;++j) {
        a[(size_t)j*n+j].i=0.0;
        for (int i=j+1;i<n;++i) {
            size_t x=(size_t)j*n+i,y=(size_t)i*n+j;
            dcomplex v={0.5*(a[x].r+a[y].r),0.5*(a[x].i-a[y].i)};
            a[x]=v; a[y]=(dcomplex){v.r,-v.i};
        }
    }
}
static int BandNonCol_RootDenseSolveOneK_Host(BandNonColState *state, int rebuild_overlap,
    int n,int n2,int MaxN,int kloop,double k1,double k2,double k3,
    const double *m_olp,const double *m_h11,const double *m_h22,
    const double *m_h12,const double *m_h12i,const double *m_i11,
    const double *m_i22,const double *m_i12,int *packed_order_GA,int *MP,
    double *ko,double ***EIGEN,int need_evec,BandNonColRootDenseWorkspace *rdw)
{
    const size_t nn=(size_t)n*n,n2n2=(size_t)n2*n2;
    dcomplex *s=rdw->s_all,*work=rdw->work,*q=rdw->hs2;
    dcomplex *h[3]={rdw->h11,rdw->h22,rdw->h12};
    const double *real[3]={m_h11,m_h22,m_h12};
    const double *imag[3]={m_i11,m_i22,m_h12i};
    if (rebuild_overlap) {
        memset(s,0,nn*sizeof(dcomplex));
        BandNonCol_ConstructDenseMsFromPacked(state, 0,m_olp,s,packed_order_GA,MP,k1,k2,k3,n,1);
        BandNonCol_HostHermitian(n,s);
        if (!openmx_mgpu_host_solve(n,n,1,s,n,ko+1)) BandNonCol_AbortWithMessage(openmx_mgpu_host_error());
#pragma omp parallel for
        for (int j=0;j<n;++j) {
            double scale=1.0/sqrt(fmax(ko[j+1],1.0e-10));
            for (int i=0;i<n;++i) { s[(size_t)j*n+i].r*=scale; s[(size_t)j*n+i].i*=scale; }
        }
    }
    for (int b=0;b<3;++b) {
        memset(h[b],0,nn*sizeof(dcomplex));
        BandNonCol_ConstructDenseMsFromPacked(state, 0,real[b],h[b],packed_order_GA,MP,k1,k2,k3,n,1);
        BandNonCol_ConstructDenseMsFromPacked(state, 1,imag[b],h[b],packed_order_GA,MP,k1,k2,k3,n,1);
        if (b==2) BandNonCol_ConstructDenseMsFromPacked(state, 1,m_i12,h[b],packed_order_GA,MP,k1,k2,k3,n,1);
        if (!openmx_mgpu_host_zgemm(CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,h[b],n,s,n,work,n) ||
            !openmx_mgpu_host_zgemm(CUBLAS_OP_C,CUBLAS_OP_N,n,n,n,s,n,work,n,h[b],n))
            BandNonCol_AbortWithMessage(openmx_mgpu_host_error());
    }
#pragma omp parallel for collapse(2)
    for (int j=0;j<n;++j) for (int i=0;i<n;++i) {
        size_t ij=(size_t)j*n+i, full=(size_t)j*n2+i;
        q[full]=h[0][ij]; q[full+(size_t)n*n2]=h[2][ij];
        q[(size_t)i*n2+j+n]=(dcomplex){h[2][ij].r,-h[2][ij].i};
        q[full+(size_t)n*n2+n]=h[1][ij];
    }
    BandNonCol_HostHermitian(n2,q);
    if (!openmx_mgpu_host_solve(n2,MaxN,1,q,n2,ko+1)) BandNonCol_AbortWithMessage(openmx_mgpu_host_error());
    for (int l=1;l<=MaxN;++l) EIGEN[0][kloop][l]=ko[l];
    if (need_evec) {
        memset(rdw->cs2,0,n2n2*sizeof(dcomplex));
        for (int half=0;half<2;++half)
            if (!openmx_mgpu_host_zgemm(CUBLAS_OP_T,CUBLAS_OP_T,MaxN,n,n,q+(size_t)half*n,n2,
                    s,n,rdw->cs2+(size_t)half*n*n2,n2)) BandNonCol_AbortWithMessage(openmx_mgpu_host_error());
    }
    rdw->cs2_on_device=0;
    return 0;
}

static int BandNonCol_RootDenseSolveOneK_FirstHalf(BandNonColState *state, int rebuild_overlap,
                                                   int n, int n2, int MaxN, int kloop,
                                                   double k1, double k2, double k3,
                                                   const double *m_olp,
                                                   const double *m_h11, const double *m_h22,
                                                   const double *m_h12, const double *m_h12i,
                                                   const double *m_i11, const double *m_i22,
                                                   const double *m_i12,
                                                   int *packed_order_GA, int *MP,
                                                   double *ko, double ***EIGEN,
                                                   int need_evec,
                                                   BandNonColRootDenseWorkspace *rdw,
                                                   int refine);

/* One k-point of the k-dense path, refined.  Pass 1 (vectors 0): the
   refinement gated with the previous step's chemical potential (the
   eigenvalues are second order in the vectors' error), EIGEN, and the basis
   kept on the host for pass 2.  Pass 2 (vectors 1): the refinement from that
   basis with the step's chemical potential, the wavefunctions into cs2.  The
   FP64 eigensolver takes over inside when the refinement fails. */
static void BandNonCol_KDenseRefinedSolveOneK(BandNonColState *state, int rebuild_overlap, int n, int n2, int MaxN, int kloop,
                                              double k1, double k2, double k3,
                                              const double *m_olp, const double *m_h11, const double *m_h22,
                                              const double *m_h12, const double *m_h12i, const double *m_i11,
                                              const double *m_i22, const double *m_i12,
                                              int *packed_order_GA, int *MP, double *ko, double ***EIGEN,
                                              BandNonColRootDenseWorkspace *rdw, BandNonColKRefineSlot *slot,
                                              int vectors, double mu)
{
    BandNonColRefineState *st = &state->refine;
    EigenRefineState *rs = &state->refine_state;
    size_t const n2n2 = (size_t)n2 * (size_t)n2;
    int pending;

    /* the slot's basis as the warm start: always between the passes of one
       step, across steps only with OPENMX_EIGEN_REFINE_WARM_BAND */
    rs->basis_valid = 0;
    state->refine_force_warm = 0;
    if (slot->valid && slot->basis != NULL && (vectors || openmx_eigen_refine_warm_band_step(st->scf_iter)) &&
        BandNonCol_RefineEnsureDeviceBasis(state, n2) && rs->lam != NULL) {
        wait_cudafunc(cudaMemcpy(state->refine_x, slot->basis, sizeof(dcomplex) * n2n2, cudaMemcpyHostToDevice));
        wait_cudafunc(cudaMemcpy(rs->lam, slot->lam, sizeof(double) * (size_t)n2, cudaMemcpyHostToDevice));
        rs->basis_valid = 1;
        rs->basis_n = n2;
        state->refine_force_warm = vectors;
    }
    pending = BandNonCol_RootDenseSolveOneK_FirstHalf(state, rebuild_overlap,n,n2,MaxN,kloop,k1,k2,k3,
                                                      m_olp,m_h11,m_h22,m_h12,m_h12i,m_i11,m_i22,m_i12,
                                                      packed_order_GA,MP,ko,EIGEN,vectors,rdw,1);
    state->refine_force_warm = 0;
    if (!vectors) slot->refined = 0;
    if (!pending) {
        slot->valid = 0;
        return;
    }
    BandNonCol_RootDenseSolveOneK_SecondHalf(state, n,n2,MaxN,kloop,ko,EIGEN,mu,rdw,vectors);
    if (st->solved && rs->basis_valid) {
        if (!vectors) slot->refined = 1;
        if ((!vectors || openmx_eigen_refine_warm_band_enabled()) && BandNonCol_KRefSlotAllocate(state, slot, n2)) {
            wait_cudafunc(cudaMemcpy(slot->basis, state->refine_x, sizeof(dcomplex) * n2n2, cudaMemcpyDeviceToHost));
            wait_cudafunc(cudaMemcpy(slot->lam, rs->lam, sizeof(double) * (size_t)n2, cudaMemcpyDeviceToHost));
            slot->valid = 1;
        }
        else {
            BandNonCol_KRefSlotFree(state, slot, n2);
        }
    }
    else {
        BandNonCol_KRefSlotFree(state, slot, n2);
    }
}

/* The solve of one k-point; with refine, the first half of a refined one
   (returns 1 while the solve is pending), else the full solve (returns 0). */
static int BandNonCol_RootDenseSolveOneK_FirstHalf(BandNonColState *state, int rebuild_overlap,
                                                   int n, int n2, int MaxN, int kloop,
                                                   double k1, double k2, double k3,
                                                   const double *m_olp,
                                                   const double *m_h11, const double *m_h22,
                                                   const double *m_h12, const double *m_h12i,
                                                   const double *m_i11, const double *m_i22,
                                                   const double *m_i12,
                                                   int *packed_order_GA, int *MP,
                                                   double *ko, double ***EIGEN,
                                                   int need_evec,
                                                   BandNonColRootDenseWorkspace *rdw,
                                                   int refine)
{
    if (state->mgpu_host_dense)
        return BandNonCol_RootDenseSolveOneK_Host(state, rebuild_overlap,n,n2,MaxN,kloop,k1,k2,k3,
            m_olp,m_h11,m_h22,m_h12,m_h12i,m_i11,m_i22,m_i12,packed_order_GA,MP,ko,EIGEN,need_evec,rdw);
    int nn = n*n;
    int n2n2 = n2*n2;
    int i,j,l;
    dcomplex *active_S = rdw->s_all;

#pragma acc enter data create(ko[0 : n2 + 1])

    if (!rebuild_overlap){
#pragma acc enter data copyin(active_S[0 : nn])
    }

    if (rebuild_overlap){
        BandNonCol_ConstructDenseMsFromPacked(state, 0,m_olp,active_S,packed_order_GA,MP,k1,k2,k3,n,1);

        BandNonCol_SymmetrizeDenseHermitian_OpenACC(n,active_S);
        BandNonCol_GpuSolver_DenseZheevx_Device(state, active_S,ko,n,n,
                                               "Band_DFT_NonCol root dense overlap",1);

if (!openmx_gpu_transform(OPENMX_GPU_INVSQRT,0,1,n,(double*)acc_deviceptr((void*)(ko))+1,1,NULL,0,(double*)acc_deviceptr((void*)(ko))+1,1,0,1.0e-10)) {
#pragma acc parallel loop present(ko[0 : n + 1])
        for (l=1; l<=n; l++){
            if (ko[l]<1.0e-10) ko[l] = 1.0e-10;
            ko[l] = 1.0/sqrt(ko[l]);
        }
}

if (!openmx_gpu_transform(OPENMX_GPU_SCALE_COLUMNS,1,n,n,acc_deviceptr((void*)(active_S)),n,(double*)acc_deviceptr((void*)(ko))+1,0,acc_deviceptr((void*)(active_S)),n,0,0.0)) {
#pragma acc parallel loop collapse(2) present(active_S[0 : nn], ko[0 : n + 1])
        for (i=0; i<n; i++){
            for (j=0; j<n; j++){
                active_S[(size_t)j*(size_t)n + (size_t)i].r *= ko[j+1];
                active_S[(size_t)j*(size_t)n + (size_t)i].i *= ko[j+1];
            }
        }
}

#pragma acc update self(active_S[0 : nn])
    }

    BandNonCol_ConstructDenseMsFromPacked(state, 0,m_h11,rdw->h11,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_ConstructDenseMsFromPacked(state, 1,m_i11,rdw->work,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_AddDense_OpenACC(n,rdw->h11,rdw->work);

    BandNonCol_ConstructDenseMsFromPacked(state, 0,m_h22,rdw->h22,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_ConstructDenseMsFromPacked(state, 1,m_i22,rdw->work,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_AddDense_OpenACC(n,rdw->h22,rdw->work);

    BandNonCol_ConstructDenseMsFromPacked(state, 0,m_h12,rdw->h12,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_ConstructDenseMsFromPacked(state, 1,m_h12i,rdw->work,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_AddDense_OpenACC(n,rdw->h12,rdw->work);
    BandNonCol_ConstructDenseMsFromPacked(state, 1,m_i12,rdw->work,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_AddDense_OpenACC(n,rdw->h12,rdw->work);

    {
        dcomplex *h11 = rdw->h11;
        dcomplex *h22 = rdw->h22;
        dcomplex *h12 = rdw->h12;
        dcomplex *work = rdw->work;
        dcomplex *hs2 = rdw->hs2;
        dcomplex *cs2 = rdw->cs2;

        /* The GEMMul8 workspace stays allocated across the turns of one
           concurrency group (the auto turn limit budgets for it) and is
           released in the group-end cleanup of each caller. */
        BandNonCol_DenseTripleTransform_PresentOpenACC(state, n,h11,active_S,work);
        BandNonCol_DenseTripleTransform_PresentOpenACC(state, n,h12,active_S,work);
        BandNonCol_DenseTripleTransform_PresentOpenACC(state, n,h22,active_S,work);

#pragma acc exit data delete(work[0 : nn])
        BandNonCol_BuildDenseHs2_OpenACC(n,n2,h11,h22,h12,hs2);
#pragma acc exit data delete(h11[0 : nn], h22[0 : nn], h12[0 : nn])

        BandNonCol_SymmetrizeDenseHermitian_OpenACC(n2,hs2);
        if (refine && BandNonCol_RefinedPrepare(state, n2,MaxN,hs2,cs2,ko,kloop,EIGEN)){
            /* pending: hs2, active_S and ko stay present for the second half */
            return 1;
        }
        BandNonCol_GpuSolver_DenseZheevx_Device(state, hs2,ko,n2,MaxN,
                                               "Band_DFT_NonCol root dense Hamiltonian",
                                               need_evec || !BandNonCol_EigenvaluesOnlyNoVectors());

#pragma acc update self(ko[0 : MaxN + 1])

        for (l=1; l<=MaxN; l++){
            EIGEN[0][kloop][l] = ko[l];
        }

        if (!need_evec){
#pragma acc exit data delete(hs2[0 : n2n2], active_S[0 : nn])
#pragma acc exit data delete(ko[0 : n2 + 1])
            BandNonCol_ClearOpenAccFreelists();
            return 0;
        }

        BandNonCol_DenseWavefunctions_PresentOpenACC(state, n,n2,MaxN,hs2,active_S,cs2);
#pragma acc exit data delete(active_S[0 : nn])

        /* cs2 stays device-resident: the 16*n2*n2 host readback used to be
           the biggest host-touched block of a k-owner rank, and both
           consumers can work from the device copy (the scatter packs
           per-destination panels there, the DM copyin turns into a
           present hit).  The copy lives until the workspace reset. */
        rdw->cs2_on_device = 1;
#pragma acc exit data delete(hs2[0 : n2n2])
#pragma acc exit data delete(ko[0 : n2 + 1])
        BandNonCol_ClearOpenAccFreelists();
    }
    return 0;
}

static void BandNonCol_RootDenseSolveOneK_OpenACC(BandNonColState *state, int rebuild_overlap,
                                                  int n, int n2, int MaxN, int kloop,
                                                  double k1, double k2, double k3,
                                                  const double *m_olp,
                                                  const double *m_h11, const double *m_h22,
                                                  const double *m_h12, const double *m_h12i,
                                                  const double *m_i11, const double *m_i22,
                                                  const double *m_i12,
                                                  int *packed_order_GA, int *MP,
                                                  double *ko, double ***EIGEN,
                                                  int need_evec,
                                                  BandNonColRootDenseWorkspace *rdw)
{
    (void)BandNonCol_RootDenseSolveOneK_FirstHalf(state, rebuild_overlap,n,n2,MaxN,kloop,k1,k2,k3,
                                                  m_olp,m_h11,m_h22,m_h12,m_h12i,m_i11,m_i22,m_i12,
                                                  packed_order_GA,MP,ko,EIGEN,need_evec,rdw,0);
}

static void BandNonCol_MakeEigenRange(int id, int numprocs, int MaxN, int *is, int *ie)
{
    if (numprocs<=MaxN){
        double av_num = (double)MaxN/(double)numprocs;
        *is = (int)(av_num*(double)id) + 1;
        *ie = (int)(av_num*(double)(id+1));
        if (id==0) *is = 1;
        if (id==numprocs-1) *ie = MaxN;
    }
    else if (id<MaxN){
        *is = id + 1;
        *ie = id + 1;
    }
    else{
        *is = 1;
        *ie = 0;
    }
}

static void BandNonCol_DistributeDenseEvecGlobal(BandNonColState *state, int active_world2, int n2, int MaxN, int myid0, int myworld2,
                                                 int *NPROCS_WD2, int *Comm_World_StartID2,
                                                 int root, const dcomplex *dense_evec, int evec_on_device,
                                                 dcomplex *EVec1)
{
    const int tag = 997;
    int target_np = NPROCS_WD2[active_world2];
    int target_start = Comm_World_StartID2[active_world2];

    if (myid0==root){
        dcomplex *send_buf = NULL;
        dcomplex *panel_dev = NULL;
        int max_count = 0;

        for (int id=0; id<target_np; id++){
            int is,ie,stride,count;

            BandNonCol_MakeEigenRange(id,target_np,MaxN,&is,&ie);
            stride = ie - is + 1;
            if (stride<0) stride = 0;
            count = n2*stride;
            if (max_count<count) max_count = count;
        }

        if (0<max_count){
            send_buf = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*(size_t)max_count,0);
            if (send_buf==NULL){
                BandNonCol_AbortWithMessage("Failed to allocate dense eigenvector send buffer in Band_DFT_NonCol.c.");
            }
        }

        if (evec_on_device && 0<max_count){
            /* pack each destination's panel on the device: the host never
               materialises the full 16*n2*n2 eigenvector block, only one
               n2 x stride panel at a time */
            panel_dev = (dcomplex*)OpenMX_Resource_AccMalloc(state->resources,sizeof(dcomplex)*(size_t)max_count);
            if (panel_dev==NULL){
                /* device momentarily full: fall back to the old full readback */
                dcomplex *ev = (dcomplex*)dense_evec;
                size_t n2n2 = (size_t)n2*(size_t)n2;

#pragma acc update self(ev[0 : n2n2])
                evec_on_device = 0;
            }
        }

        for (int id=0; id<target_np; id++){
            int is,ie,stride,count;
            int target_rank = target_start + id;
            dcomplex *dst;

            BandNonCol_MakeEigenRange(id,target_np,MaxN,&is,&ie);
            stride = ie - is + 1;
            if (stride<0) stride = 0;
            count = n2*stride;
            if (count==0) continue;

            dst = (target_rank==root) ? EVec1 : send_buf;

            if (evec_on_device){
                const size_t n2n2 = (size_t)n2*(size_t)n2;
                const int is_c = is;
                const int stride_c = stride;
                dcomplex *panel = panel_dev;

if (!openmx_gpu_transform(OPENMX_GPU_COPY,1,stride_c,n2,(dcomplex*)acc_deviceptr((void*)(dense_evec))+is_c-1,n2,NULL,0,panel,stride_c,0,0.0)) {
#pragma acc parallel loop gang vector_length(128) present(dense_evec[0 : n2n2]) deviceptr(panel)
                for (int basis=0; basis<n2; basis++){
#pragma acc loop vector
                    for (int s=0; s<stride_c; s++){
                        panel[(size_t)basis*(size_t)stride_c + (size_t)s] =
                            dense_evec[(size_t)basis*(size_t)n2 + (size_t)(is_c-1) + (size_t)s];
                    }
                }
}

                acc_memcpy_from_device(dst,panel_dev,sizeof(dcomplex)*(size_t)count);
            }
            else {
                for (int basis=0; basis<n2; basis++){
                    const dcomplex *src_col = dense_evec + (size_t)basis*(size_t)n2 + (size_t)(is-1);
                    dcomplex *dst_col = dst + (size_t)basis*(size_t)stride;
                    memcpy(dst_col,src_col,sizeof(dcomplex)*(size_t)stride);
                }
            }

            if (target_rank!=root){
                MPI_Send(send_buf,count*2,MPI_DOUBLE,target_rank,tag,mpi_comm_level1);
            }
        }

        if (panel_dev!=NULL) OpenMX_Resource_AccFree(state->resources,panel_dev);
        openmx_core_native_free(state->resources,send_buf);
    }
    else if (myworld2==active_world2){
        int is,ie,stride,count;
        MPI_Status stat;

        BandNonCol_MakeEigenRange(myid0-target_start,target_np,MaxN,&is,&ie);
        stride = ie - is + 1;
        if (stride<0) stride = 0;
        count = n2*stride;
        if (0<count){
            MPI_Recv(EVec1,count*2,MPI_DOUBLE,root,tag,mpi_comm_level1,&stat);
        }
    }
}

static void BandNonCol_BuildDenseSs2(int n, int n2, const dcomplex *S, dcomplex *S2)
{
    for (int i=0; i<n2; i++){
        for (int j=0; j<n2; j++){
            size_t idx2 = (size_t)n2*(size_t)i + (size_t)j;

            if (i<n && j<n){
                size_t idx = (size_t)n*(size_t)i + (size_t)j;
                S2[idx2] = S[idx];
            }
            else if (n<=i && i<n2 && n<=j && j<n2){
                size_t idx = (size_t)n*(size_t)(i-n) + (size_t)(j-n);
                S2[idx2] = S[idx];
            }
            else{
                S2[idx2].r = 0.0;
                S2[idx2].i = 0.0;
            }
        }
    }
}

static void BandNonCol_BuildDenseHs2(int n, int n2, const dcomplex *H11,
                                     const dcomplex *H22, const dcomplex *H12,
                                     dcomplex *H2)
{
    for (int j=0; j<n2; j++){
        for (int i=0; i<n2; i++){
            size_t idx2 = (size_t)i + (size_t)j*(size_t)n2;

            if (i<n && j<n){
                size_t idx = (size_t)i + (size_t)j*(size_t)n;
                H2[idx2] = H11[idx];
            }
            else if (i<n && n<=j){
                int jj = j - n;
                size_t idx = (size_t)i + (size_t)jj*(size_t)n;
                H2[idx2] = H12[idx];
            }
            else if (n<=i && j<n){
                int ii = i - n;
                size_t idx = (size_t)j + (size_t)ii*(size_t)n;
                H2[idx2].r =  H12[idx].r;
                H2[idx2].i = -H12[idx].i;
            }
            else{
                int ii = i - n;
                int jj = j - n;
                size_t idx = (size_t)ii + (size_t)jj*(size_t)n;
                H2[idx2] = H22[idx];
            }
        }
    }
}

static void BandNonCol_BuildDenseHs2_OpenACC(int n, int n2, const dcomplex *H11,
                                             const dcomplex *H22, const dcomplex *H12,
                                             dcomplex *H2)
{
    int nn = n*n;
    int n2n2 = n2*n2;

    if (!acc_is_present(H2,sizeof(dcomplex)*(size_t)n2n2)){
#pragma acc enter data create(H2[0 : n2n2])
    }

if (!openmx_gpu_spinor(n,0,acc_deviceptr((void*)(H11)),acc_deviceptr((void*)(H22)),acc_deviceptr((void*)(H12)),NULL,NULL,NULL,acc_deviceptr((void*)(H2)))) {
#pragma acc parallel loop collapse(2) present(H11[0 : nn], H22[0 : nn], H12[0 : nn], H2[0 : n2n2])
    for (int j=0; j<n2; j++){
        for (int i=0; i<n2; i++){
            int idx2 = i + j*n2;

            if (i<n && j<n){
                int idx = i + j*n;
                H2[idx2] = H11[idx];
            }
            else if (i<n && n<=j){
                int jj = j - n;
                int idx = i + jj*n;
                H2[idx2] = H12[idx];
            }
            else if (n<=i && j<n){
                int ii = i - n;
                int idx = j + ii*n;
                H2[idx2].r =  H12[idx].r;
                H2[idx2].i = -H12[idx].i;
            }
            else{
                int ii = i - n;
                int jj = j - n;
                int idx = ii + jj*n;
                H2[idx2] = H22[idx];
            }
        }
    }
}

}

void solve_evp_complex_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, int *na_rows2_2, 
                         int *nblk2, int *mpi_comm_rows_int, int *mpi_comm_cols_int );

void elpa_solve_evp_complex_2stage_double_impl_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, 
                                                 int *na_rows2_2, int *nblk2, int *na_cols2, 
                                                 int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld );

static void Construct_Band_Ms(BandNonColState *state,  int cpx_flag, double ****Mat, double *M1, dcomplex *Ms,
                               int *MP, double k1, double k2, double k3);

static void Construct_Band_DenseMs(BandNonColState *state,  int cpx_flag, double ****Mat, double *M1, dcomplex *Ms,
                                    int *MP, double k1, double k2, double k3,
                                    int n, int owns_dense );
static void BandNonCol_PackDenseM1(BandNonColState *state,  double ****Mat, double *M1, int *MP, int *order_GA );
static void BandNonCol_ConstructDenseMsFromPacked(BandNonColState *state,  int cpx_flag, const double *M1, dcomplex *Ms,
                                                   int *order_GA, int *MP, double k1, double k2, double k3,
                                                   int n, int owns_dense );

static double Calc_DM_Band_non_collinear(const OpenMXChemicalPotentials *chemical,
    int calc_flag,
    int store_flag,
    int myid0,
    int myid2,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    int MaxN,
    double k1,
    double k2,
    double k3,
    double *****CDM,
    double *****iDM0,
    double *****EDM,
    double *ko,
    dcomplex *EVec1,
    double *rDM11,
    double *rDM22,
    double *rDM12,
    double *iDM12,
    double *iDM11,
    double *iDM22, 
    double *rEDM11,
    double *rEDM22, const OpenMXBandSpinorScratch *spinors);

double Calc_ParDM_Band_non_collinear(BandNonColState *state,
    int calc_flag,
    int store_flag,
    int myid0,
    int myid2,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    int MaxN,
    double k1,
    double k2,
    double k3,
    double *****ParDM,
    double *****iParDM,
    double *ko,
    dcomplex *EVec1,
    double *rDM11,
    double *rDM22,
    double *rDM12,
    double *iDM12,
    double *iDM11,
    double *iDM22, const OpenMXBandSpinorScratch *spinors);

#pragma GCC optimize("O2")
typedef struct {
  BandNonColState *state;
  char *mode;
  int SCF_iter;
  int knum_i;
  int knum_j;
  int knum_k;
  int SpinP_switch;
  double *****nh;
  double *****ImNL;
  double ****CntOLP;
  double *****CDM;
  double *****EDM;
  double *Eele0;
  double *Eele1;
  int *MP;
  int *order_GA;
  double *ko;
  double *koS;
  double ***EIGEN;
  double *H1;
  dcomplex *Hs11;
  dcomplex *Hs22;
  dcomplex *Hs12;
  dcomplex **EVec1;
  dcomplex *Ss;
  dcomplex *Cs;
  dcomplex *Hs;
  dcomplex *Ss2;
  dcomplex *Cs2;
  dcomplex *Hs2;
  int ***k_op;
  int *T_k_op;
  int **T_k_ID;
  double *T_KGrids1;
  double *T_KGrids2;
  double *T_KGrids3;
  int myworld1;
  int *NPROCS_ID1;
  int *Comm_World1;
  int *NPROCS_WD1;
  int *Comm_World_StartID1;
  MPI_Comm *MPI_CommWD1;
  int myworld2;
  int *NPROCS_ID2;
  int *NPROCS_WD2;
  int *Comm_World2;
  int *Comm_World_StartID2;
  MPI_Comm *MPI_CommWD2;
  const OpenMXElectronicViews *electronic;
} Band_DFT_NonCol_arguments;
static double Band_DFT_NonCol_owned(BandNonColState *state, char *mode,
                    int SCF_iter,
                    int knum_i, int knum_j, int knum_k,
		    int SpinP_switch,
		    double *****nh,
		    double *****ImNL,
		    double ****CntOLP,
		    double *****CDM,
		    double *****EDM,
		    double Eele0[2], double Eele1[2],
		    int *MP,
		    int *order_GA,
		    double *ko,
		    double *koS,
		    double ***EIGEN,
		    double *H1,
		    dcomplex *Hs11,
		    dcomplex *Hs22,
		    dcomplex *Hs12,
		    dcomplex **EVec1,
		    dcomplex *Ss,
		    dcomplex *Cs,
                    dcomplex *Hs,
		    dcomplex *Ss2,
		    dcomplex *Cs2,
                    dcomplex *Hs2,
		    int ***k_op,
		    int *T_k_op,
		    int **T_k_ID,
		    double *T_KGrids1,
		    double *T_KGrids2,
		    double *T_KGrids3,
                    int myworld1,
		    int *NPROCS_ID1,
		    int *Comm_World1,
		    int *NPROCS_WD1,
		    int *Comm_World_StartID1,
		    MPI_Comm *MPI_CommWD1,
                    int myworld2,
		    int *NPROCS_ID2,
		    int *NPROCS_WD2,
		    int *Comm_World2,
		    int *Comm_World_StartID2,
		    MPI_Comm *MPI_CommWD2, const OpenMXElectronicViews *electronic, OpenMXSolverInvocation *solver_owner);
static int Band_DFT_NonCol_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Band_DFT_NonCol_arguments *a=(Band_DFT_NonCol_arguments*)arguments;
  *result=Band_DFT_NonCol_owned(a->state, a->mode,a->SCF_iter,a->knum_i,a->knum_j,a->knum_k,a->SpinP_switch,a->nh,a->ImNL,a->CntOLP,a->CDM,a->EDM,a->Eele0,a->Eele1,a->MP,a->order_GA,a->ko,a->koS,a->EIGEN,a->H1,a->Hs11,a->Hs22,a->Hs12,a->EVec1,a->Ss,a->Cs,a->Hs,a->Ss2,a->Cs2,a->Hs2,a->k_op,a->T_k_op,a->T_k_ID,a->T_KGrids1,a->T_KGrids2,a->T_KGrids3,a->myworld1,a->NPROCS_ID1,a->Comm_World1,a->NPROCS_WD1,a->Comm_World_StartID1,a->MPI_CommWD1,a->myworld2,a->NPROCS_ID2,a->NPROCS_WD2,a->Comm_World2,a->Comm_World_StartID2,a->MPI_CommWD2,a->electronic,owner);
  return 1;
}
double Band_DFT_NonCol(BandNonColState *state,
                    char *mode,
                    int SCF_iter,
                    int knum_i, int knum_j, int knum_k,
		    int SpinP_switch,
		    double *****nh,
		    double *****ImNL,
		    double ****CntOLP,
		    double *****CDM,
		    double *****EDM,
		    double Eele0[2], double Eele1[2], 
		    int *MP,
		    int *order_GA,
		    double *ko,
		    double *koS,
		    double ***EIGEN,
		    double *H1,   
		    dcomplex *Hs11,   
		    dcomplex *Hs22,   
		    dcomplex *Hs12,   
		    dcomplex **EVec1,
		    dcomplex *Ss,
		    dcomplex *Cs,
                    dcomplex *Hs,
		    dcomplex *Ss2,
		    dcomplex *Cs2,
                    dcomplex *Hs2,
		    int ***k_op,
		    int *T_k_op,
		    int **T_k_ID,
		    double *T_KGrids1,
		    double *T_KGrids2,
		    double *T_KGrids3,
                    int myworld1,
		    int *NPROCS_ID1,
		    int *Comm_World1,
		    int *NPROCS_WD1,
		    int *Comm_World_StartID1,
		    MPI_Comm *MPI_CommWD1,
                    int myworld2,
		    int *NPROCS_ID2,
		    int *NPROCS_WD2,
		    int *Comm_World2,
		    int *Comm_World_StartID2,
		    MPI_Comm *MPI_CommWD2, const OpenMXElectronicViews *electronic)
{
  Band_DFT_NonCol_arguments arguments={state,mode,SCF_iter,knum_i,knum_j,knum_k,SpinP_switch,nh,ImNL,CntOLP,CDM,EDM,Eele0,Eele1,MP,order_GA,ko,koS,EIGEN,H1,Hs11,Hs22,Hs12,EVec1,Ss,Cs,Hs,Ss2,Cs2,Hs2,k_op,T_k_op,T_k_ID,T_KGrids1,T_KGrids2,T_KGrids3,myworld1,NPROCS_ID1,Comm_World1,NPROCS_WD1,Comm_World_StartID1,MPI_CommWD1,myworld2,NPROCS_ID2,NPROCS_WD2,Comm_World2,Comm_World_StartID2,MPI_CommWD2,electronic};
  return OpenMX_Solver_Run(Band_DFT_NonCol_operation,&arguments);
}

static double Band_DFT_NonCol_owned(BandNonColState *state, char *mode,
                    int SCF_iter,
                    int knum_i, int knum_j, int knum_k,
		    int SpinP_switch,
		    double *****nh,
		    double *****ImNL,
		    double ****CntOLP,
		    double *****CDM,
		    double *****EDM,
		    double Eele0[2], double Eele1[2],
		    int *MP,
		    int *order_GA,
		    double *ko,
		    double *koS,
		    double ***EIGEN,
		    double *H1,
		    dcomplex *Hs11,
		    dcomplex *Hs22,
		    dcomplex *Hs12,
		    dcomplex **EVec1,
		    dcomplex *Ss,
		    dcomplex *Cs,
                    dcomplex *Hs,
		    dcomplex *Ss2,
		    dcomplex *Cs2,
                    dcomplex *Hs2,
		    int ***k_op,
		    int *T_k_op,
		    int **T_k_ID,
		    double *T_KGrids1,
		    double *T_KGrids2,
		    double *T_KGrids3,
                    int myworld1,
		    int *NPROCS_ID1,
		    int *Comm_World1,
		    int *NPROCS_WD1,
		    int *Comm_World_StartID1,
		    MPI_Comm *MPI_CommWD1,
                    int myworld2,
		    int *NPROCS_ID2,
		    int *NPROCS_WD2,
		    int *Comm_World2,
		    int *Comm_World_StartID2,
		    MPI_Comm *MPI_CommWD2, const OpenMXElectronicViews *electronic, OpenMXSolverInvocation *solver_owner)
{
  OpenMXBandSpinorScratch spinors;

  double *****ParDM=electronic->ParDM;
  double ******iDM=electronic->iDM;
  double *****iParDM=electronic->iParDM;

  int i,j,k,l,m,n,n2,p,wan,MaxN,i0,ks;
  int i1,i1s,j1,ia,jb,lmax,po,po1,spin,s1,e1;
  int num2,RnB,l1,l2,l3,loop_num,ns,ne;
  int ct_AN,h_AN,wanA,tnoA,wanB,tnoB;
  int MA_AN,GA_AN,Anum,num_kloop0,max_num_kloop0;
  int T_knum,S_knum,E_knum,kloop,kloop0;
  double av_num,lumos;
  double time0;
  int LB_AN,GB_AN,Bnum;
  double k1,k2,k3;
  double sum,sumi,sum_weights;
  double Num_State;
  double My_Num_State;
  double FermiF;
  double tmp,tmp1,eig,EV_cut0;
  double x,Dnum,Dnum2,AcP,ChemP_MAX,ChemP_MIN;
  int *is1,*ie1;
  int *is2,*ie2;
  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;

  int all_knum; 
  dcomplex Ctmp1,Ctmp2;
  int ii,ij,ik;
  int BM,BN,BK;
  double u2,v2,uv,vu;
  double d1,d2;
  double My_Eele1[2]; 
  double TZ,dum,sumE,kRn,si,co;
  double Resum,ResumE,Redum,Redum2;
  double Imsum,ImsumE,Imdum,Imdum2;
  double TStime,TEtime,SiloopTime,EiloopTime;
  double Stime,Etime,Stime0,Etime0;
  double x_cut=60.0;
  double My_Eele0[2];

  char file_EV[YOUSO10];
  FILE *fp_EV;
  char buf[fp_bsize];          /* setvbuf */

  int AN,Rn,size_H1;
  int parallel_mode;
  int use_root_dense_gpusolver;
  int use_k_dense_gpusolver;
  int root_dense_serial_gpusolver_worlds = 0;
  int owns_dense_k_rank;
  int *dense_k_owner = NULL;
  int *dense_k_order = NULL;
  int interleave_k = 1;
  dcomplex **k_evec_cache = NULL;
  size_t k_cache_bytes = 0, k_cache_used = 0;
  int k_cache_hits = 0, k_overlap_hits = 0;
  double *k_packed_m1[8] = {NULL};
  int k_packed_ready = 0;
  size_t k_packed_bytes = 0;
  int numprocs0,myid0;
  int ID,ID0,ID1;
  int numprocs1,myid1;
  int numprocs2,myid2;
  int Num_Comm_World1;
  int Num_Comm_World2;

  int tag=999,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;

  double time1,time2,time3;
  double time4,time5,time6;
  double time7,time8,time9;
  double time10,time11,time12,time13;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  FILE* file;
  char* BUF[1000];

  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  int info,ig,jg,il,jl,prow,pcol,brow,bcol;
  int ZERO=0, ONE=1;
  dcomplex alpha = {1.0,0.0}; dcomplex beta = {0.0,0.0};
  int LOCr, LOCc, node, irow, icol;
  double mC_spin_i1,C_spin_i1;

  int Max_Num_Snd_EV,Max_Num_Rcv_EV;
  int *Num_Snd_EV,*Num_Rcv_EV;
  int *index_Snd_i,*index_Snd_j,*index_Rcv_i,*index_Rcv_j;
  double *EVec_Snd,*EVec_Rcv;
  double *rDM11,*rDM22,*rDM12,*iDM12,*iDM11,*iDM22,*rEDM11,*rEDM22;

  /* for time */
  dtime(&TStime);

  /* New inputs and geometry updates both start a new SCF sequence. */
  if (SCF_iter <= 1) BandNonCol_KOverlapReset(state);

  time1 = 0.0;
  time2 = 0.0;
  time3 = 0.0;
  time4 = 0.0;
  time5 = 0.0;
  time6 = 0.0;
  time7 = 0.0;
  time8 = 0.0;
  time9 = 0.0;
  time10 = 0.0;
  time11 = 0.0;
  time12 = 0.0;
  time13 = 0.0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);
  MPI_Barrier(mpi_comm_level1);

  Num_Comm_World1 = 1;

  /*********************************************** 
       for pallalel calculations in myworld1
  ***********************************************/

  MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

  /****************************************************
   find the number of basis functions, n
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n += Spe_Total_CNO[wanA];
  }
  n2 = n*2;

  /* GPU dispatch (added by H.Kawai): assign CUDA/OpenACC device when GPUSOLVER is requested */
  if (scf_eigen_lib_flag == GPUSOLVER && n2 >= Band_DFT_NonCol_GpuSwitchNum() &&
      Set_Hamiltonian_OpenACC_Rank_Is_Selected()) {
      set_cuda_default_device_from_local_rank_noncollective();
      set_openacc_nvidia_device_from_local_rank_noncollective();
  }

  /****************************************************
   find TZ
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ += Spe_Core_Charge[wan];
  }

  /***********************************************
     find the number of states to be solved 
  ***********************************************/

  lumos = (double)(TZ-system_charge)*0.200;
  if (lumos<60.0) lumos = 200.0;
  MaxN = (TZ-system_charge) + (int)lumos;
  if (n2<MaxN) MaxN = n2;

  /***********************************************
     allocation of arrays
  ***********************************************/

  My_NZeros=OpenMX_Solver_Int(solver_owner,"My_NZeros",((size_t)1)*numprocs0);
  SP_NZeros=OpenMX_Solver_Int(solver_owner,"SP_NZeros",((size_t)1)*numprocs0);
  SP_Atoms=OpenMX_Solver_Int(solver_owner,"SP_Atoms",((size_t)1)*numprocs0);

  spinors.ReEVec_i1=OpenMX_Solver_RealView(solver_owner,"ReEVec_i1",((size_t)1)*List_YOUSO[7],((size_t)1)*(MaxN+1));
  for (i=0; i<List_YOUSO[7]; i++){
  }

  spinors.ImEVec_i1=OpenMX_Solver_RealView(solver_owner,"ImEVec_i1",((size_t)1)*List_YOUSO[7],((size_t)1)*(MaxN+1));
  for (i=0; i<List_YOUSO[7]; i++){
  }

  spinors.ReEVec_i2=OpenMX_Solver_RealView(solver_owner,"ReEVec_i2",((size_t)1)*List_YOUSO[7],((size_t)1)*(MaxN+1));
  for (i=0; i<List_YOUSO[7]; i++){
  }

  spinors.ImEVec_i2=OpenMX_Solver_RealView(solver_owner,"ImEVec_i2",((size_t)1)*List_YOUSO[7],((size_t)1)*(MaxN+1));
  for (i=0; i<List_YOUSO[7]; i++){
  }

  spinors.ReEVec_j1=OpenMX_Solver_RealView(solver_owner,"ReEVec_j1",((size_t)1)*List_YOUSO[7],((size_t)1)*(MaxN+1));
  for (i=0; i<List_YOUSO[7]; i++){
  }

  spinors.ImEVec_j1=OpenMX_Solver_RealView(solver_owner,"ImEVec_j1",((size_t)1)*List_YOUSO[7],((size_t)1)*(MaxN+1));
  for (i=0; i<List_YOUSO[7]; i++){
  }

  spinors.ReEVec_j2=OpenMX_Solver_RealView(solver_owner,"ReEVec_j2",((size_t)1)*List_YOUSO[7],((size_t)1)*(MaxN+1));
  for (i=0; i<List_YOUSO[7]; i++){
  }

  spinors.ImEVec_j2=OpenMX_Solver_RealView(solver_owner,"ImEVec_j2",((size_t)1)*List_YOUSO[7],((size_t)1)*(MaxN+1));
  for (i=0; i<List_YOUSO[7]; i++){
  }

  /****************************************************
   find size_H
  ****************************************************/

  size_H1 = Get_OneD_HS_Col(0, nh[0], &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  /***********************************************
     allocation of arrays 
  ***********************************************/

  rDM11=OpenMX_Solver_Real(solver_owner,"rDM11",((size_t)1)*size_H1);
  rDM22=OpenMX_Solver_Real(solver_owner,"rDM22",((size_t)1)*size_H1);
  rDM12=OpenMX_Solver_Real(solver_owner,"rDM12",((size_t)1)*size_H1);
  iDM11=OpenMX_Solver_Real(solver_owner,"iDM11",((size_t)1)*size_H1);
  iDM22=OpenMX_Solver_Real(solver_owner,"iDM22",((size_t)1)*size_H1);
  iDM12=OpenMX_Solver_Real(solver_owner,"iDM12",((size_t)1)*size_H1);
  rEDM11=OpenMX_Solver_Real(solver_owner,"rEDM11",((size_t)1)*size_H1);
  rEDM22=OpenMX_Solver_Real(solver_owner,"rEDM22",((size_t)1)*size_H1);

  for (i=0; i<size_H1; i++){
    rDM11[i] = 0.0;
    rDM22[i] = 0.0;
    rDM12[i] = 0.0;
    iDM11[i] = 0.0;
    iDM22[i] = 0.0;
    iDM12[i] = 0.0;
    rEDM11[i] = 0.0;
    rEDM22[i] = 0.0;
  }

  /***********************************************
          initialize CDM, EDM, and iDM
  ***********************************************/

  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];    
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
	  CDM[0][MA_AN][LB_AN][i][j] = 0.0;
	  CDM[1][MA_AN][LB_AN][i][j] = 0.0;
	  CDM[2][MA_AN][LB_AN][i][j] = 0.0;
	  CDM[3][MA_AN][LB_AN][i][j] = 0.0;
	  EDM[0][MA_AN][LB_AN][i][j] = 0.0;
	  EDM[1][MA_AN][LB_AN][i][j] = 0.0;
	  EDM[2][MA_AN][LB_AN][i][j] = 0.0;
	  EDM[3][MA_AN][LB_AN][i][j] = 0.0;
	  iDM[0][0][MA_AN][LB_AN][i][j] = 0.0;
	  iDM[0][1][MA_AN][LB_AN][i][j] = 0.0;
	}
      }
    }
  }

  /***********************************************
              k-points by regular mesh 
  ***********************************************/

  for (i=0;i<knum_i;i++) {
    for (j=0;j<knum_j;j++) {
      for (k=0;k<knum_k;k++) {
	k_op[i][j][k] = 1;
      }
    }
  }

  /***********************************
       one-dimentionalize for MPI
  ************************************/

  T_knum = 0;
  for (i=0; i<knum_i; i++){
    for (j=0; j<knum_j; j++){
      for (k=0; k<knum_k; k++){
	if (0<k_op[i][j][k]){
	  T_knum++;
	}
      }
    }
  }

  /* set T_KGrids1,2,3 and T_k_op */

  /* Added by N. Yamaguchi ***/
  if (way_of_kpoint==1){
    /* ***/

    T_knum = 0;
    for (i=0; i<knum_i; i++){

      if (knum_i==1)  k1 = 0.0;
      else            k1 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_i) + Shift_K_Point;

      for (j=0; j<knum_j; j++){

	if (knum_j==1)  k2 = 0.0;
	else            k2 = -0.5 + (2.0*(double)j+1.0)/(2.0*(double)knum_j) - Shift_K_Point;

	for (k=0; k<knum_k; k++){

	  if (knum_k==1)  k3 = 0.0;
	  else            k3 = -0.5 + (2.0*(double)k+1.0)/(2.0*(double)knum_k) + 2.0*Shift_K_Point;

	  if (0<k_op[i][j][k]){

	    T_KGrids1[T_knum] = k1;
	    T_KGrids2[T_knum] = k2;
	    T_KGrids3[T_knum] = k3;
	    T_k_op[T_knum]    = k_op[i][j][k];

	    T_knum++;
	  }
	}
      }
    }

    if (myid0==Host_ID && 0<level_stdout){

      printf(" KGrids1: ");fflush(stdout);
      for (i=0;i<=knum_i-1;i++){
	if (knum_i==1)  k1 = 0.0;
	else            k1 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_i) + Shift_K_Point;
	printf("%9.5f ",k1);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids2: ");fflush(stdout);

      for (i=0;i<=knum_j-1;i++){
	if (knum_j==1)  k2 = 0.0;
	else            k2 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_j) - Shift_K_Point;
	printf("%9.5f ",k2);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids3: ");fflush(stdout);
      for (i=0;i<=knum_k-1;i++){
	if (knum_k==1)  k3 = 0.0;
	else            k3 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_k) + 2.0*Shift_K_Point;
	printf("%9.5f ",k3);fflush(stdout);
      }
      printf("\n");fflush(stdout);
    }
  } // end of if (way_of_kpoint==1)

  /* Added by N. Yamaguchi ***/
  /***********************************************
          k-points by a Gamma-centered mesh
   ***********************************************/
  
  else if (way_of_kpoint==3){
    
    T_knum = 0;
    for (i=0; i<knum_i; i++){

      if (knum_i==1)  k1 = 0.0;
      else            k1 = ((double)i)/((double)knum_i) + Shift_K_Point;

      for (j=0; j<knum_j; j++){

        if (knum_j==1)  k2 = 0.0;
        else            k2 = ((double)j)/((double)knum_j) - Shift_K_Point;

        for (k=0; k<knum_k; k++){

          if (knum_k==1)  k3 = 0.0;
          else            k3 = ((double)k)/((double)knum_k) + 2.0*Shift_K_Point;

          if (0<k_op[i][j][k]){

            T_KGrids1[T_knum] = k1;
            T_KGrids2[T_knum] = k2;
            T_KGrids3[T_knum] = k3;
            T_k_op[T_knum]    = k_op[i][j][k];

            T_knum++;
          }
        }
      }
    }

   if (myid0==Host_ID && 0<level_stdout){

      printf(" KGrids1: ");fflush(stdout);
      for (i=0;i<=knum_i-1;i++){
        if (knum_i==1)  k1 = 0.0;
        else            k1 = ((double)i)/((double)knum_i) + Shift_K_Point;
        printf("%9.5f ",k1);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids2: ");fflush(stdout);

      for (i=0;i<=knum_j-1;i++){
        if (knum_j==1)  k2 = 0.0;
        else            k2 = ((double)i)/((double)knum_j) - Shift_K_Point;
        printf("%9.5f ",k2);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids3: ");fflush(stdout);
      for (i=0;i<=knum_k-1;i++){
        if (knum_k==1)  k3 = 0.0;
        else            k3 = ((double)i)/((double)knum_k) + 2.0*Shift_K_Point;
        printf("%9.5f ",k3);fflush(stdout);
      }
      printf("\n");fflush(stdout);
    }
  } // end of else if (way_of_kpoint==3)
  /* ***/

  /***********************************************
            calculate the sum of weights
  ***********************************************/

  sum_weights = 0.0;
  for (k=0; k<T_knum; k++){
    sum_weights += (double)T_k_op[k];
  }

  /***********************************************
           allocate k-points into processors 
  ***********************************************/

  if (numprocs1<T_knum){

    /* set parallel_mode */
    parallel_mode = 0;

    /* allocation of kloop to ID */     

    for (ID=0; ID<numprocs1; ID++){
      tmp = (double)T_knum/(double)numprocs1;
      S_knum = (int)((double)ID*(tmp+1.0e-12)); 
      E_knum = (int)((double)(ID+1)*(tmp+1.0e-12)) - 1;
      if (ID==(numprocs1-1)) E_knum = T_knum - 1;
      if (E_knum<0)          E_knum = 0;

      for (k=S_knum; k<=E_knum; k++){
        /* ID in the first level world */
        T_k_ID[myworld1][k] = ID;
      }
    }

    /* find own informations */

    tmp = (double)T_knum/(double)numprocs1; 
    S_knum = (int)((double)myid1*(tmp+1.0e-12)); 
    E_knum = (int)((double)(myid1+1)*(tmp+1.0e-12)) - 1;
    if (myid1==(numprocs1-1)) E_knum = T_knum - 1;
    if (E_knum<0)             E_knum = 0;

    num_kloop0 = E_knum - S_knum + 1;

    MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);
  }

  else {

    /* set parallel_mode */
    parallel_mode = 1;
    num_kloop0 = 1;

    Num_Comm_World2 = T_knum;
    MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);

    S_knum = myworld2;

    /* allocate k-points into processors */
    
    for (k=0; k<T_knum; k++){
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
	  MPI_Allreduce(&num_kloop0, &max_num_kloop0, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

	  /* the refined eigensolver: its switch (collective, once; the memory
	     probe counts its basis), and no refined solve until the first halves
	     of this call say so */
	  BandNonCol_RefineConfigure(state);
	  state->refine.last_refined = 0;

	  if (scf_eigen_lib_flag==GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2)){
	    (void)BandNonCol_GpuDiagFits(state, SCF_iter,n,n2,MaxN,size_H1,myid0,T_knum);
	  }
	  else {
	    /* not a GPU-dense run; keep the shared verdict neutral so the helper
	       branches follow their other conditions unchanged */
	    state->gpu_dense_verdict = 1;
	  }
	  use_root_dense_gpusolver = (scf_eigen_lib_flag==GPUSOLVER && all_knum==1 && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2) &&
	                              state->gpu_dense_verdict);
	  use_k_dense_gpusolver = (scf_eigen_lib_flag==GPUSOLVER && all_knum!=1 &&
	                          (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2) && strcasecmp(mode,"scf")==0 &&
	                          state->gpu_dense_verdict);
	  if (use_k_dense_gpusolver){
	    /* the refined eigensolver on the k-dense path (every rank) */
	    BandNonCol_RefineBeginStep(state, SCF_iter, 0, 1, 0);
	    if (state->refine.kdense) BandNonCol_KRefEnsure(state, T_knum, n2);
	  }
	  owns_dense_k_rank = (use_k_dense_gpusolver && Set_Hamiltonian_OpenACC_Rank_Is_Selected());
	  if (use_k_dense_gpusolver){
	    dense_k_owner=OpenMX_Solver_Int(solver_owner,"dense_k_owner",((size_t)1)*(size_t)T_knum);
	    dense_k_order=OpenMX_Solver_Int(solver_owner,"dense_k_order",((size_t)1)*(size_t)T_knum);
	    if (dense_k_owner==NULL || dense_k_order==NULL){
	      BandNonCol_AbortWithMessage("Failed to allocate dense k-point owner table in Band_DFT_NonCol.c.");
	    }

	    for (k=0; k<T_knum; k++){
	      int dense_owner_candidate = INT_MAX;
	      int dense_owner;

	      if (owns_dense_k_rank && S_knum<=k && k<S_knum+num_kloop0){
	        dense_owner_candidate = myid0;
	      }
	      MPI_Allreduce(&dense_owner_candidate,&dense_owner,1,MPI_INT,MPI_MIN,mpi_comm_level1);

	      if (dense_owner==INT_MAX){
	        BandNonCol_AbortWithMessage("Failed to select dense k-point GpuSolver owner in Band_DFT_NonCol.c.");
	      }

	      dense_k_owner[k] = dense_owner;
	    }
	    interleave_k = openmx_env_flag_at(OPENMX_ENV_BAND_NONCOL_INTERLEAVE_K, interleave_k);
	    if (interleave_k)
	      BandNonCol_InterleaveKOrder(state, T_knum,numprocs0,dense_k_owner,dense_k_order);
	    else
	      for (k=0; k<T_knum; ++k) dense_k_order[k] = k;
	  }
	  if (use_root_dense_gpusolver || use_k_dense_gpusolver){
	    BandNonCol_SetDenseGemmul8Defaults();
	    MPI_Barrier(mpi_comm_level1);
	  }

	  /****************************************************
	                make is1, ie1, is2, ie2
  ****************************************************/

  /* allocation */

  is1=OpenMX_Solver_Int(solver_owner,"is1",((size_t)1)*numprocs2);
  ie1=OpenMX_Solver_Int(solver_owner,"ie1",((size_t)1)*numprocs2);
  is2=OpenMX_Solver_Int(solver_owner,"is2",((size_t)1)*numprocs2);
  ie2=OpenMX_Solver_Int(solver_owner,"ie2",((size_t)1)*numprocs2);

  Num_Snd_EV=OpenMX_Solver_Int(solver_owner,"Num_Snd_EV",((size_t)1)*numprocs2);
  Num_Rcv_EV=OpenMX_Solver_Int(solver_owner,"Num_Rcv_EV",((size_t)1)*numprocs2);

  /* make is1 and ie1 */ 

  if ( numprocs2<=n ){

    av_num = (double)n/(double)numprocs2;

    for (ID=0; ID<numprocs2; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs2-1] = n; 

  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=n; ID<numprocs2; ID++){
      is1[ID] =  1;
      ie1[ID] =  0;
    }
  }

  /* make is2 and ie2 */ 

  if ( numprocs2<=MaxN ){

    av_num = (double)MaxN/(double)numprocs2;

    for (ID=0; ID<numprocs2; ID++){
      is2[ID] = (int)(av_num*(double)ID) + 1; 
      ie2[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is2[0] = 1;
    ie2[numprocs2-1] = MaxN; 
  }

  else{
    for (ID=0; ID<MaxN; ID++){
      is2[ID] = ID + 1; 
      ie2[ID] = ID + 1;
    }
    for (ID=MaxN; ID<numprocs2; ID++){
      is2[ID] = 1;
      ie2[ID] = 0;
    }
  }

  /****************************************************************
    making data structure of MPI communicaition for eigenvectors 
  ****************************************************************/

  for (ID=0; ID<numprocs2; ID++){
    Num_Snd_EV[ID] = 0;
    Num_Rcv_EV[ID] = 0;
  }

  for(i=0; i<na_rows2; i++){

    ig = np_rows2*nblk2*((i)/nblk2) + (i)%nblk2 + ((np_rows2+my_prow2)%np_rows2)*nblk2 + 1;

    po = 0;
    for (ID=0; ID<numprocs2; ID++){
      if (is2[ID]<=ig && ig <=ie2[ID]){
	po = 1;
	ID0 = ID;
	break;
      }
    }

    if (po==1) Num_Snd_EV[ID0] += na_cols2;
  }

  for (ID=0; ID<numprocs2; ID++){
    IDS = (myid2 + ID) % numprocs2;
    IDR = (myid2 - ID + numprocs2) % numprocs2;
    if (ID!=0){
      MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
      MPI_Recv(&Num_Rcv_EV[IDR],  1, MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
      MPI_Wait(&request,&stat);
    }
    else{
      Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
    }
  }

  Max_Num_Snd_EV = 0;
  Max_Num_Rcv_EV = 0;
  for (ID=0; ID<numprocs2; ID++){
    if (Max_Num_Snd_EV<Num_Snd_EV[ID]) Max_Num_Snd_EV = Num_Snd_EV[ID];
    if (Max_Num_Rcv_EV<Num_Rcv_EV[ID]) Max_Num_Rcv_EV = Num_Rcv_EV[ID];
  }  

  Max_Num_Snd_EV++;
  Max_Num_Rcv_EV++;

  index_Snd_i=OpenMX_Solver_Int(solver_owner,"index_Snd_i",((size_t)1)*Max_Num_Snd_EV);
  index_Snd_j=OpenMX_Solver_Int(solver_owner,"index_Snd_j",((size_t)1)*Max_Num_Snd_EV);
  EVec_Snd=OpenMX_Solver_Real(solver_owner,"EVec_Snd",((size_t)1)*Max_Num_Snd_EV*2);
  index_Rcv_i=OpenMX_Solver_Int(solver_owner,"index_Rcv_i",((size_t)1)*Max_Num_Rcv_EV);
  index_Rcv_j=OpenMX_Solver_Int(solver_owner,"index_Rcv_j",((size_t)1)*Max_Num_Rcv_EV);
  EVec_Rcv=OpenMX_Solver_Real(solver_owner,"EVec_Rcv",((size_t)1)*Max_Num_Rcv_EV*2);

  /****************************************************
                      PrintMemory
  ****************************************************/

  if (state->Band_DFT_NonCol_owned_firsttime && memoryusage_fileout) {
  PrintMemory("Band_DFT_NonCol: My_NZeros", sizeof(int)*numprocs0,NULL);
  PrintMemory("Band_DFT_NonCol: SP_NZeros", sizeof(int)*numprocs0,NULL);
  PrintMemory("Band_DFT_NonCol: SP_Atoms", sizeof(int)*numprocs0,NULL);
  PrintMemory("Band_DFT_NonCol: is1", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: ie1", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: is2", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: ie2", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: Num_Snd_EV", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: Num_Rcv_EV", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: index_Snd_i", sizeof(int)*Max_Num_Snd_EV,NULL);
  PrintMemory("Band_DFT_NonCol: index_Snd_j", sizeof(int)*Max_Num_Snd_EV,NULL);
  PrintMemory("Band_DFT_NonCol: EVec_Snd", sizeof(double)*Max_Num_Snd_EV*2,NULL);
  PrintMemory("Band_DFT_NonCol: index_Rcv_i", sizeof(int)*Max_Num_Rcv_EV,NULL);
  PrintMemory("Band_DFT_NonCol: index_Rcv_j", sizeof(int)*Max_Num_Rcv_EV,NULL);
  PrintMemory("Band_DFT_NonCol: EVec_Rcv", sizeof(double)*Max_Num_Rcv_EV*2,NULL);

  PrintMemory("Band_DFT_NonCol: rDM11", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: rDM22", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: rDM12", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: iDM11", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: iDM22", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: iDM12", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: rEDM11", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: rEDM22", sizeof(double)*size_H1,NULL);
  }

  /****************************************************
                      start kloop
  ****************************************************/

  dtime(&SiloopTime);

  if (use_k_dense_gpusolver){

    /* Real-space matrices are unchanged across all k groups in this SCF.
       Retain a bounded host copy on every owner only when all owners fit;
       otherwise keep the original per-group allocation and gather path. */
    int local_packed_ready = !owns_dense_k_rank;
    if (owns_dense_k_rank && size_H1 > 0 &&
        (size_t)size_H1 <= SIZE_MAX / (8U * sizeof(double))){
      size_t bytes = 8U * (size_t)size_H1 * sizeof(double);
      if (bytes <= BandNonCol_HostCacheLimit(OPENMX_ENV_BAND_NONCOL_PACK_CACHE_MB,128.0,64)){
        k_packed_m1[0] = (double*)openmx_core_native_malloc(state->resources,1,bytes,0);
        if (k_packed_m1[0] != NULL){
          for (int part=1; part<8; ++part)
            k_packed_m1[part] = k_packed_m1[0] + (size_t)part * (size_t)size_H1;
          k_packed_bytes = bytes;
          local_packed_ready = 1;
        }
      }
    }
    MPI_Allreduce(&local_packed_ready,&k_packed_ready,1,MPI_INT,MPI_MIN,mpi_comm_level1);
    if (k_packed_ready){
      double ****matrices[8] = {CntOLP,nh[0],nh[1],nh[2],nh[3],ImNL[0],ImNL[1],ImNL[2]};
      double *discard = NULL;
      if (!owns_dense_k_rank){
        discard=OpenMX_Solver_Real(solver_owner,"discard",((size_t)1)*(size_t)size_H1);
        if (discard == NULL)
          BandNonCol_AbortWithMessage("Failed to allocate dense k-point packing buffer in Band_DFT_NonCol.c.");
      }
      for (int part=0; part<8; ++part)
        BandNonCol_PackDenseM1(state, matrices[part],owns_dense_k_rank ? k_packed_m1[part] : discard,MP,order_GA);
    }
    else {
      openmx_core_native_free(state->resources,k_packed_m1[0]);
      for (int part=0; part<8; ++part) k_packed_m1[part] = NULL;
      k_packed_bytes = 0;
    }

    int local_cache_vectors = 0, cache_vectors = 0;
    if (owns_dense_k_rank) BandNonCol_KOverlapPrepare(state, n,T_knum);
    if (owns_dense_k_rank && n2 > 0 && MaxN > 0 &&
        (size_t)n2 <= SIZE_MAX / (size_t)MaxN / sizeof(dcomplex)){
      k_cache_bytes = (size_t)n2 * (size_t)MaxN * sizeof(dcomplex);
      if (k_cache_bytes <= BandNonCol_KCacheLimit())
        k_evec_cache = (dcomplex**)openmx_core_native_malloc(state->resources,(size_t)T_knum,sizeof(dcomplex*),1);
      local_cache_vectors = (k_evec_cache != NULL);
    }
    /* Every rank must use the same concurrency schedule. If any owner can
       retain vectors, budget the backtransform's larger peak in pass one. */
    MPI_Allreduce(&local_cache_vectors,&cache_vectors,1,MPI_INT,MPI_MAX,mpi_comm_level1);
    int max_concurrent_gpu_turns =
      BandNonCol_MaxConcurrentKGpuTurns(state, n,n2,MaxN,size_H1,
                                        cache_vectors ? BANDNONCOL_GPU_TURN_EIGVECS : BANDNONCOL_GPU_TURN_EIGVALS,
                                        owns_dense_k_rank,myid0,T_knum);
    BandNonColRootDenseWorkspace *rdw;
    double *pack_buffer = NULL;
    double *m_olp = k_packed_m1[0];
    double *m_h11 = k_packed_m1[1];
    double *m_h22 = k_packed_m1[2];
    double *m_h12 = k_packed_m1[3];
    double *m_h12i = k_packed_m1[4];
    double *m_i11 = k_packed_m1[5];
    double *m_i22 = k_packed_m1[6];
    double *m_i12 = k_packed_m1[7];

    for (int group_first=0; group_first<T_knum; group_first+=max_concurrent_gpu_turns){
      int group_last =
        (group_first + max_concurrent_gpu_turns < T_knum) ?
        (group_first + max_concurrent_gpu_turns) : T_knum;
      int owns_dense_k_group = 0;

      for (int gpu_turn=group_first; gpu_turn<group_last; gpu_turn++){
        if (dense_k_owner[dense_k_order[gpu_turn]]==myid0){
          owns_dense_k_group = 1;
          break;
        }
      }

      MPI_Barrier(mpi_comm_level1);

      if (!k_packed_ready){
        if (owns_dense_k_group){
          m_olp=OpenMX_Solver_Real(solver_owner,"m_olp",((size_t)1)*(size_t)size_H1);
          m_h11=OpenMX_Solver_Real(solver_owner,"m_h11",((size_t)1)*(size_t)size_H1);
          m_h22=OpenMX_Solver_Real(solver_owner,"m_h22",((size_t)1)*(size_t)size_H1);
          m_h12=OpenMX_Solver_Real(solver_owner,"m_h12",((size_t)1)*(size_t)size_H1);
          m_h12i=OpenMX_Solver_Real(solver_owner,"m_h12i",((size_t)1)*(size_t)size_H1);
          m_i11=OpenMX_Solver_Real(solver_owner,"m_i11",((size_t)1)*(size_t)size_H1);
          m_i22=OpenMX_Solver_Real(solver_owner,"m_i22",((size_t)1)*(size_t)size_H1);
          m_i12=OpenMX_Solver_Real(solver_owner,"m_i12",((size_t)1)*(size_t)size_H1);

          if (m_olp==NULL || m_h11==NULL || m_h22==NULL || m_h12==NULL ||
              m_h12i==NULL || m_i11==NULL || m_i22==NULL || m_i12==NULL){
            BandNonCol_AbortWithMessage("Failed to allocate dense k-point packed matrices in Band_DFT_NonCol.c.");
          }
        }
        else {
          pack_buffer=OpenMX_Solver_Real(solver_owner,"pack_buffer",((size_t)1)*(size_t)size_H1);
          if (pack_buffer==NULL){
            BandNonCol_AbortWithMessage("Failed to allocate dense k-point packing buffer in Band_DFT_NonCol.c.");
          }
        }

        BandNonCol_PackDenseM1(state, CntOLP, owns_dense_k_group ? m_olp : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, nh[0],  owns_dense_k_group ? m_h11 : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, nh[1],  owns_dense_k_group ? m_h22 : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, nh[2],  owns_dense_k_group ? m_h12 : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, nh[3],  owns_dense_k_group ? m_h12i : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, ImNL[0],owns_dense_k_group ? m_i11 : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, ImNL[1],owns_dense_k_group ? m_i22 : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, ImNL[2],owns_dense_k_group ? m_i12 : pack_buffer,MP,order_GA);
        pack_buffer = NULL;
      }

      for (int gpu_turn=group_first; gpu_turn<group_last; gpu_turn++){

        kloop = dense_k_order[gpu_turn];
        if (dense_k_owner[kloop]!=myid0) continue;

        k1 = T_KGrids1[kloop];
        k2 = T_KGrids2[kloop];
        k3 = T_KGrids3[kloop];

        openmx_mgpu_scope_state mgpu_scope =
          openmx_mgpu_band_scope_enter(!gpusolver2_flag,T_knum,kloop);
        rdw = BandNonCol_RootDenseWorkspace_Ensure(state, 1,n,n2,MaxN,1,SCF_iter);
        if (k_evec_cache != NULL && !state->refine.kdense){
          size_t limit = BandNonCol_KCacheLimit();
          if (k_cache_used <= limit && k_cache_bytes <= limit - k_cache_used)
            k_evec_cache[kloop] = (dcomplex*)openmx_core_native_malloc(state->resources,1,k_cache_bytes,0);
        }
        {
          int save_evec = (k_evec_cache != NULL && k_evec_cache[kloop] != NULL);
          int reuse_overlap = BandNonCol_KOverlapRestore(state, kloop,k1,k2,k3,rdw->s_all);
          k_overlap_hits += reuse_overlap;
          if (state->refine.kdense){
            /* the refined pass 2 recomputes the vectors from the kept basis */
            BandNonCol_KDenseRefinedSolveOneK(state, !reuse_overlap,n,n2,MaxN,kloop,k1,k2,k3,
                                              m_olp,m_h11,m_h22,m_h12,m_h12i,
                                              m_i11,m_i22,m_i12,
                                              order_GA,MP,ko,EIGEN,rdw,&state->kref[kloop],0,state->chemical->chemical_potential);
            if (!reuse_overlap) BandNonCol_KOverlapSave(state, kloop,k1,k2,k3,rdw->s_all);
            if (save_evec){
              openmx_core_native_free(state->resources,k_evec_cache[kloop]);
              k_evec_cache[kloop] = NULL;
            }
          }
          else {
          BandNonCol_RootDenseSolveOneK_OpenACC(state, !reuse_overlap,n,n2,MaxN,kloop,k1,k2,k3,
                                              m_olp,m_h11,m_h22,m_h12,m_h12i,
                                              m_i11,m_i22,m_i12,
                                              order_GA,MP,ko,EIGEN,save_evec,rdw);
          if (!reuse_overlap) BandNonCol_KOverlapSave(state, kloop,k1,k2,k3,rdw->s_all);
          if (save_evec){
            BandNonCol_KCacheSave(k_evec_cache[kloop],rdw);
            k_cache_used += k_cache_bytes;
          }
          }
        }
        openmx_mgpu_band_scope_leave(mgpu_scope);
      }

      if (owns_dense_k_group){
        if (!k_packed_ready){
          m_olp = NULL;
          m_h11 = NULL;
          m_h22 = NULL;
          m_h12 = NULL;
          m_h12i = NULL;
          m_i11 = NULL;
          m_i22 = NULL;
          m_i12 = NULL;
        }
        if (BandNonCol_GemmWorkspaceTurnRelease()){
          openmx_gemmul8ReleaseWorkspaces();
        }
        BandNonCol_ConstructCache_Reset(state);
        BandNonCol_RootDenseWorkspace_Reset(state);
        BandNonCol_GpuSolver_Destroy(state);
        BandNonCol_RefineDropBasis(state);
      }

      MPI_Barrier(mpi_comm_level1);
    }
  }
  else if (use_root_dense_gpusolver){

    const int serialize_gpu_turns = BandNonCol_SerializeGpuSolverGpuTurns();
    const int owns_root_dense_rank =
      (0<=myworld2 && myworld2<Num_Comm_World2 &&
       myid0==Comm_World_StartID2[myworld2] &&
       Set_Hamiltonian_OpenACC_Rank_Is_Selected());
    int root_dense_turn_group =
      serialize_gpu_turns ?
      BandNonCol_GpuTurnGroup(state, n,n2,MaxN,size_H1,
                              BANDNONCOL_GPU_TURN_EIGVECS,
                              owns_root_dense_rank,myid0,Num_Comm_World2) :
      Num_Comm_World2;
    int use_setham_packed_cache =
      (Set_Hamiltonian_GpuSolver_Packed_CacheReady(state->hamiltonian_cache) &&
       Set_Hamiltonian_GpuSolver_Packed_OrderMode(state->hamiltonian_cache)==1);
    int root_s_valid = 0;
    int rebuild_overlap;
    BandNonColRootDenseWorkspace *rdw;
    double *pack_buffer = NULL;
    double *m_olp = NULL;
    double *m_h11 = NULL;
    double *m_h22 = NULL;
    double *m_h12 = NULL;
    double *m_h12i = NULL;
    double *m_i11 = NULL;
    double *m_i22 = NULL;
    double *m_i12 = NULL;
    int *packed_order_GA = order_GA;

    if (root_dense_turn_group<1) root_dense_turn_group = 1;
    root_dense_serial_gpusolver_worlds = 0;

    /* the refined eigensolver: every owner's buffers stay on the device
       between the two halves when all k-worlds form one turn group; with
       several groups the solve is staged (pass 1 and pass 2 as on the k-dense
       path, the bases on the host between them) */
    BandNonCol_RefineBeginStep(state, SCF_iter, strcasecmp(mode,"scf")==0 && Num_Comm_World2<=root_dense_turn_group, 0,
                               strcasecmp(mode,"scf")==0 && root_dense_turn_group<Num_Comm_World2);
    if (state->refine.staged) BandNonCol_KRefEnsure(state, T_knum, n2);

    if (use_setham_packed_cache){
      Set_Hamiltonian_GpuSolver_SetMP(MP);
    }

    double root_dense_mu = 0.0;

    for (int root_dense_pass=0; root_dense_pass<(state->refine.staged ? 2 : 1); root_dense_pass++){

    if (root_dense_pass==1){
      /* the staged refined eigensolver: the chemical potential of all
         k-points from the estimates of pass 1 */

      for (kloop=0; kloop<T_knum; kloop++){
        MPI_Bcast(&EIGEN[0][kloop][0], MaxN+1, MPI_DOUBLE, Comm_World_StartID2[kloop], mpi_comm_level1);
      }
      root_dense_mu = BandNonCol_RefineChemP(EIGEN,T_knum,MaxN,T_k_op,sum_weights,TZ);
      if (!state->Band_DFT_NonCol_owned_announced && myid0==0 && 2<=level_stdout){
        printf("<Band_DFT_NonCol> refined eigensolver in serialized GPU turns (%d k-worlds in groups of %d): the "
               "bases stay on the host between the passes\n", Num_Comm_World2, root_dense_turn_group);
        fflush(stdout);
        state->Band_DFT_NonCol_owned_announced = 1;
      }
    }

    for (int group_first=0; group_first<Num_Comm_World2; group_first+=root_dense_turn_group){
      int group_last =
        (group_first + root_dense_turn_group < Num_Comm_World2) ?
        (group_first + root_dense_turn_group) : Num_Comm_World2;
      int owns_root_dense_group = 0;

      for (int root_dense_world=group_first; root_dense_world<group_last; root_dense_world++){
        if (myid0==Comm_World_StartID2[root_dense_world]){
          owns_root_dense_group = 1;
          break;
        }
      }

      if (serialize_gpu_turns){
        MPI_Barrier(mpi_comm_level1);
      }

      if (use_setham_packed_cache){
        if (owns_root_dense_group){
          if (!Set_Hamiltonian_GpuSolver_Packed_OwnsCache(state->hamiltonian_cache)){
            BandNonCol_AbortWithMessage("Set_Hamiltonian packed cache is not owned by this rank in Band_DFT_NonCol.c.");
          }
          packed_order_GA = Set_Hamiltonian_GpuSolver_Packed_OrderGA(state->hamiltonian_cache);
          m_olp = Set_Hamiltonian_GpuSolver_Packed_Overlap(state->hamiltonian_cache);
          m_h11 = Set_Hamiltonian_GpuSolver_Packed_H(state->hamiltonian_cache, 0);
          m_h22 = Set_Hamiltonian_GpuSolver_Packed_H(state->hamiltonian_cache, 1);
          m_h12 = Set_Hamiltonian_GpuSolver_Packed_H(state->hamiltonian_cache, 2);
          m_h12i = Set_Hamiltonian_GpuSolver_Packed_H(state->hamiltonian_cache, 3);
          m_i11 = Set_Hamiltonian_GpuSolver_Packed_ImNL(state->hamiltonian_cache, 0);
          m_i22 = Set_Hamiltonian_GpuSolver_Packed_ImNL(state->hamiltonian_cache, 1);
          m_i12 = Set_Hamiltonian_GpuSolver_Packed_ImNL(state->hamiltonian_cache, 2);

          if (packed_order_GA==NULL || m_olp==NULL || m_h11==NULL || m_h22==NULL ||
              m_h12==NULL || m_h12i==NULL || m_i11==NULL || m_i22==NULL || m_i12==NULL){
            BandNonCol_AbortWithMessage("Set_Hamiltonian packed matrix cache is missing in Band_DFT_NonCol.c.");
          }
        }
      }
      else if (owns_root_dense_group){
        m_olp=OpenMX_Solver_Real(solver_owner,"m_olp",((size_t)1)*(size_t)size_H1);
        m_h11=OpenMX_Solver_Real(solver_owner,"m_h11",((size_t)1)*(size_t)size_H1);
        m_h22=OpenMX_Solver_Real(solver_owner,"m_h22",((size_t)1)*(size_t)size_H1);
        m_h12=OpenMX_Solver_Real(solver_owner,"m_h12",((size_t)1)*(size_t)size_H1);
        m_h12i=OpenMX_Solver_Real(solver_owner,"m_h12i",((size_t)1)*(size_t)size_H1);
        m_i11=OpenMX_Solver_Real(solver_owner,"m_i11",((size_t)1)*(size_t)size_H1);
        m_i22=OpenMX_Solver_Real(solver_owner,"m_i22",((size_t)1)*(size_t)size_H1);
        m_i12=OpenMX_Solver_Real(solver_owner,"m_i12",((size_t)1)*(size_t)size_H1);

        if (m_olp==NULL || m_h11==NULL || m_h22==NULL || m_h12==NULL ||
            m_h12i==NULL || m_i11==NULL || m_i22==NULL || m_i12==NULL){
          BandNonCol_AbortWithMessage("Failed to allocate root dense packed matrices in Band_DFT_NonCol.c.");
        }
      }
      else {
        pack_buffer=OpenMX_Solver_Real(solver_owner,"pack_buffer",((size_t)1)*(size_t)size_H1);
        if (pack_buffer==NULL){
          BandNonCol_AbortWithMessage("Failed to allocate root dense packing buffer in Band_DFT_NonCol.c.");
        }
      }

      if (!use_setham_packed_cache){
        BandNonCol_PackDenseM1(state, CntOLP, owns_root_dense_group ? m_olp : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, nh[0],  owns_root_dense_group ? m_h11 : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, nh[1],  owns_root_dense_group ? m_h22 : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, nh[2],  owns_root_dense_group ? m_h12 : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, nh[3],  owns_root_dense_group ? m_h12i : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, ImNL[0],owns_root_dense_group ? m_i11 : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, ImNL[1],owns_root_dense_group ? m_i22 : pack_buffer,MP,order_GA);
        BandNonCol_PackDenseM1(state, ImNL[2],owns_root_dense_group ? m_i12 : pack_buffer,MP,order_GA);
        pack_buffer = NULL;
      }

      /* the solves (the first halves of refined ones), then, for the refined
         eigensolver, the chemical potential of all k-points on the estimates
         and the second halves, then the distribution of the eigenvectors */
      for (int root_dense_world=group_first; root_dense_world<group_last; root_dense_world++){
        int root_dense_owner = Comm_World_StartID2[root_dense_world];
        int owns_root_dense = (myid0==root_dense_owner);

        if (owns_root_dense || myworld2==root_dense_world){

          rdw = BandNonCol_RootDenseWorkspace_Ensure(state, owns_root_dense,n,n2,MaxN,1,SCF_iter);
          root_s_valid = 0;
          if (owns_root_dense) root_s_valid = rdw->s_valid;
          rebuild_overlap = (SCF_iter==1 || !root_s_valid);

          kloop = root_dense_world;
          k1 = T_KGrids1[kloop];
          k2 = T_KGrids2[kloop];
          k3 = T_KGrids3[kloop];

          if (owns_root_dense){
            openmx_mgpu_scope_state mgpu_scope =
              openmx_mgpu_band_scope_enter(!gpusolver2_flag,T_knum,kloop);
            if (state->refine.staged){
              /* pass 1: the refinement gated with the previous step's
                 chemical potential, the basis to the host slot; pass 2: from
                 that basis with this step's, the wavefunctions into cs2 */
              BandNonCol_KDenseRefinedSolveOneK(state, rebuild_overlap,n,n2,MaxN,kloop,k1,k2,k3,
                                                m_olp,m_h11,m_h22,m_h12,m_h12i,m_i11,m_i22,m_i12,
                                                packed_order_GA,MP,ko,EIGEN,rdw,&state->kref[kloop],
                                                root_dense_pass, root_dense_pass ? root_dense_mu : state->chemical->chemical_potential);
              rdw->s_valid = 1;
            }
            else {
              (void)BandNonCol_RootDenseSolveOneK_FirstHalf(state, rebuild_overlap,n,n2,MaxN,kloop,k1,k2,k3,
                                                            m_olp,m_h11,m_h22,m_h12,m_h12i,
                                                            m_i11,m_i22,m_i12,
                                                            packed_order_GA,MP,ko,EIGEN,1,rdw,
                                                            state->refine.last_refined);
            }
            openmx_mgpu_band_scope_leave(mgpu_scope);
          }
        }
      }

      if (state->refine.last_refined){
        double mu;

        for (kloop=0; kloop<T_knum; kloop++){
          MPI_Bcast(&EIGEN[0][kloop][0], MaxN+1, MPI_DOUBLE, Comm_World_StartID2[kloop], mpi_comm_level1);
        }
        mu = BandNonCol_RefineChemP(EIGEN,T_knum,MaxN,T_k_op,sum_weights,TZ);
        if (owns_root_dense_rank && state->refine.pending){
          kloop = myworld2;
          openmx_mgpu_scope_state mgpu_scope =
            openmx_mgpu_band_scope_enter(!gpusolver2_flag,T_knum,kloop);
          BandNonCol_RootDenseSolveOneK_SecondHalf(state, n,n2,MaxN,kloop,ko,EIGEN,mu,&state->root_dense_workspace,1);
          /* the basis (n2^2 complex numbers) stays only for a warm start of
             the next step (OPENMX_EIGEN_REFINE_WARM_BAND) */
          if (!openmx_eigen_refine_warm_band_enabled()) BandNonCol_RefineDropBasis(state);
          openmx_mgpu_band_scope_leave(mgpu_scope);
        }
        BandNonCol_RefineEndStep(state);
      }

      /* no wavefunctions after pass 1 of a staged solve */
      if (!(state->refine.staged && root_dense_pass==0))
      for (int root_dense_world=group_first; root_dense_world<group_last; root_dense_world++){
        int root_dense_owner = Comm_World_StartID2[root_dense_world];
        int owns_root_dense = (myid0==root_dense_owner);

        if (owns_root_dense || myworld2==root_dense_world){
          rdw = &state->root_dense_workspace;
          kloop = root_dense_world;
          BandNonCol_DistributeDenseEvecGlobal(state, kloop,n2,MaxN,myid0,myworld2,NPROCS_WD2,
                                               Comm_World_StartID2,root_dense_owner,
                                               owns_root_dense ? rdw->cs2 : NULL,
                                               owns_root_dense ? rdw->cs2_on_device : 0,EVec1[0]);
          if (owns_root_dense) rdw->s_valid = 1;
        }
      }

      if (!use_setham_packed_cache && owns_root_dense_group){
        m_olp = NULL;
        m_h11 = NULL;
        m_h22 = NULL;
        m_h12 = NULL;
        m_h12i = NULL;
        m_i11 = NULL;
        m_i22 = NULL;
        m_i12 = NULL;
      }

      if (owns_root_dense_group){
        if (BandNonCol_GemmWorkspaceTurnRelease()){
          openmx_gemmul8ReleaseWorkspaces();
        }
        BandNonCol_ConstructCache_Reset(state);
        BandNonCol_GpuSolver_Destroy(state);
        /* a staged solve keeps its basis in the host slot */
        if (state->refine.staged) BandNonCol_RefineDropBasis(state);
      }

      if (serialize_gpu_turns){
        MPI_Barrier(mpi_comm_level1);
      }
    }

    } /* root_dense_pass */
    /* the staged solves of this step decide the stage for all ranks */
    if (state->refine.staged) BandNonCol_RefineEndStep(state);

    if (!serialize_gpu_turns){
      MPI_Barrier(mpi_comm_level1);
    }
  }
  else {

  for (kloop0=0; kloop0<max_num_kloop0; kloop0++){

    /* get k1, k2, and k3 */

    if (kloop0<num_kloop0){

      kloop = S_knum + kloop0;
      k1 = T_KGrids1[kloop];
      k2 = T_KGrids2[kloop];
      k3 = T_KGrids3[kloop];
    }

    if (measure_time) dtime(&Stime);

    if (SCF_iter==1 || all_knum!=1){

      /* make Cs */

      for(i=0; i<na_rows*na_cols; i++){ Cs[i] = Complex(0.0,0.0);}
      Construct_Band_Ms(state, 0,CntOLP,H1,Cs,MP,k1,k2,k3);

      /* diagonalize Cs */

      if (kloop0<num_kloop0){

	MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
	MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

	mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
	mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

	if (scf_eigen_lib_flag==GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2) && na_rows==n && na_cols==n &&
	    state->gpu_dense_verdict){
	  BandNonCol_GpuSolver_DenseZheevx(state, Cs,Ss,ko,n,n,"Band_DFT_NonCol overlap");
	}
	else if (scf_eigen_lib_flag==1 || (numprocs2<5 && scf_eigen_lib_flag!=GPUSOLVER)){
	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
	    ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
	}
	
	else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){

#ifndef kcomp
	  int mpiworld;
	  mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
	  F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
	    ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols, 
	      &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#else
	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
	    ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
#endif
	  
	}

	MPI_Comm_free(&mpi_comm_rows);
	MPI_Comm_free(&mpi_comm_cols);

	/* print to the standard output */

	if (3<=level_stdout){
	  printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
		 myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	  for (i=1; i<=n; i++){
	    printf("  Eigenvalues of OLP  %2d  %15.12f\n",i,ko[i]);
	  }
	}

	/*
	  printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	  myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	  for (i=1; i<=n; i++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i,ko[i]);
	  }
	*/

	/* minus eigenvalues to 1.0e-10 */

	for (l=1; l<=n; l++){
	  if (ko[l]<1.0e-10) ko[l] = 1.0e-10;
	  ko[l] = 1.0/sqrt(ko[l]);
	}

	/* calculate S*1/sqrt(ko) */

	for(i=0; i<na_rows; i++){
	  for(j=0; j<na_cols; j++){
	    jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	    Ss[j*na_rows+i].r = Ss[j*na_rows+i].r*ko[jg];
	    Ss[j*na_rows+i].i = Ss[j*na_rows+i].i*ko[jg];
	  }
	}

	/* make Ss2 */

	Overlap_Band_NC_Ss2( Ss, Ss2, MPI_CommWD2[myworld2] );
      }
    }

    if (measure_time){
      dtime(&Etime);
      time1 += Etime - Stime;
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

    if (measure_time) dtime(&Stime);
    
    /* set Hs11, Hs22, Hs12 */

    for(i=0; i<na_rows*na_cols; i++){
      Hs11[i] = Complex(0.0,0.0);
      Hs22[i] = Complex(0.0,0.0);
      Hs12[i] = Complex(0.0,0.0);
    }

    Construct_Band_Ms(state, 0,nh[0],  H1, Hs11,MP,k1,k2,k3);
    Construct_Band_Ms(state, 0,nh[1],  H1, Hs22,MP,k1,k2,k3);
    Construct_Band_Ms(state, 0,nh[2],  H1, Hs12,MP,k1,k2,k3);
    Construct_Band_Ms(state, 1,nh[3],  H1, Hs12,MP,k1,k2,k3);
    Construct_Band_Ms(state, 1,ImNL[0],H1, Hs11,MP,k1,k2,k3);
    Construct_Band_Ms(state, 1,ImNL[1],H1, Hs22,MP,k1,k2,k3);
    Construct_Band_Ms(state, 1,ImNL[2],H1, Hs12,MP,k1,k2,k3);

    if (measure_time){
      dtime(&Etime);
      time2 += Etime - Stime;
      dtime(&Stime);
    }
    
    if (measure_time){
      dtime(&Etime);
      time13 += Etime - Stime;
    }

    if (kloop0<num_kloop0){

      if (measure_time) dtime(&Stime);

      if (BandNonCol_UseDenseGpuMatrix(state, n,n2)){
        BandNonCol_DenseTripleTransform_OpenACC(state, n,Hs11,Ss,Cs);
        BandNonCol_DenseTripleTransform_OpenACC(state, n,Hs12,Ss,Cs);
        BandNonCol_DenseTripleTransform_OpenACC(state, n,Hs22,Ss,Cs);
      }
      else {
        /* S^t x Hs11 x S */

        for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

        Cblacs_barrier(ictxt1,"A");
        F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

        for (i=0; i<na_rows*na_cols; i++) Hs11[i] = Complex(0.0,0.0);

        Cblacs_barrier(ictxt1,"C");
        F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs11,&ONE,&ONE,descH);

        /* S^t x Hs12 x S */

        for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

        Cblacs_barrier(ictxt1,"A");
        F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

        for (i=0; i<na_rows*na_cols; i++) Hs12[i] = Complex(0.0,0.0);

        Cblacs_barrier(ictxt1,"C");
        F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs12,&ONE,&ONE,descH);

        /* S^t x Hs22 x S */

        for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

        Cblacs_barrier(ictxt1,"A");
        F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

        for (i=0; i<na_rows*na_cols; i++) Hs22[i] = Complex(0.0,0.0);

        Cblacs_barrier(ictxt1,"C");
        F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs22,&ONE,&ONE,descH);
      }

      if (measure_time){
        dtime(&Etime);
        time3 += Etime - Stime;
      }

      /****************************************************
	 diagonalize the transformed H
      ****************************************************/

      if (measure_time){
        dtime(&Stime);
      }

      Hamiltonian_Band_NC_Hs2( Hs11, Hs22, Hs12, Hs2, MPI_CommWD2[myworld2] );

      MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol2,my_prow2,&mpi_comm_rows);
      MPI_Comm_split(MPI_CommWD2[myworld2],my_prow2,my_pcol2,&mpi_comm_cols);

      mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
      mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

        if (scf_eigen_lib_flag==GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2) && na_rows2==n2 && na_cols2==n2 &&
            state->gpu_dense_verdict){
          BandNonCol_GpuSolver_DenseZheevx(state, Hs2,Cs2,ko,n2,MaxN,"Band_DFT_NonCol Hamiltonian");
        }
        else if (scf_eigen_lib_flag==1 || (numprocs2<5 && scf_eigen_lib_flag!=GPUSOLVER)){
	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
	}

        else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){

#ifndef kcomp
        int mpiworld;
        mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
        F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
	  ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &na_cols2, 
            &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#else
        F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
        ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
#endif

      }

      MPI_Comm_free(&mpi_comm_rows);
      MPI_Comm_free(&mpi_comm_cols);

      if (2<=level_stdout){
	for (i1=1; i1<=MaxN; i1++){
	  printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1,ko[i1]);
	}
      }

      for (l=1; l<=MaxN; l++){
	EIGEN[0][kloop][l] = ko[l];
      }

      if (3<=level_stdout && 0<=kloop){
	printf(" myid0=%2d  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n2; i1++){
	  printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1, ko[i1]);
	}
      }

      /*
      printf(" myid0=%2d  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
	     myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
      for (i1=1; i1<=n2; i1++){
	printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1, ko[i1]);
      }
      */

    } /* end of if (kloop0<num_kloop0) */

    if (measure_time){
      dtime(&Etime);
      time4 += Etime - Stime;
    }

    /**************************************************
      if (all_knum==1), wave functions are calculated. 
    **************************************************/

    if (measure_time) dtime(&Stime);

    if (all_knum==1){

      if (BandNonCol_UseDenseGpuMatrix(state, n,n2)){
        BandNonCol_DenseWavefunctions_OpenACC(state, n,n2,MaxN,Cs2,Ss,Hs2);
      }
      else {
        for(k=0; k<na_rows2*na_cols2; k++){
	  Hs2[k].r = 0.0;
	  Hs2[k].i = 0.0;
        }

        Cblacs_barrier(ictxt1_2,"A");
        F77_NAME(pzgemm,PZGEMM)( "T","T",&n2,&n2,&n2,&alpha,Cs2,&ONE,&ONE,descC2,Ss2,
                                 &ONE,&ONE,descS2,&beta,Hs2,&ONE,&ONE,descH2);
      }

      /* MPI communications of Hs2 */

      for (ID=0; ID<numprocs2; ID++){
    
	IDS = (myid2 + ID) % numprocs2;
	IDR = (myid2 - ID + numprocs2) % numprocs2;

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
	    MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
	  }
	  if (Num_Rcv_EV[IDR]!=0){
	    MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
	  }
	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Wait(&request,&stat);
	  }

	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
	  }
	  if (Num_Rcv_EV[IDR]!=0){
	    MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
	  }
	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Wait(&request,&stat);
	  }

	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Isend(EVec_Snd, Num_Snd_EV[IDS]*2, MPI_DOUBLE, IDS, 999, MPI_CommWD2[myworld2], &request);
	  }
	  if (Num_Rcv_EV[IDR]!=0){
	    MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR]*2, MPI_DOUBLE, IDR, 999, MPI_CommWD2[myworld2], &stat);
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
	  m = (jg-1)*(ie2[myid2]-is2[myid2]+1)+ig-is2[myid2];

	  EVec1[0][m].r = EVec_Rcv[2*k  ];
	  EVec1[0][m].i = EVec_Rcv[2*k+1];
	}
      }

    } /* if (all_knum==1) */

    if (measure_time){
      dtime(&Etime);
      time5 += Etime - Stime;
    }

  } /* kloop0 */

  /* release the GEMMul8 workspace left behind by the per-k dense GEMMs */
  if (BandNonCol_UseDenseGpuMatrix(state, n,n2) && BandNonCol_GemmWorkspaceTurnRelease()){
    openmx_gemmul8ReleaseWorkspaces();
  }

  } /* fallback distributed path */

  /****************************************************
     MPI:

     EIGEN
  ****************************************************/

  if (measure_time){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Stime);
  }

  for (kloop=0; kloop<T_knum; kloop++){
    /* get ID in the zeroth world */
    if (use_k_dense_gpusolver){
      ID = dense_k_owner[kloop];
    }
    else if (use_root_dense_gpusolver){
      ID = root_dense_serial_gpusolver_worlds ? Host_ID : Comm_World_StartID2[kloop];
    }
    else {
      ID = Comm_World_StartID1[0] + T_k_ID[myworld1][kloop];
    }
    MPI_Bcast(&EIGEN[0][kloop][0], MaxN+1, MPI_DOUBLE, ID, mpi_comm_level1);
  } 

  if (measure_time){
    dtime(&Etime);
    time6 += Etime - Stime;
  }

  /**************************************
         find chemical potential
  **************************************/

  if (measure_time) dtime(&Stime);

  double Beta_trial1;
  
  /* first, find ChemP at 3000 K */

  Beta_trial1 = 1.0/kB/(3000.0/eV2Hartree);

  po = 0;
  loop_num = 0;
  ChemP_MAX = 20.0;  
  ChemP_MIN =-20.0;

  do {

    loop_num++;

    state->chemical->chemical_potential = 0.50*(ChemP_MAX + ChemP_MIN);
    Num_State = 0.0;

    for (kloop=0; kloop<T_knum; kloop++){
      for (l=1; l<=MaxN; l++){

	x = (EIGEN[0][kloop][l] - state->chemical->chemical_potential)*Beta_trial1;

	if (x<=-x_cut) x = -x_cut;
	if (x_cut<=x)  x =  x_cut;
	FermiF = FermiFunc_NC(x,l);
	Num_State += FermiF*(double)T_k_op[kloop];
      } 
    } 

    Num_State = Num_State/sum_weights;
    Dnum = TZ - Num_State - system_charge;

    if (0.0<=Dnum) ChemP_MIN = state->chemical->chemical_potential;
    else           ChemP_MAX = state->chemical->chemical_potential;
    if (fabs(Dnum)<1.0e-12) po = 1;

  }
  while (po==0 && loop_num<1000);

  /* second, find ChemP at the temperatue, starting from the previously found ChemP. */

  po = 0;
  loop_num = 0;
  ChemP_MAX = 20.0;  
  ChemP_MIN =-20.0;

  do {

    loop_num++;

    if (loop_num!=1){
      state->chemical->chemical_potential = 0.50*(ChemP_MAX + ChemP_MIN);
    }

    Num_State = 0.0;

    for (kloop=0; kloop<T_knum; kloop++){
      for (l=1; l<=MaxN; l++){

	x = (EIGEN[0][kloop][l] - state->chemical->chemical_potential)*Beta;

	if (x<=-x_cut) x = -x_cut;
	if (x_cut<=x)  x =  x_cut;
	FermiF = FermiFunc_NC(x,l);

	Num_State += FermiF*(double)T_k_op[kloop];

      } 
    } 

    Num_State = Num_State/sum_weights;
    Dnum = TZ - Num_State - system_charge;

    if (0.0<=Dnum) ChemP_MIN = state->chemical->chemical_potential;
    else           ChemP_MAX = state->chemical->chemical_potential;
    if (fabs(Dnum)<1.0e-12) po = 1;
  }
  while (po==0 && loop_num<1000);

  /* for the NEGF calculation */
  if (Solver==4 && TRAN_ChemP_Band==0) state->chemical->chemical_potential = 0.5*(ChemP_e[0]+ChemP_e[1]);

  /****************************************************
           band energy in a finite temperature
  ****************************************************/

  Eele0[0] = 0.0;
  Eele0[1] = 0.0;

  for (kloop=0; kloop<T_knum; kloop++){
    for (l=1; l<=MaxN; l++){

      x = (EIGEN[0][kloop][l] - state->chemical->chemical_potential)*Beta;

      if (x<=-x_cut) x = -x_cut;
      if (x_cut<=x)  x = x_cut;
      FermiF = FermiFunc_NC(x,l);

      Eele0[0] += FermiF*EIGEN[0][kloop][l]*(double)T_k_op[kloop];
    }
  } 

  Eele0[0] = Eele0[0]/sum_weights;
  state->energies->band = Eele0[0];

  if (2<=level_stdout){
    printf("myid0=%2d ChemP=%lf, Eele0[0]=%lf, Eele0[1]=%lf\n",myid0,state->chemical->chemical_potential,Eele0[0],Eele0[1]);
  }

  if (measure_time){
    dtime(&Etime);
    time7 += Etime - Stime;
  }

  /****************************************************
       if all_knum==1, calculate CDM and EDM

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

  if (measure_time) dtime(&Stime);

  if (all_knum==1){

    /* initialize CDM, EDM, and iDM */

    if ( strcasecmp(mode,"scf")==0 ){ 

      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
	GA_AN = M2G[MA_AN];    
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
	      CDM[0][MA_AN][LB_AN][i][j] = 0.0;
	      CDM[1][MA_AN][LB_AN][i][j] = 0.0;
	      CDM[2][MA_AN][LB_AN][i][j] = 0.0;
	      CDM[3][MA_AN][LB_AN][i][j] = 0.0;
	      EDM[0][MA_AN][LB_AN][i][j] = 0.0;
	      EDM[1][MA_AN][LB_AN][i][j] = 0.0;
	      EDM[2][MA_AN][LB_AN][i][j] = 0.0;
	      EDM[3][MA_AN][LB_AN][i][j] = 0.0;
	      iDM[0][0][MA_AN][LB_AN][i][j] = 0.0;
	      iDM[0][1][MA_AN][LB_AN][i][j] = 0.0;
	    }
	  }
	}
      }
    }

    else if ( strcasecmp(mode,"ParDM")==0 ){ 

      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
	GA_AN = M2G[MA_AN];    
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
	      ParDM[0][MA_AN][LB_AN][i][j] = 0.0;
	      ParDM[1][MA_AN][LB_AN][i][j] = 0.0;
	      ParDM[2][MA_AN][LB_AN][i][j] = 0.0;
	      ParDM[3][MA_AN][LB_AN][i][j] = 0.0;
	      iParDM[0][MA_AN][LB_AN][i][j] = 0.0;
	      iParDM[1][MA_AN][LB_AN][i][j] = 0.0;
	    }
	  }
	}
      }
    }

    /* get k1, k2, and k3 */

    kloop = S_knum;

    k1 = T_KGrids1[kloop];
    k2 = T_KGrids2[kloop];
    k3 = T_KGrids3[kloop];

    /* calculate DM, iDM, and EDM */

    if ( strcasecmp(mode,"scf")==0 ){

      if (use_root_dense_gpusolver){
        const int serialize_gpu_turns = BandNonCol_SerializeGpuSolverGpuTurns();
        const int owns_root_dense_rank =
          (0<=myworld2 && myworld2<Num_Comm_World2 &&
           myid0==Comm_World_StartID2[myworld2] &&
           Set_Hamiltonian_OpenACC_Rank_Is_Selected());
        int root_dense_turn_group =
          serialize_gpu_turns ?
          BandNonCol_GpuTurnGroup(state, n,n2,MaxN,size_H1,
                                  BANDNONCOL_GPU_TURN_DM,
                                  owns_root_dense_rank,myid0,Num_Comm_World2) :
          Num_Comm_World2;

        if (root_dense_turn_group<1) root_dense_turn_group = 1;

        memset(rDM11,0,sizeof(double)*(size_t)size_H1);
        memset(rDM22,0,sizeof(double)*(size_t)size_H1);
        memset(rDM12,0,sizeof(double)*(size_t)size_H1);
        memset(iDM12,0,sizeof(double)*(size_t)size_H1);
        memset(iDM11,0,sizeof(double)*(size_t)size_H1);
        memset(iDM22,0,sizeof(double)*(size_t)size_H1);
        memset(rEDM11,0,sizeof(double)*(size_t)size_H1);
        memset(rEDM22,0,sizeof(double)*(size_t)size_H1);

        for (int group_first=0; group_first<Num_Comm_World2; group_first+=root_dense_turn_group){
          int group_last =
            (group_first + root_dense_turn_group < Num_Comm_World2) ?
            (group_first + root_dense_turn_group) : Num_Comm_World2;

          if (serialize_gpu_turns){
            MPI_Barrier(mpi_comm_level1);
          }

          for (int root_dense_world=group_first; root_dense_world<group_last; root_dense_world++){
            int root_dense_owner = Comm_World_StartID2[root_dense_world];
            int owns_root_dense = (myid0==root_dense_owner);

            if (owns_root_dense){
              BandNonColRootDenseWorkspace *rdw = &state->root_dense_workspace;

              if (!rdw->valid || rdw->cs2==NULL){
                BandNonCol_AbortWithMessage("Root dense eigenvectors are not available for Band_DFT_NonCol DM.");
              }

              kloop = root_dense_world;
              k1 = T_KGrids1[kloop];
              k2 = T_KGrids2[kloop];
              k3 = T_KGrids3[kloop];

            }
            const int dm_k=root_dense_world;
            openmx_mgpu_scope_state mgpu_scope=openmx_mgpu_band_scope_enter(!gpusolver2_flag,T_knum,dm_k);
            BandNonCol_AccumulateDMRootDenseK_OpenACC(state, size_H1,MP,n,n2,MaxN,
                T_KGrids1[dm_k],T_KGrids2[dm_k],T_KGrids3[dm_k],EIGEN[0][dm_k],
                owns_root_dense?state->root_dense_workspace.cs2:NULL,
                rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,rEDM11,rEDM22,mpi_comm_level1);
            openmx_mgpu_band_scope_leave(mgpu_scope);
          }

          if (serialize_gpu_turns){
            MPI_Barrier(mpi_comm_level1);
          }
        }

        if (!serialize_gpu_turns){
          MPI_Barrier(mpi_comm_level1);
        }

        kloop = 0;
        k1 = T_KGrids1[0];
        k2 = T_KGrids2[0];
        k3 = T_KGrids3[0];
        Calc_DM_Band_non_collinear(state->chemical,  0,1,
				    myid0,myid2,size_H1,
				    is2,ie2,MP,n,n2,MaxN,k1,k2,k3,
				    CDM,iDM[0],EDM,EIGEN[0][0],
				    EVec1[0],
				    rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
				    rEDM11,rEDM22 , &spinors);
      }
      else if (scf_eigen_lib_flag==GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2)){
        const openmx_mgpu_scope_state dm_scope =
            openmx_mgpu_band_scope_enter(!gpusolver2_flag,T_knum,kloop);
        BandNonCol_CalcDMAllK1_OpenACC(state,  myid0,myid2,size_H1,
					is2,ie2,MP,n,n2,k1,k2,k3,
					CDM,iDM[0],EDM,EIGEN[0][kloop],
					EVec1[0],
					rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
					rEDM11,rEDM22 );
        openmx_mgpu_band_scope_leave(dm_scope);
      }
      else {
        Calc_DM_Band_non_collinear(state->chemical,  1,1,
				    myid0,myid2,size_H1,
					    is2,ie2,MP,n,n2,MaxN,k1,k2,k3,
				    CDM,iDM[0],EDM,EIGEN[0][kloop],
				    EVec1[0],
				    rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
				    rEDM11,rEDM22 , &spinors);
      }
    }

    else if ( strcasecmp(mode,"ParDM")==0 ){ 

      Calc_ParDM_Band_non_collinear(state,  1,1,
				     myid0,myid2,size_H1,
				     is2,ie2,MP,n,n2,MaxN,k1,k2,k3, 
				     ParDM,iParDM,EIGEN[0][kloop],EVec1[0],
				     rDM11,rDM22,rDM12,iDM12,iDM11,iDM22, &spinors);
    } 

  } /* if (all_knum==1) */

  if (measure_time){
    dtime(&Etime);
    time8 += Etime - Stime;
  }

  dtime(&EiloopTime);

  if (myid0==Host_ID && 0<level_stdout){
    printf("<Band_DFT>  Eigen, time=%lf\n", EiloopTime-SiloopTime);fflush(stdout);
  }

  /****************************************************
   ****************************************************
     diagonalization for calculating density matrix
   ****************************************************
  ****************************************************/

  dtime(&SiloopTime);

  if (all_knum!=1){

    /* initialize CDM, EDM, and iDM */

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];    
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
	    CDM[0][MA_AN][LB_AN][i][j] = 0.0;
	    CDM[1][MA_AN][LB_AN][i][j] = 0.0;
	    CDM[2][MA_AN][LB_AN][i][j] = 0.0;
	    CDM[3][MA_AN][LB_AN][i][j] = 0.0;
            EDM[0][MA_AN][LB_AN][i][j] = 0.0;
            EDM[1][MA_AN][LB_AN][i][j] = 0.0;
            EDM[2][MA_AN][LB_AN][i][j] = 0.0;
            EDM[3][MA_AN][LB_AN][i][j] = 0.0;
	    iDM[0][0][MA_AN][LB_AN][i][j] = 0.0;
	    iDM[0][1][MA_AN][LB_AN][i][j] = 0.0;
	  }
	}
      }
    }

    /* for kloop */

    if (use_k_dense_gpusolver){
      int max_concurrent_gpu_turns =
        BandNonCol_MaxConcurrentKGpuTurns(state, n,n2,MaxN,size_H1,
                                          BANDNONCOL_GPU_TURN_EIGVECS,
                                          owns_dense_k_rank,myid0,T_knum);
      BandNonColRootDenseWorkspace *rdw;
      double *pack_buffer = NULL;
      double *m_olp = k_packed_m1[0];
      double *m_h11 = k_packed_m1[1];
      double *m_h22 = k_packed_m1[2];
      double *m_h12 = k_packed_m1[3];
      double *m_h12i = k_packed_m1[4];
      double *m_i11 = k_packed_m1[5];
      double *m_i22 = k_packed_m1[6];
      double *m_i12 = k_packed_m1[7];

      memset(rDM11,0,sizeof(double)*(size_t)size_H1);
      memset(rDM22,0,sizeof(double)*(size_t)size_H1);
      memset(rDM12,0,sizeof(double)*(size_t)size_H1);
      memset(iDM12,0,sizeof(double)*(size_t)size_H1);
      memset(iDM11,0,sizeof(double)*(size_t)size_H1);
      memset(iDM22,0,sizeof(double)*(size_t)size_H1);
      memset(rEDM11,0,sizeof(double)*(size_t)size_H1);
      memset(rEDM22,0,sizeof(double)*(size_t)size_H1);

      for (int group_first=0; group_first<T_knum; group_first+=max_concurrent_gpu_turns){
        int group_last =
          (group_first + max_concurrent_gpu_turns < T_knum) ?
          (group_first + max_concurrent_gpu_turns) : T_knum;
        int owns_dense_k_group = 0;
        int local_needs_solve = 0, group_needs_solve = 0;

        for (int gpu_turn=group_first; gpu_turn<group_last; gpu_turn++){
          if (dense_k_owner[dense_k_order[gpu_turn]]==myid0){
            owns_dense_k_group = 1;
            if (k_evec_cache == NULL || k_evec_cache[dense_k_order[gpu_turn]] == NULL)
              local_needs_solve = 1;
          }
        }
        /* Without a shared packed copy, all ranks skip the gathers only
           when every owner in this group has retained its eigenvectors. */
        if (!k_packed_ready)
          MPI_Allreduce(&local_needs_solve,&group_needs_solve,1,MPI_INT,MPI_MAX,mpi_comm_level1);
        MPI_Barrier(mpi_comm_level1);

        if (!k_packed_ready){
          if (group_needs_solve && owns_dense_k_group){
            m_olp=OpenMX_Solver_Real(solver_owner,"m_olp",((size_t)1)*(size_t)size_H1);
            m_h11=OpenMX_Solver_Real(solver_owner,"m_h11",((size_t)1)*(size_t)size_H1);
            m_h22=OpenMX_Solver_Real(solver_owner,"m_h22",((size_t)1)*(size_t)size_H1);
            m_h12=OpenMX_Solver_Real(solver_owner,"m_h12",((size_t)1)*(size_t)size_H1);
            m_h12i=OpenMX_Solver_Real(solver_owner,"m_h12i",((size_t)1)*(size_t)size_H1);
            m_i11=OpenMX_Solver_Real(solver_owner,"m_i11",((size_t)1)*(size_t)size_H1);
            m_i22=OpenMX_Solver_Real(solver_owner,"m_i22",((size_t)1)*(size_t)size_H1);
            m_i12=OpenMX_Solver_Real(solver_owner,"m_i12",((size_t)1)*(size_t)size_H1);

            if (m_olp==NULL || m_h11==NULL || m_h22==NULL || m_h12==NULL ||
                m_h12i==NULL || m_i11==NULL || m_i22==NULL || m_i12==NULL){
              BandNonCol_AbortWithMessage("Failed to allocate dense k-point DM packed matrices in Band_DFT_NonCol.c.");
            }
          }
          else if (group_needs_solve){
            pack_buffer=OpenMX_Solver_Real(solver_owner,"pack_buffer",((size_t)1)*(size_t)size_H1);
            if (pack_buffer==NULL){
              BandNonCol_AbortWithMessage("Failed to allocate dense k-point DM packing buffer in Band_DFT_NonCol.c.");
            }
          }

          if (group_needs_solve){
            BandNonCol_PackDenseM1(state, CntOLP, owns_dense_k_group ? m_olp : pack_buffer,MP,order_GA);
            BandNonCol_PackDenseM1(state, nh[0],  owns_dense_k_group ? m_h11 : pack_buffer,MP,order_GA);
            BandNonCol_PackDenseM1(state, nh[1],  owns_dense_k_group ? m_h22 : pack_buffer,MP,order_GA);
            BandNonCol_PackDenseM1(state, nh[2],  owns_dense_k_group ? m_h12 : pack_buffer,MP,order_GA);
            BandNonCol_PackDenseM1(state, nh[3],  owns_dense_k_group ? m_h12i : pack_buffer,MP,order_GA);
            BandNonCol_PackDenseM1(state, ImNL[0],owns_dense_k_group ? m_i11 : pack_buffer,MP,order_GA);
            BandNonCol_PackDenseM1(state, ImNL[1],owns_dense_k_group ? m_i22 : pack_buffer,MP,order_GA);
            BandNonCol_PackDenseM1(state, ImNL[2],owns_dense_k_group ? m_i12 : pack_buffer,MP,order_GA);
          }
          pack_buffer = NULL;
        }

        for (int gpu_turn=group_first; gpu_turn<group_last; gpu_turn++){

          kloop = dense_k_order[gpu_turn];
          int owns_dm=dense_k_owner[kloop]==myid0;

          k1 = T_KGrids1[kloop];
          k2 = T_KGrids2[kloop];
          k3 = T_KGrids3[kloop];

          openmx_mgpu_scope_state mgpu_scope =
            openmx_mgpu_band_scope_enter(!gpusolver2_flag,T_knum,kloop);
          if(owns_dm){
          rdw = BandNonCol_RootDenseWorkspace_Ensure(state, 1,n,n2,MaxN,1,SCF_iter);
          if (k_evec_cache != NULL && k_evec_cache[kloop] != NULL){
            BandNonCol_KCacheRestore(state, k_evec_cache[kloop],rdw);
            openmx_core_native_free(state->resources,k_evec_cache[kloop]);
            k_evec_cache[kloop] = NULL;
            ++k_cache_hits;
          }
          else {
            int reuse_overlap = BandNonCol_KOverlapRestore(state, kloop,k1,k2,k3,rdw->s_all);
            k_overlap_hits += reuse_overlap;
            if (state->kref != NULL && state->kref[kloop].refined){
              BandNonCol_KDenseRefinedSolveOneK(state, !reuse_overlap,n,n2,MaxN,kloop,k1,k2,k3,
                                                m_olp,m_h11,m_h22,m_h12,m_h12i,
                                                m_i11,m_i22,m_i12,
                                                order_GA,MP,ko,EIGEN,rdw,&state->kref[kloop],1,state->chemical->chemical_potential);
            }
            else {
            BandNonCol_RootDenseSolveOneK_OpenACC(state, !reuse_overlap,n,n2,MaxN,kloop,k1,k2,k3,
                                                m_olp,m_h11,m_h22,m_h12,m_h12i,
                                                m_i11,m_i22,m_i12,
                                                order_GA,MP,ko,EIGEN,1,rdw);
            }
            if (!reuse_overlap) BandNonCol_KOverlapSave(state, kloop,k1,k2,k3,rdw->s_all);
          }
          }
          BandNonCol_AccumulateDMRootDenseK_OpenACC(state, size_H1,MP,n,n2,MaxN,k1,k2,k3,
                                                    EIGEN[0][kloop],owns_dm?state->root_dense_workspace.cs2:NULL,
                                                    rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
                                                      rEDM11,rEDM22,mpi_comm_level1);
          if(owns_dm)BandNonCol_RootDenseEigenvectorsRelease(rdw);
          openmx_mgpu_band_scope_leave(mgpu_scope);
        }

        if (owns_dense_k_group){
          if (!k_packed_ready){
            m_olp = NULL;
            m_h11 = NULL;
            m_h22 = NULL;
            m_h12 = NULL;
            m_h12i = NULL;
            m_i11 = NULL;
            m_i22 = NULL;
            m_i12 = NULL;
          }
          if (BandNonCol_GemmWorkspaceTurnRelease()){
            openmx_gemmul8ReleaseWorkspaces();
          }
          BandNonCol_ConstructCache_Reset(state);
          BandNonCol_RootDenseWorkspace_Reset(state);
          BandNonCol_GpuSolver_Destroy(state);
          BandNonCol_RefineDropBasis(state);
        }

        MPI_Barrier(mpi_comm_level1);
      }

      /* the k-dense refined solves of this step decide the stage for all ranks */
      if (state->refine.kdense) BandNonCol_RefineEndStep(state);
      kloop = 0;
      k1 = T_KGrids1[0];
      k2 = T_KGrids2[0];
      k3 = T_KGrids3[0];
      Calc_DM_Band_non_collinear(state->chemical,  0,1,
				  myid0,myid2,size_H1,
				  is2,ie2,MP,n,n2,MaxN,k1,k2,k3,
				  CDM,iDM[0],EDM,EIGEN[0][0],
				  EVec1[0],
				  rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
				  rEDM11,rEDM22 , &spinors);
    }
    else {

    for (kloop0=0; kloop0<max_num_kloop0; kloop0++){

      /* get k1, k2, and k3 */

      if (kloop0<num_kloop0){

	kloop = S_knum + kloop0;
	k1 = T_KGrids1[kloop];
	k2 = T_KGrids2[kloop];
	k3 = T_KGrids3[kloop];
      }

      if (measure_time) dtime(&Stime);

      /* make Cs */

      for(i=0; i<na_rows*na_cols; i++){ Cs[i] = Complex(0.0,0.0);}
      Construct_Band_Ms(state, 0,CntOLP,H1,Cs,MP,k1,k2,k3);

      /* diagonalize Cs */

      if (kloop0<num_kloop0){

	MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
	MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

	mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
	mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

        if (scf_eigen_lib_flag==GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2) && na_rows==n && na_cols==n &&
            state->gpu_dense_verdict){
          BandNonCol_GpuSolver_DenseZheevx(state, Cs,Ss,ko,n,n,"Band_DFT_NonCol overlap");
        }
        else if (scf_eigen_lib_flag==1 || (numprocs2<5 && scf_eigen_lib_flag!=GPUSOLVER)){
  	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
	}

        else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){

#ifndef kcomp
          int mpiworld;
          mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
          F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
          ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols, 
            &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );

#else
  	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
#endif
	}

	MPI_Comm_free(&mpi_comm_rows);
	MPI_Comm_free(&mpi_comm_cols);

	/* print to the standard output */

	if (3<=level_stdout){
	  printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
		 myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	  for (i=1; i<=n; i++){
	    printf("  Eigenvalues of OLP  %2d  %15.12f\n",i,ko[i]);
	  }
	}

	/*
	printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i=1; i<=n; i++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i,ko[i]);
	}
	*/

	/* minus eigenvalues to 1.0e-10 */

	for (l=1; l<=n; l++){
	  if (ko[l]<1.0e-10) ko[l] = 1.0e-10;
	  ko[l] = 1.0/sqrt(ko[l]);
	}

	/* calculate S*1/sqrt(ko) */

	for(i=0; i<na_rows; i++){
	  for(j=0; j<na_cols; j++){
	    jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	    Ss[j*na_rows+i].r = Ss[j*na_rows+i].r*ko[jg];
	    Ss[j*na_rows+i].i = Ss[j*na_rows+i].i*ko[jg];
	  }
	}

	/* make Ss2 */

	Overlap_Band_NC_Ss2( Ss, Ss2, MPI_CommWD2[myworld2] );

      } /* end of if (kloop0<num_kloop0) */

      if (measure_time){
        dtime(&Etime);
        time9 += Etime - Stime;
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
      
      if (measure_time) dtime(&Stime);
      
      /* set Hs */

      for(i=0; i<na_rows*na_cols; i++){
	Hs11[i] = Complex(0.0,0.0);
	Hs22[i] = Complex(0.0,0.0);
	Hs12[i] = Complex(0.0,0.0);
      }

      Construct_Band_Ms(state, 0,nh[0],  H1, Hs11,MP,k1,k2,k3);
      Construct_Band_Ms(state, 0,nh[1],  H1, Hs22,MP,k1,k2,k3);
      Construct_Band_Ms(state, 0,nh[2],  H1, Hs12,MP,k1,k2,k3);
      Construct_Band_Ms(state, 1,nh[3],  H1, Hs12,MP,k1,k2,k3);
      Construct_Band_Ms(state, 1,ImNL[0],H1, Hs11,MP,k1,k2,k3);
      Construct_Band_Ms(state, 1,ImNL[1],H1, Hs22,MP,k1,k2,k3);
      Construct_Band_Ms(state, 1,ImNL[2],H1, Hs12,MP,k1,k2,k3);

      if (kloop0<num_kloop0){

        if (BandNonCol_UseDenseGpuMatrix(state, n,n2)){
          BandNonCol_DenseTripleTransform_OpenACC(state, n,Hs11,Ss,Cs);
          BandNonCol_DenseTripleTransform_OpenACC(state, n,Hs12,Ss,Cs);
          BandNonCol_DenseTripleTransform_OpenACC(state, n,Hs22,Ss,Cs);
        }
        else {
	  /* S^t x Hs11 x S */

	  for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

	  Cblacs_barrier(ictxt1,"A");
	  F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

	  for (i=0; i<na_rows*na_cols; i++) Hs11[i] = Complex(0.0,0.0);

	  Cblacs_barrier(ictxt1,"C");
	  F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs11,&ONE,&ONE,descH);

	  /* S^t x Hs12 x S */

	  for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

	  Cblacs_barrier(ictxt1,"A");
	  F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

	  for (i=0; i<na_rows*na_cols; i++) Hs12[i] = Complex(0.0,0.0);

	  Cblacs_barrier(ictxt1,"C");
	  F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs12,&ONE,&ONE,descH);

	  /* S^t x Hs22 x S */

	  for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

	  Cblacs_barrier(ictxt1,"A");
	  F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

	  for (i=0; i<na_rows*na_cols; i++) Hs22[i] = Complex(0.0,0.0);

	  Cblacs_barrier(ictxt1,"C");
	  F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs22,&ONE,&ONE,descH);
        }

        if (measure_time){
  	  dtime(&Etime);
	  time10 += Etime - Stime;
	}

	/* ***************************************************
	   diagonalize the transformed H
	   *************************************************** */

	if (measure_time) dtime(&Stime);

	Hamiltonian_Band_NC_Hs2( Hs11, Hs22, Hs12, Hs2, MPI_CommWD2[myworld2] );

	MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol2,my_prow2,&mpi_comm_rows);
	MPI_Comm_split(MPI_CommWD2[myworld2],my_prow2,my_pcol2,&mpi_comm_cols);

	mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
	mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);
  
        if (scf_eigen_lib_flag==GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2) && na_rows2==n2 && na_cols2==n2 &&
            state->gpu_dense_verdict){
          BandNonCol_GpuSolver_DenseZheevx(state, Hs2,Cs2,ko,n2,MaxN,"Band_DFT_NonCol Hamiltonian");
        }
        else if (scf_eigen_lib_flag==1 || (numprocs2<5 && scf_eigen_lib_flag!=GPUSOLVER)){
	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
	}

        else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){

#ifndef kcomp
          int mpiworld;
          mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
          F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
	  ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &na_cols2, 
            &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );

#else
  	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
#endif
	}

	MPI_Comm_free(&mpi_comm_rows);
	MPI_Comm_free(&mpi_comm_cols);

	if (2<=level_stdout){
	  for (i1=1; i1<=MaxN; i1++){
	    printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1,ko[i1]);
	  }
	}

	for (l=1; l<=MaxN; l++){
	  EIGEN[0][kloop][l] = ko[l];
	}

	if (3<=level_stdout && 0<=kloop && kloop0<num_kloop0){
	  printf(" myid0=%2d  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
		 myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	  for (i1=1; i1<=n2; i1++){
	    printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1, ko[i1]);
	  }
	}

        if (measure_time){
          dtime(&Etime);
          time11 += Etime - Stime;
	}

        /**************************************************
                  calculation of wave functions
        **************************************************/

        if (BandNonCol_UseDenseGpuMatrix(state, n,n2)){
          BandNonCol_DenseWavefunctions_OpenACC(state, n,n2,MaxN,Cs2,Ss,Hs2);
        }
        else {
	  for(k=0; k<na_rows2*na_cols2; k++){
	    Hs2[k].r = 0.0;
	    Hs2[k].i = 0.0;
	  }

	  Cblacs_barrier(ictxt1_2,"A");
	  F77_NAME(pzgemm,PZGEMM)( "T","T",&n2,&n2,&n2,&alpha,Cs2,&ONE,&ONE,descC2,Ss2,
				   &ONE,&ONE,descS2,&beta,Hs2,&ONE,&ONE,descH2);
        }

	/* MPI communications of Hs2 */

        ID = 0;
	IDS = (myid2 + ID) % numprocs2;
	IDR = (myid2 - ID + numprocs2) % numprocs2;

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

	for(k=0; k<Num_Rcv_EV[IDR]; k++){
	  ig = index_Snd_i[k];
	  jg = index_Snd_j[k];
	  m = (jg-1)*(ie2[myid2]-is2[myid2]+1)+ig-is2[myid2];

	  EVec1[0][m].r = EVec_Snd[2*k  ];
	  EVec1[0][m].i = EVec_Snd[2*k+1];
	}

      } /* end of if (kloop0<num_kloop0) */

      /****************************************************
                     calculate DM and EDM
      ****************************************************/

      if (measure_time) dtime(&Stime);

      /* calculate DM and iDM */

      if ( strcasecmp(mode,"scf")==0 ){

        int nc_dm_done=0;
        /* the vote of every rank and k turn, then the direct device attempt;
           a declined attempt accumulates on the host (no collective) below.
           OPENMX_HOST_SIMD=legacy keeps the collective capacity recovery. */
        if (BandNonCol_UseGpuFallbackDM(n2,(size_t)n2*(size_t)(ie2[myid2]-is2[myid2]+1)*sizeof(dcomplex)) &&
            !BandNonCol_DMOnHost(n2,kloop0<num_kloop0?ie2[myid2]-is2[myid2]+1:0,size_H1)){
          /* accumulates this k on the GPU; EVec1 stays unscaled because the
             kernel applies the Fermi weights itself */
          const openmx_mgpu_scope_state dm_scope =
              openmx_mgpu_band_scope_enter(!gpusolver2_flag,T_knum,kloop);
          nc_dm_done=BandNonCol_AccumulateDMKPoint_OpenACC(state, myid2,is2,ie2,MP,n,n2,size_H1,k1,k2,k3,
                                                kloop0<num_kloop0?EIGEN[0][kloop]:NULL,EVec1[0],
                                                rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
                                                rEDM11,rEDM22,kloop0<num_kloop0,mpi_comm_level1,
                                                OpenMX_HostSimdMode()!=OPENMX_HOST_SIMD_LEGACY);
          openmx_mgpu_band_scope_leave(dm_scope);
        }
        if (!nc_dm_done){
	Calc_DM_Band_non_collinear(state->chemical,  (kloop0<num_kloop0),0,
				    myid0,myid2,size_H1,
				    is2,ie2,MP,n,n2,MaxN,k1,k2,k3,
				    CDM,iDM[0],EDM,EIGEN[0][kloop],
				    EVec1[0],
				    rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
				    rEDM11,rEDM22 , &spinors);
        }
      }

      else if ( strcasecmp(mode,"ParDM")==0 ){ 

	Calc_ParDM_Band_non_collinear(state,  (kloop0<num_kloop0),0,
				       myid0,myid2,size_H1,
				       is2,ie2,MP,n,n2,MaxN,k1,k2,k3, 
				       ParDM,iParDM,EIGEN[0][kloop],EVec1[0],
				       rDM11,rDM22,rDM12,iDM12,iDM11,iDM22, &spinors);

      }

      if (measure_time){
        dtime(&Etime);
        time12 += Etime - Stime;
      }

    } /* kloop0 */

    /* release the GEMMul8 workspace left behind by the per-k dense GEMMs */
    if (BandNonCol_UseDenseGpuMatrix(state, n,n2) && BandNonCol_GemmWorkspaceTurnRelease()){
      openmx_gemmul8ReleaseWorkspaces();
    }

    /* store DM and iDM */

    if ( strcasecmp(mode,"scf")==0 ){ 

      Calc_DM_Band_non_collinear(state->chemical,  0,1,
				  myid0,myid2,size_H1,
				  is2,ie2,MP,n,n2,MaxN,k1,k2,k3, 
				  CDM,iDM[0],EDM,EIGEN[0][kloop],
				  EVec1[0],
				  rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
				  rEDM11,rEDM22 , &spinors);

    }

    else if ( strcasecmp(mode,"ParDM")==0 ){ 

      Calc_ParDM_Band_non_collinear(state,  0,1,
				     myid0,myid2,size_H1,
				     is2,ie2,MP,n,n2,MaxN,k1,k2,k3, 
				     ParDM,iParDM,EIGEN[0][kloop],EVec1[0],
				     rDM11,rDM22,rDM12,iDM12,iDM11,iDM22, &spinors);

    }

    }

  } /* if (all_knum!=1) */

  /****************************************************
           normalization of CDM, EDM, and iDM 
  ****************************************************/

  dtime(&EiloopTime);

  if (myid0==Host_ID && 0<level_stdout){
    printf("<Band_DFT>  DM, time=%lf\n", EiloopTime-SiloopTime);fflush(stdout);
  }

  dum = 1.0/sum_weights;

  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];    
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      Bnum = MP[GB_AN];

      if ( strcasecmp(mode,"scf")==0 ){ 

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    CDM[0][MA_AN][LB_AN][i][j] = CDM[0][MA_AN][LB_AN][i][j]*dum;
	    CDM[1][MA_AN][LB_AN][i][j] = CDM[1][MA_AN][LB_AN][i][j]*dum;
	    CDM[2][MA_AN][LB_AN][i][j] = CDM[2][MA_AN][LB_AN][i][j]*dum;
	    CDM[3][MA_AN][LB_AN][i][j] = CDM[3][MA_AN][LB_AN][i][j]*dum;
	    EDM[0][MA_AN][LB_AN][i][j] = EDM[0][MA_AN][LB_AN][i][j]*dum;
	    EDM[1][MA_AN][LB_AN][i][j] = EDM[1][MA_AN][LB_AN][i][j]*dum;
	    EDM[2][MA_AN][LB_AN][i][j] = EDM[2][MA_AN][LB_AN][i][j]*dum;
	    EDM[3][MA_AN][LB_AN][i][j] = EDM[3][MA_AN][LB_AN][i][j]*dum;
	    iDM[0][0][MA_AN][LB_AN][i][j] = iDM[0][0][MA_AN][LB_AN][i][j]*dum;
	    iDM[0][1][MA_AN][LB_AN][i][j] = iDM[0][1][MA_AN][LB_AN][i][j]*dum;
	  }
	}
      } // end of if 

      else if ( strcasecmp(mode,"ParDM")==0 ){ 

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    ParDM[0][MA_AN][LB_AN][i][j] = ParDM[0][MA_AN][LB_AN][i][j]*dum;
	    ParDM[1][MA_AN][LB_AN][i][j] = ParDM[1][MA_AN][LB_AN][i][j]*dum;
	    ParDM[2][MA_AN][LB_AN][i][j] = ParDM[2][MA_AN][LB_AN][i][j]*dum;
	    ParDM[3][MA_AN][LB_AN][i][j] = ParDM[3][MA_AN][LB_AN][i][j]*dum;
	    iParDM[0][MA_AN][LB_AN][i][j] = iParDM[0][MA_AN][LB_AN][i][j]*dum;
	    iParDM[1][MA_AN][LB_AN][i][j] = iParDM[1][MA_AN][LB_AN][i][j]*dum;
	  }
	}
      } // end of else if 

    }
  }

  /****************************************************
                       bond-energies
  ****************************************************/

  if ( strcasecmp(mode,"scf")==0 ){ 

    My_Eele1[0] = 0.0;
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];    
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      for (j=0; j<=FNAN[GA_AN]; j++){
	GB_AN = natn[GA_AN][j];  
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];

	/* non-spin-orbit coupling and non-LDA+U */  
	if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
	    && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){
	  for (k=0; k<tnoA; k++){
	    for (l=0; l<tnoB; l++){
	      My_Eele1[0] = My_Eele1[0]
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
	      My_Eele1[0] = My_Eele1[0]
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

    MPI_Barrier(mpi_comm_level1);
    MPI_Allreduce(&My_Eele1[0], &Eele1[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    Eele1[1] = 0.0;

    if (3<=level_stdout && myid0==Host_ID){
      printf("Eele00=%15.12f Eele01=%15.12f\n",Eele0[0],Eele0[1]);
      printf("Eele10=%15.12f Eele11=%15.12f\n",Eele1[0],Eele1[1]);
    }

    /****************************************************
                        output
    ****************************************************/

    if (myid0==Host_ID){
  
      strcpy(file_EV,".EV");
      fnjoint(filepath,filename,file_EV);

      if ((fp_EV = fopen(file_EV,"w")) != NULL){

	setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"           Eigenvalues (Hartree) of SCF KS-eq.           \n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n\n");
	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",state->chemical->chemical_potential);
	fprintf(fp_EV,"   Number of States             = %18.14f\n",Num_State);
	fprintf(fp_EV,"   Eigenvalues\n\n");

	for (kloop=0; kloop<T_knum; kloop++){

	  if (0<T_k_op[kloop]){

	    fprintf(fp_EV,"\n");
	    fprintf(fp_EV,"   kloop=%i\n",kloop);
	    fprintf(fp_EV,"   k1=%10.5f k2=%10.5f k3=%10.5f\n\n",
		    T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);

	    for (l=1; l<=MaxN; l++){
	      fprintf(fp_EV,"%5d  %18.14f\n",l,EIGEN[0][kloop][l]);
	    }
	  }
	}
	fclose(fp_EV);
      }
      else{
	printf("Failure of saving the EV file.\n");
	fclose(fp_EV);
      }  
    }

  } // end of if ( strcasecmp(mode,"scf")==0 ){ 

  /* Ordinary workspaces are released by the C++ invocation. */
  openmx_core_native_free(state->resources,k_packed_m1[0]);
  if (k_evec_cache != NULL){
    for (int point=0; point<T_knum; ++point) openmx_core_native_free(state->resources,k_evec_cache[point]);
    openmx_core_native_free(state->resources,k_evec_cache);
  }
  if (use_k_dense_gpusolver){
    if (openmx_env_flag_at(OPENMX_ENV_BAND_NONCOL_CACHE_TRACE, 0))
      printf("BAND_NONCOL_CACHE rank=%d SCF=%d evec_hits=%d overlap_hits=%d evec=%.1f MiB overlap=%.1f MiB packed=%.1f MiB\n",
             myid0,SCF_iter,k_cache_hits,k_overlap_hits,k_cache_used/1048576.0,
             state->k_overlap_used/1048576.0,k_packed_bytes/1048576.0);
  }

  /* for PrintMemory and allocation */
  state->Band_DFT_NonCol_owned_firsttime=0;

  /* for elapsed time */

  if (measure_time){
    printf("myid0=%2d time1 =%9.4f\n",myid0,time1);fflush(stdout);
    printf("myid0=%2d time2 =%9.4f\n",myid0,time2);fflush(stdout);
    printf("myid0=%2d time3 =%9.4f\n",myid0,time3);fflush(stdout);
    printf("myid0=%2d time4 =%9.4f\n",myid0,time4);fflush(stdout);
    printf("myid0=%2d time5 =%9.4f\n",myid0,time5);fflush(stdout);
    printf("myid0=%2d time6 =%9.4f\n",myid0,time6);fflush(stdout);
    printf("myid0=%2d time7 =%9.4f\n",myid0,time7);fflush(stdout);
    printf("myid0=%2d time8 =%9.4f\n",myid0,time8);fflush(stdout);
    printf("myid0=%2d time9 =%9.4f\n",myid0,time9);fflush(stdout);
    printf("myid0=%2d time10=%9.4f\n",myid0,time10);fflush(stdout);
    printf("myid0=%2d time11=%9.4f\n",myid0,time11);fflush(stdout);
    printf("myid0=%2d time12=%9.4f\n",myid0,time12);fflush(stdout);
    printf("myid0=%2d time13=%9.4f\n",myid0,time12);fflush(stdout);
  }

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}

static void BandNonCol_PackDenseM1(BandNonColState *state,  double ****Mat, double *M1, int *MP, int *order_GA )
{
  int i,j,k;
  int MA_AN,GA_AN,LB_AN,GB_AN;
  int wanA,wanB,tnoA,tnoB,Anum;
  int num,tnum;
  int ID,myid,numprocs;
  int *My_NZeros;
  int *is1,*is2;
  int *My_Matomnum;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  My_NZeros = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  My_Matomnum = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  is1 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  is2 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);

  if (My_NZeros==NULL || My_Matomnum==NULL || is1==NULL || is2==NULL){
    openmx_core_native_free(state->resources,My_NZeros);
    openmx_core_native_free(state->resources,My_Matomnum);
    openmx_core_native_free(state->resources,is1);
    openmx_core_native_free(state->resources,is2);
    BandNonCol_AbortWithMessage("Failed to allocate dense M1 packing workspace in Band_DFT_NonCol.c.");
  }

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
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }

  is1[0] = 0;
  for (ID=1; ID<numprocs; ID++){
    is1[ID] = is1[ID-1] + My_NZeros[ID-1];
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

  for (i=0; i<tnum; i++) M1[i] = 0.0;

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
          M1[k] = Mat[MA_AN][LB_AN][i][j];
          k++;
        }
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE,&M1[0],tnum,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  openmx_core_native_free(state->resources,My_NZeros);
  openmx_core_native_free(state->resources,My_Matomnum);
  openmx_core_native_free(state->resources,is1);
  openmx_core_native_free(state->resources,is2);
}

static void BandNonCol_ConstructDenseMsFromPacked(BandNonColState *state,  int cpx_flag, const double *M1, dcomplex *Ms,
                                                   int *order_GA, int *MP, double k1, double k2, double k3,
                                                   int n, int owns_dense )
{
  int i,j,k;
  int AN,Rn,l1,l2,l3;
  int GA_AN,LB_AN,GB_AN;
  int wanA,wanB,tnoA,tnoB,Anum,Bnum;
  double kRn,si,co;

  if (!owns_dense){
    return;
  }

  if (!state->mgpu_host_dense && scf_eigen_lib_flag==GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum()<=2*n){
    BandNonCol_ConstructCache_Ensure(state, order_GA,MP,n);
    BandNonCol_ConstructDenseMs_OpenACC(state, cpx_flag,n,k1,k2,k3,M1,Ms);
    return;
  }

  k = 0;
  for (AN=1; AN<=atomnum; AN++){
    GA_AN = order_GA[AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];

    for (i=0; i<tnoA; i++){
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        Rn = ncn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
        Bnum = MP[GB_AN];

        l1 = atv_ijk[Rn][1];
        l2 = atv_ijk[Rn][2];
        l3 = atv_ijk[Rn][3];
        kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

        si = sin(2.0*PI*kRn);
        co = cos(2.0*PI*kRn);

        for (j=0; j<tnoB; j++){
          int ig = Anum + i;
          int jg = Bnum + j;
          size_t idx = (size_t)(jg-1)*(size_t)n + (size_t)(ig-1);

          if (cpx_flag==0){
            Ms[idx].r += M1[k]*co;
            Ms[idx].i += M1[k]*si;
          }
          else if (cpx_flag==1){
            Ms[idx].r -= M1[k]*si;
            Ms[idx].i += M1[k]*co;
          }

          k++;
        }
      }
    }
  }
}

static void Construct_Band_DenseMs(BandNonColState *state,  int cpx_flag, double ****Mat, double *M1, dcomplex *Ms,
                                    int *MP, double k1, double k2, double k3,
                                    int n, int owns_dense )
{

  int i,j,k;
  int MA_AN,GA_AN,LB_AN,GB_AN,AN,Rn,l1,l2,l3;
  int wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM;
  int num,tnum;
  int ID,myid,numprocs;
  int *My_NZeros;
  int *is1,*is2;
  int *My_Matomnum,*order_GA;
  double kRn,si,co;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  My_NZeros = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  My_Matomnum = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  is1 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  is2 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  order_GA = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(atomnum+2),0);

  if (state->Construct_Band_DenseMs_firsttime && memoryusage_fileout) {
    PrintMemory("Construct_Band_DenseMs: My_NZeros", sizeof(int)*numprocs,NULL);
    PrintMemory("Construct_Band_DenseMs: is1", sizeof(int)*numprocs,NULL);
    PrintMemory("Construct_Band_DenseMs: is2", sizeof(int)*numprocs,NULL);
    PrintMemory("Construct_Band_DenseMs: order_GA", sizeof(int)*(atomnum+2),NULL);
  }
  state->Construct_Band_DenseMs_firsttime = 0;

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
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }

  is1[0] = 0;
  for (ID=1; ID<numprocs; ID++){
    is1[ID] = is1[ID-1] + My_NZeros[ID-1];
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

  for (i=0; i<tnum; i++) M1[i] = 0.0;

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
          M1[k] = Mat[MA_AN][LB_AN][i][j];
          k++;
        }
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE,&M1[0],tnum,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  if (owns_dense && scf_eigen_lib_flag==GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum()<=2*n){
    BandNonCol_ConstructCache_Ensure(state, order_GA,MP,n);
    BandNonCol_ConstructDenseMs_OpenACC(state, cpx_flag,n,k1,k2,k3,M1,Ms);
  }
  else if (owns_dense){
    k = 0;
    for (AN=1; AN<=atomnum; AN++){
      GA_AN = order_GA[AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

      for (i=0; i<tnoA; i++){
        for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
          GB_AN = natn[GA_AN][LB_AN];
          Rn = ncn[GA_AN][LB_AN];
          wanB = WhatSpecies[GB_AN];
          tnoB = Spe_Total_CNO[wanB];
          Bnum = MP[GB_AN];

          l1 = atv_ijk[Rn][1];
          l2 = atv_ijk[Rn][2];
          l3 = atv_ijk[Rn][3];
          kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

          si = sin(2.0*PI*kRn);
          co = cos(2.0*PI*kRn);

          for (j=0; j<tnoB; j++){
            int ig = Anum + i;
            int jg = Bnum + j;
            size_t idx = (size_t)(jg-1)*(size_t)n + (size_t)(ig-1);

            if (cpx_flag==0){
              Ms[idx].r += M1[k]*co;
              Ms[idx].i += M1[k]*si;
            }
            else if (cpx_flag==1){
              Ms[idx].r -= M1[k]*si;
              Ms[idx].i += M1[k]*co;
            }

            k++;
          }
        }
      }
    }
  }

  openmx_core_native_free(state->resources,My_NZeros);
  openmx_core_native_free(state->resources,My_Matomnum);
  openmx_core_native_free(state->resources,is1);
  openmx_core_native_free(state->resources,is2);
  openmx_core_native_free(state->resources,order_GA);
}

void Construct_Band_Ms(BandNonColState *state,  int cpx_flag, double ****Mat, double *M1, dcomplex *Ms,
                        int *MP, double k1, double k2, double k3)
{

  int i,j,k;
  int MA_AN,GA_AN,LB_AN,GB_AN,AN,Rn,l1,l2,l3;
  int wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM;
  int num,tnum,num_orbitals;
  int ID,myid,numprocs,tag=999;
  int *My_NZeros;
  int *is1,*ie1,*is2;
  int *My_Matomnum,*order_GA;
  double sum,kRn,si,co;
  double Stime,Etime;
  double AStime,AEtime;
  MPI_Status stat;
  MPI_Request request;
  int ig,jg,il,jl,prow,pcol,brow,bcol;

  if (measure_time){
    dtime(&AStime);
    dtime(&Stime);
  }

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* allocation of arrays */

  My_NZeros = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  My_Matomnum = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  is1 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  ie1 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  is2 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  order_GA = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*(atomnum+2),0);

  if (state->Construct_Band_Ms_firsttime && memoryusage_fileout) {
  PrintMemory("Construct_Band_Ms: My_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: SP_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: SP_Atoms", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: is1", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: ie1", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: is2", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: order_GA", sizeof(int)*(atomnum+2),NULL);
  }
  state->Construct_Band_Ms_firsttime = 1;

  /* find my total number of non-zero elements in myid */

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
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }  

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;

  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }  

  /* set is2 and order_GA */

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

  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum += Spe_Total_CNO[wanA];
  }
  NUM = Anum - 1;

  /* set M1 */

  for (i=0; i<tnum; i++) M1[i] = 0.0;

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
          M1[k] = Mat[MA_AN][LB_AN][i][j]; 
          k++;
	}
      }
    }
  }

  if (measure_time){
    dtime(&Etime);
    printf("timeB1 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  if (measure_time){
    dtime(&Stime);
  }

  /* MPI M1 */

  MPI_Allreduce(MPI_IN_PLACE, &M1[0], tnum, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  if (measure_time){
    dtime(&Etime);
    printf("timeB2 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  if (measure_time){
    dtime(&Stime);
  }

  /* M1 -> Ms */

  k = 0;
  for (AN=1; AN<=atomnum; AN++){
    GA_AN = order_GA[AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];

    for (i=0; i<tnoA; i++){

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
        Rn = ncn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	Bnum = MP[GB_AN];

	l1 = atv_ijk[Rn][1];
	l2 = atv_ijk[Rn][2];
	l3 = atv_ijk[Rn][3];
	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	si = sin(2.0*PI*kRn);
	co = cos(2.0*PI*kRn);

	for (j=0; j<tnoB; j++){
	  ig = Anum+i;
	  jg = Bnum+j;
	    
	  brow = (ig-1)/nblk;
	  bcol = (jg-1)/nblk;

	  prow = brow%np_rows;
	  pcol = bcol%np_cols;

	  if(my_prow==prow && my_pcol==pcol){

	    il = (brow/np_rows+1)*nblk+1;
	    jl = (bcol/np_cols+1)*nblk+1;

	    if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
	      if(my_prow==prow){
		il = il+(ig-1)%nblk;
	      }
	      il = il-nblk;
	    }

	    if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
	      if(my_pcol==pcol){
		jl = jl+(jg-1)%nblk;
	      }
	      jl = jl-nblk;
	    }

            if (cpx_flag==0){
	      Ms[(jl-1)*na_rows+il-1].r += M1[k]*co;
	      Ms[(jl-1)*na_rows+il-1].i += M1[k]*si;
	    }
            else if (cpx_flag==1){
	      Ms[(jl-1)*na_rows+il-1].r -= M1[k]*si;
	      Ms[(jl-1)*na_rows+il-1].i += M1[k]*co;
	    }
	  }
	    
	  k++;
	}
      }
    }
  }

  if (measure_time){
    dtime(&Etime);
    printf("timeB3 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  /* freeing of arrays */

  openmx_core_native_free(state->resources,My_NZeros);
  openmx_core_native_free(state->resources,My_Matomnum);
  openmx_core_native_free(state->resources,is1);
  openmx_core_native_free(state->resources,ie1);
  openmx_core_native_free(state->resources,is2);
  openmx_core_native_free(state->resources,order_GA);

  if (measure_time){
    dtime(&AEtime);
    printf("timeB_all myid=%2d %15.12f\n",myid,AEtime-AStime);
  }
}

double Calc_DM_Band_non_collinear(const OpenMXChemicalPotentials *chemical,
    int calc_flag,
    int store_flag,
    int myid0,
    int myid2,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    int MaxN,
    double k1,
    double k2,
    double k3,
    double *****CDM,
    double *****iDM0,
    double *****EDM,
    double *ko,
    dcomplex *EVec1,
    double *rDM11,
    double *rDM22,
    double *rDM12,
    double *iDM12,
    double *iDM11,
    double *iDM22, 
    double *rEDM11,
    double *rEDM22, const OpenMXBandSpinorScratch *spinors)
{
  int i,j,k,po,p,GA_AN,MA_AN,wanA,tnoA,Anum,Rn,kmin,kmax;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j0,j1,i2,j2,ID,l1,l2,l3;
  double max_x=60.0,dum,co,si,kRn,tmp1,tmp2;
  double FermiF,x,x2,ReA,ReB,ReC,ImA,ImB,ImC;
  double d1,d2,d3,d4,d5,d6,d7,d8,d9,d10;
  double FermiEps = 1.0e-13;
  double Stime,Etime,stime,etime,time,lumos;
  MPI_Status stat;
  MPI_Request request;

  dtime(&stime);

  if (measure_time){
    dtime(&Stime);
  }

  /******************************
      calculation of DM, EDM 
  *******************************/ 

  if (calc_flag==1){

    /* pre-calculation of the Fermi function */
    
    po = 0;
    kmin = is2[myid2];
    kmax = ie2[myid2];
 
    for (k=is2[myid2]; k<=ie2[myid2]; k++){

      x = (ko[k] - chemical->chemical_potential)*Beta;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = FermiFunc_NC(x,k); 
      tmp1 = sqrt(FermiF);

      for (i1=1; i1<=n2; i1++){
	i = (i1-1)*(ie2[myid2]-is2[myid2]+1) + k - is2[myid2];
	EVec1[i].r *= tmp1;
	EVec1[i].i *= tmp1;
      }

      /* find kmax */

      if ( FermiF<FermiEps && po==0 ) {
        kmax = k;
        po = 1;         
      }
    }    

    /* the neighbour planes of this k point for the host SIMD kernel of
       openmx_nc_dm_cpu.h; NULL planes (legacy policy, no memory) keep the
       legacy loops below */
    OpenMXNcDMPanels nc_panels;
    const int nc_mode = OpenMX_HostSimdMode();
    NcDM_Band_Panels(&nc_panels, nc_mode, EVec1, ie2[myid2]-is2[myid2]+1, is2[myid2], kmin, kmax, n, List_YOUSO[7]);

    /* loop for GA_AN */
    
    p = 0;
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

      /* store EVec1 to temporal arrays */

      for (i=0; i<tnoA; i++){

	i1 = (Anum + i - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];
        i2 = (Anum + i + n - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];

	{
	  double *re_i1_row_37=openmx_solver_real_matrix_row(&spinors->ReEVec_i1,i);
	  double *im_i1_row_38=openmx_solver_real_matrix_row(&spinors->ImEVec_i1,i);
	  double *re_i2_row_39=openmx_solver_real_matrix_row(&spinors->ReEVec_i2,i);
	  double *im_i2_row_40=openmx_solver_real_matrix_row(&spinors->ImEVec_i2,i);
	  for (k=kmin; k<=kmax; k++){
	  re_i1_row_37[k] = EVec1[i1+k].r;
	  im_i1_row_38[k] = EVec1[i1+k].i;
	  re_i2_row_39[k] = EVec1[i2+k].r;
	  im_i2_row_40[k] = EVec1[i2+k].i;
	}
	}
      }

      if (nc_panels.re1 != NULL) {
        for (i=0; i<tnoA; i++){
          nc_panels.rows[0*nc_panels.max_rows+i] = openmx_solver_real_matrix_row(&spinors->ReEVec_i1,i) + kmin;
          nc_panels.rows[1*nc_panels.max_rows+i] = openmx_solver_real_matrix_row(&spinors->ImEVec_i1,i) + kmin;
          nc_panels.rows[2*nc_panels.max_rows+i] = openmx_solver_real_matrix_row(&spinors->ReEVec_i2,i) + kmin;
          nc_panels.rows[3*nc_panels.max_rows+i] = openmx_solver_real_matrix_row(&spinors->ImEVec_i2,i) + kmin;
        }
      }

      /* loop for LB_AN */

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	GB_AN = natn[GA_AN][LB_AN];
	Rn = ncn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	Bnum = MP[GB_AN];

	l1 = atv_ijk[Rn][1];
	l2 = atv_ijk[Rn][2];
	l3 = atv_ijk[Rn][3];
	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	si = sin(2.0*PI*kRn);
	co = cos(2.0*PI*kRn);

        /* every component of this block in one pass over the bands */
        if (nc_panels.re1 != NULL) {
          OpenMXEnergyNcBlock block;
          block.rows = tnoA; block.columns = tnoB; block.bands = nc_panels.bands; block.stride = nc_panels.stride;
          block.re0_1 = nc_panels.rows; block.im0_1 = nc_panels.rows + nc_panels.max_rows;
          block.re0_2 = nc_panels.rows + 2*nc_panels.max_rows; block.im0_2 = nc_panels.rows + 3*nc_panels.max_rows;
          block.re1T_1 = nc_panels.re1 + (Bnum - 1); block.im1T_1 = nc_panels.im1 + (Bnum - 1);
          block.re1T_2 = nc_panels.re2 + (Bnum - 1); block.im1T_2 = nc_panels.im2 + (Bnum - 1);
          block.weight = NULL; block.eigen = ko + kmin;
          block.phase = 1; block.co = co; block.si = si;
          block.out[0] = rDM11 + p;  block.out[1] = rDM22 + p;  block.out[2] = rDM12 + p;  block.out[3] = iDM12 + p;
          block.out[4] = iDM11 + p;  block.out[5] = iDM22 + p;  block.out[6] = rEDM11 + p; block.out[7] = rEDM22 + p;
          block.out[8] = NULL; block.out[9] = NULL;
          block.out_stride = (size_t)tnoB;
          if (openmx_core_energy_nc_block(&block, nc_mode)) {
            p += tnoA*tnoB;
            continue;
          }
        }

        /* store EVec1 to temporal arrays */

        for (j=0; j<tnoB; j++){
	  j1 = (Bnum + j - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];
	  j2 = (Bnum + j + n - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];
          {
            double *re_j1_row_33=openmx_solver_real_matrix_row(&spinors->ReEVec_j1,j);
            double *im_j1_row_34=openmx_solver_real_matrix_row(&spinors->ImEVec_j1,j);
            double *re_j2_row_35=openmx_solver_real_matrix_row(&spinors->ReEVec_j2,j);
            double *im_j2_row_36=openmx_solver_real_matrix_row(&spinors->ImEVec_j2,j);
            for (k=kmin; k<=kmax; k++){
	    re_j1_row_33[k] = EVec1[j1+k].r;
	    im_j1_row_34[k] = EVec1[j1+k].i;
	    re_j2_row_35[k] = EVec1[j2+k].r;
	    im_j2_row_36[k] = EVec1[j2+k].i;
          }
          } 
        }

        /* loops for i and j */

	for (i=0; i<tnoA; i++){
	  {
	    double *re_i1_row_29=openmx_solver_real_matrix_row(&spinors->ReEVec_i1,i);
	    double *im_i1_row_30=openmx_solver_real_matrix_row(&spinors->ImEVec_i1,i);
	    double *re_i2_row_31=openmx_solver_real_matrix_row(&spinors->ReEVec_i2,i);
	    double *im_i2_row_32=openmx_solver_real_matrix_row(&spinors->ImEVec_i2,i);
	    for (j=0; j<tnoB; j++){

	    d1 = 0.0;
	    d2 = 0.0;
	    d3 = 0.0;
	    d4 = 0.0;
	    d5 = 0.0;
	    d6 = 0.0;
	    d7 = 0.0;
	    d8 = 0.0;
	    d9 = 0.0;
	    d10= 0.0;

	    {
	      double *re_j1_row_22=openmx_solver_real_matrix_row(&spinors->ReEVec_j1,j);
	      double *im_j1_row_24=openmx_solver_real_matrix_row(&spinors->ImEVec_j1,j);
	      double *re_j2_row_26=openmx_solver_real_matrix_row(&spinors->ReEVec_j2,j);
	      double *im_j2_row_28=openmx_solver_real_matrix_row(&spinors->ImEVec_j2,j);
	      for (k=kmin; k<=kmax; k++){

	      ReA = re_i1_row_29[k]*re_j1_row_22[k] + im_i1_row_30[k]*im_j1_row_24[k];
	      ImA = re_i1_row_29[k]*im_j1_row_24[k] - im_i1_row_30[k]*re_j1_row_22[k];
	      ReB = re_i2_row_31[k]*re_j2_row_26[k] + im_i2_row_32[k]*im_j2_row_28[k];
	      ImB = re_i2_row_31[k]*im_j2_row_28[k] - im_i2_row_32[k]*re_j2_row_26[k];
	      ReC = re_i1_row_29[k]*re_j2_row_26[k] + im_i1_row_30[k]*im_j2_row_28[k];
	      ImC = re_i1_row_29[k]*im_j2_row_28[k] - im_i1_row_30[k]*re_j2_row_26[k];

	      d1 += ReA;
	      d2 += ImA;
	      d3 += ReB;
	      d4 += ImB;
	      d5 += ReC;
	      d6 += ImC;
	      d7 += ReA*ko[k];
	      d8 += ImA*ko[k];
	      d9 += ReB*ko[k];
	      d10 += ImB*ko[k];
	    }
	    }

	    /* Re DM11 */
	    rDM11[p] += co*d1 - si*d2; 

	    /* Re DM22 */
	    rDM22[p] += co*d3 - si*d4;

	    /* Re DM12 */
	    rDM12[p] += co*d5 - si*d6; 

	    /* Im DM12 */
	    iDM12[p] += co*d6 + si*d5;

	    /* Im DM11 */
	    iDM11[p] += co*d2 + si*d1;

	    /* Im DM22 */
	    iDM22[p] += co*d4 + si*d3;

	    /* ReEDM11 */
	    rEDM11[p] += co*d7 - si*d8;

	    /* rEDM22 */
	    rEDM22[p] += co*d9 - si*d10;

	    /* increment of p */
	    p++;  

	  }
	  }
	}
      }
    } /* GA_AN */
    NcDM_PanelsFree(&nc_panels);

  } /* if (calc_flag==1) */

  if (measure_time){
    dtime(&Etime);
    printf("timeA1 myid0=%2d myid2=%2d ie2-is2+1=%2d  %15.12f\n",
            myid0,myid2,ie2[myid2]-is2[myid2]+1,Etime-Stime);
  }

  /***********************************
     store the data to proper arrays
  ************************************/ 

  if (store_flag==1){

    /* MPI_Allreduce */

    if (measure_time){
      dtime(&Stime);
    }

    MPI_Allreduce(MPI_IN_PLACE, rDM11, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, rDM22, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, rDM12, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, iDM11, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, iDM22, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, iDM12, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, rEDM11,size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, rEDM22,size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    if (measure_time){
      dtime(&Etime);
      printf("timeA2 %15.12f\n",Etime-Stime);
    }

    /* store DM1 to a proper place */

    if (measure_time){
      dtime(&Stime);
    }

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

	      CDM[0][MA_AN][LB_AN][i][j] += rDM11[p];  /* Re11 */ 
	      CDM[1][MA_AN][LB_AN][i][j] += rDM22[p];  /* Re22 */ 
	      CDM[2][MA_AN][LB_AN][i][j] += rDM12[p];  /* Re12 */ 
	      CDM[3][MA_AN][LB_AN][i][j] += iDM12[p];  /* Im12 */ 
	      iDM0[0][MA_AN][LB_AN][i][j] += iDM11[p]; /* Im11 */ 
	      iDM0[1][MA_AN][LB_AN][i][j] += iDM22[p]; /* Im22 */ 

	      EDM[0][MA_AN][LB_AN][i][j] += rEDM11[p]; /* ReEDM11 */ 
	      EDM[1][MA_AN][LB_AN][i][j] += rEDM22[p]; /* ReEDM22 */ 

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

    if (measure_time){
      dtime(&Etime);
      printf("timeA3 %15.12f\n",Etime-Stime);
    }

  } /* if (store_flag==1) */

  dtime(&etime);
  return (etime-stime);
}

double Calc_ParDM_Band_non_collinear(BandNonColState *state,
    int calc_flag,
    int store_flag,
    int myid0,
    int myid2,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    int MaxN,
    double k1,
    double k2,
    double k3,
    double *****ParDM,
    double *****iParDM,
    double *ko,
    dcomplex *EVec1,
    double *rDM11,
    double *rDM22,
    double *rDM12,
    double *iDM12,
    double *iDM11,
    double *iDM22, const OpenMXBandSpinorScratch *spinors)
{
  int i,j,k,po,p,GA_AN,MA_AN,wanA,tnoA,Anum,Rn,kmin,kmax;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j0,j1,i2,j2,ID,l1,l2,l3;
  double max_x=60.0,dum,co,si,kRn,tmp1,tmp2;
  double FermiF,x,x2,ReA,ReB,ReC,ImA,ImB,ImC;
  double d1,d2,d3,d4,d5,d6;
  double FermiEps = 1.0e-13;
  double *windowF,e,e0,e1,b0,b1;
  double Stime,Etime,stime,etime,time,lumos;
  MPI_Status stat;
  MPI_Request request;

  dtime(&stime);

  if (measure_time){
    dtime(&Stime);
  }

  /* allocation of windowF */

  windowF = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(n2+1),0);

  /******************************
       calculation of ParDM
  *******************************/ 

  if (calc_flag==1){

    /* set parameters */
    
    po = 0;
    kmin = is2[myid2];
    kmax = ie2[myid2];

    /* pre-calculation of the window function */
 
    for (k=kmin; k<=kmax; k++){

      /* polynomial */
      if (CWF_WindowF_switch==0){

        double in0,in1,out0,out1; 
        out0 = CWF_disentangling_Erange[0];
        in0 = CWF_disentangling_Erange[1];
        in1 = CWF_disentangling_Erange[2];
        out1 = CWF_disentangling_Erange[3];
        e = ko[k];
        windowF[k] = weight_CWF2(e,state->chemical->chemical_potential,in0,in1,out0,out1);
      }

      /* Fermi */
      else if (CWF_WindowF_switch==1){

        e = ko[k];
        b0 = 1.0/CWF_disentangling_smearing_kBT0;
        b1 = 1.0/CWF_disentangling_smearing_kBT1;
        e0 = CWF_disentangling_Erange[0] + state->chemical->chemical_potential;
        e1 = CWF_disentangling_Erange[1] + state->chemical->chemical_potential;

        windowF[k] = (1.0/(exp(b0*(e0-e))+1.0))*(1.0/(exp(b1*(e-e1))+1.0)) + CWF_disentangling_smearing_bound;

      }

    } // k    

    /* loop for GA_AN */
    
    p = 0;
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

      /* store EVec1 to temporal arrays */

      for (i=0; i<tnoA; i++){

	i1 = (Anum + i - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];
        i2 = (Anum + i + n - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];

	{
	  double *re_i1_row_17=openmx_solver_real_matrix_row(&spinors->ReEVec_i1,i);
	  double *im_i1_row_18=openmx_solver_real_matrix_row(&spinors->ImEVec_i1,i);
	  double *re_i2_row_19=openmx_solver_real_matrix_row(&spinors->ReEVec_i2,i);
	  double *im_i2_row_20=openmx_solver_real_matrix_row(&spinors->ImEVec_i2,i);
	  for (k=kmin; k<=kmax; k++){
	  re_i1_row_17[k] = EVec1[i1+k].r;
	  im_i1_row_18[k] = EVec1[i1+k].i;
	  re_i2_row_19[k] = EVec1[i2+k].r;
	  im_i2_row_20[k] = EVec1[i2+k].i;
	}
	}
      }

      /* loop for LB_AN */

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	GB_AN = natn[GA_AN][LB_AN];
	Rn = ncn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	Bnum = MP[GB_AN];

	l1 = atv_ijk[Rn][1];
	l2 = atv_ijk[Rn][2];
	l3 = atv_ijk[Rn][3];
	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	si = sin(2.0*PI*kRn);
	co = cos(2.0*PI*kRn);

        /* store EVec1 to temporal arrays */

        for (j=0; j<tnoB; j++){
	  j1 = (Bnum + j - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];
	  j2 = (Bnum + j + n - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];
          {
            double *re_j1_row_13=openmx_solver_real_matrix_row(&spinors->ReEVec_j1,j);
            double *im_j1_row_14=openmx_solver_real_matrix_row(&spinors->ImEVec_j1,j);
            double *re_j2_row_15=openmx_solver_real_matrix_row(&spinors->ReEVec_j2,j);
            double *im_j2_row_16=openmx_solver_real_matrix_row(&spinors->ImEVec_j2,j);
            for (k=kmin; k<=kmax; k++){
	    re_j1_row_13[k] = EVec1[j1+k].r;
	    im_j1_row_14[k] = EVec1[j1+k].i;
	    re_j2_row_15[k] = EVec1[j2+k].r;
	    im_j2_row_16[k] = EVec1[j2+k].i;
          }
          } 
        }

        /* loops for i and j */

	for (i=0; i<tnoA; i++){
	  {
	    double *re_i1_row_9=openmx_solver_real_matrix_row(&spinors->ReEVec_i1,i);
	    double *im_i1_row_10=openmx_solver_real_matrix_row(&spinors->ImEVec_i1,i);
	    double *re_i2_row_11=openmx_solver_real_matrix_row(&spinors->ReEVec_i2,i);
	    double *im_i2_row_12=openmx_solver_real_matrix_row(&spinors->ImEVec_i2,i);
	    for (j=0; j<tnoB; j++){

	    d1 = 0.0;
	    d2 = 0.0;
	    d3 = 0.0;
	    d4 = 0.0;
	    d5 = 0.0;
	    d6 = 0.0;

	    {
	      double *re_j1_row_2=openmx_solver_real_matrix_row(&spinors->ReEVec_j1,j);
	      double *im_j1_row_4=openmx_solver_real_matrix_row(&spinors->ImEVec_j1,j);
	      double *re_j2_row_6=openmx_solver_real_matrix_row(&spinors->ReEVec_j2,j);
	      double *im_j2_row_8=openmx_solver_real_matrix_row(&spinors->ImEVec_j2,j);
	      for (k=kmin; k<=kmax; k++){

	      ReA = re_i1_row_9[k]*re_j1_row_2[k] + im_i1_row_10[k]*im_j1_row_4[k];
	      ImA = re_i1_row_9[k]*im_j1_row_4[k] - im_i1_row_10[k]*re_j1_row_2[k];
	      ReB = re_i2_row_11[k]*re_j2_row_6[k] + im_i2_row_12[k]*im_j2_row_8[k];
	      ImB = re_i2_row_11[k]*im_j2_row_8[k] - im_i2_row_12[k]*re_j2_row_6[k];
	      ReC = re_i1_row_9[k]*re_j2_row_6[k] + im_i1_row_10[k]*im_j2_row_8[k];
	      ImC = re_i1_row_9[k]*im_j2_row_8[k] - im_i1_row_10[k]*re_j2_row_6[k];

	      d1 += windowF[k]*ReA;
	      d2 += windowF[k]*ImA;
	      d3 += windowF[k]*ReB;
	      d4 += windowF[k]*ImB;
	      d5 += windowF[k]*ReC;
	      d6 += windowF[k]*ImC;

	    }
	    } // k

	    /* Re DM11 */
	    rDM11[p] += co*d1 - si*d2; 

	    /* Re DM22 */
	    rDM22[p] += co*d3 - si*d4;

	    /* Re DM12 */
	    rDM12[p] += co*d5 - si*d6; 

	    /* Im DM12 */
	    iDM12[p] += co*d6 + si*d5;

	    /* Im DM11 */
	    iDM11[p] += co*d2 + si*d1;

	    /* increment of p */
	    p++;  

	  }
	  }
	}
      }
    } /* GA_AN */

  } /* if (calc_flag==1) */

  if (measure_time){
    dtime(&Etime);
    printf("timeA1 myid0=%2d myid2=%2d ie2-is2+1=%2d  %15.12f\n",
            myid0,myid2,ie2[myid2]-is2[myid2]+1,Etime-Stime);
  }

  /***********************************
     store the data to proper arrays
  ************************************/ 

  if (store_flag==1){

    /* MPI_Allreduce */

    if (measure_time){
      dtime(&Stime);
    }

    MPI_Allreduce(MPI_IN_PLACE, rDM11, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, rDM22, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, rDM12, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, iDM11, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, iDM22, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, iDM12, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    if (measure_time){
      dtime(&Etime);
      printf("timeA2 %15.12f\n",Etime-Stime);
    }

    /* store DM1 to a proper place */

    if (measure_time){
      dtime(&Stime);
    }

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

	      ParDM[0][MA_AN][LB_AN][i][j] += rDM11[p];  /* Re11 */ 
	      ParDM[1][MA_AN][LB_AN][i][j] += rDM22[p];  /* Re22 */ 
	      ParDM[2][MA_AN][LB_AN][i][j] += rDM12[p];  /* Re12 */ 
	      ParDM[3][MA_AN][LB_AN][i][j] += iDM12[p];  /* Im12 */ 
	      iParDM[0][MA_AN][LB_AN][i][j] += iDM11[p]; /* Im11 */ 
	      iParDM[1][MA_AN][LB_AN][i][j] += iDM22[p]; /* Im22 */ 

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

    if (measure_time){
      dtime(&Etime);
      printf("timeA3 %15.12f\n",Etime-Stime);
    }

  } /* if (store_flag==1) */

  /* freeing of windowF */

  openmx_core_native_free(state->resources,windowF);

  dtime(&etime);
  return (etime-stime);
}

/* Run-boundary reset: geometry and device state must not survive into the
   next -runtest input, whose host allocations can reuse the same addresses.
   Called on every rank before Free_Arrays(0) frees the current system. */
static void BandNonCol_ReleaseCaches(BandNonColState *state)
{
    state->mgpu_host_dense=0;
    BandNonCol_RefineRelease(state);
    BandNonCol_KOverlapReset(state);
    BandNonCol_ConstructCache_Reset(state);
    BandNonCol_DMEntryCache_Reset(state);
    BandNonCol_RootDenseWorkspace_Reset(state);
    BandNonCol_GpuSolver_Destroy(state);
    BandNonCol_DMGpu_Destroy(state);
    openmx_core_native_free(state->resources,state->dm_occ_workspace.occ);
    openmx_core_native_free(state->resources,state->dm_occ_workspace.eig_occ);
    memset(&state->dm_occ_workspace, 0, sizeof(state->dm_occ_workspace));
    /* The next input owns a fresh refinement policy and no prior basis. */
    const BandNonColRefineState initial = {-1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    state->refine = initial;
}
