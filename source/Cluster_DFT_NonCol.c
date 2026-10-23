/**********************************************************************
  Cluster_DFT_NonCol.c:

     Cluster_DFT_NonCol.c is a subroutine to perform non-collinear
     cluster calculations.

  Log of Cluster_DFT_NonCol.c:

     21/Feb./2019  Released by T. Ozaki

***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_solver_workspace.h"
#include "openmx_native_resources.h"
#include "openmx_mgpu_runtime.h"
#include "openmx_gpu_transform.h"
#include "openmx_gpu_dm_collective.h"
#include "openmx_gpu_workers.h"
#include "openmx_mgpu_density.h"
#include "lapack_prototypes.h"
#include "set_cuda_default_device_from_local_rank.h"
#include "set_openacc_device_from_local_rank.h"
#include "elpa_cosma_bridge.h"
#include "openmx_cluster_capacity.h"
#include "eigen_refine_gpu.h"
#include "openmx_precision.h"
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* Persistent metadata belongs to the explicit calculation. Buffer fields
   borrow storage owned by its native resource group. */
enum { NCPROF_OVERLAP, NCPROF_GATHER, NCPROF_FORWARD, NCPROF_EIGEN, NCPROF_BACK, NCPROF_RELEASE, NCPROF_DM,
       NCPROF_COUNT };

typedef struct
{
    int            valid;
    int            n;
    int            n2;
    size_t         tnum_capacity;   /* elements covered by o_dindex/o_stage */
    size_t         total_bytes;
    unsigned char *base;
    size_t         o_S;
    size_t         o_ko;
    size_t         o_r[7];          /* rHs11,rHs12,rHs22,iHs11,iHs12,iHs22,Cs */
    size_t         o_Hs2;           /* Hs2 and its eigenvectors, then the back-transformed dense_evec */
    size_t         o_vt;            /* the eigenvectors transposed for the back transform */
    size_t         o_dindex;
    size_t         o_stage;
} ClusterNonColDenseArena;

typedef struct
{
    int       n;
    int       n2;
    int       transformed_s_valid;
    int       s_on_device;
    double   *S;
} ClusterNonColDenseSCache;

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
} ClusterNonColRootDMWorkspace;

typedef struct
{
    int                inited;
    int                device;
    int                last_n;
    int                last_maxn;
    cusolverDnHandle_t handle;
    cudaStream_t       stream;
    size_t             d_work_bytes;
    void              *d_work;
    size_t             h_work_bytes;
    void              *h_work;
    int32_t           *d_info;
} ClusterNonColGpuEigenCtx;

typedef struct
{
    int     iterations;    /* -1 until the environment is read */
    double  until;         /* printed NormRD below which the FP64 eigensolver takes over; 0 off */
    int     final_stage;   /* the FP64 eigensolver for the rest of the SCF cycle */
    int     last_refined;  /* the latest SCF eigensolve was a refined one */
    int     cycle_refined; /* some eigensolve of this SCF cycle was */
    int     defaulted;     /* on because of the device, not of OPENMX_EIGEN_REFINE */
    int     force_fp64;    /* mode "fp64" of Cluster_DFT_NonCol: this solve in FP64 whatever the stage */
} ClusterNonColRefineState;

typedef struct
{
    double mu;
} ClusterNonColOccupationCtx;

typedef struct {
    int valid;
    int device_valid;
    int n;
    int tnum;
    unsigned long long fingerprint;
    int *dense_index;
} ClusterNonColDenseIndexCache;

typedef struct
{
    int       n;
    int       n2;
    double   *Cs;
    double   *rHs11;
    double   *rHs12;
    double   *rHs22;
    double   *iHs11;
    double   *iHs12;
    double   *iHs22;
    dcomplex *Hs2;
} ClusterNonColRootSolveWorkspace;

struct OpenMXClusterNonColState {
  OpenMXChemicalPotentials *chemical; /* borrowed from this calculation */
  OpenMXThermodynamicEnergies *thermo; /* borrowed from this calculation */
  OpenMXNativeResources *resources; /* borrowed from this calculation */
  int initialized;
  ClusterNonColDenseArena dense_arena;
  int mgpu_host_dense;
  int gs2_fallback_n2;
  int cpu_fallback_n2;
  int cpu_overlap_n2;
  ClusterNonColDenseSCache dense_scache;
  ClusterNonColDMEntryCache    dm_entry_cache;
  ClusterNonColRootDMWorkspace root_dm_workspace;
  dcomplex *CachedGpuSolverDenseEVec;
  int CachedGpuSolverDenseN2;
  int CachedGpuSolverDenseOnDevice;
  ClusterNonColGpuEigenCtx gpu_eigen_ctx;
  ClusterNonColRefineState refine;
  EigenRefineState         refine_state;
  EigenRefineDevice        refine_dev;
  double prof[NCPROF_COUNT];
  long prof_solves;
  ClusterNonColDenseIndexCache dense_index_cache;
  ClusterNonColRootSolveWorkspace root_solve_workspace;
  int firsttime;
  int GpuDiagFits_verdict_n2;
  int GpuDiagFits_verdict;
};
typedef OpenMXClusterNonColState ClusterNonColState;

static void ClusterNonCol_ReleaseCaches(ClusterNonColState *state);
static void ClusterNonCol_Reset(OpenMXNativeResources *owner, void *metadata)
{
  ClusterNonColState *state = (ClusterNonColState *)metadata;
  (void)owner;
  if (state->initialized) ClusterNonCol_ReleaseCaches(state);
}

OpenMXClusterNonColState *Cluster_DFT_NonCol_Bind_Resources(OpenMXCoreContext *core)
{
  OpenMXNativeResources *owner = NULL;
  void *metadata = NULL;
  char error[256];
  if (!openmx_core_native_resources_state(core, "Cluster_DFT_NonCol",
      sizeof(ClusterNonColState), ClusterNonCol_Reset, &owner, &metadata, error, sizeof(error))) {
    fprintf(stderr, "OpenMX noncollinear cluster state: %s\n", error);
    abort();
  }
  ClusterNonColState *state = (ClusterNonColState *)metadata;
  if (!state->initialized) {
    state->resources = owner;
    const ClusterNonColRefineState initial = {-1, 0.0, 0, 0, 0, 0, 0};
    state->refine = initial;
    state->firsttime = 1;
    state->GpuDiagFits_verdict = -1;
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

#define  measure_time   0

static void ClusterNonCol_AbortWithMessage(const char *message)
{
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1, 1);
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

static void *ClusterNonCol_MallocArray(ClusterNonColState *state, size_t count, size_t elem_size, const char *label)
{
    size_t bytes = ClusterNonCol_CheckedMulCount(count, elem_size, label);
    void *ptr = openmx_core_native_malloc(state->resources,1,(bytes == 0) ? 1 : bytes,0);

    if (ptr == NULL) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Out of memory in Cluster_DFT_NonCol.c: %s (%zu bytes)", label, bytes);
        ClusterNonCol_AbortWithMessage(msg);
    }
    return ptr;
}

/* Bounded device allocation for the root dense path (mirrors
   Cluster_DFT_Col.c): wait_cudafunc() retries forever, which suits transient
   pressure but deadlocks on a device that stays full.  Absorb short races
   with a few retries and report a persistent failure instead. */
static cudaError_t ClusterNonCol_TryDeviceMalloc(ClusterNonColState *state, void **ptr, size_t bytes)
{
    cudaError_t err = cudaErrorMemoryAllocation;

    for (int attempt = 0; attempt < 8; attempt++) {
        err = OpenMX_Resource_GpuMalloc(state->resources,ptr,bytes);
        if (err == cudaSuccess) return cudaSuccess;
        (void)cudaGetLastError();

        double wait_time = drand48() * WAITTIME;
        double start_time = MPI_Wtime();
        double current_time = start_time;
        while ((current_time - start_time) < wait_time) {
            current_time = MPI_Wtime();
        }
    }

    *ptr = NULL;
    return err;
}

/* Device arena of the root dense solve.  The dense owner reserves the whole
   device footprint of a diagonalization cycle in ONE allocation during the
   GPU-diag preflight and maps every device buffer of the dense path into it
   with acc_map_data.  A positive preflight verdict therefore cannot be
   invalidated later in the SCF cycle: no allocation the dense path depends
   on can fail once the arena exists (the OpenACC enter-data creates this
   replaces abort fatally on a full device). */

/* spinor order for which the automatic capacity fallback of gpusolver
   chose the distributed gpusolver2 path (0: none); see
   openmx_cluster_capacity.h */

/* the distributed ELPA/COSMA path: requested by the input, or the capacity
   fallback of gpusolver for this spinor order */
static int ClusterNonCol_UseGs2(ClusterNonColState *state, int n2)
{
    return gpusolver2_flag || (0 < n2 && state->gs2_fallback_n2 == n2);
}

static void ClusterNonCol_RootHostBuffers_Release(ClusterNonColState *state);

static void ClusterNonCol_ResetGs2Fallback(ClusterNonColState *state)
{
    if (state->gs2_fallback_n2 != 0) openmx_gs2_set_fallback_report(0);
    state->gs2_fallback_n2 = 0;
    state->cpu_fallback_n2 = 0;
}

static size_t ClusterNonCol_ArenaOff(size_t *pos, size_t bytes)
{
    size_t off = *pos;
    size_t padded = ClusterNonCol_CheckedAddCount(bytes, 511u, "dense arena segment");

    padded &= ~(size_t)511u;
    *pos = ClusterNonCol_CheckedAddCount(off, padded, "dense arena layout");
    return off;
}

static void *ClusterNonCol_ArenaPtr(ClusterNonColState *state, size_t off)
{
    return (void *)(state->dense_arena.base + off);
}

static void ClusterNonCol_ArenaMap(ClusterNonColState *state, const void *host, size_t off, size_t bytes)
{
    acc_map_data((void *)host, ClusterNonCol_ArenaPtr(state, off), bytes);
}

static void ClusterNonCol_ArenaUnmap(const void *host)
{
    acc_unmap_data((void *)host);
}

/* Transformed-overlap cache of the root dense path, hoisted to file scope so
   the arena release can invalidate it.  The host buffer persists across
   cycles; the device copy lives in the arena. */

static void ClusterNonCol_DenseArena_Release(ClusterNonColState *state);

static void ClusterNonCol_DMEntryCache_Reset(ClusterNonColState *state)
{
    openmx_core_native_free(state->resources,state->dm_entry_cache.basis0);
    openmx_core_native_free(state->resources,state->dm_entry_cache.basis1);
    memset(&state->dm_entry_cache, 0, sizeof(state->dm_entry_cache));
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

static void ClusterNonCol_DMEntryCache_Ensure(ClusterNonColState *state, int *MP, int n)
{
    ClusterNonColDMEntryCache *cache = &state->dm_entry_cache;
    unsigned long long fingerprint = ClusterNonCol_DMEntryFingerprint(MP, n);
    int entry_count = 0;

    if (cache->valid && cache->n == n && cache->fingerprint == fingerprint) {
        return;
    }

    ClusterNonCol_DMEntryCache_Reset(state);

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]];

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int tnoB  = Spe_Total_CNO[WhatSpecies[GB_AN]];
            entry_count += tnoA * tnoB;
        }
    }

    cache->basis0 = (int *)ClusterNonCol_MallocArray(state, (size_t)(entry_count + 1), sizeof(int), "DM basis0 cache");
    cache->basis1 = (int *)ClusterNonCol_MallocArray(state, (size_t)(entry_count + 1), sizeof(int), "DM basis1 cache");

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

static void ClusterNonCol_RootDMWorkspace_Reset(ClusterNonColState *state)
{
    openmx_core_native_free(state->resources,state->root_dm_workspace.occ);
    openmx_core_native_free(state->resources,state->root_dm_workspace.occ_e);
    openmx_core_native_free(state->resources,state->root_dm_workspace.dm_buffer);
    memset(&state->root_dm_workspace, 0, sizeof(state->root_dm_workspace));
}

static void ClusterNonCol_RootDMWorkspace_Ensure(ClusterNonColState *state, int max_state, size_t dm_buffer_elems)
{
    ClusterNonColRootDMWorkspace *ws = &state->root_dm_workspace;

    if (max_state <= ws->max_state_capacity && dm_buffer_elems <= ws->dm_buffer_elems) {
        return;
    }

    ClusterNonCol_RootDMWorkspace_Reset(state);

    ws->occ       = (double *)ClusterNonCol_MallocArray(state, (size_t)max_state, sizeof(double), "root DM occupations");
    ws->occ_e     = (double *)ClusterNonCol_MallocArray(state, (size_t)max_state, sizeof(double), "root DM energy occupations");
    ws->dm_buffer = (double *)ClusterNonCol_MallocArray(state, dm_buffer_elems, sizeof(double), "root DM buffer");

    ws->max_state_capacity = max_state;
    ws->dm_buffer_elems    = dm_buffer_elems;
}

static int ClusterNonCol_PrecomputeOccupations(const OpenMXChemicalPotentials *chemical, int kmin, int kmax, const double *ko, double *occ, double *occ_e)
{
    int nk = 0;
    double max_x = 60.0;

    for (int local_k = 0, global_k = kmin; global_k <= kmax; local_k++, global_k++) {
        double x = (ko[global_k] - chemical->chemical_potential) * Beta;
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

static cublasHandle_t ClusterNonCol_CublasHandle(ClusterNonColState *state);

/* OPENMX_CLUSTER_GPU_DM_GEMM (default 1): the density matrix of the root
   dense path as one product P = W^H W, W = diag(sqrt f) V over the occupied
   states, followed by a gather of the sparse entries, instead of the
   per-entry sums over the states (sidia333 on an RTX 5080: 432 ms per step
   for 3.2 million entries x 888 states; the product, 2.3e11 flops through
   GEMMul8, and the gather take a few tens of ms).  0 keeps the sums. */
static int ClusterNonCol_DmByGemm(void)
{
    return openmx_env_flag_at(OPENMX_ENV_CLUSTER_GPU_DM_GEMM, 1);
}

static double ClusterNonCol_CalcDMRootDense_OpenACC(ClusterNonColState *state, int myid, int size_H1, int *MP, int n, int n2, int max_state,
                                                    double *****CDM, double *****iDM0, double *****EDM, double *ko,
                                                    const dcomplex *dense_evec, int calc_edm)
{
    ClusterNonColRootDMWorkspace *ws = &state->root_dm_workspace;
    const int dm_components = calc_edm ? 10 : 6;
    const int dm_chunk_size = 131072;
    size_t dm_elems = (size_t)size_H1 * (size_t)dm_components;
    size_t evec_count = (size_t)n2 * (size_t)n2;
    int nk_occ = 0,recover=0;
    double stime, etime;

    if ((size_t)INT_MAX < dm_elems) {
        ClusterNonCol_AbortWithMessage("Root DM buffer exceeds MPI count limit in Cluster_DFT_NonCol.c.");
    }

    ClusterNonCol_RootDMWorkspace_Ensure(state, max_state, dm_elems);

    dtime(&stime);
    if (myid == Host_ID) memset(ws->dm_buffer, 0, sizeof(double) * dm_elems);

    if (myid == Host_ID) {
        ClusterNonColDMEntryCache *cache;
        const double *occ;
        const double *occ_e;
        double *rDM11;
        double *rDM22;
        double *rDM12;
        double *iDM12;
        double *iDM11;
        double *iDM22;
        double *rEDM11;
        double *rEDM22;
        double *rEDM12;
        double *iEDM12;

        ClusterNonCol_DMEntryCache_Ensure(state, MP, n);
        cache = &state->dm_entry_cache;
        if (cache->entry_count != size_H1) {
            ClusterNonCol_AbortWithMessage("DM entry cache size mismatch in Cluster_DFT_NonCol.c.");
        }

        nk_occ = ClusterNonCol_PrecomputeOccupations(state->chemical, 1, max_state, ko, ws->occ, ws->occ_e);
        occ = ws->occ;
        occ_e = ws->occ_e;

        rDM11  = ws->dm_buffer;
        rDM22  = ws->dm_buffer + (size_t)size_H1;
        rDM12  = ws->dm_buffer + (size_t)size_H1 * 2;
        iDM12  = ws->dm_buffer + (size_t)size_H1 * 3;
        iDM11  = ws->dm_buffer + (size_t)size_H1 * 4;
        iDM22  = ws->dm_buffer + (size_t)size_H1 * 5;
        rEDM11 = ws->dm_buffer + (size_t)size_H1 * 6;
        rEDM22 = ws->dm_buffer + (size_t)size_H1 * 7;
        rEDM12 = ws->dm_buffer + (size_t)size_H1 * 8;
        iEDM12 = ws->dm_buffer + (size_t)size_H1 * 9;

        double *dm_outputs[10]={rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
            calc_edm?rEDM11:NULL,calc_edm?rEDM22:NULL,calc_edm?rEDM12:NULL,calc_edm?iEDM12:NULL};

        /* the product form needs W (nk_occ x n2 complex) in the forward-transform
           segments of the arena, idle now, and P (n2 x n2 complex) in a buffer
           of its own for the step (the transpose segment keeps the vectors of
           the refined eigensolver for its warm start) */
        void *pm_buffer = NULL;
        int const by_gemm =
            (0 < nk_occ && !openmx_gpu_capacity_test_recovery() && ClusterNonCol_DmByGemm() && state->dense_arena.valid &&
             (size_t)nk_occ * (size_t)n2 * sizeof(dcomplex) <= 7 * (size_t)n * (size_t)n * sizeof(double) &&
             ClusterNonCol_TryDeviceMalloc(state, &pm_buffer, evec_count * sizeof(dcomplex)) == cudaSuccess);

        if (state->mgpu_host_dense && 0<nk_occ && !openmx_gpu_capacity_test_recovery()) {
            dcomplex *pm=ClusterNonCol_MallocArray(state, evec_count,sizeof(dcomplex),"host density matrix");
            for (int pass=0;pass<(calc_edm?2:1);++pass) {
                if (!OpenMX_MgpuDensityHost(n2,nk_occ,pass?occ_e:occ,dense_evec,pm))
                    ClusterNonCol_AbortWithMessage(openmx_mgpu_host_error());
#pragma omp parallel for
                for (int p=0;p<cache->entry_count;++p) {
                    size_t ia=cache->basis0[p],ib=cache->basis1[p];
                    dcomplex p11=pm[ia+ib*n2],p22=pm[ia+n+(ib+n)*n2],p12=pm[ia+(ib+n)*n2];
                    if (!pass) {
                        rDM11[p]=p11.r; iDM11[p]=p11.i;
                        rDM22[p]=p22.r; iDM22[p]=p22.i;
                        rDM12[p]=p12.r; iDM12[p]=p12.i;
                    } else {
                        rEDM11[p]=p11.r; rEDM22[p]=p22.r;
                        rEDM12[p]=p12.r; iEDM12[p]=p12.i;
                    }
                }
            }
            openmx_core_native_free(state->resources,pm);
        }
        else if (by_gemm) {
            ClusterNonColDenseArena *arena = &state->dense_arena;
            cublasHandle_t handle = ClusterNonCol_CublasHandle(state);
            cuDoubleComplex const one  = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex const zero = make_cuDoubleComplex(0.0, 0.0);
            dcomplex *w  = (dcomplex *)ClusterNonCol_ArenaPtr(state, arena->o_r[0]);
            dcomplex *pm = (dcomplex *)pm_buffer;
            double *wt = (double *)ClusterNonCol_ArenaPtr(state, arena->o_ko);   /* the weights, n2 + 2 doubles */
            const dcomplex *ev = (const dcomplex *)acc_deviceptr((void *)dense_evec);
            const int dm_gather_chunk = 1 << 20;
            size_t const nkn2 = (size_t)nk_occ * (size_t)n2;

            for (int pass = 0; pass < (calc_edm ? 2 : 1); pass++) {
                /* pass 0: P = W^H W with W = diag(sqrt f) V, the density matrix;
                   pass 1: W^H V with W = diag(f e) V, the energy density matrix */
                double *weights = (double *)ClusterNonCol_MallocArray(state, (size_t)nk_occ, sizeof(double), "DM weights");

                for (int k = 0; k < nk_occ; k++) weights[k] = pass == 0 ? sqrt(fmax(0.0, occ[k])) : occ_e[k];
                wait_cudafunc(cudaMemcpy(wt, weights, (size_t)nk_occ * sizeof(double), cudaMemcpyHostToDevice));
                openmx_core_native_free(state->resources,weights);

if (!openmx_gpu_transform(OPENMX_GPU_SCALE_ROWS,1,nk_occ,n2,ev,n2,wt,0,w,nk_occ,0,0.0)) {
#pragma acc parallel loop deviceptr(w, ev, wt)
                for (size_t i = 0; i < nkn2; i++) {
                    size_t const k = i % (size_t)nk_occ, ia = i / (size_t)nk_occ;
                    dcomplex const v = ev[k + ia * (size_t)n2];

                    w[i].r = wt[k] * v.r;
                    w[i].i = wt[k] * v.i;
                }
}

                OpenMXBlasWait(openmx_gemmul8Zgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, n2, n2, nk_occ, &one,
                                                  (cuDoubleComplex const *)w, nk_occ,
                                                  (cuDoubleComplex const *)(pass == 0 ? w : ev), pass == 0 ? nk_occ : n2,
                                                  &zero, (cuDoubleComplex *)pm, n2));
                wait_cudafunc(cudaDeviceSynchronize());

                double *gather_outputs[10]={0};
                if (pass==0) {
                    for (int i=0;i<6;i++) gather_outputs[i]=dm_outputs[i];
                } else {
                    for (int i=0;i<4;i++) gather_outputs[i]=dm_outputs[i+6];
                }
                if (openmx_gpu_dm_entries(pass?OPENMX_GPU_EDM_GATHER:OPENMX_GPU_DM_GATHER,
                        n,0,n2,cache->entry_count,pm,0,cache->basis0,cache->basis1,
                        NULL,0,NULL,NULL,NULL,NULL,NULL,gather_outputs,0)) continue;

                for (int offset = 0; offset < cache->entry_count; offset += dm_gather_chunk) {
                    const int chunk_count =
                        (cache->entry_count - offset < dm_gather_chunk) ? (cache->entry_count - offset) : dm_gather_chunk;
                    const int *basis0_chunk = cache->basis0 + offset;
                    const int *basis1_chunk = cache->basis1 + offset;

                    if (pass == 0) {
                        double *rDM11_chunk = rDM11 + offset;
                        double *rDM22_chunk = rDM22 + offset;
                        double *rDM12_chunk = rDM12 + offset;
                        double *iDM12_chunk = iDM12 + offset;
                        double *iDM11_chunk = iDM11 + offset;
                        double *iDM22_chunk = iDM22 + offset;

#pragma acc parallel loop copyin(basis0_chunk[0 : chunk_count], basis1_chunk[0 : chunk_count]) \
    copyout(rDM11_chunk[0 : chunk_count], rDM22_chunk[0 : chunk_count], rDM12_chunk[0 : chunk_count], \
            iDM12_chunk[0 : chunk_count], iDM11_chunk[0 : chunk_count], iDM22_chunk[0 : chunk_count]) deviceptr(pm)
                        for (int p = 0; p < chunk_count; p++) {
                            const size_t ia = (size_t)basis0_chunk[p];
                            const size_t ib = (size_t)basis1_chunk[p];
                            const dcomplex p11 = pm[ia + ib * (size_t)n2];
                            const dcomplex p22 = pm[(ia + (size_t)n) + (ib + (size_t)n) * (size_t)n2];
                            const dcomplex p12 = pm[ia + (ib + (size_t)n) * (size_t)n2];

                            rDM11_chunk[p] = p11.r;
                            iDM11_chunk[p] = p11.i;
                            rDM22_chunk[p] = p22.r;
                            iDM22_chunk[p] = p22.i;
                            rDM12_chunk[p] = p12.r;
                            iDM12_chunk[p] = p12.i;
                        }
                    }
                    else {
                        double *rEDM11_chunk = rEDM11 + offset;
                        double *rEDM22_chunk = rEDM22 + offset;
                        double *rEDM12_chunk = rEDM12 + offset;
                        double *iEDM12_chunk = iEDM12 + offset;

#pragma acc parallel loop copyin(basis0_chunk[0 : chunk_count], basis1_chunk[0 : chunk_count]) \
    copyout(rEDM11_chunk[0 : chunk_count], rEDM22_chunk[0 : chunk_count], rEDM12_chunk[0 : chunk_count], \
            iEDM12_chunk[0 : chunk_count]) deviceptr(pm)
                        for (int p = 0; p < chunk_count; p++) {
                            const size_t ia = (size_t)basis0_chunk[p];
                            const size_t ib = (size_t)basis1_chunk[p];
                            const dcomplex p11 = pm[ia + ib * (size_t)n2];
                            const dcomplex p22 = pm[(ia + (size_t)n) + (ib + (size_t)n) * (size_t)n2];
                            const dcomplex p12 = pm[ia + (ib + (size_t)n) * (size_t)n2];

                            rEDM11_chunk[p] = p11.r;
                            rEDM22_chunk[p] = p22.r;
                            rEDM12_chunk[p] = p12.r;
                            iEDM12_chunk[p] = p12.i;
                        }
                    }
                }
            }
            wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,pm_buffer));
        }
        else if (0 < nk_occ) recover=1;

    }

    ClusterNonColDMEntryCache *cache=&state->dm_entry_cache;
    double *outputs[10]={0};
    for(int c=0;c<dm_components;c++)outputs[c]=ws->dm_buffer+(size_t)c*size_H1;
    const void *vectors=NULL;int host=0;
    if(recover){vectors=acc_deviceptr((void*)dense_evec);host=vectors==NULL;if(host)vectors=dense_evec;}
    openmx_gpu_dm_collective(mpi_comm_level1,OPENMX_GPU_DM_NC,n,recover?nk_occ:0,n2,
        recover?cache->entry_count:0,vectors,host,cache->basis0,cache->basis1,NULL,0,NULL,NULL,
        ws->occ,calc_edm?ws->occ_e:NULL,NULL,outputs,0);

    /* only the root wrote entries: the sum over the ranks is its buffer */
    MPI_Bcast(ws->dm_buffer, (int)dm_elems, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    ClusterNonCol_StoreDMBuffer(myid, size_H1, MP, CDM, iDM0, ws->dm_buffer);

    if (calc_edm) {
        ClusterNonCol_StoreEDMBuffer(myid, size_H1, MP, EDM, ws->dm_buffer + (size_t)size_H1 * 6);
    }

    dtime(&etime);
    return etime - stime;
}

static double ClusterNonCol_ScatterDenseEVecToLocal(ClusterNonColState *state, int myid, int numprocs, int n2, const int *is2, const int *ie2,
                                                    dcomplex *dense_evec, int dense_evec_on_device,
                                                    dcomplex *EVec1)
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
        size_t evec_count = (size_t)n2 * (size_t)n2;
        size_t max_send_count = (size_t)max_local_states * (size_t)n2;
        dcomplex *panel_dev = NULL;
        int pack_on_device = dense_evec_on_device;

        if (0 < max_send_count) {
            if (max_send_count > (size_t)INT_MAX / 2u) {
                ClusterNonCol_AbortWithMessage("GPUSOLVER dense eigenvector send buffer exceeds MPI count limit.");
            }
            sendbuf = (dcomplex *)ClusterNonCol_MallocArray(state, max_send_count, sizeof(dcomplex),
                                                            "GPUSOLVER dense eigenvector send buffer");
        }

        if (pack_on_device && 0 < max_send_count) {
            /* pack each destination's state-major panel on the device so the
               host never materialises the full 16*n2*n2 eigenvector block
               (2.8 GiB at n2=13312) -- only one panel travels at a time */
            panel_dev = (dcomplex *)OpenMX_Resource_AccMalloc(state->resources,sizeof(dcomplex) * max_send_count);
            if (panel_dev == NULL) {
                pack_on_device = 0;
            }
        }

        if (dense_evec_on_device && !pack_on_device) {
            /* device momentarily full: fall back to the old full readback */
#pragma acc update self(dense_evec[0 : evec_count])
        }

        for (int ID = 0; ID < numprocs; ID++) {
            int states = ie2[ID] - is2[ID] + 1;
            dcomplex *target = (ID == Host_ID) ? EVec1 : sendbuf;

            if (states <= 0) continue;

            if (pack_on_device) {
                const int is_c = is2[ID];
                const int states_c = states;
                const int n2_c = n2;
                dcomplex *panel = panel_dev;

if (!openmx_gpu_transform(OPENMX_GPU_TRANSPOSE,1,n2_c,states_c,(dcomplex*)acc_deviceptr((void*)(dense_evec))+is_c-1,n2_c,NULL,0,panel,n2_c,0,0.0)) {
#pragma acc parallel loop gang vector_length(128) present(dense_evec[0 : evec_count]) deviceptr(panel)
                for (int k = 0; k < states_c; k++) {
#pragma acc loop vector
                    for (int basis = 0; basis < n2_c; basis++) {
                        panel[(size_t)k * (size_t)n2_c + (size_t)basis] =
                            dense_evec[(size_t)(is_c + k - 1) + (size_t)basis * (size_t)n2_c];
                    }
                }
}

                acc_memcpy_from_device(target, panel_dev,
                                       sizeof(dcomplex) * (size_t)states * (size_t)n2);
            }
            else {
                for (int k = 0; k < states; k++) {
                    int state = is2[ID] + k - 1;

                    for (int basis = 0; basis < n2; basis++) {
                        /* Repack the dense column-major gpuSOLVER vectors into the
                           state-major EVec1 blocks used by the ELPA/ScaLAPACK DM path. */
                        target[(size_t)k * (size_t)n2 + (size_t)basis] =
                            dense_evec[(size_t)state + (size_t)basis * (size_t)n2];
                    }
                }
            }

            if (ID != Host_ID) {
                size_t send_count = (size_t)states * (size_t)n2;
                MPI_Send(sendbuf, (int)(send_count * 2u), MPI_DOUBLE, ID, tag, mpi_comm_level1);
            }
        }

        if (panel_dev != NULL) OpenMX_Resource_AccFree(state->resources,panel_dev);
    }
    else {
        int states = ie2[myid] - is2[myid] + 1;

        if (0 < states) {
            size_t recv_count = (size_t)states * (size_t)n2;
            MPI_Status stat;

            if (recv_count > (size_t)INT_MAX / 2u) {
                ClusterNonCol_AbortWithMessage("GPUSOLVER dense eigenvector receive buffer exceeds MPI count limit.");
            }
            MPI_Recv(EVec1, (int)(recv_count * 2u), MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &stat);
        }
    }

    openmx_core_native_free(state->resources,sendbuf);
    dtime(&etime);
    return etime - stime;
}

static void ClusterNonCol_ReleaseGpuSolverCachedEVec(ClusterNonColState *state, int myid)
{
    dcomplex *dense_evec = state->CachedGpuSolverDenseEVec;

    if (myid == Host_ID && dense_evec != NULL) {
        if (state->CachedGpuSolverDenseOnDevice) {
            /* the device copy is a mapped arena segment, not an allocation */
            ClusterNonCol_ArenaUnmap(dense_evec);
        }
        openmx_core_native_free(state->resources,dense_evec);
    }

    state->CachedGpuSolverDenseEVec = NULL;
    state->CachedGpuSolverDenseN2 = 0;
    state->CachedGpuSolverDenseOnDevice = 0;
}

static void ClusterNonCol_StashGpuSolverDenseEVec(ClusterNonColState *state, int myid, int n2, dcomplex **dense_evec, int *on_device)
{
    ClusterNonCol_ReleaseGpuSolverCachedEVec(state, myid);

    if (myid == Host_ID) {
        state->CachedGpuSolverDenseEVec = *dense_evec;
        state->CachedGpuSolverDenseN2 = n2;
        state->CachedGpuSolverDenseOnDevice = *on_device;
    }

    *dense_evec = NULL;
    *on_device = 0;
}

void Cluster_DFT_NonCol_DemoteGpuSolverCachedEVec(ClusterNonColState *state)
{
    dcomplex *dense_evec = state->CachedGpuSolverDenseEVec;
    int n2 = state->CachedGpuSolverDenseN2;

    if (dense_evec != NULL && state->CachedGpuSolverDenseOnDevice) {
        size_t evec_count = ClusterNonCol_CheckedMulCount((size_t)n2, (size_t)n2,
                                                          "cached GPUSOLVER dense eigenvectors");

        /* Preserve the vectors for the final post-SCF scatter, then drop the
           mapping onto the arena segment (the arena itself keeps the device
           memory reserved until the pre-Force release). */
#pragma acc update self(dense_evec[0 : evec_count])
        ClusterNonCol_ArenaUnmap(dense_evec);
        state->CachedGpuSolverDenseOnDevice = 0;
    }
}

/* Reuse the host eigenvectors for the final force EDM before the legacy
   scatter releases them. Otherwise a capacity solve would end with four
   serial CPU sums over every occupied state and every sparse matrix entry. */
int Cluster_DFT_NonCol_MgpuCachedEDM(ClusterNonColState *state, int n, int size_H1, int *MP, double *ko,
                                   double *****EDM, double *elapsed)
{
    const int n2 = 2*n;
    const size_t count = (size_t)size_H1*4;
    int myid, valid = 0;
    double start, end;
    *elapsed = 0.0;
    if (!state->mgpu_host_dense) return 0;
    MPI_Comm_rank(mpi_comm_level1,&myid);
    if (myid==Host_ID)
        valid = (state->CachedGpuSolverDenseEVec != NULL &&
                 state->CachedGpuSolverDenseN2 == n2 &&
                 !state->CachedGpuSolverDenseOnDevice);
    MPI_Bcast(&valid,1,MPI_INT,Host_ID,mpi_comm_level1);
    if (!valid) return 0;
    if (count>INT_MAX) ClusterNonCol_AbortWithMessage("Post-SCF EDM exceeds MPI count limit");
    dtime(&start);
    const int previous = openmx_mgpu_scope_enter(1);
    ClusterNonCol_RootDMWorkspace_Ensure(state, n2,count);
    ClusterNonColRootDMWorkspace *ws=&state->root_dm_workspace;
    if (myid==Host_ID) {
        memset(ws->dm_buffer,0,count*sizeof(double));
        ClusterNonCol_DMEntryCache_Ensure(state, MP,n);
        ClusterNonColDMEntryCache *cache=&state->dm_entry_cache;
        if (cache->entry_count!=size_H1) ClusterNonCol_AbortWithMessage("Post-SCF EDM entry count mismatch");
        const int occupied=ClusterNonCol_PrecomputeOccupations(state->chemical, 1,n2,ko,ws->occ,ws->occ_e);
        if (occupied>0) {
            dcomplex *pm=ClusterNonCol_MallocArray(state, (size_t)n2*n2,sizeof(dcomplex),"host force EDM");
            if (!OpenMX_MgpuDensityHost(n2,occupied,ws->occ_e,state->CachedGpuSolverDenseEVec,pm))
                ClusterNonCol_AbortWithMessage(openmx_mgpu_host_error());
#pragma omp parallel for
            for (int p=0;p<size_H1;++p) {
                size_t ia=cache->basis0[p],ib=cache->basis1[p];
                ws->dm_buffer[p]=pm[ia+ib*n2].r;
                ws->dm_buffer[(size_t)size_H1+p]=pm[ia+n+(ib+n)*n2].r;
                ws->dm_buffer[(size_t)size_H1*2+p]=pm[ia+(ib+n)*n2].r;
                ws->dm_buffer[(size_t)size_H1*3+p]=pm[ia+(ib+n)*n2].i;
            }
            openmx_core_native_free(state->resources,pm);
        }
    }
    MPI_Bcast(ws->dm_buffer,(int)count,MPI_DOUBLE,Host_ID,mpi_comm_level1);
    ClusterNonCol_StoreEDMBuffer(myid,size_H1,MP,EDM,ws->dm_buffer);
    openmx_mgpu_scope_leave(previous);
    dtime(&end);
    *elapsed=end-start;
    return 1;
}

double Cluster_DFT_NonCol_ScatterGpuSolverCachedEVec(ClusterNonColState *state, int n2, int *is2, int *ie2, dcomplex *EVec1)
{
    int myid, numprocs;
    int valid = 0;
    double elapsed = 0.0;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    if (myid == Host_ID) {
        valid = (state->CachedGpuSolverDenseEVec != NULL &&
                 state->CachedGpuSolverDenseN2 == n2);
    }
    MPI_Bcast(&valid, 1, MPI_INT, Host_ID, mpi_comm_level1);

    if (valid) {
        elapsed = ClusterNonCol_ScatterDenseEVecToLocal(state, myid, numprocs, n2, is2, ie2,
                                                        state->CachedGpuSolverDenseEVec,
                                                        state->CachedGpuSolverDenseOnDevice, EVec1);
    }

    ClusterNonCol_ReleaseGpuSolverCachedEVec(state, myid);

    return elapsed;
}

/* GPU GEMM wrappers (added by H.Kawai, ported from 3.9.9 GPU) */
static cublasHandle_t ClusterNonCol_CublasHandle(ClusterNonColState *state)
{
    /* One cached handle per selected device instead of a create/destroy pair
       around every GEMM (14 calls and a transpose per SCF iteration in the
       root dense path). */
    struct HandleCache { cublasHandle_t handle; int device; };
    struct HandleCache *cache=(struct HandleCache*)OpenMX_Resource_State(
        state->resources,"dense-gemm-handle",sizeof(struct HandleCache));
    int current_device = -1;

    wait_cudafunc(cudaGetDevice(&current_device));
    if (cache->handle != NULL && cache->device == current_device) {
        return cache->handle;
    }
    if (cache->handle != NULL) {
        OpenMXBlasWait(OpenMX_Resource_BlasDestroy(state->resources,cache->handle));
        cache->handle = NULL;
    }
    OpenMXBlasWait(OpenMX_Resource_BlasCreate(state->resources,&cache->handle));
    cache->device = current_device;
    return cache->handle;
}

static int ClusterNonCol_GemmWorkspaceTurnRelease(void)
{
    /* Releasing the GEMMul8 workspace once the diagonalization is done keeps
       it from occupying the shared device through the grid/force phases of
       the SCF step; opt out with OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE=0
       (falls back to the band-path knob for a single switch). */
    return openmx_env_flag_at(OPENMX_ENV_CLUSTER_GEMMUL8_TURN_RELEASE, openmx_env_flag_at(OPENMX_ENV_BAND_GEMMUL8_TURN_RELEASE, 1));
}

/* Persistent cuSOLVER state for the root dense zheevx.  The shared
   gpusolver_Syevdx_Complex_openacc helper creates a fresh handle, stream and
   multi-GiB device workspace on every call; at one call per SCF iteration
   that churn is pure overhead, so cache them across iterations instead. */

static void ClusterNonCol_GpuEigenCtx_Release(ClusterNonColState *state)
{
    ClusterNonColGpuEigenCtx *ctx = &state->gpu_eigen_ctx;

    if (!ctx->inited) return;

    if (ctx->d_work != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_work));
    if (ctx->d_info != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_info));
    openmx_core_native_free(state->resources,ctx->h_work);
    OpenMXSolverWait(OpenMX_Resource_SolverDestroy(state->resources,ctx->handle));
    wait_cudafunc(OpenMX_Resource_StreamDestroy(state->resources,ctx->stream));
    memset(ctx, 0, sizeof(*ctx));
}

static void ClusterNonCol_RefineRelease(ClusterNonColState *state);

/* Pre-Force release: the dense arena and the cached zheevdx workspace can
   hold many GiB of the shared device between SCF cycles; return them before
   Force() runs. */
void Cluster_DFT_NonCol_Release_GPU_Solver(ClusterNonColState *state)
{
    ClusterNonCol_DenseArena_Release(state);
    ClusterNonCol_GpuEigenCtx_Release(state);
    ClusterNonCol_RefineRelease(state);
}

static ClusterNonColGpuEigenCtx *ClusterNonCol_GpuEigenCtx_Get(ClusterNonColState *state)
{
    ClusterNonColGpuEigenCtx *ctx = &state->gpu_eigen_ctx;
    int current_device = -1;

    wait_cudafunc(cudaGetDevice(&current_device));
    if (ctx->inited && ctx->device != current_device) {
        ClusterNonCol_GpuEigenCtx_Release(state);
    }
    if (!ctx->inited) {
        OpenMXSolverWait(OpenMX_Resource_SolverCreate(state->resources,&ctx->handle));
        wait_cudafunc(OpenMX_Resource_StreamCreate(state->resources,&ctx->stream,cudaStreamNonBlocking));
        OpenMXSolverWait(cusolverDnSetStream(ctx->handle, ctx->stream));
        wait_cudafunc(OpenMX_Resource_GpuMalloc(state->resources,(void **)&ctx->d_info,sizeof(int32_t)));
        ctx->device = current_device;
        ctx->inited = 1;
    }
    return ctx;
}

/* Ensure the cached zheevdx workspaces cover an (n, maxn) solve.  A_dev and
   W_dev are DEVICE pointers.  Returns 0 (with *need_bytes_out set) when the
   device workspace cannot be allocated, leaving the context consistent; the
   GPU-diag preflight uses this to reserve the full-range workspace up front
   at the size cusolver actually reports. */
static int ClusterNonCol_EigenWorkspace_TryEnsure(ClusterNonColState *state, void *A_dev, double *W_dev, int n, int maxn,
                                                  size_t *need_bytes_out)
{
    ClusterNonColGpuEigenCtx *ctx = ClusterNonCol_GpuEigenCtx_Get(state);
    cusolverEigMode_t const   jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t const    uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t const  range = (n == maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;
    double vl = 0.0, vu = 0.0;
    int64_t h_meig = 0;
    size_t d_bytes = 0;
    size_t h_bytes = 0;

    if (need_bytes_out != NULL) *need_bytes_out = 0;
    if (ctx->last_n == n && ctx->last_maxn == maxn) return 1;

    OpenMXSolverWait(cusolverDnXsyevdx_bufferSize(ctx->handle, NULL, jobz, range, uplo, n, CUDA_C_64F, A_dev, n,
                                               &vl, &vu, 1L, (int64_t)maxn, &h_meig, CUDA_R_64F, W_dev,
                                               CUDA_C_64F, &d_bytes, &h_bytes));

    if (ctx->d_work_bytes < d_bytes) {
        if (ctx->d_work != NULL) wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,ctx->d_work));
        ctx->d_work = NULL;
        ctx->d_work_bytes = 0;
        if (0 < d_bytes && ClusterNonCol_TryDeviceMalloc(state, &ctx->d_work, d_bytes) != cudaSuccess) {
            if (need_bytes_out != NULL) *need_bytes_out = d_bytes;
            return 0;
        }
        ctx->d_work_bytes = d_bytes;
    }
    if (ctx->h_work_bytes < h_bytes) {
        openmx_core_native_free(state->resources,ctx->h_work);
        ctx->h_work = ClusterNonCol_MallocArray(state, h_bytes, 1u, "cuSOLVER host workspace");
        ctx->h_work_bytes = h_bytes;
    }
    ctx->last_n = n;
    ctx->last_maxn = maxn;
    return 1;
}

/* In-place zheevx on OpenACC-present data: the eigenvectors land in the first
   maxn columns of A and ko[1..maxn] receives the eigenvalues shifted by one,
   exactly like EigenBand_lapack_openacc.  Workspace sizes are re-queried only
   when the problem size changes. */
static void ClusterNonCol_ZheevdxPresent(ClusterNonColState *state, dcomplex *A, double *W, int n, int maxn)
{
    ClusterNonColGpuEigenCtx *ctx = ClusterNonCol_GpuEigenCtx_Get(state);
    cusolverEigMode_t const   jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t const    uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t const  range = (n == maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;
    double vl = 0.0, vu = 0.0;
    int64_t h_meig = 0;
    int32_t info = 0;

#pragma acc data      present(A[0 : (size_t)n * (size_t)n], W[0 : n + 1])
#pragma acc host_data use_device(A, W)
    {
        size_t need_bytes = 0;

        if (!ClusterNonCol_EigenWorkspace_TryEnsure(state, A, W, n, maxn, &need_bytes)) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "Cluster_DFT_NonCol.c: out of GPU memory for the %.1f MiB eigensolver workspace; "
                     "reduce the MPI ranks sharing the device.",
                     (double)need_bytes / (1024.0 * 1024.0));
            ClusterNonCol_AbortWithMessage(msg);
        }

        OpenMXSolverWait(cusolverDnXsyevdx(ctx->handle, NULL, jobz, range, uplo, n, CUDA_C_64F, A, n, &vl, &vu, 1L,
                                        (int64_t)maxn, &h_meig, CUDA_R_64F, W, CUDA_C_64F, ctx->d_work,
                                        ctx->d_work_bytes, ctx->h_work, ctx->h_work_bytes, ctx->d_info));

        wait_cudafunc(cudaMemcpyAsync(&info, ctx->d_info, sizeof(int32_t), cudaMemcpyDeviceToHost, ctx->stream));
        wait_cudafunc(cudaStreamSynchronize(ctx->stream));
    }

    if (info != 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "cusolverDnXsyevdx failed in Cluster_DFT_NonCol.c: info=%d", (int)info);
        ClusterNonCol_AbortWithMessage(msg);
    }

#pragma acc data present(W[0 : n + 1])
    {
#pragma acc kernels
#pragma acc loop seq
        for (int i = n; i >= 1; i--) {
            W[i] = W[i - 1];
        }
    }
}

/* The refined FP32 eigensolver (eigen_refine_gpu.c) in the root dense path.

   OPENMX_EIGEN_REFINE=K (0: off): every step of an SCF cycle solves the
   Hamiltonian eigenproblem for all 2n pairs in FP32 (or starts from the
   previous step's vectors) and refines the MaxN wanted ones by K steps.
   Unset, K = 2 on GPUs whose FP64 throughput is a small fraction of their
   FP32 one (cudaDevAttrSingleToDoublePrecisionPerfRatio of at least 8:
   GeForce, RTX PRO 6000 Blackwell and the like report 32 or 64, A100 and
   H100 2), as long as the complex GEMMul8 products are on; the root rank's
   device decides for all ranks.  On an RTX 5080 with 18 ranks,
   sidia333_nc_cluster (2n = 5616) took 197.9 s instead of 247.4 s before
   the warm start, the eigensolver 847 ms instead of 3183 ms per step, with
   the same SCF steps and Utot 1e-12 Ha off.

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
   products fell back to plain cuBLAS FP64 ends for the rest of the run.
   Orbital optimization (Cnt_switch) keeps the FP64 eigensolver. */

/* every rank, once, before the first root dense solve: OPENMX_EIGEN_REFINE,
   or the default the root rank's device implies, for all ranks alike */
static void ClusterNonCol_RefineConfigure(ClusterNonColState *state)
{
    ClusterNonColRefineState *st = &state->refine;
    int myid, config[2] = {0, 0};   /* iterations, FP32:FP64 ratio when defaulted on */

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

/* the steps of the refinement, 0 when it is off or not configured yet */
static int ClusterNonCol_RefineIterations(ClusterNonColState *state)
{
    return (openmx_fp32_enabled() && 0 < state->refine.iterations) ? state->refine.iterations : 0;
}

static void ClusterNonCol_RefineRelease(ClusterNonColState *state)
{
    openmx_eigen_refine_state_release(&state->refine_state);
    openmx_eigen_refine_device_release(&state->refine_dev);
}

/* every rank, before each root dense solve: a new SCF cycle starts refined.
   The last step scf.maxIter allows is always FP64, as the energy density
   matrix after the SCF comes from it whether the SCF converged or not.
   Orbital optimization (Cnt_switch) restarts the step count within a cycle
   and has stop rules of its own, so it keeps the FP64 eigensolver. */
static void ClusterNonCol_RefineBeginSolve(ClusterNonColState *state, int SCF_iter)
{
    ClusterNonColRefineState *st = &state->refine;

    ClusterNonCol_RefineConfigure(state);
    if (SCF_iter == 1) {
        st->final_stage = 0;
        st->cycle_refined = 0;
        state->refine_state.basis_valid = 0;   /* a new cycle may come with a new geometry */
    }
    st->last_refined = (0 < ClusterNonCol_RefineIterations(state) && !st->final_stage && Cnt_switch == 0 &&
                        SCF_iter < DFTSCF_loop);
}

/* DFT.c: whether the latest eigensolve was a refined FP32 one (the same on
   every rank).  An SCF that ends on such a solve, by its stop condition or
   by scf.maxIter, has the same Hamiltonian solved once more in FP64 (mode
   "fp64" of Cluster_DFT_NonCol, one eigensolve without the density matrix)
   so that the vectors kept for the post-SCF energy density matrix are FP64
   ones; the density matrix of the step, refined to 1e-11, stands. */
int Cluster_DFT_NonCol_RefineLastRefined(ClusterNonColState *state)
{
    return state->refine.last_refined;
}

/* DFT.c, after the mixing of an SCF step that goes on: normrd is the printed
   NormRD */
void Cluster_DFT_NonCol_RefineAfterMixing(ClusterNonColState *state, double normrd)
{
    ClusterNonColRefineState *st = &state->refine;
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

/* The refined solve of A (n x n complex, present): see eigen_refine_gpu.h
   for the buffers (fp32, region, x) and the warm start; on success the
   refined vectors are in the first maxn columns of A and W[1..maxn] holds
   the refined eigenvalues, as in ClusterNonCol_ZheevdxPresent.  Returns 1,
   0 (A still holds the Hermitian matrix: the caller solves it in FP64,
   *persistent says for how long), or -1 when the warm start found the
   previous vectors too far from A (nothing changed). */
static cudaError_t ClusterNonCol_RefineDeviceMalloc(void *context, void **ptr, size_t bytes)
{
    return ClusterNonCol_TryDeviceMalloc((ClusterNonColState *)context, ptr, bytes);
}
static void *ClusterNonCol_RefineHostAllocate(void *context, size_t count, size_t bytes, const char *label)
{
    return ClusterNonCol_MallocArray((ClusterNonColState *)context, count, bytes, label);
}

static int ClusterNonCol_RefinedEigenPresent(ClusterNonColState *state, dcomplex *A, double *W, int n, int maxn, int iterations, void *fp32,
                                             size_t region, dcomplex *x, int transient, int warm, int *persistent)
{
    ClusterNonColGpuEigenCtx *ctx = ClusterNonCol_GpuEigenCtx_Get(state);
    EigenRefineDevice *dev = &state->refine_dev;
    EigenRefineProblem pb;
    EigenRefineReport rep;
    ClusterNonColOccupationCtx occ = {0.0};
    double *e0 = (double *)ClusterNonCol_MallocArray(state, (size_t)n, sizeof(double), "refinement eigenvalues");
    double *f0 = (double *)ClusterNonCol_MallocArray(state, (size_t)n, sizeof(double), "refinement occupations");
    int rc;

    dev->cublas = ClusterNonCol_CublasHandle(state);
    dev->cusolver = ctx->handle;
    dev->stream = ctx->stream;
    dev->d_work = &ctx->d_work;
    dev->d_work_bytes = &ctx->d_work_bytes;
    dev->h_work = &ctx->h_work;
    dev->h_work_bytes = &ctx->h_work_bytes;
    dev->d_info = ctx->d_info;
    dev->allocation_context = state;
    dev->try_malloc_owned = ClusterNonCol_RefineDeviceMalloc;
    dev->resources = state->resources;
    dev->host_malloc_owned = ClusterNonCol_RefineHostAllocate;

    memset(&pb, 0, sizeof(pb));
    pb.cplx = 1;
    pb.n = n;
    pb.maxn = maxn;
    pb.iterations = iterations;
    pb.a = acc_deviceptr(A);
    pb.w = (double *)acc_deviceptr(W) + 1;
    pb.fp32 = fp32;
    pb.region = region;
    pb.x = x;
    pb.transient = transient;
    pb.warm = warm;
    pb.defaulted = state->refine.defaulted;
    pb.occupation = ClusterNonCol_RefineFermi;
    pb.occupation_ctx = &occ;

    *persistent = 0;
    rc = openmx_eigen_refine_prepare(&state->refine_state, dev, &pb, e0, &rep);
    if (rc == 1) {
        ClusterNonCol_RefineOccupations(e0, n, maxn, f0, &occ.mu);
        rc = openmx_eigen_refine_finish(&state->refine_state, dev, &pb, f0, &rep);
    }
    openmx_core_native_free(state->resources,e0);
    openmx_core_native_free(state->resources,f0);
    *persistent = rep.persistent;
    if (rc == 1 && 3 <= level_stdout) {
        printf("<Cluster_DFT_NonCol> refinement step %d of %d columns: max |r| %.1e, max |s| %.1e, delta %.1e; "
               "Rayleigh-Ritz in %d partially occupied clusters (largest %d, chained within %.1e)\n",
               iterations, rep.columns, rep.max_r, rep.max_s, rep.delta, rep.rr_clusters, rep.rr_largest, rep.chain);
        fflush(stdout);
    }
    return rc;
}

static void ClusterNonCol_GEMMul8Dgemm_OpenACC(ClusterNonColState *state, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                               int k, double const * A, double const * B, double * C)
{
    cublasHandle_t handle = ClusterNonCol_CublasHandle(state);

#pragma acc data      present(A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        double const alpha = 1.0;
        double const beta  = 0.0;

        OpenMXBlasWait(openmx_gemmul8Dgemm(handle, transa, transb, m, n, k, &alpha, A, m, B, k, &beta, C, m));
    }
}

static void ClusterNonCol_GEMMul8Zgemm_OpenACC(ClusterNonColState *state, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                               int k, dcomplex const * A, dcomplex const * B, dcomplex * C)
{
    cublasHandle_t handle = ClusterNonCol_CublasHandle(state);

#pragma acc data      present(A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        cuDoubleComplex const alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex const beta  = make_cuDoubleComplex(0.0, 0.0);

        OpenMXBlasWait(openmx_gemmul8Zgemm(handle, transa, transb, m, n, k, &alpha, (cuDoubleComplex const *)A, m,
                                          (cuDoubleComplex const *)B, k, &beta, (cuDoubleComplex *)C, m));
    }
}

/* OPENMX_CLUSTER_PROFILE=1 (as in Cluster_DFT_Col.c): cumulative wall time of
   the phases of the root dense solve on its owner rank, one NCCLUSTERPROF
   line per solve: the transformed overlap (rebuilt at SCF_iter 1 and after a
   size change), the gather of the Hamiltonian blocks, the forward transform
   with the assembly of Hs2, the eigensolver, the back transform, the
   release (eigenvalue download, remapping and the release of the GEMMul8
   workspace) and the density matrices.  The device is synchronized around
   each phase, so the times are not those of an unprofiled run. */

static int ClusterNonCol_ProfileEnabled(void)
{
    return openmx_env_flag_at(OPENMX_ENV_CLUSTER_PROFILE, 0);
}

/* charges the time since *t0 to phase (none if negative) after synchronizing
   the device, and restarts *t0 */
static void ClusterNonCol_ProfileLap(ClusterNonColState *state, int phase, double *t0)
{
    double now;

    wait_cudafunc(cudaDeviceSynchronize());
    now = MPI_Wtime();
    if (0 <= phase) state->prof[phase] += now - *t0;
    *t0 = now;
}

static void ClusterNonCol_ProfileReport(ClusterNonColState *state, int myid, int n2, int MaxN, double dm_seconds)
{
    if (myid != Host_ID || !ClusterNonCol_ProfileEnabled()) return;

    state->prof[NCPROF_DM] += dm_seconds;
    state->prof_solves++;
    printf("NCCLUSTERPROF rank=%d solves=%ld n2=%d maxn=%d overlap=%.6f gather=%.6f forward=%.6f eigen=%.6f "
           "back=%.6f release=%.6f dm=%.6f (s, cumulative)\n",
           myid, state->prof_solves, n2, MaxN, state->prof[NCPROF_OVERLAP],
           state->prof[NCPROF_GATHER], state->prof[NCPROF_FORWARD], state->prof[NCPROF_EIGEN],
           state->prof[NCPROF_BACK], state->prof[NCPROF_RELEASE], state->prof[NCPROF_DM]);
    fflush(stdout);
}

/* OPENMX_GEMM_SAMPLE_DIR=<existing directory>, as in Cluster_DFT_Col.c: writes
   the transformed Hamiltonian the eigensolver receives, Hs2 (n2 x n2 complex,
   column-major, real and imaginary parts interleaved), for the SCF steps
   listed in OPENMX_GEMM_SAMPLE_ITERS (comma separated, default "1"), as
   Hs2_scf<step>.bin.  The files feed the offline eigensolver precision study
   tests/eigen_precision_probe.cu.  Returns 1 when it wrote a file. */
static int ClusterNonCol_SampleHamiltonian(int SCF_iter, dcomplex *Hs2, int n2, int MaxN)
{
    const char *dir = openmx_env_at(OPENMX_ENV_GEMM_SAMPLE_DIR);
    const char *list = openmx_env_at(OPENMX_ENV_GEMM_SAMPLE_ITERS);
    size_t count = (size_t)n2 * (size_t)n2;
    char path[4096];
    int wanted = 0;
    FILE *fp;

    if (dir == NULL || dir[0] == '\0') return 0;
    if (list == NULL || list[0] == '\0') list = "1";

    while (*list != '\0' && !wanted) {
        char *end;
        long step = strtol(list, &end, 10);
        if (end == list) break;
        wanted = (step == SCF_iter);
        list = (*end == ',') ? end + 1 : end;
    }
    if (!wanted) return 0;

#pragma acc update self(Hs2[0 : count]) if_present
    snprintf(path, sizeof(path), "%s/Hs2_scf%03d.bin", dir, SCF_iter);
    fp = fopen(path, "wb");
    if (fp == NULL || fwrite(Hs2, sizeof(dcomplex), count, fp) != count || fclose(fp) != 0) {
        char msg[4200];
        snprintf(msg, sizeof(msg), "Cluster_DFT_NonCol.c: could not write the sampled matrix %s", path);
        ClusterNonCol_AbortWithMessage(msg);
    }
    printf("<Cluster_DFT_NonCol> sampled %s (n2=%d, MaxN=%d)\n", path, n2, MaxN);
    fflush(stdout);
    return 1;
}

/* Back transform of the root dense solve, on device pointers.  On entry the
   n2 x n2 segment zv holds the eigenvectors of the transformed problem in
   its first MaxN columns; on return it holds dense_evec, state l in the
   original basis on row l,

       dense_evec[l + i n2] = sum_j zv[j + l n2] Ss2[i + j n2],   l < MaxN,

   and zeros on the rows from MaxN on.  Ss2 = diag(S, S) carries the real
   transformed overlap S (n x n) on both spin blocks, so a complex GEMM
   against the n2 x n2 Ss2 does four times the work of the product.  Here
   the vectors are transposed into vt (MaxN x n2); the real view of vt
   (2 MaxN x n2, the real and imaginary parts of a state on consecutive
   rows) times S^T is the real view of a spin half of dense_evec, one real
   GEMM per half.  Being real GEMMs, they follow the GEMMul8 settings of the
   real products (OPENMX_GEMMUL8_NUM_MOD_D, _FASTMODE_D, _DISABLE_D), like
   the forward transform; the complex settings (_Z) no longer apply here. */
static void ClusterNonCol_BackTransform(ClusterNonColState *state, dcomplex *zv, dcomplex *vt, double const *S, int n, int n2, int MaxN)
{
    cublasHandle_t handle = ClusterNonCol_CublasHandle(state);
    cuDoubleComplex const one  = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex const zero = make_cuDoubleComplex(0.0, 0.0);
    double const alpha = 1.0;
    double const beta  = 0.0;

    /* vt = zv(:, 0:MaxN)^T; the second operand (zv again, scaled by 0) only
       completes the call */
    if (!openmx_gpu_transform(OPENMX_GPU_TRANSPOSE,1,MaxN,n2,zv,n2,NULL,0,vt,MaxN,0,0.0))
    OpenMXBlasWait(cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, MaxN, n2, &one, (cuDoubleComplex const *)zv, n2,
                              &zero, (cuDoubleComplex const *)zv, n2, (cuDoubleComplex *)vt, MaxN));

    if (MaxN < n2) {
        /* the rows of the unsolved states are never computed; clear them so
           no consumer ever sees stale values */
if (!openmx_gpu_transform(OPENMX_GPU_ZERO,1,n2-MaxN,n2,NULL,0,NULL,0,(dcomplex*)zv+MaxN,n2,0,0.0)) {
#pragma acc parallel loop collapse(2) deviceptr(zv)
        for (int col = 0; col < n2; col++) {
            for (int row = MaxN; row < n2; row++) {
                zv[(size_t)row + (size_t)col * (size_t)n2].r = 0.0;
                zv[(size_t)row + (size_t)col * (size_t)n2].i = 0.0;
            }
        }
}

    }

    for (int half = 0; half < 2; half++) {
        size_t const col = (size_t)half * (size_t)n;

        OpenMXBlasWait(openmx_gemmul8Dgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2 * MaxN, n, n, &alpha,
                                          (double const *)vt + 2 * (size_t)MaxN * col, 2 * MaxN, S, n, &beta,
                                          (double *)zv + 2 * (size_t)n2 * col, 2 * n2));
    }
}

static void ClusterNonCol_GpuSolver_DenseDsyevx(double *A, double *Z, double *ko, int n, int maxn,
                                               const char *where)
{
    int info,l,copy_cols;

    info = gpusolver_Syevdx(A, ko, n, maxn);
    if (info!=0){
        fprintf(stderr,"%s: gpusolver_Syevdx failed, info=%d\n",where,info);
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
    int info,l,copy_cols;

    info = gpusolver_Syevdx_Complex(A, ko, n, maxn);
    if (info!=0){
        fprintf(stderr,"%s: gpusolver_Syevdx_Complex failed, info=%d\n",where,info);
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

static void ClusterNonCol_BuildDenseFromGathered(ClusterNonColState *state, const double *H1, const int *order_GA, int *MP, double *H,
                                                 int n, int tnum)
{
    int AN, GA_AN, LB_AN, GB_AN;
    int wanA, wanB, tnoA, tnoB;
    size_t *atom_offsets;
    size_t dense_count;
    size_t offset;

    atom_offsets = (size_t *)ClusterNonCol_MallocArray(state, (size_t)atomnum + 2u, sizeof(size_t), "atom offsets");

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

    openmx_core_native_free(state->resources,atom_offsets);
}

static void Patch2Full_Cluster_NonCol_Owner(ClusterNonColState *state, double ****RH, double *H, int *MP, int owns_dense, int n)
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
    local_H1 = (double *)ClusterNonCol_MallocArray(state, (size_t)local_nzeros, sizeof(double), "local H1 segment");
    ClusterNonCol_PackLocalSegment(RH, local_H1, local_nzeros);

    if (myid == dense_root) {
        recv_nzeros = (int *)ClusterNonCol_MallocArray(state, (size_t)numprocs, sizeof(int), "recv_nzeros");
        recv_matomnum = (int *)ClusterNonCol_MallocArray(state, (size_t)numprocs, sizeof(int), "recv_matomnum");
    }

    MPI_Gather(&local_nzeros, 1, MPI_INT, recv_nzeros, 1, MPI_INT, dense_root, mpi_comm_level1);
    MPI_Gather(&local_matomnum, 1, MPI_INT, recv_matomnum, 1, MPI_INT, dense_root, mpi_comm_level1);

    if (myid == dense_root) {
        size_t total_nzeros = 0;
        size_t total_atoms = 0;

        h1_displs = (int *)ClusterNonCol_MallocArray(state, (size_t)numprocs, sizeof(int), "h1_displs");
        atom_displs = (int *)ClusterNonCol_MallocArray(state, (size_t)numprocs, sizeof(int), "atom_displs");

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

        gathered_H1 = (double *)ClusterNonCol_MallocArray(state, total_nzeros, sizeof(double), "gathered H1");
        order_GA = (int *)ClusterNonCol_MallocArray(state, (size_t)atomnum + 1u, sizeof(int), "order_GA");
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

        ClusterNonCol_BuildDenseFromGathered(state, gathered_H1, order_GA, MP, H, n, gathered_nzeros);
    }

    openmx_core_native_free(state->resources,gathered_H1);
    openmx_core_native_free(state->resources,order_GA);
    openmx_core_native_free(state->resources,atom_displs);
    openmx_core_native_free(state->resources,h1_displs);
    openmx_core_native_free(state->resources,recv_matomnum);
    openmx_core_native_free(state->resources,recv_nzeros);
    openmx_core_native_free(state->resources,local_H1);
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
            /* Match the host assembler: atom A supplies the row in column-major storage. */
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

static unsigned long long ClusterNonCol_DenseIndexFingerprint(const int *order_GA, int *MP, int n)
{
    unsigned long long h = 1469598103934665603ULL;

#define CLUSTERNONCOL_HASH_INT(v) \
    do { \
        h ^= (unsigned long long)(unsigned int)(v); \
        h *= 1099511628211ULL; \
    } while (0)

    CLUSTERNONCOL_HASH_INT(n);
    CLUSTERNONCOL_HASH_INT(atomnum);
    for (int AN = 1; AN <= atomnum; AN++) {
        int GA_AN = order_GA[AN];

        CLUSTERNONCOL_HASH_INT(GA_AN);
        CLUSTERNONCOL_HASH_INT(MP[GA_AN]);
        CLUSTERNONCOL_HASH_INT(WhatSpecies[GA_AN]);
        CLUSTERNONCOL_HASH_INT(Spe_Total_CNO[WhatSpecies[GA_AN]]);
        CLUSTERNONCOL_HASH_INT(FNAN[GA_AN]);
        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            CLUSTERNONCOL_HASH_INT(natn[GA_AN][LB_AN]);
        }
    }

#undef CLUSTERNONCOL_HASH_INT

    return h;
}

/* The sparse-to-dense index table only depends on the geometry, so build it
   once on the host and keep a device copy in the arena instead of rebuilding
   and re-uploading it for each of the eight dense loads of every SCF
   iteration.  Returns the DEVICE copy (the o_dindex arena segment).  Only the
   dense-owner rank calls this. */
static const int *ClusterNonCol_DenseIndexDevice_Get(ClusterNonColState *state, const int *order_GA, int *MP, int n, int tnum)
{
    ClusterNonColDenseIndexCache *cache = &state->dense_index_cache;
    ClusterNonColDenseArena *arena = &state->dense_arena;
    unsigned long long fingerprint = ClusterNonCol_DenseIndexFingerprint(order_GA, MP, n);

    if (!arena->valid || arena->tnum_capacity < (size_t)tnum) {
        ClusterNonCol_AbortWithMessage("The dense arena does not cover the sparse index table in Cluster_DFT_NonCol.c.");
    }

    if (!(cache->valid && cache->n == n && cache->tnum == tnum && cache->fingerprint == fingerprint)) {
        openmx_core_native_free(state->resources,cache->dense_index);
        memset(cache, 0, sizeof(*cache));

        cache->dense_index = (int *)ClusterNonCol_MallocArray(state, (size_t)tnum, sizeof(int), "dense index cache");
        ClusterNonCol_BuildDenseIndexFromGathered(order_GA, MP, n, tnum, cache->dense_index);

        cache->valid = 1;
        cache->n = n;
        cache->tnum = tnum;
        cache->fingerprint = fingerprint;
    }

    if (!cache->device_valid) {
        acc_memcpy_to_device(ClusterNonCol_ArenaPtr(state, arena->o_dindex), cache->dense_index,
                             (size_t)tnum * sizeof(int));
        cache->device_valid = 1;
    }
    return (const int *)ClusterNonCol_ArenaPtr(state, arena->o_dindex);
}

/* Stage the gathered sparse matrix through the arena and scatter it into the
   present dense matrix H.  H1_host is a HOST buffer; the upload goes through
   the o_stage arena segment, so no device allocation happens here (the old
   per-call enter-data copyin aborted fatally on a full device). */
static void ClusterNonCol_BuildDeviceDenseFromGathered(ClusterNonColState *state, const double *H1_host, const int *dense_index_dev,
                                                       int tnum, int n, double *H)
{
    ClusterNonColDenseArena *arena = &state->dense_arena;
    size_t nn = ClusterNonCol_CheckedMulCount((size_t)n, (size_t)n, "device dense H");

    if (!arena->valid || arena->tnum_capacity < (size_t)tnum) {
        ClusterNonCol_AbortWithMessage("The dense arena does not cover the gathered matrix in Cluster_DFT_NonCol.c.");
    }
    acc_memcpy_to_device(ClusterNonCol_ArenaPtr(state, arena->o_stage), (void *)H1_host,
                         (size_t)tnum * sizeof(double));
    if (openmx_gpu_scatter(n,tnum,0,dense_index_dev,tnum,ClusterNonCol_ArenaPtr(state, arena->o_stage),0,NULL,NULL,0,acc_deviceptr((void*)(H)))) return;

if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(H)),n,0,0.0)) {
#pragma acc parallel loop present(H[0 : nn])
    for (size_t p = 0; p < nn; p++) {
        H[p] = 0.0;
    }
}

    {
        const double *H1 = (const double *)ClusterNonCol_ArenaPtr(state, arena->o_stage);
        const int *dense_index = dense_index_dev;

#pragma acc parallel loop present(H[0 : nn]) deviceptr(H1, dense_index)
        for (int p = 0; p < tnum; p++) {
            int idx = dense_index[p];
#pragma acc atomic update
            H[idx] += H1[p];
        }
    }
}

static void ClusterNonCol_LoadDeviceDenseFromSetHamCache(ClusterNonColState *state, const double *H1, int *MP, int n, double *H)
{
    int tnum = Set_Hamiltonian_GpuSolver_Packed_Size();
    int *order_GA = Set_Hamiltonian_GpuSolver_Packed_OrderGA();
    const int *dense_index_dev;

    if (H1 == NULL || order_GA == NULL) {
        ClusterNonCol_AbortWithMessage("Set_Hamiltonian packed matrix cache is missing in Cluster_DFT_NonCol.c.");
    }

    dense_index_dev = ClusterNonCol_DenseIndexDevice_Get(state, order_GA, MP, n, tnum);
    ClusterNonCol_BuildDeviceDenseFromGathered(state, H1, dense_index_dev, tnum, n, H);
}

static void Patch2Device_Cluster_NonCol_Owner(ClusterNonColState *state, double ****RH, double *H, int *MP, int owns_dense, int n)
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
    const int *dense_index = NULL;
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
    local_H1 = (double *)ClusterNonCol_MallocArray(state, (size_t)local_nzeros, sizeof(double), "local H1 segment");
    ClusterNonCol_PackLocalSegment(RH, local_H1, local_nzeros);

    if (myid == dense_root) {
        recv_nzeros = (int *)ClusterNonCol_MallocArray(state, (size_t)numprocs, sizeof(int), "recv_nzeros");
        recv_matomnum = (int *)ClusterNonCol_MallocArray(state, (size_t)numprocs, sizeof(int), "recv_matomnum");
    }

    MPI_Gather(&local_nzeros, 1, MPI_INT, recv_nzeros, 1, MPI_INT, dense_root, mpi_comm_level1);
    MPI_Gather(&local_matomnum, 1, MPI_INT, recv_matomnum, 1, MPI_INT, dense_root, mpi_comm_level1);

    if (myid == dense_root) {
        size_t total_nzeros = 0;
        size_t total_atoms = 0;

        h1_displs = (int *)ClusterNonCol_MallocArray(state, (size_t)numprocs, sizeof(int), "h1_displs");
        atom_displs = (int *)ClusterNonCol_MallocArray(state, (size_t)numprocs, sizeof(int), "atom_displs");

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

        gathered_H1 = (double *)ClusterNonCol_MallocArray(state, total_nzeros, sizeof(double), "gathered H1");
        order_GA = (int *)ClusterNonCol_MallocArray(state, (size_t)atomnum + 1u, sizeof(int), "order_GA");
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

        dense_index = ClusterNonCol_DenseIndexDevice_Get(state, order_GA, MP, n, gathered_nzeros);
        ClusterNonCol_BuildDeviceDenseFromGathered(state, gathered_H1, dense_index, gathered_nzeros, n, H);
    }

    openmx_core_native_free(state->resources,gathered_H1);
    openmx_core_native_free(state->resources,order_GA);
    openmx_core_native_free(state->resources,atom_displs);
    openmx_core_native_free(state->resources,h1_displs);
    openmx_core_native_free(state->resources,recv_matomnum);
    openmx_core_native_free(state->resources,recv_nzeros);
    openmx_core_native_free(state->resources,local_H1);
}

static size_t ClusterNonCol_GpuDiagReserveBytes(void)
{
    return openmx_env_mib_bytes_at(OPENMX_ENV_CLUSTER_GPU_DIAG_RESERVE_MB, 1024);
}

/* Lay out and reserve the dense arena for an (n, n2) solve whose gathered
   sparse matrices hold at most tnum elements.  Keeps an already-sufficient
   arena.  Returns 0 when the device cannot hold the reservation. */
static int ClusterNonCol_DenseArena_TryEnsure(ClusterNonColState *state, int n, int n2, size_t tnum)
{
    ClusterNonColDenseArena *arena = &state->dense_arena;
    size_t nn8 = ClusterNonCol_CheckedMulCount(ClusterNonCol_CheckedMulCount((size_t)n, (size_t)n, "dense arena S"),
                                               sizeof(double), "dense arena S bytes");
    size_t n2n2z = ClusterNonCol_CheckedMulCount(ClusterNonCol_CheckedMulCount((size_t)n2, (size_t)n2,
                                                                               "dense arena NC matrix"),
                                                 sizeof(dcomplex), "dense arena NC matrix bytes");
    size_t ko8 = (size_t)(n2 + 2) * sizeof(double);
    size_t pos = 0;

    if (arena->valid && arena->n == n && arena->n2 == n2 && tnum <= arena->tnum_capacity) return 1;

    ClusterNonCol_DenseArena_Release(state);

    arena->o_S   = ClusterNonCol_ArenaOff(&pos, nn8);
    arena->o_ko  = ClusterNonCol_ArenaOff(&pos, ko8);
    for (int i = 0; i < 7; i++) {
        arena->o_r[i] = ClusterNonCol_ArenaOff(&pos, nn8);
    }
    arena->o_Hs2    = ClusterNonCol_ArenaOff(&pos, n2n2z);
    arena->o_vt     = ClusterNonCol_ArenaOff(&pos, n2n2z);
    arena->o_dindex = ClusterNonCol_ArenaOff(&pos, ClusterNonCol_CheckedMulCount(tnum, sizeof(int),
                                                                                 "dense arena index"));
    arena->o_stage  = ClusterNonCol_ArenaOff(&pos, ClusterNonCol_CheckedMulCount(tnum, sizeof(double),
                                                                                 "dense arena stage"));

    if (ClusterNonCol_TryDeviceMalloc(state, (void **)&arena->base, pos) != cudaSuccess) {
        memset(arena, 0, sizeof(*arena));
        return 0;
    }
    arena->valid = 1;
    arena->n = n;
    arena->n2 = n2;
    arena->tnum_capacity = tnum;
    arena->total_bytes = pos;
    return 1;
}

static void ClusterNonCol_DenseArena_Release(ClusterNonColState *state)
{
    ClusterNonColDenseArena *arena = &state->dense_arena;
    ClusterNonColDenseSCache *scache = &state->dense_scache;

    if (!arena->valid) return;

    /* the stashed eigenvectors may still be mapped into the arena; preserve
       their host copy for the post-SCF scatter before the memory goes away */
    Cluster_DFT_NonCol_DemoteGpuSolverCachedEVec(state);

    if (scache->s_on_device) {
        ClusterNonCol_ArenaUnmap(scache->S);
        scache->s_on_device = 0;
    }
    scache->transformed_s_valid = 0;

    state->dense_index_cache.device_valid = 0;
    state->refine_state.basis_valid = 0;

    wait_cudafunc(OpenMX_Resource_GpuFree(state->resources,arena->base));
    memset(arena, 0, sizeof(*arena));
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
    long parsed = openmx_env_long_at(OPENMX_ENV_CLUSTER_NONCOL_GPU_SWITCH_NUM, CLUSTER_NONCOL_GPU_CPU_SWITCH_NUM);

    return 0 < parsed && parsed <= INT_MAX ? (int)parsed : CLUSTER_NONCOL_GPU_CPU_SWITCH_NUM;
}

/* Collectively decide whether the root dense GPU diagonalization fits on the
   device (mirrors ClusterCol_GpuDiagFits).  The dense owner RESERVES the
   whole device footprint of the solve up front — one arena every device
   buffer of the dense path is mapped into, plus the zheevdx workspace at the
   size cusolverDnXsyevdx_bufferSize() actually reports — so a positive
   verdict cannot be invalidated later in the SCF cycle.  On top of the
   reservation the transient workspace of the SCF_iter 1 overlap
   diagonalization and a configurable margin must still be free.  The verdict
   is agreed on over mpi_comm_level1 before selecting native or host-backed
   MAGMA storage; it is cached until the next SCF restart or a size change.
   A capacity failure in both paths terminates collectively.
   OPENMX_CLUSTER_GPU_DIAG=0 explicitly selects the legacy CPU branch;
   neither =1 nor forced MAGMA bypasses memory admission. */
static int ClusterNonCol_GpuDiagFits(ClusterNonColState *state, int SCF_iter, int n, int n2, int myid)
{

    ClusterNonColDenseArena *arena = &state->dense_arena;
    long long my_nzeros = 0;
    long long total_nzeros = 0;
    int my_fit = 1;
    int fit = 0;

    if (SCF_iter==1 || state->GpuDiagFits_verdict_n2!=n2) {
        state->mgpu_host_dense=0;
        ClusterNonCol_ResetGs2Fallback(state);
    }
    if (openmx_env_flag_at(OPENMX_ENV_CLUSTER_GPU_DIAG, -1)==0 && !openmx_mgpu_host_required()) {
        state->cpu_fallback_n2=n2;
        return 0;
    }
    if (openmx_mgpu_host_required()) goto host_capacity;

    if (state->GpuDiagFits_verdict != -1 && state->GpuDiagFits_verdict_n2 == n2 && 1 < SCF_iter) return state->GpuDiagFits_verdict;

    /* capacity of the sparse-to-dense staging segments: the gathered matrix
       and its index table scale with the total nonzero count */
    my_nzeros = (long long)ClusterNonCol_ComputeLocalNZeroCount();
    MPI_Allreduce(&my_nzeros, &total_nzeros, 1, MPI_LONG_LONG, MPI_SUM, mpi_comm_level1);

    if (myid == Host_ID) {
        my_fit = ClusterNonCol_DenseArena_TryEnsure(state, n, n2, (size_t)total_nzeros);

        if (my_fit) {
            my_fit = ClusterNonCol_EigenWorkspace_TryEnsure(state, ClusterNonCol_ArenaPtr(state, arena->o_Hs2),
                                                            (double *)ClusterNonCol_ArenaPtr(state, arena->o_ko),
                                                            n2, n2, NULL);
        }

        if (my_fit) {
            /* the SCF_iter 1 overlap diagonalization allocates its own
               cusolver workspace transiently; require room for it plus the
               configured margin on top of the reservation */
            ClusterNonColGpuEigenCtx *ctx = ClusterNonCol_GpuEigenCtx_Get(state);
            cusolverEigMode_t const jobz = CUSOLVER_EIG_MODE_VECTOR;
            cublasFillMode_t const uplo = CUBLAS_FILL_MODE_LOWER;
            double vl = 0.0, vu = 0.0;
            int64_t h_meig = 0;
            size_t overlap_d_bytes = 0;
            size_t overlap_h_bytes = 0;
            size_t free_bytes = 0;
            size_t total_bytes = 0;

            OpenMXSolverWait(cusolverDnXsyevdx_bufferSize(ctx->handle, NULL, jobz, CUSOLVER_EIG_RANGE_I, uplo, n,
                                                       CUDA_R_64F, ClusterNonCol_ArenaPtr(state, arena->o_S), n,
                                                       &vl, &vu, 1L, (int64_t)n, &h_meig, CUDA_R_64F,
                                                       ClusterNonCol_ArenaPtr(state, arena->o_ko), CUDA_R_64F,
                                                       &overlap_d_bytes, &overlap_h_bytes));

            if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
                (void)cudaGetLastError();
                my_fit = 0;
            }
            else {
                my_fit = (ClusterNonCol_CheckedAddCount(overlap_d_bytes, ClusterNonCol_GpuDiagReserveBytes(),
                                                        "GPU diag margin") <= free_bytes);
            }
        }

    }

    MPI_Allreduce(&my_fit, &fit, 1, MPI_INT, MPI_MIN, mpi_comm_level1);

    if (fit == 0) {
host_capacity:
        ClusterNonCol_DenseArena_Release(state);
        ClusterNonCol_GpuEigenCtx_Release(state);
        ClusterNonCol_ReleaseGpuSolverCachedEVec(state, myid);
        openmx_gemmul8ReleaseWorkspaces();
        acc_wait_all();
        acc_clear_freelists();
        /* the automatic decision (once per cycle) starts from the baseline
           its host estimate was calibrated on: the root host matrices of an
           earlier step go now (MAGMA re-creates them; the distributed path
           does not need them).  Forced MAGMA comes here every step and keeps
           them. */
        if (!openmx_mgpu_host_required()) ClusterNonCol_RootHostBuffers_Release(state);
        MPI_Barrier(mpi_comm_level1);
        /* Host_ID plans node-local MAGMA and, in a multi-node run, falls
           back to the distributed gpusolver2 path; the choice is broadcast
           before any rank commits to a path */
        const int capacity = openmx_cluster_capacity_select(mpi_comm_level1,n2,1,"Cluster NC");
        state->gs2_fallback_n2=0;
        state->mgpu_host_dense=0;
        fit=0;
        if (capacity==OPENMX_CLUSTER_CAPACITY_GS2) {
            /* verdict 0 takes every rank to the distributed family, whose
               gpusolver2 branches then run (ClusterNonCol_UseGs2).  The
               decision is made at SCF_iter 1 only (cached below), where that
               family also diagonalizes the overlap. */
            if (openmx_cluster_gs2_admit(mpi_comm_level1,ictxt1_2,n2,nblk2,1,"Cluster NC"))
                state->gs2_fallback_n2 = n2;
        }
        else if (capacity==OPENMX_CLUSTER_CAPACITY_MAGMA) {
            if (myid==Host_ID) {
                printf("<OpenMX_MGPU> Cluster NC n=%d: %s MAGMA, host-backed dense matrices; scf.gpu.num upper bound respected.\n",
                       n2,openmx_mgpu_host_required()?"forced":"cuSOLVER memory limit ->");
                fflush(stdout);
            }
            state->mgpu_host_dense=1;
            fit=1;
        }
    }

    state->GpuDiagFits_verdict = fit;
    state->GpuDiagFits_verdict_n2 = n2;
    state->cpu_fallback_n2 = !fit && !ClusterNonCol_UseGs2(state, n2) ? n2 : 0;
    return state->GpuDiagFits_verdict;
}

/* Host-side scratch matrices of the root dense solve.  They are refilled
   from scratch every SCF iteration, so keep the ~1.6 GiB of allocations
   alive across iterations instead of paying malloc/free plus first-touch
   faults each time. */

/* the persistent host matrices of the root dense paths (native and MAGMA)
   on Host_ID: the dense overlap cache and the root solve workspace */
static void ClusterNonCol_RootHostBuffers_Release(ClusterNonColState *state)
{
    ClusterNonColRootSolveWorkspace *ws = &state->root_solve_workspace;

    openmx_core_native_free(state->resources,state->dense_scache.S);
    memset(&state->dense_scache, 0, sizeof(state->dense_scache));
    openmx_core_native_free(state->resources,ws->Cs);
    openmx_core_native_free(state->resources,ws->rHs11);
    openmx_core_native_free(state->resources,ws->rHs12);
    openmx_core_native_free(state->resources,ws->rHs22);
    openmx_core_native_free(state->resources,ws->iHs11);
    openmx_core_native_free(state->resources,ws->iHs12);
    openmx_core_native_free(state->resources,ws->iHs22);
    openmx_core_native_free(state->resources,ws->Hs2);
    memset(ws, 0, sizeof(*ws));
}

static void ClusterNonCol_RootSolveWorkspace_Ensure(ClusterNonColState *state, int n, int n2)
{
    ClusterNonColRootSolveWorkspace *ws = &state->root_solve_workspace;
    size_t nn;
    size_t n2n2;

    if (ws->n == n && ws->n2 == n2) return;

    openmx_core_native_free(state->resources,ws->Cs);
    openmx_core_native_free(state->resources,ws->rHs11);
    openmx_core_native_free(state->resources,ws->rHs12);
    openmx_core_native_free(state->resources,ws->rHs22);
    openmx_core_native_free(state->resources,ws->iHs11);
    openmx_core_native_free(state->resources,ws->iHs12);
    openmx_core_native_free(state->resources,ws->iHs22);
    openmx_core_native_free(state->resources,ws->Hs2);
    memset(ws, 0, sizeof(*ws));

    nn   = ClusterNonCol_CheckedMulCount((size_t)n, (size_t)n, "root dense real matrix");
    n2n2 = ClusterNonCol_CheckedMulCount((size_t)n2, (size_t)n2, "root dense complex matrix");

    ws->Cs    = (double *)ClusterNonCol_MallocArray(state, nn, sizeof(double), "root dense work Cs");
    ws->rHs11 = (double *)ClusterNonCol_MallocArray(state, nn, sizeof(double), "root dense rHs11");
    ws->rHs12 = (double *)ClusterNonCol_MallocArray(state, nn, sizeof(double), "root dense rHs12");
    ws->rHs22 = (double *)ClusterNonCol_MallocArray(state, nn, sizeof(double), "root dense rHs22");
    ws->iHs11 = (double *)ClusterNonCol_MallocArray(state, nn, sizeof(double), "root dense iHs11");
    ws->iHs12 = (double *)ClusterNonCol_MallocArray(state, nn, sizeof(double), "root dense iHs12");
    ws->iHs22 = (double *)ClusterNonCol_MallocArray(state, nn, sizeof(double), "root dense iHs22");
    ws->Hs2   = (dcomplex *)ClusterNonCol_MallocArray(state, n2n2, sizeof(dcomplex), "root dense Hs2");

    ws->n  = n;
    ws->n2 = n2;
}

/* Full dense matrices stay on the host. Only bounded GEMM tiles and MAGMA's
   block-cyclic partitions are resident on each selected GPU. */
static void ClusterNonCol_MagmaRootDensePath(ClusterNonColState *state, int SCF_iter, double *ko, double *****nh, double *****ImNL,
                                             double ****CntOLP, int *MP, int n, int n2, int MaxN,
                                             int myid, dcomplex **dense_evec_out,
                                             int *dense_evec_on_device_out)
{
    const int owner=(myid==Host_ID);
    const size_t nn=(size_t)n*n, n2n2=(size_t)n2*n2;
    ClusterNonColDenseSCache *cache=&state->dense_scache;
    ClusterNonColRootSolveWorkspace *ws=&state->root_solve_workspace;
    int rebuild=(SCF_iter==1 || !cache->transformed_s_valid || cache->n!=n || cache->n2!=n2);
    double *S=NULL, *blocks[6]={NULL,NULL,NULL,NULL,NULL,NULL};
    dcomplex *q=NULL, *out=NULL;
    double t0=0.0;
    const int profile=owner && ClusterNonCol_ProfileEnabled();
    if (profile) dtime(&t0);
    *dense_evec_out=NULL; *dense_evec_on_device_out=0;
    state->refine.last_refined=0;
    if (owner) {
        if (cache->n!=n || cache->n2!=n2) { openmx_core_native_free(state->resources,cache->S); cache->S=NULL; }
        if (!cache->S) cache->S=ClusterNonCol_MallocArray(state, nn,sizeof(double),"host overlap");
        cache->n=n; cache->n2=n2; cache->s_on_device=0;
        S=cache->S;
        ClusterNonCol_RootSolveWorkspace_Ensure(state, n,n2);
        blocks[0]=ws->rHs11; blocks[1]=ws->rHs12; blocks[2]=ws->rHs22;
        blocks[3]=ws->iHs11; blocks[4]=ws->iHs12; blocks[5]=ws->iHs22;
        q=ws->Hs2;
    }
    MPI_Bcast(&rebuild,1,MPI_INT,Host_ID,mpi_comm_level1);
    if (rebuild) {
        Patch2Full_Cluster_NonCol_Owner(state, CntOLP,S,MP,owner,n);
        if (owner) {
            if (!openmx_mgpu_host_solve(n,n,0,S,n,ko+1))
                ClusterNonCol_AbortWithMessage(openmx_mgpu_host_error());
#pragma omp parallel for
            for (int j=0;j<n;++j) {
                double scale=1.0/sqrt(fmax(ko[j+1],1.0e-10));
                for (int i=0;i<n;++i) S[(size_t)j*n+i]*=scale;
            }
            cache->transformed_s_valid=1;
        }
    }
    if (profile) ClusterNonCol_ProfileLap(state, NCPROF_OVERLAP,&t0);
    Patch2Full_Cluster_NonCol_Owner(state, nh[0],blocks[0],MP,owner,n);
    Patch2Full_Cluster_NonCol_Owner(state, nh[2],blocks[1],MP,owner,n);
    Patch2Full_Cluster_NonCol_Owner(state, nh[1],blocks[2],MP,owner,n);
    Patch2Full_Cluster_NonCol_Owner(state, ImNL[0],blocks[3],MP,owner,n);
    Patch2Full_Cluster_NonCol_Owner(state, nh[3],blocks[4],MP,owner,n);
    Patch2Full_Cluster_NonCol_Owner(state, ImNL[1],blocks[5],MP,owner,n);
    Patch2Full_Cluster_NonCol_Owner(state, ImNL[2],owner?ws->Cs:NULL,MP,owner,n);
    if (profile) ClusterNonCol_ProfileLap(state, NCPROF_GATHER,&t0);
    if (owner) {
#pragma omp parallel for
        for (size_t i=0;i<nn;++i) blocks[4][i]+=ws->Cs[i];
        for (int b=0;b<6;++b) {
            if (!openmx_mgpu_host_dgemm(CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,blocks[b],n,S,n,ws->Cs,n) ||
                !openmx_mgpu_host_dgemm(CUBLAS_OP_T,CUBLAS_OP_N,n,n,n,S,n,ws->Cs,n,blocks[b],n))
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
        if (profile) ClusterNonCol_ProfileLap(state, NCPROF_FORWARD,&t0);
        if (ClusterNonCol_SampleHamiltonian(SCF_iter,q,n2,MaxN) && profile)
            ClusterNonCol_ProfileLap(state, -1,&t0);
        if (!openmx_mgpu_host_solve(n2,MaxN,1,q,n2,ko+1))
            ClusterNonCol_AbortWithMessage(openmx_mgpu_host_error());
        if (profile) ClusterNonCol_ProfileLap(state, NCPROF_EIGEN,&t0);
        dcomplex *vt=ClusterNonCol_MallocArray(state, (size_t)MaxN*n2,sizeof(dcomplex),"host backtransform transpose");
        out=ClusterNonCol_MallocArray(state, n2n2,sizeof(dcomplex),"host dense eigenvectors");
        memset(out,0,n2n2*sizeof(dcomplex));
#pragma omp parallel for collapse(2)
        for (int j=0;j<n2;++j) for (int i=0;i<MaxN;++i)
            vt[(size_t)j*MaxN+i]=q[(size_t)i*n2+j];
        for (int half=0;half<2;++half) {
            if (!openmx_mgpu_host_dgemm(CUBLAS_OP_N,CUBLAS_OP_T,2*MaxN,n,n,
                (double *)(vt+(size_t)half*n*MaxN),2*MaxN,S,n,
                (double *)(out+(size_t)half*n*n2),2*n2))
                ClusterNonCol_AbortWithMessage(openmx_mgpu_host_error());
        }
        openmx_core_native_free(state->resources,vt);
        if (profile) ClusterNonCol_ProfileLap(state, NCPROF_BACK,&t0);
        *dense_evec_out=out;
    }
    MPI_Bcast(ko,n2+1,MPI_DOUBLE,Host_ID,mpi_comm_level1);
}

static void ClusterNonCol_GpuSolverRootDensePath(ClusterNonColState *state, int SCF_iter, double *ko, double *****nh, double *****ImNL,
                                                double ****CntOLP, int *MP, int n, int n2, int MaxN,
                                                int myid, dcomplex **dense_evec_out,
                                                int *dense_evec_on_device_out)
{
    if (state->mgpu_host_dense) {
        ClusterNonCol_MagmaRootDensePath(state, SCF_iter,ko,nh,ImNL,CntOLP,MP,n,n2,MaxN,myid,
                                        dense_evec_out,dense_evec_on_device_out);
        return;
    }
    ClusterNonColDenseSCache *scache = &state->dense_scache;
    ClusterNonColDenseArena *arena = &state->dense_arena;
    const int owns_dense = (myid == Host_ID);
    const int profile = owns_dense && ClusterNonCol_ProfileEnabled();
    double t0 = 0.0;
    const int use_setham_packed_cache =
        (Set_Hamiltonian_GpuSolver_Packed_CacheReady() &&
         Set_Hamiltonian_GpuSolver_Packed_OrderMode() == 1);
    int rebuild_s = (SCF_iter == 1 || !scache->transformed_s_valid || scache->n != n || scache->n2 != n2);
    double *S = NULL;
    double *Cs = NULL;
    double *rHs11 = NULL, *rHs12 = NULL, *rHs22 = NULL;
    double *iHs11 = NULL, *iHs12 = NULL, *iHs22 = NULL;
    dcomplex *Hs2 = NULL;
    dcomplex *dense_evec = NULL;
    size_t nn = 0;
    size_t n2n2 = 0;
    int solve_state[2] = {0, 0};  /* an FP32 solve was used; FP64 for the rest of the cycle (1) or run (2) */

    *dense_evec_out = NULL;
    *dense_evec_on_device_out = 0;

    if (use_setham_packed_cache) {
        Set_Hamiltonian_GpuSolver_SetMP(MP);
    }
    if (Set_Hamiltonian_OpenACC_Rank_Is_Selected()) {
        set_cuda_default_device_from_local_rank_noncollective();
        set_openacc_nvidia_device_from_local_rank_noncollective();
    }

    if (owns_dense) {
        nn = ClusterNonCol_CheckedMulCount((size_t)n, (size_t)n, "root dense real matrix");
        n2n2 = ClusterNonCol_CheckedMulCount((size_t)n2, (size_t)n2, "root dense complex matrix");

        if (!arena->valid || arena->n != n || arena->n2 != n2) {
            ClusterNonCol_AbortWithMessage("The dense arena reservation is missing in Cluster_DFT_NonCol.c.");
        }

        if (scache->n != n || scache->n2 != n2) {
            if (scache->s_on_device) {
                ClusterNonCol_ArenaUnmap(scache->S);
                scache->s_on_device = 0;
            }
            openmx_core_native_free(state->resources,scache->S);
            scache->S = NULL;
            scache->transformed_s_valid = 0;
            rebuild_s = 1;
        }

        if (scache->S == NULL) {
            scache->S = (double *)ClusterNonCol_MallocArray(state, nn, sizeof(double), "cached transformed overlap");
        }

        scache->n = n;
        scache->n2 = n2;
        S = scache->S;

        ClusterNonCol_RootSolveWorkspace_Ensure(state, n, n2);
        Cs    = state->root_solve_workspace.Cs;
        rHs11 = state->root_solve_workspace.rHs11;
        rHs12 = state->root_solve_workspace.rHs12;
        rHs22 = state->root_solve_workspace.rHs22;
        iHs11 = state->root_solve_workspace.iHs11;
        iHs12 = state->root_solve_workspace.iHs12;
        iHs22 = state->root_solve_workspace.iHs22;
        Hs2   = state->root_solve_workspace.Hs2;

        /* ownership of dense_evec moves to the eigenvector stash, so it
           cannot live in the persistent workspace */
        dense_evec = (dcomplex *)ClusterNonCol_MallocArray(state, n2n2, sizeof(dcomplex), "root dense eigenvectors");
    }

    MPI_Bcast(&rebuild_s, 1, MPI_INT, Host_ID, mpi_comm_level1);

    if (profile) ClusterNonCol_ProfileLap(state, -1, &t0);

    if (rebuild_s) {
        if (owns_dense && !scache->s_on_device) {
            /* The transformed overlap only changes with the geometry, so it
               stays resident on the device across SCF iterations instead of
               being re-uploaded every step. */
            ClusterNonCol_ArenaMap(state, S, arena->o_S, nn * sizeof(double));
            scache->s_on_device = 1;
        }

        if (use_setham_packed_cache) {
            if (owns_dense) {
                if (!Set_Hamiltonian_GpuSolver_Packed_OwnsCache()) {
                    ClusterNonCol_AbortWithMessage("Set_Hamiltonian packed overlap cache is not owned by this rank.");
                }
                ClusterNonCol_LoadDeviceDenseFromSetHamCache(state, Set_Hamiltonian_GpuSolver_Packed_Overlap(), MP, n, S);
            }
        }
        else {
            Patch2Device_Cluster_NonCol_Owner(state, CntOLP, S, MP, owns_dense, n);
        }

        if (owns_dense) {
            ClusterNonCol_ArenaMap(state, ko, arena->o_ko, (size_t)(n2 + 1) * sizeof(double));
            Eigen_gpusolver_x_openacc2(S, ko, n, n);

if (!openmx_gpu_transform(OPENMX_GPU_INVSQRT,0,1,n,(double*)acc_deviceptr((void*)(ko))+1,1,NULL,0,(double*)acc_deviceptr((void*)(ko))+1,1,0,1.0e-10)) {
#pragma acc parallel loop present(ko[0 : n2 + 1])
            for (int l = 1; l <= n; l++) {
                if (ko[l] < 1.0e-10) ko[l] = 1.0e-10;
                ko[l] = 1.0 / sqrt(ko[l]);
            }
}

if (!openmx_gpu_transform(OPENMX_GPU_SCALE_COLUMNS,0,n,n,acc_deviceptr((void*)(S)),n,(double*)acc_deviceptr((void*)(ko))+1,0,acc_deviceptr((void*)(S)),n,0,0.0)) {
#pragma acc parallel loop collapse(2) present(S[0 : n * n], ko[0 : n2 + 1])
            for (int col = 0; col < n; col++) {
                for (int row = 0; row < n; row++) {
                    S[(size_t)row + (size_t)col * (size_t)n] *= ko[col + 1];
                }
            }
}

            ClusterNonCol_ArenaUnmap(ko);

            scache->transformed_s_valid = 1;
        }
    }

    if (profile) ClusterNonCol_ProfileLap(state, NCPROF_OVERLAP, &t0);

    if (owns_dense) {
        ClusterNonCol_ArenaMap(state, rHs11, arena->o_r[0], nn * sizeof(double));
        ClusterNonCol_ArenaMap(state, rHs12, arena->o_r[1], nn * sizeof(double));
        ClusterNonCol_ArenaMap(state, rHs22, arena->o_r[2], nn * sizeof(double));
        ClusterNonCol_ArenaMap(state, iHs11, arena->o_r[3], nn * sizeof(double));
        ClusterNonCol_ArenaMap(state, iHs12, arena->o_r[4], nn * sizeof(double));
        ClusterNonCol_ArenaMap(state, iHs22, arena->o_r[5], nn * sizeof(double));
        ClusterNonCol_ArenaMap(state, Cs,    arena->o_r[6], nn * sizeof(double));
    }

    if (use_setham_packed_cache) {
        if (owns_dense) {
            if (!Set_Hamiltonian_GpuSolver_Packed_OwnsCache()) {
                ClusterNonCol_AbortWithMessage("Set_Hamiltonian packed Hamiltonian cache is not owned by this rank.");
            }
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(state, Set_Hamiltonian_GpuSolver_Packed_H(0), MP, n, rHs11);
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(state, Set_Hamiltonian_GpuSolver_Packed_H(1), MP, n, rHs22);
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(state, Set_Hamiltonian_GpuSolver_Packed_H(2), MP, n, rHs12);
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(state, Set_Hamiltonian_GpuSolver_Packed_H(3), MP, n, iHs12);
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(state, Set_Hamiltonian_GpuSolver_Packed_ImNL(0), MP, n, iHs11);
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(state, Set_Hamiltonian_GpuSolver_Packed_ImNL(1), MP, n, iHs22);
            ClusterNonCol_LoadDeviceDenseFromSetHamCache(state, Set_Hamiltonian_GpuSolver_Packed_ImNL(2), MP, n, Cs);
        }
    }
    else {
        Patch2Device_Cluster_NonCol_Owner(state, nh[0],   rHs11, MP, owns_dense, n);
        Patch2Device_Cluster_NonCol_Owner(state, nh[1],   rHs22, MP, owns_dense, n);
        Patch2Device_Cluster_NonCol_Owner(state, nh[2],   rHs12, MP, owns_dense, n);
        Patch2Device_Cluster_NonCol_Owner(state, nh[3],   iHs12, MP, owns_dense, n);
        Patch2Device_Cluster_NonCol_Owner(state, ImNL[0], iHs11, MP, owns_dense, n);
        Patch2Device_Cluster_NonCol_Owner(state, ImNL[1], iHs22, MP, owns_dense, n);
        Patch2Device_Cluster_NonCol_Owner(state, ImNL[2], Cs,    MP, owns_dense, n);
    }

    if (profile) ClusterNonCol_ProfileLap(state, NCPROF_GATHER, &t0);

    if (owns_dense) {
if (!openmx_gpu_transform(OPENMX_GPU_ADD,0,n,n,acc_deviceptr((void*)(iHs12)),n,acc_deviceptr((void*)(Cs)),n,acc_deviceptr((void*)(iHs12)),n,0,0.0)) {
#pragma acc parallel loop present(iHs12[0 : n * n], Cs[0 : n * n])
        for (size_t i = 0; i < nn; i++) {
            iHs12[i] += Cs[i];
        }
}

        /* S (the scaled overlap eigenvectors) is already resident on the
           device from the rebuild step. */

if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(Cs)),n,0,0.0)) {
#pragma acc parallel loop present(Cs[0 : n * n])
        for (int i = 0; i < n * n; i++) Cs[i] = 0.0;
}

        ClusterNonCol_GEMMul8Dgemm_OpenACC(state, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, rHs11, S, Cs);
if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(rHs11)),n,0,0.0)) {
#pragma acc parallel loop present(rHs11[0 : n * n])
        for (int i = 0; i < n * n; i++) rHs11[i] = 0.0;
}

        ClusterNonCol_GEMMul8Dgemm_OpenACC(state, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, S, Cs, rHs11);

if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(Cs)),n,0,0.0)) {
#pragma acc parallel loop present(Cs[0 : n * n])
        for (int i = 0; i < n * n; i++) Cs[i] = 0.0;
}

        ClusterNonCol_GEMMul8Dgemm_OpenACC(state, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, rHs12, S, Cs);
if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(rHs12)),n,0,0.0)) {
#pragma acc parallel loop present(rHs12[0 : n * n])
        for (int i = 0; i < n * n; i++) rHs12[i] = 0.0;
}

        ClusterNonCol_GEMMul8Dgemm_OpenACC(state, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, S, Cs, rHs12);

if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(Cs)),n,0,0.0)) {
#pragma acc parallel loop present(Cs[0 : n * n])
        for (int i = 0; i < n * n; i++) Cs[i] = 0.0;
}

        ClusterNonCol_GEMMul8Dgemm_OpenACC(state, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, rHs22, S, Cs);
if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(rHs22)),n,0,0.0)) {
#pragma acc parallel loop present(rHs22[0 : n * n])
        for (int i = 0; i < n * n; i++) rHs22[i] = 0.0;
}

        ClusterNonCol_GEMMul8Dgemm_OpenACC(state, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, S, Cs, rHs22);

if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(Cs)),n,0,0.0)) {
#pragma acc parallel loop present(Cs[0 : n * n])
        for (int i = 0; i < n * n; i++) Cs[i] = 0.0;
}

        ClusterNonCol_GEMMul8Dgemm_OpenACC(state, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, iHs11, S, Cs);
if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(iHs11)),n,0,0.0)) {
#pragma acc parallel loop present(iHs11[0 : n * n])
        for (int i = 0; i < n * n; i++) iHs11[i] = 0.0;
}

        ClusterNonCol_GEMMul8Dgemm_OpenACC(state, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, S, Cs, iHs11);

if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(Cs)),n,0,0.0)) {
#pragma acc parallel loop present(Cs[0 : n * n])
        for (int i = 0; i < n * n; i++) Cs[i] = 0.0;
}

        ClusterNonCol_GEMMul8Dgemm_OpenACC(state, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, iHs12, S, Cs);
if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(iHs12)),n,0,0.0)) {
#pragma acc parallel loop present(iHs12[0 : n * n])
        for (int i = 0; i < n * n; i++) iHs12[i] = 0.0;
}

        ClusterNonCol_GEMMul8Dgemm_OpenACC(state, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, S, Cs, iHs12);

if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(Cs)),n,0,0.0)) {
#pragma acc parallel loop present(Cs[0 : n * n])
        for (int i = 0; i < n * n; i++) Cs[i] = 0.0;
}

        ClusterNonCol_GEMMul8Dgemm_OpenACC(state, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, iHs22, S, Cs);
if (!openmx_gpu_transform(OPENMX_GPU_ZERO,0,n,n,NULL,0,NULL,0,acc_deviceptr((void*)(iHs22)),n,0,0.0)) {
#pragma acc parallel loop present(iHs22[0 : n * n])
        for (int i = 0; i < n * n; i++) iHs22[i] = 0.0;
}

        ClusterNonCol_GEMMul8Dgemm_OpenACC(state, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, S, Cs, iHs22);

        ClusterNonCol_ArenaUnmap(Cs);
        ClusterNonCol_ArenaMap(state, Hs2, arena->o_Hs2, n2n2 * sizeof(dcomplex));
        ClusterNonCol_ArenaMap(state, ko, arena->o_ko, (size_t)(n2 + 1) * sizeof(double));

if (!openmx_gpu_spinor(n,1,acc_deviceptr((void*)(rHs11)),acc_deviceptr((void*)(rHs22)),acc_deviceptr((void*)(rHs12)),acc_deviceptr((void*)(iHs11)),acc_deviceptr((void*)(iHs22)),acc_deviceptr((void*)(iHs12)),acc_deviceptr((void*)(Hs2)))) {
#pragma acc parallel loop collapse(2) present(Hs2[0 : n2 * n2], rHs11[0 : n * n], rHs12[0 : n * n], rHs22[0 : n * n], \
                                             iHs11[0 : n * n], iHs12[0 : n * n], iHs22[0 : n * n])
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
}

        ClusterNonCol_ArenaUnmap(rHs11);
        ClusterNonCol_ArenaUnmap(rHs12);
        ClusterNonCol_ArenaUnmap(rHs22);
        ClusterNonCol_ArenaUnmap(iHs11);
        ClusterNonCol_ArenaUnmap(iHs12);
        ClusterNonCol_ArenaUnmap(iHs22);
        if (profile) ClusterNonCol_ProfileLap(state, NCPROF_FORWARD, &t0);

        /* the sampled steps' copy and write are charged to no phase */
        if (ClusterNonCol_SampleHamiltonian(SCF_iter, Hs2, n2, MaxN) && profile) ClusterNonCol_ProfileLap(state, -1, &t0);

        if (state->refine.last_refined && !state->refine.force_fp64) {
            /* the forward-transform segments (o_r, contiguous) are free until
               the next step, the transpose segment of the back transform
               until then; the first step (all states) needs product buffers
               of its own */
            size_t const region = arena->o_r[6] + nn * sizeof(double) - arena->o_r[0];
            int const warm = openmx_eigen_refine_warm_enabled() && state->refine_state.basis_valid &&
                             state->refine_state.basis_n == n2 && 1 < SCF_iter;
            int rc;

            if (3 <= level_stdout) {
                printf("<Cluster_DFT_NonCol> SCF step %d: %s, %d refinement steps%s (OPENMX_EIGEN_REFINE)\n", SCF_iter,
                       warm ? "warm start from the previous vectors" : "FP32 eigensolver",
                       ClusterNonCol_RefineIterations(state), SCF_iter == 1 ? " of all states" : "");
                fflush(stdout);
            }
            rc = ClusterNonCol_RefinedEigenPresent(state, Hs2, ko, n2, MaxN, ClusterNonCol_RefineIterations(state),
                                                   ClusterNonCol_ArenaPtr(state, arena->o_r[0]), region,
                                                   (dcomplex *)ClusterNonCol_ArenaPtr(state, arena->o_vt), SCF_iter == 1,
                                                   warm, &solve_state[1]);
            if (rc < 0) {
                rc = ClusterNonCol_RefinedEigenPresent(state, Hs2, ko, n2, MaxN, ClusterNonCol_RefineIterations(state),
                                                       ClusterNonCol_ArenaPtr(state, arena->o_r[0]), region,
                                                       (dcomplex *)ClusterNonCol_ArenaPtr(state, arena->o_vt), SCF_iter == 1,
                                                       0, &solve_state[1]);
            }
            if (rc == 1) {
                solve_state[0] = 1;
            }
            else {
                ClusterNonCol_ZheevdxPresent(state, Hs2, ko, n2, MaxN);
            }
        }
        else {
            ClusterNonCol_ZheevdxPresent(state, Hs2, ko, n2, MaxN);
        }
        if (profile) ClusterNonCol_ProfileLap(state, NCPROF_EIGEN, &t0);

        /* Back-transform only the MaxN solved states (the eigensolver already
           limits itself to MaxN).  The result takes the place of the solved
           vectors in their arena segment, where dense_evec is then mapped.
           The transposed vectors go into the forward-transform segments
           when they fit (MaxN <= 0.875 n2), which keeps the refined basis
           in the transpose segment for the next step's warm start; a solve
           that was not refined leaves no basis to start from. */
        {
            int const vt_in_r = ((size_t)2 * (size_t)MaxN * (size_t)n2 * sizeof(double) <= 7 * nn * sizeof(double));

            if (!vt_in_r || !solve_state[0]) state->refine_state.basis_valid = 0;
            ClusterNonCol_BackTransform(state, (dcomplex *)ClusterNonCol_ArenaPtr(state, arena->o_Hs2),
                                        (dcomplex *)ClusterNonCol_ArenaPtr(state, vt_in_r ? arena->o_r[0] : arena->o_vt),
                                        (double const *)ClusterNonCol_ArenaPtr(state, arena->o_S), n, n2, MaxN);
        }
        if (profile) ClusterNonCol_ProfileLap(state, NCPROF_BACK, &t0);

#pragma acc update self(ko[0 : n2 + 1])
        ClusterNonCol_ArenaUnmap(Hs2);
        ClusterNonCol_ArenaUnmap(ko);
        ClusterNonCol_ArenaMap(state, dense_evec, arena->o_Hs2, n2n2 * sizeof(dcomplex));

        /* free the GEMMul8 workspace (the back transform's, with 2 MaxN
           rows, is the largest) so the DM phase and the grid/force phases
           of this SCF step see the memory. */
        if (ClusterNonCol_GemmWorkspaceTurnRelease()) {
            openmx_gemmul8ReleaseWorkspaces();
        }
        if (profile) ClusterNonCol_ProfileLap(state, NCPROF_RELEASE, &t0);

        *dense_evec_out = dense_evec;
        *dense_evec_on_device_out = 1;
    }

    MPI_Bcast(ko, n2 + 1, MPI_DOUBLE, Host_ID, mpi_comm_level1);

    /* every rank learns how the root solved, for the stop test of DFT.c: an
       FP64 fallback is no refined step, and a persistent reason for it ends
       the refined stage of this SCF cycle */
    MPI_Bcast(solve_state, 2, MPI_INT, Host_ID, mpi_comm_level1);
    if (0 < ClusterNonCol_RefineIterations(state)) {
        state->refine.last_refined = solve_state[0];
        if (solve_state[0]) state->refine.cycle_refined = 1;
        if (1 <= solve_state[1]) state->refine.final_stage = 1;
        if (2 <= solve_state[1]) {
            state->refine.iterations = 0;
            state->refine.defaulted = 0;
        }
    }
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

static void Save_DOS_NonCol(ClusterNonColState *state, int n, int n2, int MaxN, int *MP, double ****OLP0, dcomplex *EVec1, double *ko);
static void Save_LCAO_NonCol(ClusterNonColState *state, int n, int n2, int MaxN, int *MP, double ****OLP0, dcomplex *EVec1, double *ko);
static dcomplex Lapack_LU_Zinverse(ClusterNonColState *state, int n, dcomplex *A);

static double Calc_Oscillator_Strength(ClusterNonColState *state,  int n, int n2, int UMOmax, int Nocc[2], int *MP2,
                                        int *ind2ind,
                                        dcomplex *Z0, dcomplex *A, dcomplex *Z,
					dcomplex *Ax, dcomplex *Ay, dcomplex *Az,
					double Utot1, double Utot2,
					double XANES_Res[10] );

static void Calc_XANES_NonCol(ClusterNonColState *state, const OpenMXAtomicViews *atomic,  int n, int n2, int MaxN, int *MP,
			       int *is2, int *ie2, dcomplex *EVec1, double *ko, const OpenMXElectronicViews *electronic);

double Calc_DM_Cluster_non_collinear_ScaLAPACK(const OpenMXChemicalPotentials *chemical,
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
    dcomplex *EVec1 , const OpenMXElectronicViews *electronic);

double Calc_ParDM_Cluster_non_collinear_ScaLAPACK(const OpenMXChemicalPotentials *chemical,
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
    dcomplex *EVec1 , const OpenMXElectronicViews *electronic);

typedef struct {
  ClusterNonColState *state;
  const OpenMXAtomicViews *atomic;
  char *mode;
  int SCF_iter;
  int SpinP_switch;
  double *ko;
  double *****nh;
  double *****ImNL;
  double ****CntOLP;
  double *****CDM;
  double *****EDM;
  double *Eele0;
  double *Eele1;
  int *MP;
  int *is2;
  int *ie2;
  double *Ss;
  double *Cs;
  double *rHs11;
  double *rHs12;
  double *rHs22;
  double *iHs11;
  double *iHs12;
  double *iHs22;
  dcomplex *Ss2;
  dcomplex *Hs2;
  dcomplex *Cs2;
  double *DM1;
  int size_H1;
  dcomplex *EVec1;
  double *Work1;
  const OpenMXElectronicViews *electronic;
} Cluster_DFT_NonCol_arguments;
static double Cluster_DFT_NonCol_owned(ClusterNonColState *state, const OpenMXAtomicViews *atomic,
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
                   double *Work1, const OpenMXElectronicViews *electronic, OpenMXSolverInvocation *solver_owner);
static int Cluster_DFT_NonCol_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Cluster_DFT_NonCol_arguments *a=(Cluster_DFT_NonCol_arguments*)arguments;
  *result=Cluster_DFT_NonCol_owned(a->state, a->atomic,a->mode,a->SCF_iter,a->SpinP_switch,a->ko,a->nh,a->ImNL,a->CntOLP,a->CDM,a->EDM,a->Eele0,a->Eele1,a->MP,a->is2,a->ie2,a->Ss,a->Cs,a->rHs11,a->rHs12,a->rHs22,a->iHs11,a->iHs12,a->iHs22,a->Ss2,a->Hs2,a->Cs2,a->DM1,a->size_H1,a->EVec1,a->Work1,a->electronic,owner);
  return 1;
}
double Cluster_DFT_NonCol(ClusterNonColState *state, const OpenMXAtomicViews *atomic,
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
                   double *Work1, const OpenMXElectronicViews *electronic)
{
  Cluster_DFT_NonCol_arguments arguments={state,atomic,mode,SCF_iter,SpinP_switch,ko,nh,ImNL,CntOLP,CDM,EDM,Eele0,Eele1,MP,is2,ie2,Ss,Cs,rHs11,rHs12,rHs22,iHs11,iHs12,iHs22,Ss2,Hs2,Cs2,DM1,size_H1,EVec1,Work1,electronic};
  return OpenMX_Solver_Run(Cluster_DFT_NonCol_operation,&arguments);
}

static double Cluster_DFT_NonCol_owned(ClusterNonColState *state, const OpenMXAtomicViews *atomic,
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
                   double *Work1, const OpenMXElectronicViews *electronic, OpenMXSolverInvocation *solver_owner)
{

  double ******iDM=electronic->iDM;

  int mgpu_scope_previous = openmx_mgpu_scope_enter(
      scf_eigen_lib_flag == GPUSOLVER && !gpusolver2_flag);

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
  int gpusolver_direct_evec_on_device = 0;
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
    n  = n + atomic->species.contracted_orbitals[wanA];
  }
  n2 = 2*n;

  if (SCF_iter==1) state->cpu_overlap_n2=0;

  /* GPU dispatch (added by H.Kawai): assign CUDA/OpenACC device when GPUSOLVER is requested */
    if (scf_eigen_lib_flag == GPUSOLVER && n2 >= Cluster_DFT_NonCol_GpuSwitchNum() &&
        Set_Hamiltonian_OpenACC_Rank_Is_Selected()) {
      set_cuda_default_device_from_local_rank_noncollective();
      set_openacc_nvidia_device_from_local_rank_noncollective();
    }

  use_gpusolver_direct_cluster_dm =
    (scf_eigen_lib_flag == GPUSOLVER && gpusolver2_flag == 0 && (openmx_mgpu_host_required() || Cluster_DFT_NonCol_GpuSwitchNum() <= n2) &&
     (strcasecmp(mode,"scf") == 0 || strcasecmp(mode,"fp64") == 0) &&
     MO_fileout != 1 && xanes_calc != 1 && xanes_gs_fileout != 1 &&
     !cal_partial_charge && !Dos_fileout && !DosGauss_fileout &&
     ClusterNonCol_GpuDiagFits(state, SCF_iter,n,n2,myid));

  /****************************************************
                  allocation of arrays
  ****************************************************/

  Num_Snd_EV=OpenMX_Solver_Int(solver_owner,"Num_Snd_EV",((size_t)1)*numprocs);
  Num_Rcv_EV=OpenMX_Solver_Int(solver_owner,"Num_Rcv_EV",((size_t)1)*numprocs);

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
    TZ += atomic->species.core_charge[wan];
  }

  /* ***************************************************
         find the numbers of partions for MPI
  *************************************************** */

  /* find the maximum states in solved eigenvalues */

  if (SCF_iter==1){
    MaxN = n2;
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

  index_Snd_i=OpenMX_Solver_Int(solver_owner,"index_Snd_i",((size_t)1)*Max_Num_Snd_EV);
  index_Snd_j=OpenMX_Solver_Int(solver_owner,"index_Snd_j",((size_t)1)*Max_Num_Snd_EV);
  EVec_Snd=OpenMX_Solver_Real(solver_owner,"EVec_Snd",((size_t)1)*Max_Num_Snd_EV*2);
  index_Rcv_i=OpenMX_Solver_Int(solver_owner,"index_Rcv_i",((size_t)1)*Max_Num_Rcv_EV);
  index_Rcv_j=OpenMX_Solver_Int(solver_owner,"index_Rcv_j",((size_t)1)*Max_Num_Rcv_EV);
  EVec_Rcv=OpenMX_Solver_Real(solver_owner,"EVec_Rcv",((size_t)1)*Max_Num_Rcv_EV*2);

  /* print memory size */

  if (state->firsttime && memoryusage_fileout){
    PrintMemory("Cluster_DFT_NonCol: Num_Snd_EV",sizeof(int)*numprocs,NULL);
    PrintMemory("Cluster_DFT_NonCol: Num_Ecv_EV",sizeof(int)*numprocs,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Snd_i",sizeof(int)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Snd_j",sizeof(int)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Rcv_i",sizeof(int)*Max_Num_Rcv_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Rcv_j",sizeof(int)*Max_Num_Rcv_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: EVec_Snd",sizeof(double)*Max_Num_Snd_EV*2,NULL);
    PrintMemory("Cluster_DFT_NonCol: EVec_Rcv",sizeof(double)*Max_Num_Rcv_EV*2,NULL);
  }
  state->firsttime=0;

  if (use_gpusolver_direct_cluster_dm){

    if (measure_time) dtime(&stime);

    ClusterNonCol_ReleaseGpuSolverCachedEVec(state, myid);

    for (i1=0; i1<=n2; i1++){ ko[i1] = 1.0e+5; }

    ClusterNonCol_RefineBeginSolve(state, SCF_iter);
    state->refine.force_fp64 = (strcasecmp(mode,"fp64")==0);
    ClusterNonCol_GpuSolverRootDensePath(state, SCF_iter,ko,nh,ImNL,CntOLP,MP,n,n2,MaxN,myid,
                                        &gpusolver_dense_evec,&gpusolver_direct_evec_on_device);
    state->refine.force_fp64 = 0;

    if (measure_time){
      dtime(&etime);
      time3 += etime - stime;
    }
  }
  else {

  state->refine.last_refined = 0;
  state->refine.cycle_refined = 0;

  /* ***************************************************
            diagonalize the overlap matrix
  *************************************************** */

  if (state->cpu_overlap_n2!=n2){

    if (measure_time) dtime(&stime);

    Overlap_Cluster_Ss(CntOLP,Cs,MP,0);

    MPI_Comm_split(mpi_comm_level1,my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(mpi_comm_level1,my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

    /* diagonalize Cs */

    if (ClusterNonCol_UseGs2(state, n2)){

      /* distributed multi-GPU eigensolver (ELPA, NVIDIA GPU kernels) */
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
    else if (scf_eigen_lib_flag==GPUSOLVER && state->cpu_fallback_n2!=n2 &&
             Cluster_DFT_NonCol_GpuSwitchNum()<=n2 && na_rows==n && na_cols==n){
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
    state->cpu_overlap_n2=n2;

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
  if (ClusterNonCol_UseGs2(state, n2))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&Re_alpha,rHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,rHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs11[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  if (ClusterNonCol_UseGs2(state, n2))
    openmx_gs2_pdgemm_("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs11,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs11,&ONE,&ONE,descH);

  /* S^t x rHs12 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  if (ClusterNonCol_UseGs2(state, n2))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&Re_alpha,rHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,rHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs12[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  if (ClusterNonCol_UseGs2(state, n2))
    openmx_gs2_pdgemm_("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs12,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs12,&ONE,&ONE,descH);

  /* S^t x rHs22 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  if (ClusterNonCol_UseGs2(state, n2))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&Re_alpha,rHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,rHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs22[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  if (ClusterNonCol_UseGs2(state, n2))
    openmx_gs2_pdgemm_("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs22,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs22,&ONE,&ONE,descH);

  /* S^t x iHs11 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  if (ClusterNonCol_UseGs2(state, n2))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&Re_alpha,iHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,iHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs11[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  if (ClusterNonCol_UseGs2(state, n2))
    openmx_gs2_pdgemm_("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs11,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs11,&ONE,&ONE,descH);

  /* S^t x iHs12 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  if (ClusterNonCol_UseGs2(state, n2))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&Re_alpha,iHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,iHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs12[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  if (ClusterNonCol_UseGs2(state, n2))
    openmx_gs2_pdgemm_("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs12,&ONE,&ONE,descH);
  else
    F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs12,&ONE,&ONE,descH);

  /* S^t x iHs22 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  if (ClusterNonCol_UseGs2(state, n2))
    openmx_gs2_pdgemm_("N","N",&n,&n,&n,&Re_alpha,iHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);
  else
    F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,iHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs22[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  if (ClusterNonCol_UseGs2(state, n2))
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

  if (ClusterNonCol_UseGs2(state, n2)){

    /* distributed multi-GPU eigensolver (ELPA, NVIDIA GPU kernels); all n2
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
  else if (scf_eigen_lib_flag==GPUSOLVER && state->cpu_fallback_n2!=n2 &&
           Cluster_DFT_NonCol_GpuSwitchNum()<=n2 && na_rows2==n2 && na_cols2==n2){
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
  if (ClusterNonCol_UseGs2(state, n2))
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
      ClusterNonCol_StashGpuSolverDenseEVec(state, myid,n2,&gpusolver_dense_evec,&gpusolver_direct_evec_on_device);
    }

    ;

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
      state->chemical->chemical_potential = 0.50*(ChemP_MAX + ChemP_MIN);
      Num_State = 0.0;

      for (i1=1; i1<=MaxN; i1++){
	x = (ko[i1] - state->chemical->chemical_potential)*Beta_trial1;
	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = FermiFunc_NC(x,i1);
	Num_State = Num_State + FermiF;
	if (0.5<FermiF) Cluster_HOMO[0] = i1;
      }

      Dnum = (TZ - Num_State) - system_charge;
      if (0.0<=Dnum) ChemP_MIN = state->chemical->chemical_potential;
      else           ChemP_MAX = state->chemical->chemical_potential;
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
	state->chemical->chemical_potential = 0.50*(ChemP_MAX + ChemP_MIN);
      }

      Num_State = 0.0;

      for (i1=1; i1<=MaxN; i1++){
	x = (ko[i1] - state->chemical->chemical_potential)*Beta;
	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = FermiFunc_NC(x,i1);
	Num_State = Num_State + FermiF;
	if (0.5<FermiF) Cluster_HOMO[0] = i1;
      }

      Dnum = (TZ - Num_State) - system_charge;
      if (0.0<=Dnum) ChemP_MIN = state->chemical->chemical_potential;
      else           ChemP_MAX = state->chemical->chemical_potential;
      if (fabs(Dnum)<1.0e-12) po = 1;
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

    /****************************************************
            Energies by summing up eigenvalues
    ****************************************************/

    Eele0[0] = 0.0;
    Eele0[1] = 0.0;

    for (i1=1; i1<=MaxN; i1++){

      x = (ko[i1] - state->chemical->chemical_potential)*Beta;
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

      array0=OpenMX_Solver_Real(solver_owner,"array0",((size_t)1)*(n2+2)*2);
      is3=OpenMX_Solver_Int(solver_owner,"is3",((size_t)1)*numprocs);
      ie3=OpenMX_Solver_Int(solver_owner,"ie3",((size_t)1)*numprocs);

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
	  tnoA = atomic->species.contracted_orbitals[wanA];
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
	  tnoA = atomic->species.contracted_orbitals[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    LUMOs_Coef[0][0][j][GA_AN][i].r = array0[ (Anum+i-1)*2      ];
	    LUMOs_Coef[0][0][j][GA_AN][i].i = array0[ (Anum+i-1)*2+1    ];
	    LUMOs_Coef[0][1][j][GA_AN][i].r = array0[ (Anum+i-1)*2  +n2 ];
	    LUMOs_Coef[0][1][j][GA_AN][i].i = array0[ (Anum+i-1)*2+1+n2 ];
	  }
	}
      }

      ;

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

      if (myid==Host_ID && !gpusolver_direct_evec_on_device && !state->mgpu_host_dense){
        ClusterNonCol_AbortWithMessage("GPUSOLVER device eigenvectors are not available in Cluster_DFT_NonCol.c.");
      }

      {
        double dm_time = ClusterNonCol_CalcDMRootDense_OpenACC(state, myid,size_H1,MP,n,n2,MaxN,CDM,iDM[0],EDM,ko,
                                                               gpusolver_dense_evec,(Cnt_switch==1));

        time6 += dm_time;
        ClusterNonCol_ProfileReport(state, myid,n2,MaxN,dm_time);
      }

      /* Keep the latest direct gpuSOLVER eigenvectors for the post-SCF EDM/force
         rebuild.  Scatter to EVec1 only once after SCF convergence to avoid
         paying this MPI/data-transfer cost every SCF step. */
      ClusterNonCol_StashGpuSolverDenseEVec(state, myid,n2,&gpusolver_dense_evec,&gpusolver_direct_evec_on_device);
    }
    else {

      /* DM */

      time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK(state->chemical,  1, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                        CDM, iDM[0], EDM, ko, DM1, Work1, EVec1, electronic);

      time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK(state->chemical,  2, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                        CDM, iDM[0], EDM, ko, DM1, Work1, EVec1, electronic);

      time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK(state->chemical,  3, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                        CDM, iDM[0], EDM, ko, DM1, Work1, EVec1, electronic);

      time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK(state->chemical,  4, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                        CDM, iDM[0], EDM, ko, DM1, Work1, EVec1, electronic);

      time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK(state->chemical,  5, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                        CDM, iDM[0], EDM, ko, DM1, Work1, EVec1, electronic);

      time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK(state->chemical,  6, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                        CDM, iDM[0], EDM, ko, DM1, Work1, EVec1, electronic);

      /* EDM */

      if (Cnt_switch==1){

        time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK(state->chemical,  7, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                          CDM, iDM[0], EDM, ko, DM1, Work1, EVec1, electronic);

        time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK(state->chemical,  8, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                          CDM, iDM[0], EDM, ko, DM1, Work1, EVec1, electronic);

        time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK(state->chemical,  9, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                          CDM, iDM[0], EDM, ko, DM1, Work1, EVec1, electronic);

        time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK(state->chemical,  10, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                          CDM, iDM[0], EDM, ko, DM1, Work1, EVec1, electronic);
      }

      /* calculation of ParDM */

      if ( strcasecmp(mode,"ParDM")==0 ){
        for (i=1; i<=6; i++){
          time6 += Calc_ParDM_Cluster_non_collinear_ScaLAPACK(state->chemical,  i, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                               ko, DM1, Work1, EVec1, electronic);
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
      tnoA = atomic->species.contracted_orbitals[wanA];
      for (j=0; j<=FNAN[GA_AN]; j++){
	wanB = WhatSpecies[natn[GA_AN][j]];
	tnoB = atomic->species.contracted_orbitals[wanB];

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

	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",state->chemical->chemical_potential);
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
	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",state->chemical->chemical_potential);
	fprintf(fp_EV,"   HOMO = %2d\n",Cluster_HOMO[0]);

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"   Real (Re) and imaginary (Im) parts of LCAO coefficients\n\n");

      }

      /* allocation of arrays */
      double *array0;
      int *is3,*ie3;

      array0=OpenMX_Solver_Real(solver_owner,"array0",((size_t)1)*(n2+2)*4);
      is3=OpenMX_Solver_Int(solver_owner,"is3",((size_t)1)*numprocs);
      ie3=OpenMX_Solver_Int(solver_owner,"ie3",((size_t)1)*numprocs);

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

      ;

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
    Save_DOS_NonCol(state, n,n2,MaxN,MP,CntOLP,EVec1,ko);
  }
  else if ( strcasecmp(mode,"lcaoout")==0 ){
    Save_LCAO_NonCol(state, n,n2,MaxN,MP,CntOLP,EVec1,ko);
  }
  else if ( strcasecmp(mode,"xanes")==0 ){
    Calc_XANES_NonCol(state, atomic, n,n2,MaxN,MP,is2,ie2,EVec1,ko, electronic);
  }

  if (measure_time){
    printf("Cluster_DFT myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f\n",
            myid,time1,time2,time3,time4,time5,time6,time7);fflush(stdout);
  }

  /****************************************************
                          Free
  ****************************************************/

  ;

  ;

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  openmx_mgpu_scope_leave(mgpu_scope_previous);
  return time0;
}

double Calc_DM_Cluster_non_collinear_ScaLAPACK(const OpenMXChemicalPotentials *chemical,
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
    dcomplex *EVec1 , const OpenMXElectronicViews *electronic)
{
  double *****Partial_DM=electronic->Partial_DM;

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

    x = (ko[k] - chemical->chemical_potential)*Beta;
    if (x<=-max_x) x = -max_x;
    if (max_x<=x)  x = max_x;
    FermiF = FermiFunc_NC(x,k);

    if (11<=calc_flag && cal_partial_charge){
      x2 = (ko[k] - (chemical->chemical_potential+ene_win_partial_charge))*Beta;
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

double Calc_ParDM_Cluster_non_collinear_ScaLAPACK(const OpenMXChemicalPotentials *chemical,
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
    dcomplex *EVec1 , const OpenMXElectronicViews *electronic)
{
  double *****ParDM=electronic->ParDM;
  double *****iParDM=electronic->iParDM;

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
      windowF = weight_CWF2(e,chemical->chemical_potential,in0,in1,out0,out1);
    }

    else if (CWF_WindowF_switch==1){ // Fermi
      e = ko[k];
      b0 = 1.0/CWF_disentangling_smearing_kBT0;
      b1 = 1.0/CWF_disentangling_smearing_kBT1;
      e0 = CWF_disentangling_Erange[0] + chemical->chemical_potential;
      e1 = CWF_disentangling_Erange[1] + chemical->chemical_potential;
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

void Save_DOS_NonCol(ClusterNonColState *state, int n, int n2, int MaxN, int *MP, double ****OLP0, dcomplex *EVec1, double *ko)
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

  array0 = (float*)openmx_core_native_malloc(state->resources,1,sizeof(float)*(n2+2)*2,0);
  is3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  ie3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  SD = (float*)openmx_core_native_malloc(state->resources,1,sizeof(float)*2*(atomnum+1)*List_YOUSO[7],0);
  Re11 = (float*)openmx_core_native_malloc(state->resources,1,sizeof(float)*(atomnum+1)*List_YOUSO[7],0);
  Re22 = (float*)openmx_core_native_malloc(state->resources,1,sizeof(float)*(atomnum+1)*List_YOUSO[7],0);
  Re12 = (float*)openmx_core_native_malloc(state->resources,1,sizeof(float)*(atomnum+1)*List_YOUSO[7],0);
  Im12 = (float*)openmx_core_native_malloc(state->resources,1,sizeof(float)*(atomnum+1)*List_YOUSO[7],0);

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
    if ( ((state->chemical->chemical_potential+Dos_Erange[0])<ko[i]) && (i-1)<iemin ) {
      iemin = i - 1;
      break;
    }
  }
  if (iemin<1) iemin = 1;

  /* find iemax */

  iemax = 1;
  for (i=1; i<=2*n; i++) {
    if ( ((state->chemical->chemical_potential+Dos_Erange[1])<ko[i])) {
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
    fprintf(fp_eig,"ChemP       %lf\n",state->chemical->chemical_potential);

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

  openmx_core_native_free(state->resources,array0);
  openmx_core_native_free(state->resources,is3);
  openmx_core_native_free(state->resources,ie3);
  openmx_core_native_free(state->resources,SD);
  openmx_core_native_free(state->resources,Re11);
  openmx_core_native_free(state->resources,Re22);
  openmx_core_native_free(state->resources,Re12);
  openmx_core_native_free(state->resources,Im12);
}

void Save_LCAO_NonCol(ClusterNonColState *state, int n, int n2, int MaxN, int *MP, double ****OLP0, dcomplex *EVec1, double *ko)
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

  array0 = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*(n2+2)*2,0);
  is3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  ie3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);

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
    if ( ((state->chemical->chemical_potential+Dos_Erange[0])<ko[i]) && (i-1)<iemin ) {
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
    x = (ko[i] - state->chemical->chemical_potential)*Beta;
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
    fwrite(&state->thermo->total, sizeof(double), 1, fp1);
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

  openmx_core_native_free(state->resources,array0);
  openmx_core_native_free(state->resources,is3);
  openmx_core_native_free(state->resources,ie3);
}

void Calc_XANES_NonCol(ClusterNonColState *state, const OpenMXAtomicViews *atomic,  int n, int n2, int MaxN, int *MP,
			int *is2, int *ie2, dcomplex *EVec1, double *ko, const OpenMXElectronicViews *electronic)

#define C1_ref(i,j) C1[ (n2)*(i)+(j) ]
#define C2_ref(i,j) C2[ (n2)*(i)+(j) ]

{
  double *****OLP=electronic->OLP;
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

  array0 = (double*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*(n2+2)*2,0);
  is3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  ie3 = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*numprocs,0);
  C2 = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*n2*MaxN,0);

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

    C1 = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*n2*MaxN1,0);

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
	tno0 = Spe_Total_CNO[Cwan];
      }

      OLPmo[direction][Mc_AN] = (double***)openmx_core_native_malloc(state->resources,1,sizeof(double**)*(FNAN[Gc_AN]+1),0);
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
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

  XANES_Out = (double**)openmx_core_native_malloc(state->resources,1,sizeof(double*)*Num_XANES_Out,0);
  for (i=0; i<Num_XANES_Out; i++){
    XANES_Out[i] = (double*)openmx_core_native_malloc(state->resources,1,sizeof(double)*10,0);
  }

  Z = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*UMOmax*UMOmax,0);
  Ax = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*UMOmax*UMOmax,0);
  Ay = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*UMOmax*UMOmax,0);
  Az = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*UMOmax*UMOmax,0);

  ind2ind = (int*)openmx_core_native_malloc(state->resources,1,sizeof(int)*UMOmax,0);

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
  Utot2 = state->thermo->total;

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

    Calc_Oscillator_Strength(state,  n, n2, UMOmax, HOMO_XANES, MP2, ind2ind, C1, C2, Z, Ax, Ay, Az, Utot1, Utot3, XANES_Res );

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
    openmx_core_native_free(state->resources,XANES_Out[i]);
  }
  openmx_core_native_free(state->resources,XANES_Out);

  openmx_core_native_free(state->resources,Z);
  openmx_core_native_free(state->resources,Ax);
  openmx_core_native_free(state->resources,Ay);
  openmx_core_native_free(state->resources,Az);
  openmx_core_native_free(state->resources,ind2ind);
  openmx_core_native_free(state->resources,MP2);
  openmx_core_native_free(state->resources,C2);
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

double Calc_Oscillator_Strength(ClusterNonColState *state,  int n, int n2, int UMOmax, int Nocc[2], int *MP2,
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

  det = Lapack_LU_Zinverse(state, Nocc[0], Z0);

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

dcomplex Lapack_LU_Zinverse(ClusterNonColState *state, int n, dcomplex *A)
{
    const char *thisprogram="Lapack_LU_inverse";
    int *ipiv;
    dcomplex *work,tmp,det;
    int lwork;
    int info,i,j;

    /* L*U factorization */

    ipiv = (int*) openmx_core_native_malloc(state->resources,1,sizeof(int)*n,0);

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
    work = (dcomplex*)openmx_core_native_malloc(state->resources,1,sizeof(dcomplex)*lwork,0);

    F77_NAME(zgetri,ZGETRI)(&n, A, &n, ipiv, work, &lwork, &info);

    if ( info !=0 ) {
      printf("zgetrf failed, info=%i, %s\n",info,thisprogram);
    }

    openmx_core_native_free(state->resources,work); openmx_core_native_free(state->resources,ipiv);

    return det;
}

/* Run-boundary reset: geometry and device state must not survive into the
   next -runtest input, whose host allocations can reuse the same addresses.
   Called on every rank before Free_Arrays(0) frees the current system. */
static void ClusterNonCol_ReleaseCaches(ClusterNonColState *state)
{
    state->mgpu_host_dense=0;
    state->cpu_overlap_n2=0;
    ClusterNonCol_ResetGs2Fallback(state);
    ClusterNonCol_DMEntryCache_Reset(state);
    ClusterNonCol_RootDMWorkspace_Reset(state);
    /* Discard the completed input's vectors before returning its arena. */
    ClusterNonCol_ReleaseGpuSolverCachedEVec(state, Host_ID);
    Cluster_DFT_NonCol_Release_GPU_Solver(state);
    openmx_core_native_free(state->resources,state->dense_index_cache.dense_index);
    memset(&state->dense_index_cache, 0, sizeof(state->dense_index_cache));
    ClusterNonCol_RootHostBuffers_Release(state);
    /* The next input owns a fresh refinement policy and no prior basis. */
    const ClusterNonColRefineState initial = {-1, 0.0, 0, 0, 0, 0, 0};
    state->refine = initial;
}
