/**********************************************************************
  Band_DFT_NonCol.c:

     Band_DFT_NonCol.c is a subroutine to perform band calculations
     based on a non-collinear DFT. 

  Log of Band_DFT_NonCol.c:

     16/Feb./2019  Released by T. Ozaki

***********************************************************************/

/* madvise/MADV_DONTNEED under -std=c11 */
#define _DEFAULT_SOURCE 1

#include "mpi.h"
#include "openmx_common.h"
#include "openmx_mgpu_runtime.h"
#include "openmx_gpu_transform.h"
#include "openmx_mgpu_band_plan.h"
#include "openmx_gpu_dm_collective.h"
#include "openmx_gpu_workers.h"
static int BandNonCol_mgpu_host_dense=0;

#include "eigen_refine_gpu.h"
#include "openmx_precision.h"
#include "eigen_refine_hooks.h"
#include "lapack_prototypes.h"
#include "tran_variables.h"
#include "set_hip_default_device_from_local_rank.h"
#include <math.h>
#include <omp.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <time.h>

/* the OpenMP device pass parses the whole TU with a stricter libc surface
   and loses the MADV_* macros; the value is the Linux ABI constant */
#ifndef MADV_DONTNEED
#define MADV_DONTNEED 4
#endif

#define  measure_time  0

extern int openmx_magma_zheevdx_gpu(int n, int maxn, void *d_A, double *w, int *mout);
extern int openmx_magma_release_z_workspace(void);
extern int openmx_hipsolver_zheevd_gpu(int n, int maxn, void *d_A, double *w, int *mout);
extern int openmx_hipsolver_release_workspace(void);
extern int openmx_gpu_eigensolver_use_hipsolver(void);
extern int openmx_gpu_eigen_backend(int n, int maxn, int cplx);
extern int openmx_gpu_is_apu(void);
extern int openmx_gpu_is_mi300a(void);

static int BandNonCol_SolveTimingEnabled(void)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("OPENMX_SOLVE_TIMING");
        enabled = (value != NULL && atoi(value) != 0);
    }
    return enabled;
}

/* GPU workspace for band non-collinear DM (added by H.Kawai, ported from 3.9.9 GPU) */
typedef struct
{
    int            initialized;
    int            device_id;
    int            max_tno;
    int            max_cols;
    int            max_nk;
    hipStream_t   stream;
    hipblasHandle_t hipblas;
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

static BandNonColDMGpuWorkspace BandNonCol_dm_gpu_workspace = {0};

typedef struct
{
    int m_index;
    int dense_index;
    int phase_index;
} BandNonColConstructEntry;

extern int BandNonCol_BuildDenseMs_HIP(int cpx_flag, int count, int h_count, int phase_count, int n,
                                       const BandNonColConstructEntry *entries, const double *phase_r,
                                       const double *phase_i, const double *M1, dcomplex *d_M);
extern int BandNonCol_AddDense_HIP(int count, dcomplex *d_dst, const dcomplex *d_src);
extern int BandNonCol_SymmetrizeDenseHermitian_HIP(int n, dcomplex *d_A);
extern int BandNonCol_BuildDenseHs2_HIP(int n, int n2, const dcomplex *d_H11,
                                        const dcomplex *d_H22, const dcomplex *d_H12,
                                        dcomplex *d_H2);
extern int BandNonCol_BuildDenseSs2_HIP(int n, int n2, const dcomplex *d_S, dcomplex *d_S2);
extern int BandNonCol_CalcDMRootDense_HIP(int entry_count, int pair_count, int size_H1,
                                          int n, int n2, int nk_occ,
                                          const int *basis0, const int *basis1,
                                          const int *phase_index, const double *phase_r,
                                          const double *phase_i, const double *occ,
                                          const double *eig_occ, const dcomplex *dense_evec,
                                          double *dm_buffer);
extern int BandNonCol_CalcDMRootDenseDevice_HIP(int entry_count, int pair_count, int size_H1,
                                                int n, int n2, int nk_occ,
                                                const int *basis0, const int *basis1,
                                                const int *phase_index, const double *phase_r,
                                                const double *phase_i, const double *occ,
                                                const double *eig_occ,
                                                const dcomplex *d_dense_evec,
                                                int dense_evec_ld, double *dm_buffer);

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

static BandNonColConstructCache BandNonCol_construct_cache = {0};

static void BandNonCol_AbortWithMessage(const char *msg);
extern void openmx_gemmul8TrimWorkspaces(void);
static void BandNonCol_RefineRelease(void);
static void BandNonCol_RefineResetStage(void);
static void BandNonCol_KOverlapReset(void);

typedef struct
{
    int                initialized;
    int                device_id;
    int                matrix_dim;
    size_t             d_work_bytes;
    size_t             h_work_bytes;
    hipStream_t       stream;
    hipsolverDnHandle_t gpusolver;
    dcomplex *         d_A;
    double *           d_W;
    int32_t *          d_info;
    void *             d_work;
    void *             h_work;
} BandNonColGpuSolverWorkspace;

static BandNonColGpuSolverWorkspace BandNonCol_gpusolver_workspace = {0};

typedef struct
{
    int            initialized;
    int            device_id;
    int            n_dim;
    int            n2_dim;
    hipblasHandle_t hipblas;
    dcomplex *     d_S;
    dcomplex *     d_H11;
    dcomplex *     d_H22;
    dcomplex *     d_H12;
    dcomplex *     d_tmp;
    dcomplex *     d_H2;
    dcomplex *     d_C2;
    size_t         d_C2_elems;
    int            d_C2_valid;
    int            d_C2_n2;
    int            d_C2_ld;
    dcomplex *     d_scale;
    dcomplex *     h_scale;
} BandNonColDenseGpuWorkspace;

static BandNonColDenseGpuWorkspace BandNonCol_dense_gpu_workspace = {0};
static int BandNonCol_dense_gpu_atexit_registered = 0;

static void BandNonCol_DenseGpu_EnsureHipblas(void)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;

    if (w->hipblas==NULL){
        wait_hipfunc(hipblasCreate(&w->hipblas));
    }
}

static void BandNonCol_DenseGpu_ReleaseHipblas(void)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;

    if (w->hipblas!=NULL){
        wait_hipfunc(hipDeviceSynchronize());
        wait_hipfunc(hipblasDestroy(w->hipblas));
        w->hipblas = NULL;
    }
}

static void BandNonCol_DenseGpu_Destroy(void)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;

    if (w->d_S     != NULL) wait_hipfunc(hipFree(w->d_S));
    if (w->d_H11   != NULL) wait_hipfunc(hipFree(w->d_H11));
    if (w->d_H22   != NULL) wait_hipfunc(hipFree(w->d_H22));
    if (w->d_H12   != NULL) wait_hipfunc(hipFree(w->d_H12));
    if (w->d_tmp   != NULL) wait_hipfunc(hipFree(w->d_tmp));
    if (w->d_H2    != NULL) wait_hipfunc(hipFree(w->d_H2));
    if (w->d_C2    != NULL) wait_hipfunc(hipFree(w->d_C2));
    if (w->d_scale != NULL) wait_hipfunc(hipFree(w->d_scale));
    if (w->hipblas  != NULL) wait_hipfunc(hipblasDestroy(w->hipblas));
    free(w->h_scale);

    memset(w,0,sizeof(*w));
    w->device_id = -1;
}

static int BandNonCol_DeviceEigenvectorDMRequested(void)
{
    if (BandNonCol_mgpu_host_dense) return 0;
    const char *value = getenv("OPENMX_NC_DEVICE_EIGENVECTOR_DM");

    return (value!=NULL && atoi(value)!=0);
}

static int BandNonCol_BacktransformBarrierRequested(void)
{
    if (BandNonCol_mgpu_host_dense) return 0;
    const char *value = getenv("OPENMX_NC_BACKTRANSFORM_BARRIER");

    return (value!=NULL && atoi(value)!=0);
}

static int BandNonCol_DenseBufferCacheEnabled(void)
{
    const char *value = getenv("OPENMX_NC_DENSE_BUFFER_CACHE");

    /* The direct DM path consumes d_C2 after the global chemical-potential
       phase, so its opt-in necessarily extends the dense workspace lifetime. */
    return BandNonCol_DeviceEigenvectorDMRequested() ||
           (value!=NULL && atoi(value)!=0);
}

static void BandNonCol_DenseGpu_ProcessCleanup(void)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;

    /* atexit handlers run after OpenMX has finalized MPI.  Do not use
       wait_hipfunc here, since its retry/error path calls MPI routines. */
    if (w->device_id>=0) (void)hipSetDevice(w->device_id);
    (void)hipDeviceSynchronize();
    if (w->d_S     != NULL) (void)hipFree(w->d_S);
    if (w->d_H11   != NULL) (void)hipFree(w->d_H11);
    if (w->d_H22   != NULL) (void)hipFree(w->d_H22);
    if (w->d_H12   != NULL) (void)hipFree(w->d_H12);
    if (w->d_tmp   != NULL) (void)hipFree(w->d_tmp);
    if (w->d_H2    != NULL) (void)hipFree(w->d_H2);
    if (w->d_C2    != NULL) (void)hipFree(w->d_C2);
    if (w->d_scale != NULL) (void)hipFree(w->d_scale);
    if (w->hipblas != NULL) (void)hipblasDestroy(w->hipblas);
    free(w->h_scale);

    memset(w,0,sizeof(*w));
    w->device_id = -1;
}

static void BandNonCol_DenseGpu_RegisterProcessCleanup(void)
{
    if (!BandNonCol_dense_gpu_atexit_registered){
        if (atexit(BandNonCol_DenseGpu_ProcessCleanup)==0){
            BandNonCol_dense_gpu_atexit_registered = 1;
        }
        else {
            fprintf(stderr,
                    "<Band>  Warning: failed to register non-collinear dense GPU cache cleanup.\n");
            fflush(stderr);
        }
    }
}

static void BandNonCol_DenseGpu_Ensure(int n, int n2)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;
    int current_device;
    size_t nn;
    size_t n2n2;

    if (n<=0 || n2<=0){
        BandNonCol_AbortWithMessage("Invalid dense GPU workspace dimensions in Band_DFT_NonCol.c.");
    }

    wait_hipfunc(hipGetDevice(&current_device));
    if (w->initialized && w->device_id!=current_device){
      int workspace_device = w->device_id;

      /* HIP allocations and the hipBLAS handle belong to the device on
         which this workspace was created.  Tear them down in that context
         before following a rank/device remap. */
      wait_hipfunc(hipSetDevice(workspace_device));
      BandNonCol_DenseGpu_Destroy();
      wait_hipfunc(hipSetDevice(current_device));
    }
    if (!(w->initialized && w->device_id==current_device && n<=w->n_dim && n2<=w->n2_dim)){
      if (w->initialized){
        BandNonCol_DenseGpu_Destroy();
      }
      w->device_id = current_device;
      w->n_dim = n;
      w->n2_dim = n2;
    }

    nn = (size_t)n*(size_t)n;
    n2n2 = (size_t)n2*(size_t)n2;

    if (w->d_S     == NULL) wait_hipfunc(hipMalloc((void**)&w->d_S,     sizeof(dcomplex)*nn));
    if (w->d_H11   == NULL) wait_hipfunc(hipMalloc((void**)&w->d_H11,   sizeof(dcomplex)*nn));
    if (w->d_H22   == NULL) wait_hipfunc(hipMalloc((void**)&w->d_H22,   sizeof(dcomplex)*nn));
    if (w->d_H12   == NULL) wait_hipfunc(hipMalloc((void**)&w->d_H12,   sizeof(dcomplex)*nn));
    if (w->d_tmp   == NULL) wait_hipfunc(hipMalloc((void**)&w->d_tmp,   sizeof(dcomplex)*nn));
    if (w->d_H2    == NULL) wait_hipfunc(hipMalloc((void**)&w->d_H2,    sizeof(dcomplex)*n2n2));
    if (w->d_scale == NULL) wait_hipfunc(hipMalloc((void**)&w->d_scale, sizeof(dcomplex)*(size_t)n2));

    if (w->h_scale == NULL) w->h_scale = (dcomplex*)malloc(sizeof(dcomplex)*(size_t)n2);
    if (w->h_scale==NULL){
        BandNonCol_DenseGpu_Destroy();
        BandNonCol_AbortWithMessage("Failed to allocate dense GPU scale workspace in Band_DFT_NonCol.c.");
    }

    w->initialized = 1;
    w->device_id = current_device;
    if (w->n_dim<n) w->n_dim = n;
    if (w->n2_dim<n2) w->n2_dim = n2;

    BandNonCol_DenseGpu_EnsureHipblas();
}

static void BandNonCol_DenseGpu_ReleaseWaveBuffers(void)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;

    if (w->d_C2 != NULL){
        wait_hipfunc(hipFree(w->d_C2));
        w->d_C2 = NULL;
    }
    w->d_C2_elems = 0U;
    w->d_C2_valid = 0;
    w->d_C2_n2 = 0;
    w->d_C2_ld = 0;
}

static void BandNonCol_DenseGpu_ReleaseOverlapMagmaScratch(void)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;

    wait_hipfunc(hipDeviceSynchronize());
    BandNonCol_DenseGpu_ReleaseWaveBuffers();

    if (w->d_H11 != NULL){
        wait_hipfunc(hipFree(w->d_H11));
        w->d_H11 = NULL;
    }
    if (w->d_H22 != NULL){
        wait_hipfunc(hipFree(w->d_H22));
        w->d_H22 = NULL;
    }
    if (w->d_H12 != NULL){
        wait_hipfunc(hipFree(w->d_H12));
        w->d_H12 = NULL;
    }
    if (w->d_tmp != NULL){
        wait_hipfunc(hipFree(w->d_tmp));
        w->d_tmp = NULL;
    }
    if (w->d_H2 != NULL){
        wait_hipfunc(hipFree(w->d_H2));
        w->d_H2 = NULL;
    }

    BandNonCol_DenseGpu_ReleaseHipblas();
}

static void BandNonCol_DenseGpu_EnsureWaveBuffers(int n2, int MaxN)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;
    size_t compact_count;

    if (n2<=0 || MaxN<=0 || n2<MaxN){
        BandNonCol_AbortWithMessage("Invalid compact wavefunction dimensions in Band_DFT_NonCol.c.");
    }

    if ((size_t)MaxN>SIZE_MAX/(size_t)n2){
        BandNonCol_AbortWithMessage("Compact wavefunction size overflow in Band_DFT_NonCol.c.");
    }
    compact_count = (size_t)n2*(size_t)MaxN;
    if (compact_count>SIZE_MAX/sizeof(dcomplex)){
        BandNonCol_AbortWithMessage("Compact wavefunction byte-size overflow in Band_DFT_NonCol.c.");
    }

    if (w->d_C2!=NULL && w->d_C2_elems<compact_count){
        wait_hipfunc(hipFree(w->d_C2));
        w->d_C2 = NULL;
        w->d_C2_elems = 0U;
    }
    if (w->d_C2==NULL){
        wait_hipfunc(hipMalloc((void**)&w->d_C2,sizeof(dcomplex)*compact_count));
        w->d_C2_elems = compact_count;
    }
    w->d_C2_valid = 0;
}

static const dcomplex *BandNonCol_DenseGpu_DeviceEigenvectors(int n2, int leading_dim)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;
    int current_device;
    size_t required;

    if (n2<=0 || leading_dim<=0 || (size_t)leading_dim>SIZE_MAX/(size_t)n2) return NULL;
    required = (size_t)n2*(size_t)leading_dim;

    if (hipGetDevice(&current_device)!=hipSuccess || !w->initialized ||
        w->device_id!=current_device || !w->d_C2_valid || w->d_C2==NULL || w->d_C2_n2!=n2 ||
        w->d_C2_ld!=leading_dim || w->d_C2_elems<required){
        return NULL;
    }
    return w->d_C2;
}

static void BandNonCol_DenseGpu_ReleasePreMagmaBuffers(int release_s)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;

    wait_hipfunc(hipDeviceSynchronize());
    BandNonCol_DenseGpu_ReleaseWaveBuffers();

    if (w->d_H11 != NULL){
        wait_hipfunc(hipFree(w->d_H11));
        w->d_H11 = NULL;
    }
    if (w->d_H22 != NULL){
        wait_hipfunc(hipFree(w->d_H22));
        w->d_H22 = NULL;
    }
    if (w->d_H12 != NULL){
        wait_hipfunc(hipFree(w->d_H12));
        w->d_H12 = NULL;
    }
    if (w->d_tmp != NULL){
        wait_hipfunc(hipFree(w->d_tmp));
        w->d_tmp = NULL;
    }
    if (w->d_scale != NULL){
        wait_hipfunc(hipFree(w->d_scale));
        w->d_scale = NULL;
    }
    if (release_s && w->d_S != NULL){
        wait_hipfunc(hipFree(w->d_S));
        w->d_S = NULL;
    }
    if (w->h_scale != NULL){
        free(w->h_scale);
        w->h_scale = NULL;
    }

    BandNonCol_DenseGpu_ReleaseHipblas();
}

static void BandNonCol_DenseGpu_RestoreS(int n, const dcomplex *h_S)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;
    const size_t nn = (size_t)n*(size_t)n;

    if (h_S==NULL){
        BandNonCol_AbortWithMessage("Missing cached overlap transform in Band_DFT_NonCol.c.");
    }
    if (w->d_S==NULL){
        wait_hipfunc(hipMalloc((void**)&w->d_S,sizeof(dcomplex)*nn));
    }
    wait_hipfunc(hipMemcpy(w->d_S,h_S,sizeof(dcomplex)*nn,hipMemcpyHostToDevice));
}

static void BandNonCol_DMGpu_Destroy(void)
{
    BandNonColDMGpuWorkspace * w = &BandNonCol_dm_gpu_workspace;
    if (w->d_vec_up   != NULL) wait_hipfunc(hipFree(w->d_vec_up));
    if (w->d_vec_up_e != NULL) wait_hipfunc(hipFree(w->d_vec_up_e));
    if (w->d_vec_dn   != NULL) wait_hipfunc(hipFree(w->d_vec_dn));
    if (w->d_vec_dn_e != NULL) wait_hipfunc(hipFree(w->d_vec_dn_e));
    if (w->d_cols_up  != NULL) wait_hipfunc(hipFree(w->d_cols_up));
    if (w->d_cols_dn  != NULL) wait_hipfunc(hipFree(w->d_cols_dn));
    if (w->d_mat11    != NULL) wait_hipfunc(hipFree(w->d_mat11));
    if (w->d_mat22    != NULL) wait_hipfunc(hipFree(w->d_mat22));
    if (w->d_mat12    != NULL) wait_hipfunc(hipFree(w->d_mat12));
    if (w->d_mat11e   != NULL) wait_hipfunc(hipFree(w->d_mat11e));
    if (w->d_mat22e   != NULL) wait_hipfunc(hipFree(w->d_mat22e));
    if (w->hipblas     != NULL) wait_hipfunc(hipblasDestroy(w->hipblas));
    if (w->stream     != NULL) wait_hipfunc(hipStreamDestroy(w->stream));
    memset(w, 0, sizeof(*w));
    w->device_id = -1;
}

static void BandNonCol_GpuSolver_Destroy(void)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;

    if (w->d_A      != NULL) wait_hipfunc(hipFree(w->d_A));
    if (w->d_W      != NULL) wait_hipfunc(hipFree(w->d_W));
    if (w->d_info   != NULL) wait_hipfunc(hipFree(w->d_info));
    if (w->d_work   != NULL) wait_hipfunc(hipFree(w->d_work));
    if (w->h_work   != NULL) free(w->h_work);
    if (w->gpusolver != NULL) wait_hipfunc(hipsolverDnDestroy(w->gpusolver));
    if (w->stream   != NULL) wait_hipfunc(hipStreamDestroy(w->stream));

    memset(w, 0, sizeof(*w));
    w->device_id = -1;
}

static void BandNonCol_DMGpu_Init(void)
{
    BandNonColDMGpuWorkspace * w = &BandNonCol_dm_gpu_workspace;
    int current_device;
    wait_hipfunc(hipGetDevice(&current_device));
    if (w->initialized && w->device_id == current_device) return;
    if (w->initialized) BandNonCol_DMGpu_Destroy();
    wait_hipfunc(hipStreamCreateWithFlags(&w->stream, hipStreamNonBlocking));
    wait_hipfunc(hipblasCreate(&w->hipblas));
    wait_hipfunc(hipblasSetStream(w->hipblas, w->stream));
    w->initialized = 1;
    w->device_id   = current_device;
}

static void BandNonCol_GpuSolver_Init(void)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;
    int current_device;

    wait_hipfunc(hipGetDevice(&current_device));
    if (w->initialized && w->device_id == current_device) return;
    if (w->initialized) BandNonCol_GpuSolver_Destroy();

    wait_hipfunc(hipStreamCreateWithFlags(&w->stream, hipStreamNonBlocking));
    wait_hipfunc(hipsolverDnCreate(&w->gpusolver));
    wait_hipfunc(hipsolverDnSetStream(w->gpusolver, w->stream));

    w->initialized = 1;
    w->device_id   = current_device;
}

static void BandNonCol_GpuSolver_EnsureInfo(void)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;

    BandNonCol_GpuSolver_Init();
    if (w->d_info==NULL) wait_hipfunc(hipMalloc((void**)&w->d_info,sizeof(int32_t)));
}

static void BandNonCol_GpuSolver_ReleaseDeviceWorkspace(void)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;

    if (w->d_work!=NULL) wait_hipfunc(hipFree(w->d_work));
    w->d_work = NULL;
    w->d_work_bytes = 0;
}

static void BandNonCol_SetDenseGemmul8Defaults(void)
{
    if (getenv("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT")==NULL &&
        getenv("GEMMUL8_MAX_WORKSPACE_PERCENT")==NULL){
        setenv("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT","50",0);
    }
}

static int BandNonCol_GemmWorkspaceTurnRelease(void)
{
    const char *value = getenv("OPENMX_BAND_GEMMUL8_TURN_RELEASE");

    if (value==NULL){
        return 1;
    }
    return (atoi(value)!=0);
}

static int BandNonCol_HipSolverWorkspaceTurnRelease(void)
{
    const char *value = getenv("OPENMX_NC_HIPSOLVER_TURN_RELEASE");

    if (value==NULL){
        /* On a unified-memory APU, retaining one large workspace in every
           MPI owner leaves less physical memory for the following owners and
           for the eigenvector back-transform. */
        return openmx_gpu_is_apu();
    }
    return (atoi(value)!=0);
}

static int BandNonCol_MagmaWorkspaceTurnRelease(void)
{
    const char *value = getenv("OPENMX_NC_MAGMA_TURN_RELEASE");

    if (value==NULL) return openmx_gpu_is_apu();
    return (atoi(value)!=0);
}

/* The policy wrapper may select native FP64 even when the legacy entry
   route below is MAGMA-named.  Query that policy for MI300A accounting;
   explicit OPENMX_GPU_EIGENSOLVER choices remain authoritative. */
static int BandNonCol_MI300ANativePolicy(int n, int maxn)
{
    return openmx_gpu_is_mi300a() && openmx_gpu_eigen_backend(n,maxn,1);
}

static void BandNonCol_ReleaseSolverTurnWorkspaces(int n, int maxn, int use_hipsolver)
{
    if (BandNonCol_MI300ANativePolicy(n,maxn)){
        /* A native attempt can retry through MAGMA.  Respect each library's
           release override independently, including scratch left by a
           successful MAGMA fallback.  In particular a MAGMA-named entry
           no longer implies that rocSOLVER's workspace is unused. */
        if (BandNonCol_HipSolverWorkspaceTurnRelease())
            (void)openmx_hipsolver_release_workspace();
        if (BandNonCol_MagmaWorkspaceTurnRelease())
            (void)openmx_magma_release_z_workspace();
    }
    else if (use_hipsolver && BandNonCol_HipSolverWorkspaceTurnRelease()){
        (void)openmx_hipsolver_release_workspace();
    }
    else if (!use_hipsolver && BandNonCol_MagmaWorkspaceTurnRelease()){
        (void)openmx_magma_release_z_workspace();
    }
}

static void BandNonCol_ReleaseOverlapWorkspace(int n)
{
    if (BandNonCol_MI300ANativePolicy(n,n)){
        BandNonCol_ReleaseSolverTurnWorkspaces(n,n,openmx_gpu_eigensolver_use_hipsolver());
    }
    else if (openmx_gpu_eigensolver_use_hipsolver() && BandNonCol_HipSolverWorkspaceTurnRelease()){
        /* Preserve the existing non-MI300A / explicit-MAGMA overlap path. */
        (void)openmx_hipsolver_release_workspace();
    }
}

/* Select the legacy explicit-hipSOLVER entry route, also used by host and
   OpenMP-mapped callers.  This is not the actual backend of the default
   HIP-owned solve: the MAGMA-named policy wrapper chooses native FP64 on
   MI300A and retains its memory preflight, phase transform and retry.
   Keep the route opt-in unchanged rather than passing HIP-owned matrices
   through the foreign-pointer entry merely to describe that backend. */
int BandNonCol_HamiltonianUseHipSolver(void)
{
    const char *value;

    if (!openmx_gpu_eigensolver_use_hipsolver()) return 0;
    value = getenv("OPENMX_NC_HAMILTONIAN_EIGENSOLVER");
    return (value!=NULL && strcasecmp(value,"hipsolver")==0);
}

static int BandNonCol_SerializeGpuSolverGpuTurns(void)
{
    const char *value = getenv("OPENMX_GPUSOLVER_SERIAL_GPU_TURNS");

    if (value==NULL){
        return 1;
    }
    return (atoi(value)!=0);
}

static int BandNonCol_AutoGpuTurnLimit(int requested, int n, int n2, int MaxN, int size_H1,
                                       int owns_dense_rank, int myid0);

static int BandNonCol_GpuTurnGroup(int n, int n2, int MaxN, int size_H1,
                                   int owns_dense_rank, int myid0)
{
    const char *value = getenv("OPENMX_BAND_GPU_TURN_GROUP");
    long group;

    if (value==NULL){
        value = getenv("OPENMX_BAND_GPU_MAX_RANKS_PER_DEVICE");
        if (value==NULL){
            return BandNonCol_AutoGpuTurnLimit(4,n,n2,MaxN,size_H1,
                                               owns_dense_rank,myid0);
        }
        else {
            group = strtol(value,NULL,10);
        }
    }
    else {
        group = strtol(value,NULL,10);
    }

    if (group<1L){
        return 1;
    }
    if (1024L<group){
        return 1024;
    }
    return (int)group;
}

static int BandNonCol_MaxConcurrentKGpuTurns(int n, int n2, int MaxN, int size_H1,
                                             int owns_dense_rank, int myid0)
{
    const int   max_requested = 4;
    const char *value = getenv("OPENMX_BAND_GPU_MAX_CONCURRENT_K");
    long limit;
    int requested = max_requested;

    if (value != NULL) {
        char *endp = NULL;
        limit = strtol(value, &endp, 10);
        if (endp == value || limit < 1L) {
            return 1;
        }
        if ((long)max_requested < limit) {
            requested = max_requested;
        }
        else {
            requested = (int)limit;
        }
        return BandNonCol_AutoGpuTurnLimit(requested,n,n2,MaxN,size_H1,
                                           owns_dense_rank,myid0);
    }

    value = getenv("OPENMX_BAND_GPU_MAX_RANKS_PER_DEVICE");
    if (value!=NULL){
        limit = strtol(value,NULL,10);
        if (limit<1L){
            return 1;
        }
        if ((long)max_requested<limit){
            requested = max_requested;
        }
        else {
            requested = (int)limit;
        }
        return BandNonCol_AutoGpuTurnLimit(requested,n,n2,MaxN,size_H1,
                                           owns_dense_rank,myid0);
    }

    return BandNonCol_AutoGpuTurnLimit(requested,n,n2,MaxN,size_H1,
                                       owns_dense_rank,myid0);
}

static void BandNonCol_GpuSolver_EnsureMatrixCapacity(int n)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;
    size_t nn;

    if (n<=0) BandNonCol_AbortWithMessage("Invalid GPU solver matrix size in Band_DFT_NonCol.c.");

    BandNonCol_GpuSolver_Init();
    if (n<=w->matrix_dim) return;

    if (w->d_A    != NULL) wait_hipfunc(hipFree(w->d_A));
    if (w->d_W    != NULL) wait_hipfunc(hipFree(w->d_W));
    if (w->d_info != NULL) wait_hipfunc(hipFree(w->d_info));

    nn = (size_t)n*(size_t)n;
    wait_hipfunc(hipMalloc((void**)&w->d_A,    sizeof(dcomplex)*nn));
    wait_hipfunc(hipMalloc((void**)&w->d_W,    sizeof(double)*(size_t)n));
    wait_hipfunc(hipMalloc((void**)&w->d_info, sizeof(int32_t)));

    w->matrix_dim = n;
}

static void BandNonCol_GpuSolver_EnsureWorkspace(int n, int maxn)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;
    hipsolverEigMode_t  jobz = HIPSOLVER_EIG_MODE_VECTOR;
    hipblasFillMode_t   uplo = HIPBLAS_FILL_MODE_LOWER;
    hipsolverEigRange_t range;
    double vl = 0.0;
    double vu = 0.0;
    int64_t h_meig = 0;
    size_t d_bytes = 0;
    size_t h_bytes = 0;

    if (n<=0 || maxn<=0 || n<maxn){
        BandNonCol_AbortWithMessage("Invalid GPU solver eigensolver dimensions in Band_DFT_NonCol.c.");
    }

    BandNonCol_GpuSolver_EnsureMatrixCapacity(n);
    range = (n==maxn) ? HIPSOLVER_EIG_RANGE_ALL : HIPSOLVER_EIG_RANGE_I;

    wait_hipfunc(hipsolverDnXsyevdx_bufferSize(w->gpusolver,NULL,jobz,range,uplo,n,
                                               HIP_C_64F,(hipDoubleComplex*)w->d_A,n,&vl,&vu,1L,maxn,&h_meig,
                                               HIP_R_64F,w->d_W,HIP_C_64F,&d_bytes,&h_bytes));

    if (w->d_work_bytes<d_bytes){
        if (w->d_work!=NULL) wait_hipfunc(hipFree(w->d_work));
        w->d_work = NULL;
        if (0<d_bytes) wait_hipfunc(hipMalloc((void**)&w->d_work,d_bytes));
        w->d_work_bytes = d_bytes;
    }

    if (h_bytes==0){
        if (w->h_work!=NULL) free(w->h_work);
        w->h_work = NULL;
        w->h_work_bytes = 0;
    }
    else if (w->h_work_bytes<h_bytes){
        if (w->h_work!=NULL) free(w->h_work);
        w->h_work = malloc(h_bytes);
        if (w->h_work==NULL) BandNonCol_AbortWithMessage("Failed to allocate GPU solver host workspace in Band_DFT_NonCol.c.");
        w->h_work_bytes = h_bytes;
    }
}

static int BandNonCol_OmpTargetIsPresent(const void *ptr, size_t bytes)
{
    if (ptr == NULL || bytes == 0) return 0;
    return omp_target_is_present(ptr, omp_get_default_device()) != 0;
}

static void BandNonCol_GpuSolver_DenseZheevx_Device(dcomplex *A, double *ko, int n, int maxn,
                                                   int use_hipsolver, const char *where)
{
    dcomplex *d_A = NULL;
    int info = 0;
    int mout = 0;
    int l;
    size_t nn = (size_t)n*(size_t)n;
    size_t bytes = sizeof(dcomplex)*nn;

    if (n<=0 || maxn<=0 || n<maxn){
        BandNonCol_AbortWithMessage("Invalid GPU eigensolver dimensions in Band_DFT_NonCol.c.");
    }

    {
        void *mapped = omp_get_mapped_ptr(A, omp_get_default_device());
        const int real_copy = (mapped != NULL && mapped != (void *)A);
        const int unified = (!real_copy && openmx_gpu_is_apu());
        int solved_on_device_side = 0;

        if (use_hipsolver && real_copy){
            /* The OpenMP mapping is a real device copy (MI300X and other
               discrete GPUs): the device side is authoritative, so solve in
               place on the mapped buffer and refresh the host array once. */
            const int tdbg = BandNonCol_SolveTimingEnabled();
            double tz0 = 0.0, tz1 = 0.0, tz2 = 0.0;

            if (tdbg) dtime(&tz0);

            info = openmx_hipsolver_zheevd_gpu(n,maxn,mapped,ko,&mout);

            if (tdbg) { wait_hipfunc(hipDeviceSynchronize()); dtime(&tz1); }

#pragma omp target update from(A[0 : nn])

            solved_on_device_side = 1;

            if (tdbg) {
                dtime(&tz2);
                printf("NC_SOLVE_TIMING(dev-mapped) zheevd=%.3f updfrom=%.3f\n", tz1 - tz0, tz2 - tz1);
                fflush(stdout);
            }
        }
        else if (use_hipsolver && unified){
            /* APU (MI300A) zero-copy: the host array IS the device data; solve
               on it directly with no staging and no transfers. */
            const int tdbg = BandNonCol_SolveTimingEnabled();
            double tz0 = 0.0, tz1 = 0.0;

            if (tdbg) dtime(&tz0);

            info = openmx_hipsolver_zheevd_gpu(n,maxn,(mapped != NULL) ? mapped : (void *)A,ko,&mout);

            solved_on_device_side = 1;

            if (tdbg) {
                wait_hipfunc(hipDeviceSynchronize());
                dtime(&tz1);
                printf("NC_SOLVE_TIMING(dev-unified) zheevd=%.3f\n", tz1 - tz0);
                fflush(stdout);
            }
        }
        else {
            /* Staging path: discrete GPU without a live mapping, or MAGMA. */
            const int tdbg = BandNonCol_SolveTimingEnabled();
            double tt0 = 0.0, tt1 = 0.0, tt2 = 0.0, tt3 = 0.0, tt4 = 0.0;

            if (tdbg) dtime(&tt0);

#pragma omp target update from(A[0 : nn])

            if (tdbg) dtime(&tt1);

            wait_hipfunc(hipMalloc((void**)&d_A,bytes));
            wait_hipfunc(hipMemcpy(d_A,A,bytes,hipMemcpyHostToDevice));

            if (tdbg) { wait_hipfunc(hipDeviceSynchronize()); dtime(&tt2); }

            if (use_hipsolver){
                info = openmx_hipsolver_zheevd_gpu(n,maxn,d_A,ko,&mout);
            }
            else {
                info = openmx_magma_zheevdx_gpu(n,maxn,d_A,ko,&mout);
            }

            if (tdbg) { wait_hipfunc(hipDeviceSynchronize()); dtime(&tt3); }

            wait_hipfunc(hipMemcpy(A,d_A,bytes,hipMemcpyDeviceToHost));
            wait_hipfunc(hipFree(d_A));

            if (tdbg) {
                dtime(&tt4);
                printf("NC_SOLVE_TIMING(dev-staged) updfrom=%.3f h2d=%.3f zheevd=%.3f d2h_free=%.3f\n",
                       tt1 - tt0, tt2 - tt1, tt3 - tt2, tt4 - tt3);
                fflush(stdout);
            }
        }

        if (info!=0){
            fprintf(stderr,"%s: GPU eigensolver (zheevd/zheevdx) failed, info=%d\n",where,info);
            fflush(stderr);
            MPI_Abort(mpi_comm_level1,1);
        }
        if (mout!=maxn){
            fprintf(stderr,"%s: GPU eigensolver returned %d eigenpairs, expected %d\n",
                    where,mout,maxn);
            fflush(stderr);
            MPI_Abort(mpi_comm_level1,1);
        }

        BandNonCol_ReleaseSolverTurnWorkspaces(n,maxn,use_hipsolver);

        for (l=maxn; 1<=l; l--) ko[l] = ko[l-1];

        {
            const int tdbg = BandNonCol_SolveTimingEnabled();
            double tu0 = 0.0, tu1 = 0.0;

            if (tdbg) dtime(&tu0);

            if (!solved_on_device_side){
                /* Staging path left the authoritative data on the host. */
#pragma omp target update to(A[0 : nn])
            }
#pragma omp target update to(ko[0 : n + 1])

            if (tdbg) {
                dtime(&tu1);
                printf("NC_SOLVE_TIMING(dev) updto=%.3f\n", tu1 - tu0);
                fflush(stdout);
            }
        }
    }
}

static void BandNonCol_GEMMul8Zgemm_OpenMP(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k,
                                            dcomplex const * A, dcomplex const * B, dcomplex * C)
{
    hipblasHandle_t handle;

    wait_hipfunc(hipblasCreate(&handle));
#pragma omp target data map(present, alloc: A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma omp target data use_device_ptr(A, B, C)
    {
        hipDoubleComplex const alpha = make_hipDoubleComplex(1.0, 0.0);
        hipDoubleComplex const beta  = make_hipDoubleComplex(0.0, 0.0);
        wait_hipfunc(openmx_gemmul8Zgemm(handle, transa, transb, m, n, k, &alpha,
                                          (hipDoubleComplex const *)A, m, (hipDoubleComplex const *)B, k, &beta,
                                          (hipDoubleComplex *)C, m));
    }
    wait_hipfunc(hipblasDestroy(handle));
}

static void BandNonCol_HipblasZgemm_OpenMP(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k,
                                           dcomplex const *A, dcomplex const *B, dcomplex *C)
{
    const int lda = (transa==HIPBLAS_OP_N) ? m : k;
    const int ldb = (transb==HIPBLAS_OP_N) ? k : n;
    const size_t a_count = (size_t)lda * (size_t)((transa==HIPBLAS_OP_N) ? k : m);
    const size_t b_count = (size_t)ldb * (size_t)((transb==HIPBLAS_OP_N) ? n : k);
    const size_t c_count = (size_t)m * (size_t)n;

    BandNonCol_DMGpu_Init();
#pragma omp target data map(present, alloc: A[0 : a_count], B[0 : b_count], C[0 : c_count])
#pragma omp target data use_device_ptr(A, B, C)
    {
        hipDoubleComplex const alpha = make_hipDoubleComplex(1.0, 0.0);
        hipDoubleComplex const beta  = make_hipDoubleComplex(0.0, 0.0);
        wait_hipfunc(hipblasZgemm(BandNonCol_dm_gpu_workspace.hipblas, transa, transb, m, n, k, &alpha,
                                  (hipDoubleComplex const *)A, lda,
                                  (hipDoubleComplex const *)B, ldb,
                                  &beta, (hipDoubleComplex *)C, m));
        wait_hipfunc(hipStreamSynchronize(BandNonCol_dm_gpu_workspace.stream));
    }
}

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
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("OPENMX_BAND_NONCOL_GPU_SWITCH_NUM");
        int parsed;

        cached = BAND_NONCOL_GPU_CPU_SWITCH_NUM;
        if (value != NULL && 0 < (parsed = atoi(value))) {
            cached = parsed;
        }

        /* When one GPU is shared by too many ranks the per-rank dense solve
           serializes on the device and its large packed-matrix gather can
           trip a UCX shared-memory assertion (observed on GGFF at 24 ranks
           on one MI300A); the CPU ELPA2 path is both faster and stable
           there.  Report an unreachable threshold so every n2 takes it.
           An explicit OPENMX_BAND_NONCOL_GPU_SWITCH_NUM still stands unless
           it too is above the matrix, i.e. the user can lower it but we do
           not silently re-enable the GPU path here. */
        if (openmx_band_gpu_dense_oversubscribed() && cached < INT_MAX) {
            cached = INT_MAX;
        }
    }
    return cached;
}

static int BandNonCol_UseDenseGpuMatrix(int n, int n2)
{
    return (scf_eigen_lib_flag == GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum() <= n2 &&
            na_rows == n && na_cols == n && na_rows2 == n2 && na_cols2 == n2);
}

static void BandNonCol_DenseTripleTransform_OpenMP(int n, dcomplex *A, dcomplex *S, dcomplex *Work)
{
    size_t nn = (size_t)n * (size_t)n;

#pragma omp target data map(tofrom: A[0 : nn]) map(to: S[0 : nn]) map(alloc: Work[0 : nn])
    {
#pragma omp target teams distribute parallel for
        for (size_t idx = 0; idx < nn; idx++) {
            Work[idx].r = 0.0;
            Work[idx].i = 0.0;
        }

        BandNonCol_HipblasZgemm_OpenMP(HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, A, S, Work);

#pragma omp target teams distribute parallel for
        for (size_t idx = 0; idx < nn; idx++) {
            A[idx].r = 0.0;
            A[idx].i = 0.0;
        }

        BandNonCol_HipblasZgemm_OpenMP(HIPBLAS_OP_C, HIPBLAS_OP_N, n, n, n, S, Work, A);
    }
}

static void BandNonCol_DenseTripleTransform_PresentOpenMP(int n, dcomplex *A, dcomplex *S, dcomplex *Work)
{
    int nn = n*n;

#pragma omp target teams distribute parallel for map(present, alloc: Work[0 : nn])
    for (int idx=0; idx<nn; idx++){
        Work[idx].r = 0.0;
        Work[idx].i = 0.0;
    }

    BandNonCol_GEMMul8Zgemm_OpenMP(HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,A,S,Work);

#pragma omp target teams distribute parallel for map(present, alloc: A[0 : nn])
    for (int idx=0; idx<nn; idx++){
        A[idx].r = 0.0;
        A[idx].i = 0.0;
    }

    BandNonCol_GEMMul8Zgemm_OpenMP(HIPBLAS_OP_C,HIPBLAS_OP_N,n,n,n,S,Work,A);
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

static void BandNonCol_SymmetrizeDenseHermitian_OpenMP(int n, dcomplex *A)
{
    int nn = n*n;

#pragma omp target teams distribute parallel for map(present, alloc: A[0 : nn])
    for (int i=0; i<n; i++){
        A[(size_t)i + (size_t)i*(size_t)n].i = 0.0;
    }

#pragma omp target teams distribute parallel for map(present, alloc: A[0 : nn])
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
   GEMMul8 product against S: half the work for all n2 states, and MaxN / n2
   of that for MaxN of them, and no Ss2 on the device (ported from the
   NVIDIA version).  Only the first MaxN columns of Cs2 need to be present;
   Hs2 is mapped here unless it already is. */
static void BandNonCol_DenseWavefunctions_PresentOpenMP(int n, int n2, int MaxN, dcomplex *Cs2, dcomplex *S,
                                                        dcomplex *Hs2)
{
    size_t const nn = (size_t)n*(size_t)n;
    size_t const n2n2 = (size_t)n2*(size_t)n2;
    size_t const c_count = (size_t)n2*(size_t)MaxN;

    if (!BandNonCol_OmpTargetIsPresent(Hs2,sizeof(dcomplex)*n2n2)){
#pragma omp target enter data map(alloc: Hs2[0 : n2n2])
    }

    if (MaxN<n2){
#pragma omp target teams distribute parallel for collapse(2) map(present, alloc: Hs2[0 : n2n2])
        for (int col=0; col<n2; col++){
            for (int row=MaxN; row<n2; row++){
                Hs2[(size_t)row + (size_t)col*(size_t)n2].r = 0.0;
                Hs2[(size_t)row + (size_t)col*(size_t)n2].i = 0.0;
            }
        }
    }

    BandNonCol_DMGpu_Init();
#pragma omp target data map(present, alloc: Cs2[0 : c_count], S[0 : nn], Hs2[0 : n2n2])
#pragma omp target data use_device_ptr(Cs2, S, Hs2)
    {
        hipDoubleComplex const alpha = make_hipDoubleComplex(1.0, 0.0);
        hipDoubleComplex const beta  = make_hipDoubleComplex(0.0, 0.0);

        for (int half=0; half<2; half++){
            size_t const off = (size_t)half*(size_t)n;

            wait_hipfunc(openmx_gemmul8Zgemm(BandNonCol_dm_gpu_workspace.hipblas, HIPBLAS_OP_T, HIPBLAS_OP_T,
                                              MaxN, n, n, &alpha,
                                              (hipDoubleComplex const *)(Cs2 + off), n2,
                                              (hipDoubleComplex const *)S, n, &beta,
                                              (hipDoubleComplex *)(Hs2 + off*(size_t)n2), n2));
        }
        wait_hipfunc(hipStreamSynchronize(BandNonCol_dm_gpu_workspace.stream));
    }
}

static void BandNonCol_DenseWavefunctions_OpenMP(int n, int n2, int MaxN, dcomplex *Cs2, dcomplex *S,
                                                 dcomplex *Hs2)
{
    size_t const nn = (size_t)n*(size_t)n;
    size_t const n2n2 = (size_t)n2*(size_t)n2;
    size_t const c_count = (size_t)n2*(size_t)MaxN;

#pragma omp target data map(to: Cs2[0 : c_count], S[0 : nn]) map(from: Hs2[0 : n2n2])
    {
        BandNonCol_DenseWavefunctions_PresentOpenMP(n,n2,MaxN,Cs2,S,Hs2);
    }
}

static void BandNonCol_AddDense_OpenMP(int n, dcomplex *dst, const dcomplex *src)
{
    int nn = n*n;

#pragma omp target teams distribute parallel for map(present, alloc: dst[0 : nn], src[0 : nn])
    for (int idx=0; idx<nn; idx++){
        dst[idx].r += src[idx].r;
        dst[idx].i += src[idx].i;
    }
}

static void BandNonCol_GpuSolver_DenseZheevx(dcomplex *A, dcomplex *Z, double *ko, int n, int maxn,
                                            int use_hipsolver, const char *where)
{
    dcomplex *d_A = NULL;
    int info = 0;
    int mout = 0;
    int l;
    int copy_cols;
    size_t nn;
    size_t bytes;

    if (n<=0 || maxn<=0 || n<maxn){
        BandNonCol_AbortWithMessage("Invalid GPU eigensolver dimensions in Band_DFT_NonCol.c.");
    }

    nn = (size_t)n*(size_t)n;
    bytes = sizeof(dcomplex)*nn;

    if (use_hipsolver && openmx_gpu_is_apu()){
        /* APU (MI300A): the host array lives in the same physical memory the
           GPU uses, so solve on it directly with no staging copies. */
        const int tdbg = BandNonCol_SolveTimingEnabled();
        double th0 = 0.0, th1 = 0.0;

        if (tdbg) dtime(&th0);

        info = openmx_hipsolver_zheevd_gpu(n,maxn,A,ko,&mout);

        if (tdbg) {
            wait_hipfunc(hipDeviceSynchronize());
            dtime(&th1);
            printf("NC_SOLVE_TIMING(host-unified) zheevd=%.3f\n", th1 - th0);
            fflush(stdout);
        }
    }
    else {
        const int tdbg = BandNonCol_SolveTimingEnabled();
        double th0 = 0.0, th1 = 0.0, th2 = 0.0, th3 = 0.0;

        if (tdbg) dtime(&th0);

        wait_hipfunc(hipMalloc((void**)&d_A,bytes));
        wait_hipfunc(hipMemcpy(d_A,A,bytes,hipMemcpyHostToDevice));

        if (tdbg) { wait_hipfunc(hipDeviceSynchronize()); dtime(&th1); }

        if (use_hipsolver){
            info = openmx_hipsolver_zheevd_gpu(n,maxn,d_A,ko,&mout);
        }
        else {
            info = openmx_magma_zheevdx_gpu(n,maxn,d_A,ko,&mout);
        }

        if (tdbg) { wait_hipfunc(hipDeviceSynchronize()); dtime(&th2); }

        wait_hipfunc(hipMemcpy(A,d_A,bytes,hipMemcpyDeviceToHost));
        wait_hipfunc(hipFree(d_A));

        if (tdbg) {
            dtime(&th3);
            printf("NC_SOLVE_TIMING(host-staged) h2d=%.3f zheevd=%.3f d2h_free=%.3f\n",
                   th1 - th0, th2 - th1, th3 - th2);
            fflush(stdout);
        }
    }

    if (info!=0){
        fprintf(stderr,"%s: GPU eigensolver (zheevd/zheevdx) failed, info=%d\n",where,info);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1,1);
    }
    if (mout!=maxn){
        fprintf(stderr,"%s: GPU eigensolver returned %d eigenpairs, expected %d\n",
                where,mout,maxn);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1,1);
    }

    BandNonCol_ReleaseSolverTurnWorkspaces(n,maxn,use_hipsolver);

    for (l=maxn; 1<=l; l--) ko[l] = ko[l-1];

    if (Z!=NULL){
        copy_cols = maxn;
        if (n<copy_cols) copy_cols = n;
        memcpy(Z,A,sizeof(dcomplex)*(size_t)n*(size_t)copy_cols);
    }
}

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

static BandNonColDMEntryCache BandNonCol_dm_entry_cache = {0};
static BandNonColDMOccWorkspace BandNonCol_dm_occ_workspace = {0};

static void BandNonCol_AbortWithMessage(const char *msg)
{
    fprintf(stderr,"%s\n",msg);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1,1);
}

static int BandNonCol_GpuVerbose(void)
{
    const char *value = getenv("OPENMX_GPU_VERBOSE");

    return (value != NULL && value[0] == '1');
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
    size_t nn = BandNonCol_CheckedMul((size_t)n,(size_t)n,"GPU solver fallback workspace matrix count");
    size_t matrix_bytes = BandNonCol_ArrayBytes(nn,sizeof(dcomplex),"GPU solver fallback workspace matrix");

    return BandNonCol_CheckedMul(matrix_bytes,4U,"GPU solver fallback workspace");
}

static size_t BandNonCol_QueryGpuSolverWorkBytes(int n, int maxn)
{
    /* Native Zheevd solves the full spectrum regardless of maxn.  Keep the
       two successful overlap/Hamiltonian queries by device and dimension;
       no matrix data or failed-query fallback is cached across inputs. */
    static struct {
        int valid, device, n;
        size_t bytes;
    } native_cache[2] = {{0}};
    static unsigned native_next = 0;
    const int native = BandNonCol_MI300ANativePolicy(n,maxn);
    int device = -1;
    hipsolverDnHandle_t handle = NULL;
    hipStream_t stream = NULL;
    hipError_t hip_status;
    hipsolverStatus_t solver_status;
    hipsolverEigMode_t jobz = HIPSOLVER_EIG_MODE_VECTOR;
    hipblasFillMode_t uplo = HIPBLAS_FILL_MODE_LOWER;
    hipsolverEigRange_t range = (n==maxn) ? HIPSOLVER_EIG_RANGE_ALL : HIPSOLVER_EIG_RANGE_I;
    double vl = 0.0;
    double vu = 0.0;
    int64_t h_meig = 0;
    size_t d_bytes = 0;
    size_t h_bytes = 0;
    size_t fallback = BandNonCol_GpuSolverWorkFallbackBytes(n);

    if (native){
        if (hipGetDevice(&device)!=hipSuccess) return fallback;
        for (int i=0; i<2; ++i)
            if (native_cache[i].valid && native_cache[i].device==device && native_cache[i].n==n)
                return native_cache[i].bytes;
    }

    hip_status = hipStreamCreateWithFlags(&stream,hipStreamNonBlocking);
    if (hip_status!=hipSuccess) return fallback;

    solver_status = hipsolverDnCreate(&handle);
    if (solver_status==HIPSOLVER_STATUS_SUCCESS){
        solver_status = hipsolverDnSetStream(handle,stream);
    }

    if (solver_status==HIPSOLVER_STATUS_SUCCESS){
        if (native){
            int lwork = 0;
            solver_status = hipsolverDnZheevd_bufferSize(handle,jobz,uplo,
                                                         n,NULL,n,NULL,&lwork);
            if (solver_status==HIPSOLVER_STATUS_SUCCESS && 0<lwork){
                d_bytes = BandNonCol_ArrayBytes((size_t)lwork,sizeof(hipDoubleComplex),
                                                  "native Zheevd workspace");
            }
        }
        else {
            solver_status = hipsolverDnXsyevdx_bufferSize(handle,NULL,jobz,range,uplo,n,
                                                     HIP_C_64F,NULL,n,&vl,&vu,1L,maxn,&h_meig,
                                                     HIP_R_64F,NULL,HIP_C_64F,&d_bytes,&h_bytes);
        }
    }

    if (handle!=NULL) hipsolverDnDestroy(handle);
    hipStreamDestroy(stream);

    if (solver_status!=HIPSOLVER_STATUS_SUCCESS || d_bytes==0) return fallback;

    if (native){
        unsigned slot = native_next++ % 2U;
        native_cache[slot].valid = 1;
        native_cache[slot].device = device;
        native_cache[slot].n = n;
        native_cache[slot].bytes = d_bytes;
    }
    return d_bytes;
}

/*
  Fixed per-rank device cost of the BLAS/eigensolver libraries beyond the
  matrices estimated below (handle workspaces, kernel code objects): measured
  at 200-300 MiB per rank on ROCm 7.2 for the Krylov solver, independent of
  the matrix size, and an allocation that fails inside hipBLASLt terminates
  the process instead of returning an error.  Every rank that may solve
  concurrently pays it, so it belongs to the per-rank estimate that the
  concurrency guard multiplies.  OPENMX_BAND_GPU_RESERVE_MB overrides the
  default of 512 MiB.
*/
static size_t BandNonCol_GpuLibraryReserveBytes(void)
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

/* Verdict of the last BandNonCol_GpuDiagFits (uniform over the ranks):
   1 while the device is too short of memory for the dense GPU
   diagonalization.  The per-k-point GPU eigensolves of the fallback loop
   consult it too -- MAGMA creates and destroys a BLAS handle inside every
   zheevdx call, which is exactly the per-rank cost that exhausted the
   device, so they have to take the ELPA/LAPACK path along with the dense
   solve instead of running unguarded. */
static int BandNonCol_gpu_dense_refused = 0;

/* Ranks of mpi_comm_level1 that share this rank's GPU.  Collective on the
   first call; every rank reaches BandNonCol_GpuDiagFits together. */
static int BandNonCol_device_ranks_cached = 0;

static int BandNonCol_DeviceRanks(void)
{
    if (BandNonCol_device_ranks_cached==0){
        MPI_Comm node_comm, device_comm;
        int hip_device = 0;
        int device_ranks = 0;

        MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
        if (hipGetDevice(&hip_device)!=hipSuccess) hip_device = 0;
        MPI_Comm_split(node_comm,hip_device,0,&device_comm);
        MPI_Comm_size(device_comm,&device_ranks);
        MPI_Comm_free(&device_comm);
        MPI_Comm_free(&node_comm);
        BandNonCol_device_ranks_cached = (device_ranks<1) ? 1 : device_ranks;
    }
    return BandNonCol_device_ranks_cached;
}

/* non-collective: for the per-k-point paths, which not every rank reaches
   the same number of times (BandNonCol_GpuDiagFits fills the cache first) */
static int BandNonCol_DeviceRanksCached(void)
{
    return (BandNonCol_device_ranks_cached<1) ? 1 : BandNonCol_device_ranks_cached;
}

static size_t BandNonCol_RootDenseDeviceBytes(int n, int n2, int MaxN, int size_H1)
{
    size_t nn = BandNonCol_CheckedMul((size_t)n,(size_t)n,"root dense n*n");
    size_t n2n2 = BandNonCol_CheckedMul((size_t)n2,(size_t)n2,"root dense n2*n2");
    size_t n2maxn = BandNonCol_CheckedMul((size_t)n2,(size_t)MaxN,"root dense n2*MaxN");
    size_t matrix_bytes = BandNonCol_ArrayBytes(nn,sizeof(dcomplex),"root dense n*n matrix");
    size_t matrix2_bytes = BandNonCol_ArrayBytes(n2n2,sizeof(dcomplex),"root dense n2*n2 matrix");
    size_t compact_evec_bytes = BandNonCol_ArrayBytes(n2maxn,sizeof(dcomplex),
                                                       "root dense compact eigenvectors");
    size_t scale_bytes = BandNonCol_ArrayBytes((size_t)n2,sizeof(dcomplex),
                                                "root dense scale vector");
    size_t packed_count = (0<size_H1) ? (size_t)size_H1 : 0U;
    size_t construct_bytes = 0;
    size_t overlap_peak = 0;
    size_t hamiltonian_peak = 0;
    size_t wavefunction_peak = 0;
    size_t work_n = BandNonCol_QueryGpuSolverWorkBytes(n,n);
    size_t work_n2 = BandNonCol_QueryGpuSolverWorkBytes(n2,MaxN);
    size_t ko_bytes = BandNonCol_ArrayBytes((size_t)n2+1U,sizeof(double),"root dense eigenvalue vector");

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
    BandNonCol_AddBytes(&overlap_peak,work_n,"root dense overlap GPU solver workspace");

    BandNonCol_AddBytes(&hamiltonian_peak,construct_bytes,"root dense Hamiltonian peak construct cache");
    BandNonCol_AddBytes(&hamiltonian_peak,ko_bytes,"root dense Hamiltonian peak eigenvalues");
    BandNonCol_AddBytes(&hamiltonian_peak,
                        BandNonCol_CheckedMul(matrix_bytes,5U,"root dense Hamiltonian n*n matrices"),
                        "root dense Hamiltonian n*n matrices");
    BandNonCol_AddBytes(&hamiltonian_peak,matrix2_bytes,"root dense Hamiltonian n2*n2 matrix");
    BandNonCol_AddBytes(&hamiltonian_peak,work_n2,"root dense Hamiltonian GPU solver workspace");
    BandNonCol_AddBytes(&hamiltonian_peak,sizeof(int32_t),"root dense Hamiltonian GPU solver info");

    BandNonCol_AddBytes(&wavefunction_peak,construct_bytes,"root dense wavefunction peak construct cache");
    BandNonCol_AddBytes(&wavefunction_peak,ko_bytes,"root dense wavefunction peak eigenvalues");
    BandNonCol_AddBytes(&wavefunction_peak,matrix_bytes,"root dense wavefunction overlap matrix");
    BandNonCol_AddBytes(&wavefunction_peak,matrix2_bytes,"root dense wavefunction Hamiltonian matrix");
    BandNonCol_AddBytes(&wavefunction_peak,compact_evec_bytes,
                        "root dense wavefunction compact eigenvectors");

    if (BandNonCol_DenseBufferCacheEnabled()){
        /* Cached mode intentionally carries every dense buffer across phase
           and SCF boundaries.  Include that retained memory in the automatic
           per-device concurrency guard, not just the normal phase peak. */
        BandNonCol_AddBytes(&overlap_peak,
                            BandNonCol_CheckedMul(matrix_bytes,4U,
                                                  "root dense cached overlap n*n matrices"),
                            "root dense cached overlap n*n matrices");
        BandNonCol_AddBytes(&overlap_peak,matrix2_bytes,
                            "root dense cached overlap Hamiltonian matrix");
        BandNonCol_AddBytes(&overlap_peak,compact_evec_bytes,
                            "root dense cached overlap eigenvectors");
        BandNonCol_AddBytes(&overlap_peak,scale_bytes,
                            "root dense cached overlap scale vector");

        BandNonCol_AddBytes(&hamiltonian_peak,compact_evec_bytes,
                            "root dense cached Hamiltonian eigenvectors");
        BandNonCol_AddBytes(&hamiltonian_peak,scale_bytes,
                            "root dense cached Hamiltonian scale vector");

        BandNonCol_AddBytes(&wavefunction_peak,
                            BandNonCol_CheckedMul(matrix_bytes,4U,
                                                  "root dense cached wavefunction n*n matrices"),
                            "root dense cached wavefunction n*n matrices");
        BandNonCol_AddBytes(&wavefunction_peak,scale_bytes,
                            "root dense cached wavefunction scale vector");
    }

    return BandNonCol_MaxBytes(overlap_peak,BandNonCol_MaxBytes(hamiltonian_peak,wavefunction_peak));
}

static size_t BandNonCol_RootDenseReserveBytes(size_t total_bytes, size_t required_bytes)
{
    size_t reserve = total_bytes/10U;
    size_t min_reserve = 512U*1024U*1024U;
    size_t required_margin = required_bytes/5U;

    if (reserve<min_reserve) reserve = min_reserve;
    if (reserve<required_margin) reserve = required_margin;

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
    reserve = BandNonCol_RootDenseReserveBytes(total_bytes,required);

    /* HIP/OpenMP target and GEMMul8 need scratch space beyond the explicitly
       estimated dense matrices. Keep one full group-sized cushion. */
    if (reserve<required) reserve = required;

    if (group_required!=NULL) *group_required = required;
    if (reserve_bytes!=NULL) *reserve_bytes = reserve;

    return (required<=free_bytes && reserve<=free_bytes-required);
}

static int BandNonCol_AutoGpuTurnLimit(int requested, int n, int n2, int MaxN, int size_H1,
                                       int owns_dense_rank, int myid0)
{
    if (BandNonCol_mgpu_host_dense) return 1;
    MPI_Comm node_comm;
    MPI_Comm device_comm = MPI_COMM_NULL;
    int local_limit = requested;
    int global_limit = requested;
    int hip_ok = 0;
    int hip_device = -1;
    size_t required_bytes = 0;
    size_t free_bytes = 0;
    size_t total_bytes = 0;

    (void)myid0;

    if (requested<=2){
        return requested;
    }

    if (owns_dense_rank){
        hipError_t hip_status = hipGetDevice(&hip_device);

        if (hip_status==hipSuccess){
            required_bytes = BandNonCol_RootDenseDeviceBytes(n,n2,MaxN,size_H1);
            hip_status = hipMemGetInfo(&free_bytes,&total_bytes);
        }

        if (hip_status==hipSuccess && 0<required_bytes){
            hip_ok = 1;
        }
        else {
            local_limit = 2;
        }
    }

    MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
    MPI_Comm_split(node_comm,hip_ok ? hip_device : MPI_UNDEFINED,0,&device_comm);
    MPI_Comm_free(&node_comm);

    if (hip_ok){
        int device_ranks = 0;

        MPI_Comm_size(device_comm,&device_ranks);

        {
            int concurrent_ranks = (device_ranks<requested) ? device_ranks : requested;
            size_t published_required = 0U;
            size_t published_reserve = 0U;
            int lacks_memory =
              !BandNonCol_GpuTurnMemoryFits(free_bytes,total_bytes,required_bytes,
                                            concurrent_ranks,NULL,NULL);

            /* Tell long-lived GPU caches how much room the full band
               diagonalization group needs before they decide residency. */
            (void)BandNonCol_GpuTurnMemoryFits(SIZE_MAX,total_bytes,required_bytes,
                                               concurrent_ranks,
                                               &published_required,
                                               &published_reserve);
            if (published_required!=SIZE_MAX &&
                published_required<=SIZE_MAX-published_reserve){
                OpenMX_GpuPhaseNeed_Register("band_noncol_vec",
                                             published_required+published_reserve);
            }

            if (lacks_memory){
                int trial_ranks = (device_ranks<2) ? device_ranks : 2;

                if (2<requested &&
                    BandNonCol_GpuTurnMemoryFits(free_bytes,total_bytes,required_bytes,
                                                 trial_ranks,NULL,NULL)){
                    local_limit = 2;
                }
                else {
                    local_limit = 1;
                }
            }
        }

        MPI_Comm_free(&device_comm);
    }

    MPI_Allreduce(&local_limit,&global_limit,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    return global_limit;
}

static int BandNonCol_RootDenseParallelKWorldsFit(int n, int n2, int MaxN, int size_H1,
                                                  int myid0, int myworld2, int Num_Comm_World2,
                                                  int *Comm_World_StartID2)
{
    MPI_Comm node_comm;
    MPI_Comm device_comm = MPI_COMM_NULL;
    int potential_owner;
    int selected_owner;
    int hip_ok = 0;
    int hip_device = -1;
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
    selected_owner = potential_owner && Set_Hamiltonian_OpenMP_Rank_Is_Selected();

    if (potential_owner && !selected_owner){
        if (myid0==Comm_World_StartID2[myworld2]){
            if (BandNonCol_GpuVerbose()) {
                fprintf(stderr,
                        "<Band>  Rank %d is a non-collinear dense GPU solver k-world owner, "
                        "but it is not selected for HIP/OpenMP target offload; using serialized k-worlds.\n",
                        myid0);
                fflush(stderr);
            }
        }
        local_fit = 0;
    }

    if (selected_owner){
        hipError_t hip_status;

        hip_status = hipGetDevice(&hip_device);
        if (hip_status==hipSuccess){
            required_bytes = BandNonCol_RootDenseDeviceBytes(n,n2,MaxN,size_H1);
            hip_status = hipMemGetInfo(&free_bytes,&total_bytes);
        }

        if (hip_status==hipSuccess){
            reserve_bytes = BandNonCol_RootDenseReserveBytes(total_bytes,required_bytes);
            hip_ok = 1;
        }
        else {
            if (2 <= level_stdout) {
                fprintf(stderr,
                        "<Band>  Rank %d failed to query GPU memory for non-collinear dense GPU solver (%s); "
                        "using serialized k-worlds.\n",
                        myid0,hipGetErrorString(hip_status));
                fflush(stderr);
            }
            local_fit = 0;
        }
    }

    MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
    MPI_Comm_split(node_comm,hip_ok ? hip_device : MPI_UNDEFINED,0,&device_comm);
    MPI_Comm_free(&node_comm);

    if (hip_ok){
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
            if (BandNonCol_GpuVerbose() && device_rank==0){
                printf("<Band>  Serializing non-collinear dense GPU solver k-worlds: GPU device %d is shared by %d owner rank(s), "
                       "free %.3f MiB, need %.3f MiB plus %.3f MiB reserve.\n",
                       hip_device,device_ranks,
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

static int BandNonCol_NativeGpuDiagFits(int n,int n2,int MaxN,int size_H1,int owns_dense)
{
    const char *env=getenv("OPENMX_BAND_GPU_DIAG");
    size_t required=BandNonCol_RootDenseDeviceBytes(n,n2,MaxN,size_H1);
    size_t free_bytes=0,total_bytes=0,reserve=0,fixed_all=0;
    int local_fit=1,fit=1,device_ranks;
    if (env!=NULL && atoi(env)==0) local_fit=0;
    if (required==0 || required==SIZE_MAX) local_fit=0;
    /* Every rank sharing the device keeps its own solver workspace and
       library context alive during the k-point solves, so their fixed
       costs add up however many of them solve at a time (the concurrency
       guard only spaces the matrix transients).  Collective: keep it
       before the owner-only check. */
    device_ranks=BandNonCol_DeviceRanks();
    fixed_all=BandNonCol_GpuLibraryReserveBytes()*(size_t)device_ranks;
    if (local_fit && owns_dense){
        if (hipMemGetInfo(&free_bytes,&total_bytes)!=hipSuccess) local_fit=0;
        else {
            reserve=BandNonCol_RootDenseReserveBytes(total_bytes,required);
            if (required>free_bytes || reserve>free_bytes-required ||
                fixed_all>free_bytes-required-reserve) local_fit=0;
            else OpenMX_GpuPhaseNeed_Register("band_noncol_ev",required+reserve+fixed_all);
            if (!local_fit){
                int myid;
                MPI_Comm_rank(mpi_comm_level1,&myid);
                if (myid==Host_ID){
                    const double MiB=1024.0*1024.0;
                    fprintf(stderr,
                            "Band_DFT_NonCol: GPU device is shared by %d rank(s), has %.3f MiB free "
                            "(%.3f MiB total), but %.3f MiB is required for the dense GPU diagonalization "
                            "(%.3f MiB library reserve per rank); switching to CPU path.\n",
                            device_ranks,(double)free_bytes/MiB,(double)total_bytes/MiB,
                            (double)(required+reserve+fixed_all)/MiB,
                            (double)BandNonCol_GpuLibraryReserveBytes()/MiB);
                    fflush(stderr);
                }
            }
        }
    }
    MPI_Allreduce(&local_fit,&fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);
    BandNonCol_gpu_dense_refused = !fit;
    return fit;
}

static int BandNonCol_GpuDiagFits(int n,int n2,int MaxN,int size_H1,int owns_dense)
{
    const char *enabled=getenv("OPENMX_BAND_GPU_DIAG");
    BandNonCol_mgpu_host_dense=0;
    if (enabled && atoi(enabled)==0 && !openmx_mgpu_host_required()) {
        BandNonCol_gpu_dense_refused=1;
        return 0;
    }
    int fit=openmx_mgpu_host_required()?0:BandNonCol_NativeGpuDiagFits(n,n2,MaxN,size_H1,owns_dense);
    if (!fit && !gpusolver2_flag) {
        BandNonCol_DenseGpu_ReleaseWaveBuffers();
        BandNonCol_DenseGpu_Destroy();
        openmx_gemmul8ReleaseWorkspaces();
        MPI_Barrier(mpi_comm_level1);
        int rank;
        MPI_Comm_rank(mpi_comm_level1,&rank);
        if (rank==Host_ID) {
            int total=openmx_mgpu_band_kpoint_count(SpinP_switch,way_of_kpoint,
                Kspace_grid1,Kspace_grid2,Kspace_grid3,num_non_eq_kpt);
            fit=openmx_mgpu_host_plan(total,n2,1)!=0;
            if (level_stdout>=1) {
                if (fit) printf("<OpenMX_MGPU> Band noncollinear n=%d: host-backed MAGMA capacity path.\n",n2);
                else printf("<OpenMX_MGPU> Band noncollinear n=%d: %s; CPU solve.\n",n2,openmx_mgpu_host_error());
                fflush(stdout);
            }
        }
        MPI_Bcast(&fit,1,MPI_INT,Host_ID,mpi_comm_level1);
        BandNonCol_mgpu_host_dense=fit;
    }
    BandNonCol_gpu_dense_refused=!fit;
    return fit;
}


static unsigned long long BandNonCol_HashInt(unsigned long long h, int value)
{
    h ^= (unsigned long long)(unsigned int)value;
    h *= 1099511628211ULL;
    return h;
}

static void BandNonCol_ConstructCache_Reset(void)
{
    BandNonColConstructCache *cache = &BandNonCol_construct_cache;

    if (cache->dense_device_valid){
        BandNonColConstructEntry *entries = cache->dense_entries;
        double *phase_r = cache->phase_r;
        double *phase_i = cache->phase_i;
        int dense_count = cache->dense_count;
        int phase_count = cache->phase_count;

        if (entries!=NULL && 0<dense_count){
#pragma omp target exit data map(delete: entries[0 : dense_count])
        }
        if (phase_r!=NULL && phase_i!=NULL && 0<phase_count){
#pragma omp target exit data map(delete: phase_r[0 : phase_count], phase_i[0 : phase_count])
        }
    }

    free(cache->phase_l1);
    free(cache->phase_l2);
    free(cache->phase_l3);
    free(cache->phase_r);
    free(cache->phase_i);
    free(cache->dense_entries);
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

static void BandNonCol_ConstructCache_Ensure(int *order_GA, int *MP, int n)
{
    BandNonColConstructCache *cache = &BandNonCol_construct_cache;
    unsigned long long fingerprint = BandNonCol_ConstructFingerprint(order_GA,MP);
    int dense_count = 0;
    int phase_count = 0;

    if (cache->valid && cache->n==n && cache->fingerprint==fingerprint) return;

    BandNonCol_ConstructCache_Reset();

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

    cache->dense_entries = (BandNonColConstructEntry*)malloc(sizeof(BandNonColConstructEntry)*(size_t)(dense_count+1));
    cache->phase_l1 = (int*)malloc(sizeof(int)*(size_t)(phase_count+1));
    cache->phase_l2 = (int*)malloc(sizeof(int)*(size_t)(phase_count+1));
    cache->phase_l3 = (int*)malloc(sizeof(int)*(size_t)(phase_count+1));
    cache->phase_r = (double*)malloc(sizeof(double)*(size_t)(phase_count+1));
    cache->phase_i = (double*)malloc(sizeof(double)*(size_t)(phase_count+1));

    if (cache->dense_entries==NULL || cache->phase_l1==NULL || cache->phase_l2==NULL ||
        cache->phase_l3==NULL || cache->phase_r==NULL || cache->phase_i==NULL){
        BandNonCol_ConstructCache_Reset();
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
   kernel while keeping its virtual range (the present-table key).  On an
   APU (MI300A) the OpenMP map is zero-copy -- the "device copy" IS these
   pages -- so there is no duplicate to reclaim and discarding would wipe
   the mapped data; only discrete GPUs take this path. */
static void BandNonCol_DiscardHostPages(void *ptr, size_t bytes)
{
    const size_t page = 4096;
    uintptr_t lo = ((uintptr_t)ptr + page - 1) & ~(uintptr_t)(page - 1);
    uintptr_t hi = ((uintptr_t)ptr + bytes) & ~(uintptr_t)(page - 1);

    if (openmx_gpu_is_apu()) return;
    if (ptr == NULL || hi <= lo) return;
    (void)madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
}

static void BandNonCol_ConstructCache_EnsureDenseDevice(void)
{
    BandNonColConstructCache *cache = &BandNonCol_construct_cache;

    if (cache->dense_device_valid) return;

    if (0<cache->dense_count){
        BandNonColConstructEntry *entries = cache->dense_entries;
        int dense_count = cache->dense_count;
#pragma omp target enter data map(to: entries[0 : dense_count])
        /* only the device kernels read the entry table from here on (the
           CPU construct fallback is unreachable once this path is live);
           the host mirror is ~100 MB per k-owner, so give the pages back */
        BandNonCol_DiscardHostPages(entries, sizeof(BandNonColConstructEntry)*(size_t)dense_count);
    }

    if (0<cache->phase_count){
        double *phase_r = cache->phase_r;
        double *phase_i = cache->phase_i;
        int phase_count = cache->phase_count;
#pragma omp target enter data map(alloc: phase_r[0 : phase_count], phase_i[0 : phase_count])
    }

    cache->dense_device_valid = 1;
}

static void BandNonCol_ConstructDenseMs_OpenMP(int cpx_flag, int n, double k1, double k2, double k3,
                                                const double *M2, dcomplex *Ms)
{
    BandNonColConstructCache *cache = &BandNonCol_construct_cache;
    BandNonColConstructEntry *entries = cache->dense_entries;
    double *phase_r = cache->phase_r;
    double *phase_i = cache->phase_i;
    int dense_count = cache->dense_count;
    int phase_count = cache->phase_count;
    int h_count = cache->h_count;
    int matrix_count = n*n;

    BandNonCol_ConstructCache_EnsureDenseDevice();

    for (int p=0; p<phase_count; p++){
        double kRn = k1*(double)cache->phase_l1[p]
                   + k2*(double)cache->phase_l2[p]
                   + k3*(double)cache->phase_l3[p];
        phase_i[p] = sin(2.0*PI*kRn);
        phase_r[p] = cos(2.0*PI*kRn);
    }

    if (0<phase_count){
#pragma omp target update to(phase_r[0 : phase_count], phase_i[0 : phase_count])
    }

    if (!BandNonCol_OmpTargetIsPresent(Ms,sizeof(dcomplex)*(size_t)matrix_count)){
#pragma omp target enter data map(alloc: Ms[0 : matrix_count])
    }

#pragma omp target data map(to: M2[0 : h_count]) map(present, alloc: Ms[0 : matrix_count], entries[0 : dense_count], phase_r[0 : phase_count], phase_i[0 : phase_count])
    {
#pragma omp target teams distribute parallel for
        for (int idx=0; idx<matrix_count; idx++){
            Ms[idx].r = 0.0;
            Ms[idx].i = 0.0;
        }

#pragma omp target teams distribute parallel for
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

#pragma omp atomic update
            Ms[dense_index].r += val_r;
#pragma omp atomic update
            Ms[dense_index].i += val_i;
        }
    }
}

static void BandNonCol_DenseGpuEigen(dcomplex *d_A, double *ko, int n, int maxn,
                                     int use_hipsolver, const char *where)
{
    int info;
    int mout = 0;
    int l;

    if (d_A==NULL || ko==NULL || n<=0 || maxn<=0 || n<maxn){
        BandNonCol_AbortWithMessage("Invalid dense GPU eigensolver arguments in Band_DFT_NonCol.c.");
    }

    {
        const int tdbg = BandNonCol_SolveTimingEnabled();
        double td0 = 0.0, td1 = 0.0;

        if (tdbg) dtime(&td0);

        if (use_hipsolver){
            info = openmx_hipsolver_zheevd_gpu(n,maxn,d_A,ko,&mout);
        }
        else {
            /* HIP-owned storage: keep the policy wrapper's native memory
               preflight and retry contract, even when it selects rocSOLVER. */
            info = openmx_magma_zheevdx_gpu(n,maxn,d_A,ko,&mout);
        }

        if (tdbg) {
            wait_hipfunc(hipDeviceSynchronize());
            dtime(&td1);
            printf("NC_SOLVE_TIMING(dense) zheevd=%.3f\n", td1 - td0);
            fflush(stdout);
        }
    }
    if (info!=0){
        fprintf(stderr,"%s: GPU eigensolver (zheevd/zheevdx) failed, info=%d\n",where,info);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1,1);
    }
    if (mout!=maxn){
        fprintf(stderr,"%s: GPU eigensolver returned %d eigenpairs, expected %d\n",
                where,mout,maxn);
        fflush(stderr);
        MPI_Abort(mpi_comm_level1,1);
    }

    for (l=maxn; 1<=l; l--) ko[l] = ko[l-1];
}

static void BandNonCol_ConstructDenseMsFromPacked_HIP(int cpx_flag, const double *M1, dcomplex *d_M,
                                                      int *order_GA, int *MP,
                                                      double k1, double k2, double k3, int n)
{
    BandNonColConstructCache *cache = &BandNonCol_construct_cache;

    if (M1==NULL || d_M==NULL){
        BandNonCol_AbortWithMessage("Invalid HIP dense matrix construction arguments in Band_DFT_NonCol.c.");
    }

    BandNonCol_ConstructCache_Ensure(order_GA,MP,n);
    cache = &BandNonCol_construct_cache;

    for (int p=0; p<cache->phase_count; p++){
        double kRn = k1*(double)cache->phase_l1[p]
                   + k2*(double)cache->phase_l2[p]
                   + k3*(double)cache->phase_l3[p];
        cache->phase_i[p] = sin(2.0*PI*kRn);
        cache->phase_r[p] = cos(2.0*PI*kRn);
    }

    if (BandNonCol_BuildDenseMs_HIP(cpx_flag,cache->dense_count,cache->h_count,
                                    cache->phase_count,n,cache->dense_entries,
                                    cache->phase_r,cache->phase_i,M1,d_M) != 0){
        BandNonCol_AbortWithMessage("Band_DFT_NonCol HIP dense matrix generation failed.");
    }
}

static void BandNonCol_DenseGpuPrepareOverlap(int rebuild_overlap, int n, int n2, double *ko,
                                              dcomplex *h_S, int cache_buffers)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;
    const size_t nn = (size_t)n*(size_t)n;

    BandNonCol_DenseGpu_Ensure(n,n2);

    if (rebuild_overlap){
        if (BandNonCol_SymmetrizeDenseHermitian_HIP(n,w->d_S) != 0){
            BandNonCol_AbortWithMessage("Band_DFT_NonCol HIP overlap symmetrization failed.");
        }

        if (cache_buffers){
            /* The eigensolver uses a separate synchronous workspace.  With
               ample device memory, retaining these matrices avoids freeing
               and recreating about a GiB per k-point owner. */
            wait_hipfunc(hipDeviceSynchronize());
        }
        else {
            BandNonCol_DenseGpu_ReleaseOverlapMagmaScratch();
        }
        BandNonCol_DenseGpuEigen(w->d_S,ko,n,n,openmx_gpu_eigensolver_use_hipsolver(),
                                 "Band_DFT_NonCol root dense overlap");

        for (int l=1; l<=n; l++){
            if (ko[l]<1.0e-10) ko[l] = 1.0e-10;
            w->h_scale[l-1].r = 1.0/sqrt(ko[l]);
            w->h_scale[l-1].i = 0.0;
        }

        wait_hipfunc(hipMemcpy(w->d_scale,w->h_scale,sizeof(dcomplex)*(size_t)n,hipMemcpyHostToDevice));
        BandNonCol_DenseGpu_EnsureHipblas();
        if (w->d_tmp==NULL){
            wait_hipfunc(hipMalloc((void**)&w->d_tmp,sizeof(dcomplex)*nn));
        }
        if (!openmx_gpu_transform ||
            !openmx_gpu_transform(OPENMX_GPU_SCALE_COLUMNS,1,n,n,w->d_S,n,w->d_scale,0,w->d_tmp,n,1,0.0))
        wait_hipfunc(hipblasZdgmm(w->hipblas,HIPBLAS_SIDE_RIGHT,n,n,
                                  (const hipDoubleComplex*)w->d_S,n,
                                  (const hipDoubleComplex*)w->d_scale,1,
                                  (hipDoubleComplex*)w->d_tmp,n));
        wait_hipfunc(hipMemcpy(w->d_S,w->d_tmp,sizeof(dcomplex)*nn,hipMemcpyDeviceToDevice));
        wait_hipfunc(hipMemcpy(h_S,w->d_S,sizeof(dcomplex)*nn,hipMemcpyDeviceToHost));
    }
    else {
        wait_hipfunc(hipMemcpy(w->d_S,h_S,sizeof(dcomplex)*nn,hipMemcpyHostToDevice));
    }
}

static void BandNonCol_DenseGpuTripleTransform(int n, dcomplex *d_A)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;
    const hipDoubleComplex alpha = make_hipDoubleComplex(1.0,0.0);
    const hipDoubleComplex beta  = make_hipDoubleComplex(0.0,0.0);

    BandNonCol_DenseGpu_EnsureHipblas();
    wait_hipfunc(openmx_gemmul8Zgemm(w->hipblas,HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,&alpha,
                                      (const hipDoubleComplex*)d_A,n,
                                      (const hipDoubleComplex*)w->d_S,n,&beta,
                                      (hipDoubleComplex*)w->d_tmp,n));
    wait_hipfunc(openmx_gemmul8Zgemm(w->hipblas,HIPBLAS_OP_C,HIPBLAS_OP_N,n,n,n,&alpha,
                                      (const hipDoubleComplex*)w->d_S,n,
                                      (const hipDoubleComplex*)w->d_tmp,n,&beta,
                                      (hipDoubleComplex*)d_A,n));
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

static void BandNonCol_DMEntryCache_Reset(void)
{
    free(BandNonCol_dm_entry_cache.basis0);
    free(BandNonCol_dm_entry_cache.basis1);
    free(BandNonCol_dm_entry_cache.phase_index);
    free(BandNonCol_dm_entry_cache.pair_l1);
    free(BandNonCol_dm_entry_cache.pair_l2);
    free(BandNonCol_dm_entry_cache.pair_l3);
    free(BandNonCol_dm_entry_cache.phase_r);
    free(BandNonCol_dm_entry_cache.phase_i);
    memset(&BandNonCol_dm_entry_cache,0,sizeof(BandNonCol_dm_entry_cache));
}

static void BandNonCol_DMEntryCache_Ensure(int *MP, int n)
{
    BandNonColDMEntryCache *cache = &BandNonCol_dm_entry_cache;
    unsigned long long fingerprint = BandNonCol_DMLayoutFingerprint(MP,n);
    int entry_count = 0;
    int pair_count = 0;

    if (cache->valid && cache->n==n && cache->fingerprint==fingerprint) return;

    BandNonCol_DMEntryCache_Reset();

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

    cache->basis0 = (int*)malloc(sizeof(int)*(size_t)(entry_count+1));
    cache->basis1 = (int*)malloc(sizeof(int)*(size_t)(entry_count+1));
    cache->phase_index = (int*)malloc(sizeof(int)*(size_t)(entry_count+1));
    cache->pair_l1 = (int*)malloc(sizeof(int)*(size_t)(pair_count+1));
    cache->pair_l2 = (int*)malloc(sizeof(int)*(size_t)(pair_count+1));
    cache->pair_l3 = (int*)malloc(sizeof(int)*(size_t)(pair_count+1));
    cache->phase_r = (double*)malloc(sizeof(double)*(size_t)(pair_count+1));
    cache->phase_i = (double*)malloc(sizeof(double)*(size_t)(pair_count+1));

    if (cache->basis0==NULL || cache->basis1==NULL || cache->phase_index==NULL ||
        cache->pair_l1==NULL || cache->pair_l2==NULL || cache->pair_l3==NULL ||
        cache->phase_r==NULL || cache->phase_i==NULL){
        BandNonCol_DMEntryCache_Reset();
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

static void BandNonCol_UpdateDMEntryPhases(double k1, double k2, double k3)
{
    BandNonColDMEntryCache *cache = &BandNonCol_dm_entry_cache;

    for (int pair=0; pair<cache->pair_count; pair++){
        double kRn = k1*(double)cache->pair_l1[pair]
                   + k2*(double)cache->pair_l2[pair]
                   + k3*(double)cache->pair_l3[pair];
        cache->phase_i[pair] = sin(2.0*PI*kRn);
        cache->phase_r[pair] = cos(2.0*PI*kRn);
    }
}

static void BandNonCol_DMOccWorkspace_Ensure(int nk)
{
    BandNonColDMOccWorkspace *ws = &BandNonCol_dm_occ_workspace;

    if (nk<=0) BandNonCol_AbortWithMessage("Invalid DM occupation workspace size in Band_DFT_NonCol.c.");
    if (ws->max_occ_nk>=nk) return;

    free(ws->occ);
    free(ws->eig_occ);

    ws->occ = (double*)malloc(sizeof(double)*(size_t)nk);
    ws->eig_occ = (double*)malloc(sizeof(double)*(size_t)nk);

    if (ws->occ==NULL || ws->eig_occ==NULL){
        free(ws->occ);
        free(ws->eig_occ);
        ws->occ = NULL;
        ws->eig_occ = NULL;
        ws->max_occ_nk = 0;
        BandNonCol_AbortWithMessage("Failed to allocate DM occupation workspace in Band_DFT_NonCol.c.");
    }

    ws->max_occ_nk = nk;
}

static int BandNonCol_BuildOccupationWeightsDense(int kmin, int kmax, const double *eigen, double *occ, double *eig_occ)
{
    const double max_x = 60.0;
    const double FermiEps = 1.0e-13;
    int nk = 0;

    if (kmax<kmin) return 0;

    for (int k=kmin; k<=kmax; k++){
        double x = (eigen[k] - ChemP)*Beta;
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

/* GPU per-k DM accumulation for the legacy multi-k second loop, i.e. the
   ScaLAPACK/ELPA fallback (the k-dense GPU flow already accumulates its DM
   on the device through AccumulateDMRootDenseK).  The eigenvector panel
   still fits on the device even when the preflight refused the full dense
   solve.  Sizes below the noncollinear band threshold keep the CPU loop.
   Opt out with OPENMX_BAND_NONCOL_GPU_DM=0. */
static int BandNonCol_UseGpuFallbackDM(int n2, size_t evec_bytes)
{
    const char *value = getenv("OPENMX_BAND_NONCOL_GPU_DM");
    size_t free_bytes = 0, total_bytes = 0;

    if (value != NULL && atoi(value) == 0) return 0;
    if (!(scf_eigen_lib_flag == GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum() <= n2)) return 0;

    /* The device upload (an OpenMP target map) aborts instead of failing
       softly, so refuse up front: every rank sharing the device maps its
       own panel at about the same time and pays the library reserve for
       the kernel, hence the device-wide sum. */
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        (void)hipGetLastError();
        return 0;
    }
    if (evec_bytes > SIZE_MAX / (size_t)BandNonCol_DeviceRanksCached()) return 0;
    return ((evec_bytes + BandNonCol_GpuLibraryReserveBytes()) * (size_t)BandNonCol_DeviceRanksCached() <= free_bytes);
}

static void BandNonCol_AccumulateDMKPoint_OpenMP(int myid2, int *is2, int *ie2, int *MP,
                                                  int n, int n2, int size_H1,
                                                  double k1, double k2, double k3, const double *ko,
                                                  const dcomplex *EVec1,
                                                  double *rDM11, double *rDM22, double *rDM12,
                                                  double *iDM12, double *iDM11, double *iDM22,
                                                  double *rEDM11, double *rEDM22,int active,MPI_Comm dm_comm)
{
    BandNonColDMEntryCache *cache=&BandNonCol_dm_entry_cache;
    BandNonColDMOccWorkspace *ws=&BandNonCol_dm_occ_workspace;
    int stride=active?ie2[myid2]-is2[myid2]+1:0,nk=0;
    if(stride>0){
        BandNonCol_DMOccWorkspace_Ensure(stride);
        nk=BandNonCol_BuildOccupationWeightsDense(is2[myid2],ie2[myid2],ko,ws->occ,ws->eig_occ);
        BandNonCol_DMEntryCache_Ensure(MP,n);BandNonCol_UpdateDMEntryPhases(k1,k2,k3);
        if(cache->entry_count!=size_H1)BandNonCol_AbortWithMessage("DM entry count mismatch");
    }
    const void *vectors=NULL;int host=0;
    if(nk){vectors=EVec1;host=1;}
    double *outputs[10]={rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,rEDM11,rEDM22};
    openmx_gpu_dm_collective(dm_comm,OPENMX_GPU_DM_NC,n,nk,stride,nk?cache->entry_count:0,
        vectors,host,cache->basis0,cache->basis1,nk?cache->phase_index:NULL,cache->pair_count,
        cache->phase_r,cache->phase_i,ws->occ,ws->eig_occ,NULL,outputs,1);
}

static void BandNonCol_AccumulateDMRootDense_OpenMP(int *MP, int n, int n2, int max_state, int size_H1,
                                                     double k1, double k2, double k3, const double *ko,
                                                     const dcomplex *dense_evec,
                                                     double *rDM11, double *rDM22, double *rDM12,
                                                     double *iDM12, double *iDM11, double *iDM22,
                                                     double *rEDM11, double *rEDM22)
{
    BandNonColDMEntryCache *cache;
    BandNonColDMOccWorkspace *ws = &BandNonCol_dm_occ_workspace;
    const int *basis0;
    const int *basis1;
    const int *phase_index;
    const double *phase_r;
    const double *phase_i;
    const double *occ;
    const double *eig_occ;
    int nk;
    int entry_count;
    int pair_count;
    size_t evec_count;

    if (max_state<=0 || dense_evec==NULL) return;

    BandNonCol_DMOccWorkspace_Ensure(max_state);
    nk = BandNonCol_BuildOccupationWeightsDense(1,max_state,ko,ws->occ,ws->eig_occ);

    if (nk<=0) return;

    BandNonCol_DMEntryCache_Ensure(MP,n);
    BandNonCol_UpdateDMEntryPhases(k1,k2,k3);

    cache = &BandNonCol_dm_entry_cache;

    if (cache->entry_count!=size_H1){
        BandNonCol_AbortWithMessage("DM entry cache size mismatch in Band_DFT_NonCol.c.");
    }

    basis0 = cache->basis0;
    basis1 = cache->basis1;
    phase_index = cache->phase_index;
    phase_r = cache->phase_r;
    phase_i = cache->phase_i;
    occ = ws->occ;
    eig_occ = ws->eig_occ;
    entry_count = cache->entry_count;
    pair_count = cache->pair_count;
    evec_count = (size_t)n2*(size_t)n2;

#pragma omp target data map(to: dense_evec[0:evec_count], phase_r[0:pair_count], phase_i[0:pair_count], occ[0:nk], eig_occ[0:nk])
    {
        const int dm_chunk_size = 131072;

        for (int offset=0; offset<entry_count; offset+=dm_chunk_size){
            const int chunk_count = (entry_count-offset<dm_chunk_size) ? (entry_count-offset) : dm_chunk_size;
            const int *basis0_chunk = basis0 + offset;
            const int *basis1_chunk = basis1 + offset;
            const int *phase_index_chunk = phase_index + offset;
            double *rDM11_chunk = rDM11 + offset;
            double *rDM22_chunk = rDM22 + offset;
            double *rDM12_chunk = rDM12 + offset;
            double *iDM12_chunk = iDM12 + offset;
            double *iDM11_chunk = iDM11 + offset;
            double *iDM22_chunk = iDM22 + offset;
            double *rEDM11_chunk = rEDM11 + offset;
            double *rEDM22_chunk = rEDM22 + offset;

/* "tofrom" (not "from") + "+=" below: the kernel ACCUMULATES on top of the
   running sums the callers carry (the legacy multi-k loop across its k
   points, like Calc_DM_Band_non_collinear with store_flag=0; the dense
   paths zero the arrays before their first call). */
#pragma omp target data map(to: basis0_chunk[0:chunk_count], basis1_chunk[0:chunk_count], phase_index_chunk[0:chunk_count]) \
                        map(tofrom: rDM11_chunk[0:chunk_count], rDM22_chunk[0:chunk_count], rDM12_chunk[0:chunk_count], \
                                    iDM12_chunk[0:chunk_count], iDM11_chunk[0:chunk_count], iDM22_chunk[0:chunk_count], \
                                    rEDM11_chunk[0:chunk_count], rEDM22_chunk[0:chunk_count])
            {
#pragma omp target teams distribute parallel for
                for (int p=0; p<chunk_count; p++){
                    const int ia = basis0_chunk[p];
                    const int ib = basis1_chunk[p];
                    const int ph = phase_index_chunk[p];
                    const double co = phase_r[ph];
                    const double si = phase_i[ph];
                    double dm11_r = 0.0, dm11_i = 0.0;
                    double dm22_r = 0.0, dm22_i = 0.0;
                    double dm12_r = 0.0, dm12_i = 0.0;
                    double edm11_r = 0.0, edm11_i = 0.0;
                    double edm22_r = 0.0, edm22_i = 0.0;

                    for (int k=0; k<nk; k++){
                        const double w = occ[k];
                        const double ew = eig_occ[k];
                        const dcomplex va_up = dense_evec[(size_t)ia*(size_t)n2 + (size_t)k];
                        const dcomplex vb_up = dense_evec[(size_t)ib*(size_t)n2 + (size_t)k];
                        const dcomplex va_dn = dense_evec[(size_t)(ia+n)*(size_t)n2 + (size_t)k];
                        const dcomplex vb_dn = dense_evec[(size_t)(ib+n)*(size_t)n2 + (size_t)k];
                        const double re11 = va_up.r*vb_up.r + va_up.i*vb_up.i;
                        const double im11 = va_up.r*vb_up.i - va_up.i*vb_up.r;
                        const double re22 = va_dn.r*vb_dn.r + va_dn.i*vb_dn.i;
                        const double im22 = va_dn.r*vb_dn.i - va_dn.i*vb_dn.r;
                        const double re12 = va_up.r*vb_dn.r + va_up.i*vb_dn.i;
                        const double im12 = va_up.r*vb_dn.i - va_up.i*vb_dn.r;

                        dm11_r += w*re11;
                        dm11_i += w*im11;
                        dm22_r += w*re22;
                        dm22_i += w*im22;
                        dm12_r += w*re12;
                        dm12_i += w*im12;
                        edm11_r += ew*re11;
                        edm11_i += ew*im11;
                        edm22_r += ew*re22;
                        edm22_i += ew*im22;
                    }

                    rDM11_chunk[p]  += co*dm11_r - si*dm11_i;
                    iDM11_chunk[p]  += co*dm11_i + si*dm11_r;
                    rDM22_chunk[p]  += co*dm22_r - si*dm22_i;
                    iDM22_chunk[p]  += co*dm22_i + si*dm22_r;
                    rDM12_chunk[p]  += co*dm12_r - si*dm12_i;
                    iDM12_chunk[p]  += co*dm12_i + si*dm12_r;
                    rEDM11_chunk[p] += co*edm11_r - si*edm11_i;
                    rEDM22_chunk[p] += co*edm22_r - si*edm22_i;
                }
            }
        }
    }
}

static int BandNonCol_DMCapacity(int n,int nk,int stride,int size_H1,
    const dcomplex *vectors,int host,BandNonColDMEntryCache *cache,
    BandNonColDMOccWorkspace *ws,double *buffer)
{
    double *outputs[10]={0};
    for (int c=0;c<8;c++) outputs[c]=buffer+(size_t)c*size_H1;
    return openmx_gpu_dm_entries(OPENMX_GPU_DM_NC,n,nk,stride,size_H1,vectors,host,
        cache->basis0,cache->basis1,cache->phase_index,cache->pair_count,cache->phase_r,cache->phase_i,
        ws->occ,ws->eig_occ,NULL,outputs,0);
}

static void BandNonCol_AccumulateDMRootDenseK_HIP(int size_H1, int *MP, int n, int n2, int MaxN,
                                                  double k1, double k2, double k3,
                                                  double *ko, const dcomplex *dense_evec,
                                                  const dcomplex *d_dense_evec, int dense_evec_ld,
                                                  double *rDM11, double *rDM22, double *rDM12,
                                                  double *iDM12, double *iDM11, double *iDM22,
                                                  double *rEDM11, double *rEDM22)
{
    BandNonColDMEntryCache *cache=&BandNonCol_dm_entry_cache;
    BandNonColDMOccWorkspace *ws=&BandNonCol_dm_occ_workspace;
    int nk=0,done=0;
    double *buffer=NULL;
    double *outputs[10]={rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,rEDM11,rEDM22};
    if(dense_evec || d_dense_evec){
        if(d_dense_evec && dense_evec_ld<=0)BandNonCol_AbortWithMessage("Invalid NC eigenvector stride");
        BandNonCol_DMOccWorkspace_Ensure(MaxN);
        nk=BandNonCol_BuildOccupationWeightsDense(1,MaxN,ko,ws->occ,ws->eig_occ);
        BandNonCol_DMEntryCache_Ensure(MP,n);BandNonCol_UpdateDMEntryPhases(k1,k2,k3);
        if(cache->entry_count!=size_H1)BandNonCol_AbortWithMessage("NC DM entry count mismatch");
    }
    if(nk && !BandNonCol_mgpu_host_dense && !openmx_gpu_capacity_test_recovery()){
        buffer=calloc((size_t)size_H1*8,sizeof(double));
        if(buffer)done=(d_dense_evec ?
            BandNonCol_CalcDMRootDenseDevice_HIP(cache->entry_count,cache->pair_count,size_H1,
                n,n2,nk,cache->basis0,cache->basis1,cache->phase_index,cache->phase_r,cache->phase_i,
                ws->occ,ws->eig_occ,d_dense_evec,dense_evec_ld,buffer):
            BandNonCol_CalcDMRootDense_HIP(cache->entry_count,cache->pair_count,size_H1,
                n,n2,nk,cache->basis0,cache->basis1,cache->phase_index,cache->phase_r,cache->phase_i,
                ws->occ,ws->eig_occ,dense_evec,buffer))==0;
        if(done)for(int c=0;c<8;c++)for(int i=0;i<size_H1;i++)outputs[c][i]+=buffer[(size_t)c*size_H1+i];
        free(buffer);
    }
    openmx_gpu_dm_collective(mpi_comm_level1,OPENMX_GPU_DM_NC,n,done?0:nk,
        d_dense_evec?dense_evec_ld:n2,(!done && nk)?cache->entry_count:0,
        d_dense_evec?d_dense_evec:dense_evec,d_dense_evec==NULL,cache->basis0,cache->basis1,
        (!done && nk)?cache->phase_index:NULL,cache->pair_count,cache->phase_r,cache->phase_i,
        ws->occ,ws->eig_occ,NULL,outputs,1);
}

static void BandNonCol_CalcDMRootDenseAllK1_HIP(int myid0, int owns_root_dense, int size_H1,
                                                int *MP, int n, int n2, int MaxN,
                                                double k1, double k2, double k3,
                                                double *****CDM, double *****iDM0, double *****EDM,
                                                double *ko, dcomplex *dense_evec,
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

    BandNonCol_AccumulateDMRootDenseK_HIP(size_H1,MP,n,n2,MaxN,k1,k2,k3,ko,
        owns_root_dense?dense_evec:NULL,NULL,0,rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,rEDM11,rEDM22);

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



static void BandNonCol_CalcDMAllK1_OpenMP(int myid0, int myid2, int size_H1,
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

    BandNonCol_AccumulateDMKPoint_OpenMP(myid2,is2,ie2,MP,n,n2,size_H1,k1,k2,k3,ko,EVec1,
                                          rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,rEDM11,rEDM22,1,mpi_comm_level1);

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

typedef struct
{
    int       valid;
    int       s_valid;
    int       cs2_valid;
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

static BandNonColRootDenseWorkspace BandNonCol_root_dense_workspace = {0};

static void BandNonCol_RootDenseWorkspace_Reset(void)
{
    BandNonColRootDenseWorkspace *ws = &BandNonCol_root_dense_workspace;

    free(ws->s_all);
    free(ws->h11);
    free(ws->h22);
    free(ws->h12);
    free(ws->work);
    free(ws->hs2);
    free(ws->cs2);
    memset(ws,0,sizeof(*ws));
}

/* Hard reset for a run boundary (between -runtest/-runtestL inputs): drop
   every device buffer, handle and geometry-derived table of this module so
   no validity check can accept data of the previous input after its host
   arrays were freed and their addresses reused.  Idempotent. */
void Band_DFT_NonCol_Release_GPU_Caches(void)
{
    BandNonCol_mgpu_host_dense=0;
    BandNonCol_RefineRelease();
    /* the refinement's stage (iterations, a cycle or run turned to FP64)
       starts over with the next input, read again by every rank */
    BandNonCol_RefineResetStage();
    BandNonCol_KOverlapReset();
    BandNonCol_ConstructCache_Reset();
    BandNonCol_DMEntryCache_Reset();
    BandNonCol_RootDenseWorkspace_Reset();
    BandNonCol_DenseGpu_ReleaseWaveBuffers();
    BandNonCol_DenseGpu_ReleaseOverlapMagmaScratch();
    BandNonCol_GpuSolver_ReleaseDeviceWorkspace();
    BandNonCol_GpuSolver_Destroy();
    BandNonCol_DMGpu_Destroy();
    BandNonCol_DenseGpu_Destroy();
}

static BandNonColRootDenseWorkspace *BandNonCol_RootDenseWorkspace_Ensure(int owns_root_dense,
                                                                           int n, int n2, int MaxN,
                                                                           int T_knum, int SCF_iter)
{
    BandNonColRootDenseWorkspace *ws = &BandNonCol_root_dense_workspace;
    size_t nn;
    size_t n2n2;

    if (!owns_root_dense) return ws;

    if (!ws->valid || ws->n!=n || ws->n2!=n2 || ws->maxn!=MaxN || ws->t_knum!=T_knum){
        BandNonCol_RootDenseWorkspace_Reset();

        nn = (size_t)n*(size_t)n;
        n2n2 = (size_t)n2*(size_t)n2;

        ws->s_all = (dcomplex*)malloc(sizeof(dcomplex)*nn*(size_t)T_knum);
        ws->h11   = (dcomplex*)malloc(sizeof(dcomplex)*nn);
        ws->h22   = (dcomplex*)malloc(sizeof(dcomplex)*nn);
        ws->h12   = (dcomplex*)malloc(sizeof(dcomplex)*nn);
        ws->work  = (dcomplex*)malloc(sizeof(dcomplex)*nn);
        ws->hs2   = (dcomplex*)malloc(sizeof(dcomplex)*n2n2);
        ws->cs2   = (dcomplex*)malloc(sizeof(dcomplex)*n2n2);

        if (ws->s_all==NULL || ws->h11==NULL || ws->h22==NULL || ws->h12==NULL ||
            ws->work==NULL || ws->hs2==NULL || ws->cs2==NULL){
            BandNonCol_RootDenseWorkspace_Reset();
            BandNonCol_AbortWithMessage("Failed to allocate root dense GPU solver workspace in Band_DFT_NonCol.c.");
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

/* The back transform of the first MaxN vectors in d_H2 into d_C2, and from
   there into the host: rdw->cs2 (ld n2) with copy_evec_to_host, or, given
   evec_panel, the k-point eigenvector cache's panel (n2 x MaxN, ld MaxN),
   which is d_C2's own layout, in one copy. */
static void BandNonCol_RootDenseBacktransform_HIP(int n, int n2, int MaxN,
                                                   int copy_evec_to_host,
                                                   BandNonColRootDenseWorkspace *rdw,
                                                   dcomplex *evec_panel)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;
    const hipDoubleComplex alpha = make_hipDoubleComplex(1.0,0.0);
    const hipDoubleComplex beta  = make_hipDoubleComplex(0.0,0.0);
    const int cache_buffers = BandNonCol_DenseBufferCacheEnabled();

    if (rdw==NULL || !rdw->valid || rdw->s_all==NULL ||
        !w->initialized || w->d_H2==NULL){
        BandNonCol_AbortWithMessage("Deferred root dense eigenvectors are not available for backtransform.");
    }

    BandNonCol_DenseGpu_RestoreS(n,rdw->s_all);
    BandNonCol_DenseGpu_EnsureWaveBuffers(n2,MaxN);

    /* S2 is block diagonal, diag(S,S).  Multiplying the two spin blocks
       independently halves the back-transform work and avoids materializing
       the n2-by-n2 S2 matrix.  d_C2 uses a compact MaxN leading dimension;
       an optional host copy preserves the legacy padded layout consumed by
       the density-matrix and fallback distribution paths. */
    BandNonCol_DenseGpu_EnsureHipblas();
    wait_hipfunc(openmx_gemmul8Zgemm(w->hipblas,HIPBLAS_OP_T,HIPBLAS_OP_T,MaxN,n,n,&alpha,
                                      (const hipDoubleComplex*)w->d_H2,n2,
                                      (const hipDoubleComplex*)w->d_S,n,&beta,
                                      (hipDoubleComplex*)w->d_C2,MaxN));
    wait_hipfunc(openmx_gemmul8Zgemm(w->hipblas,HIPBLAS_OP_T,HIPBLAS_OP_T,MaxN,n,n,&alpha,
                                      (const hipDoubleComplex*)(w->d_H2 + n),n2,
                                      (const hipDoubleComplex*)w->d_S,n,&beta,
                                      (hipDoubleComplex*)(w->d_C2 + (size_t)n*(size_t)MaxN),MaxN));
    if (evec_panel!=NULL){
        wait_hipfunc(hipMemcpy(evec_panel,w->d_C2,sizeof(dcomplex)*(size_t)n2*(size_t)MaxN,
                               hipMemcpyDeviceToHost));
    }
    else if (copy_evec_to_host){
        wait_hipfunc(hipMemcpy2D(rdw->cs2,sizeof(dcomplex)*(size_t)n2,
                                  w->d_C2,sizeof(dcomplex)*(size_t)MaxN,
                                  sizeof(dcomplex)*(size_t)MaxN,(size_t)n2,
                                  hipMemcpyDeviceToHost));
        rdw->cs2_valid = 1;
    }
    else {
        /* Match the synchronization formerly supplied by the blocking D2H
           copy before exposing d_C2 to the later density-matrix phase. */
        wait_hipfunc(hipDeviceSynchronize());
    }
    w->d_C2_valid = 1;
    w->d_C2_n2 = n2;
    w->d_C2_ld = MaxN;
    if (!cache_buffers){
        BandNonCol_DenseGpu_ReleaseWaveBuffers();
        BandNonCol_DenseGpu_Destroy();
    }
}

/* ------------------------------------------------------------------ */
/* The refined FP32 eigensolver (eigen_refine_gpu.cu) on the dense GPU
   paths of the SCF solves.  OPENMX_EIGEN_REFINE=K (0: off), unset: on by
   default on GPUs whose FP64 throughput is a small fraction of their FP32
   one when the complex GEMMul8 products are on; the global root's device
   decides for all ranks.  The refinement's occupations need the chemical
   potential of all k-points, which the solver only knows once the
   eigenvalues are gathered:
   - one k-point per k-world with every owner in one GPU turn group and room
     on the device for all of their refined solves at once (resident): every
     owner solves in FP32 (or starts from its previous vectors), the
     estimates are gathered, a chemical potential found on them, and the
     owners finish their refinements (two halves of one solve), from the
     first SCF step on (whose failures do not condemn the cycle);
   - the same path in serialized GPU turn groups (staged) and the k-dense
     path: pass 1 refines every k-point with the previous step's chemical
     potential (the eigenvalues are second order in the vectors' error) and
     keeps its basis on the host, pass 2 starts from that basis with the
     step's chemical potential and builds the vectors, from the second SCF
     step on.  Staged, pass 2 follows pass 1 in the same serialized GPU turn
     groups; on the k-dense path it is the density-matrix pass.
   Between the SCF steps the basis of every path waits in the host slot of
   its k-point: the refinement's device buffers go after every k-point, on
   the resident path too (whose NVIDIA original keeps the basis on the
   device), since the turn groups of the next step are sized on the free
   memory of the shared device.  An SCF that ends on a refined step is
   solved once more in FP64 (Band_DFT_NonCol_RefineForceFP64 around a
   repeated call, which recomputes the density matrices), and the last step
   scf.maxIter allows is FP64 in any case.  Ported from the NVIDIA version
   (openmx4.0_gpu, Band_DFT_NonCol.c). */
typedef struct
{
    int     iterations;    /* -1 until the environment is read */
    double  until;         /* printed NormRD below which the FP64 eigensolver takes over; 0 off */
    int     final_stage;   /* the FP64 eigensolver for the rest of the SCF cycle */
    int     last_refined;  /* the latest SCF eigensolve was a refined one (any k-point); from BeginStep to
                              EndStep: this call refines the resident path */
    int     cycle_refined; /* some eigensolve of this SCF cycle was */
    int     defaulted;     /* on because of the device, not of OPENMX_EIGEN_REFINE */
    int     force_fp64;    /* Band_DFT_NonCol_RefineForceFP64: this call in FP64 whatever the stage */
    int     transient;     /* the first step of a cycle: its failures do not condemn the cycle */
    int     pending;       /* this rank's k-point of the resident path: prepared, waiting for the chemical
                              potential */
    int     solved;        /* a k-point of this rank was refined this step */
    int     persistent;    /* the strongest persistent failure of this step on this rank */
    int     kdense;        /* this call refines the k-dense path */
    int     staged;        /* this call refines the one-k-point-per-k-world path in serialized turn groups */
    int     scf_iter;      /* the step of the call */
} BandNonColRefineState;

/* per k-point: the basis of the latest refined solve on the host (the device
   buffers go with every k-point); the budget is OPENMX_BAND_REFINE_BASIS_MB
   per rank (default 1024), at most 1/8 of the node's available memory over
   its ranks */
typedef struct
{
    dcomplex *basis;       /* n2 x n2, host */
    double   *lam;         /* n2 eigenvalue estimates */
    int       valid;       /* basis and lam hold the latest refined solve */
    int       refined;     /* pass 1 of this step refined the k-point */
} BandNonColKRefineSlot;

typedef struct
{
    double mu;
} BandNonColOccupationCtx;

/* the device side of a refined solve, allocated per k-point */
typedef struct
{
    void    *d_scratch;    /* the FP32 solve, then the product blocks */
    size_t   scratch_bytes;
    double  *d_w;          /* n2 refined eigenvalues */
    int32_t *d_info;
    void    *d_work;       /* the FP32 eigensolver's workspace */
    size_t   d_work_bytes;
} BandNonColRefineBuffers;

static BandNonColKRefineSlot  *BandNonCol_kref = NULL;
static int                     BandNonCol_kref_slots = 0, BandNonCol_kref_n2 = 0;
static size_t                  BandNonCol_kref_used = 0;
static const BandNonColRefineState BandNonCol_refine_initial = {-1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static BandNonColRefineState   BandNonCol_refine = {-1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static EigenRefineState        BandNonCol_refine_state;
static EigenRefineDevice       BandNonCol_refine_dev;
static EigenRefineProblem      BandNonCol_refine_pb;         /* the solve in progress, prepare to finish */
static EigenRefineReport       BandNonCol_refine_rep;        /* its report */
static BandNonColOccupationCtx BandNonCol_refine_occ = {0.0};
static BandNonColRefineBuffers BandNonCol_refine_buf;
static dcomplex               *BandNonCol_refine_x = NULL;   /* the device basis, n2 x n2 */
static int                     BandNonCol_refine_x_n = 0;
static double                 *BandNonCol_refine_e0 = NULL;  /* the estimates of the solve in progress, n2 */
static int                     BandNonCol_refine_e0_n = 0;

static hipError_t BandNonCol_TryDeviceMalloc(void **ptr, size_t bytes)
{
    hipError_t status = hipMalloc(ptr, (bytes == 0) ? 1 : bytes);

    if (status != hipSuccess) {
        *ptr = NULL;
        (void)hipGetLastError();
    }
    return status;
}

static void *BandNonCol_MallocArray(size_t count, size_t elem_size, const char *label)
{
    void *ptr = malloc((count * elem_size == 0) ? 1 : count * elem_size);

    if (ptr == NULL) {
        char msg[256];

        snprintf(msg, sizeof(msg), "Failed to allocate %s (%zu bytes) in Band_DFT_NonCol.c.", label,
                 count * elem_size);
        BandNonCol_AbortWithMessage(msg);
    }
    return ptr;
}

static void BandNonCol_RefineFreeDevice(void **ptr)
{
    if (*ptr != NULL) {
        wait_hipfunc(hipFree(*ptr));
        *ptr = NULL;
    }
}

/* the device buffers of a refined solve go after every k-point (the basis
   waits in the host slot); the products' unblocked scratch with them */
static void BandNonCol_RefineDropDevice(void)
{
    BandNonColRefineBuffers *b = &BandNonCol_refine_buf;

    BandNonCol_RefineFreeDevice(&b->d_scratch);
    BandNonCol_RefineFreeDevice((void **)&b->d_w);
    BandNonCol_RefineFreeDevice((void **)&b->d_info);
    BandNonCol_RefineFreeDevice(&b->d_work);
    b->scratch_bytes = 0;
    b->d_work_bytes = 0;
    BandNonCol_RefineFreeDevice((void **)&BandNonCol_refine_x);
    BandNonCol_refine_x_n = 0;
    BandNonCol_refine_state.basis_valid = 0;
    /* the products' workspace as the FP64 path treats its own before the
       eigensolver (an FP64 solve may follow a failed refinement) */
    if (BandNonCol_GemmWorkspaceTurnRelease()) openmx_gemmul8ReleaseWorkspaces();
    else openmx_gemmul8TrimWorkspaces();
}

static void BandNonCol_KRefReset(void)
{
    if (BandNonCol_kref != NULL) {
        for (int slot = 0; slot < BandNonCol_kref_slots; slot++) {
            free(BandNonCol_kref[slot].basis);
            free(BandNonCol_kref[slot].lam);
        }
        free(BandNonCol_kref);
    }
    BandNonCol_kref = NULL;
    BandNonCol_kref_slots = 0;
    BandNonCol_kref_n2 = 0;
    BandNonCol_kref_used = 0;
}

/* the run boundary (Band_DFT_NonCol_Release_GPU_Caches) */
static void BandNonCol_RefineRelease(void)
{
    openmx_eigen_refine_state_release(&BandNonCol_refine_state);
    BandNonCol_KRefReset();
    BandNonCol_RefineDropDevice();
    openmx_eigen_refine_device_release(&BandNonCol_refine_dev);
    free(BandNonCol_refine_e0);
    BandNonCol_refine_e0 = NULL;
    BandNonCol_refine_e0_n = 0;
}

/* the run boundary, after the release: the stage of the next input starts
   over (a refinement turned off for the rest of a run does not reach the
   next input of -runtest/-runtestL); every rank calls it, so the next
   configuration is read collectively again */
static void BandNonCol_RefineResetStage(void)
{
    BandNonCol_refine = BandNonCol_refine_initial;
}

/* every rank, once: OPENMX_EIGEN_REFINE, or the default the global root's
   device implies, and OPENMX_EIGEN_REFINE_UNTIL, the global root's for all
   ranks alike (a stage that differed between the ranks would take them
   into different collectives) */
static void BandNonCol_RefineConfigure(void)
{
    BandNonColRefineState *st = &BandNonCol_refine;
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

static int BandNonCol_RefineIterations(void)
{
    return (openmx_fp32_enabled() && 0 < BandNonCol_refine.iterations) ? BandNonCol_refine.iterations : 0;
}

/* the FP32 scratch of a refined solve: the FP32 matrix and eigenvalues, and
   room for the three product blocks so that none takes a buffer of its own */
static size_t BandNonCol_RefineScratchBytes(int n2, int maxn)
{
    size_t const n2n2 = (size_t)n2 * (size_t)n2;
    size_t const fp32 = (2 * n2n2 + (size_t)n2 + 1) * sizeof(float);
    size_t const block = ((size_t)n2 * (size_t)maxn * 2 * sizeof(double) + 511u) / 512u * 512u;

    return ((fp32 < 3 * block) ? 3 * block : fp32) + 4096;
}

/* every rank: whether the device of each owner has room for the refined
   solves of the turns that solve at once, turns of them (beside the FP64
   path's buffers of the same turns); base is this rank's view of whether
   the call may refine at all (OPENMX_EIGEN_REFINE_MIN_N is read per rank),
   folded into the common verdict so that every rank takes the collective
   and the same path; a refinement that would not fit leaves the step in
   FP64 */
static int BandNonCol_RefineFits(int base, int n2, int maxn, int turns, int owns_dense)
{
    if (BandNonCol_mgpu_host_dense) return 0;
    size_t const n2n2 = (size_t)n2 * (size_t)n2;
    int          local_fit = base ? 1 : 0, fit = 0;

    if (BandNonCol_RefineIterations() <= 0) return 0;
    if (turns < 1) turns = 1;
    if (local_fit && owns_dense) {
        size_t const extra = sizeof(dcomplex) * n2n2 + BandNonCol_RefineScratchBytes(n2, maxn) +
                             openmx_eigen_refine_fp32_work_bytes(1, n2);
        size_t free_bytes = 0, total_bytes = 0;

        if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != 0 ||
            free_bytes < (extra + 3 * sizeof(dcomplex) * n2n2) * (size_t)turns + (size_t)512 * 1024 * 1024)
            local_fit = 0;
    }
    MPI_Allreduce(&local_fit, &fit, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
    return fit;
}

/* every rank, before the solves of a call: possible says whether the call
   takes the one-k-point-per-k-world path with every owner resident (from
   the first SCF step on), possible_staged the same path in serialized turn
   groups and possible_kdense the k-dense path (both from the second step
   on) */
static void BandNonCol_RefineBeginStep(int SCF_iter, int possible, int possible_kdense, int possible_staged)
{
    if (BandNonCol_mgpu_host_dense) possible=possible_kdense=possible_staged=0;
    BandNonColRefineState *st = &BandNonCol_refine;
    int common;

    BandNonCol_RefineConfigure();
    if (SCF_iter == 1) {
        st->final_stage = 0;
        st->cycle_refined = 0;
        BandNonCol_refine_state.basis_valid = 0;   /* a new cycle may come with a new geometry */
        BandNonCol_KRefReset();
    }
    st->transient = (SCF_iter == 1);
    st->scf_iter = SCF_iter;
    st->pending = 0;
    st->solved = 0;
    st->persistent = 0;
    /* the occupations of the refinement are FermiFunc_NC at one chemical
       potential: XANES and emptied states keep the FP64 eigensolver */
    common = (0 < BandNonCol_RefineIterations() && !st->final_stage && !st->force_fp64 && Cnt_switch == 0 &&
              SCF_iter < DFTSCF_loop && xanes_calc == 0 && empty_states_flag == 0);
    st->last_refined = (possible && common);
    /* the two-pass solves gate their first pass with the previous step's
       chemical potential, so they start at the second step */
    st->kdense = (possible_kdense && 2 <= SCF_iter && common);
    st->staged = (possible_staged && 2 <= SCF_iter && common);
    if (BandNonCol_kref != NULL) {
        for (int slot = 0; slot < BandNonCol_kref_slots; slot++) BandNonCol_kref[slot].refined = 0;
    }
}

/* every rank, after the refined solves of a call: the owners' solves decide
   the stage for all ranks alike */
static void BandNonCol_RefineEndStep(void)
{
    BandNonColRefineState *st = &BandNonCol_refine;
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

int Band_DFT_NonCol_RefineLastRefined(void)
{
    return BandNonCol_refine.last_refined;
}

void Band_DFT_NonCol_RefineForceFP64(int on)
{
    BandNonCol_refine.force_fp64 = on;
}

/* the end of an SCF cycle: the warm-start bases go (the device one and the
   host slots), so the forces and the next cycle start with the memory free */
void Band_DFT_NonCol_RefineEndCycle(void)
{
    BandNonCol_RefineDropDevice();
    BandNonCol_KRefReset();
    BandNonCol_refine_state.basis_valid = 0;
}

/* DFT.c, after the mixing of an SCF step that goes on: normrd is the printed
   NormRD */
void Band_DFT_NonCol_RefineAfterMixing(double normrd)
{
    BandNonColRefineState *st = &BandNonCol_refine;
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

static void BandNonCol_KRefEnsure(int slots, int n2)
{
    if (BandNonCol_kref != NULL && (BandNonCol_kref_slots != slots || BandNonCol_kref_n2 != n2))
        BandNonCol_KRefReset();
    if (BandNonCol_kref == NULL) {
        BandNonCol_kref = (BandNonColKRefineSlot *)BandNonCol_MallocArray((size_t)slots, sizeof(BandNonColKRefineSlot),
                                                                          "refined basis slots");
        memset(BandNonCol_kref, 0, sizeof(BandNonColKRefineSlot) * (size_t)slots);
        BandNonCol_kref_slots = slots;
        BandNonCol_kref_n2 = n2;
    }
}

/* the budget of the slots of one rank */
static size_t BandNonCol_RefineBasisLimit(void)
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

static int BandNonCol_KRefSlotAllocate(BandNonColKRefineSlot *slot, int n2)
{
    size_t const bytes = sizeof(dcomplex) * (size_t)n2 * (size_t)n2 + sizeof(double) * (size_t)n2;
    size_t const limit = BandNonCol_RefineBasisLimit();

    if (slot->basis != NULL) return 1;
    if (limit < BandNonCol_kref_used || bytes > limit - BandNonCol_kref_used) return 0;
    slot->basis = (dcomplex *)malloc(sizeof(dcomplex) * (size_t)n2 * (size_t)n2);
    slot->lam = (double *)malloc(sizeof(double) * (size_t)n2);
    if (slot->basis == NULL || slot->lam == NULL) {
        free(slot->basis);
        free(slot->lam);
        slot->basis = NULL;
        slot->lam = NULL;
        return 0;
    }
    BandNonCol_kref_used += bytes;
    return 1;
}

static void BandNonCol_KRefSlotFree(BandNonColKRefineSlot *slot, int n2)
{
    if (slot->basis != NULL) {
        BandNonCol_kref_used -= sizeof(dcomplex) * (size_t)n2 * (size_t)n2 + sizeof(double) * (size_t)n2;
    }
    free(slot->basis);
    free(slot->lam);
    slot->basis = NULL;
    slot->lam = NULL;
    slot->valid = 0;
}

/* The device buffers of a refined solve of an n2 x n2 problem: 1 when they
   are there, 0 with everything released again when the device has no room
   (condemn: the failure takes the FP64 eigensolver for the rest of the SCF
   cycle) */
static int BandNonCol_RefineEnsureBuffers(int n2, int maxn, int kloop, int condemn)
{
    BandNonColRefineState *st = &BandNonCol_refine;
    BandNonColRefineBuffers *b = &BandNonCol_refine_buf;
    size_t const n2n2 = (size_t)n2 * (size_t)n2;
    size_t const scratch = BandNonCol_RefineScratchBytes(n2, maxn);

    BandNonCol_DenseGpu_EnsureHipblas();
    if (BandNonCol_refine_x != NULL && BandNonCol_refine_x_n != n2) {
        BandNonCol_RefineFreeDevice((void **)&BandNonCol_refine_x);
        BandNonCol_refine_x_n = 0;
    }
    if (b->scratch_bytes < scratch) {
        BandNonCol_RefineFreeDevice(&b->d_scratch);
        b->scratch_bytes = 0;
        if (BandNonCol_TryDeviceMalloc(&b->d_scratch, scratch) == hipSuccess) b->scratch_bytes = scratch;
    }
    if (b->d_w == NULL) (void)BandNonCol_TryDeviceMalloc((void **)&b->d_w, sizeof(double) * (size_t)n2);
    if (b->d_info == NULL) (void)BandNonCol_TryDeviceMalloc((void **)&b->d_info, sizeof(int32_t));
    if (BandNonCol_refine_x == NULL &&
        BandNonCol_TryDeviceMalloc((void **)&BandNonCol_refine_x, sizeof(dcomplex) * n2n2) == hipSuccess)
        BandNonCol_refine_x_n = n2;
    if (b->d_scratch == NULL || b->d_w == NULL || b->d_info == NULL || BandNonCol_refine_x == NULL) {
        if (2 <= level_stdout) {
            printf("<Band_DFT_NonCol> k-point %d: no room for the refined eigensolver's %.1f MiB; FP64 eigensolver%s\n",
                   kloop, (double)(scratch + sizeof(dcomplex) * n2n2) / 1048576.0,
                   condemn ? " for the rest of the SCF cycle" : "");
            fflush(stdout);
        }
        BandNonCol_RefineDropDevice();
        if (condemn && st->persistent < 1) st->persistent = 1;
        return 0;
    }
    return 1;
}

/* The first half of a refined solve of the transformed Hamiltonian in d_H2
   (n2 x n2, Hermitian): the warm start from the slot's basis when warm
   allows and the slot holds one, else the FP32 solve; the estimates to
   BandNonCol_refine_e0.  Returns 1 with the solve prepared
   (BandNonCol_RefineFinishH2 completes it), 0 when the FP64 eigensolver has
   to take over (d_H2 still the Hermitian matrix, the refinement's device
   buffers released). */
static int BandNonCol_RefinePrepareH2(int n2, int maxn, int kloop, BandNonColKRefineSlot *slot, int warm,
                                      int transient, int condemn)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;
    BandNonColRefineState *st = &BandNonCol_refine;
    BandNonColRefineBuffers *b = &BandNonCol_refine_buf;
    EigenRefineDevice *dev = &BandNonCol_refine_dev;
    EigenRefineState *rs = &BandNonCol_refine_state;
    EigenRefineProblem *pb = &BandNonCol_refine_pb;
    EigenRefineReport *rep = &BandNonCol_refine_rep;
    size_t const n2n2 = (size_t)n2 * (size_t)n2;
    int rc;

    if (!BandNonCol_RefineEnsureBuffers(n2, maxn, kloop, condemn)) {
        slot->valid = 0;
        return 0;
    }
    if (BandNonCol_refine_e0_n < n2) {
        free(BandNonCol_refine_e0);
        BandNonCol_refine_e0 = NULL;
        BandNonCol_refine_e0_n = 0;
        BandNonCol_refine_e0 = (double *)BandNonCol_MallocArray((size_t)n2, sizeof(double), "refinement eigenvalues");
        BandNonCol_refine_e0_n = n2;
    }

    memset(dev, 0, sizeof(*dev));
    dev->hipblas = w->hipblas;
    dev->d_work = &b->d_work;
    dev->d_work_bytes = &b->d_work_bytes;
    dev->d_info = b->d_info;
    dev->try_malloc = BandNonCol_TryDeviceMalloc;
    dev->host_malloc = BandNonCol_MallocArray;
    dev->shared = 1;   /* the k-point owners' turns share the device */

    memset(pb, 0, sizeof(*pb));
    memset(rep, 0, sizeof(*rep));
    pb->cplx = 1;
    pb->n = n2;
    pb->maxn = maxn;
    pb->iterations = BandNonCol_RefineIterations();
    pb->a = w->d_H2;
    pb->w = b->d_w;
    pb->fp32 = b->d_scratch;
    pb->region = b->scratch_bytes;
    pb->x = BandNonCol_refine_x;
    pb->transient = transient;
    pb->warm = (warm && slot->valid && slot->basis != NULL && rs->lam != NULL && rs->capacity >= n2);
    pb->defaulted = st->defaulted;
    pb->occupation = BandNonCol_RefineFermi;
    pb->occupation_ctx = &BandNonCol_refine_occ;
    if (pb->warm) {
        wait_hipfunc(hipMemcpy(BandNonCol_refine_x, slot->basis, sizeof(dcomplex) * n2n2, hipMemcpyHostToDevice));
        wait_hipfunc(hipMemcpy(rs->lam, slot->lam, sizeof(double) * (size_t)n2, hipMemcpyHostToDevice));
        rs->basis_valid = 1;
        rs->basis_n = n2;
    }
    else {
        rs->basis_valid = 0;
    }
    rc = openmx_eigen_refine_prepare(rs, dev, pb, BandNonCol_refine_e0, rep);
    if (rc < 0) {
        pb->warm = 0;
        rs->basis_valid = 0;
        rc = openmx_eigen_refine_prepare(rs, dev, pb, BandNonCol_refine_e0, rep);
    }
    if (rc != 1) {
        if (st->persistent < rep->persistent) st->persistent = rep->persistent;
        rs->basis_valid = 0;
        slot->valid = 0;
        BandNonCol_RefineDropDevice();
        return 0;
    }
    return 1;
}

/* The second half: the refinement with the occupations at mu.  Returns 1
   with ko[1..maxn] the refined eigenvalues, d_H2's first maxn columns the
   refined vectors and the basis kept in the slot when keep is set and there
   is room; 0 when the FP64 eigensolver has to take over (d_H2 still the
   Hermitian matrix).  The refinement's device buffers stay for the caller. */
static int BandNonCol_RefineFinishH2(int n2, int maxn, int kloop, double *ko, BandNonColKRefineSlot *slot, double mu,
                                     int keep)
{
    BandNonColRefineState *st = &BandNonCol_refine;
    BandNonColRefineBuffers *b = &BandNonCol_refine_buf;
    EigenRefineState *rs = &BandNonCol_refine_state;
    EigenRefineProblem *pb = &BandNonCol_refine_pb;
    EigenRefineReport *rep = &BandNonCol_refine_rep;
    size_t const n2n2 = (size_t)n2 * (size_t)n2;
    double *f0 = (double *)BandNonCol_MallocArray((size_t)n2, sizeof(double), "refinement occupations");
    int rc;

    BandNonCol_refine_occ.mu = mu;
    for (int i = 0; i < n2; i++) f0[i] = BandNonCol_RefineFermi(BandNonCol_refine_e0[i], i + 1, &BandNonCol_refine_occ);
    rc = openmx_eigen_refine_finish(rs, &BandNonCol_refine_dev, pb, f0, rep);
    free(f0);
    if (rc != 1) {
        if (st->persistent < rep->persistent) st->persistent = rep->persistent;
        rs->basis_valid = 0;
        slot->valid = 0;
        return 0;
    }
    wait_hipfunc(hipMemcpy(ko + 1, b->d_w, sizeof(double) * (size_t)maxn, hipMemcpyDeviceToHost));
    if (keep && BandNonCol_KRefSlotAllocate(slot, n2)) {
        wait_hipfunc(hipMemcpy(slot->basis, BandNonCol_refine_x, sizeof(dcomplex) * n2n2, hipMemcpyDeviceToHost));
        wait_hipfunc(hipMemcpy(slot->lam, rs->lam, sizeof(double) * (size_t)n2, hipMemcpyDeviceToHost));
        slot->valid = 1;
    }
    else {
        BandNonCol_KRefSlotFree(slot, n2);
    }
    if (3 <= level_stdout) {
        printf("<Band_DFT_NonCol> k-point %d: %s, %d refinement steps of %d columns: max |r| %.1e, max |s| %.1e, "
               "delta %.1e; Rayleigh-Ritz in %d partially occupied clusters (largest %d)\n",
               kloop, pb->warm ? "warm start" : "FP32 eigensolver", pb->iterations, rep->columns, rep->max_r,
               rep->max_s, rep->delta, rep->rr_clusters, rep->rr_largest);
        fflush(stdout);
    }
    st->solved = 1;
    return 1;
}

/* The refined solve of the two-pass paths in one go: warm from the slot's
   basis when warm_allowed, the occupations at mu, the basis kept in the
   slot with keep.  Returns 1 when d_H2's first maxn columns hold the refined
   vectors, 0 when the FP64 eigensolver has to take over (d_H2 still the
   Hermitian matrix). */
static int BandNonCol_RefinedSolveH2(int n2, int maxn, int kloop, double *ko, BandNonColKRefineSlot *slot,
                                     int warm_allowed, double mu, int keep)
{
    if (!BandNonCol_RefinePrepareH2(n2, maxn, kloop, slot, warm_allowed, 0, 0)) return 0;
    return BandNonCol_RefineFinishH2(n2, maxn, kloop, ko, slot, mu, keep);
}

/* ------------------------------------------------------------------ */

/* The transformed Hamiltonian of one k-point of the dense paths, built on
   the device and made Hermitian in d_H2, with the transformed overlap
   built (rebuild_overlap) or taken from rdw->s_all; the buffers that only
   served the build go before the eigensolver. */
static void BandNonCol_RootDenseBuildH2(int rebuild_overlap,
                                        int n, int n2, int MaxN,
                                        double k1, double k2, double k3,
                                        const double *m_olp,
                                        const double *m_h11, const double *m_h22,
                                        const double *m_h12, const double *m_h12i,
                                        const double *m_i11, const double *m_i22,
                                        const double *m_i12,
                                        int *packed_order_GA, int *MP, double *ko,
                                        BandNonColRootDenseWorkspace *rdw)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;
    const int cache_buffers = BandNonCol_DenseBufferCacheEnabled();

    rdw->cs2_valid = 0;
    w->d_C2_valid = 0;

    if (n<=0 || (size_t)n2!=2U*(size_t)n || MaxN<=0 || n2<MaxN){
        BandNonCol_AbortWithMessage("Invalid spin-block dimensions in Band_DFT_NonCol root dense solver.");
    }

    if (cache_buffers){
        BandNonCol_DenseGpu_RegisterProcessCleanup();
    }
    BandNonCol_DenseGpu_Ensure(n,n2);

    if (rebuild_overlap){
        BandNonCol_ConstructDenseMsFromPacked_HIP(0,m_olp,w->d_S,packed_order_GA,MP,k1,k2,k3,n);
    }
    BandNonCol_DenseGpuPrepareOverlap(rebuild_overlap,n,n2,ko,rdw->s_all,cache_buffers);

    /* Return requested overlap scratch before assembling H.  The default
       Hamiltonian policy also selects native FP64 on MI300A. */
    if (rebuild_overlap) BandNonCol_ReleaseOverlapWorkspace(n);
    BandNonCol_DenseGpu_Ensure(n,n2);

    BandNonCol_ConstructDenseMsFromPacked_HIP(0,m_h11,w->d_H11,packed_order_GA,MP,k1,k2,k3,n);
    BandNonCol_ConstructDenseMsFromPacked_HIP(1,m_i11,w->d_tmp,packed_order_GA,MP,k1,k2,k3,n);
    if (BandNonCol_AddDense_HIP(n*n,w->d_H11,w->d_tmp) != 0){
        BandNonCol_AbortWithMessage("Band_DFT_NonCol HIP H11 add failed.");
    }

    BandNonCol_ConstructDenseMsFromPacked_HIP(0,m_h22,w->d_H22,packed_order_GA,MP,k1,k2,k3,n);
    BandNonCol_ConstructDenseMsFromPacked_HIP(1,m_i22,w->d_tmp,packed_order_GA,MP,k1,k2,k3,n);
    if (BandNonCol_AddDense_HIP(n*n,w->d_H22,w->d_tmp) != 0){
        BandNonCol_AbortWithMessage("Band_DFT_NonCol HIP H22 add failed.");
    }

    BandNonCol_ConstructDenseMsFromPacked_HIP(0,m_h12,w->d_H12,packed_order_GA,MP,k1,k2,k3,n);
    BandNonCol_ConstructDenseMsFromPacked_HIP(1,m_h12i,w->d_tmp,packed_order_GA,MP,k1,k2,k3,n);
    if (BandNonCol_AddDense_HIP(n*n,w->d_H12,w->d_tmp) != 0){
        BandNonCol_AbortWithMessage("Band_DFT_NonCol HIP H12 add failed.");
    }
    BandNonCol_ConstructDenseMsFromPacked_HIP(1,m_i12,w->d_tmp,packed_order_GA,MP,k1,k2,k3,n);
    if (BandNonCol_AddDense_HIP(n*n,w->d_H12,w->d_tmp) != 0){
        BandNonCol_AbortWithMessage("Band_DFT_NonCol HIP H12 ImNL add failed.");
    }

    BandNonCol_DenseGpuTripleTransform(n,w->d_H11);
    BandNonCol_DenseGpuTripleTransform(n,w->d_H12);
    BandNonCol_DenseGpuTripleTransform(n,w->d_H22);

    if (BandNonCol_BuildDenseHs2_HIP(n,n2,w->d_H11,w->d_H22,w->d_H12,w->d_H2) != 0){
        BandNonCol_AbortWithMessage("Band_DFT_NonCol HIP Hs2 build failed.");
    }
    if (BandNonCol_SymmetrizeDenseHermitian_HIP(n2,w->d_H2) != 0){
        BandNonCol_AbortWithMessage("Band_DFT_NonCol HIP Hamiltonian symmetrization failed.");
    }

    if (cache_buffers){
        wait_hipfunc(hipDeviceSynchronize());
    }
    else {
        BandNonCol_DenseGpu_ReleasePreMagmaBuffers(1);
    }
    if (BandNonCol_GemmWorkspaceTurnRelease()){
        openmx_gemmul8ReleaseWorkspaces();
    }
}

/* One k-point of the dense paths once its Hamiltonian is built (and solved
   by the refinement when refined): the FP64 eigensolver unless refined,
   EIGEN, and the back transform unless deferred or no vectors are needed
   (into rdw->cs2 with copy_evec_to_host, or into evec_panel). */
static void BandNonCol_RootDenseCompleteOneK(int n, int n2, int MaxN, int kloop, int refined,
                                             double *ko, double ***EIGEN,
                                             int need_evec, int copy_evec_to_host,
                                             int defer_backtransform,
                                             BandNonColRootDenseWorkspace *rdw, dcomplex *evec_panel)
{
    BandNonColDenseGpuWorkspace *w = &BandNonCol_dense_gpu_workspace;
    const int cache_buffers = BandNonCol_DenseBufferCacheEnabled();
    const int h_use_hipsolver = BandNonCol_HamiltonianUseHipSolver();

    if (!refined){
        BandNonCol_DenseGpuEigen(w->d_H2,ko,n2,MaxN,h_use_hipsolver,
                                 "Band_DFT_NonCol root dense Hamiltonian");
    }
    for (int l=1; l<=MaxN; l++){
        EIGEN[0][kloop][l] = ko[l];
    }

    /* The solve above is synchronous.  Return rocSOLVER's large process-wide
       scratch before this rank enters the n2-by-n2 back-transform; otherwise
       completed turns accumulate scratch and squeeze later turns on MI300A. */
    if (!refined) BandNonCol_ReleaseSolverTurnWorkspaces(n2,MaxN,h_use_hipsolver);

    if (!need_evec){
        if (!cache_buffers){
            BandNonCol_DenseGpu_Destroy();
        }
        return;
    }

    if (!defer_backtransform){
        BandNonCol_RootDenseBacktransform_HIP(n,n2,MaxN,copy_evec_to_host,rdw,evec_panel);
    }
}

/* One k-point of the dense paths: the transformed Hamiltonian built on the
   device and solved, in FP64 or, with refine_slot, refined (warm from the
   slot when refine_warm allows, occupations at refine_mu, the basis kept in
   the slot with refine_keep; FP64 when the refinement fails), then
   completed as above.  Returns 1 when the solve was refined. */
static void BandNonCol_ConstructDenseMsFromPacked(int cpx_flag,const double *M1,dcomplex *Ms,
    int *order_GA,int *MP,double k1,double k2,double k3,int n,int owns_dense);
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

static int BandNonCol_RootDenseSolveOneK_Host(int rebuild_overlap,
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
        BandNonCol_ConstructDenseMsFromPacked(0,m_olp,s,packed_order_GA,MP,k1,k2,k3,n,1);
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
        BandNonCol_ConstructDenseMsFromPacked(0,real[b],h[b],packed_order_GA,MP,k1,k2,k3,n,1);
        BandNonCol_ConstructDenseMsFromPacked(1,imag[b],h[b],packed_order_GA,MP,k1,k2,k3,n,1);
        if (b==2) BandNonCol_ConstructDenseMsFromPacked(1,m_i12,h[b],packed_order_GA,MP,k1,k2,k3,n,1);
        if (!openmx_mgpu_host_zgemm(HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,h[b],n,s,n,work,n) ||
            !openmx_mgpu_host_zgemm(HIPBLAS_OP_C,HIPBLAS_OP_N,n,n,n,s,n,work,n,h[b],n))
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
            if (!openmx_mgpu_host_zgemm(HIPBLAS_OP_T,HIPBLAS_OP_T,MaxN,n,n,q+(size_t)half*n,n2,
                    s,n,rdw->cs2+(size_t)half*n*n2,n2)) BandNonCol_AbortWithMessage(openmx_mgpu_host_error());
    }
    rdw->cs2_valid=need_evec;
    return 0;
}

static int BandNonCol_RootDenseSolveOneK_HIP(int rebuild_overlap,
                                              int n, int n2, int MaxN, int kloop,
                                              double k1, double k2, double k3,
                                              const double *m_olp,
                                              const double *m_h11, const double *m_h22,
                                              const double *m_h12, const double *m_h12i,
                                              const double *m_i11, const double *m_i22,
                                              const double *m_i12,
                                              int *packed_order_GA, int *MP,
                                              double *ko, double ***EIGEN,
                                              int need_evec, int copy_evec_to_host,
                                              int defer_backtransform,
                                              BandNonColRootDenseWorkspace *rdw, dcomplex *evec_panel,
                                              BandNonColKRefineSlot *refine_slot, int refine_warm,
                                              double refine_mu, int refine_keep)
{
    if (BandNonCol_mgpu_host_dense) {
        int status=BandNonCol_RootDenseSolveOneK_Host(rebuild_overlap,n,n2,MaxN,kloop,k1,k2,k3,
            m_olp,m_h11,m_h22,m_h12,m_h12i,m_i11,m_i22,m_i12,packed_order_GA,MP,ko,EIGEN,need_evec,rdw);
        if (need_evec && evec_panel) for (int basis=0;basis<n2;basis++)
            memcpy(evec_panel+(size_t)basis*MaxN,rdw->cs2+(size_t)basis*n2,(size_t)MaxN*sizeof(dcomplex));
        return status;
    }

    int refined = 0;

    BandNonCol_RootDenseBuildH2(rebuild_overlap,n,n2,MaxN,k1,k2,k3,
                                m_olp,m_h11,m_h22,m_h12,m_h12i,m_i11,m_i22,m_i12,
                                packed_order_GA,MP,ko,rdw);
    if (refine_slot!=NULL){
        refined = BandNonCol_RefinedSolveH2(n2,MaxN,kloop,ko,refine_slot,refine_warm,refine_mu,refine_keep);
        /* the refinement's device buffers go before the back transform (the
           basis waits in the slot) */
        BandNonCol_RefineDropDevice();
    }
    BandNonCol_RootDenseCompleteOneK(n,n2,MaxN,kloop,refined,ko,EIGEN,need_evec,copy_evec_to_host,
                                     defer_backtransform,rdw,evec_panel);
    return refined;
}

/* The first half of a refined solve of this rank's k-point on the resident
   path, where every owner keeps its d_H2 and refinement buffers until the
   chemical potential of all k-points is known: the transformed Hamiltonian,
   then the FP32 solve, or the warm start from the slot's basis of the
   previous step, whose estimates go to ko and EIGEN.  Returns 1 with the
   solve pending (BandNonCol_RootDenseRefinedSecondHalf completes it), 0 when
   the k-point was solved in FP64 instead (and completed as an unrefined
   solve with the vectors). */
static int BandNonCol_RootDenseRefinedFirstHalf(int rebuild_overlap,
                                                int n, int n2, int MaxN, int kloop,
                                                double k1, double k2, double k3,
                                                const double *m_olp,
                                                const double *m_h11, const double *m_h22,
                                                const double *m_h12, const double *m_h12i,
                                                const double *m_i11, const double *m_i22,
                                                const double *m_i12,
                                                int *packed_order_GA, int *MP,
                                                double *ko, double ***EIGEN,
                                                int copy_evec_to_host, int defer_backtransform,
                                                BandNonColRootDenseWorkspace *rdw)
{
    BandNonColRefineState *st = &BandNonCol_refine;

    st->pending = 0;
    BandNonCol_RootDenseBuildH2(rebuild_overlap,n,n2,MaxN,k1,k2,k3,
                                m_olp,m_h11,m_h22,m_h12,m_h12i,m_i11,m_i22,m_i12,
                                packed_order_GA,MP,ko,rdw);
    if (BandNonCol_RefinePrepareH2(n2,MaxN,kloop,&BandNonCol_kref[kloop],
                                   openmx_eigen_refine_warm_band_step(st->scf_iter) && !st->transient,
                                   st->transient,!st->transient)){
        for (int l=1; l<=MaxN; l++){
            ko[l] = BandNonCol_refine_e0[l-1];
            EIGEN[0][kloop][l] = ko[l];
        }
        st->pending = 1;
        return 1;
    }
    BandNonCol_RootDenseCompleteOneK(n,n2,MaxN,kloop,0,ko,EIGEN,1,copy_evec_to_host,
                                     defer_backtransform,rdw,NULL);
    return 0;
}

/* The second half on the resident path: the refinement with the
   occupations at mu (the FP64 eigensolver when it fails), the eigenvalues
   to ko and EIGEN, the basis to the slot for the next step's warm start
   (OPENMX_EIGEN_REFINE_WARM_BAND), the refinement's device buffers
   released, then the back transform as for an unrefined solve. */
static void BandNonCol_RootDenseRefinedSecondHalf(int n, int n2, int MaxN, int kloop, double *ko,
                                                  double ***EIGEN, double mu,
                                                  int copy_evec_to_host, int defer_backtransform,
                                                  BandNonColRootDenseWorkspace *rdw)
{
    BandNonColRefineState *st = &BandNonCol_refine;
    int refined;

    st->pending = 0;
    refined = BandNonCol_RefineFinishH2(n2,MaxN,kloop,ko,&BandNonCol_kref[kloop],mu,
                                        openmx_eigen_refine_warm_band_enabled());
    BandNonCol_RefineDropDevice();
    BandNonCol_RootDenseCompleteOneK(n,n2,MaxN,kloop,refined,ko,EIGEN,1,copy_evec_to_host,
                                     defer_backtransform,rdw,NULL);
}

/* The SCF driver already gathers these eight sparse matrices in the same
   rank/atom/orbital order as PackDenseM1. Borrow them on dense k owners
   instead of repeating the gathers for every GPU turn in both passes.
   Agree collectively before skipping any of the fallback collectives. */
static int BandNonCol_PackedKCacheReady(int owns_dense, int size_H1)
{
    const char *env = getenv("OPENMX_BAND_NONCOL_PACKED_CACHE");
    int ready = (env == NULL || atoi(env) != 0) &&
        Set_Hamiltonian_GpuSolver_Packed_CacheReady() &&
        Set_Hamiltonian_GpuSolver_Packed_OrderMode() == 1 &&
        Set_Hamiltonian_GpuSolver_Packed_Size() == size_H1;
    int all_ready = 0;

    if (ready && owns_dense) {
        ready = Set_Hamiltonian_GpuSolver_Packed_OwnsCache() &&
                Set_Hamiltonian_GpuSolver_Packed_OrderGA() != NULL &&
                Set_Hamiltonian_GpuSolver_Packed_Overlap() != NULL;
        for (int spin = 0; spin < 4; ++spin)
            ready = ready && Set_Hamiltonian_GpuSolver_Packed_H(spin) != NULL;
        for (int comp = 0; comp < 3; ++comp)
            ready = ready && Set_Hamiltonian_GpuSolver_Packed_ImNL(comp) != NULL;
    }
    MPI_Allreduce(&ready, &all_ready, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
    return all_ready;
}

/* The multi-k GPU path normally diagonalizes each H(k) twice: first for
   occupations, then for density matrices. Keep compact backtransformed
   eigenvectors within this SCF call when host memory permits. The node-wide
   cache is limited to 1/32 of currently available RAM, additionally capped
   at 1024 MiB per rank (OPENMX_BAND_NONCOL_KCACHE_MB=0 disables it). */
static size_t BandNonCol_KCacheLimit(void)
{
    const char *env = getenv("OPENMX_BAND_NONCOL_KCACHE_MB");
    double cap = env == NULL ? 1024.0 : atof(env);
    FILE *file;
    char line[256];
    unsigned long long available_kib = 0;
    int ranks = openmx_gpu_local_size_noncollective();
    size_t limit;
    if (!(cap > 0.0) || cap >= (double)SIZE_MAX / (1024.0 * 1024.0)) return 0;
    limit = (size_t)(cap * 1024.0 * 1024.0);
    file = fopen("/proc/meminfo", "r");
    if (file == NULL) return 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (sscanf(line, "MemAvailable: %llu kB", &available_kib) == 1) break;
    }
    fclose(file);
    if (ranks < 1) ranks = 1;
    if (available_kib / (unsigned)ranks * (1024ULL / 32ULL) < limit)
        limit = (size_t)(available_kib / (unsigned)ranks * (1024ULL / 32ULL));
    return limit;
}

/* The per-rank budget of a host cache: the variable's MiB (default_mib when
   unset, 0 disables), at most 1/fraction of the node's available memory over
   its ranks; 0 when /proc/meminfo cannot be read. */
static size_t BandNonCol_HostCacheLimit(const char *name, double default_mib, unsigned fraction)
{
    const char *env = getenv(name);
    double cap = (env == NULL) ? default_mib : atof(env);
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
    if (fraction < 1u) fraction = 1u;
    available_kib /= (unsigned long long)ranks * (unsigned long long)fraction;
    if (available_kib < limit / 1024U) limit = (size_t)available_kib * 1024U;
    return limit;
}

/* The transformed overlap of a k-point depends on the geometry and k, not
   on the SCF Hamiltonian: the k-dense path keeps a bounded host copy per
   k-point across the SCF steps (independently of the device workspaces,
   which go with every concurrency group), so that from the second step on
   neither pass builds and solves S(k) again.  OPENMX_BAND_NONCOL_OVERLAP_CACHE_MB
   per rank (default 512, at most 1/64 of the node's available memory over
   its ranks; 0 disables); reset with every new SCF cycle and at the run
   boundary.  Ported from the NVIDIA version. */
typedef struct {
    dcomplex *matrix;
    double k1, k2, k3;
} BandNonColKOverlap;

static BandNonColKOverlap *BandNonCol_k_overlap = NULL;
static int BandNonCol_k_overlap_n = 0, BandNonCol_k_overlap_points = 0;
static size_t BandNonCol_k_overlap_bytes = 0, BandNonCol_k_overlap_used = 0;

static void BandNonCol_KOverlapReset(void)
{
    for (int k = 0; k < BandNonCol_k_overlap_points; ++k)
        free(BandNonCol_k_overlap[k].matrix);
    free(BandNonCol_k_overlap);
    BandNonCol_k_overlap = NULL;
    BandNonCol_k_overlap_n = BandNonCol_k_overlap_points = 0;
    BandNonCol_k_overlap_bytes = BandNonCol_k_overlap_used = 0;
}

static size_t BandNonCol_KOverlapLimit(void)
{
    return BandNonCol_HostCacheLimit("OPENMX_BAND_NONCOL_OVERLAP_CACHE_MB", 512.0, 64);
}

static void BandNonCol_KOverlapPrepare(int n, int points)
{
    size_t bytes;

    if (BandNonCol_k_overlap_n != n || BandNonCol_k_overlap_points != points)
        BandNonCol_KOverlapReset();
    if (BandNonCol_k_overlap != NULL || n <= 0 || points <= 0 ||
        (size_t)n > SIZE_MAX / (size_t)n / sizeof(dcomplex)) return;
    bytes = (size_t)n * (size_t)n * sizeof(dcomplex);
    if (bytes > BandNonCol_KOverlapLimit()) return;
    BandNonCol_k_overlap = (BandNonColKOverlap*)calloc((size_t)points, sizeof(BandNonColKOverlap));
    if (BandNonCol_k_overlap == NULL) return;
    BandNonCol_k_overlap_n = n;
    BandNonCol_k_overlap_points = points;
    BandNonCol_k_overlap_bytes = bytes;
}

/* 1 with the k-point's transformed overlap copied to matrix (n x n), 0 when
   it is not cached */
static int BandNonCol_KOverlapRestore(int point, double k1, double k2, double k3, dcomplex *matrix)
{
    BandNonColKOverlap *entry;

    if (BandNonCol_k_overlap == NULL || point < 0 || point >= BandNonCol_k_overlap_points) return 0;
    entry = &BandNonCol_k_overlap[point];
    if (entry->matrix == NULL || entry->k1 != k1 || entry->k2 != k2 || entry->k3 != k3) return 0;
    memcpy(matrix, entry->matrix, BandNonCol_k_overlap_bytes);
    return 1;
}

static void BandNonCol_KOverlapSave(int point, double k1, double k2, double k3, const dcomplex *matrix)
{
    BandNonColKOverlap *entry;

    if (BandNonCol_k_overlap == NULL || point < 0 || point >= BandNonCol_k_overlap_points) return;
    entry = &BandNonCol_k_overlap[point];
    if (entry->matrix == NULL) {
        size_t limit = BandNonCol_KOverlapLimit();

        if (BandNonCol_k_overlap_used > limit ||
            BandNonCol_k_overlap_bytes > limit - BandNonCol_k_overlap_used) return;
        entry->matrix = (dcomplex*)malloc(BandNonCol_k_overlap_bytes);
        if (entry->matrix == NULL) return;
        BandNonCol_k_overlap_used += BandNonCol_k_overlap_bytes;
    }
    memcpy(entry->matrix, matrix, BandNonCol_k_overlap_bytes);
    entry->k1 = k1;
    entry->k2 = k2;
    entry->k3 = k3;
}

/* K points belong to contiguous per-rank ranges. Visiting consecutive k
   indices would serialize most GPU groups on one owner. Interleave owners
   while preserving each owner's ascending k order and hence its DM sums.
   (tests/run_band_noncol_k_order_cpu.py builds this function as it is.) */
static void BandNonCol_InterleaveKOrder(int points, int ranks,
                                       const int *owner, int *order)
{
    int *head = (int*)malloc(sizeof(int)*(size_t)ranks);
    int *next = (int*)malloc(sizeof(int)*(size_t)points);
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
    free(next);
    free(head);
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

static void BandNonCol_DistributeDenseEvecGlobal(int active_world2, int n2, int MaxN, int myid0, int myworld2,
                                                 int *NPROCS_WD2, int *Comm_World_StartID2,
                                                 int root, const dcomplex *dense_evec, dcomplex *EVec1)
{
    const int tag = 997;
    int target_np = NPROCS_WD2[active_world2];
    int target_start = Comm_World_StartID2[active_world2];

    if (myid0==root){
        dcomplex *send_buf = NULL;
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
            send_buf = (dcomplex*)malloc(sizeof(dcomplex)*(size_t)max_count);
            if (send_buf==NULL){
                BandNonCol_AbortWithMessage("Failed to allocate dense eigenvector send buffer in Band_DFT_NonCol.c.");
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
            for (int basis=0; basis<n2; basis++){
                const dcomplex *src_col = dense_evec + (size_t)basis*(size_t)n2 + (size_t)(is-1);
                dcomplex *dst_col = dst + (size_t)basis*(size_t)stride;
                memcpy(dst_col,src_col,sizeof(dcomplex)*(size_t)stride);
            }

            if (target_rank!=root){
                MPI_Send(send_buf,count*2,MPI_DOUBLE,target_rank,tag,mpi_comm_level1);
            }
        }

        free(send_buf);
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

static void BandNonCol_BuildDenseHs2_OpenMP(int n, int n2, const dcomplex *H11,
                                             const dcomplex *H22, const dcomplex *H12,
                                             dcomplex *H2)
{
    int nn = n*n;
    int n2n2 = n2*n2;

    if (!BandNonCol_OmpTargetIsPresent(H2,sizeof(dcomplex)*(size_t)n2n2)){
#pragma omp target enter data map(alloc: H2[0 : n2n2])
    }

#pragma omp target teams distribute parallel for collapse(2) map(present, alloc: H11[0 : nn], H22[0 : nn], H12[0 : nn], H2[0 : n2n2])
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

double **ReEVec_i1,**ImEVec_i1,**ReEVec_i2,**ImEVec_i2;
double **ReEVec_j1,**ImEVec_j1,**ReEVec_j2,**ImEVec_j2;

void solve_evp_complex_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, int *na_rows2_2, 
                         int *nblk2, int *mpi_comm_rows_int, int *mpi_comm_cols_int );

void elpa_solve_evp_complex_2stage_double_impl_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, 
                                                 int *na_rows2_2, int *nblk2, int *na_cols2, 
                                                 int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld );


static void Construct_Band_Ms( int cpx_flag, double ****Mat, double *M1, dcomplex *Ms, 
                               int *MP, double k1, double k2, double k3);

static void Construct_Band_DenseMs( int cpx_flag, double ****Mat, double *M1, dcomplex *Ms,
                                    int *MP, double k1, double k2, double k3,
                                    int n, int owns_dense );
static void BandNonCol_PackDenseM1( double ****Mat, double *M1, int *MP, int *order_GA );
static void BandNonCol_ConstructDenseMsFromPacked( int cpx_flag, const double *M1, dcomplex *Ms,
                                                   int *order_GA, int *MP, double k1, double k2, double k3,
                                                   int n, int owns_dense );

static double Calc_DM_Band_non_collinear(
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
    double *rEDM22);


double Calc_ParDM_Band_non_collinear(
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
    double *iDM22);



#pragma GCC optimize("O2")
double Band_DFT_NonCol(
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
		    MPI_Comm *MPI_CommWD2)
{
  openmx_mgpu_scope_state input_scope=openmx_mgpu_band_scope_enter(0,0,0);
  static int firsttime=1;
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
  int use_k_packed_cache = 0;
  int *packed_k_order = order_GA;
  int use_device_eigenvector_dm;
  int use_backtransform_barrier;
  int root_dense_serial_gpusolver_worlds = 0;
  int owns_dense_k_rank;
  int *dense_k_owner = NULL;
  int *dense_k_order = NULL;
  dcomplex **k_evec_cache = NULL;
  size_t k_cache_bytes = 0, k_cache_used = 0, k_cache_limit = 0;
  int k_cache_hits = 0, k_overlap_hits = 0;
  int k_gather_groups = 0, k_gather_skips = 0;
  int k_dense_gpu_turns = 1;
  int root_dense_turn_group = 1;
  int owns_root_dense_rank = 0;
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

  /* the transformed overlaps of the k-dense path are kept across the SCF
     steps of one geometry; new inputs and geometry updates both start a new
     SCF cycle */
  if (SCF_iter <= 1) BandNonCol_KOverlapReset();

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

  /* GPU dispatch (added by H.Kawai): assign HIP/OpenMP target device when GPUSOLVER is requested */
  if (scf_eigen_lib_flag == GPUSOLVER && n2 >= Band_DFT_NonCol_GpuSwitchNum() &&
      Set_Hamiltonian_OpenMP_Rank_Is_Selected()) {
      set_hip_default_device_from_local_rank_noncollective();
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

  My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

  ReEVec_i1 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ReEVec_i1[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ImEVec_i1 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ImEVec_i1[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ReEVec_i2 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ReEVec_i2[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ImEVec_i2 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ImEVec_i2[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ReEVec_j1 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ReEVec_j1[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ImEVec_j1 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ImEVec_j1[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ReEVec_j2 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ReEVec_j2[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ImEVec_j2 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ImEVec_j2[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  /****************************************************
   find size_H
  ****************************************************/

  size_H1 = Get_OneD_HS_Col(0, nh[0], &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  /***********************************************
     allocation of arrays 
  ***********************************************/

  rDM11 = (double*)malloc(sizeof(double)*size_H1);
  rDM22= (double*)malloc(sizeof(double)*size_H1);
  rDM12 = (double*)malloc(sizeof(double)*size_H1);
  iDM11 = (double*)malloc(sizeof(double)*size_H1);
  iDM22 = (double*)malloc(sizeof(double)*size_H1);
  iDM12 = (double*)malloc(sizeof(double)*size_H1);
  rEDM11 = (double*)malloc(sizeof(double)*size_H1);
  rEDM22 = (double*)malloc(sizeof(double)*size_H1);

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

			  {
			    int gpu_diag_fit=(scf_eigen_lib_flag==GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2)) ?
			      BandNonCol_GpuDiagFits(n,n2,MaxN,size_H1,Set_Hamiltonian_OpenMP_Rank_Is_Selected()) : 0;
			    use_root_dense_gpusolver = (gpu_diag_fit && all_knum==1);
			    use_k_dense_gpusolver = (gpu_diag_fit && all_knum!=1 && strcasecmp(mode,"scf")==0);
			  }
			  {
			    int local_requests[2];
			    int request_sums[2];

			    local_requests[0] = BandNonCol_DeviceEigenvectorDMRequested() ? 1 : 0;
			    local_requests[1] = BandNonCol_BacktransformBarrierRequested() ? 1 : 0;
			    MPI_Allreduce(local_requests,request_sums,2,MPI_INT,MPI_SUM,mpi_comm_level1);
			    if ((request_sums[0]!=0 && request_sums[0]!=numprocs0) ||
			        (request_sums[1]!=0 && request_sums[1]!=numprocs0)){
			      BandNonCol_AbortWithMessage("Non-collinear GPU opt-ins must be consistent across MPI ranks.");
			    }
			    use_device_eigenvector_dm =
			      (request_sums[0]==numprocs0 && use_root_dense_gpusolver && strcasecmp(mode,"scf")==0);
			    use_backtransform_barrier =
			      (request_sums[1]==numprocs0 && use_root_dense_gpusolver && strcasecmp(mode,"scf")==0);
			  }
			  owns_dense_k_rank = (use_k_dense_gpusolver && Set_Hamiltonian_OpenMP_Rank_Is_Selected());
			  /* the GPU turns that solve at once (collective, before the
			     refinement's memory test that depends on them): the k-dense
			     path's concurrency groups, or the root-dense path's turn group
			     (every k-world when its turns are not serialized) */
			  if (use_k_dense_gpusolver){
			    k_dense_gpu_turns =
			      BandNonCol_MaxConcurrentKGpuTurns(n,n2,MaxN,size_H1,owns_dense_k_rank,myid0);
			  }
			  else if (use_root_dense_gpusolver){
			    owns_root_dense_rank =
			      (0<=myworld2 && myworld2<Num_Comm_World2 &&
			       myid0==Comm_World_StartID2[myworld2] &&
			       Set_Hamiltonian_OpenMP_Rank_Is_Selected());
			    root_dense_turn_group =
			      BandNonCol_SerializeGpuSolverGpuTurns() ?
			      BandNonCol_GpuTurnGroup(n,n2,MaxN,size_H1,owns_root_dense_rank,myid0) :
			      Num_Comm_World2;
			    if (root_dense_turn_group<1) root_dense_turn_group = 1;
			    /* the group decides the resident refined path and its
			       collectives: one value for all ranks, whatever their
			       OPENMX_BAND_GPU_TURN_GROUP */
			    MPI_Allreduce(MPI_IN_PLACE,&root_dense_turn_group,1,MPI_INT,MPI_MIN,mpi_comm_level1);
			  }
			  /* the refined eigensolver's paths of this call, alike on every
			     rank (see BandNonCol_RefineBeginStep): room for the refined
			     solves of the turns that solve at once, the resident two halves
			     when every k-world owner is in one turn group, else the two
			     passes */
			  {
			    const int refine_base = (strcasecmp(mode,"scf")==0 && openmx_eigen_refine_size_ok(n2));
			    const int refine_resident = (use_root_dense_gpusolver && Num_Comm_World2<=root_dense_turn_group);
			    int refine_fits = 0;

			    BandNonCol_RefineConfigure();
			    if (use_k_dense_gpusolver){
			      int turns = k_dense_gpu_turns;

			      if (T_knum<turns) turns = T_knum;
			      if (numprocs0<turns) turns = numprocs0;
			      refine_fits = BandNonCol_RefineFits(refine_base,n2,MaxN,turns,owns_dense_k_rank);
			    }
			    else if (use_root_dense_gpusolver){
			      refine_fits = BandNonCol_RefineFits(refine_base,n2,MaxN,
			                                          (Num_Comm_World2<root_dense_turn_group) ?
			                                          Num_Comm_World2 : root_dense_turn_group,
			                                          owns_root_dense_rank);
			    }
			    BandNonCol_RefineBeginStep(SCF_iter,refine_fits && refine_resident,
			                               refine_fits && use_k_dense_gpusolver,
			                               refine_fits && use_root_dense_gpusolver && !refine_resident);
			    if (BandNonCol_refine.kdense || BandNonCol_refine.staged || BandNonCol_refine.last_refined)
			      BandNonCol_KRefEnsure(T_knum,n2);
			  }
			  if (use_k_dense_gpusolver){
			    dense_k_owner = (int*)malloc(sizeof(int)*(size_t)T_knum);
			    if (dense_k_owner==NULL){
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
			  }
  if (use_k_dense_gpusolver) {
    int interleave_k = 1;

    use_k_packed_cache = BandNonCol_PackedKCacheReady(owns_dense_k_rank, size_H1);
    if (use_k_packed_cache) {
      Set_Hamiltonian_GpuSolver_SetMP(MP);
      if (owns_dense_k_rank)
        packed_k_order = Set_Hamiltonian_GpuSolver_Packed_OrderGA();
      if (myid0 == Host_ID && BandNonCol_GpuVerbose())
        printf("<Band> SCF %d: reusing shared packed H/S for all non-collinear k points.\n", SCF_iter);
    }
    dense_k_order = (int*)malloc(sizeof(int)*(size_t)T_knum);
    if (dense_k_order == NULL)
      BandNonCol_AbortWithMessage("Cannot allocate GPU k-point turn order.");
    /* Consecutive k points normally belong to the same MPI rank. A
       consecutive group therefore serialized all solves even when the
       memory guard admitted four owners. Round-robin owner queues make
       that existing concurrency limit effective, without changing the
       k-point assignment or the global eigenvalue indices.  The order
       makes the turn groups of both passes, so the root's
       OPENMX_BAND_NONCOL_INTERLEAVE_K (default 1) holds for every rank. */
    if (myid0 == Host_ID) {
      const char *interleave = getenv("OPENMX_BAND_NONCOL_INTERLEAVE_K");
      if (interleave != NULL) interleave_k = (atoi(interleave) != 0);
    }
    MPI_Bcast(&interleave_k, 1, MPI_INT, Host_ID, mpi_comm_level1);
    if (interleave_k)
      BandNonCol_InterleaveKOrder(T_knum, numprocs0, dense_k_owner, dense_k_order);
    else
      for (int point=0; point<T_knum; ++point) dense_k_order[point] = point;
  }
  if (use_k_dense_gpusolver && strcasecmp(mode,"scf")==0 &&
      (size_t)n2 <= SIZE_MAX / (size_t)MaxN / sizeof(dcomplex)) {
    k_cache_bytes = (size_t)n2 * (size_t)MaxN * sizeof(dcomplex);
    k_cache_limit = BandNonCol_KCacheLimit();
    if (k_cache_bytes <= k_cache_limit)
      k_evec_cache = (dcomplex**)calloc((size_t)T_knum, sizeof(dcomplex*));
  }
				  if (use_root_dense_gpusolver || use_k_dense_gpusolver){
				    BandNonCol_SetDenseGemmul8Defaults();
				    MPI_Barrier(mpi_comm_level1);
				  }

	  /****************************************************
	                make is1, ie1, is2, ie2
  ****************************************************/

  /* allocation */

  is1 = (int*)malloc(sizeof(int)*numprocs2);
  ie1 = (int*)malloc(sizeof(int)*numprocs2);
  is2 = (int*)malloc(sizeof(int)*numprocs2);
  ie2 = (int*)malloc(sizeof(int)*numprocs2);

  Num_Snd_EV = (int*)malloc(sizeof(int)*numprocs2);
  Num_Rcv_EV = (int*)malloc(sizeof(int)*numprocs2);

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

  index_Snd_i = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  index_Snd_j = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  EVec_Snd = (double*)malloc(sizeof(double)*Max_Num_Snd_EV*2);
  index_Rcv_i = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  index_Rcv_j = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  EVec_Rcv = (double*)malloc(sizeof(double)*Max_Num_Rcv_EV*2);

  /****************************************************
                      PrintMemory
  ****************************************************/

  if (firsttime && memoryusage_fileout) {
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

	    int max_concurrent_gpu_turns = k_dense_gpu_turns;
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

	    if (owns_dense_k_rank) BandNonCol_KOverlapPrepare(n,T_knum);

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

	      if (use_k_packed_cache){
	        if (owns_dense_k_group){
	          m_olp = Set_Hamiltonian_GpuSolver_Packed_Overlap();
	          m_h11 = Set_Hamiltonian_GpuSolver_Packed_H(0);
	          m_h22 = Set_Hamiltonian_GpuSolver_Packed_H(1);
	          m_h12 = Set_Hamiltonian_GpuSolver_Packed_H(2);
	          m_h12i = Set_Hamiltonian_GpuSolver_Packed_H(3);
	          m_i11 = Set_Hamiltonian_GpuSolver_Packed_ImNL(0);
	          m_i22 = Set_Hamiltonian_GpuSolver_Packed_ImNL(1);
	          m_i12 = Set_Hamiltonian_GpuSolver_Packed_ImNL(2);
	        }
	      }
	      else if (owns_dense_k_group){
	        m_olp = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_h11 = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_h22 = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_h12 = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_h12i = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_i11 = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_i22 = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_i12 = (double*)malloc(sizeof(double)*(size_t)size_H1);

	        if (m_olp==NULL || m_h11==NULL || m_h22==NULL || m_h12==NULL ||
	            m_h12i==NULL || m_i11==NULL || m_i22==NULL || m_i12==NULL){
	          free(m_olp);
	          free(m_h11);
	          free(m_h22);
	          free(m_h12);
	          free(m_h12i);
	          free(m_i11);
	          free(m_i22);
	          free(m_i12);
	          BandNonCol_AbortWithMessage("Failed to allocate dense k-point packed matrices in Band_DFT_NonCol.c.");
	        }
	      }
	      else {
	        pack_buffer = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        if (pack_buffer==NULL){
	          BandNonCol_AbortWithMessage("Failed to allocate dense k-point packing buffer in Band_DFT_NonCol.c.");
	        }
	      }

	      if (!use_k_packed_cache){
	        BandNonCol_PackDenseM1(CntOLP, owns_dense_k_group ? m_olp : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(nh[0],  owns_dense_k_group ? m_h11 : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(nh[1],  owns_dense_k_group ? m_h22 : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(nh[2],  owns_dense_k_group ? m_h12 : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(nh[3],  owns_dense_k_group ? m_h12i : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(ImNL[0],owns_dense_k_group ? m_i11 : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(ImNL[1],owns_dense_k_group ? m_i22 : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(ImNL[2],owns_dense_k_group ? m_i12 : pack_buffer,MP,order_GA);
	        free(pack_buffer);
	        pack_buffer = NULL;
	      }

	      for (int gpu_turn=group_first; gpu_turn<group_last; gpu_turn++){

	        kloop = dense_k_order[gpu_turn];
	        if (dense_k_owner[kloop]!=myid0) continue;

	        (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);

	        k1 = T_KGrids1[kloop];
	        k2 = T_KGrids2[kloop];
        k3 = T_KGrids3[kloop];

        rdw = BandNonCol_RootDenseWorkspace_Ensure(1,n,n2,MaxN,1,SCF_iter);
        {
          /* the transformed overlap of an earlier step of this cycle skips
             the build and the eigensolve of S(k) */
          const int reuse_overlap = BandNonCol_KOverlapRestore(kloop,k1,k2,k3,rdw->s_all);

          k_overlap_hits += reuse_overlap;
          if (BandNonCol_refine.kdense) {
            /* pass 1 of a refined step: the eigenvalues with the previous
               step's chemical potential, the basis to the host slot; pass 2
               builds the vectors, so nothing goes to the eigenvector cache */
            BandNonCol_kref[kloop].refined =
              BandNonCol_RootDenseSolveOneK_HIP(!reuse_overlap,n,n2,MaxN,kloop,k1,k2,k3,
                                                m_olp,m_h11,m_h22,m_h12,m_h12i,
                                                m_i11,m_i22,m_i12,
                                                packed_k_order,MP,ko,EIGEN,0,0,0,rdw,NULL,
                                                &BandNonCol_kref[kloop],
                                                openmx_eigen_refine_warm_band_step(SCF_iter),ChemP,1);
          }
          else {
            dcomplex *panel = NULL;

            if (k_evec_cache != NULL && k_cache_bytes <= k_cache_limit - k_cache_used)
              panel = k_evec_cache[kloop] = (dcomplex*)malloc(k_cache_bytes);
            /* the cached vectors go from d_C2 into the panel in one copy
               (the same compact layout, ld MaxN) */
            BandNonCol_RootDenseSolveOneK_HIP(!reuse_overlap,n,n2,MaxN,kloop,k1,k2,k3,
                                              m_olp,m_h11,m_h22,m_h12,m_h12i,
                                              m_i11,m_i22,m_i12,
                                              packed_k_order,MP,ko,EIGEN,panel!=NULL,0,0,rdw,panel,
                                              NULL,0,0.0,0);
            if (panel != NULL) k_cache_used += k_cache_bytes;
          }
          if (!reuse_overlap) BandNonCol_KOverlapSave(kloop,k1,k2,k3,rdw->s_all);
        }
	      }

	      if (owns_dense_k_group){
	        if (!use_k_packed_cache){
	          free(m_olp);
	          free(m_h11);
	          free(m_h22);
	          free(m_h12);
	          free(m_h12i);
	          free(m_i11);
	          free(m_i22);
	          free(m_i12);
	          m_olp = NULL;
	          m_h11 = NULL;
	          m_h22 = NULL;
	          m_h12 = NULL;
	          m_h12i = NULL;
	          m_i11 = NULL;
	          m_i22 = NULL;
	          m_i12 = NULL;
	        }

	        BandNonCol_ConstructCache_Reset();
	        BandNonCol_RootDenseWorkspace_Reset();
	      }

	      MPI_Barrier(mpi_comm_level1);
	    }
		  }
		  else if (use_root_dense_gpusolver){

		    /* owns_root_dense_rank and root_dense_turn_group were set before
		       the refinement's memory test, which depends on them */
		    const int serialize_gpu_turns = BandNonCol_SerializeGpuSolverGpuTurns();
		    /* the resident refined solves of this call (every k-world owner in
		       one turn group): the first halves, the chemical potential of
		       their estimates, the second halves */
		    const int refine_resident = BandNonCol_refine.last_refined;
		    int resident_kloop = -1;
		    int use_setham_packed_cache =
		      (Set_Hamiltonian_GpuSolver_Packed_CacheReady() &&
	       Set_Hamiltonian_GpuSolver_Packed_OrderMode()==1);
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

		    root_dense_serial_gpusolver_worlds = 0;

	    if (use_setham_packed_cache){
	      Set_Hamiltonian_GpuSolver_SetMP(MP);
	    }

	    if (refine_resident){
	      static int announced = 0;

	      if (!announced && myid0==Host_ID && 2 <= level_stdout){
	        printf("<Band_DFT_NonCol> refined eigensolver with every k-world owner resident (%d k-worlds, turn "
	               "group %d): the estimates of all k-points give the chemical potential of the refinement\n",
	               Num_Comm_World2,root_dense_turn_group);
	        fflush(stdout);
	        announced = 1;
	      }
	    }

	    /* a refined step (staged) solves every k-point twice: pass 0 with the
	       previous step's chemical potential, its basis to the host slot;
	       pass 1 from that basis with the chemical potential of pass 0's
	       eigenvalues, with the vectors */
	    for (int root_pass=(BandNonCol_refine.staged ? 0 : 1); root_pass<2; root_pass++){
	    double root_refine_mu = ChemP;

	    if (root_pass==1 && BandNonCol_refine.staged){
	      for (int kb=0; kb<T_knum; kb++){
	        MPI_Bcast(&EIGEN[0][kb][0],MaxN+1,MPI_DOUBLE,Comm_World_StartID2[kb],mpi_comm_level1);
	      }
	      root_refine_mu = BandNonCol_RefineChemP(EIGEN,T_knum,MaxN,T_k_op,sum_weights,TZ);
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
	          if (!Set_Hamiltonian_GpuSolver_Packed_OwnsCache()){
	            BandNonCol_AbortWithMessage("Set_Hamiltonian packed cache is not owned by this rank in Band_DFT_NonCol.c.");
	          }
	          packed_order_GA = Set_Hamiltonian_GpuSolver_Packed_OrderGA();
	          m_olp = Set_Hamiltonian_GpuSolver_Packed_Overlap();
	          m_h11 = Set_Hamiltonian_GpuSolver_Packed_H(0);
	          m_h22 = Set_Hamiltonian_GpuSolver_Packed_H(1);
	          m_h12 = Set_Hamiltonian_GpuSolver_Packed_H(2);
	          m_h12i = Set_Hamiltonian_GpuSolver_Packed_H(3);
	          m_i11 = Set_Hamiltonian_GpuSolver_Packed_ImNL(0);
	          m_i22 = Set_Hamiltonian_GpuSolver_Packed_ImNL(1);
	          m_i12 = Set_Hamiltonian_GpuSolver_Packed_ImNL(2);

	          if (packed_order_GA==NULL || m_olp==NULL || m_h11==NULL || m_h22==NULL ||
	              m_h12==NULL || m_h12i==NULL || m_i11==NULL || m_i22==NULL || m_i12==NULL){
	            BandNonCol_AbortWithMessage("Set_Hamiltonian packed matrix cache is missing in Band_DFT_NonCol.c.");
	          }
	        }
	      }
	      else if (owns_root_dense_group){
	        m_olp = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_h11 = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_h22 = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_h12 = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_h12i = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_i11 = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_i22 = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        m_i12 = (double*)malloc(sizeof(double)*(size_t)size_H1);

	        if (m_olp==NULL || m_h11==NULL || m_h22==NULL || m_h12==NULL ||
	            m_h12i==NULL || m_i11==NULL || m_i22==NULL || m_i12==NULL){
	          free(m_olp);
	          free(m_h11);
	          free(m_h22);
	          free(m_h12);
	          free(m_h12i);
	          free(m_i11);
	          free(m_i22);
	          free(m_i12);
	          BandNonCol_AbortWithMessage("Failed to allocate root dense packed matrices in Band_DFT_NonCol.c.");
	        }
	      }
	      else {
	        pack_buffer = (double*)malloc(sizeof(double)*(size_t)size_H1);
	        if (pack_buffer==NULL){
	          BandNonCol_AbortWithMessage("Failed to allocate root dense packing buffer in Band_DFT_NonCol.c.");
	        }
	      }

	      if (!use_setham_packed_cache){
	        BandNonCol_PackDenseM1(CntOLP, owns_root_dense_group ? m_olp : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(nh[0],  owns_root_dense_group ? m_h11 : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(nh[1],  owns_root_dense_group ? m_h22 : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(nh[2],  owns_root_dense_group ? m_h12 : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(nh[3],  owns_root_dense_group ? m_h12i : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(ImNL[0],owns_root_dense_group ? m_i11 : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(ImNL[1],owns_root_dense_group ? m_i22 : pack_buffer,MP,order_GA);
	        BandNonCol_PackDenseM1(ImNL[2],owns_root_dense_group ? m_i12 : pack_buffer,MP,order_GA);
	        free(pack_buffer);
	        pack_buffer = NULL;
	      }

	      for (int root_dense_world=group_first; root_dense_world<group_last; root_dense_world++){
	        int root_dense_owner = Comm_World_StartID2[root_dense_world];
	        int owns_root_dense = (myid0==root_dense_owner);

	        if (owns_root_dense || myworld2==root_dense_world){

	          rdw = BandNonCol_RootDenseWorkspace_Ensure(owns_root_dense,n,n2,MaxN,1,SCF_iter);
	          root_s_valid = 0;
	          if (owns_root_dense) root_s_valid = rdw->s_valid;
	          rebuild_overlap = (SCF_iter==1 || !root_s_valid);

	          kloop = root_dense_world;
	          (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);
	          k1 = T_KGrids1[kloop];
	          k2 = T_KGrids2[kloop];
	          k3 = T_KGrids3[kloop];

	          if (owns_root_dense && root_pass==0){
	            /* pass 1 of a refined step: the eigenvalues, the basis to the
	               host slot, no vectors */
	            BandNonCol_kref[kloop].refined =
	              BandNonCol_RootDenseSolveOneK_HIP(rebuild_overlap,n,n2,MaxN,kloop,k1,k2,k3,
	                                                m_olp,m_h11,m_h22,m_h12,m_h12i,
	                                                m_i11,m_i22,m_i12,
	                                                packed_order_GA,MP,ko,EIGEN,0,0,0,rdw,NULL,
	                                                &BandNonCol_kref[kloop],
	                                                openmx_eigen_refine_warm_band_step(SCF_iter),ChemP,1);
	            rdw->s_valid = 1;
	          }
	          else if (owns_root_dense && refine_resident){
	            /* the first half of a resident refined solve: pending with
	               d_H2 and the refinement's buffers on the device until the
	               chemical potential is known, or solved in FP64 */
	            if (BandNonCol_RootDenseRefinedFirstHalf(rebuild_overlap,n,n2,MaxN,kloop,k1,k2,k3,
	                                                     m_olp,m_h11,m_h22,m_h12,m_h12i,
	                                                     m_i11,m_i22,m_i12,
	                                                     packed_order_GA,MP,ko,EIGEN,
	                                                     !use_device_eigenvector_dm,
	                                                     use_backtransform_barrier,rdw)){
	              resident_kloop = kloop;
	            }
	            rdw->s_valid = 1;
	          }
	          else if (owns_root_dense){
	            BandNonColKRefineSlot *refine_slot =
	              (BandNonCol_refine.staged && BandNonCol_kref[kloop].refined) ? &BandNonCol_kref[kloop] : NULL;

	            BandNonCol_RootDenseSolveOneK_HIP(rebuild_overlap,n,n2,MaxN,kloop,k1,k2,k3,
	                                              m_olp,m_h11,m_h22,m_h12,m_h12i,
	                                              m_i11,m_i22,m_i12,
	                                              packed_order_GA,MP,ko,EIGEN,1,
	                                              !use_device_eigenvector_dm,
	                                              use_backtransform_barrier,rdw,NULL,
	                                              refine_slot,1,root_refine_mu,
	                                              openmx_eigen_refine_warm_band_enabled());
	          }

	          /* the resident solves distribute their vectors after the second
	             halves */
	          if (root_pass==1 && !use_backtransform_barrier && !refine_resident){
	            if (!use_device_eigenvector_dm){
	              if (owns_root_dense && !rdw->cs2_valid){
	                BandNonCol_AbortWithMessage("Root dense host eigenvectors are not available for distribution.");
	              }
	              BandNonCol_DistributeDenseEvecGlobal(kloop,n2,MaxN,myid0,myworld2,NPROCS_WD2,
	                                                   Comm_World_StartID2,root_dense_owner,
	                                                   owns_root_dense ? rdw->cs2 : NULL,EVec1[0]);
	            }

	            if (owns_root_dense) rdw->s_valid = 1;
	          }
	        }
	      }

	      if (refine_resident){
	        /* every rank: the estimates of all k-points (the eigenvalues of an
	           owner that solved in FP64 instead) give the chemical potential
	           with which the pending owners finish their refinements; the
	           vectors follow as after an unrefined solve */
	        double resident_mu;

	        for (int kb=0; kb<T_knum; kb++){
	          MPI_Bcast(&EIGEN[0][kb][0],MaxN+1,MPI_DOUBLE,Comm_World_StartID2[kb],mpi_comm_level1);
	        }
	        resident_mu = BandNonCol_RefineChemP(EIGEN,T_knum,MaxN,T_k_op,sum_weights,TZ);
	        if (BandNonCol_refine.pending && 0<=resident_kloop){
	          BandNonCol_RootDenseRefinedSecondHalf(n,n2,MaxN,resident_kloop,ko,EIGEN,resident_mu,
	                                                !use_device_eigenvector_dm,use_backtransform_barrier,
	                                                &BandNonCol_root_dense_workspace);
	        }
	        BandNonCol_RefineEndStep();

	        if (!use_backtransform_barrier){
	          for (int root_dense_world=group_first; root_dense_world<group_last; root_dense_world++){
	            int root_dense_owner = Comm_World_StartID2[root_dense_world];
	            int owns_root_dense = (myid0==root_dense_owner);

	            if (owns_root_dense || myworld2==root_dense_world){
	              rdw = &BandNonCol_root_dense_workspace;
	              if (!use_device_eigenvector_dm){
	                if (owns_root_dense && !rdw->cs2_valid){
	                  BandNonCol_AbortWithMessage("Root dense host eigenvectors are not available for distribution.");
	                }
	                BandNonCol_DistributeDenseEvecGlobal(root_dense_world,n2,MaxN,myid0,myworld2,NPROCS_WD2,
	                                                     Comm_World_StartID2,root_dense_owner,
	                                                     owns_root_dense ? rdw->cs2 : NULL,EVec1[0]);
	              }
	              if (owns_root_dense) rdw->s_valid = 1;
	            }
	          }
	        }
	      }

	      if (root_pass==1 && use_backtransform_barrier){
	        /* Every root owner has completed its H eigensolve before any rank
	           starts the two bandwidth-heavy backtransform GEMMs. */
	        MPI_Barrier(mpi_comm_level1);

	        for (int root_dense_world=group_first; root_dense_world<group_last; root_dense_world++){
	          int root_dense_owner = Comm_World_StartID2[root_dense_world];
	          int owns_root_dense = (myid0==root_dense_owner);

	          if (owns_root_dense || myworld2==root_dense_world){
	            rdw = BandNonCol_RootDenseWorkspace_Ensure(owns_root_dense,n,n2,MaxN,1,SCF_iter);

	            if (owns_root_dense){
	              BandNonCol_RootDenseBacktransform_HIP(n,n2,MaxN,
	                                                     !use_device_eigenvector_dm,rdw,NULL);
	            }

	            if (!use_device_eigenvector_dm){
	              if (owns_root_dense && !rdw->cs2_valid){
	                BandNonCol_AbortWithMessage("Deferred root dense host eigenvectors are not available for distribution.");
	              }
	              BandNonCol_DistributeDenseEvecGlobal(root_dense_world,n2,MaxN,
	                                                   myid0,myworld2,NPROCS_WD2,
	                                                   Comm_World_StartID2,root_dense_owner,
	                                                   owns_root_dense ? rdw->cs2 : NULL,EVec1[0]);
	            }

	            if (owns_root_dense) rdw->s_valid = 1;
	          }
	        }
	      }

	      if (!use_setham_packed_cache && owns_root_dense_group){
	        free(m_olp);
	        free(m_h11);
	        free(m_h22);
	        free(m_h12);
	        free(m_h12i);
	        free(m_i11);
	        free(m_i22);
	        free(m_i12);
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
	        BandNonCol_ConstructCache_Reset();
	      }

	      if (serialize_gpu_turns || use_backtransform_barrier){
	        MPI_Barrier(mpi_comm_level1);
	      }
	    }

	    } /* root_pass */
	    if (BandNonCol_refine.staged){
	      BandNonCol_RefineEndStep();
	    }

	    if (!serialize_gpu_turns && !use_backtransform_barrier){
	      MPI_Barrier(mpi_comm_level1);
	    }
	  }
	  else {

  for (kloop0=0; kloop0<max_num_kloop0; kloop0++){

    /* get k1, k2, and k3 */

    if (kloop0<num_kloop0){

      kloop = S_knum + kloop0;
      (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);
      k1 = T_KGrids1[kloop];
      k2 = T_KGrids2[kloop];
      k3 = T_KGrids3[kloop];
    }

    if (measure_time) dtime(&Stime);

    if (SCF_iter==1 || all_knum!=1){

      /* make Cs */

      for(i=0; i<na_rows*na_cols; i++){ Cs[i] = Complex(0.0,0.0);}
      Construct_Band_Ms(0,CntOLP,H1,Cs,MP,k1,k2,k3);

      /* diagonalize Cs */

      if (kloop0<num_kloop0){

	MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
	MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

	mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
	mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

	if (scf_eigen_lib_flag==GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2) &&
	    !BandNonCol_gpu_dense_refused && na_rows==n && na_cols==n){
	  BandNonCol_GpuSolver_DenseZheevx(Cs,Ss,ko,n,n,openmx_gpu_eigensolver_use_hipsolver(),
                                            "Band_DFT_NonCol overlap");
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

    Construct_Band_Ms(0,nh[0],  H1, Hs11,MP,k1,k2,k3);
    Construct_Band_Ms(0,nh[1],  H1, Hs22,MP,k1,k2,k3);
    Construct_Band_Ms(0,nh[2],  H1, Hs12,MP,k1,k2,k3);
    Construct_Band_Ms(1,nh[3],  H1, Hs12,MP,k1,k2,k3);
    Construct_Band_Ms(1,ImNL[0],H1, Hs11,MP,k1,k2,k3);
    Construct_Band_Ms(1,ImNL[1],H1, Hs22,MP,k1,k2,k3);
    Construct_Band_Ms(1,ImNL[2],H1, Hs12,MP,k1,k2,k3);

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

      if (BandNonCol_UseDenseGpuMatrix(n,n2)){
        BandNonCol_DenseTripleTransform_OpenMP(n,Hs11,Ss,Cs);
        BandNonCol_DenseTripleTransform_OpenMP(n,Hs12,Ss,Cs);
        BandNonCol_DenseTripleTransform_OpenMP(n,Hs22,Ss,Cs);
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

        if (scf_eigen_lib_flag==GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2) &&
            !BandNonCol_gpu_dense_refused && na_rows2==n2 && na_cols2==n2){
          BandNonCol_GpuSolver_DenseZheevx(Hs2,Cs2,ko,n2,MaxN,
                                           BandNonCol_HamiltonianUseHipSolver(),
                                           "Band_DFT_NonCol Hamiltonian");
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

      if (BandNonCol_UseDenseGpuMatrix(n,n2)){
        BandNonCol_DenseWavefunctions_OpenMP(n,n2,MaxN,Cs2,Ss,Hs2);
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

    ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
    Num_State = 0.0;

    for (kloop=0; kloop<T_knum; kloop++){
      for (l=1; l<=MaxN; l++){

	x = (EIGEN[0][kloop][l] - ChemP)*Beta_trial1;

	if (x<=-x_cut) x = -x_cut;
	if (x_cut<=x)  x =  x_cut;
	FermiF = FermiFunc_NC(x,l);
	Num_State += FermiF*(double)T_k_op[kloop];
      } 
    } 

    Num_State = Num_State/sum_weights;
    Dnum = TZ - Num_State - system_charge;

    if (0.0<=Dnum) ChemP_MIN = ChemP;
    else           ChemP_MAX = ChemP;
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
      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
    }

    Num_State = 0.0;

    for (kloop=0; kloop<T_knum; kloop++){
      for (l=1; l<=MaxN; l++){

	x = (EIGEN[0][kloop][l] - ChemP)*Beta;

	if (x<=-x_cut) x = -x_cut;
	if (x_cut<=x)  x =  x_cut;
	FermiF = FermiFunc_NC(x,l);

	Num_State += FermiF*(double)T_k_op[kloop];

      } 
    } 

    Num_State = Num_State/sum_weights;
    Dnum = TZ - Num_State - system_charge;

    if (0.0<=Dnum) ChemP_MIN = ChemP;
    else           ChemP_MAX = ChemP;
    if (fabs(Dnum)<1.0e-12) po = 1;
  }
  while (po==0 && loop_num<1000);

  /* for the NEGF calculation */
  if (Solver==4 && TRAN_ChemP_Band==0) ChemP = 0.5*(ChemP_e[0]+ChemP_e[1]);

  /****************************************************
           band energy in a finite temperature
  ****************************************************/

  Eele0[0] = 0.0;
  Eele0[1] = 0.0;

  for (kloop=0; kloop<T_knum; kloop++){
    for (l=1; l<=MaxN; l++){

      x = (EIGEN[0][kloop][l] - ChemP)*Beta;

      if (x<=-x_cut) x = -x_cut;
      if (x_cut<=x)  x = x_cut;
      FermiF = FermiFunc_NC(x,l);

      Eele0[0] += FermiF*EIGEN[0][kloop][l]*(double)T_k_op[kloop];
    }
  } 

  Eele0[0] = Eele0[0]/sum_weights;
  Uele = Eele0[0];

  if (2<=level_stdout){
    printf("myid0=%2d ChemP=%lf, Eele0[0]=%lf, Eele0[1]=%lf\n",myid0,ChemP,Eele0[0],Eele0[1]);
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

    (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);

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
	           Set_Hamiltonian_OpenMP_Rank_Is_Selected());
	        int root_dense_turn_group =
	          serialize_gpu_turns ?
	          BandNonCol_GpuTurnGroup(n,n2,MaxN,size_H1,owns_root_dense_rank,myid0) :
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
	              BandNonColRootDenseWorkspace *rdw = &BandNonCol_root_dense_workspace;
	              const dcomplex *d_dense_evec = NULL;

	              if (!rdw->valid){
	                BandNonCol_AbortWithMessage("Root dense eigenvectors are not available for Band_DFT_NonCol DM.");
	              }
	              if (use_device_eigenvector_dm){
	                d_dense_evec = BandNonCol_DenseGpu_DeviceEigenvectors(n2,MaxN);
	                if (d_dense_evec==NULL){
	                  BandNonCol_AbortWithMessage("Root dense device eigenvectors are not available for Band_DFT_NonCol DM.");
	                }
	              }
	              else if (!rdw->cs2_valid || rdw->cs2==NULL){
	                BandNonCol_AbortWithMessage("Root dense host eigenvectors are not available for Band_DFT_NonCol DM.");
	              }

	              kloop = root_dense_world;
	              (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);
	              k1 = T_KGrids1[kloop];
	              k2 = T_KGrids2[kloop];
	              k3 = T_KGrids3[kloop];

	              BandNonCol_AccumulateDMRootDenseK_HIP(size_H1,MP,n,n2,MaxN,k1,k2,k3,
	                                                    EIGEN[0][kloop],
	                                                    use_device_eigenvector_dm ? NULL : rdw->cs2,
	                                                    d_dense_evec,
	                                                    use_device_eigenvector_dm ? MaxN : 0,
	                                                    rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
	                                                    rEDM11,rEDM22);
	            } else { BandNonCol_AccumulateDMRootDenseK_HIP(size_H1,MP,n,n2,MaxN,0,0,0,NULL,
                      NULL,NULL,0,rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,rEDM11,rEDM22); }
	          }

	          if (serialize_gpu_turns){
	            MPI_Barrier(mpi_comm_level1);
	          }
	        }

	        if (!serialize_gpu_turns){
	          MPI_Barrier(mpi_comm_level1);
	        }

	        kloop = 0;
	        (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,0);
	        k1 = T_KGrids1[0];
	        k2 = T_KGrids2[0];
	        k3 = T_KGrids3[0];
	        Calc_DM_Band_non_collinear( 0,1,
					    myid0,myid2,size_H1,
					    is2,ie2,MP,n,n2,MaxN,k1,k2,k3,
					    CDM,iDM[0],EDM,EIGEN[0][0],
					    EVec1[0],
					    rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
					    rEDM11,rEDM22 );
	      }
      else if (scf_eigen_lib_flag==GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2)){
        BandNonCol_CalcDMAllK1_OpenMP( myid0,myid2,size_H1,
					is2,ie2,MP,n,n2,k1,k2,k3,
					CDM,iDM[0],EDM,EIGEN[0][kloop],
					EVec1[0],
					rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
					rEDM11,rEDM22 );
      }
      else {
        Calc_DM_Band_non_collinear( 1,1,
				    myid0,myid2,size_H1,
					    is2,ie2,MP,n,n2,MaxN,k1,k2,k3,
				    CDM,iDM[0],EDM,EIGEN[0][kloop],
				    EVec1[0],
				    rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
				    rEDM11,rEDM22 );
      }
    }

    else if ( strcasecmp(mode,"ParDM")==0 ){ 

      Calc_ParDM_Band_non_collinear( 1,1,
				     myid0,myid2,size_H1,
				     is2,ie2,MP,n,n2,MaxN,k1,k2,k3, 
				     ParDM,iParDM,EIGEN[0][kloop],EVec1[0],
				     rDM11,rDM22,rDM12,iDM12,iDM11,iDM22);
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

  /* Debug hook: exit cleanly right after the first eigen phase so that
     profilers (rocprofv3) flush their traces.  OPENMX_ABORT_AFTER_EIGEN=1. */
  {
    const char *abort_after_eigen = getenv("OPENMX_ABORT_AFTER_EIGEN");
    if (abort_after_eigen != NULL && atoi(abort_after_eigen) != 0){
      fflush(stdout);
      fflush(stderr);
      MPI_Barrier(mpi_comm_level1);
      MPI_Finalize();
      exit(0);
    }
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
	        BandNonCol_MaxConcurrentKGpuTurns(n,n2,MaxN,size_H1,owns_dense_k_rank,myid0);
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
		        int local_needs_solve = 0, group_needs_solve = 1;

		        for (int gpu_turn=group_first; gpu_turn<group_last; gpu_turn++){
		          if (dense_k_owner[dense_k_order[gpu_turn]]==myid0){
		            owns_dense_k_group = 1;
		            if (k_evec_cache == NULL || k_evec_cache[dense_k_order[gpu_turn]] == NULL)
		              local_needs_solve = 1;
		          }
		        }
		        /* Without the shared packed copy, all ranks skip the gathers of
		           a group whose owners all kept the eigenvectors of their
		           k-points from pass 1 (collective, as the gathers are). */
		        if (!use_k_packed_cache){
		          MPI_Allreduce(&local_needs_solve,&group_needs_solve,1,MPI_INT,MPI_MAX,mpi_comm_level1);
		          if (group_needs_solve) k_gather_groups++;
		          else k_gather_skips++;
		        }

		        MPI_Barrier(mpi_comm_level1);

		        if (use_k_packed_cache){
		          if (owns_dense_k_group){
		            m_olp = Set_Hamiltonian_GpuSolver_Packed_Overlap();
		            m_h11 = Set_Hamiltonian_GpuSolver_Packed_H(0);
		            m_h22 = Set_Hamiltonian_GpuSolver_Packed_H(1);
		            m_h12 = Set_Hamiltonian_GpuSolver_Packed_H(2);
		            m_h12i = Set_Hamiltonian_GpuSolver_Packed_H(3);
		            m_i11 = Set_Hamiltonian_GpuSolver_Packed_ImNL(0);
		            m_i22 = Set_Hamiltonian_GpuSolver_Packed_ImNL(1);
		            m_i12 = Set_Hamiltonian_GpuSolver_Packed_ImNL(2);
		          }
		        }
		        else if (group_needs_solve && owns_dense_k_group){
		          m_olp = (double*)malloc(sizeof(double)*(size_t)size_H1);
		          m_h11 = (double*)malloc(sizeof(double)*(size_t)size_H1);
		          m_h22 = (double*)malloc(sizeof(double)*(size_t)size_H1);
		          m_h12 = (double*)malloc(sizeof(double)*(size_t)size_H1);
		          m_h12i = (double*)malloc(sizeof(double)*(size_t)size_H1);
		          m_i11 = (double*)malloc(sizeof(double)*(size_t)size_H1);
		          m_i22 = (double*)malloc(sizeof(double)*(size_t)size_H1);
		          m_i12 = (double*)malloc(sizeof(double)*(size_t)size_H1);

		          if (m_olp==NULL || m_h11==NULL || m_h22==NULL || m_h12==NULL ||
		              m_h12i==NULL || m_i11==NULL || m_i22==NULL || m_i12==NULL){
		            free(m_olp);
		            free(m_h11);
		            free(m_h22);
		            free(m_h12);
		            free(m_h12i);
		            free(m_i11);
		            free(m_i22);
		            free(m_i12);
		            BandNonCol_AbortWithMessage("Failed to allocate dense k-point DM packed matrices in Band_DFT_NonCol.c.");
		          }
		        }
		        else if (group_needs_solve){
		          pack_buffer = (double*)malloc(sizeof(double)*(size_t)size_H1);
		          if (pack_buffer==NULL){
		            BandNonCol_AbortWithMessage("Failed to allocate dense k-point DM packing buffer in Band_DFT_NonCol.c.");
		          }
		        }

		        if (!use_k_packed_cache && group_needs_solve){
		          BandNonCol_PackDenseM1(CntOLP, owns_dense_k_group ? m_olp : pack_buffer,MP,order_GA);
		          BandNonCol_PackDenseM1(nh[0],  owns_dense_k_group ? m_h11 : pack_buffer,MP,order_GA);
		          BandNonCol_PackDenseM1(nh[1],  owns_dense_k_group ? m_h22 : pack_buffer,MP,order_GA);
		          BandNonCol_PackDenseM1(nh[2],  owns_dense_k_group ? m_h12 : pack_buffer,MP,order_GA);
		          BandNonCol_PackDenseM1(nh[3],  owns_dense_k_group ? m_h12i : pack_buffer,MP,order_GA);
		          BandNonCol_PackDenseM1(ImNL[0],owns_dense_k_group ? m_i11 : pack_buffer,MP,order_GA);
		          BandNonCol_PackDenseM1(ImNL[1],owns_dense_k_group ? m_i22 : pack_buffer,MP,order_GA);
		          BandNonCol_PackDenseM1(ImNL[2],owns_dense_k_group ? m_i12 : pack_buffer,MP,order_GA);
		          free(pack_buffer);
		          pack_buffer = NULL;
		        }

		        for (int gpu_turn=group_first; gpu_turn<group_last; gpu_turn++){

		          kloop = dense_k_order[gpu_turn];
		          (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);
              if (dense_k_owner[kloop]!=myid0) { BandNonCol_AccumulateDMRootDenseK_HIP(size_H1,MP,n,n2,MaxN,0,0,0,NULL,
                      NULL,NULL,0,rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,rEDM11,rEDM22); continue; }

		          (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);

		          k1 = T_KGrids1[kloop];
		          k2 = T_KGrids2[kloop];
	          k3 = T_KGrids3[kloop];

	          rdw = BandNonCol_RootDenseWorkspace_Ensure(1,n,n2,MaxN,1,SCF_iter);
              if (k_evec_cache != NULL && k_evec_cache[kloop] != NULL) {
                for (int basis=0; basis<n2; ++basis)
                  memcpy(rdw->cs2 + (size_t)basis*n2,
                         k_evec_cache[kloop] + (size_t)basis*MaxN,
                         sizeof(dcomplex)*(size_t)MaxN);
                rdw->cs2_valid = 1;
                free(k_evec_cache[kloop]);
                k_evec_cache[kloop] = NULL;
                ++k_cache_hits;
              }
              else {
                /* pass 2 of a k-point refined in pass 1: from its basis, with
                   the step's chemical potential */
                BandNonColKRefineSlot *refine_slot =
                  (BandNonCol_refine.kdense && BandNonCol_kref != NULL && BandNonCol_kref[kloop].refined) ?
                  &BandNonCol_kref[kloop] : NULL;
                const int reuse_overlap = BandNonCol_KOverlapRestore(kloop,k1,k2,k3,rdw->s_all);

                k_overlap_hits += reuse_overlap;
                BandNonCol_RootDenseSolveOneK_HIP(!reuse_overlap,n,n2,MaxN,kloop,k1,k2,k3,
                                                  m_olp,m_h11,m_h22,m_h12,m_h12i,
                                                  m_i11,m_i22,m_i12,
                                                  packed_k_order,MP,ko,EIGEN,1,1,0,rdw,NULL,
                                                  refine_slot,1,ChemP,
                                                  openmx_eigen_refine_warm_band_enabled());
                if (!reuse_overlap) BandNonCol_KOverlapSave(kloop,k1,k2,k3,rdw->s_all);
              }
	          BandNonCol_AccumulateDMRootDenseK_HIP(size_H1,MP,n,n2,MaxN,k1,k2,k3,
	                                                EIGEN[0][kloop],rdw->cs2,NULL,0,
		                                                rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
		                                                rEDM11,rEDM22);
		        }

		        if (owns_dense_k_group){
		          if (!use_k_packed_cache){
		            free(m_olp);
		            free(m_h11);
		            free(m_h22);
		            free(m_h12);
		            free(m_h12i);
		            free(m_i11);
		            free(m_i22);
		            free(m_i12);
		            m_olp = NULL;
		            m_h11 = NULL;
		            m_h22 = NULL;
		            m_h12 = NULL;
		            m_h12i = NULL;
		            m_i11 = NULL;
		            m_i22 = NULL;
		            m_i12 = NULL;
		          }

		          BandNonCol_ConstructCache_Reset();
		          BandNonCol_RootDenseWorkspace_Reset();
		        }

		        MPI_Barrier(mpi_comm_level1);
		      }

		      /* every rank: the refined solves of the two passes decide the
		         stage */
		      if (BandNonCol_refine.kdense){
		        BandNonCol_RefineEndStep();
		      }

		      kloop = 0;
	      (void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,0);
	      k1 = T_KGrids1[0];
	      k2 = T_KGrids2[0];
	      k3 = T_KGrids3[0];
	      Calc_DM_Band_non_collinear( 0,1,
					  myid0,myid2,size_H1,
					  is2,ie2,MP,n,n2,MaxN,k1,k2,k3,
					  CDM,iDM[0],EDM,EIGEN[0][0],
					  EVec1[0],
					  rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
					  rEDM11,rEDM22 );
	    }
	    else {

	    for (kloop0=0; kloop0<max_num_kloop0; kloop0++){

      /* get k1, k2, and k3 */

      if (kloop0<num_kloop0){

	kloop = S_knum + kloop0;
	(void)openmx_mgpu_band_scope_enter(scf_eigen_lib_flag==GPUSOLVER && !gpusolver2_flag,T_knum,kloop);
	k1 = T_KGrids1[kloop];
	k2 = T_KGrids2[kloop];
	k3 = T_KGrids3[kloop];
      }

      if (measure_time) dtime(&Stime);

      /* make Cs */

      for(i=0; i<na_rows*na_cols; i++){ Cs[i] = Complex(0.0,0.0);}
      Construct_Band_Ms(0,CntOLP,H1,Cs,MP,k1,k2,k3);

      /* diagonalize Cs */

      if (kloop0<num_kloop0){

	MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
	MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

	mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
	mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

        if (scf_eigen_lib_flag==GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2) &&
	    !BandNonCol_gpu_dense_refused && na_rows==n && na_cols==n){
          BandNonCol_GpuSolver_DenseZheevx(Cs,Ss,ko,n,n,openmx_gpu_eigensolver_use_hipsolver(),
                                           "Band_DFT_NonCol overlap");
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

      Construct_Band_Ms(0,nh[0],  H1, Hs11,MP,k1,k2,k3);
      Construct_Band_Ms(0,nh[1],  H1, Hs22,MP,k1,k2,k3);
      Construct_Band_Ms(0,nh[2],  H1, Hs12,MP,k1,k2,k3);
      Construct_Band_Ms(1,nh[3],  H1, Hs12,MP,k1,k2,k3);
      Construct_Band_Ms(1,ImNL[0],H1, Hs11,MP,k1,k2,k3);
      Construct_Band_Ms(1,ImNL[1],H1, Hs22,MP,k1,k2,k3);
      Construct_Band_Ms(1,ImNL[2],H1, Hs12,MP,k1,k2,k3);

      if (kloop0<num_kloop0){

        if (BandNonCol_UseDenseGpuMatrix(n,n2)){
          BandNonCol_DenseTripleTransform_OpenMP(n,Hs11,Ss,Cs);
          BandNonCol_DenseTripleTransform_OpenMP(n,Hs12,Ss,Cs);
          BandNonCol_DenseTripleTransform_OpenMP(n,Hs22,Ss,Cs);
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
  
        if (scf_eigen_lib_flag==GPUSOLVER && (openmx_mgpu_host_required() || Band_DFT_NonCol_GpuSwitchNum()<=n2) &&
            !BandNonCol_gpu_dense_refused && na_rows2==n2 && na_cols2==n2){
          BandNonCol_GpuSolver_DenseZheevx(Hs2,Cs2,ko,n2,MaxN,
                                           BandNonCol_HamiltonianUseHipSolver(),
                                           "Band_DFT_NonCol Hamiltonian");
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

        if (BandNonCol_UseDenseGpuMatrix(n,n2)){
          BandNonCol_DenseWavefunctions_OpenMP(n,n2,MaxN,Cs2,Ss,Hs2);
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

        if (BandNonCol_UseGpuFallbackDM(n2,(size_t)n2*(size_t)(ie2[myid2]-is2[myid2]+1)*sizeof(dcomplex))){
          /* accumulates this k on the GPU; EVec1 stays unscaled because the
             kernel applies the Fermi weights itself */
          BandNonCol_AccumulateDMKPoint_OpenMP(myid2,is2,ie2,MP,n,n2,size_H1,k1,k2,k3,
                                               EIGEN[0][kloop],EVec1[0],
                                               rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
                                               rEDM11,rEDM22,kloop0<num_kloop0,mpi_comm_level1);
        }
        else{
	Calc_DM_Band_non_collinear( (kloop0<num_kloop0),0,
				    myid0,myid2,size_H1,
				    is2,ie2,MP,n,n2,MaxN,k1,k2,k3,
				    CDM,iDM[0],EDM,EIGEN[0][kloop],
				    EVec1[0],
				    rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
				    rEDM11,rEDM22 );
        }
      }

      else if ( strcasecmp(mode,"ParDM")==0 ){

	Calc_ParDM_Band_non_collinear( (kloop0<num_kloop0),0,
				       myid0,myid2,size_H1,
				       is2,ie2,MP,n,n2,MaxN,k1,k2,k3, 
				       ParDM,iParDM,EIGEN[0][kloop],EVec1[0],
				       rDM11,rDM22,rDM12,iDM12,iDM11,iDM22);

      }

      if (measure_time){
        dtime(&Etime);
        time12 += Etime - Stime;
      }

    } /* kloop0 */

    /* store DM and iDM */

    if ( strcasecmp(mode,"scf")==0 ){ 

      Calc_DM_Band_non_collinear( 0,1,
				  myid0,myid2,size_H1,
				  is2,ie2,MP,n,n2,MaxN,k1,k2,k3, 
				  CDM,iDM[0],EDM,EIGEN[0][kloop],
				  EVec1[0],
				  rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
				  rEDM11,rEDM22 );

    }

	    else if ( strcasecmp(mode,"ParDM")==0 ){ 

	      Calc_ParDM_Band_non_collinear( 0,1,
				     myid0,myid2,size_H1,
				     is2,ie2,MP,n,n2,MaxN,k1,k2,k3, 
				     ParDM,iParDM,EIGEN[0][kloop],EVec1[0],
				     rDM11,rDM22,rDM12,iDM12,iDM11,iDM22);

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
	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",ChemP);
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

  /****************************************************
                       free arrays
  ****************************************************/

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ReEVec_i1[i]);
  }
  free(ReEVec_i1);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ImEVec_i1[i]);
  }
  free(ImEVec_i1);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ReEVec_i2[i]);
  }
  free(ReEVec_i2);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ImEVec_i2[i]);
  }
  free(ImEVec_i2);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ReEVec_j1[i]);
  }
  free(ReEVec_j1);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ImEVec_j1[i]);
  }
  free(ImEVec_j1);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ReEVec_j2[i]);
  }
  free(ReEVec_j2);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ImEVec_j2[i]);
  }
  free(ImEVec_j2);

  free(rDM11);
  free(rDM22);
  free(rDM12);
  free(iDM11);
  free(iDM22);
  free(iDM12);
  free(rEDM11);
  free(rEDM22);

  free(is1);
  free(ie1);

  free(is2);
  free(ie2);

  free(Num_Snd_EV);
  free(Num_Rcv_EV);

  free(index_Snd_i);
  free(index_Snd_j);
  free(EVec_Snd); 
  free(index_Rcv_i);
  free(index_Rcv_j);
  free(EVec_Rcv);
  free(dense_k_owner);
  free(dense_k_order);
  if (k_evec_cache != NULL) {
    for (int point=0; point<T_knum; ++point) free(k_evec_cache[point]);
    free(k_evec_cache);
    if (BandNonCol_GpuVerbose())
      printf("<Band> rank %d SCF %d: reused %d k-point eigensolves, host cache %.1f MiB (limit %.1f MiB).\n",
             myid0, SCF_iter, k_cache_hits, k_cache_used/(1024.0*1024.0),
             k_cache_limit/(1024.0*1024.0));
  }
  if (use_k_dense_gpusolver) {
    /* OPENMX_BAND_NONCOL_CACHE_TRACE=1: the k-dense path's caches per rank
       and call (the packed matrices are Set_Hamiltonian's shared copy, or
       gathered per turn group of the density-matrix pass unless every
       owner of the group kept its eigenvectors) */
    const char *trace = getenv("OPENMX_BAND_NONCOL_CACHE_TRACE");

    if (trace != NULL && atoi(trace) != 0) {
      printf("BAND_NONCOL_CACHE rank=%d SCF=%d evec_hits=%d overlap_hits=%d evec=%.1f MiB overlap=%.1f MiB "
             "packed=%s dm_gathers=%d dm_gathers_skipped=%d\n",
             myid0, SCF_iter, k_cache_hits, k_overlap_hits, k_cache_used/1048576.0,
             BandNonCol_k_overlap_used/1048576.0, use_k_packed_cache ? "shared" : "gathered",
             k_gather_groups, k_gather_skips);
      fflush(stdout);
    }
  }

  /* for PrintMemory and allocation */
  firsttime=0;

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
  openmx_mgpu_band_scope_leave(input_scope);
  return time0;
}




static void BandNonCol_PackDenseM1( double ****Mat, double *M1, int *MP, int *order_GA )
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

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  My_Matomnum = (int*)malloc(sizeof(int)*numprocs);
  is1 = (int*)malloc(sizeof(int)*numprocs);
  is2 = (int*)malloc(sizeof(int)*numprocs);

  if (My_NZeros==NULL || My_Matomnum==NULL || is1==NULL || is2==NULL){
    free(My_NZeros);
    free(My_Matomnum);
    free(is1);
    free(is2);
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

  free(My_NZeros);
  free(My_Matomnum);
  free(is1);
  free(is2);
}

static void BandNonCol_ConstructDenseMsFromPacked( int cpx_flag, const double *M1, dcomplex *Ms,
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

  if (!BandNonCol_mgpu_host_dense && scf_eigen_lib_flag==GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum()<=2*n){
    BandNonCol_ConstructCache_Ensure(order_GA,MP,n);
    BandNonCol_ConstructDenseMs_OpenMP(cpx_flag,n,k1,k2,k3,M1,Ms);
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

static void Construct_Band_DenseMs( int cpx_flag, double ****Mat, double *M1, dcomplex *Ms,
                                    int *MP, double k1, double k2, double k3,
                                    int n, int owns_dense )
{
  static int firsttime=1;
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

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  My_Matomnum = (int*)malloc(sizeof(int)*numprocs);
  is1 = (int*)malloc(sizeof(int)*numprocs);
  is2 = (int*)malloc(sizeof(int)*numprocs);
  order_GA = (int*)malloc(sizeof(int)*(atomnum+2));

  if (firsttime && memoryusage_fileout) {
    PrintMemory("Construct_Band_DenseMs: My_NZeros", sizeof(int)*numprocs,NULL);
    PrintMemory("Construct_Band_DenseMs: is1", sizeof(int)*numprocs,NULL);
    PrintMemory("Construct_Band_DenseMs: is2", sizeof(int)*numprocs,NULL);
    PrintMemory("Construct_Band_DenseMs: order_GA", sizeof(int)*(atomnum+2),NULL);
  }
  firsttime = 0;

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
    BandNonCol_ConstructCache_Ensure(order_GA,MP,n);
    BandNonCol_ConstructDenseMs_OpenMP(cpx_flag,n,k1,k2,k3,M1,Ms);
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

  free(My_NZeros);
  free(My_Matomnum);
  free(is1);
  free(is2);
  free(order_GA);
}



void Construct_Band_Ms( int cpx_flag, double ****Mat, double *M1, dcomplex *Ms, 
                        int *MP, double k1, double k2, double k3)
{
  static int firsttime=1;
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

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  My_Matomnum = (int*)malloc(sizeof(int)*numprocs);
  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);
  is2 = (int*)malloc(sizeof(int)*numprocs);
  order_GA = (int*)malloc(sizeof(int)*(atomnum+2));

  if (firsttime && memoryusage_fileout) {
  PrintMemory("Construct_Band_Ms: My_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: SP_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: SP_Atoms", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: is1", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: ie1", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: is2", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: order_GA", sizeof(int)*(atomnum+2),NULL);
  }
  firsttime = 1;

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

  free(My_NZeros);
  free(My_Matomnum);
  free(is1);
  free(ie1);
  free(is2);
  free(order_GA);

  if (measure_time){
    dtime(&AEtime);
    printf("timeB_all myid=%2d %15.12f\n",myid,AEtime-AStime);
  }
}




double Calc_DM_Band_non_collinear(
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
    double *rEDM22)
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

      x = (ko[k] - ChemP)*Beta;
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

	for (k=kmin; k<=kmax; k++){
	  ReEVec_i1[i][k] = EVec1[i1+k].r;
	  ImEVec_i1[i][k] = EVec1[i1+k].i;
	  ReEVec_i2[i][k] = EVec1[i2+k].r;
	  ImEVec_i2[i][k] = EVec1[i2+k].i;
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
          for (k=kmin; k<=kmax; k++){
	    ReEVec_j1[j][k] = EVec1[j1+k].r;
	    ImEVec_j1[j][k] = EVec1[j1+k].i;
	    ReEVec_j2[j][k] = EVec1[j2+k].r;
	    ImEVec_j2[j][k] = EVec1[j2+k].i;
          } 
        }

        /* loops for i and j */

	for (i=0; i<tnoA; i++){
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

	    for (k=kmin; k<=kmax; k++){

	      ReA = ReEVec_i1[i][k]*ReEVec_j1[j][k] + ImEVec_i1[i][k]*ImEVec_j1[j][k]; 
	      ImA = ReEVec_i1[i][k]*ImEVec_j1[j][k] - ImEVec_i1[i][k]*ReEVec_j1[j][k];
	      ReB = ReEVec_i2[i][k]*ReEVec_j2[j][k] + ImEVec_i2[i][k]*ImEVec_j2[j][k];
	      ImB = ReEVec_i2[i][k]*ImEVec_j2[j][k] - ImEVec_i2[i][k]*ReEVec_j2[j][k];
	      ReC = ReEVec_i1[i][k]*ReEVec_j2[j][k] + ImEVec_i1[i][k]*ImEVec_j2[j][k];
	      ImC = ReEVec_i1[i][k]*ImEVec_j2[j][k] - ImEVec_i1[i][k]*ReEVec_j2[j][k]; 

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




double Calc_ParDM_Band_non_collinear(
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
    double *iDM22) 
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

  windowF = (double*)malloc(sizeof(double)*(n2+1));

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
        windowF[k] = weight_CWF2(e,ChemP,in0,in1,out0,out1); 
      }

      /* Fermi */
      else if (CWF_WindowF_switch==1){

        e = ko[k];
        b0 = 1.0/CWF_disentangling_smearing_kBT0;
        b1 = 1.0/CWF_disentangling_smearing_kBT1;
        e0 = CWF_disentangling_Erange[0] + ChemP; 
        e1 = CWF_disentangling_Erange[1] + ChemP; 

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

	for (k=kmin; k<=kmax; k++){
	  ReEVec_i1[i][k] = EVec1[i1+k].r;
	  ImEVec_i1[i][k] = EVec1[i1+k].i;
	  ReEVec_i2[i][k] = EVec1[i2+k].r;
	  ImEVec_i2[i][k] = EVec1[i2+k].i;
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
          for (k=kmin; k<=kmax; k++){
	    ReEVec_j1[j][k] = EVec1[j1+k].r;
	    ImEVec_j1[j][k] = EVec1[j1+k].i;
	    ReEVec_j2[j][k] = EVec1[j2+k].r;
	    ImEVec_j2[j][k] = EVec1[j2+k].i;
          } 
        }

        /* loops for i and j */

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

	    d1 = 0.0;
	    d2 = 0.0;
	    d3 = 0.0;
	    d4 = 0.0;
	    d5 = 0.0;
	    d6 = 0.0;

	    for (k=kmin; k<=kmax; k++){

	      ReA = ReEVec_i1[i][k]*ReEVec_j1[j][k] + ImEVec_i1[i][k]*ImEVec_j1[j][k]; 
	      ImA = ReEVec_i1[i][k]*ImEVec_j1[j][k] - ImEVec_i1[i][k]*ReEVec_j1[j][k];
	      ReB = ReEVec_i2[i][k]*ReEVec_j2[j][k] + ImEVec_i2[i][k]*ImEVec_j2[j][k];
	      ImB = ReEVec_i2[i][k]*ImEVec_j2[j][k] - ImEVec_i2[i][k]*ReEVec_j2[j][k];
	      ReC = ReEVec_i1[i][k]*ReEVec_j2[j][k] + ImEVec_i1[i][k]*ImEVec_j2[j][k];
	      ImC = ReEVec_i1[i][k]*ImEVec_j2[j][k] - ImEVec_i1[i][k]*ReEVec_j2[j][k]; 

	      d1 += windowF[k]*ReA;
	      d2 += windowF[k]*ImA;
	      d3 += windowF[k]*ReB;
	      d4 += windowF[k]*ImB;
	      d5 += windowF[k]*ReC;
	      d6 += windowF[k]*ImC;

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

  free(windowF);

  dtime(&etime);
  return (etime-stime);
}
