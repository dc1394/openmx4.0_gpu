#include <limits.h>
/**********************************************************************
  Krylov.c:

     Krylov.c is a subroutine to perform a Krylov subspace
     method developed by T.Ozaki.

  Log of Krylov.c

     10/June/2005  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>
#include "mpi.h"
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_solver_workspace.h"
#include "openmx_native_resources.h"
#include "lapack_prototypes.h"
#include "set_hip_default_device_from_local_rank.h"
#include <omp.h>
#include <rocblas/rocblas.h>

/* Stage timing of the Krylov solver (the time1..time16 blocks below, printed
   per rank at the end of each call): set OPENMX_KRYLOV_MEASURE_TIME=1.
   Diagnostic; replaces the old compile-time "#define measure_time 0". */

/* Persistent metadata belongs to the explicit calculation owner. */
typedef struct {
  int active;
  hipblasHandle_t hipblas;
  hipStream_t stream;
  double *d_A;
  double *d_B;
  double *d_C;
  double *h_A;
  size_t d_A_count;
  size_t d_B_count;
  size_t d_C_count;
  size_t h_A_count;
} Krylov_GPU_Workspace;

typedef struct {
  int state;
  int matomnum;
  int nspin;
  int *dim_n;
  int *dim_m;
  int *valid;
  size_t *off;
  double *d_arena;
  size_t arena_elems;
} Krylov_KUCache;

struct OpenMXKrylovState {
  OpenMXChemicalPotentials *chemical; /* borrowed from this calculation */
  OpenMXMixingHistory *history; /* borrowed from this calculation */
  OpenMXNativeResources *resources; /* borrowed from this calculation */
  int initialized;
  Krylov_KUCache kucache;
  int gpu_ws_pool_size;
  Krylov_GPU_Workspace *gpu_ws_pool;
  unsigned long long gpu_need_total;
  int gpu_device_ranks;
  int gpu_alloc_failure_reported;
  int gpu_fits;
  int gpu_gemmul8;
  int gpu_fused;
  int gpu_eigen_min;
  double gpu_dgemm_min_flops;
  int measure_time_env;
  double Col_trd_owned_TZ;
  int Col_trd_owned_firsttime;
  int Col_trd_owned_recalc_firsttime;
  int Col_trd_owned_recalc_flag;
  double Col_owned_TZ;
  int Col_owned_firsttime;
  int Col_owned_recalc_firsttime;
  int Col_owned_recalc_flag;
  int ProjectedSolve_GPU_profile;
  int GPU_Preflight_env_checked;
  size_t GPU_Preflight_reserve_bytes;
  int GPU_Preflight_hip_device;
  MPI_Comm GPU_Preflight_device_comm;
  int GPU_InitOnce_initialized;
  int GPU_Enabled_requested;
};
typedef OpenMXKrylovState KrylovState;

static int Krylov_MeasureTime(KrylovState *state)
{
    (void)state;
    return openmx_env_flag_at(OPENMX_ENV_KRYLOV_MEASURE_TIME, 0);
}
#define  measure_time   (Krylov_MeasureTime(state))
#define  error_check    0
#define  cutoff_value   Threshold_OLP_Eigen
#define  KRYLOV_ENABLE_GPU  1
/* GPU/CPU crossover for the projected eigensolve, compared against Msize3;
   overridable via OPENMX_KRYLOV_GPU_EIGEN_MIN */
#define  KRYLOV_GPU_EIGEN_MIN  300
#define  KRYLOV_GPU_DGEMM_MIN_FLOPS  50000000.0




/* The device GEMMs use the common bridge, whose architecture, input and
   environment policy selects Ozaki-II emulation or native FP64 and may
   fall back to native FP64 when workspace is unavailable. Setting
   OPENMX_KRYLOV_GEMMUL8=0 bypasses the bridge for native hipBLAS FP64. */


#include "tran_prototypes.h"

#if defined(nosse) || !defined(__SSE2__)
#include "mimic_sse.h"
#else
#include <emmintrin.h>
#endif
  
#ifdef kcomp
#define _mm_loadu_pd  _mm_load_pd
#define _mm_storeu_pd _mm_store_pd
#endif

static void Generate_pMatrix(KrylovState *state,  int myid, int spin, int Mc_AN, double *****Hks, double ****OLP0, double **invS,
                              double ***Krylov_U, double ***Krylov_U_OLP, double **inv_RS, int *MP,
                              int *Msize, int *Msize2, int *Msize3, int *Msize4, int Msize2_max, 
                              double **tmpvec0, double **tmpvec1, double **tmpvec2 ); 
                              
static void Generate_pMatrix2( int myid, int spin, int Mc_AN, double *****Hks, double ****OLP0, double ***Krylov_U, 
                               int *MP, int *Msize, int *Msize2, int *Msize3, 
                               double **tmpvec1 );
static void Krylov_IOLP( int Mc_AN, double ****OLP0, double ***Krylov_U_OLP, double **inv_RS, 
                         int *MP, int *Msize2, int *Msize4, int Msize2_max, 
                         double **tmpvec0, double **tmpvec1 );
static void S_orthonormalize_vec( int Mc_AN, int ct_on, double **vec,
                                  double **workvec, double ****OLP0,
                                  double **tmpmat0, double *ko, double *iko, int *MP, int *Msize2 );
static void Embedding_Matrix(KrylovState *state, int spin, int Mc_AN, double *****Hks,
                             double ***Krylov_U, double ****EC_matrix, 
			     int *MP, int *Msize, int *Msize2, int *Msize3, 
                             double **tmpvec1);

static void Inverse_S_by_Cholesky(int Mc_AN, double ****OLP0, double **invS, int *MP, int NUM, double *LoS);

static void Save_DOS_Col(const OpenMXChemicalPotentials *chemical, OpenMXSolverResidueView Residues, double ****OLP0, OpenMXSolverRealFieldView EVal, OpenMXSolverIntTensor LO_TC, OpenMXSolverIntTensor HO_TC);

static double Krylov_Col(KrylovState *state, char *mode,
			 int SCF_iter,
			 double *****Hks,
			 double ****OLP0,
			 double *****CDM,
			 double *****EDM,
			 double Eele0[2],
			 double Eele1[2]);


/*******************************************************
  The following subroutines are called only when 
  Matomnum==1 && openmp_threads_num>1
*******************************************************/

static double Krylov_Col_trd(KrylovState *state, char *mode,
			 int SCF_iter,
			 double *****Hks,
			 double ****OLP0,
			 double *****CDM,
			 double *****EDM,
			 double Eele0[2],
			 double Eele1[2]);

static void Generate_pMatrix_trd(KrylovState *state,  int myid, int spin, int Mc_AN, double *****Hks, double ****OLP0, double **invS,
                              double ***Krylov_U, double ***Krylov_U_OLP, double **inv_RS, int *MP,
                              int *Msize, int *Msize2, int *Msize3, int *Msize4, int Msize2_max, 
                              double **tmpvec0, double **tmpvec1, double **tmpvec2 ); 
                              
static void Generate_pMatrix2_trd( int myid, int spin, int Mc_AN, double *****Hks, double ****OLP0, double ***Krylov_U, 
                               int *MP, int *Msize, int *Msize2, int *Msize3, 
                               double **tmpvec1 );
static void Krylov_IOLP_trd( int Mc_AN, double ****OLP0, double ***Krylov_U_OLP, double **inv_RS, 
                         int *MP, int *Msize2, int *Msize4, int Msize2_max, 
                         double **tmpvec0, double **tmpvec1 );
static void S_orthonormalize_vec_trd( int Mc_AN, int ct_on, double **vec,
                                  double **workvec, double ****OLP0,
                                  double **tmpmat0, double *ko, double *iko, int *MP, int *Msize2 );
static void Embedding_Matrix_trd(int spin, int Mc_AN, double *****Hks,
                             double ***Krylov_U, double ****EC_matrix, 
			     int *MP, int *Msize, int *Msize2, int *Msize3, 
                             double **tmpvec1, int EKC_core_size_max, int Msize2_max);


static int Eigen_lapack_x(double **a, double *ko, int n, int EVmax);
static int Eigen_lapack_d(double **a, double *ko, int n, int EVmax);
static int Eigen_lapack_r(double **a, double *ko, int n, int EVmax);
extern int openmx_magma_dsyevdx_gpu(int n, int maxn, double *d_A, double *w, int *mout);
extern int openmx_gemmul8DgemmEnabled(void);

static int Krylov_GPU_Enabled(KrylovState *state)
{
    (void)state;
    return KRYLOV_ENABLE_GPU && scf_eigen_lib_flag == GPUSOLVER &&
           openmx_env_flag_at(OPENMX_ENV_KRYLOV_GPU, !openmx_band_gpu_dense_oversubscribed());
}

/*
  Verdict of the device-memory preflight of the current Krylov call
  (Krylov_GPU_Preflight): 0 routes every GPU helper below to its CPU path.
  A failed device allocation inside a call also clears it, so the rest of
  the call stays on the CPU instead of racing the other ranks for the last
  free bytes.  It has to be decided up front: hipBLASLt terminates the
  process when an allocation inside a GEMM fails, it does not return an
  error the caller could recover from.
*/


/* set by the preflight; the KU cache of the same call is sized from them */





static void Krylov_GPU_InitOnce(KrylovState *state);
static void Krylov_GPU_Workspace_Init(KrylovState *state, Krylov_GPU_Workspace *ws);
static void Krylov_GPU_Workspace_Free(KrylovState *state, Krylov_GPU_Workspace *ws);
static int  Krylov_GPU_EnsureHipblas(KrylovState *state, Krylov_GPU_Workspace *ws);
static int  Krylov_GPU_EnsureDouble(KrylovState *state, double **ptr, size_t *capacity, size_t count);
static int  Krylov_GPU_Usable(KrylovState *state, const Krylov_GPU_Workspace *ws);
static void Krylov_GPU_Preflight(KrylovState *state, int matomnum, const int *Msize, const int *Msize3);
static void Krylov_GPU_PreparePool(KrylovState *state, int nthrds);
static Krylov_GPU_Workspace *Krylov_GPU_GetWorkspace(KrylovState *state, int thread_id);
static void Krylov_Dgemm(KrylovState *state, Krylov_GPU_Workspace *ws,
                         hipblasOperation_t transa, hipblasOperation_t transb,
                         int m, int n, int k,
                         const double *A, int lda,
                         const double *B, int ldb,
                         double *C, int ldc);
static void Krylov_Eigen2(KrylovState *state, Krylov_GPU_Workspace *ws, double *a, int csize, double *ko, int n, int EVmax);


static void Krylov_GPU_InitOnce(KrylovState *state)
{
    if (state->GPU_InitOnce_initialized) return;
    state->GPU_InitOnce_initialized = 1;
    double flops = openmx_env_double_at(OPENMX_ENV_KRYLOV_GPU_DGEMM_MIN_FLOPS, state->gpu_dgemm_min_flops);
    long eigen = openmx_env_long_at(OPENMX_ENV_KRYLOV_GPU_EIGEN_MIN, state->gpu_eigen_min);
    if (flops >= 0) state->gpu_dgemm_min_flops = flops;
    if (eigen > 0) state->gpu_eigen_min = eigen > INT_MAX ? INT_MAX : (int)eigen;
    state->gpu_fused = openmx_env_flag_at(OPENMX_ENV_KRYLOV_GPU_FUSED, state->gpu_fused);
    state->gpu_gemmul8 = openmx_env_flag_at(OPENMX_ENV_KRYLOV_GEMMUL8, state->gpu_gemmul8);
    if (Krylov_GPU_Enabled(state)){
      wait_hipfunc(OpenMX_Resource_GpuFree(state->resources, 0));
      /* Initialize the device's Tensile kernel library before OpenMP
         workers enter their first GEMM. Concurrent lazy initialization
         can leave hipErrorNoBinaryForGpu even when GEMM and sync succeed. */
      rocblas_initialize();
    }
}




static void Krylov_GPU_PreparePool(KrylovState *state, int nthrds)
{
  int i;

  if (!Krylov_GPU_Enabled(state) || nthrds <= state->gpu_ws_pool_size) return;

  Krylov_GPU_Workspace *new_pool =
    (Krylov_GPU_Workspace*)openmx_core_native_realloc(state->resources,state->gpu_ws_pool,sizeof(Krylov_GPU_Workspace)*(size_t)nthrds);
  if (new_pool == NULL){
    fprintf(stderr,"Krylov: could not allocate GPU workspace pool.\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  state->gpu_ws_pool = new_pool;
  for (i=state->gpu_ws_pool_size; i<nthrds; i++){
    memset(&state->gpu_ws_pool[i],0,sizeof(Krylov_GPU_Workspace));
  }
  state->gpu_ws_pool_size = nthrds;
}

static Krylov_GPU_Workspace *Krylov_GPU_GetWorkspace(KrylovState *state, int thread_id)
{
  Krylov_GPU_Workspace *ws;

  if (!Krylov_GPU_Enabled(state) ||
      thread_id < 0 || state->gpu_ws_pool_size <= thread_id) return NULL;

  /* HIP's current device belongs to the host thread. OpenMP workers can
     start on device 0 or retain a device from an earlier region. Bind every
     entry before creating a stream or using the rank's resident KU cache. */
  if (hipSetDevice(state->GPU_Preflight_hip_device) != hipSuccess){
    (void)hipGetLastError();
    return NULL;
  }

  ws = &state->gpu_ws_pool[thread_id];
  if (!ws->active && ws->stream == NULL){
    Krylov_GPU_Workspace_Init(state, ws);
  }

  return ws;
}

static void Krylov_GPU_ReportAllocFailure(KrylovState *state, const char *what, size_t bytes, const char *reason)
{
  if (state->gpu_alloc_failure_reported) return;
  state->gpu_alloc_failure_reported = 1;
  if (bytes != 0)
    fprintf(stderr,
            "Krylov: %s of %.3f MiB on the GPU failed (%s); "
            "the remaining local solves of this call run on the CPU.\n",
            what, (double)bytes/(1024.0*1024.0), reason);
  else
    fprintf(stderr,
            "Krylov: %s on the GPU failed (%s); "
            "the remaining local solves of this call run on the CPU.\n",
            what, reason);
  fflush(stderr);
}

/* Grows a device buffer; returns 0 (and demotes the rest of the call to
   the CPU) when the device is out of memory instead of aborting. */
static int Krylov_GPU_EnsureDouble(KrylovState *state, double **ptr, size_t *capacity, size_t count)
{
  if (*capacity < count){
    hipError_t err;

    if (*ptr != NULL) wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,*ptr));
    *ptr = NULL;
    *capacity = 0;
    err = OpenMX_Resource_GpuMalloc(state->resources,(void**)ptr,sizeof(double)*count);
    if (err != hipSuccess){
      (void)hipGetLastError();
      *ptr = NULL;
      state->gpu_fits = 0;
      Krylov_GPU_ReportAllocFailure(state, "hipMalloc", sizeof(double)*count, hipGetErrorString(err));
      return 0;
    }
    *capacity = count;
  }
  return 1;
}

static int Krylov_GPU_Usable(KrylovState *state, const Krylov_GPU_Workspace *ws)
{
  return ws != NULL && ws->active && state->gpu_fits;
}

static void Krylov_GPU_EnsureHostMatrix(KrylovState *state, Krylov_GPU_Workspace *ws, size_t count)
{
  if (ws->h_A_count < count){
    double *p = (double*)openmx_core_native_realloc(state->resources,ws->h_A,sizeof(double)*count);
    if (p == NULL){
      fprintf(stderr,"Krylov: could not allocate host staging matrix.\n");
      MPI_Abort(MPI_COMM_WORLD,1);
    }
    ws->h_A = p;
    ws->h_A_count = count;
  }
}

static void Krylov_GPU_Workspace_Init(KrylovState *state, Krylov_GPU_Workspace *ws)
{
  memset(ws,0,sizeof(Krylov_GPU_Workspace));

  if (!Krylov_GPU_Enabled(state)) return;

  ws->active = 1;
  wait_hipfunc(OpenMX_Resource_StreamCreate(state->resources,&ws->stream,hipStreamNonBlocking));
}

static void Krylov_GPU_Workspace_Free(KrylovState *state, Krylov_GPU_Workspace *ws)
{
  if (ws->stream != NULL) wait_hipfunc(hipStreamSynchronize(ws->stream));
  if (ws->d_A    != NULL) wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ws->d_A));
  if (ws->d_B    != NULL) wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ws->d_B));
  if (ws->d_C    != NULL) wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,ws->d_C));
  if (ws->h_A    != NULL) openmx_core_native_free(state->resources,ws->h_A);
  if (ws->hipblas   != NULL) OpenMXBlasWait(OpenMX_Resource_BlasDestroy(state->resources,ws->hipblas));
  if (ws->stream   != NULL) wait_hipfunc(OpenMX_Resource_StreamDestroy(state->resources,ws->stream));
  memset(ws,0,sizeof(Krylov_GPU_Workspace));
}

static int Krylov_GPU_EnsureHipblas(KrylovState *state, Krylov_GPU_Workspace *ws)
{
  if (ws->hipblas == NULL){
    hipblasHandle_t handle = NULL;
    hipblasStatus_t status = OpenMX_Resource_BlasCreate(state->resources,&handle);

    /* the handle owns the BLAS workspace pool, so this is where a full
       device shows up first */
    if (status != HIPBLAS_STATUS_SUCCESS){
      (void)hipGetLastError();
      state->gpu_fits = 0;
      Krylov_GPU_ReportAllocFailure(state, "hipblasCreate", 0, "hipBLAS handle creation failed");
      return 0;
    }
    OpenMXBlasWait(hipblasSetStream(handle, ws->stream));
    ws->hipblas = handle;
  }
  return 1;
}

/* C = op(A) op(B) on the device, all operands resident, on the stream of
   ws: openmx_gemmul8Dgemm unless OPENMX_KRYLOV_GEMMUL8=0 */
static hipblasStatus_t Krylov_GPU_DeviceDgemm(KrylovState *state, Krylov_GPU_Workspace *ws,
                                              hipblasOperation_t transa, hipblasOperation_t transb,
                                              int m, int n, int k,
                                              const double *dA, int lda,
                                              const double *dB, int ldb,
                                              double *dC, int ldc)
{
  const double alpha = 1.0;
  const double beta  = 0.0;

  if (state->gpu_gemmul8)
    return openmx_gemmul8Dgemm(ws->hipblas, transa, transb, m, n, k,
                               &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
  return hipblasDgemm(ws->hipblas, transa, transb, m, n, k,
                      &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
}

/* A device GEMM that returned an error: the host operands are intact, so
   the caller redoes the product on the CPU, and the rest of the call stays
   there (like a failed allocation).  The stream is drained first, so that
   no copy of this call is still in flight when the host buffers are
   reused. */
static void Krylov_GPU_GemmFailed(KrylovState *state, Krylov_GPU_Workspace *ws, hipblasStatus_t status,
                                  int m, int n, int k)
{
  char reason[96];

  (void)hipStreamSynchronize(ws->stream);
  (void)hipGetLastError();
  state->gpu_fits = 0;
  snprintf(reason, sizeof(reason), "%s status %d, m=%d n=%d k=%d",
           state->gpu_gemmul8 ? "GPU GEMM bridge" : "native hipBLAS FP64", (int)status, m, n, k);
  Krylov_GPU_ReportAllocFailure(state, "DGEMM", 0, reason);
}

static void Krylov_Dgemm(KrylovState *state, Krylov_GPU_Workspace *ws,
                         hipblasOperation_t transa, hipblasOperation_t transb,
                         int m, int n, int k,
                         const double *A, int lda,
                         const double *B, int ldb,
                         double *C, int ldc)
{
  const double alpha = 1.0;
  const double beta  = 0.0;
  size_t cols_A = (transa == HIPBLAS_OP_N) ? (size_t)k : (size_t)m;
  size_t cols_B = (transb == HIPBLAS_OP_N) ? (size_t)n : (size_t)k;
  size_t count_A = (size_t)lda*cols_A;
  size_t count_B = (size_t)ldb*cols_B;
  size_t count_C = (size_t)ldc*(size_t)n;
  hipblasStatus_t status;
  int on_gpu = Krylov_GPU_Usable(state, ws) &&
               !((double)m*(double)n*(double)k < state->gpu_dgemm_min_flops);

  if (on_gpu){
    on_gpu = Krylov_GPU_EnsureHipblas(state, ws) &&
             Krylov_GPU_EnsureDouble(state, &ws->d_A, &ws->d_A_count, count_A) &&
             Krylov_GPU_EnsureDouble(state, &ws->d_B, &ws->d_B_count, count_B) &&
             Krylov_GPU_EnsureDouble(state, &ws->d_C, &ws->d_C_count, count_C);
  }

  if (on_gpu){
    wait_hipfunc(hipMemcpyAsync(ws->d_A, A, sizeof(double)*count_A,
                                  hipMemcpyHostToDevice, ws->stream));
    wait_hipfunc(hipMemcpyAsync(ws->d_B, B, sizeof(double)*count_B,
                                  hipMemcpyHostToDevice, ws->stream));

    status = Krylov_GPU_DeviceDgemm(state, ws, transa, transb, m, n, k,
                                    ws->d_A, lda, ws->d_B, ldb, ws->d_C, ldc);
    if (status == HIPBLAS_STATUS_SUCCESS){
      wait_hipfunc(hipMemcpyAsync(C, ws->d_C, sizeof(double)*count_C,
                                    hipMemcpyDeviceToHost, ws->stream));
      wait_hipfunc(hipStreamSynchronize(ws->stream));
      return;
    }
    Krylov_GPU_GemmFailed(state, ws, status, m, n, k);
  }

  {
    char ta = (transa == HIPBLAS_OP_N) ? 'N' : 'T';
    char tb = (transb == HIPBLAS_OP_N) ? 'N' : 'T';
    F77_NAME(dgemm,DGEMM)(&ta, &tb, &m, &n, &k, (double*)&alpha,
                          (double*)A, &lda, (double*)B, &ldb, (double*)&beta, C, &ldc);
  }
}

static void Krylov_Eigen2(KrylovState *state, Krylov_GPU_Workspace *ws, double *a, int csize, double *ko, int n, int EVmax)
{
  int i,j;
  int info, mout;

  if (!Krylov_GPU_Usable(state, ws) || n < state->gpu_eigen_min){
    Eigen_lapack2(a,csize,ko,n,EVmax);
    return;
  }

  Krylov_GPU_EnsureHostMatrix(state, ws,(size_t)n*(size_t)n);
  if (!Krylov_GPU_EnsureDouble(state, &ws->d_A, &ws->d_A_count, (size_t)n*(size_t)n)){
    Eigen_lapack2(a,csize,ko,n,EVmax);
    return;
  }

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      ws->h_A[i*n+j] = a[(i+1)*csize+j+1];
    }
  }

  wait_hipfunc(hipMemcpyAsync(ws->d_A, ws->h_A, sizeof(double)*(size_t)n*(size_t)n,
                                hipMemcpyHostToDevice, ws->stream));
  wait_hipfunc(hipStreamSynchronize(ws->stream));

  mout = 0;
  info = openmx_magma_dsyevdx_gpu(n, EVmax, ws->d_A, ko+1, &mout);
  if (info != 0 || mout != EVmax){
    if (info != 0){
      if (2 <= level_stdout) {
          fprintf(stderr,"Krylov: dense GPU eigensolver failed, info=%d; falling back to LAPACK.\n",info);
      }
    }
    else{
      if (2 <= level_stdout) {
          fprintf(stderr,"Krylov: dense GPU eigensolver returned %d eigenpairs, expected %d; falling back to LAPACK.\n",
                  mout,EVmax);
      }
    }
    Eigen_lapack2(a,csize,ko,n,EVmax);
    return;
  }

  wait_hipfunc(hipMemcpyAsync(ws->h_A, ws->d_A, sizeof(double)*(size_t)n*(size_t)n,
                                hipMemcpyDeviceToHost, ws->stream));
  wait_hipfunc(hipStreamSynchronize(ws->stream));

  for (i=0; i<EVmax; i++){
    for (j=0; j<n; j++){
      a[i*n+j] = ws->h_A[i*n+j];
    }
  }
}





void Krylov_Release_GPU_KUCache(KrylovState *state)
{
  if (state->kucache.d_arena != NULL) wait_hipfunc(OpenMX_Resource_GpuFree(state->resources,state->kucache.d_arena));
  openmx_core_native_free(state->resources,state->kucache.dim_n);
  openmx_core_native_free(state->resources,state->kucache.dim_m);
  openmx_core_native_free(state->resources,state->kucache.valid);
  openmx_core_native_free(state->resources,state->kucache.off);
  memset(&state->kucache,0,sizeof(state->kucache));
}

/* Run-boundary reset (between -runtest/-runtestL inputs): also return the
   per-thread GPU workspaces, whose sizes derive from the previous system.
   Idempotent; called from serial code, never inside the OpenMP regions
   that use the pool. */
static void Krylov_Release_GPU_Caches(KrylovState *state)
{
  int i;

  for (i=0; i<state->gpu_ws_pool_size; i++){
    Krylov_GPU_Workspace_Free(state, &state->gpu_ws_pool[i]);
  }
  openmx_core_native_free(state->resources,state->gpu_ws_pool);
  state->gpu_ws_pool = NULL;
  state->gpu_ws_pool_size = 0;

  Krylov_Release_GPU_KUCache(state);

  /* Preflight creates this group collectively; calculation teardown is
     collective on the same ranks and runs before MPI finalization. */
  if (state->GPU_Preflight_device_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&state->GPU_Preflight_device_comm);
  }
}

/* keep_free: device memory the preflight of this call has already claimed
   for the solves themselves, summed over the device_ranks ranks sharing
   the GPU.  The cache is sized from what is left beyond that and the
   reserve; otherwise it would starve the very solves it accelerates. */
static void Krylov_KUCache_Prepare(KrylovState *state, int matomnum, int nspin, const int *Msize,
                                   const int *Msize3, int scf_iter,
                                   int device_ranks, unsigned long long keep_free)
{


  size_t nslot = (size_t)nspin*(size_t)(matomnum+1);
  size_t free_bytes=0,total_bytes=0,reserve=(size_t)4096*1024*1024;
  size_t budget,used=0;
  int admitted=0;

  if (!openmx_env_flag_at(OPENMX_ENV_KRYLOV_GPU_KU_CACHE, 1)) return;
  reserve = openmx_env_mib_bytes_at(OPENMX_ENV_KRYLOV_GPU_KU_RESERVE_MB, reserve / (1024ULL * 1024ULL));
  if (state->kucache.state==1 && state->kucache.matomnum==matomnum &&
      state->kucache.nspin==nspin){
    int same=1;
    for (int spin=0;spin<nspin;spin++) for (int a=1;a<=matomnum;a++){
      size_t s=(size_t)spin*(size_t)(matomnum+1)+(size_t)a;
      if (state->kucache.dim_n[s]!=0 &&
          (state->kucache.dim_n[s]!=Msize[a] || state->kucache.dim_m[s]!=Msize3[a])) same=0;
    }
    if (same){
      if (scf_iter<=1) memset(state->kucache.valid,0,sizeof(int)*nslot);
      return;
    }
    Krylov_Release_GPU_KUCache(state);
  }
  if (state->kucache.state==-1) return;

  state->kucache.dim_n=(int*)openmx_core_native_malloc(state->resources,nslot,sizeof(int),1);
  state->kucache.dim_m=(int*)openmx_core_native_malloc(state->resources,nslot,sizeof(int),1);
  state->kucache.valid=(int*)openmx_core_native_malloc(state->resources,nslot,sizeof(int),1);
  state->kucache.off=(size_t*)openmx_core_native_malloc(state->resources,nslot,sizeof(size_t),1);
  if (state->kucache.dim_n==NULL || state->kucache.dim_m==NULL ||
      state->kucache.valid==NULL || state->kucache.off==NULL){
    fprintf(stderr,"Krylov: could not allocate KU-cache tables.\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  if (device_ranks<1) device_ranks=1;
  if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=hipSuccess){
    (void)hipGetLastError();
    Krylov_Release_GPU_KUCache(state);
    state->kucache.state=-1;
    return;
  }
  reserve += (size_t)keep_free;
  budget=free_bytes>reserve ? (free_bytes-reserve)/(size_t)device_ranks : 0;
  for (int spin=0;spin<nspin;spin++) for (int a=1;a<=matomnum;a++){
    size_t slot=(size_t)spin*(size_t)(matomnum+1)+(size_t)a;
    size_t elems=(size_t)Msize[a]*(size_t)Msize3[a];
    size_t bytes=elems*sizeof(double);
    if (bytes>budget || used*sizeof(double)>budget-bytes) continue;
    state->kucache.dim_n[slot]=Msize[a];
    state->kucache.dim_m[slot]=Msize3[a];
    state->kucache.off[slot]=used;
    used+=elems;
    admitted++;
  }
  if (used!=0 && OpenMX_Resource_GpuMalloc(state->resources,(void**)&state->kucache.d_arena,used*sizeof(double))!=hipSuccess){
    (void)hipGetLastError();
    Krylov_Release_GPU_KUCache(state);
    state->kucache.state=-1;
    return;
  }
  state->kucache.state=1;
  state->kucache.matomnum=matomnum;
  state->kucache.nspin=nspin;
  state->kucache.arena_elems=used;

  if (2<=level_stdout && 0<admitted){
    int myid=-1;
    MPI_Comm_rank(mpi_comm_level1,&myid);
    if (myid==Host_ID){
      printf("<Krylov> GPU KU-cache: rank %d keeps %d of %d basis panels on the device (%.1f MiB)\n",
             myid,admitted,(int)(nslot-(size_t)nspin),(double)(used*sizeof(double))/(1024.0*1024.0));
      fflush(stdout);
    }
  }
}

/*
  Device-memory preflight of one Krylov call.  The projected solves of the
  ranks sharing a GPU run concurrently, so the estimate is summed over
  those ranks and compared with the free memory of the device: the growth
  of the per-thread device buffers (H_DC, KU and the product, the largest
  local atom), the MAGMA workspace of the projected eigenproblem, the
  per-thread GEMMul8 workspace of the products, and a per-rank reserve for
  the BLAS library workspaces (OPENMX_KRYLOV_GPU_RESERVE_MB, default
  512 MiB).  It runs at every call
  because the free memory shrinks over the SCF loop (mixing history, ...),
  before the KU cache of the call is sized so that the cache only takes
  what the solves leave over, and its verdict is collective over the ranks
  of the device so that they take the same path.  The default reserve is
  deliberately generous: the failure it guards against is not a returned
  error but the process exiting inside hipBLASLt.
*/
static void Krylov_GPU_Preflight(KrylovState *state, int matomnum, const int *Msize, const int *Msize3)
{




  const double MiB = 1024.0*1024.0;
  size_t free_bytes = 0, total_bytes = 0;
  size_t nn_max = 0, nm_max = 0, mm_max = 0, m3_max = 0, n_max = 0;
  size_t growth = 0, magma_bytes, lib_bytes, gemmul8_bytes = 0, need_rank;
  unsigned long long need_local, need_total, free_local, free_min;
  int myid, Mc_AN, t;

  state->gpu_fits = 1;
  state->gpu_need_total = 0;
  if (!Krylov_GPU_Enabled(state)) return;

  state->GPU_Preflight_reserve_bytes = openmx_env_mib_bytes_at(OPENMX_ENV_KRYLOV_GPU_RESERVE_MB,
      state->GPU_Preflight_reserve_bytes / (1024ULL * 1024ULL));

  if (state->GPU_Preflight_device_comm == MPI_COMM_NULL){
    MPI_Comm node_comm;
    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    if (hipGetDevice(&state->GPU_Preflight_hip_device) != hipSuccess) state->GPU_Preflight_hip_device = 0;
    MPI_Comm_split(node_comm, state->GPU_Preflight_hip_device, 0, &state->GPU_Preflight_device_comm);
    MPI_Comm_free(&node_comm);
    MPI_Comm_size(state->GPU_Preflight_device_comm, &state->gpu_device_ranks);
  }

  for (Mc_AN=1; Mc_AN<=matomnum; Mc_AN++){
    size_t n = (size_t)Msize[Mc_AN], m3 = (size_t)Msize3[Mc_AN];
    if (nn_max < n*n)   nn_max = n*n;
    if (nm_max < n*m3)  nm_max = n*m3;
    if (mm_max < m3*m3) mm_max = m3*m3;
    if (m3_max < m3)    m3_max = m3;
    if (n_max < n)      n_max = n;
  }

  /* buffers already grown in an earlier call are not needed again */
  for (t=0; t<state->gpu_ws_pool_size; t++){
    const Krylov_GPU_Workspace *ws = &state->gpu_ws_pool[t];
    size_t a = nn_max > mm_max ? nn_max : mm_max;
    if (ws->d_A_count < a)      growth += a - ws->d_A_count;
    if (ws->d_B_count < nm_max) growth += nm_max - ws->d_B_count;
    if (ws->d_C_count < nm_max) growth += nm_max - ws->d_C_count;
  }
  growth *= sizeof(double);

  /* magma_dsyevdx_gpu: lddc*n plus the tridiagonalization panels */
  magma_bytes = sizeof(double)*(2*m3_max*m3_max + 256*m3_max);
  /* Measured on a 16 GB gfx1200 with n=492, m=320 (operands of ~2 MB):
     the solves crashed inside hipBLASLt with 171 MiB per rank available
     and held with 313 MiB, so the library side costs a fixed 200-300 MiB
     per rank regardless of the operand size -- that is what the reserve
     covers.  Four copies of the largest operand are added on top so the
     margin also grows with the cluster size. */
  lib_bytes = 4*sizeof(double)*nn_max;

  /* The GEMMul8 workspace of the device GEMMs (C=H*u1, u1^T*C, the back
     transformation): the bridge keeps one per stream, i.e. per thread of
     the pool, grown to the largest product the thread has run, and it is
     allocated inside the GEMM call, where hipBLASLt would terminate the
     process if the memory ran out.  0 when the products are native
     (bridge architecture policy, OPENMX_KRYLOV_GEMMUL8=0,
     scf.gemmul8.enable off, GEMMUL8_DISABLE*). */
  if (state->gpu_gemmul8){
    size_t per_thread = 0;

    for (Mc_AN=1; Mc_AN<=matomnum; Mc_AN++){
      const int n = Msize[Mc_AN], m3 = Msize3[Mc_AN];
      const int shape[3][3] = {{n,m3,n}, {m3,m3,n}, {m3,n,m3}};
      int s;

      for (s=0; s<3; s++){
        size_t bytes;
        if ((double)shape[s][0]*(double)shape[s][1]*(double)shape[s][2] < state->gpu_dgemm_min_flops)
          continue;
        bytes = openmx_gemmul8DWorkspaceSize(shape[s][0], shape[s][1], shape[s][2]);
        if (per_thread < bytes) per_thread = bytes;
      }
    }
    gemmul8_bytes = per_thread*(size_t)state->gpu_ws_pool_size;
  }
  need_rank = growth + magma_bytes + lib_bytes + gemmul8_bytes + state->GPU_Preflight_reserve_bytes;

  /* the workspaces still held from an earlier call (of any GEMMul8 user of
     this rank) are returned first and counted in full above; the barrier
     makes every rank of the device measure after all of them did so */
  if (gemmul8_bytes != 0) openmx_gemmul8ReleaseWorkspaces();
  MPI_Barrier(state->GPU_Preflight_device_comm);

  if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != hipSuccess){
    (void)hipGetLastError();
    free_bytes = 0;
  }

  need_local = (unsigned long long)need_rank;
  free_local = (unsigned long long)free_bytes;
  MPI_Allreduce(&need_local, &need_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, state->GPU_Preflight_device_comm);
  MPI_Allreduce(&free_local, &free_min,   1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, state->GPU_Preflight_device_comm);

  state->gpu_fits = (need_total <= free_min);
  state->gpu_need_total = need_total;

  if (!state->gpu_fits){
    /* a cache kept from an earlier call would only pin memory the other
       stages need now; a later call that fits rebuilds it from scratch */
    Krylov_Release_GPU_KUCache(state);
    MPI_Comm_rank(mpi_comm_level1, &myid);
    if (myid == Host_ID && 2 <= level_stdout){
      fprintf(stderr,
              "Krylov: GPU device %d is shared by %d rank(s), has %.3f MiB free "
              "(%.3f MiB total on rank %d), but %.3f MiB is required for the GPU projected "
              "solves (largest local cluster n=%d, m=%d; %.3f MiB reserve and %.3f MiB "
              "GEMMul8 workspace on rank %d); switching to CPU path.\n",
              state->GPU_Preflight_hip_device, state->gpu_device_ranks, (double)free_min/MiB, (double)total_bytes/MiB,
              myid, (double)need_total/MiB, (int)n_max, (int)m3_max, (double)state->GPU_Preflight_reserve_bytes/MiB,
              (double)gemmul8_bytes/MiB, myid);
      fflush(stderr);
    }
  }
}

static const double *Krylov_KUCache_Get(KrylovState *state, Krylov_GPU_Workspace *ws,int spin,int Mc_AN,
                                        int n,int m3,int msize2,double ***Krylov_U,double *KU)
{
  size_t slot;
  double *dst;
  if (state->kucache.state!=1 || Mc_AN<1 || state->kucache.matomnum<Mc_AN ||
      spin<0 || state->kucache.nspin<=spin) return NULL;
  slot=(size_t)spin*(size_t)(state->kucache.matomnum+1)+(size_t)Mc_AN;
  if (state->kucache.dim_n[slot]!=n || state->kucache.dim_m[slot]!=m3) return NULL;
  dst=state->kucache.d_arena+state->kucache.off[slot];
  if (!state->kucache.valid[slot]){
    for (int i=0;i<n;i++) for (int j=0;j<m3;j++)
      KU[(size_t)j*(size_t)n+(size_t)i]=Krylov_U[spin][Mc_AN][j*msize2+i+1];
    wait_hipfunc(hipMemcpyAsync(dst,KU,sizeof(double)*(size_t)n*(size_t)m3,
                                hipMemcpyHostToDevice,ws->stream));
    wait_hipfunc(hipStreamSynchronize(ws->stream));
    state->kucache.valid[slot]=1;
  }
  return dst;
}

static int Krylov_ProjectedSolve_GPU(KrylovState *state, Krylov_GPU_Workspace *ws,int spin,int Mc_AN,
                                     const int *Msize,const int *Msize2,const int *Msize3,
                                     double ***Krylov_U,double ****EC_matrix,
                                     double *H_DC,double *KU,double *C,double *ko)
{
  const double *d_KU;
  int n=Msize[Mc_AN],m3=Msize3[Mc_AN],info,mout=0,eig_on_device=0;
  size_t nn=(size_t)n*(size_t)n;
  size_t nm=(size_t)n*(size_t)m3;
  size_t mm=(size_t)m3*(size_t)m3;
  hipblasStatus_t status;
  /* OPENMX_KRYLOV_GPU_PROFILE: the stage times of the fused path (the
     KRYFUSE line, plus pack, the compact panel the GPU eigensolver reads);
     with the profile on every stage ends with a stream
     synchronization so that the GPU time lands in its own stage */

  double t0=0.0,t1=0.0,t2=0.0,t3=0.0,t4=0.0,t5=0.0,t6=0.0,t7=0.0,t8=0.0;

  if (!Krylov_GPU_Usable(state, ws) || !state->gpu_fused) return 0;
  if ((double)n*(double)m3*(double)n<state->gpu_dgemm_min_flops ||
      (double)m3*(double)m3*(double)n<state->gpu_dgemm_min_flops ||
      (double)m3*(double)n*(double)m3<state->gpu_dgemm_min_flops) return 0;

  if (state->ProjectedSolve_GPU_profile<0){

    state->ProjectedSolve_GPU_profile=(openmx_env_flag_at(OPENMX_ENV_KRYLOV_GPU_PROFILE, 0));
  }
  if (state->ProjectedSolve_GPU_profile) dtime(&t0);

  /* a failed allocation or product hands the whole atom to the CPU path
     (return 0); nothing computed so far is reused there */
  if (!Krylov_GPU_EnsureHipblas(state, ws) ||
      !Krylov_GPU_EnsureDouble(state, &ws->d_A,&ws->d_A_count,nn>mm?nn:mm) ||
      !Krylov_GPU_EnsureDouble(state, &ws->d_C,&ws->d_C_count,nm)) return 0;
  d_KU=Krylov_KUCache_Get(state, ws,spin,Mc_AN,n,m3,Msize2[Mc_AN],Krylov_U,KU);
  if (d_KU==NULL){
    for (int i=0;i<n;i++) for (int j=0;j<m3;j++)
      KU[(size_t)j*(size_t)n+(size_t)i]=Krylov_U[spin][Mc_AN][j*Msize2[Mc_AN]+i+1];
    if (!Krylov_GPU_EnsureDouble(state, &ws->d_B,&ws->d_B_count,nm)) return 0;
    wait_hipfunc(hipMemcpyAsync(ws->d_B,KU,sizeof(double)*nm,hipMemcpyHostToDevice,ws->stream));
    d_KU=ws->d_B;
  }
  if (state->ProjectedSolve_GPU_profile){ wait_hipfunc(hipStreamSynchronize(ws->stream)); dtime(&t1); }
  wait_hipfunc(hipMemcpyAsync(ws->d_A,H_DC,sizeof(double)*nn,hipMemcpyHostToDevice,ws->stream));
  if (state->ProjectedSolve_GPU_profile){ wait_hipfunc(hipStreamSynchronize(ws->stream)); dtime(&t2); }
  /* C = H_DC * u1 */
  status=Krylov_GPU_DeviceDgemm(state, ws,HIPBLAS_OP_N,HIPBLAS_OP_N,n,m3,n,ws->d_A,n,d_KU,n,ws->d_C,n);
  if (status!=HIPBLAS_STATUS_SUCCESS){ Krylov_GPU_GemmFailed(state, ws,status,n,m3,n); return 0; }
  if (state->ProjectedSolve_GPU_profile){ wait_hipfunc(hipStreamSynchronize(ws->stream)); dtime(&t3); }
  /* u1^+ * H_DC * u1 (the projected Hamiltonian replaces H_DC on the host) */
  status=Krylov_GPU_DeviceDgemm(state, ws,HIPBLAS_OP_T,HIPBLAS_OP_N,m3,m3,n,d_KU,n,ws->d_C,n,ws->d_A,m3);
  if (status!=HIPBLAS_STATUS_SUCCESS){ Krylov_GPU_GemmFailed(state, ws,status,m3,m3,n); return 0; }
  if (state->ProjectedSolve_GPU_profile){ wait_hipfunc(hipStreamSynchronize(ws->stream)); dtime(&t4); }
  wait_hipfunc(hipMemcpyAsync(H_DC,ws->d_A,sizeof(double)*mm,hipMemcpyDeviceToHost,ws->stream));
  wait_hipfunc(hipStreamSynchronize(ws->stream));

  for (int i=0;i<(int)Krylov_U[spin][Mc_AN][0];i++) H_DC[i*m3+i]=1.0e3;
  for (int i=m3-1;0<=i;i--) for (int j=0;j<m3;j++)
    H_DC[(i+1)*(m3+1)+(j+1)]=H_DC[i*m3+j]+EC_matrix[spin][Mc_AN][i+1][j+1];
  if (state->ProjectedSolve_GPU_profile) dtime(&t5);

  /* diagonalize; the eigenvector panel ends up in ws->d_A either way.
     Below Krylov_gpu_eigen_min LAPACK solves it, as in Krylov_Eigen2 (the
     dispatch of the NVIDIA version). */
  if (state->gpu_eigen_min<=m3){
    /* The projected order can exceed the local basis after block rounding.
       KU holds only (Msize2+2)*m3 entries; use the square staging buffer
       for the compact column-major panel consumed by the eigensolver. */
    Krylov_GPU_EnsureHostMatrix(state, ws, mm);
    for (int i=0;i<m3;i++) for (int j=0;j<m3;j++) ws->h_A[(size_t)i*(size_t)m3+(size_t)j]=H_DC[(i+1)*(m3+1)+(j+1)];
    wait_hipfunc(hipMemcpyAsync(ws->d_A,ws->h_A,sizeof(double)*mm,hipMemcpyHostToDevice,ws->stream));
    wait_hipfunc(hipStreamSynchronize(ws->stream));
    if (state->ProjectedSolve_GPU_profile) dtime(&t6);
    info=openmx_magma_dsyevdx_gpu(m3,m3,ws->d_A,ko+1,&mout);
    eig_on_device=(info==0 && mout==m3);
    if (2<=level_stdout) {
      if (info!=0)
        fprintf(stderr,"Krylov: dense GPU eigensolver failed, info=%d; falling back to LAPACK.\n",info);
      else if (mout!=m3)
        fprintf(stderr,"Krylov: dense GPU eigensolver returned %d eigenpairs, expected %d; falling back to LAPACK.\n",
                mout,m3);
    }
  }
  else t6=t5;
  if (!eig_on_device){
    Eigen_lapack2(H_DC,m3+1,ko,m3,m3);
    wait_hipfunc(hipMemcpyAsync(ws->d_A,H_DC,sizeof(double)*mm,hipMemcpyHostToDevice,ws->stream));
  }
  if (state->ProjectedSolve_GPU_profile){ wait_hipfunc(hipStreamSynchronize(ws->stream)); dtime(&t7); }
  /* back transformation of eigenvectors: c = u1 * b */
  status=Krylov_GPU_DeviceDgemm(state, ws,HIPBLAS_OP_T,HIPBLAS_OP_T,m3,n,m3,ws->d_A,m3,d_KU,n,ws->d_C,m3);
  if (status!=HIPBLAS_STATUS_SUCCESS){
    /* H_DC no longer holds the input: this atom is finished on the CPU
       from the eigenvector panel (on the device after MAGMA) and the
       basis, as the per-call path would do it */
    const double alpha=1.0,beta=0.0;
    int lda=m3,ldb=n,ldc=m3;

    Krylov_GPU_GemmFailed(state, ws,status,m3,n,m3);
    if (eig_on_device){
      wait_hipfunc(hipMemcpyAsync(H_DC,ws->d_A,sizeof(double)*mm,hipMemcpyDeviceToHost,ws->stream));
      wait_hipfunc(hipStreamSynchronize(ws->stream));
    }
    for (int i=0;i<n;i++) for (int j=0;j<m3;j++)
      KU[(size_t)j*(size_t)n+(size_t)i]=Krylov_U[spin][Mc_AN][j*Msize2[Mc_AN]+i+1];
    F77_NAME(dgemm,DGEMM)("T","T",&m3,&n,&m3,(double*)&alpha,H_DC,&lda,KU,&ldb,(double*)&beta,C,&ldc);
    return 1;
  }
  wait_hipfunc(hipMemcpyAsync(C,ws->d_C,sizeof(double)*nm,hipMemcpyDeviceToHost,ws->stream));
  wait_hipfunc(hipStreamSynchronize(ws->stream));
  if (state->ProjectedSolve_GPU_profile){
    dtime(&t8);
    if (Mc_AN==1 && spin==0){
      /* This describes the GEMM policy, not a workspace fallback of an
         individual product. The common solver reports its actual backend
         and retries through GPUEIGENPROF. Query only after GPU work ran. */
      printf("KRYFUSE n=%d m3=%d ku=%.4f hup=%.4f g1=%.4f g2=%.4f h33=%.4f pack=%.4f eig=%.4f g3=%.4f tot=%.4f"
             " gemm_policy=%s eigensolver=%s\n",
             n,m3,t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6,t8-t7,t8-t0,
             (state->gpu_gemmul8 && openmx_gemmul8DgemmEnabled()) ? "gemmul8" : "native-fp64",
             eig_on_device ? "gpu-dispatch" : "lapack");
      fflush(stdout);
    }
  }
  return 1;
}




double Krylov(KrylovState *state, char *mode,
              int SCF_iter,
              double *****Hks,
              double *****ImNL,
              double ****OLP0,
              double *****CDM,
              double *****EDM,
              double Eele0[2], double Eele1[2])
{
  double time0;

  /****************************************************
         collinear without spin-orbit coupling
  ****************************************************/

  if ( (SpinP_switch==0 || SpinP_switch==1) && SO_switch==0 ){

    if(atomnum<=NUMPROCS_MPI_COMM_WORLD && 1<openmp_threads_num) 
      time0 = Krylov_Col_trd(state, mode, SCF_iter, Hks, OLP0, CDM, EDM, Eele0, Eele1);
    else
      time0 = Krylov_Col(state, mode, SCF_iter, Hks, OLP0, CDM, EDM, Eele0, Eele1);
  }

  /****************************************************
         collinear with spin-orbit coupling
  ****************************************************/

  else if ( (SpinP_switch==0 || SpinP_switch==1) && SO_switch==1 ){
    printf("Spin-orbit coupling is not supported for collinear DFT calculations.\n");
    MPI_Finalize();
    exit(1);
  }

  /****************************************************
   non-collinear with and without spin-orbit coupling
  ****************************************************/

  else if (SpinP_switch==3){
    printf("The O(N) Krylov subspace method is not supported for non-collinear DFT calculations.\n");
    MPI_Finalize();
    exit(1);
  }

  return time0;
}







typedef struct {
  KrylovState *state;
  char *mode;
  int SCF_iter;
  double *****Hks;
  double ****OLP0;
  double *****CDM;
  double *****EDM;
  double *Eele0;
  double *Eele1;
} Krylov_Col_arguments;
static double Krylov_Col_owned(KrylovState *state, char *mode,
			 int SCF_iter,
			 double *****Hks,
			 double ****OLP0,
			 double *****CDM,
			 double *****EDM,
			 double Eele0[2],
			 double Eele1[2], OpenMXSolverInvocation *solver_owner);
static int Krylov_Col_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Krylov_Col_arguments *a=(Krylov_Col_arguments*)arguments;
  *result=Krylov_Col_owned(a->state, a->mode,a->SCF_iter,a->Hks,a->OLP0,a->CDM,a->EDM,a->Eele0,a->Eele1,owner);
  return 1;
}
static double Krylov_Col(KrylovState *state, char *mode,
			 int SCF_iter,
			 double *****Hks,
			 double ****OLP0,
			 double *****CDM,
			 double *****EDM,
			 double Eele0[2],
			 double Eele1[2])
{
  Krylov_Col_arguments arguments={state,mode,SCF_iter,Hks,OLP0,CDM,EDM,Eele0,Eele1};
  return OpenMX_Solver_Run(Krylov_Col_operation,&arguments);
}

static double Krylov_Col_owned(KrylovState *state, char *mode,
			 int SCF_iter,
			 double *****Hks,
			 double ****OLP0,
			 double *****CDM,
			 double *****EDM,
			 double Eele0[2],
			 double Eele1[2], OpenMXSolverInvocation *solver_owner)
{


  int Mc_AN,Gc_AN,i,is,js,Gi,wan,wanA,wanB,Anum;
  int num,NUM0,NUM,NUM1,n2,Cwan,Hwan,Rn2;
  int size1,size2,max_size1,max_size2;
  int ih,ig,ian,j,kl,jg,jan,Bnum,m,n,spin,i2,ip;
  int k,l,i1,j1,P_min,m_size,q1,q2,csize,Residues_size;
  int h_AN1,Mh_AN1,h_AN2,Gh_AN1,Gh_AN2,wan1,wan2;
  int po,po1,loopN,tno1,tno2,h_AN,Gh_AN,rl1,rl2,rl;
  int MA_AN,GA_AN,tnoA,GB_AN,tnoB,ct_on;
  int Msize2_max;

  double My_TZ,sum,FermiF,time0,srt;
  double sum00,sum10,sum20,sum30;
  double sum01,sum11,sum21,sum31;
  double sum02,sum12,sum22,sum32;
  double sum03,sum13,sum23,sum33;
  double tmp0,tmp1,tmp2,tmp3,b2,co,x0,Sx,Dx,xmin,xmax;
  double My_Num_State,Num_State,x,Dnum;
  double emin,emax,de;
  double TStime,TEtime,Stime1,Etime1;
  double Stime2,Etime2;
  double time1,time2,time3,time4,time5;
  double time6,time7,time8,time9,time10;
  double time11,time12,time13,time14,time15,time16;
  double Erange;
  double My_Eele0[2],My_Eele1[2];
  double max_x=30.0;
  double ChemP_MAX,ChemP_MIN,spin_degeneracy;
  double spetrum_radius;
  OpenMXSolverRealFieldView EVal;
  OpenMXSolverResidueView Residues;
  OpenMXSolverRealFieldView PDOS_DC;
  double *tmp_array;
  double *tmp_array2;

  int *MP;

  /*****************************************************
  Msize:   \sum_FNAN Spe_Total_CNO  
  Msize2:  \sum_FNAN+SNAN Spe_Total_CNO  
  Msize3:  rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]
  Msize4:  rlmax_EC2[Mc_AN]*EKC_core_size[Mc_AN]
  Msize5:  dimension for the last column of Residues
  *****************************************************/

  int *Msize;
  int *Msize2;
  int *Msize3;
  int *Msize4;
  int *Msize5;
  OpenMXSolverIntTensor LO_TC;
  OpenMXSolverIntTensor HO_TC;
  int numprocs,myid,ID,IDS,IDR,tag=999;
  double Stime_atom, Etime_atom;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Barrier(mpi_comm_level1);
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  Krylov_GPU_InitOnce(state);
  Krylov_GPU_PreparePool(state, omp_get_max_threads());

  dtime(&TStime);

  if (measure_time==1){ 
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
    time14 = 0.0;
    time15 = 0.0;
  }

  /****************************************************
                   allocation of arrays:
  ****************************************************/

  Msize=OpenMX_Solver_Int(solver_owner,"Msize",((size_t)1)*(Matomnum+1));
  Msize2=OpenMX_Solver_Int(solver_owner,"Msize2",((size_t)1)*(Matomnum+1));
  Msize3=OpenMX_Solver_Int(solver_owner,"Msize3",((size_t)1)*(Matomnum+1));
  Msize4=OpenMX_Solver_Int(solver_owner,"Msize4",((size_t)1)*(Matomnum+1));
  Msize2_max = 0;

  /* find Msize */

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Msize[Mc_AN] = 1;
    }
    else{

      Gc_AN = M2G[Mc_AN];

      NUM = 0;
      for (i=0; i<=FNAN[Gc_AN]; i++){
	Gi = natn[Gc_AN][i];
	wanA = WhatSpecies[Gi];
	NUM += Spe_Total_CNO[wanA];
      }
      Msize[Mc_AN] = NUM;
    }
  }

  /* find Msize2 and Msize2_max */

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Msize2[Mc_AN] = 1;
    }
    else{

      Gc_AN = M2G[Mc_AN];

      NUM = 0;
      for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
	Gi = natn[Gc_AN][i];
	wanA = WhatSpecies[Gi];
	NUM += Spe_Total_CNO[wanA];
      }
      Msize2[Mc_AN] = NUM;
    }
 
    if (Msize2_max<Msize2[Mc_AN]) Msize2_max = Msize2[Mc_AN] + 4;
  }

  /* find Msize3 and Msize4 */

  Msize3[0] = 1;
  Msize4[0] = 1;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wan = WhatSpecies[Gc_AN];
    ct_on = Spe_Total_CNO[wan];
    Msize3[Mc_AN] = rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN];
    Msize4[Mc_AN] = rlmax_EC2[Mc_AN]*EKC_core_size[Mc_AN];
  }

  if (Krylov_GPU_Enabled(state)){
    Krylov_GPU_Preflight(state, Matomnum,Msize,Msize3);
    if (state->gpu_fits)
      Krylov_KUCache_Prepare(state, Matomnum,SpinP_switch+1,Msize,Msize3,SCF_iter,
                             state->gpu_device_ranks,state->gpu_need_total);
  }

  m_size = 0;

  EVal=OpenMX_Solver_Field(solver_owner,((size_t)1)*(SpinP_switch+1),((size_t)1)*(Matomnum+1));
  for (spin=0; spin<=SpinP_switch; spin++){

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      n2 = Msize3[Mc_AN] + 2;
      m_size += n2;
      OpenMX_Solver_FieldResize(EVal,spin,Mc_AN,((size_t)1)*n2);
    }
  }

  if (state->Col_owned_firsttime){
  PrintMemory("Krylov: EVal",  sizeof(double)*m_size,NULL);
  }

  if (2<=level_stdout){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        printf("<Krylov> myid=%4d Mc_AN=%4d Gc_AN=%4d Msize=%4d\n",
        myid,Mc_AN,M2G[Mc_AN],Msize[Mc_AN]);
    }
  }

  /****************************************************
    allocation of arrays:

    double PDOS[SpinP_switch+1]
               [Matomnum+1]
               [n2]
  ****************************************************/

  m_size = 0;
  PDOS_DC=OpenMX_Solver_Field(solver_owner,((size_t)1)*(SpinP_switch+1),((size_t)1)*(Matomnum+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      n2 = Msize3[Mc_AN] + 4;
      m_size += n2;
      OpenMX_Solver_FieldResize(PDOS_DC,spin,Mc_AN,((size_t)1)*n2);
    }
  }

  if (state->Col_owned_firsttime){
  PrintMemory("Krylov: PDOS_DC",sizeof(double)*m_size,NULL);
  }

  /****************************************************
    allocation of arrays:

    int LO_TC[SpinP_switch+1][Matomnum+1]
    int HO_TC[SpinP_switch+1][Matomnum+1]
  ****************************************************/

  LO_TC = OpenMX_Solver_IntView(solver_owner,"LO_TC",SpinP_switch+1,Matomnum+1);
  HO_TC = OpenMX_Solver_IntView(solver_owner,"HO_TC",SpinP_switch+1,Matomnum+1);

  /****************************************************
    allocation of array:

    double Residues[SpinP_switch+1]
                   [Matomnum+1]
                   [FNAN[Gc_AN]+1]
                   [Spe_Total_CNO[Gc_AN]] 
                   [Spe_Total_CNO[Gh_AN]] 
                   [HO_TC-LO_TC+3]
  ****************************************************/

  Residues=OpenMX_Solver_Residues(solver_owner,((size_t)1)*(SpinP_switch+1),1,0);
  FNAN[0] = 0;
  for (spin=0; spin<=SpinP_switch; spin++)
    OpenMX_Solver_ResidueResize(Residues,spin,0,1);

  /****************************************************
     initialize density and energy density matrices
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      wanA = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wanA];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	wanB = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wanB];
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
	    CDM[spin][Mc_AN][h_AN][i][j] = 0.0;
	    EDM[spin][Mc_AN][h_AN][i][j] = 0.0;
	  }
	}
      }
    }
  }

  /****************************************************
   MPI

   Hks
  ****************************************************/

  if (measure_time==1) dtime(&Stime1);

  if (SCF_iter==1){

    /***********************************
               set data size
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){
	tag = 999;

	/* find data size to send block data */
	if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	  size1 = 0;
	  for (spin=0; spin<=SpinP_switch; spin++){
	    for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
	      Mc_AN = Snd_MAN[IDS][n];
	      Gc_AN = Snd_GAN[IDS][n];
	      Cwan = WhatSpecies[Gc_AN]; 
	      tno1 = Spe_Total_CNO[Cwan];
	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
		Gh_AN = natn[Gc_AN][h_AN];        
		Hwan = WhatSpecies[Gh_AN];
		tno2 = Spe_Total_CNO[Hwan];
		size1 += tno1*tno2;
	      }
	    }
	  }

	  Snd_HFS_Size[IDS] = size1;
	  MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
	}
	else{
	  Snd_HFS_Size[IDS] = 0;
	}

	/* receiving of size of data */

	if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

	  MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	  Rcv_HFS_Size[IDR] = size2;
	}
	else{
	  Rcv_HFS_Size[IDR] = 0;
	}

	if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
      }
      else{
	Snd_HFS_Size[IDS] = 0;
	Rcv_HFS_Size[IDR] = 0;
      }
    }
  }

  /***********************************
             data transfer
  ************************************/

  /* find maximum size of size1 */

  max_size1 = 0;
  max_size2 = 0;
  for (ID=0; ID<numprocs; ID++){
    size1 = Snd_HFS_Size[ID];
    if (max_size1<size1) max_size1 = size1;
    size2 = Rcv_HFS_Size[ID];
    if (max_size2<size2) max_size2 = size2;
  }

  /* allocation of arrays */

  tmp_array=OpenMX_Solver_Real(solver_owner,"tmp_array",((size_t)1)*max_size1);
  tmp_array2=OpenMX_Solver_Real(solver_owner,"tmp_array2",((size_t)1)*max_size2);

  /* MPI communication */

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      /*****************************
              sending of data 
      *****************************/

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

        size1 = Snd_HFS_Size[IDS];

        /* multidimentional array to vector array */

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
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

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

        size2 = Rcv_HFS_Size[IDR];
        
        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          Mc_AN = S_TopMAN[IDR] - 1;  /* S_TopMAN should be used. */
          for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];

            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  Hks[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
		}
	      }
	    }
	  }        
	}
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
        MPI_Wait(&request,&stat);
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

  if (SCF_iter==1){

    /***********************************
               data transfer
    ************************************/

    tag = 999;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){

        /*****************************
                sending of data 
        *****************************/

        if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

          size1 = Snd_HFS_Size[IDS]/(1+SpinP_switch);

          /* multidimentional array to vector array */

          num = 0;

          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
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

        if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
          
          size2 = Rcv_HFS_Size[IDR]/(1+SpinP_switch);
        
          MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

          num = 0;
          Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
          for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];

            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  OLP0[Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
   	        }
	      }
	    }
	  }        
        }

        if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
          MPI_Wait(&request,&stat);
	}
      }
    }
  }

  /* freeing of arrays */
  ;

  if (measure_time==1){ 
    dtime(&Etime1);
    time1 = Etime1 - Stime1;      
  }

  /***********************************************
    for regeneration of the buffer matrix
  ***********************************************/

  if (sqrt(fabs(state->history->norm[0]))<(0.2+0.10*(double)atomnum) && state->Col_owned_recalc_firsttime){
    state->Col_owned_recalc_flag = 1;
    state->Col_owned_recalc_firsttime = 0;
  }
  else{
    state->Col_owned_recalc_flag = 0;
  }

  if (SCF_iter==1) state->Col_owned_recalc_firsttime = 1;

  if (error_check==1){
    printf("SCF_iter=%2d recalc_firsttime=%2d\n",SCF_iter,state->Col_owned_recalc_firsttime);
  }

  if (measure_time==1) dtime(&Stime2);

#pragma omp parallel shared(EKC_invS_flag,List_YOUSO,Residues,EDM,CDM,HO_TC,LO_TC,EVal,RMI1,S_G2M,EC_matrix,state,recalc_EM,Krylov_U,SpinP_switch,EKC_core_size,EKC_core_size_max,rlmax_EC,rlmax_EC2,time11,time10,time9,time8,time7,time6,time5,time4,time3,time2,Hks,OLP0,EKC_Exact_invS_flag,SCF_iter,Msize4,Msize3,Msize2,Msize,Msize2_max,natn,FNAN,SNAN,Spe_Total_CNO,WhatSpecies,M2G,Matomnum,myid,time_per_atom)
  {
    int OMPID,Nthrds,Nprocs;
    int Mc_AN,Gc_AN,wan,spin;
    int ig,ian,ih,kl,jg,jan,Bnum,m,n,rl;
    int Anum,i,j,k,Gi,wanA,NUM,n2,csize,is,i2;
    int i1,rl1,js,ip,po1,tno1,h_AN,Gh_AN,wanB,tno2;
    int *MP;
    int KU_d1, KU_d2,lda,ldb,ldc,M,N,K;
    double alpha, beta;
    OpenMXSolverRealTensor invS = {0};
    double *LoS;
    double *C,*KU;
    double *H_DC,*ko;
    double ***Krylov_U_OLP = NULL;
    OpenMXSolverRealTensor inv_RS = {0};
    double sum00,sum10,sum20,sum30,sum;
    double tmp0,tmp1,tmp2,tmp3;
    double Erange;
    double Stime_atom,Etime_atom;
    double Stime1,Etime1;
    OpenMXSolverRealTensor tmpvec0 = {0};
    OpenMXSolverRealTensor tmpvec1 = {0};
    OpenMXSolverRealTensor tmpvec2 = {0};
    Krylov_GPU_Workspace *gpu_ws;

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();
    gpu_ws = Krylov_GPU_GetWorkspace(state, OMPID);

    /* allocation of arrays */

    MP=OpenMX_Solver_Int(solver_owner,"MP",((size_t)1)*List_YOUSO[2]);

    tmpvec0=OpenMX_Solver_RealView(solver_owner,"tmpvec0",((size_t)1)*EKC_core_size_max,((size_t)1)*Msize2_max);


    tmpvec1=OpenMX_Solver_RealView(solver_owner,"tmpvec1",((size_t)1)*EKC_core_size_max,((size_t)1)*Msize2_max);


    tmpvec2=OpenMX_Solver_RealView(solver_owner,"tmpvec2",((size_t)1)*EKC_core_size_max,((size_t)1)*Msize2_max);


    if (state->Col_owned_firsttime && OMPID==0){
      PrintMemory("Krylov: tmpvec0",sizeof(double)*EKC_core_size_max*Msize2_max,NULL);
      PrintMemory("Krylov: tmpvec1",sizeof(double)*EKC_core_size_max*Msize2_max,NULL);
      PrintMemory("Krylov: tmpvec2",sizeof(double)*EKC_core_size_max*Msize2_max,NULL);
    }

    /***********************************************
     main loop of calculation 
    ***********************************************/

    for (Mc_AN=1+OMPID; Mc_AN<=Matomnum; Mc_AN+=Nthrds){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wan = WhatSpecies[Gc_AN];

      /* MP array */

      Anum = 1;
      for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
	MP[i] = Anum;
	Gi = natn[Gc_AN][i];
	wanA = WhatSpecies[Gi];
	Anum += Spe_Total_CNO[wanA];
      }
      NUM = Anum - 1;
      n2 = NUM + 40;

      /***********************************************
                 allocation of arrays:
      ***********************************************/

      if (Msize[Mc_AN]<Msize3[Mc_AN])
	csize = Msize3[Mc_AN] + 40;
      else
	csize = Msize[Mc_AN] + 40;

      KU_d1 = EKC_core_size[Mc_AN]*Msize2[Mc_AN];
      KU_d2 = Msize2[Mc_AN];

      H_DC=OpenMX_Solver_Real(solver_owner,"H_DC",((size_t)1)*csize*csize);
      ko=OpenMX_Solver_Real(solver_owner,"ko",((size_t)1)*csize);
      C=OpenMX_Solver_Real(solver_owner,"C",((size_t)1)*csize*csize);
      KU=OpenMX_Solver_Real(solver_owner,"KU",((size_t)1)*(Msize2[Mc_AN]+2)*Msize3[Mc_AN]);

      /***********************************************
          calculate the inverse of overlap matrix
      ***********************************************/

      if (SCF_iter==1 && Msize3[Mc_AN]<Msize2[Mc_AN] && EKC_Exact_invS_flag==1){

	LoS=OpenMX_Solver_Real(solver_owner,"LoS",((size_t)1)*(Msize2[Mc_AN]+3)*(Msize2[Mc_AN]+3));

	invS=OpenMX_Solver_RealView(solver_owner,"invS",((size_t)1)*(Msize2[Mc_AN]+3),((size_t)1)*(Msize2[Mc_AN]+3));


	if (measure_time==1) dtime(&Stime1);

	Inverse_S_by_Cholesky(Mc_AN, OLP0, OpenMX_Solver_RealMatrix(solver_owner,"invS",invS.extent[0],invS.extent[1]), MP, NUM, LoS);

	if (measure_time==1 && OMPID==0){ 
	  dtime(&Etime1);
	  time2 += Etime1 - Stime1;      
	}
      }

      else if (SCF_iter==1 && Msize3[Mc_AN]<Msize2[Mc_AN] && EKC_invS_flag==1){

        Krylov_U_OLP = OpenMX_Solver_RealVolume(solver_owner,"Krylov_U_OLP",
            rlmax_EC2[Mc_AN],EKC_core_size[Mc_AN],Msize2[Mc_AN]+3);
        for (i=0; i<rlmax_EC2[Mc_AN]; i++)
          for (j=0; j<EKC_core_size[Mc_AN]; j++)
            for (k=0; k<Msize2[Mc_AN]+3; k++) Krylov_U_OLP[i][j][k] = 0.0;

	inv_RS=OpenMX_Solver_RealView(solver_owner,"inv_RS",((size_t)1)*(rlmax_EC2[Mc_AN]+1)*EKC_core_size[Mc_AN],((size_t)1)*(rlmax_EC2[Mc_AN]+1)*EKC_core_size[Mc_AN]);


	if (measure_time==1) dtime(&Stime1);

	Krylov_IOLP( Mc_AN, OLP0, Krylov_U_OLP, OpenMX_Solver_RealMatrix(solver_owner,"inv_RS",inv_RS.extent[0],inv_RS.extent[1]), MP, Msize2, Msize4, Msize2_max, OpenMX_Solver_RealMatrix(solver_owner,"tmpvec0",tmpvec0.extent[0],tmpvec0.extent[1]), OpenMX_Solver_RealMatrix(solver_owner,"tmpvec1",tmpvec1.extent[0],tmpvec1.extent[1]) );

	if (measure_time==1 && OMPID==0){ 
	  dtime(&Etime1);
	  time2 += Etime1 - Stime1;      
	}
      }

      for (spin=0; spin<=SpinP_switch; spin++){

	/****************************************************
                generate a preconditioning matrix
	****************************************************/

	if (measure_time==1) dtime(&Stime1);

	if (SCF_iter==1 && Msize3[Mc_AN]<Msize2[Mc_AN]){

	  Generate_pMatrix(state,  myid, spin, Mc_AN, Hks, OLP0, OpenMX_Solver_RealMatrix(solver_owner,"invS",invS.extent[0],invS.extent[1]), Krylov_U, Krylov_U_OLP, OpenMX_Solver_RealMatrix(solver_owner,"inv_RS",inv_RS.extent[0],inv_RS.extent[1]), MP,
                            Msize, Msize2, Msize3, Msize4, Msize2_max, OpenMX_Solver_RealMatrix(solver_owner,"tmpvec0",tmpvec0.extent[0],tmpvec0.extent[1]), OpenMX_Solver_RealMatrix(solver_owner,"tmpvec1",tmpvec1.extent[0],tmpvec1.extent[1]), OpenMX_Solver_RealMatrix(solver_owner,"tmpvec2",tmpvec2.extent[0],tmpvec2.extent[1]) );

	}
	else if (SCF_iter==1){
	  Generate_pMatrix2( myid, spin, Mc_AN, Hks, OLP0, Krylov_U, MP, Msize, Msize2, Msize3, OpenMX_Solver_RealMatrix(solver_owner,"tmpvec1",tmpvec1.extent[0],tmpvec1.extent[1]) );
	}

	if (measure_time==1 && OMPID==0){ 
	  dtime(&Etime1);
	  time3 += Etime1 - Stime1;      
	}

	if (measure_time==1) dtime(&Stime1);

	if (recalc_EM==1 || SCF_iter==1 || state->Col_owned_recalc_flag==1){

	  Embedding_Matrix(state,  spin, Mc_AN, Hks, Krylov_U, EC_matrix, MP, Msize, Msize2, Msize3, OpenMX_Solver_RealMatrix(solver_owner,"tmpvec1",tmpvec1.extent[0],tmpvec1.extent[1]));
	}

	if (measure_time==1 && OMPID==0){ 
	  dtime(&Etime1);
	  time4 += Etime1 - Stime1;      
	}

	/****************************************************
                construct the Hamiltonian matrix
	****************************************************/

	if (measure_time==1) dtime(&Stime1);
        
	for (i=0; i<=FNAN[Gc_AN]; i++){
	  ig = natn[Gc_AN][i];
	  ian = Spe_Total_CNO[WhatSpecies[ig]];
	  Anum = MP[i] - 1;
	  ih = S_G2M[ig]; /* S_G2M should be used */

	  for (j=0; j<=FNAN[Gc_AN]; j++){

	    kl = RMI1[Mc_AN][i][j];
	    jg = natn[Gc_AN][j];
	    jan = Spe_Total_CNO[WhatSpecies[jg]];
	    Bnum = MP[j] - 1;

	    if (0<=kl){
	      for (m=0; m<ian; m++){
		for (n=0; n<jan; n++){
		  H_DC[(Anum+m)*Msize[Mc_AN]+Bnum+n] = Hks[spin][ih][kl][m][n];
		}
	      }
	    }

	    else{
	      for (m=0; m<ian; m++){
		for (n=0; n<jan; n++){
		  H_DC[(Anum+m)*Msize[Mc_AN]+Bnum+n] = 0.0;
		}
	      }
	    }
	  }
	}

	if (measure_time==1 && OMPID==0){ 
	  dtime(&Etime1);
	  time5 += Etime1 - Stime1;      
	}

	/****************************************************
                   transform u1^+ * H_DC * u1
	****************************************************/

	/* H_DC * u1 */

	if (measure_time==1) dtime(&Stime1);

	/* original version

	for (i=1; i<=Msize[Mc_AN]; i++){
	  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (j=1; j<=Msize[Mc_AN]; j++){
		sum += H_DC[i][j]*Krylov_U[spin][Mc_AN][rl][n][j];  
	      }

	      C[rl*EKC_core_size[Mc_AN]+n+1][i] = sum;
	    }      
	  }      
	} 
	*/

	/* BLAS3 version */

	if (Krylov_ProjectedSolve_GPU(state, gpu_ws,spin,Mc_AN,Msize,Msize2,Msize3,
	                              Krylov_U,EC_matrix,H_DC,KU,C,ko)==0){

	for (i=0; i<Msize[Mc_AN]; i++){
	  for (j=0; j<Msize3[Mc_AN]; j++){
	    KU[j*Msize[Mc_AN]+i] = Krylov_U[spin][Mc_AN][j*Msize2[Mc_AN]+i+1];
	  }
	}

	alpha = 1.0; beta = 0.0; 
	M = Msize[Mc_AN]; N = Msize3[Mc_AN]; K = Msize[Mc_AN];
	lda = M; ldb = K; ldc = Msize[Mc_AN]; 
	
	Krylov_Dgemm(state, gpu_ws, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K,
                     H_DC, lda, KU, ldb, C, ldc);

	if (measure_time==1 && OMPID==0){ 
	  dtime(&Etime1);
	  time6 += Etime1 - Stime1;      
	}

	/* u1^+ * H_DC * u1 */

	if (measure_time==1) dtime(&Stime1);

	/* original version */

	/*
	for (rl1=0; rl1<rlmax_EC[Mc_AN]; rl1++){                                                     
          for (m=0; m<EKC_core_size[Mc_AN]; m++){                                                
	    for (rl2=rl1; rl2<rlmax_EC[Mc_AN]; rl2++){                            
	      for (n=0; n<EKC_core_size[Mc_AN]; n++){                                      
		sum = 0.0;                                                                              
		i2 = rl2*EKC_core_size[Mc_AN] + n + 1;                                                  
		for (i=1; i<=Msize[Mc_AN]; i++){                                                    
		  sum += Krylov_U[spin][Mc_AN][rl1][m][i]*C[i2][i];                          
		}                                                                                  

		H_DC[rl1*EKC_core_size[Mc_AN]+m+1][rl2*EKC_core_size[Mc_AN]+n+1] = sum;                 
		H_DC[rl2*EKC_core_size[Mc_AN]+n+1][rl1*EKC_core_size[Mc_AN]+m+1] = sum;          
	      }                                                                                         
	    }                                                                              
          }                                                                               
	}                                            
	*/

	/* BLAS3 version */

	alpha = 1.0; beta = 0.0; 
	M = Msize3[Mc_AN]; N = Msize3[Mc_AN]; K = Msize[Mc_AN];
	lda = K; ldb = K; ldc = Msize3[Mc_AN]; 

	Krylov_Dgemm(state, gpu_ws, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K,
                     KU, lda, C, ldb, H_DC, ldc);

	if (measure_time==1 && OMPID==0){ 
	  dtime(&Etime1);
	  time7 += Etime1 - Stime1;      
	}

	/* correction for ZeroNum */

	m = (int)Krylov_U[spin][Mc_AN][0];
	for (i=0; i<m; i++){
	  H_DC[i*Msize3[Mc_AN]+i] = 1.0e+3;
	}

	/****************************************************
            H0 = u1^+ * H_DC * u1 + D 
	****************************************************/

	if (measure_time==1) dtime(&Stime1);

	for (i=(Msize3[Mc_AN]-1); 0<=i; i--){
	  for (j=0; j<Msize3[Mc_AN]; j++){
	    H_DC[(i+1)*(Msize3[Mc_AN]+1)+(j+1)] = H_DC[i*Msize3[Mc_AN]+j] + EC_matrix[spin][Mc_AN][i+1][j+1];
	  }
	}

	if (measure_time==1 && OMPID==0){ 
	  dtime(&Etime1);
	  time8 += Etime1 - Stime1;      
	}

	/****************************************************
           diagonalize
	****************************************************/

	if (measure_time==1) dtime(&Stime1);

	Krylov_Eigen2(state, gpu_ws,H_DC,Msize3[Mc_AN]+1,ko,Msize3[Mc_AN],Msize3[Mc_AN]);

	if (measure_time==1 && OMPID==0){ 
	  dtime(&Etime1);
	  time9 += Etime1 - Stime1;      
	}

	/********************************************
           back transformation of eigenvectors
                      c = u1 * b
	*********************************************/

	if (measure_time==1) dtime(&Stime1);

	/* original version */

	/*
	for (i=1; i<=Msize[Mc_AN]; i++){                                                              
          for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){                                                   
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){  

	      tmp1 = Krylov_U[spin][Mc_AN][rl][n][i]; 
	      i1 = rl*EKC_core_size[Mc_AN] + n + 1; 

	      for (j=1; j<=Msize3[Mc_AN]; j++){  
		C[i][j] += tmp1*H_DC[i1][j];                                                     
	      }
	    }   
          }                                                                                           
	}                                                                                              
	*/

	/* BLAS3 version */

	alpha = 1.0; beta = 0.0; 
        M = Msize3[Mc_AN]; N = Msize[Mc_AN]; K = Msize3[Mc_AN];
        lda = K; ldb = N; ldc = Msize3[Mc_AN]; 

	Krylov_Dgemm(state, gpu_ws, HIPBLAS_OP_T, HIPBLAS_OP_T, M, N, K,
                     H_DC, lda, KU, ldb, C, ldc);

	} /* fallback of fused GPU projected solve */

	if (measure_time==1 && OMPID==0){ 
	  dtime(&Etime1);
	  time10 += Etime1 - Stime1;      
	}

	if (measure_time==1) dtime(&Stime1);

	/***********************************************
          store eigenvalues and residues of poles
	***********************************************/

	{
	  double *eval_row_41=openmx_solver_field_row(&EVal,spin,Mc_AN);
	  for (i=1; i<=Msize3[Mc_AN]; i++){
	  eval_row_41[i-1] = ko[i];
	}
	}

	/******************************************************
        set an energy range (-Erange+ChemP to Erange+ChemP)
        of eigenvalues used to store the Residues.
	******************************************************/

	Erange = 0.367493245;  /* in hartree, corresponds to 10 eV */

	/***********************************************
          find LO_TC and HO_TC
	***********************************************/

	/* LO_TC */ 
	i = 0;
	ip = 0;
	po1 = 0;
	do{
	  if ( (state->chemical->chemical_potential-Erange)<openmx_solver_field_row(&EVal,spin,Mc_AN)[i]){
	    ip = i;
	    po1 = 1; 
	  }
	  i++;
	} while (po1==0 && i<Msize3[Mc_AN]);

	openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN] = ip;

	/* HO_TC */ 
	i = 0;
	ip = Msize3[Mc_AN]-1;
	po1 = 0;
	do{
	  if ( (state->chemical->chemical_potential+Erange)<openmx_solver_field_row(&EVal,spin,Mc_AN)[i]){
	    ip = i;
	    po1 = 1; 
	  }
	  i++;
	} while (po1==0 && i<Msize3[Mc_AN]);

	openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN] = ip;

	/***********************************************
          store residues of poles
	***********************************************/

	n2 = openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN] - openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN] + 3;
	if (n2<1) n2 = 1;

	OpenMX_Solver_ResidueResize(Residues,spin,Mc_AN,n2);

	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];

	{
	  double *eval_row_40=openmx_solver_field_row(&EVal,spin,Mc_AN);
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  wanB = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wanB];
	  Bnum = MP[h_AN] - 1;

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){

	      for (i1=0; i1<openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]; i1++){
		tmp1 = C[i*Msize3[Mc_AN]+i1]*C[(Bnum+j)*Msize3[Mc_AN]+i1];
		CDM[spin][Mc_AN][h_AN][i][j] += tmp1;
		EDM[spin][Mc_AN][h_AN][i][j] += tmp1*eval_row_40[i1];
	      }      
      


	      {
	        double *residues_row_36=openmx_solver_residue_row(&Residues,spin,Mc_AN,h_AN,i,j);
	        for (i1=openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]; i1<=openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN]; i1++){
		residues_row_36[i1-openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]]
		  = C[i*Msize3[Mc_AN]+i1]*C[(Bnum+j)*Msize3[Mc_AN]+i1];
	      }
	      }
	    }
	  }
	}
	}

	if (measure_time==1 && OMPID==0){ 
	  dtime(&Etime1);
	  time11 += Etime1 - Stime1;      
	}

      } /* spin */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

  } /* #pragma omp parallel */

  if (measure_time==1){ 
    dtime(&Etime2);
    time16 = Etime2 - Stime2;      
  }

  if (state->Col_owned_firsttime){

    Residues_size = 1;

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];
	wan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wanB = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wanB];
	  n2 = openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN] - openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN] + 3;
	  Residues_size += tno1*tno2*n2;
	}
      }
    }

    PrintMemory("Krylov: Residues",sizeof(double)*Residues_size,NULL);
  }

  /****************************************************
                calculate the projected DOS
  ****************************************************/

  if (measure_time==1) dtime(&Stime1);

#pragma omp parallel shared(time_per_atom,Residues,LO_TC,HO_TC,EDM,CDM,OLP0,natn,FNAN,PDOS_DC,Msize3,Spe_Total_CNO,WhatSpecies,M2G,SpinP_switch,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,spin,Gc_AN,wanA,tno1,i1,i,h_AN,Gh_AN,wanB,tno2,j,tmp1,Etime_atom)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=1+OMPID; Mc_AN<=Matomnum; Mc_AN+=Nthrds){

      dtime(&Stime_atom);

      for (spin=0; spin<=SpinP_switch; spin++){

	Gc_AN = M2G[Mc_AN];
	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];

	{
	  double *pdos_dc_row_35=openmx_solver_field_row(&PDOS_DC,spin,Mc_AN);
	  for (i1=0; i1<=(Msize3[Mc_AN]+1); i1++){
	  pdos_dc_row_35[i1] = 0.0;
	}
	}

	{
	  double *pdos_dc_row_34=openmx_solver_field_row(&PDOS_DC,spin,Mc_AN);
	  for (i=0; i<tno1; i++){
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wanB = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wanB];
	    for (j=0; j<tno2; j++){

	      tmp1 = OLP0[Mc_AN][h_AN][i][j];

	      pdos_dc_row_34[0] += tmp1*CDM[spin][Mc_AN][h_AN][i][j];
	      pdos_dc_row_34[1] += tmp1*EDM[spin][Mc_AN][h_AN][i][j];

	      {
	        double *residues_row_31=openmx_solver_residue_row(&Residues,spin,Mc_AN,h_AN,i,j);
	        for (i1=0; i1<(openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN]-openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]+1); i1++){
		pdos_dc_row_34[i1+2] += residues_row_31[i1]*tmp1;
	      }
	      }

	    }            
	  }        
	}
	}

      } /* spin */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

  } /* #pragma omp parallel */

  if (measure_time==1){ 
    dtime(&Etime1);
    time12 += Etime1 - Stime1;      
  }

  /****************************************************
           find the total number of electrons 
  ****************************************************/

  if (measure_time==1) dtime(&Stime1);

  My_TZ = 0.0;
  for (i=1; i<=Matomnum; i++){
    Gc_AN = M2G[i];
    wan = WhatSpecies[Gc_AN];
    My_TZ += Spe_Core_Charge[wan];
  }

  /* MPI, My_TZ */

  MPI_Barrier(mpi_comm_level1);
  MPI_Allreduce(&My_TZ, &state->Col_owned_TZ, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /****************************************************
                find the chemical potential
  ****************************************************/

  po = 0;
  loopN = 0;

  ChemP_MAX = 10.0;  
  ChemP_MIN =-10.0;
  if      (SpinP_switch==0) spin_degeneracy = 2.0;
  else if (SpinP_switch==1) spin_degeneracy = 1.0;

  do {
    state->chemical->chemical_potential = 0.50*(ChemP_MAX + ChemP_MIN);

    My_Num_State = 0.0;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      for (spin=0; spin<=SpinP_switch; spin++){

        dtime(&Stime_atom);

        Gc_AN = M2G[Mc_AN];

        My_Num_State += spin_degeneracy*openmx_solver_field_row(&PDOS_DC,spin,Mc_AN)[0];

        {
          double *eval_row_28=openmx_solver_field_row(&EVal,spin,Mc_AN);
          double *pdos_dc_row_29=openmx_solver_field_row(&PDOS_DC,spin,Mc_AN);
          for (i=0; i<(openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN]-openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]+1); i++){

          x = (eval_row_28[i+openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]] - state->chemical->chemical_potential)*Beta;
          if (x<=-max_x) x = -max_x;
          if (max_x<=x)  x = max_x;
          FermiF = 1.0/(1.0 + exp(x));
          My_Num_State += spin_degeneracy*FermiF*pdos_dc_row_29[i+2];
	}
        }

        dtime(&Etime_atom);
        time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      }
    }

    /* MPI, My_Num_State */

    MPI_Barrier(mpi_comm_level1);
    MPI_Allreduce(&My_Num_State, &Num_State, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    Dnum = (state->Col_owned_TZ - Num_State) - system_charge;
    if (0.0<=Dnum) ChemP_MIN = state->chemical->chemical_potential;
    else           ChemP_MAX = state->chemical->chemical_potential;
    if (fabs(Dnum)<1.0e-11) po = 1;


    if (myid==Host_ID && 3<=level_stdout){
      printf("  ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",state->chemical->chemical_potential,state->Col_owned_TZ,Num_State);
    }

    loopN++;
  }
  while (po==0 && loopN<1000); 

  if (measure_time==1){ 
    dtime(&Etime1);
    time13 += Etime1 - Stime1;      
  }

  /****************************************************
        eigenenergy by summing up eigenvalues
  ****************************************************/

  if (measure_time==1) dtime(&Stime1);

  My_Eele0[0] = 0.0;
  My_Eele0[1] = 0.0;
  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      My_Eele0[spin] += openmx_solver_field_row(&PDOS_DC,spin,Mc_AN)[1];

      {
        double *eval_row_26=openmx_solver_field_row(&EVal,spin,Mc_AN);
        double *pdos_dc_row_27=openmx_solver_field_row(&PDOS_DC,spin,Mc_AN);
        for (i=0; i<(openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN]-openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]+1); i++){

        x = (eval_row_26[i+openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]] - state->chemical->chemical_potential)*Beta;

        if (x<=-max_x) x = -max_x;
        if (max_x<=x)  x = max_x;
        FermiF = 1.0/(1.0 + exp(x));
        My_Eele0[spin] += FermiF*eval_row_26[i+openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]]*pdos_dc_row_27[i+2];
      }
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }
  }

  /* MPI, My_Eele0 */
  for (spin=0; spin<=SpinP_switch; spin++){
    MPI_Barrier(mpi_comm_level1);
    MPI_Allreduce(&My_Eele0[spin], &Eele0[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  }

  if (SpinP_switch==0){
    Eele0[1] = Eele0[0];
  }

  if (measure_time==1){ 
    dtime(&Etime1);
    time14 += Etime1 - Stime1;      
  }

  if (measure_time==1) dtime(&Stime1);

#pragma omp parallel shared(FNAN,time_per_atom,EDM,CDM,Residues,natn,max_x,Beta,state,EVal,LO_TC,HO_TC,Spe_Total_CNO,WhatSpecies,M2G,SpinP_switch,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,spin,Stime_atom,Gc_AN,wanA,tno1,i1,x,FermiF,h_AN,Gh_AN,wanB,tno2,i,j,tmp1,Etime_atom)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=1+OMPID; Mc_AN<=Matomnum; Mc_AN+=Nthrds){
      for (spin=0; spin<=SpinP_switch; spin++){

	dtime(&Stime_atom);

	Gc_AN = M2G[Mc_AN];
	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];

	{
	  double *eval_row_25=openmx_solver_field_row(&EVal,spin,Mc_AN);
	  for (i1=0; i1<(openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN]-openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]+1); i1++){

	  x = (eval_row_25[i1+openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]] - state->chemical->chemical_potential)*Beta;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;
	  FermiF = 1.0/(1.0 + exp(x));

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wanB = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wanB];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		tmp1 = FermiF*openmx_solver_residue_row(&Residues,spin,Mc_AN,h_AN,i,j)[i1];
		CDM[spin][Mc_AN][h_AN][i][j] += tmp1;
		EDM[spin][Mc_AN][h_AN][i][j] += tmp1*eval_row_25[i1+openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]];
	      }
	    }
	  }
	}
	}

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      }
    }

  } /* #pragma omp parallel */

  /****************************************************
                      bond energies
  ****************************************************/

  My_Eele1[0] = 0.0;
  My_Eele1[1] = 0.0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];    
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    for (j=0; j<=FNAN[GA_AN]; j++){
      GB_AN = natn[GA_AN][j];  
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];

      for (k=0; k<tnoA; k++){
	for (l=0; l<tnoB; l++){
	  for (spin=0; spin<=SpinP_switch; spin++){
	    My_Eele1[spin] += CDM[spin][MA_AN][j][k][l]*Hks[spin][MA_AN][j][k][l];
	  }
	}
      }
  
    }
  }

  /* MPI, My_Eele1 */
  MPI_Barrier(mpi_comm_level1);
  for (spin=0; spin<=SpinP_switch; spin++){
    MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE,
                   MPI_SUM, mpi_comm_level1);
  }

  if (SpinP_switch==0){
    Eele1[1] = Eele1[0];
  }

  if (3<=level_stdout && myid==Host_ID){
    printf("  Eele00=%15.12f Eele01=%15.12f\n",Eele0[0],Eele0[1]);
    printf("  Eele10=%15.12f Eele11=%15.12f\n",Eele1[0],Eele1[1]);
  }

  if (measure_time==1){ 
    dtime(&Etime1);
    time15 += Etime1 - Stime1;      
  }

  if ( strcasecmp(mode,"dos")==0 ){
    Save_DOS_Col(state->chemical, Residues,OLP0,EVal,LO_TC,HO_TC);
  }

  if (measure_time==1){ 
    printf("myid=%2d time1 =%5.3f time2 =%5.3f time3 =%5.3f time4 =%5.3f time5 =%5.3f\n",
            myid,time1,time2,time3,time4,time5);
    printf("myid=%2d time6 =%5.3f time7 =%5.3f time8 =%5.3f time9 =%5.3f time10=%5.3f\n",
            myid,time6,time7,time8,time9,time10);
    printf("myid=%2d time11=%5.3f time12=%5.3f time13=%5.3f time14=%5.3f time15=%5.3f\n",
            myid,time11,time12,time13,time14,time15);
    printf("myid=%2d time16=%5.3f\n",myid,time16);
  }

  /* The C++ invocation releases all per-solve workspaces. */
  FNAN[0] = 0;

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  if (measure_time==1){ 
  printf("total time=%15.12f\n",time0);
  }

  /* for PrintMemory */
  state->Col_owned_firsttime=0;

  return time0;
}









typedef struct {
  KrylovState *state;
  int myid;
  int spin;
  int Mc_AN;
  double *****Hks;
  double ****OLP0;
  double **invS;
  double ***Krylov_U;
  double ***Krylov_U_OLP;
  double **inv_RS;
  int *MP;
  int *Msize;
  int *Msize2;
  int *Msize3;
  int *Msize4;
  int Msize2_max;
  double **tmpvec0;
  double **tmpvec1;
  double **tmpvec2;
} Generate_pMatrix_scratch_arguments;
static void Generate_pMatrix_raii(KrylovState *state,  int myid, int spin, int Mc_AN, double *****Hks,
                       double ****OLP0, double **invS,
                       double ***Krylov_U, double ***Krylov_U_OLP, double **inv_RS, int *MP,
                       int *Msize, int *Msize2, int *Msize3, int *Msize4, int Msize2_max,
                       double **tmpvec0, double **tmpvec1, double **tmpvec2 ,OpenMXSolverInvocation *solver_owner);
static int Generate_pMatrix_scratch_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Generate_pMatrix_scratch_arguments *a=(Generate_pMatrix_scratch_arguments*)arguments;
  Generate_pMatrix_raii(a->state, a->myid,a->spin,a->Mc_AN,a->Hks,a->OLP0,a->invS,a->Krylov_U,a->Krylov_U_OLP,a->inv_RS,a->MP,a->Msize,a->Msize2,a->Msize3,a->Msize4,a->Msize2_max,a->tmpvec0,a->tmpvec1,a->tmpvec2,owner);
  *result=0;
  return 1;
}
void Generate_pMatrix(KrylovState *state,  int myid, int spin, int Mc_AN, double *****Hks,
                       double ****OLP0, double **invS,
                       double ***Krylov_U, double ***Krylov_U_OLP, double **inv_RS, int *MP, 
                       int *Msize, int *Msize2, int *Msize3, int *Msize4, int Msize2_max, 
                       double **tmpvec0, double **tmpvec1, double **tmpvec2 )
{
  Generate_pMatrix_scratch_arguments arguments={state,myid,spin,Mc_AN,Hks,OLP0,invS,Krylov_U,Krylov_U_OLP,inv_RS,MP,Msize,Msize2,Msize3,Msize4,Msize2_max,tmpvec0,tmpvec1,tmpvec2};
  OpenMX_Solver_Run(Generate_pMatrix_scratch_operation,&arguments);
}

static void Generate_pMatrix_raii(KrylovState *state,  int myid, int spin, int Mc_AN, double *****Hks,
                       double ****OLP0, double **invS,
                       double ***Krylov_U, double ***Krylov_U_OLP, double **inv_RS, int *MP,
                       int *Msize, int *Msize2, int *Msize3, int *Msize4, int Msize2_max,
                       double **tmpvec0, double **tmpvec1, double **tmpvec2 ,OpenMXSolverInvocation *solver_owner)
{
  int rl,rl0,rl1,ct_AN,fan,san,can,wan,ct_on,i,j;
  int n,Anum,Bnum,k,ian,ih,kl,jg,ig,jan,m,m1,n1;
  int ZeroNum,Gh_AN,wanB,m1s,is;
  int rl00,rl01,rl02,rl03,rl04,rl05,rl06,rl07;
  int mm0,mm1,mm2,mm3,mm4,mm5,mm6,mm7;

  int KU_d1, KU_d2, csize; 
  __m128d mmSum00,mmSum01,mmSum10,mmSum11,mmSum20,mmSum21,mmSum30,mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5; 

  double mmArr[8];
  double time1,time2,time3,time4,time5;
  double time6,time7,time8,time9,time10;
  double Stime1,Etime1;
  double sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7;
  double sum,dum,tmp0,tmp1,tmp2,tmp3,rcutA,r0;
  double **Utmp,**matRS0,**matRS1;
  double **tmpmat0;
  double *ko,*iko;
  double **FS;
  double ***U0;

  ct_AN = M2G[Mc_AN];
  fan = FNAN[ct_AN];
  san = SNAN[ct_AN];
  can = fan + san;
  wan = WhatSpecies[ct_AN];
  ct_on = Spe_Total_CNO[wan];
  rcutA = Spe_Atom_Cut1[wan]; 

  if (Msize[Mc_AN]<Msize3[Mc_AN])
    csize = Msize3[Mc_AN] + 40;
  else
    csize = Msize[Mc_AN] + 40;

  KU_d1 = EKC_core_size[Mc_AN]*Msize2[Mc_AN];
  KU_d2 = Msize2[Mc_AN];

  /* allocation of arrays */

  Utmp=OpenMX_Solver_AlignedRealMatrix(solver_owner,"Utmp-1",((size_t)1)*rlmax_EC[Mc_AN],((size_t)1)*EKC_core_size[Mc_AN]);
  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
  }

  U0=OpenMX_Solver_AlignedRealVolume(solver_owner,"U0-2",((size_t)1)*rlmax_EC[Mc_AN],((size_t)1)*EKC_core_size[Mc_AN],((size_t)1)*(Msize2[Mc_AN]+3));
  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
    for (j=0; j<EKC_core_size[Mc_AN]; j++){
      ;
      for (k=0; k<(Msize2[Mc_AN]+3); k++)  U0[i][j][k] = 0.0;
    }
  }  

  tmpmat0=OpenMX_Solver_AlignedRealMatrix(solver_owner,"tmpmat0-3",((size_t)1)*(EKC_core_size[Mc_AN]+4),((size_t)1)*(EKC_core_size[Mc_AN]+4));
  for (i=0; i<(EKC_core_size[Mc_AN]+4); i++){
    ;
  }

  FS=OpenMX_Solver_AlignedRealMatrix(solver_owner,"FS-4",((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN],((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  for (i=0; i<(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]; i++){
    ;
  }

  ko=(double*)OpenMX_Solver_Real(solver_owner,"ko-7",((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  iko=(double*)OpenMX_Solver_Real(solver_owner,"iko-8",((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]);

  matRS0=OpenMX_Solver_AlignedRealMatrix(solver_owner,"matRS0-5",((size_t)1)*(EKC_core_size[Mc_AN]+2),((size_t)1)*(Msize4[Mc_AN]+3));
  for (i=0; i<(EKC_core_size[Mc_AN]+2); i++){
    ;
  }

  matRS1=OpenMX_Solver_AlignedRealMatrix(solver_owner,"matRS1-6",((size_t)1)*(Msize4[Mc_AN]+3),((size_t)1)*(EKC_core_size[Mc_AN]+2));
  for (i=0; i<(Msize4[Mc_AN]+3); i++){
    ;
  }

  /****************************************************
      initialize 
  ****************************************************/

  if (measure_time==1){ 
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
  }

  if (measure_time==1) dtime(&Stime1);

  for (i=0; i<EKC_core_size_max; i++){
    for (j=0; j<Msize2_max; j++){
      tmpvec0[i][j] = 0.0;
      tmpvec1[i][j] = 0.0;
    }
  }

  /* find the nearest atom with distance of r0 */   

  r0 = 1.0e+10;
  for (k=1; k<=FNAN[ct_AN]; k++){
    Gh_AN = natn[ct_AN][k];
    wanB = WhatSpecies[Gh_AN];
    if (Dis[ct_AN][k]<r0)  r0 = Dis[ct_AN][k]; 
  }

  /* starting vector */

  m = 0;  
  for (k=0; k<=FNAN[ct_AN]; k++){

    Gh_AN = natn[ct_AN][k];
    wanB = WhatSpecies[Gh_AN];

    if ( Dis[ct_AN][k]<(scale_rc_EKC[Mc_AN]*r0) ){

      Anum = MP[k] - 1; 

      for (i=0; i<Spe_Total_CNO[wanB]; i++){

        tmpvec0[m][Anum+i] = 1.0;

        m++;  
      }
    }
  }

  S_orthonormalize_vec( Mc_AN, ct_on, tmpvec0, tmpvec1, OLP0, tmpmat0, ko, iko, MP, Msize2 );

  for (n=0; n<EKC_core_size[Mc_AN]; n++){
    for (i=0; i<Msize2[Mc_AN]; i++){
      U0[0][n][i] = tmpvec0[n][i];
    }
  }

  if (measure_time==1){ 
    dtime(&Etime1);
    time1 = Etime1 - Stime1;      
  }

  /****************************************************
           generate Krylov subspace vectors
  ****************************************************/

  for (rl=0; rl<(rlmax_EC[Mc_AN]-1); rl++){

    if (measure_time==1) dtime(&Stime1);

    /*******************************************************
                            H * |Wn)
    *******************************************************/

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=0; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=0; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
        Bnum = MP[j] - 1;

        if (0<=kl){

#ifdef nosse

      /* original version */
	  	  
      	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += Hks[spin][ih][kl][m][k]*tmpvec0[n][Bnum+k];
	      }

              tmpvec1[n][Anum+m] += sum;
	    }
	  }
	  
#else
      /* Loop Unrolling + SSE version */
	  
	  for (m=0; m<(ian-3); m+=4){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      mmSum00 = _mm_setzero_pd();
	      mmSum01 = _mm_setzero_pd();
	      mmSum10 = _mm_setzero_pd();
	      mmSum11 = _mm_setzero_pd();
	      mmSum20 = _mm_setzero_pd();
	      mmSum21 = _mm_setzero_pd();
	      mmSum30 = _mm_setzero_pd();
	      mmSum31 = _mm_setzero_pd();

	      for (k=0; k<(jan-3); k+=4){
		mmTmp0 = _mm_loadu_pd(&tmpvec0[n][Bnum+k+0]);
		mmTmp1 = _mm_loadu_pd(&tmpvec0[n][Bnum+k+2]);

		mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+0][k+0]),mmTmp0));  
		mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+0][k+2]),mmTmp1));  

		mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+1][k+0]),mmTmp0));  
		mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+1][k+2]),mmTmp1));  

		mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+2][k+0]),mmTmp0));  
		mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+2][k+2]),mmTmp1));  

		mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+3][k+0]),mmTmp0));  
		mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+3][k+2]),mmTmp1));  
	      }

	      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	      _mm_storeu_pd(&mmArr[0], mmSum00);
	      _mm_storeu_pd(&mmArr[2], mmSum10);
	      _mm_storeu_pd(&mmArr[4], mmSum20);
	      _mm_storeu_pd(&mmArr[6], mmSum30);

	      sum0 = mmArr[0] + mmArr[1];
	      sum1 = mmArr[2] + mmArr[3];
	      sum2 = mmArr[4] + mmArr[5];
	      sum3 = mmArr[6] + mmArr[7];

	      for (; k<jan; k++){
		sum0 += Hks[spin][ih][kl][m+0][k]*tmpvec0[n][Bnum+k];
		sum1 += Hks[spin][ih][kl][m+1][k]*tmpvec0[n][Bnum+k];
		sum2 += Hks[spin][ih][kl][m+2][k]*tmpvec0[n][Bnum+k];
		sum3 += Hks[spin][ih][kl][m+3][k]*tmpvec0[n][Bnum+k];
	      }

              tmpvec1[n][Anum+m+0] += sum0;
              tmpvec1[n][Anum+m+1] += sum1;
              tmpvec1[n][Anum+m+2] += sum2;
              tmpvec1[n][Anum+m+3] += sum3;
	    }
	  }

	  for (; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += Hks[spin][ih][kl][m][k]*tmpvec0[n][Bnum+k];
	      }

              tmpvec1[n][Anum+m] += sum;
	    }
	  } 
#endif	   
	}
      }
    }

    if (measure_time==1){ 
      dtime(&Etime1);
      time2 += Etime1 - Stime1;      
    }

    /*******************************************************
           S^{-1} * H * |Wn)
    *******************************************************/

    if (EKC_Exact_invS_flag==1){ 

      if (measure_time==1) dtime(&Stime1);

#ifdef nosse

      /* original version */
      /*      
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
        for (i=0; i<Msize2[Mc_AN]; i++){
	  sum = 0.0;
	  for (j=0; j<Msize2[Mc_AN]; j++){
	    sum += invS[i][j]*tmpvec1[n][j];
	  }
	  tmpvec0[n][i] = sum;
        } 
      }
      */
      /* unrolling version */

      for (i=0; i<(Msize2[Mc_AN]-3); i+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  sum0 = 0.0;
	  sum1 = 0.0;
	  sum2 = 0.0;
	  sum3 = 0.0;

	  for (j=0; j<Msize2[Mc_AN]; j++){
	    sum0 += invS[i+0][j]*tmpvec1[n][j];
	    sum1 += invS[i+1][j]*tmpvec1[n][j];
	    sum2 += invS[i+2][j]*tmpvec1[n][j];
	    sum3 += invS[i+3][j]*tmpvec1[n][j];
	  }

	  tmpvec0[n][i+0] = sum0;
	  tmpvec0[n][i+1] = sum1;
	  tmpvec0[n][i+2] = sum2;
	  tmpvec0[n][i+3] = sum3;
        } 
      }

      is = Msize2[Mc_AN] - Msize2[Mc_AN]%4;

      for (i=is; i<Msize2[Mc_AN]; i++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  sum = 0.0;
	  for (j=0; j<Msize2[Mc_AN]; j++){
	    sum += invS[i][j]*tmpvec1[n][j];
	  }
	  tmpvec0[n][i] = sum;
        } 
      }

#else
      /* unrolling + SSE version */
      
      for (i=0; i<(Msize2[Mc_AN]-3); i+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  mmSum00 = _mm_setzero_pd();
	  mmSum01 = _mm_setzero_pd();
	  mmSum10 = _mm_setzero_pd();
	  mmSum11 = _mm_setzero_pd();
	  mmSum20 = _mm_setzero_pd();
	  mmSum21 = _mm_setzero_pd();
	  mmSum30 = _mm_setzero_pd();
	  mmSum31 = _mm_setzero_pd();

	  for (j=0; j<(Msize2[Mc_AN]-3); j+=4){
	    mmTmp0 = _mm_loadu_pd(&tmpvec1[n][j+0]);
	    mmTmp1 = _mm_loadu_pd(&tmpvec1[n][j+2]);

	    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&invS[i+0][j+0]),mmTmp0));  
	    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&invS[i+0][j+2]),mmTmp1));  

	    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&invS[i+1][j+0]),mmTmp0));  
	    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&invS[i+1][j+2]),mmTmp1));  

	    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&invS[i+2][j+0]),mmTmp0));  
	    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&invS[i+2][j+2]),mmTmp1));  

	    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&invS[i+3][j+0]),mmTmp0));  
	    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&invS[i+3][j+2]),mmTmp1));  
	  }

	  mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	  mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	  mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	  mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	  _mm_storeu_pd(&mmArr[0], mmSum00);
	  _mm_storeu_pd(&mmArr[2], mmSum10);
	  _mm_storeu_pd(&mmArr[4], mmSum20);
	  _mm_storeu_pd(&mmArr[6], mmSum30);

	  sum0 = mmArr[0] + mmArr[1];
	  sum1 = mmArr[2] + mmArr[3];
	  sum2 = mmArr[4] + mmArr[5];
	  sum3 = mmArr[6] + mmArr[7];

	  for (; j<Msize2[Mc_AN]; j++){
	    sum0 += invS[i+0][j]*tmpvec1[n][j];
	    sum1 += invS[i+1][j]*tmpvec1[n][j];
	    sum2 += invS[i+2][j]*tmpvec1[n][j];
	    sum3 += invS[i+3][j]*tmpvec1[n][j];
	  }

	  tmpvec0[n][i+0] = sum0;
	  tmpvec0[n][i+1] = sum1;
	  tmpvec0[n][i+2] = sum2;
	  tmpvec0[n][i+3] = sum3;
        } 
      }
                  
      is = Msize2[Mc_AN] - Msize2[Mc_AN]%4;

      for (i=is; i<Msize2[Mc_AN]; i++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  sum = 0.0;
	  for (j=0; j<Msize2[Mc_AN]; j++){
	    sum += invS[i][j]*tmpvec1[n][j];
	  }
	  tmpvec0[n][i] = sum;
        } 
      }
      
#endif      

      if (measure_time==1){ 
	dtime(&Etime1);
	time3 += Etime1 - Stime1;      
      }
    }

    /*******************************************************
          U * RS^-1 * U^+ * H * |Wn)
    *******************************************************/

    else if (EKC_invS_flag==1){ 

      /*  U^+ * H * |Wn) */

      for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
	for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  for (m=0; m<EKC_core_size[Mc_AN]; m++){

	    sum = 0.0;
	    for (i=0; i<Msize2[Mc_AN]; i++){
	      sum += Krylov_U_OLP[rl0][n][i]*tmpvec1[m][i];            
	    }

	    /* transpose the later calcualtion */
	    matRS0[m][rl0*EKC_core_size[Mc_AN]+n] = sum;
	  }
	}  
      }

      /*  RS^-1 * U^+ * H * |Wn) */

      for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
	for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  for (m=0; m<EKC_core_size[Mc_AN]; m++){

	    sum = 0.0;
	    for (i=0; i<Msize4[Mc_AN]; i++){
	      sum += inv_RS[rl0*EKC_core_size[Mc_AN]+n][i]*matRS0[m][i];
	    }

	    matRS1[rl0*EKC_core_size[Mc_AN]+n][m] = sum;
	  }
	}
      }         

      /*  U * RS^-1 * U^+ * H * |Wn) */

      for (n=0; n<EKC_core_size[Mc_AN]; n++){
	for (i=0; i<Msize2[Mc_AN]; i++){
	  tmpvec0[n][i] = 0.0;
	}
      }

      for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
	for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  for (m=0; m<EKC_core_size[Mc_AN]; m++){
	    tmp0 = matRS1[rl0*EKC_core_size[Mc_AN]+n][m];
	    for (i=0; i<Msize2[Mc_AN]; i++){
	      tmpvec0[m][i] += Krylov_U_OLP[rl0][n][i]*tmp0;
	    }
	  }
	}
      }
    }

    else {
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
        for (i=0; i<Msize2[Mc_AN]; i++){
	  tmpvec0[n][i]  = tmpvec1[n][i];
        }
      }
    }

    if (measure_time==1) dtime(&Stime1);

    /*************************************************************
     S-orthogonalization by a classical block Gram-Schmidt method 
    *************************************************************/

    /* |tmpvec2) = S * |tmpvec0) */

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec2[n][i] = 0.0;
      }
    }

    for (i=0; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=0; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j] - 1;

	if (0<=kl){

#ifdef nosse

	  /* Original version */
	  	    
	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*tmpvec0[n][Bnum+k];
	      }

	      tmpvec2[n][Anum+m] += sum;
	    }
	  }
	  
#else

	  /* Unrolling + SSE version */
	  
	  for (m=0; m<(ian-3); m+=4){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      mmSum00 = _mm_setzero_pd();
	      mmSum01 = _mm_setzero_pd();
	      mmSum10 = _mm_setzero_pd();
	      mmSum11 = _mm_setzero_pd();
	      mmSum20 = _mm_setzero_pd();
	      mmSum21 = _mm_setzero_pd();
	      mmSum30 = _mm_setzero_pd();
	      mmSum31 = _mm_setzero_pd();

	      for (k=0; k<(jan-3); k+=4){
		mmTmp0 = _mm_loadu_pd(&tmpvec0[n][Bnum+k+0]);
		mmTmp1 = _mm_loadu_pd(&tmpvec0[n][Bnum+k+2]);

		mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+0]),mmTmp0));  
		mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+2]),mmTmp1));  

		mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+0]),mmTmp0));  
		mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+2]),mmTmp1));  

		mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+0]),mmTmp0));  
		mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+2]),mmTmp1));  

		mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+0]),mmTmp0));  
		mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+2]),mmTmp1));  
	      }

	      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	      _mm_storeu_pd(&mmArr[0], mmSum00);
	      _mm_storeu_pd(&mmArr[2], mmSum10);
	      _mm_storeu_pd(&mmArr[4], mmSum20);
	      _mm_storeu_pd(&mmArr[6], mmSum30);

	      sum0 = mmArr[0] + mmArr[1];
	      sum1 = mmArr[2] + mmArr[3];
	      sum2 = mmArr[4] + mmArr[5];
	      sum3 = mmArr[6] + mmArr[7];

	      for (; k<jan; k++){
		sum0 += OLP0[ih][kl][m+0][k]*tmpvec0[n][Bnum+k];
		sum1 += OLP0[ih][kl][m+1][k]*tmpvec0[n][Bnum+k];
		sum2 += OLP0[ih][kl][m+2][k]*tmpvec0[n][Bnum+k];
		sum3 += OLP0[ih][kl][m+3][k]*tmpvec0[n][Bnum+k];
	      }

	      tmpvec2[n][Anum+m+0] += sum0;
	      tmpvec2[n][Anum+m+1] += sum1;
	      tmpvec2[n][Anum+m+2] += sum2;
	      tmpvec2[n][Anum+m+3] += sum3;
	    }
	  }

	  for (; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*tmpvec0[n][Bnum+k];
	      }

	      tmpvec2[n][Anum+m] += sum;
	    }
	  } 
	  
#endif

	}
      }
    }

    if (measure_time==1){ 
      dtime(&Etime1);
      time4 += Etime1 - Stime1;      
    }

    if (measure_time==1) dtime(&Stime1);

#ifdef nosse

    /* Original version */
        
    for (rl0=0; rl0<=rl; rl0++){

      /* (U_rl0|tmpvec2) */
        
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
          sum = 0.0;
  	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += U0[rl0][m][i]*tmpvec2[n][i];
	  }
          tmpmat0[m][n] = sum;
	}
      }
    
      /* |tmpvec0) - |U_rl0) * (U_rl0|tmpvec2) */
        
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
	for (k=0; k<EKC_core_size[Mc_AN]; k++){
	  dum = tmpmat0[k][n];
	  for (i=0; i<Msize2[Mc_AN]; i++)  tmpvec0[n][i] -= U0[rl0][k][i]*dum;
        }
      }

    }
    
#else

    /* Unrolling + SSE version */
    
    for (rl0=0; rl0<=rl; rl0++){

      /* (U_rl0|tmpvec2) */
    
      for (m=0; m<(EKC_core_size[Mc_AN]-3); m+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  mmSum00 = _mm_setzero_pd();
	  mmSum01 = _mm_setzero_pd();
	  mmSum10 = _mm_setzero_pd();
	  mmSum11 = _mm_setzero_pd();
	  mmSum20 = _mm_setzero_pd();
	  mmSum21 = _mm_setzero_pd();
	  mmSum30 = _mm_setzero_pd();
	  mmSum31 = _mm_setzero_pd();

  	  for (i=0; i<(Msize2[Mc_AN]-3); i+=4){
	    mmTmp0 = _mm_loadu_pd(&tmpvec2[n][i+0]);
	    mmTmp1 = _mm_loadu_pd(&tmpvec2[n][i+2]);

	    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+0][i+0]),mmTmp0));  
	    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+0][i+2]),mmTmp1));  

	    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+1][i+0]),mmTmp0));  
	    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+1][i+2]),mmTmp1));  

	    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+2][i+0]),mmTmp0));  
	    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+2][i+2]),mmTmp1));  

	    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+3][i+0]),mmTmp0));  
	    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+3][i+2]),mmTmp1));  
	  }

	  mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	  mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	  mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	  mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	  _mm_storeu_pd(&mmArr[0], mmSum00);
	  _mm_storeu_pd(&mmArr[2], mmSum10);
	  _mm_storeu_pd(&mmArr[4], mmSum20);
	  _mm_storeu_pd(&mmArr[6], mmSum30);

	  sum0 = mmArr[0] + mmArr[1];
	  sum1 = mmArr[2] + mmArr[3];
	  sum2 = mmArr[4] + mmArr[5];
	  sum3 = mmArr[6] + mmArr[7];

  	  for (; i<Msize2[Mc_AN]; i++){
            sum0 += U0[rl0][m+0][i]*tmpvec2[n][i];
            sum1 += U0[rl0][m+1][i]*tmpvec2[n][i];
            sum2 += U0[rl0][m+2][i]*tmpvec2[n][i];
            sum3 += U0[rl0][m+3][i]*tmpvec2[n][i];
	  }

          tmpmat0[m+0][n] = sum0;
          tmpmat0[m+1][n] = sum1;
          tmpmat0[m+2][n] = sum2;
          tmpmat0[m+3][n] = sum3;
	}
      }

      for (; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
          sum = 0.0;
  	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += U0[rl0][m][i]*tmpvec2[n][i];
	  }
          tmpmat0[m][n] = sum;
	}
      }

      /* |tmpvec0) - |U_rl0) * (U_rl0|tmpvec2) */
    
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
	for (k=0; k<EKC_core_size[Mc_AN]; k++){
	  dum = tmpmat0[k][n];
	  for (i=0; i<Msize2[Mc_AN]; i++)  tmpvec0[n][i] -= U0[rl0][k][i]*dum;
        }
      }

    }
    
#endif

    if (measure_time==1){ 
      dtime(&Etime1);
      time5 += Etime1 - Stime1;      
    }

    /*************************************************************
                   S-orthonormalization of tmpvec0
    *************************************************************/

    if (measure_time==1) dtime(&Stime1);

    S_orthonormalize_vec( Mc_AN, ct_on, tmpvec0, tmpvec1, OLP0, tmpmat0, ko, iko, MP, Msize2 );

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
        U0[rl+1][n][i] = tmpvec0[n][i];
      }
    }

    if (measure_time==1){ 
      dtime(&Etime1);
      time6 += Etime1 - Stime1;      
    }

  } /* rl */

  /************************************************************
              orthogonalization by diagonalization
  ************************************************************/

  if (measure_time==1) dtime(&Stime1);

  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

    /*  S * |Vn) */

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=0; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=0; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j] - 1;

	if (0<=kl){

#ifdef nosse

	  /* Original version */
	  /**/	  
	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*U0[rl][n][Bnum+k];
	      }
	      tmpvec1[n][Anum+m] += sum;
	    }
	  } 
	  /**/

#else

	  /* Unrolling + SSE version */
	  /**/
	  for (m=0; m<(ian-3); m+=4){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      mmSum00 = _mm_setzero_pd();
	      mmSum01 = _mm_setzero_pd();
	      mmSum10 = _mm_setzero_pd();
	      mmSum11 = _mm_setzero_pd();
	      mmSum20 = _mm_setzero_pd();
	      mmSum21 = _mm_setzero_pd();
	      mmSum30 = _mm_setzero_pd();
	      mmSum31 = _mm_setzero_pd();

	      for (k=0; k<(jan-3); k+=4){
		mmTmp0 = _mm_loadu_pd(&U0[rl][n][Bnum+k+0]);
		mmTmp1 = _mm_loadu_pd(&U0[rl][n][Bnum+k+2]);

		mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+0]),mmTmp0));  
		mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+2]),mmTmp1));  

		mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+0]),mmTmp0));  
		mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+2]),mmTmp1));  

		mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+0]),mmTmp0));  
		mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+2]),mmTmp1));  

		mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+0]),mmTmp0));  
		mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+2]),mmTmp1));  
	      }

	      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	      _mm_storeu_pd(&mmArr[0], mmSum00);
	      _mm_storeu_pd(&mmArr[2], mmSum10);
	      _mm_storeu_pd(&mmArr[4], mmSum20);
	      _mm_storeu_pd(&mmArr[6], mmSum30);

	      sum0 = mmArr[0] + mmArr[1];
	      sum1 = mmArr[2] + mmArr[3];
	      sum2 = mmArr[4] + mmArr[5];
	      sum3 = mmArr[6] + mmArr[7];

	      for (; k<jan; k++){
		sum0 += OLP0[ih][kl][m+0][k]*U0[rl][n][Bnum+k];
		sum1 += OLP0[ih][kl][m+1][k]*U0[rl][n][Bnum+k];
		sum2 += OLP0[ih][kl][m+2][k]*U0[rl][n][Bnum+k];
		sum3 += OLP0[ih][kl][m+3][k]*U0[rl][n][Bnum+k];
	      }

	      tmpvec1[n][Anum+m+0] += sum0;
	      tmpvec1[n][Anum+m+1] += sum1;
	      tmpvec1[n][Anum+m+2] += sum2;
	      tmpvec1[n][Anum+m+3] += sum3;
	    }
	  } 

	  for (; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*U0[rl][n][Bnum+k];
	      }
	      tmpvec1[n][Anum+m] += sum;
	    }
	  } 
	  /**/
#endif

	}
      }
    }

#ifdef nosse

    /* Original version */
    /**/    
    for (rl0=rl; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += U0[rl0][m][i]*tmpvec1[n][i]; 
	  }
          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum;
	}
      }
    }
    /**/

#else

    /* Unrolling + SSE version */
    /**/
    for (rl0=rl; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<(EKC_core_size[Mc_AN]-3); m+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  mmSum00 = _mm_setzero_pd();
	  mmSum01 = _mm_setzero_pd();
	  mmSum10 = _mm_setzero_pd();
	  mmSum11 = _mm_setzero_pd();
	  mmSum20 = _mm_setzero_pd();
	  mmSum21 = _mm_setzero_pd();
	  mmSum30 = _mm_setzero_pd();
	  mmSum31 = _mm_setzero_pd();

	  for (i=0; i<(Msize2[Mc_AN]-3); i+=4){
	    mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i+0]);
	    mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i+2]);

	    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+0][i+0]),mmTmp0));  
	    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+0][i+2]),mmTmp1));  

	    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+1][i+0]),mmTmp0));  
	    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+1][i+2]),mmTmp1));  

	    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+2][i+0]),mmTmp0));  
	    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+2][i+2]),mmTmp1));  

	    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+3][i+0]),mmTmp0));  
	    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+3][i+2]),mmTmp1));  
	  }

	  mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	  mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	  mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	  mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	  _mm_storeu_pd(&mmArr[0], mmSum00);
	  _mm_storeu_pd(&mmArr[2], mmSum10);
	  _mm_storeu_pd(&mmArr[4], mmSum20);
	  _mm_storeu_pd(&mmArr[6], mmSum30);

	  sum0 = mmArr[0] + mmArr[1];
	  sum1 = mmArr[2] + mmArr[3];
	  sum2 = mmArr[4] + mmArr[5];
	  sum3 = mmArr[6] + mmArr[7];

	  for (; i<Msize2[Mc_AN]; i++){
            sum0 += U0[rl0][m+0][i]*tmpvec1[n][i]; 
            sum1 += U0[rl0][m+1][i]*tmpvec1[n][i]; 
            sum2 += U0[rl0][m+2][i]*tmpvec1[n][i]; 
            sum3 += U0[rl0][m+3][i]*tmpvec1[n][i]; 
	  }

          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum0;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum0;

          FS[rl0*EKC_core_size[Mc_AN]+m+2][rl*EKC_core_size[Mc_AN]+n+1] = sum1;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+2] = sum1;

          FS[rl0*EKC_core_size[Mc_AN]+m+3][rl*EKC_core_size[Mc_AN]+n+1] = sum2;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+3] = sum2;

          FS[rl0*EKC_core_size[Mc_AN]+m+4][rl*EKC_core_size[Mc_AN]+n+1] = sum3;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+4] = sum3;
	}
      }

      for (; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += U0[rl0][m][i]*tmpvec1[n][i]; 
	  }
          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum;
	}
      }

    }
    /**/

#endif

  }

  if (measure_time==1){ 
    dtime(&Etime1);
    time7 += Etime1 - Stime1;      
  }

  if (measure_time==1) dtime(&Stime1);

  Eigen_lapack(FS,ko,Msize3[Mc_AN],Msize3[Mc_AN]);

  if (measure_time==1){ 
    dtime(&Etime1);
    time8 += Etime1 - Stime1;      
  }

  ZeroNum = 0;

  for (i=1; i<=Msize3[Mc_AN]; i++){

    if (error_check==1){
      printf("spin=%2d Mc_AN=%2d i=%3d ko[i]=%18.15f\n",spin,Mc_AN,i,ko[i]);
    }

    if (cutoff_value<ko[i]){
      ko[i]  = sqrt(fabs(ko[i]));
      iko[i] = 1.0/ko[i];
    }
    else{
      ZeroNum++;
      ko[i]  = 0.0;
      iko[i] = 0.0;
    }
  }  

  if (error_check==1){
    printf("spin=%2d Mc_AN=%2d ZeroNum=%2d\n",spin,Mc_AN,ZeroNum);
  }

  for (i=1; i<=Msize3[Mc_AN]; i++){
    for (j=1; j<=Msize3[Mc_AN]; j++){
      FS[i][j] = FS[i][j]*iko[j];
    }
  }

  /* transpose for later calculation */
  for (i=1; i<=Msize3[Mc_AN]; i++){
    for (j=i+1; j<=Msize3[Mc_AN]; j++){
      tmp1 = FS[i][j];
      tmp2 = FS[j][i];
      FS[i][j] = tmp2;
      FS[j][i] = tmp1;
    }
  }

  if (measure_time==1) dtime(&Stime1);

  /* U0 * U * lamda^{-1/2} */ 

#ifdef nosse

  /* original version */
  /*
  for (i=0; i<Msize2[Mc_AN]; i++){
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){

        m1 = rl0*EKC_core_size[Mc_AN] + m + 1;

	sum = 0.0; 
	for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

          n1 = rl*EKC_core_size[Mc_AN] + 1;

	  for (n=0; n<EKC_core_size[Mc_AN]; n++){
	    sum += U0[rl][n][i]*FS[m1][n1+n]; 
	  }
	}

        Utmp[rl0][m] = sum;        
      }
    }

    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        U0[rl0][m][i] = Utmp[rl0][m];
      }
    }
  }
  */

  /* unrolling version */

  for (i=0; i<Msize2[Mc_AN]; i++){

    for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
        Utmp[rl][n] = U0[rl][n][i];
      }
    }

    for (m1=1; m1<=(rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]-3); m1+=4){

      rl00 = (m1+0-1)/EKC_core_size[Mc_AN];
      rl01 = (m1+1-1)/EKC_core_size[Mc_AN];
      rl02 = (m1+2-1)/EKC_core_size[Mc_AN];
      rl03 = (m1+3-1)/EKC_core_size[Mc_AN];

      mm0 = (m1+0-1)%EKC_core_size[Mc_AN];
      mm1 = (m1+1-1)%EKC_core_size[Mc_AN];
      mm2 = (m1+2-1)%EKC_core_size[Mc_AN];
      mm3 = (m1+3-1)%EKC_core_size[Mc_AN];

      sum0 = 0.0; 
      sum1 = 0.0; 
      sum2 = 0.0; 
      sum3 = 0.0; 

      for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

	n1 = rl*EKC_core_size[Mc_AN] + 1;

	for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  sum0 += Utmp[rl][n]*FS[m1+0][n1+n]; 
	  sum1 += Utmp[rl][n]*FS[m1+1][n1+n]; 
	  sum2 += Utmp[rl][n]*FS[m1+2][n1+n]; 
	  sum3 += Utmp[rl][n]*FS[m1+3][n1+n]; 
	}
      }

      U0[rl00][mm0][i] = sum0;        
      U0[rl01][mm1][i] = sum1;        
      U0[rl02][mm2][i] = sum2;        
      U0[rl03][mm3][i] = sum3;        
    }

    m1s = rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN])%4 + 1;

    for (m1=m1s; m1<=rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]; m1++){

      rl0 = (m1-1)/EKC_core_size[Mc_AN];
      m = (m1-1)%EKC_core_size[Mc_AN];

      sum = 0.0; 

      for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

	n1 = rl*EKC_core_size[Mc_AN] + 1;

	for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  sum += Utmp[rl][n]*FS[m1][n1+n]; 
	}
      }

      U0[rl0][m][i] = sum;        
    }
  } /* i */

#else

  /* Unrolling + SSE version */
  /**/

  for (i=0; i<Msize2[Mc_AN]; i++){

    for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
        Utmp[rl][n] = U0[rl][n][i];
      }
    }

    for (m1=1; m1<=(rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]-3); m1+=4){

      rl00 = (m1+0-1)/EKC_core_size[Mc_AN];
      rl01 = (m1+1-1)/EKC_core_size[Mc_AN];
      rl02 = (m1+2-1)/EKC_core_size[Mc_AN];
      rl03 = (m1+3-1)/EKC_core_size[Mc_AN];

      mm0 = (m1+0-1)%EKC_core_size[Mc_AN];
      mm1 = (m1+1-1)%EKC_core_size[Mc_AN];
      mm2 = (m1+2-1)%EKC_core_size[Mc_AN];
      mm3 = (m1+3-1)%EKC_core_size[Mc_AN];

      sum0 = 0.0; 
      sum1 = 0.0; 
      sum2 = 0.0; 
      sum3 = 0.0; 

      mmSum00 = _mm_setzero_pd();
      mmSum01 = _mm_setzero_pd();
      mmSum10 = _mm_setzero_pd();
      mmSum11 = _mm_setzero_pd();
      mmSum20 = _mm_setzero_pd();
      mmSum21 = _mm_setzero_pd();
      mmSum30 = _mm_setzero_pd();
      mmSum31 = _mm_setzero_pd();

      for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

	n1 = rl*EKC_core_size[Mc_AN] + 1;

	if (0){
	mmTmp0 = _mm_loadu_pd(&Utmp[rl][n+0]);
	mmTmp1 = _mm_loadu_pd(&Utmp[rl][n+2]);
	}

	for (n=0; n<(EKC_core_size[Mc_AN]-3); n+=4){
	  mmTmp0 = _mm_loadu_pd(&Utmp[rl][n+0]);
	  mmTmp1 = _mm_loadu_pd(&Utmp[rl][n+2]);
	  
	  mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&FS[m1+0][n1+n+0]),mmTmp0));  
	  mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&FS[m1+0][n1+n+2]),mmTmp1));  

	  mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&FS[m1+1][n1+n+0]),mmTmp0));  
	  mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&FS[m1+1][n1+n+2]),mmTmp1));  

	  mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&FS[m1+2][n1+n+0]),mmTmp0));  
	  mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&FS[m1+2][n1+n+2]),mmTmp1));  

	  mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&FS[m1+3][n1+n+0]),mmTmp0));  
	  mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&FS[m1+3][n1+n+2]),mmTmp1));  
	}

	for (; n<EKC_core_size[Mc_AN]; n++){
	  sum0 += Utmp[rl][n]*FS[m1+0][n1+n]; 
	  sum1 += Utmp[rl][n]*FS[m1+1][n1+n]; 
	  sum2 += Utmp[rl][n]*FS[m1+2][n1+n]; 
	  sum3 += Utmp[rl][n]*FS[m1+3][n1+n]; 
	}

      }
      
      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
      _mm_storeu_pd(&mmArr[0], mmSum00);
      _mm_storeu_pd(&mmArr[2], mmSum10);
      _mm_storeu_pd(&mmArr[4], mmSum20);
      _mm_storeu_pd(&mmArr[6], mmSum30);

      sum0 += mmArr[0] + mmArr[1];
      sum1 += mmArr[2] + mmArr[3];
      sum2 += mmArr[4] + mmArr[5];
      sum3 += mmArr[6] + mmArr[7];

      U0[rl00][mm0][i] = sum0;        
      U0[rl01][mm1][i] = sum1;        
      U0[rl02][mm2][i] = sum2;        
      U0[rl03][mm3][i] = sum3;        
    }

    m1s = rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN])%4 + 1;

    for (m1=m1s; m1<=rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]; m1++){

      rl0 = (m1-1)/EKC_core_size[Mc_AN];
      m = (m1-1)%EKC_core_size[Mc_AN];

      sum = 0.0; 

      for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

	n1 = rl*EKC_core_size[Mc_AN] + 1;

	for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  sum += Utmp[rl][n]*FS[m1][n1+n]; 
	}
      }

      U0[rl0][m][i] = sum;        
    }
  } /* i */

#endif

  Krylov_U[spin][Mc_AN][0] = ZeroNum;

  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
        Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+i+1] = U0[rl][n][i];
      }
    }
  }

  if (measure_time==1){ 
    dtime(&Etime1);
    time9 += Etime1 - Stime1;      
  }

  /************************************************************
         check the orthonormality of Krylov vectors
  ************************************************************/

  if (error_check==1){

    for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

      for (n=0; n<EKC_core_size[Mc_AN]; n++){
	for (i=0; i<Msize2[Mc_AN]; i++){
	  tmpvec1[n][i]  = 0.0;
	}
      }

      for (i=0; i<=can; i++){

	ig = natn[ct_AN][i];
	ian = Spe_Total_CNO[WhatSpecies[ig]];
	Anum = MP[i] - 1;
	ih = S_G2M[ig];

	for (j=0; j<=can; j++){

	  kl = RMI1[Mc_AN][i][j];
	  jg = natn[ct_AN][j];
	  jan = Spe_Total_CNO[WhatSpecies[jg]];
	  Bnum = MP[j] - 1;

	  if (0<=kl){

	    for (m=0; m<ian; m++){
	      for (n=0; n<EKC_core_size[Mc_AN]; n++){

		sum = 0.0;
		for (k=0; k<jan; k++){
		  sum += OLP0[ih][kl][m][k]*U0[rl][n][Bnum+k];
		}

		tmpvec1[n][Anum+m] += sum;
	      }
	    } 
	  }
	}
      }

      for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
	for (m=0; m<EKC_core_size[Mc_AN]; m++){
	  for (n=0; n<EKC_core_size[Mc_AN]; n++){
	    sum = 0.0;
	    for (i=0; i<Msize2[Mc_AN]; i++){
	      sum += U0[rl0][m][i]*tmpvec1[n][i]; 
	    }

	    if (rl==rl0 && m==n){
	      if ( 1.0e-10<fabs(sum-1.0) ) {
		printf("A spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
		       spin,Mc_AN,rl,rl0,m,n,sum);
	      }
	    }
	    else{
	      if ( 1.0e-10<fabs(sum) ) {
		printf("B spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
		       spin,Mc_AN,rl,rl0,m,n,sum);
	      }
	    }

	  }
	}
      }
    }
  }

  if (measure_time==1){ 
    printf("pMatrix myid=%2d time1 =%5.3f time2 =%5.3f time3 =%5.3f time4 =%5.3f\n",
            myid,time1,time2,time3,time4);
    printf("pMatrix myid=%2d time5 =%5.3f time6 =%5.3f time7 =%5.3f time8 =%5.3f\n",
            myid,time5,time6,time7,time8);
    printf("pMatrix myid=%2d time9 =%5.3f\n",myid,time9);
  }

  /* freeing of arrays */

  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
  }
  ;

  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    for (j=0; j<EKC_core_size[Mc_AN]; j++){
      ;
    }
    ;
  }  
  ;

  for (i=0; i<(EKC_core_size[Mc_AN]+4); i++){
    ;
  }
  ;

  for (i=0; i<(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]; i++){
    ;
  }
  ;

  ;
  ;

  for (i=0; i<(EKC_core_size[Mc_AN]+2); i++){
    ;
  }
  ;

  for (i=0; i<(Msize4[Mc_AN]+3); i++){
    ;
  }
  ;
}








typedef struct {
  int myid;
  int spin;
  int Mc_AN;
  double *****Hks;
  double ****OLP0;
  double ***Krylov_U;
  int *MP;
  int *Msize;
  int *Msize2;
  int *Msize3;
  double **tmpvec1;
} Generate_pMatrix2_scratch_arguments;
static void Generate_pMatrix2_raii( int myid, int spin, int Mc_AN, double *****Hks, double ****OLP0,
                        double ***Krylov_U, int *MP, int *Msize, int *Msize2, int *Msize3,
                        double **tmpvec1,OpenMXSolverInvocation *solver_owner);
static int Generate_pMatrix2_scratch_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Generate_pMatrix2_scratch_arguments *a=(Generate_pMatrix2_scratch_arguments*)arguments;
  Generate_pMatrix2_raii(a->myid,a->spin,a->Mc_AN,a->Hks,a->OLP0,a->Krylov_U,a->MP,a->Msize,a->Msize2,a->Msize3,a->tmpvec1,owner);
  *result=0;
  return 1;
}
void Generate_pMatrix2( int myid, int spin, int Mc_AN, double *****Hks, double ****OLP0, 
                        double ***Krylov_U, int *MP, int *Msize, int *Msize2, int *Msize3, 
                        double **tmpvec1)
{
  Generate_pMatrix2_scratch_arguments arguments={myid,spin,Mc_AN,Hks,OLP0,Krylov_U,MP,Msize,Msize2,Msize3,tmpvec1};
  OpenMX_Solver_Run(Generate_pMatrix2_scratch_operation,&arguments);
}

static void Generate_pMatrix2_raii( int myid, int spin, int Mc_AN, double *****Hks, double ****OLP0,
                        double ***Krylov_U, int *MP, int *Msize, int *Msize2, int *Msize3,
                        double **tmpvec1,OpenMXSolverInvocation *solver_owner)
{
  int rl,rl0,rl1,ct_AN,fan,san,can,wan,ct_on,i,j;
  int n,Anum,Bnum,k,ian,ih,kl,jg,ig,jan,m,m1,n1;
  int ZeroNum,rl_half;
  int KU_d1,KU_d2,csize;
  double sum,dum,tmp0,tmp1,tmp2,tmp3;
  double **Utmp;
  double *ko,*iko;
  double **FS;
  double ***U0;

  int rl00,rl01,rl02,rl03,rl04,rl05,rl06,rl07;
  int mm0,mm1,mm2,mm3,mm4,mm5,mm6,mm7,m1s;
  __m128d mmSum00,mmSum01,mmSum10,mmSum11,mmSum20,mmSum21,mmSum30,mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5; 

  double mmArr[8];
  double sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7;

  ct_AN = M2G[Mc_AN];
  fan = FNAN[ct_AN];
  san = SNAN[ct_AN];
  can = fan + san;
  wan = WhatSpecies[ct_AN];
  ct_on = Spe_Total_CNO[wan];

  if (Msize[Mc_AN]<Msize3[Mc_AN])
    csize = Msize3[Mc_AN] + 40;
  else
    csize = Msize[Mc_AN] + 40;

  KU_d1 = EKC_core_size[Mc_AN]*Msize2[Mc_AN];
  KU_d2 = Msize2[Mc_AN];

  /* allocation of arrays */

  Utmp=OpenMX_Solver_AlignedRealMatrix(solver_owner,"Utmp-1",((size_t)1)*rlmax_EC[Mc_AN],((size_t)1)*EKC_core_size[Mc_AN]);
  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
  }

  U0=OpenMX_Solver_AlignedRealVolume(solver_owner,"U0-2",((size_t)1)*rlmax_EC[Mc_AN],((size_t)1)*EKC_core_size[Mc_AN],((size_t)1)*(Msize2[Mc_AN]+3));
  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
    for (j=0; j<EKC_core_size[Mc_AN]; j++){
      ;
      for (k=0; k<(Msize2[Mc_AN]+3); k++)  U0[i][j][k] = 0.0;
    }
  }  

  FS=OpenMX_Solver_AlignedRealMatrix(solver_owner,"FS-3",((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN],((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  for (i=0; i<(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]; i++){
    ;
  }

  ko=(double*)OpenMX_Solver_Real(solver_owner,"ko-4",((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  iko=(double*)OpenMX_Solver_Real(solver_owner,"iko-5",((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]);

  /****************************************************
   initialize 
  ****************************************************/

  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
        U0[rl][n][i] = 0.0;
      }
    }
  }

  i = 0;
  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      if (i<Msize2[Mc_AN]) U0[rl][n][i] = 1.0;
      i++;
    }
  }

  /************************************************************
              orthogonalization by diagonalization
  ************************************************************/

  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

    /*  S * |Vn) */

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=0; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=0; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j] - 1;

	if (0<=kl){

#ifdef nosse

	  /* Original version */
	  /**/
	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*U0[rl][n][Bnum+k];
	      }
	      tmpvec1[n][Anum+m] += sum;
	    }
	  }
	  /**/

#else

	  /* Unrolling + SSE version */

	  for (m=0; m<(ian-3); m+=4){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      mmSum00 = _mm_setzero_pd();
	      mmSum01 = _mm_setzero_pd();
	      mmSum10 = _mm_setzero_pd();
	      mmSum11 = _mm_setzero_pd();
	      mmSum20 = _mm_setzero_pd();
	      mmSum21 = _mm_setzero_pd();
	      mmSum30 = _mm_setzero_pd();
	      mmSum31 = _mm_setzero_pd();

	      for (k=0; k<(jan-3); k+=4){
		mmTmp0 = _mm_loadu_pd(&U0[rl][n][Bnum+k+0]);
		mmTmp1 = _mm_loadu_pd(&U0[rl][n][Bnum+k+2]);

		mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+0]),mmTmp0));  
		mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+2]),mmTmp1));  

		mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+0]),mmTmp0));  
		mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+2]),mmTmp1));  

		mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+0]),mmTmp0));  
		mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+2]),mmTmp1));  

		mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+0]),mmTmp0));  
		mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+2]),mmTmp1));  
	      }

	      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	      _mm_storeu_pd(&mmArr[0], mmSum00);
	      _mm_storeu_pd(&mmArr[2], mmSum10);
	      _mm_storeu_pd(&mmArr[4], mmSum20);
	      _mm_storeu_pd(&mmArr[6], mmSum30);

	      sum0 = mmArr[0] + mmArr[1];
	      sum1 = mmArr[2] + mmArr[3];
	      sum2 = mmArr[4] + mmArr[5];
	      sum3 = mmArr[6] + mmArr[7];

	      for (; k<jan; k++){
		sum0 += OLP0[ih][kl][m+0][k]*U0[rl][n][Bnum+k];
		sum1 += OLP0[ih][kl][m+1][k]*U0[rl][n][Bnum+k];
		sum2 += OLP0[ih][kl][m+2][k]*U0[rl][n][Bnum+k];
		sum3 += OLP0[ih][kl][m+3][k]*U0[rl][n][Bnum+k];
	      }

	      tmpvec1[n][Anum+m+0] += sum0;
	      tmpvec1[n][Anum+m+1] += sum1;
	      tmpvec1[n][Anum+m+2] += sum2;
	      tmpvec1[n][Anum+m+3] += sum3;
	    }
	  } 

	  for (; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*U0[rl][n][Bnum+k];
	      }
	      tmpvec1[n][Anum+m] += sum;
	    }
	  } 

#endif

	}
      }
    }

#ifdef nosse

    /* Original version */
    /**/
    for (rl0=rl; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += U0[rl0][m][i]*tmpvec1[n][i]; 
	  }
          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum;
	}
      }
    }
    /**/

#else

    /* Unrolling + SSE version */

    for (rl0=rl; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<(EKC_core_size[Mc_AN]-3); m+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  mmSum00 = _mm_setzero_pd();
	  mmSum01 = _mm_setzero_pd();
	  mmSum10 = _mm_setzero_pd();
	  mmSum11 = _mm_setzero_pd();
	  mmSum20 = _mm_setzero_pd();
	  mmSum21 = _mm_setzero_pd();
	  mmSum30 = _mm_setzero_pd();
	  mmSum31 = _mm_setzero_pd();

	  for (i=0; i<(Msize2[Mc_AN]-3); i+=4){
	    mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i+0]);
	    mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i+2]);

	    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+0][i+0]),mmTmp0));  
	    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+0][i+2]),mmTmp1));  

	    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+1][i+0]),mmTmp0));  
	    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+1][i+2]),mmTmp1));  

	    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+2][i+0]),mmTmp0));  
	    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+2][i+2]),mmTmp1));  

	    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+3][i+0]),mmTmp0));  
	    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+3][i+2]),mmTmp1));  
	  }

	  mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	  mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	  mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	  mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	  _mm_storeu_pd(&mmArr[0], mmSum00);
	  _mm_storeu_pd(&mmArr[2], mmSum10);
	  _mm_storeu_pd(&mmArr[4], mmSum20);
	  _mm_storeu_pd(&mmArr[6], mmSum30);

	  sum0 = mmArr[0] + mmArr[1];
	  sum1 = mmArr[2] + mmArr[3];
	  sum2 = mmArr[4] + mmArr[5];
	  sum3 = mmArr[6] + mmArr[7];

	  for (; i<Msize2[Mc_AN]; i++){
            sum0 += U0[rl0][m+0][i]*tmpvec1[n][i]; 
            sum1 += U0[rl0][m+1][i]*tmpvec1[n][i]; 
            sum2 += U0[rl0][m+2][i]*tmpvec1[n][i]; 
            sum3 += U0[rl0][m+3][i]*tmpvec1[n][i]; 
	  }

          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum0;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum0;

          FS[rl0*EKC_core_size[Mc_AN]+m+2][rl*EKC_core_size[Mc_AN]+n+1] = sum1;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+2] = sum1;

          FS[rl0*EKC_core_size[Mc_AN]+m+3][rl*EKC_core_size[Mc_AN]+n+1] = sum2;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+3] = sum2;

          FS[rl0*EKC_core_size[Mc_AN]+m+4][rl*EKC_core_size[Mc_AN]+n+1] = sum3;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+4] = sum3;
	}
      }

      for (; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += U0[rl0][m][i]*tmpvec1[n][i]; 
	  }
          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum;
	}
      }
    }

#endif

  }

  Eigen_lapack(FS,ko,Msize3[Mc_AN],Msize3[Mc_AN]);
  ZeroNum = 0;

  for (i=1; i<=Msize3[Mc_AN]; i++){

    if (error_check==1){
      printf("spin=%2d Mc_AN=%2d i=%3d ko[i]=%18.15f\n",spin,Mc_AN,i,ko[i]);
    }

    if (cutoff_value<ko[i]){
      ko[i]  = sqrt(fabs(ko[i]));
      iko[i] = 1.0/ko[i];
    }
    else{
      ZeroNum++;
      ko[i]  = 0.0;
      iko[i] = 0.0;
    }
  }  

  if (error_check==1){
    printf("spin=%2d Mc_AN=%2d ZeroNum=%2d\n",spin,Mc_AN,ZeroNum);
  }

  for (i=1; i<=Msize3[Mc_AN]; i++){
    for (j=1; j<=Msize3[Mc_AN]; j++){
      FS[i][j] = FS[i][j]*iko[j];
    }
  }

  /* transpose for later calculation */
  for (i=1; i<=Msize3[Mc_AN]; i++){
    for (j=i+1; j<=Msize3[Mc_AN]; j++){
      tmp1 = FS[i][j];
      tmp2 = FS[j][i];
      FS[i][j] = tmp2;
      FS[j][i] = tmp1;
    }
  }

  /* U0 * U * lamda^{-1/2} */ 

  
  

  for (i=0; i<Msize2[Mc_AN]; i++){
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){

        m1 = rl0*EKC_core_size[Mc_AN] + m + 1;

	sum = 0.0; 
	for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

          n1 = rl*EKC_core_size[Mc_AN] + 1;

	  for (n=0; n<EKC_core_size[Mc_AN]; n++){
	    sum += U0[rl][n][i]*FS[m1][n1+n]; 
	  }
	}

        Utmp[rl0][m] = sum;        
      }
    }

    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        U0[rl0][m][i] = Utmp[rl0][m];
      }
    }
  }




  if (0){

#ifdef nosse

  /* original version */

  /**/
  for (i=0; i<Msize2[Mc_AN]; i++){
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){

        m1 = rl0*EKC_core_size[Mc_AN] + m + 1;

	sum = 0.0; 
	for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

          n1 = rl*EKC_core_size[Mc_AN] + 1;

	  for (n=0; n<EKC_core_size[Mc_AN]; n++){
	    sum += U0[rl][n][i]*FS[m1][n1+n]; 
	  }
	}

        Utmp[rl0][m] = sum;        
      }
    }

    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        U0[rl0][m][i] = Utmp[rl0][m];
      }
    }
  }
  /**/

#else

  /* Unrolling + SSE version */
  /**/
  for (i=0; i<Msize2[Mc_AN]; i++){

    for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
        Utmp[rl][n] = U0[rl][n][i];
      }
    }

    for (m1=1; m1<=(rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]-3); m1+=4){

      rl00 = (m1+0-1)/EKC_core_size[Mc_AN];
      rl01 = (m1+1-1)/EKC_core_size[Mc_AN];
      rl02 = (m1+2-1)/EKC_core_size[Mc_AN];
      rl03 = (m1+3-1)/EKC_core_size[Mc_AN];

      mm0 = (m1+0-1)%EKC_core_size[Mc_AN];
      mm1 = (m1+1-1)%EKC_core_size[Mc_AN];
      mm2 = (m1+2-1)%EKC_core_size[Mc_AN];
      mm3 = (m1+3-1)%EKC_core_size[Mc_AN];

      sum0 = 0.0; 
      sum1 = 0.0; 
      sum2 = 0.0; 
      sum3 = 0.0; 

      mmSum00 = _mm_setzero_pd();
      mmSum01 = _mm_setzero_pd();
      mmSum10 = _mm_setzero_pd();
      mmSum11 = _mm_setzero_pd();
      mmSum20 = _mm_setzero_pd();
      mmSum21 = _mm_setzero_pd();
      mmSum30 = _mm_setzero_pd();
      mmSum31 = _mm_setzero_pd();

      for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

	n1 = rl*EKC_core_size[Mc_AN] + 1;

	for (n=0; n<(EKC_core_size[Mc_AN]-3); n+=4){

	  mmTmp0 = _mm_loadu_pd(&Utmp[rl][n+0]);
	  mmTmp1 = _mm_loadu_pd(&Utmp[rl][n+2]);

	  mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&FS[m1+0][n1+n+0]),mmTmp0));  
	  mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&FS[m1+0][n1+n+2]),mmTmp1));  

	  mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&FS[m1+1][n1+n+0]),mmTmp0));  
	  mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&FS[m1+1][n1+n+2]),mmTmp1));  

	  mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&FS[m1+2][n1+n+0]),mmTmp0));  
	  mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&FS[m1+2][n1+n+2]),mmTmp1));  

	  mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&FS[m1+3][n1+n+0]),mmTmp0));  
	  mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&FS[m1+3][n1+n+2]),mmTmp1));  
	}

	for (; n<EKC_core_size[Mc_AN]; n++){
	  sum0 += Utmp[rl][n]*FS[m1+0][n1+n]; 
	  sum1 += Utmp[rl][n]*FS[m1+1][n1+n]; 
	  sum2 += Utmp[rl][n]*FS[m1+2][n1+n]; 
	  sum3 += Utmp[rl][n]*FS[m1+3][n1+n]; 
	}
      }

      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  

      _mm_storeu_pd(&mmArr[0], mmSum00);
      _mm_storeu_pd(&mmArr[2], mmSum10);
      _mm_storeu_pd(&mmArr[4], mmSum20);
      _mm_storeu_pd(&mmArr[6], mmSum30);

      sum0 += mmArr[0] + mmArr[1];
      sum1 += mmArr[2] + mmArr[3];
      sum2 += mmArr[4] + mmArr[5];
      sum3 += mmArr[6] + mmArr[7];

      U0[rl00][mm0][i] = sum0;        
      U0[rl01][mm1][i] = sum1;        
      U0[rl02][mm2][i] = sum2;        
      U0[rl03][mm3][i] = sum3;        
    }

    m1s = rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN])%4 + 1;

    for (m1=m1s; m1<=rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]; m1++){

      rl0 = (m1-1)/EKC_core_size[Mc_AN];
      m = (m1-1)%EKC_core_size[Mc_AN];

      sum = 0.0; 

      for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

	n1 = rl*EKC_core_size[Mc_AN] + 1;

	for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  sum += Utmp[rl][n]*FS[m1][n1+n]; 
	}
      }

      U0[rl0][m][i] = sum;        
    }
  } /* i */

#endif

  }






  Krylov_U[spin][Mc_AN][0] = ZeroNum;

  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
        Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+i+1] = U0[rl][n][i];
      }
    }
  }

  /************************************************************
         check the orthonormality of Krylov vectors
  ************************************************************/

  if (error_check==1){

    for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

      for (n=0; n<EKC_core_size[Mc_AN]; n++){
	for (i=0; i<Msize2[Mc_AN]; i++){
	  tmpvec1[n][i]  = 0.0;
	}
      }

      for (i=0; i<=can; i++){

	ig = natn[ct_AN][i];
	ian = Spe_Total_CNO[WhatSpecies[ig]];
	Anum = MP[i] - 1;
	ih = S_G2M[ig];

	for (j=0; j<=can; j++){

	  kl = RMI1[Mc_AN][i][j];
	  jg = natn[ct_AN][j];
	  jan = Spe_Total_CNO[WhatSpecies[jg]];
	  Bnum = MP[j] - 1;

	  if (0<=kl){

	    for (m=0; m<ian; m++){
	      for (n=0; n<EKC_core_size[Mc_AN]; n++){

		sum = 0.0;
		for (k=0; k<jan; k++){
		  sum += OLP0[ih][kl][m][k]*U0[rl][n][Bnum+k];
		}

		tmpvec1[n][Anum+m] += sum;
	      }
	    } 
	  }
	}
      }

      for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
	for (m=0; m<EKC_core_size[Mc_AN]; m++){
	  for (n=0; n<EKC_core_size[Mc_AN]; n++){
	    sum = 0.0;
	    for (i=0; i<Msize2[Mc_AN]; i++){
	      sum += U0[rl0][m][i]*tmpvec1[n][i]; 
	    }

	    if (rl==rl0 && m==n){
	      if ( 1.0e-10<fabs(sum-1.0) ) {
		printf("A spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
		       spin,Mc_AN,rl,rl0,m,n,sum);
	      }
	    }
	    else{
	      if ( 1.0e-10<fabs(sum) ) {
		printf("B spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
		       spin,Mc_AN,rl,rl0,m,n,sum);
	      }
	    }

	  }
	}
      }
    }
  }

  /* freeing of arrays */

  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
  }
  ;

  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    for (j=0; j<EKC_core_size[Mc_AN]; j++){
      ;
    }
    ;
  }  
  ;

  for (i=0; i<(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]; i++){
    ;
  }
  ;

  ;
  ;
}







void Embedding_Matrix(KrylovState *state, int spin, int Mc_AN, double *****Hks,
                      double ***Krylov_U, double ****EC_matrix, 
                      int *MP, int *Msize, int *Msize2, int *Msize3, 
                      double **tmpvec1 )
{
  int ct_AN,fan,san,can,wan,ct_on;
  int rl,rl0,m,n,i,j,k,kl,jg,jan,ih,ian;
  int Anum,Bnum,ig;
  int KU_d1,KU_d2,csize;
  double sum,tmp1,tmp2,tmp3;
  __m128d mmSum00,mmSum01,mmSum10,mmSum11,mmSum20,mmSum21,mmSum30,mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5; 

  double mmArr[8];
  double sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7;
  double stime,etime;
  double time1,time2,time3,time4,time5,time6,time7,time8;

  ct_AN = M2G[Mc_AN];
  fan = FNAN[ct_AN];
  san = SNAN[ct_AN];
  can = fan + san;
  wan = WhatSpecies[ct_AN];
  ct_on = Spe_Total_CNO[wan];

  if (Msize[Mc_AN]<Msize3[Mc_AN])
    csize = Msize3[Mc_AN] + 40;
  else
    csize = Msize[Mc_AN] + 40;

  KU_d1 = EKC_core_size[Mc_AN]*Msize2[Mc_AN];
  KU_d2 = Msize2[Mc_AN];

  if (measure_time==1){ 
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time4 = 0.0;
    time5 = 0.0;
    time6 = 0.0;
    time7 = 0.0;
    time8 = 0.0;
  }

  /*******************************
            u1^+ C^+ u2 
  *******************************/
    
  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

    /* C^+ u2 */

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=0; i<=fan; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=fan+1; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j];

	if (0<=kl){

#ifdef nosse

          if (measure_time==1) dtime(&stime);

	  /* Original version */
	  /**/	  
	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += Hks[spin][ih][kl][m][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
	      }

	      tmpvec1[n][Anum+m] += sum;
	    }
	  }
	  /**/

          if (measure_time==1){ 
            dtime(&etime);
            time1 += etime - stime;
	  }

#else
          if (measure_time==1) dtime(&stime);

	  /* Unrolling + SSE version */
	  /**/
	  for (m=0; m<(ian-3); m+=4){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      mmSum00 = _mm_setzero_pd();
	      mmSum01 = _mm_setzero_pd();
	      mmSum10 = _mm_setzero_pd();
	      mmSum11 = _mm_setzero_pd();
	      mmSum20 = _mm_setzero_pd();
	      mmSum21 = _mm_setzero_pd();
	      mmSum30 = _mm_setzero_pd();
	      mmSum31 = _mm_setzero_pd();

	      for (k=0; k<(jan-3); k+=4){
		mmTmp0 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+0]);
		mmTmp1 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+2]);

		mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+0][k+0]),mmTmp0));  
		mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+0][k+2]),mmTmp1));  

		mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+1][k+0]),mmTmp0));  
		mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+1][k+2]),mmTmp1));  

		mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+2][k+0]),mmTmp0));  
		mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+2][k+2]),mmTmp1));  

		mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+3][k+0]),mmTmp0));  
		mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+3][k+2]),mmTmp1));  
	      }

	      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	      _mm_storeu_pd(&mmArr[0], mmSum00);
	      _mm_storeu_pd(&mmArr[2], mmSum10);
	      _mm_storeu_pd(&mmArr[4], mmSum20);
	      _mm_storeu_pd(&mmArr[6], mmSum30);

	      sum0 = mmArr[0] + mmArr[1];
	      sum1 = mmArr[2] + mmArr[3];
	      sum2 = mmArr[4] + mmArr[5];
	      sum3 = mmArr[6] + mmArr[7];

	      for (; k<jan; k++){
		sum0 += Hks[spin][ih][kl][m+0][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
		sum1 += Hks[spin][ih][kl][m+1][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
		sum2 += Hks[spin][ih][kl][m+2][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
		sum3 += Hks[spin][ih][kl][m+3][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
	      }

	      tmpvec1[n][Anum+m+0] += sum0;
	      tmpvec1[n][Anum+m+1] += sum1;
	      tmpvec1[n][Anum+m+2] += sum2;
	      tmpvec1[n][Anum+m+3] += sum3;
	    }
	  } 

	  for (; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += Hks[spin][ih][kl][m][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
	      }

	      tmpvec1[n][Anum+m] += sum;
	    }
	  } 

          if (measure_time==1){ 
            dtime(&etime);
            time1 += etime - stime;
	  }
#endif

	}
      }
    }

    /* u1^+ C^+ u2 */

#ifdef nosse

    if (measure_time==1) dtime(&stime);

    /* Original version */
    /**/    
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize[Mc_AN]; i++){
            sum += Krylov_U[spin][Mc_AN][rl0*KU_d1+m*KU_d2+i+1]*tmpvec1[n][i]; 
	  }
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
	}
      }
    }
    /**/

    if (measure_time==1){ 
      dtime(&etime);
      time2 += etime - stime;
    }

#else

    /*
    printf("ABC1 Mc_AN=%2d %2d %2d %2d\n",Mc_AN,rlmax_EC[Mc_AN],EKC_core_size[Mc_AN],Msize[Mc_AN]);
    */

    if (measure_time==1) dtime(&stime);

    /* Unrolling + SSE version */
    /**/
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<(EKC_core_size[Mc_AN]-3); m+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  mmSum00 = _mm_setzero_pd();
	  mmSum01 = _mm_setzero_pd();
	  mmSum10 = _mm_setzero_pd();
	  mmSum11 = _mm_setzero_pd();
	  mmSum20 = _mm_setzero_pd();
	  mmSum21 = _mm_setzero_pd();
	  mmSum30 = _mm_setzero_pd();
	  mmSum31 = _mm_setzero_pd();

	  for (i=0; i<(Msize[Mc_AN]-3); i+=4){
	    mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i+0]);
	    mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i+2]);

	    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+1]),mmTmp0));  
	    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+3]),mmTmp1));  

	    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+1)*KU_d2+i+1]),mmTmp0));  
	    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+1)*KU_d2+i+3]),mmTmp1));  

	    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+2)*KU_d2+i+1]),mmTmp0));  
	    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+2)*KU_d2+i+3]),mmTmp1));  

	    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+3)*KU_d2+i+1]),mmTmp0));  
	    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+3)*KU_d2+i+3]),mmTmp1));  
	  }

	  mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	  mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	  mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	  mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	  _mm_storeu_pd(&mmArr[0], mmSum00);
	  _mm_storeu_pd(&mmArr[2], mmSum10);
	  _mm_storeu_pd(&mmArr[4], mmSum20);
	  _mm_storeu_pd(&mmArr[6], mmSum30);

	  sum0 = mmArr[0] + mmArr[1];
	  sum1 = mmArr[2] + mmArr[3];
	  sum2 = mmArr[4] + mmArr[5];
	  sum3 = mmArr[6] + mmArr[7];

	  for (; i<Msize[Mc_AN]; i++){
            sum0 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+1]*tmpvec1[n][i]; 
            sum1 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+1)*KU_d2+i+1]*tmpvec1[n][i]; 
            sum2 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+2)*KU_d2+i+1]*tmpvec1[n][i]; 
            sum3 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+3)*KU_d2+i+1]*tmpvec1[n][i]; 
	  }

          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum0;
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+2][rl*EKC_core_size[Mc_AN]+n+1] = sum1;
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+3][rl*EKC_core_size[Mc_AN]+n+1] = sum2;
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+4][rl*EKC_core_size[Mc_AN]+n+1] = sum3;
	}
      }

      for (; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize[Mc_AN]; i++){
            sum += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+1]*tmpvec1[n][i]; 
	  }
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
	}
      }

    }
    /**/

    if (measure_time==1){ 
      dtime(&etime);
      time2 += etime - stime;
    }

#endif

  } /* rl */

  /*******************************
            u2^+ C u1 
  *******************************/
  
  for (i=1; i<=Msize3[Mc_AN]; i++){
    for (j=i; j<=Msize3[Mc_AN]; j++){

      tmp1 = EC_matrix[spin][Mc_AN][i][j];
      tmp2 = EC_matrix[spin][Mc_AN][j][i];
      tmp3 = tmp1 + tmp2;   

      EC_matrix[spin][Mc_AN][i][j] = tmp3; 
      EC_matrix[spin][Mc_AN][j][i] = tmp3;
    }
  }

  /*******************************
             u2^+ B u2 
  *******************************/

  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

    /* B u2 */

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=fan+1; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=fan+1; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j];

	if (0<=kl){

#ifdef nosse

          if (measure_time==1) dtime(&stime);

	  /* Original version */
	  /**/	  
	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += Hks[spin][ih][kl][m][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
	      }

	      tmpvec1[n][Anum+m] += sum;
	    }
	  }
	  /**/

	  if (measure_time==1){ 
	    dtime(&etime);
	    time3 += etime - stime;
	  }

#else

          if (measure_time==1) dtime(&stime);

	  /* Unrolling + SSE version */
	  /**/
	  for (m=0; m<(ian-3); m+=4){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      mmSum00 = _mm_setzero_pd();
	      mmSum01 = _mm_setzero_pd();
	      mmSum10 = _mm_setzero_pd();
	      mmSum11 = _mm_setzero_pd();
	      mmSum20 = _mm_setzero_pd();
	      mmSum21 = _mm_setzero_pd();
	      mmSum30 = _mm_setzero_pd();
	      mmSum31 = _mm_setzero_pd();

	      for (k=0; k<(jan-3); k+=4){
		mmTmp0 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+0]);
		mmTmp1 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+2]);

		mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+0][k+0]),mmTmp0));  
		mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+0][k+2]),mmTmp1));  

		mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+1][k+0]),mmTmp0));  
		mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+1][k+2]),mmTmp1));  

		mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+2][k+0]),mmTmp0));  
		mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+2][k+2]),mmTmp1));  

		mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+3][k+0]),mmTmp0));  
		mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+3][k+2]),mmTmp1));  
	      }

	      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	      _mm_storeu_pd(&mmArr[0], mmSum00);
	      _mm_storeu_pd(&mmArr[2], mmSum10);
	      _mm_storeu_pd(&mmArr[4], mmSum20);
	      _mm_storeu_pd(&mmArr[6], mmSum30);

	      sum0 = mmArr[0] + mmArr[1];
	      sum1 = mmArr[2] + mmArr[3];
	      sum2 = mmArr[4] + mmArr[5];
	      sum3 = mmArr[6] + mmArr[7];

	      for (; k<jan; k++){
		sum0 += Hks[spin][ih][kl][m+0][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+0];
		sum1 += Hks[spin][ih][kl][m+1][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+0];
		sum2 += Hks[spin][ih][kl][m+2][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+0];
		sum3 += Hks[spin][ih][kl][m+3][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+0];
	      }

	      tmpvec1[n][Anum+m+0] += sum0;
	      tmpvec1[n][Anum+m+1] += sum1;
	      tmpvec1[n][Anum+m+2] += sum2;
	      tmpvec1[n][Anum+m+3] += sum3;
	    }
	  }

	  for (; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += Hks[spin][ih][kl][m][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
	      }

	      tmpvec1[n][Anum+m] += sum;
	    }
	  } 
	  /**/ 

	  if (measure_time==1){ 
	    dtime(&etime);
	    time3 += etime - stime;
	  }

#endif

	}
      }
    }

    /* u2^+ B u2 */

#ifdef nosse

    if (measure_time==1) dtime(&stime);

    /* Original version */
    /**/    
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=Msize[Mc_AN]; i<Msize2[Mc_AN]; i++){
            sum += Krylov_U[spin][Mc_AN][rl0*KU_d1+m*KU_d2+i+1]*tmpvec1[n][i]; 
	  }
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] += sum;
	}
      }
    }
    /**/

    if (measure_time==1){ 
      dtime(&etime);
      time4 += etime - stime;
    }

#else

    if (measure_time==1) dtime(&stime);

    /* Unrolling + SSE version */
    /**/
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<(EKC_core_size[Mc_AN]-3); m+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  mmSum00 = _mm_setzero_pd();
	  mmSum01 = _mm_setzero_pd();
	  mmSum10 = _mm_setzero_pd();
	  mmSum11 = _mm_setzero_pd();
	  mmSum20 = _mm_setzero_pd();
	  mmSum21 = _mm_setzero_pd();
	  mmSum30 = _mm_setzero_pd();
	  mmSum31 = _mm_setzero_pd();

	  for (i=Msize[Mc_AN]; i<(Msize2[Mc_AN]-3); i+=4){
	    mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i+0]);
	    mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i+2]);

	    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+1]),mmTmp0));  
	    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+3]),mmTmp1));  

	    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+1)*KU_d2+i+1]),mmTmp0));  
	    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+1)*KU_d2+i+3]),mmTmp1));  

	    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+2)*KU_d2+i+1]),mmTmp0));  
	    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+2)*KU_d2+i+3]),mmTmp1));  

	    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+3)*KU_d2+i+1]),mmTmp0));  
	    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+3)*KU_d2+i+3]),mmTmp1));  
	  }

	  mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	  mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	  mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	  mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	  _mm_storeu_pd(&mmArr[0], mmSum00);
	  _mm_storeu_pd(&mmArr[2], mmSum10);
	  _mm_storeu_pd(&mmArr[4], mmSum20);
	  _mm_storeu_pd(&mmArr[6], mmSum30);

	  sum0 = mmArr[0] + mmArr[1];
	  sum1 = mmArr[2] + mmArr[3];
	  sum2 = mmArr[4] + mmArr[5];
	  sum3 = mmArr[6] + mmArr[7];

	  for (; i<Msize2[Mc_AN]; i++){
            sum0 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+1]*tmpvec1[n][i]; 
            sum1 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+1)*KU_d2+i+1]*tmpvec1[n][i]; 
            sum2 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+2)*KU_d2+i+1]*tmpvec1[n][i]; 
            sum3 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+3)*KU_d2+i+1]*tmpvec1[n][i]; 
	  }

          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] += sum0;
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+2][rl*EKC_core_size[Mc_AN]+n+1] += sum1;
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+3][rl*EKC_core_size[Mc_AN]+n+1] += sum2;
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+4][rl*EKC_core_size[Mc_AN]+n+1] += sum3;
	}
      }

      for (; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=Msize[Mc_AN]; i<Msize2[Mc_AN]; i++){
            sum += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+1]*tmpvec1[n][i]; 
	  }
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] += sum;
	}
      }

    }
    /**/

    if (measure_time==1){ 
      dtime(&etime);
      time4 += etime - stime;
    }

#endif

  } /* rl */

  if (measure_time==1){ 
    printf("Embedding_Matrix time1=%5.3f time2=%5.3f time3=%5.3f time4=%5.3f\n",
            time1,time2,time3,time4);
  }
}










typedef struct {
  int Mc_AN;
  double ****OLP0;
  double ***Krylov_U_OLP;
  double **inv_RS;
  int *MP;
  int *Msize2;
  int *Msize4;
  int Msize2_max;
  double **tmpvec0;
  double **tmpvec1;
} Krylov_IOLP_scratch_arguments;
static void Krylov_IOLP_raii( int Mc_AN, double ****OLP0, double ***Krylov_U_OLP, double **inv_RS, int *MP,
                  int *Msize2, int *Msize4, int Msize2_max, double **tmpvec0, double **tmpvec1 ,OpenMXSolverInvocation *solver_owner);
static int Krylov_IOLP_scratch_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Krylov_IOLP_scratch_arguments *a=(Krylov_IOLP_scratch_arguments*)arguments;
  Krylov_IOLP_raii(a->Mc_AN,a->OLP0,a->Krylov_U_OLP,a->inv_RS,a->MP,a->Msize2,a->Msize4,a->Msize2_max,a->tmpvec0,a->tmpvec1,owner);
  *result=0;
  return 1;
}
void Krylov_IOLP( int Mc_AN, double ****OLP0, double ***Krylov_U_OLP, double **inv_RS, int *MP, 
                  int *Msize2, int *Msize4, int Msize2_max, double **tmpvec0, double **tmpvec1 )
{
  Krylov_IOLP_scratch_arguments arguments={Mc_AN,OLP0,Krylov_U_OLP,inv_RS,MP,Msize2,Msize4,Msize2_max,tmpvec0,tmpvec1};
  OpenMX_Solver_Run(Krylov_IOLP_scratch_operation,&arguments);
}

static void Krylov_IOLP_raii( int Mc_AN, double ****OLP0, double ***Krylov_U_OLP, double **inv_RS, int *MP,
                  int *Msize2, int *Msize4, int Msize2_max, double **tmpvec0, double **tmpvec1 ,OpenMXSolverInvocation *solver_owner)
{
  int rl,ct_AN,fan,san,can,wan,ct_on,i,j;
  int n,Anum,Bnum,k,ian,ih,kl,jg,ig,jan,m,m1,n1;
  int rl0,ZeroNum,Neumann_series,ns,Gh_AN,wanB;
  double sum,dum,tmp0,tmp1,tmp2,tmp3,rcutA,r0;
  double **tmpmat0;
  double *ko,*iko;
  double **FS;

  ct_AN = M2G[Mc_AN];
  fan = FNAN[ct_AN];
  san = SNAN[ct_AN];
  can = fan + san;
  wan = WhatSpecies[ct_AN];
  ct_on = Spe_Total_CNO[wan];
  rcutA = Spe_Atom_Cut1[wan]; 

  /* allocation of arrays */

  tmpmat0=OpenMX_Solver_AlignedRealMatrix(solver_owner,"tmpmat0-1",((size_t)1)*(EKC_core_size[Mc_AN]+4),((size_t)1)*(EKC_core_size[Mc_AN]+4));
  for (i=0; i<(EKC_core_size[Mc_AN]+4); i++){
    ;
  }

  FS=OpenMX_Solver_AlignedRealMatrix(solver_owner,"FS-2",((size_t)1)*(rlmax_EC2[Mc_AN]+2)*EKC_core_size[Mc_AN],((size_t)1)*(rlmax_EC2[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  for (i=0; i<(rlmax_EC2[Mc_AN]+2)*EKC_core_size[Mc_AN]; i++){
    ;
  }

  ko=(double*)OpenMX_Solver_Real(solver_owner,"ko-3",((size_t)1)*(rlmax_EC2[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  iko=(double*)OpenMX_Solver_Real(solver_owner,"iko-4",((size_t)1)*(rlmax_EC2[Mc_AN]+2)*EKC_core_size[Mc_AN]);

  /****************************************************
      initialize 
  ****************************************************/

  for (i=0; i<EKC_core_size_max; i++){
    for (j=0; j<Msize2_max; j++){
      tmpvec0[i][j] = 0.0;
      tmpvec1[i][j] = 0.0;
    }
  }

  /* find the nearest atom with distance of r0 */   

  r0 = 1.0e+10;
  for (k=1; k<=FNAN[ct_AN]; k++){
    Gh_AN = natn[ct_AN][k];
    wanB = WhatSpecies[Gh_AN];
    if (Dis[ct_AN][k]<r0)  r0 = Dis[ct_AN][k]; 
  }

  /* starting vector */

  m = 0;  
  for (k=0; k<=FNAN[ct_AN]; k++){

    Gh_AN = natn[ct_AN][k];
    wanB = WhatSpecies[Gh_AN];

    if ( Dis[ct_AN][k]<(scale_rc_EKC[Mc_AN]*r0) ){

      Anum = MP[k] - 1; 
      
      for (i=0; i<Spe_Total_CNO[wanB]; i++){

        tmpvec0[m][Anum+i]         = 1.0;
        Krylov_U_OLP[0][m][Anum+i] = 1.0;

        m++;  
      }
    }
  }

  /*
  for (i=0; i<EKC_core_size[Mc_AN]; i++){
    tmpvec0[i][i]         = 1.0;
    Krylov_U_OLP[0][i][i] = 1.0;
  }
  */

  /****************************************************
           generate Krylov subspace vectors
  ****************************************************/

  for (rl=0; rl<(rlmax_EC2[Mc_AN]-1); rl++){

    /*******************************************************
                            S * |Wn)
    *******************************************************/

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=0; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=0; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
        Bnum = MP[j] - 1;

        if (0<=kl){

	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*tmpvec0[n][Bnum+k];
	      }

              tmpvec1[n][Anum+m] += sum;
	    }
	  } 
	}
      }
    }

    /*************************************************************
      orthogonalization by a modified block Gram-Schmidt method 
    *************************************************************/

    /* |tmpvec1) := (I - \sum_{rl0} |U_rl0)(U_rl0|)|tmpvec1) */

    for (rl0=0; rl0<=rl; rl0++){

      /* (U_rl0|tmpvec1) */

      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
          sum = 0.0;
  	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += Krylov_U_OLP[rl0][m][i]*tmpvec1[n][i];
	  }
          tmpmat0[m][n] = sum;
	}
      }

      /* |tmpvec1) := |tmpvec1) - |U_rl0)(U_rl0|tmpvec1) */

      for (n=0; n<EKC_core_size[Mc_AN]; n++){
	for (k=0; k<EKC_core_size[Mc_AN]; k++){
	  dum = tmpmat0[k][n];
	  for (i=0; i<Msize2[Mc_AN]; i++)  tmpvec1[n][i] -= Krylov_U_OLP[rl0][k][i]*dum;
        }
      }
    }

    /*************************************************************
                       normalization of tmpvec1
    *************************************************************/

    for (m=0; m<EKC_core_size[Mc_AN]; m++){
      for (n=0; n<EKC_core_size[Mc_AN]; n++){

        sum = 0.0;
        for (i=0; i<Msize2[Mc_AN]; i++){
          sum += tmpvec1[m][i]*tmpvec1[n][i]; 
	}

        tmpmat0[m+1][n+1] = sum;
      }
    }

    /* diagonalize tmpmat0 */ 

    if ( EKC_core_size[Mc_AN]==1){
      ko[1] = tmpmat0[1][1];
      tmpmat0[1][1] = 1.0;
    }
    else{
      Eigen_lapack( tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN] );
    }

    ZeroNum = 0;

    for (n=1; n<=EKC_core_size[Mc_AN]; n++){
      if (cutoff_value<ko[n]){
	ko[n]  = sqrt(fabs(ko[n]));
	iko[n] = 1.0/ko[n];
      }
      else{
	ZeroNum++;
	ko[n]  = 0.0;
	iko[n] = 0.0;
      }

      if (error_check==1){
        printf("rl=%3d ko=%18.15f\n",rl,ko[n]);
      }
    }

    if (error_check==1){
      printf("rl=%3d ZeroNum=%3d\n",rl,ZeroNum);
    } 

    /* tmpvec0 = tmpvec1 * tmpmat0^{-1/2} */ 

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec0[n][i] = 0.0;
      }
    }

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (k=0; k<EKC_core_size[Mc_AN]; k++){
	dum = tmpmat0[k+1][n+1]*iko[n+1];
	for (i=0; i<Msize2[Mc_AN]; i++)  tmpvec0[n][i] += tmpvec1[k][i]*dum;
      }
    }

    /*************************************************************
                         store Krylov vectors
    *************************************************************/

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
        Krylov_U_OLP[rl+1][n][i] = tmpvec0[n][i];
      }
    }

  } /* rl */

  /************************************************************
       calculate the inverse of the reduced overlap matrix
  ************************************************************/

  /*  construct the reduced overlap matrix */

  for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){

    /*  S * |Vn) */

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=0; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=0; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j] - 1;

	if (0<=kl){

	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*Krylov_U_OLP[rl][n][Bnum+k];
	      }
	      tmpvec1[n][Anum+m] += sum;
	    }
	  } 
	}
      }
    }

    for (rl0=rl; rl0<rlmax_EC2[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += Krylov_U_OLP[rl0][m][i]*tmpvec1[n][i]; 
	  }
          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum;
	}
      }
    }
  }

  /* diagonalize FS */

  Eigen_lapack(FS,ko,Msize4[Mc_AN],Msize4[Mc_AN]);

  /* find ill-conditioned eigenvalues */

  ZeroNum = 0;
  for (i=1; i<=Msize4[Mc_AN]; i++){

    if (error_check==1){
      printf("Mc_AN=%2d i=%3d ko[i]=%18.15f\n",Mc_AN,i,ko[i]);
    }

    if (cutoff_value<ko[i]){
      iko[i] = 1.0/ko[i];
    }
    else{
      ZeroNum++;
      ko[i]  = 0.0;
      iko[i] = 0.0;
    }
  }  

  if (error_check==1){
    printf("Mc_AN=%2d ZeroNum=%2d\n",Mc_AN,ZeroNum);
  }

  /* construct the inverse */

  for (i=1; i<=Msize4[Mc_AN]; i++){
    for (j=1; j<=Msize4[Mc_AN]; j++){
      sum = 0.0;
      for (k=1; k<=Msize4[Mc_AN]; k++){
        sum += FS[i][k]*iko[k]*FS[j][k];
      }
      inv_RS[i-1][j-1] = sum;
    }
  }

  /* symmetrization of inv_RS */

  for (i=1; i<=Msize4[Mc_AN]; i++){
    for (j=i+1; j<=Msize4[Mc_AN]; j++){
      tmp0 = inv_RS[i-1][j-1];
      tmp1 = inv_RS[j-1][i-1];
      tmp2 = 0.5*(tmp0 + tmp1);
      inv_RS[i-1][j-1] = tmp2;
      inv_RS[j-1][i-1] = tmp2;
    }
  }



  /*
  {


    double mat1[1000][1000];
    double tsum;
    int myid;

    MPI_Comm_rank(mpi_comm_level1,&myid);

    if (myid==0){

      printf("check normalization\n");

      for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){
	for (m=0; m<EKC_core_size[Mc_AN]; m++){
	  for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

              sum = 0.0; 
   	      for (i=0; i<Msize2[Mc_AN]; i++){
                sum += Krylov_U_OLP[rl][m][i]*Krylov_U_OLP[rl0][n][i]; 
	      }
              printf("rl=%3d rl0=%3d m=%3d n=%3d  <|>=%18.15f\n",rl,rl0,m,n,sum);
	    }
	  }
	}
      }


      printf("\n\ninvS\n");


      for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){
	for (m=0; m<EKC_core_size[Mc_AN]; m++){

	  for (i=0; i<Msize2[Mc_AN]; i++){

	    sum = 0.0;

	    for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
	      for (n=0; n<EKC_core_size[Mc_AN]; n++){
		sum += inv_RS[rl*EKC_core_size[Mc_AN]+m][rl0*EKC_core_size[Mc_AN]+n]*Krylov_U_OLP[rl0][n][i]; 
	      }
	    }

	    mat1[rl*EKC_core_size[Mc_AN]+m][i] = sum; 
	  }
	}
      }


      tsum = 0.0;

      for (i=0; i<Msize2[Mc_AN]; i++){
	for (j=0; j<Msize2[Mc_AN]; j++){

	  sum = 0.0;
	  for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){
	    for (m=0; m<EKC_core_size[Mc_AN]; m++){
	      sum += Krylov_U_OLP[rl][m][i]*mat1[rl*EKC_core_size[Mc_AN]+m][j];
	    }
	  }

	  printf("i=%4d j=%4d %18.15f\n",i,j,sum);

          tsum += fabs(sum);

	}    
      }    


      printf("\n\ntsum=%18.15f\n",tsum);

    }


    MPI_Finalize();
    exit(0);
  }
  */

  /* freeing of arrays */

  for (i=0; i<(EKC_core_size[Mc_AN]+4); i++){
    ;
  }
  ;

  for (i=0; i<(rlmax_EC2[Mc_AN]+2)*EKC_core_size[Mc_AN]; i++){
    ;
  }
  ;

  ;
  ;
}












void S_orthonormalize_vec( int Mc_AN, int ct_on, double **vec,
                           double **workvec, double ****OLP0,
                           double **tmpmat0, double *ko, double *iko, 
                           int *MP, int *Msize2 )
{
  int n,i,j,can,san,fan,ct_AN;
  int k,m,ZeroNum,Anum,Bnum,ih;
  int kl,jg,jan,ig,ian;
  double dum,sum;
  double tmp0,tmp1,tmp2,tmp3;
  __m128d mmSum00,mmSum01,mmSum10,mmSum11,mmSum20,mmSum21,mmSum30,mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5; 
  double mmArr[8];
  double sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7;

  ct_AN = M2G[Mc_AN];
  fan = FNAN[ct_AN];
  san = SNAN[ct_AN];
  can = fan + san;

  /* S|Vn) */

  for (n=0; n<EKC_core_size[Mc_AN]; n++){
    for (i=0; i<Msize2[Mc_AN]; i++){
      workvec[n][i]  = 0.0;
    }
  }

  for (i=0; i<=can; i++){

    ig = natn[ct_AN][i];
    ian = Spe_Total_CNO[WhatSpecies[ig]];
    Anum = MP[i] - 1;
    ih = S_G2M[ig];

    for (j=0; j<=can; j++){

      kl = RMI1[Mc_AN][i][j];
      jg = natn[ct_AN][j];
      jan = Spe_Total_CNO[WhatSpecies[jg]];
      Bnum = MP[j] - 1;

      if (0<=kl){

#ifdef nosse

	/* Original version */
	/**/	
	for (m=0; m<ian; m++){
	  for (n=0; n<EKC_core_size[Mc_AN]; n++){

	    sum = 0.0;
	    for (k=0; k<jan; k++){
	      sum += OLP0[ih][kl][m][k]*vec[n][Bnum+k];
	    }

	    workvec[n][Anum+m] += sum;
	  }
	}
	/**/

#else

	/* Unrolling + SSE version */
	/**/
	for (m=0; m<(ian-3); m+=4){
	  for (n=0; n<EKC_core_size[Mc_AN]; n++){

	    mmSum00 = _mm_setzero_pd();
	    mmSum01 = _mm_setzero_pd();
	    mmSum10 = _mm_setzero_pd();
	    mmSum11 = _mm_setzero_pd();
	    mmSum20 = _mm_setzero_pd();
	    mmSum21 = _mm_setzero_pd();
	    mmSum30 = _mm_setzero_pd();
	    mmSum31 = _mm_setzero_pd();
	    
	    for (k=0; k<(jan-3); k+=4){
	      mmTmp0 = _mm_loadu_pd(&vec[n][Bnum+k+0]);
	      mmTmp1 = _mm_loadu_pd(&vec[n][Bnum+k+2]);

	      mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+0]),mmTmp0));  
	      mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+2]),mmTmp1));  

	      mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+0]),mmTmp0));  
	      mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+2]),mmTmp1));  

	      mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+0]),mmTmp0));  
	      mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+2]),mmTmp1));  

	      mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+0]),mmTmp0));  
	      mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+2]),mmTmp1));  
	    }

	    mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	    mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	    mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	    mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	    _mm_storeu_pd(&mmArr[0], mmSum00);
	    _mm_storeu_pd(&mmArr[2], mmSum10);
	    _mm_storeu_pd(&mmArr[4], mmSum20);
	    _mm_storeu_pd(&mmArr[6], mmSum30);

	    sum0 = mmArr[0] + mmArr[1];
	    sum1 = mmArr[2] + mmArr[3];
	    sum2 = mmArr[4] + mmArr[5];
	    sum3 = mmArr[6] + mmArr[7];

	    for (; k<jan; k++){
	      sum0 += OLP0[ih][kl][m+0][k]*vec[n][Bnum+k];
	      sum1 += OLP0[ih][kl][m+1][k]*vec[n][Bnum+k];
	      sum2 += OLP0[ih][kl][m+2][k]*vec[n][Bnum+k];
	      sum3 += OLP0[ih][kl][m+3][k]*vec[n][Bnum+k];
	    }

	    workvec[n][Anum+m+0] += sum0;
	    workvec[n][Anum+m+1] += sum1;
	    workvec[n][Anum+m+2] += sum2;
	    workvec[n][Anum+m+3] += sum3;
	  }
	}

	for (; m<ian; m++){
	  for (n=0; n<EKC_core_size[Mc_AN]; n++){

	    sum = 0.0;
	    for (k=0; k<jan; k++){
	      sum += OLP0[ih][kl][m][k]*vec[n][Bnum+k];
	    }

	    workvec[n][Anum+m] += sum;
	  }
	}
	/**/ 

#endif

      }
    }
  }

  /*  (Vn|S|Vn) */

#ifdef nosse

  /* Original version */
  /**/  
  for (m=0; m<EKC_core_size[Mc_AN]; m++){
    for (n=m; n<EKC_core_size[Mc_AN]; n++){
      sum = 0.0;
      for (i=0; i<Msize2[Mc_AN]; i++){
	sum += vec[m][i]*workvec[n][i];
      }
      tmpmat0[m+1][n+1] = sum;
      tmpmat0[n+1][m+1] = sum;
    }
  }
  /**/

#else

  /* Unrolling + SSE version */
  /**/
  for (m=0; m<(EKC_core_size[Mc_AN]-3); m+=4){
    for (n=m; n<EKC_core_size[Mc_AN]; n++){
      mmSum00 = _mm_setzero_pd();
      mmSum01 = _mm_setzero_pd();
      mmSum10 = _mm_setzero_pd();
      mmSum11 = _mm_setzero_pd();
      mmSum20 = _mm_setzero_pd();
      mmSum21 = _mm_setzero_pd();
      mmSum30 = _mm_setzero_pd();
      mmSum31 = _mm_setzero_pd();

      for (i=0; i<(Msize2[Mc_AN]-3); i+=4){
	mmTmp0 = _mm_loadu_pd(&workvec[n][i+0]);
	mmTmp1 = _mm_loadu_pd(&workvec[n][i+2]);

	mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&vec[m+0][i+0]),mmTmp0));  
	mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&vec[m+0][i+2]),mmTmp1));  

	mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&vec[m+1][i+0]),mmTmp0));  
	mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&vec[m+1][i+2]),mmTmp1));  

	mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&vec[m+2][i+0]),mmTmp0));  
	mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&vec[m+2][i+2]),mmTmp1));  

	mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&vec[m+3][i+0]),mmTmp0));  
	mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&vec[m+3][i+2]),mmTmp1));  
      }

      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
      
      _mm_storeu_pd(&mmArr[0], mmSum00);
      _mm_storeu_pd(&mmArr[2], mmSum10);
      _mm_storeu_pd(&mmArr[4], mmSum20);
      _mm_storeu_pd(&mmArr[6], mmSum30);

      sum0 = mmArr[0] + mmArr[1];
      sum1 = mmArr[2] + mmArr[3];
      sum2 = mmArr[4] + mmArr[5];
      sum3 = mmArr[6] + mmArr[7];

      for (; i<Msize2[Mc_AN]; i++){
	sum0 += vec[m+0][i]*workvec[n][i];
	sum1 += vec[m+1][i]*workvec[n][i];
	sum2 += vec[m+2][i]*workvec[n][i];
	sum3 += vec[m+3][i]*workvec[n][i];
      }

      tmpmat0[m+1][n+1] = sum0;
      tmpmat0[n+1][m+1] = sum0;

      tmpmat0[m+2][n+1] = sum1;
      tmpmat0[n+1][m+2] = sum1;

      tmpmat0[m+3][n+1] = sum2;
      tmpmat0[n+1][m+3] = sum2;

      tmpmat0[m+4][n+1] = sum3;
      tmpmat0[n+1][m+4] = sum3;
    }
  }

  for (; m<EKC_core_size[Mc_AN]; m++){
    for (n=m; n<EKC_core_size[Mc_AN]; n++){
      sum = 0.0;
      for (i=0; i<Msize2[Mc_AN]; i++){
	sum += vec[m][i]*workvec[n][i];
      }
      tmpmat0[m+1][n+1] = sum;
      tmpmat0[n+1][m+1] = sum;
    }
  }
  /**/

#endif

  /* diagonalize tmpmat0 */ 

  if ( EKC_core_size[Mc_AN]==1){
    ko[1] = tmpmat0[1][1];
    tmpmat0[1][1] = 1.0;
  }
  else{
    Eigen_lapack( tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN] );
  }

  ZeroNum = 0;

  for (n=1; n<=EKC_core_size[Mc_AN]; n++){
    if (cutoff_value<ko[n]){
      ko[n]  = sqrt(fabs(ko[n]));
      iko[n] = 1.0/ko[n];
    }
    else{
      ZeroNum++;
      ko[n]  = 0.0;
      iko[n] = 0.0;
    }
  }

  /* U0 = vec * tmpmat0^{-1/2} */ 

  for (n=0; n<EKC_core_size[Mc_AN]; n++){
    for (i=0; i<Msize2[Mc_AN]; i++){
      workvec[n][i] = 0.0;
    }
  }

  for (n=0; n<EKC_core_size[Mc_AN]; n++){
    for (k=0; k<EKC_core_size[Mc_AN]; k++){
      dum = tmpmat0[k+1][n+1]*iko[n+1];
      for (i=0; i<Msize2[Mc_AN]; i++)  workvec[n][i] += vec[k][i]*dum;
    }
  }
  
  for (n=0; n<EKC_core_size[Mc_AN]; n++){
    for (i=0; i<Msize2[Mc_AN]; i++){
      vec[n][i] = workvec[n][i];
    }
  }
}



void Inverse_S_by_Cholesky(int Mc_AN, double ****OLP0, double **invS, int *MP, int NUM, double *LoS)
{
  char  *UPLO="U";
  INTEGER N,lda,info,lwork;
  int Gc_AN,i,j,k,Anum,Gi,wanA,time1,time2;
  int ig,ian,ih,kl,jg,jan,Bnum,m,n;
  double tmp0,tmp1;

  N = NUM;
  Gc_AN = M2G[Mc_AN];

  /* OLP0 to invS */

  for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){

    ig = natn[Gc_AN][i];
    ian = Spe_Total_CNO[WhatSpecies[ig]];
    Anum = MP[i] - 1;
    ih = S_G2M[ig];

    for (j=0; j<=(FNAN[Gc_AN]+SNAN[Gc_AN]); j++){

      kl = RMI1[Mc_AN][i][j];
      jg = natn[Gc_AN][j];
      jan = Spe_Total_CNO[WhatSpecies[jg]];
      Bnum = MP[j] - 1;

      if (0<=kl){
	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){
	    invS[Anum+m][Bnum+n] = OLP0[ih][kl][m][n];
	  }
	}
      }

      else{
	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){
	    invS[Anum+m][Bnum+n] = 0.0;
	  }
	}
      }
    }
  }

  /* invS -> LoS */

  for (i=0;i<N;i++) {
    for (j=0;j<N;j++) {
       LoS[j*N+i]= invS[i][j];
    }
  }

  /* call dpotrf_() in clapack */

  lda = N;
  F77_NAME(dpotrf,DPOTRF)(UPLO, &N, LoS, &lda, &info);

  if (info!=0){
    printf("Error in dpotrf_() which is called from Embedding_Cluster.c  info=%2d\n",info);
  }

  /* call dpotri_() in clapack */

  lwork = N;
  F77_NAME(dpotri,DPOTRI)(UPLO, &N, LoS, &lda, &info);

  if (info!=0){
    printf("Error in dpotri_() which is called from Embedding_Cluster.c  info=%2d\n",info);
  }

  /* LoS -> invS */

  for (j=0; j<N; j++) {
    m = j*N; 
    for (i=0; i<=j; i++) {
      invS[i][j] = LoS[m+i];
      invS[j][i] = LoS[m+i];
    }
  }

}


void Save_DOS_Col(const OpenMXChemicalPotentials *chemical, OpenMXSolverResidueView Residues, double ****OLP0, OpenMXSolverRealFieldView EVal, OpenMXSolverIntTensor LO_TC, OpenMXSolverIntTensor HO_TC)
{
  int spin,Mc_AN,wanA,Gc_AN,tno1;
  int i1,i,j,MaxL,l,h_AN,Gh_AN,wanB,tno2;
  double Stime_atom,Etime_atom; 
  double sum;
  int i_vec[10];  
  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;
  int numprocs,myid,ID,tag;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){
    printf("The DOS is supported for a range from -10 to 10 eV for the O(N) Krylov subspace method.\n");
  }

  /* open file pointers */

  if (myid==Host_ID){

    sprintf(file_eig,"%s%s.Dos.val",filepath,filename);
    if ( (fp_eig=fopen(file_eig,"w"))==NULL ) {
      printf("cannot open a file %s\n",file_eig);
    }
  }
  
  sprintf(file_ev, "%s%s.Dos.vec%i",filepath,filename,myid);
  if ( (fp_ev=fopen(file_ev,"w"))==NULL ) {
    printf("cannot open a file %s\n",file_ev);
  }

  /****************************************************
                   save *.Dos.vec
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wanA = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wanA];

      fprintf(fp_ev,"<AN%dAN%d\n",Gc_AN,spin);
      fprintf(fp_ev,"%d\n",(openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN]-openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]+1));

      {
        double *eval_row_21=openmx_solver_field_row(&EVal,spin,Mc_AN);
        for (i1=0; i1<(openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN]-openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]+1); i1++){

	fprintf(fp_ev,"%4d  %10.6f  ",i1,eval_row_21[i1-1+openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]]);

	for (i=0; i<tno1; i++){

	  sum = 0.0;
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wanB = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wanB];
	    for (j=0; j<tno2; j++){
	      sum += openmx_solver_residue_row(&Residues,spin,Mc_AN,h_AN,i,j)[i1]*
		               OLP0[Mc_AN][h_AN][i][j];
	    }
	  }

	  fprintf(fp_ev,"%8.5f",sum);
	}
	fprintf(fp_ev,"\n");
      }
      }

      fprintf(fp_ev,"AN%dAN%d>\n",Gc_AN,spin);

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }
  }

  /****************************************************
                   save *.Dos.val
  ****************************************************/

  if ( (fp_eig=fopen(file_eig,"w")) != NULL ) {

    fprintf(fp_eig,"mode        5\n");
    fprintf(fp_eig,"NonCol      0\n");
    /*      fprintf(fp_eig,"N           %d\n",n); */
    fprintf(fp_eig,"Nspin       %d\n",SpinP_switch);
    fprintf(fp_eig,"Erange      %lf %lf\n",Dos_Erange[0],Dos_Erange[1]);
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
    fprintf(fp_eig,"ChemP       %lf\n",chemical->chemical_potential);

  }

  /* close file pointers */

  if (myid==Host_ID){
    if (fp_eig) fclose(fp_eig);
  }

  if (fp_ev)  fclose(fp_ev);
}


typedef struct {
  KrylovState *state;
  char *mode;
  int SCF_iter;
  double *****Hks;
  double ****OLP0;
  double *****CDM;
  double *****EDM;
  double *Eele0;
  double *Eele1;
} Krylov_Col_trd_arguments;
static double Krylov_Col_trd_owned(KrylovState *state, char *mode,
			 int SCF_iter,
			 double *****Hks,
			 double ****OLP0,
			 double *****CDM,
			 double *****EDM,
			 double Eele0[2],
			 double Eele1[2], OpenMXSolverInvocation *solver_owner);
static int Krylov_Col_trd_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Krylov_Col_trd_arguments *a=(Krylov_Col_trd_arguments*)arguments;
  *result=Krylov_Col_trd_owned(a->state, a->mode,a->SCF_iter,a->Hks,a->OLP0,a->CDM,a->EDM,a->Eele0,a->Eele1,owner);
  return 1;
}
static double Krylov_Col_trd(KrylovState *state, char *mode,
			 int SCF_iter,
			 double *****Hks,
			 double ****OLP0,
			 double *****CDM,
			 double *****EDM,
			 double Eele0[2],
			 double Eele1[2])
{
  Krylov_Col_trd_arguments arguments={state,mode,SCF_iter,Hks,OLP0,CDM,EDM,Eele0,Eele1};
  return OpenMX_Solver_Run(Krylov_Col_trd_operation,&arguments);
}

static double Krylov_Col_trd_owned(KrylovState *state, char *mode,
			 int SCF_iter,
			 double *****Hks,
			 double ****OLP0,
			 double *****CDM,
			 double *****EDM,
			 double Eele0[2],
			 double Eele1[2], OpenMXSolverInvocation *solver_owner)
{

/**********************************************************************************
  The structure of this subroutine is exactly the same as the original Krylov_Col,
  except for the main loop where OMP parallelization at the atom level is removed.
**********************************************************************************/



  int Mc_AN,Gc_AN,i,is,js,Gi,wan,wanA,wanB,Anum;
  int num,NUM0,NUM,NUM1,n2,Cwan,Hwan,Rn2;
  int size1,size2,max_size1,max_size2;
  int ih,ig,ian,j,kl,jg,jan,Bnum,m,n,spin,i2,ip;
  int k,l,i1,j1,P_min,m_size,q1,q2,csize,Residues_size;
  int h_AN1,Mh_AN1,h_AN2,Gh_AN1,Gh_AN2,wan1,wan2;
  int po,po1,loopN,tno1,tno2,h_AN,Gh_AN,rl1,rl2,rl;
  int MA_AN,GA_AN,tnoA,GB_AN,tnoB,ct_on;
  int Msize2_max;

  double My_TZ,sum,FermiF,time0,srt;
  double sum00,sum10,sum20,sum30;
  double sum01,sum11,sum21,sum31;
  double sum02,sum12,sum22,sum32;
  double sum03,sum13,sum23,sum33;
  double tmp0,tmp1,tmp2,tmp3,b2,co,x0,Sx,Dx,xmin,xmax;
  double My_Num_State,Num_State,x,Dnum;
  double emin,emax,de;
  double TStime,TEtime,Stime1,Etime1;
  double Stime2,Etime2;
  double time1,time2,time3,time4,time5;
  double time6,time7,time8,time9,time10;
  double time11,time12,time13,time14,time15,time16;
  double Erange;
  double My_Eele0[2],My_Eele1[2];
  double max_x=30.0;
  double ChemP_MAX,ChemP_MIN,spin_degeneracy;
  double spetrum_radius;
  OpenMXSolverRealFieldView EVal;
  OpenMXSolverResidueView Residues;
  OpenMXSolverRealFieldView PDOS_DC;
  double *tmp_array;
  double *tmp_array2;
  int *MP;

  /*****************************************************
  Msize:   \sum_FNAN Spe_Total_CNO  
  Msize2:  \sum_FNAN+SNAN Spe_Total_CNO  
  Msize3:  rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]
  Msize4:  rlmax_EC2[Mc_AN]*EKC_core_size[Mc_AN]
  Msize5:  dimension for the last column of Residues
  *****************************************************/

  int *Msize;
  int *Msize2;
  int *Msize3;
  int *Msize4;
  int *Msize5;
  OpenMXSolverIntTensor LO_TC;
  OpenMXSolverIntTensor HO_TC;
  int numprocs,myid,ID,IDS,IDR,tag=999;
  double Stime_atom, Etime_atom;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Barrier(mpi_comm_level1);
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  Krylov_GPU_InitOnce(state);
  Krylov_GPU_PreparePool(state, omp_get_max_threads());

  printf("ABC1 myid=%2d\n",myid);

  dtime(&TStime);

  if (measure_time==1){ 
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
    time14 = 0.0;
    time15 = 0.0;
  }

  /****************************************************
                   allocation of arrays:
  ****************************************************/

  Msize=OpenMX_Solver_Int(solver_owner,"Msize",((size_t)1)*(Matomnum+1));
  Msize2=OpenMX_Solver_Int(solver_owner,"Msize2",((size_t)1)*(Matomnum+1));
  Msize3=OpenMX_Solver_Int(solver_owner,"Msize3",((size_t)1)*(Matomnum+1));
  Msize4=OpenMX_Solver_Int(solver_owner,"Msize4",((size_t)1)*(Matomnum+1));
  Msize2_max = 0;

  /* find Msize */

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Msize[Mc_AN] = 1;
    }
    else{

      Gc_AN = M2G[Mc_AN];

      NUM = 0;
      for (i=0; i<=FNAN[Gc_AN]; i++){
	Gi = natn[Gc_AN][i];
	wanA = WhatSpecies[Gi];
	NUM += Spe_Total_CNO[wanA];
      }
      Msize[Mc_AN] = NUM;
    }
  }

  /* find Msize2 and Msize2_max */

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Msize2[Mc_AN] = 1;
    }
    else{

      Gc_AN = M2G[Mc_AN];

      NUM = 0;
      for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
	Gi = natn[Gc_AN][i];
	wanA = WhatSpecies[Gi];
	NUM += Spe_Total_CNO[wanA];
      }
      Msize2[Mc_AN] = NUM;
    }
 
    if (Msize2_max<Msize2[Mc_AN]) Msize2_max = Msize2[Mc_AN] + 4;
  }

  /* find Msize3 and Msize4 */

  Msize3[0] = 1;
  Msize4[0] = 1;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wan = WhatSpecies[Gc_AN];
    ct_on = Spe_Total_CNO[wan];
    Msize3[Mc_AN] = rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN];
    Msize4[Mc_AN] = rlmax_EC2[Mc_AN]*EKC_core_size[Mc_AN];
  }

  if (Krylov_GPU_Enabled(state)){
    Krylov_GPU_Preflight(state, Matomnum,Msize,Msize3);
    if (state->gpu_fits)
      Krylov_KUCache_Prepare(state, Matomnum,SpinP_switch+1,Msize,Msize3,SCF_iter,
                             state->gpu_device_ranks,state->gpu_need_total);
  }

  m_size = 0;

  EVal=OpenMX_Solver_Field(solver_owner,((size_t)1)*(SpinP_switch+1),((size_t)1)*(Matomnum+1));
  for (spin=0; spin<=SpinP_switch; spin++){

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      n2 = Msize3[Mc_AN] + 2;
      m_size += n2;
      OpenMX_Solver_FieldResize(EVal,spin,Mc_AN,((size_t)1)*n2);
    }
  }

  if (state->Col_trd_owned_firsttime){
  PrintMemory("Krylov: EVal",  sizeof(double)*m_size,NULL);
  }

  if (2<=level_stdout){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        printf("<Krylov> myid=%4d Mc_AN=%4d Gc_AN=%4d Msize=%4d\n",
        myid,Mc_AN,M2G[Mc_AN],Msize[Mc_AN]);
    }
  }

  /****************************************************
    allocation of arrays:

    double PDOS[SpinP_switch+1]
               [Matomnum+1]
               [n2]
  ****************************************************/

  m_size = 0;
  PDOS_DC=OpenMX_Solver_Field(solver_owner,((size_t)1)*(SpinP_switch+1),((size_t)1)*(Matomnum+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      n2 = Msize3[Mc_AN] + 4;
      m_size += n2;
      OpenMX_Solver_FieldResize(PDOS_DC,spin,Mc_AN,((size_t)1)*n2);
    }
  }

  if (state->Col_trd_owned_firsttime){
  PrintMemory("Krylov: PDOS_DC",sizeof(double)*m_size,NULL);
  }

  /****************************************************
    allocation of arrays:

    int LO_TC[SpinP_switch+1][Matomnum+1]
    int HO_TC[SpinP_switch+1][Matomnum+1]
  ****************************************************/

  LO_TC = OpenMX_Solver_IntView(solver_owner,"LO_TC",SpinP_switch+1,Matomnum+1);
  HO_TC = OpenMX_Solver_IntView(solver_owner,"HO_TC",SpinP_switch+1,Matomnum+1);

  /****************************************************
    allocation of array:

    double Residues[SpinP_switch+1]
                   [Matomnum+1]
                   [FNAN[Gc_AN]+1]
                   [Spe_Total_CNO[Gc_AN]] 
                   [Spe_Total_CNO[Gh_AN]] 
                   [HO_TC-LO_TC+3]
  ****************************************************/

  Residues=OpenMX_Solver_Residues(solver_owner,((size_t)1)*(SpinP_switch+1),1,0);
  FNAN[0] = 0;
  for (spin=0; spin<=SpinP_switch; spin++)
    OpenMX_Solver_ResidueResize(Residues,spin,0,1);

  /****************************************************
     initialize density and energy density matrices
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      wanA = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wanA];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	wanB = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wanB];
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
	    CDM[spin][Mc_AN][h_AN][i][j] = 0.0;
	    EDM[spin][Mc_AN][h_AN][i][j] = 0.0;
	  }
	}
      }
    }
  }

  /****************************************************
   MPI

   Hks
  ****************************************************/

  if (measure_time==1) dtime(&Stime1);

  if (SCF_iter==1){

    /***********************************
               set data size
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){
	tag = 999;

	/* find data size to send block data */
	if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	  size1 = 0;
	  for (spin=0; spin<=SpinP_switch; spin++){
	    for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
	      Mc_AN = Snd_MAN[IDS][n];
	      Gc_AN = Snd_GAN[IDS][n];
	      Cwan = WhatSpecies[Gc_AN]; 
	      tno1 = Spe_Total_CNO[Cwan];
	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
		Gh_AN = natn[Gc_AN][h_AN];        
		Hwan = WhatSpecies[Gh_AN];
		tno2 = Spe_Total_CNO[Hwan];
		size1 += tno1*tno2;
	      }
	    }
	  }

	  Snd_HFS_Size[IDS] = size1;
	  MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
	}
	else{
	  Snd_HFS_Size[IDS] = 0;
	}

	/* receiving of size of data */

	if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

	  MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	  Rcv_HFS_Size[IDR] = size2;
	}
	else{
	  Rcv_HFS_Size[IDR] = 0;
	}

	if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
      }
      else{
	Snd_HFS_Size[IDS] = 0;
	Rcv_HFS_Size[IDR] = 0;
      }
    }
  }

  /***********************************
             data transfer
  ************************************/

  /* find maximum size of size1 */

  max_size1 = 0;
  max_size2 = 0;
  for (ID=0; ID<numprocs; ID++){
    size1 = Snd_HFS_Size[ID];
    if (max_size1<size1) max_size1 = size1;
    size2 = Rcv_HFS_Size[ID];
    if (max_size2<size2) max_size2 = size2;
  }

  /* allocation of arrays */

  tmp_array=OpenMX_Solver_Real(solver_owner,"tmp_array",((size_t)1)*max_size1);
  tmp_array2=OpenMX_Solver_Real(solver_owner,"tmp_array2",((size_t)1)*max_size2);

  /* MPI communication */

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      /*****************************
              sending of data 
      *****************************/

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

        size1 = Snd_HFS_Size[IDS];

        /* multidimentional array to vector array */

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
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

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

        size2 = Rcv_HFS_Size[IDR];
        
        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          Mc_AN = S_TopMAN[IDR] - 1;  /* S_TopMAN should be used. */
          for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];

            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  Hks[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
		}
	      }
	    }
	  }        
	}
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
        MPI_Wait(&request,&stat);
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

  if (SCF_iter==1){

    /***********************************
               data transfer
    ************************************/

    tag = 999;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){

        /*****************************
                sending of data 
        *****************************/

        if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

          size1 = Snd_HFS_Size[IDS]/(1+SpinP_switch);

          /* multidimentional array to vector array */

          num = 0;

          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
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

        if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
          
          size2 = Rcv_HFS_Size[IDR]/(1+SpinP_switch);
        
          MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

          num = 0;
          Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
          for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];

            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  OLP0[Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
   	        }
	      }
	    }
	  }        
        }

        if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
          MPI_Wait(&request,&stat);
	}
      }
    }
  }

  /* freeing of arrays */
  ;

  if (measure_time==1){ 
    dtime(&Etime1);
    time1 = Etime1 - Stime1;      
  }

  /***********************************************
    for regeneration of the buffer matrix
  ***********************************************/

  if (sqrt(fabs(state->history->norm[0]))<(0.2+0.10*(double)atomnum) && state->Col_trd_owned_recalc_firsttime){
    state->Col_trd_owned_recalc_flag = 1;
    state->Col_trd_owned_recalc_firsttime = 0;
  }
  else{
    state->Col_trd_owned_recalc_flag = 0;
  }

  if (SCF_iter==1) state->Col_trd_owned_recalc_firsttime = 1;

  if (error_check==1){
    printf("SCF_iter=%2d recalc_firsttime=%2d\n",SCF_iter,state->Col_trd_owned_recalc_firsttime);
  }

  if (measure_time==1) dtime(&Stime2);

  /*
#pragma omp parallel shared(List_YOUSO,Residues,EDM,CDM,HO_TC,LO_TC,ChemP,EVal,RMI1,S_G2M,EC_matrix,recalc_flag,recalc_EM,Krylov_U,SpinP_switch,EKC_core_size,rlmax_EC,rlmax_EC2,time11,time10,time9,time8,time7,time6,time5,time4,time3,time2,Hks,OLP0,EKC_Exact_invS_flag,SCF_iter,Msize3,Msize2,Msize,natn,FNAN,SNAN,Spe_Total_CNO,WhatSpecies,M2G,Matomnum,myid,time_per_atom,firsttime) 
  */

  {
    int OMPID,Nthrds,Nprocs;
    int Mc_AN,Gc_AN,wan,spin;
    int ig,ian,ih,kl,jg,jan,Bnum,m,n,rl;
    int Anum,i,j,k,Gi,wanA,NUM,n2,csize,is,i2;
    int i1,rl1,js,ip,po1,tno1,h_AN,Gh_AN,wanB,tno2;
    int *MP;
    int KU_d1, KU_d2,lda,ldb,ldc,M,N,K;
    double alpha, beta;
    OpenMXSolverRealTensor invS = {0};
    double *LoS;
    double *C,*KU;
    double *H_DC,*ko;
    double ***Krylov_U_OLP = NULL;
    OpenMXSolverRealTensor inv_RS = {0};
    double sum00,sum10,sum20,sum30,sum;
    double tmp0,tmp1,tmp2,tmp3;
    double Erange;
    double Stime_atom,Etime_atom;
    double Stime1,Etime1;
    OpenMXSolverRealTensor tmpvec0 = {0};
    OpenMXSolverRealTensor tmpvec1 = {0};
    OpenMXSolverRealTensor tmpvec2 = {0};
    Krylov_GPU_Workspace *gpu_ws;

    __m128d mmSum00,mmSum01,mmSum10,mmSum11;
    __m128d mmSum20,mmSum21,mmSum30,mmSum31;
    __m128d mmTmp0, mmTmp1, mmTmp2, mmTmp3;
    __m128d mmTmp4, mmTmp5; 

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    gpu_ws = Krylov_GPU_GetWorkspace(state, OMPID);

    /*
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();
    */

    /* allocation of arrays */

    MP=OpenMX_Solver_Int(solver_owner,"MP",((size_t)1)*List_YOUSO[2]);

    tmpvec0=OpenMX_Solver_RealView(solver_owner,"tmpvec0",((size_t)1)*EKC_core_size_max,((size_t)1)*Msize2_max);


    tmpvec1=OpenMX_Solver_RealView(solver_owner,"tmpvec1",((size_t)1)*EKC_core_size_max,((size_t)1)*Msize2_max);


    tmpvec2=OpenMX_Solver_RealView(solver_owner,"tmpvec2",((size_t)1)*EKC_core_size_max,((size_t)1)*Msize2_max);


    if (state->Col_trd_owned_firsttime && OMPID==0){
      PrintMemory("Krylov: tmpvec0",sizeof(double)*EKC_core_size_max*Msize2_max,NULL);
      PrintMemory("Krylov: tmpvec1",sizeof(double)*EKC_core_size_max*Msize2_max,NULL);
      PrintMemory("Krylov: tmpvec2",sizeof(double)*EKC_core_size_max*Msize2_max,NULL);
    }

    /***********************************************
     main loop of calculation 
    ***********************************************/

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wan = WhatSpecies[Gc_AN];

      /* MP array */

      Anum = 1;
      for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
	MP[i] = Anum;
	Gi = natn[Gc_AN][i];
	wanA = WhatSpecies[Gi];
	Anum += Spe_Total_CNO[wanA];
      }
      NUM = Anum - 1;
      n2 = NUM + 40;

      /***********************************************
                 allocation of arrays:
      ***********************************************/

      if (Msize[Mc_AN]<Msize3[Mc_AN])
	csize = Msize3[Mc_AN] + 40;
      else
	csize = Msize[Mc_AN] + 40;

      KU_d1 = EKC_core_size[Mc_AN]*Msize2[Mc_AN];
      KU_d2 = Msize2[Mc_AN];

      H_DC=OpenMX_Solver_Real(solver_owner,"H_DC",((size_t)1)*csize*csize);
      ko=OpenMX_Solver_Real(solver_owner,"ko",((size_t)1)*csize);
      C=OpenMX_Solver_Real(solver_owner,"C",((size_t)1)*csize*csize);
      KU=OpenMX_Solver_Real(solver_owner,"KU",((size_t)1)*(Msize2[Mc_AN]+2)*Msize3[Mc_AN]);

      /***********************************************
        calculate the inverse of overlap matrix
      ***********************************************/

      if (SCF_iter==1 && Msize3[Mc_AN]<Msize2[Mc_AN] && EKC_Exact_invS_flag==1){

	LoS=OpenMX_Solver_Real(solver_owner,"LoS",((size_t)1)*(Msize2[Mc_AN]+3)*(Msize2[Mc_AN]+3));

	invS=OpenMX_Solver_RealView(solver_owner,"invS",((size_t)1)*(Msize2[Mc_AN]+3),((size_t)1)*(Msize2[Mc_AN]+3));


	if (measure_time==1) dtime(&Stime1);

	Inverse_S_by_Cholesky(Mc_AN, OLP0, OpenMX_Solver_RealMatrix(solver_owner,"invS",invS.extent[0],invS.extent[1]), MP, NUM, LoS);

	if (measure_time==1){ 
	  dtime(&Etime1);
	  time2 += Etime1 - Stime1;      
	}
      }

      else if (SCF_iter==1 && Msize3[Mc_AN]<Msize2[Mc_AN] && EKC_invS_flag==1){

        Krylov_U_OLP = OpenMX_Solver_RealVolume(solver_owner,"Krylov_U_OLP",
            rlmax_EC2[Mc_AN],EKC_core_size[Mc_AN],Msize2[Mc_AN]+3);
        for (i=0; i<rlmax_EC2[Mc_AN]; i++)
          for (j=0; j<EKC_core_size[Mc_AN]; j++)
            for (k=0; k<Msize2[Mc_AN]+3; k++) Krylov_U_OLP[i][j][k] = 0.0;

	inv_RS=OpenMX_Solver_RealView(solver_owner,"inv_RS",((size_t)1)*(rlmax_EC2[Mc_AN]+1)*EKC_core_size[Mc_AN],((size_t)1)*(rlmax_EC2[Mc_AN]+1)*EKC_core_size[Mc_AN]);


	if (measure_time==1) dtime(&Stime1);

	Krylov_IOLP_trd( Mc_AN, OLP0, Krylov_U_OLP, OpenMX_Solver_RealMatrix(solver_owner,"inv_RS",inv_RS.extent[0],inv_RS.extent[1]), MP, Msize2, Msize4, Msize2_max, OpenMX_Solver_RealMatrix(solver_owner,"tmpvec0",tmpvec0.extent[0],tmpvec0.extent[1]), OpenMX_Solver_RealMatrix(solver_owner,"tmpvec1",tmpvec1.extent[0],tmpvec1.extent[1]) );

	if (measure_time==1){ 
	  dtime(&Etime1);
	  time2 += Etime1 - Stime1;      
	}
      }

      for (spin=0; spin<=SpinP_switch; spin++){

	/****************************************************
                generate a preconditioning matrix
	****************************************************/

	if (measure_time==1) dtime(&Stime1);

	if (SCF_iter==1 && Msize3[Mc_AN]<Msize2[Mc_AN]){

	  Generate_pMatrix_trd(state,  myid, spin, Mc_AN, Hks, OLP0, OpenMX_Solver_RealMatrix(solver_owner,"invS",invS.extent[0],invS.extent[1]), Krylov_U, Krylov_U_OLP, OpenMX_Solver_RealMatrix(solver_owner,"inv_RS",inv_RS.extent[0],inv_RS.extent[1]), MP,
                            Msize, Msize2, Msize3, Msize4, Msize2_max, OpenMX_Solver_RealMatrix(solver_owner,"tmpvec0",tmpvec0.extent[0],tmpvec0.extent[1]), OpenMX_Solver_RealMatrix(solver_owner,"tmpvec1",tmpvec1.extent[0],tmpvec1.extent[1]), OpenMX_Solver_RealMatrix(solver_owner,"tmpvec2",tmpvec2.extent[0],tmpvec2.extent[1]) );
	}
	else if (SCF_iter==1){

	  Generate_pMatrix2_trd( myid, spin, Mc_AN, Hks, OLP0, Krylov_U, MP, Msize, Msize2, Msize3, OpenMX_Solver_RealMatrix(solver_owner,"tmpvec1",tmpvec1.extent[0],tmpvec1.extent[1]) );
	}

	if (measure_time==1){ 
	  dtime(&Etime1);
	  time3 += Etime1 - Stime1;      
	}

	if (measure_time==1) dtime(&Stime1);

	if (recalc_EM==1 || SCF_iter<=3 || state->Col_trd_owned_recalc_flag==1){

	  Embedding_Matrix_trd( spin, Mc_AN, Hks, Krylov_U, EC_matrix, MP, Msize, Msize2, Msize3, OpenMX_Solver_RealMatrix(solver_owner,"tmpvec1",tmpvec1.extent[0],tmpvec1.extent[1]), EKC_core_size_max, Msize2_max);
	}

	if (measure_time==1){ 
	  dtime(&Etime1);
	  time4 += Etime1 - Stime1;      
	}

	/****************************************************
                construct the Hamiltonian matrix
	****************************************************/

	if (measure_time==1) dtime(&Stime1);
        
	for (i=0; i<=FNAN[Gc_AN]; i++){
	  ig = natn[Gc_AN][i];
	  ian = Spe_Total_CNO[WhatSpecies[ig]];
	  Anum = MP[i] - 1;
	  ih = S_G2M[ig]; /* S_G2M should be used */

	  for (j=0; j<=FNAN[Gc_AN]; j++){

	    kl = RMI1[Mc_AN][i][j];
	    jg = natn[Gc_AN][j];
	    jan = Spe_Total_CNO[WhatSpecies[jg]];
	    Bnum = MP[j] - 1;

	    if (0<=kl){
	      for (m=0; m<ian; m++){
		for (n=0; n<jan; n++){
		  H_DC[(Anum+m)*Msize[Mc_AN]+Bnum+n] = Hks[spin][ih][kl][m][n];
		}
	      }
	    }

	    else{
	      for (m=0; m<ian; m++){
		for (n=0; n<jan; n++){
		  H_DC[(Anum+m)*Msize[Mc_AN]+Bnum+n] = 0.0;
		}
	      }
	    }
	  }
	}

	if (measure_time==1){ 
	  dtime(&Etime1);
	  time5 += Etime1 - Stime1;      
	}

	/****************************************************
                   transform u1^+ * H_DC * u1
	****************************************************/

	/* H_DC * u1 */

	if (measure_time==1) dtime(&Stime1);

	/* original version

	for (i=1; i<=Msize[Mc_AN]; i++){
	  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (j=1; j<=Msize[Mc_AN]; j++){
		sum += H_DC[i][j]*Krylov_U[spin][Mc_AN][rl][n][j];  
	      }

	      C[rl*EKC_core_size[Mc_AN]+n+1][i] = sum;
	    }      
	  }      
	} 
	*/

	/* BLAS3 version */

	if (Krylov_ProjectedSolve_GPU(state, gpu_ws,spin,Mc_AN,Msize,Msize2,Msize3,
	                              Krylov_U,EC_matrix,H_DC,KU,C,ko)==0){

	for (i=0; i<Msize[Mc_AN]; i++){
	  for (j=0; j<Msize3[Mc_AN]; j++){
	    KU[j*Msize[Mc_AN]+i] = Krylov_U[spin][Mc_AN][j*Msize2[Mc_AN]+i+1];
	  }
	}

	alpha = 1.0; beta = 0.0; 
	M = Msize[Mc_AN]; N = Msize3[Mc_AN]; K = Msize[Mc_AN];
	lda = M; ldb = K; ldc = Msize[Mc_AN]; 
	
	Krylov_Dgemm(state, gpu_ws, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K,
                     H_DC, lda, KU, ldb, C, ldc);

	if (measure_time==1){ 
	  dtime(&Etime1);
	  time6 += Etime1 - Stime1;      
	}

	/* u1^+ * H_DC * u1 */

	if (measure_time==1) dtime(&Stime1);

	/* original version */

	/*
	for (rl1=0; rl1<rlmax_EC[Mc_AN]; rl1++){                                                     
          for (m=0; m<EKC_core_size[Mc_AN]; m++){                                                
	    for (rl2=rl1; rl2<rlmax_EC[Mc_AN]; rl2++){                            
	      for (n=0; n<EKC_core_size[Mc_AN]; n++){                                      
		sum = 0.0;                                                                              
		i2 = rl2*EKC_core_size[Mc_AN] + n + 1;                                                  
		for (i=1; i<=Msize[Mc_AN]; i++){                                                    
		  sum += Krylov_U[spin][Mc_AN][rl1][m][i]*C[i2][i];                          
		}                                                                                  

		H_DC[rl1*EKC_core_size[Mc_AN]+m+1][rl2*EKC_core_size[Mc_AN]+n+1] = sum;                 
		H_DC[rl2*EKC_core_size[Mc_AN]+n+1][rl1*EKC_core_size[Mc_AN]+m+1] = sum;          
	      }                                                                                         
	    }                                                                              
          }                                                                               
	}                                            
	*/

	/* BLAS3 version */

	alpha = 1.0; beta = 0.0; 
	M = Msize3[Mc_AN]; N = Msize3[Mc_AN]; K = Msize[Mc_AN];
	lda = K; ldb = K; ldc = Msize3[Mc_AN]; 

	Krylov_Dgemm(state, gpu_ws, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K,
                     KU, lda, C, ldb, H_DC, ldc);

	if (measure_time==1){ 
	  dtime(&Etime1);
	  time7 += Etime1 - Stime1;      
	}

	/* correction for ZeroNum */

	m = (int)Krylov_U[spin][Mc_AN][0];
	for (i=0; i<m; i++){
	  H_DC[i*Msize3[Mc_AN]+i] = 1.0e+3;
	}

	/****************************************************
            H0 = u1^+ * H_DC * u1 + D 
	****************************************************/

	if (measure_time==1) dtime(&Stime1);

	for (i=(Msize3[Mc_AN]-1); 0<=i; i--){
	  for (j=0; j<Msize3[Mc_AN]; j++){
	    H_DC[(i+1)*(Msize3[Mc_AN]+1)+(j+1)] = H_DC[i*Msize3[Mc_AN]+j] + EC_matrix[spin][Mc_AN][i+1][j+1];
	  }
	}

	if (measure_time==1){ 
	  dtime(&Etime1);
	  time8 += Etime1 - Stime1;      
	}

	/****************************************************
           diagonalize
	****************************************************/

	if (measure_time==1) dtime(&Stime1);

	Krylov_Eigen2(state, gpu_ws,H_DC,Msize3[Mc_AN]+1,ko,Msize3[Mc_AN],Msize3[Mc_AN]);

	if (measure_time==1){ 
	  dtime(&Etime1);
	  time9 += Etime1 - Stime1;      
	}

	/********************************************
           back transformation of eigenvectors
                      c = u1 * b
	*********************************************/

	if (measure_time==1) dtime(&Stime1);

	/* original version */

	/*
	for (i=1; i<=Msize[Mc_AN]; i++){                                                              
          for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){                                                   
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){  

	      tmp1 = Krylov_U[spin][Mc_AN][rl][n][i]; 
	      i1 = rl*EKC_core_size[Mc_AN] + n + 1; 

	      for (j=1; j<=Msize3[Mc_AN]; j++){  
		C[i][j] += tmp1*H_DC[i1][j];                                                     
	      }
	    }   
          }                                                                                           
	}                                                                                              
	*/

	/* BLAS3 version */

	alpha = 1.0; beta = 0.0; 
        M = Msize3[Mc_AN]; N = Msize[Mc_AN]; K = Msize3[Mc_AN];
        lda = K; ldb = N; ldc = Msize3[Mc_AN]; 

	Krylov_Dgemm(state, gpu_ws, HIPBLAS_OP_T, HIPBLAS_OP_T, M, N, K,
                     H_DC, lda, KU, ldb, C, ldc);

	} /* fallback of fused GPU projected solve */

	if (measure_time==1){ 
	  dtime(&Etime1);
	  time10 += Etime1 - Stime1;      
	}

	if (measure_time==1) dtime(&Stime1);

	/***********************************************
          store eigenvalues and residues of poles
	***********************************************/

	{
	  double *eval_row_20=openmx_solver_field_row(&EVal,spin,Mc_AN);
	  for (i=1; i<=Msize3[Mc_AN]; i++){
	  eval_row_20[i-1] = ko[i];
	}
	}

	/******************************************************
        set an energy range (-Erange+ChemP to Erange+ChemP)
        of eigenvalues used to store the Residues.
	******************************************************/

	Erange = 0.367493245;  /* in hartree, corresponds to 10 eV */

	/***********************************************
          find LO_TC and HO_TC
	***********************************************/

	/* LO_TC */ 
	i = 0;
	ip = 0;
	po1 = 0;
	do{
	  if ( (state->chemical->chemical_potential-Erange)<openmx_solver_field_row(&EVal,spin,Mc_AN)[i]){
	    ip = i;
	    po1 = 1; 
	  }
	  i++;
	} while (po1==0 && i<Msize3[Mc_AN]);

	openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN] = ip;

	/* HO_TC */ 
	i = 0;
	ip = Msize3[Mc_AN]-1;
	po1 = 0;
	do{
	  if ( (state->chemical->chemical_potential+Erange)<openmx_solver_field_row(&EVal,spin,Mc_AN)[i]){
	    ip = i;
	    po1 = 1; 
	  }
	  i++;
	} while (po1==0 && i<Msize3[Mc_AN]);

	openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN] = ip;

	/***********************************************
          store residues of poles
	***********************************************/

	n2 = openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN] - openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN] + 3;
	if (n2<1) n2 = 1;
	OpenMX_Solver_ResidueResize(Residues,spin,Mc_AN,n2);

	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];

	{
	  double *eval_row_19=openmx_solver_field_row(&EVal,spin,Mc_AN);
	  for (i=0; i<tno1; i++){
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
	    wanB = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wanB];
	    Bnum = MP[h_AN] - 1;
	    for (j=0; j<tno2; j++){

	      for (i1=0; i1<openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]; i1++){
		tmp1 = C[i*Msize3[Mc_AN]+i1]*C[(Bnum+j)*Msize3[Mc_AN]+i1];
		CDM[spin][Mc_AN][h_AN][i][j] += tmp1;
		EDM[spin][Mc_AN][h_AN][i][j] += tmp1*eval_row_19[i1];
	      }      
      

	      {
	        double *residues_row_15=openmx_solver_residue_row(&Residues,spin,Mc_AN,h_AN,i,j);
	        for (i1=openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]; i1<=openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN]; i1++){
		residues_row_15[i1-openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]]
                          = C[i*Msize3[Mc_AN]+i1]*C[(Bnum+j)*Msize3[Mc_AN]+i1];

	      }
	      }
	    }
	  }
	}
	}

	if (measure_time==1){ 
	  dtime(&Etime1);
	  time11 += Etime1 - Stime1;      
	}

      } /* spin */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

  } /* #pragma omp parallel */

  if (measure_time==1){ 
    dtime(&Etime2);
    time16 = Etime2 - Stime2;      
  }

  if (state->Col_trd_owned_firsttime){

    Residues_size = 1;

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];
	wan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wanB = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wanB];
	  n2 = openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN] - openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN] + 3;
	  Residues_size += tno1*tno2*n2;
	}
      }
    }

    PrintMemory("Krylov: Residues",sizeof(double)*Residues_size,NULL);
  }

  /****************************************************
                calculate the projected DOS
  ****************************************************/

  if (measure_time==1) dtime(&Stime1);

  /*
#pragma omp parallel shared(time_per_atom,Residues,LO_TC,HO_TC,EDM,CDM,OLP0,natn,FNAN,PDOS_DC,Msize3,Spe_Total_CNO,WhatSpecies,M2G,SpinP_switch,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,spin,Gc_AN,wanA,tno1,i1,i,h_AN,Gh_AN,wanB,tno2,j,tmp1,Etime_atom)
  */

  {

    /* get info. on OpenMP */ 

    /*
    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();
    */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      for (spin=0; spin<=SpinP_switch; spin++){

	Gc_AN = M2G[Mc_AN];
	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];

	{
	  double *pdos_dc_row_14=openmx_solver_field_row(&PDOS_DC,spin,Mc_AN);
	  for (i1=0; i1<=(Msize3[Mc_AN]+1); i1++){
	  pdos_dc_row_14[i1] = 0.0;
	}
	}

	{
	  double *pdos_dc_row_13=openmx_solver_field_row(&PDOS_DC,spin,Mc_AN);
	  for (i=0; i<tno1; i++){

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wanB = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wanB];
	    for (j=0; j<tno2; j++){

	      tmp1 = OLP0[Mc_AN][h_AN][i][j];

	      pdos_dc_row_13[0] += tmp1*CDM[spin][Mc_AN][h_AN][i][j];
	      pdos_dc_row_13[1] += tmp1*EDM[spin][Mc_AN][h_AN][i][j];

	      {
	        double *residues_row_10=openmx_solver_residue_row(&Residues,spin,Mc_AN,h_AN,i,j);
	        for (i1=0; i1<(openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN]-openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]+1); i1++){
		pdos_dc_row_13[i1+2] += residues_row_10[i1]*tmp1;
	      }
	      }

	    }            
	  }        
	}
	}

      } /* spin */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

  } /* #pragma omp parallel */

  if (measure_time==1){ 
    dtime(&Etime1);
    time12 += Etime1 - Stime1;      
  }

  /****************************************************
           find the total number of electrons 
  ****************************************************/

  if (measure_time==1) dtime(&Stime1);

  My_TZ = 0.0;
  for (i=1; i<=Matomnum; i++){
    Gc_AN = M2G[i];
    wan = WhatSpecies[Gc_AN];
    My_TZ += Spe_Core_Charge[wan];
  }

  /* MPI, My_TZ */

  MPI_Barrier(mpi_comm_level1);
  MPI_Allreduce(&My_TZ, &state->Col_trd_owned_TZ, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /****************************************************
                find the chemical potential
  ****************************************************/

  po = 0;
  loopN = 0;

  ChemP_MAX = 10.0;  
  ChemP_MIN =-10.0;
  if      (SpinP_switch==0) spin_degeneracy = 2.0;
  else if (SpinP_switch==1) spin_degeneracy = 1.0;

  do {
    state->chemical->chemical_potential = 0.50*(ChemP_MAX + ChemP_MIN);

    My_Num_State = 0.0;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      for (spin=0; spin<=SpinP_switch; spin++){

        dtime(&Stime_atom);

        Gc_AN = M2G[Mc_AN];

        My_Num_State += spin_degeneracy*openmx_solver_field_row(&PDOS_DC,spin,Mc_AN)[0];

        {
          double *eval_row_7=openmx_solver_field_row(&EVal,spin,Mc_AN);
          double *pdos_dc_row_8=openmx_solver_field_row(&PDOS_DC,spin,Mc_AN);
          for (i=0; i<(openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN]-openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]+1); i++){

          x = (eval_row_7[i+openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]] - state->chemical->chemical_potential)*Beta;
          if (x<=-max_x) x = -max_x;
          if (max_x<=x)  x = max_x;
          FermiF = 1.0/(1.0 + exp(x));
          My_Num_State += spin_degeneracy*FermiF*pdos_dc_row_8[i+2];
	}
        }

        dtime(&Etime_atom);
        time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      }
    }

    /* MPI, My_Num_State */

    MPI_Barrier(mpi_comm_level1);
    MPI_Allreduce(&My_Num_State, &Num_State, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    Dnum = (state->Col_trd_owned_TZ - Num_State) - system_charge;
    if (0.0<=Dnum) ChemP_MIN = state->chemical->chemical_potential;
    else           ChemP_MAX = state->chemical->chemical_potential;
    if (fabs(Dnum)<1.0e-11) po = 1;


    if (myid==Host_ID && 3<=level_stdout){
      printf("  ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",state->chemical->chemical_potential,state->Col_trd_owned_TZ,Num_State);
    }

    loopN++;
  }
  while (po==0 && loopN<1000); 

  if (measure_time==1){ 
    dtime(&Etime1);
    time13 += Etime1 - Stime1;      
  }

  /****************************************************
        eigenenergy by summing up eigenvalues
  ****************************************************/

  if (measure_time==1) dtime(&Stime1);

  My_Eele0[0] = 0.0;
  My_Eele0[1] = 0.0;
  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      My_Eele0[spin] += openmx_solver_field_row(&PDOS_DC,spin,Mc_AN)[1];

      {
        double *eval_row_5=openmx_solver_field_row(&EVal,spin,Mc_AN);
        double *pdos_dc_row_6=openmx_solver_field_row(&PDOS_DC,spin,Mc_AN);
        for (i=0; i<(openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN]-openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]+1); i++){

        x = (eval_row_5[i+openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]] - state->chemical->chemical_potential)*Beta;

        if (x<=-max_x) x = -max_x;
        if (max_x<=x)  x = max_x;
        FermiF = 1.0/(1.0 + exp(x));
        My_Eele0[spin] += FermiF*eval_row_5[i+openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]]*pdos_dc_row_6[i+2];
      }
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }
  }

  /* MPI, My_Eele0 */
  for (spin=0; spin<=SpinP_switch; spin++){
    MPI_Barrier(mpi_comm_level1);
    MPI_Allreduce(&My_Eele0[spin], &Eele0[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  }

  if (SpinP_switch==0){
    Eele0[1] = Eele0[0];
  }

  if (measure_time==1){ 
    dtime(&Etime1);
    time14 += Etime1 - Stime1;      
  }

  if (measure_time==1) dtime(&Stime1);

  /*
#pragma omp parallel shared(time_per_atom,EDM,CDM,Residues,natn,max_x,Beta,ChemP,EVal,LO_TC,HO_TC,Spe_Total_CNO,WhatSpecies,M2G,SpinP_switch,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,spin,Stime_atom,Gc_AN,wanA,tno1,i1,x,FermiF,h_AN,Gh_AN,wanB,tno2,i,j,tmp1,Etime_atom)
  */

  {

    /* get info. on OpenMP */ 

    /*
    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();
    */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      for (spin=0; spin<=SpinP_switch; spin++){

	dtime(&Stime_atom);

	Gc_AN = M2G[Mc_AN];
	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];

	{
	  double *eval_row_4=openmx_solver_field_row(&EVal,spin,Mc_AN);
	  for (i1=0; i1<(openmx_solver_int_matrix_row(&HO_TC,spin)[Mc_AN]-openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]+1); i1++){

	  x = (eval_row_4[i1+openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]] - state->chemical->chemical_potential)*Beta;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;
	  FermiF = 1.0/(1.0 + exp(x));

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wanB = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wanB];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		tmp1 = FermiF*openmx_solver_residue_row(&Residues,spin,Mc_AN,h_AN,i,j)[i1];
		CDM[spin][Mc_AN][h_AN][i][j] += tmp1;
		EDM[spin][Mc_AN][h_AN][i][j] += tmp1*eval_row_4[i1+openmx_solver_int_matrix_row(&LO_TC,spin)[Mc_AN]];
	      }
	    }
	  }
	}
	}

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      }
    }

  } /* #pragma omp parallel */

  /****************************************************
                      bond energies
  ****************************************************/

  My_Eele1[0] = 0.0;
  My_Eele1[1] = 0.0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];    
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    for (j=0; j<=FNAN[GA_AN]; j++){
      GB_AN = natn[GA_AN][j];  
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];

      for (k=0; k<tnoA; k++){
	for (l=0; l<tnoB; l++){
	  for (spin=0; spin<=SpinP_switch; spin++){
	    My_Eele1[spin] += CDM[spin][MA_AN][j][k][l]*Hks[spin][MA_AN][j][k][l];
	  }
	}
      }
  
    }
  }

  /* MPI, My_Eele1 */
  MPI_Barrier(mpi_comm_level1);
  for (spin=0; spin<=SpinP_switch; spin++){
    MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE,
                   MPI_SUM, mpi_comm_level1);
  }

  if (SpinP_switch==0){
    Eele1[1] = Eele1[0];
  }

  if (3<=level_stdout && myid==Host_ID){
    printf("  Eele00=%15.12f Eele01=%15.12f\n",Eele0[0],Eele0[1]);
    printf("  Eele10=%15.12f Eele11=%15.12f\n",Eele1[0],Eele1[1]);
  }

  if (measure_time==1){ 
    dtime(&Etime1);
    time15 += Etime1 - Stime1;      
  }

  if ( strcasecmp(mode,"dos")==0 ){
    Save_DOS_Col(state->chemical, Residues,OLP0,EVal,LO_TC,HO_TC);
  }

  if (measure_time==1){ 
    printf("myid=%2d time1 =%5.3f time2 =%5.3f time3 =%5.3f time4 =%5.3f time5 =%5.3f\n",
            myid,time1,time2,time3,time4,time5);
    printf("myid=%2d time6 =%5.3f time7 =%5.3f time8 =%5.3f time9 =%5.3f time10=%5.3f\n",
            myid,time6,time7,time8,time9,time10);
    printf("myid=%2d time11=%5.3f time12=%5.3f time13=%5.3f time14=%5.3f time15=%5.3f\n",
            myid,time11,time12,time13,time14,time15);
    printf("myid=%2d time16=%5.3f\n",myid,time16);
  }

  /* The C++ invocation releases all per-solve workspaces. */
  FNAN[0] = 0;

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  if (measure_time==1){ 
  printf("total time=%15.12f\n",time0);
  }

  /* for PrintMemory */
  state->Col_trd_owned_firsttime=0;

  return time0;
}









typedef struct {
  KrylovState *state;
  int myid;
  int spin;
  int Mc_AN;
  double *****Hks;
  double ****OLP0;
  double **invS;
  double ***Krylov_U;
  double ***Krylov_U_OLP;
  double **inv_RS;
  int *MP;
  int *Msize;
  int *Msize2;
  int *Msize3;
  int *Msize4;
  int Msize2_max;
  double **tmpvec0;
  double **tmpvec1;
  double **tmpvec2;
} Generate_pMatrix_trd_scratch_arguments;
static void Generate_pMatrix_trd_raii(KrylovState *state,  int myid, int spin, int Mc_AN, double *****Hks,
                       double ****OLP0, double **invS,
                       double ***Krylov_U, double ***Krylov_U_OLP, double **inv_RS, int *MP,
                       int *Msize, int *Msize2, int *Msize3, int *Msize4, int Msize2_max,
                       double **tmpvec0, double **tmpvec1, double **tmpvec2 ,OpenMXSolverInvocation *solver_owner);
static int Generate_pMatrix_trd_scratch_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Generate_pMatrix_trd_scratch_arguments *a=(Generate_pMatrix_trd_scratch_arguments*)arguments;
  Generate_pMatrix_trd_raii(a->state, a->myid,a->spin,a->Mc_AN,a->Hks,a->OLP0,a->invS,a->Krylov_U,a->Krylov_U_OLP,a->inv_RS,a->MP,a->Msize,a->Msize2,a->Msize3,a->Msize4,a->Msize2_max,a->tmpvec0,a->tmpvec1,a->tmpvec2,owner);
  *result=0;
  return 1;
}
void Generate_pMatrix_trd(KrylovState *state,  int myid, int spin, int Mc_AN, double *****Hks,
                       double ****OLP0, double **invS,
                       double ***Krylov_U, double ***Krylov_U_OLP, double **inv_RS, int *MP, 
                       int *Msize, int *Msize2, int *Msize3, int *Msize4, int Msize2_max, 
                       double **tmpvec0, double **tmpvec1, double **tmpvec2 )
{
  Generate_pMatrix_trd_scratch_arguments arguments={state,myid,spin,Mc_AN,Hks,OLP0,invS,Krylov_U,Krylov_U_OLP,inv_RS,MP,Msize,Msize2,Msize3,Msize4,Msize2_max,tmpvec0,tmpvec1,tmpvec2};
  OpenMX_Solver_Run(Generate_pMatrix_trd_scratch_operation,&arguments);
}

static void Generate_pMatrix_trd_raii(KrylovState *state,  int myid, int spin, int Mc_AN, double *****Hks,
                       double ****OLP0, double **invS,
                       double ***Krylov_U, double ***Krylov_U_OLP, double **inv_RS, int *MP,
                       int *Msize, int *Msize2, int *Msize3, int *Msize4, int Msize2_max,
                       double **tmpvec0, double **tmpvec1, double **tmpvec2 ,OpenMXSolverInvocation *solver_owner)
{
  int rl,rl0,rl1,ct_AN,fan,san,can,wan,ct_on,i,j;
  int n,Anum,Bnum,k,ian,ih,kl,jg,ig,jan,m,m1,n1;
  int ZeroNum,Gh_AN,wanB,m1s,is,info;
  int rl00,rl01,rl02,rl03,rl04,rl05,rl06,rl07;
  int mm0,mm1,mm2,mm3,mm4,mm5,mm6,mm7;

  int KU_d1, KU_d2, csize; 
  __m128d mmSum00,mmSum01,mmSum10,mmSum11,mmSum20,mmSum21,mmSum30,mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5; 

  double mmArr[8];
  double time1,time2,time3,time4,time5;
  double time6,time7,time8,time9,time10;
  double Stime1,Etime1;
  double sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7;
  double sum,dum,tmp0,tmp1,tmp2,tmp3,rcutA,r0;
  double **Utmp,**matRS0,**matRS1;
  double **tmpmat0;
  double *ko,*iko;
  double **FS;
  double ***U0;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;
 
  ct_AN = M2G[Mc_AN];
  fan = FNAN[ct_AN];
  san = SNAN[ct_AN];
  can = fan + san;
  wan = WhatSpecies[ct_AN];
  ct_on = Spe_Total_CNO[wan];
  rcutA = Spe_Atom_Cut1[wan]; 

  if (Msize[Mc_AN]<Msize3[Mc_AN])
    csize = Msize3[Mc_AN] + 40;
  else
    csize = Msize[Mc_AN] + 40;

  KU_d1 = EKC_core_size[Mc_AN]*Msize2[Mc_AN];
  KU_d2 = Msize2[Mc_AN];

  /* allocation of arrays */

  Utmp=OpenMX_Solver_AlignedRealMatrix(solver_owner,"Utmp-1",((size_t)1)*rlmax_EC[Mc_AN],((size_t)1)*EKC_core_size[Mc_AN]);
  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
  }

  U0=OpenMX_Solver_AlignedRealVolume(solver_owner,"U0-2",((size_t)1)*rlmax_EC[Mc_AN],((size_t)1)*EKC_core_size[Mc_AN],((size_t)1)*(Msize2[Mc_AN]+3));
  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
    for (j=0; j<EKC_core_size[Mc_AN]; j++){
      ;
      for (k=0; k<(Msize2[Mc_AN]+3); k++)  U0[i][j][k] = 0.0;
    }
  }  

  tmpmat0=OpenMX_Solver_AlignedRealMatrix(solver_owner,"tmpmat0-3",((size_t)1)*(EKC_core_size[Mc_AN]+4),((size_t)1)*(EKC_core_size[Mc_AN]+4));
  for (i=0; i<(EKC_core_size[Mc_AN]+4); i++){
    ;
  }

  FS=OpenMX_Solver_AlignedRealMatrix(solver_owner,"FS-4",((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN],((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  for (i=0; i<(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]; i++){
    ;
  }

  ko=(double*)OpenMX_Solver_Real(solver_owner,"ko-9",((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  iko=(double*)OpenMX_Solver_Real(solver_owner,"iko-10",((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]);

  matRS0=OpenMX_Solver_AlignedRealMatrix(solver_owner,"matRS0-5",((size_t)1)*(EKC_core_size[Mc_AN]+2),((size_t)1)*(Msize4[Mc_AN]+3));
  for (i=0; i<(EKC_core_size[Mc_AN]+2); i++){
    ;
  }

  matRS1=OpenMX_Solver_AlignedRealMatrix(solver_owner,"matRS1-6",((size_t)1)*(Msize4[Mc_AN]+3),((size_t)1)*(EKC_core_size[Mc_AN]+2));
  for (i=0; i<(Msize4[Mc_AN]+3); i++){
    ;
  }

  /****************************************************
      initialize 
  ****************************************************/

  if (measure_time==1){ 
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
  }

  if (measure_time==1) dtime(&Stime1);

  for (i=0; i<EKC_core_size_max; i++){
    for (j=0; j<Msize2_max; j++){
      tmpvec0[i][j] = 0.0;
      tmpvec1[i][j] = 0.0;
    }
  }

  /* find the nearest atom with distance of r0 */   

  r0 = 1.0e+10;
  for (k=1; k<=FNAN[ct_AN]; k++){
    Gh_AN = natn[ct_AN][k];
    wanB = WhatSpecies[Gh_AN];
    if (Dis[ct_AN][k]<r0)  r0 = Dis[ct_AN][k]; 
  }

  /* starting vector */

  m = 0;  
  for (k=0; k<=FNAN[ct_AN]; k++){

    Gh_AN = natn[ct_AN][k];
    wanB = WhatSpecies[Gh_AN];

    if ( Dis[ct_AN][k]<(scale_rc_EKC[Mc_AN]*r0) ){

      Anum = MP[k] - 1; 

      for (i=0; i<Spe_Total_CNO[wanB]; i++){

        tmpvec0[m][Anum+i] = 1.0;

        m++;  
      }
    }
  }

  S_orthonormalize_vec_trd( Mc_AN, ct_on, tmpvec0, tmpvec1, OLP0, tmpmat0, ko, iko, MP, Msize2 );

  for (n=0; n<EKC_core_size[Mc_AN]; n++){
    for (i=0; i<Msize2[Mc_AN]; i++){
      U0[0][n][i] = tmpvec0[n][i];
    }
  }

  if (measure_time==1){ 
    dtime(&Etime1);
    time1 = Etime1 - Stime1;      
  }

  /****************************************************
           generate Krylov subspace vectors
  ****************************************************/

  for (rl=0; rl<(rlmax_EC[Mc_AN]-1); rl++){

    if (measure_time==1) dtime(&Stime1);

    /*******************************************************
                            H * |Wn)
    *******************************************************/

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=0; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=0; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
        Bnum = MP[j] - 1;

        if (0<=kl){

#ifdef nosse

      /* original version */
	  	  
      	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += Hks[spin][ih][kl][m][k]*tmpvec0[n][Bnum+k];
	      }

              tmpvec1[n][Anum+m] += sum;
	    }
	  }
	  
#else
      /* Loop Unrolling + SSE version */
	  
	  for (m=0; m<(ian-3); m+=4){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      mmSum00 = _mm_setzero_pd();
	      mmSum01 = _mm_setzero_pd();
	      mmSum10 = _mm_setzero_pd();
	      mmSum11 = _mm_setzero_pd();
	      mmSum20 = _mm_setzero_pd();
	      mmSum21 = _mm_setzero_pd();
	      mmSum30 = _mm_setzero_pd();
	      mmSum31 = _mm_setzero_pd();

	      for (k=0; k<(jan-3); k+=4){
		mmTmp0 = _mm_loadu_pd(&tmpvec0[n][Bnum+k+0]);
		mmTmp1 = _mm_loadu_pd(&tmpvec0[n][Bnum+k+2]);

		mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+0][k+0]),mmTmp0));  
		mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+0][k+2]),mmTmp1));  

		mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+1][k+0]),mmTmp0));  
		mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+1][k+2]),mmTmp1));  

		mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+2][k+0]),mmTmp0));  
		mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+2][k+2]),mmTmp1));  

		mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+3][k+0]),mmTmp0));  
		mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+3][k+2]),mmTmp1));  
	      }

	      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	      _mm_storeu_pd(&mmArr[0], mmSum00);
	      _mm_storeu_pd(&mmArr[2], mmSum10);
	      _mm_storeu_pd(&mmArr[4], mmSum20);
	      _mm_storeu_pd(&mmArr[6], mmSum30);

	      sum0 = mmArr[0] + mmArr[1];
	      sum1 = mmArr[2] + mmArr[3];
	      sum2 = mmArr[4] + mmArr[5];
	      sum3 = mmArr[6] + mmArr[7];

	      for (; k<jan; k++){
		sum0 += Hks[spin][ih][kl][m+0][k]*tmpvec0[n][Bnum+k];
		sum1 += Hks[spin][ih][kl][m+1][k]*tmpvec0[n][Bnum+k];
		sum2 += Hks[spin][ih][kl][m+2][k]*tmpvec0[n][Bnum+k];
		sum3 += Hks[spin][ih][kl][m+3][k]*tmpvec0[n][Bnum+k];
	      }

              tmpvec1[n][Anum+m+0] += sum0;
              tmpvec1[n][Anum+m+1] += sum1;
              tmpvec1[n][Anum+m+2] += sum2;
              tmpvec1[n][Anum+m+3] += sum3;
	    }
	  }

	  for (; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += Hks[spin][ih][kl][m][k]*tmpvec0[n][Bnum+k];
	      }

              tmpvec1[n][Anum+m] += sum;
	    }
	  } 
#endif	   
	}
      }
    }

    if (measure_time==1){ 
      dtime(&Etime1);
      time2 += Etime1 - Stime1;      
    }

    /*******************************************************
           S^{-1} * H * |Wn)
    *******************************************************/

    if (EKC_Exact_invS_flag==1){ 

      if (measure_time==1) dtime(&Stime1);

#ifdef nosse

      /* original version */
      /*      
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
        for (i=0; i<Msize2[Mc_AN]; i++){
	  sum = 0.0;
	  for (j=0; j<Msize2[Mc_AN]; j++){
	    sum += invS[i][j]*tmpvec1[n][j];
	  }
	  tmpvec0[n][i] = sum;
        } 
      }
      */
      /* unrolling version */

      for (i=0; i<(Msize2[Mc_AN]-3); i+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  sum0 = 0.0;
	  sum1 = 0.0;
	  sum2 = 0.0;
	  sum3 = 0.0;

	  for (j=0; j<Msize2[Mc_AN]; j++){
	    sum0 += invS[i+0][j]*tmpvec1[n][j];
	    sum1 += invS[i+1][j]*tmpvec1[n][j];
	    sum2 += invS[i+2][j]*tmpvec1[n][j];
	    sum3 += invS[i+3][j]*tmpvec1[n][j];
	  }

	  tmpvec0[n][i+0] = sum0;
	  tmpvec0[n][i+1] = sum1;
	  tmpvec0[n][i+2] = sum2;
	  tmpvec0[n][i+3] = sum3;
        } 
      }

      is = Msize2[Mc_AN] - Msize2[Mc_AN]%4;

      for (i=is; i<Msize2[Mc_AN]; i++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  sum = 0.0;
	  for (j=0; j<Msize2[Mc_AN]; j++){
	    sum += invS[i][j]*tmpvec1[n][j];
	  }
	  tmpvec0[n][i] = sum;
        } 
      }

#else
      /* unrolling + SSE version */
      
      for (i=0; i<(Msize2[Mc_AN]-3); i+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  mmSum00 = _mm_setzero_pd();
	  mmSum01 = _mm_setzero_pd();
	  mmSum10 = _mm_setzero_pd();
	  mmSum11 = _mm_setzero_pd();
	  mmSum20 = _mm_setzero_pd();
	  mmSum21 = _mm_setzero_pd();
	  mmSum30 = _mm_setzero_pd();
	  mmSum31 = _mm_setzero_pd();

	  for (j=0; j<(Msize2[Mc_AN]-3); j+=4){
	    mmTmp0 = _mm_loadu_pd(&tmpvec1[n][j+0]);
	    mmTmp1 = _mm_loadu_pd(&tmpvec1[n][j+2]);

	    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&invS[i+0][j+0]),mmTmp0));  
	    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&invS[i+0][j+2]),mmTmp1));  

	    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&invS[i+1][j+0]),mmTmp0));  
	    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&invS[i+1][j+2]),mmTmp1));  

	    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&invS[i+2][j+0]),mmTmp0));  
	    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&invS[i+2][j+2]),mmTmp1));  

	    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&invS[i+3][j+0]),mmTmp0));  
	    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&invS[i+3][j+2]),mmTmp1));  
	  }

	  mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	  mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	  mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	  mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	  _mm_storeu_pd(&mmArr[0], mmSum00);
	  _mm_storeu_pd(&mmArr[2], mmSum10);
	  _mm_storeu_pd(&mmArr[4], mmSum20);
	  _mm_storeu_pd(&mmArr[6], mmSum30);

	  sum0 = mmArr[0] + mmArr[1];
	  sum1 = mmArr[2] + mmArr[3];
	  sum2 = mmArr[4] + mmArr[5];
	  sum3 = mmArr[6] + mmArr[7];

	  for (; j<Msize2[Mc_AN]; j++){
	    sum0 += invS[i+0][j]*tmpvec1[n][j];
	    sum1 += invS[i+1][j]*tmpvec1[n][j];
	    sum2 += invS[i+2][j]*tmpvec1[n][j];
	    sum3 += invS[i+3][j]*tmpvec1[n][j];
	  }

	  tmpvec0[n][i+0] = sum0;
	  tmpvec0[n][i+1] = sum1;
	  tmpvec0[n][i+2] = sum2;
	  tmpvec0[n][i+3] = sum3;
        } 
      }
                  
      is = Msize2[Mc_AN] - Msize2[Mc_AN]%4;

      for (i=is; i<Msize2[Mc_AN]; i++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  sum = 0.0;
	  for (j=0; j<Msize2[Mc_AN]; j++){
	    sum += invS[i][j]*tmpvec1[n][j];
	  }
	  tmpvec0[n][i] = sum;
        } 
      }
      
#endif      

      if (measure_time==1){ 
	dtime(&Etime1);
	time3 += Etime1 - Stime1;      
      }
    }

    /*******************************************************
          U * RS^-1 * U^+ * H * |Wn)
    *******************************************************/

    else if (EKC_invS_flag==1){ 

      /*  U^+ * H * |Wn) */

      for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
	for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  for (m=0; m<EKC_core_size[Mc_AN]; m++){

	    sum = 0.0;
	    for (i=0; i<Msize2[Mc_AN]; i++){
	      sum += Krylov_U_OLP[rl0][n][i]*tmpvec1[m][i];            
	    }

	    /* transpose the later calcualtion */
	    matRS0[m][rl0*EKC_core_size[Mc_AN]+n] = sum;
	  }
	}  
      }

      /*  RS^-1 * U^+ * H * |Wn) */

      for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
	for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  for (m=0; m<EKC_core_size[Mc_AN]; m++){

	    sum = 0.0;
	    for (i=0; i<Msize4[Mc_AN]; i++){
	      sum += inv_RS[rl0*EKC_core_size[Mc_AN]+n][i]*matRS0[m][i];
	    }

	    matRS1[rl0*EKC_core_size[Mc_AN]+n][m] = sum;
	  }
	}
      }         

      /*  U * RS^-1 * U^+ * H * |Wn) */

      for (n=0; n<EKC_core_size[Mc_AN]; n++){
	for (i=0; i<Msize2[Mc_AN]; i++){
	  tmpvec0[n][i] = 0.0;
	}
      }

      for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
	for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  for (m=0; m<EKC_core_size[Mc_AN]; m++){
	    tmp0 = matRS1[rl0*EKC_core_size[Mc_AN]+n][m];
	    for (i=0; i<Msize2[Mc_AN]; i++){
	      tmpvec0[m][i] += Krylov_U_OLP[rl0][n][i]*tmp0;
	    }
	  }
	}
      }
    }

    else {
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
        for (i=0; i<Msize2[Mc_AN]; i++){
	  tmpvec0[n][i]  = tmpvec1[n][i];
        }
      }
    }

    if (measure_time==1) dtime(&Stime1);

    /*************************************************************
     S-orthogonalization by a classical block Gram-Schmidt method 
    *************************************************************/

    /* |tmpvec2) = S * |tmpvec0) */

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec2[n][i] = 0.0;
      }
    }

    for (i=0; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=0; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j] - 1;

	if (0<=kl){

#ifdef nosse

	  /* Original version */
	  	    
	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*tmpvec0[n][Bnum+k];
	      }

	      tmpvec2[n][Anum+m] += sum;
	    }
	  }
	  
#else

	  /* Unrolling + SSE version */
	  
	  for (m=0; m<(ian-3); m+=4){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      mmSum00 = _mm_setzero_pd();
	      mmSum01 = _mm_setzero_pd();
	      mmSum10 = _mm_setzero_pd();
	      mmSum11 = _mm_setzero_pd();
	      mmSum20 = _mm_setzero_pd();
	      mmSum21 = _mm_setzero_pd();
	      mmSum30 = _mm_setzero_pd();
	      mmSum31 = _mm_setzero_pd();

	      for (k=0; k<(jan-3); k+=4){
		mmTmp0 = _mm_loadu_pd(&tmpvec0[n][Bnum+k+0]);
		mmTmp1 = _mm_loadu_pd(&tmpvec0[n][Bnum+k+2]);

		mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+0]),mmTmp0));  
		mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+2]),mmTmp1));  

		mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+0]),mmTmp0));  
		mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+2]),mmTmp1));  

		mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+0]),mmTmp0));  
		mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+2]),mmTmp1));  

		mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+0]),mmTmp0));  
		mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+2]),mmTmp1));  
	      }

	      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	      _mm_storeu_pd(&mmArr[0], mmSum00);
	      _mm_storeu_pd(&mmArr[2], mmSum10);
	      _mm_storeu_pd(&mmArr[4], mmSum20);
	      _mm_storeu_pd(&mmArr[6], mmSum30);

	      sum0 = mmArr[0] + mmArr[1];
	      sum1 = mmArr[2] + mmArr[3];
	      sum2 = mmArr[4] + mmArr[5];
	      sum3 = mmArr[6] + mmArr[7];

	      for (; k<jan; k++){
		sum0 += OLP0[ih][kl][m+0][k]*tmpvec0[n][Bnum+k];
		sum1 += OLP0[ih][kl][m+1][k]*tmpvec0[n][Bnum+k];
		sum2 += OLP0[ih][kl][m+2][k]*tmpvec0[n][Bnum+k];
		sum3 += OLP0[ih][kl][m+3][k]*tmpvec0[n][Bnum+k];
	      }

	      tmpvec2[n][Anum+m+0] += sum0;
	      tmpvec2[n][Anum+m+1] += sum1;
	      tmpvec2[n][Anum+m+2] += sum2;
	      tmpvec2[n][Anum+m+3] += sum3;
	    }
	  }

	  for (; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*tmpvec0[n][Bnum+k];
	      }

	      tmpvec2[n][Anum+m] += sum;
	    }
	  } 
	  
#endif

	}
      }
    }

    if (measure_time==1){ 
      dtime(&Etime1);
      time4 += Etime1 - Stime1;      
    }

    if (measure_time==1) dtime(&Stime1);

#ifdef nosse

    /* Original version */
        
    for (rl0=0; rl0<=rl; rl0++){

      /* (U_rl0|tmpvec2) */
        
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
          sum = 0.0;
  	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += U0[rl0][m][i]*tmpvec2[n][i];
	  }
          tmpmat0[m][n] = sum;
	}
      }
    
      /* |tmpvec0) - |U_rl0) * (U_rl0|tmpvec2) */
        
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
	for (k=0; k<EKC_core_size[Mc_AN]; k++){
	  dum = tmpmat0[k][n];
	  for (i=0; i<Msize2[Mc_AN]; i++)  tmpvec0[n][i] -= U0[rl0][k][i]*dum;
        }
      }

    }
    
#else

    /* Unrolling + SSE version */
    
    for (rl0=0; rl0<=rl; rl0++){

      /* (U_rl0|tmpvec2) */
    
      for (m=0; m<(EKC_core_size[Mc_AN]-3); m+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  mmSum00 = _mm_setzero_pd();
	  mmSum01 = _mm_setzero_pd();
	  mmSum10 = _mm_setzero_pd();
	  mmSum11 = _mm_setzero_pd();
	  mmSum20 = _mm_setzero_pd();
	  mmSum21 = _mm_setzero_pd();
	  mmSum30 = _mm_setzero_pd();
	  mmSum31 = _mm_setzero_pd();

  	  for (i=0; i<(Msize2[Mc_AN]-3); i+=4){
	    mmTmp0 = _mm_loadu_pd(&tmpvec2[n][i+0]);
	    mmTmp1 = _mm_loadu_pd(&tmpvec2[n][i+2]);

	    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+0][i+0]),mmTmp0));  
	    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+0][i+2]),mmTmp1));  

	    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+1][i+0]),mmTmp0));  
	    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+1][i+2]),mmTmp1));  

	    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+2][i+0]),mmTmp0));  
	    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+2][i+2]),mmTmp1));  

	    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+3][i+0]),mmTmp0));  
	    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+3][i+2]),mmTmp1));  
	  }

	  mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	  mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	  mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	  mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	  _mm_storeu_pd(&mmArr[0], mmSum00);
	  _mm_storeu_pd(&mmArr[2], mmSum10);
	  _mm_storeu_pd(&mmArr[4], mmSum20);
	  _mm_storeu_pd(&mmArr[6], mmSum30);

	  sum0 = mmArr[0] + mmArr[1];
	  sum1 = mmArr[2] + mmArr[3];
	  sum2 = mmArr[4] + mmArr[5];
	  sum3 = mmArr[6] + mmArr[7];

  	  for (; i<Msize2[Mc_AN]; i++){
            sum0 += U0[rl0][m+0][i]*tmpvec2[n][i];
            sum1 += U0[rl0][m+1][i]*tmpvec2[n][i];
            sum2 += U0[rl0][m+2][i]*tmpvec2[n][i];
            sum3 += U0[rl0][m+3][i]*tmpvec2[n][i];
	  }

          tmpmat0[m+0][n] = sum0;
          tmpmat0[m+1][n] = sum1;
          tmpmat0[m+2][n] = sum2;
          tmpmat0[m+3][n] = sum3;
	}
      }

      for (; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
          sum = 0.0;
  	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += U0[rl0][m][i]*tmpvec2[n][i];
	  }
          tmpmat0[m][n] = sum;
	}
      }

      /* |tmpvec0) - |U_rl0) * (U_rl0|tmpvec2) */
    
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
	for (k=0; k<EKC_core_size[Mc_AN]; k++){
	  dum = tmpmat0[k][n];
	  for (i=0; i<Msize2[Mc_AN]; i++)  tmpvec0[n][i] -= U0[rl0][k][i]*dum;
        }
      }

    }
    
#endif

    if (measure_time==1){ 
      dtime(&Etime1);
      time5 += Etime1 - Stime1;      
    }

    /*************************************************************
                   S-orthonormalization of tmpvec0
    *************************************************************/

    if (measure_time==1) dtime(&Stime1);

    S_orthonormalize_vec_trd( Mc_AN, ct_on, tmpvec0, tmpvec1, OLP0, tmpmat0, ko, iko, MP, Msize2 );

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
        U0[rl+1][n][i] = tmpvec0[n][i];
      }
    }

    if (measure_time==1){ 
      dtime(&Etime1);
      time6 += Etime1 - Stime1;      
    }

  } /* rl */

  /************************************************************
              orthogonalization by diagonalization
  ************************************************************/

  if (measure_time==1) dtime(&Stime1);

#pragma omp parallel shared(EKC_core_size_max,Msize2_max,rlmax_EC,Mc_AN,EKC_core_size,Msize2,can,natn,ct_AN,ct_on,Spe_Total_CNO,WhatSpecies,MP,S_G2M,RMI1,Hks,spin,Msize,Msize4,OLP0,U0,rlmax_EC2,FS)   private(OMPID,Nthrds,Nprocs,rl,n,i,ig,ian,Anum,ih,j,kl,jg,jan,Bnum,m,sum,k,mmSum00,mmSum01,mmSum10,mmSum11,mmSum20,mmSum21,mmSum30,mmSum31,mmTmp0,mmTmp1,mmArr,sum0,sum1,sum2,sum3,rl0)

  {

    double **tmpvec1;

    tmpvec1=OpenMX_Solver_AlignedRealMatrix(solver_owner,"tmpvec1-7",((size_t)1)*EKC_core_size_max,((size_t)1)*Msize2_max);
    for (i=0; i<EKC_core_size_max; i++){
      ;
    }


    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (rl=0+OMPID; rl<rlmax_EC[Mc_AN]; rl+=Nthrds){

    /*  S * |Vn) */

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=0; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=0; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j] - 1;

	if (0<=kl){

#ifdef nosse

	  /* Original version */
	  /**/	  
	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*U0[rl][n][Bnum+k];
	      }
	      tmpvec1[n][Anum+m] += sum;
	    }
	  } 
	  /**/

#else

	  /* Unrolling + SSE version */
	  /**/
	  for (m=0; m<(ian-3); m+=4){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      mmSum00 = _mm_setzero_pd();
	      mmSum01 = _mm_setzero_pd();
	      mmSum10 = _mm_setzero_pd();
	      mmSum11 = _mm_setzero_pd();
	      mmSum20 = _mm_setzero_pd();
	      mmSum21 = _mm_setzero_pd();
	      mmSum30 = _mm_setzero_pd();
	      mmSum31 = _mm_setzero_pd();

	      for (k=0; k<(jan-3); k+=4){
		mmTmp0 = _mm_loadu_pd(&U0[rl][n][Bnum+k+0]);
		mmTmp1 = _mm_loadu_pd(&U0[rl][n][Bnum+k+2]);

		mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+0]),mmTmp0));  
		mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+2]),mmTmp1));  

		mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+0]),mmTmp0));  
		mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+2]),mmTmp1));  

		mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+0]),mmTmp0));  
		mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+2]),mmTmp1));  

		mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+0]),mmTmp0));  
		mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+2]),mmTmp1));  
	      }

	      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	      _mm_storeu_pd(&mmArr[0], mmSum00);
	      _mm_storeu_pd(&mmArr[2], mmSum10);
	      _mm_storeu_pd(&mmArr[4], mmSum20);
	      _mm_storeu_pd(&mmArr[6], mmSum30);

	      sum0 = mmArr[0] + mmArr[1];
	      sum1 = mmArr[2] + mmArr[3];
	      sum2 = mmArr[4] + mmArr[5];
	      sum3 = mmArr[6] + mmArr[7];

	      for (; k<jan; k++){
		sum0 += OLP0[ih][kl][m+0][k]*U0[rl][n][Bnum+k];
		sum1 += OLP0[ih][kl][m+1][k]*U0[rl][n][Bnum+k];
		sum2 += OLP0[ih][kl][m+2][k]*U0[rl][n][Bnum+k];
		sum3 += OLP0[ih][kl][m+3][k]*U0[rl][n][Bnum+k];
	      }

	      tmpvec1[n][Anum+m+0] += sum0;
	      tmpvec1[n][Anum+m+1] += sum1;
	      tmpvec1[n][Anum+m+2] += sum2;
	      tmpvec1[n][Anum+m+3] += sum3;
	    }
	  } 

	  for (; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*U0[rl][n][Bnum+k];
	      }
	      tmpvec1[n][Anum+m] += sum;
	    }
	  } 
	  /**/
#endif

	}
      }
    }

#ifdef nosse

    /* Original version */
    /**/    
    for (rl0=rl; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += U0[rl0][m][i]*tmpvec1[n][i]; 
	  }
          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum;
	}
      }
    }
    /**/

#else

    /* Unrolling + SSE version */
    /**/
    for (rl0=rl; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<(EKC_core_size[Mc_AN]-3); m+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  mmSum00 = _mm_setzero_pd();
	  mmSum01 = _mm_setzero_pd();
	  mmSum10 = _mm_setzero_pd();
	  mmSum11 = _mm_setzero_pd();
	  mmSum20 = _mm_setzero_pd();
	  mmSum21 = _mm_setzero_pd();
	  mmSum30 = _mm_setzero_pd();
	  mmSum31 = _mm_setzero_pd();

	  for (i=0; i<(Msize2[Mc_AN]-3); i+=4){
	    mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i+0]);
	    mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i+2]);

	    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+0][i+0]),mmTmp0));  
	    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+0][i+2]),mmTmp1));  

	    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+1][i+0]),mmTmp0));  
	    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+1][i+2]),mmTmp1));  

	    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+2][i+0]),mmTmp0));  
	    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+2][i+2]),mmTmp1));  

	    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+3][i+0]),mmTmp0));  
	    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+3][i+2]),mmTmp1));  
	  }

	  mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	  mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	  mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	  mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	  _mm_storeu_pd(&mmArr[0], mmSum00);
	  _mm_storeu_pd(&mmArr[2], mmSum10);
	  _mm_storeu_pd(&mmArr[4], mmSum20);
	  _mm_storeu_pd(&mmArr[6], mmSum30);

	  sum0 = mmArr[0] + mmArr[1];
	  sum1 = mmArr[2] + mmArr[3];
	  sum2 = mmArr[4] + mmArr[5];
	  sum3 = mmArr[6] + mmArr[7];

	  for (; i<Msize2[Mc_AN]; i++){
            sum0 += U0[rl0][m+0][i]*tmpvec1[n][i]; 
            sum1 += U0[rl0][m+1][i]*tmpvec1[n][i]; 
            sum2 += U0[rl0][m+2][i]*tmpvec1[n][i]; 
            sum3 += U0[rl0][m+3][i]*tmpvec1[n][i]; 
	  }

          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum0;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum0;

          FS[rl0*EKC_core_size[Mc_AN]+m+2][rl*EKC_core_size[Mc_AN]+n+1] = sum1;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+2] = sum1;

          FS[rl0*EKC_core_size[Mc_AN]+m+3][rl*EKC_core_size[Mc_AN]+n+1] = sum2;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+3] = sum2;

          FS[rl0*EKC_core_size[Mc_AN]+m+4][rl*EKC_core_size[Mc_AN]+n+1] = sum3;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+4] = sum3;
	}
      }

      for (; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += U0[rl0][m][i]*tmpvec1[n][i]; 
	  }
          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum;
	}
      }

    }
    /**/

#endif

  }

    for (i=0; i<EKC_core_size_max; i++){
      ;
    }
    ;

  } /* #pragma omp parallel */

  if (measure_time==1){ 
    dtime(&Etime1);
    time7 += Etime1 - Stime1;      
  }

  if (measure_time==1) dtime(&Stime1);

  info = Eigen_lapack_d(FS,ko,Msize3[Mc_AN],Msize3[Mc_AN]);
  if (info!=0){
    info = Eigen_lapack_x(FS,ko,Msize3[Mc_AN],Msize3[Mc_AN]);
    if (info!=0){
      Eigen_lapack_r(FS,ko,Msize3[Mc_AN],Msize3[Mc_AN]);
    }
  }

  if (measure_time==1){ 
    dtime(&Etime1);
    time8 += Etime1 - Stime1;      
  }

  ZeroNum = 0;

  for (i=1; i<=Msize3[Mc_AN]; i++){

    if (error_check==1){
      printf("spin=%2d Mc_AN=%2d i=%3d ko[i]=%18.15f\n",spin,Mc_AN,i,ko[i]);
    }

    if (cutoff_value<ko[i]){
      ko[i]  = sqrt(fabs(ko[i]));
      iko[i] = 1.0/ko[i];
    }
    else{
      ZeroNum++;
      ko[i]  = 0.0;
      iko[i] = 0.0;
    }
  }  

  if (error_check==1){
    printf("spin=%2d Mc_AN=%2d ZeroNum=%2d\n",spin,Mc_AN,ZeroNum);
  }

  for (i=1; i<=Msize3[Mc_AN]; i++){
    for (j=1; j<=Msize3[Mc_AN]; j++){
      FS[i][j] = FS[i][j]*iko[j];
    }
  }

  /* transpose for later calculation */
  for (i=1; i<=Msize3[Mc_AN]; i++){
    for (j=i+1; j<=Msize3[Mc_AN]; j++){
      tmp1 = FS[i][j];
      tmp2 = FS[j][i];
      FS[i][j] = tmp2;
      FS[j][i] = tmp1;
    }
  }

  if (measure_time==1) dtime(&Stime1);

  /* U0 * U * lamda^{-1/2} */ 

#ifdef nosse

  /* original version */
  /*
  for (i=0; i<Msize2[Mc_AN]; i++){
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){

        m1 = rl0*EKC_core_size[Mc_AN] + m + 1;

	sum = 0.0; 
	for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

          n1 = rl*EKC_core_size[Mc_AN] + 1;

	  for (n=0; n<EKC_core_size[Mc_AN]; n++){
	    sum += U0[rl][n][i]*FS[m1][n1+n]; 
	  }
	}

        Utmp[rl0][m] = sum;        
      }
    }

    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        U0[rl0][m][i] = Utmp[rl0][m];
      }
    }
  }
  */

  /* unrolling version */

  for (i=0; i<Msize2[Mc_AN]; i++){

    for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
        Utmp[rl][n] = U0[rl][n][i];
      }
    }

    for (m1=1; m1<=(rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]-3); m1+=4){

      rl00 = (m1+0-1)/EKC_core_size[Mc_AN];
      rl01 = (m1+1-1)/EKC_core_size[Mc_AN];
      rl02 = (m1+2-1)/EKC_core_size[Mc_AN];
      rl03 = (m1+3-1)/EKC_core_size[Mc_AN];

      mm0 = (m1+0-1)%EKC_core_size[Mc_AN];
      mm1 = (m1+1-1)%EKC_core_size[Mc_AN];
      mm2 = (m1+2-1)%EKC_core_size[Mc_AN];
      mm3 = (m1+3-1)%EKC_core_size[Mc_AN];

      sum0 = 0.0; 
      sum1 = 0.0; 
      sum2 = 0.0; 
      sum3 = 0.0; 

      for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

	n1 = rl*EKC_core_size[Mc_AN] + 1;

	for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  sum0 += Utmp[rl][n]*FS[m1+0][n1+n]; 
	  sum1 += Utmp[rl][n]*FS[m1+1][n1+n]; 
	  sum2 += Utmp[rl][n]*FS[m1+2][n1+n]; 
	  sum3 += Utmp[rl][n]*FS[m1+3][n1+n]; 
	}
      }

      U0[rl00][mm0][i] = sum0;        
      U0[rl01][mm1][i] = sum1;        
      U0[rl02][mm2][i] = sum2;        
      U0[rl03][mm3][i] = sum3;        
    }

    m1s = rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN])%4 + 1;

    for (m1=m1s; m1<=rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]; m1++){

      rl0 = (m1-1)/EKC_core_size[Mc_AN];
      m = (m1-1)%EKC_core_size[Mc_AN];

      sum = 0.0; 

      for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

	n1 = rl*EKC_core_size[Mc_AN] + 1;

	for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  sum += Utmp[rl][n]*FS[m1][n1+n]; 
	}
      }

      U0[rl0][m][i] = sum;        
    }
  } /* i */

#else

  /* Unrolling + SSE version */
  /**/

#pragma omp parallel shared(rlmax_EC,Mc_AN,EKC_core_size,Msize2,U0,FS)   private(OMPID,Nthrds,Nprocs,rl,n,i,j,m,sum,k,mmSum00,mmSum01,mmSum10,mmSum11,mmSum20,mmSum21,mmSum30,mmSum31,mmTmp0,mmTmp1,mmArr,sum0,sum1,sum2,sum3,rl0,rl00,rl01,rl02,rl03,mm0,mm1,mm2,mm3,n1,m1,m1s,Utmp)

  {

    double ** Utmp;
    Utmp=OpenMX_Solver_AlignedRealMatrix(solver_owner,"Utmp-8",((size_t)1)*rlmax_EC[Mc_AN],((size_t)1)*EKC_core_size[Mc_AN]);
    for (i=0; i<rlmax_EC[Mc_AN]; i++){
      ;
    }

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

  for (i=0+OMPID; i<Msize2[Mc_AN]; i+=Nthrds){

    for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
        Utmp[rl][n] = U0[rl][n][i];
      }
    }

    for (m1=1; m1<=(rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]-3); m1+=4){

      rl00 = (m1+0-1)/EKC_core_size[Mc_AN];
      rl01 = (m1+1-1)/EKC_core_size[Mc_AN];
      rl02 = (m1+2-1)/EKC_core_size[Mc_AN];
      rl03 = (m1+3-1)/EKC_core_size[Mc_AN];

      mm0 = (m1+0-1)%EKC_core_size[Mc_AN];
      mm1 = (m1+1-1)%EKC_core_size[Mc_AN];
      mm2 = (m1+2-1)%EKC_core_size[Mc_AN];
      mm3 = (m1+3-1)%EKC_core_size[Mc_AN];

      sum0 = 0.0; 
      sum1 = 0.0; 
      sum2 = 0.0; 
      sum3 = 0.0; 

      mmSum00 = _mm_setzero_pd();
      mmSum01 = _mm_setzero_pd();
      mmSum10 = _mm_setzero_pd();
      mmSum11 = _mm_setzero_pd();
      mmSum20 = _mm_setzero_pd();
      mmSum21 = _mm_setzero_pd();
      mmSum30 = _mm_setzero_pd();
      mmSum31 = _mm_setzero_pd();

      for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

	n1 = rl*EKC_core_size[Mc_AN] + 1;

	if (0){
	mmTmp0 = _mm_loadu_pd(&Utmp[rl][n+0]);
	mmTmp1 = _mm_loadu_pd(&Utmp[rl][n+2]);
	}

	for (n=0; n<(EKC_core_size[Mc_AN]-3); n+=4){
	  mmTmp0 = _mm_loadu_pd(&Utmp[rl][n+0]);
	  mmTmp1 = _mm_loadu_pd(&Utmp[rl][n+2]);
	  
	  mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&FS[m1+0][n1+n+0]),mmTmp0));  
	  mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&FS[m1+0][n1+n+2]),mmTmp1));  

	  mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&FS[m1+1][n1+n+0]),mmTmp0));  
	  mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&FS[m1+1][n1+n+2]),mmTmp1));  

	  mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&FS[m1+2][n1+n+0]),mmTmp0));  
	  mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&FS[m1+2][n1+n+2]),mmTmp1));  

	  mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&FS[m1+3][n1+n+0]),mmTmp0));  
	  mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&FS[m1+3][n1+n+2]),mmTmp1));  
	}

	for (; n<EKC_core_size[Mc_AN]; n++){
	  sum0 += Utmp[rl][n]*FS[m1+0][n1+n]; 
	  sum1 += Utmp[rl][n]*FS[m1+1][n1+n]; 
	  sum2 += Utmp[rl][n]*FS[m1+2][n1+n]; 
	  sum3 += Utmp[rl][n]*FS[m1+3][n1+n]; 
	}

      }
      
      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
      _mm_storeu_pd(&mmArr[0], mmSum00);
      _mm_storeu_pd(&mmArr[2], mmSum10);
      _mm_storeu_pd(&mmArr[4], mmSum20);
      _mm_storeu_pd(&mmArr[6], mmSum30);

      sum0 += mmArr[0] + mmArr[1];
      sum1 += mmArr[2] + mmArr[3];
      sum2 += mmArr[4] + mmArr[5];
      sum3 += mmArr[6] + mmArr[7];

      U0[rl00][mm0][i] = sum0;        
      U0[rl01][mm1][i] = sum1;        
      U0[rl02][mm2][i] = sum2;        
      U0[rl03][mm3][i] = sum3;        
    }

    m1s = rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN])%4 + 1;

    for (m1=m1s; m1<=rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]; m1++){

      rl0 = (m1-1)/EKC_core_size[Mc_AN];
      m = (m1-1)%EKC_core_size[Mc_AN];

      sum = 0.0; 

      for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

	n1 = rl*EKC_core_size[Mc_AN] + 1;

	for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  sum += Utmp[rl][n]*FS[m1][n1+n]; 
	}
      }

      U0[rl0][m][i] = sum;        
    }
  } /* i */

  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
  }
  ;


  } /* #pragma omp parallel */

#endif

  Krylov_U[spin][Mc_AN][0] = ZeroNum;

  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
        Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+i+1] = U0[rl][n][i];
      }
    }
  }

  if (measure_time==1){ 
    dtime(&Etime1);
    time9 += Etime1 - Stime1;      
  }

  /************************************************************
         check the orthonormality of Krylov vectors
  ************************************************************/

  if (error_check==1){

    for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

      for (n=0; n<EKC_core_size[Mc_AN]; n++){
	for (i=0; i<Msize2[Mc_AN]; i++){
	  tmpvec1[n][i]  = 0.0;
	}
      }

      for (i=0; i<=can; i++){

	ig = natn[ct_AN][i];
	ian = Spe_Total_CNO[WhatSpecies[ig]];
	Anum = MP[i] - 1;
	ih = S_G2M[ig];

	for (j=0; j<=can; j++){

	  kl = RMI1[Mc_AN][i][j];
	  jg = natn[ct_AN][j];
	  jan = Spe_Total_CNO[WhatSpecies[jg]];
	  Bnum = MP[j] - 1;

	  if (0<=kl){

	    for (m=0; m<ian; m++){
	      for (n=0; n<EKC_core_size[Mc_AN]; n++){

		sum = 0.0;
		for (k=0; k<jan; k++){
		  sum += OLP0[ih][kl][m][k]*U0[rl][n][Bnum+k];
		}

		tmpvec1[n][Anum+m] += sum;
	      }
	    } 
	  }
	}
      }

      for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
	for (m=0; m<EKC_core_size[Mc_AN]; m++){
	  for (n=0; n<EKC_core_size[Mc_AN]; n++){
	    sum = 0.0;
	    for (i=0; i<Msize2[Mc_AN]; i++){
	      sum += U0[rl0][m][i]*tmpvec1[n][i]; 
	    }

	    if (rl==rl0 && m==n){
	      if ( 1.0e-10<fabs(sum-1.0) ) {
		printf("A spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
		       spin,Mc_AN,rl,rl0,m,n,sum);
	      }
	    }
	    else{
	      if ( 1.0e-10<fabs(sum) ) {
		printf("B spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
		       spin,Mc_AN,rl,rl0,m,n,sum);
	      }
	    }

	  }
	}
      }
    }
  }

  if (measure_time==1){ 
    printf("pMatrix myid=%2d time1 =%5.3f time2 =%5.3f time3 =%5.3f time4 =%5.3f\n",
            myid,time1,time2,time3,time4);
    printf("pMatrix myid=%2d time5 =%5.3f time6 =%5.3f time7 =%5.3f time8 =%5.3f\n",
            myid,time5,time6,time7,time8);
    printf("pMatrix myid=%2d time9 =%5.3f\n",myid,time9);
  }

  /* freeing of arrays */

  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
  }
  ;

  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    for (j=0; j<EKC_core_size[Mc_AN]; j++){
      ;
    }
    ;
  }  
  ;

  for (i=0; i<(EKC_core_size[Mc_AN]+4); i++){
    ;
  }
  ;

  for (i=0; i<(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]; i++){
    ;
  }
  ;

  ;
  ;

  for (i=0; i<(EKC_core_size[Mc_AN]+2); i++){
    ;
  }
  ;

  for (i=0; i<(Msize4[Mc_AN]+3); i++){
    ;
  }
  ;
}








typedef struct {
  int myid;
  int spin;
  int Mc_AN;
  double *****Hks;
  double ****OLP0;
  double ***Krylov_U;
  int *MP;
  int *Msize;
  int *Msize2;
  int *Msize3;
  double **tmpvec1;
} Generate_pMatrix2_trd_scratch_arguments;
static void Generate_pMatrix2_trd_raii( int myid, int spin, int Mc_AN, double *****Hks, double ****OLP0,
                        double ***Krylov_U, int *MP, int *Msize, int *Msize2, int *Msize3,
                        double **tmpvec1,OpenMXSolverInvocation *solver_owner);
static int Generate_pMatrix2_trd_scratch_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Generate_pMatrix2_trd_scratch_arguments *a=(Generate_pMatrix2_trd_scratch_arguments*)arguments;
  Generate_pMatrix2_trd_raii(a->myid,a->spin,a->Mc_AN,a->Hks,a->OLP0,a->Krylov_U,a->MP,a->Msize,a->Msize2,a->Msize3,a->tmpvec1,owner);
  *result=0;
  return 1;
}
void Generate_pMatrix2_trd( int myid, int spin, int Mc_AN, double *****Hks, double ****OLP0, 
                        double ***Krylov_U, int *MP, int *Msize, int *Msize2, int *Msize3, 
                        double **tmpvec1)
{
  Generate_pMatrix2_trd_scratch_arguments arguments={myid,spin,Mc_AN,Hks,OLP0,Krylov_U,MP,Msize,Msize2,Msize3,tmpvec1};
  OpenMX_Solver_Run(Generate_pMatrix2_trd_scratch_operation,&arguments);
}

static void Generate_pMatrix2_trd_raii( int myid, int spin, int Mc_AN, double *****Hks, double ****OLP0,
                        double ***Krylov_U, int *MP, int *Msize, int *Msize2, int *Msize3,
                        double **tmpvec1,OpenMXSolverInvocation *solver_owner)
{
  int rl,rl0,rl1,ct_AN,fan,san,can,wan,ct_on,i,j;
  int n,Anum,Bnum,k,ian,ih,kl,jg,ig,jan,m,m1,n1,info;
  int ZeroNum,rl_half;
  int KU_d1,KU_d2,csize;
  double sum,dum,tmp0,tmp1,tmp2,tmp3;
  double **Utmp;
  double *ko,*iko;
  double **FS;
  double ***U0;

  int rl00,rl01,rl02,rl03,rl04,rl05,rl06,rl07;
  int mm0,mm1,mm2,mm3,mm4,mm5,mm6,mm7,m1s;
  __m128d mmSum00,mmSum01,mmSum10,mmSum11,mmSum20,mmSum21,mmSum30,mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5; 

  double mmArr[8];
  double sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7;

  ct_AN = M2G[Mc_AN];
  fan = FNAN[ct_AN];
  san = SNAN[ct_AN];
  can = fan + san;
  wan = WhatSpecies[ct_AN];
  ct_on = Spe_Total_CNO[wan];

  if (Msize[Mc_AN]<Msize3[Mc_AN])
    csize = Msize3[Mc_AN] + 40;
  else
    csize = Msize[Mc_AN] + 40;

  KU_d1 = EKC_core_size[Mc_AN]*Msize2[Mc_AN];
  KU_d2 = Msize2[Mc_AN];

  /* allocation of arrays */

  Utmp=OpenMX_Solver_AlignedRealMatrix(solver_owner,"Utmp-1",((size_t)1)*rlmax_EC[Mc_AN],((size_t)1)*EKC_core_size[Mc_AN]);
  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
  }

  U0=OpenMX_Solver_AlignedRealVolume(solver_owner,"U0-2",((size_t)1)*rlmax_EC[Mc_AN],((size_t)1)*EKC_core_size[Mc_AN],((size_t)1)*(Msize2[Mc_AN]+3));
  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
    for (j=0; j<EKC_core_size[Mc_AN]; j++){
      ;
      for (k=0; k<(Msize2[Mc_AN]+3); k++)  U0[i][j][k] = 0.0;
    }
  }  

  FS=OpenMX_Solver_AlignedRealMatrix(solver_owner,"FS-3",((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN],((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  for (i=0; i<(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]; i++){
    ;
  }

  ko=(double*)OpenMX_Solver_Real(solver_owner,"ko-4",((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  iko=(double*)OpenMX_Solver_Real(solver_owner,"iko-5",((size_t)1)*(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  
  /****************************************************
   initialize 
  ****************************************************/

  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
        U0[rl][n][i] = 0.0;
      }
    }
  }

  i = 0;
  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      if (i<Msize2[Mc_AN]) U0[rl][n][i] = 1.0;
      i++;
    }
  }

  /************************************************************
              orthogonalization by diagonalization
  ************************************************************/

  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

    /*  S * |Vn) */

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=0; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=0; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j] - 1;

	if (0<=kl){

#ifdef nosse

	  /* Original version */
	  /**/
	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*U0[rl][n][Bnum+k];
	      }
	      tmpvec1[n][Anum+m] += sum;
	    }
	  }
	  /**/

#else

	  /* Unrolling + SSE version */

	  for (m=0; m<(ian-3); m+=4){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      mmSum00 = _mm_setzero_pd();
	      mmSum01 = _mm_setzero_pd();
	      mmSum10 = _mm_setzero_pd();
	      mmSum11 = _mm_setzero_pd();
	      mmSum20 = _mm_setzero_pd();
	      mmSum21 = _mm_setzero_pd();
	      mmSum30 = _mm_setzero_pd();
	      mmSum31 = _mm_setzero_pd();

	      for (k=0; k<(jan-3); k+=4){
		mmTmp0 = _mm_loadu_pd(&U0[rl][n][Bnum+k+0]);
		mmTmp1 = _mm_loadu_pd(&U0[rl][n][Bnum+k+2]);

		mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+0]),mmTmp0));  
		mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+2]),mmTmp1));  

		mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+0]),mmTmp0));  
		mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+2]),mmTmp1));  

		mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+0]),mmTmp0));  
		mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+2]),mmTmp1));  

		mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+0]),mmTmp0));  
		mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+2]),mmTmp1));  
	      }

	      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	      _mm_storeu_pd(&mmArr[0], mmSum00);
	      _mm_storeu_pd(&mmArr[2], mmSum10);
	      _mm_storeu_pd(&mmArr[4], mmSum20);
	      _mm_storeu_pd(&mmArr[6], mmSum30);

	      sum0 = mmArr[0] + mmArr[1];
	      sum1 = mmArr[2] + mmArr[3];
	      sum2 = mmArr[4] + mmArr[5];
	      sum3 = mmArr[6] + mmArr[7];

	      for (; k<jan; k++){
		sum0 += OLP0[ih][kl][m+0][k]*U0[rl][n][Bnum+k];
		sum1 += OLP0[ih][kl][m+1][k]*U0[rl][n][Bnum+k];
		sum2 += OLP0[ih][kl][m+2][k]*U0[rl][n][Bnum+k];
		sum3 += OLP0[ih][kl][m+3][k]*U0[rl][n][Bnum+k];
	      }

	      tmpvec1[n][Anum+m+0] += sum0;
	      tmpvec1[n][Anum+m+1] += sum1;
	      tmpvec1[n][Anum+m+2] += sum2;
	      tmpvec1[n][Anum+m+3] += sum3;
	    }
	  } 

	  for (; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*U0[rl][n][Bnum+k];
	      }
	      tmpvec1[n][Anum+m] += sum;
	    }
	  } 

#endif

	}
      }
    }

#ifdef nosse

    /* Original version */
    /**/
    for (rl0=rl; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += U0[rl0][m][i]*tmpvec1[n][i]; 
	  }
          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum;
	}
      }
    }
    /**/

#else

    /* Unrolling + SSE version */

    for (rl0=rl; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<(EKC_core_size[Mc_AN]-3); m+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  mmSum00 = _mm_setzero_pd();
	  mmSum01 = _mm_setzero_pd();
	  mmSum10 = _mm_setzero_pd();
	  mmSum11 = _mm_setzero_pd();
	  mmSum20 = _mm_setzero_pd();
	  mmSum21 = _mm_setzero_pd();
	  mmSum30 = _mm_setzero_pd();
	  mmSum31 = _mm_setzero_pd();

	  for (i=0; i<(Msize2[Mc_AN]-3); i+=4){
	    mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i+0]);
	    mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i+2]);

	    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+0][i+0]),mmTmp0));  
	    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+0][i+2]),mmTmp1));  

	    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+1][i+0]),mmTmp0));  
	    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+1][i+2]),mmTmp1));  

	    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+2][i+0]),mmTmp0));  
	    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+2][i+2]),mmTmp1));  

	    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+3][i+0]),mmTmp0));  
	    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&U0[rl0][m+3][i+2]),mmTmp1));  
	  }

	  mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	  mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	  mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	  mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	  _mm_storeu_pd(&mmArr[0], mmSum00);
	  _mm_storeu_pd(&mmArr[2], mmSum10);
	  _mm_storeu_pd(&mmArr[4], mmSum20);
	  _mm_storeu_pd(&mmArr[6], mmSum30);

	  sum0 = mmArr[0] + mmArr[1];
	  sum1 = mmArr[2] + mmArr[3];
	  sum2 = mmArr[4] + mmArr[5];
	  sum3 = mmArr[6] + mmArr[7];

	  for (; i<Msize2[Mc_AN]; i++){
            sum0 += U0[rl0][m+0][i]*tmpvec1[n][i]; 
            sum1 += U0[rl0][m+1][i]*tmpvec1[n][i]; 
            sum2 += U0[rl0][m+2][i]*tmpvec1[n][i]; 
            sum3 += U0[rl0][m+3][i]*tmpvec1[n][i]; 
	  }

          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum0;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum0;

          FS[rl0*EKC_core_size[Mc_AN]+m+2][rl*EKC_core_size[Mc_AN]+n+1] = sum1;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+2] = sum1;

          FS[rl0*EKC_core_size[Mc_AN]+m+3][rl*EKC_core_size[Mc_AN]+n+1] = sum2;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+3] = sum2;

          FS[rl0*EKC_core_size[Mc_AN]+m+4][rl*EKC_core_size[Mc_AN]+n+1] = sum3;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+4] = sum3;
	}
      }

      for (; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += U0[rl0][m][i]*tmpvec1[n][i]; 
	  }
          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum;
	}
      }
    }

#endif

  }


  info = Eigen_lapack_d(FS,ko,Msize3[Mc_AN],Msize3[Mc_AN]);
  if (info!=0){
    info = Eigen_lapack_x(FS,ko,Msize3[Mc_AN],Msize3[Mc_AN]);
    if (info!=0){
      Eigen_lapack_r(FS,ko,Msize3[Mc_AN],Msize3[Mc_AN]);
    }
  }

  ZeroNum = 0;

  for (i=1; i<=Msize3[Mc_AN]; i++){

    if (error_check==1){
      printf("spin=%2d Mc_AN=%2d i=%3d ko[i]=%18.15f\n",spin,Mc_AN,i,ko[i]);
    }

    if (cutoff_value<ko[i]){
      ko[i]  = sqrt(fabs(ko[i]));
      iko[i] = 1.0/ko[i];
    }
    else{
      ZeroNum++;
      ko[i]  = 0.0;
      iko[i] = 0.0;
    }
  }  

  if (error_check==1){
    printf("spin=%2d Mc_AN=%2d ZeroNum=%2d\n",spin,Mc_AN,ZeroNum);
  }

  for (i=1; i<=Msize3[Mc_AN]; i++){
    for (j=1; j<=Msize3[Mc_AN]; j++){
      FS[i][j] = FS[i][j]*iko[j];
    }
  }

  /* transpose for later calculation */
  for (i=1; i<=Msize3[Mc_AN]; i++){
    for (j=i+1; j<=Msize3[Mc_AN]; j++){
      tmp1 = FS[i][j];
      tmp2 = FS[j][i];
      FS[i][j] = tmp2;
      FS[j][i] = tmp1;
    }
  }

  /* U0 * U * lamda^{-1/2} */ 

#ifdef nosse

  /* original version */

  /**/
  for (i=0; i<Msize2[Mc_AN]; i++){
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){

        m1 = rl0*EKC_core_size[Mc_AN] + m + 1;

	sum = 0.0; 
	for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

          n1 = rl*EKC_core_size[Mc_AN] + 1;

	  for (n=0; n<EKC_core_size[Mc_AN]; n++){
	    sum += U0[rl][n][i]*FS[m1][n1+n]; 
	  }
	}

        Utmp[rl0][m] = sum;        
      }
    }

    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        U0[rl0][m][i] = Utmp[rl0][m];
      }
    }
  }
  /**/

#else

  /* Unrolling + SSE version */
  /**/
  for (i=0; i<Msize2[Mc_AN]; i++){

    for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
      for (n=0; n<EKC_core_size[Mc_AN]; n++){
        Utmp[rl][n] = U0[rl][n][i];
      }
    }

    for (m1=1; m1<=(rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]-3); m1+=4){

      rl00 = (m1+0-1)/EKC_core_size[Mc_AN];
      rl01 = (m1+1-1)/EKC_core_size[Mc_AN];
      rl02 = (m1+2-1)/EKC_core_size[Mc_AN];
      rl03 = (m1+3-1)/EKC_core_size[Mc_AN];

      mm0 = (m1+0-1)%EKC_core_size[Mc_AN];
      mm1 = (m1+1-1)%EKC_core_size[Mc_AN];
      mm2 = (m1+2-1)%EKC_core_size[Mc_AN];
      mm3 = (m1+3-1)%EKC_core_size[Mc_AN];

      sum0 = 0.0; 
      sum1 = 0.0; 
      sum2 = 0.0; 
      sum3 = 0.0; 

      mmSum00 = _mm_setzero_pd();
      mmSum01 = _mm_setzero_pd();
      mmSum10 = _mm_setzero_pd();
      mmSum11 = _mm_setzero_pd();
      mmSum20 = _mm_setzero_pd();
      mmSum21 = _mm_setzero_pd();
      mmSum30 = _mm_setzero_pd();
      mmSum31 = _mm_setzero_pd();

      for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

	n1 = rl*EKC_core_size[Mc_AN] + 1;

        if (0){
	  mmTmp0 = _mm_loadu_pd(&Utmp[rl][n+0]); /*???????????*/
  	  mmTmp1 = _mm_loadu_pd(&Utmp[rl][n+2]); /*???????????*/
        }

	for (n=0; n<(EKC_core_size[Mc_AN]-3); n+=4){
	  mmTmp0 = _mm_loadu_pd(&Utmp[rl][n+0]);
	  mmTmp1 = _mm_loadu_pd(&Utmp[rl][n+2]);
	  mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&FS[m1+0][n1+n+0]),mmTmp0));  
	  mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&FS[m1+0][n1+n+2]),mmTmp1));  

	  mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&FS[m1+1][n1+n+0]),mmTmp0));  
	  mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&FS[m1+1][n1+n+2]),mmTmp1));  

	  mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&FS[m1+2][n1+n+0]),mmTmp0));  
	  mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&FS[m1+2][n1+n+2]),mmTmp1));  

	  mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&FS[m1+3][n1+n+0]),mmTmp0));  
	  mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&FS[m1+3][n1+n+2]),mmTmp1));  
	}

	for (; n<EKC_core_size[Mc_AN]; n++){
	  sum0 += Utmp[rl][n]*FS[m1+0][n1+n]; 
	  sum1 += Utmp[rl][n]*FS[m1+1][n1+n]; 
	  sum2 += Utmp[rl][n]*FS[m1+2][n1+n]; 
	  sum3 += Utmp[rl][n]*FS[m1+3][n1+n]; 
	}
      }

      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
      _mm_storeu_pd(&mmArr[0], mmSum00);
      _mm_storeu_pd(&mmArr[2], mmSum10);
      _mm_storeu_pd(&mmArr[4], mmSum20);
      _mm_storeu_pd(&mmArr[6], mmSum30);

      sum0 += mmArr[0] + mmArr[1];
      sum1 += mmArr[2] + mmArr[3];
      sum2 += mmArr[4] + mmArr[5];
      sum3 += mmArr[6] + mmArr[7];

      U0[rl00][mm0][i] = sum0;        
      U0[rl01][mm1][i] = sum1;        
      U0[rl02][mm2][i] = sum2;        
      U0[rl03][mm3][i] = sum3;        
    }

    m1s = rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN])%4 + 1;

    for (m1=m1s; m1<=rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]; m1++){

      rl0 = (m1-1)/EKC_core_size[Mc_AN];
      m = (m1-1)%EKC_core_size[Mc_AN];

      sum = 0.0; 

      for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

	n1 = rl*EKC_core_size[Mc_AN] + 1;

	for (n=0; n<EKC_core_size[Mc_AN]; n++){
	  sum += Utmp[rl][n]*FS[m1][n1+n]; 
	}
      }

      U0[rl0][m][i] = sum;        
    }
  } /* i */

#endif

  Krylov_U[spin][Mc_AN][0] = ZeroNum;

  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
        Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+i+1] = U0[rl][n][i];
      }
    }
  }

  /************************************************************
         check the orthonormality of Krylov vectors
  ************************************************************/

  if (error_check==1){

    for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){

      for (n=0; n<EKC_core_size[Mc_AN]; n++){
	for (i=0; i<Msize2[Mc_AN]; i++){
	  tmpvec1[n][i]  = 0.0;
	}
      }

      for (i=0; i<=can; i++){

	ig = natn[ct_AN][i];
	ian = Spe_Total_CNO[WhatSpecies[ig]];
	Anum = MP[i] - 1;
	ih = S_G2M[ig];

	for (j=0; j<=can; j++){

	  kl = RMI1[Mc_AN][i][j];
	  jg = natn[ct_AN][j];
	  jan = Spe_Total_CNO[WhatSpecies[jg]];
	  Bnum = MP[j] - 1;

	  if (0<=kl){

	    for (m=0; m<ian; m++){
	      for (n=0; n<EKC_core_size[Mc_AN]; n++){

		sum = 0.0;
		for (k=0; k<jan; k++){
		  sum += OLP0[ih][kl][m][k]*U0[rl][n][Bnum+k];
		}

		tmpvec1[n][Anum+m] += sum;
	      }
	    } 
	  }
	}
      }

      for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
	for (m=0; m<EKC_core_size[Mc_AN]; m++){
	  for (n=0; n<EKC_core_size[Mc_AN]; n++){
	    sum = 0.0;
	    for (i=0; i<Msize2[Mc_AN]; i++){
	      sum += U0[rl0][m][i]*tmpvec1[n][i]; 
	    }

	    if (rl==rl0 && m==n){
	      if ( 1.0e-10<fabs(sum-1.0) ) {
		printf("A spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
		       spin,Mc_AN,rl,rl0,m,n,sum);
	      }
	    }
	    else{
	      if ( 1.0e-10<fabs(sum) ) {
		printf("B spin=%2d Mc_AN=%2d rl=%2d rl0=%2d m=%2d n=%2d sum=%18.15f\n",
		       spin,Mc_AN,rl,rl0,m,n,sum);
	      }
	    }

	  }
	}
      }
    }
  }

  /* freeing of arrays */

  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    ;
  }
  ;

  for (i=0; i<rlmax_EC[Mc_AN]; i++){
    for (j=0; j<EKC_core_size[Mc_AN]; j++){
      ;
    }
    ;
  }  
  ;

  for (i=0; i<(rlmax_EC[Mc_AN]+2)*EKC_core_size[Mc_AN]; i++){
    ;
  }
  ;

  ;
  ;
}







typedef struct {
  int spin;
  int Mc_AN;
  double *****Hks;
  double ***Krylov_U;
  double ****EC_matrix;
  int *MP;
  int *Msize;
  int *Msize2;
  int *Msize3;
  double **tmpvec;
  int EKC_core_size_max;
  int Msize2_max;
} Embedding_Matrix_trd_scratch_arguments;
static void Embedding_Matrix_trd_raii(int spin, int Mc_AN, double *****Hks,
                      double ***Krylov_U, double ****EC_matrix,
                      int *MP, int *Msize, int *Msize2, int *Msize3,
                      double **tmpvec, int EKC_core_size_max, int Msize2_max,OpenMXSolverInvocation *solver_owner);
static int Embedding_Matrix_trd_scratch_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Embedding_Matrix_trd_scratch_arguments *a=(Embedding_Matrix_trd_scratch_arguments*)arguments;
  Embedding_Matrix_trd_raii(a->spin,a->Mc_AN,a->Hks,a->Krylov_U,a->EC_matrix,a->MP,a->Msize,a->Msize2,a->Msize3,a->tmpvec,a->EKC_core_size_max,a->Msize2_max,owner);
  *result=0;
  return 1;
}
void Embedding_Matrix_trd(int spin, int Mc_AN, double *****Hks,
                      double ***Krylov_U, double ****EC_matrix, 
                      int *MP, int *Msize, int *Msize2, int *Msize3, 
                      double **tmpvec, int EKC_core_size_max, int Msize2_max)
{
  Embedding_Matrix_trd_scratch_arguments arguments={spin,Mc_AN,Hks,Krylov_U,EC_matrix,MP,Msize,Msize2,Msize3,tmpvec,EKC_core_size_max,Msize2_max};
  OpenMX_Solver_Run(Embedding_Matrix_trd_scratch_operation,&arguments);
}

static void Embedding_Matrix_trd_raii(int spin, int Mc_AN, double *****Hks,
                      double ***Krylov_U, double ****EC_matrix,
                      int *MP, int *Msize, int *Msize2, int *Msize3,
                      double **tmpvec, int EKC_core_size_max, int Msize2_max,OpenMXSolverInvocation *solver_owner)
{
  int ct_AN,fan,san,can,wan,ct_on;
  int rl,rl0,m,n,i,j,k,kl,jg,jan,ih,ian;
  int Anum,Bnum,ig;
  int KU_d1,KU_d2,csize;
  double sum,tmp1,tmp2,tmp3;

  __m128d mmSum00,mmSum01,mmSum10,mmSum11,mmSum20,mmSum21,mmSum30,mmSum31, mmTmp0, mmTmp1; 
  double mmArr[8];
  double sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  ct_AN = M2G[Mc_AN];
  fan = FNAN[ct_AN];
  san = SNAN[ct_AN];
  can = fan + san;
  wan = WhatSpecies[ct_AN];
  ct_on = Spe_Total_CNO[wan];

  if (Msize[Mc_AN]<Msize3[Mc_AN])
    csize = Msize3[Mc_AN] + 40;
  else
    csize = Msize[Mc_AN] + 40;

  KU_d1 = EKC_core_size[Mc_AN]*Msize2[Mc_AN];
  KU_d2 = Msize2[Mc_AN];

  /*******************************
            u1^+ C^+ u2 
  *******************************/

#pragma omp parallel shared(EKC_core_size_max,Msize2_max,rlmax_EC,Mc_AN,EKC_core_size,Msize2,fan,can,natn,ct_AN,Spe_Total_CNO,WhatSpecies,MP,S_G2M,RMI1,Hks,Krylov_U,spin,KU_d1,KU_d2,Msize,EC_matrix)   private(OMPID,Nthrds,Nprocs,rl,n,i,ig,ian,Anum,ih,j,kl,jg,jan,Bnum,m,sum,k,mmSum00,mmSum01,mmSum10,mmSum11,mmSum20,mmSum21,mmSum30,mmSum31,mmTmp0,mmTmp1,mmArr,sum0,sum1,sum2,sum3,rl0)

  {

    double **tmpvec1;
    tmpvec1=OpenMX_Solver_AlignedRealMatrix(solver_owner,"tmpvec1-1",((size_t)1)*EKC_core_size_max,((size_t)1)*Msize2_max);
    for (i=0; i<EKC_core_size_max; i++){
      ;
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (rl=0+OMPID; rl<rlmax_EC[Mc_AN]; rl+=Nthrds){

    /* C^+ u2 */

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=0; i<=fan; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=fan+1; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j];

	if (0<=kl){

#ifdef nosse

	  /* Original version */
	  /**/	  
	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += Hks[spin][ih][kl][m][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
	      }

	      tmpvec1[n][Anum+m] += sum;
	    }
	  }
	  /**/

#else

	  /* Unrolling + SSE version */
	  /**/
	  for (m=0; m<(ian-3); m+=4){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      mmSum00 = _mm_setzero_pd();
	      mmSum01 = _mm_setzero_pd();
	      mmSum10 = _mm_setzero_pd();
	      mmSum11 = _mm_setzero_pd();
	      mmSum20 = _mm_setzero_pd();
	      mmSum21 = _mm_setzero_pd();
	      mmSum30 = _mm_setzero_pd();
	      mmSum31 = _mm_setzero_pd();

	      for (k=0; k<(jan-3); k+=4){
		mmTmp0 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+0]);
		mmTmp1 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+2]);

		mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+0][k+0]),mmTmp0));  
		mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+0][k+2]),mmTmp1));  

		mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+1][k+0]),mmTmp0));  
		mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+1][k+2]),mmTmp1));  

		mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+2][k+0]),mmTmp0));  
		mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+2][k+2]),mmTmp1));  

		mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+3][k+0]),mmTmp0));  
		mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+3][k+2]),mmTmp1));  
	      }

	      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	      _mm_storeu_pd(&mmArr[0], mmSum00);
	      _mm_storeu_pd(&mmArr[2], mmSum10);
	      _mm_storeu_pd(&mmArr[4], mmSum20);
	      _mm_storeu_pd(&mmArr[6], mmSum30);

	      sum0 = mmArr[0] + mmArr[1];
	      sum1 = mmArr[2] + mmArr[3];
	      sum2 = mmArr[4] + mmArr[5];
	      sum3 = mmArr[6] + mmArr[7];

	      for (; k<jan; k++){
		sum0 += Hks[spin][ih][kl][m+0][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
		sum1 += Hks[spin][ih][kl][m+1][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
		sum2 += Hks[spin][ih][kl][m+2][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
		sum3 += Hks[spin][ih][kl][m+3][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
	      }

	      tmpvec1[n][Anum+m+0] += sum0;
	      tmpvec1[n][Anum+m+1] += sum1;
	      tmpvec1[n][Anum+m+2] += sum2;
	      tmpvec1[n][Anum+m+3] += sum3;
	    }
	  } 

	  for (; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += Hks[spin][ih][kl][m][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
	      }

	      tmpvec1[n][Anum+m] += sum;
	    }
	  } 
	  /**/
#endif

	}
      }
    }

    /* u1^+ C^+ u2 */

#ifdef nosse

    /* Original version */
    /**/    
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize[Mc_AN]; i++){
            sum += Krylov_U[spin][Mc_AN][rl0*KU_d1+m*KU_d2+i+1]*tmpvec1[n][i]; 
	  }
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
	}
      }
    }
    /**/

#else

    /* Unrolling + SSE version */
    /**/
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<(EKC_core_size[Mc_AN]-3); m+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  mmSum00 = _mm_setzero_pd();
	  mmSum01 = _mm_setzero_pd();
	  mmSum10 = _mm_setzero_pd();
	  mmSum11 = _mm_setzero_pd();
	  mmSum20 = _mm_setzero_pd();
	  mmSum21 = _mm_setzero_pd();
	  mmSum30 = _mm_setzero_pd();
	  mmSum31 = _mm_setzero_pd();

	  for (i=0; i<(Msize[Mc_AN]-3); i+=4){
	    mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i+0]);
	    mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i+2]);

	    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+1]),mmTmp0));  
	    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+3]),mmTmp1));  

	    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+1)*KU_d2+i+1]),mmTmp0));  
	    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+1)*KU_d2+i+3]),mmTmp1));  

	    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+2)*KU_d2+i+1]),mmTmp0));  
	    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+2)*KU_d2+i+3]),mmTmp1));  

	    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+3)*KU_d2+i+1]),mmTmp0));  
	    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+3)*KU_d2+i+3]),mmTmp1));  
	  }

	  mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	  mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	  mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	  mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	  _mm_storeu_pd(&mmArr[0], mmSum00);
	  _mm_storeu_pd(&mmArr[2], mmSum10);
	  _mm_storeu_pd(&mmArr[4], mmSum20);
	  _mm_storeu_pd(&mmArr[6], mmSum30);

	  sum0 = mmArr[0] + mmArr[1];
	  sum1 = mmArr[2] + mmArr[3];
	  sum2 = mmArr[4] + mmArr[5];
	  sum3 = mmArr[6] + mmArr[7];

	  for (; i<Msize[Mc_AN]; i++){
            sum0 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+1]*tmpvec1[n][i]; 
            sum1 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+1)*KU_d2+i+1]*tmpvec1[n][i]; 
            sum2 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+2)*KU_d2+i+1]*tmpvec1[n][i]; 
            sum3 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+3)*KU_d2+i+1]*tmpvec1[n][i]; 
	  }

          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum0;
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+2][rl*EKC_core_size[Mc_AN]+n+1] = sum1;
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+3][rl*EKC_core_size[Mc_AN]+n+1] = sum2;
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+4][rl*EKC_core_size[Mc_AN]+n+1] = sum3;
	}
      }

      for (; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize[Mc_AN]; i++){
            sum += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+1]*tmpvec1[n][i]; 
	  }
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
	}
      }

    }
    /**/

#endif

  } /* rl */

  for (i=0; i<EKC_core_size_max; i++){
    ;
  }
  ;
    

  } /* #pragma omp parallel */

  /*******************************
            u2^+ C u1 
  *******************************/
  
  for (i=1; i<=Msize3[Mc_AN]; i++){
    for (j=i; j<=Msize3[Mc_AN]; j++){

      tmp1 = EC_matrix[spin][Mc_AN][i][j];
      tmp2 = EC_matrix[spin][Mc_AN][j][i];
      tmp3 = tmp1 + tmp2;   

      EC_matrix[spin][Mc_AN][i][j] = tmp3; 
      EC_matrix[spin][Mc_AN][j][i] = tmp3;
    }
  }

  /*******************************
             u2^+ B u2 
  *******************************/

#pragma omp parallel shared(EKC_core_size_max,Msize2_max,rlmax_EC,Mc_AN,EKC_core_size,Msize2,fan,can,natn,ct_AN,Spe_Total_CNO,WhatSpecies,MP,S_G2M,RMI1,Hks,Krylov_U,spin,KU_d1,KU_d2,Msize,EC_matrix)   private(OMPID,Nthrds,Nprocs,rl,n,i,ig,ian,Anum,ih,j,kl,jg,jan,Bnum,m,sum,k,mmSum00,mmSum01,mmSum10,mmSum11,mmSum20,mmSum21,mmSum30,mmSum31,mmTmp0,mmTmp1,mmArr,sum0,sum1,sum2,sum3,rl0)

  {

    double **tmpvec1;
    tmpvec1=OpenMX_Solver_AlignedRealMatrix(solver_owner,"tmpvec1-2",((size_t)1)*EKC_core_size_max,((size_t)1)*Msize2_max);
    for (i=0; i<EKC_core_size_max; i++){
      ;
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();
   
  for (rl=0+OMPID; rl<rlmax_EC[Mc_AN]; rl+=Nthrds){

    /* B u2 */

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=fan+1; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=fan+1; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j];

	if (0<=kl){

#ifdef nosse

	  /* Original version */
	  /**/	  
	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += Hks[spin][ih][kl][m][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
	      }

	      tmpvec1[n][Anum+m] += sum;
	    }
	  }
	  /**/

#else

	  /* Unrolling + SSE version */
	  /**/
	  for (m=0; m<(ian-3); m+=4){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      mmSum00 = _mm_setzero_pd();
	      mmSum01 = _mm_setzero_pd();
	      mmSum10 = _mm_setzero_pd();
	      mmSum11 = _mm_setzero_pd();
	      mmSum20 = _mm_setzero_pd();
	      mmSum21 = _mm_setzero_pd();
	      mmSum30 = _mm_setzero_pd();
	      mmSum31 = _mm_setzero_pd();

	      for (k=0; k<(jan-3); k+=4){
		mmTmp0 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+0]);
		mmTmp1 = _mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+2]);

		mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+0][k+0]),mmTmp0));  
		mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+0][k+2]),mmTmp1));  

		mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+1][k+0]),mmTmp0));  
		mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+1][k+2]),mmTmp1));  

		mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+2][k+0]),mmTmp0));  
		mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+2][k+2]),mmTmp1));  

		mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+3][k+0]),mmTmp0));  
		mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Hks[spin][ih][kl][m+3][k+2]),mmTmp1));  
	      }

	      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	      _mm_storeu_pd(&mmArr[0], mmSum00);
	      _mm_storeu_pd(&mmArr[2], mmSum10);
	      _mm_storeu_pd(&mmArr[4], mmSum20);
	      _mm_storeu_pd(&mmArr[6], mmSum30);

	      sum0 = mmArr[0] + mmArr[1];
	      sum1 = mmArr[2] + mmArr[3];
	      sum2 = mmArr[4] + mmArr[5];
	      sum3 = mmArr[6] + mmArr[7];

	      for (; k<jan; k++){
		sum0 += Hks[spin][ih][kl][m+0][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+0];
		sum1 += Hks[spin][ih][kl][m+1][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+0];
		sum2 += Hks[spin][ih][kl][m+2][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+0];
		sum3 += Hks[spin][ih][kl][m+3][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k+0];
	      }

	      tmpvec1[n][Anum+m+0] += sum0;
	      tmpvec1[n][Anum+m+1] += sum1;
	      tmpvec1[n][Anum+m+2] += sum2;
	      tmpvec1[n][Anum+m+3] += sum3;
	    }
	  }

	  for (; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += Hks[spin][ih][kl][m][k]*Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+Bnum+k];
	      }

	      tmpvec1[n][Anum+m] += sum;
	    }
	  } 
	  /**/ 

#endif

	}
      }
    }

    /* u2^+ B u2 */

#ifdef nosse

    /* Original version */
    /**/    
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=Msize[Mc_AN]; i<Msize2[Mc_AN]; i++){
            sum += Krylov_U[spin][Mc_AN][rl0*KU_d1+m*KU_d2+i+1]*tmpvec1[n][i]; 
	  }
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] += sum;
	}
      }
    }
    /**/

#else

    /* Unrolling + SSE version */
    /**/
    for (rl0=0; rl0<rlmax_EC[Mc_AN]; rl0++){
      for (m=0; m<(EKC_core_size[Mc_AN]-3); m+=4){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  mmSum00 = _mm_setzero_pd();
	  mmSum01 = _mm_setzero_pd();
	  mmSum10 = _mm_setzero_pd();
	  mmSum11 = _mm_setzero_pd();
	  mmSum20 = _mm_setzero_pd();
	  mmSum21 = _mm_setzero_pd();
	  mmSum30 = _mm_setzero_pd();
	  mmSum31 = _mm_setzero_pd();

	  for (i=Msize[Mc_AN]; i<(Msize2[Mc_AN]-3); i+=4){
	    mmTmp0 = _mm_loadu_pd(&tmpvec1[n][i+0]);
	    mmTmp1 = _mm_loadu_pd(&tmpvec1[n][i+2]);

	    mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+1]),mmTmp0));  
	    mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+3]),mmTmp1));  

	    mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+1)*KU_d2+i+1]),mmTmp0));  
	    mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+1)*KU_d2+i+3]),mmTmp1));  

	    mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+2)*KU_d2+i+1]),mmTmp0));  
	    mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+2)*KU_d2+i+3]),mmTmp1));  

	    mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+3)*KU_d2+i+1]),mmTmp0));  
	    mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+3)*KU_d2+i+3]),mmTmp1));  
	  }

	  mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	  mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	  mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	  mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	  _mm_storeu_pd(&mmArr[0], mmSum00);
	  _mm_storeu_pd(&mmArr[2], mmSum10);
	  _mm_storeu_pd(&mmArr[4], mmSum20);
	  _mm_storeu_pd(&mmArr[6], mmSum30);

	  sum0 = mmArr[0] + mmArr[1];
	  sum1 = mmArr[2] + mmArr[3];
	  sum2 = mmArr[4] + mmArr[5];
	  sum3 = mmArr[6] + mmArr[7];

	  for (; i<Msize2[Mc_AN]; i++){
            sum0 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+1]*tmpvec1[n][i]; 
            sum1 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+1)*KU_d2+i+1]*tmpvec1[n][i]; 
            sum2 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+2)*KU_d2+i+1]*tmpvec1[n][i]; 
            sum3 += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+3)*KU_d2+i+1]*tmpvec1[n][i]; 
	  }

          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] += sum0;
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+2][rl*EKC_core_size[Mc_AN]+n+1] += sum1;
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+3][rl*EKC_core_size[Mc_AN]+n+1] += sum2;
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+4][rl*EKC_core_size[Mc_AN]+n+1] += sum3;
	}
      }

      for (; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=Msize[Mc_AN]; i<Msize2[Mc_AN]; i++){
            sum += Krylov_U[spin][Mc_AN][rl0*KU_d1+(m+0)*KU_d2+i+1]*tmpvec1[n][i]; 
	  }
          EC_matrix[spin][Mc_AN][rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] += sum;
	}
      }

    }
    /**/

#endif

  } /* rl */

    for (i=0; i<EKC_core_size_max; i++){
      ;
    }
    ;

  } /* #pragma omp parallel */

}










typedef struct {
  int Mc_AN;
  double ****OLP0;
  double ***Krylov_U_OLP;
  double **inv_RS;
  int *MP;
  int *Msize2;
  int *Msize4;
  int Msize2_max;
  double **tmpvec0;
  double **tmpvec1;
} Krylov_IOLP_trd_scratch_arguments;
static void Krylov_IOLP_trd_raii( int Mc_AN, double ****OLP0, double ***Krylov_U_OLP, double **inv_RS, int *MP,
                  int *Msize2, int *Msize4, int Msize2_max, double **tmpvec0, double **tmpvec1 ,OpenMXSolverInvocation *solver_owner);
static int Krylov_IOLP_trd_scratch_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Krylov_IOLP_trd_scratch_arguments *a=(Krylov_IOLP_trd_scratch_arguments*)arguments;
  Krylov_IOLP_trd_raii(a->Mc_AN,a->OLP0,a->Krylov_U_OLP,a->inv_RS,a->MP,a->Msize2,a->Msize4,a->Msize2_max,a->tmpvec0,a->tmpvec1,owner);
  *result=0;
  return 1;
}
void Krylov_IOLP_trd( int Mc_AN, double ****OLP0, double ***Krylov_U_OLP, double **inv_RS, int *MP, 
                  int *Msize2, int *Msize4, int Msize2_max, double **tmpvec0, double **tmpvec1 )
{
  Krylov_IOLP_trd_scratch_arguments arguments={Mc_AN,OLP0,Krylov_U_OLP,inv_RS,MP,Msize2,Msize4,Msize2_max,tmpvec0,tmpvec1};
  OpenMX_Solver_Run(Krylov_IOLP_trd_scratch_operation,&arguments);
}

static void Krylov_IOLP_trd_raii( int Mc_AN, double ****OLP0, double ***Krylov_U_OLP, double **inv_RS, int *MP,
                  int *Msize2, int *Msize4, int Msize2_max, double **tmpvec0, double **tmpvec1 ,OpenMXSolverInvocation *solver_owner)
{
  int rl,ct_AN,fan,san,can,wan,ct_on,i,j;
  int n,Anum,Bnum,k,ian,ih,kl,jg,ig,jan,m,m1,n1,info;
  int rl0,ZeroNum,Neumann_series,ns,Gh_AN,wanB;
  double sum,dum,tmp0,tmp1,tmp2,tmp3,rcutA,r0;
  double **tmpmat0;
  double *ko,*iko;
  double **FS;

  ct_AN = M2G[Mc_AN];
  fan = FNAN[ct_AN];
  san = SNAN[ct_AN];
  can = fan + san;
  wan = WhatSpecies[ct_AN];
  ct_on = Spe_Total_CNO[wan];
  rcutA = Spe_Atom_Cut1[wan]; 

  /* allocation of arrays */

  tmpmat0=OpenMX_Solver_AlignedRealMatrix(solver_owner,"tmpmat0-1",((size_t)1)*(EKC_core_size[Mc_AN]+4),((size_t)1)*(EKC_core_size[Mc_AN]+4));
  for (i=0; i<(EKC_core_size[Mc_AN]+4); i++){
    ;
  }

  FS=OpenMX_Solver_AlignedRealMatrix(solver_owner,"FS-2",((size_t)1)*(rlmax_EC2[Mc_AN]+2)*EKC_core_size[Mc_AN],((size_t)1)*(rlmax_EC2[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  for (i=0; i<(rlmax_EC2[Mc_AN]+2)*EKC_core_size[Mc_AN]; i++){
    ;
  }

  ko=(double*)OpenMX_Solver_Real(solver_owner,"ko-3",((size_t)1)*(rlmax_EC2[Mc_AN]+2)*EKC_core_size[Mc_AN]);
  iko=(double*)OpenMX_Solver_Real(solver_owner,"iko-4",((size_t)1)*(rlmax_EC2[Mc_AN]+2)*EKC_core_size[Mc_AN]);

  /****************************************************
      initialize 
  ****************************************************/

  for (i=0; i<EKC_core_size_max; i++){
    for (j=0; j<Msize2_max; j++){
      tmpvec0[i][j] = 0.0;
      tmpvec1[i][j] = 0.0;
    }
  }

  /* find the nearest atom with distance of r0 */   

  r0 = 1.0e+10;
  for (k=1; k<=FNAN[ct_AN]; k++){
    Gh_AN = natn[ct_AN][k];
    wanB = WhatSpecies[Gh_AN];
    if (Dis[ct_AN][k]<r0)  r0 = Dis[ct_AN][k]; 
  }

  /* starting vector */

  m = 0;  
  for (k=0; k<=FNAN[ct_AN]; k++){

    Gh_AN = natn[ct_AN][k];
    wanB = WhatSpecies[Gh_AN];

    if ( Dis[ct_AN][k]<(scale_rc_EKC[Mc_AN]*r0) ){

      Anum = MP[k] - 1; 
      
      for (i=0; i<Spe_Total_CNO[wanB]; i++){

        tmpvec0[m][Anum+i]         = 1.0;
        Krylov_U_OLP[0][m][Anum+i] = 1.0;

        m++;  
      }
    }
  }

  /*
  for (i=0; i<EKC_core_size[Mc_AN]; i++){
    tmpvec0[i][i]         = 1.0;
    Krylov_U_OLP[0][i][i] = 1.0;
  }
  */

  /****************************************************
           generate Krylov subspace vectors
  ****************************************************/

  for (rl=0; rl<(rlmax_EC2[Mc_AN]-1); rl++){

    /*******************************************************
                            S * |Wn)
    *******************************************************/

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=0; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=0; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
        Bnum = MP[j] - 1;

        if (0<=kl){

	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*tmpvec0[n][Bnum+k];
	      }

              tmpvec1[n][Anum+m] += sum;
	    }
	  } 
	}
      }
    }

    /*************************************************************
      orthogonalization by a modified block Gram-Schmidt method 
    *************************************************************/

    /* |tmpvec1) := (I - \sum_{rl0} |U_rl0)(U_rl0|)|tmpvec1) */

    for (rl0=0; rl0<=rl; rl0++){

      /* (U_rl0|tmpvec1) */

      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
          sum = 0.0;
  	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += Krylov_U_OLP[rl0][m][i]*tmpvec1[n][i];
	  }
          tmpmat0[m][n] = sum;
	}
      }

      /* |tmpvec1) := |tmpvec1) - |U_rl0)(U_rl0|tmpvec1) */

      for (n=0; n<EKC_core_size[Mc_AN]; n++){
	for (k=0; k<EKC_core_size[Mc_AN]; k++){
	  dum = tmpmat0[k][n];
	  for (i=0; i<Msize2[Mc_AN]; i++)  tmpvec1[n][i] -= Krylov_U_OLP[rl0][k][i]*dum;
        }
      }
    }

    /*************************************************************
                       normalization of tmpvec1
    *************************************************************/

    for (m=0; m<EKC_core_size[Mc_AN]; m++){
      for (n=0; n<EKC_core_size[Mc_AN]; n++){

        sum = 0.0;
        for (i=0; i<Msize2[Mc_AN]; i++){
          sum += tmpvec1[m][i]*tmpvec1[n][i]; 
	}

        tmpmat0[m+1][n+1] = sum;
      }
    }

    /* diagonalize tmpmat0 */ 

    if ( EKC_core_size[Mc_AN]==1){
      ko[1] = tmpmat0[1][1];
      tmpmat0[1][1] = 1.0;
    }
    else{

      info = Eigen_lapack_d(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
      if (info!=0){
	info = Eigen_lapack_x(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
	if (info!=0){
	  Eigen_lapack_r(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
	}
      }
    }

    ZeroNum = 0;

    for (n=1; n<=EKC_core_size[Mc_AN]; n++){
      if (cutoff_value<ko[n]){
	ko[n]  = sqrt(fabs(ko[n]));
	iko[n] = 1.0/ko[n];
      }
      else{
	ZeroNum++;
	ko[n]  = 0.0;
	iko[n] = 0.0;
      }

      if (error_check==1){
        printf("rl=%3d ko=%18.15f\n",rl,ko[n]);
      }
    }

    if (error_check==1){
      printf("rl=%3d ZeroNum=%3d\n",rl,ZeroNum);
    } 

    /* tmpvec0 = tmpvec1 * tmpmat0^{-1/2} */ 

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec0[n][i] = 0.0;
      }
    }

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (k=0; k<EKC_core_size[Mc_AN]; k++){
	dum = tmpmat0[k+1][n+1]*iko[n+1];
	for (i=0; i<Msize2[Mc_AN]; i++)  tmpvec0[n][i] += tmpvec1[k][i]*dum;
      }
    }

    /*************************************************************
                         store Krylov vectors
    *************************************************************/

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
        Krylov_U_OLP[rl+1][n][i] = tmpvec0[n][i];
      }
    }

  } /* rl */

  /************************************************************
       calculate the inverse of the reduced overlap matrix
  ************************************************************/

  /*  construct the reduced overlap matrix */

  for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){

    /*  S * |Vn) */

    for (n=0; n<EKC_core_size[Mc_AN]; n++){
      for (i=0; i<Msize2[Mc_AN]; i++){
	tmpvec1[n][i]  = 0.0;
      }
    }

    for (i=0; i<=can; i++){

      ig = natn[ct_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i] - 1;
      ih = S_G2M[ig];

      for (j=0; j<=can; j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[ct_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j] - 1;

	if (0<=kl){

	  for (m=0; m<ian; m++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (k=0; k<jan; k++){
		sum += OLP0[ih][kl][m][k]*Krylov_U_OLP[rl][n][Bnum+k];
	      }
	      tmpvec1[n][Anum+m] += sum;
	    }
	  } 
	}
      }
    }

    for (rl0=rl; rl0<rlmax_EC2[Mc_AN]; rl0++){
      for (m=0; m<EKC_core_size[Mc_AN]; m++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){
  	  sum = 0.0;
	  for (i=0; i<Msize2[Mc_AN]; i++){
            sum += Krylov_U_OLP[rl0][m][i]*tmpvec1[n][i]; 
	  }
          FS[rl0*EKC_core_size[Mc_AN]+m+1][rl*EKC_core_size[Mc_AN]+n+1] = sum;
          FS[rl*EKC_core_size[Mc_AN]+n+1][rl0*EKC_core_size[Mc_AN]+m+1] = sum;
	}
      }
    }
  }

  /* diagonalize FS */

  info = Eigen_lapack_d(FS,ko,Msize4[Mc_AN],Msize4[Mc_AN]);
  if (info!=0){
    info = Eigen_lapack_x(FS,ko,Msize4[Mc_AN],Msize4[Mc_AN]);
    if (info!=0){
      Eigen_lapack_r(FS,ko,Msize4[Mc_AN],Msize4[Mc_AN]);
    }
  }

  /* find ill-conditioned eigenvalues */

  ZeroNum = 0;
  for (i=1; i<=Msize4[Mc_AN]; i++){

    if (error_check==1){
      printf("Mc_AN=%2d i=%3d ko[i]=%18.15f\n",Mc_AN,i,ko[i]);
    }

    if (cutoff_value<ko[i]){
      iko[i] = 1.0/ko[i];
    }
    else{
      ZeroNum++;
      ko[i]  = 0.0;
      iko[i] = 0.0;
    }
  }  

  if (error_check==1){
    printf("Mc_AN=%2d ZeroNum=%2d\n",Mc_AN,ZeroNum);
  }

  /* construct the inverse */

  for (i=1; i<=Msize4[Mc_AN]; i++){
    for (j=1; j<=Msize4[Mc_AN]; j++){
      sum = 0.0;
      for (k=1; k<=Msize4[Mc_AN]; k++){
        sum += FS[i][k]*iko[k]*FS[j][k];
      }
      inv_RS[i-1][j-1] = sum;
    }
  }

  /* symmetrization of inv_RS */

  for (i=1; i<=Msize4[Mc_AN]; i++){
    for (j=i+1; j<=Msize4[Mc_AN]; j++){
      tmp0 = inv_RS[i-1][j-1];
      tmp1 = inv_RS[j-1][i-1];
      tmp2 = 0.5*(tmp0 + tmp1);
      inv_RS[i-1][j-1] = tmp2;
      inv_RS[j-1][i-1] = tmp2;
    }
  }

  /*
  {


    double mat1[1000][1000];
    double tsum;
    int myid;

    MPI_Comm_rank(mpi_comm_level1,&myid);

    if (myid==0){

      printf("check normalization\n");

      for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){
	for (m=0; m<EKC_core_size[Mc_AN]; m++){
	  for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

              sum = 0.0; 
   	      for (i=0; i<Msize2[Mc_AN]; i++){
                sum += Krylov_U_OLP[rl][m][i]*Krylov_U_OLP[rl0][n][i]; 
	      }
              printf("rl=%3d rl0=%3d m=%3d n=%3d  <|>=%18.15f\n",rl,rl0,m,n,sum);
	    }
	  }
	}
      }


      printf("\n\ninvS\n");


      for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){
	for (m=0; m<EKC_core_size[Mc_AN]; m++){

	  for (i=0; i<Msize2[Mc_AN]; i++){

	    sum = 0.0;

	    for (rl0=0; rl0<rlmax_EC2[Mc_AN]; rl0++){
	      for (n=0; n<EKC_core_size[Mc_AN]; n++){
		sum += inv_RS[rl*EKC_core_size[Mc_AN]+m][rl0*EKC_core_size[Mc_AN]+n]*Krylov_U_OLP[rl0][n][i]; 
	      }
	    }

	    mat1[rl*EKC_core_size[Mc_AN]+m][i] = sum; 
	  }
	}
      }


      tsum = 0.0;

      for (i=0; i<Msize2[Mc_AN]; i++){
	for (j=0; j<Msize2[Mc_AN]; j++){

	  sum = 0.0;
	  for (rl=0; rl<rlmax_EC2[Mc_AN]; rl++){
	    for (m=0; m<EKC_core_size[Mc_AN]; m++){
	      sum += Krylov_U_OLP[rl][m][i]*mat1[rl*EKC_core_size[Mc_AN]+m][j];
	    }
	  }

	  printf("i=%4d j=%4d %18.15f\n",i,j,sum);

          tsum += fabs(sum);

	}    
      }    


      printf("\n\ntsum=%18.15f\n",tsum);

    }


    MPI_Finalize();
    exit(0);
  }
  */

  /* freeing of arrays */

  for (i=0; i<(EKC_core_size[Mc_AN]+4); i++){
    ;
  }
  ;

  for (i=0; i<(rlmax_EC2[Mc_AN]+2)*EKC_core_size[Mc_AN]; i++){
    ;
  }
  ;

  ;
  ;
}












void S_orthonormalize_vec_trd( int Mc_AN, int ct_on, double **vec,
                           double **workvec, double ****OLP0,
                           double **tmpmat0, double *ko, double *iko, 
                           int *MP, int *Msize2 )

/* This subroutine is exactly the same as the original S_orthonormalize_vec,
except that Eigen_lapack is replaced with Eigen_lapack_d, _x, or _r
*/

{
  int n,i,j,can,san,fan,ct_AN;
  int k,m,ZeroNum,Anum,Bnum,ih;
  int kl,jg,jan,ig,ian,info;
  double dum,sum;
  double tmp0,tmp1,tmp2,tmp3;
  __m128d mmSum00,mmSum01,mmSum10,mmSum11,mmSum20,mmSum21,mmSum30,mmSum31, mmTmp0, mmTmp1, mmTmp2, mmTmp3, mmTmp4, mmTmp5; 
  double mmArr[8];
  double sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7;

  ct_AN = M2G[Mc_AN];
  fan = FNAN[ct_AN];
  san = SNAN[ct_AN];
  can = fan + san;

  /* S|Vn) */

  for (n=0; n<EKC_core_size[Mc_AN]; n++){
    for (i=0; i<Msize2[Mc_AN]; i++){
      workvec[n][i]  = 0.0;
    }
  }

  for (i=0; i<=can; i++){

    ig = natn[ct_AN][i];
    ian = Spe_Total_CNO[WhatSpecies[ig]];
    Anum = MP[i] - 1;
    ih = S_G2M[ig];

    for (j=0; j<=can; j++){

      kl = RMI1[Mc_AN][i][j];
      jg = natn[ct_AN][j];
      jan = Spe_Total_CNO[WhatSpecies[jg]];
      Bnum = MP[j] - 1;

      if (0<=kl){

#ifdef nosse

	/* Original version */
	/**/	
	for (m=0; m<ian; m++){
	  for (n=0; n<EKC_core_size[Mc_AN]; n++){

	    sum = 0.0;
	    for (k=0; k<jan; k++){
	      sum += OLP0[ih][kl][m][k]*vec[n][Bnum+k];
	    }

	    workvec[n][Anum+m] += sum;
	  }
	}
	/**/

#else

	/* Unrolling + SSE version */
	/**/
	for (m=0; m<(ian-3); m+=4){
	  for (n=0; n<EKC_core_size[Mc_AN]; n++){

	    mmSum00 = _mm_setzero_pd();
	    mmSum01 = _mm_setzero_pd();
	    mmSum10 = _mm_setzero_pd();
	    mmSum11 = _mm_setzero_pd();
	    mmSum20 = _mm_setzero_pd();
	    mmSum21 = _mm_setzero_pd();
	    mmSum30 = _mm_setzero_pd();
	    mmSum31 = _mm_setzero_pd();
	    
	    for (k=0; k<(jan-3); k+=4){
	      mmTmp0 = _mm_loadu_pd(&vec[n][Bnum+k+0]);
	      mmTmp1 = _mm_loadu_pd(&vec[n][Bnum+k+2]);

	      mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+0]),mmTmp0));  
	      mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+0][k+2]),mmTmp1));  

	      mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+0]),mmTmp0));  
	      mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+1][k+2]),mmTmp1));  

	      mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+0]),mmTmp0));  
	      mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+2][k+2]),mmTmp1));  

	      mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+0]),mmTmp0));  
	      mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&OLP0[ih][kl][m+3][k+2]),mmTmp1));  
	    }

	    mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
	    mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
	    mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
	    mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
	  
	    _mm_storeu_pd(&mmArr[0], mmSum00);
	    _mm_storeu_pd(&mmArr[2], mmSum10);
	    _mm_storeu_pd(&mmArr[4], mmSum20);
	    _mm_storeu_pd(&mmArr[6], mmSum30);

	    sum0 = mmArr[0] + mmArr[1];
	    sum1 = mmArr[2] + mmArr[3];
	    sum2 = mmArr[4] + mmArr[5];
	    sum3 = mmArr[6] + mmArr[7];

	    for (; k<jan; k++){
	      sum0 += OLP0[ih][kl][m+0][k]*vec[n][Bnum+k];
	      sum1 += OLP0[ih][kl][m+1][k]*vec[n][Bnum+k];
	      sum2 += OLP0[ih][kl][m+2][k]*vec[n][Bnum+k];
	      sum3 += OLP0[ih][kl][m+3][k]*vec[n][Bnum+k];
	    }

	    workvec[n][Anum+m+0] += sum0;
	    workvec[n][Anum+m+1] += sum1;
	    workvec[n][Anum+m+2] += sum2;
	    workvec[n][Anum+m+3] += sum3;
	  }
	}

	for (; m<ian; m++){
	  for (n=0; n<EKC_core_size[Mc_AN]; n++){

	    sum = 0.0;
	    for (k=0; k<jan; k++){
	      sum += OLP0[ih][kl][m][k]*vec[n][Bnum+k];
	    }

	    workvec[n][Anum+m] += sum;
	  }
	}
	/**/ 

#endif

      }
    }
  }

  /*  (Vn|S|Vn) */

#ifdef nosse

  /* Original version */
  /**/  
  for (m=0; m<EKC_core_size[Mc_AN]; m++){
    for (n=m; n<EKC_core_size[Mc_AN]; n++){
      sum = 0.0;
      for (i=0; i<Msize2[Mc_AN]; i++){
	sum += vec[m][i]*workvec[n][i];
      }
      tmpmat0[m+1][n+1] = sum;
      tmpmat0[n+1][m+1] = sum;
    }
  }
  /**/

#else

  /* Unrolling + SSE version */
  /**/
  for (m=0; m<(EKC_core_size[Mc_AN]-3); m+=4){
    for (n=m; n<EKC_core_size[Mc_AN]; n++){
      mmSum00 = _mm_setzero_pd();
      mmSum01 = _mm_setzero_pd();
      mmSum10 = _mm_setzero_pd();
      mmSum11 = _mm_setzero_pd();
      mmSum20 = _mm_setzero_pd();
      mmSum21 = _mm_setzero_pd();
      mmSum30 = _mm_setzero_pd();
      mmSum31 = _mm_setzero_pd();

      for (i=0; i<(Msize2[Mc_AN]-3); i+=4){
	mmTmp0 = _mm_loadu_pd(&workvec[n][i+0]);
	mmTmp1 = _mm_loadu_pd(&workvec[n][i+2]);

	mmSum00 = _mm_add_pd(mmSum00, _mm_mul_pd(_mm_loadu_pd(&vec[m+0][i+0]),mmTmp0));  
	mmSum01 = _mm_add_pd(mmSum01, _mm_mul_pd(_mm_loadu_pd(&vec[m+0][i+2]),mmTmp1));  

	mmSum10 = _mm_add_pd(mmSum10, _mm_mul_pd(_mm_loadu_pd(&vec[m+1][i+0]),mmTmp0));  
	mmSum11 = _mm_add_pd(mmSum11, _mm_mul_pd(_mm_loadu_pd(&vec[m+1][i+2]),mmTmp1));  

	mmSum20 = _mm_add_pd(mmSum20, _mm_mul_pd(_mm_loadu_pd(&vec[m+2][i+0]),mmTmp0));  
	mmSum21 = _mm_add_pd(mmSum21, _mm_mul_pd(_mm_loadu_pd(&vec[m+2][i+2]),mmTmp1));  

	mmSum30 = _mm_add_pd(mmSum30, _mm_mul_pd(_mm_loadu_pd(&vec[m+3][i+0]),mmTmp0));  
	mmSum31 = _mm_add_pd(mmSum31, _mm_mul_pd(_mm_loadu_pd(&vec[m+3][i+2]),mmTmp1));  
      }

      mmSum00 = _mm_add_pd(mmSum00, mmSum01);  
      mmSum10 = _mm_add_pd(mmSum10, mmSum11);  
      mmSum20 = _mm_add_pd(mmSum20, mmSum21);  
      mmSum30 = _mm_add_pd(mmSum30, mmSum31);  
      
      _mm_storeu_pd(&mmArr[0], mmSum00);
      _mm_storeu_pd(&mmArr[2], mmSum10);
      _mm_storeu_pd(&mmArr[4], mmSum20);
      _mm_storeu_pd(&mmArr[6], mmSum30);

      sum0 = mmArr[0] + mmArr[1];
      sum1 = mmArr[2] + mmArr[3];
      sum2 = mmArr[4] + mmArr[5];
      sum3 = mmArr[6] + mmArr[7];

      for (; i<Msize2[Mc_AN]; i++){
	sum0 += vec[m+0][i]*workvec[n][i];
	sum1 += vec[m+1][i]*workvec[n][i];
	sum2 += vec[m+2][i]*workvec[n][i];
	sum3 += vec[m+3][i]*workvec[n][i];
      }

      tmpmat0[m+1][n+1] = sum0;
      tmpmat0[n+1][m+1] = sum0;

      tmpmat0[m+2][n+1] = sum1;
      tmpmat0[n+1][m+2] = sum1;

      tmpmat0[m+3][n+1] = sum2;
      tmpmat0[n+1][m+3] = sum2;

      tmpmat0[m+4][n+1] = sum3;
      tmpmat0[n+1][m+4] = sum3;
    }
  }

  for (; m<EKC_core_size[Mc_AN]; m++){
    for (n=m; n<EKC_core_size[Mc_AN]; n++){
      sum = 0.0;
      for (i=0; i<Msize2[Mc_AN]; i++){
	sum += vec[m][i]*workvec[n][i];
      }
      tmpmat0[m+1][n+1] = sum;
      tmpmat0[n+1][m+1] = sum;
    }
  }
  /**/

#endif

  /* diagonalize tmpmat0 */ 

  if ( EKC_core_size[Mc_AN]==1){
    ko[1] = tmpmat0[1][1];
    tmpmat0[1][1] = 1.0;
  }
  else{

    info = Eigen_lapack_d(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
    if (info!=0){
      info = Eigen_lapack_x(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
      if (info!=0){
	Eigen_lapack_r(tmpmat0, ko, EKC_core_size[Mc_AN], EKC_core_size[Mc_AN]);
      }
    }
  }

  ZeroNum = 0;

  for (n=1; n<=EKC_core_size[Mc_AN]; n++){
    if (cutoff_value<ko[n]){
      ko[n]  = sqrt(fabs(ko[n]));
      iko[n] = 1.0/ko[n];
    }
    else{
      ZeroNum++;
      ko[n]  = 0.0;
      iko[n] = 0.0;
    }
  }

  /* U0 = vec * tmpmat0^{-1/2} */ 

  for (n=0; n<EKC_core_size[Mc_AN]; n++){
    for (i=0; i<Msize2[Mc_AN]; i++){
      workvec[n][i] = 0.0;
    }
  }

  for (n=0; n<EKC_core_size[Mc_AN]; n++){
    for (k=0; k<EKC_core_size[Mc_AN]; k++){
      dum = tmpmat0[k+1][n+1]*iko[n+1];
      for (i=0; i<Msize2[Mc_AN]; i++)  workvec[n][i] += vec[k][i]*dum;
    }
  }
  
  for (n=0; n<EKC_core_size[Mc_AN]; n++){
    for (i=0; i<Msize2[Mc_AN]; i++){
      vec[n][i] = workvec[n][i];
    }
  }
}


typedef struct {
  double **a;
  double *ko;
  int n0;
  int EVmax;
} Eigen_lapack_x_scratch_arguments;
static int Eigen_lapack_x_raii(double **a, double *ko, int n0, int EVmax,OpenMXSolverInvocation *solver_owner);
static int Eigen_lapack_x_scratch_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Eigen_lapack_x_scratch_arguments *a=(Eigen_lapack_x_scratch_arguments*)arguments;
  *result=(double)Eigen_lapack_x_raii(a->a,a->ko,a->n0,a->EVmax,owner);
  return 1;
}
int Eigen_lapack_x(double **a, double *ko, int n0, int EVmax)
{
  Eigen_lapack_x_scratch_arguments arguments={a,ko,n0,EVmax};
  return (int)OpenMX_Solver_Run(Eigen_lapack_x_scratch_operation,&arguments);
}

static int Eigen_lapack_x_raii(double **a, double *ko, int n0, int EVmax,OpenMXSolverInvocation *solver_owner)
{

  /*
    F77_NAME(dsyevx,DSYEVX)()
  
    input:  n;
    input:  a[n][n];  matrix A
    output: a[n][n];  eigevectors
    output: ko[n];    eigenvalues 
  */
    
  char *name="Eigen_lapack";

  char  *JOBZ="V";
  char  *RANGE="I";
  char  *UPLO="L";

  INTEGER n=n0;
  INTEGER LDA=n0;
  double VL,VU; /* dummy */
  INTEGER IL,IU; 
  double ABSTOL=1.0e-13;
  INTEGER M;

  double *A,*Z;
  INTEGER LDZ=n;
  INTEGER LWORK;
  double *WORK;
  INTEGER *IWORK;
  INTEGER *IFAIL, INFO;

  int i,j;

  A=(double*)OpenMX_Solver_Real(solver_owner,"A-1",((size_t)1)*n*n);
  Z=(double*)OpenMX_Solver_Real(solver_owner,"Z-2",((size_t)1)*n*n);

  LWORK=n*8;
  WORK=(double*)OpenMX_Solver_Real(solver_owner,"WORK-3",((size_t)1)*LWORK);
  IWORK=(INTEGER*)OpenMX_Solver_Bytes(solver_owner,"IWORK-4",((size_t)1)*n*5,sizeof(INTEGER));
  IFAIL=(INTEGER*)OpenMX_Solver_Bytes(solver_owner,"IFAIL-5",((size_t)1)*n,sizeof(INTEGER));

  IL = 1;
  IU = EVmax;
 
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
       A[i*n+j] = a[i+1][j+1];
    }
  }

#if 0
  printf("A=\n");
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
       printf("%f ",A[i*n+j]);
    }
    printf("\n");
  }
  fflush(stdout);
#endif

  F77_NAME(dsyevx,DSYEVX)( JOBZ, RANGE, UPLO, &n, A, &LDA, &VL, &VU, &IL, &IU,
           &ABSTOL, &M, ko, Z, &LDZ, WORK, &LWORK, IWORK,
           IFAIL, &INFO ); 

  if (INFO==0){

    /* store eigenvectors */
    for (i=0;i<EVmax;i++) {
      for (j=0;j<n;j++) {
	/*  a[i+1][j+1]= Z[i*n+j]; */
	a[j+1][i+1]= Z[i*n+j];
      }
    }

    /* shift ko by 1 */
    for (i=EVmax; i>=1; i--){
      ko[i]= ko[i-1];
    }
  }

  if (INFO!=0) {
    /* printf("\n%s: error in dsyevx_, info=%d\n\n",name,INFO); */
  }
   
  ; ; ; ; ;
  return INFO;

}


typedef struct {
  double **a;
  double *ko;
  int n0;
  int EVmax;
} Eigen_lapack_d_scratch_arguments;
static int Eigen_lapack_d_raii(double **a, double *ko, int n0, int EVmax,OpenMXSolverInvocation *solver_owner);
static int Eigen_lapack_d_scratch_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Eigen_lapack_d_scratch_arguments *a=(Eigen_lapack_d_scratch_arguments*)arguments;
  *result=(double)Eigen_lapack_d_raii(a->a,a->ko,a->n0,a->EVmax,owner);
  return 1;
}
int Eigen_lapack_d(double **a, double *ko, int n0, int EVmax)
{
  Eigen_lapack_d_scratch_arguments arguments={a,ko,n0,EVmax};
  return (int)OpenMX_Solver_Run(Eigen_lapack_d_scratch_operation,&arguments);
}

static int Eigen_lapack_d_raii(double **a, double *ko, int n0, int EVmax,OpenMXSolverInvocation *solver_owner)
{

  /* 
    F77_NAME(dsyevd,DSYEVD)()
  
    input:  n;
    input:  a[n][n];  matrix A
    output: a[n][n];  eigevectors
    output: ko[n];    eigenvalues 
  */
    
  const char *name="Eigen_lapack";

  char  *JOBZ="V";
  char  *UPLO="L";

  INTEGER n=n0;
  INTEGER LDA=n;
  double VL,VU; /* dummy */
  INTEGER IL,IU; 
  double ABSTOL=1.0e-13;
  INTEGER M;

  double *A;
  INTEGER LDZ=n;
  INTEGER LWORK,LIWORK;
  double *WORK;
  INTEGER *IWORK;
  INTEGER INFO;

  int i,j;

  A=(double*)OpenMX_Solver_Real(solver_owner,"A-1",((size_t)1)*n*n);

  LWORK=  1 + 6*n + 2*n*n;
  WORK=(double*)OpenMX_Solver_Real(solver_owner,"WORK-2",((size_t)1)*LWORK);

  LIWORK = 3 + 5*n;
  IWORK=(INTEGER*)OpenMX_Solver_Bytes(solver_owner,"IWORK-3",((size_t)1)*LIWORK,sizeof(INTEGER));


  IL = 1;
  IU = EVmax; 
 
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
       A[i*n+j]= a[i+1][j+1];
    }
  }

#if 0
  printf("A=\n");
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
       printf("%f ",A[i*n+j]);
    }
    printf("\n");
  }
  fflush(stdout);
#endif

  F77_NAME(dsyevd,DSYEVD)( JOBZ, UPLO, &n, A, &LDA, ko, WORK, &LWORK, IWORK, &LIWORK, &INFO ); 

  if (INFO==0){

    /* store eigenvectors */
    for (i=0;i<EVmax;i++) {
      for (j=0;j<n;j++) {
	/* a[i+1][j+1]= Z[i*n+j]; */
	a[j+1][i+1]= A[i*n+j];
      }
    }

    /* shift ko by 1 */
    for (i=EVmax; i>=1; i--){
      ko[i]= ko[i-1];
    }
  }

  if (INFO!=0) {
    /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
  }
   
  ; ; ;
  return INFO;
}


typedef struct {
  double **a;
  double *ko;
  int n0;
  int EVmax;
} Eigen_lapack_r_scratch_arguments;
static int Eigen_lapack_r_raii(double **a, double *ko, int n0, int EVmax,OpenMXSolverInvocation *solver_owner);
static int Eigen_lapack_r_scratch_operation(OpenMXSolverInvocation *owner,void *arguments,double *result)
{
  Eigen_lapack_r_scratch_arguments *a=(Eigen_lapack_r_scratch_arguments*)arguments;
  *result=(double)Eigen_lapack_r_raii(a->a,a->ko,a->n0,a->EVmax,owner);
  return 1;
}
int Eigen_lapack_r(double **a, double *ko, int n0, int EVmax)
{
  Eigen_lapack_r_scratch_arguments arguments={a,ko,n0,EVmax};
  return (int)OpenMX_Solver_Run(Eigen_lapack_r_scratch_operation,&arguments);
}

static int Eigen_lapack_r_raii(double **a, double *ko, int n0, int EVmax,OpenMXSolverInvocation *solver_owner)
{

  /* 
    F77_NAME(dsyevr,DSYEVR)()
  
    input:  n;
    input:  a[n][n];  matrix A
    output: a[n][n];  eigevectors
    output: ko[n];    eigenvalues 
  */
    
  const char *name="Eigen_lapack";

  char  *JOBZ="V";
  char  *RANGE="I";
  char  *UPLO="L";

  INTEGER n=n0;
  INTEGER LDA=n;
  double VL,VU; /* dummy */
  INTEGER IL,IU; 
  double ABSTOL=1.0e-13;
  INTEGER M;

  double *A,*Z;
  INTEGER LDZ=n;
  INTEGER LWORK,LIWORK;
  double *WORK;
  INTEGER *IWORK;
  INTEGER *ISUPPZ; 
  INTEGER INFO;

  int i,j;

  A=(double*)OpenMX_Solver_Real(solver_owner,"A-1",((size_t)1)*n*n);
  Z=(double*)OpenMX_Solver_Real(solver_owner,"Z-2",((size_t)1)*n*n);

  LWORK= (n+16)*n;   /*  n*26 of (n+6)*n */
  WORK=(double*)OpenMX_Solver_Real(solver_owner,"WORK-3",((size_t)1)*LWORK);

  LIWORK = (n+1)*n;  /*  n*10 or ??? */
  IWORK=(INTEGER*)OpenMX_Solver_Bytes(solver_owner,"IWORK-4",((size_t)1)*LIWORK,sizeof(INTEGER));

  ISUPPZ=(INTEGER*)OpenMX_Solver_Bytes(solver_owner,"ISUPPZ-5",((size_t)1)*n*2,sizeof(INTEGER));

  IL = 1;
  IU = EVmax; 
 
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
       A[i*n+j]= a[i+1][j+1];
    }
  }

#if 0
  printf("A=\n");
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
       printf("%f ",A[i*n+j]);
    }
    printf("\n");
  }
  fflush(stdout);
#endif

  F77_NAME(dsyevr,DSYEVR)( JOBZ, RANGE, UPLO, &n, A, &LDA, &VL, &VU, &IL, &IU,
           &ABSTOL, &M, ko, Z, &LDZ, ISUPPZ, WORK, &LWORK,
           IWORK, &LIWORK, &INFO ); 

  if (INFO==0){

    /* store eigenvectors */
    for (i=0;i<EVmax;i++) {
      for (j=0;j<n;j++) {
	/*  a[i+1][j+1]= Z[i*n+j]; */
	a[j+1][i+1]= Z[i*n+j];
      }
    }

    /* shift ko by 1 */
    for (i=EVmax; i>=1; i--){
      ko[i]= ko[i-1];
    }
  }

  if (INFO!=0) {
     printf("\n%s: error in dsyevr_, info=%d\n\n",name,INFO);
     MPI_Finalize();
     exit(10);
  }
   
  ; ; ; ; ;

  return INFO;
}

static void Krylov_Reset(OpenMXNativeResources *owner, void *metadata)
{
  KrylovState *state = (KrylovState *)metadata;
  (void)owner;
  if (state->initialized) Krylov_Release_GPU_Caches(state);
}

OpenMXKrylovState *Krylov_Bind_Resources(OpenMXCoreContext *core)
{
  OpenMXNativeResources *owner = NULL;
  void *metadata = NULL;
  char error[256];
  if (!openmx_core_native_resources_state(core, "Krylov", sizeof(KrylovState),
      Krylov_Reset, &owner, &metadata, error, sizeof(error))) {
    fprintf(stderr, "OpenMX Krylov state: %s\n", error);
    abort();
  }
  KrylovState *state = (KrylovState *)metadata;
  if (!state->initialized) {
    state->resources = owner;
    if(!openmx_core_mixing_history(core,&state->history,error,sizeof(error))) {
      fprintf(stderr,"OpenMX Krylov mixing state: %s\n",error);
      abort();
    }
    state->gpu_device_ranks = 1;
    state->gpu_fits = 1;
    state->gpu_gemmul8 = 1;
    state->gpu_fused = 1;
    state->gpu_eigen_min = KRYLOV_GPU_EIGEN_MIN;
    state->gpu_dgemm_min_flops = KRYLOV_GPU_DGEMM_MIN_FLOPS;
    state->measure_time_env = -1;
    state->Col_trd_owned_firsttime = 1;
    state->Col_trd_owned_recalc_firsttime = 1;
    state->Col_owned_firsttime = 1;
    state->Col_owned_recalc_firsttime = 1;
    state->ProjectedSolve_GPU_profile = -1;
    state->GPU_Preflight_reserve_bytes = (size_t)512*1024*1024;
    state->GPU_Preflight_device_comm = MPI_COMM_NULL;
    state->GPU_Enabled_requested = -1;
    if(!openmx_core_chemical_potentials(core,&state->chemical,error,sizeof(error))) {
      fprintf(stderr,"OpenMX chemical potentials: %s\n",error);
      abort();
    }
    state->initialized = 1;
  }
  return state;
}
